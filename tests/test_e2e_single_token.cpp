/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// test_e2e_single_token -- ONE decode step, end to end, through the whole
// engine: ModelConfig -> arena -> descriptors -> planned stages -> injected
// MoE block -> 43-layer DecodeStep -> argmax readback.
//
// What is asserted (the probe contract):
//   * zero deadlocks      -- the step returns; every wait in the graph is
//                           event-ordered and every stream is synchronized
//                           exactly once at the readback. A hang here IS the
//                           failure signal (the harness times the binary out).
//   * zero stream errors  -- every ACL call in the path is status-checked and
//                           throws; reaching the checks below means every
//                           status on the way was 0.
//   * strictly 0 dynamic memory allocations inside the step (the pipeline's
//     own counter, which the static-runtime contract pins at 0), and 0
//     descriptors built inside the step.
//
// Model geometry comes from config.json when one is reachable (--config,
// then $DSV4_MODEL_DIR/config.json, then /mnt/c/models/DeepSeek-V4-Flash/
// config.json) and falls back to the compiled contract. Weights: on the
// symbolic mock runtime the graph runs over pure address bookkeeping with a
// deterministic seeded readback; on real hardware the probe runs the same
// graph over deterministic synthetic weights at an explicit --coverage
// bring-up subset (full coverage pins ~137 GiB of host RAM, which is a
// production run's decision, not a probe's default). No device at all: SKIP.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/model_config.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/pipeline.hpp"

#if ASCEND_MOCK_RUNTIME
#include "mock_allocator.hpp"
#include "mock_ops_api.hpp"
#endif

namespace {

using namespace ascend_moe;

constexpr const char* kDefaultConfigPath = "/mnt/c/models/DeepSeek-V4-Flash/config.json";

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  std::printf("  [ %s ] %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition) {
    ++g_failures;
  }
}

void Skip(const std::string& what) { std::printf("  [ SKIP ] %s\n", what.c_str()); }

std::string ConfigPathFrom(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::strcmp(argv[index], "--config") == 0) {
      return argv[index + 1];
    }
  }
  if (const char* from_env = std::getenv("DSV4_MODEL_DIR")) {
    const std::string beside = std::string(from_env) + "/config.json";
    std::ifstream probe(beside);
    if (probe.is_open()) {
      return beside;
    }
  }
  std::ifstream probe(kDefaultConfigPath);
  return probe.is_open() ? std::string(kDefaultConfigPath) : std::string();
}

// The mock's forced D2H reads are seeded: the per-layer top-6 vector picks
// experts 0..5 (already resident at init) and the greedy token comes back 0.
// Deterministic, in range, and enough to drive the real residency machinery.
void SeedDeviceToHost(void* destination, size_t count) {
  if (count == Int32Bytes(kNumExpertsPerTok)) {
    auto* ids = static_cast<int32_t*>(destination);
    for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
      ids[index] = static_cast<int32_t>(index);
    }
    return;
  }
  if (count == sizeof(int64_t)) {
    *static_cast<int64_t*>(destination) = 0;
  }
}

// A weight source that moves no byte: the mock's transfers are interval
// checks, so the graph runs over pure address bookkeeping.
class SymbolicWeightSource : public WeightByteSource {
 public:
  const char* source_name() const override { return "symbolic"; }
  bool Contains(int32_t layer, int32_t expert) const override {
    return layer >= 0 && layer < kNumLayers && expert >= 0 && expert < kNumRoutedExperts;
  }
  void ReadExpertSlotRange(uint8_t*, size_t destination_capacity, size_t slot_offset, size_t count, int32_t layer,
                           int32_t expert) override {
    RefuseIfClosed("ReadExpertSlotRange");
    DSV4_REQUIRE(Contains(layer, expert),
                 "expert (layer=" << layer << ", id=" << expert << ") is outside the symbolic coverage");
    DSV4_REQUIRE(slot_offset + count <= ExpertSlotLayout::ForDeepSeekV4Flash().slot_num_bytes(),
                 "symbolic read leaves the slot");
    DSV4_REQUIRE(count <= destination_capacity, "symbolic read exceeds the destination capacity");
  }
  bool HasNamed(const std::string&) const override { return true; }
  void ReadNamed(const std::string& name, uint8_t*, size_t destination_capacity, size_t, size_t count) override {
    RefuseIfClosed("ReadNamed");
    DSV4_REQUIRE(count <= destination_capacity, "named read '" << name << "' exceeds the capacity");
  }
  size_t NamedByteSize(const std::string&) const override { return 0; }  // 0 = do not size-check
  void Close() override { closed_ = true; }
};

}  // namespace

int main(int argc, char** argv) {
  std::printf("test_e2e_single_token -- one decode step through the whole engine\n");

  // ---- model configuration ------------------------------------------------
  ModelConfig model;
  std::string model_source;
  const std::string config_path = ConfigPathFrom(argc, argv);
  if (!config_path.empty()) {
    model = ModelConfig::FromJsonFile(config_path);
    model_source = config_path;
  } else {
    Skip("no config.json reachable; the compiled DSV4-Flash contract stands in");
  }
  try {
    model.Validate();
    model.AssertMatchesBinaryContract();
    std::printf("  model config: %s (%d layers, hidden %d, top-%d of %d)\n",
                model_source.empty() ? "<compiled contract>" : model_source.c_str(),
                static_cast<int>(model.num_hidden_layers), static_cast<int>(model.hidden_size),
                static_cast<int>(model.num_experts_per_tok), static_cast<int>(model.n_routed_experts));
  } catch (const std::exception& error) {
    std::printf("  [FAIL] model configuration rejected: %s\n", error.what());
    return 1;
  }

  // ---- device -------------------------------------------------------------
  AclDeviceOps* device = nullptr;
  try {
    device = new AclDeviceOps(0);
  } catch (const std::exception& error) {
    Skip(std::string("no usable NPU device on this host: ") + error.what());
    return 0;
  }
  std::unique_ptr<AclDeviceOps> device_guard(device);
  std::printf("  device SoC: %s\n", device->soc_name().c_str());

  try {
    RuntimeConfig config;
    config.synthetic_weights = true;
    config.max_context_len = 256;  // bring-up reservation; the probe runs one position

#if ASCEND_MOCK_RUNTIME
    // Symbolic everything: 64 GiB of reported HBM, a symbolic host half, and
    // seeded readbacks. Full 11,008 coverage costs zero physical bytes.
    ascend_moe::mock::MockResetAllocatorForTest();
    ascend_moe::mock::MockSetReportedHbm(64ull << 30);
    const bool mock_backend = true;
    const int64_t coverage = kTotalRoutedExperts;
#else
    const bool mock_backend = std::string(device->soc_name()).find("Mock") != std::string::npos;
    // Real hardware: synthetic weights are REAL bytes, so the hierarchy's host
    // half is real pinned RAM. The probe never guesses a footprint; coverage
    // is a command-line decision.
    int64_t coverage = kTotalRoutedExperts;
    bool coverage_given = false;
    for (int index = 1; index + 1 < argc; ++index) {
      if (std::strcmp(argv[index], "--coverage") == 0) {
        coverage = std::strtoll(argv[index + 1], nullptr, 10);
        coverage_given = true;
      }
    }
    if (!mock_backend && !coverage_given) {
      Skip("real hardware without --coverage: full residency pins ~137 GiB of host RAM, which is a "
           "production decision. Re-run with --coverage <experts> for a bring-up subset.");
      return 0;
    }
#endif
    config.routed_coverage = coverage;

    const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);

    ExclusiveExpertManager::Options options;
    options.num_layers = model.num_hidden_layers;
    options.num_experts = model.n_routed_experts;
    options.top_k = model.num_experts_per_tok;
    options.routed_coverage = coverage;
#if ASCEND_MOCK_RUNTIME
    options.host_available_bytes = 256ull << 30;  // the host half is symbolic; skip the meminfo guard
    options.transfer_chunk_bytes = 512 * 1024;    // keep the transit scratch in the real-memory tier
#endif
    // K: the largest device slot count the (reported) HBM holds after the
    // backbone and the reserves, the same planner the runner drives.
    size_t free_hbm = 0;
    size_t total_hbm = 0;
    DSV4_REQUIRE(device->QueryDeviceMemory(&free_hbm, &total_hbm), "aclrtGetMemInfo(ACL_HBM_MEM) failed");
    const size_t backbone_bytes =
        Dsv4Pipeline::BackboneDeviceBytes(model.mla, config.block_size, config.max_context_len);
    options.device_slots = ExclusiveExpertManager::PlanDeviceSlots(
        free_hbm, backbone_bytes, layout.slot_num_bytes(), coverage, kDeviceReserveBytes, kTransferChunkBytes,
        model.num_experts_per_tok, -1);
    ExclusiveExpertManager experts(*device, *device, layout, options);

    OpTable ops;
    DSV4_REQUIRE(ops.runtime_reachable(), "the aclnn/operator table resolved nothing from the loader path");

    MoeRouterEngine router(*device, *device);
    Dsv4Pipeline pipeline(*device, *device, ops, experts, router, config);

    std::unique_ptr<WeightByteSource> source;
#if ASCEND_MOCK_RUNTIME
    source = std::make_unique<SymbolicWeightSource>();
#else
    source = std::make_unique<SyntheticWeightSource>(layout, model.num_hidden_layers, model.n_routed_experts);
#endif
    pipeline.Build(*source);
    experts.Ingest(*source, {});
    experts.ValidateResidency();

    // ---- the one step -----------------------------------------------------
#if ASCEND_MOCK_RUNTIME
    ascend_moe::mock::MockSetD2HSeed(&SeedDeviceToHost);
#endif
    const int32_t prompt_token = 7;
    pipeline.DecodeStep(prompt_token, 0);  // the embedding row for token 7 lands in the arena here
    const int32_t argmax_token = pipeline.ReadArgmaxToken();
#if ASCEND_MOCK_RUNTIME
    ascend_moe::mock::MockSetD2HSeed(nullptr);
#endif

    const StepCounters& counters = pipeline.counters();
    Check(argmax_token >= 0 && argmax_token < kVocabSize,
          "the greedy readback returned a valid token id (" + std::to_string(argmax_token) + ")");
    Check(counters.steps == 1 && counters.layers == kNumLayers,
          "exactly one step covered all " + std::to_string(kNumLayers) + " layers");
    Check(counters.device_allocations_in_step == 0,
          "strictly 0 device allocations inside the step");
    Check(counters.descriptors_built_in_step == 0, "strictly 0 descriptors built inside the step");
    Check(counters.host_synchronizations == kNumLayers + 1,
          "host synchronizations: " + std::to_string(counters.host_synchronizations) + " (the forced one per MoE "
          "layer plus the argmax readback -- " + std::to_string(kNumLayers + 1) + ")");
    Check(counters.launches > 0, "the graph enqueued " + std::to_string(counters.launches) + " launches");
    Check(experts.poisoned() == false, "the exclusive hierarchy never poisoned");
    (void)mock_backend;

    experts.Synchronize();
    experts.ValidateResidency();
    Check(true, "the exclusive residency invariant holds after the step");

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    // Any status error anywhere in build or decode unwinds here: that is the
    // "stream error" the probe exists to catch.
    std::printf("  [FAIL] %s\n", error.what());
    return 1;
  }
}
