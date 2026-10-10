#include "moe/core/acl_guard.hpp"
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

// npu_e2e_single_token -- ONE decode step, end to end, through the whole
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
// real hardware the probe runs the graph over deterministic synthetic weights
// at an explicit --coverage
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

#include "npu_test_support.hpp"

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

}  // namespace

int RunMain(int argc, char** argv) {
  bool require_device = false;
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--require-device") == 0) require_device = true;
  if (!PhysicalNpuPresent()) return require_device ? 1 : 0;
  std::printf("npu_e2e_single_token -- one decode step through the whole engine\n");

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
    std::fprintf(stderr, "[FAIL] detected NPU could not initialize: %s\n", error.what());
    return 1;
  }
  std::unique_ptr<AclDeviceOps> device_guard(device);
  std::printf("  device SoC: %s\n", device->soc_name().c_str());

  try {
    RuntimeConfig config;
    config.synthetic_weights = true;
    config.max_context_len = 256;  // bring-up reservation; the probe runs one position

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
    if (!coverage_given) {
      Skip("real hardware without --coverage: full residency pins ~137 GiB of host RAM, which is a "
           "production decision. Re-run with --coverage <experts> for a bring-up subset.");
      return require_device ? 1 : 0;
    }
    config.routed_coverage = coverage;

    const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);

    ExclusiveExpertManager::Options options;
    options.num_layers = model.num_hidden_layers;
    options.num_experts = model.n_routed_experts;
    options.top_k = model.num_experts_per_tok;
    options.routed_coverage = coverage;
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
    source = std::make_unique<SyntheticWeightSource>(layout, model.num_hidden_layers, model.n_routed_experts);
    pipeline.Build(*source);
    experts.Ingest(*source, {});
    experts.ValidateResidency();

    // ---- the one step -----------------------------------------------------
    const int32_t prompt_token = 7;
    pipeline.DecodeStep(prompt_token, 0);  // the embedding row for token 7 lands in the arena here
    const int32_t argmax_token = pipeline.ReadArgmaxToken();

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

int main(int argc, char** argv) {
  return ascend_moe::GuardedMain([&] { return RunMain(argc, argv); });
}
