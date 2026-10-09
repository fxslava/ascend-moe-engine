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

// mock_pipeline_e2e -- the zero-NPU, zero-allocation contract suite.
//
// Runs natively on an x86 host (WSL) against libopapi_mock: the symbolic
// allocator's interval registry, the shadow descriptor engine, the DeepSeek-V4
// Flash operator contracts in the GetWorkspaceSize stubs, and the FULL
// 43-layer decode graph -- arena, paged KV, exclusive expert hierarchy at the
// complete 11,008-expert coverage -- all as pure address bookkeeping. Physical
// footprint stays at the few pinned mailboxes; the final section reads
// /proc/self/status to prove it.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <unistd.h>
#include <string>
#include <type_traits>
#include <vector>

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"
#include "mock_ops_api.hpp"

#include "moe/core/error.hpp"
#include "moe/core/json.hpp"
#include "moe/core/model_config.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/lattice_moe_block.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/core/weight_source.hpp"

// The dlsym'd mock entry points are ABI-compatible with the product's
// transcribed plan typedefs -- pinned here so a drift in either side is a
// compile error, not a silent wrong-register read.
static_assert(std::is_same<decltype(&aclnnSoftplusGetWorkspaceSize),
                           ascend_moe::SoftplusPlanFn>::value,
              "mock aclnnSoftplus must match SoftplusPlanFn");
static_assert(std::is_same<decltype(&aclnnSqrtGetWorkspaceSize),
                           ascend_moe::UnaryPlanFn>::value,
              "mock aclnnSqrt must match UnaryPlanFn");
static_assert(std::is_same<decltype(&aclnnMoeGatingTopKV2GetWorkspaceSize),
                           ascend_moe::MoeGatingTopKV2PlanFn>::value,
              "mock aclnnMoeGatingTopKV2 must match MoeGatingTopKV2PlanFn");
static_assert(std::is_same<decltype(&aclnnMoeInitRoutingV4GetWorkspaceSize),
                           ascend_moe::MoeInitRoutingV4PlanFn>::value,
              "mock aclnnMoeInitRoutingV4 must match MoeInitRoutingV4PlanFn");
static_assert(std::is_same<decltype(&aclnnGroupedMatmulV5GetWorkspaceSize),
                           ascend_moe::GroupedMatmulV5PlanFn>::value,
              "mock aclnnGroupedMatmulV5 must match GroupedMatmulV5PlanFn");

namespace ascend_moe {
namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (condition) {
    std::printf("  [ ok ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

template <typename Action>
void CheckRefuses(Action action, const std::string& what) {
  ++g_checks;
  try {
    action();
  } catch (const Dsv4Error&) {
    std::printf("  [ ok ] refused: %s\n", what.c_str());
    return;
  } catch (const std::exception& error) {
    ++g_failures;
    std::printf("  [FAIL] %s threw the wrong type: %s\n", what.c_str(), error.what());
    return;
  }
  ++g_failures;
  std::printf("  [FAIL] %s was allowed\n", what.c_str());
}

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

using namespace mock;  // NOLINT(build/namespaces) -- test-local brevity

// ---------------------------------------------------------------------------
// 1. The interval registry
// ---------------------------------------------------------------------------

void TestIntervalRegistry() {
  Section("symbolic allocator: fake addresses and the interval registry");
  MockResetAllocatorForTest();

  const uintptr_t first = MockDeviceMalloc(100);
  const uintptr_t second = MockDeviceMalloc(13369344);
  const uintptr_t host = MockHostMalloc(24);  // a mailbox: real
  Check(first % kMockAllocAlignBytes == 0 && second % kMockAllocAlignBytes == 0 &&
            host % kMockAllocAlignBytes == 0,
        "every base address is 4096-byte aligned (aclrtMalloc HUGE_FIRST parity)");
  Check(first < second, "fake addresses are monotonically increasing");
  Check(MockSpanContains(first, 100), "a registered span contains its own bytes");
  Check(MockSpanContains(first + 4096 - 100, 100) || !MockSpanContains(first + 1, 100),
        "containment is strict: [first, first + size) does not spill past the span end");

  Check(!MockSpanContains(first + 100, 1), "containment is exact: one byte past the 100-byte span is outside");
  Check(!MockSpanContains(second + 13369344, 1), "a byte past the expert slot's span end is outside");
  Check(MockSpanContains(second, 13369344), "a full 13,369,344-byte expert slot fits its span");

  std::string reason;
  Check(MockCheckTransfer(second, first, 100, &reason), "a fully contained transfer passes");
  Check(!MockCheckTransfer(second, first, 200, &reason),
        "an out-of-bounds source interval is refused");
  Check(!MockCheckTransfer(second + 13369344 - 50, first, 100, &reason),
        "an out-of-bounds destination interval is refused");
  const bool overlap_refused = !MockCheckTransfer(second + 50, second, 100, &reason);
  Check(overlap_refused && reason.find("overlap") != std::string::npos,
        "partially overlapping source/destination is refused as non-inplace aliasing: " + reason);
  Check(MockCheckTransfer(first, first, 100, &reason), "an identical in-place interval is legal");

  Check(MockUnregisterSpan(first) && !MockSpanContains(first, 100), "aclrtFree deregisters the span");
  Check(!MockUnregisterSpan(first), "a double free is refused");
  MockUnregisterSpan(second);
  MockUnregisterSpan(host);
  Check(MockMemoryStatistics().rejected_operations > 0, "refusals were counted");
}

// ---------------------------------------------------------------------------
// 2. The shadow descriptor engine
// ---------------------------------------------------------------------------

aclTensor* MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* data) {
  return aclCreateTensor(shape.data(), shape.size(), dtype,
                         nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(), data);
}

void TestDescriptorEngine() {
  Section("shadow descriptor engine: MockAclTensor and aclSetTensorAddr");
  MockResetAllocatorForTest();

  const uintptr_t arena = MockDeviceMalloc(1ull << 20);
  aclTensor* tensor = MakeTensor({1, kNumRoutedExperts}, ACL_FLOAT32, reinterpret_cast<void*>(arena));
  MockAclTensor* meta = AsMockTensor(tensor);
  Check(meta != nullptr && meta->magic == kMockAclTensorMagic, "aclCreateTensor yields a MockAclTensor");
  Check(meta->total_bytes == 1024, "metadata carries the storage size (256 fp32 = 1024 bytes)");
  Check(meta->shape[1] == kNumRoutedExperts, "metadata carries the shape");

  float one = 1.0f;
  float twenty = 20.0f;
  aclScalar* beta = aclCreateScalar(&one, ACL_FLOAT32);
  aclScalar* threshold = aclCreateScalar(&twenty, ACL_FLOAT32);
  Check(AsMockScalar(beta)->as_f64() == 1.0f, "aclCreateScalar keeps fp32 bytes (1.0f)");

  // A small executor to repoint.
  uint64_t workspace = 0;
  aclOpExecutor* executor = nullptr;
  const bool planned = aclnnSoftplusGetWorkspaceSize(tensor, beta, threshold, tensor, &workspace, &executor) == 0 &&
                       workspace > 0 && executor != nullptr;
  Check(planned, "a softplus plan over [1, 256] fp32 succeeds and sizes a workspace");

  Check(aclSetTensorAddr(executor, 0, tensor, reinterpret_cast<void*>(arena + 512)) == 0,
        "aclSetTensorAddr accepts an in-arena address");
  Check(AsMockTensor(tensor)->device_addr == reinterpret_cast<void*>(arena + 512),
        "the repoint is visible through the same handle");
  const bool overruns = aclSetTensorAddr(executor, 0, tensor, reinterpret_cast<void*>(arena + (1 << 20) - 4)) != 0;
  Check(overruns, "aclSetTensorAddr refuses an address whose tensor would overrun the span");
  Check(AsMockTensor(MakeTensor({3}, ACL_FP4X2_E2M1, nullptr))->total_bytes == 2,
        "FP4 packs two E2M1 values per byte");

  aclDestroyTensor(tensor);
  aclDestroyScalar(beta);
  aclDestroyScalar(threshold);
  aclDestroyAclOpExecutor(executor);
  MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 3. The DeepSeek-V4 Flash operator contracts (positive and negative)
// ---------------------------------------------------------------------------

void TestOperatorContracts() {
  Section("operator contracts: sqrtsoftplus scoring, gating, routing, grouped GEMM");
  MockResetAllocatorForTest();
  const uintptr_t arena = MockDeviceMalloc(6ull * 13369344 + (1ull << 22));  // six full expert slots + slack
  uint64_t workspace = 0;
  aclOpExecutor* executor = nullptr;

  // -- softplus / sqrt over [tokens, 256] ------------------------------------
  aclTensor* scores = MakeTensor({4, 256}, ACL_FLOAT32, reinterpret_cast<void*>(arena));
  aclTensor* scores_out = MakeTensor({4, 256}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 4096));
  float one = 1.0f;
  float twenty = 20.0f;
  aclScalar* beta = aclCreateScalar(&one, ACL_FLOAT32);
  aclScalar* threshold = aclCreateScalar(&twenty, ACL_FLOAT32);
  Check(aclnnSoftplusGetWorkspaceSize(scores, beta, threshold, scores_out, &workspace, &executor) == 0,
        "softplus accepts [4, 256] fp32 with beta 1.0 / threshold 20.0");
  aclDestroyAclOpExecutor(executor);
  executor = nullptr;
  Check(aclnnSqrtGetWorkspaceSize(scores, scores_out, &workspace, &executor) == 0, "sqrt accepts the same contract");
  aclDestroyAclOpExecutor(executor);
  executor = nullptr;
  bool refused = aclnnSoftplusGetWorkspaceSize(MakeTensor({4, 128}, ACL_FLOAT32, nullptr), beta, threshold,
                                               scores_out, &workspace, &executor) != 0;
  std::string why = MockLastContractFailure();
  Check(refused && why.find("[tokens, 256]") != std::string::npos, "softplus refuses [4, 128]: " + why);
  executor = nullptr;
  refused = aclnnSqrtGetWorkspaceSize(scores, MakeTensor({4, 255}, ACL_FLOAT32, nullptr), &workspace, &executor) != 0;
  Check(refused, "sqrt refuses a shape mismatch");

  float bad_beta = 2.0f;
  aclScalar* wrong_beta = aclCreateScalar(&bad_beta, ACL_FLOAT32);
  refused = aclnnSoftplusGetWorkspaceSize(scores, wrong_beta, threshold, scores_out, &workspace, &executor) != 0;
  Check(refused, "softplus refuses beta != 1.0");
  executor = nullptr;
  aclDestroyScalar(wrong_beta);

  // -- gating over pre-normalized scores --------------------------------------
  aclTensor* bias = MakeTensor({256}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 8192));
  aclTensor* y = MakeTensor({4, 6}, ACL_FLOAT32, reinterpret_cast<void*>(arena + 12288));
  aclTensor* idx = MakeTensor({4, 6}, ACL_INT32, reinterpret_cast<void*>(arena + 13500));
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, y, idx) == 0,
        "gating accepts k=6, groupCount=1, normType=-1, renorm=1, scaling=1.5 over [4, 256]");
  Check(MockValidateGatingForTest(scores, bias, 8, 1, 1, 0, 1, -1, 1.5, y, idx) != 0,
        "gating refuses k=8: " + MockLastContractFailure());
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 2, 0, 1, -1, 1.5, y, idx) != 0,
        "gating refuses groupCount=2");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 0, -1, 1.5, y, idx) != 0,
        "gating refuses renorm=0 (norm_topk_prob is true)");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, 0, 1.5, y, idx) != 0,
        "gating refuses normType=0 (softmax): the scores arrive pre-normalized");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 2.0, y, idx) != 0,
        "gating refuses routedScalingFactor=2.0");
  Check(MockValidateGatingForTest(MakeTensor({4, 128}, ACL_FLOAT32, nullptr), bias, 6, 1, 1, 0, 1, -1, 1.5, y,
                                  idx) != 0,
        "gating refuses [4, 128] scores");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, MakeTensor({4, 8}, ACL_FLOAT32, nullptr),
                                  idx) != 0,
        "gating refuses yOut [4, 8]");
  Check(MockValidateGatingForTest(scores, bias, 6, 1, 1, 0, 1, -1, 1.5, y,
                                  MakeTensor({4, 6}, ACL_FLOAT32, nullptr)) != 0,
        "gating refuses a float expertIdxOut");

  // -- init routing: the device cumsum groupList --------------------------------
  aclTensor* expert_idx = MakeTensor({4, 6}, ACL_INT32, reinterpret_cast<void*>(arena + 13500));
  aclTensor* group_list = MakeTensor({256}, ACL_INT64, reinterpret_cast<void*>(arena + 20480));
  Check(MockValidateRoutingForTest(expert_idx, 256, group_list) == 0,
        "routing accepts a [256] INT64 cumsum groupList");
  refused = MockValidateRoutingForTest(expert_idx, 256, MakeTensor({255}, ACL_INT64, nullptr)) != 0;
  why = MockLastContractFailure();
  Check(refused && why.find("[expert_num]") != std::string::npos, "routing refuses a [255] groupList: " + why);
  Check(MockValidateRoutingForTest(expert_idx, 256, MakeTensor({256}, ACL_FLOAT32, nullptr)) != 0,
        "routing refuses a non-INT64 groupList");
  // Aliasing: expertIdx bytes and groupList bytes sharing storage.
  aclTensor* aliased = MakeTensor({256}, ACL_INT64, reinterpret_cast<void*>(arena + 13500 + 48));
  Check(MockValidateRoutingForTest(expert_idx, 256, aliased) != 0,
        "routing refuses expertIdx/groupList aliasing");

  // -- grouped matmul: FP4 weights, E8M0 block-32 scales ------------------------
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(layout.slot_num_bytes() == 13369344,
        "the DSV4-Flash expert slot is exactly 13,369,344 bytes (12.75 MiB)");
  const ExpertRegionSpec& gate_up = layout.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = layout.region(ExpertRegionId::kGateUpScale);
  std::vector<aclTensor*> weights;
  std::vector<aclTensor*> scales;
  for (int slot = 0; slot < 6; ++slot) {
    weights.push_back(MakeTensor({gate_up.rows, gate_up.cols}, ACL_FP4X2_E2M1,
                                 reinterpret_cast<void*>(arena + layout.slot_num_bytes() * slot)));
    scales.push_back(MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())},
                                ACL_FLOAT8_E8M0,
                                reinterpret_cast<void*>(arena + layout.slot_num_bytes() * slot +
                                                        gate_up_scale.offset_bytes)));
  }
  aclTensorList* weight_list = aclCreateTensorList(weights.data(), weights.size());
  aclTensorList* scale_list = aclCreateTensorList(scales.data(), scales.size());
  Check(MockValidateGmmForTest(weight_list, scale_list, 3, 0) == 0,
        "grouped matmul accepts FP4 weights + E8M0 scales with splitItem=3, groupType=0");
  Check(MockValidateGmmForTest(weight_list, scale_list, 2, 0) != 0, "grouped matmul refuses splitItem=2");
  Check(MockValidateGmmForTest(weight_list, scale_list, 3, 1) != 0, "grouped matmul refuses groupType=1");

  aclTensor* fp8_weights_raw[] = {MakeTensor({gate_up.rows, gate_up.cols}, ACL_FLOAT8_E4M3FN, nullptr),
                                  MakeTensor({gate_up.rows, gate_up.cols}, ACL_FLOAT8_E4M3FN, nullptr)};
  aclTensorList* fp8_weights = aclCreateTensorList(fp8_weights_raw, 2);
  refused = MockValidateGmmForTest(fp8_weights, scale_list, 3, 0) != 0;
  why = MockLastContractFailure();
  Check(refused && why.find("ACL_FLOAT4_E2M1") != std::string::npos,
        "grouped matmul refuses FP8 weights bound to the FP4 slot layout: " + why);
  aclTensor* fp32_scale_raw[] = {
      MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())}, ACL_FLOAT32, nullptr),
      MakeTensor({gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())}, ACL_FLOAT32, nullptr)};
  aclTensorList* fp32_scales = aclCreateTensorList(fp32_scale_raw, 2);
  Check(MockValidateGmmForTest(weight_list, fp32_scales, 3, 0) != 0, "grouped matmul refuses FP32 scales");

  // The slot-region offsets the expert pool hands the GEMM must tile the slot.
  for (int slot = 0; slot < 6; ++slot) {
    const uintptr_t slot_base = arena + layout.slot_num_bytes() * static_cast<uintptr_t>(slot);
    for (const ExpertRegionSpec& spec : layout.regions()) {
      Check(MockSpanContains(slot_base + spec.offset_bytes, spec.num_bytes()),
            std::string("slot ") + std::to_string(slot) + " region " + ExpertRegionName(spec.id) +
                " sits inside its registered span at the documented offset");
    }
  }

  // Manual 4.16: list destruction also destroys its tensor handles.
  aclDestroyTensorList(weight_list);
  aclDestroyTensorList(scale_list);
  aclDestroyTensorList(fp8_weights);
  aclDestroyTensorList(fp32_scales);
  aclDestroyTensor(scores);
  aclDestroyTensor(scores_out);
  aclDestroyTensor(bias);
  aclDestroyTensor(y);
  aclDestroyTensor(idx);
  aclDestroyTensor(expert_idx);
  aclDestroyTensor(group_list);
  aclDestroyTensor(aliased);
  aclDestroyScalar(beta);
  aclDestroyScalar(threshold);
  MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 4. ModelConfig: config.json parsing and the DSV4 contract asserts
// ---------------------------------------------------------------------------

// The published DeepSeek-V4 Flash configuration, reduced to the fields the
// runner consumes (structure and values preserved, including the nested
// quantization object and the array inside it).
const char* kDsv4FlashConfigJson =
    "{\n"
    "  \"architectures\": [\"DeepseekV4ForCausalLM\"],\n"
    "  \"expert_dtype\": \"fp4\",\n"
    "  \"hidden_size\": 4096,\n"
    "  \"index_n_heads\": 64,\n"
    "  \"index_topk\": 512,\n"
    "  \"moe_intermediate_size\": 2048,\n"
    "  \"n_routed_experts\": 256,\n"
    "  \"n_shared_experts\": 1,\n"
    "  \"norm_topk_prob\": true,\n"
    "  \"num_attention_heads\": 64,\n"
    "  \"num_experts_per_tok\": 6,\n"
    "  \"num_hidden_layers\": 43,\n"
    "  \"q_lora_rank\": 1024,\n"
    "  \"qk_rope_head_dim\": 64,\n"
    "  \"quantization_config\": {\n"
    "    \"activation_scheme\": \"dynamic\",\n"
    "    \"quant_method\": \"fp8\",\n"
    "    \"scale_fmt\": \"ue8m0\",\n"
    "    \"weight_block_size\": [128, 128]\n"
    "  },\n"
    "  \"routed_scaling_factor\": 1.5,\n"
    "  \"rms_norm_eps\": 1e-06,\n"
    "  \"rope_theta\": 10000,\n"
    "  \"scoring_func\": \"sqrtsoftplus\",\n"
    "  \"swiglu_limit\": 10.0,\n"
    "  \"topk_method\": \"noaux_tc\",\n"
    "  \"vocab_size\": 129280\n"
    "}\n";

std::string WithReplaced(const std::string& text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  DSV4_REQUIRE(at != std::string::npos, "test fixture lost its anchor: " << from);
  return text.substr(0, at) + to + text.substr(at + from.size());
}

void TestModelConfig() {
  Section("model config: config.json parsing, router asserts, binary-contract match");
  const ModelConfig model = ModelConfig::FromJsonText(kDsv4FlashConfigJson, "fixture config.json");
  Check(model.num_hidden_layers == 43 && model.n_routed_experts == 256 && model.num_experts_per_tok == 6 &&
            model.hidden_size == 4096 && model.moe_intermediate_size == 2048 &&
            model.routed_scaling_factor == 1.5,
        "the DSV4-Flash topology parses to the documented values");
  Check(model.topk_method == "noaux_tc" && model.scoring_func == "sqrtsoftplus" && model.norm_topk_prob,
        "the three router contract fields parse");
  Check(model.mla.provenance == GeometryProvenance::kCheckpointConfig && model.mla.qk_rope_head_dim == 64 &&
            model.mla.kv_lora_rank == kDefaultKvLoraRank,
        "qk_rope_head_dim comes from the checkpoint; kv_lora_rank keeps the family default "
        "(the checkpoint does not publish it)");
  Check(model.dense_weight_block == 128 && model.scale_fmt == "ue8m0",
        "the nested quantization_config object is read (block 128, ue8m0 scales)");
  model.AssertMatchesBinaryContract();
  Check(true, "the parsed topology matches the compiled graph geometry exactly");

  // Escape and number grammar, on fields that must survive to the assert.
  const std::string escaped = WithReplaced(std::string(kDsv4FlashConfigJson), "\"noaux_tc\"",
                                           "\"noaux_\\u0074c\"");  // \u0074 = 't'
  Check(ModelConfig::FromJsonText(escaped, "escaped").contains("topk_method"),
        "\\uXXXX escapes decode before the contract asserts see them");
  const std::string exponent =
      WithReplaced(std::string(kDsv4FlashConfigJson), "\"routed_scaling_factor\": 1.5",
                   "\"routed_scaling_factor\": 15e-1");
  Check(ModelConfig::FromJsonText(exponent, "exponent").routed_scaling_factor == 1.5,
        "exponent-notation numbers parse");

  // Every refusal names its field; a silently-defaulted wrong router would
  // reach the device.
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"scoring_func\": \"sqrtsoftplus\"",
                                                            "\"scoring_func\": \"sigmoid\""), "t"); },
               "scoring_func sigmoid: the decomposed chain computes sqrt(softplus)");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"topk_method\": \"noaux_tc\"",
                                                            "\"topk_method\": \"group_limited_greedy\""), "t"); },
               "topk_method group_limited_greedy: the gating stage plans no group constraint");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"norm_topk_prob\": true",
                                                            "\"norm_topk_prob\": false"), "t"); },
               "norm_topk_prob false: the gating stage renorm=1");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"num_hidden_layers\": 43", "\"vocab_size\": 129280"),
                                              "t"); },
               "a missing required field (num_hidden_layers) does not silently default");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"num_hidden_layers\": 43",
                                                            "\"num_hidden_layers\": \"43\""), "t"); },
               "a string where a number belongs is a hard type error");
  CheckRefuses([] { ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                            "\"hidden_size\": 4096",
                                                            "\"hidden_size\": 4097"), "t"); },
               "hidden_size 4097 does not tile the block-32 microscale");
  CheckRefuses([] { ModelConfig::FromJsonText("{not json", "t"); }, "a malformed document is a parse error");

  // Topology that is structurally valid but not what this binary compiled
  // for: refused with the divergence named, before any reservation.
  CheckRefuses([] {
    const ModelConfig other = ModelConfig::FromJsonText(WithReplaced(std::string(kDsv4FlashConfigJson),
                                                                     "\"hidden_size\": 4096",
                                                                     "\"hidden_size\": 2048"), "t");
    other.AssertMatchesBinaryContract();
  }, "a structurally valid but divergent topology is refused by the binary-contract check");

  // A default-constructed ModelConfig IS the compiled contract.
  ModelConfig{}.AssertMatchesBinaryContract();
  Check(true, "the default-constructed contract (no config.json anywhere) matches the binary");
}

// ---------------------------------------------------------------------------
// 5. The MoE block seam: slot contracts and the lattice skeleton
// ---------------------------------------------------------------------------

void TestMoeBlockSeam(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config) {
  Section("IRoutedMoeBlock: the slot byte contract and the lattice skeleton");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(Lattice24MoeBlock::SlotBytes() == 7077888,
        "the 2-bit Leech slot is exactly 7,077,888 bytes (6.75 MiB): weights halve, E8M0 scales do not");
  Check(Lattice24MoeBlock::SlotBytes() * 2 > layout.slot_num_bytes() &&
            static_cast<double>(Lattice24MoeBlock::SlotBytes()) / static_cast<double>(layout.slot_num_bytes()) < 0.55,
        "the lattice slot is ~50% of the FP4 slot (weights halve, the 0.75 MiB of scales do not)");

  Lattice24MoeBlock lattice;
  CheckRefuses([&] {
    StaticOpSlotTable table(8, "lattice-test");
    StaticArenaManager probe(allocator, streams, config);
    ExpertSlotAddresses addresses;
    lattice.PlanStages(table, probe, addresses);
  }, "planning the lattice skeleton names the missing aclnnLatticeUnpackAndGroupedMatmul");
  Check(lattice.DescribeBackend().find("SKELETON") != std::string::npos,
        "the lattice backend describes itself as a skeleton");
}

// ---------------------------------------------------------------------------
// 6. The 43-layer decode graph, end to end, at full coverage
// ---------------------------------------------------------------------------

// A weight source that moves no byte: the mock's transfers are interval
// checks, so the graph runs over pure address bookkeeping.
class SymbolicWeightSource : public WeightByteSource {
 public:
  SymbolicWeightSource(int64_t num_layers, int64_t num_experts)
      : num_layers_(num_layers), num_experts_(num_experts) {}

  const char* source_name() const override { return "symbolic"; }
  bool Contains(int32_t layer, int32_t expert) const override {
    return layer >= 0 && layer < num_layers_ && expert >= 0 && expert < num_experts_;
  }
  void ReadExpertSlotRange(uint8_t*, size_t destination_capacity, size_t slot_offset, size_t count, int32_t layer,
                           int32_t expert) override {
    RefuseIfClosed("ReadExpertSlotRange");
    DSV4_REQUIRE(Contains(layer, expert),
                 "expert (layer=" << layer << ", id=" << expert << ") is outside the symbolic source coverage");
    DSV4_REQUIRE(slot_offset + count <= layout_.slot_num_bytes(),
                 "symbolic read leaves the " << layout_.slot_num_bytes() << "-byte slot");
    DSV4_REQUIRE(count <= destination_capacity, "symbolic read exceeds the destination capacity");
    bytes_read_ += count;
    ++read_requests_;
  }
  bool HasNamed(const std::string&) const override { return true; }
  void ReadNamed(const std::string& name, uint8_t*, size_t destination_capacity, size_t byte_offset,
                 size_t count) override {
    RefuseIfClosed("ReadNamed");
    (void)byte_offset;
    DSV4_REQUIRE(count <= destination_capacity, "named read '" << name << "' exceeds the destination capacity");
    bytes_read_ += count;
    ++read_requests_;
  }
  size_t NamedByteSize(const std::string&) const override { return 0; }  // 0 = do not size-check
  void Close() override { closed_ = true; }

 private:
  int64_t num_layers_;
  int64_t num_experts_;
};

// The symbolic oracle for the two forced device-to-host reads: the per-layer
// top-6 expert ids and the greedy token. Without operators that compute, the
// graph's own outputs are fabricated here -- deterministically, in range.
void SeedDeviceToHost(void* destination, size_t count) {
  if (count == sizeof(int32_t) * static_cast<size_t>(kNumExpertsPerTok)) {
    int32_t* ids = static_cast<int32_t*>(destination);
    for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
      ids[index] = static_cast<int32_t>(index);
    }
    return;
  }
  if (count == sizeof(int64_t)) {
    *static_cast<int64_t*>(destination) = 0;  // greedy token id 0
  }
}

size_t ReadResidentKb() {
  std::FILE* file = std::fopen("/proc/self/status", "r");
  if (file == nullptr) {
    return 0;
  }
  char line[256];
  size_t kb = 0;
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    unsigned long long value = 0;
    if (std::sscanf(line, "VmRSS: %llu kB", &value) == 1) {
      kb = static_cast<size_t>(value);
      break;
    }
  }
  std::fclose(file);
  return kb;
}

// The layer topology this test dispatches, as three contiguous runs. Spelled
// once so the expected per-path counters below are arithmetic on these names
// rather than three magic numbers.
constexpr int64_t kSwaLayers = 21;
constexpr int64_t kCsaLayers = 20;
constexpr int64_t kHcaLayers = kNumLayers - kSwaLayers - kCsaLayers;  // 2
// Four decode steps, so position 3 closes one CSA window (ratio 4) and the
// emission half of the cadence is really executed, not just the hold half.
// HCA at ratio 128 cannot close inside a test this short, which is itself
// worth asserting: its layers must hold on every step and write nothing.
constexpr int64_t kDecodeSteps = 4;

void TestFullPipeline() {
  Section("the 43-layer decode graph: build, plan, decode, residency (full 11,008 coverage)");
  MockResetAllocatorForTest();
  MockSetReportedHbm(64ull << 30);

  OpTable ops;
  Check(ops.runtime_reachable(), "the operator table resolves from libopapi_mock via dlsym");
  for (OpId id : {OpId::kSoftplus, OpId::kSqrt, OpId::kMoeGatingTopKV2, OpId::kMoeInitRoutingV4,
                  OpId::kGroupedMatmulV5, OpId::kFusedInferAttentionScoreV5, OpId::kQuantMatmulV5,
                  OpId::kGroupedMatmulSwigluQuantV2}) {
    Check(ops.available(id), std::string("resolved: ") + OpName(id));
  }

  AclDeviceOps device(0);  // every aclrt* call lands in the symbolic mock
  Check(std::string(device.soc_name()).find("Mock") != std::string::npos,
        "the device backend reports the mock SoC");

  RuntimeConfig seam_config;
  seam_config.block_size = 128;
  seam_config.max_context_len = 256;
  TestMoeBlockSeam(device, device, seam_config);

  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  // The ratios matter here: a CSA layer's compressor and indexer projections
  // are part of the backbone the slot planner has to subtract.
  std::vector<int64_t> planning_ratios(static_cast<size_t>(kNumLayers), kCompressRatioSwa);
  for (int64_t layer = kSwaLayers; layer < kSwaLayers + kCsaLayers; ++layer) {
    planning_ratios[static_cast<size_t>(layer)] = kCompressRatioCsa;
  }
  for (int64_t layer = kSwaLayers + kCsaLayers; layer < kNumLayers; ++layer) {
    planning_ratios[static_cast<size_t>(layer)] = kCompressRatioHca;
  }
  const size_t backbone_bytes =
      Dsv4Pipeline::BackboneDeviceBytes(MlaGeometry(), 128, 256, planning_ratios);
  const size_t swa_only_bytes = Dsv4Pipeline::BackboneDeviceBytes(MlaGeometry(), 128, 256);
  Check(backbone_bytes > swa_only_bytes,
        "the compressed schedule enlarges the backbone the slot planner subtracts (" +
            std::to_string((backbone_bytes - swa_only_bytes) >> 20) + " MiB of compressor and indexer "
            "projections), so K is chosen against what will really be resident");
  const int64_t slots = ExclusiveExpertManager::PlanDeviceSlots(
      64ull << 30, backbone_bytes, layout.slot_num_bytes(), kTotalRoutedExperts, kDeviceReserveBytes,
      kTransferChunkBytes, kNumExpertsPerTok, -1);
  Check(slots > kNumExpertsPerTok && slots < kTotalRoutedExperts,
        "the slot planner fits K between one top-k and full residency on the reported 64 GiB");

  ExclusiveExpertManager::Options options;
  options.routed_coverage = kTotalRoutedExperts;  // FULL: 11,008 experts as spans, not bytes
  options.device_slots = slots;
  options.transfer_chunk_bytes = 512 * 1024;  // keeps the pinned transit scratch in the real tier
  options.host_available_bytes = 256ull << 30;  // the host half is symbolic; skip the meminfo guard
  ExclusiveExpertManager experts(device, device, layout, options);
  std::printf("%s", experts.DescribeHierarchy().c_str());
  Check(experts.host_slot_count() + experts.device_slot_count() == kTotalRoutedExperts,
        "|Set_Device| + |Set_Host| = 11,008 with zero physical bytes behind them");

  SymbolicWeightSource source(kNumLayers, kNumRoutedExperts);
  RuntimeConfig config;
  config.synthetic_weights = true;
  config.block_size = 128;
  config.max_context_len = 256;
  // A schedule that exercises ALL THREE attention paths in one graph, laid out
  // in contiguous runs so the per-path counters below are checkable by hand:
  // 21 SWA, then 20 CSA (4:1), then 2 HCA (128:1).
  config.compress_ratios.assign(static_cast<size_t>(kNumLayers), kCompressRatioSwa);
  for (int64_t layer = kSwaLayers; layer < kSwaLayers + kCsaLayers; ++layer) {
    config.compress_ratios[static_cast<size_t>(layer)] = kCompressRatioCsa;
  }
  for (int64_t layer = kSwaLayers + kCsaLayers; layer < kNumLayers; ++layer) {
    config.compress_ratios[static_cast<size_t>(layer)] = kCompressRatioHca;
  }
  config.compress_ratios_provenance = GeometryProvenance::kCommandLine;
  // The router is the runner's shared dependency now: the pipeline and the
  // injected MoE block both consume it.
  MoeRouterEngine router(device, device);
  Dsv4Pipeline pipeline(device, device, ops, experts, router, config);
  pipeline.Build(source);  // reserves, ingests, descriptors, PLANS EVERY STAGE
  experts.Ingest(source, {});

  Check(pipeline.diagnostics().decoded_steps == 0 && !pipeline.diagnostics().ttft_ms &&
            pipeline.diagnostics().paged_attention.total_blocks_allocated == 86 &&
            pipeline.diagnostics().paged_attention.active_context_tokens == 0,
        "planning reserves two KV blocks per layer without claiming decoded tokens or latency");

  Check(pipeline.arena_manager().arena().sealed(), "the arena is sealed after Build");
  Check(pipeline.arena_manager().arena().workspace_bytes() > 0, "the shared workspace holds the plan high-water mark");
  Check(pipeline.moe_block().GetExpertSlotBytes() == layout.slot_num_bytes(),
        "the injected standard block carries the exact 13,369,344-byte FP4/E8M0 slot contract");
  const std::string stage_report = pipeline.DescribeStages();
  Check(stage_report.find("expert_gemm1") != std::string::npos && stage_report.find("expert_combine") != std::string::npos,
        "the MoE block planned its expert stages into the pipeline's shared stage table");
  Check(stage_report.find("GroupedMatmulSwigluQuantV2") != std::string::npos,
        "the fused grouped-GEMM entry point appears in the planned stage list (fused default path)");
  const size_t planned_stages = [&] {
    const std::string report = pipeline.DescribeStages();
    return report.size();  // non-empty iff the stages planned
  }();
  Check(planned_stages > 0, "every stage of the 43-layer graph planned against the mock contracts");
  std::printf("%s", pipeline.DescribeStages().c_str());

  MockSetD2HSeed(&SeedDeviceToHost);
  int32_t token = 7;  // prompt token 7 at position 0
  for (int64_t position = 0; position < kDecodeSteps; ++position) {
    pipeline.DecodeStep(token, position);
    token = pipeline.ReadArgmaxToken();
    Check(token >= 0 && token < kVocabSize,
          "the position-" + std::to_string(position) + " readback returns a valid greedy token id");
  }
  MockSetD2HSeed(nullptr);

  experts.ValidateResidency();
  Check(true, "the exclusive residency invariant holds after every 43-layer step");

  const StepCounters& counters = pipeline.counters();
  Check(counters.steps == static_cast<uint64_t>(kDecodeSteps) &&
            counters.layers == static_cast<uint64_t>(kDecodeSteps * kNumLayers),
        std::to_string(kDecodeSteps) + " decode steps covered all 43 layers each (" +
            std::to_string(kDecodeSteps * kNumLayers) + " layer executions)");
  Check(counters.device_allocations_in_step == 0, "zero device allocations inside a step");
  Check(counters.descriptors_built_in_step == 0, "zero descriptors built inside a step");
  Check(counters.host_synchronizations == static_cast<uint64_t>(kDecodeSteps * (kNumLayers + 1)),
        "exactly the forced one-per-MoE-layer readback plus one per step");

  // ---- mHC: two hyper-connections per layer, every layer, every step ----
  Check(counters.mhc_rounds == static_cast<uint64_t>(2 * kNumLayers * kDecodeSteps),
        "every layer ran TWO mHC rounds per step (attention and MoE), " +
            std::to_string(2 * kNumLayers * kDecodeSteps) + " in total");
  // The point of the fused path: the Sinkhorn runs inside HcPre's vector
  // epilogue, so the decode loop issues NO host plans at all. On the staged
  // fallback this would be one per round -- TestMhcFusedVsStaged measures both.
  Check(counters.sinkhorn_replans == 0,
        "the fused path issues ZERO host plans inside the decode loop: HcPre's kernel normalizes B_l in "
        "registers, so there is no standalone Sinkhorn to re-plan");

  // ---- layer topology: the dispatch really followed compress_ratios ----
  Check(counters.swa_layers == static_cast<uint64_t>(kSwaLayers * kDecodeSteps) &&
            counters.csa_layers == static_cast<uint64_t>(kCsaLayers * kDecodeSteps) &&
            counters.hca_layers == static_cast<uint64_t>(kHcaLayers * kDecodeSteps),
        "the attention cores dispatched per compress_ratios: " + std::to_string(counters.swa_layers) +
            " SWA, " + std::to_string(counters.csa_layers) + " CSA, " + std::to_string(counters.hca_layers) +
            " HCA layer executions");

  // ---- the compressor cadence ------------------------------------------
  // Positions 0..3. A CSA layer (ratio 4) closes its window exactly once, at
  // position 3; an HCA layer (ratio 128) cannot close at all in four steps.
  const int64_t compressed_layers = kCsaLayers + kHcaLayers;
  const int64_t expected_emissions = kCsaLayers;  // one window per CSA layer
  const int64_t expected_holds = compressed_layers * kDecodeSteps - expected_emissions;
  Check(counters.compressor_emissions == static_cast<uint64_t>(expected_emissions),
        "exactly one window closed per CSA layer over " + std::to_string(kDecodeSteps) +
            " steps at ratio 4 (" + std::to_string(counters.compressor_emissions) +
            " emissions), and no HCA window closed at ratio 128");
  Check(counters.compressor_holds == static_cast<uint64_t>(expected_holds),
        "every other compressed-layer step was a HOLD (" + std::to_string(counters.compressor_holds) +
            "): a partial window launches the empty-descriptor plan and writes nothing");
  // The invariant the cadence exists to guarantee: a hold step must never
  // reach the paged cache, so entries written and windows closed are equal.
  Check(counters.compressed_entries_written == counters.compressor_emissions,
        "the paged compressed cache was written exactly once per closed window (" +
            std::to_string(counters.compressed_entries_written) +
            "), so no partial window ever touched it");
  Check(counters.indexer_selections == static_cast<uint64_t>(kCsaLayers * kDecodeSteps),
        "the lightning indexer ran on every CSA layer of every step (" +
            std::to_string(counters.indexer_selections) + " top-512 selections), HCA layers skipping it");

  const std::string decode_report = pipeline.DescribeStages();
  Check(decode_report.find("pure TND") != std::string::npos &&
            decode_report.find("kFusedHcPre") != std::string::npos &&
            decode_report.find("aclnnHcPre") != std::string::npos,
        "the stage report states the mHC layout and that B_l comes from the fused HcPre kernel");
  Check(decode_report.find("sparse_attn_csa") != std::string::npos &&
            decode_report.find("sparse_attn_hca") != std::string::npos &&
            decode_report.find("cmp_hold_csa") == std::string::npos,
        "both compressed cores are planned without a zero-row HOLD executor");
  Check(counters.expert_slot_misses > 0 && counters.expert_slot_hits > 0,
        "the seeded routing produced both hits and misses across the swap engine");
  const auto& diag = pipeline.diagnostics();
  Check(diag.decoded_steps == static_cast<uint64_t>(kDecodeSteps) &&
            diag.moe_cache.total_expert_requests ==
                static_cast<uint64_t>(kDecodeSteps * kNumLayers * kNumExpertsPerTok) &&
            diag.moe_cache.hbm_slot_hits == counters.expert_slot_hits &&
            diag.moe_cache.host_promotions == counters.expert_slot_misses &&
            diag.moe_cache.evictions_to_host == diag.moe_cache.host_promotions,
        "DecodeStep snapshots exact cache telemetry for every layers-of-experts sweep");
  Check(diag.paged_attention.active_context_tokens == static_cast<uint64_t>(kDecodeSteps) &&
            diag.paged_attention.block_size == 128 &&
            diag.paged_attention.kv_cache_utilization == static_cast<double>(kDecodeSteps) / 256.0 &&
            !diag.attention.attention_entropy && diag.attention.sparsity_ratio == 0.0,
        "KV utilization reflects active tokens; neither core reports an entropy or sparsity it never "
        "measured");

  const MockMemoryStats& stats = MockMemoryStatistics();
  Check(stats.rejected_operations == 0,
        "ZERO rejected operations: every DMA, memset and SetTensorAddr stayed inside a registered span");
  Check(stats.memcpy_checks > 0 && stats.setaddr_checks > 0,
        "the interval registry actually exercised transfers (" + std::to_string(stats.memcpy_checks) +
            " memcpy checks, " + std::to_string(stats.setaddr_checks) + " SetTensorAddr checks)");
  Check(stats.symbolic_device_bytes > 45ull << 30,
        "the symbolic device tier alone carries the K-slot pool and the arena (" +
            std::to_string(stats.symbolic_device_bytes >> 30) + " GiB of spans; the host half adds " +
            std::to_string(((static_cast<uint64_t>(experts.host_slot_count()) * layout.slot_num_bytes()) >> 30)) +
            " GiB more, all at zero physical cost)");
  std::printf("  slot-map index mismatches (derive-and-verify tally): %" PRIu64 "\n", stats.slot_map_mismatches);
  const size_t rss_kb = ReadResidentKb();
  Check(rss_kb < 512 * 1024, "physical RSS stayed under 512 MiB (read " + std::to_string(rss_kb) + " kB)");
}

// ---------------------------------------------------------------------------
// The two ways B_l can be produced, measured against each other
// ---------------------------------------------------------------------------
//
// This is the test that justifies the default. Both modes compute the same
// doubly-stochastic residual map; they differ in what that costs per decode
// step, and the difference is the whole reason the fused kernel is preferred:
//
//   kFusedHcPre         2 launches per round, 0 host plans
//   kStagedMhcSinkhorn  3 launches per round, 1 host plan per round
//
// It also keeps the staged fallback from rotting. It is documented as the
// escape hatch if the fused kernel needs its own bring-up, so it has to stay
// dispatchable, not merely compilable.
void TestMhcMixingModes() {
  Section("mHC mixing: the fused HcPre kernel against the staged standalone Sinkhorn");
  MockResetAllocatorForTest();
  MockSetReportedHbm(64ull << 30);

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Check(false, "the operator table resolves from libopapi_mock");
    return;
  }
  for (OpId id : {OpId::kHcPre, OpId::kHcPost, OpId::kMhcPre, OpId::kMhcSinkhorn, OpId::kMhcPost}) {
    Check(ops.available(id), std::string("resolved: ") + OpName(id));
  }

  AclDeviceOps device(0);
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();

  struct Measured {
    uint64_t launches = 0;
    uint64_t replans = 0;
    uint64_t rounds = 0;
    size_t stages = 0;
  };
  const auto measure = [&](MhcMixingMode mode) {
    ExclusiveExpertManager::Options options;
    // Full coverage: the key space is flat over layer x expert, so a subset
    // would refuse the first layer outside it. The spans are symbolic, so the
    // only real cost is the small resident slot pool.
    options.routed_coverage = kTotalRoutedExperts;
    options.device_slots = 64;
    options.transfer_chunk_bytes = 512 * 1024;
    options.host_available_bytes = 256ull << 30;
    ExclusiveExpertManager experts(device, device, layout, options);

    SymbolicWeightSource source(kNumLayers, kNumRoutedExperts);
    RuntimeConfig config;
    config.synthetic_weights = true;
    config.block_size = 128;
    config.max_context_len = 256;
    config.mhc_mixing_mode = mode;
    MoeRouterEngine router(device, device);
    Dsv4Pipeline pipeline(device, device, ops, experts, router, config);
    pipeline.Build(source);
    experts.Ingest(source, {});

    MockSetD2HSeed(&SeedDeviceToHost);
    pipeline.DecodeStep(7, 0);
    (void)pipeline.ReadArgmaxToken();
    MockSetD2HSeed(nullptr);

    Measured out;
    out.launches = pipeline.counters().launches;
    out.replans = pipeline.counters().sinkhorn_replans;
    out.rounds = pipeline.counters().mhc_rounds;
    // The LABELLED line, not a bare mention: the staged report also names
    // kFusedHcPre when it explains why that mode is the default, so a plain
    // substring search would match both.
    out.stages = pipeline.DescribeStages().find("B_l SOURCE:   kFusedHcPre") != std::string::npos ? 1 : 0;
    return out;
  };

  const Measured fused = measure(MhcMixingMode::kFusedHcPre);
  const Measured staged = measure(MhcMixingMode::kStagedMhcSinkhorn);

  const uint64_t rounds = static_cast<uint64_t>(2 * kNumLayers);
  Check(fused.rounds == rounds && staged.rounds == rounds,
        "both modes run the same " + std::to_string(rounds) + " mHC rounds per step, so the comparison "
        "below is like for like");
  Check(fused.replans == 0,
        "kFusedHcPre issues NO host plan inside the decode loop: the Sinkhorn is interior to HcPre's kernel");
  Check(staged.replans == rounds,
        "kStagedMhcSinkhorn pays exactly one host plan per round (" + std::to_string(staged.replans) +
            "), because aclnnMhcSinkhorn has no repeatable form");
  Check(staged.launches == fused.launches + rounds,
        "the staged path costs one EXTRA launch per round (" + std::to_string(fused.launches) + " fused vs " +
            std::to_string(staged.launches) + " staged): HcPre folds the normalization into the launch that "
            "produces hIn, instead of dispatching it separately");
  Check(fused.stages == 1 && staged.stages == 0,
        "each mode's stage report names the B_l source it actually used");
}

void TestDiagnosticsJson() {
  Section("structured diagnostics: null measurements, counters, finite JSON, write failures");
  struct TempFile {
    char path[64] = "/tmp/dsv4-diag-XXXXXX";
    TempFile() {
      const int fd = ::mkstemp(path);
      DSV4_REQUIRE(fd >= 0, "cannot create diagnostics fixture");
      ::close(fd);
    }
    ~TempFile() { ::unlink(path); }
  } file;
  auto read = [&] {
    std::ifstream input(file.path);
    std::ostringstream text;
    text << input.rdbuf();
    return moe_json::Parse(text.str());
  };
  InferenceDiagnostics diag;
  Check(diag.moe_cache.HitRate() == 0.0, "zero requests have a defined zero hit rate");
  diag.DumpJson(file.path);
  auto json = read();
  Check(json.find("ttft_ms")->is_null() && json.find("tpot_ms")->is_null() &&
            json.find("perplexity")->is_null() &&
            json.find("attention")->find("attention_entropy")->is_null(), "unmeasured values serialize as null");
  diag.moe_cache = {12, 3, 9, 9};
  diag.ttft_ms = 1.25;
  diag.tpot_ms = 0.5;
  diag.perplexity = std::numeric_limits<double>::infinity();
  diag.attention.attention_entropy = std::numeric_limits<double>::quiet_NaN();
  diag.DumpJson(file.path);
  json = read();
  Check(json.find("ttft_ms")->as_number() == 1.25 && json.find("tpot_ms")->as_number() == 0.5 &&
            json.find("moe_cache")->find("hit_rate")->as_number() == 0.25 &&
            json.find("perplexity")->is_null() &&
            json.find("attention")->find("attention_entropy")->is_null(),
        "measured numbers are preserved; non-finite values never produce invalid JSON");
  bool refused = false;
  try { diag.DumpJson(std::string(file.path) + "/unwritable.json"); }
  catch (const Dsv4Error&) { refused = true; }
  Check(refused, "a failed diagnostics export is reported");
}

}  // namespace
}  // namespace ascend_moe

int RunMain() {
  using namespace ascend_moe;
  std::printf("mock_pipeline_e2e -- zero-NPU, zero-allocation contract suite over libopapi_mock\n");
  try {
    TestIntervalRegistry();
    TestDescriptorEngine();
    TestOperatorContracts();
    TestModelConfig();
    TestFullPipeline();
    TestMhcMixingModes();
    TestDiagnosticsJson();
  } catch (const std::exception& error) {
    std::printf("\nunexpected exception: %s\n", error.what());
    ++g_failures;
  }
  const ascend_moe::mock::MockMemoryStats& stats = ascend_moe::mock::MockMemoryStatistics();
  std::printf("\nmock runtime tally: %" PRIu64 " device spans (%" PRIu64 " GiB symbolic), %" PRIu64
              " real host bytes,\n  %" PRIu64 " memcpy checks, %" PRIu64 " setaddr checks, %" PRIu64
              " refusals, %" PRIu64 " slot-map notes\n",
              stats.device_allocations, stats.symbolic_device_bytes >> 30, stats.real_host_bytes,
              stats.memcpy_checks, stats.setaddr_checks, stats.rejected_operations, stats.slot_map_mismatches);
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}

int main() {
  return ascend_moe::GuardedMain([&] { return RunMain(); });
}
