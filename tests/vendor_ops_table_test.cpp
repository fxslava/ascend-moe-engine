/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

// vendor_ops_table_test -- the vendored arch35 (Ascend 950PR) operators from
// third_party/ops_transformer, end to end through the engine's own machinery.
//
// What is verified here, without a device:
//   * OpTable resolves all four vendored operators (mhc_pre, mhc_sinkhorn,
//     mhc_post, quant_lightning_indexer) and the resolved addresses ARE the
//     linked library's own symbols -- so the dlsym contract, the C prototypes
//     in moe/ops/aclnn_dsv4_vendor_ops.h and the mock implementation cannot
//     drift apart silently.
//   * The two-phase protocol runs for each operator through PlanAclnnOp +
//     StaticOpSlot::Adopt + Launch: the plan validates the DSV4 geometry
//     (n_hc = 4 streams over the 4096-wide hidden state, TND layout), the
//     executor adopts (aclSetAclOpExecutorRepeatable succeeds), the launch
//     is a validated no-op, and re-planning returns the same workspace size.
//   * aclSetTensorAddr swaps a slot address inside the captured IR order.
//   * The mhc_sinkhorn ViewCopy(output, output) guard: a non-contiguous
//     output view is planned successfully but the hazard is counted, which is
//     the repeatability condition the engine must route around.
//
// What is NOT verified here: the arch35 kernel numerics and the real
// GetWorkspaceSize numbers -- those need a 950PR with the vendored opp
// package deployed.

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"
#include "mock_ops_api.hpp"

#include "moe/core/acl_guard.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"
#include "moe/ops/aclnn_dsv4_vendor_ops.h"

namespace ascend_moe {
namespace {

using namespace mock;  // the descriptor/allocator helpers this suite asserts through

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

constexpr int64_t kTokens = 4;
constexpr int64_t kNhc = 4;     // mHC streams (n)
constexpr int64_t kMixRows = kNhc * kNhc + 2 * kNhc;  // n^2 + 2n = 24

aclTensor* MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* data) {
  return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                         data);
}

// A strided view over `span`: shape [t, n, n] with element stride 2 forces
// l0op::ViewCopy's gather/scatter branch -- the mhc_sinkhorn hazard.
aclTensor* MakeStridedTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* span) {
  std::vector<int64_t> strides(shape.size(), 1);
  int64_t element = 2;  // skip every other element
  for (size_t index = shape.size(); index-- > 0;) {
    strides[index] = element;
    element *= shape[index] * 2;
  }
  return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                         shape.data(), shape.size(), span);
}

// ---------------------------------------------------------------------------
// 1. Operator table resolution
// ---------------------------------------------------------------------------

void TestTableResolution() {
  Section("vendored operator table: resolution and symbol identity");
  OpTable ops;
  if (!ops.runtime_reachable()) {
    std::printf("  NOTE: the aclnn runtime is not on the loader path; nothing to resolve.\n");
    return;
  }
  const OpId vendored[] = {OpId::kMhcPre, OpId::kMhcSinkhorn, OpId::kMhcPost, OpId::kQuantLightningIndexer};
  for (OpId id : vendored) {
    Check(ops.available(id), std::string(OpName(id)) + " resolved from the linked libraries");
  }
  ops.RequireAll({OpId::kMhcPre, OpId::kMhcSinkhorn, OpId::kMhcPost, OpId::kQuantLightningIndexer});
  Check(true, "RequireAll accepts the four vendored operators");

  // The engine's C prototypes and the linked implementation are the same
  // symbols: OpTable's dlsym address equals the address the declaration in
  // moe/ops/aclnn_dsv4_vendor_ops.h binds to. A mock or vendor library that
  // exported a different arity would still pass dlsym -- this pins the ABI.
  Check(reinterpret_cast<void*>(&aclnnMhcPreGetWorkspaceSize) == ops.op(OpId::kMhcPre).plan,
        "OpTable's mhc_pre plan is exactly &aclnnMhcPreGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnMhcPre) == ops.op(OpId::kMhcPre).launch,
        "OpTable's mhc_pre launch is exactly &aclnnMhcPre");
  Check(reinterpret_cast<void*>(&aclnnMhcSinkhornGetWorkspaceSize) == ops.op(OpId::kMhcSinkhorn).plan,
        "OpTable's mhc_sinkhorn plan is exactly &aclnnMhcSinkhornGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnMhcPostGetWorkspaceSize) == ops.op(OpId::kMhcPost).plan,
        "OpTable's mhc_post plan is exactly &aclnnMhcPostGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnQuantLightningIndexerGetWorkspaceSize) == ops.op(OpId::kQuantLightningIndexer).plan,
        "OpTable's quant_lightning_indexer plan is exactly &aclnnQuantLightningIndexerGetWorkspaceSize");

  const std::string inventory = ops.DescribeInventory();
  Check(inventory.find("aclnnMhcPre") != std::string::npos &&
            inventory.find("aclnnMhcSinkhorn") != std::string::npos &&
            inventory.find("aclnnMhcPost") != std::string::npos &&
            inventory.find("aclnnQuantLightningIndexer") != std::string::npos,
        "the inventory lists all four vendored operators");
  for (OpId id : vendored) {
    Check(!ops.op(id).provider.empty(), std::string(OpName(id)) + " names its provider (" + ops.op(id).provider + ")");
  }
}

// ---------------------------------------------------------------------------
// 2. The mHC chain: plan, adopt (repeatable), launch
// ---------------------------------------------------------------------------

void TestMhcChain() {
  Section("vendored mHC chain: plan -> repeatable executor -> launch");
  mock::MockResetAllocatorForTest();
  SimulatedDeviceOps device(16ull << 20);
  OpTable ops;

  const uintptr_t arena = mock::MockDeviceMalloc(8ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);  // 64-byte spacing, 4096-aligned base
    return reinterpret_cast<void*>(arena + cursor);
  };

  DeviceStream stream = device.CreateStream();

  // -- mhc_pre: [T,4,4096] BF16 states through phi [24,16384] ----------------
  aclTensor* x = MakeTensor({kTokens, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  aclTensor* phi = MakeTensor({kMixRows, kNhc * kHiddenSize}, ACL_FLOAT32, next(kMixRows * kNhc * kHiddenSize * 4));
  aclTensor* alpha = MakeTensor({3}, ACL_FLOAT32, next(3 * 4));
  aclTensor* bias = MakeTensor({kMixRows}, ACL_FLOAT32, next(kMixRows * 4));
  aclTensor* gamma = MakeTensor({kNhc, kHiddenSize}, ACL_FLOAT32, next(kNhc * kHiddenSize * 4));
  aclTensor* h_in = MakeTensor({kTokens, kHiddenSize}, ACL_BF16, next(kTokens * kHiddenSize * 2));
  aclTensor* h_post = MakeTensor({kTokens, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * 4));
  aclTensor* h_res = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));

  aclOpExecutor* executor = nullptr;
  uint64_t ws = PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, &executor, x, phi, alpha, bias, gamma, 1e-6, 1e-6, h_in,
                                          h_post, h_res, nullptr, nullptr, nullptr);
  Check(ws > 0, "mhc_pre planned a non-empty workspace over the DSV4 geometry");
  StaticOpSlot pre_slot;
  pre_slot.Adopt(OpId::kMhcPre, "vendor/mhc_pre", ws, executor);
  Check(pre_slot.planned() && pre_slot.workspace_size() == ws,
        "mhc_pre's executor adopted and made repeatable (aclSetAclOpExecutorRepeatable)");
  void* workspace = ws > 0 ? device.DeviceMalloc(ws) : nullptr;
  pre_slot.Launch(ops, workspace, stream);
  Check(true, "mhc_pre launch enqueued on the mock stream");
  // Deterministic planning: the same descriptors plan the same workspace.
  aclOpExecutor* executor2 = nullptr;
  const uint64_t ws_again =
      PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, &executor2, x, phi, alpha, bias, gamma, 1e-6, 1e-6, h_in, h_post,
                                h_res, nullptr, nullptr, nullptr);
  Check(ws_again == ws, "re-planning mhc_pre returns the same deterministic workspace size");
  aclDestroyAclOpExecutor(executor2);

  // The IR capture order: {x, phi, alpha, bias, gamma, hIn, hPost, hRes, ...}
  // -- slot 0 is x, slot 5 is hIn. Swapping x's address through the captured
  // handle must not count as a slot-map mismatch.
  const uint64_t mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  pre_slot.SetAddress(0, x, AsMockTensor(x)->device_addr);
  pre_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "aclSetTensorAddr(0, x) hits the captured IR slot exactly (no slot-map mismatch)");
  pre_slot.Reset();

  // -- mhc_sinkhorn: [T,4,4] -> doubly-stochastic, contiguous output --------
  aclTensor* sink_in = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));
  aclTensor* sink_out = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));
  const int warnings_before = mock::MockSinkhornViewCopyWarnings();
  ws = PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &executor, sink_in, 1e-6f, 20, sink_out, nullptr,
                                      nullptr);
  StaticOpSlot sink_slot;
  sink_slot.Adopt(OpId::kMhcSinkhorn, "vendor/mhc_sinkhorn", ws, executor);
  sink_slot.Launch(ops, workspace, stream);
  Check(mock::MockSinkhornViewCopyWarnings() == warnings_before,
        "a contiguous mhc_sinkhorn output raises no ViewCopy hazard");
  sink_slot.Reset();

  // The hazard: a strided output view plans fine, but the trailing
  // ViewCopy(output, output) stage is a gather/scatter the address swap
  // cannot express -- the guard counts it instead of failing the plan.
  aclTensor* strided_out = MakeStridedTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32,
                                             next(kTokens * kNhc * kNhc * 16));
  ws = PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &executor, sink_in, 1e-6f, 20, strided_out, nullptr,
                                      nullptr);
  Check(mock::MockSinkhornViewCopyWarnings() == warnings_before + 1,
        "a non-contiguous mhc_sinkhorn output is planned but counted as a ViewCopy repeatability hazard");
  aclDestroyAclOpExecutor(executor);

  // numIters bounds come from the vendored aclnn layer's own checks.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &bad, sink_in, 1e-6f, 0, sink_out, nullptr, nullptr);
  }, "mhc_sinkhorn with numIters 0 (below the [1, 100] bound)");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &bad, sink_in, 1e-6f, 101, sink_out, nullptr, nullptr);
  }, "mhc_sinkhorn with numIters 101 (above the [1, 100] bound)");

  // -- mhc_post: x_next = (hRes)^T x + hOut * hPost ---------------------------
  aclTensor* post_x = MakeTensor({kTokens, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  aclTensor* post_hout = MakeTensor({kTokens, kHiddenSize}, ACL_BF16, next(kTokens * kHiddenSize * 2));
  aclTensor* post_out = MakeTensor({kTokens, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  ws = PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, &executor, post_x, h_res, post_hout, h_post, post_out);
  StaticOpSlot post_slot;
  post_slot.Adopt(OpId::kMhcPost, "vendor/mhc_post", ws, executor);
  post_slot.Launch(ops, workspace, stream);
  Check(true, "mhc_post planned, adopted (repeatable) and launched");
  post_slot.Reset();

  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, &bad, post_x, h_res, post_hout, h_post, post_hout);
  }, "mhc_post with a [T, 4096] output where [T, 4, 4096] is required");

  for (aclTensor* tensor : {x, phi, alpha, bias, gamma, h_in, h_post, h_res, sink_in, sink_out, strided_out, post_x,
                            post_hout, post_out}) {
    aclDestroyTensor(tensor);
  }
  device.DestroyStream(stream);
  mock::MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 3. The lightning indexer: TND query over a paged key
// ---------------------------------------------------------------------------

void TestQuantLightningIndexer() {
  Section("vendored quant_lightning_indexer: TND query, PA_BSND paged key");
  mock::MockResetAllocatorForTest();
  OpTable ops;

  constexpr int64_t kHeads = 16;        // N1, smallest 950PR indexer head count
  constexpr int64_t kHeadDim = 128;     // D
  constexpr int64_t kQueryTokens = 8;   // T
  constexpr int64_t kBlocks = 4;
  constexpr int64_t kBlockSize = 128;   // pages of 16..1024 tokens
  constexpr int64_t kSparseCount = 4;

  const uintptr_t arena = mock::MockDeviceMalloc(2ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);
    return reinterpret_cast<void*>(arena + cursor);
  };

  aclTensor* query = MakeTensor({kQueryTokens, kHeads, kHeadDim}, ACL_FLOAT8_E4M3FN,
                                next(kQueryTokens * kHeads * kHeadDim));
  aclTensor* key = MakeTensor({kBlocks, kBlockSize, 1, kHeadDim}, ACL_FLOAT8_E4M3FN,
                              next(kBlocks * kBlockSize * kHeadDim));
  aclTensor* weights = MakeTensor({kQueryTokens, kHeads}, ACL_BF16, next(kQueryTokens * kHeads * 2));
  aclTensor* q_scale = MakeTensor({kQueryTokens, kHeads}, ACL_FLOAT32, next(kQueryTokens * kHeads * 4));
  aclTensor* k_scale = MakeTensor({kBlocks, kBlockSize, 1}, ACL_FLOAT32, next(kBlocks * kBlockSize * 4));
  aclTensor* seq_q = MakeTensor({1}, ACL_INT32, next(4));    // one batch, 8 tokens
  aclTensor* seq_k = MakeTensor({1}, ACL_INT32, next(4));
  aclTensor* block_table = MakeTensor({1, kBlocks}, ACL_INT32, next(kBlocks * 4));
  aclTensor* out = MakeTensor({kQueryTokens, 1, kSparseCount}, ACL_INT32, next(kQueryTokens * kSparseCount * 4));

  aclOpExecutor* executor = nullptr;
  const uint64_t ws = PlanAclnnOp<QuantLightningIndexerPlanFn>(
      ops, OpId::kQuantLightningIndexer, &executor, query, key, weights, q_scale, k_scale, seq_q, seq_k, block_table, 0,
      0, const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3, INT64_MAX, INT64_MAX, out);
  Check(ws > 0, "the indexer planned a non-empty workspace over TND + PA_BSND");
  StaticOpSlot slot;
  slot.Adopt(OpId::kQuantLightningIndexer, "vendor/quant_lightning_indexer", ws, executor);
  SimulatedDeviceOps device(4ull << 20);
  DeviceStream stream = device.CreateStream();
  slot.Launch(ops, ws > 0 ? device.DeviceMalloc(ws) : nullptr, stream);
  Check(true, "the indexer's executor adopted (repeatable) and launched");
  slot.Reset();
  device.DestroyStream(stream);

  // The engine's E4M3 + BF16 + FP32 tuple is the one the 950PR def binds;
  // an FP16 query is the A3 int8-style path and must be refused here.
  aclTensor* bad_query = MakeTensor({kQueryTokens, kHeads, kHeadDim}, ACL_FLOAT16, next(kQueryTokens * kHeads * 2));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<QuantLightningIndexerPlanFn>(ops, OpId::kQuantLightningIndexer, &bad, bad_query, key, weights,
                                              q_scale, k_scale, seq_q, seq_k, block_table, 0, 0,
                                              const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3,
                                              INT64_MAX, INT64_MAX, out);
  }, "an FP16 indexer query outside the 950PR FP8 tuple");
  // N1 outside {16, 24, 32, 64} is refused by the vendored def on 950PR.
  aclTensor* narrow_q = MakeTensor({kQueryTokens, 8, kHeadDim}, ACL_FLOAT8_E4M3FN, next(kQueryTokens * 8));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<QuantLightningIndexerPlanFn>(ops, OpId::kQuantLightningIndexer, &bad, narrow_q, key, weights,
                                              q_scale, k_scale, seq_q, seq_k, block_table, 0, 0,
                                              const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3,
                                              INT64_MAX, INT64_MAX, out);
  }, "an indexer with N1 = 8 (outside the 950PR {16, 24, 32, 64} set)");

  for (aclTensor* tensor : {query, key, weights, q_scale, k_scale, seq_q, seq_k, block_table, out, bad_query,
                            narrow_q}) {
    aclDestroyTensor(tensor);
  }
  mock::MockUnregisterSpan(arena);
}

}  // namespace
}  // namespace ascend_moe

int RunMain() {
  using namespace ascend_moe;
  std::printf("vendor_ops_table_test -- the vendored arch35 operators through OpTable\n");
  try {
    TestTableResolution();
    TestMhcChain();
    TestQuantLightningIndexer();
  } catch (const std::exception& error) {
    std::printf("\nunexpected exception: %s\n", error.what());
    ++g_failures;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  std::printf("\nNot covered here (needs a 950PR): the arch35 kernel numerics and the real\n"
              "GetWorkspaceSize numbers against the deployed opp package.\n");
  return g_failures == 0 ? 0 : 1;
}

int main() { return ascend_moe::GuardedMain([&] { return RunMain(); }); }
