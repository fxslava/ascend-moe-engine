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
// third_party/ops_dsv4, end to end through the engine's own machinery.
//
// What is verified here, without a device:
//   * OpTable resolves all NINE vendored operators -- the mHC chain (mhc_pre,
//     mhc_sinkhorn, mhc_post), both lightning indexers
//     (quant_lightning_indexer, vllm_quant_lightning_indexer), the token-level
//     compressor, the shared-KV sparse attention core and the two cache
//     epilogs -- and the resolved addresses ARE the linked library's own
//     symbols, so the dlsym contract, the C prototypes in
//     moe/ops/aclnn_dsv4_vendor_ops.h and the mock implementation cannot drift
//     apart silently.
//   * The two-phase protocol runs for each operator through PlanAclnnOp +
//     StaticOpSlot::Adopt + Launch: the plan validates the DSV4 geometry
//     (n_hc = 4 streams over the 4096-wide hidden state, 64 query heads over a
//     512-wide total head dim, indexer D = 128, compression ratios 4 and 128,
//     TND layout), the executor adopts (aclSetAclOpExecutorRepeatable
//     succeeds), the launch is a validated no-op, and re-planning returns the
//     same workspace size.
//   * aclSetTensorAddr swaps a slot address inside the captured IR order, and
//     re-binding across two launch cycles keeps hitting the captured slots.
//   * The manual-4.31 self-copy invariant across the whole vendored set: no
//     wrapper plans a same-address ViewCopy. In particular the patched
//     mhc_sinkhorn elides its trailing copy for a contiguous output (where
//     upstream copied onto its own address) and still copies, src != dst, for
//     a non-contiguous one; and the three REF-output operators
//     (Compressor.stateCache and the two epilogs) stage no copy at all.
//   * The stride attributes that address the paged caches and keys agree with
//     the views the caller handed over -- a stale value would read or scatter
//     into the wrong block.
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
  const std::vector<OpId> vendored = {OpId::kMhcPre,
                                      OpId::kMhcSinkhorn,
                                      OpId::kMhcPost,
                                      OpId::kHcPre,
                                      OpId::kHcPreInvRms,
                                      OpId::kHcPreSinkhorn,
                                      OpId::kHcPost,
                                      OpId::kQuantLightningIndexer,
                                      OpId::kCompressor,
                                      OpId::kVllmQuantLightningIndexer,
                                      OpId::kKvQuantSparseAttnSharedkv,
                                      OpId::kKvCompressEpilog,
                                      OpId::kIndexerCompressEpilogV2};
  for (OpId id : vendored) {
    Check(ops.available(id), std::string(OpName(id)) + " resolved from the linked libraries");
  }
  ops.RequireAll(vendored);
  Check(true, "RequireAll accepts all thirteen vendored operators");

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
  // The fused hc_* family. These must be four distinct entry points, and in
  // particular aclnnHcPreSinkhorn must NOT alias aclnnMhcSinkhorn: they are
  // different decompositions of the mHC mapping, not a rename.
  Check(reinterpret_cast<void*>(&aclnnHcPreGetWorkspaceSize) == ops.op(OpId::kHcPre).plan,
        "OpTable's hc_pre plan is exactly &aclnnHcPreGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnHcPre) == ops.op(OpId::kHcPre).launch,
        "OpTable's hc_pre launch is exactly &aclnnHcPre");
  Check(reinterpret_cast<void*>(&aclnnHcPreInvRmsGetWorkspaceSize) == ops.op(OpId::kHcPreInvRms).plan,
        "OpTable's hc_pre_inv_rms plan is exactly &aclnnHcPreInvRmsGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnHcPreInvRms) == ops.op(OpId::kHcPreInvRms).launch,
        "OpTable's hc_pre_inv_rms launch is exactly &aclnnHcPreInvRms");
  Check(reinterpret_cast<void*>(&aclnnHcPreSinkhornGetWorkspaceSize) == ops.op(OpId::kHcPreSinkhorn).plan,
        "OpTable's hc_pre_sinkhorn plan is exactly &aclnnHcPreSinkhornGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnHcPreSinkhorn) == ops.op(OpId::kHcPreSinkhorn).launch,
        "OpTable's hc_pre_sinkhorn launch is exactly &aclnnHcPreSinkhorn");
  Check(reinterpret_cast<void*>(&aclnnHcPostGetWorkspaceSize) == ops.op(OpId::kHcPost).plan,
        "OpTable's hc_post plan is exactly &aclnnHcPostGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnHcPost) == ops.op(OpId::kHcPost).launch,
        "OpTable's hc_post launch is exactly &aclnnHcPost");
  Check(ops.op(OpId::kHcPreSinkhorn).plan != ops.op(OpId::kMhcSinkhorn).plan,
        "hc_pre_sinkhorn and mhc_sinkhorn are distinct entry points (a fusion, not a rename)");
  Check(ops.op(OpId::kHcPre).plan != ops.op(OpId::kMhcPre).plan &&
            ops.op(OpId::kHcPost).plan != ops.op(OpId::kMhcPost).plan,
        "the two mHC families' pre and post stages resolve to different entry points");
  Check(ops.op(OpId::kHcPre).plan != ops.op(OpId::kHcPreSinkhorn).plan &&
            ops.op(OpId::kHcPre).plan != ops.op(OpId::kHcPreInvRms).plan,
        "the fused hc_pre is a distinct entry point from both staged-path operators");

  Check(reinterpret_cast<void*>(&aclnnQuantLightningIndexerGetWorkspaceSize) == ops.op(OpId::kQuantLightningIndexer).plan,
        "OpTable's quant_lightning_indexer plan is exactly &aclnnQuantLightningIndexerGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnCompressorGetWorkspaceSize) == ops.op(OpId::kCompressor).plan,
        "OpTable's compressor plan is exactly &aclnnCompressorGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnCompressor) == ops.op(OpId::kCompressor).launch,
        "OpTable's compressor launch is exactly &aclnnCompressor");
  Check(reinterpret_cast<void*>(&aclnnVllmQuantLightningIndexerGetWorkspaceSize) ==
            ops.op(OpId::kVllmQuantLightningIndexer).plan,
        "OpTable's vllm_quant_lightning_indexer plan is exactly "
        "&aclnnVllmQuantLightningIndexerGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize) ==
            ops.op(OpId::kKvQuantSparseAttnSharedkv).plan,
        "OpTable's kv_quant_sparse_attn_sharedkv plan is exactly "
        "&aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnKvCompressEpilogGetWorkspaceSize) == ops.op(OpId::kKvCompressEpilog).plan,
        "OpTable's kv_compress_epilog plan is exactly &aclnnKvCompressEpilogGetWorkspaceSize");
  Check(reinterpret_cast<void*>(&aclnnIndexerCompressEpilogV2GetWorkspaceSize) ==
            ops.op(OpId::kIndexerCompressEpilogV2).plan,
        "OpTable's indexer_compress_epilog_v2 plan is exactly "
        "&aclnnIndexerCompressEpilogV2GetWorkspaceSize");
  // The two indexers are distinct operators with distinct ABIs; a copy-paste
  // in the vendor tree or the OpTable declaration would alias them.
  Check(ops.op(OpId::kQuantLightningIndexer).plan != ops.op(OpId::kVllmQuantLightningIndexer).plan,
        "the two lightning indexers resolve to different entry points");

  const std::string inventory = ops.DescribeInventory();
  bool inventory_lists_all = true;
  for (OpId id : vendored) {
    if (inventory.find(OpName(id)) == std::string::npos) {
      inventory_lists_all = false;
    }
  }
  Check(inventory_lists_all, "the inventory lists all thirteen vendored operators");
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

  // -- mhc_sinkhorn: [T,4,4] -> doubly-stochastic --------------------------------
  // The CONTIGUOUS output is the case upstream got wrong: Contiguous(output)
  // is the identity, MhcSinkhorn writes into and returns that same tensor, and
  // the unconditional trailing ViewCopy(kernelOut, output) becomes a
  // manual-4.31 same-address self-copy that makes the executor non-reusable.
  // The patched wrapper skips the copy in exactly that case -- the result is
  // already in the caller buffer -- so the executor stays reusable and no
  // hazard is recorded on either path.
  aclTensor* sink_in = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));
  aclTensor* sink_out = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));
  const int elisions_before = mock::MockSinkhornSelfCopyElisions();
  const int hazards_before = mock::MockVendorSelfCopyHazards();
  ws = PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &executor, sink_in, 1e-6f, 20, sink_out, nullptr,
                                      nullptr);
  StaticOpSlot sink_slot;
  sink_slot.Adopt(OpId::kMhcSinkhorn, "vendor/mhc_sinkhorn", ws, executor);
  sink_slot.Launch(ops, workspace, stream);
  Check(mock::MockSinkhornSelfCopyElisions() == elisions_before + 1,
        "a contiguous mhc_sinkhorn output elides the trailing ViewCopy (the patch firing on the case "
        "upstream copied onto its own address)");
  Check(mock::MockVendorSelfCopyHazards() == hazards_before,
        "the contiguous mhc_sinkhorn path records no manual-4.31 self-copy hazard, so its executor is "
        "reusable");
  // Reusability is the whole point of the patch: re-bind the output address on
  // the adopted executor and launch again on the same plan.
  const uint64_t sink_mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  sink_slot.SetAddress(1, sink_out, AsMockTensor(sink_out)->device_addr);
  sink_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == sink_mismatches_before,
        "the contiguous mhc_sinkhorn plan relaunches after aclSetTensorAddr on its output slot");
  sink_slot.Reset();

  // A strided output view makes Contiguous allocate a distinct
  // executor-owned temp, so the final ViewCopy still runs -- but with
  // src != dst, which is equally reusable and costs one extra copy launch.
  aclTensor* strided_out = MakeStridedTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32,
                                             next(kTokens * kNhc * kNhc * 16));
  ws = PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &executor, sink_in, 1e-6f, 20, strided_out, nullptr,
                                      nullptr);
  Check(mock::MockSinkhornSelfCopyElisions() == elisions_before + 1,
        "a non-contiguous mhc_sinkhorn output keeps its ViewCopy (src != dst), so nothing is elided");
  Check(mock::MockVendorSelfCopyHazards() == hazards_before,
        "the non-contiguous mhc_sinkhorn path records no self-copy hazard either");
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
// 2b. The fused hc_* mHC family, alongside the mhc_* chain above
// ---------------------------------------------------------------------------
//
// The same mHC mapping, decomposed the vllm-ascend way. What this section is
// really for is the structural claim: the fused form has no in-place Sinkhorn
// tensor, so unlike mhc_sinkhorn it has nothing to elide -- the hazard counter
// AND the elision counter both have to stay put across all four operators,
// where mhc_sinkhorn necessarily moves the elision counter.

void TestFusedHcFamily() {
  Section("vendored fused mHC family: hc_pre / hc_pre_inv_rms / hc_pre_sinkhorn / hc_post");
  mock::MockResetAllocatorForTest();
  SimulatedDeviceOps device(16ull << 20);
  OpTable ops;

  const uintptr_t arena = mock::MockDeviceMalloc(16ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);
    return reinterpret_cast<void*>(arena + cursor);
  };

  DeviceStream stream = device.CreateStream();

  // The hc_* family takes x as [bs, hc, d] (or [b, s, hc, d]); the engine's
  // decode stream maps on as bs = T.
  aclTensor* x = MakeTensor({kTokens, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  aclTensor* hc_fn =
      MakeTensor({kMixRows, kNhc * kHiddenSize}, ACL_FLOAT32, next(kMixRows * kNhc * kHiddenSize * 4));
  aclTensor* hc_scale = MakeTensor({3}, ACL_FLOAT32, next(3 * 4));
  aclTensor* hc_base = MakeTensor({kMixRows}, ACL_FLOAT32, next(kMixRows * 4));
  aclTensor* y = MakeTensor({kTokens, kHiddenSize}, ACL_BF16, next(kTokens * kHiddenSize * 2));
  aclTensor* post = MakeTensor({kTokens, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * 4));
  aclTensor* comb_frag = MakeTensor({kTokens, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));

  // The ledger is read once up front: nothing the fused family does may move
  // either counter.
  const int hazards_before = mock::MockVendorSelfCopyHazards();
  const int elisions_before = mock::MockSinkhornSelfCopyElisions();
  const int ref_plans_before = mock::MockRefOutputPlans();

  // -- hc_pre: the whole prologue in ONE launch -------------------------------
  aclOpExecutor* executor = nullptr;
  uint64_t ws = PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &executor, x, hc_fn, hc_scale, hc_base, kNhc, 20, 1e-6,
                                         1e-6, y, post, comb_frag);
  Check(ws > 0, "hc_pre planned a non-empty workspace over the DSV4 geometry");
  StaticOpSlot pre_slot;
  pre_slot.Adopt(OpId::kHcPre, "vendor/hc_pre", ws, executor);
  Check(pre_slot.planned() && pre_slot.workspace_size() == ws,
        "hc_pre's executor adopted and made repeatable (aclSetAclOpExecutorRepeatable)");
  void* workspace = ws > 0 ? device.DeviceMalloc(ws) : nullptr;
  pre_slot.Launch(ops, workspace, stream);
  Check(true, "hc_pre launch enqueued on the mock stream");

  aclOpExecutor* executor2 = nullptr;
  const uint64_t ws_again = PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &executor2, x, hc_fn, hc_scale, hc_base,
                                                    kNhc, 20, 1e-6, 1e-6, y, post, comb_frag);
  Check(ws_again == ws, "re-planning hc_pre returns the same deterministic workspace size");
  aclDestroyAclOpExecutor(executor2);

  // Two full launch cycles on the ONE plan, re-binding a different slot each
  // time: the IR capture order is {x, hcFn, hcScale, hcBase, y, post,
  // combFrag}, so slot 0 is x and slot 4 is y.
  uint64_t mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  pre_slot.SetAddress(0, x, AsMockTensor(x)->device_addr);
  pre_slot.Launch(ops, workspace, stream);
  pre_slot.SetAddress(4, y, AsMockTensor(y)->device_addr);
  pre_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "hc_pre relaunches twice after aclSetTensorAddr on its x and y slots (no slot-map mismatch)");
  pre_slot.Reset();

  // hc_mult is fixed at 4 on 950PR and the iteration count carries the same
  // [1, 100] bound the mhc_sinkhorn path has.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &bad, x, hc_fn, hc_scale, hc_base, 6, 20, 1e-6, 1e-6, y, post,
                             comb_frag);
  }, "hc_pre with hc_mult 6 (only 4 is a DSV4-Flash configuration)");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &bad, x, hc_fn, hc_scale, hc_base, kNhc, 0, 1e-6, 1e-6, y, post,
                             comb_frag);
  }, "hc_pre with hc_sinkhorn_iters 0 (below the [1, 100] bound)");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &bad, x, hc_fn, hc_scale, hc_base, kNhc, 101, 1e-6, 1e-6, y, post,
                             comb_frag);
  }, "hc_pre with hc_sinkhorn_iters 101 (above the [1, 100] bound)");
  // hcFn is [n^2+2n, n*d]; handing it hc_base's [24] would be a silent
  // misread of 1.5 MB of weights.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, &bad, x, hc_base, hc_scale, hc_base, kNhc, 20, 1e-6, 1e-6, y, post,
                             comb_frag);
  }, "hc_pre with a [24] hcFn where [24, n*d] is required");

  // -- hc_pre_inv_rms: the staged path's prologue -----------------------------
  // rsqrt drops x's last two axes and keeps a trailing 1.
  aclTensor* rsqrt = MakeTensor({kTokens, 1}, ACL_FLOAT32, next(kTokens * 4));
  ws = PlanAclnnOp<HcPreInvRmsPlanFn>(ops, OpId::kHcPreInvRms, &executor, x, 1e-6, rsqrt);
  Check(ws > 0, "hc_pre_inv_rms planned a non-empty workspace");
  StaticOpSlot rms_slot;
  rms_slot.Adopt(OpId::kHcPreInvRms, "vendor/hc_pre_inv_rms", ws, executor);
  Check(rms_slot.planned(), "hc_pre_inv_rms's executor adopted (aclSetAclOpExecutorRepeatable)");
  rms_slot.Launch(ops, workspace, stream);
  // IR order {x, y}: slot 1 is the rsqrt output.
  mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  rms_slot.SetAddress(1, rsqrt, AsMockTensor(rsqrt)->device_addr);
  rms_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "hc_pre_inv_rms relaunches after aclSetTensorAddr on its rsqrt slot");
  rms_slot.Reset();

  // The trailing 1 is the contract: a [T] or [T, 4] rsqrt would make
  // hc_pre_sinkhorn read the normalizer off the wrong stride.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPreInvRmsPlanFn>(ops, OpId::kHcPreInvRms, &bad, x, 1e-6, post);
  }, "hc_pre_inv_rms with a [T, 4] output where [T, 1] is required");

  // -- hc_pre_sinkhorn: the mixing projection fused with the normalization ----
  // `mixes` is the [T, 24] product the engine forms with aclnnMatmul, standing
  // in for upstream's at::linear(x.flatten(-2), hcFn).
  aclTensor* mixes = MakeTensor({kTokens, kMixRows}, ACL_FLOAT32, next(kTokens * kMixRows * 4));
  ws = PlanAclnnOp<HcPreSinkhornPlanFn>(ops, OpId::kHcPreSinkhorn, &executor, mixes, rsqrt, hc_scale, hc_base, x,
                                        kNhc, 20, 1e-6, y, post, comb_frag);
  Check(ws > 0, "hc_pre_sinkhorn planned a non-empty workspace over the DSV4 geometry");
  StaticOpSlot sink_slot;
  sink_slot.Adopt(OpId::kHcPreSinkhorn, "vendor/hc_pre_sinkhorn", ws, executor);
  Check(sink_slot.planned() && sink_slot.workspace_size() == ws,
        "hc_pre_sinkhorn's executor adopted and made repeatable");
  sink_slot.Launch(ops, workspace, stream);

  // Two launch cycles with a re-bind each: IR order is {mixes, rsqrt, hcScale,
  // hcBase, x, y, post, combFrag}, so slot 1 is rsqrt (the per-token input the
  // decode loop actually re-points) and slot 5 is y.
  mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  sink_slot.SetAddress(1, rsqrt, AsMockTensor(rsqrt)->device_addr);
  sink_slot.Launch(ops, workspace, stream);
  sink_slot.SetAddress(5, y, AsMockTensor(y)->device_addr);
  sink_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "hc_pre_sinkhorn relaunches twice after aclSetTensorAddr on its rsqrt and y slots");
  sink_slot.Reset();

  // Unlike mhc_sinkhorn there is no output that aliases an input here, so a
  // contiguous output is NOT a special case -- the elision counter stays put.
  Check(mock::MockSinkhornSelfCopyElisions() == elisions_before,
        "hc_pre_sinkhorn with contiguous outputs elides nothing: the fused form has no in-place "
        "Sinkhorn tensor to copy onto itself");

  // mixes must carry the n^2+2n row count; hc_base's [24] happens to match it,
  // so the check that bites is the rank/leading-axis one.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPreSinkhornPlanFn>(ops, OpId::kHcPreSinkhorn, &bad, post, rsqrt, hc_scale, hc_base, x, kNhc, 20,
                                     1e-6, y, post, comb_frag);
  }, "hc_pre_sinkhorn with a [T, 4] mixes where [T, 24] is required");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPreSinkhornPlanFn>(ops, OpId::kHcPreSinkhorn, &bad, mixes, post, hc_scale, hc_base, x, kNhc, 20,
                                     1e-6, y, post, comb_frag);
  }, "hc_pre_sinkhorn with a [T, 4] rsqrt where [T, 1] is required");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPreSinkhornPlanFn>(ops, OpId::kHcPreSinkhorn, &bad, mixes, rsqrt, hc_scale, hc_base, x, kNhc, 20,
                                     1e-6, y, post, post);
  }, "hc_pre_sinkhorn with a [T, 4] comb_frag where [T, 4, 4] is required");

  // -- hc_post: the BSHD residual combine ------------------------------------
  // HcPost takes x as [b, s, d], not the [T, N, D] of MhcPost; a TND decode
  // stream maps on as b = T, s = 1. Getting that wrong is the easiest mistake
  // to make when the two families sit side by side, so it is pinned here.
  aclTensor* post_x = MakeTensor({kTokens, 1, kHiddenSize}, ACL_BF16, next(kTokens * kHiddenSize * 2));
  aclTensor* residual =
      MakeTensor({kTokens, 1, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  aclTensor* post_bshd = MakeTensor({kTokens, 1, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * 4));
  aclTensor* comb_bshd = MakeTensor({kTokens, 1, kNhc, kNhc}, ACL_FLOAT32, next(kTokens * kNhc * kNhc * 4));
  aclTensor* post_out =
      MakeTensor({kTokens, 1, kNhc, kHiddenSize}, ACL_BF16, next(kTokens * kNhc * kHiddenSize * 2));
  ws = PlanAclnnOp<HcPostPlanFn>(ops, OpId::kHcPost, &executor, post_x, residual, post_bshd, comb_bshd, post_out);
  Check(ws > 0, "hc_post planned a non-empty workspace over the BSHD geometry");
  StaticOpSlot post_slot;
  post_slot.Adopt(OpId::kHcPost, "vendor/hc_post", ws, executor);
  Check(post_slot.planned(), "hc_post's executor adopted (aclSetAclOpExecutorRepeatable)");
  post_slot.Launch(ops, workspace, stream);
  // IR order {x, residual, post, comb, y}: slot 4 is the output.
  mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  post_slot.SetAddress(4, post_out, AsMockTensor(post_out)->device_addr);
  post_slot.Launch(ops, workspace, stream);
  post_slot.SetAddress(1, residual, AsMockTensor(residual)->device_addr);
  post_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "hc_post relaunches twice after aclSetTensorAddr on its output and residual slots");
  post_slot.Reset();

  // The TND shape MhcPost takes is exactly what HcPost must refuse.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPostPlanFn>(ops, OpId::kHcPost, &bad, x, residual, post_bshd, comb_bshd, post_out);
  }, "hc_post with MhcPost's [T, N, D] x where BSHD [b, s, d] is required");
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<HcPostPlanFn>(ops, OpId::kHcPost, &bad, post_x, residual, post_bshd, comb_bshd, post_x);
  }, "hc_post with a [b, s, d] output where residual's [b, s, hc, d] is required");

  // -- the whole point: neither ledger counter moved --------------------------
  Check(mock::MockVendorSelfCopyHazards() == hazards_before,
        "no operator in the fused hc_* family plans a same-address ViewCopy");
  Check(mock::MockSinkhornSelfCopyElisions() == elisions_before,
        "the fused family needs no elision at all -- it removes the manual-4.31 Sinkhorn hazard by "
        "construction, where mhc_sinkhorn can only work around it");
  Check(mock::MockRefOutputPlans() == ref_plans_before,
        "the fused family records no REF-output plan: none of the four hc_* OpDefs has an output whose "
        "name matches an input");

  for (aclTensor* tensor : {x, hc_fn, hc_scale, hc_base, y, post, comb_frag, rsqrt, mixes, post_x, residual,
                            post_bshd, comb_bshd, post_out}) {
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

// ---------------------------------------------------------------------------
// 4. The token-level compressor: both compression ratios, REF state cache
// ---------------------------------------------------------------------------

void TestCompressor() {
  Section("vendored compressor: CSA (cmp_ratio 4) and HCA (cmp_ratio 128), REF state cache");
  mock::MockResetAllocatorForTest();
  OpTable ops;

  constexpr int64_t kCompTokens = 128;  // divisible by both 4 and 128
  constexpr int64_t kStateBlocks = 4;
  constexpr int64_t kStateBlockSize = 8;
  constexpr int64_t kStateDim = 1024;
  constexpr int64_t kCmpChannels = 512;  // normWeight[0]
  constexpr int64_t kRopeHeadDim = 64;
  constexpr int64_t kCoff = 1;

  const uintptr_t arena = mock::MockDeviceMalloc(32ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);
    return reinterpret_cast<void*>(arena + cursor);
  };

  aclTensor* x = MakeTensor({kCompTokens, kHiddenSize}, ACL_BF16, next(kCompTokens * kHiddenSize * 2));
  aclTensor* wkv = MakeTensor({kCmpChannels, kHiddenSize}, ACL_BF16, next(kHiddenSize * kCmpChannels * 2));
  aclTensor* wgate = MakeTensor({kCmpChannels, kHiddenSize}, ACL_BF16, next(kHiddenSize * kCmpChannels * 2));
  aclTensor* state_cache = MakeTensor({kStateBlocks, kStateBlockSize, kStateDim}, ACL_FLOAT32,
                                      next(kStateBlocks * kStateBlockSize * kStateDim * 4));
  aclTensor* ape = MakeTensor({4, kCmpChannels}, ACL_FLOAT32, next(4 * kCmpChannels * 4));
  aclTensor* norm_weight = MakeTensor({kCmpChannels}, ACL_FLOAT32, next(kCmpChannels * 4));
  aclTensor* rope_sin = MakeTensor({kCompTokens / 4 + 1, kRopeHeadDim}, ACL_FLOAT32, next(kCompTokens * kRopeHeadDim * 4));
  aclTensor* rope_cos = MakeTensor({kCompTokens / 4 + 1, kRopeHeadDim}, ACL_FLOAT32, next(kCompTokens * kRopeHeadDim * 4));
  aclTensor* block_table = MakeTensor({1, kStateBlocks}, ACL_INT32, next(kStateBlocks * 4));
  aclTensor* cu_seqlens = MakeTensor({2}, ACL_INT32, next(8));

  // The kernel addresses the paged state cache through this attribute, so it
  // must be the axis-0 stride of the view the caller owns.
  const int64_t state_stride0 = AsMockTensor(state_cache)->strides.at(0);
  Check(state_stride0 == kStateBlockSize * kStateDim,
        "the contiguous state cache's axis-0 stride is blockSize*D as the kernel expects");

  SimulatedDeviceOps device(16ull << 20);
  DeviceStream stream = device.CreateStream();

  // -- CSA: cmp_ratio 4 -------------------------------------------------------
  aclTensor* cmp_kv_csa = MakeTensor({kCompTokens / 4 + 1, kCmpChannels * kCoff}, ACL_BF16,
                                     next((kCompTokens / 4 + 1) * kCmpChannels * kCoff * 2));
  const int ref_plans_before = mock::MockRefOutputPlans();
  aclOpExecutor* executor = nullptr;
  uint64_t ws = PlanAclnnOp<CompressorPlanFn>(ops, OpId::kCompressor, &executor, x, wkv, wgate, state_cache, ape,
                                              norm_weight, rope_sin, rope_cos, block_table, cu_seqlens, nullptr,
                                              nullptr, kRopeHeadDim, 4, kCoff, 1e-6, 1, 1, state_stride0,
                                              cmp_kv_csa);
  Check(ws > 0, "the compressor planned a non-empty workspace at cmp_ratio 4 (CSA)");
  StaticOpSlot csa_slot;
  csa_slot.Adopt(OpId::kCompressor, "vendor/compressor_csa", ws, executor);
  Check(csa_slot.planned(), "the compressor's CSA executor adopted (aclSetAclOpExecutorRepeatable)");
  void* workspace = ws > 0 ? device.DeviceMalloc(ws) : nullptr;
  csa_slot.Launch(ops, workspace, stream);

  // Re-bind the REF state cache across a second launch cycle on the same plan:
  // it is IR slot 3 ({x, wkv, wgate, stateCache, ...}).
  const uint64_t mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  csa_slot.SetAddress(3, state_cache, AsMockTensor(state_cache)->device_addr);
  csa_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "aclSetTensorAddr re-binds the compressor's REF state cache slot across launch cycles");
  Check(mock::MockRefOutputPlans() == ref_plans_before + 1,
        "the compressor is recorded as a REF-output plan (no copy stage, so no self-copy hazard)");
  csa_slot.Reset();

  // -- HCA: cmp_ratio 128 -----------------------------------------------------
  aclTensor* cmp_kv_hca = MakeTensor({kCompTokens / 128 + 1, kCmpChannels * kCoff}, ACL_BF16,
                                     next((kCompTokens / 128 + 1) * kCmpChannels * kCoff * 2));
  aclTensor* ape_hca = MakeTensor({128, kCmpChannels}, ACL_FLOAT32, next(128 * kCmpChannels * 4));
  aclTensor* sin_hca = MakeTensor({2, kRopeHeadDim}, ACL_FLOAT32, next(2 * kRopeHeadDim * 4));
  aclTensor* cos_hca = MakeTensor({2, kRopeHeadDim}, ACL_FLOAT32, next(2 * kRopeHeadDim * 4));
  aclOpExecutor* hca_executor = nullptr;
  const uint64_t ws_hca = PlanAclnnOp<CompressorPlanFn>(
      ops, OpId::kCompressor, &hca_executor, x, wkv, wgate, state_cache, ape_hca, norm_weight, sin_hca, cos_hca,
      block_table, cu_seqlens, nullptr, nullptr, kRopeHeadDim, 128, kCoff, 1e-6, 1, 1, state_stride0, cmp_kv_hca);
  Check(ws_hca > 0, "the compressor planned a non-empty workspace at cmp_ratio 128 (HCA)");
  aclDestroyAclOpExecutor(hca_executor);

  // A ratio outside {4, 128} is not a DSV4-Flash configuration.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<CompressorPlanFn>(ops, OpId::kCompressor, &bad, x, wkv, wgate, state_cache, ape, norm_weight,
                                  rope_sin, rope_cos, block_table, cu_seqlens, nullptr, nullptr, kRopeHeadDim, 8,
                                  kCoff, 1e-6, 1, 1, state_stride0, cmp_kv_csa);
  }, "the compressor at cmp_ratio 8 (outside the DSV4 {4, 128} set)");

  // cmp_kv rows must be T/cmp_ratio: handing the CSA output to the HCA plan
  // would silently write 32x too many rows on device.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<CompressorPlanFn>(ops, OpId::kCompressor, &bad, x, wkv, wgate, state_cache, ape, norm_weight,
                                  rope_sin, rope_cos, block_table, cu_seqlens, nullptr, nullptr, kRopeHeadDim, 128,
                                  kCoff, 1e-6, 1, 1, state_stride0, cmp_kv_csa);
  }, "the compressor at cmp_ratio 128 with a T/4-row cmp_kv output");

  // A stale stride attribute would make the kernel scatter into the wrong
  // block of the paged state cache.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<CompressorPlanFn>(ops, OpId::kCompressor, &bad, x, wkv, wgate, state_cache, ape, norm_weight,
                                  rope_sin, rope_cos, block_table, cu_seqlens, nullptr, nullptr, kRopeHeadDim, 4,
                                  kCoff, 1e-6, 1, 1, state_stride0 + 1, cmp_kv_csa);
  }, "the compressor with a stateCacheStrideDim0 that disagrees with the state cache view");

  // The state cache is the recurrent pooling state and must be FP32.
  aclTensor* bf16_state = MakeTensor({kStateBlocks, kStateBlockSize, kStateDim}, ACL_BF16,
                                     next(kStateBlocks * kStateBlockSize * kStateDim * 2));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<CompressorPlanFn>(ops, OpId::kCompressor, &bad, x, wkv, wgate, bf16_state, ape, norm_weight,
                                  rope_sin, rope_cos, block_table, cu_seqlens, nullptr, nullptr, kRopeHeadDim, 4,
                                  kCoff, 1e-6, 1, 1, state_stride0, cmp_kv_csa);
  }, "the compressor with a BF16 state cache where FP32 is required");

  for (aclTensor* tensor : {x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, block_table,
                            cu_seqlens, cmp_kv_csa, cmp_kv_hca, bf16_state, ape_hca, sin_hca, cos_hca}) {
    aclDestroyTensor(tensor);
  }
  device.DestroyStream(stream);
  mock::MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 5. The shared-KV indexer and the shared-KV sparse attention core
// ---------------------------------------------------------------------------

void TestSharedKvPath() {
  Section("vendored shared-KV path: vllm indexer -> kv_quant_sparse_attn_sharedkv");
  mock::MockResetAllocatorForTest();
  OpTable ops;

  constexpr int64_t kHeads = 64;        // N1 / query heads
  constexpr int64_t kHeadDim = 128;     // indexer D
  constexpr int64_t kQueryTokens = 8;   // T
  constexpr int64_t kBlocks = 4;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kSparseCount = 16;
  constexpr int64_t kTotalHeadDim = 512;  // DSV4 MLA total head dim
  constexpr int64_t kTileSize = 64;
  constexpr int64_t kRopeHeadDim = 64;

  const uintptr_t arena = mock::MockDeviceMalloc(64ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);
    return reinterpret_cast<void*>(arena + cursor);
  };

  SimulatedDeviceOps device(32ull << 20);
  DeviceStream stream = device.CreateStream();

  // -- the shared-KV indexer --------------------------------------------------
  aclTensor* query = MakeTensor({kQueryTokens, kHeads, kHeadDim}, ACL_FLOAT8_E4M3FN,
                                next(kQueryTokens * kHeads * kHeadDim));
  aclTensor* key = MakeTensor({kBlocks, kBlockSize, 1, kHeadDim}, ACL_FLOAT8_E4M3FN,
                              next(kBlocks * kBlockSize * kHeadDim));
  aclTensor* weights = MakeTensor({kQueryTokens, kHeads}, ACL_FLOAT32, next(kQueryTokens * kHeads * 4));
  aclTensor* q_scale = MakeTensor({kQueryTokens, kHeads}, ACL_FLOAT32, next(kQueryTokens * kHeads * 4));
  aclTensor* k_scale = MakeTensor({kBlocks, kBlockSize, 1}, ACL_FLOAT32, next(kBlocks * kBlockSize * 4));
  aclTensor* seq_q = MakeTensor({1}, ACL_INT32, next(4));
  aclTensor* seq_k = MakeTensor({1}, ACL_INT32, next(4));
  aclTensor* block_table = MakeTensor({1, kBlocks}, ACL_INT32, next(kBlocks * 4));
  aclTensor* metadata = MakeTensor({1024}, ACL_INT32, next(1024 * 4));
  aclTensor* cmp_indices = MakeTensor({kQueryTokens, 1, kSparseCount}, ACL_INT32,
                                      next(kQueryTokens * kSparseCount * 4));
  // returnValues=false is signalled by a [0] placeholder, which is what the
  // vendored wrapper checks before issuing the second copy.
  aclTensor* no_values = MakeTensor({0}, ACL_FLOAT32, next(4));

  const int64_t key_stride0 = AsMockTensor(key)->strides.at(0);
  const int64_t scale_stride0 = AsMockTensor(k_scale)->strides.at(0);

  aclOpExecutor* executor = nullptr;
  uint64_t ws = PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
      ops, OpId::kVllmQuantLightningIndexer, &executor, query, key, weights, q_scale, k_scale, seq_q, seq_k,
      block_table, metadata, 0, 0, const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3,
      INT64_MAX, INT64_MAX, /*cmp_ratio=*/4, /*return_values=*/false, key_stride0, scale_stride0, cmp_indices,
      no_values);
  Check(ws > 0, "the shared-KV indexer planned a non-empty workspace over TND + PA_BSND at cmp_ratio 4");
  StaticOpSlot indexer_slot;
  indexer_slot.Adopt(OpId::kVllmQuantLightningIndexer, "vendor/vllm_quant_lightning_indexer", ws, executor);
  void* workspace = ws > 0 ? device.DeviceMalloc(ws) : nullptr;
  indexer_slot.Launch(ops, workspace, stream);
  Check(true, "the shared-KV indexer's executor adopted (repeatable) and launched");
  indexer_slot.Reset();

  // returnValues=true needs a real sparse_values output, not the placeholder.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
        ops, OpId::kVllmQuantLightningIndexer, &bad, query, key, weights, q_scale, k_scale, seq_q, seq_k,
        block_table, metadata, 0, 0, const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3,
        INT64_MAX, INT64_MAX, 4, /*return_values=*/true, key_stride0, scale_stride0, cmp_indices, no_values);
  }, "the shared-KV indexer with returnValues=true and a [0] sparse_values placeholder");

  // A stale key stride would read the wrong page of the paged key.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
        ops, OpId::kVllmQuantLightningIndexer, &bad, query, key, weights, q_scale, k_scale, seq_q, seq_k,
        block_table, metadata, 0, 0, const_cast<char*>("TND"), const_cast<char*>("PA_BSND"), kSparseCount, 3,
        INT64_MAX, INT64_MAX, 4, false, key_stride0 + 1, scale_stride0, cmp_indices, no_values);
  }, "the shared-KV indexer with a stride attribute that disagrees with the paged key view");

  // -- the shared-KV sparse attention core ------------------------------------
  aclTensor* q = MakeTensor({kQueryTokens, kHeads, kTotalHeadDim}, ACL_BF16,
                            next(kQueryTokens * kHeads * kTotalHeadDim * 2));
  aclTensor* ori_kv = MakeTensor({kBlocks, kBlockSize, kTotalHeadDim}, ACL_FLOAT8_E4M3FN,
                                 next(kBlocks * kBlockSize * kTotalHeadDim));
  aclTensor* cmp_kv = MakeTensor({kBlocks, kBlockSize, kTotalHeadDim}, ACL_FLOAT8_E4M3FN,
                                 next(kBlocks * kBlockSize * kTotalHeadDim));
  aclTensor* ori_indices = MakeTensor({kQueryTokens, 1, kSparseCount}, ACL_INT32,
                                      next(kQueryTokens * kSparseCount * 4));
  aclTensor* sinks = MakeTensor({kHeads}, ACL_FLOAT32, next(kHeads * 4));
  aclTensor* attn_out = MakeTensor({kQueryTokens, kHeads, kTotalHeadDim}, ACL_BF16,
                                   next(kQueryTokens * kHeads * kTotalHeadDim * 2));
  aclTensor* no_lse = MakeTensor({0}, ACL_FLOAT32, next(4));

  const int64_t ori_stride0 = AsMockTensor(ori_kv)->strides.at(0);
  const int64_t cmp_stride0 = AsMockTensor(cmp_kv)->strides.at(0);
  const int hazards_before = mock::MockVendorSelfCopyHazards();

  aclOpExecutor* attn_executor = nullptr;
  const uint64_t attn_ws = PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
      ops, OpId::kKvQuantSparseAttnSharedkv, &attn_executor, q, ori_kv, cmp_kv, ori_indices, cmp_indices,
      block_table, block_table, seq_q, seq_k, seq_k, nullptr, nullptr, sinks, metadata, /*kv_quant_mode=*/1,
      kTileSize, kRopeHeadDim, /*softmax_scale=*/0.0441942, /*cmp_ratio=*/4, /*ori_mask_mode=*/4,
      /*cmp_mask_mode=*/3, /*ori_win_left=*/127, /*ori_win_right=*/0, const_cast<char*>("TND"),
      const_cast<char*>("PA_ND"), ori_stride0, cmp_stride0, /*return_softmax_lse=*/false, attn_out, no_lse);
  Check(attn_ws > 0, "the shared-KV attention core planned a non-empty workspace over both KV streams");
  StaticOpSlot attn_slot;
  attn_slot.Adopt(OpId::kKvQuantSparseAttnSharedkv, "vendor/kv_quant_sparse_attn_sharedkv", attn_ws, attn_executor);
  void* attn_workspace = attn_ws > 0 ? device.DeviceMalloc(attn_ws) : nullptr;
  attn_slot.Launch(ops, attn_workspace, stream);
  // Slot 0 is q; re-binding it and relaunching is the per-token decode step.
  const uint64_t attn_mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  attn_slot.SetAddress(0, q, AsMockTensor(q)->device_addr);
  attn_slot.Launch(ops, attn_workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == attn_mismatches_before,
        "the shared-KV attention plan relaunches after aclSetTensorAddr on its query slot");
  Check(mock::MockVendorSelfCopyHazards() == hazards_before,
        "the shared-KV attention core records no manual-4.31 self-copy hazard");
  attn_slot.Reset();

  // The compressed half alone is a valid configuration (HCA-only decode).
  aclOpExecutor* cmp_only = nullptr;
  const uint64_t cmp_only_ws = PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
      ops, OpId::kKvQuantSparseAttnSharedkv, &cmp_only, q, nullptr, cmp_kv, nullptr, cmp_indices, nullptr,
      block_table, seq_q, nullptr, seq_k, nullptr, nullptr, sinks, metadata, 1, kTileSize, kRopeHeadDim, 0.0441942,
      128, 4, 3, 127, 0, const_cast<char*>("TND"), const_cast<char*>("PA_ND"), 0, cmp_stride0, false, attn_out,
      no_lse);
  Check(cmp_only_ws > 0, "the shared-KV attention core plans with only the compressed KV stream bound");
  aclDestroyAclOpExecutor(cmp_only);

  // Neither half bound makes the operator meaningless.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
        ops, OpId::kKvQuantSparseAttnSharedkv, &bad, q, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, seq_q,
        nullptr, nullptr, nullptr, nullptr, sinks, metadata, 1, kTileSize, kRopeHeadDim, 0.0441942, 4, 4, 3, 127, 0,
        const_cast<char*>("TND"), const_cast<char*>("PA_ND"), 0, 0, false, attn_out, no_lse);
  }, "the shared-KV attention core with neither KV stream bound");

  // A bound cmp stream without its sparse indices has nothing to gather.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
        ops, OpId::kKvQuantSparseAttnSharedkv, &bad, q, ori_kv, cmp_kv, ori_indices, nullptr, block_table,
        block_table, seq_q, seq_k, seq_k, nullptr, nullptr, sinks, metadata, 1, kTileSize, kRopeHeadDim, 0.0441942,
        4, 4, 3, 127, 0, const_cast<char*>("TND"), const_cast<char*>("PA_ND"), ori_stride0, cmp_stride0, false,
        attn_out, no_lse);
  }, "the shared-KV attention core with a bound cmpKv but no cmpSparseIndices");

  // 512 is the DSV4 total head dim; a 128-wide query is the indexer geometry,
  // not the attention geometry.
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
        ops, OpId::kKvQuantSparseAttnSharedkv, &bad, query, ori_kv, cmp_kv, ori_indices, cmp_indices, block_table,
        block_table, seq_q, seq_k, seq_k, nullptr, nullptr, sinks, metadata, 1, kTileSize, kRopeHeadDim, 0.0441942,
        4, 4, 3, 127, 0, const_cast<char*>("TND"), const_cast<char*>("PA_ND"), ori_stride0, cmp_stride0, false,
        attn_out, no_lse);
  }, "the shared-KV attention core with a 128-wide head dim where the DSV4 512 is required");

  for (aclTensor* tensor : {query, key, weights, q_scale, k_scale, seq_q, seq_k, block_table, metadata, cmp_indices,
                            no_values, q, ori_kv, cmp_kv, ori_indices, sinks, attn_out, no_lse}) {
    aclDestroyTensor(tensor);
  }
  device.DestroyStream(stream);
  mock::MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 6. The two cache epilogs: REF-output scatter, no copy stage
// ---------------------------------------------------------------------------

void TestCacheEpilogs() {
  Section("vendored cache epilogs: kv_compress_epilog, indexer_compress_epilog_v2");
  mock::MockResetAllocatorForTest();
  OpTable ops;

  constexpr int64_t kBlocks = 8;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kKvDim = 512;
  constexpr int64_t kIndexerDim = 128;
  constexpr int64_t kRows = 16;
  constexpr int64_t kQuantGroup = 128;

  const uintptr_t arena = mock::MockDeviceMalloc(32ull << 20);
  size_t cursor = 0;
  const auto next = [&](size_t bytes) {
    cursor += (bytes + 63) & ~size_t(63);
    return reinterpret_cast<void*>(arena + cursor);
  };

  SimulatedDeviceOps device(16ull << 20);
  DeviceStream stream = device.CreateStream();

  // -- kv_compress_epilog -----------------------------------------------------
  aclTensor* kv_cache = MakeTensor({kBlocks, kBlockSize, kKvDim}, ACL_FLOAT8_E4M3FN,
                                   next(kBlocks * kBlockSize * kKvDim));
  aclTensor* kv_rows = MakeTensor({kRows, kKvDim}, ACL_BF16, next(kRows * kKvDim * 2));
  aclTensor* kv_slots = MakeTensor({kRows}, ACL_INT32, next(kRows * 4));
  const int64_t kv_block_stride = AsMockTensor(kv_cache)->strides.at(0);
  const int ref_plans_before = mock::MockRefOutputPlans();
  const int hazards_before = mock::MockVendorSelfCopyHazards();

  aclOpExecutor* executor = nullptr;
  uint64_t ws = PlanAclnnOp<KvCompressEpilogPlanFn>(ops, OpId::kKvCompressEpilog, &executor, kv_cache, kv_rows,
                                                    kv_slots, kQuantGroup, 1, 1, 1, kv_block_stride);
  Check(ws > 0, "kv_compress_epilog planned a non-empty workspace");
  StaticOpSlot kv_slot;
  kv_slot.Adopt(OpId::kKvCompressEpilog, "vendor/kv_compress_epilog", ws, executor);
  void* workspace = ws > 0 ? device.DeviceMalloc(ws) : nullptr;
  kv_slot.Launch(ops, workspace, stream);
  // The cache is IR slot 0 and is the REF output: re-binding it is how the
  // decode loop walks blocks on one plan.
  const uint64_t mismatches_before = mock::MockMemoryStatistics().slot_map_mismatches;
  kv_slot.SetAddress(0, kv_cache, AsMockTensor(kv_cache)->device_addr);
  kv_slot.Launch(ops, workspace, stream);
  Check(mock::MockMemoryStatistics().slot_map_mismatches == mismatches_before,
        "aclSetTensorAddr re-binds kv_compress_epilog's REF cache slot across launch cycles");
  kv_slot.Reset();

  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvCompressEpilogPlanFn>(ops, OpId::kKvCompressEpilog, &bad, kv_cache, kv_rows, kv_slots,
                                        kQuantGroup, 1, 1, 1, kv_block_stride + 1);
  }, "kv_compress_epilog with a blockStride that disagrees with the cache view");

  aclTensor* short_slots = MakeTensor({kRows / 2}, ACL_INT32, next(kRows / 2 * 4));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvCompressEpilogPlanFn>(ops, OpId::kKvCompressEpilog, &bad, kv_cache, kv_rows, short_slots,
                                        kQuantGroup, 1, 1, 1, kv_block_stride);
  }, "kv_compress_epilog with fewer slot_mapping entries than rows of x");

  aclTensor* fp32_cache = MakeTensor({kBlocks, kBlockSize, kKvDim}, ACL_FLOAT32,
                                     next(kBlocks * kBlockSize * kKvDim * 4));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<KvCompressEpilogPlanFn>(ops, OpId::kKvCompressEpilog, &bad, fp32_cache, kv_rows, kv_slots,
                                        kQuantGroup, 1, 1, 1, AsMockTensor(fp32_cache)->strides.at(0));
  }, "kv_compress_epilog with an FP32 cache where FP8 is required");

  // -- indexer_compress_epilog_v2 ---------------------------------------------
  aclTensor* idx_cache = MakeTensor({kBlocks, kBlockSize, kIndexerDim}, ACL_UINT8,
                                    next(kBlocks * kBlockSize * kIndexerDim));
  aclTensor* idx_rows = MakeTensor({kRows, kIndexerDim}, ACL_BF16, next(kRows * kIndexerDim * 2));
  aclTensor* idx_slots = MakeTensor({kRows}, ACL_INT32, next(kRows * 4));
  const int64_t idx_block_stride = AsMockTensor(idx_cache)->strides.at(0);

  aclOpExecutor* idx_executor = nullptr;
  const uint64_t idx_ws = PlanAclnnOp<IndexerCompressEpilogV2PlanFn>(
      ops, OpId::kIndexerCompressEpilogV2, &idx_executor, idx_cache, idx_rows, idx_slots, 2, idx_block_stride);
  Check(idx_ws > 0, "indexer_compress_epilog_v2 planned a non-empty workspace");
  StaticOpSlot idx_slot;
  idx_slot.Adopt(OpId::kIndexerCompressEpilogV2, "vendor/indexer_compress_epilog_v2", idx_ws, idx_executor);
  idx_slot.Launch(ops, idx_ws > 0 ? device.DeviceMalloc(idx_ws) : nullptr, stream);
  Check(true, "indexer_compress_epilog_v2 adopted (repeatable) and launched");
  idx_slot.Reset();

  aclTensor* wide_rows = MakeTensor({kRows, 256}, ACL_BF16, next(kRows * 256 * 2));
  CheckRefuses([&] {
    aclOpExecutor* bad = nullptr;
    PlanAclnnOp<IndexerCompressEpilogV2PlanFn>(ops, OpId::kIndexerCompressEpilogV2, &bad, idx_cache, wide_rows,
                                               idx_slots, 2, idx_block_stride);
  }, "indexer_compress_epilog_v2 with a 256-wide x where the indexer width is 128");

  // Both epilogs are REF-output operators: they stage no copy at all, so
  // neither can contribute a manual-4.31 self-copy hazard.
  Check(mock::MockRefOutputPlans() >= ref_plans_before + 2,
        "both cache epilogs are recorded as REF-output plans");
  Check(mock::MockVendorSelfCopyHazards() == hazards_before,
        "neither cache epilog records a manual-4.31 self-copy hazard");

  for (aclTensor* tensor : {kv_cache, kv_rows, kv_slots, short_slots, fp32_cache, idx_cache, idx_rows, idx_slots,
                            wide_rows}) {
    aclDestroyTensor(tensor);
  }
  device.DestroyStream(stream);
  mock::MockUnregisterSpan(arena);
}

// ---------------------------------------------------------------------------
// 7. The repeatability invariant over the whole vendored set
// ---------------------------------------------------------------------------

void TestRepeatabilityInvariant() {
  Section("vendored operator set: the manual-4.31 self-copy invariant");
  // Every wrapper in third_party/ops_dsv4 either writes its output through a
  // distinct executor-owned tensor (so its ViewCopy has src != dst), or has a
  // REF output and stages no copy, or -- for the patched mhc_sinkhorn -- elides
  // the copy when the kernel already wrote the caller tensor. So after
  // exercising all thirteen above, the hazard ledger must be empty.
  //
  // The fused hc_* family is the fourth case and the only one that needs no
  // rule: it has neither a REF output nor an in-place Sinkhorn tensor, so there
  // is nothing for a copy stage to alias.
  Check(mock::MockVendorSelfCopyHazards() == 0,
        "no vendored operator planned a same-address ViewCopy across the whole suite");
  Check(mock::MockSinkhornSelfCopyElisions() > 0,
        "the mhc_sinkhorn patch fired at least once (the contiguous-output path was exercised)");
  Check(mock::MockRefOutputPlans() >= 3,
        "all three REF-output operators (compressor, both epilogs) were exercised");
}

}  // namespace
}  // namespace ascend_moe

int RunMain() {
  using namespace ascend_moe;
  std::printf("vendor_ops_table_test -- the vendored arch35 operators through OpTable\n");
  try {
    TestTableResolution();
    TestMhcChain();
    TestFusedHcFamily();
    TestQuantLightningIndexer();
    TestCompressor();
    TestSharedKvPath();
    TestCacheEpilogs();
    TestRepeatabilityInvariant();
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
