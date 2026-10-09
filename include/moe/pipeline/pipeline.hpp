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

// The 43-layer DeepSeek-V4 Flash decode graph on ACLNN V5, as an ORCHESTRATOR.
//
// Dsv4Pipeline no longer does everything itself; it wires the pieces together
// and owns the per-layer replay:
//
//   StaticArenaManager       what is allocated and which descriptors cover it
//   MoeRouterEngine          scoring -> selection -> dropless dispatch
//   ExclusiveExpertManager   routed-expert residency (passed in)
//   IDeviceAllocator / IStreamEngine   the backend contract (passed in)
//
// What remains here is WHEN things run: the attention half of every layer, the
// expert GEMMs and the shared expert, the head, and the step boundaries.
//
// ONE EXECUTOR PER STAGE, NOT PER LAYER
// -------------------------------------
// All 43 layers have identical shapes, so each stage is planned exactly once
// and replayed 43 times with `aclSetTensorAddr` / `aclSetDynamicTensorAddr`
// swapping that layer's weights and the step's activations into the retained,
// repeatable executor. That is ~26 planned executors for the whole model
// instead of 26 x 43, and it is what the brief's "all descriptors created ONCE
// during initialization, dynamic addresses swapped via aclSetTensorAddr"
// describes.
//
// THREE DEVIATIONS FROM THE BRIEF, ALL FORCED, ALL REPORTED
// ---------------------------------------------------------
// 1. ONE HOST SYNCHRONIZATION PER MoE LAYER, not one per step.
//
//    The brief asks for "zero CPU-device synchronization calls inside the
//    43-layer loop" (3.4) *and* for a host-managed exclusive swap engine whose
//    residency decisions are made in C++ tables (2). Those cannot both hold:
//    the engine has to know which six experts the router chose before it can
//    decide what to promote and what to evict, and the router runs on the
//    device. So each MoE layer copies the 24-byte top-6 index vector D2H and
//    synchronizes (MoeRouterEngine::ScoreAndSelect). `StepCounters` counts
//    them and the report prints the expected total, so the cost is visible
//    rather than hidden. Removing it needs either a device-side residency
//    table the GEMM indexes itself -- which means a custom Ascend C kernel,
//    excluded by 3 -- or routing the next layer one layer early. Both are
//    named in README.md.
//
// 2. ONE HOST PLAN PER mHC ROUND, BECAUSE SINKHORN CANNOT BE REPEATABLE.
//
//    Every sub-block is wrapped by an mHC round: `aclnnMhcPre` mixes the four
//    residual streams into the layer input, the 4x4 mixing map is normalized
//    into B_l, and `aclnnMhcPost` folds the sub-block's output back. Pre and
//    Post hold retained, repeatable executors -- the 950PR run confirmed
//    Repeatable=true for both at this engine's TND shapes. The normalization
//    between them cannot: `aclSetAclOpExecutorRepeatable` fails
//    `aclnnMhcSinkhorn` with 561000, and the non-contiguous-output workaround
//    the CANN 4.31 note once recommended is now refused outright with 561103.
//
//    So the standalone operator is planned and launched once per round, its
//    executor consumed by its own launch: 86 host plans per step, counted as
//    `StepCounters::sinkhorn_replans` and printed. Removing them needs the
//    normalization fused into a kernel that never materializes B_l at the
//    aclnn layer -- `aclnnHcPreSinkhorn` is the vendored operator that does
//    this, and `MhcMixingMode::kFusedExternal` is the seam it plugs into.
//
//    THE LIGHTNING INDEXER, by contrast, IS now applied: a CSA layer runs the
//    compressor, `aclnnVllmQuantLightningIndexer` at top-512 and
//    `aclnnKvQuantSparseAttnSharedkv`. What remains unresolved there is the
//    indexer's per-token-head dequant scales, which no operator in this graph
//    produces; the report says so.
//
// 3. THE COMBINE IS A [1, 6] x [6, hidden] MATMUL, NOT MoeTokenUnpermute.
//
//    `aclnnMoeTokenUnpermute` has no `ascend950` kernel directory in CANN
//    9.2.0-beta.2, and `aclnnGroupedMatmulFinalizeRoutingV3` -- which does --
//    takes its expert weights as ONE tensor, i.e. contiguous, which an
//    exclusive LRU slot pool cannot promise. For a single-token step the six
//    expanded rows are the same token, so the weighted sum IS a small matmul
//    over the permuted routing weights, which is exact and runs on the part.
//    See the comment on the `expert_combine` stage.

#pragma once
#include "moe/core/resource_scope.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/stream_engine.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/pipeline/moe_router_engine.hpp"
#include "moe/pipeline/routed_moe_block.hpp"
#include "moe/memory/static_arena_manager.hpp"
#include "moe/pipeline/static_op_slot_table.hpp"

namespace ascend_moe {

struct StepCounters {
  uint64_t steps = 0;
  uint64_t layers = 0;
  uint64_t launches = 0;
  // Must be exactly `kNumLayers` per step (deviation 1) plus one for the
  // argmax readback. Anything above that is a defect.
  uint64_t host_synchronizations = 0;
  uint64_t device_allocations_in_step = 0;  // must stay 0
  uint64_t descriptors_built_in_step = 0;   // must stay 0
  uint64_t expert_slot_hits = 0;
  uint64_t expert_slot_misses = 0;
  // mHC rounds executed: two per layer per step (attention, MoE).
  uint64_t mhc_rounds = 0;
  // Host plans issued inside the decode loop because `aclnnMhcSinkhorn`
  // cannot hold a repeatable executor on a 950PR (561000). One per mHC round
  // on MhcMixingMode::kPerUsePlan, zero on kFusedExternal. This is a FORCED
  // violation of the engine's zero-host-work-in-the-loop rule, so it is
  // counted and printed rather than hidden.
  uint64_t sinkhorn_replans = 0;
  // Compressor cadence. `compressor_holds` counts the no-op launches on
  // partial windows and `compressor_emissions` the windows that closed and
  // wrote a Dsv4CompressedKvEntry; `compressed_entries_written` must equal
  // the emissions exactly, because a hold step must never reach the cache.
  uint64_t compressor_holds = 0;
  uint64_t compressor_emissions = 0;
  uint64_t compressed_entries_written = 0;
  uint64_t indexer_selections = 0;
  // Per-path layer executions, so a report can show the dispatch really
  // followed `compress_ratios`.
  uint64_t swa_layers = 0;
  uint64_t csa_layers = 0;
  uint64_t hca_layers = 0;
};

// Which tensor index inside an op's retained executor each swappable address
// occupies. ACLNN numbers an executor's tensors in IR order -- every input in
// declaration order, then every output -- counting only tensor and tensor-list
// arguments and skipping the scalars. Each constant below is derived that way
// from the signature in op_table.hpp, with the argument it refers to
// named.
//
// This derivation is the one part of the design the plan phase cannot confirm:
// `GetWorkspaceSize` succeeds whatever indices a later `aclSetTensorAddr`
// uses, and a wrong index silently repoints the wrong tensor.
// `DescribeSlotIndexMap()` prints the whole map for exactly that reason, so a
// device bring-up can check it against each op's IR definition before trusting
// a number.
namespace slot {

// aclnnRmsNorm(x, gamma, eps, yOut, rstdOut)
inline constexpr size_t kRmsNormX = 0;
inline constexpr size_t kRmsNormGamma = 1;
inline constexpr size_t kRmsNormY = 2;
inline constexpr size_t kRmsNormRstd = 3;

// aclnnRmsNormDynamicMxQuant(x, gamma, beta, ..., yOut, mxscaleOut, rstdOut)
inline constexpr size_t kMxQuantX = 0;
inline constexpr size_t kMxQuantGamma = 1;
inline constexpr size_t kMxQuantBeta = 2;
inline constexpr size_t kMxQuantY = 3;
inline constexpr size_t kMxQuantScale = 4;
inline constexpr size_t kMxQuantRstd = 5;

// aclnnMatmul(self, mat2, out, cubeMathType)
inline constexpr size_t kMatmulSelf = 0;
inline constexpr size_t kMatmulMat2 = 1;
inline constexpr size_t kMatmulOut = 2;

// aclnnQuantMatmulV5(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset,
//                    yOffset, bias, ..., out)
inline constexpr size_t kQuantMmX1 = 0;
inline constexpr size_t kQuantMmX2 = 1;
inline constexpr size_t kQuantMmX1Scale = 2;
inline constexpr size_t kQuantMmX2Scale = 3;
inline constexpr size_t kQuantMmOut = 9;

// aclnnSoftplus(x, beta, threshold, out) and aclnnSqrt(x, out): the two host
// scalars are skipped by the IR numbering, so both unary stages have out at
// tensor index 1. Neither address is ever swapped -- the scoring buffers are
// step-static activations -- but the indices are recorded so a bring-up can
// check them like every other stage.
inline constexpr size_t kSoftplusX = 0;
inline constexpr size_t kSoftplusOut = 1;
inline constexpr size_t kSqrtX = 0;
inline constexpr size_t kSqrtOut = 1;

// aclnnMoeGatingTopKV2(x, bias, inputIds, tid2eid, ..., yOut, expertIdxOut,
//                      outOut)
inline constexpr size_t kGatingX = 0;
inline constexpr size_t kGatingBias = 1;
inline constexpr size_t kGatingY = 4;
inline constexpr size_t kGatingExpertIdx = 5;

// aclnnApplyRotaryPosEmbV2(queryRef, keyRef, cos, sin, ...)
inline constexpr size_t kRotaryCos = 2;
inline constexpr size_t kRotarySin = 3;

// aclnnScatterPaKvCache(key, keyCacheRef, slotMapping, value, valueCacheRef, ..)
inline constexpr size_t kScatterKeyCache = 1;
inline constexpr size_t kScatterValueCache = 4;

// aclnnFusedInferAttentionScoreV5: query, key[], value[] then 22 more tensor
// arguments before queryRope (24) and keyRope (25).
inline constexpr size_t kFiaKeyList = 1;
inline constexpr size_t kFiaValueList = 2;
inline constexpr size_t kFiaKeyRope = 25;

// GroupedMatmulV5(x[], weight[], bias[], scale[], ...) and
// GroupedMatmulSwigluQuantV2(x, weight[], weightScale[], ...)
inline constexpr size_t kGmmWeightList = 1;
inline constexpr size_t kGmmV5ScaleList = 3;
inline constexpr size_t kGmmSwigluScaleList = 2;

// aclnnInplaceAdd(selfRef, other, alpha): alpha is a host scalar and is
// skipped, so `other` is tensor index 1.
inline constexpr size_t kInplaceAddSelf = 0;
inline constexpr size_t kInplaceAddOther = 1;

// aclnnMhcPre(x, phi, alpha, bias, gammaOptional, normEps, hcEps,
//             hInOut, hPostOut, hResOut, invRmsOut, hMixOut, hPreOut).
// The two epsilons are host doubles and are skipped; the three trailing
// optional outputs are bound null but still occupy their indices.
inline constexpr size_t kMhcPreX = 0;
inline constexpr size_t kMhcPrePhi = 1;
inline constexpr size_t kMhcPreAlpha = 2;
inline constexpr size_t kMhcPreBias = 3;
inline constexpr size_t kMhcPreGamma = 4;

// aclnnMhcSinkhorn(x, eps, numIters, output, normOut, sumOut). H2 rebinds
// index 1 and relaunches, which is what confirms this numbering empirically.
inline constexpr size_t kMhcSinkhornX = 0;
inline constexpr size_t kMhcSinkhornOut = 1;

// aclnnMhcPost(x, hRes, hOut, hPost, out)
inline constexpr size_t kMhcPostX = 0;
inline constexpr size_t kMhcPostHRes = 1;
inline constexpr size_t kMhcPostHOut = 2;
inline constexpr size_t kMhcPostHPost = 3;
inline constexpr size_t kMhcPostOut = 4;

// aclnnCompressor(x, wkv, wgate, stateCacheRef, ape, normWeight, ropeSin,
//                 ropeCos, stateBlockTable, cuSeqlens, seqused, startPos,
//                 ...attrs..., cmpKvOut)
inline constexpr size_t kCompressorX = 0;
inline constexpr size_t kCompressorWkv = 1;
inline constexpr size_t kCompressorWgate = 2;
inline constexpr size_t kCompressorStateCache = 3;
inline constexpr size_t kCompressorApe = 4;
inline constexpr size_t kCompressorNormWeight = 5;
inline constexpr size_t kCompressorRopeSin = 6;
inline constexpr size_t kCompressorRopeCos = 7;
inline constexpr size_t kCompressorCmpKvOut = 12;

// aclnnKvCompressEpilog(kvCompressCacheRef, x, slotMapping, ...attrs...) and
// aclnnIndexerCompressEpilogV2(indexerCompressCacheRef, x, slotMapping, ...).
// Both leading parameters are REF: input and output in one slot.
inline constexpr size_t kCompressEpilogCacheRef = 0;

// aclnnVllmQuantLightningIndexer(query, key, weights, queryDequantScale,
//                                keyDequantScale, actualSeqLenQ,
//                                actualSeqLenK, blockTable, metadata, ...)
inline constexpr size_t kIndexerKey = 1;
inline constexpr size_t kIndexerWeights = 2;
inline constexpr size_t kIndexerKeyDequantScale = 4;

// aclnnKvQuantSparseAttnSharedkv(q, oriKv, cmpKv, oriSparseIndices,
//                                cmpSparseIndices, oriBlockTable,
//                                cmpBlockTable, ...)
inline constexpr size_t kSparseAttnCmpKv = 2;

}  // namespace slot

// The orchestrator. Allocation lives in StaticArenaManager, router scoring and
// dispatch in MoeRouterEngine, expert residency in the ExclusiveExpertManager
// passed in, and the EXPERT EXECUTION ITSELF in the injected IRoutedMoeBlock
// (default StandardAclnnMoeBlock); this class sequences the 43-layer replay
// over all of them, owning attention, the shared expert and the head.
class Dsv4Pipeline {
 public:
  // `router` is constructed by the owner and shared with the MoE block, which
  // consumes its dispatch output; `moe_block` defaults to the stock
  // StandardAclnnMoeBlock constructed here. The pipeline references all
  // services and outlives none of them.
  Dsv4Pipeline(IDeviceAllocator& allocator, IStreamEngine& streams, const OpTable& ops,
               ExclusiveExpertManager& experts, MoeRouterEngine& router, const RuntimeConfig& config,
               std::unique_ptr<IRoutedMoeBlock> moe_block = nullptr);
  ~Dsv4Pipeline();

  Dsv4Pipeline(const Dsv4Pipeline&) = delete;
  Dsv4Pipeline& operator=(const Dsv4Pipeline&) = delete;

  // Reserves the arena, ingests the backbone weights from `source`, creates
  // every descriptor, plans every stage and seals. `source` is NOT closed here:
  // the expert manager closes it, after it has taken its own bytes, so there is
  // exactly one place that seals the only copy.
  void Build(WeightByteSource& source);

  // Runs the 43 layers for one token at `position`. Enqueues work; reads
  // nothing back except the per-layer routing vector (deviation 1).
  void DecodeStep(int32_t token_id, int64_t position);

  // The single `aclrtSynchronizeStream` that ends a step, plus the 8-byte
  // readback of the greedy token.
  int32_t ReadArgmaxToken();

  const StepCounters& counters() const { return counters_; }
  const InferenceDiagnostics& diagnostics() const { return diagnostics_; }
  const StaticArenaManager& arena_manager() const { return arena_manager_; }
  const MoeRouterEngine& router() const { return router_; }
  const IRoutedMoeBlock& moe_block() const { return *moe_block_; }
  std::string DescribeStages() const;
  std::string DescribeSlotIndexMap() const;

  // The memory the backbone needs, so the slot planner can subtract it from
  // free HBM before choosing K. Pure arithmetic; no allocation.
  static size_t BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len,
                                    const std::vector<int64_t>& compress_ratios = {});

 private:
  // One mHC round's planned stages plus the weights and the sub-block output
  // they bind, so RunMhcPre / RunMhcPost are written once and parameterized
  // rather than duplicated for attention and MoE.
  struct MhcRound {
    const char* pre = nullptr;
    const char* sinkhorn = nullptr;
    const char* post = nullptr;
  };

  // ---- build helpers ----
  void PlanStages();
  void PlanMhcStages();
  void PlanCompressionStages();
  ExpertSlotAddresses CollectExpertSlotAddresses();

  // ---- per-layer helpers ----
  // The mHC half of a sub-block: Pre folds the four streams into h_in and
  // Sinkhorn normalizes the residual map; Post folds the sub-block's output
  // back into the streams and advances the ping-pong parity.
  void RunMhcPre(const MhcRound& round, const MhcRoundTensors& tensors,
                 const BackboneWeights::Layer::MhcWeights& weights);
  // Post needs no tensor set: hRes, hOut and hPost are bound at plan time and
  // never move; only the ping-pong stream buffers are repointed.
  void RunMhcPost(const MhcRound& round);
  void RunAttention(int32_t layer, int64_t position);
  // The attention CORE, chosen by this layer's compress_ratios entry. The
  // prologue (q/kv projections, rope, the uncompressed cache write) and the
  // epilogue (attn_quant, o_proj) are shared by all three paths.
  void RunAttentionCore(int32_t layer, int64_t position);
  void RunSlidingWindowAttention();
  void RunCompressedAttention(int32_t layer, int64_t position, AttentionPath path);
  // Advances this layer's window ring and, on a closing step, emits one
  // Dsv4CompressedKvEntry into the paged compressed cache. Returns true when
  // the window closed.
  bool AdvanceCompressorCadence(int32_t layer, int64_t position, int64_t ratio, AttentionPath path);
  void RunMoe(int32_t layer);
  void RunSharedExpert(int32_t layer);
  void Launch(PipelineStage& stage_entry);

  // Byte base of `layer`'s slice of a per-layer compression reservation.
  uint8_t* LayerSlice(ArenaHandle handle, size_t layer_stride_bytes, int32_t layer) const;

  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  ResourceScope resources_{allocator_, streams_};
  const OpTable& ops_;
  ExclusiveExpertManager& experts_;
  RuntimeConfig config_;

  StaticArenaManager arena_manager_;
  MoeRouterEngine& router_;
  // The expert-execution seam. Never null after construction: a null
  // injection becomes the stock StandardAclnnMoeBlock.
  std::unique_ptr<IRoutedMoeBlock> moe_block_;

  DeviceStream compute_stream_ = nullptr;
  DeviceEvent compute_done_ = nullptr;

  // The orchestrator's planned stages, born at full capacity (StaticOpSlot
  // is not movable, see StaticOpSlotTable); the injected MoE block plans its
  // expert stages into the same table, so one DescribeStages covers the
  // whole graph.
  //
  // The ceiling covers the widest graph: the shared dense backbone, two mHC
  // rounds, the three-stream reduction at the head, and -- on a checkpoint
  // that marks layers both CSA and HCA -- both compressed cores, both
  // compressor cadences and the indexer chain.
  static constexpr size_t kMaxPipelineStages = 56;
  StaticOpSlotTable stages_ = StaticOpSlotTable(kMaxPipelineStages, "pipeline");
  StepCounters counters_;
  InferenceDiagnostics diagnostics_;

  // The six active experts' region addresses at descriptor time; the MoE
  // block plans against the same map.
  ExpertSlotAddresses slot_addrs_;

  // Shapes, fixed at construction from the config.
  int64_t heads_ = kNumAttentionHeads;

  // Which of the two mHC stream buffers holds the live residual stream. Every
  // `aclnnMhcPost` reads this one and writes the other, so no call aliases its
  // own input; 86 posts per step leaves it back where it started, but the
  // parity is tracked rather than assumed.
  size_t stream_parity_ = 0;

  // Whether the compressed paths were planned at all. False on a SWA-only
  // checkpoint, where they cost nothing: no reservation, no descriptor, no
  // executor.
  bool csa_planned_ = false;
  bool hca_planned_ = false;
  bool swa_planned_ = false;

  // The pinned host mailboxes for the per-step embedding and cache-slot writes.
  int64_t* token_mailbox_ = nullptr;  // [1] greedy token, D2H
  int32_t* slot_mailbox_ = nullptr;   // [1] paged cache slot, H2D
  // The compressed stream's per-emission scalars: the destination slot, the
  // compressed context length, and the compressor's {cuSeqlens, seqused,
  // startPos} triple, which share one reservation so one copy writes all three.
  int32_t* cmp_slot_mailbox_ = nullptr;    // [1] compressed cache slot, H2D
  int32_t* cmp_seq_mailbox_ = nullptr;     // [1] compressed context length, H2D
  int32_t* cmp_window_mailbox_ = nullptr;  // [3] cuSeqlens, seqused, startPos
};

}  // namespace ascend_moe
