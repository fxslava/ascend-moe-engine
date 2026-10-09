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

// The orchestrator's half of the graph: attention, the expert GEMMs, the
// shared expert and the head. Memory lives in StaticArenaManager, the router
// chain in MoeRouterEngine; this file sequences them across the 43 layers.

#include "moe/pipeline/pipeline.hpp"

#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "moe/core/error.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/standard_aclnn_moe_block.hpp"

namespace ascend_moe {
namespace {

// This runner decodes one token at a time (greedy, batch 1). Every buffer the
// manager reserves is sized for it, and the routing combine is only correct
// for it -- see the note on the `expert_combine` stage.
constexpr int64_t kTokensPerStep = 1;

// Scale dtype for the dense block-128 weight scales and for the MX block-32
// activation scales: both are OCP E8M0, one byte per block.
constexpr int32_t kScaleDtype = kAclFloat8E8m0;

// `aclnnRmsNormDynamicMxQuant` round mode. FP8 destinations support "rint"
// only (artifacts/cann92_audit/REPORT.md section 0). Non-const because the
// aclnn signatures take `char*`.
char kMxRoundModeRint[] = "rint";
char kFiaLayout[] = "TND";
char kRotaryMode[] = "half";
char kScatterCacheMode[] = "Norm";

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Dsv4Pipeline::Dsv4Pipeline(IDeviceAllocator& allocator, IStreamEngine& streams, const OpTable& ops,
                           ExclusiveExpertManager& experts, MoeRouterEngine& router, const RuntimeConfig& config,
                           std::unique_ptr<IRoutedMoeBlock> moe_block)
    : allocator_(allocator),
      streams_(streams),
      ops_(ops),
      experts_(experts),
      config_(config),
      arena_manager_(allocator, streams, config),
      router_(router),
      moe_block_(std::move(moe_block)) {
  // The default backend: the stock ACLNN V5 expert path. An explicit
  // injection (the lattice skeleton, a bring-up fork) is taken as given.
  if (moe_block_ == nullptr) {
    moe_block_ = std::make_unique<StandardAclnnMoeBlock>(ops_, router_, experts_, config_);
  }
  compute_stream_ = resources_.CreateStream();
  compute_done_ = resources_.CreateEvent();
  // Primed for the same reason the staging engine primes its events: the first
  // layer's ScoreAndSelect waits on it before anything has recorded it.
  streams_.RecordEvent(compute_done_, compute_stream_);
  streams_.SynchronizeStream(compute_stream_);

  token_mailbox_ = static_cast<int64_t*>(resources_.HostPinnedMalloc(Int64Bytes(1)));
  slot_mailbox_ = static_cast<int32_t*>(resources_.HostPinnedMalloc(Int32Bytes(1)));
  *token_mailbox_ = 0;
  *slot_mailbox_ = 0;

  // The compressed paths' per-emission scalars. Pinned for the same reason
  // slot_mailbox_ is: they are DMA sources, so the copy can be async without a
  // stack-lifetime problem.
  cmp_slot_mailbox_ = static_cast<int32_t*>(resources_.HostPinnedMalloc(Int32Bytes(1)));
  cmp_seq_mailbox_ = static_cast<int32_t*>(resources_.HostPinnedMalloc(Int32Bytes(1)));
  cmp_window_mailbox_ = static_cast<int32_t*>(resources_.HostPinnedMalloc(Int32Bytes(3)));
  *cmp_slot_mailbox_ = 0;
  *cmp_seq_mailbox_ = 0;
  cmp_window_mailbox_[0] = 0;
  cmp_window_mailbox_[1] = 0;
  cmp_window_mailbox_[2] = 0;
}

Dsv4Pipeline::~Dsv4Pipeline() {
  resources_.Reset();
  router_.ResetStages();
}

size_t Dsv4Pipeline::BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len,
                                         const std::vector<int64_t>& compress_ratios) {
  return StaticArenaManager::BackboneDeviceBytes(mla, block_size, max_context_len, compress_ratios);
}

uint8_t* Dsv4Pipeline::LayerSlice(ArenaHandle handle, size_t layer_stride_bytes, int32_t layer) const {
  return arena_manager_.arena().AddressAs<uint8_t>(handle) +
         layer_stride_bytes * static_cast<size_t>(layer);
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

ExpertSlotAddresses Dsv4Pipeline::CollectExpertSlotAddresses() {
  // The six experts start in slots 0..5 so the plan phase sees real, resident
  // addresses; every layer afterwards repoints them via
  // aclSetDynamicTensorAddr. Kept in `slot_addrs_`: the MoE block's planning
  // consumes the same map.
  slot_addrs_ = ExpertSlotAddresses{};
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    const int32_t slot = static_cast<int32_t>(index);
    const size_t at = static_cast<size_t>(index);
    slot_addrs_.gate_up_weight[at] = experts_.RegionAddress(slot, ExpertRegionId::kGateUpWeight);
    slot_addrs_.gate_up_scale[at] = experts_.RegionAddress(slot, ExpertRegionId::kGateUpScale);
    slot_addrs_.down_weight[at] = experts_.RegionAddress(slot, ExpertRegionId::kDownWeight);
    slot_addrs_.down_scale[at] = experts_.RegionAddress(slot, ExpertRegionId::kDownScale);
  }
  return slot_addrs_;
}

void Dsv4Pipeline::Build(WeightByteSource& source) {
  arena_manager_.ReserveActivations();
  arena_manager_.ReserveBackbone();
  arena_manager_.Commit();
  arena_manager_.IngestBackbone(source);
  arena_manager_.CreateDescriptors(experts_.layout(), CollectExpertSlotAddresses());
  router_.PlanStages(ops_, arena_manager_, config_);
  PlanStages();
  arena_manager_.CommitWorkspace();
  arena_manager_.Seal();
  diagnostics_.synthetic_weights = config_.synthetic_weights;
  diagnostics_.dry_run = config_.dry_run;
#if ASCEND_MOCK_RUNTIME
  diagnostics_.mock_runtime = true;
#endif
  diagnostics_.paged_attention.block_size = static_cast<uint64_t>(config_.block_size);
  diagnostics_.paged_attention.total_blocks_allocated =
      static_cast<uint64_t>(arena_manager_.num_blocks() * kNumLayers);
}

// ---------------------------------------------------------------------------
// Planning (the orchestrator's stages; the router's five live in
// MoeRouterEngine::PlanStages, the expert GEMM stages in the injected
// IRoutedMoeBlock -- all into ONE StaticOpSlotTable)
// ---------------------------------------------------------------------------

void Dsv4Pipeline::PlanStages() {
  const MlaGeometry& mla = config_.mla;
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();

  // Which attention cores this checkpoint's `compress_ratios` actually asks
  // for. A path that no layer selects is NOT planned: it would otherwise cost
  // reservations (a CSA layer's indexer projection alone is 16 MiB per layer),
  // descriptors and retained executors for a core that can never be
  // dispatched.
  swa_planned_ = config_.any_layer_uses(AttentionPath::kSlidingWindow);
  csa_planned_ = arena_manager_.uses_csa();
  hca_planned_ = arena_manager_.uses_hca();

  // Every required op must exist before anything is planned, so a toolkit gap
  // is one clear message rather than a failure part-way through the graph.
  // The MoE block checks its own grouped-GEMM surface in its PlanStages.
  std::vector<OpId> required = {
      OpId::kRmsNorm,       OpId::kRmsNormDynamicMxQuant, OpId::kDynamicMxQuant,
      OpId::kQuantMatmulV5, OpId::kApplyRotaryPosEmbV2,   OpId::kScatterPaKvCache,
      OpId::kInplaceAdd,    OpId::kSwiGlu,                OpId::kArgMax,
      // mHC is the residual stream itself, so the trio is unconditional.
      OpId::kMhcPre,        OpId::kMhcSinkhorn,           OpId::kMhcPost,
  };
  if (swa_planned_) {
    required.push_back(OpId::kFusedInferAttentionScoreV5);
  }
  if (csa_planned_ || hca_planned_) {
    required.push_back(OpId::kCompressor);
    required.push_back(OpId::kKvCompressEpilog);
    required.push_back(OpId::kKvQuantSparseAttnSharedkv);
  }
  if (csa_planned_) {
    required.push_back(OpId::kMatmul);
    required.push_back(OpId::kVllmQuantLightningIndexer);
    required.push_back(OpId::kIndexerCompressEpilogV2);
  }
  ops_.RequireAll(required);

  // The stage table is born at full capacity (StaticOpSlot is not movable,
  // see stages_ in the header); planning only fills slots in place.
  auto add = [&](const char* name, OpId op) -> PipelineStage& { return stages_.Add(name, op); };
  auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* executor) {
    entry.slot.Adopt(entry.op, entry.name, workspace, executor);
    arena.NoteWorkspace(workspace);
  };

  aclOpExecutor* executor = nullptr;

  // 0. the mHC prologue / epilogue of both sub-blocks, and the compressed
  //    attention cores. Planned first so a refused geometry is reported
  //    before the dense backbone is half planned.
  PlanMhcStages();
  PlanCompressionStages();

  // 1. input RMSNorm fused with the FP8 activation quantization. Its input is
  //    the mHC attention round's h_in, bound AS IS: TND hIn is [1, 4096],
  //    which is already the activation this stage is planned for, so there is
  //    no squeeze, no unsqueeze and no intermediate view.
  {
    PipelineStage& entry = add("input_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.mhc_attn.h_in, t.w_input_norm, nullptr, kRmsNormEpsilon,
        kSwigluScaleAlgOcp, kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.normed_fp8,
        t.normed_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 2. q down-projection.
  {
    PipelineStage& entry = add("q_a_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_q_a, t.normed_mx_scale, t.w_q_a_scale, nullptr, nullptr,
        nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.q_a);
    adopt(entry, workspace, executor);
  }
  // 3. q LoRA norm, fused quantization again.
  {
    PipelineStage& entry = add("q_a_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.q_a, t.w_q_a_norm, nullptr, kRmsNormEpsilon, kSwigluScaleAlgOcp,
        kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.q_a_fp8, t.q_a_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 4. q up-projection with W_UK folded in: emits [heads, kv_lora | rope], so
  //    attention happens in the compressed latent space.
  {
    PipelineStage& entry = add("q_b_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.q_a_fp8, t.w_q_b, t.q_a_mx_scale, t.w_q_b_scale, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, config_.dense_group_size, t.q_b);
    adopt(entry, workspace, executor);
  }
  // 5. kv down-projection with MQA rope: emits [kv_lora | rope].
  {
    PipelineStage& entry = add("kv_a_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_kv_a, t.normed_mx_scale, t.w_kv_a_scale, nullptr, nullptr,
        nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.kv_a);
    adopt(entry, workspace, executor);
  }
  // 6. the compressed latent's own RMSNorm (no quantization: it is cached bf16).
  {
    PipelineStage& entry = add("kv_a_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.kv_latent, t.w_kv_a_norm,
                                                          kRmsNormEpsilon, t.kv_latent_normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  // 7. partial RoPE over the q and k rope slices, in place.
  {
    PipelineStage& entry = add("rope", OpId::kApplyRotaryPosEmbV2);
    const uint64_t workspace = PlanAclnnOp<ApplyRotaryPosEmbV2PlanFn>(
        ops_, entry.op, &executor, t.q_rope, t.k_rope, t.rope_cos, t.rope_sin, kRotaryLayoutBsnd, kRotaryMode);
    adopt(entry, workspace, executor);
  }
  // 8. paged cache write. The two co-indexed caches carry the latent and the
  //    rope slice, so one scatter writes both halves of the MLA row.
  {
    PipelineStage& entry = add("kv_cache_write", OpId::kScatterPaKvCache);
    const uint64_t workspace = PlanAclnnOp<ScatterPaKvCachePlanFn>(
        ops_, entry.op, &executor, t.kv_latent_normed, t.kv_latent_cache, t.slot_mapping, t.k_rope,
        t.kv_rope_cache, nullptr, nullptr, nullptr, kScatterCacheMode, nullptr, nullptr, nullptr);
    adopt(entry, workspace, executor);
  }
  // 9. paged MLA decode attention.
  {
    PipelineStage& entry = add("attention", OpId::kFusedInferAttentionScoreV5);
    const double scale = 1.0 / std::sqrt(static_cast<double>(mla.q_head_dim()));
    const uint64_t workspace = PlanAclnnOp<FusedInferAttentionScoreV5PlanFn>(
        ops_, entry.op, &executor, t.q_latent, t.key_list, t.value_list, nullptr, nullptr, t.actual_seq_q,
        t.actual_seq_kv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t.block_table, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t.q_rope, t.key_rope_cache_view,
        nullptr, nullptr, nullptr, nullptr, nullptr, heads_, scale, kFiaPreTokensAll, kFiaNextTokensCausal,
        kFiaLayout, 1, kFiaSparseModeBand, kFiaInnerPreciseHighPrecision, config_.block_size,
        kFiaAntiquantModeNone, false, kFiaAntiquantModeNone, kFiaAntiquantModeNone, kFiaQueryQuantModeNone,
        kFiaPseTypeNone, t.attn_out, t.softmax_lse);
    // Only call the public, signature-checked planner. Geometry stays fixed after Build.
    adopt(entry, workspace, executor);
  }
  // 10. quantize the latent attention output for the folded o_proj.
  {
    PipelineStage& entry = add("attn_quant", OpId::kDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<DynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.attn_flat, kSwigluAxisLast, kMxRoundModeRint,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t.attn_fp8,
        t.attn_mx_scale);
    adopt(entry, workspace, executor);
  }
  // 11. output projection, with W_UV folded in.
  {
    PipelineStage& entry = add("o_proj", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.attn_fp8, t.w_o, t.attn_mx_scale, t.w_o_scale, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, config_.dense_group_size, t.proj_out);
    adopt(entry, workspace, executor);
  }
  // 12. the attention sub-block's residual is NOT an add any more: the four
  //     streams absorb o_proj's output through `mhc_post_attn` (planned in
  //     PlanMhcStages), whose `hRes^T x` term carries the input forward. That
  //     is why `residual_attn` is gone rather than moved.
  //
  // 13. post-attention RMSNorm, quantized, for the expert and shared GEMMs.
  //     Reads the MoE round's own h_in -- each sub-block gets its own
  //     hyper-connection, so the MoE input is mixed from the streams afresh
  //     rather than inherited from the attention half.
  {
    PipelineStage& entry = add("post_norm_quant", OpId::kRmsNormDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<RmsNormDynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.mhc_moe.h_in, t.w_post_norm, nullptr, kRmsNormEpsilon, kSwigluScaleAlgOcp,
        kMxRoundModeRint, static_cast<int64_t>(kAclFloat8E4m3Fn), false, t.normed_fp8, t.normed_mx_scale, nullptr);
    adopt(entry, workspace, executor);
  }
  // 14. the same norm unquantized, because the router reads bf16.
  {
    PipelineStage& entry = add("post_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.mhc_moe.h_in,
                                                          t.w_post_norm, kRmsNormEpsilon, t.normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  // 15.-18. live in MoeRouterEngine (router matmul, sqrtsoftplus scoring,
  //     gating, dropless dispatch).
  //
  // 19.-21. the routed experts: planned by the injected IRoutedMoeBlock into
  //     this same table -- expert_gemm1 (+ expert_swiglu on the decomposed
  //     path), expert_gemm2 and expert_combine. The orchestrator no longer
  //     names a single grouped-GEMM entry point; the backend decides.
  moe_block_->PlanStages(stages_, arena_manager_, slot_addrs_);
  // 22-25. the shared expert: every token uses it, so it is never routed.
  {
    PipelineStage& entry = add("shared_gate_up", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.normed_fp8, t.w_shared_gate_up, t.normed_mx_scale, t.w_shared_gate_up_scale,
        nullptr, nullptr, nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.shared_gate_up);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_act", OpId::kSwiGlu);
    const uint64_t workspace =
        PlanAclnnOp<SwiGluPlanFn>(ops_, entry.op, &executor, t.shared_gate_up, -1, t.shared_act);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_act_quant", OpId::kDynamicMxQuant);
    const uint64_t workspace = PlanAclnnOp<DynamicMxQuantPlanFn>(
        ops_, entry.op, &executor, t.shared_act, kSwigluAxisLast, kMxRoundModeRint,
        static_cast<int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t.shared_act_fp8,
        t.shared_act_scale);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("shared_down", OpId::kQuantMatmulV5);
    const uint64_t workspace = PlanAclnnOp<QuantMatmulV5PlanFn>(
        ops_, entry.op, &executor, t.shared_act_fp8, t.w_shared_down, t.shared_act_scale, t.w_shared_down_scale,
        nullptr, nullptr, nullptr, nullptr, nullptr, false, true, config_.dense_group_size, t.shared_out);
    adopt(entry, workspace, executor);
  }
  // 26. routed + shared. The sum IS the MoE sub-block's output, so it is what
  //     `mhc_post_moe` folds back into the streams; the old `residual_moe`
  //     add is gone for the same reason `residual_attn` is.
  {
    PipelineStage& entry = add("add_shared", OpId::kInplaceAdd);
    const uint64_t workspace =
        PlanAclnnOp<InplaceAddPlanFn>(ops_, entry.op, &executor, t.routed_out, t.shared_out, nullptr);
    adopt(entry, workspace, executor);
  }
  // 27. the stream reduction. After 43 layers the hidden state is spread over
  //     four residual streams and the head takes one [1, 4096] vector, so the
  //     four are summed: stream 0 is copied into `hidden` (a D2D copy, no
  //     launch) and the other three are added onto it. Summation is the
  //     reduction the hyper-connection expansion is initialized against -- the
  //     n streams start as copies of the one residual, so their sum is what
  //     reduces to the original pre-mHC network at initialization.
  for (int64_t stream = 1; stream < kNhcStreams; ++stream) {
    static const char* const kReduceNames[kNhcStreams] = {"", "stream_reduce_1", "stream_reduce_2",
                                                          "stream_reduce_3"};
    PipelineStage& entry = add(kReduceNames[static_cast<size_t>(stream)], OpId::kInplaceAdd);
    const uint64_t workspace = PlanAclnnOp<InplaceAddPlanFn>(ops_, entry.op, &executor, t.hidden,
                                                             t.stream_slice[0][stream], nullptr);
    adopt(entry, workspace, executor);
  }
  // 28-30. the head.
  {
    PipelineStage& entry = add("final_norm", OpId::kRmsNorm);
    const uint64_t workspace = PlanAclnnOp<RmsNormPlanFn>(ops_, entry.op, &executor, t.hidden, t.w_final_norm,
                                                          kRmsNormEpsilon, t.final_normed, t.rstd);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("lm_head", OpId::kMatmul);
    const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.final_normed, t.w_lm_head,
                                                         t.logits, kCubeMathTypeKeepDtype);
    adopt(entry, workspace, executor);
  }
  {
    PipelineStage& entry = add("argmax", OpId::kArgMax);
    const uint64_t workspace = PlanAclnnOp<ArgMaxPlanFn>(ops_, entry.op, &executor, t.logits, -1, false, t.argmax);
    adopt(entry, workspace, executor);
  }
}

// ---------------------------------------------------------------------------
// The mHC residual chain: two rounds per layer, pure TND
// ---------------------------------------------------------------------------
//
// Per round: `aclnnMhcPre` mixes the four streams into hIn, the mixing map is
// normalized into B_l, the sub-block runs, and `aclnnMhcPost` folds its output
// back as `x_next = B_l^T x + hOut * hPost`.
//
// Pre and Post hold RETAINED, repeatable executors -- the 950PR run confirmed
// Repeatable=true for both at exactly these TND shapes. The normalization in
// between does NOT: see the MhcMixingMode note in config.hpp. On kPerUsePlan
// the Sinkhorn stage is only PROBED here, to measure its workspace and prove
// the geometry is accepted; the decode loop plans and launches it per use.
void Dsv4Pipeline::PlanMhcStages() {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  aclOpExecutor* executor = nullptr;

  struct RoundPlan {
    const char* pre;
    const char* sinkhorn;
    const char* post;
    const MhcRoundTensors* tensors;
  };
  const RoundPlan rounds[2] = {
      {"mhc_pre_attn", "mhc_sinkhorn_attn", "mhc_post_attn", &t.mhc_attn},
      {"mhc_pre_moe", "mhc_sinkhorn_moe", "mhc_post_moe", &t.mhc_moe},
  };

  for (const RoundPlan& round : rounds) {
    const MhcRoundTensors& set = *round.tensors;
    {
      PipelineStage& entry = stages_.Add(round.pre, OpId::kMhcPre);
      // The three optional outputs (invRms, hMix, hPre) stay null: nothing in
      // the kPerUsePlan chain reads them. They are what a kFusedExternal
      // producer would need, which is why they are called out here rather
      // than silently omitted.
      const uint64_t workspace = PlanAclnnOp<MhcPrePlanFn>(
          ops_, entry.op, &executor, t.residual_stream[0], t.w_mhc_phi, t.w_mhc_alpha, t.w_mhc_bias,
          t.w_mhc_gamma, kRmsNormEpsilon, kMhcHcEpsilon, set.h_in, set.h_post, set.h_res, nullptr, nullptr,
          nullptr);
      entry.slot.Adopt(entry.op, entry.name, workspace, executor);
      arena.NoteWorkspace(workspace);
    }
    {
      PipelineStage& entry = stages_.Add(round.sinkhorn, OpId::kMhcSinkhorn);
      // A PROBE plan: it establishes that the operator accepts this geometry
      // and measures the workspace the shared arena must cover, then is
      // destroyed. It is deliberately never adopted -- adopting calls
      // aclSetAclOpExecutorRepeatable, which this operator fails with 561000
      // on a 950PR and which would therefore abort Build.
      const uint64_t workspace = PlanAclnnOp<MhcSinkhornPlanFn>(ops_, entry.op, &executor, set.h_res,
                                                                kMhcSinkhornEps, kMhcSinkhornIters,
                                                                set.h_res_sink, nullptr, nullptr);
      DSV4_ACL_CHECK(aclDestroyAclOpExecutor(executor));
      executor = nullptr;
      entry.slot.NoteSingleUse(entry.op, entry.name, workspace);
      arena.NoteWorkspace(workspace);
    }
    {
      PipelineStage& entry = stages_.Add(round.post, OpId::kMhcPost);
      const uint64_t workspace = PlanAclnnOp<MhcPostPlanFn>(ops_, entry.op, &executor, t.residual_stream[0],
                                                            set.h_res_sink, set.h_out, set.h_post,
                                                            t.residual_stream[1]);
      entry.slot.Adopt(entry.op, entry.name, workspace, executor);
      arena.NoteWorkspace(workspace);
    }
  }
}

// ---------------------------------------------------------------------------
// The compressed attention cores
// ---------------------------------------------------------------------------
//
// Only the paths some layer's `compress_ratios` entry selects are planned. On
// a SWA-only checkpoint this function returns immediately and the compressor,
// the indexer and the shared-KV core cost nothing at all.
//
// EVERY STRIDE ATTRIBUTE HERE IS DERIVED, NEVER WRITTEN. `stateCacheStrideDim0`,
// `blockStride`, `cmpKvStride0` and the indexer's `stride` / `scaleStride` are
// read off the descriptor being bound, through DeriveDimensionStrideElements
// (which interrogates the handle with aclGetViewStrides / aclGetDataType). The
// vendored wrappers forward these to the kernel with no check of their own --
// hypotheses H3 and H4 record the attributes as UNGUARDED on the CANN backend
// -- so a hand-written constant that drifted from the view would address the
// wrong block with no diagnostic.
void Dsv4Pipeline::PlanCompressionStages() {
  if (!csa_planned_ && !hca_planned_) {
    return;
  }
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const MlaGeometry& mla = config_.mla;
  aclOpExecutor* executor = nullptr;

  const auto adopt = [&](PipelineStage& entry, uint64_t workspace, aclOpExecutor* planned) {
    entry.slot.Adopt(entry.op, entry.name, workspace, planned);
    arena.NoteWorkspace(workspace);
  };

  // Derived once, from the descriptors the stages below bind.
  const int64_t state_stride = DeriveDimensionStrideElements(t.cmp_state_cache, 0);
  const int64_t cmp_kv_stride = DeriveDimensionStrideElements(t.cmp_kv_cache, 0);

  // THE STRIDE TRAPS, CLOSED HERE RATHER THAN DISCOVERED ON DEVICE.
  //
  // Deriving a stride removes the risk of a stale CONSTANT, but not the risk
  // of a stale LAYOUT: the reservation arithmetic and the descriptor are
  // written in two different places, and if they ever disagree the operators
  // would address a cache that is laid out differently from the one the arena
  // sized. These three assertions are where that disagreement has to surface.
  DSV4_REQUIRE(static_cast<size_t>(DeriveDimensionStrideBytes(t.cmp_kv_cache, 0)) ==
                   static_cast<size_t>(config_.block_size) * sizeof(Dsv4CompressedKvEntry),
               "the compressed KV cache's axis-0 byte stride is "
                   << DeriveDimensionStrideBytes(t.cmp_kv_cache, 0) << " but one block holds "
                   << config_.block_size << " x " << sizeof(Dsv4CompressedKvEntry)
                   << " bytes. The descriptor and Dsv4CompressedKvEntry have diverged.");
  AssertDimensionStride(t.cmp_kv_cache, 1, kCompressedKvEntryBytes,
                        "the compressed KV cache's per-slot stride");
  if (csa_planned_) {
    // The indexer reads the key cache as FP8 [blocks, block_size, 1, 128] and
    // its epilog writes the same bytes as UINT8 [blocks, block_size, 128]. If
    // their block strides ever differed, the epilog would scatter into slots
    // the indexer never looks at -- silently, since both operators take the
    // stride as an unchecked attribute.
    DSV4_REQUIRE(DeriveDimensionStrideBytes(t.index_k_cache, 0) ==
                     DeriveDimensionStrideBytes(t.index_k_cache_u8, 0),
                 "the indexer key cache's two views disagree on their block stride: "
                     << DeriveDimensionStrideBytes(t.index_k_cache, 0) << " bytes as FP8 against "
                     << DeriveDimensionStrideBytes(t.index_k_cache_u8, 0)
                     << " bytes as UINT8, so the epilog would write where the indexer does not read");
  }

  // One compressor plan per ratio, plus one genuinely empty HOLD plan per
  // ratio. The hold plan is what a partial window launches: zero tokens in,
  // zero rows out, so the wrapper's `if (x->IsEmpty() || cmpKvOut->IsEmpty())`
  // early return fires before any l0 call and the step writes nothing -- not
  // into the compressed cache and not into the state ring (H3).
  const auto plan_compressor = [&](const char* name, int64_t ratio, const aclTensor* ape,
                                   const aclTensor* window, const aclTensor* rope_sin,
                                   const aclTensor* rope_cos, const aclTensor* destination) {
    PipelineStage& entry = stages_.Add(name, OpId::kCompressor);
    const uint64_t workspace = PlanAclnnOp<CompressorPlanFn>(
        ops_, entry.op, &executor, window, t.w_cmp_wkv, t.w_cmp_wgate, t.cmp_state_cache, ape,
        t.w_cmp_norm_weight, rope_sin, rope_cos, t.cmp_state_block_table, t.cmp_cu_seqlens, t.cmp_seqused,
        t.cmp_start_pos, mla.qk_rope_head_dim, ratio, kCompressorCoff, kRmsNormEpsilon,
        kCompressorRotaryModeHalf, kCompressorCacheModePaged, state_stride, destination);
    adopt(entry, workspace, executor);
  };

  if (csa_planned_) {
    plan_compressor("cmp_emit_csa", kCompressRatioCsa, t.w_cmp_ape_csa, t.cmp_window_csa, t.cmp_rope_sin_csa,
                    t.cmp_rope_cos_csa, t.cmp_kv_out);
    plan_compressor("cmp_hold_csa", kCompressRatioCsa, t.w_cmp_ape_csa, t.cmp_window_empty,
                    t.cmp_rope_sin_empty, t.cmp_rope_cos_empty, t.cmp_kv_out_empty);
  }
  if (hca_planned_) {
    plan_compressor("cmp_emit_hca", kCompressRatioHca, t.w_cmp_ape_hca, t.cmp_window_hca, t.cmp_rope_sin_hca,
                    t.cmp_rope_cos_hca, t.cmp_kv_out);
    plan_compressor("cmp_hold_hca", kCompressRatioHca, t.w_cmp_ape_hca, t.cmp_window_empty,
                    t.cmp_rope_sin_empty, t.cmp_rope_cos_empty, t.cmp_kv_out_empty);
  }

  // The emitted BF16 row, quantized and scattered into its 604-byte slot.
  {
    PipelineStage& entry = stages_.Add("cmp_kv_scatter", OpId::kKvCompressEpilog);
    const uint64_t workspace = PlanAclnnOp<KvCompressEpilogPlanFn>(
        ops_, entry.op, &executor, t.cmp_kv_cache, t.cmp_kv_out, t.cmp_slot_mapping, kCompressEpilogQuantGroup,
        kCompressEpilogQuantMode, kCompressEpilogRoundScalePow2, kCompressEpilogLayoutPaged, cmp_kv_stride);
    adopt(entry, workspace, executor);
  }

  const double softmax_scale = 1.0 / std::sqrt(static_cast<double>(mla.q_head_dim()));
  const auto plan_sparse_core = [&](const char* name, int64_t ratio, const aclTensor* indices) {
    PipelineStage& entry = stages_.Add(name, OpId::kKvQuantSparseAttnSharedkv);
    // oriKv is left UNBOUND, and that is a limitation, not an oversight: this
    // operator requires the uncompressed half of the hybrid cache in FP8
    // E4M3, while this engine's paged latent cache is BF16. Binding it is a
    // cache-layout change, not a binding change, so the compressed layers
    // attend their compressed stream only. DescribeStages says so.
    const uint64_t workspace = PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
        ops_, entry.op, &executor, t.sparse_q, /*oriKv=*/nullptr, t.cmp_kv_cache,
        /*oriSparseIndices=*/nullptr, indices, /*oriBlockTable=*/nullptr, t.cmp_block_table,
        /*cuSeqlensQ=*/nullptr, /*cuSeqlensOriKv=*/nullptr, t.cmp_seq_k, /*sequsedQ=*/nullptr, t.cmp_seq_k,
        /*sinks=*/nullptr, /*metadata=*/nullptr, kSparseAttnKvQuantMode, kSparseAttnTileSize,
        mla.qk_rope_head_dim, softmax_scale, ratio, kSparseAttnOriMaskSwa, kSparseAttnCmpMaskCausal,
        kSlidingWindowLeft, kSlidingWindowRight, const_cast<char*>(kSparseAttnLayoutQ),
        const_cast<char*>(kSparseAttnLayoutKv), /*oriKvStride0=*/0, cmp_kv_stride,
        /*returnSoftmaxLse=*/false, t.sparse_attn_out, t.sparse_lse_empty);
    adopt(entry, workspace, executor);
  };

  if (csa_planned_) {
    // The indexer query: a projection of the q-LoRA state, quantized to the
    // FP8 E4M3 stream the indexer scores.
    {
      PipelineStage& entry = stages_.Add("index_q_proj", OpId::kMatmul);
      const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.q_a, t.w_index_q,
                                                           t.index_q_bf16, kCubeMathTypeKeepDtype);
      adopt(entry, workspace, executor);
    }
    {
      PipelineStage& entry = stages_.Add("index_q_quant", OpId::kDynamicMxQuant);
      const uint64_t workspace = PlanAclnnOp<DynamicMxQuantPlanFn>(
          ops_, entry.op, &executor, t.index_q_bf16, kSwigluAxisLast, kMxRoundModeRint,
          static_cast<int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t.index_q_fp8,
          t.index_q_mx_scale);
      adopt(entry, workspace, executor);
    }
    // The indexer key: a projection of the compressed entry the window just
    // emitted, scattered into the indexer's own paged cache at the same slot.
    {
      PipelineStage& entry = stages_.Add("index_k_proj", OpId::kMatmul);
      const uint64_t workspace = PlanAclnnOp<MatmulPlanFn>(ops_, entry.op, &executor, t.cmp_kv_out, t.w_index_k,
                                                           t.index_k_bf16, kCubeMathTypeKeepDtype);
      adopt(entry, workspace, executor);
    }
    {
      PipelineStage& entry = stages_.Add("index_k_scatter", OpId::kIndexerCompressEpilogV2);
      const uint64_t workspace = PlanAclnnOp<IndexerCompressEpilogV2PlanFn>(
          ops_, entry.op, &executor, t.index_k_cache_u8, t.index_k_bf16, t.cmp_slot_mapping,
          kCompressEpilogLayoutPaged, DeriveDimensionStrideElements(t.index_k_cache_u8, 0));
      adopt(entry, workspace, executor);
    }
    {
      PipelineStage& entry = stages_.Add("indexer", OpId::kVllmQuantLightningIndexer);
      const uint64_t workspace = PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
          ops_, entry.op, &executor, t.index_q, t.index_k_cache, t.index_head_weights, t.index_q_dequant,
          t.index_k_dequant, /*actualSeqLengthsQuery=*/nullptr, t.cmp_seq_k, t.cmp_block_table,
          /*metadata=*/nullptr, kIndexerQuantModePerTokenHead, kIndexerQuantModePerTokenHead,
          const_cast<char*>(kIndexerLayoutQuery), const_cast<char*>(kIndexerLayoutKeyPaged), kIndexTopK,
          kIndexerSparseModeCausal, INT64_MAX, INT64_MAX, kCompressRatioCsa, /*returnValues=*/false,
          DeriveDimensionStrideElements(t.index_k_cache, 0),
          DeriveDimensionStrideElements(t.index_k_dequant, 0), t.index_sparse_indices,
          t.index_sparse_values_empty);
      adopt(entry, workspace, executor);
    }
    plan_sparse_core("sparse_attn_csa", kCompressRatioCsa, t.index_sparse_indices);
  }
  if (hca_planned_) {
    // No indexer on the HCA path: at 128:1 the compressed stream is already
    // shorter than the top-512 window, so the identity selection written at
    // Build IS "attend everything there is".
    plan_sparse_core("sparse_attn_hca", kCompressRatioHca, t.cmp_sparse_indices_dense);
  }
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

void Dsv4Pipeline::Launch(PipelineStage& entry) {
  entry.slot.Launch(ops_, arena_manager_.arena().workspace(), compute_stream_);
  ++counters_.launches;
}

// ---------------------------------------------------------------------------
// The mHC half of a sub-block
// ---------------------------------------------------------------------------

void Dsv4Pipeline::RunMhcPre(const MhcRound& round, const MhcRoundTensors& tensors,
                             const BackboneWeights::Layer::MhcWeights& weights) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  PipelineStage& pre = stages_.stage(round.pre);

  // This round's weights and the live stream buffer.
  pre.slot.SetAddress(slot::kMhcPreX, t.residual_stream[0], arena.Address(t.h_residual_stream[stream_parity_]));
  pre.slot.SetAddress(slot::kMhcPrePhi, t.w_mhc_phi, arena.Address(weights.phi));
  pre.slot.SetAddress(slot::kMhcPreAlpha, t.w_mhc_alpha, arena.Address(weights.alpha));
  pre.slot.SetAddress(slot::kMhcPreBias, t.w_mhc_bias, arena.Address(weights.bias));
  pre.slot.SetAddress(slot::kMhcPreGamma, t.w_mhc_gamma, arena.Address(weights.gamma));
  Launch(pre);

  // B_l. On kFusedExternal the slot is expected to already hold a doubly
  // stochastic map from a fused producer, so nothing is launched here.
  if (config_.mhc_mixing_mode == MhcMixingMode::kFusedExternal) {
    return;
  }
  // kPerUsePlan: plan and launch the standalone operator, its executor
  // consumed by its own launch. This is a HOST PLAN INSIDE THE DECODE LOOP,
  // which the rest of this engine goes to some length to avoid -- see the
  // MhcMixingMode note in config.hpp for why there is no alternative on this
  // toolkit, and `sinkhorn_replans` for the count.
  const PipelineStage& sinkhorn = stages_.stage(round.sinkhorn);
  aclOpExecutor* executor = nullptr;
  const uint64_t workspace =
      PlanAclnnOp<MhcSinkhornPlanFn>(ops_, sinkhorn.op, &executor, tensors.h_res, kMhcSinkhornEps,
                                     kMhcSinkhornIters, tensors.h_res_sink, nullptr, nullptr);
  arena.AssertWorkspaceFits(workspace, sinkhorn.name);
  sinkhorn.slot.LaunchSingleUse(ops_, arena.workspace(), compute_stream_, executor, workspace);
  ++counters_.launches;
  ++counters_.sinkhorn_replans;
}

void Dsv4Pipeline::RunMhcPost(const MhcRound& round) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  PipelineStage& post = stages_.stage(round.post);

  // Read the live stream, write the other one: no mHC call ever aliases its
  // own input, so the executor stays reusable across all 86 rounds of a step.
  const size_t source = stream_parity_;
  const size_t destination = 1 - stream_parity_;
  post.slot.SetAddress(slot::kMhcPostX, t.residual_stream[0], arena.Address(t.h_residual_stream[source]));
  post.slot.SetAddress(slot::kMhcPostOut, t.residual_stream[1], arena.Address(t.h_residual_stream[destination]));
  Launch(post);
  stream_parity_ = destination;
  ++counters_.mhc_rounds;
}

void Dsv4Pipeline::RunAttention(int32_t layer, int64_t position) {
  const MlaGeometry& mla = config_.mla;
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  const BackboneWeights::Layer& layer_weights = weights.layers[static_cast<size_t>(layer)];

  // Repoint this layer's weights. The descriptors and the executors are the
  // ones planned at init; only the addresses move.
  stages_.stage("input_norm_quant")
      .slot.SetAddress(slot::kMxQuantGamma, t.w_input_norm, arena.Address(layer_weights.input_norm));
  stages_.stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_a, arena.Address(layer_weights.q_a_weight));
  stages_.stage("q_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_a_scale, arena.Address(layer_weights.q_a_scale));
  stages_.stage("q_a_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_q_a_norm, arena.Address(layer_weights.q_a_norm));
  stages_.stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2, t.w_q_b, arena.Address(layer_weights.q_b_weight));
  stages_.stage("q_b_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_q_b_scale, arena.Address(layer_weights.q_b_scale));
  stages_.stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2, t.w_kv_a, arena.Address(layer_weights.kv_a_weight));
  stages_.stage("kv_a_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_kv_a_scale, arena.Address(layer_weights.kv_a_scale));
  stages_.stage("kv_a_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_kv_a_norm, arena.Address(layer_weights.kv_a_norm));
  stages_.stage("o_proj").slot.SetAddress(slot::kQuantMmX2, t.w_o, arena.Address(layer_weights.o_weight));
  stages_.stage("o_proj").slot.SetAddress(slot::kQuantMmX2Scale, t.w_o_scale, arena.Address(layer_weights.o_scale));

  // This layer's slice of the paged cache.
  const size_t latent_stride = Bf16Bytes(arena_manager_.num_blocks() * config_.block_size * mla.kv_lora_rank);
  const size_t rope_stride = Bf16Bytes(arena_manager_.num_blocks() * config_.block_size * mla.qk_rope_head_dim);
  uint8_t* latent =
      arena.AddressAs<uint8_t>(weights.kv_latent_cache) + latent_stride * static_cast<size_t>(layer);
  uint8_t* rope = arena.AddressAs<uint8_t>(weights.kv_rope_cache) + rope_stride * static_cast<size_t>(layer);
  stages_.stage("kv_cache_write").slot.SetAddress(slot::kScatterKeyCache, t.kv_latent_cache, latent);
  stages_.stage("kv_cache_write").slot.SetAddress(slot::kScatterValueCache, t.kv_rope_cache, rope);
  // The dense core's cache bindings, only when this layer will actually run
  // it. Guarding on the PATH as well as on `swa_planned_` matters twice over:
  // an all-compressed checkpoint plans no `attention` stage at all, so an
  // unconditional lookup here would throw, and a compressed layer would
  // otherwise pay three address swaps for a core it never launches.
  if (swa_planned_ && config_.attention_path(layer) == AttentionPath::kSlidingWindow) {
    PipelineStage& dense = stages_.stage("attention");
    dense.slot.SetTensorListAddress(slot::kFiaKeyList, 0, t.key_list, latent);
    dense.slot.SetTensorListAddress(slot::kFiaValueList, 0, t.value_list, latent);
    dense.slot.SetAddress(slot::kFiaKeyRope, t.key_rope_cache_view, rope);
  }

  // The rope table row for this position.
  uint8_t* cos_row = arena.AddressAs<uint8_t>(weights.rope_cos) + Bf16Bytes(position * mla.qk_rope_head_dim);
  uint8_t* sin_row = arena.AddressAs<uint8_t>(weights.rope_sin) + Bf16Bytes(position * mla.qk_rope_head_dim);
  stages_.stage("rope").slot.SetAddress(slot::kRotaryCos, t.rope_cos, cos_row);
  stages_.stage("rope").slot.SetAddress(slot::kRotarySin, t.rope_sin, sin_row);

  // The prologue is shared by all three attention paths: every layer projects
  // q and kv, applies the partial RoPE and writes the uncompressed paged
  // cache, whatever its compression ratio.
  Launch(stages_.stage("input_norm_quant"));
  Launch(stages_.stage("q_a_proj"));
  Launch(stages_.stage("q_a_norm_quant"));
  Launch(stages_.stage("q_b_proj"));
  Launch(stages_.stage("kv_a_proj"));
  Launch(stages_.stage("kv_a_norm"));
  Launch(stages_.stage("rope"));
  Launch(stages_.stage("kv_cache_write"));

  // Only the CORE differs per layer.
  RunAttentionCore(layer, position);

  // And the epilogue is shared again: both cores leave their result in
  // act.attn_out, which attn_quant and the folded o_proj consume unchanged.
  Launch(stages_.stage("attn_quant"));
  Launch(stages_.stage("o_proj"));
}

// ---------------------------------------------------------------------------
// Attention core dispatch: the layer topology, straight from compress_ratios
// ---------------------------------------------------------------------------

void Dsv4Pipeline::RunAttentionCore(int32_t layer, int64_t position) {
  const AttentionPath path = config_.attention_path(layer);
  switch (path) {
    case AttentionPath::kCompressedSparse:
      ++counters_.csa_layers;
      RunCompressedAttention(layer, position, path);
      return;
    case AttentionPath::kHyperCompressed:
      ++counters_.hca_layers;
      RunCompressedAttention(layer, position, path);
      return;
    case AttentionPath::kSlidingWindow:
      ++counters_.swa_layers;
      RunSlidingWindowAttention();
      return;
  }
}

void Dsv4Pipeline::RunSlidingWindowAttention() { Launch(stages_.stage("attention")); }

// The compressor cadence. Returns true when this step closed a window and a
// Dsv4CompressedKvEntry reached the paged cache.
//
// A PARTIAL WINDOW NEVER TOUCHES THE PAGED CACHE. The only state it advances
// is the rolling ring buffer -- a plain D2D copy of this layer's h_in into
// slot (position % ratio), no launch, no descriptor, no mapping -- and then
// the hold stage, whose empty cmpKvOut makes the wrapper return before any l0
// call. The ring is where the window lives because the operator cannot hold
// it: cmpKvOut's row count is pinned at T / cmpRatio, so a one-token call can
// never accumulate toward a window of 4 or 128 (hypotheses H3).
bool Dsv4Pipeline::AdvanceCompressorCadence(int32_t layer, int64_t position, int64_t ratio,
                                            AttentionPath path) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  const BackboneWeights::Layer& layer_weights = weights.layers[static_cast<size_t>(layer)];
  const MlaGeometry& mla = config_.mla;
  const bool csa = path == AttentionPath::kCompressedSparse;

  // This layer's own slice of every per-layer compression buffer.
  uint8_t* window = LayerSlice(weights.cmp_window, arena_manager_.CompressedWindowLayerStrideBytes(), layer);
  uint8_t* state = LayerSlice(weights.cmp_state_cache, arena_manager_.CompressedStateLayerStrideBytes(), layer);
  uint8_t* cache = LayerSlice(weights.cmp_kv_cache, arena_manager_.CompressedKvLayerStrideBytes(), layer);

  // 1. the ring update, every step, for every compressed layer.
  const int64_t window_position = position % ratio;
  streams_.MemcpyAsync(window + Bf16Bytes(window_position * kHiddenSize), Bf16Bytes(kHiddenSize),
                       arena.Address(t.mhc_attn.h_h_in), Bf16Bytes(kHiddenSize),
                       MemcpyKind::kDeviceToDevice, compute_stream_);

  const char* emit_stage = csa ? "cmp_emit_csa" : "cmp_emit_hca";
  const char* hold_stage = csa ? "cmp_hold_csa" : "cmp_hold_hca";
  const aclTensor* window_descriptor = csa ? t.cmp_window_csa : t.cmp_window_hca;

  // 2. the cadence itself, exactly as the brief states it.
  const bool is_emission = ((position + 1) % ratio == 0);
  if (!is_emission) {
    // The hold stage: a validated, completely EMPTY executor. The wrapper's
    // empty-tensor early return fired at plan time, so it staged no kernel and
    // registered no tensors -- which is why nothing is rebound here, not even
    // the state ring. There is nothing in it to repoint, and nothing it can
    // write: no allocation, no mapping, and no touch of the paged cache.
    Launch(stages_.stage(hold_stage));
    ++counters_.compressor_holds;
    return false;
  }

  // The state ring is a REF parameter: the emitting plan updates this layer's
  // slice in place.
  PipelineStage& emit_stage_entry = stages_.stage(emit_stage);
  emit_stage_entry.slot.SetAddress(slot::kCompressorStateCache, t.cmp_state_cache, state);

  // 3. the window closed. Bind the destination slot and the window's weights,
  //    then emit.
  const int64_t compressed_length = (position + 1) / ratio;
  const int64_t compressed_slot = compressed_length - 1;
  DSV4_REQUIRE(compressed_slot < arena_manager_.compressed_slots(),
               "the compressed KV cache holds " << arena_manager_.compressed_slots()
                                                << " entries but layer " << layer << " at position " << position
                                                << " (ratio " << ratio << ") wants slot " << compressed_slot);

  PipelineStage& emit = emit_stage_entry;
  emit.slot.SetAddress(slot::kCompressorX, const_cast<aclTensor*>(window_descriptor), window);
  emit.slot.SetAddress(slot::kCompressorWkv, t.w_cmp_wkv, arena.Address(layer_weights.cmp_wkv));
  emit.slot.SetAddress(slot::kCompressorWgate, t.w_cmp_wgate, arena.Address(layer_weights.cmp_wgate));
  emit.slot.SetAddress(slot::kCompressorNormWeight, t.w_cmp_norm_weight,
                       arena.Address(layer_weights.cmp_norm_weight));
  aclTensor* ape = csa ? t.w_cmp_ape_csa : t.w_cmp_ape_hca;
  emit.slot.SetAddress(slot::kCompressorApe, ape, arena.Address(layer_weights.cmp_ape));

  // The window spans positions [position - ratio + 1, position], which are
  // consecutive rows of the rope tables -- so the [ratio, rope] slice IS a
  // view at the window's first row and no copy is needed.
  const int64_t window_start = position - ratio + 1;
  uint8_t* cos_row = arena.AddressAs<uint8_t>(weights.rope_cos) + Bf16Bytes(window_start * mla.qk_rope_head_dim);
  uint8_t* sin_row = arena.AddressAs<uint8_t>(weights.rope_sin) + Bf16Bytes(window_start * mla.qk_rope_head_dim);
  emit.slot.SetAddress(slot::kCompressorRopeCos,
                       csa ? t.cmp_rope_cos_csa : t.cmp_rope_cos_hca, cos_row);
  emit.slot.SetAddress(slot::kCompressorRopeSin,
                       csa ? t.cmp_rope_sin_csa : t.cmp_rope_sin_hca, sin_row);

  // The window metadata and the destination slot: one 12-byte and two 4-byte
  // H2D copies from pinned mailboxes, so nothing depends on a stack lifetime.
  cmp_window_mailbox_[0] = static_cast<int32_t>(ratio);         // cuSeqlens
  cmp_window_mailbox_[1] = static_cast<int32_t>(ratio);         // seqused
  cmp_window_mailbox_[2] = static_cast<int32_t>(window_start);  // startPos
  streams_.MemcpyAsync(arena.Address(weights.cmp_window_meta), Int32Bytes(3), cmp_window_mailbox_,
                       Int32Bytes(3), MemcpyKind::kHostToDevice, compute_stream_);
  *cmp_slot_mailbox_ = static_cast<int32_t>(compressed_slot);
  *cmp_seq_mailbox_ = static_cast<int32_t>(compressed_length);
  streams_.MemcpyAsync(arena.Address(weights.cmp_slot_mapping), Int32Bytes(1), cmp_slot_mailbox_, Int32Bytes(1),
                       MemcpyKind::kHostToDevice, compute_stream_);
  streams_.MemcpyAsync(arena.Address(weights.cmp_seq_k), Int32Bytes(1), cmp_seq_mailbox_, Int32Bytes(1),
                       MemcpyKind::kHostToDevice, compute_stream_);

  Launch(emit);
  ++counters_.compressor_emissions;

  // 4. quantize the emitted row and scatter it into its 604-byte slot. THIS,
  //    and only this, is what writes the paged compressed cache.
  PipelineStage& scatter = stages_.stage("cmp_kv_scatter");
  scatter.slot.SetAddress(slot::kCompressEpilogCacheRef, t.cmp_kv_cache, cache);
  Launch(scatter);
  ++counters_.compressed_entries_written;
  return true;
}

void Dsv4Pipeline::RunCompressedAttention(int32_t layer, int64_t position, AttentionPath path) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  const BackboneWeights::Layer& layer_weights = weights.layers[static_cast<size_t>(layer)];
  const bool csa = path == AttentionPath::kCompressedSparse;
  const int64_t ratio = csa ? kCompressRatioCsa : kCompressRatioHca;

  uint8_t* cache = LayerSlice(weights.cmp_kv_cache, arena_manager_.CompressedKvLayerStrideBytes(), layer);
  const bool emitted = AdvanceCompressorCadence(layer, position, ratio, path);

  if (csa) {
    // The indexer key follows the compressed entry: it is a projection OF that
    // entry, so it only exists on an emitting step and lands in the same slot.
    if (emitted) {
      uint8_t* index_cache =
          LayerSlice(weights.index_k_cache, arena_manager_.IndexerKeyLayerStrideBytes(), layer);
      PipelineStage& key_proj = stages_.stage("index_k_proj");
      key_proj.slot.SetAddress(slot::kMatmulMat2, t.w_index_k, arena.Address(layer_weights.index_k_weight));
      Launch(key_proj);
      PipelineStage& key_scatter = stages_.stage("index_k_scatter");
      key_scatter.slot.SetAddress(slot::kCompressEpilogCacheRef, t.index_k_cache_u8, index_cache);
      Launch(key_scatter);
    }
    // The query is this token's, so it is projected and quantized every step.
    PipelineStage& query_proj = stages_.stage("index_q_proj");
    query_proj.slot.SetAddress(slot::kMatmulMat2, t.w_index_q, arena.Address(layer_weights.index_q_weight));
    Launch(query_proj);
    Launch(stages_.stage("index_q_quant"));

    PipelineStage& indexer = stages_.stage("indexer");
    uint8_t* index_cache =
        LayerSlice(weights.index_k_cache, arena_manager_.IndexerKeyLayerStrideBytes(), layer);
    uint8_t* index_scale =
        LayerSlice(weights.index_k_dequant, arena_manager_.IndexerKeyScaleLayerStrideBytes(), layer);
    indexer.slot.SetAddress(slot::kIndexerKey, t.index_k_cache, index_cache);
    indexer.slot.SetAddress(slot::kIndexerKeyDequantScale, t.index_k_dequant, index_scale);
    indexer.slot.SetAddress(slot::kIndexerWeights, t.index_head_weights,
                            arena.Address(layer_weights.index_head_weight));
    Launch(indexer);
    ++counters_.indexer_selections;
  }

  PipelineStage& core = stages_.stage(csa ? "sparse_attn_csa" : "sparse_attn_hca");
  core.slot.SetAddress(slot::kSparseAttnCmpKv, t.cmp_kv_cache, cache);
  Launch(core);
}

void Dsv4Pipeline::RunSharedExpert(int32_t layer) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights::Layer& weights = arena_manager_.backbone().layers[static_cast<size_t>(layer)];
  stages_.stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_gate_up, arena.Address(weights.shared_gate_up_weight));
  stages_.stage("shared_gate_up")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_gate_up_scale,
                       arena.Address(weights.shared_gate_up_scale));
  stages_.stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2, t.w_shared_down, arena.Address(weights.shared_down_weight));
  stages_.stage("shared_down")
      .slot.SetAddress(slot::kQuantMmX2Scale, t.w_shared_down_scale, arena.Address(weights.shared_down_scale));
  Launch(stages_.stage("shared_gate_up"));
  Launch(stages_.stage("shared_act"));
  Launch(stages_.stage("shared_act_quant"));
  Launch(stages_.stage("shared_down"));
}

void Dsv4Pipeline::RunMoe(int32_t layer) {
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights::Layer& weights = arena_manager_.backbone().layers[static_cast<size_t>(layer)];

  stages_.stage("post_norm_quant").slot.SetAddress(slot::kMxQuantGamma, t.w_post_norm, arena.Address(weights.post_norm));
  stages_.stage("post_norm").slot.SetAddress(slot::kRmsNormGamma, t.w_post_norm, arena.Address(weights.post_norm));

  Launch(stages_.stage("post_norm_quant"));
  Launch(stages_.stage("post_norm"));

  // Score, select, and read the top-6 back (deviation 1: the one forced host
  // round trip per MoE layer).
  const int32_t* expert_ids =
      router_.ScoreAndSelect(layer, arena_manager_, ops_, weights, compute_stream_, compute_done_);
  ++counters_.host_synchronizations;

  // The exclusive hierarchy makes the six chosen experts resident, batching
  // every miss into one event-ordered duplex exchange before the GEMM. The
  // residency plan travels into the dispatch context: binding the weight
  // lists is the backend's job now.
  const LayerSwapPlan plan = experts_.PrepareLayer(layer, expert_ids, static_cast<int32_t>(kNumExpertsPerTok),
                                                   compute_stream_, compute_done_);
  counters_.expert_slot_hits += static_cast<uint64_t>(plan.hit_count);
  counters_.expert_slot_misses += static_cast<uint64_t>(plan.miss_count);

  MoeDispatchContext dispatch;
  dispatch.layer = layer;
  dispatch.token_count = kTokensPerStep;
  dispatch.top_k = kNumExpertsPerTok;
  dispatch.expanded_x = t.expanded_x;
  dispatch.expanded_scale = t.expanded_scale;
  dispatch.group_list = t.group_list;
  dispatch.gemm1_out = t.gemm1_out;
  dispatch.gemm1_scale = t.gemm1_scale;
  dispatch.gemm2_out = t.gemm2_out;
  dispatch.expanded_weights_row = t.expanded_weights_row;
  dispatch.routed_out = t.routed_out;
  dispatch.swap_plan = &plan;
  dispatch.compute_stream = compute_stream_;

  const uint64_t block_launches_before = moe_block_->launches();
  moe_block_->ExecuteMoe(dispatch, streams_);
  counters_.launches += moe_block_->launches() - block_launches_before;

  RunSharedExpert(layer);
  // routed + shared IS the MoE sub-block's output; `mhc_post_moe` (launched by
  // the caller) is what folds it back into the four streams, so there is no
  // residual add here any more.
  Launch(stages_.stage("add_shared"));

  // Publish the compute boundary: the next layer's swap batch waits on it
  // before overwriting any slot this layer's GEMMs read.
  streams_.RecordEvent(compute_done_, compute_stream_);
  ++counters_.layers;
}

void Dsv4Pipeline::DecodeStep(int32_t token_id, int64_t position) {
  DSV4_REQUIRE(arena_manager_.arena().sealed(), "DecodeStep before Build()");
  DSV4_REQUIRE(token_id >= 0 && token_id < kVocabSize, "token id " << token_id << " outside the vocabulary");
  DSV4_REQUIRE(position >= 0 && position < config_.max_context_len,
               "position " << position << " outside the reserved context of " << config_.max_context_len);

  const uint64_t allocations_before = allocator_.DeviceAllocationCount();
  const size_t descriptors_before = arena_manager_.arena().descriptors().size();

  // Embedding lookup for a single greedy token is one row copy, not a gather
  // kernel: no descriptor, no workspace, no launch.
  ArenaTensors& t = arena_manager_.tensors();
  StaticMemoryArena& arena = arena_manager_.arena();
  const BackboneWeights& weights = arena_manager_.backbone();
  uint8_t* embed_row =
      arena.AddressAs<uint8_t>(weights.embed_tokens) + Bf16Bytes(static_cast<int64_t>(token_id) * kHiddenSize);

  // The mHC stream starts the step as four copies of the embedding. That is
  // the hyper-connection expansion at its initialization: n identical streams
  // whose sum is the original residual, which is what makes the stream
  // reduction at the head a plain sum. Four row copies, no launch.
  stream_parity_ = 0;
  uint8_t* stream_base = arena.AddressAs<uint8_t>(t.h_residual_stream[stream_parity_]);
  for (int64_t stream = 0; stream < kNhcStreams; ++stream) {
    streams_.MemcpyAsync(stream_base + Bf16Bytes(stream * kHiddenSize), Bf16Bytes(kHiddenSize), embed_row,
                         Bf16Bytes(kHiddenSize), MemcpyKind::kDeviceToDevice, compute_stream_);
  }

  // The slot this token occupies in the paged cache. The pinned mailbox is the
  // DMA source, so the copy can be async without a stack lifetime problem.
  *slot_mailbox_ = static_cast<int32_t>(position);
  streams_.MemcpyAsync(arena.Address(t.h_slot_mapping), Int32Bytes(1), slot_mailbox_, Int32Bytes(1),
                       MemcpyKind::kHostToDevice, compute_stream_);

  static const MhcRound kAttnRound{"mhc_pre_attn", "mhc_sinkhorn_attn", "mhc_post_attn"};
  static const MhcRound kMoeRound{"mhc_pre_moe", "mhc_sinkhorn_moe", "mhc_post_moe"};

  for (int32_t layer = 0; layer < static_cast<int32_t>(kNumLayers); ++layer) {
    const BackboneWeights::Layer& layer_weights = weights.layers[static_cast<size_t>(layer)];
    // Two hyper-connections per layer: the streams are mixed into a fresh
    // sub-block input before attention and again before the MoE, and each
    // sub-block's output is folded straight back into them.
    RunMhcPre(kAttnRound, t.mhc_attn, layer_weights.mhc_attn);
    RunAttention(layer, position);
    RunMhcPost(kAttnRound);

    RunMhcPre(kMoeRound, t.mhc_moe, layer_weights.mhc_moe);
    RunMoe(layer);
    RunMhcPost(kMoeRound);
  }

  // Reduce the four streams to the one [1, 4096] vector the head takes:
  // stream 0 copied in, the other three added on. The operands are repointed
  // from the LIVE buffer rather than assuming the 86 posts left the parity
  // even, so an odd layer or sub-block count could never read a stale stream.
  uint8_t* live = arena.AddressAs<uint8_t>(t.h_residual_stream[stream_parity_]);
  streams_.MemcpyAsync(arena.Address(t.h_hidden), Bf16Bytes(kHiddenSize), live, Bf16Bytes(kHiddenSize),
                       MemcpyKind::kDeviceToDevice, compute_stream_);
  for (int64_t stream = 1; stream < kNhcStreams; ++stream) {
    static const char* const kReduceNames[kNhcStreams] = {"", "stream_reduce_1", "stream_reduce_2",
                                                          "stream_reduce_3"};
    PipelineStage& entry = stages_.stage(kReduceNames[static_cast<size_t>(stream)]);
    entry.slot.SetAddress(slot::kInplaceAddOther, t.stream_slice[0][stream],
                          live + Bf16Bytes(stream * kHiddenSize));
    Launch(entry);
  }

  Launch(stages_.stage("final_norm"));
  Launch(stages_.stage("lm_head"));
  Launch(stages_.stage("argmax"));

  ++counters_.steps;
  diagnostics_.decoded_steps = counters_.steps;
  diagnostics_.moe_cache = experts_.cache_stats();
  diagnostics_.paged_attention.active_context_tokens = static_cast<uint64_t>(position + 1);
  diagnostics_.paged_attention.kv_cache_utilization = static_cast<double>(position + 1) /
      static_cast<double>(arena_manager_.num_blocks() * config_.block_size);
  counters_.device_allocations_in_step += allocator_.DeviceAllocationCount() - allocations_before;
  counters_.descriptors_built_in_step += arena_manager_.arena().descriptors().size() - descriptors_before;
}

int32_t Dsv4Pipeline::ReadArgmaxToken() {
  ArenaTensors& t = arena_manager_.tensors();
  // The one synchronization that ends a step.
  streams_.MemcpyAsync(token_mailbox_, Int64Bytes(1), arena_manager_.arena().Address(t.h_argmax), Int64Bytes(1),
                       MemcpyKind::kDeviceToHost, compute_stream_);
  streams_.SynchronizeStream(compute_stream_);
  ++counters_.host_synchronizations;
  const int64_t token = *token_mailbox_;
  DSV4_REQUIRE(token >= 0 && token < kVocabSize,
               "the LM head returned token " << token << ", outside the vocabulary");
  return static_cast<int32_t>(token);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

std::string Dsv4Pipeline::DescribeStages() const {
  std::ostringstream out;
  // Counted from the stages that were really planned, not from the containers'
  // capacity: the router's vector is sized to its ceiling and only partly
  // filled. The header is emitted after the tables for that reason.
  size_t live_router_stages = 0;
  std::ostringstream table;
  uint64_t high_water = 0;
  const auto describe_stage = [&table, &high_water](const PipelineStage& entry) {
    table << "  " << std::left << std::setw(20) << entry.name << std::right << std::setw(40)
          << OpName(entry.op) << "  workspace=" << std::setw(10) << entry.slot.workspace_size()
          << (entry.slot.single_use() ? "  (single-use plan)" : "") << "\n";
    high_water = std::max(high_water, entry.slot.workspace_size());
  };
  for (size_t i = 0; i < stages_.size(); ++i) {
    describe_stage(stages_[i]);
  }
  // The router hands back its RAW vector, which -- like every PipelineStage
  // container here -- is born at full capacity because StaticOpSlot cannot be
  // moved. Its unfilled entries carry `name == nullptr`, and streaming a null
  // `const char*` sets badbit, which since C++11 silently discards EVERY
  // subsequent insertion. That is what used to swallow the whole report below
  // this table, the rope-tables correctness warning included, so the guard is
  // not defensive noise: it is the fix.
  for (const PipelineStage& entry : router_.stages()) {
    if (entry.name == nullptr) {
      continue;
    }
    describe_stage(entry);
    ++live_router_stages;
  }
  out << "pipeline: " << (stages_.size() + live_router_stages) << " planned stages (" << stages_.size()
      << " in the orchestrator table, " << live_router_stages
      << " in the router engine), replayed " << kNumLayers << " times with address swaps\n";
  out << "  moe block:    " << moe_block_->DescribeBackend() << "\n";
  out << table.str();
  out << "  shared workspace high-water: " << high_water << " bytes\n";

  // ---- mHC -------------------------------------------------------------
  out << "  mHC:          pure TND, " << kNhcStreams << " residual streams, two rounds per layer ("
      << 2 * kNumLayers << " per step)\n";
  out << "                x [1, " << kNhcStreams << ", " << kHiddenSize << "] BF16, hRes/B_l [1, "
      << kNhcStreams << ", " << kNhcStreams << "] FP32, hIn/hOut [1, " << kHiddenSize << "] BF16, hPost [1, "
      << kNhcStreams << "] FP32 -- no rank mixing, and hIn feeds RMSNorm as is\n";
  if (config_.mhc_mixing_mode == MhcMixingMode::kFusedExternal) {
    out << "  B_l SOURCE:   kFusedExternal -- the pipeline launches NOTHING for the Sinkhorn link and takes\n"
           "                whatever the B_l slot holds. This is only correct with a fused producer wired in\n"
           "                (aclnnHcPreSinkhorn is the vendored candidate); with none, B_l is whatever was\n"
           "                last written there.\n";
  } else {
    out << "  B_l SOURCE:   kPerUsePlan -- aclnnMhcSinkhorn is planned and launched once per mHC round, its\n"
           "                executor consumed by its own launch. FORCED: on a 950PR\n"
           "                aclSetAclOpExecutorRepeatable fails this operator with 561000 (the wrapper builds a\n"
           "                dynamic internal iteration graph CANN will not mark reusable), and a\n"
           "                non-contiguous output -- once the documented workaround -- is refused outright with\n"
           "                561103. So it cannot hold a retained executor at all, and this engine pays "
        << 2 * kNumLayers
        << " host\n                plans per step for it. Removing that cost needs the normalization fused into a\n"
           "                kernel that never materializes B_l at the aclnn layer; aclnnHcPreSinkhorn does\n"
           "                exactly that and would be driven from mHC's optional hMix / invRms outputs.\n";
  }
  if (!arena_manager_.backbone().mhc_weights_populated) {
    out << "  WARNING: the mHC projection weights (hc.phi / alpha / bias / gamma) were NOT populated by the\n"
           "           checkpoint and are zero. aclnnMhcPre would fold the four streams into a zero layer\n"
           "           input. Supply model.layers.*.{self_attn,mlp}.hc.*\n";
  }

  // ---- layer topology --------------------------------------------------
  int64_t swa = 0, csa = 0, hca = 0;
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    switch (config_.attention_path(layer)) {
      case AttentionPath::kCompressedSparse: ++csa; break;
      case AttentionPath::kHyperCompressed: ++hca; break;
      default: ++swa; break;
    }
  }
  out << "  topology:     " << swa << " SWA, " << csa << " CSA (4:1), " << hca << " HCA (128:1)";
  if (config_.compress_ratios.empty()) {
    out << " -- compress_ratios was not supplied, so every layer is SWA and the\n"
           "                compressed cores are NOT planned (no reservation, no descriptor, no executor). No\n"
           "                interleave is guessed: --compress-ratios or a checkpoint schedule activates them.\n";
  } else {
    out << " from " << (config_.compress_ratios_provenance == GeometryProvenance::kCommandLine
                            ? "the command line"
                            : "the checkpoint")
        << "\n";
    out << "  compressed KV entry: " << kCompressedKvEntryBytes << " bytes ("
        << kCompressedKvNopeChannels << " nope FP8 + " << kCompressedKvScaleCount << " UE8M0 block scales + "
        << static_cast<int64_t>(sizeof(Dsv4CompressedKvEntry::padding)) << " pad + "
        << kCompressedKvRopeChannels << " rope BF16), " << arena_manager_.compressed_slots()
        << " slots per layer\n";
    out << "                every stride attribute (stateCacheStrideDim0, blockStride, cmpKvStride0, the\n"
           "                indexer stride/scaleStride) is DERIVED from the bound descriptor via\n"
           "                aclGetViewStrides; none is a constant. The wrappers forward them unchecked.\n";
    out << "  UNRESOLVED:   whether the deployed compressor kernel writes this " << kCompressedKvEntryBytes
        << "-byte packing. H3 measured\n                cmpKvOut as D x sizeof(dtype) and recorded the "
        << kCompressedKvEntryBytes
        << "-byte claim as unsettled; the entry is the\n                host-side authority for the cache row, "
           "and the derived strides are what keep a\n                disagreement loud instead of silent.\n";
    out << "  NOT BOUND:    the uncompressed (ori) half of the hybrid cache. "
           "aclnnKvQuantSparseAttnSharedkv wants\n                oriKv in FP8 E4M3 and this engine's paged "
           "latent cache is BF16, so a compressed layer\n                attends its compressed stream only. "
           "Binding it is a cache-layout change, not a\n                binding change.\n";
  }
  if (csa > 0) {
    out << "  INDEXER SCALES: queryDequantScale / keyDequantScale are bound at UNIT scale. The operator\n"
           "                documents them as per-token-head FP32, and nothing in this graph produces that:\n"
           "                aclnnDynamicMxQuant and aclnnRmsNormDynamicMxQuant emit OCP E8M0 block-32\n"
           "                microscales, and aclnnIndexerCompressEpilogV2 has no scale output. The top-512\n"
           "                selection RUNS and its indices are real, but the score ORDERING is only correct\n"
           "                once a per-token-head scale producer exists.\n";
  }
  out << "  DECOMPOSED SCORING: the router scoring is sqrt(softplus(logits)) -- aclnnSoftplus (beta="
      << kSoftplusBeta << ",\n               threshold=" << kSoftplusThreshold << ") then aclnnSqrt -- because the "
         "stock gating\n               operator's normType offers softmax/sigmoid only. aclnnMoeGatingTopKV2 receives "
         "the\n               pre-normalized scores (normType="
      << config_.gating_norm_type << ", renorm=" << kGatingRenormL1 << ", eps=" << kGatingEps
      << ").\n               The pass-through normType value still needs a device A/B against the host reference in\n"
         "               v5_ops_moe.py (sqrt_softplus_routing); --gating-norm-type overrides it.\n";
  out << "  UNVERIFIED: aclnnQuantMatmulV5 groupSize=" << config_.dense_group_size
      << " (the block-128 dense scale encoding).\n";
  if (!arena_manager_.backbone().rope_tables_populated) {
    out << "  WARNING: the rope cos/sin tables were NOT populated by the checkpoint and are zero. Attention\n"
           "           would compute without positional information. Supply model.rotary_emb.{cos,sin}_cached.\n";
  }
  return out.str();
}

std::string Dsv4Pipeline::DescribeSlotIndexMap() const {
  std::ostringstream out;
  out << "aclSetTensorAddr index map (derive-and-verify; the plan phase cannot check these)\n";
  out << "  RmsNorm                 x=" << slot::kRmsNormX << " gamma=" << slot::kRmsNormGamma
      << " y=" << slot::kRmsNormY << " rstd=" << slot::kRmsNormRstd << "\n";
  out << "  RmsNormDynamicMxQuant   x=" << slot::kMxQuantX << " gamma=" << slot::kMxQuantGamma
      << " beta=" << slot::kMxQuantBeta << " y=" << slot::kMxQuantY << " mxscale=" << slot::kMxQuantScale << "\n";
  out << "  Matmul                  self=" << slot::kMatmulSelf << " mat2=" << slot::kMatmulMat2
      << " out=" << slot::kMatmulOut << "\n";
  out << "  QuantMatmulV5           x1=" << slot::kQuantMmX1 << " x2=" << slot::kQuantMmX2
      << " x1Scale=" << slot::kQuantMmX1Scale << " x2Scale=" << slot::kQuantMmX2Scale
      << " out=" << slot::kQuantMmOut << "\n";
  out << "  Softplus                x=" << slot::kSoftplusX << " out=" << slot::kSoftplusOut
      << " (beta/threshold are scalars, skipped by the IR numbering)\n";
  out << "  Sqrt                    x=" << slot::kSqrtX << " out=" << slot::kSqrtOut << "\n";
  out << "  MoeGatingTopKV2         x=" << slot::kGatingX << " bias=" << slot::kGatingBias
      << " y=" << slot::kGatingY << " expertIdx=" << slot::kGatingExpertIdx << "\n";
  out << "  ApplyRotaryPosEmbV2     cos=" << slot::kRotaryCos << " sin=" << slot::kRotarySin << "\n";
  out << "  ScatterPaKvCache        keyCacheRef=" << slot::kScatterKeyCache
      << " valueCacheRef=" << slot::kScatterValueCache << "\n";
  out << "  FusedInferAttentionV5   key[0]=" << slot::kFiaKeyList << " value[0]=" << slot::kFiaValueList
      << " keyRope=" << slot::kFiaKeyRope << "\n";
  out << "  GroupedMatmulV5         weight[i]=" << slot::kGmmWeightList << " scale[i]=" << slot::kGmmV5ScaleList
      << "\n";
  out << "  GmmSwigluQuantV2        weight[i]=" << slot::kGmmWeightList
      << " weightScale[i]=" << slot::kGmmSwigluScaleList << "\n";
  out << "  InplaceAdd              selfRef=" << slot::kInplaceAddSelf << " other=" << slot::kInplaceAddOther
      << " (alpha is a scalar, skipped)\n";
  out << "  MhcPre                  x=" << slot::kMhcPreX << " phi=" << slot::kMhcPrePhi
      << " alpha=" << slot::kMhcPreAlpha << " bias=" << slot::kMhcPreBias << " gamma=" << slot::kMhcPreGamma
      << "\n";
  out << "  MhcSinkhorn             x=" << slot::kMhcSinkhornX << " out=" << slot::kMhcSinkhornOut
      << " (never rebound: single-use plan, see the B_l note above)\n";
  out << "  MhcPost                 x=" << slot::kMhcPostX << " hRes=" << slot::kMhcPostHRes
      << " hOut=" << slot::kMhcPostHOut << " hPost=" << slot::kMhcPostHPost << " out=" << slot::kMhcPostOut
      << "\n";
  out << "  Compressor              x=" << slot::kCompressorX << " wkv=" << slot::kCompressorWkv
      << " wgate=" << slot::kCompressorWgate << " stateCacheRef=" << slot::kCompressorStateCache
      << " ape=" << slot::kCompressorApe << " normWeight=" << slot::kCompressorNormWeight
      << " ropeSin=" << slot::kCompressorRopeSin << " ropeCos=" << slot::kCompressorRopeCos
      << " cmpKvOut=" << slot::kCompressorCmpKvOut << "\n";
  out << "  KvCompressEpilog        cacheRef=" << slot::kCompressEpilogCacheRef
      << " (also IndexerCompressEpilogV2; both leading params are REF)\n";
  out << "  VllmQuantLightningIdx   key=" << slot::kIndexerKey << " weights=" << slot::kIndexerWeights
      << " keyDequantScale=" << slot::kIndexerKeyDequantScale << "\n";
  out << "  KvQuantSparseAttnShrdkv cmpKv=" << slot::kSparseAttnCmpKv << "\n";
  out << "\n  THE ONE AMBIGUITY, NARROWED. Two numberings are defensible for an op that takes\n"
         "  aclIntArray parameters, and they disagree only for such ops:\n"
         "    counting aclIntArray among the tensors  -> FIA keyRope is 25 (what this engine uses)\n"
         "    skipping it, as the scalars are skipped  -> FIA keyRope is 22\n"
         "  aclnnFusedInferAttentionScoreV5 is the ONLY op in this graph with aclIntArray arguments\n"
         "  (actualSeqLengths, actualSeqLengthsKv, actualSharedPrefixLen, qStartIdx, kvStartIdx), so it is\n"
         "  the only index that can be wrong: the mock's slot-map tally flags exactly one swap per\n"
         "  SWA layer per step, and nothing else. Every mHC, compressor, epilog, indexer and shared-KV\n"
         "  index above is unambiguous under BOTH conventions, because none of those ops takes an\n"
         "  aclIntArray. A device bring-up therefore has ONE number to settle, not twelve.\n";
  return out.str();
}

}  // namespace ascend_moe

