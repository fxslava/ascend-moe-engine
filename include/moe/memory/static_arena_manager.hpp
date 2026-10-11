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

// StaticArenaManager: the memory half of the decode graph (SRP).
//
// Everything about WHAT is allocated and WHICH descriptors describe it lives
// here; everything about WHEN it runs lives in Dsv4Pipeline and
// MoeRouterEngine. The manager owns the StaticMemoryArena, the reservation /
// ingestion / descriptor phases of its three-phase lifecycle, and the two
// descriptor sets (activations + backbone weights) the stages consume. It
// knows nothing about streams, operators or the 43-layer replay.

#pragma once
#include <memory>

#include <cstdint>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_allocator.hpp"
#include "moe/core/device_types.hpp"
#include "moe/core/kv_cache_layout.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/memory/static_arena.hpp"

namespace ascend_moe {

// Shared byte-arithmetic of the graph's shapes (reservations, descriptors and
// per-layer addressing all speak in these).
int64_t DivideUp(int64_t value, int64_t divisor);
size_t Fp8Bytes(int64_t elements);
size_t Bf16Bytes(int64_t elements);
size_t Fp32Bytes(int64_t elements);
size_t Int32Bytes(int64_t elements);
size_t Int64Bytes(int64_t elements);
// Block-128 scale columns for a dense weight whose reduction axis is `k`.
int64_t DenseScaleCols(int64_t k);
// Block-32 microscale columns.
int64_t MxScaleCols(int64_t k);

// Where the six active experts' regions sit inside their HBM slots at
// descriptor time. The pipeline resolves these from the exclusive manager
// (slots 0..5 are resident at init) and hands them to the manager, which is
// what keeps this class free of any staging dependency (DIP).
struct ExpertSlotAddresses {
  void* gate_up_weight[kNumExpertsPerTok] = {};
  void* gate_up_scale[kNumExpertsPerTok] = {};
  void* down_weight[kNumExpertsPerTok] = {};
  void* down_scale[kNumExpertsPerTok] = {};
};

// One mHC round's tensor set: everything `aclnnMhcPre` ->
// `aclnnMhcSinkhorn` -> `aclnnMhcPost` needs for ONE sub-block. There are two
// of these per layer (attention and MoE), because each sub-block gets its own
// hyper-connection.
//
// PURE TND, AS THE 950PR RUN SETTLED IT (see the kMhcStreamRank note in
// config.hpp). Rank 3 for the stream tensors, rank 2 for the per-token ones,
// and no mixing anywhere -- which also means `h_in` at [1, 4096] IS the
// activation RMSNorm / attention / MoE are planned for, so it is bound to them
// directly, with no squeeze, no unsqueeze and no second view.
struct MhcRoundTensors {
  aclTensor* h_in = nullptr;        // [1, 4096] bf16  (pre's y / hIn, fed on as is)
  aclTensor* h_post = nullptr;      // [1, 4]    fp32
  aclTensor* h_res = nullptr;       // [1, 4, 4] fp32  (mhc_pre's RAW map;
                                    //  unused on the fused path, where the
                                    //  kernel never materializes it)
  aclTensor* h_res_sink = nullptr;  // [1, 4, 4] fp32  (B_l, CONTIGUOUS)
  aclTensor* h_out = nullptr;       // [1, 4096] bf16  (a descriptor over the
                                    //  sub-block's own output buffer)

  // The SAME four buffers with a unit leading axis, for `aclnnHcPost`.
  //
  // HcPre takes x as TND [bs, hc, d] and emits y / post / comb_frag at
  // [1, 4096] / [1, 4] / [1, 4, 4] -- byte-identical to the TND set above, so
  // y feeds RMSNorm with no reshape. HcPost, however, is strictly BSHD: x rank
  // 3, residual / comb / y rank 4, post rank 3 (aclnn_hc_post.cpp kXDim = 3,
  // kResidualDim = 4, kPostDim = 3). Its own header says how the two meet --
  // "a TND token stream maps on as b = T, s = 1" -- so these are the same
  // addresses, the same element counts and the same contiguous strides with a
  // 1 prepended. Created once at Build; never a runtime squeeze or unsqueeze.
  aclTensor* h_post_bshd = nullptr;  // [1, 1, 4]    fp32
  aclTensor* b_l_bshd = nullptr;     // [1, 1, 4, 4] fp32
  aclTensor* h_out_bshd = nullptr;   // [1, 1, 4096] bf16

  ArenaHandle h_h_in = kInvalidArenaHandle;  // for the window-ring copies
};

// Every activation descriptor the graph uses. Created once, in
// CreateDescriptors; the stages only ever repoint addresses.
struct ArenaTensors {
  // ---- mHC residual stream ---------------------------------------------
  //
  // Two [1, 4, 4096] BF16 buffers, ping-ponged: every `aclnnMhcPost` reads one
  // and writes the other, so no call ever aliases its own input. 43 layers x 2
  // sub-blocks = 86 writes, so the finished stream lands back in slot 0 -- but
  // the pipeline tracks the parity rather than relying on that.
  aclTensor* residual_stream[2] = {nullptr, nullptr};
  // The same two buffers as [1, 1, 4, 4096], for HcPost's rank-4 residual and
  // y. Same addresses, a 1 prepended; see the note on MhcRoundTensors.
  aclTensor* residual_stream_bshd[2] = {nullptr, nullptr};
  // The four streams of each buffer as [1, 4096] views, for the embedding
  // broadcast at the start of a step and the summation at the head.
  aclTensor* stream_slice[2][kNhcStreams] = {};
  ArenaHandle h_residual_stream[2] = {kInvalidArenaHandle, kInvalidArenaHandle};
  MhcRoundTensors mhc_attn;
  MhcRoundTensors mhc_moe;

  // activations
  aclTensor* hidden = nullptr;            // [1, 4096] bf16
  aclTensor* normed = nullptr;           // [1, 4096] bf16
  aclTensor* normed_fp8 = nullptr;       // [1, 4096] fp8
  aclTensor* normed_mx_scale = nullptr;  // [1, 128] e8m0
  aclTensor* rstd = nullptr;             // [1, 1] fp32
  aclTensor* q_a = nullptr;              // [1, 1024] bf16
  aclTensor* q_a_fp8 = nullptr;          // [1, 1024] fp8
  aclTensor* q_a_mx_scale = nullptr;     // [1, 32] e8m0
  aclTensor* q_b = nullptr;              // [1, heads*(kv_lora+rope)] bf16
  aclTensor* q_latent = nullptr;         // [1, heads, kv_lora] strided view of q_b
  aclTensor* q_rope = nullptr;           // [1, heads, rope] strided view of q_b
  aclTensor* kv_a = nullptr;             // [1, kv_lora+rope] bf16
  aclTensor* kv_latent = nullptr;        // [1, kv_lora] view of kv_a
  aclTensor* k_rope = nullptr;           // [1, 1, rope] view of kv_a
  aclTensor* kv_latent_normed = nullptr;  // [1, kv_lora] bf16
  aclTensor* attn_out = nullptr;          // [1, heads, kv_lora] bf16
  aclTensor* attn_flat = nullptr;         // [1, heads*kv_lora] bf16
  aclTensor* attn_fp8 = nullptr;          // [1, heads*kv_lora] fp8
  aclTensor* attn_mx_scale = nullptr;     // [1, heads*kv_lora/32] e8m0
  aclTensor* proj_out = nullptr;          // [1, 4096] bf16
  aclTensor* softmax_lse = nullptr;       // [1, heads, 1] fp32

  // routing
  aclTensor* router_matmul = nullptr;      // [1, 256] BF16 before explicit cast
  aclTensor* router_logits = nullptr;      // [1, 256] fp32
  aclTensor* router_softplus = nullptr;    // [1, 256] fp32, softplus(logits)
  aclTensor* router_scores = nullptr;      // [1, 256] fp32, sqrt(softplus(logits))
  aclScalar* softplus_beta = nullptr;      // host scalar, kSoftplusBeta as fp32
  aclScalar* softplus_threshold = nullptr; // host scalar, kSoftplusThreshold as fp32
  aclTensor* gating_weights = nullptr;     // [1, 6] fp32
  aclTensor* gating_indices = nullptr;     // [1, 6] int32
  aclTensor* local_indices = nullptr;      // [1, 6] int32 (0..5, host-written)
  aclTensor* expanded_x = nullptr;         // [6, 4096] fp8
  aclTensor* expanded_row_idx = nullptr;   // [6] int32
  aclTensor* expanded_scale = nullptr;     // [6, 128] e8m0
  aclTensor* expanded_weights = nullptr;   // [6] fp32
  aclTensor* combine_weights = nullptr;   // [1, 6] BF16 cast, matching GEMM2
  aclTensor* expanded_weights_row = nullptr;  // [1, 6] fp32 view, for the combine
  aclTensor* group_list = nullptr;         // [6] int64 cumsum
  aclTensor* gemm1_raw = nullptr;          // [6, 2*2048] bf16 (decomposed path)
  aclTensor* gemm1_out = nullptr;          // [6, 2048] fp8
  aclTensor* gemm1_scale = nullptr;        // [6, 64] e8m0
  aclTensor* gemm2_out = nullptr;          // [6, 4096] bf16
  aclTensor* routed_out = nullptr;         // [1, 4096] bf16
  aclTensorList* gemm1_x_list = nullptr;
  aclTensorList* gemm1_x_scale_list = nullptr;
  aclTensorList* gemm1_raw_out_list = nullptr;
  aclTensorList* gemm2_x_list = nullptr;
  aclTensorList* gemm2_x_scale_list = nullptr;
  aclTensorList* gemm2_out_list = nullptr;

  // shared expert
  aclTensor* shared_gate_up = nullptr;    // [1, 2*2048] bf16
  aclTensor* shared_act = nullptr;        // [1, 2048] bf16
  aclTensor* shared_act_fp8 = nullptr;    // [1, 2048] fp8
  aclTensor* shared_act_scale = nullptr;  // [1, 64] e8m0
  aclTensor* shared_out = nullptr;        // [1, 4096] bf16

  // head
  aclTensor* final_normed = nullptr;  // [1, 4096] bf16
  aclTensor* logits = nullptr;        // [1, vocab] bf16
  aclTensor* argmax = nullptr;        // [1] int64

  // paged KV
  aclTensor* kv_latent_cache = nullptr;
  aclTensor* kv_rope_cache = nullptr;
  aclTensorList* key_list = nullptr;
  aclTensorList* value_list = nullptr;
  aclTensor* key_rope_cache_view = nullptr;
  aclTensor* block_table = nullptr;
  aclTensor* slot_mapping = nullptr;
  aclIntArray* actual_seq_q = nullptr;
  aclIntArray* actual_seq_kv = nullptr;

  // rope tables
  aclTensor* rope_cos = nullptr;
  aclTensor* rope_sin = nullptr;

  // ---- token compression (CSA / HCA) -----------------------------------
  //
  // Null throughout when no layer selects a compressed path: the stages are
  // then not planned either, so nothing reserves the ~GB of projection
  // weights a compressed layer needs. See StaticArenaManager::uses_compression.
  //
  // Each step copies h_in to (position % ratio) in the device window.
  // Only a full window launches Compressor; HOLD has no executor or empty
  // descriptor. TH output storage includes one padding row.
  aclTensor* cmp_window_csa = nullptr;     // [4, 4096]   bf16, per-layer view
  aclTensor* cmp_window_hca = nullptr;     // [128, 4096] bf16, per-layer view
  aclTensor* cmp_rope_sin_csa = nullptr;   // [2, 64] fp32, emitted row + padding
  aclTensor* cmp_rope_cos_csa = nullptr;
  aclTensor* cmp_rope_sin_hca = nullptr;   // [2, 64] fp32
  aclTensor* cmp_rope_cos_hca = nullptr;
  aclTensor* cmp_kv_padded = nullptr;
  aclTensor* cmp_rope_cos_row = nullptr;
  aclTensor* cmp_rope_sin_row = nullptr;
  aclTensor* index_metadata = nullptr;
  aclTensor* index_seq_k = nullptr;
  aclTensor* index_head_weights_bf16 = nullptr;
  aclTensor* cmp_state_cache = nullptr;    // [4, 8, 1024] fp32 REF ring, per layer
  aclTensor* cmp_state_block_table = nullptr;  // [1, 4]   int32
  aclTensor* cmp_cu_seqlens = nullptr;     // [2] int32
  aclTensor* cmp_seqused = nullptr;        // [1] int32
  aclTensor* cmp_start_pos = nullptr;      // [1] int32
  aclTensor* cmp_kv_out = nullptr;         // [1, 512] bf16, the emitted row
  // The paged hybrid cache, one Dsv4CompressedKvEntry per compressed slot.
  // Two views over the same bytes because the epilog and the attention core
  // disagree on rank/dtype spelling, never two allocations.
  aclTensor* cmp_kv_cache = nullptr;       // [blocks, block_size, 604] fp8, per layer
  aclTensor* cmp_slot_mapping = nullptr;   // [1] int32
  aclTensor* cmp_block_table = nullptr;    // [1, blocks] int32
  aclTensor* cmp_seq_k = nullptr;          // [1] int32, compressed context length
  // 0..kIndexTopK-1, written once at Build. The HCA path has no indexer, but
  // the attention core still requires an INT32 index vector for a bound
  // cmpKv, so "attend everything in the compressed stream" is spelled as the
  // identity selection.
  aclTensor* cmp_sparse_indices_dense = nullptr;  // [1, 1, 1, 512] int32
  aclTensor* sparse_q = nullptr;           // [1, 1, 64, 512] bf16 view of act.q_b
  aclTensor* sparse_attn_out = nullptr;    // [1, 1, 64, 512] bf16 view of act.attn_out
  aclTensor* sparse_lse_empty = nullptr;   // [0] fp32 placeholder

  // ---- quantized lightning indexer (CSA only) --------------------------
  aclTensor* index_q_bf16 = nullptr;       // [1, 8192] bf16 (64 heads x 128)
  aclTensor* index_q_fp8 = nullptr;        // [1, 8192] fp8 e4m3
  aclTensor* index_q_mx_scale = nullptr;   // [1, 256]  e8m0 (discarded, see report)
  aclTensor* index_q = nullptr;            // [1, 1, 64, 128] fp8 view of index_q_fp8
  aclTensor* index_k_bf16 = nullptr;       // [1, 128] bf16
  aclTensor* index_k_cache_u8 = nullptr;   // [blocks, block_size, 128] uint8, per layer
  aclTensor* index_k_cache = nullptr;      // [blocks, block_size, 1, 128] fp8, same bytes
  aclTensor* index_k_dequant = nullptr;    // [blocks, block_size, 1] fp32, per layer
  aclTensor* index_q_dequant = nullptr;    // [1, 1, 64] fp32
  aclTensor* index_head_weights = nullptr; // [1, 1, 64] fp32 cast output
  aclTensor* index_sparse_indices = nullptr;   // [1, 1, 1, 512] int32
  aclTensor* index_sparse_values_empty = nullptr;  // [0] fp32 placeholder

  // per-layer weights, repointed by aclSetTensorAddr
  aclTensor* w_input_norm = nullptr;
  aclTensor* w_q_a = nullptr;
  aclTensor* w_q_a_scale = nullptr;
  aclTensor* w_q_a_norm = nullptr;
  aclTensor* w_q_b = nullptr;
  aclTensor* w_q_b_scale = nullptr;
  aclTensor* w_kv_a = nullptr;
  aclTensor* w_kv_a_scale = nullptr;
  aclTensor* w_kv_a_norm = nullptr;
  aclTensor* w_o = nullptr;
  aclTensor* w_o_scale = nullptr;
  aclTensor* w_post_norm = nullptr;
  aclTensor* w_router = nullptr;
  aclTensor* w_router_bias = nullptr;
  aclTensor* w_shared_gate_up = nullptr;
  aclTensor* w_shared_gate_up_scale = nullptr;
  aclTensor* w_shared_down = nullptr;
  aclTensor* w_shared_down_scale = nullptr;
  aclTensor* w_final_norm = nullptr;
  aclTensor* w_lm_head = nullptr;

  // mHC weights. One descriptor set, repointed per layer AND per sub-block
  // round -- each round is its own stage with its own executor, so the shared
  // descriptors are rebound on both, exactly as w_post_norm already is for
  // post_norm_quant and post_norm.
  aclTensor* w_mhc_phi = nullptr;    // [24, 16384] fp32
  aclTensor* w_mhc_alpha = nullptr;  // [3] fp32
  aclTensor* w_mhc_bias = nullptr;   // [24] fp32
  aclTensor* w_mhc_gamma = nullptr;  // [4, 4096] fp32

  // Compressor and indexer weights, null on a SWA-only checkpoint. The
  // positional bias has one row per window position, so CSA and HCA need
  // different SHAPES of it, not just different addresses.
  aclTensor* w_cmp_wkv = nullptr;          // [512, 4096] bf16
  aclTensor* w_cmp_wgate = nullptr;        // [512, 4096] bf16
  aclTensor* w_cmp_ape_csa = nullptr;      // [4, 512] fp32
  aclTensor* w_cmp_ape_hca = nullptr;      // [128, 512] fp32
  aclTensor* w_cmp_norm_weight = nullptr;  // [512] fp32
  aclTensor* w_index_q = nullptr;          // [1024, 8192] bf16 ([K, N] for Matmul)
  aclTensor* w_index_k = nullptr;          // [512, 128] bf16

  // the six active experts, repointed per layer
  std::vector<aclTensor*> expert_gate_up;
  std::vector<aclTensor*> expert_gate_up_scale;
  std::vector<aclTensor*> expert_down;
  std::vector<aclTensor*> expert_down_scale;
  aclTensorList* expert_gate_up_list = nullptr;
  aclTensorList* expert_gate_up_scale_list = nullptr;
  aclTensorList* expert_down_list = nullptr;
  aclTensorList* expert_down_scale_list = nullptr;

  // arena handles the step needs addresses from
  ArenaHandle h_hidden = kInvalidArenaHandle;
  ArenaHandle h_gating_indices = kInvalidArenaHandle;
  ArenaHandle h_local_indices = kInvalidArenaHandle;
  ArenaHandle h_group_list = kInvalidArenaHandle;
  ArenaHandle h_argmax = kInvalidArenaHandle;
  ArenaHandle h_slot_mapping = kInvalidArenaHandle;
};

// Per-layer backbone weight reservations. kv_b_proj never appears: its two
// up-projections are absorbed into q_b_proj and o_proj at checkpoint-conversion
// time, which is what makes the decode graph attend in the compressed latent
// space and the paged cache `kv_lora_rank + qk_rope_head_dim` wide.
struct BackboneWeights {
  struct Layer {
    ArenaHandle input_norm = kInvalidArenaHandle;
    ArenaHandle q_a_weight = kInvalidArenaHandle;
    ArenaHandle q_a_scale = kInvalidArenaHandle;
    ArenaHandle q_a_norm = kInvalidArenaHandle;
    ArenaHandle q_b_weight = kInvalidArenaHandle;  // folded: emits q_latent | q_rope
    ArenaHandle q_b_scale = kInvalidArenaHandle;
    ArenaHandle kv_a_weight = kInvalidArenaHandle;  // emits kv_latent | k_rope
    ArenaHandle kv_a_scale = kInvalidArenaHandle;
    ArenaHandle kv_a_norm = kInvalidArenaHandle;
    ArenaHandle o_weight = kInvalidArenaHandle;  // folded: consumes the latent attn out
    ArenaHandle o_scale = kInvalidArenaHandle;
    ArenaHandle post_norm = kInvalidArenaHandle;
    ArenaHandle router_weight = kInvalidArenaHandle;
    ArenaHandle router_bias = kInvalidArenaHandle;  // noaux_tc selection bias
    ArenaHandle shared_gate_up_weight = kInvalidArenaHandle;
    ArenaHandle shared_gate_up_scale = kInvalidArenaHandle;
    ArenaHandle shared_down_weight = kInvalidArenaHandle;
    ArenaHandle shared_down_scale = kInvalidArenaHandle;

    // mHC: one hyper-connection per sub-block, so two independent weight sets
    // per layer. phi is [24, 16384] FP32 -- 1.5 MiB each, 129 MiB over the
    // model -- which is why it is reserved rather than rebuilt.
    struct MhcWeights {
      ArenaHandle phi = kInvalidArenaHandle;    // [24, 16384] fp32
      ArenaHandle alpha = kInvalidArenaHandle;  // [3] fp32
      ArenaHandle bias = kInvalidArenaHandle;   // [24] fp32
      ArenaHandle gamma = kInvalidArenaHandle;  // [4, 4096] fp32
    };
    MhcWeights mhc_attn;
    MhcWeights mhc_moe;

    // Compressor, reserved only on a build whose checkpoint marks some layer
    // CSA or HCA.
    ArenaHandle cmp_wkv = kInvalidArenaHandle;          // [512, 4096] bf16
    ArenaHandle cmp_wgate = kInvalidArenaHandle;        // [512, 4096] bf16
    ArenaHandle cmp_ape = kInvalidArenaHandle;          // [max_ratio, 512] fp32
    ArenaHandle cmp_norm_weight = kInvalidArenaHandle;  // [512] fp32

    // Lightning indexer, reserved only when some layer is CSA.
    ArenaHandle index_q_weight = kInvalidArenaHandle;     // [1024, 8192] bf16
    ArenaHandle index_k_weight = kInvalidArenaHandle;     // [512, 128] bf16
    ArenaHandle index_head_weight = kInvalidArenaHandle;  // [64] bf16
  };
  std::vector<Layer> layers;
  ArenaHandle embed_tokens = kInvalidArenaHandle;
  ArenaHandle final_norm = kInvalidArenaHandle;
  ArenaHandle lm_head = kInvalidArenaHandle;
  ArenaHandle rope_cos = kInvalidArenaHandle;
  ArenaHandle rope_sin = kInvalidArenaHandle;
  ArenaHandle kv_latent_cache = kInvalidArenaHandle;
  ArenaHandle kv_rope_cache = kInvalidArenaHandle;
  ArenaHandle block_table = kInvalidArenaHandle;
  ArenaHandle slot_mapping = kInvalidArenaHandle;

  // Compression-side reservations, one slice per layer carved out of each.
  ArenaHandle cmp_window = kInvalidArenaHandle;
  ArenaHandle cmp_state_cache = kInvalidArenaHandle;
  ArenaHandle cmp_kv_cache = kInvalidArenaHandle;
  ArenaHandle cmp_block_table = kInvalidArenaHandle;
  ArenaHandle cmp_slot_mapping = kInvalidArenaHandle;
  ArenaHandle cmp_window_meta = kInvalidArenaHandle;
  ArenaHandle cmp_seq_k = kInvalidArenaHandle;
  ArenaHandle cmp_state_block_table = kInvalidArenaHandle;
  ArenaHandle cmp_sparse_indices_dense = kInvalidArenaHandle;
  ArenaHandle index_k_cache = kInvalidArenaHandle;
  ArenaHandle index_k_dequant = kInvalidArenaHandle;
  ArenaHandle index_q_dequant = kInvalidArenaHandle;
  ArenaHandle index_sparse_indices = kInvalidArenaHandle;

  bool rope_tables_populated = false;
  // False when the checkpoint supplied no mHC projection weights. phi / alpha
  // / bias then read as zero and `aclnnMhcPre` folds the four streams into a
  // zero layer input -- a wrong answer with no symptom, so every report says
  // so, exactly as it does for the rope tables.
  bool mhc_weights_populated = false;
};

class StaticArenaManager {
 public:
  // The stream engine is used exactly once per phase: the ordered H2D
  // transfers of backbone ingestion. Everything else is allocation.
  StaticArenaManager(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config);
  ~StaticArenaManager();

  StaticArenaManager(const StaticArenaManager&) = delete;
  StaticArenaManager& operator=(const StaticArenaManager&) = delete;

  // The memory the backbone needs, so the slot planner can subtract it from
  // free HBM before choosing K. Pure arithmetic; no allocation.
  //
  // `compress_ratios` is the checkpoint's per-layer schedule: an empty list
  // means no layer compresses, so the compressor and indexer projections are
  // neither reserved nor counted. Passing it matters -- a CSA layer's indexer
  // query projection alone is 16 MiB, and under-reporting the backbone would
  // let the slot planner choose a K that does not fit.
  static size_t BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size, int64_t max_context_len,
                                    const std::vector<int64_t>& compress_ratios = {});

  // ---- phase 1: RESERVE -------------------------------------------------
  void ReserveActivations();
  void ReserveBackbone();

  // ---- phase 2: BUILD ---------------------------------------------------
  void Commit();
  void IngestBackbone(WeightByteSource& source);
  void CreateDescriptors(const ExpertSlotLayout& slots, const ExpertSlotAddresses& experts);
  void CommitWorkspace();

  // ---- phase 3: SEALED --------------------------------------------------
  void Seal();

  // ---- accessors ---------------------------------------------------------
  // The non-const overloads exist for the planning phase (workspace notes,
  // descriptor creation); the decode loop sees const only.
  const StaticMemoryArena& arena() const { return arena_; }
  StaticMemoryArena& arena() { return arena_; }
  const ArenaTensors& tensors() const { return *tensors_; }
  ArenaTensors& tensors() { return *tensors_; }
  const BackboneWeights& backbone() const { return *backbone_; }
  int64_t num_blocks() const { return num_blocks_; }

  // Which compressed paths this checkpoint actually asks for. The pipeline
  // plans a path's stages iff the answer here is yes, so a SWA-only
  // checkpoint pays nothing for CSA/HCA -- not reservations, not descriptors,
  // not executors.
  bool uses_csa() const { return uses_csa_; }
  bool uses_hca() const { return uses_hca_; }
  bool uses_compression() const { return uses_csa_ || uses_hca_; }
  // The widest window any layer holds; 1 when nothing compresses.
  int64_t max_compress_ratio() const { return max_compress_ratio_; }
  // How many compressed entries one layer's paged cache can hold.
  int64_t compressed_slots() const { return num_blocks_ * config_.block_size; }

  // Byte stride between consecutive layers' slices of a shared reservation.
  // Host-side pointer arithmetic only; the strides the OPERATORS receive all
  // come from DeriveDimensionStrideElements on the bound descriptor.
  size_t CompressedWindowLayerStrideBytes() const;
  size_t CompressedStateLayerStrideBytes() const;
  size_t CompressedKvLayerStrideBytes() const;
  size_t IndexerKeyLayerStrideBytes() const;
  size_t IndexerKeyScaleLayerStrideBytes() const;

 private:
  void ReserveMhc();
  void ReserveCompression();
  void CreateMhcDescriptors();
  void CreateCompressionDescriptors();
  // Writes the constants the compressed paths need once and only once: the
  // identity block tables and the identity top-k selection.
  void SeedStaticTables();
  void* ReservationAddress(const char* name) const;

  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  RuntimeConfig config_;
  StaticMemoryArena arena_;
  std::unique_ptr<ArenaTensors> tensors_;
  std::unique_ptr<BackboneWeights> backbone_;
  int64_t num_blocks_ = 0;
  int64_t heads_ = kNumAttentionHeads;
  bool uses_csa_ = false;
  bool uses_hca_ = false;
  int64_t max_compress_ratio_ = 1;
};

}  // namespace ascend_moe
