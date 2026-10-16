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

// dsv4_layers.h -- the surrounding layer machinery the semi-end-to-end
// emulation needs: quantized GEMVs, shared-KV sliding-window attention, and
// the MoE block.
//
// cpu_kernels.h emulates the three FUSED operators. A layer is more than
// those three, so this header adds the pieces between them. Everything is
// ported from the reference forward pass in vllm-ascend
// vllm_ascend/models/deepseek_v4.py -- DeepseekV2DecoderLayer.forward,
// DeepseekV4Attention.__init__, DeepseekV4MoE.forward, DeepseekV2MLP -- and
// cross-checked against the shard headers for every shape.
//
// THE LAYER, AS THE REFERENCE WRITES IT
//   residual        = x                                  [n, D]
//   h_in,post,comb  = hc_pre(x, hc_attn_*)               h_in [D]
//   h_in            = input_layernorm(h_in)              attn_norm.weight
//   attn_out        = self_attn(h_in)                    [D]
//   x               = hc_post(attn_out, residual, post, comb)
//   residual        = x
//   h_in,post,comb  = hc_pre(x, hc_ffn_*)
//   h_in            = post_attention_layernorm(h_in)     ffn_norm.weight
//   ffn_out         = mlp(h_in)                          MoE or dense
//   x               = hc_post(ffn_out, residual, post, comb)
//
// Note where the activation RMSNorm sits: on h_in, the [1, D] mHC output.
// NOT on the [1, n, D] bundle. The bundle's own RMS reciprocal is computed
// inside hc_pre and applied to the 24-row projection; attn_norm.weight is a
// separate, later norm. See the note in cpu_kernels.h.

#ifndef MOE_EMU_DSV4_LAYERS_H
#define MOE_EMU_DSV4_LAYERS_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "moe/emu/cpu_kernels.h"

namespace moe::emu {

// ---------------------------------------------------------------------------
// Geometry -- config.json, cross-checked against the shard headers
// ---------------------------------------------------------------------------

// Attention. "num_attention_heads": 64, "head_dim": 512, "q_lora_rank": 1024,
// "o_lora_rank": 1024, "o_groups": 8, "sliding_window": 128.
//   wq_a [1024, 4096]   wq_b [32768, 1024]   wkv  [512, 4096]
//   wo_a [8192, 4096]   wo_b [4096, 8192]
// wq_b's 32768 is 64 heads x 512. wkv's 512 is head_dim, not n_heads*head_dim:
// DSV4 has "num_key_value_heads": 1, so ALL 64 query heads attend against one
// shared 512-wide key/value row per token. That is what the vendored operator
// name kv_quant_sparse_attn_sharedkv refers to, and it is why the KV cache is
// 512 floats per token instead of 64 x 512.
inline constexpr int kAttnHeads = 64;
inline constexpr int kAttnHeadDim = 512;
inline constexpr int kQLoraRank = 1024;
inline constexpr int kOLoraRank = 1024;
inline constexpr int kOGroups = 8;
inline constexpr int kAttnRopeHeadDim = 64;   // qk_rope_head_dim
inline constexpr int kSlidingWindow = 128;    // sliding_window
// wo_a is grouped: in_features = n_heads * head_dim / n_groups = 4096,
// out_features = n_groups * o_lora_rank = 8192.
inline constexpr int kWoAGroupInput = kAttnHeads * kAttnHeadDim / kOGroups;  // 4096
inline constexpr int kWoAOutput = kOGroups * kOLoraRank;                     // 8192

// MoE. "n_routed_experts": 256, "num_experts_per_tok": 6,
// "moe_intermediate_size": 2048, "routed_scaling_factor": 1.5,
// "swiglu_limit": 10.0, "num_hash_layers": 3.
inline constexpr int kNumRoutedExperts = 256;
inline constexpr int kNumExpertsPerTok = 6;
inline constexpr int kMoeIntermediate = 2048;
inline constexpr float kRoutedScalingFactor = 1.5F;
inline constexpr float kSwigluLimit = 10.0F;
inline constexpr int kNumHashLayers = 3;

// Quantization block geometry. The two routed storage kinds differ, which is
// easy to miss and fatal to get wrong:
//   dense / shared-expert FP8 : E4M3 values, E8M0 scale per 128 x 128 BLOCK
//                              ("weight_block_size": [128, 128]).
//                              wq_a [1024, 4096] -> scale [8, 32].
//   routed-expert FP4        : E2M1 nibbles, E8M0 scale per ROW per GROUP OF
//                              32 along the input axis.
//                              w1 logical [2048, 4096] -> scale [2048, 128].
inline constexpr int kFp8BlockSize = 128;
inline constexpr int kMxfp4GroupSize = 32;

// ---------------------------------------------------------------------------
// Quantized matrix views
// ---------------------------------------------------------------------------

// E4M3 weights with a 2-D grid of E8M0 block scales.
struct Fp8BlockMatrix {
  const std::uint8_t* weight = nullptr;  // [rows, cols] E4M3
  const std::uint8_t* scale = nullptr;   // [ceil(rows/128), ceil(cols/128)] E8M0
  int rows = 0;
  int cols = 0;
  int scale_cols = 0;  // the scale grid's row stride
};

// E2M1 nibbles with per-row, per-32-group E8M0 scales.
//
// NIBBLE ORDER -- confirmed by the official DeepSeek-V4-Flash
// inference/convert.py cast_e2m1fn_to_e4m3fn: low then high are stacked and
// flattened along the logical input dimension. The all-ones GEMV cannot
// distinguish a within-pair swap, so scripts/verify_mxfp4_packing.py also
// checks every decoded value and a nonuniform probe.
struct Mxfp4Matrix {
  const std::uint8_t* packed = nullptr;  // [rows, cols/2] packed nibbles
  const std::uint8_t* scale = nullptr;   // [rows, cols/32] E8M0
  int rows = 0;
  int cols = 0;  // LOGICAL columns, i.e. twice the stored byte width
};

// See the source-backed convention above. Change here and nowhere else.
inline constexpr bool kMxfp4LowNibbleFirst = true;

// The eight magnitudes of OCP E2M1: 1-2-1 with bias 1, no infinities, no NaN.
// code = s eem, value = {0, 0.5, 1, 1.5, 2, 3, 4, 6} with the sign bit on top.
float DecodeFp4E2M1(std::uint8_t code) noexcept;

// ---------------------------------------------------------------------------
// GEMV -- always y[rows] = W @ x[cols], FP32 accumulate
// ---------------------------------------------------------------------------
//
// Dequantization happens one output row at a time into a small scratch
// buffer, never into a full FP32 copy of the matrix. That is what keeps the
// resident set at roughly the size of the mapped quantized weights: a decode
// step is a GEMV, so each weight byte is touched exactly once and there is
// nothing to gain from materializing the matrix.

void fp8_block_gemv_cpu(const Fp8BlockMatrix& weight, const float* x, float* y);
void mxfp4_gemv_cpu(const Mxfp4Matrix& weight, const float* x, float* y);
void bf16_gemv_cpu(const std::uint16_t* weight, int rows, int cols, const float* x,
                   float* y);

// A single row of an FP8 block matrix, dequantized. Exposed for the tests.
void fp8_block_dequant_row_cpu(const Fp8BlockMatrix& weight, int row, float* out);
void mxfp4_dequant_row_cpu(const Mxfp4Matrix& weight, int row, float* out);

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

// out = weight * x / sqrt(mean(x^2) + eps). `weight_bf16` may be null for the
// has_weight=False variant (q_norm_without_weight). Returns the reciprocal.
float rms_norm_cpu(const float* x, const void* weight_bf16, float* out, int count,
                   float eps);

// ---------------------------------------------------------------------------
// Shared-KV sliding-window attention
// ---------------------------------------------------------------------------

struct SharedKvAttentionWeights {
  Fp8BlockMatrix wq_a;  // [1024, 4096]
  Fp8BlockMatrix wq_b;  // [32768, 1024]
  Fp8BlockMatrix wkv;   // [512, 4096]
  Fp8BlockMatrix wo_a;  // [8192, 4096], read as 8 groups of [1024, 4096]
  Fp8BlockMatrix wo_b;  // [4096, 8192]
  const std::uint16_t* q_norm = nullptr;   // [1024] BF16
  const std::uint16_t* kv_norm = nullptr;  // [512] BF16
  const float* attn_sink = nullptr;        // [64] FP32, one logit per head
};

// The sliding-window KV ring. One shared 512-wide row per token, so the whole
// window is 128 * 512 * 4 = 256 KB -- small enough to keep in FP32 and avoid
// a quantization step the emulation would then have to model.
class SharedKvWindowCache {
 public:
  explicit SharedKvWindowCache(int window = kSlidingWindow,
                               int head_dim = kAttnHeadDim);

  // Writes one token's post-RoPE KV row at `position`, evicting the row
  // `window` steps older. Returns the number of live tokens afterwards.
  int Append(const float* kv_row, int position);

  // Row for absolute `position`, or null when it has fallen out of the
  // window.
  const float* At(int position) const;

  int window() const { return window_; }
  int head_dim() const { return head_dim_; }
  int live_tokens() const { return live_; }
  int oldest_position() const;
  std::size_t bytes() const { return storage_.size() * sizeof(float); }

 private:
  int window_;
  int head_dim_;
  int live_ = 0;
  int newest_position_ = -1;
  std::vector<float> storage_;
};

struct AttentionStats {
  float q_lora_inv_rms = 0.0F;
  float kv_inv_rms = 0.0F;
  int attended_tokens = 0;
  // Softmax mass that the per-head attention sink absorbed, averaged over
  // heads. A sink near 1 means the head declined to attend to anything in the
  // window, which is the signal that the window or the position is wrong.
  float mean_sink_weight = 0.0F;
  float max_logit = 0.0F;
};

// One decode step of shared-KV SWA attention.
//
//   h_in       [D] FP32, already RMSNorm'd by the caller (attn_norm)
//   position   absolute token position, used for RoPE and the window
//   cache      updated in place with this token's KV row
//   out        [D] FP32, the sub-block FUNCTION output -- feed it to
//              mhc_post_cpu as h_out
//
// The pipeline, matching DeepseekV4Attention:
//   q_lora = rms_norm(wq_a @ h_in, q_norm)                      [1024]
//   q      = wq_b @ q_lora                                      [64, 512]
//   q[h]   = rms_norm(q[h], no weight)   (q_norm_without_weight)
//   q[h]   = rope_half(q[h], last 64 lanes)
//   kv     = rope_half(rms_norm(wkv @ h_in, kv_norm), last 64)  [512]
//   append kv to the window cache
//   logit[h][t] = (q[h] . kv[t]) * head_dim^-0.5
//   p[h]        = softmax([logit[h][.], attn_sink[h]])  -- the sink is an
//                 extra column whose weight is then DISCARDED, so the
//                 attended weights sum to less than 1
//   ctx[h]      = sum_t p[h][t] * kv[t]                          [512]
//   o_a[g]      = wo_a_group[g] @ ctx_slice[g]                   8 x [1024]
//   out         = wo_b @ concat(o_a)                             [4096]
void shared_kv_swa_attention_decode_cpu(const float* h_in,
                                        const SharedKvAttentionWeights& weights,
                                        SharedKvWindowCache* cache, int position,
                                        const float* rope_cos, const float* rope_sin,
                                        float* out, AttentionStats* stats);

// Populate the window cache for one position WITHOUT computing an output.
//
// Priming the window is the only thing a warmup token is for, and the KV row
// depends on none of the query or output machinery: it is wkv -> kv_norm ->
// RoPE, 2.1M MACs, against the ~107M a full decode step costs (wq_b alone is
// 32768x1024). Running the full step per warmup token made the attention
// stage 9x more expensive than the step being measured.
void shared_kv_swa_prefill_kv_cpu(const float* h_in,
                                  const SharedKvAttentionWeights& weights,
                                  SharedKvWindowCache* cache, int position,
                                  const float* rope_cos, const float* rope_sin);

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

// sqrt(softplus(x)). config.json "scoring_func": "sqrtsoftplus", which the
// device runs as the decomposed aclnnSoftplus -> aclnnSqrt chain with
// beta = 1 and threshold = 20 (softplus is linear above the threshold, which
// is what keeps exp() from overflowing).
float SqrtSoftplus(float logit) noexcept;

struct MoeRouting {
  int experts[kNumExpertsPerTok] = {};
  float weights[kNumExpertsPerTok] = {};
  int count = 0;
  bool from_hash_table = false;
};

// Layers [0, num_hash_layers): the selection is a pure function of the token
// id -- indices = tid2eid[token_id] -- and only the WEIGHTS come from the
// gate. See vllm-ascend tools/dsv4_moe_runtime/routing/hash_router.py and the
// `hash` branch of DeepseekV4MoE.__init__, where e_score_correction_bias is
// explicitly None for these layers. The checkpoint agrees: ffn.gate.tid2eid
// exists on exactly 3 layers and ffn.gate.bias on the other 40.
//
// `tid2eid_row` is the token's row of the table, `count` entries wide.
MoeRouting moe_route_hash_cpu(const void* tid2eid_row, const char* tid2eid_dtype,
                              int count, const float* gate_logits, int num_experts,
                              bool renormalize);

// Layers [num_hash_layers, num_layers): bias-shifted top-k selection
// ("topk_method": "noaux_tc"). The bias shifts the SELECTION only; the
// returned weight is the unshifted score, which is what noaux_tc means.
MoeRouting moe_route_score_cpu(const float* gate_logits, const float* bias,
                               int num_experts, int top_k, bool renormalize);

// SwiGLU with the clamp DSV4 uses:
//   gate = clamp(w1 @ x, max = limit)
//   up   = clamp(w3 @ x, -limit, limit)
//   h    = silu(gate) * up          (silu(z) = z * sigmoid(z))
//   out  = w2 @ h
// The clamp is asymmetric -- the gate is bounded ABOVE only -- which is what
// AscendSiluAndMulWithClamp.forward_oot does. Clamping the gate from below
// too would change the activation wherever w1 @ x is very negative.
void moe_expert_swiglu_mxfp4_cpu(const Mxfp4Matrix& w1, const Mxfp4Matrix& w3,
                                 const Mxfp4Matrix& w2, const float* x, float* out,
                                 float limit, std::vector<float>* scratch);

void moe_expert_swiglu_fp8_cpu(const Fp8BlockMatrix& w1, const Fp8BlockMatrix& w3,
                               const Fp8BlockMatrix& w2, const float* x, float* out,
                               float limit, std::vector<float>* scratch);

// ---------------------------------------------------------------------------
// Whole-MoE-block oracles: decode (M = 1, top-k experts) and prefill (M >> 1)
// ---------------------------------------------------------------------------
// These are the goldens for the two dsv4_moe_expert entry points. Both are
// thin compositions over moe_expert_swiglu_mxfp4_cpu above, deliberately:
// that function is already the authority for the asymmetric clamp and the
// per-group scale order, and a second transcription of the expert body here
// would be a second thing to keep in agreement with it.
//
// `w1`, `w3`, `w2` are ARRAYS of `num_experts` matrices, indexed by expert
// id. On device the same thing is one contiguous GM buffer per role with a
// fixed per-expert stride; see the tiling header.

// Decode: one token, `top_k` routed experts, router-weighted sum.
//
//   out = sum over slot s of  routing_weights[s] * Expert(topk_expert_ids[s])(x)
//
// The sum here runs over slots in ascending order, but that order is NOT part
// of the contract: the device kernel gives each slot its own AIV and
// accumulates with an atomic-add DataCopy, so the hardware order is whatever
// the cores finish in and varies run to run. Six partials of comparable
// magnitude reassociate within ~1e-7 relative, far inside the 1e-2 the suites
// allow, so this is a real but irrelevant difference -- stated because a
// comment claiming a fixed order would be false.
//
// The caller must zero `out` before an atomic accumulation on device; this
// oracle fills it, so the two agree without the caller doing anything.
void moe_decode_topk_mxfp4_cpu(const Mxfp4Matrix* w1, const Mxfp4Matrix* w3,
                               const Mxfp4Matrix* w2, const float* x,
                               const std::int32_t* topk_expert_ids,
                               const float* routing_weights, int top_k,
                               int num_experts, float* out, float limit,
                               std::vector<float>* scratch);

// Prefill: `total_tokens` tokens, each routed to exactly one expert.
//
//   y_tokens[t] = Expert(token_to_expert[t])(x_tokens[t])
//
// No router weighting: in the prefill path the dispatch has already grouped
// tokens by expert and the combine stage applies the weights, so a token here
// carries exactly one expert and an unweighted output. Rows are independent,
// so this is a per-token loop with no cross-token order to preserve.
void moe_prefill_batched_mxfp4_cpu(const Mxfp4Matrix* w1, const Mxfp4Matrix* w3,
                                   const Mxfp4Matrix* w2, const float* x_tokens,
                                   const std::int32_t* token_to_expert,
                                   int total_tokens, int num_experts,
                                   float* y_tokens, float limit,
                                   std::vector<float>* scratch);

// ---------------------------------------------------------------------------
// W4A16: the oracle for the CUBE prefill entry point
// ---------------------------------------------------------------------------
// The oracle above is W4A32 -- MXFP4 weights against FP32 activations, which
// is what the AIV kernels compute because they decode nibbles into FP32
// registers. `dsv4_moe_prefill_cube` cannot: the Cube's BF16 tensor path takes
// BF16 on BOTH sides, so the activations are rounded too and the answer is a
// DIFFERENT function. Diffing that kernel against the W4A32 oracle would
// charge it for a rounding the hardware forces, so this family mirrors the
// rounding instead.
//
// WHERE THE TWO DIFFER, PRECISELY -- there is exactly one new lossy step and
// it happens twice:
//
//   X    -> BF16   before the gate/up GEMMs      (RNE, <= 2^-9 relative)
//   act  -> BF16   before the down GEMM          (RNE, <= 2^-9 relative)
//
// Everything else is bit-for-bit the same arithmetic:
//
//   * MXFP4 -> BF16 is EXACT. An E2M1 magnitude is one of {0, .5, 1, 1.5, 2,
//     3, 4, 6}: three significant bits at most, against BF16's eight. The
//     E8M0 scale is a power of two, so the product only moves the exponent
//     field. Exact for scale codes 1..254 wherever the product stays inside
//     BF16's normal range -- and `Mxfp4DequantIsExactInBf16` below is how a
//     test states that domain instead of assuming it.
//   * the group scale's PLACEMENT is free. The W4A32 oracle scales each
//     32-element group SUBTOTAL; a Cube GEMM can only distribute the scale
//     into every product. Scaling by a power of two commutes with rounding,
//     so the two agree bit-exactly WITHIN a group. Across groups they differ
//     only by FP32 reassociation (<= cols * 2^-24), because the additions
//     then happen in a different order -- a real difference, bounded and
//     unrelated to the scale.
//   * the accumulation is FP32 on both sides. A product of two BF16 values
//     needs at most 16 significand bits, so every product is exact in FP32
//     and only the summation order is in question.

// RNE round of an FP32 value to BF16 precision, returned widened back to
// FP32. This is `Cast(..., RoundMode::CAST_RINT)` and torch's
// `.to(torch.bfloat16)`; it is the kernel's one lossy operation.
float bf16_round_cpu(float value) noexcept;

// True where `DecodeFp4E2M1(nibble) * DecodeFp8E8M0(scale_code)` is exactly
// representable in BF16 AND the device agrees -- i.e. where the "MXFP4 -> BF16
// is lossless" claim actually holds. It is the DOMAIN OF AGREEMENT, not just a
// representability test, so it excludes three things:
//
//   * the NaN scale code 0xFF. The device's decode turns it into +inf where
//     this oracle yields NaN -- documented on DecodeE8M0 in
//     include/moe/kernels/arch35/arch35_common.hpp, and the Cube path's
//     E8M0ToBf16 truncation does the same.
//   * a product that overflows. The largest E2M1 magnitude is 6 = 1.5 * 2^2,
//     so scale codes at the top of the range leave FP32, never mind BF16.
//   * a product that lands in BF16's SUBNORMAL range. The bit layouts happen
//     to align, so such a value is representable -- but the device flushes
//     subnormals to zero, so the two sides disagree there regardless. Scale
//     code 0 (2^-127) is the case that matters, and it is the same flush the
//     AIV path records.
//
// Real MXFP4 checkpoints do not place scales at either extreme; a synthetic
// weight set can, which is why a test states this domain rather than assuming
// the whole code range.
bool Mxfp4DequantIsExactInBf16(std::uint8_t nibble, std::uint8_t scale_code) noexcept;

// One row of an MXFP4 matrix dequantized the way the Cube path does it: each
// nibble's magnitude times its group's scale, the product then held at BF16
// precision. Inside the exact domain above the rounding is a no-op, which is
// the whole claim.
void mxfp4_dequant_row_bf16_cpu(const Mxfp4Matrix& weight, int row, float* out);

// y[row] = sum over c of W_bf16[row, c] * x_bf16[c], FP32 accumulate, natural
// column order, the group scale distributed into every product.
//
// `x_bf16` holds FP32 words that must ALREADY be BF16-representable -- pass
// them through bf16_round_cpu first. Taking FP32 here rather than packed
// uint16 keeps this callable from the same buffers the device wrapper is
// handed, and an un-rounded input would silently make this a W4A32 GEMM.
void mxfp4_gemv_w4a16_cpu(const Mxfp4Matrix& weight, const float* x_bf16, float* y);

// The clamped-SwiGLU expert over the W4A16 path. `x` is FP32 as the caller
// holds it; the rounding to BF16 happens inside, because it is part of the
// function the kernel computes and not a preprocessing step the caller may
// skip.
void moe_expert_swiglu_mxfp4_w4a16_cpu(const Mxfp4Matrix& w1, const Mxfp4Matrix& w3,
                                       const Mxfp4Matrix& w2, const float* x,
                                       float* out, float limit,
                                       std::vector<float>* scratch);

// Prefill, W4A16: the golden for `dsv4_moe_prefill_cube`. Same signature and
// same token-to-expert contract as moe_prefill_batched_mxfp4_cpu, so a suite
// can run both and attribute the difference to the activation rounding alone.
void moe_prefill_batched_mxfp4_w4a16_cpu(const Mxfp4Matrix* w1,
                                         const Mxfp4Matrix* w3,
                                         const Mxfp4Matrix* w2,
                                         const float* x_tokens,
                                         const std::int32_t* token_to_expert,
                                         int total_tokens, int num_experts,
                                         float* y_tokens, float limit,
                                         std::vector<float>* scratch);

// ---------------------------------------------------------------------------
// FLASH-STYLE TILED W4A16: the same function, never materializing the
// intermediate
// ---------------------------------------------------------------------------
// `moe_prefill_batched_mxfp4_w4a16_cpu` above is the monolithic shape: all
// `inter` gate/up values, then the activation, then the down GEMM. The tiled
// shape slices the CONTRACTION axis into chunks of R and keeps one chunk
// alive, dequantizing W1/W3/W2 at the moment each element is multiplied:
//
//   Y = 0
//   for r0 in 0, R, 2R, ... < inter:
//     gate/up = W1b[r0:r0+R, :] @ xb,  W3b[r0:r0+R, :] @ xb
//     act     = bf16(silu(min(gate, L)) * clamp(up, -L, L))
//     Y      += W2b[:, r0:r0+R] @ act
//
// There is no GM workspace of dequantized weights and no M x inter
// intermediate at any point; the scratch footprint is
// `moe_prefill_flash_scratch_floats()` and does not grow with `inter`.
//
// IT IS THE SAME FUNCTION UP TO ONE REASSOCIATION, and that is the precise
// claim tests/emu checks rather than assumes:
//
//   * gate, up and act are BIT-IDENTICAL to the monolithic oracle at every R.
//     gate[r] reduces over the hidden axis of ROW r alone and the rows are
//     independent, so slicing W1/W3's output rows reorders nothing.
//   * the DOWN GEMM's additions are reordered: the monolithic form sums
//     `inter` products into one running FP32 accumulator, the tiled form sums
//     each chunk into its own and adds the subtotal to Y. A two-level tree
//     against a flat sum -- pure FP32 reassociation, and the only difference.
//   * so the two extremes are bit-exact for opposite reasons: R >= inter is a
//     single chunk whose accumulator IS the flat sum; R == 1 adds each
//     product to Y on its own, which is also the flat sum. Only 1 < R < inter
//     is genuinely reassociated.
//
// `inter % R != 0` is supported: the final chunk is short. R is NOT required
// to be a multiple of the 32-element MXFP4 group -- the group scale is
// distributed into every product, so a chunk that splits a group changes
// nothing about the arithmetic -- but the device tiling will use a multiple
// of 32, and nothing here depends on that choice.

// The scratch requirement in floats: `hidden + 3 * min(R, inter)`, i.e.
// xb[hidden] plus gate/up/act over one chunk. Exposed so a test can hold the
// "nothing is materialized" claim to an arithmetic statement rather than to a
// comment: at fixed R it must be the same number for every `inter`. Returns 0
// for a nonpositive extent.
std::size_t moe_prefill_flash_scratch_floats(int hidden, int inter, int chunk) noexcept;

// One expert, tiled over the intermediate axis in chunks of `chunk` (= R).
// `x` is FP32 as the caller holds it; the BF16 rounding happens inside and
// ONCE for the whole token, not once per chunk. Throws std::invalid_argument
// on a nonpositive R -- a zero chunk would skip the reduction and return a
// zeroed output, which is the one failure mode an oracle must not have.
void moe_expert_swiglu_mxfp4_w4a16_flash_cpu(const Mxfp4Matrix& w1,
                                             const Mxfp4Matrix& w3,
                                             const Mxfp4Matrix& w2,
                                             const float* x, float* out,
                                             float limit, int chunk,
                                             std::vector<float>* scratch);

// Prefill over a batch of tokens, tiled. Same token-to-expert contract as
// moe_prefill_batched_mxfp4_w4a16_cpu plus the chunk size, so a suite can run
// both over one fixture and attribute the whole difference to the down GEMM's
// reassociation.
void moe_prefill_flash_ffn_w4a16_cpu(const Mxfp4Matrix* w1,
                                     const Mxfp4Matrix* w3,
                                     const Mxfp4Matrix* w2,
                                     const float* x_tokens,
                                     const std::int32_t* token_to_expert,
                                     int total_tokens, int num_experts,
                                     float* y_tokens, float limit, int chunk,
                                     std::vector<float>* scratch);

// One logical element of an MXFP4 matrix, dequantized exactly as the Cube
// path does it: decode the nibble, multiply by the group's power-of-two E8M0
// scale, hold the product at BF16. Shared by the 1D-strip reference above and
// the 2D block-tiled Cube model below so that the nibble order and the
// group-scale indexing exist in ONE place. Unchecked -- it is an inner-loop
// primitive and its callers validate the matrix and the indices first.
float mxfp4_dequant_element_bf16_cpu(const Mxfp4Matrix& w, int row, int col) noexcept;

// ---------------------------------------------------------------------------
// TWO-PHASE ON-CHIP FUSED CUBE PIPELINE (Ascend 950PR, DaVinci v3 / dav-3510)
// ---------------------------------------------------------------------------
// `moe_prefill_flash_ffn_w4a16_cpu` above tiles ONE axis -- the contraction --
// and reduces each output row whole. That is the AIV/Vector shape and it is
// reserved for that path. The Cube cannot be driven that way: it computes
// fractal GEMMs out of L0A/L0B into an L0C accumulator, so the schedule that
// fits it streams three contractions through fixed-size on-chip buffers.
//
// THIS MODEL REPLACED A NESTED (m, h) OUTPUT GRID, and the reason is the whole
// point of the rewrite. The previous schedule put the gate/up GEMMs INSIDE the
// hidden-tile loop, which recomputed the entire intermediate once per H tile
// -- 64x the gate/up MACs at hidden 4096, H_tile 64. Correct, and unusable.
// The two-phase schedule below pays for a resident `act[M_tile, inter]` plane
// instead and evaluates gate and up EXACTLY ONCE, which is a property the
// suite asserts as an arithmetic identity on the trace rather than a claim in
// a comment:
//
//     gate_up_macs == 2 * M * inter * hidden          (multiplier 1.0x)
//     act_elements_written == M * inter               (each stored once)
//
// THE SCHEDULE MODELLED, exactly as a MIX AIC+AIV kernel would run it. There
// are TWO phases, separated in time, over a grid whose only axis is the token
// tile -- there is no (m, h) grid any more:
//
//   for each token tile m0 (M_tile = 16 rows)     <- the grid; no barriers
//     act[M_tile, inter]                          <- BF16, resident in UB
//
//     PHASE 1  gate / up / SwiGLU
//     for r0 in 0, R_sub, ... < inter             <- R_sub = 128
//       gate/up[M_tile, R_sub] = 0                <- FP32 accumulator
//       for k0 in 0, K_sub, ... < hidden          <- K_sub = 128
//         X[M_tile, K_sub] -> L0A                 <- ONE read, used TWICE
//         W1[R_sub, K_sub], W3[R_sub, K_sub]      <- dequantized per sub-tile
//         gate += X * W1^T,  up += X * W3^T       <- mmad, FP32
//       act[:, r0 : r0 + R_sub] =                 <- Vector, clamped SwiGLU
//           bf16(silu(min(gate, L)) * clamp(up, -L, L))
//
//     PHASE 2  down projection
//     for h0 in 0, H_sub, ... < hidden            <- H_sub = 64
//       Y_acc[M_tile, H_sub] = 0                  <- FP32 accumulator
//       for r0 in 0, R_sub2, ... < inter          <- R_sub2 = 256
//         W2[H_sub, R_sub2] -> L0B                <- dequantized per sub-tile
//         Y_acc += act[:, r0 : r0 + R_sub2] * W2^T
//       write Y_acc to GM                         <- once per (m0, h0)
//
// THREE-LEVEL TILE DECOMPOSITION, which is what the sub-tile extents name:
//   macro   the token tile in GM           M_tile x hidden, M_tile x inter
//   sub     what is staged on chip         L1 / UB: the R_sub, K_sub, H_sub,
//                                          R_sub2 blocks above
//   micro   what one mmad consumes         the 16x16x16 fractal, kCubeFractal
//
// ZERO GM INTERMEDIATE. `act` never reaches GM, and neither does a
// dequantized weight: every W1/W3/W2 sub-tile is decoded from MXFP4 at the
// moment it is staged. The host scratch IS the modelled buffer set, so "no
// global workspace" is a property of this file -- `moe_prefill_cube_2d_scratch_floats`
// is the formula, and it grows with M_tile * inter (the `act` residency) and
// with the sub-tile extents, and with NOTHING else.
//
// WHAT IS MODELLED, AND WHAT IS DELIBERATELY NOT
//
// Modelled, because each one can change the answer or the feasibility:
//   * the 1-D token-tile grid and the fact that tiles share NOTHING -- the
//     visit order is a parameter precisely so a test can prove it does not
//     matter;
//   * the FP32 accumulator, written to GM exactly once per (m0, h0) block;
//   * the three-level reduction (fractal C0 = 16, then the sub-tile block,
//     then the axis), each level accumulating in FP32;
//   * on-the-fly dequantization into per-sub-tile fragments;
//   * the BF16 dtypes the budget is charged at, including the UB residency of
//     `act`, even though the host holds those fragments in float.
//
// NOT modelled, stated because a golden that pretends otherwise stops being
// one:
//   * the order of the 16 products inside one mmad. The ISA does not specify
//     it; natural order is one admissible choice, and the error bound the
//     suite derives covers any of them.
//   * MTE1/MTE2 timing, BufferID ownership, double-buffer ping-pong. Buffer
//     depth enters only as a multiplier on the footprint budget, which is the
//     one way it can make a geometry infeasible.
//   * WHICH on-chip buffer holds the level-2 subtotal. The schedule above
//     writes it L0C-resident; a kernel driven through the high-level Matmul
//     API with a per-sub-tile B must instead accumulate the subtotals in an
//     FP32 UB buffer, because L0C does not survive between separately driven
//     single-core GEMMs. The ARITHMETIC is the same either way -- an FP32 sum
//     of per-sub-tile subtotals -- and that is all this file claims.
//   * the device's `Silu`. It is a vector intrinsic over the hardware's
//     exp/reciprocal, not std::exp, and its error is not derivable from the
//     format -- the one term a device comparison must measure rather than
//     derive.
//
// THE SCHEDULE'S RESIDUAL COST, recorded here because it is the price of
// removing the 64x recompute and the kernel author must see it before writing
// the kernel rather than after: with r outer and k inner, the X macro-tile is
// re-read from GM once per r0 step, i.e. ceil(inter / R_sub) times per token
// tile. That is a DMA cost and not a MAC cost -- gate and up are still
// evaluated exactly once -- and `CubeTileTrace::x_fragment_reads` reports it
// as an observed number.
//
// FIVE PROPERTIES THIS MODEL EXISTS TO PIN DOWN, each checked in
// tests/emu/test_dsv4_moe_prefill_cube_2d_emu.cpp rather than asserted here:
//
//   1. SPATIAL INDEPENDENCE, operationally. Token tiles share no state, so
//      the VISIT ORDER cannot change the answer. `CubeTileGeometry::visit`
//      exists for exactly that test: all four traversals must agree BIT FOR
//      BIT. An implementation that leaked a scratchpad or an accumulator
//      between tiles would fail it, which is a stronger statement than "the
//      source contains no barrier".
//   2. WRITE-ONCE COVERAGE. The tiles partition the output exactly: the trace
//      counts the elements written and the suite poisons the destination
//      first, so neither a gap nor an overlap can pass. `act` is covered the
//      same way, which is what makes the 1.0x multiplier an observation.
//   3. THE BUFFER BUDGET IS A HARD CONSTRAINT, not a comment. The footprint
//      of every fragment actually staged is computed during the run and
//      reported through `CubeTileTrace`, so the suite asserts an OBSERVED
//      peak against the platform's L0A/L0B/L0C/L1 sizes AND against UB, which
//      the resident `act` plane now puts in the budget.
//   4. THE REDUCTION HIERARCHY IS THREE-LEVEL, and that is what separates this
//      model from the 1D one numerically. See the note on `CubeTileGeometry`.
//   5. ZERO REDUNDANT COMPUTATION, as the two identities above.
//
// W4A16 semantics are unchanged from the rest of the family: X rounded to
// BF16 (once -- the rounding is idempotent, so doing it per fragment is the
// same function), exact power-of-two E8M0 scaling, lossless E2M1 -> BF16, and
// FP32 accumulation.

// DaVinci v3 per-AI-core capacities, from the toolkit's own platform configs
// (x86_64-linux/data/platform_config/Ascend950PR_*.ini: l0_a/l0_b = 65536,
// l0_c_size = 262144, l1_size = 524288, ub_size = 253952) and matching the
// whitepaper's Table in ascend_950PR_950DT_specs.markdown. The ini is the
// authority: its UB figure (253,952 B) is the one the whitepaper's
// per-subsystem "512 KB UB" contradicts, so the two are not interchangeable
// as sources even where they agree on the L0 sizes. UB is PER VECTOR CORE and
// a subsystem holds two of them, which is the factor of two the paper hides.
inline constexpr std::size_t kCubeL0ABytes = 65536;   // 64 KiB
inline constexpr std::size_t kCubeL0BBytes = 65536;   // 64 KiB
inline constexpr std::size_t kCubeL0CBytes = 262144;  // 256 KiB
inline constexpr std::size_t kCubeL1Bytes = 524288;   // 512 KiB
inline constexpr std::size_t kCubeUbBytes = 253952;   // 248 KiB, per AIV

// The fractal unit. C0 for BF16 is 16 ELEMENTS (32 B) -- 64 is the C0 of the
// b4 family, which does not apply here because this path dequantizes to BF16
// before the Cube sees anything. The MAC array's K depth is the same 16, so
// one `mmad` reduces 16 products and the accumulator accumulates between
// fractals: that is where the three-level reduction below comes from.
inline constexpr int kCubeFractal = 16;

// How the grid walks the work. Scheduling only: a correct implementation must
// give the same answer for all four, and the suite checks that rather than
// taking it on trust.
//
// The grid's only independent axis is the TOKEN TILE, so these permute the
// token tiles. Phase 2's h0 loop is inside a tile body and cannot be a grid
// axis -- `act` must exist before any of it runs -- but its DIRECTION is just
// as irrelevant to the answer, so the enum carries that too and no two of the
// four name the same traversal:
//
//   kRowMajor      token tiles ascending,  phase-2 h blocks ascending
//   kColumnMajor   token tiles ascending,  phase-2 h blocks DESCENDING
//   kReversed      token tiles descending, phase-2 h blocks descending
//   kShuffled      token tiles permuted,   phase-2 h blocks ascending
enum class CubeTileVisit {
  kRowMajor,
  kColumnMajor,
  kReversed,
  kShuffled,
};

// THE REDUCTION HIERARCHY, because it is the whole numerical difference
// between this model and the flat 1D one:
//
//   level 0   one mmad, C0 = 16 products, summed in natural order
//   level 1   one sub-tile block (K_sub for gate/up, R_sub2 for down), the
//             fractals accumulated in FP32
//   level 2   the whole contraction axis, the blocks accumulated in FP32
//
// The monolithic oracle sums the same products FLAT. So the two differ by
// FP32 reassociation on every GEMM -- bounded, and bounded by the depth of
// the trees rather than by anything measured. Level 0's internal order is NOT
// specified by the ISA; natural order is one admissible choice and the bound
// the suite derives covers any of them.
//
// WHICH EXTENT BLOCKS WHICH REDUCTION, since the two phases no longer share
// one: gate and up reduce over `hidden` in blocks of `k_tile` (K_sub), and
// the down projection reduces over `inter` in blocks of `r_tile_down`
// (R_sub2). `r_tile` and `h_tile` are OUTPUT extents and enter no reduction,
// so they cannot change the answer -- which is claim 1 of the suite.
//
// One consequence is worth stating because a tolerance that ignores it is
// wrong: the gate/up reassociation is amplified by the BF16 store of `act`.
// A gate value within the reassociation of a BF16 tie rounds the other way,
// so `act` can differ by ONE BF16 ULP -- 2^-9 relative, four orders above the
// FP32 reassociation that caused it. The suite measures that flip set and
// bounds its contribution separately instead of hiding it in one number.
struct CubeTileGeometry {
  // The grid: tokens per token tile. 16 at the production geometry, which is
  // what keeps `act[16, inter]` at exactly 64 KiB there (16 * 2048 * 2 B).
  //
  // Stated without overclaiming: M_tile = 32 doubles BOTH the `act` plane
  // (to 128 KiB) and the FP32 accumulators, and at R_sub = 128 this model's
  // budget still admits it -- the 64 KiB saved by the smaller sub-tiles is
  // what buys the room. It is not admitted because it is a good idea: a real
  // kernel also carries the Matmul framework's own UB and the queue depths
  // this model does not charge, so 32 is close enough to the ceiling that the
  // production geometry does not take it.
  int m_tile = 16;
  // PHASE 1. r_tile is the inter-axis OUTPUT sub-tile (R_sub); k_tile is the
  // hidden-axis CONTRACTION sub-tile (K_sub).
  int r_tile = 128;
  int k_tile = 128;
  // PHASE 2. h_tile is the hidden-axis OUTPUT sub-tile (H_sub); r_tile_down
  // is the inter-axis CONTRACTION sub-tile (R_sub2). They are separate knobs
  // from phase 1's because the two phases are separated in time and share no
  // buffer except `act`, so there is no reason to tie them -- and the down
  // GEMM's B fragment is W2[H_sub, R_sub2], a different shape from
  // W1[R_sub, K_sub].
  int h_tile = 64;
  int r_tile_down = 256;
  // Buffer slots per operand: 1 for a serial loop, 2 for the double buffering
  // a real kernel needs to overlap MTE1/MTE2 with the Cube. The budget is
  // charged for all of them, which is why it is part of the geometry and not
  // an implementation detail. `act` is charged at depth 1 whatever this says:
  // it is a single resident plane, not a streamed fragment.
  int buffer_depth = 2;
  CubeTileVisit visit = CubeTileVisit::kRowMajor;
  std::uint64_t visit_seed = 0;
};

// Bytes resident in each on-chip buffer, charged at the hardware's dtypes:
// the A/B fragments are BF16 (2 B) even though this host model holds them in
// float, and the accumulators are FP32 (4 B).
//
// THE TWO PHASES ARE SEPARATED IN TIME, so each buffer is charged the LARGER
// of the two phases and not their sum. That is the one modelling decision in
// here that could let an infeasible geometry look feasible, so it is stated:
// it is sound only because phase 2 does not start until the last `act` column
// is stored, which is exactly what makes the schedule non-redundant.
//
//   L0A   phase 1  X[M, K_sub] * 2 B * depth
//         phase 2  act[M, R_sub2] * 2 B * depth
//   L0B   phase 1  W1/W3[R_sub, K_sub] * 2 B * depth   (one slot: W1 then W3)
//         phase 2  W2[H_sub, R_sub2] * 2 B * depth
//   L0C   phase 1  (gate[M, R_sub] + up[M, R_sub]) * 4 B    together, not max
//         phase 2  Y_acc[M, H_sub] * 4 B
//   L1    phase 1  (2 * R_sub * K_sub + M * K_sub) * 2 B * depth
//         phase 2  (H_sub * R_sub2) * 2 B * depth
//   UB    BOTH     act[M_tile, inter] * 2 B, at depth 1, plus the phase's own
//                  FP32 accumulators, plus the phase's dequant staging (the
//                  weight sub-tile as packed nibbles and as BF16, since the
//                  expansion is a VECTOR operation and happens in UB before
//                  anything reaches L1) -- the resident plane is why `inter`
//                  is a parameter of the budget at all.
struct CubeBufferFootprint {
  std::size_t l0a_bytes = 0;
  std::size_t l0b_bytes = 0;
  std::size_t l0c_bytes = 0;
  std::size_t l1_bytes = 0;
  std::size_t ub_bytes = 0;

  bool Fits() const {
    return l0a_bytes <= kCubeL0ABytes && l0b_bytes <= kCubeL0BBytes &&
           l0c_bytes <= kCubeL0CBytes && l1_bytes <= kCubeL1Bytes &&
           ub_bytes <= kCubeUbBytes;
  }
};

// The footprint of ONE phase's fragment set. Separate entry points rather
// than one with a mode flag, because the two phases stage different shapes
// and a caller that passed the wrong extents to a merged function would get a
// plausible number back.
//
// `m_extent` and the contraction extents are the ACTUAL (possibly ragged)
// ones; `inter` is the problem's, since `act` is allocated for the whole axis
// whatever the ragged tail does. A ragged fragment is a strict subset of a
// full one, so the full-sub-tile figure is the peak -- which the trace's
// observed peak confirms rather than assumes.
CubeBufferFootprint moe_prefill_cube_phase1_footprint(int m_extent, int r_extent,
                                                      int k_extent, int inter,
                                                      int buffer_depth) noexcept;
CubeBufferFootprint moe_prefill_cube_phase2_footprint(int m_extent, int h_extent,
                                                      int r_down_extent, int inter,
                                                      int buffer_depth) noexcept;

// The elementwise maximum of the two phases at FULL sub-tile extents: the
// peak the geometry must fit. Takes `inter` because the `act` residency does.
CubeBufferFootprint moe_prefill_cube_tile_footprint(const CubeTileGeometry& geometry,
                                                    int inter) noexcept;

// Is this geometry one the hardware can run at this `inter`? Three separate
// conditions, kept separate because they fail for different reasons:
//   * every sub-tile extent positive and a whole multiple of the 16x16x16
//     fractal;
//   * buffer_depth in 1..2;
//   * the full-sub-tile peak inside L0A/L0B/L0C/L1 AND inside UB, the last of
//     which is the constraint the resident `act` plane adds and the one that
//     fixes M_tile at 16 for the production geometry.
bool moe_prefill_cube_tile_geometry_admissible(const CubeTileGeometry& geometry,
                                               int inter) noexcept;

// Scratch, in floats: the host mirror of the modelled buffer set, which is
// what makes "no global workspace" checkable. Sized by the SUB-TILE extents
// plus the one term that is deliberately not sub-tile sized -- act[M_tile,
// inter], the resident plane the whole schedule is built around. It does not
// grow with `hidden` or the token count, and it grows with `inter` LINEARLY
// IN M_TILE and not in M: that difference is the claim.
std::size_t moe_prefill_cube_2d_scratch_floats(const CubeTileGeometry& geometry,
                                               int inter) noexcept;

// What the run actually did. Everything here is OBSERVED during the loop, not
// recomputed from the geometry afterwards, so the suite's assertions are about
// the execution and not about the formula.
struct CubeTileTrace {
  // The largest footprint any single fragment set reached, per phase and
  // overall. Kept per phase as well as merged because the two phases load
  // different shapes and a single peak would not say which one is tight.
  CubeBufferFootprint peak;
  CubeBufferFootprint phase1_peak;
  CubeBufferFootprint phase2_peak;
  // Grid shape and coverage.
  std::uint64_t token_tiles = 0;               // the grid's extent
  std::uint64_t output_tiles = 0;              // (m0, h0) GM writes
  std::uint64_t output_elements_written = 0;   // must equal M * hidden exactly
  // `act` stores. The TWO-PHASE schedule's defining property: each of the
  // M * inter activations is computed and stored EXACTLY ONCE. The previous
  // (m, h)-grid schedule would report M * inter * ceil(hidden / H_tile) here,
  // so this counter is the redundancy check in its sharpest form.
  std::uint64_t act_elements_written = 0;      // must equal M * inter exactly
  // The largest and smallest actual fragment extents seen, so a ragged run
  // can be told apart from a run that quietly skipped its tails.
  int max_m_extent = 0;
  int max_h_extent = 0;
  int max_r_extent = 0;
  int max_k_extent = 0;
  int max_r_down_extent = 0;
  int min_m_extent = 0;
  int min_h_extent = 0;
  int min_r_extent = 0;
  int min_k_extent = 0;
  int min_r_down_extent = 0;
  // Multiply-accumulates issued, by phase. gate_up_macs must now be EXACTLY
  // 2 * M * inter * hidden -- the single-pass work, multiplier 1.0x -- and
  // down_macs exactly M * hidden * inter.
  std::uint64_t gate_up_macs = 0;
  std::uint64_t down_macs = 0;
  // The X macro-tile is re-read once per phase-1 r0 step, which is this
  // schedule's residual cost. Observed, so the suite asserts the factor
  // instead of the comment above being the only record of it.
  std::uint64_t x_fragment_reads = 0;
  // How many times a weight SUB-TILE was dequantized into its L0B slot. This
  // is the B-fragment reuse the Cube exists for, as a number: a token tile
  // whose tokens share one expert stages each sub-tile ONCE and reduces all M
  // rows against it, while a tile straddling an expert boundary must restage.
  // The golden stays per-token correct either way -- it does not assume the
  // dispatch grouped the tokens -- and this counter is how the suite shows
  // what that assumption would have bought.
  std::uint64_t gate_up_fragment_stagings = 0;
  std::uint64_t down_fragment_stagings = 0;
};

// The two-phase Cube golden. Same token-to-expert contract as
// `moe_prefill_batched_mxfp4_w4a16_cpu`, so a suite can run both over one
// fixture and attribute the whole difference to the reduction hierarchy.
// `trace` may be null. Throws std::invalid_argument on a null operand, a
// ragged expert set, an out-of-range expert id, or an inadmissible geometry --
// never silently returns a zeroed output, which for a golden is the one
// unacceptable failure mode.
void moe_prefill_cube_flash_2d_w4a16_cpu(const Mxfp4Matrix* w1,
                                         const Mxfp4Matrix* w3,
                                         const Mxfp4Matrix* w2,
                                         const float* x_tokens,
                                         const std::int32_t* token_to_expert,
                                         int total_tokens, int num_experts,
                                         float* y_tokens, float limit,
                                         const CubeTileGeometry& geometry,
                                         CubeTileTrace* trace,
                                         std::vector<float>* scratch);

// ---------------------------------------------------------------------------
// Activation block quantizer (oracle for the bf16_block_quant_fp8 operator)
// ---------------------------------------------------------------------------
// BF16/FP32 -> FP8 E4M3 with one FP32 scale per 128 elements:
//
//   m        = max over the block of |x|
//   scale[b] = (m > 0) ? m / 448 : 1        FP32, NOT a power of two
//   q[i]     = e4m3_rne_sat( x[i] * (448 / m) )
//
// This replaces the aclnnDynamicMxQuant + aclnnCast pair the five activation
// quantizer sites used. DynamicMxQuant emitted OCP E8M0 power-of-two scales;
// these are arbitrary FP32, which is the dtype QuantBatchMatmulV3 takes
// directly and which uses the E4M3 range strictly better. Outputs therefore
// differ from the old path by design.
//
// `width` must be a multiple of 128. `q_out` holds `width` codes, `scale_out`
// holds `width / 128` floats. BF16 input is STORAGE BITS so the same buffer
// can be handed to the device wrapper unchanged.
inline constexpr int kBlockQuantRefBlockSize = 128;

void bf16_block_quant_fp8_cpu(const std::uint16_t* x_bf16, int width,
                              std::uint8_t* q_out, float* scale_out);

// FP32 activations are consumed at full precision, without rounding to BF16.
void bf16_block_quant_fp8_cpu(const float* x, int width,
                            std::uint8_t* q_out, float* scale_out);

// Decode one code against its block scale. The dequantization the consuming
// matmul performs, factored out so tests state the round trip once.
float bf16_block_quant_fp8_dequant(std::uint8_t code, float scale) noexcept;

// ---------------------------------------------------------------------------
// Resident-set reporting
// ---------------------------------------------------------------------------

struct MemoryUsage {
  std::size_t current_rss_bytes = 0;  // VmRSS
  std::size_t peak_rss_bytes = 0;     // VmHWM -- the number the budget is on
  bool available = false;
};

// Reads /proc/self/status. `available` is false where that is not present.
MemoryUsage ReadMemoryUsage();

}  // namespace moe::emu

#endif  // MOE_EMU_DSV4_LAYERS_H
