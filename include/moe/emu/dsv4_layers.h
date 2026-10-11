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
// NIBBLE ORDER -- AN ASSUMPTION THAT THE CHECKPOINT CANNOT SETTLE
//   `packed` holds two logical elements per byte. This emulation reads logical
//   column 2i from the LOW nibble and 2i+1 from the HIGH nibble, which is the
//   dominant MX packing convention.
//
//   It cannot be confirmed from the data. Swapping the order permutes elements
//   within each adjacent pair, which leaves the value MULTISET -- and therefore
//   every distributional statistic, including the per-group max that sets the
//   E8M0 scale -- completely unchanged. No test on the weights alone can tell
//   the two apart; only a device run or a documented packing can. The order is
//   therefore isolated in `kMxfp4LowNibbleFirst` below so it can be flipped in
//   exactly one place, and the bench reports which convention it used.
struct Mxfp4Matrix {
  const std::uint8_t* packed = nullptr;  // [rows, cols/2] packed nibbles
  const std::uint8_t* scale = nullptr;   // [rows, cols/32] E8M0
  int rows = 0;
  int cols = 0;  // LOGICAL columns, i.e. twice the stored byte width
};

// See the note on Mxfp4Matrix. Change here and nowhere else.
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
