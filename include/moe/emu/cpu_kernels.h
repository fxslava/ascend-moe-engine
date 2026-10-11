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

// cpu_kernels.h -- host CPU emulation of the three fused DeepSeek-V4-Flash
// decode operators, in plain C++17 with no CANN, no NPU runtime and no torch.
//
// WHY THIS LAYER EXISTS
//   The CAModel NPU simulator takes minutes per iteration. These kernels run
//   a Layer 0 decode step in milliseconds, which makes them usable as (a) the
//   CPU fallback path, (b) the golden numerical reference the Ascend C kernels
//   are checked against, and (c) a place to settle layout questions before
//   paying for a device build.
//
// WHERE THE MATH COMES FROM
//   Every formula here is ported from an upstream CPU/NumPy golden, not
//   reinvented. The provenance, file by file:
//
//     mHC + Sinkhorn   ops-transformer
//                        experimental/mhc/mhc_pre_sinkhorn_premix/tests/
//                        assets/golden.py  (`_compute`)
//                      and mhc/mhc_post/tests/assets/golden.py
//                        (`_mhc_post_compute`)
//     Compressor       ops-transformer
//                        attention/compressor/tests/pytest/
//                        compressor_golden.py  (`cpu_compressor`,
//                        `softmax_columns`), with the RMSNorm + RoPE epilog
//                        from third_party/ops_dsv4/attention/compressor
//                        (README formula 6-7, op_kernel/arch35/vf/vf_rope.h)
//     Indexer top-k    ops-transformer
//                        attention/quant_lightning_indexer/tests/pytest/
//                        quant_lightning_indexer_golden.py
//                        (`GeneralizedQLI::cal_atten_per_batch_fp8`)
//
//   Three behaviours in those goldens look like noise and are not. They are
//   reproduced deliberately and each is commented at its site:
//     - the Sinkhorn `+ hc_eps` cadence is asymmetric (added once after the
//       initial row softmax, then only ever used as a denominator guard);
//     - the compressor's column softmax flushes FP32 denormals to zero, which
//       is what the NPU `exp` does;
//     - the indexer rounds scores to BF16 *before* selecting, so ties are
//       common and the tie-break (lower index wins) is part of the contract.
//
// ACCURACY CONTRACT
//   All reductions accumulate in FP32, matching the NPU kernels, which is why
//   this is a bit-comparable reference rather than merely a close one. BF16
//   inputs are widened to FP32 on read and the single BF16 store at the end
//   rounds to nearest even. No FMA contraction is assumed: build with
//   -ffp-contract=off if you need the host result to be reproducible across
//   compilers (the emu CMake target does this).
//
// THREADING
//   Every function is pure and reentrant except for the explicit
//   `CompressorRingState` the caller owns. Nothing here keeps global state.

#ifndef MOE_EMU_CPU_KERNELS_H
#define MOE_EMU_CPU_KERNELS_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace moe::emu {

// ---------------------------------------------------------------------------
// Geometry -- every number below is from the published DeepSeek-V4-Flash
// config.json, cross-checked against the shard-2/shard-4 safetensors headers.
// ---------------------------------------------------------------------------

// mHC residual streams. config.json: "hc_mult": 4, "hidden_size": 4096,
// "hc_sinkhorn_iters": 20, "hc_eps": 1e-06.
inline constexpr int kHcStreams = 4;                              // n
inline constexpr int kHiddenSize = 4096;                          // D
inline constexpr int kHcFlatWidth = kHcStreams * kHiddenSize;      // 16384
inline constexpr int kHcProjectionRows = kHcStreams * kHcStreams + 2 * kHcStreams;  // 24
inline constexpr int kHcAlphaElements = 3;
inline constexpr int kHcSinkhornIters = 20;
inline constexpr float kHcEpsilon = 1e-6F;
inline constexpr float kRmsNormEpsilon = 1e-6F;
// Confirmed by `layers.0.hc_attn_fn` = F32 [24, 16384] in shard 2.
inline constexpr int kHcPhiRows = kHcProjectionRows;
inline constexpr int kHcPhiCols = kHcFlatWidth;

// Token compressor, "C4A" configuration. From the vendored op's constraint
// table (third_party/ops_dsv4/attention/compressor/README.md): C4A is
// D=512, coff=2, cmp_ratio=4. Confirmed by shard 4:
//   layers.2.attn.compressor.wkv.weight   BF16 [1024, 4096] = [coff*D, H]
//   layers.2.attn.compressor.ape          F32  [4, 1024]    = [ratio, coff*D]
//   layers.2.attn.compressor.norm.weight  BF16 [512]        = [D]
inline constexpr int kCompressorHeadDim = 512;    // D
inline constexpr int kCompressorRatio = 4;        // cmp_ratio
inline constexpr int kCompressorCoff = 2;         // coff
inline constexpr int kCompressorStateWidth = kCompressorCoff * kCompressorHeadDim;  // 1024
inline constexpr int kCompressorReduceRows = kCompressorCoff * kCompressorRatio;    // 8
// config.json: "qk_rope_head_dim": 64. The vendored kernel applies RoPE to
// the LAST ropeHeadDim lanes: vf_rope.h is called with
// baseAddr = headDim - ropeHeadDim = 448.
inline constexpr int kRopeHeadDim = 64;
inline constexpr int kRopeBaseOffset = kCompressorHeadDim - kRopeHeadDim;  // 448

// Lightning indexer. config.json: "index_n_heads": 64, "index_head_dim": 128,
// "index_topk": 512.
inline constexpr int kIndexerHeads = 64;
inline constexpr int kIndexerHeadDim = 128;
inline constexpr int kIndexerTopK = 512;

// RoPE rotary modes, matching the vendored op's `rotary_mode` attribute.
enum class RotaryMode : int {
  kHalf = 1,        // vf_rope.h HalfModeRopeVF   -- the DSV4 default
  kInterleave = 2,  // vf_rope.h InterleaveModeRopeVF
};

// ---------------------------------------------------------------------------
// BF16 <-> FP32
// ---------------------------------------------------------------------------
//
// Round-to-nearest-even on the store, which is what the Ascend `Cast` with
// RoundMode::CAST_RINT does and what torch's `.to(torch.bfloat16)` does.
// A NaN is kept a NaN rather than being allowed to round into an infinity.

inline float Bf16ToFloat(std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline std::uint16_t FloatToBf16(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7F800000U;
  const std::uint32_t mantissa = bits & 0x007FFFFFU;
  if (exponent == 0x7F800000U && mantissa != 0U) {
    // Quiet the NaN into the top mantissa bit so it survives the truncation.
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7FFFU + ((bits >> 16U) & 1U);  // round to nearest even
  return static_cast<std::uint16_t>(bits >> 16U);
}

// ===========================================================================
// 1. mHC fused Sinkhorn GEMV
// ===========================================================================

// The intermediates the fused operator produces on the way to `x_next`. The
// pipeline needs `h_in` (it is the attention / MoE sub-block input) and the
// tests need `birkhoff` (to assert double stochasticity), so the extended
// entry point below hands them back rather than discarding them.
struct MhcSinkhornTrace {
  // RMSNorm reciprocal over the flattened [n*D] stream: the golden's
  // `inv_rms = 1/sqrt(mean(x*x) + norm_eps)`.
  float inv_rms = 0.0F;
  // phi @ flat_x, BEFORE the RMS scale (the golden's `hc_before_norm`).
  float projection[kHcProjectionRows] = {};
  // sigmoid(normalized[:n] * alpha0 + bias[:n]) + hc_eps  -- the pre-mix gains.
  float h_pre[kHcStreams] = {};
  // 2 * sigmoid(normalized[n:2n] * alpha1 + bias[n:2n]) -- the post gains.
  float h_post[kHcStreams] = {};
  // normalized[2n:] * alpha2 + bias[2n:], reshaped [n, n]: the raw matrix M
  // that Sinkhorn is run on.
  float residual_logits[kHcStreams * kHcStreams] = {};
  // The doubly stochastic result, row-major [n, n]. This is `hRes`.
  float birkhoff[kHcStreams * kHcStreams] = {};
  // sum_i x[i, :] * h_pre[i]  -- the [1, D] sub-block input.
  float h_in[kHiddenSize] = {};
};

// The full operator, exposing the trace.
//
//   x            [1, n, D] BF16, TND          the four residual streams
//   phi          [24, 16384] FP32             mixing projection
//   alpha        [3] FP32                     per-segment gains
//   bias         [24] FP32                    per-row biases
//   h_out        [1, D] BF16                  the sub-block's FUNCTION output
//   x_next       [1, n, D] BF16               written: B^T x + h_out (x) h_post
//   trace        written: see MhcSinkhornTrace
//
// `num_iters` Sinkhorn iterations are run, each one a row normalization
// followed by a column one; the first row step is the initial softmax. With
// num_iters = 20 the result is doubly stochastic to within 1e-5, which
// test_mhc_sinkhorn_emu asserts directly.
void mhc_fused_sinkhorn_gemv_traced(
    const void* x_bf16,
    const float* phi_fp32,
    const float* alpha_fp32,
    const float* bias_fp32,
    const void* h_out_bf16,
    void* x_next_bf16,
    MhcSinkhornTrace* trace,
    float eps = kHcEpsilon,
    int num_iters = kHcSinkhornIters);

// THE PRE AND POST HALVES, SPLIT
// ------------------------------
// The fused entry point takes `h_out` as an INPUT, which is fine for an
// operator-level contract but cannot express a real layer: the decoder does
//
//   h_in, h_post, B = hc_pre(x)        <- needs x only
//   h_in            = RMSNorm(h_in)
//   h_out           = sub_block(h_in)  <- needs h_in
//   x_next          = hc_post(...)     <- needs h_out
//
// so h_out does not exist until after the sub-block has run on h_in. (See
// DeepseekV2DecoderLayer.forward in vllm-ascend
// vllm_ascend/models/deepseek_v4.py, which calls hc_pre, then
// input_layernorm, then self_attn, then hc_post with the post/comb that
// hc_pre returned.) Calling the fused operator with a dummy h_out just to
// recover the trace would redo the 24x16384 projection and the Sinkhorn
// solve for nothing, so the two halves are exposed separately.
//
// NOTE ON THE NORM'S POSITION: the RMSNorm between them applies to `h_in`,
// the [1, D] mHC output -- NOT to the [1, n, D] stream bundle on the way in.
// The bundle's own RMS reciprocal is computed inside hc_pre and applied to
// the projection; `attn_norm.weight` / `ffn_norm.weight` are a separate,
// later norm over 4096 lanes. Normalizing the bundle first instead would
// produce a running pipeline with the wrong activation scale at every layer.

// Everything up to and including the Sinkhorn solve and h_in. Writes the
// trace; does not touch any output stream.
void mhc_pre_sinkhorn_cpu(
    const void* x_bf16,        // [1, n, D] BF16
    const float* phi_fp32,     // [24, 16384] FP32
    const float* alpha_fp32,   // [3] FP32
    const float* bias_fp32,    // [24] FP32
    MhcSinkhornTrace* trace,
    float eps = kHcEpsilon,
    int num_iters = kHcSinkhornIters,
    float norm_eps = kRmsNormEpsilon);

// The fold-back: x_next[j, :] = sum_i B[i, j] x[i, :] + h_post[j] * h_out[:].
// `birkhoff` and `h_post` come from a preceding mhc_pre_sinkhorn_cpu call.
// `x_next` may alias `x`.
void mhc_post_cpu(
    const void* x_bf16,        // [1, n, D] BF16, the residual going in
    const float* birkhoff,     // [n, n] FP32, row-major
    const float* h_post,       // [n] FP32
    const void* h_out_bf16,    // [1, D] BF16, the sub-block FUNCTION output
    void* x_next_bf16);        // [1, n, D] BF16

// The declared entry point. Identical math; discards the trace.
void mhc_fused_sinkhorn_gemv_cpu(
    const void* x_bf16,        // Input residual stream: [1, 4, 4096] BF16 (TND)
    const float* phi_fp32,     // Mixing projection weight: [24, 16384] FP32
    const float* alpha_fp32,   // Gain vector: [3] FP32
    const float* bias_fp32,    // Bias vector: [24] FP32
    const void* h_out_bf16,    // Attention/FFN output: [1, 4096] BF16
    void* x_next_bf16,         // Output residual stream: [1, 4, 4096] BF16
    float eps = kHcEpsilon,
    int num_iters = kHcSinkhornIters);

// Sinkhorn-Knopp on its own, over a stack 4x4. Exposed because the Birkhoff
// invariant tests drive it directly with adversarial matrices (near-zero
// entries, huge dynamic range) that no phi projection would produce.
//
// `logits` and `out` are row-major [n, n] and may alias. The cadence is the
// golden's exactly: initial row softmax with `+ eps`, a column normalization,
// then (num_iters - 1) row/column pairs.
void sinkhorn_knopp_4x4(const float* logits, float* out, float eps, int num_iters);

// ===========================================================================
// 2. Paged KV compressor, decode cadence
// ===========================================================================

// The persistent cadence state one sequence needs.
//
// WHY A STRUCT AND NOT A BARE [4, 512] ARRAY
//   The C4A compressor is coff=2: the eight rows reduced into one compressed
//   token are the PREVIOUS group's four "a"-half state rows followed by the
//   CURRENT group's four "b"-half rows (README formula 2, and the
//   `coff == 2` branch of cpu_compressor). So the cadence has to carry the
//   previous group's a-halves across the group boundary; four rows of the
//   current group alone are not sufficient state, and a [4, 512] buffer
//   cannot express it. `paged_compressor_decode_cpu` below keeps the declared
//   flat-buffer signature for the coff=1 case; this struct is what the real
//   DSV4 geometry needs.
struct CompressorRingState {
  // The current group's full [ratio, coff*D] projected state rows.
  float kv_rows[kCompressorRatio][kCompressorStateWidth] = {};
  float score_rows[kCompressorRatio][kCompressorStateWidth] = {};
  // The previous group's a-halves ([ratio, D]), i.e. coff_id = 0 of the next
  // reduction. Valid only once `have_previous` is true; before that the
  // golden initializes those rows to kv 0 / score -inf, so they contribute
  // exactly nothing to the column softmax.
  float previous_kv_a[kCompressorRatio][kCompressorHeadDim] = {};
  float previous_score_a[kCompressorRatio][kCompressorHeadDim] = {};
  bool have_previous = false;
  // Tokens accepted so far; the cadence is driven by the caller's `step`, and
  // this is kept only so the state can report/validate itself.
  std::int32_t tokens_seen = 0;
};

// What one decode step did. `emitted` is false on a HOLD step, in which case
// the destination slot was not touched at all.
struct CompressorStepResult {
  bool emitted = false;
  // Index of the compressed token this group produced, i.e. step / ratio.
  std::int32_t compressed_index = -1;
  // The reciprocal RMS used by the epilog's RMSNorm, for cross-checking
  // against the kernel's `norm_rstd` output.
  float inv_rms = 0.0F;
};

// The declared entry point, for the coff=1 cadence.
//
//   cur_kv            [1, 512] BF16 -- one token's projected state row
//   step              generation step, 0-indexed
//   ring_buffer       [ratio, 512] FP32, persistent across calls
//   rope_cos/rope_sin [kRopeHeadDim] FP32 tables for THIS compressed token
//   paged_cache_slot  destination row in the paged KV cache (PA_BSND), BF16
//                     [kCompressorHeadDim]; written only on an EMIT step
//   ratio             HOLD/EMIT period
//
// HOLD  (step % ratio != ratio - 1): copies `cur_kv` into
//       ring_buffer[step % ratio] and returns without touching the slot.
// EMIT  (step % ratio == ratio - 1): reduces the `ratio` buffered rows into
//       one token and writes it to `paged_cache_slot`.
//
// With coff=1 the state row carries no separate gate projection, so the
// 512-wide row is read as [kv half | score half] of 256 each -- exactly the
// `half_dim = state_cache.shape[-1] / 2` split that
// compressor_golden.py's `run_cpu_compressor` performs. RoPE is applied to
// the last kRopeHeadDim lanes of the 256-wide result. For the real C4A
// geometry (coff=2, a full 512-wide compressed token) use
// `paged_compressor_decode_c4a` below; `apps/dsv4_layer0_emu_bench` does.
bool paged_compressor_decode_cpu(
    const void* cur_kv,          // Current token KV vector: [1, 512]
    std::int32_t step,           // Generation token step (0-indexed)
    float* ring_buffer,          // Persistent buffer: [4, 512]
    const float* rope_cos,       // RoPE tables: FP32
    const float* rope_sin,
    void* paged_cache_slot,      // Destination slot in paged KV cache (PA_BSND)
    int ratio = kCompressorRatio);

// The real C4A operator: coff=2, cmp_ratio=4, D=512.
//
//   x              [1, H] BF16          the token's hidden state
//   wkv, wgate     [coff*D, H] BF16     compressor projections
//   ape            [ratio, coff*D] FP32 positional biases, added to the gate
//                                       projection at the token's slot in the
//                                       group
//   norm_weight    [D] BF16             RMSNorm gain for the epilog
//   rope_cos/sin   [kRopeHeadDim] FP32  tables for the emitted token
//   state          persistent, see CompressorRingState
//   paged_cache_slot [D] BF16           written only on an EMIT step
//
// Returns what happened. The pipeline:
//   1. kv_state  = x @ wkv^T            [coff*D]
//      score_state = x @ wgate^T + ape[step % ratio]
//   2. on EMIT, assemble [8, D]: previous group's a-halves then this group's
//      b-halves
//   3. column softmax over the 8 rows, per feature lane
//   4. Hadamard with the kv rows, reduce the 8 rows
//   5. RMSNorm(D) with norm_weight and norm_eps
//   6. half-mode RoPE on the last kRopeHeadDim lanes
CompressorStepResult paged_compressor_decode_c4a(
    const void* x_bf16,
    const void* wkv_bf16,
    const void* wgate_bf16,
    const float* ape_fp32,
    const void* norm_weight_bf16,
    std::int32_t step,
    CompressorRingState* state,
    const float* rope_cos,
    const float* rope_sin,
    void* paged_cache_slot,
    int hidden_size = kHiddenSize,
    int head_dim = kCompressorHeadDim,
    int ratio = kCompressorRatio,
    float norm_eps = kRmsNormEpsilon,
    RotaryMode rotary_mode = RotaryMode::kHalf);

// The pieces of the above, exposed so the tests can check each stage against
// the golden independently of the cadence.

// `softmax_columns` from compressor_golden.py: softmax down `rows` (axis 0)
// for each of `cols` lanes independently, max-subtracted, with FP32 denormal
// results flushed to zero to match the NPU `exp`. A lane whose every row is
// -inf would be 0/0; those lanes are written as zero rather than NaN.
void compressor_column_softmax(const float* logits, float* out, int rows, int cols);

// Formula 6: out = norm_weight * x / sqrt(mean(x^2) + eps). Returns the
// reciprocal RMS it used.
float compressor_rms_norm(const float* x, const void* norm_weight_bf16, float* out,
                          int head_dim, float eps);

// vf_rope.h, applied in place to `token[base .. base + rope_head_dim)`:
//   half mode        lo = x[base+i], hi = x[base+half+i]
//                    x[base+i]      = cos[i]*lo      - sin[i]*hi
//                    x[base+half+i] = sin[half+i]*lo + cos[half+i]*hi
//   interleave mode  pairs (2i, 2i+1) rotated by (cos[i], sin[i])
// The tables are indexed exactly as the kernel indexes them, so a table that
// does NOT duplicate its two halves still matches the hardware.
void compressor_apply_rope(float* token, const float* rope_cos, const float* rope_sin,
                           int head_dim, int rope_head_dim, RotaryMode mode);

// ===========================================================================
// 3. Fast FP8 lightning indexer top-k
// ===========================================================================

// The declared entry point.
//
//   query_fp8        [1, 1, 64, 128] FP8 E4M3 -- one decode query, 64 heads
//   key_cache_fp8    PA_BSND [num_blocks, block_size, 1, 128] FP8 E4M3
//   dequant_scales   the fused scale vector, see below
//   block_tables     logical block -> physical block, INT32. It MUST hold at
//                    least ceil(seq_len / block_size) entries: key j is read
//                    through block_tables[j / block_size], so a shorter array
//                    is an out-of-bounds read this signature cannot detect
//                    (it carries no length). Pass nullptr for the unpaged
//                    case, which maps logical to physical identically.
//   seq_len          number of keys to score (0 .. seq_len-1)
//   top_k            how many indices to select (e.g. 512)
//   out_indices      [1, 1, 1, top_k] INT32, written
//
// SCORING (cal_atten_per_batch_fp8, verbatim)
//   logit[j] = k_scale[j] * sum_h  qw[h] * relu( q[h] . k[j] )
//   The ReLU is per head and comes BEFORE the head weighting -- a plain
//   Q.K^T followed by a weighted sum is a different operator.
//
// DEQUANT SCALES LAYOUT
//   `dequant_scales` is one FP32 array of 64 + seq_len entries:
//     [0, 64)              qw[h] -- the per-head query dequant scale ALREADY
//                          multiplied by the indexer's per-head weight
//                          (`cur_w = cur_wt * cur_q_scale` in the golden)
//     [64, 64 + seq_len)   k_scale[j] -- the per-key dequant scale
//   Folding `weights` into the query scales is exactly what the golden does
//   before the matmul, so this single array carries the full contract without
//   a separate weights argument.
//
// SELECTION
//   Scores are rounded to BF16 before selection, as the golden does, then the
//   top `top_k` are taken under a total order: higher score first, lower
//   index first on a tie. Ties are the common case after BF16 rounding, so
//   that second key is load-bearing, not a detail. Selection is
//   `std::nth_element` + a partial sort of the winners -- O(N) expected
//   rather than the golden's full O(N log N) sort, with the identical result.
//
// When seq_len < top_k the remaining slots are filled with -1, matching the
// golden's `torch.full(..., -1)` initialization.
void fast_fp8_indexer_topk_cpu(
    const std::uint8_t* query_fp8,      // [1, 1, 64, 128] FP8_E4M3
    const std::uint8_t* key_cache_fp8,  // PA_BSND [num_blocks, block_size, 1, 128]
    const float* dequant_scales,
    const std::int32_t* block_tables,
    std::int32_t seq_len,
    std::int32_t top_k,                 // e.g. 512
    std::int32_t* out_indices,          // [1, 1, 1, 512] INT32
    std::int32_t block_size = 128,
    int heads = kIndexerHeads,
    int head_dim = kIndexerHeadDim);

// The scoring half on its own, before selection: `out_scores[j]` for
// j in [0, seq_len). FP32, NOT yet rounded to BF16. The naive-baseline test
// compares against this.
void fast_fp8_indexer_scores_cpu(
    const std::uint8_t* query_fp8,
    const std::uint8_t* key_cache_fp8,
    const float* dequant_scales,
    const std::int32_t* block_tables,
    std::int32_t seq_len,
    float* out_scores,
    std::int32_t block_size = 128,
    int heads = kIndexerHeads,
    int head_dim = kIndexerHeadDim);

}  // namespace moe::emu

#endif  // MOE_EMU_CPU_KERNELS_H
