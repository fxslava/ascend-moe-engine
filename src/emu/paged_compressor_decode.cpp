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

// Paged KV compressor, decode cadence, on the host CPU.
//
// Ported from ops-transformer attention/compressor/tests/pytest/
// compressor_golden.py (`cpu_compressor`, `softmax_columns`) for the
// compression stage, plus the RMSNorm/RoPE epilog that the vendored arch35
// operator adds (third_party/ops_dsv4/attention/compressor: README formulas
// 6-7 and op_kernel/arch35/vf/vf_rope.h).
//
// THE COMPRESSION IS A GATED REDUCTION, NOT A LEARNED CONVOLUTION
//   "4:1 compression" is easy to read as a strided conv or a GEMV over the
//   window. It is neither. Each token is projected twice -- once through wkv
//   and once through wgate -- and the gate projections of the whole window
//   are turned into weights by a softmax taken DOWN the window (axis 0,
//   per feature lane, not across features). The compressed token is the
//   gate-weighted sum of the window's kv projections:
//
//     kv_state[t]    = x[t] @ wkv^T
//     score_state[t] = x[t] @ wgate^T + ape[t mod ratio]
//     S              = softmax_columns(score_state over the window)
//     C              = sum_over_window( S * kv_state )
//
//   So every one of the D lanes picks its own mixture of the window's tokens.
//   A row-wise softmax, or a mean, would both produce something finite and
//   wrong.
//
// COFF=2: THE WINDOW IS EIGHT ROWS, NOT FOUR
//   The deployed C4A configuration is coff=2 (D=512, cmp_ratio=4). The
//   projections are [2*D, H], so each token yields a 1024-wide state row that
//   splits into an "a" half and a "b" half. The eight rows reduced for group
//   i are:
//
//     rows 0..3  the a-halves of group i-1's four tokens
//     rows 4..7  the b-halves of group i's four tokens
//
//   i.e. the windows OVERLAP by one group, which is what the README calls
//   "overlap 数据重排" and what the golden's `coff == 2` branch does by
//   reading `pre_start_offset = start_offset - cmp_ratio`. For the very first
//   group there is no predecessor, and the golden's initialization -- kv rows
//   zero, score rows -inf -- makes those four rows contribute exactly zero
//   weight through the softmax. That is why the state struct carries
//   `have_previous` instead of zero-filling.

#include "moe/emu/cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(std::string("paged_compressor_decode: ") + message);
  }
}

// y = W @ x for a BF16 [rows, cols] row-major W and a BF16 [cols] x,
// accumulating in FP32. Unit stride on both operands.
void ProjectBf16(const std::uint16_t* weight, const float* x, float* out, int rows,
                 int cols) {
  for (int row = 0; row < rows; ++row) {
    const std::uint16_t* weight_row = weight + static_cast<std::size_t>(row) * cols;
    float dot = 0.0F;
    for (int col = 0; col < cols; ++col) {
      dot += Bf16ToFloat(weight_row[col]) * x[col];
    }
    out[row] = dot;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// softmax down the window
// ---------------------------------------------------------------------------

void compressor_column_softmax(const float* logits, float* out, int rows, int cols) {
  Require(rows > 0 && cols > 0, "column softmax needs positive extents");
  // FP32's smallest normal. The golden zeroes any exp() result below it:
  //   exp_z[exp_z < np.finfo(np.float32).tiny] = 0.0
  // with the comment that this matches the NPU's flush-to-zero exp. Keeping
  // the subnormal instead changes the denominator in the last few bits, so
  // this clamp is part of the numerical contract, not a tidy-up.
  const float tiny = std::numeric_limits<float>::min();

  for (int col = 0; col < cols; ++col) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int row = 0; row < rows; ++row) {
      maximum = std::max(maximum, logits[row * cols + col]);
    }

    // An all--inf lane (every row masked) would be 0/0. The golden never
    // produces one -- at least the current group's rows are always finite --
    // but a zero lane is the only defensible answer, so write it rather than
    // propagating a NaN into the paged cache.
    if (!std::isfinite(maximum)) {
      for (int row = 0; row < rows; ++row) {
        out[row * cols + col] = 0.0F;
      }
      continue;
    }

    float sum = 0.0F;
    for (int row = 0; row < rows; ++row) {
      float value = std::exp(logits[row * cols + col] - maximum);
      if (value < tiny) {
        value = 0.0F;
      }
      out[row * cols + col] = value;
      sum += value;
    }
    // sum >= 1 because the max-subtracted row contributes exp(0) == 1.
    const float scale = 1.0F / sum;
    for (int row = 0; row < rows; ++row) {
      out[row * cols + col] *= scale;
    }
  }
}

// ---------------------------------------------------------------------------
// the epilog: RMSNorm then RoPE
// ---------------------------------------------------------------------------

float compressor_rms_norm(const float* x, const void* norm_weight_bf16, float* out,
                          int head_dim, float eps) {
  Require(head_dim > 0, "rms norm needs a positive head_dim");
  const auto* norm_weight = static_cast<const std::uint16_t*>(norm_weight_bf16);
  double sum_squares = 0.0;
  for (int i = 0; i < head_dim; ++i) {
    sum_squares += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float mean_square =
      static_cast<float>(sum_squares / static_cast<double>(head_dim));
  const float inv_rms = 1.0F / std::sqrt(mean_square + eps);
  for (int i = 0; i < head_dim; ++i) {
    const float gain = (norm_weight == nullptr) ? 1.0F : Bf16ToFloat(norm_weight[i]);
    out[i] = gain * x[i] * inv_rms;
  }
  return inv_rms;
}

void compressor_apply_rope(float* token, const float* rope_cos, const float* rope_sin,
                           int head_dim, int rope_head_dim, RotaryMode mode) {
  Require(rope_head_dim > 0 && rope_head_dim % 2 == 0,
          "rope_head_dim must be positive and even");
  Require(rope_head_dim <= head_dim, "rope_head_dim must not exceed head_dim");
  if (rope_cos == nullptr || rope_sin == nullptr) {
    return;  // no tables: the token passes through, as rotary_mode absent
  }

  // vf_rope.h is invoked with baseAddr = headDim - ropeHeadDim, so the
  // rotation covers the LAST rope_head_dim lanes and lanes [0, base) are only
  // cast through. Rotating the first 64 lanes instead is the single most
  // likely place to diverge from the device, so the offset is derived here
  // rather than passed in.
  const int base = head_dim - rope_head_dim;
  const int half = rope_head_dim / 2;

  if (mode == RotaryMode::kHalf) {
    // HalfModeRopeVF, index for index:
    //   out[base+i]      = cos[i]      * lo - sin[i]      * hi
    //   out[base+half+i] = sin[half+i] * lo + cos[half+i] * hi
    // The second line reads the tables at half+i, not i. For the usual table
    // whose halves are duplicated these are equal; honouring the kernel's
    // indexing keeps a non-duplicated table correct too.
    for (int i = 0; i < half; ++i) {
      const float lo = token[base + i];
      const float hi = token[base + half + i];
      token[base + i] = rope_cos[i] * lo - rope_sin[i] * hi;
      token[base + half + i] = rope_sin[half + i] * lo + rope_cos[half + i] * hi;
    }
  } else {
    // InterleaveModeRopeVF: adjacent lanes form each rotation pair.
    for (int i = 0; i < half; ++i) {
      const float even = token[base + 2 * i];
      const float odd = token[base + 2 * i + 1];
      token[base + 2 * i] = rope_cos[i] * even - rope_sin[i] * odd;
      token[base + 2 * i + 1] = rope_sin[i] * even + rope_cos[i] * odd;
    }
  }
}

// ---------------------------------------------------------------------------
// the declared entry point (coff = 1)
// ---------------------------------------------------------------------------

bool paged_compressor_decode_cpu(
    const void* cur_kv,
    std::int32_t step,
    float* ring_buffer,
    const float* rope_cos,
    const float* rope_sin,
    void* paged_cache_slot,
    int ratio) {
  Require(cur_kv != nullptr, "cur_kv is null");
  Require(ring_buffer != nullptr, "ring_buffer is null");
  // The declared contract sizes `ring_buffer` at [kCompressorRatio, 512], and
  // this function indexes it at `step % ratio`. A ratio above that bound would
  // therefore write past the end of a correctly sized caller buffer, so it is
  // refused rather than trusted -- the signature carries no length to check
  // against, which makes the bound the only thing standing between a wrong
  // `ratio` and heap corruption.
  Require(ratio > 0 && ratio <= kCompressorRatio,
          "ratio must be in (0, 4]: the ring buffer is [4, 512]");
  Require(step >= 0, "step must be non-negative");

  constexpr int width = kCompressorStateWidth / kCompressorCoff;  // 512
  // coff=1 splits the 512-wide row into a kv half and a score half, exactly
  // as run_cpu_compressor derives `half_dim = state_cache.shape[-1] / 2`.
  constexpr int head_dim = width / 2;  // 256
  static_assert(head_dim * 2 == width, "the state row must split evenly");

  const auto* row = static_cast<const std::uint16_t*>(cur_kv);
  const int slot = step % ratio;
  float* destination = ring_buffer + static_cast<std::size_t>(slot) * width;

  // --- HOLD ---------------------------------------------------------------
  for (int i = 0; i < width; ++i) {
    destination[i] = Bf16ToFloat(row[i]);
  }
  if (slot != ratio - 1) {
    return false;  // the slot is not touched on a HOLD step
  }

  // --- EMIT ---------------------------------------------------------------
  Require(paged_cache_slot != nullptr, "paged_cache_slot is null on an EMIT step");

  std::vector<float> kv(static_cast<std::size_t>(ratio) * head_dim);
  std::vector<float> score(static_cast<std::size_t>(ratio) * head_dim);
  for (int r = 0; r < ratio; ++r) {
    const float* source = ring_buffer + static_cast<std::size_t>(r) * width;
    for (int i = 0; i < head_dim; ++i) {
      kv[static_cast<std::size_t>(r) * head_dim + i] = source[i];
      score[static_cast<std::size_t>(r) * head_dim + i] = source[head_dim + i];
    }
  }

  std::vector<float> weights(score.size());
  compressor_column_softmax(score.data(), weights.data(), ratio, head_dim);

  std::vector<float> compressed(head_dim, 0.0F);
  for (int r = 0; r < ratio; ++r) {
    for (int i = 0; i < head_dim; ++i) {
      compressed[i] += weights[static_cast<std::size_t>(r) * head_dim + i] *
                       kv[static_cast<std::size_t>(r) * head_dim + i];
    }
  }

  // No norm_weight in this signature, so the RMSNorm gain is 1.
  std::vector<float> normalized(head_dim);
  compressor_rms_norm(compressed.data(), nullptr, normalized.data(), head_dim,
                      kRmsNormEpsilon);
  compressor_apply_rope(normalized.data(), rope_cos, rope_sin, head_dim, kRopeHeadDim,
                        RotaryMode::kHalf);

  auto* out = static_cast<std::uint16_t*>(paged_cache_slot);
  for (int i = 0; i < head_dim; ++i) {
    out[i] = FloatToBf16(normalized[i]);
  }
  return true;
}

// ---------------------------------------------------------------------------
// the real C4A operator (coff = 2)
// ---------------------------------------------------------------------------

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
    int hidden_size,
    int head_dim,
    int ratio,
    float norm_eps,
    RotaryMode rotary_mode) {
  Require(x_bf16 != nullptr, "x is null");
  Require(wkv_bf16 != nullptr && wgate_bf16 != nullptr, "wkv/wgate is null");
  Require(state != nullptr, "state is null");
  Require(step >= 0, "step must be non-negative");
  Require(hidden_size > 0, "hidden_size must be positive");
  Require(head_dim > 0 && head_dim <= kCompressorHeadDim,
          "head_dim must be in (0, 512]");
  Require(ratio > 0 && ratio <= kCompressorRatio, "ratio must be in (0, 4]");

  const int state_width = kCompressorCoff * head_dim;  // 1024 for C4A
  const auto* x = static_cast<const std::uint16_t*>(x_bf16);

  std::vector<float> x_f32(static_cast<std::size_t>(hidden_size));
  for (int i = 0; i < hidden_size; ++i) {
    x_f32[static_cast<std::size_t>(i)] = Bf16ToFloat(x[i]);
  }

  // --- stage 1: the two projections ---------------------------------------
  const int slot = step % ratio;
  float* kv_row = state->kv_rows[slot];
  float* score_row = state->score_rows[slot];
  ProjectBf16(static_cast<const std::uint16_t*>(wkv_bf16), x_f32.data(), kv_row,
              state_width, hidden_size);
  ProjectBf16(static_cast<const std::uint16_t*>(wgate_bf16), x_f32.data(), score_row,
              state_width, hidden_size);

  // ape is [ratio, coff*D] and is added at the token's position WITHIN the
  // group, so it is a function of step % ratio and not of the absolute step.
  if (ape_fp32 != nullptr) {
    const float* ape_row = ape_fp32 + static_cast<std::size_t>(slot) * state_width;
    for (int i = 0; i < state_width; ++i) {
      score_row[i] += ape_row[i];
    }
  }

  state->tokens_seen = step + 1;

  CompressorStepResult result;
  if (slot != ratio - 1) {
    return result;  // HOLD: emitted stays false, the slot is untouched
  }

  // --- stage 2: assemble the eight-row window ------------------------------
  Require(paged_cache_slot != nullptr, "paged_cache_slot is null on an EMIT step");
  const int rows = kCompressorCoff * ratio;  // 8
  std::vector<float> window_kv(static_cast<std::size_t>(rows) * head_dim);
  std::vector<float> window_score(static_cast<std::size_t>(rows) * head_dim);

  for (int r = 0; r < ratio; ++r) {
    float* kv_dst = window_kv.data() + static_cast<std::size_t>(r) * head_dim;
    float* score_dst = window_score.data() + static_cast<std::size_t>(r) * head_dim;
    if (state->have_previous) {
      // coff_id = 0: the previous group's a-halves.
      std::copy_n(state->previous_kv_a[r], head_dim, kv_dst);
      std::copy_n(state->previous_score_a[r], head_dim, score_dst);
    } else {
      // The golden's first-group initialization: kv 0, score -inf. exp(-inf)
      // is 0, so these rows get exactly zero softmax weight and the first
      // compressed token is a pure four-row reduction.
      std::fill_n(kv_dst, head_dim, 0.0F);
      std::fill_n(score_dst, head_dim, -std::numeric_limits<float>::infinity());
    }
    // coff_id = 1: this group's b-halves, i.e. the upper half of the row.
    std::copy_n(state->kv_rows[r] + head_dim, head_dim,
                window_kv.data() + static_cast<std::size_t>(ratio + r) * head_dim);
    std::copy_n(state->score_rows[r] + head_dim, head_dim,
                window_score.data() + static_cast<std::size_t>(ratio + r) * head_dim);
  }

  // --- stages 3-5: column softmax, Hadamard, reduce ------------------------
  std::vector<float> weights(window_score.size());
  compressor_column_softmax(window_score.data(), weights.data(), rows, head_dim);

  std::vector<float> compressed(static_cast<std::size_t>(head_dim), 0.0F);
  for (int r = 0; r < rows; ++r) {
    const std::size_t offset = static_cast<std::size_t>(r) * head_dim;
    for (int i = 0; i < head_dim; ++i) {
      compressed[static_cast<std::size_t>(i)] +=
          weights[offset + static_cast<std::size_t>(i)] *
          window_kv[offset + static_cast<std::size_t>(i)];
    }
  }

  // --- stages 6-7: RMSNorm then RoPE --------------------------------------
  std::vector<float> normalized(static_cast<std::size_t>(head_dim));
  result.inv_rms = compressor_rms_norm(compressed.data(), norm_weight_bf16,
                                       normalized.data(), head_dim, norm_eps);
  compressor_apply_rope(normalized.data(), rope_cos, rope_sin, head_dim, kRopeHeadDim,
                        rotary_mode);

  auto* out = static_cast<std::uint16_t*>(paged_cache_slot);
  for (int i = 0; i < head_dim; ++i) {
    out[i] = FloatToBf16(normalized[static_cast<std::size_t>(i)]);
  }

  // --- roll the window ----------------------------------------------------
  // This group's a-halves become the next group's "previous" rows. Done after
  // the reduction so a caller that inspects the state mid-step sees the rows
  // the reduction actually used.
  for (int r = 0; r < ratio; ++r) {
    std::copy_n(state->kv_rows[r], head_dim, state->previous_kv_a[r]);
    std::copy_n(state->score_rows[r], head_dim, state->previous_score_a[r]);
  }
  state->have_previous = true;

  result.emitted = true;
  result.compressed_index = step / ratio;
  return result;
}

}  // namespace moe::emu
