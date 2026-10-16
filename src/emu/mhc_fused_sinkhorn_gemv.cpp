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

// mHC pre-mix + Sinkhorn-Knopp + fused post update, on the host CPU.
//
// Ported from two upstream goldens, composed into the single operator the
// decode graph actually launches:
//
//   ops-transformer experimental/mhc/mhc_pre_sinkhorn_premix/tests/assets/
//     golden.py :: _compute      -- RMSNorm, phi projection, the three gate
//                                   segments, and the Sinkhorn cadence
//   ops-transformer mhc/mhc_post/tests/assets/golden.py :: _mhc_post_compute
//                                -- y = hRes^T x + hPost (x) hOut
//
// THE STAGE ORDER, AND THE ONE THING THAT IS EASY TO GET WRONG
//   inv_rms is computed over the FLATTENED [n*D] stream and applied to the
//   PROJECTION, not to x:
//
//     inv_rms    = 1 / sqrt(mean(flat_x^2) + norm_eps)       scalar
//     normalized = (phi @ flat_x) * inv_rms                  [24]
//
//   Normalizing x first and then projecting gives the same number only
//   because the projection is linear and inv_rms is a scalar -- but the
//   golden's order is the one the kernel implements and the one that keeps
//   `hc_before_norm` available as a backward output, so it is the order used
//   here.
//
//   The 24 projection rows are then split 4 / 4 / 16:
//     rows [0, 4)    h_pre  = sigmoid(z * alpha[0] + bias) + hc_eps
//     rows [4, 8)    h_post = 2 * sigmoid(z * alpha[1] + bias)
//     rows [8, 24)   M      = z * alpha[2] + bias, reshaped [4, 4]
//
//   The sigmoids and the factor 2 are NOT decoration: h_pre is a gate in
//   (0, 1] + eps and h_post is a gain in (0, 2). An implementation that uses
//   the raw projection rows instead produces an unbounded residual map, and
//   nothing downstream will flag it.

#include "moe/emu/cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace moe::emu {
namespace {

// The golden clamps before exp (`np.clip(value, -80, 80)`) so that a large
// negative logit cannot produce inf/0 from the exponential. 80 is safely
// inside FP32's range and the clamp is saturating, matching numpy.
float Sigmoid(float value) noexcept {
  const float clamped = std::min(80.0F, std::max(-80.0F, value));
  return 1.0F / (1.0F + std::exp(-clamped));
}

void RequirePositive(int value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string("mhc_fused_sinkhorn_gemv: ") + name +
                                " must be positive, got " + std::to_string(value));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Sinkhorn-Knopp
// ---------------------------------------------------------------------------
//
// THE CADENCE IS THE GOLDEN'S, INCLUDING ITS ASYMMETRY
//   iteration 0 : row softmax (max-subtracted), then `+ eps` on every entry,
//                 then one column normalization
//   iterations 1 .. num_iters-1 : a row normalization, then a column one
//
//   So there are num_iters row steps and num_iters column steps, and the last
//   operation performed is always a column normalization. `+ eps` is applied
//   ONCE, right after the initial softmax; from then on eps appears only as
//   `/(sum + eps)`, a division guard. Adding eps after every step instead
//   would drift the row sums by num_iters * n * eps and break the 1e-5
//   Birkhoff bound this is required to meet.
//
// WHY A STACK 4x4
//   16 floats. It stays in registers across all 40 normalizations, so the
//   whole solve is a few hundred cycles with no memory traffic -- which is
//   the entire reason the fused operator is worth having.
void sinkhorn_knopp_4x4(const float* logits, float* out, float eps, int num_iters) {
  RequirePositive(num_iters, "num_iters");
  constexpr int n = kHcStreams;
  float b[n][n];

  // --- initial row softmax -------------------------------------------------
  for (int row = 0; row < n; ++row) {
    float maximum = logits[row * n];
    for (int col = 1; col < n; ++col) {
      maximum = std::max(maximum, logits[row * n + col]);
    }
    float sum = 0.0F;
    for (int col = 0; col < n; ++col) {
      b[row][col] = std::exp(logits[row * n + col] - maximum);
      sum += b[row][col];
    }
    // The max-subtraction guarantees one entry is exp(0) == 1, so sum >= 1
    // and this division needs no guard. Keeping the guard anyway costs
    // nothing and makes the function safe for a NaN-bearing input.
    const float scale = (sum > 0.0F && std::isfinite(sum)) ? (1.0F / sum) : 0.0F;
    for (int col = 0; col < n; ++col) {
      b[row][col] = b[row][col] * scale + eps;  // the one additive eps
    }
  }

  // --- the first column normalization --------------------------------------
  for (int col = 0; col < n; ++col) {
    float sum = 0.0F;
    for (int row = 0; row < n; ++row) {
      sum += b[row][col];
    }
    const float denominator = sum + eps;
    for (int row = 0; row < n; ++row) {
      b[row][col] /= denominator;
    }
  }

  // --- the remaining row/column pairs --------------------------------------
  for (int iteration = 1; iteration < num_iters; ++iteration) {
    for (int row = 0; row < n; ++row) {
      float sum = 0.0F;
      for (int col = 0; col < n; ++col) {
        sum += b[row][col];
      }
      const float denominator = sum + eps;
      for (int col = 0; col < n; ++col) {
        b[row][col] /= denominator;
      }
    }
    for (int col = 0; col < n; ++col) {
      float sum = 0.0F;
      for (int row = 0; row < n; ++row) {
        sum += b[row][col];
      }
      const float denominator = sum + eps;
      for (int row = 0; row < n; ++row) {
        b[row][col] /= denominator;
      }
    }
  }

  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      out[row * n + col] = b[row][col];
    }
  }
}

// ---------------------------------------------------------------------------
// The fused operator
// ---------------------------------------------------------------------------

void mhc_pre_sinkhorn_cpu(
    const void* x_bf16,
    const float* phi_fp32,
    const float* alpha_fp32,
    const float* bias_fp32,
    MhcSinkhornTrace* trace,
    float eps,
    int num_iters,
    float norm_eps) {
  if (x_bf16 == nullptr || phi_fp32 == nullptr || alpha_fp32 == nullptr ||
      bias_fp32 == nullptr || trace == nullptr) {
    throw std::invalid_argument("mhc_pre_sinkhorn: null argument");
  }
  RequirePositive(num_iters, "num_iters");

  constexpr int n = kHcStreams;
  constexpr int d = kHiddenSize;
  constexpr int flat = kHcFlatWidth;

  const auto* x = static_cast<const std::uint16_t*>(x_bf16);

  // --- widen x once --------------------------------------------------------
  // Every stage below reads x at least twice (RMS, projection, the h_in
  // reduction), so widening once is both faster and the only way to keep all
  // the reads bit-identical.
  static thread_local float pre_x_f32[flat];
  for (int i = 0; i < flat; ++i) {
    pre_x_f32[i] = Bf16ToFloat(x[i]);
  }

  // --- inv_rms over the flattened stream -----------------------------------
  double sum_squares = 0.0;  // FP64 for the 16384-term reduction; the golden
                             // uses numpy's pairwise mean, which is at least
                             // this accurate, and FP32 here would drift.
  for (int i = 0; i < flat; ++i) {
    sum_squares += static_cast<double>(pre_x_f32[i]) * static_cast<double>(pre_x_f32[i]);
  }
  const float mean_square = static_cast<float>(sum_squares / static_cast<double>(flat));
  trace->inv_rms = 1.0F / std::sqrt(mean_square + norm_eps);

  // --- phi @ flat_x, then the RMS scale ------------------------------------
  // phi is [24, 16384] row-major, so each row is a contiguous 16384-element
  // dot product: unit stride on both operands, which is what makes this
  // auto-vectorize into an AVX2 FMA chain.
  float normalized[kHcProjectionRows];
  for (int row = 0; row < kHcProjectionRows; ++row) {
    const float* phi_row = phi_fp32 + static_cast<std::size_t>(row) * flat;
    float dot = 0.0F;
    for (int col = 0; col < flat; ++col) {
      dot += phi_row[col] * pre_x_f32[col];
    }
    trace->projection[row] = dot;
    normalized[row] = dot * trace->inv_rms;
  }

  // --- the three gate segments ---------------------------------------------
  for (int i = 0; i < n; ++i) {
    trace->h_pre[i] = Sigmoid(normalized[i] * alpha_fp32[0] + bias_fp32[i]) + eps;
    trace->h_post[i] =
        2.0F * Sigmoid(normalized[n + i] * alpha_fp32[1] + bias_fp32[n + i]);
  }
  for (int i = 0; i < n * n; ++i) {
    trace->residual_logits[i] =
        normalized[2 * n + i] * alpha_fp32[2] + bias_fp32[2 * n + i];
  }

  // --- Sinkhorn ------------------------------------------------------------
  sinkhorn_knopp_4x4(trace->residual_logits, trace->birkhoff, eps, num_iters);

  // --- h_in: the sub-block input ------------------------------------------
  // golden: hin = sum(x * h_pre[..., :, None], axis=-2), i.e. the h_pre-gated
  // sum of the four streams. This is what RMSNorm, the attention projections
  // and the MoE router consume.
  for (int feature = 0; feature < d; ++feature) {
    float accumulator = 0.0F;
    for (int stream = 0; stream < n; ++stream) {
      accumulator += pre_x_f32[stream * d + feature] * trace->h_pre[stream];
    }
    trace->h_in[feature] = accumulator;
  }
}

void mhc_post_cpu(
    const void* x_bf16,
    const float* birkhoff,
    const float* h_post,
    const void* h_out_bf16,
    void* x_next_bf16) {
  if (x_bf16 == nullptr || birkhoff == nullptr || h_post == nullptr ||
      h_out_bf16 == nullptr || x_next_bf16 == nullptr) {
    throw std::invalid_argument("mhc_post: null argument");
  }

  constexpr int n = kHcStreams;
  constexpr int d = kHiddenSize;
  constexpr int flat = kHcFlatWidth;

  const auto* x = static_cast<const std::uint16_t*>(x_bf16);
  const auto* h_out = static_cast<const std::uint16_t*>(h_out_bf16);
  auto* x_next = static_cast<std::uint16_t*>(x_next_bf16);

  // Widened into its own buffer, NOT the one mhc_pre_sinkhorn_cpu uses: the
  // declared contract allows x_next to alias x, so the source has to be read
  // out before the first store lands.
  static thread_local float post_x_f32[flat];
  for (int i = 0; i < flat; ++i) {
    post_x_f32[i] = Bf16ToFloat(x[i]);
  }

  // y[j, :] = sum_i B[i, j] * x[i, :] + h_post[j] * h_out[:]
  //
  // Note B is indexed [i][j] while j is the output stream: this is B^T x, and
  // transposing it by accident is a silent, plausible-looking error -- the
  // result stays finite and the norms stay sane, because B^T is doubly
  // stochastic too. test_mhc_sinkhorn_emu pins the orientation with an
  // asymmetric B.
  for (int out_stream = 0; out_stream < n; ++out_stream) {
    const float gain = h_post[out_stream];
    float mix[n];
    for (int in_stream = 0; in_stream < n; ++in_stream) {
      mix[in_stream] = birkhoff[in_stream * n + out_stream];
    }
    std::uint16_t* out_row = x_next + static_cast<std::size_t>(out_stream) * d;
    for (int feature = 0; feature < d; ++feature) {
      float accumulator = gain * Bf16ToFloat(h_out[feature]);
      for (int in_stream = 0; in_stream < n; ++in_stream) {
        accumulator += mix[in_stream] * post_x_f32[in_stream * d + feature];
      }
      out_row[feature] = FloatToBf16(accumulator);
    }
  }
}

void mhc_fused_sinkhorn_gemv_traced(
    const void* x_bf16,
    const float* phi_fp32,
    const float* alpha_fp32,
    const float* bias_fp32,
    const void* h_out_bf16,
    void* x_next_bf16,
    MhcSinkhornTrace* trace,
    float eps,
    int num_iters) {
  if (h_out_bf16 == nullptr || x_next_bf16 == nullptr) {
    throw std::invalid_argument("mhc_fused_sinkhorn_gemv: null argument");
  }
  mhc_pre_sinkhorn_cpu(x_bf16, phi_fp32, alpha_fp32, bias_fp32, trace, eps, num_iters,
                       kRmsNormEpsilon);
  mhc_post_cpu(x_bf16, trace->birkhoff, trace->h_post, h_out_bf16, x_next_bf16);
}

void mhc_fused_sinkhorn_gemv_cpu(
    const void* x_bf16,
    const float* phi_fp32,
    const float* alpha_fp32,
    const float* bias_fp32,
    const void* h_out_bf16,
    void* x_next_bf16,
    float eps,
    int num_iters) {
  // The trace is ~16 KB, almost all of it h_in. thread_local rather than a
  // stack object so the declared entry point stays usable from a thread with
  // a small stack.
  static thread_local MhcSinkhornTrace trace;
  mhc_fused_sinkhorn_gemv_traced(x_bf16, phi_fp32, alpha_fp32, bias_fp32, h_out_bf16,
                                 x_next_bf16, &trace, eps, num_iters);
}

}  // namespace moe::emu
