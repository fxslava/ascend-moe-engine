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

// Quantized GEMV and RMSNorm for the host emulation.
//
// TWO QUANTIZATION SCHEMES, AND WHY THE DIFFERENCE MATTERS
//   The checkpoint uses both, with different block geometry:
//
//     dense + shared expert   F8_E4M3 values, one E8M0 scale per 128 x 128
//                             block. wq_a is [1024, 4096] with scale [8, 32].
//     routed experts          E2M1 nibbles, one E8M0 scale per ROW per
//                             32-element GROUP along the input axis. w1 is
//                             logical [2048, 4096] with scale [2048, 128].
//
//   Reading the second with the first's indexing (or vice versa) produces
//   finite, plausibly-scaled garbage, because both are "a byte array plus
//   E8M0 scales". The only thing that catches it is deriving the grid from
//   the declared shapes, which both GEMVs below do, and asserting it.
//
// MEMORY DISCIPLINE
//   Dequantization is per output row, into a caller-owned scratch row. A
//   decode step is a GEMV, so every weight byte is read exactly once and
//   materializing an FP32 copy of the matrix would cost 4x (FP8) or 8x (FP4)
//   the resident set for no reuse. This is what holds the whole harness under
//   its memory budget.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("dsv4_layers: " + message);
  }
}

// OCP E2M1: 1 sign, 2 exponent, 1 mantissa, bias 1. No inf, no NaN, so all 16
// codes are finite and the table is exact.
//   exp 0 : subnormal, value = mantissa * 0.5
//   else  : (1 + mantissa/2) * 2^(exp - 1)
constexpr float kFp4Magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

// One PACKED BYTE -> both of its decoded elements, in logical column order.
//
// The inner FP4 loop reads a byte and needs two floats from it. Decoding each
// nibble separately costs a mask, a shift, a table index and a sign branch per
// element; this table turns the whole byte into one 8-byte load of two
// adjacent floats, which also lets the compiler keep the loop vectorized. The
// nibble-order choice (see Mxfp4Matrix) is baked in here once, at table build
// time, instead of being branched on 67 million times per expert.
struct Fp4Pair {
  float even;  // logical column 2i
  float odd;   // logical column 2i+1
};

using Fp4PairTable = std::array<Fp4Pair, 256>;

Fp4PairTable BuildFp4PairTable() noexcept {
  Fp4PairTable table{};
  for (std::size_t byte = 0; byte < table.size(); ++byte) {
    const std::uint8_t low = static_cast<std::uint8_t>(byte & 0x0FU);
    const std::uint8_t high = static_cast<std::uint8_t>(byte >> 4U);
    const std::uint8_t even = kMxfp4LowNibbleFirst ? low : high;
    const std::uint8_t odd = kMxfp4LowNibbleFirst ? high : low;
    table[byte].even = DecodeFp4E2M1(even);
    table[byte].odd = DecodeFp4E2M1(odd);
  }
  return table;
}

const Fp4PairTable& Fp4Pairs() noexcept {
  static const Fp4PairTable table = BuildFp4PairTable();
  return table;
}

}  // namespace

float DecodeFp4E2M1(std::uint8_t code) noexcept {
  const float magnitude = kFp4Magnitudes[code & 0x07U];
  return (code & 0x08U) != 0 ? -magnitude : magnitude;
}

// ---------------------------------------------------------------------------
// FP8 block-scaled
// ---------------------------------------------------------------------------

void fp8_block_dequant_row_cpu(const Fp8BlockMatrix& weight, int row, float* out) {
  Require(weight.weight != nullptr && weight.scale != nullptr, "fp8 matrix is null");
  Require(row >= 0 && row < weight.rows, "fp8 row out of range");
  const Fp8Table& values = Fp8E4M3Table();
  const Fp8Table& scales = Fp8E8M0Table();

  const std::uint8_t* weight_row =
      weight.weight + static_cast<std::size_t>(row) * weight.cols;
  const std::uint8_t* scale_row =
      weight.scale +
      static_cast<std::size_t>(row / kFp8BlockSize) * weight.scale_cols;

  // One scale per 128 input columns, so hoist it out of the inner loop and
  // walk 128 columns at a time.
  for (int block_start = 0; block_start < weight.cols; block_start += kFp8BlockSize) {
    const int block_end = std::min(block_start + kFp8BlockSize, weight.cols);
    const float scale = scales[scale_row[block_start / kFp8BlockSize]];
    for (int col = block_start; col < block_end; ++col) {
      out[col] = values[weight_row[col]] * scale;
    }
  }
}

void fp8_block_gemv_cpu(const Fp8BlockMatrix& weight, const float* x, float* y) {
  Require(weight.weight != nullptr && weight.scale != nullptr, "fp8 matrix is null");
  Require(x != nullptr && y != nullptr, "fp8 gemv operand is null");
  Require(weight.rows > 0 && weight.cols > 0, "fp8 matrix has an empty extent");
  Require(weight.scale_cols >= (weight.cols + kFp8BlockSize - 1) / kFp8BlockSize,
          "fp8 scale grid is too narrow for the declared column count");

  const Fp8Table& values = Fp8E4M3Table();
  const Fp8Table& scales = Fp8E8M0Table();

  for (int row = 0; row < weight.rows; ++row) {
    const std::uint8_t* weight_row =
        weight.weight + static_cast<std::size_t>(row) * weight.cols;
    const std::uint8_t* scale_row =
        weight.scale +
        static_cast<std::size_t>(row / kFp8BlockSize) * weight.scale_cols;
    // Accumulate per 128-column block so the block scale is applied once to a
    // partial sum rather than to each of the 128 products. Fewer multiplies
    // and a shorter dependency chain, same value.
    float total = 0.0F;
    for (int block_start = 0; block_start < weight.cols; block_start += kFp8BlockSize) {
      const int block_end = std::min(block_start + kFp8BlockSize, weight.cols);
      float partial = 0.0F;
      for (int col = block_start; col < block_end; ++col) {
        partial += values[weight_row[col]] * x[col];
      }
      total += partial * scales[scale_row[block_start / kFp8BlockSize]];
    }
    y[row] = total;
  }
}

// ---------------------------------------------------------------------------
// MXFP4 group-scaled
// ---------------------------------------------------------------------------

void mxfp4_dequant_row_cpu(const Mxfp4Matrix& weight, int row, float* out) {
  Require(weight.packed != nullptr && weight.scale != nullptr, "fp4 matrix is null");
  Require(row >= 0 && row < weight.rows, "fp4 row out of range");
  Require(weight.cols % 2 == 0, "fp4 logical columns must be even");

  const Fp8Table& scales = Fp8E8M0Table();
  const Fp4PairTable& pairs = Fp4Pairs();
  const int groups = weight.cols / kMxfp4GroupSize;
  const std::uint8_t* packed_row =
      weight.packed + static_cast<std::size_t>(row) * (weight.cols / 2);
  const std::uint8_t* scale_row =
      weight.scale + static_cast<std::size_t>(row) * groups;

  for (int group = 0; group < groups; ++group) {
    const float scale = scales[scale_row[group]];
    const int first = group * kMxfp4GroupSize;
    for (int col = first; col < first + kMxfp4GroupSize; col += 2) {
      // The nibble order is folded into the table; see Fp4Pairs().
      const Fp4Pair pair = pairs[packed_row[col / 2]];
      out[col] = pair.even * scale;
      out[col + 1] = pair.odd * scale;
    }
  }
}

void mxfp4_gemv_cpu(const Mxfp4Matrix& weight, const float* x, float* y) {
  Require(weight.packed != nullptr && weight.scale != nullptr, "fp4 matrix is null");
  Require(x != nullptr && y != nullptr, "fp4 gemv operand is null");
  Require(weight.rows > 0 && weight.cols > 0, "fp4 matrix has an empty extent");
  Require(weight.cols % kMxfp4GroupSize == 0,
          "fp4 logical columns must be a multiple of the 32-element group");

  const Fp8Table& scales = Fp8E8M0Table();
  const Fp4PairTable& pairs = Fp4Pairs();
  const int groups = weight.cols / kMxfp4GroupSize;

  for (int row = 0; row < weight.rows; ++row) {
    const std::uint8_t* packed_row =
        weight.packed + static_cast<std::size_t>(row) * (weight.cols / 2);
    const std::uint8_t* scale_row =
        weight.scale + static_cast<std::size_t>(row) * groups;
    float total = 0.0F;
    for (int group = 0; group < groups; ++group) {
      const int first = group * kMxfp4GroupSize;
      float partial = 0.0F;
      for (int col = first; col < first + kMxfp4GroupSize; col += 2) {
        const Fp4Pair pair = pairs[packed_row[col / 2]];
        partial += pair.even * x[col];
        partial += pair.odd * x[col + 1];
      }
      total += partial * scales[scale_row[group]];
    }
    y[row] = total;
  }
}

// ---------------------------------------------------------------------------
// BF16
// ---------------------------------------------------------------------------

void bf16_gemv_cpu(const std::uint16_t* weight, int rows, int cols, const float* x,
                   float* y) {
  Require(weight != nullptr && x != nullptr && y != nullptr,
          "bf16 gemv operand is null");
  Require(rows > 0 && cols > 0, "bf16 matrix has an empty extent");
  for (int row = 0; row < rows; ++row) {
    const std::uint16_t* weight_row = weight + static_cast<std::size_t>(row) * cols;
    float total = 0.0F;
    for (int col = 0; col < cols; ++col) {
      total += Bf16ToFloat(weight_row[col]) * x[col];
    }
    y[row] = total;
  }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

float rms_norm_cpu(const float* x, const void* weight_bf16, float* out, int count,
                   float eps) {
  Require(x != nullptr && out != nullptr, "rms norm operand is null");
  Require(count > 0, "rms norm needs a positive count");
  double sum_squares = 0.0;
  for (int i = 0; i < count; ++i) {
    sum_squares += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float mean_square = static_cast<float>(sum_squares / static_cast<double>(count));
  const float inv_rms = 1.0F / std::sqrt(mean_square + eps);
  // weight_bf16 == nullptr is the has_weight=False variant the reference calls
  // q_norm_without_weight; it normalizes with no learned gain.
  const auto* weight = static_cast<const std::uint16_t*>(weight_bf16);
  for (int i = 0; i < count; ++i) {
    const float gain = (weight == nullptr) ? 1.0F : Bf16ToFloat(weight[i]);
    out[i] = gain * x[i] * inv_rms;
  }
  return inv_rms;
}

// ---------------------------------------------------------------------------
// Resident set
// ---------------------------------------------------------------------------

MemoryUsage ReadMemoryUsage() {
  MemoryUsage usage;
  std::ifstream status("/proc/self/status");
  if (!status.good()) {
    return usage;  // available stays false
  }
  std::string line;
  while (std::getline(status, line)) {
    // Both fields are reported in kB by the kernel.
    const auto parse = [&line](const char* key, std::size_t* destination) {
      if (line.compare(0, std::strlen(key), key) != 0) {
        return false;
      }
      std::istringstream parts(line.substr(std::strlen(key)));
      std::size_t kilobytes = 0;
      parts >> kilobytes;
      *destination = kilobytes * 1024;
      return true;
    };
    if (parse("VmRSS:", &usage.current_rss_bytes)) {
      usage.available = true;
    } else if (parse("VmHWM:", &usage.peak_rss_bytes)) {
      usage.available = true;
    }
  }
  return usage;
}

}  // namespace moe::emu
