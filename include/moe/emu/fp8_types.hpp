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

// Standalone 8-bit float conversions for the host emulation layer.
//
// Three formats, because a DeepSeek-V4-Flash checkpoint carries all three and
// `model.safetensors.index.json` spells them F8_E4M3, F8_E5M2 and F8_E8M0:
//
//   E4M3 (OCP "E4M3FN")  1-4-3, bias 7.  No infinities. The ONLY NaN
//                        encodings are 0x7F and 0xFF, which is what makes
//                        0x7E/0xFE ordinary finite values and pushes the
//                        maximum to 1.75 * 2^8 = 448. A reader that treats
//                        exp==15 as IEEE "inf/NaN" loses the top binade of
//                        every quantized weight, so this distinction is not
//                        pedantry -- it is the difference between 448 and NaN.
//   E5M2                 1-5-2, bias 15. IEEE-754-shaped: exp==31 really is
//                        inf (mantissa 0) or NaN (mantissa != 0), max 57344.
//   E8M0                 8 exponent bits, no sign and no mantissa: the OCP
//                        micro-scale type behind `"scale_fmt": "ue8m0"`. The
//                        value is 2^(byte - 127); 0xFF is the NaN code.
//
// Both signed formats decode through a 256-entry FP32 lookup table built at
// namespace scope. Dequantizing an FP8 weight block is a pure table walk --
// no branches in the inner loop, and the table is 1 KiB, so it stays resident
// in L1 next to the AVX2 accumulators.
//
// Zero dependencies beyond the C++17 standard library: no CANN, no torch, no
// compiler FP8 extension. `float` is assumed to be IEEE binary32, which is
// checked by a static_assert.

#ifndef MOE_EMU_FP8_TYPES_HPP
#define MOE_EMU_FP8_TYPES_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace moe::emu {

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "the FP8 conversions assume IEEE binary32 floats");

// ---------------------------------------------------------------------------
// bit helpers
// ---------------------------------------------------------------------------

inline float BitsToFloat(std::uint32_t bits) noexcept {
  float out = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline std::uint32_t FloatToBits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// ---------------------------------------------------------------------------
// E4M3 (OCP E4M3FN)
// ---------------------------------------------------------------------------

inline constexpr int kFp8E4M3MantissaBits = 3;
inline constexpr int kFp8E4M3ExponentBias = 7;
// 0x7E == (exp 15, mantissa 6) -> 1.75 * 2^8. Not NaN: see the header note.
inline constexpr float kFp8E4M3Max = 448.0F;

// Decode one E4M3FN byte. Returns a quiet NaN for 0x7F / 0xFF only.
inline float DecodeFp8E4M3(std::uint8_t code) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80U) << 24U;
  const int exponent = static_cast<int>((code >> kFp8E4M3MantissaBits) & 0x0FU);
  const std::uint32_t mantissa = code & 0x07U;

  if (exponent == 0x0F && mantissa == 0x07U) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (exponent == 0) {
    if (mantissa == 0) {
      return BitsToFloat(sign);  // preserves -0.0
    }
    // Subnormal: mantissa/8 * 2^-6. Representable exactly in binary32.
    const float magnitude =
        static_cast<float>(mantissa) * (1.0F / 8.0F) * (1.0F / 64.0F);
    return BitsToFloat(sign | FloatToBits(magnitude));
  }
  // Normal: rebias into binary32 and shift the mantissa into place.
  const std::uint32_t biased =
      static_cast<std::uint32_t>(exponent - kFp8E4M3ExponentBias + 127);
  return BitsToFloat(sign | (biased << 23U) | (mantissa << 20U));
}

// ---------------------------------------------------------------------------
// E5M2
// ---------------------------------------------------------------------------

inline constexpr int kFp8E5M2MantissaBits = 2;
inline constexpr int kFp8E5M2ExponentBias = 15;
inline constexpr float kFp8E5M2Max = 57344.0F;

inline float DecodeFp8E5M2(std::uint8_t code) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80U) << 24U;
  const int exponent = static_cast<int>((code >> kFp8E5M2MantissaBits) & 0x1FU);
  const std::uint32_t mantissa = code & 0x03U;

  if (exponent == 0x1F) {
    return mantissa == 0 ? BitsToFloat(sign | 0x7F800000U)
                         : std::numeric_limits<float>::quiet_NaN();
  }
  if (exponent == 0) {
    if (mantissa == 0) {
      return BitsToFloat(sign);
    }
    const float magnitude =
        static_cast<float>(mantissa) * (1.0F / 4.0F) * (1.0F / 16384.0F);
    return BitsToFloat(sign | FloatToBits(magnitude));
  }
  const std::uint32_t biased =
      static_cast<std::uint32_t>(exponent - kFp8E5M2ExponentBias + 127);
  return BitsToFloat(sign | (biased << 23U) | (mantissa << 21U));
}

// ---------------------------------------------------------------------------
// E8M0 micro-scales ("scale_fmt": "ue8m0")
// ---------------------------------------------------------------------------

// 2^(code - 127) for code in [0, 254]; code 255 is the NaN code -- NOT 2^128
// -- per the OCP micro-scaling spec.
//
// THE CODE-0 SPECIAL CASE
//   For code >= 1 the value is just the binary32 whose biased exponent field
//   IS the code and whose mantissa is zero, so `code << 23` produces it
//   directly. Code 0 is not: a zero exponent field with a zero mantissa is
//   binary32's encoding of +0.0, whereas E8M0 code 0 means 2^-127. That value
//   is still representable -- binary32 subnormals reach down to 2^-149 -- but
//   only as a SUBNORMAL (bits 0x00400000), which the shift cannot generate.
//   Returning 0.0 here would silently annihilate every weight in a block
//   carrying the smallest scale.
inline float DecodeFp8E8M0(std::uint8_t code) noexcept {
  if (code == 0xFFU) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (code == 0U) {
    return 5.87747175411143754e-39F;  // 2^-127, a binary32 subnormal
  }
  return BitsToFloat(static_cast<std::uint32_t>(code) << 23U);
}

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

using Fp8Table = std::array<float, 256>;

namespace detail {

template <float (*Decode)(std::uint8_t)>
inline Fp8Table BuildTable() noexcept {
  Fp8Table table{};
  for (std::size_t code = 0; code < table.size(); ++code) {
    table[code] = Decode(static_cast<std::uint8_t>(code));
  }
  return table;
}

}  // namespace detail

// The dequantization inner loops index these directly.
inline const Fp8Table& Fp8E4M3Table() noexcept {
  static const Fp8Table table = detail::BuildTable<&DecodeFp8E4M3>();
  return table;
}

inline const Fp8Table& Fp8E5M2Table() noexcept {
  static const Fp8Table table = detail::BuildTable<&DecodeFp8E5M2>();
  return table;
}

inline const Fp8Table& Fp8E8M0Table() noexcept {
  static const Fp8Table table = detail::BuildTable<&DecodeFp8E8M0>();
  return table;
}

// ---------------------------------------------------------------------------
// Encoding (round-to-nearest-even, saturating)
// ---------------------------------------------------------------------------
//
// Only the tests need to go this direction -- they build FP8 fixtures from
// FP32 and then assert the round trip. Saturating rather than overflowing to
// NaN matches the checkpoint's `"activation_scheme": "dynamic"` quantizer.
//
// TIE-BREAKING, AND A BUG THAT LIVED HERE
// On an EXACT tie the two candidates are adjacent codes, and the rule below
// keeps the one whose low bit is zero -- round-to-nearest-EVEN, which is what
// IEEE-style converters and the 950PR's own
// __asc_cvt_float2_to_fp8x2 do.
//
// Until 2026-10-13 these loops used a bare `error < best_error`, which with
// an ascending scan over a monotone table keeps the SMALLER magnitude on a
// tie -- round-half-toward-zero, not round-to-nearest-even, despite the
// comment that used to claim otherwise. It differs from RNE on exactly those
// ties whose lower code is odd, i.e. half of them: a host probe over the
// E4M3 table found 63 disagreements in 126 exact ties. That is what produced
// 3 code mismatches in 4,096 when bf16_block_quant_fp8 was first run on an
// Ascend950PR_9589 -- the GOLDEN was wrong on halfway values, not the
// hardware. Fixed here rather than absorbed into a +/-1 tolerance in the
// device test, because a blanket one-code tolerance would also hide a
// genuine off-by-one in the encoder.
//
// Helper rather than an inline condition so both encoders share one rule.
namespace detail {

// True when `candidate` should displace `best` at equal error: prefer the
// even code. Only ever called with two adjacent codes, so "even" is a total
// tie-break.
inline bool PrefersEvenOnTie(std::uint32_t candidate, std::uint8_t best) noexcept {
  return (candidate % 2U == 0U) && (best % 2U != 0U);
}

}  // namespace detail

inline std::uint8_t EncodeFp8E4M3(float value) noexcept {
  if (std::isnan(value)) {
    return 0x7FU;
  }
  const std::uint8_t sign = std::signbit(value) ? 0x80U : 0x00U;
  float magnitude = std::fabs(value);
  if (magnitude > kFp8E4M3Max) {
    return static_cast<std::uint8_t>(sign | 0x7EU);  // saturate to 448
  }

  // Exhaustive nearest-even search over the 127 finite magnitudes. The table
  // is monotone in the code, so this is a correctness-first implementation
  // the tests can trust; nothing on a hot path calls it.
  const Fp8Table& table = Fp8E4M3Table();
  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint32_t code = 0; code <= 0x7EU; ++code) {
    const float candidate = table[code];
    if (std::isnan(candidate)) {
      continue;
    }
    const float error = std::fabs(candidate - magnitude);
    if (error < best_error) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    } else if (error == best_error && detail::PrefersEvenOnTie(code, best)) {
      // Exact tie: take the even code (round-to-nearest-even). See the
      // tie-breaking note above for why a bare `<` was wrong here.
      best = static_cast<std::uint8_t>(code);
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

inline std::uint8_t EncodeFp8E5M2(float value) noexcept {
  if (std::isnan(value)) {
    return 0x7FU;
  }
  const std::uint8_t sign = std::signbit(value) ? 0x80U : 0x00U;
  const float magnitude = std::fabs(value);
  if (std::isinf(magnitude)) {
    return static_cast<std::uint8_t>(sign | 0x7CU);
  }
  if (magnitude > kFp8E5M2Max) {
    return static_cast<std::uint8_t>(sign | 0x7BU);  // saturate to 57344
  }
  const Fp8Table& table = Fp8E5M2Table();
  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint32_t code = 0; code <= 0x7BU; ++code) {
    const float error = std::fabs(table[code] - magnitude);
    if (error < best_error) {
      best_error = error;
      best = static_cast<std::uint8_t>(code);
    } else if (error == best_error && detail::PrefersEvenOnTie(code, best)) {
      // Round-to-nearest-even on an exact tie, as in EncodeFp8E4M3.
      best = static_cast<std::uint8_t>(code);
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

}  // namespace moe::emu

#endif  // MOE_EMU_FP8_TYPES_HPP
