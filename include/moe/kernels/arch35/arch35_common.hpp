// arch35_common.hpp -- device-side shared helpers for the Stage-1 arch35
// kernels (fused_mhc_pre_norm_sinkhorn..mxfp4_expert_gemv).
//
// Included only from device translation units after "kernel_operator.h".
// Everything here is register-level arithmetic: no UB traffic, no GM traffic,
// and no branches on data (selects only), so the same helpers serve both the
// SIMD (vector) phases and the SIMT (`__simt_vf__`) phases.
//
// Numerical contracts mirror the host oracle (src/emu/, include/moe/emu/):
//   * OCP E2M1 has no inf/NaN; the eight magnitudes are exact.
//   * E8M0 is 2^(code-127); code 255 is the NaN code (ldexpf -> inf here, the
//     oracle returns NaN -- garbage-in-garbage-in either way, documented).
//   * FP32->BF16 rounds to nearest even and quiets NaNs into the top mantissa
//     bit, exactly like moe::emu::FloatToBf16.
//   * The fast_fp8_indexer_topk selection composite packs (orderable BF16 score, ~index)
//     into one uint32 so a single unsigned compare realizes the golden's
//     "score descending, then logical index ascending" total order.

#ifndef MOE_KERNELS_ARCH35_COMMON_HPP
#define MOE_KERNELS_ARCH35_COMMON_HPP

#include <cstdint>

#include "kernel_operator.h"
// The SIMT entry points in the kernels use these; pulling them in here makes
// the common helpers self-contained. NOTE for kernel authors: once these are
// visible, the SIMT libm names (::expf, ::sqrtf, ::ldexpf, ...) shadow the
// scalar ones and reject non-SIMT callers -- the SIMD/vector phases must use
// the AscendC vector intrinsics (Exp, Rsqrt, ...) and plain C arithmetic.
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_functions.h"

namespace moe::kernels::arch35 {

// ---------------------------------------------------------------------------
// UB budget (DaVinci v3 / arch35 AIV)
// ---------------------------------------------------------------------------
// Physical UB is 256 KiB. The architecture reserves 8 KiB for the compiler
// runtime and requires a SIMT DataCache of at least 32 KiB (DataCache =
// min(256 KiB - static - dynamic - 8 KiB, 128 KiB), and a DataCache below
// 32 KiB is a link-time verification error). The usable ceiling for
// static+dynamic UB is therefore strictly:
inline constexpr std::uint32_t kUbTotalBytes = 256u * 1024u;
inline constexpr std::uint32_t kUbReservedBytes = 8u * 1024u;
inline constexpr std::uint32_t kUbDataCacheMinBytes = 32u * 1024u;
inline constexpr std::uint32_t kUsableUbBytes =
    kUbTotalBytes - kUbReservedBytes - kUbDataCacheMinBytes;  // 221,184

// HardEvent IDs 0..3 only: EVENT_ID4..7 are physically absent on arch35.
// Per-kernel allocation policy lives in each kernel's UB layout header.
inline constexpr std::uint32_t kMaxEventId = 3;

// ---------------------------------------------------------------------------
// bit helpers
// ---------------------------------------------------------------------------
// __simt_callee__ marks these as callable from SIMT entry functions: every
// one of them is register-only arithmetic. They must NOT be called from the
// SIMD (vector) phases, whose math goes through the vector unary intrinsics
// instead (the SIMT math headers shadow the scalar libm names).
__simt_callee__ __aicore__ inline float BitsToFloat(std::uint32_t bits) {
  float out;
  __builtin_memcpy(&out, &bits, sizeof(out));
  return out;
}

__simt_callee__ __aicore__ inline std::uint32_t FloatToBits(float value) {
  std::uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// ---------------------------------------------------------------------------
// OCP E2M1 (MXFP4 nibble), branch-free
// ---------------------------------------------------------------------------
// code = s eem, bias 1: exp 0 -> subnormal m*0.5, else (1+m/2)*2^(e-1).
// The magnitudes are {0,.5,1,1.5,2,3,4,6}; both nibble orders resolve to the
// same decode -- the ORDER is applied by the caller when it splits a byte.
__simt_callee__ __aicore__ inline float DecodeFp4E2M1(std::uint32_t code) {
  const std::uint32_t mag_code = code & 0x7u;
  // subnormal (mag_code < 2): (mag_code & 1) * 0.5
  // normal: ldexpf(1 + (mag_code & 1) * 0.5, (mag_code >> 1) - 1)
  const float normal =
      ldexpf(1.0f + static_cast<float>(mag_code & 1u) * 0.5f,
             static_cast<int>((mag_code >> 1u) & 0x3u) - 1);
  const float subnormal = static_cast<float>(mag_code & 1u) * 0.5f;
  const float magnitude =
      (mag_code < 2u) ? subnormal : normal;  // select, not a branch on data
  const float sign = (code & 0x8u) != 0u ? -1.0f : 1.0f;
  return magnitude * sign;
}

// ---------------------------------------------------------------------------
// E8M0 micro-scale
// ---------------------------------------------------------------------------
// 2^(code - 127) for code in [0, 254]. Code 255 is the NaN code in the OCP
// spec; ldexpf(1,128) yields +inf here where the host oracle yields NaN.
// Code 0 (2^-127) flushes to zero under device FTZ; a group carrying the
// smallest scale contributes < 1e-38 either way.
__simt_callee__ __aicore__ inline float DecodeE8M0(std::uint8_t code) {
  return ldexpf(1.0f, static_cast<int>(code) - 127);
}

// ---------------------------------------------------------------------------
// FP8 E4M3FN decode (exact oracle match, register arithmetic)
// ---------------------------------------------------------------------------
// 1-4-3, bias 7, no infinities; 0x7F/0xFF are the only NaN codes. Subnormals
// are m/8 * 2^-6. Matches moe::emu::DecodeFp8E4M3 bit for bit.
__simt_callee__ __aicore__ inline float DecodeFp8E4M3(std::uint8_t code) {
  const std::uint32_t sign = static_cast<std::uint32_t>(code & 0x80u) << 24u;
  const std::uint32_t exponent = (static_cast<std::uint32_t>(code) >> 3u) & 0x0Fu;
  const std::uint32_t mantissa = code & 0x07u;
  if (exponent == 0x0Fu && mantissa == 0x07u) {
    return BitsToFloat(0x7FC00000u);  // quiet NaN
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return BitsToFloat(sign);  // +-0.0
    }
    const float magnitude =
        static_cast<float>(mantissa) * (1.0f / 8.0f) * (1.0f / 64.0f);
    return BitsToFloat(sign | FloatToBits(magnitude));
  }
  const std::uint32_t biased = exponent - 7u + 127u;
  return BitsToFloat(sign | (biased << 23u) | (mantissa << 20u));
}

// ---------------------------------------------------------------------------
// BF16 rounding (fast_fp8_indexer_topk selection): round-to-nearest-even, NaN quieted
// ---------------------------------------------------------------------------
__simt_callee__ __aicore__ inline std::uint16_t FloatToBf16Bits(float value) {
  std::uint32_t bits = FloatToBits(value);
  const std::uint32_t exponent = bits & 0x7F800000u;
  const std::uint32_t mantissa = bits & 0x007FFFFFu;
  if (exponent == 0x7F800000u && mantissa != 0u) {
    return static_cast<std::uint16_t>((bits >> 16u) | 0x0040u);
  }
  bits += 0x7FFFu + ((bits >> 16u) & 1u);  // round to nearest even
  return static_cast<std::uint16_t>(bits >> 16u);
}

// ---------------------------------------------------------------------------
// fast_fp8_indexer_topk selection composite
// ---------------------------------------------------------------------------
// Order a (possibly negative) BF16 score with one unsigned compare by
// flipping the sign bit for positives and all bits for negatives, then pack
// the inverted logical index into the low 16 bits: a LARGER uint32 is a
// higher score, and on an exact BF16 tie a SMALLER logical index. Padding
// entries use kSelectionPadKey, which no real entry can equal because the
// inverted index of any j < 0x10000 is nonzero.
inline constexpr std::uint32_t kSelectionPadKey = 0u;

__simt_callee__ __aicore__ inline std::uint32_t MakeSelectionKey(
    float score_f32, std::uint32_t index) {
  const std::uint16_t bf16 = FloatToBf16Bits(score_f32);
  const std::uint16_t mono =
      bf16 ^ ((bf16 & 0x8000u) != 0u ? 0xFFFFu : 0x8000u);
  const std::uint32_t inverted_index = 0xFFFFu ^ (index & 0xFFFFu);
  return (static_cast<std::uint32_t>(mono) << 16u) | inverted_index;
}

__simt_callee__ __aicore__ inline std::uint32_t SelectionKeyToIndex(
    std::uint32_t key) {
  return 0xFFFFu ^ (key & 0xFFFFu);
}

}  // namespace moe::kernels::arch35

#endif  // MOE_KERNELS_ARCH35_COMMON_HPP
