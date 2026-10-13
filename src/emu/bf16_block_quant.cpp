// -----------------------------------------------------------------------------
// bf16_block_quant.cpp -- host oracle for the bf16_block_quant_fp8 operator.
//
// This is the source of truth the device kernel is compared against
// (CLAUDE.md §4.1). It deliberately does NOT mirror the kernel's structure:
// the kernel reduces a 32-element block through a 16-lane register butterfly
// because that is what the hardware FP8 encoder's float2 granularity forces,
// while this walks the block linearly. Two different shapes computing the
// same function is the point -- a transcription of the kernel would agree
// with it even when both are wrong.
//
// The quantizer:
//   m        = max over the 32-element block of |x|
//   scale[b] = (m > 0) ? m / 448 : 1
//   q[i]     = e4m3_rne_sat( x[i] * (448 / m) )
//
// Scales are FP32 and NOT powers of two. That is the divergence from the
// aclnnDynamicMxQuant path this operator replaces, which emitted OCP E8M0
// (power-of-two) scales; see op_host/bf16_block_quant_fp8_tiling.h.
//
// An all-zero block takes scale 1 and codes 0, so the dequantized value is
// exactly zero and no downstream stage multiplies by a zero or denormal
// scale.
// -----------------------------------------------------------------------------

#include "moe/emu/dsv4_layers.h"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"

#include <cmath>
#include <cstddef>

namespace moe::emu {

void bf16_block_quant_fp8_cpu(const std::uint16_t* x_bf16, int width,
                              std::uint8_t* q_out, float* scale_out) {
  if (x_bf16 == nullptr || q_out == nullptr || scale_out == nullptr ||
      width <= 0) {
    return;
  }

  const int blocks = width / kBlockQuantRefBlockSize;
  for (int b = 0; b < blocks; ++b) {
    const int base = b * kBlockQuantRefBlockSize;

    // BF16 -> FP32 is exact (BF16 is FP32 with the low 16 mantissa bits
    // cleared), so widening here introduces no error and the kernel's SIMD
    // Cast is not a source of divergence.
    float magnitude_max = 0.0F;
    for (int i = 0; i < kBlockQuantRefBlockSize; ++i) {
      const float value = Bf16ToFloat(x_bf16[base + i]);
      const float magnitude = std::fabs(value);
      if (magnitude > magnitude_max) {
        magnitude_max = magnitude;
      }
    }

    const bool nonzero = magnitude_max > 0.0F;
    // Written as a divide rather than a multiply by a precomputed reciprocal
    // so the scale is the correctly rounded FP32 quotient of the block max
    // and 448. The device forms it as m * (1/448); 1/448 is not exactly
    // representable, so the two can differ by one ulp -- the oracle test
    // allows exactly that and no more.
    scale_out[b] = nonzero ? (magnitude_max / kFp8E4M3Max) : 1.0F;
    const float inverse = nonzero ? (kFp8E4M3Max / magnitude_max) : 0.0F;

    for (int i = 0; i < kBlockQuantRefBlockSize; ++i) {
      const float value = Bf16ToFloat(x_bf16[base + i]);
      // EncodeFp8E4M3 is round-to-nearest-even and saturates at +/-448,
      // which is what __ASC_SATFINITE asks the hardware encoder for.
      q_out[base + i] = EncodeFp8E4M3(value * inverse);
    }
  }
}

float bf16_block_quant_fp8_dequant(std::uint8_t code, float scale) noexcept {
  return Fp8E4M3Table()[code] * scale;
}

}  // namespace moe::emu
