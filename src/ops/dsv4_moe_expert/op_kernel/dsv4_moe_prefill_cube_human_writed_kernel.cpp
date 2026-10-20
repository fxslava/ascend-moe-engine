// Educational, deliberately synchronous MXFP4 MoE prefill kernel.
// Geometry is fixed so the dataflow is visible without tiling or framework code.
#include "kernel_operator.h"

#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
#include "utils/debug/asc_printf.h"
#define DSV4_PIPELINE_TRACE(...) __asc_aicore::printf(__VA_ARGS__)
#else
#define DSV4_PIPELINE_TRACE(...)
#endif

namespace AscendC {
namespace {

constexpr uint32_t kTokens = 16;
constexpr uint32_t kHidden = 4096;
constexpr uint32_t kInter = 2048;
constexpr uint32_t kGroup = 32;
constexpr uint32_t kColumnBand = 64;

__aicore__ inline float Bf16ToFloat(uint16_t bits) {
  union Bits {
    uint32_t u;
    float f;
  } value{};
  value.u = static_cast<uint32_t>(bits) << 16;
  return value.f;
}

// Scalar range-reduced exp for the educational scalar path. The seventh-order
// Taylor polynomial is evaluated on [0, ln(2)); exponent scaling is exact.
__aicore__ inline float ExpApprox(float x) {
  const float y = x * 1.4426950408889634F;
  int32_t exponent = static_cast<int32_t>(y);
  if (static_cast<float>(exponent) > y) --exponent;
  if (exponent < -126) return 0.0F;
  if (exponent > 127) return 3.402823466e+38F;
  const float f = y - static_cast<float>(exponent);
  const float z = f * 0.6931471805599453F;
  float polynomial = 1.0F / 5040.0F;
  polynomial = 1.0F / 720.0F + z * polynomial;
  polynomial = 1.0F / 120.0F + z * polynomial;
  polynomial = 1.0F / 24.0F + z * polynomial;
  polynomial = 1.0F / 6.0F + z * polynomial;
  polynomial = 0.5F + z * polynomial;
  polynomial = 1.0F + z * polynomial;
  polynomial = 1.0F + z * polynomial;
  union Bits {
    uint32_t u;
    float f;
  } scale{};
  scale.u = static_cast<uint32_t>(exponent + 127) << 23;
  return scale.f * polynomial;
}

__aicore__ inline float E2M1Magnitude(uint8_t code) {
  switch (code & 0x7U) {
    case 0: return 0.0F;
    case 1: return 0.5F;
    case 2: return 1.0F;
    case 3: return 1.5F;
    case 4: return 2.0F;
    case 5: return 3.0F;
    case 6: return 4.0F;
    default: return 6.0F;
  }
}

__aicore__ inline float E8M0ToFloat(uint8_t code) {
  if (code == 0U) return 0.0F;
  if (code == 255U) return 0.0F;
  union Bits {
    uint32_t u;
    float f;
  } value{};
  value.u = static_cast<uint32_t>(code) << 23;
  return value.f;
}

__aicore__ inline float DequantWeight(const __gm__ uint8_t* packed,
                                      const __gm__ uint8_t* scales,
                                      uint32_t element_index) {
  const uint8_t byte = packed[element_index >> 1];
  const uint8_t nibble =
      (element_index & 1U) == 0U ? (byte & 0x0fU) : (byte >> 4);
  const float sign = (nibble & 0x08U) == 0U ? 1.0F : -1.0F;
  const float magnitude = E2M1Magnitude(nibble);
  const float scale = E8M0ToFloat(scales[element_index / kGroup]);
  return sign * magnitude * scale;
}

__aicore__ inline void TraceWeights(const __gm__ uint8_t* packed,
                                    const __gm__ uint8_t* scales) {
  DSV4_PIPELINE_TRACE("[dsv4-human] dequantized weight strip:");
  for (uint32_t i = 0; i < 8; ++i) {
    DSV4_PIPELINE_TRACE(" %f", DequantWeight(packed, scales, i));
  }
  DSV4_PIPELINE_TRACE("\n");
}

}  // namespace

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_human_writed(
    GM_ADDR x_gm, GM_ADDR gate_packed_gm, GM_ADDR gate_scales_gm,
    GM_ADDR up_packed_gm, GM_ADDR up_scales_gm, GM_ADDR down_packed_gm,
    GM_ADDR down_scales_gm, GM_ADDR gate_gm, GM_ADDR up_gm, GM_ADDR inter_gm,
    GM_ADDR y_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

  const __gm__ uint16_t* x = reinterpret_cast<const __gm__ uint16_t*>(x_gm);
  const __gm__ uint8_t* gate_packed =
      reinterpret_cast<const __gm__ uint8_t*>(gate_packed_gm);
  const __gm__ uint8_t* gate_scales =
      reinterpret_cast<const __gm__ uint8_t*>(gate_scales_gm);
  const __gm__ uint8_t* up_packed =
      reinterpret_cast<const __gm__ uint8_t*>(up_packed_gm);
  const __gm__ uint8_t* up_scales =
      reinterpret_cast<const __gm__ uint8_t*>(up_scales_gm);
  const __gm__ uint8_t* down_packed =
      reinterpret_cast<const __gm__ uint8_t*>(down_packed_gm);
  const __gm__ uint8_t* down_scales =
      reinterpret_cast<const __gm__ uint8_t*>(down_scales_gm);
  __gm__ float* gate = reinterpret_cast<__gm__ float*>(gate_gm);
  __gm__ float* up = reinterpret_cast<__gm__ float*>(up_gm);
  __gm__ float* inter = reinterpret_cast<__gm__ float*>(inter_gm);
  __gm__ float* y = reinterpret_cast<__gm__ float*>(y_gm);

  const uint32_t row = GetBlockIdx();
  if (row >= kTokens) return;

  if (row == 0) {
    DSV4_PIPELINE_TRACE("[dsv4-human] input tokens bf16: %f %f %f %f",
                        Bf16ToFloat(x[0]), Bf16ToFloat(x[1]),
                        Bf16ToFloat(x[2]), Bf16ToFloat(x[3]));
    DSV4_PIPELINE_TRACE("\n");
    DSV4_PIPELINE_TRACE("[dsv4-human] gate weights\n");
    TraceWeights(gate_packed, gate_scales);
    DSV4_PIPELINE_TRACE("[dsv4-human] up weights\n");
    TraceWeights(up_packed, up_scales);
    DSV4_PIPELINE_TRACE("[dsv4-human] down weights\n");
    TraceWeights(down_packed, down_scales);
  }

  // Stage 1: Gate = X * W_gate and Up = X * W_up.
  // We visit 64-column bands in order. Each scalar dot product is intentionally
  // explicit: packed values and their E8M0 group scales are decoded at the use.
  for (uint32_t n0 = 0; n0 < kInter; n0 += kColumnBand) {
    const uint32_t n_end = n0 + kColumnBand;
    {
      for (uint32_t n = n0; n < n_end; ++n) {
        float gate_sum = 0.0F;
        float up_sum = 0.0F;
        for (uint32_t k = 0; k < kHidden; ++k) {
          const float xv = Bf16ToFloat(x[row * kHidden + k]);
          const uint32_t wi = k * kInter + n;
          gate_sum += xv * DequantWeight(gate_packed, gate_scales, wi);
          up_sum += xv * DequantWeight(up_packed, up_scales, wi);
        }
        const uint32_t oi = row * kInter + n;
        gate[oi] = gate_sum;
        up[oi] = up_sum;
      }
    }
  }
  PipeBarrier<PIPE_ALL>();
  if (row == 0) {
    DSV4_PIPELINE_TRACE("[dsv4-human] gate gemm row0: %f %f %f %f",
                        gate[0], gate[1], gate[2], gate[3]);
    DSV4_PIPELINE_TRACE("\n");
  }

  // Stage 2: Inter = SiLU(Gate) * Up. Each block owns one token row.
  for (uint32_t i = row * kInter; i < (row + 1) * kInter; ++i) {
    const float g = gate[i];
    const float sigmoid = 1.0F / (1.0F + ExpApprox(-g));
    inter[i] = (g * sigmoid) * up[i];
  }
  PipeBarrier<PIPE_ALL>();
  if (row == 0) {
    DSV4_PIPELINE_TRACE("[dsv4-human] swiglu row0: %f %f %f %f",
                        inter[0], inter[1], inter[2], inter[3]);
    DSV4_PIPELINE_TRACE("\n");
  }

  // Stage 3: Y = Inter * W_down, again in readable output-column bands.
  for (uint32_t n0 = 0; n0 < kHidden; n0 += kColumnBand) {
    const uint32_t n_end = n0 + kColumnBand;
    {
      for (uint32_t n = n0; n < n_end; ++n) {
        float sum = 0.0F;
        for (uint32_t k = 0; k < kInter; ++k) {
          const float xv = inter[row * kInter + k];
          const uint32_t wi = k * kHidden + n;
          sum += xv * DequantWeight(down_packed, down_scales, wi);
        }
        y[row * kHidden + n] = sum;
      }
    }
  }
  PipeBarrier<PIPE_ALL>();
  if (row == 0) {
    DSV4_PIPELINE_TRACE("[dsv4-human] final y row0: %f %f %f %f",
                        y[0], y[1], y[2], y[3]);
    DSV4_PIPELINE_TRACE("\n");
  }
}

}  // namespace AscendC
