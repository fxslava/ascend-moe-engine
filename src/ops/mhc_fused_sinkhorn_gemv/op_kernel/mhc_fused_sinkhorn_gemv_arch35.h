#ifndef MHC_FUSED_SINKHORN_GEMV_ARCH35_H
#define MHC_FUSED_SINKHORN_GEMV_ARCH35_H

#include "mhc_fused_sinkhorn_gemv_tiling.h"

namespace moe::ops::mhc_fused_sinkhorn_gemv {

inline constexpr std::uint32_t kDeviceHeads = 4;
inline constexpr std::uint32_t kDeviceProjectionRows = 24;
inline constexpr std::uint32_t kDeviceResidualElements = 16;

class MhcFusedSinkhornGemvArch35 {
 public:
  __aicore__ inline explicit MhcFusedSinkhornGemvArch35(
      const TilingPlan& plan) : plan_(plan) {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha,
                              GM_ADDR bias, GM_ADDR h_out, GM_ADDR y) {
    x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    phi_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(phi));
    alpha_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    hOut_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(h_out));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
  }

  __aicore__ inline void Process() {
    float projection[kDeviceProjectionRows];
    for (std::uint32_t row = 0; row < kDeviceProjectionRows; ++row) {
      float dot = 0.0F;
      const std::uint32_t offset = row * plan_.hidden_size * plan_.head_count;
      for (std::uint32_t col = 0; col < plan_.hidden_size * plan_.head_count; ++col) {
        dot += phi_.GetValue(offset + col) *
               static_cast<float>(x_.GetValue(col));
      }
      const std::uint32_t alpha_index =
          row < kDeviceHeads ? 0 : (row < 2 * kDeviceHeads ? 1 : 2);
      projection[row] = dot * alpha_.GetValue(alpha_index) + bias_.GetValue(row);
    }

    float matrix[kDeviceResidualElements];
    float maximum = projection[2 * kDeviceHeads];
    for (std::uint32_t index = 1; index < kDeviceResidualElements; ++index) {
      if (projection[2 * kDeviceHeads + index] > maximum) {
        maximum = projection[2 * kDeviceHeads + index];
      }
    }
    for (std::uint32_t index = 0; index < kDeviceResidualElements; ++index) {
      matrix[index] = expf(projection[2 * kDeviceHeads + index] - maximum);
    }

    for (std::uint32_t iteration = 0;
         iteration < plan_.sinkhorn_iterations; ++iteration) {
      for (std::uint32_t row = 0; row < kDeviceHeads; ++row) {
        float sum = 0.0F;
        for (std::uint32_t col = 0; col < kDeviceHeads; ++col) {
          sum += matrix[row * kDeviceHeads + col];
        }
        for (std::uint32_t col = 0; col < kDeviceHeads; ++col) {
          matrix[row * kDeviceHeads + col] /= sum;
        }
      }
      for (std::uint32_t col = 0; col < kDeviceHeads; ++col) {
        float sum = 0.0F;
        for (std::uint32_t row = 0; row < kDeviceHeads; ++row) {
          sum += matrix[row * kDeviceHeads + col];
        }
        for (std::uint32_t row = 0; row < kDeviceHeads; ++row) {
          matrix[row * kDeviceHeads + col] /= sum;
        }
      }
    }

    for (std::uint32_t row = 0; row < kDeviceHeads; ++row) {
      for (std::uint32_t feature = 0; feature < plan_.hidden_size; ++feature) {
        float mixed = 0.0F;
        for (std::uint32_t source = 0; source < kDeviceHeads; ++source) {
          const std::uint32_t input_index =
              source * plan_.hidden_size + feature;
          mixed += matrix[source * kDeviceHeads + row] *
                   static_cast<float>(x_.GetValue(input_index));
        }
        const float post = projection[kDeviceHeads + row] *
                           static_cast<float>(hOut_.GetValue(feature));
        const std::uint32_t output_index = row * plan_.hidden_size + feature;
        y_.SetValue(output_index, bfloat16_t(mixed + post));
      }
    }
  }

 private:
  const TilingPlan& plan_;
  AscendC::GlobalTensor<bfloat16_t> x_;
  AscendC::GlobalTensor<float> phi_;
  AscendC::GlobalTensor<float> alpha_;
  AscendC::GlobalTensor<float> bias_;
  AscendC::GlobalTensor<bfloat16_t> hOut_;
  AscendC::GlobalTensor<bfloat16_t> y_;
};

}  // namespace moe::ops::mhc_fused_sinkhorn_gemv

#endif  // MHC_FUSED_SINKHORN_GEMV_ARCH35_H
