#include "mhc_fused_sinkhorn_gemv_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace moe::ops::mhc_fused_sinkhorn_gemv {
namespace {

void RequireFinite(const float* values, std::size_t count, const char* name) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
  }
}

}  // namespace

std::uint16_t FloatToBf16(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7f800000U;
  const std::uint32_t mantissa = bits & 0x007fffffU;
  if (exponent == 0x7f800000U && mantissa != 0) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float Bf16ToFloat(std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void MhcFusedSinkhornGemvReference::Validate(const Inputs& inputs) {
  if (inputs.x_bf16.size() != kInputElements) {
    throw std::invalid_argument("x must contain exactly [1,4,4096] BF16 elements");
  }
  if (inputs.phi.size() != kPhiElements) {
    throw std::invalid_argument("phi must contain exactly [24,16384] FP32 elements");
  }
  if (inputs.h_out_bf16.size() != kHiddenSize) {
    throw std::invalid_argument("h_out must contain exactly [1,4096] BF16 elements");
  }
  RequireFinite(inputs.phi.data(), inputs.phi.size(), "phi");
  RequireFinite(inputs.alpha.data(), inputs.alpha.size(), "alpha");
  RequireFinite(inputs.bias.data(), inputs.bias.size(), "bias");
  for (const auto value : inputs.x_bf16) {
    if (!std::isfinite(Bf16ToFloat(value))) {
      throw std::invalid_argument("x must contain only finite BF16 values");
    }
  }
  for (const auto value : inputs.h_out_bf16) {
    if (!std::isfinite(Bf16ToFloat(value))) {
      throw std::invalid_argument("h_out must contain only finite BF16 values");
    }
  }
}

Result MhcFusedSinkhornGemvReference::Compute(
    const Inputs& inputs, std::size_t sinkhorn_iterations) {
  Validate(inputs);
  if (sinkhorn_iterations == 0) {
    throw std::invalid_argument("Sinkhorn iteration count must be positive");
  }

  std::array<float, kProjectionRows> projected{};
  for (std::size_t row = 0; row < kProjectionRows; ++row) {
    float dot = 0.0F;
    const std::size_t row_offset = row * kInputElements;
    for (std::size_t col = 0; col < kInputElements; ++col) {
      dot += inputs.phi[row_offset + col] *
             Bf16ToFloat(inputs.x_bf16[col]);
    }
    const std::size_t segment = row < kHeads ? 0 : (row < 2 * kHeads ? 1 : 2);
    projected[row] = dot * inputs.alpha[segment] + inputs.bias[row];
  }

  Result result;
  for (std::size_t i = 0; i < kHeads; ++i) {
    result.h_in[i] = projected[i];
    result.h_post[i] = projected[kHeads + i];
  }

  float maximum = projected[2 * kHeads];
  for (std::size_t i = 1; i < kResidualElements; ++i) {
    maximum = std::max(maximum, projected[2 * kHeads + i]);
  }
  for (std::size_t i = 0; i < kResidualElements; ++i) {
    result.residual_matrix[i] =
        std::exp(projected[2 * kHeads + i] - maximum);
  }

  for (std::size_t iteration = 0; iteration < sinkhorn_iterations; ++iteration) {
    for (std::size_t row = 0; row < kHeads; ++row) {
      float sum = 0.0F;
      for (std::size_t col = 0; col < kHeads; ++col) {
        sum += result.residual_matrix[row * kHeads + col];
      }
      if (!(sum > 0.0F) || !std::isfinite(sum)) {
        throw std::runtime_error("Sinkhorn row normalization encountered an invalid sum");
      }
      for (std::size_t col = 0; col < kHeads; ++col) {
        result.residual_matrix[row * kHeads + col] /= sum;
      }
    }
    for (std::size_t col = 0; col < kHeads; ++col) {
      float sum = 0.0F;
      for (std::size_t row = 0; row < kHeads; ++row) {
        sum += result.residual_matrix[row * kHeads + col];
      }
      if (!(sum > 0.0F) || !std::isfinite(sum)) {
        throw std::runtime_error("Sinkhorn column normalization encountered an invalid sum");
      }
      for (std::size_t row = 0; row < kHeads; ++row) {
        result.residual_matrix[row * kHeads + col] /= sum;
      }
    }
  }

  result.y_bf16.resize(kInputElements);
  for (std::size_t row = 0; row < kHeads; ++row) {
    for (std::size_t feature = 0; feature < kHiddenSize; ++feature) {
      float mixed = 0.0F;
      for (std::size_t source = 0; source < kHeads; ++source) {
        mixed += result.residual_matrix[source * kHeads + row] *
                 Bf16ToFloat(inputs.x_bf16[source * kHiddenSize + feature]);
      }
      const float post = result.h_post[row] * Bf16ToFloat(inputs.h_out_bf16[feature]);
      result.y_bf16[row * kHiddenSize + feature] = FloatToBf16(mixed + post);
    }
  }
  return result;
}

}  // namespace moe::ops::mhc_fused_sinkhorn_gemv
