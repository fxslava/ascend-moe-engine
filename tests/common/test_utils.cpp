#include "test_utils.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace test_common {
namespace {
using namespace moe::ops::mhc_fused_sinkhorn_gemv;
}

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void RequireNear(float actual, float expected, float tolerance,
                 const std::string& label) {
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
    std::ostringstream message;
    message << label << ": expected " << expected << ", got " << actual
            << " (tolerance " << tolerance << ')';
    throw std::runtime_error(message.str());
  }
}

void RequireBirkhoff(
    const std::array<float, kResidualElements>& matrix, float tolerance) {
  for (std::size_t row = 0; row < kHeads; ++row) {
    float sum = 0.0F;
    for (std::size_t col = 0; col < kHeads; ++col) {
      sum += matrix[row * kHeads + col];
    }
    RequireNear(sum, 1.0F, tolerance, "Birkhoff row sum");
  }
  for (std::size_t col = 0; col < kHeads; ++col) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < kHeads; ++row) {
      sum += matrix[row * kHeads + col];
    }
    RequireNear(sum, 1.0F, tolerance, "Birkhoff column sum");
  }
}

Inputs MakeUniformSinkhornCase() {
  Inputs inputs;
  inputs.x_bf16.resize(kInputElements);
  inputs.phi.assign(kPhiElements, 0.0F);
  inputs.alpha = {2.0F, 2.0F, 2.0F};
  inputs.bias.fill(0.0F);
  inputs.h_out_bf16.assign(kHiddenSize, FloatToBf16(0.25F));

  for (std::size_t head = 0; head < kHeads; ++head) {
    const float value = static_cast<float>(head + 1);
    const auto encoded = FloatToBf16(value);
    for (std::size_t feature = 0; feature < kHiddenSize; ++feature) {
      inputs.x_bf16[head * kHiddenSize + feature] = encoded;
    }

    inputs.phi[head * kInputElements] = 0.25F;
    inputs.phi[(kHeads + head) * kInputElements] =
        0.25F * static_cast<float>(head + 1);
  }
  return inputs;
}

}  // namespace test_common
