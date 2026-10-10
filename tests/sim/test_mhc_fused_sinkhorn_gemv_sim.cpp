#include "test_utils.hpp"
#include "tensor_factory.hpp"
#include "moe/ops/mhc_fused_sinkhorn_gemv.hpp"

#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace moe::ops::mhc_fused_sinkhorn_gemv;
using test_common::Require;
using test_common::RequireBirkhoff;
using test_common::RequireNear;

void TestParameterValidation() {
  auto inputs = test_common::MakeDeterministicMhcFusedSinkhornGemvInputs();
  MhcFusedSinkhornGemvReference::Validate(inputs);

  inputs.phi.pop_back();
  bool rejected = false;
  try {
    MhcFusedSinkhornGemvReference::Validate(inputs);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "invalid phi shape was not rejected");
}

void TestBirkhoffInvariantAndReferenceVectors() {
  const auto inputs = test_common::MakeDeterministicMhcFusedSinkhornGemvInputs();
  const auto result = MhcFusedSinkhornGemvReference::Compute(inputs);

  RequireBirkhoff(result.residual_matrix, 1.0e-4F);
  for (const float value : result.residual_matrix) {
    RequireNear(value, 0.25F, 1.0e-6F, "uniform Sinkhorn reference value");
  }
  for (std::size_t head = 0; head < kHeads; ++head) {
    RequireNear(result.h_in[head], 0.5F, 1.0e-6F, "h_in projection");
    RequireNear(result.h_post[head],
                0.5F * static_cast<float>(head + 1), 1.0e-6F,
                "h_post projection");
  }

  const std::array<float, kHeads> expected{2.625F, 2.75F, 2.875F, 3.0F};
  const std::array<std::size_t, 3> features{0, 17, kHiddenSize - 1};
  for (std::size_t head = 0; head < kHeads; ++head) {
    for (const std::size_t feature : features) {
      const float actual =
          Bf16ToFloat(result.y_bf16[head * kHiddenSize + feature]);
      RequireNear(actual, expected[head], 0.0F, "BF16 reference output");
    }
  }
}

void TestNonUniformSinkhornReferenceVector() {
  auto inputs = test_common::MakeDeterministicMhcFusedSinkhornGemvInputs();
  const std::array<float, kResidualElements> logits{
      0.0F, 0.2F, -0.3F, 0.5F,
      0.6F, -0.1F, 0.4F, -0.2F,
      -0.4F, 0.3F, 0.0F, -0.1F,
      0.2F, -0.5F, 0.7F, 0.0F};
  for (std::size_t i = 0; i < logits.size(); ++i) {
    inputs.bias[2 * kHeads + i] = logits[i];
  }
  const auto result = MhcFusedSinkhornGemvReference::Compute(inputs);
  const std::array<float, kResidualElements> expected{
      0.210644693F, 0.284853272F, 0.138186950F, 0.366315085F,
      0.363801416F, 0.200018443F, 0.263760856F, 0.172419285F,
      0.167380764F, 0.373184397F, 0.221120085F, 0.238314754F,
      0.258173128F, 0.141943888F, 0.376932108F, 0.222950876F};
  RequireBirkhoff(result.residual_matrix, 1.0e-4F);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    RequireNear(result.residual_matrix[i], expected[i], 1.0e-6F,
                "non-uniform Sinkhorn reference value");
  }
}

void TestBf16Rounding() {
  Require(FloatToBf16(1.00390625F) == 0x3f80U,
          "BF16 ties-to-even rounding failed for an even lower value");
  Require(FloatToBf16(1.01171875F) == 0x3f82U,
          "BF16 ties-to-even rounding failed for an odd lower value");
}

void TestHostTilingPlanContract() {
  moe::ops::MhcFusedSinkhornGemvTilingPlan plan{};
  Require(moe::ops::MhcFusedSinkhornGemvGetTilingPlan(&plan) == 0,
          "host tiling-plan query failed");
  Require(plan.hidden_size == kHiddenSize && plan.head_count == kHeads &&
              plan.sinkhorn_iterations == kSinkhornIterations &&
              plan.workspace_bytes == 0,
          "host tiling-plan fields do not match the static operator contract");
  Require(moe::ops::MhcFusedSinkhornGemvGetTilingPlan(nullptr) != 0,
          "null host tiling-plan output was not rejected");
}

void TestHostCpuFallback() {
  const auto inputs = test_common::MakeDeterministicMhcFusedSinkhornGemvInputs();
  std::vector<std::uint16_t> output(kInputElements);
  const auto status = moe::ops::MhcFusedSinkhornGemvCpuFallback(
      inputs.x_bf16.data(), inputs.x_bf16.size(), inputs.phi.data(),
      inputs.phi.size(), inputs.alpha.data(), inputs.alpha.size(),
      inputs.bias.data(), inputs.bias.size(), inputs.h_out_bf16.data(),
      inputs.h_out_bf16.size(), output.data(), output.size());
  Require(status == moe::ops::MhcFusedSinkhornGemvStatus::kSuccess,
          "host CPU fallback failed");

  const auto reference = MhcFusedSinkhornGemvReference::Compute(inputs);
  Require(output == reference.y_bf16,
          "host CPU fallback differs from the portable reference");
  Require(moe::ops::MhcFusedSinkhornGemvCpuFallback(
              nullptr, inputs.x_bf16.size(), inputs.phi.data(),
              inputs.phi.size(), inputs.alpha.data(), inputs.alpha.size(),
              inputs.bias.data(), inputs.bias.size(),
              inputs.h_out_bf16.data(), inputs.h_out_bf16.size(),
              output.data(), output.size()) ==
              moe::ops::MhcFusedSinkhornGemvStatus::kInvalidArgument,
          "host CPU fallback did not reject a null input");
}

}  // namespace

int main() {
  try {
    TestParameterValidation();
    TestBirkhoffInvariantAndReferenceVectors();
    TestNonUniformSinkhornReferenceVector();
    TestBf16Rounding();
    TestHostTilingPlanContract();
    TestHostCpuFallback();
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] MhcFusedSim: " << error.what() << '\n';
    return 1;
  }
  std::cout << "[PASS] MhcFusedSim\n";
  return 0;
}
