#include "moe/ops/mhc_fused_sinkhorn_gemv.hpp"
#include "mhc_fused_sinkhorn_gemv_ref.hpp"
#include "mhc_fused_sinkhorn_gemv_tiling.h"

#include <algorithm>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

namespace moe::ops {

extern "C" std::uint32_t MhcFusedSinkhornGemvGetTilingPlan(
    MhcFusedSinkhornGemvTilingPlan* plan) {
  if (plan == nullptr) {
    return 1;
  }
  const auto internal_plan = mhc_fused_sinkhorn_gemv::MakeTilingPlan();
  plan->hidden_size = internal_plan.hidden_size;
  plan->head_count = internal_plan.head_count;
  plan->sinkhorn_iterations = internal_plan.sinkhorn_iterations;
  plan->reserved = internal_plan.reserved;
  plan->workspace_bytes = internal_plan.workspace_bytes;
  return 0;
}

extern "C" MhcFusedSinkhornGemvStatus MhcFusedSinkhornGemvCpuFallback(
    const std::uint16_t* x_bf16, std::size_t x_count, const float* phi,
    std::size_t phi_count, const float* alpha, std::size_t alpha_count,
    const float* bias, std::size_t bias_count,
    const std::uint16_t* h_out_bf16, std::size_t h_out_count,
    std::uint16_t* y_bf16, std::size_t y_count) {
  if (x_bf16 == nullptr || phi == nullptr || alpha == nullptr ||
      bias == nullptr || h_out_bf16 == nullptr || y_bf16 == nullptr ||
      x_count != mhc_fused_sinkhorn_gemv::kInputElements ||
      phi_count != mhc_fused_sinkhorn_gemv::kPhiElements || alpha_count != 3 ||
      bias_count != mhc_fused_sinkhorn_gemv::kProjectionRows ||
      h_out_count != mhc_fused_sinkhorn_gemv::kHiddenSize ||
      y_count != mhc_fused_sinkhorn_gemv::kInputElements) {
    return MhcFusedSinkhornGemvStatus::kInvalidArgument;
  }

  try {
    mhc_fused_sinkhorn_gemv::Inputs inputs;
    inputs.x_bf16.assign(x_bf16, x_bf16 + x_count);
    inputs.phi.assign(phi, phi + phi_count);
    std::copy_n(alpha, inputs.alpha.size(), inputs.alpha.begin());
    std::copy_n(bias, inputs.bias.size(), inputs.bias.begin());
    inputs.h_out_bf16.assign(h_out_bf16, h_out_bf16 + h_out_count);
    auto result = mhc_fused_sinkhorn_gemv::MhcFusedSinkhornGemvReference::Compute(inputs);
    std::copy(result.y_bf16.begin(), result.y_bf16.end(), y_bf16);
  } catch (const std::invalid_argument&) {
    return MhcFusedSinkhornGemvStatus::kInvalidArgument;
  } catch (const std::runtime_error&) {
    return MhcFusedSinkhornGemvStatus::kExecutionFailure;
  } catch (const std::bad_alloc&) {
    return MhcFusedSinkhornGemvStatus::kExecutionFailure;
  }
  return MhcFusedSinkhornGemvStatus::kSuccess;
}

}  // namespace moe::ops
