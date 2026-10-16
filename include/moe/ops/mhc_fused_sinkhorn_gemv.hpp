#ifndef MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP
#define MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP

#include <cstddef>
#include <cstdint>

namespace moe::ops {

struct MhcFusedSinkhornGemvTilingPlan {
  std::uint32_t hidden_size;
  std::uint32_t head_count;
  std::uint32_t sinkhorn_iterations;
  std::uint32_t reserved;
  std::uint64_t workspace_bytes;
};

extern "C" std::uint32_t MhcFusedSinkhornGemvGetTilingPlan(
    MhcFusedSinkhornGemvTilingPlan* plan);

enum class MhcFusedSinkhornGemvStatus : std::uint32_t {
  kSuccess = 0,
  kInvalidArgument = 1,
  kExecutionFailure = 2,
};

extern "C" MhcFusedSinkhornGemvStatus MhcFusedSinkhornGemvCpuFallback(
    const std::uint16_t* x_bf16, std::size_t x_count,
    const float* phi, std::size_t phi_count,
    const float* alpha, std::size_t alpha_count,
    const float* bias, std::size_t bias_count,
    const std::uint16_t* h_out_bf16, std::size_t h_out_count,
    std::uint16_t* y_bf16, std::size_t y_count);

}  // namespace moe::ops

#endif  // MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP
