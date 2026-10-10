#ifndef MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP
#define MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP

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

}  // namespace moe::ops

#endif  // MOE_OPS_MHC_FUSED_SINKHORN_GEMV_HPP
