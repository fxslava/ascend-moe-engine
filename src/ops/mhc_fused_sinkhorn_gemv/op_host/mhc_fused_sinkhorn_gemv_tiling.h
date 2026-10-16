#ifndef MHC_FUSED_SINKHORN_GEMV_TILING_H
#define MHC_FUSED_SINKHORN_GEMV_TILING_H

#include <cstddef>
#include <cstdint>

namespace moe::ops::mhc_fused_sinkhorn_gemv {

inline constexpr std::uint32_t kTilingSinkhornIterations = 20;
inline constexpr std::size_t kResidualBufferBytes = 4 * 4 * sizeof(float);
inline constexpr std::size_t kVectorBufferBudgetBytes = 64 * 1024;

struct TilingPlan {
  std::uint32_t hidden_size;
  std::uint32_t head_count;
  std::uint32_t sinkhorn_iterations;
  std::uint32_t reserved;
  std::uint64_t workspace_bytes;
};

TilingPlan MakeTilingPlan();

}  // namespace moe::ops::mhc_fused_sinkhorn_gemv

#endif  // MHC_FUSED_SINKHORN_GEMV_TILING_H
