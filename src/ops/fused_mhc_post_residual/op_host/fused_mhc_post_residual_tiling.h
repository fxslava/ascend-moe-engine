#ifndef FUSED_MHC_POST_RESIDUAL_TILING_H
#define FUSED_MHC_POST_RESIDUAL_TILING_H

// Tiling POD and v1 contract geometry for fused_mhc_post_residual.
// Included by both op_kernel/ and op_host/; no <acl/acl.h>.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kMhcPostStreams = 4;
inline constexpr std::uint32_t kMhcPostHidden = 4096;
inline constexpr std::uint32_t kMhcPostFeatureTile = 512;  // 8 tiles per stream

// ---------------------------------------------------------------------------
// sizeof == 32
// ---------------------------------------------------------------------------
struct MhcPostTilingData {
  std::uint32_t abi_version;      // 1
  std::uint32_t block_dim;        // 4 AIV, one output stream per block
  std::uint32_t hidden;           // 4096
  std::uint32_t streams;          // 4
  std::uint32_t feature_tile;     // 512 elements, 8 tiles per stream
  std::uint32_t x_stream_stride;  // 4096
  std::uint32_t y_stream_stride;  // 4096
  std::uint32_t b_row_stride;     // 4
};

MOE_ARCH35_ASSERT_TILING_POD(MhcPostTilingData, 32);

Status MakeMhcPostTiling(std::uint32_t x_stream_stride,
                         std::uint32_t y_stream_stride,
                         std::uint32_t b_row_stride,
                         MhcPostTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // FUSED_MHC_POST_RESIDUAL_TILING_H
