#ifndef MXFP4_EXPERT_GEMV_TILING_H
#define MXFP4_EXPERT_GEMV_TILING_H

// Tiling POD and v1 contract geometry for mxfp4_expert_gemv.
// Included by both op_kernel/ and op_host/; no <acl/acl.h>.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kMxfp4RowsPerTile = 32;
inline constexpr std::uint32_t kMxfp4GroupSize = 32;
inline constexpr std::uint32_t kMxfp4BlockDim = 32;  // 32 AIVs, grid-stride

// ---------------------------------------------------------------------------
// sizeof == 40
// ---------------------------------------------------------------------------
struct Mxfp4GemvTilingData {
  std::uint32_t abi_version;        // 1
  std::uint32_t block_dim;          // 32 AIVs; rows assigned in 32-row tiles
  std::uint32_t rows;               // 2048 or 4096
  std::uint32_t cols;               // 4096 or 2048
  std::uint32_t rows_per_block;     // 32
  std::uint32_t group_size;         // 32
  std::uint32_t packed_row_stride;  // cols/2 bytes
  std::uint32_t scale_row_stride;   // cols/32 bytes
  std::uint32_t low_nibble_first;   // 1; reject 0
  std::uint32_t reserved;           // 0
};

MOE_ARCH35_ASSERT_TILING_POD(Mxfp4GemvTilingData, 40);

Status MakeMxfp4GemvTiling(std::uint32_t packed_row_stride,
                           std::uint32_t scale_row_stride, std::uint32_t rows,
                           std::uint32_t cols,
                           Mxfp4GemvTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // MXFP4_EXPERT_GEMV_TILING_H
