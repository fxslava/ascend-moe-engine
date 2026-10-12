#ifndef FUSED_MHC_PRE_NORM_SINKHORN_TILING_H
#define FUSED_MHC_PRE_NORM_SINKHORN_TILING_H

// Tiling POD and v1 contract geometry for fused_mhc_pre_norm_sinkhorn.
//
// Included by BOTH sides: op_kernel/ reads the POD out of GM, op_host/ fills
// and validates it. It therefore stays free of <acl/acl.h> -- see
// include/moe/kernels/arch35/status.hpp for why `Status` lives apart.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

// ---------------------------------------------------------------------------
// v1 contract geometry (docs/kernel_contracts_spec.md, "Fixed token geometry")
// ---------------------------------------------------------------------------
inline constexpr std::uint32_t kMhcStreams = 4;
inline constexpr std::uint32_t kMhcHidden = 4096;
// phi is [24, 16384]: n*n mixing logits + 2n gain/bias rows.
inline constexpr std::uint32_t kMhcProjectionRows =
    kMhcStreams * kMhcStreams + 2 * kMhcStreams;  // 24
inline constexpr std::uint32_t kMhcProjectionTile = 4096;
inline constexpr std::uint32_t kMhcProjectionChunks = 4;
inline constexpr std::uint32_t kMhcQueueDepth = 2;
inline constexpr std::uint32_t kMhcSinkhornIterations = 20;
inline constexpr std::uint32_t kMhcFlatWidth = kMhcStreams * kMhcHidden;  // 16384
inline constexpr float kMhcHcEps = 1e-6F;
inline constexpr float kMhcNormEps = 1e-6F;

// Dynamic UB layout for SIMT-visible scalar buffers (u[24] + pre[4])
inline constexpr std::uint32_t MhcPreNormDynamicUbBytes() noexcept {
  return 256;  // 24 dots + 4 pre outputs + slack, 32-byte aligned
}

// ---------------------------------------------------------------------------
// sizeof == 64
// ---------------------------------------------------------------------------
struct MhcPreNormTilingData {
  std::uint32_t abi_version;          // 1
  std::uint32_t block_dim;            // 1 AIV
  std::uint32_t hidden;               // 4096
  std::uint32_t streams;              // 4
  std::uint32_t projection_rows;      // 24
  std::uint32_t projection_tile;      // 4096 FP32 elements
  std::uint32_t projection_chunks;    // 4 per row
  std::uint32_t queue_depth;          // 2
  std::uint32_t sinkhorn_iterations;  // 20
  std::uint32_t x_stream_stride;      // 4096 BF16 elements
  std::uint32_t phi_row_stride;       // 16384 FP32 elements
  std::uint32_t reserved;             // 0
  float hc_eps;                       // 1e-6
  float bundle_norm_eps;              // 1e-6
  float activation_norm_eps;          // 1e-6
  float reserved_fp32;                // 0
};

MOE_ARCH35_ASSERT_TILING_POD(MhcPreNormTilingData, 64);

Status MakeMhcPreNormTiling(std::uint32_t x_stream_stride,
                            std::uint32_t phi_row_stride,
                            MhcPreNormTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // FUSED_MHC_PRE_NORM_SINKHORN_TILING_H