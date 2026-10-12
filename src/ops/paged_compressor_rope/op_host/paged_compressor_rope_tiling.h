#ifndef PAGED_COMPRESSOR_ROPE_TILING_H
#define PAGED_COMPRESSOR_ROPE_TILING_H

// Tiling POD and C4A (coff=2) contract geometry for paged_compressor_rope.
// Included by both op_kernel/ and op_host/; no <acl/acl.h>.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kCompressorHidden = 4096;
inline constexpr std::uint32_t kCompressorHeadDim = 512;
inline constexpr std::uint32_t kCompressorCoff = 2;   // the kernel rejects 1
inline constexpr std::uint32_t kCompressorRatio = 4;  // HOLD/EMIT period
inline constexpr std::uint32_t kCompressorReduceRows =
    kCompressorCoff * kCompressorRatio;  // 8
inline constexpr std::uint32_t kCompressorStateWidthElems =
    kCompressorCoff * kCompressorHeadDim;  // 1024
inline constexpr std::uint32_t kRopeHeadDim = 64;
inline constexpr std::uint32_t kRotaryModeHalf = 1;  // not interleave
inline constexpr float kCompressorNormEps = 1e-6F;

// ---------------------------------------------------------------------------
// sizeof == 64
// ---------------------------------------------------------------------------
struct CompressorTilingData {
  std::uint32_t abi_version;          // 1
  std::uint32_t block_dim;            // 1 AIV
  std::uint32_t hidden;               // 4096
  std::uint32_t head_dim;             // 512
  std::uint32_t coff;                 // 2, reject 1
  std::uint32_t ratio;                // 4
  std::uint32_t reduction_rows;       // 8
  std::uint32_t rope_head_dim;        // 64
  std::uint32_t rotary_mode;          // 1 = half, not interleave
  std::uint32_t projection_row_tile;  // 1 row at a time
  std::uint32_t wkv_row_stride;       // 4096 BF16 elements
  std::uint32_t wgate_row_stride;     // 4096 BF16 elements
  std::uint32_t ape_row_stride;       // 1024 FP32 elements
  std::uint32_t reserved;             // 0
  float norm_eps;                     // 1e-6
  float reserved_fp32;                // 0
};

MOE_ARCH35_ASSERT_TILING_POD(CompressorTilingData, 64);

Status MakeCompressorTiling(std::uint32_t wkv_row_stride,
                            std::uint32_t wgate_row_stride,
                            std::uint32_t ape_row_stride,
                            CompressorTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // PAGED_COMPRESSOR_ROPE_TILING_H
