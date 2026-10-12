// fused_mhc_post_residual_ub_layout.hpp -- unified-buffer layout and event plan for fused_mhc_post_residual
// fused_mhc_post_residual (4 AIVs, one output stream per block, pure SIMD).
//
// Segment table (per core; 512-element feature tiles, 8 tiles per stream):
//
//   offset   size    segment
//   0        1,024   xTileBf16     BF16 [512] staging tile (reused per stream)
//   1,024    8,192   xF32[4]       four widened source tiles, 4 x 2,048 B
//   9,216    1,024   hOutTileBf16  BF16 [512] h_out staging tile
//   10,240   2,048   hOutF32       widened h_out tile
//   12,288   2,048   acc           FP32 accumulator tile
//   14,336   1,024   yTileBf16     BF16 [512] output tile
//   15,360   128     scalarState   16 B coefficients + B[i][j] mix scalars
//
// Total: 15,488 B. The contract's v1 budget of 13,440 B counts only the five
// resident compute tiles (8,192 + 2,048 + 2,048 + 1,024 + 128); the two BF16
// MTE2 landing tiles above are realized staging, so the implemented bound is
// 15,488 B. Both are far below kUsableUbBytes (221,184 B); per-core DataCache
// after the reserve is >= 32 KiB. The per-stream mix is applied in place on
// the widened source tiles (Muls then Add), so no product scratch is needed.
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    x tiles and h_out tile arrivals
//   EVENT_ID1  V    -> MTE3 y tile store readiness
//
// Four blocks write disjoint y rows; there is no inter-core barrier and no
// SyncAll. In-place x==y is rejected host-side (contract: forbidden).

#ifndef MOE_KERNELS_ARCH35_FUSED_MHC_POST_RESIDUAL_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_FUSED_MHC_POST_RESIDUAL_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::fused_mhc_post_residual {

inline constexpr std::uint32_t kStreams = 4;
inline constexpr std::uint32_t kHidden = 4096;
inline constexpr std::uint32_t kFeatureTile = 512;
inline constexpr std::uint32_t kTilesPerStream = kHidden / kFeatureTile;  // 8

inline constexpr std::uint32_t kXTileBytes = kFeatureTile * 2;    // 1,024
inline constexpr std::uint32_t kXF32TileBytes = kFeatureTile * 4;  // 2,048
inline constexpr std::uint32_t kAccBytes = kFeatureTile * 4;       // 2,048
inline constexpr std::uint32_t kYTileBytes = kFeatureTile * 2;     // 1,024
inline constexpr std::uint32_t kScalarStateBytes = 128;

inline constexpr std::uint32_t kTotalUbBytes =
    kXTileBytes + kStreams * kXF32TileBytes + kXTileBytes /*h_out staging*/ +
    kXF32TileBytes + kAccBytes + kYTileBytes + kScalarStateBytes;

// The contract's v1 budget for the five resident compute tiles.
inline constexpr std::uint32_t kContractUbBudgetBytes = 13440;

static_assert(kTotalUbBytes == 15488, "fused_mhc_post_residual UB layout must sum to 15,488 B");
static_assert(kTotalUbBytes <= kUsableUbBytes,
              "fused_mhc_post_residual UB layout exceeds the arch35 usable UB ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kTotalUbBytes >=
                  kUbDataCacheMinBytes,
              "fused_mhc_post_residual UB layout starves the SIMT DataCache below 32 KiB");

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;
inline constexpr std::int32_t kEvVToMte2 = 2;

}  // namespace moe::kernels::arch35::fused_mhc_post_residual

#endif  // MOE_KERNELS_ARCH35_FUSED_MHC_POST_RESIDUAL_UB_LAYOUT_HPP
