// mxfp4_expert_gemv_ub_layout.hpp -- unified-buffer layout and event plan for mxfp4_expert_gemv
// mxfp4 expert GEMV (32 AIVs, grid-stride over 32-row tiles, hybrid).
//
// Segment table (per core; cols = tiling.cols, worst case 4,096):
//
//   offset          size (cols=4096)  segment
//   0               16,384            xF32       resident FP32 x[cols]
//   16,384          2,048             partials   [32 rows][16 lanes] FP32
//   18,432          128               yTile      32 FP32 row results
//   18,560          128               scalarState loop bounds
//
// Total: 18,688 B at cols=4096 (10,752 B at cols=2048) -- inside the
// contract's padded v1 bound of 19,456 B and the directive's 35 KiB target.
// Packed nibbles and E8M0 scales are NOT staged in UB: SIMT lanes read them
// from GM through the SIMT DataCache (asc_ldca), decode OCP E2M1
// branch-free in registers and form the E8M0 scale with ldexpf, exactly as
// the group-of-32 partial is scaled in the host oracle.
//
// MAC order matches the oracle: each lane accumulates its 16 pairs (32
// logical columns, one scale group) into a register partial, then applies
// the group scale once (partial * scale), so the per-group
// scale-after-subtotal grouping of moe::emu::mxfp4_gemv_cpu is preserved.
// The 16 lane partials of each row reduce through an in-register butterfly
// (xor masks 1,2,4,8 within the aligned 16-lane row group); lane 0 writes
// the row result into the UB y tile, and the SIMD side stores each 32-row
// y tile with one V->MTE3-guarded 128 B MTE3. (A vector ReduceSum cannot
// express 32 independent 16-lane row reductions on this API generation --
// no sub-tensor views -- which is why the dense reduction is in registers.)
//
// kMxfp4LowNibbleFirst == 1 is enforced: the even logical column is decoded
// from bits 0..3 of the packed byte. Tiling low_nibble_first=0 is rejected.
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    x tile arrival (once per kernel)
//   EVENT_ID1  V    -> MTE3 yTile store readiness (once per 32-row tile)
// Phase boundaries around each SIMT launch use DataSyncBarrier<ALL>.

#ifndef MOE_KERNELS_ARCH35_MXFP4_EXPERT_GEMV_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_MXFP4_EXPERT_GEMV_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::mxfp4_expert_gemv {

inline constexpr std::uint32_t kRowsPerTile = 32;  // one tile per SIMT launch
inline constexpr std::uint32_t kLanesPerRow = 16;  // 512 threads / 32 rows
inline constexpr std::uint32_t kGroupSize = 32;

inline constexpr std::uint32_t kXF32BytesMax = 4096 * 4;                  // 16,384
inline constexpr std::uint32_t kPartialsBytes = kRowsPerTile * kLanesPerRow * 4;  // 2,048
inline constexpr std::uint32_t kYTileBytes = kRowsPerTile * 4;            // 128
inline constexpr std::uint32_t kScalarStateBytes = 128;

inline constexpr std::uint32_t kTotalUbBytes =
    kXF32BytesMax + kPartialsBytes + kYTileBytes + kScalarStateBytes;

// The contract's padded v1 bound at cols=4096.
inline constexpr std::uint32_t kContractUbBudgetBytes = 19456;
// The directive's Stage-1 target footprint.
inline constexpr std::uint32_t kDirectiveUbTargetBytes = 35 * 1024;

static_assert(kTotalUbBytes == 18688, "mxfp4_expert_gemv UB layout must sum to 18,688 B");
static_assert(kTotalUbBytes <= kContractUbBudgetBytes,
              "mxfp4_expert_gemv UB layout exceeds the contract v1 budget");
static_assert(kTotalUbBytes <= kDirectiveUbTargetBytes,
              "mxfp4_expert_gemv UB layout exceeds the 35 KiB directive target");

// SIMT launch geometry: 32 rows x 16 lanes.
inline constexpr std::uint32_t kSimtThreads = kRowsPerTile * kLanesPerRow;  // 512

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;

}  // namespace moe::kernels::arch35::mxfp4_expert_gemv

#endif  // MOE_KERNELS_ARCH35_MXFP4_EXPERT_GEMV_UB_LAYOUT_HPP
