// mxfp4_expert_gemv_ub_layout.hpp -- unified-buffer layout and event plan for mxfp4_expert_gemv
// mxfp4 expert GEMV (32 AIVs, grid-stride over 32-row tiles, hybrid).
//
// UB REGION: all DYNAMIC, static is 0 B.
//
// Both buffers are addressed by the SIMT VF, so they are allocated from the
// dynamic UB region through LocalMemAllocator rather than from TPipe/TBuf.
// The dynamic region is the one whose size the host declares as dyn_ub_size
// in the chevron; the static region is never declared at launch. The byte
// count is derived once, in Mxfp4DynUbBytes() in the tiling header, because
// host and device must agree on it exactly.
//
// Segment table (per core; cols = tiling.cols, worst case 4,096), in
// LocalMemAllocator allocation order -- the order IS the layout:
//
//   offset          size (cols=4096)  segment
//   0               16,384            xF32   resident FP32 x[cols]
//   16,384          128               yTile  32 FP32 row results
//
// Dynamic total: 16,512 B at cols=4096 (8,320 B at cols=2048). Static: 0.
// DataCache = min(256 KiB - 0 - 16,512 - 8 KiB, 128 KiB) = 131,072 B.
//
// Two segments from the previous revision are gone, and both were fictional:
// `partials` (2,048 B) was counted in the total but no buffer was ever
// allocated for it -- the 16-lane row reduction is an in-register butterfly,
// as the note below already said -- and `scalarState` (128 B) was allocated
// but never read. The old total of 18,688 B therefore overstated a real
// footprint of 16,640 B by 2,048 B of buffer that did not exist.
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

inline constexpr std::uint32_t kXF32BytesMax = 4096 * 4;       // 16,384
inline constexpr std::uint32_t kYTileBytes = kRowsPerTile * 4;  // 128

// Static UB is zero: this kernel has no TPipe and no TBuf.
inline constexpr std::uint32_t kStaticUbBytes = 0;
// Worst-case dynamic footprint, at cols = 4,096.
inline constexpr std::uint32_t kDynUbBytesMax = kXF32BytesMax + kYTileBytes;

// The contract's padded v1 bound at cols=4096.
inline constexpr std::uint32_t kContractUbBudgetBytes = 19456;
// The directive's Stage-1 target footprint.
inline constexpr std::uint32_t kDirectiveUbTargetBytes = 35 * 1024;

static_assert(kDynUbBytesMax == 16512,
              "mxfp4_expert_gemv dynamic UB must sum to 16,512 B");
static_assert(kStaticUbBytes + kDynUbBytesMax <= kContractUbBudgetBytes,
              "mxfp4_expert_gemv UB layout exceeds the contract v1 budget");
static_assert(kStaticUbBytes + kDynUbBytesMax <= kDirectiveUbTargetBytes,
              "mxfp4_expert_gemv UB layout exceeds the 35 KiB directive target");
static_assert(kStaticUbBytes + kDynUbBytesMax <= kUsableUbBytes,
              "mxfp4_expert_gemv UB layout exceeds the arch35 usable ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kStaticUbBytes -
                      kDynUbBytesMax >= kUbDataCacheMinBytes,
              "mxfp4_expert_gemv UB layout starves the SIMT DataCache below "
              "32 KiB");

// SIMT launch geometry: 32 rows x 16 lanes.
inline constexpr std::uint32_t kSimtThreads = kRowsPerTile * kLanesPerRow;  // 512

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;

}  // namespace moe::kernels::arch35::mxfp4_expert_gemv

#endif  // MOE_KERNELS_ARCH35_MXFP4_EXPERT_GEMV_UB_LAYOUT_HPP
