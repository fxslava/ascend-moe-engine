// dsv4_moe_block_arch35.h -- UB layout and event plan for the two whole-block
// entry points (Ascend 950PR, arch35 / dav-3510, hybrid SIMD/SIMT).
//
//   dsv4_moe_decode_topk      M = 1,  one AIV per routed expert slot
//   dsv4_moe_prefill_batched  M >> 1, 32 AIVs grid-striding over tokens
//
// UB REGION: all DYNAMIC, static is 0 B, for both entry points. Every buffer
// here is addressed by a SIMT VF, so it comes from the dynamic region through
// LocalMemAllocator rather than from TPipe/TBuf -- the dynamic region is the
// one whose size the host declares as dyn_ub_size in the chevron, and a TBuf
// lives in the static region, which the launch never declares. Handing a SIMT
// VF a pointer into an undeclared region is the error-341 / 507035 class.
//
// WHY ONE ENTRY POINT EACH, AND WHY `activated` NEVER REACHES GM
// The retired dsv4_moe_expert split one expert across 32 cores by row tile,
// which made `activated` a cross-core operand: `down` reduces over all of it
// while `gate_up` produced it tiled. That forced two dispatches, a GM
// workspace and stream ordering as the barrier.
//
// Both kernels here instead give ONE core a whole expert-token pair, so the
// gate/up legs and the down leg run back to back on the same core and
// `activated` is a UB buffer with no sharer. Decode takes its parallelism
// from the top_k expert slots; prefill from the token batch. Neither needs a
// barrier and neither writes `activated` to GM.
//
// Segment tables (per core), in LocalMemAllocator allocation order -- the
// order IS the layout. Production geometry, hidden = 4,096, inter = 2,048:
//
//     offset  size     segment
//     0       16,384   xF32     resident FP32 x[hidden] for this token
//     16,384  8,192    actF32   `activated`[inter], gate/up -> down
//     24,576  128      tile[0]  32 FP32 results staged for one MTE3
//     24,704  32       skew     breaks the 256 B UB bank alias (kTileSkewBytes)
//     24,736  128      tile[1]  the ping-pong partner
//     dynamic total 24,864 B  of the 221,184 B ceiling
//
// The skew is a real allocation, not padding to be reclaimed: back-to-back
// 128 B slots would give a 256 B stride, the bank period. Allocating it as a
// segment is what keeps LocalMemAllocator's order == the layout.
//
// Packed nibbles and E8M0 scales are NOT staged in UB: SIMT lanes read them
// from GM through the SIMT DataCache with a 16 B uint4 asc_ldca per scale
// group, decode OCP E2M1 branch-free in registers and form the E8M0 scale
// with ldexpf. The coalesced group-to-lane map that makes those loads one
// contiguous 256 B run per warp step is documented in the kernel.
//
// MAC order matches the oracle (src/emu/dsv4_quant_gemv.cpp::mxfp4_gemv_cpu):
// each lane accumulates its 16 pairs (32 logical columns, one scale group)
// into a register partial, then applies the group scale once, so the
// per-group scale-after-subtotal grouping is preserved. The 16 lane partials
// of each row reduce through an in-register butterfly (xor masks 1,2,4,8
// within the aligned 16-lane row group).
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    x arrival (once per token)
//   EVENT_ID1  V    -> MTE3 tile store readiness (once per 32-row tile)
//   EVENT_ID2  MTE3 -> V    tile buffer reuse after its store
// Phase boundaries around each SIMT launch use DataSyncBarrier<ALL>.
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under the
// CAModel only. No claim is made here about hardware behaviour (§5.7).

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_BLOCK_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_BLOCK_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::dsv4_moe_block {

inline constexpr std::uint32_t kRowsPerTile = 32;  // one tile per SIMT launch
inline constexpr std::uint32_t kLanesPerRow = 16;  // 512 threads / 32 rows
inline constexpr std::uint32_t kGroupSize = 32;    // logical cols per E8M0

// One scale group is 32 logical columns = 16 MXFP4 pairs = 16 packed bytes,
// which is exactly one `uint4` -- so a group's weights are ONE asc_ldca, and
// a lane's load quantum equals its stride between lanes.
inline constexpr std::uint32_t kGroupBytes = kGroupSize / 2;  // 16
static_assert(kGroupBytes == 16,
              "a scale group must be exactly one uint4 of packed nibbles");

// Groups consumed per warp step: lane l takes group s*16 + l, so one step of
// the 16 lanes spans 16 consecutive groups = 256 contiguous bytes.
inline constexpr std::uint32_t kGroupsPerWarpStep = kLanesPerRow;
inline constexpr std::uint32_t kWarpStepBytes = kGroupsPerWarpStep * kGroupBytes;
static_assert(kWarpStepBytes == 256,
              "one warp step must span 256 contiguous packed bytes");

inline constexpr std::uint32_t kHiddenMax = 4096;
inline constexpr std::uint32_t kInterMax = 2048;

inline constexpr std::uint32_t kXF32BytesMax = kHiddenMax * 4;   // 16,384
inline constexpr std::uint32_t kActF32BytesMax = kInterMax * 4;  // 8,192
inline constexpr std::uint32_t kTileBytes = kRowsPerTile * 4;    // 128

// --- the down-phase tile ring ----------------------------------------------
//
// Phase 2 double-buffers the output tile: the VF for one slot is published and
// stored while the other slot holds the previous tile's result, so the loop
// unrolls by kTileSlots and pays ONE DataSyncBarrier per unrolled pair rather
// than one per tile.
//
// WHY ONLY TWO SLOTS, AND WHAT THAT COSTS
// With two slots the next pair's VFs cannot be issued before the current
// pair's MTE3 stores have retired: they would write the very slots MTE3 is
// still reading, and that WAR hazard is not covered by any fence until
// MTE3_V. So two slots buy a HALVED barrier and event count, not compute/DMA
// overlap across the pair. Overlapping a VF with the store of a different
// tile needs a third slot, which is a separate change and a separate proof.
inline constexpr std::uint32_t kTileSlots = 2;

// 32 B of skew between slots. Two 128 B slots placed back to back have a
// 256 B stride, which is exactly the UB bank alias to avoid; 160 B is not a
// multiple of 256 B or 4 KiB. This is stride arithmetic, not a measured win --
// no claim is made here about observed bank-conflict behaviour, and nothing in
// this layout was ranked by a static analyzer (CLAUDE.md section 5.6).
inline constexpr std::uint32_t kTileSkewBytes = 32;
inline constexpr std::uint32_t kTileSlotStrideBytes = kTileBytes + kTileSkewBytes;
static_assert(kTileSlotStrideBytes % 256u != 0u,
              "the tile slot stride must not alias the 256 B UB bank period");
static_assert(kTileSlotStrideBytes % 4096u != 0u,
              "the tile slot stride must not alias the 4 KiB UB bank period");
static_assert(kTileSkewBytes % 32u == 0u,
              "the skew must keep every slot base 32 B aligned for MTE3");

// The region: kTileSlots payloads with a skew between (not after) them.
inline constexpr std::uint32_t kTileRegionBytes =
    kTileSlots * kTileBytes + (kTileSlots - 1u) * kTileSkewBytes;  // 288
static_assert(kTileRegionBytes == 288, "tile ring must occupy 288 B");

// Static UB is zero: neither entry point has a TPipe or a TBuf.
inline constexpr std::uint32_t kStaticUbBytes = 0;
inline constexpr std::uint32_t kDynUbBytesMax =
    kXF32BytesMax + kActF32BytesMax + kTileRegionBytes;

static_assert(kDynUbBytesMax == 24864,
              "dsv4_moe_block dynamic UB must sum to 24,864 B "
              "(16,384 x + 8,192 act + 288 tile ring)");
static_assert(kStaticUbBytes + kDynUbBytesMax <= kUsableUbBytes,
              "dsv4_moe_block UB layout exceeds the arch35 usable ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kStaticUbBytes -
                      kDynUbBytesMax >= kUbDataCacheMinBytes,
              "dsv4_moe_block UB layout starves the SIMT DataCache below "
              "32 KiB");

// SIMT launch geometry: 32 rows x 16 lanes.
inline constexpr std::uint32_t kSimtThreads = kRowsPerTile * kLanesPerRow;  // 512

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;
inline constexpr std::int32_t kEvMte3ToV = 2;

}  // namespace moe::kernels::arch35::dsv4_moe_block

#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_BLOCK_UB_LAYOUT_HPP
