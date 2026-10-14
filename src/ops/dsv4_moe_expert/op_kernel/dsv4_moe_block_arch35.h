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
//     24,576  128      tile     32 FP32 results staged for one MTE3
//     dynamic total 24,704 B  of the 221,184 B ceiling
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

// Static UB is zero: neither entry point has a TPipe or a TBuf.
inline constexpr std::uint32_t kStaticUbBytes = 0;
inline constexpr std::uint32_t kDynUbBytesMax =
    kXF32BytesMax + kActF32BytesMax + kTileBytes;

static_assert(kDynUbBytesMax == 24704,
              "dsv4_moe_block dynamic UB must sum to 24,704 B");
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
