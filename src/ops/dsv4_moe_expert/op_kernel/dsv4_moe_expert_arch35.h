// dsv4_moe_expert_arch35.h -- UB layout and event plan for dsv4_moe_expert,
// the fused DeepSeek-V4 routed-expert runner (32 AIVs, hybrid SIMD/SIMT).
//
// UB REGION: all DYNAMIC, static is 0 B, for both entry points. Every buffer
// here is addressed by a SIMT VF, so it comes from the dynamic region through
// LocalMemAllocator rather than from TPipe/TBuf: the dynamic region is the one
// whose size the host declares as dyn_ub_size in the chevron, and a TBuf lives
// in the static region, which the launch never declares. Handing a SIMT VF a
// pointer into an undeclared region is the error-341 / 507035 class --
// paged_compressor_rope faulted with aclError=507035 until it was given a real
// dyn_ub_size (see the note in mxfp4_expert_gemv.cpp).
//
// WHY TWO ENTRY POINTS
// The down leg reduces over the FULL `activated` vector, but the gate/up legs
// are split across 32 cores by row tile, so no single core holds all of
// `activated` when its own tiles are done. That is a cross-core dependency,
// and the cheapest correct barrier available to this launch layer is stream
// ordering: the wrapper dispatches gate_up, then down, on one stream.
// `activated` lives in GM workspace between them. fast_fp8_indexer_topk
// already establishes the multi-entry-point shape for one operator (one
// directory, one host ABI, one tiling POD).
//
// Segment tables (per core), in LocalMemAllocator allocation order -- the
// order IS the layout:
//
//   dsv4_moe_expert_gate_up    hidden = 4,096
//     offset  size     segment
//     0       16,384   xF32     resident FP32 x[hidden]
//     16,384  128      actTile  32 FP32 activated results
//     dynamic total 16,512 B
//
//   dsv4_moe_expert_down       inter = 2,048
//     offset  size     segment
//     0       8,192    actF32   resident FP32 activated[inter]
//     8,192   128      yTile    32 FP32 row results
//     dynamic total 8,320 B
//
// Packed nibbles and E8M0 scales are NOT staged in UB: SIMT lanes read them
// from GM through the SIMT DataCache (asc_ldca), decode OCP E2M1 branch-free
// in registers and form the E8M0 scale with ldexpf -- the same arrangement
// mxfp4_expert_gemv uses, and for the same reason.
//
// MAC order matches the oracle (src/emu/dsv4_quant_gemv.cpp::mxfp4_gemv_cpu):
// each lane accumulates its 16 pairs (32 logical columns, one scale group)
// into a register partial, then applies the group scale once, so the
// per-group scale-after-subtotal grouping is preserved. The 16 lane partials
// of each row reduce through an in-register butterfly (xor masks 1,2,4,8
// within the aligned 16-lane row group).
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    x / activated arrival (once per kernel)
//   EVENT_ID1  V    -> MTE3 tile store readiness (once per 32-row tile)
// Phase boundaries around each SIMT launch use DataSyncBarrier<ALL>.

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_EXPERT_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_EXPERT_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::dsv4_moe_expert {

inline constexpr std::uint32_t kRowsPerTile = 32;  // one tile per SIMT launch
inline constexpr std::uint32_t kLanesPerRow = 16;  // 512 threads / 32 rows
inline constexpr std::uint32_t kGroupSize = 32;

// One scale group is 32 logical columns = 16 MXFP4 pairs = 16 packed bytes,
// which is exactly one `uint4` -- so a group's weights are ONE asc_ldca, and
// a lane's load quantum equals its stride between lanes. That identity is
// what makes the warp's 16 group loads cover 256 contiguous bytes; see the
// coalescing note in the kernel.
inline constexpr std::uint32_t kGroupBytes = kGroupSize / 2;  // 16
static_assert(kGroupBytes == 16,
              "a scale group must be exactly one uint4 of packed nibbles");

// Groups consumed per warp step: lane l takes group s*16 + l, so one step of
// the 16 lanes spans 16 consecutive groups.
inline constexpr std::uint32_t kGroupsPerWarpStep = kLanesPerRow;
inline constexpr std::uint32_t kWarpStepBytes = kGroupsPerWarpStep * kGroupBytes;
static_assert(kWarpStepBytes == 256,
              "one warp step must span 256 contiguous packed bytes");

inline constexpr std::uint32_t kHiddenMax = 4096;
inline constexpr std::uint32_t kInterMax = 2048;

inline constexpr std::uint32_t kXF32BytesMax = kHiddenMax * 4;    // 16,384
inline constexpr std::uint32_t kActF32BytesMax = kInterMax * 4;   // 8,192
inline constexpr std::uint32_t kTileBytes = kRowsPerTile * 4;     // 128

// Static UB is zero: neither entry point has a TPipe or a TBuf.
inline constexpr std::uint32_t kStaticUbBytes = 0;
inline constexpr std::uint32_t kGateUpDynUbBytesMax = kXF32BytesMax + kTileBytes;
inline constexpr std::uint32_t kDownDynUbBytesMax = kActF32BytesMax + kTileBytes;
inline constexpr std::uint32_t kDynUbBytesMax =
    kGateUpDynUbBytesMax > kDownDynUbBytesMax ? kGateUpDynUbBytesMax
                                              : kDownDynUbBytesMax;

static_assert(kGateUpDynUbBytesMax == 16512,
              "dsv4_moe_expert gate_up dynamic UB must sum to 16,512 B");
static_assert(kDownDynUbBytesMax == 8320,
              "dsv4_moe_expert down dynamic UB must sum to 8,320 B");
static_assert(kStaticUbBytes + kDynUbBytesMax <= kUsableUbBytes,
              "dsv4_moe_expert UB layout exceeds the arch35 usable ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kStaticUbBytes -
                      kDynUbBytesMax >= kUbDataCacheMinBytes,
              "dsv4_moe_expert UB layout starves the SIMT DataCache below "
              "32 KiB");

// SIMT launch geometry: 32 rows x 16 lanes.
inline constexpr std::uint32_t kSimtThreads = kRowsPerTile * kLanesPerRow;  // 512

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;

}  // namespace moe::kernels::arch35::dsv4_moe_expert

#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_EXPERT_UB_LAYOUT_HPP
