// paged_compressor_rope_ub_layout.hpp -- unified-buffer layout and event plan for paged_compressor_rope
// fused_paged_compressor_rope (single AIV, hybrid SIMD/SIMT, coff=2 C4A).
//
// UB REGION SPLIT -- this is load-bearing, not bookkeeping.
//
// Both SIMT VFs (CompressorProjectSimt, CompressorEmitSimt) dereference UB
// pointers, and a buffer a SIMT VF touches must live in the DYNAMIC region
// whose size the host declares as the second chevron argument. TPipe/TBuf
// allocates from the STATIC region, so the two BF16 staging buffers below --
// the only ones no SIMT VF ever sees -- are the only TBufs this kernel has.
// Everything else comes from LocalMemAllocator, and the byte total is derived
// once in op_host/paged_compressor_rope_tiling.h so host and device cannot
// disagree about it.
//
// DYNAMIC segments (LocalMemAllocator; allocation order IS the layout, and it
// must match the order documented in the tiling header):
//
//   offset   size    segment
//   0        16,384  kvCur        current-group KV state [4,1024] FP32
//   16,384   16,384  gateCur      current-group gate state [4,1024] FP32
//   32,768   8,192   prevKvA      previous group a-KV halves [4,512] FP32
//   40,960   8,192   prevGateA    previous group a-gate halves [4,512] FP32
//   49,152   16,384  apeF32       all four ape rows [4,1024] FP32 (resident)
//   65,536   2,048   normWF32     widened norm gain [512]
//   67,584   256     ropeCos      cos[64] FP32
//   67,840   256     ropeSin      sin[64] FP32
//   68,096   2,048   outF32       compressed token [512] FP32
//   70,144   256     reduceSlack  16 warp partials + block inv-rms scalar
//
//   dynamic total: 70,400 B
//
// STATIC segments (TPipe/TBuf, SIMD-private):
//
//   1,024  normWBf16  BF16 norm gain [512], cast into normWF32
//   1,024  outBf16    BF16 [512] cache-slot token, MTE3 source
//
//   static total: 2,048 B
//
// Total: 72,448 B -- inside the contract's padded emit-phase figure of
// 84,992 B and far below kUsableUbBytes (221,184 B). A `scalarState` segment
// of 768 B used to sit at the end of this table; the TBuf existed but nothing
// ever read or wrote it, so it is gone rather than carried as a static-region
// allocation this kernel does not use. The four state regions
// mirror the GM ring-state ABI one-to-one (kv [0,16K), gate [16K,32K),
// prev-a kv [32K,40K), prev-a gate [40K,48K) in GM), so the state crosses
// the boundary as four whole-region MTE2/MTE3 copies. Weight rows are NOT
// staged: the SIMT projection streams wkv/wgate rows from GM through the
// SIMT DataCache and decodes BF16 by bit shift in registers, re-reading x
// (8 KiB) out of the DataCache for each of the 2,048 row dots.
//
// The emit phase is one SIMT launch of 512 threads, one feature lane per
// thread: the eight window scores/KV values live in registers, the column
// softmax / weighted reduction / RMSNorm block-reduce / half-mode RoPE on
// lanes [448,512) / the previous-a roll all run without touching GM, and
// only the finished token lands in outF32. The softmax flushes FP32
// denormals to zero explicitly, matching the oracle's NPU exp emulation.
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    state regions / ape / norm / rope arrivals
//   EVENT_ID1  V    -> MTE3 outBf16 cache-slot store readiness
//   EVENT_ID2  MTE2 -> MTE3 WAR guard: state MTE2 traffic drained before
//                           the state writeback may issue
//   EVENT_ID3  MTE3 -> MTE2 reload guard (directive): the state MTE2 reload
//                           is ordered after any in-flight MTE3 store of the
//                           same buffer on this core
// Phase boundaries around each SIMT launch use DataSyncBarrier<ALL>.
//
// The persistent GM state is 49,168 B (state + meta), an owned
// input/output allocation, NOT scratch: GM scratch is 0 B.

#ifndef MOE_KERNELS_ARCH35_PAGED_COMPRESSOR_ROPE_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_PAGED_COMPRESSOR_ROPE_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"
// Carries kCompressorDynUbBytes, the single formula the host also uses for
// the chevron's dyn_ub_size; the asserts below tie this table to it.
#include "paged_compressor_rope_tiling.h"

namespace moe::kernels::arch35::paged_compressor_rope {

inline constexpr std::uint32_t kHidden = 4096;
inline constexpr std::uint32_t kHeadDim = 512;
inline constexpr std::uint32_t kCoff = 2;
inline constexpr std::uint32_t kRatio = 4;
inline constexpr std::uint32_t kStateWidth = kCoff * kHeadDim;    // 1024
inline constexpr std::uint32_t kReduceRows = kCoff * kRatio;      // 8
inline constexpr std::uint32_t kRopeHeadDim = 64;
inline constexpr std::uint32_t kRopeBaseOffset = kHeadDim - kRopeHeadDim;  // 448

// GM state ABI (FP32 elements), mirrored by the four UB regions.
inline constexpr std::uint32_t kStateFloats = 3 * kRatio * kStateWidth;    // 12,288
inline constexpr std::uint32_t kKvCurRegionFloats = kRatio * kStateWidth;  // 4,096
inline constexpr std::uint32_t kGateCurRegionFloats = kRatio * kStateWidth;
inline constexpr std::uint32_t kPrevARegionFloats = kRatio * kHeadDim;     // 2,048
static_assert(2 * kKvCurRegionFloats + 2 * kPrevARegionFloats == kStateFloats,
              "paged_compressor_rope state ABI regions must tile the 12,288-float buffer");

inline constexpr std::uint32_t kApeBytes = kRatio * kStateWidth * 4;        // 16,384
inline constexpr std::uint32_t kOutF32Bytes = kHeadDim * 4;                 // 2,048
inline constexpr std::uint32_t kNormWBytes = kHeadDim * 2;                  // 1,024
inline constexpr std::uint32_t kNormWF32Bytes = kHeadDim * 4;               // 2,048
inline constexpr std::uint32_t kRopeTableBytes = 2 * kRopeHeadDim * 4;      // 512
inline constexpr std::uint32_t kOutBf16Bytes = kHeadDim * 2;                // 1,024
inline constexpr std::uint32_t kReduceSlackFloats = kCompressorReduceSlackFloats;
inline constexpr std::uint32_t kReduceSlackBytes = kReduceSlackFloats * 4;  // 256

// Dynamic region: every segment a SIMT VF dereferences.
inline constexpr std::uint32_t kDynamicUbBytes =
    2 * kKvCurRegionFloats * 4 + 2 * kPrevARegionFloats * 4 + kApeBytes +
    kNormWF32Bytes + kRopeTableBytes + kOutF32Bytes + kReduceSlackBytes;

// Static region: the two SIMD-private BF16 staging buffers.
inline constexpr std::uint32_t kStaticUbBytes = kNormWBytes + kOutBf16Bytes;

inline constexpr std::uint32_t kTotalUbBytes = kStaticUbBytes + kDynamicUbBytes;

// The contract's padded emit-phase figure (83,968 B + 1 KiB); see above.
inline constexpr std::uint32_t kContractUbBudgetBytes = 84992;

static_assert(kDynamicUbBytes == 70400,
              "paged_compressor_rope dynamic UB must sum to 70,400 B");
static_assert(kStaticUbBytes == 2048,
              "paged_compressor_rope static UB must sum to 2,048 B");
static_assert(kTotalUbBytes == 72448,
              "paged_compressor_rope UB layout must sum to 72,448 B");

// The host declares dyn_ub_size from the tiling header's constant, so a
// segment added here without updating that formula would under-declare the
// dynamic region and reintroduce the out-of-bounds SIMT access this split
// exists to prevent. Tie the two together at compile time.
static_assert(kDynamicUbBytes == kCompressorDynUbBytes,
              "paged_compressor_rope dynamic UB layout disagrees with the "
              "dyn_ub_size formula the host launches with");
static_assert(kStaticUbBytes == kCompressorStaticUbBytes,
              "paged_compressor_rope static UB layout disagrees with the "
              "tiling header's static-region figure");

static_assert(kTotalUbBytes <= kContractUbBudgetBytes,
              "paged_compressor_rope UB layout exceeds the contract emit-phase budget");
static_assert(kTotalUbBytes <= kUsableUbBytes,
              "paged_compressor_rope UB layout exceeds the arch35 usable UB ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kTotalUbBytes >=
                  kUbDataCacheMinBytes,
              "paged_compressor_rope UB layout starves the SIMT DataCache below 32 KiB");

// SIMT launch geometry: one thread per feature lane.
inline constexpr std::uint32_t kSimtThreads = kHeadDim;  // 512

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;
inline constexpr std::int32_t kEvMte2ToMte3 = 2;
inline constexpr std::int32_t kEvMte3ToMte2 = 3;

}  // namespace moe::kernels::arch35::paged_compressor_rope

#endif  // MOE_KERNELS_ARCH35_PAGED_COMPRESSOR_ROPE_UB_LAYOUT_HPP
