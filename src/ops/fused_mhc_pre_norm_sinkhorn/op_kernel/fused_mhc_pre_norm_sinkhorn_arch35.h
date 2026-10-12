// fused_mhc_pre_norm_sinkhorn_ub_layout.hpp -- unified-buffer layout and event plan for fused_mhc_pre_norm_sinkhorn
// fused_mhc_pre_norm_sinkhorn (single AIV, hybrid SIMD/SIMT).
//
// Segment table (all offsets 512-B aligned, distinct bank offsets; every
// segment size is a multiple of 32 B):
//
//   offset     size       segment
//   0          8,192      xStreamTile   one BF16 [4096] stream staging tile
//   8,192      65,536     xF32[4]       widened FP32 bundle, 4 x 16,384 B
//   73,728     32,768     phiQueue      2 x 16,384 B FP32 [4096] tiles
//   106,496    16,384     product       elementwise product vector
//   122,880    16,384     hRaw          pre-gated stream reduction [4096]
//   139,264    16,384     norm          activation-norm workspace [4096]
//   155,648    8,192      gammaTile     BF16 gamma [4096]
//   163,840    8,192      normOutTile   BF16 h_norm [4096]
//   172,032    2,048      reduceScratch ReduceSum shared temp
//   174,080    256        scalarState   24 dots + 16 logits + pre/post
//   174,336    8,192      (tail slack, unallocated)
//
// Total allocated: 174,336 B <= the contract's v1 budget of 182,464 B and
// <= kUsableUbBytes (221,184 B). After the 8 KiB runtime reserve, the SIMT
// DataCache gets min(256 KiB - 174,336 - 8 KiB, 128 KiB)
// = min(79,616, 131,072) = 79,616 B >= 32 KiB.
// (An earlier revision of this comment said 75,264 B, which was an
// arithmetic slip: 262,144 - 174,336 - 8,192 is 79,616. The static_assert
// below was always computing the correct value; only the prose was wrong.)
//
// Event plan (EVENT_ID0..3 only; arch35 has no EVENT_ID4..7). The two phi
// tiles alternate but are driven one-at-a-time (issue, wait, consume), so
// each dependency direction needs exactly one channel:
//   EVENT_ID0  MTE2 -> V    x stream tile, phi tile arrivals, gamma tile
//   EVENT_ID1  V    -> MTE2 WAR guard before reissuing a reused tile
//   EVENT_ID2  V    -> S    scalar read of ReduceSum results
//   EVENT_ID3  V    -> MTE3 h_raw / h_norm store readiness
// Phase boundaries between the SIMD phases and the SIMT Sinkhorn launch use
// DataSyncBarrier<MemDsbT::ALL>() (the asc_vf_call executes inline in the
// vector instruction stream on __DAV_VEC__).

#ifndef MOE_KERNELS_ARCH35_FUSED_MHC_PRE_NORM_SINKHORN_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_FUSED_MHC_PRE_NORM_SINKHORN_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::fused_mhc_pre_norm_sinkhorn {

inline constexpr std::uint32_t kStreams = 4;
inline constexpr std::uint32_t kHidden = 4096;
inline constexpr std::uint32_t kFlat = kStreams * kHidden;         // 16,384
inline constexpr std::uint32_t kProjectionRows = 24;
inline constexpr std::uint32_t kProjectionTile = 4096;             // FP32 elems
inline constexpr std::uint32_t kProjectionChunks = kFlat / kProjectionTile;
inline constexpr std::uint32_t kQueueDepth = 2;
inline constexpr std::uint32_t kSinkhornIters = 20;

// Segment sizes in bytes.
inline constexpr std::uint32_t kXStreamTileBytes = kHidden * 2;             // 8,192
inline constexpr std::uint32_t kXF32StreamBytes = kHidden * 4;              // 16,384
inline constexpr std::uint32_t kPhiTileBytes = kProjectionTile * 4;         // 16,384
inline constexpr std::uint32_t kProductBytes = kHidden * 4;                 // 16,384
inline constexpr std::uint32_t kHRawBytes = kHidden * 4;                    // 16,384
inline constexpr std::uint32_t kNormBytes = kHidden * 4;                    // 16,384
inline constexpr std::uint32_t kGammaTileBytes = kHidden * 2;               // 8,192
inline constexpr std::uint32_t kNormOutTileBytes = kHidden * 2;             // 8,192
inline constexpr std::uint32_t kReduceScratchBytes = 512 * 4;               // 2,048
inline constexpr std::uint32_t kScalarStateBytes = 256;

inline constexpr std::uint32_t kTotalUbBytes =
    kXStreamTileBytes + kStreams * kXF32StreamBytes +
    kQueueDepth * kPhiTileBytes + kProductBytes + kHRawBytes + kNormBytes +
    kGammaTileBytes + kNormOutTileBytes + kReduceScratchBytes + kScalarStateBytes;

// The contract's correctness-first v1 budget for this kernel.
inline constexpr std::uint32_t kContractUbBudgetBytes = 182464;

static_assert(kTotalUbBytes == 174'336, "fused_mhc_pre_norm_sinkhorn UB layout must sum to 174,336 B");
static_assert(kTotalUbBytes <= kContractUbBudgetBytes,
              "fused_mhc_pre_norm_sinkhorn UB layout exceeds the contract v1 budget");
static_assert(kTotalUbBytes <= kUsableUbBytes,
              "fused_mhc_pre_norm_sinkhorn UB layout exceeds the arch35 usable UB ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kTotalUbBytes >=
                  kUbDataCacheMinBytes,
              "fused_mhc_pre_norm_sinkhorn UB layout starves the SIMT DataCache below 32 KiB");

// Event IDs (arch35: only EVENT_ID0..EVENT_ID3 exist).
inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte2 = 1;
inline constexpr std::int32_t kEvVToS = 2;
inline constexpr std::int32_t kEvVToMte3 = 3;

}  // namespace moe::kernels::arch35::fused_mhc_pre_norm_sinkhorn

#endif  // MOE_KERNELS_ARCH35_FUSED_MHC_PRE_NORM_SINKHORN_UB_LAYOUT_HPP
