// fast_fp8_indexer_topk_ub_layout.hpp -- unified-buffer layout and event plan for fast_fp8_indexer_topk
// fused_fp8_indexer_topk (single AIV, hybrid SIMD/SIMT, deterministic order).
//
// Segment table:
//
//   offset   size    segment
//   0        8,192   queryFp8     E4M3 query [64,128] MTE2 landing tile
//   8,192    32,768  queryF32     dequantized query [64,128] (decoded once)
//   40,960   256     headScales  qw[64] FP32 head scales
//   41,216   8,192   sortKeys    2,048 uint32 selection composites
//
// Total: 49,408 B <= kUsableUbBytes (221,184 B). A `scalarState` segment of
// 256 B used to close this table, but Process() never allocated it and the
// host's dyn_ub_size formula never counted it, so the table was 256 B larger
// than the region the kernel actually uses; it is dropped rather than carried.
// The contract's v1 cap of
// 44,064 B assumed a vector-scoring shape (FP8 query 8,192 + widened query
// 32,768 + FP8 key tile 256 + widened key 512 + head partials 256 + score
// scan 2,048 + scalar 32); this kernel scores warp-per-key straight out of
// the GM page cache through the SIMT DataCache, so the key tiles disappear
// and the 8,192 B bitonic key buffer replaces the 2,048 B score scan. Both
// bounds hold the same ceiling. DataCache after reserve: 194,048 B.
//
// Scoring: one warp per key, lane pair {l, l+32} covers the 64 heads; the
// per-key score is fully warp-local (butterfly masks 1..16, no cross-warp
// traffic). FP32 scores are written to the GM workspace, then re-read and
// rounded through BF16 exactly as the contract's v1 prescribes, so the
// selection input is a stable GM copy.
//
// Selection: bitonic sort of the 2,048 uint32 composites descending (66
// uniform stages, asc_syncthreads between stages, no divergence). A larger
// composite = higher BF16 score, then LOWER logical index; padding keys are
// exactly 0 and can never collide with a real key (inverted index != 0).
// Output lanes [min(top_k, seq_len), top_k) are -1.
//
// Event plan (EVENT_ID0..3 only):
//   EVENT_ID0  MTE2 -> V    query tile arrival
//   EVENT_ID1  V    -> MTE3 indices store readiness
// SIMT-internal ordering uses asc_syncthreads / DataSyncBarrier.

#ifndef MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_TOPK_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_TOPK_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"
// Carries the score/select grid sizes and the per-kernel dyn_ub_size formulas
// the host launches with; the asserts below tie this table to them.
#include "fast_fp8_indexer_topk_tiling.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

inline constexpr std::uint32_t kHeads = 64;
inline constexpr std::uint32_t kHeadDim = 128;
inline constexpr std::uint32_t kTopK = 512;
inline constexpr std::uint32_t kMaxSeqLen = 2048;  // v1 UB cap; host rejects more
inline constexpr std::uint32_t kSortLength = kMaxSeqLen;

inline constexpr std::uint32_t kQueryFp8Bytes = kHeads * kHeadDim;      // 8,192
inline constexpr std::uint32_t kQueryF32Bytes = kHeads * kHeadDim * 4;  // 32,768
inline constexpr std::uint32_t kHeadScalesBytes = kHeads * 4;           // 256
inline constexpr std::uint32_t kSortKeysBytes = kSortLength * 4;        // 8,192

inline constexpr std::uint32_t kTotalUbBytes =
    kQueryFp8Bytes + kQueryF32Bytes + kHeadScalesBytes + kSortKeysBytes;

// The contract's v1 UB cap (vector-scoring shape); see the header comment.
inline constexpr std::uint32_t kContractUbCapBytes = 44064;

static_assert(kTotalUbBytes == 49408, "fast_fp8_indexer_topk UB layout must sum to 49,408 B");
static_assert(kTotalUbBytes == IndexerDynamicUbBytes(),
              "the fused kernel's UB layout disagrees with the dyn_ub_size "
              "the host launches it with");
static_assert(kTotalUbBytes <= kUsableUbBytes,
              "fast_fp8_indexer_topk UB layout exceeds the arch35 usable UB ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kTotalUbBytes >=
                  kUbDataCacheMinBytes,
              "fast_fp8_indexer_topk UB layout starves the SIMT DataCache below 32 KiB");

// SIMT launch geometry.
inline constexpr std::uint32_t kSimtThreads = 512;  // 16 warps
inline constexpr std::uint32_t kWarpSize = 32;

// ---------------------------------------------------------------------------
// The two-kernel split
// ---------------------------------------------------------------------------
// Scoring runs on all 32 AIVs, selection on one. The segment table above is
// the FUSED kernel's layout; the split kernels each take a prefix of it:
//
//   fast_fp8_indexer_score   queryFp8 + queryF32 + headScales   41,216 B
//   fast_fp8_indexer_select  sortKeys                            8,192 B
//
// The grid sizes and those two byte counts live in the tiling header, which
// both this file and the host include, so neither side can compute a
// different one. The stride below is the device's half of the key
// distribution: a core's warp w takes global warp id
// kScoreBlockDim-many-cores * 16 warps apart, i.e.
//
//   j = (block_idx * kWarpsPerCore + warp) + n * kScoreWarpStride
//
// and because kScoreWarpStride == kScoreBlockDim * kWarpsPerCore, the global
// warp ids cover [0, kScoreWarpStride) exactly once -- which is what makes the
// union of the per-warp key sequences a partition of [0, seq_len).
inline constexpr std::uint32_t kWarpsPerCore = kSimtThreads / kWarpSize;  // 16
inline constexpr std::uint32_t kScoreBlockDim = kIndexerScoreBlockDim;    // 32
inline constexpr std::uint32_t kScoreWarpStride = kScoreBlockDim * kWarpsPerCore;

static_assert(kWarpsPerCore == kIndexerWarpsPerCore,
              "the device's warps-per-core must match the host's figure, or "
              "the key-distribution stride is wrong on one side");
static_assert(kScoreWarpStride == kIndexerScoreWarpStride,
              "the device's grid-stride step must match the host's figure");

// Per-kernel UB totals, each asserted against the host's dyn_ub_size formula.
inline constexpr std::uint32_t kScoreUbBytes =
    kQueryFp8Bytes + kQueryF32Bytes + kHeadScalesBytes;  // 41,216
inline constexpr std::uint32_t kSelectUbBytes = kSortKeysBytes;  // 8,192

static_assert(kScoreUbBytes == IndexerScoreDynamicUbBytes(),
              "the score kernel's UB layout disagrees with the dyn_ub_size "
              "the host launches it with");
static_assert(kSelectUbBytes == IndexerSelectDynamicUbBytes(),
              "the select kernel's UB layout disagrees with the dyn_ub_size "
              "the host launches it with");
static_assert(kScoreUbBytes <= kUsableUbBytes && kSelectUbBytes <= kUsableUbBytes,
              "a split kernel exceeds the arch35 usable UB ceiling");

// Register blocking for the selection network: thread t holds sequence
// indices [t*kKeysPerThread, (t+1)*kKeysPerThread) in registers. This is what
// lets SelectTopKSimt split the 66 bitonic levels into a register tier
// (jj < kKeysPerThread), a warp-shuffle tier
// (kKeysPerThread <= jj < kKeysPerThread*kWarpSize) and a cross-warp UB tier.
inline constexpr std::uint32_t kKeysPerThread = kSortLength / kSimtThreads;

// The tiering above is only valid under these four properties, so they are
// asserted rather than assumed:
static_assert(kSortLength == kSimtThreads * kKeysPerThread,
              "every sort slot must be owned by exactly one thread");
static_assert(kKeysPerThread >= 2u && (kKeysPerThread &
                                       (kKeysPerThread - 1u)) == 0u,
              "kKeysPerThread must be a power of two >= 2, or the register "
              "tier's r ^ jj partner indexing is not a permutation");
static_assert((kSortLength & (kSortLength - 1u)) == 0u,
              "a bitonic network requires a power-of-two length");
static_assert(kSimtThreads % kWarpSize == 0u,
              "the warp-shuffle tier assumes whole warps");
// asc_shfl_xor takes lane_mask in [0, 32) and it must be < width. The largest
// mask the warp tier emits is (kKeysPerThread*kWarpSize/2)/kKeysPerThread.
static_assert(kWarpSize / 2u < kWarpSize,
              "the largest warp-tier lane_mask must stay below warpSize");

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk

#endif  // MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_TOPK_UB_LAYOUT_HPP
