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
//   49,408   256     scalarState tie-break / count scalars
//
// Total: 49,664 B <= kUsableUbBytes (221,184 B). The contract's v1 cap of
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
inline constexpr std::uint32_t kScalarStateBytes = 256;

inline constexpr std::uint32_t kTotalUbBytes =
    kQueryFp8Bytes + kQueryF32Bytes + kHeadScalesBytes + kSortKeysBytes +
    kScalarStateBytes;

// The contract's v1 UB cap (vector-scoring shape); see the header comment.
inline constexpr std::uint32_t kContractUbCapBytes = 44064;

static_assert(kTotalUbBytes == 49664, "fast_fp8_indexer_topk UB layout must sum to 49,664 B");
static_assert(kTotalUbBytes <= kUsableUbBytes,
              "fast_fp8_indexer_topk UB layout exceeds the arch35 usable UB ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kTotalUbBytes >=
                  kUbDataCacheMinBytes,
              "fast_fp8_indexer_topk UB layout starves the SIMT DataCache below 32 KiB");

// SIMT launch geometry.
inline constexpr std::uint32_t kSimtThreads = 512;  // 16 warps
inline constexpr std::uint32_t kWarpSize = 32;

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
