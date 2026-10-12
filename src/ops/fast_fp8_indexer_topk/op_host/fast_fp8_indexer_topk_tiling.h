#ifndef FAST_FP8_INDEXER_TOPK_TILING_H
#define FAST_FP8_INDEXER_TOPK_TILING_H

// Tiling POD and v1 contract geometry for fast_fp8_indexer_topk.
// Included by both op_kernel/ and op_host/; no <acl/acl.h>.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstddef>
#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kIndexerHeads = 64;
inline constexpr std::uint32_t kIndexerHeadDim = 128;
inline constexpr std::uint32_t kIndexerTopK = 512;
inline constexpr std::uint32_t kIndexerScoreTile = 512;
inline constexpr std::uint32_t kIndexerMaxSeqLen = 2048;
// GM scratch alignment, and the alignment the tiling staging buffer uses.
inline constexpr std::size_t kIndexerGmAlignBytes = 32;

// ---------------------------------------------------------------------------
// The two-kernel pipeline
// ---------------------------------------------------------------------------
// Scoring and selection are launched as two kernels on one stream, because
// they want different grids. Scoring is warp-per-key and embarrassingly
// parallel across keys, so it runs on all 32 AIV cores; selection is a
// 2,048-slot bitonic network that is a total order and stays on one core.
// Hardware stream ordering makes the score kernel's GM workspace writes
// visible to the select kernel with no host-side synchronization between them.
//
// The grid size is a compile-time constant shared by host and device for the
// same reason the dyn_ub figures are: the device derives the key-distribution
// stride from it, so a host that launched a different grid would silently skip
// or double-score keys. Neither side may hardcode it separately.
inline constexpr std::uint32_t kIndexerScoreBlockDim = 32;  // all 32 AIVs
inline constexpr std::uint32_t kIndexerSelectBlockDim = 1;  // total order
inline constexpr std::uint32_t kIndexerWarpsPerCore = 16;   // 512 threads / 32

// Global warps across the score grid; the device's grid-stride step.
inline constexpr std::uint32_t kIndexerScoreWarpStride =
    kIndexerScoreBlockDim * kIndexerWarpsPerCore;  // 512

// Dynamic UB, per kernel. Both are SIMT-visible-only figures: every buffer
// below is dereferenced by a SIMT VF and therefore must come from the dynamic
// region, whose size the host declares as the second chevron argument.
//
// Splitting the kernels also splits this budget -- the score kernel never
// touches the sort buffer and the select kernel never touches the query -- so
// each declares only what it uses and leaves the rest to the DataCache.
inline constexpr std::uint32_t IndexerScoreDynamicUbBytes() noexcept {
  return (kIndexerHeads * kIndexerHeadDim) * sizeof(std::uint8_t) +  // query_fp8:   8,192 B
         (kIndexerHeads * kIndexerHeadDim) * sizeof(float) +         // query_f32:  32,768 B
         (kIndexerHeads) * sizeof(float);                            // head_scales:   256 B
}                                                                    // Total: 41,216 B

inline constexpr std::uint32_t IndexerSelectDynamicUbBytes() noexcept {
  return (kIndexerMaxSeqLen) * sizeof(std::uint32_t);                // sort_keys:   8,192 B
}                                                                    // Total:  8,192 B

// The fused single-core kernel needs both halves at once. It is currently
// unlaunched (see its file header); this figure exists so that it still
// compiles against one shared formula rather than its own.
inline constexpr std::uint32_t IndexerDynamicUbBytes() noexcept {
  return IndexerScoreDynamicUbBytes() + IndexerSelectDynamicUbBytes();
}                                                                    // Total: 49,408 B

static_assert(IndexerScoreDynamicUbBytes() == 41216u,
              "fast_fp8_indexer score dynamic UB must sum to 41,216 B");
static_assert(IndexerSelectDynamicUbBytes() == 8192u,
              "fast_fp8_indexer select dynamic UB must sum to 8,192 B");
static_assert(IndexerDynamicUbBytes() == 49408u,
              "fast_fp8_indexer fused dynamic UB must sum to 49,408 B");

// ---------------------------------------------------------------------------
// sizeof == 48
// ---------------------------------------------------------------------------
struct IndexerTilingData {
  std::uint32_t abi_version;        // 1
  std::uint32_t block_dim;          // 1 AIV, deterministic total order
  std::uint32_t heads;              // 64
  std::uint32_t head_dim;           // 128
  std::uint32_t top_k;              // 512
  std::uint32_t query_head_stride;  // 128 FP8 elements
  std::uint32_t key_block_stride;   // block_size*128 FP8 elements
  std::uint32_t block_size;         // 128 in v3 semi-E2E
  std::uint32_t seq_len;            // 2048 in v3 semi-E2E
  std::uint32_t block_table_len;    // >= ceil(seq_len/block_size)
  std::uint32_t num_physical_blocks;
  std::uint32_t score_tile;  // 512 FP32 elements
};

MOE_ARCH35_ASSERT_TILING_POD(IndexerTilingData, 48);

Status MakeIndexerTiling(std::uint32_t query_head_stride,
                         std::uint32_t key_block_stride,
                         std::uint32_t block_table_len,
                         std::uint32_t num_physical_blocks,
                         std::uint32_t block_size, std::uint32_t seq_len,
                         IndexerTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // FAST_FP8_INDEXER_TOPK_TILING_H