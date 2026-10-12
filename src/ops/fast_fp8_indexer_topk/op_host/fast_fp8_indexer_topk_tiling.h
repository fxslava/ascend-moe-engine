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

// Every shape guard the v1 contract states, in one place. `seq_len == 0` is
// legal (the output is then all -1) and the key cache and page table are not
// read, so the block geometry is only constrained when there is something to
// score.
Status MakeIndexerTiling(std::uint32_t query_head_stride,
                         std::uint32_t key_block_stride,
                         std::uint32_t block_table_len,
                         std::uint32_t num_physical_blocks,
                         std::uint32_t block_size, std::uint32_t seq_len,
                         IndexerTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // FAST_FP8_INDEXER_TOPK_TILING_H
