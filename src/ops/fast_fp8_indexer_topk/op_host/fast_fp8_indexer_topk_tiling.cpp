#include "fast_fp8_indexer_topk_tiling.h"

namespace moe::kernels::arch35 {
namespace {

std::size_t AlignUp(std::size_t value, std::size_t align) noexcept {
  return (value + align - 1) / align * align;
}

}  // namespace

// Declared in moe/kernels/arch35/launch.hpp (the public API) but defined here,
// because the formula is this operator's geometry and nothing else's.
std::size_t Fp8IndexerWorkspaceBytes(std::uint32_t seq_len) noexcept {
  if (seq_len == 0) {
    return 0;
  }
  return AlignUp(static_cast<std::size_t>(seq_len) * 4, kIndexerGmAlignBytes);
}

Status MakeIndexerTiling(std::uint32_t query_head_stride,
                         std::uint32_t key_block_stride,
                         std::uint32_t block_table_len,
                         std::uint32_t num_physical_blocks,
                         std::uint32_t block_size, std::uint32_t seq_len,
                         IndexerTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (query_head_stride != kIndexerHeadDim) {
    return Status::unsupported_shape;
  }
  if (seq_len > kIndexerMaxSeqLen) {
    return Status::unsupported_shape;
  }
  // seq_len == 0 writes top_k slots of -1 and reads neither the key cache nor
  // the page table, so the block geometry is unconstrained in that case.
  if (seq_len > 0) {
    if (block_size == 0) {
      return Status::unsupported_shape;
    }
    // Widened before multiplying: block_size is caller-supplied, and
    // block_size * 128 overflows uint32 above 2^25.
    const std::uint64_t min_stride =
        static_cast<std::uint64_t>(block_size) * kIndexerHeadDim;
    if (static_cast<std::uint64_t>(key_block_stride) < min_stride) {
      return Status::unsupported_shape;
    }
    // "block_table_len >= ceil(seq_len/block_size)". Key j is read through
    // block_table[j / block_size], so a shorter table is an out-of-bounds
    // read the kernel cannot detect.
    const std::uint32_t needed = (seq_len + block_size - 1) / block_size;
    if (block_table_len < needed || num_physical_blocks == 0) {
      return Status::unsupported_shape;
    }
  }

  IndexerTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = 1;  // one core: a deterministic total order
  t.heads = kIndexerHeads;
  t.head_dim = kIndexerHeadDim;
  t.top_k = kIndexerTopK;
  t.query_head_stride = query_head_stride;
  t.key_block_stride = key_block_stride;
  t.block_size = block_size;
  t.seq_len = seq_len;
  t.block_table_len = block_table_len;
  t.num_physical_blocks = num_physical_blocks;
  t.score_tile = kIndexerScoreTile;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
