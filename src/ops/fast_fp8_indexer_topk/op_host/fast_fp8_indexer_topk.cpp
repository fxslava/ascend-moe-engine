// -----------------------------------------------------------------------------
// fast_fp8_indexer_topk.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "fast_fp8_indexer_topk_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void fast_fp8_indexer_topk(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR dequant_scales,
    GM_ADDR block_table, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

constexpr const char* kOp = "fast_fp8_indexer_topk";

}  // namespace

Status LaunchFp8IndexerTopK(
    const Fp8E4M3* query_gm, std::uint32_t query_head_stride,
    const Fp8E4M3* key_cache_gm, std::uint32_t key_block_stride,
    const float* dequant_scales_gm, const std::int32_t* block_table_gm,
    std::uint32_t block_table_len, std::uint32_t num_physical_blocks,
    std::uint32_t block_size, std::uint32_t seq_len, std::int32_t* indices_gm,
    void* workspace, std::size_t workspace_bytes, aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (query_gm == nullptr || dequant_scales_gm == nullptr ||
      indices_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  if (seq_len > 0 && (key_cache_gm == nullptr || block_table_gm == nullptr)) {
    return Status::invalid_argument;
  }
  const std::size_t required_ws = Fp8IndexerWorkspaceBytes(seq_len);
  if (required_ws > 0 &&
      (workspace == nullptr || workspace_bytes < required_ws)) {
    return Status::invalid_argument;
  }

  IndexerTilingData tiling{};
  const Status shape =
      MakeIndexerTiling(query_head_stride, key_block_stride, block_table_len,
                        num_physical_blocks, block_size, seq_len, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  const std::uint32_t dyn_ub_bytes = IndexerDynamicUbBytes();

  fast_fp8_indexer_topk<<<tiling.block_dim, dyn_ub_bytes, stream>>>(
      detail::Gm(query_gm), detail::Gm(key_cache_gm),
      detail::Gm(dequant_scales_gm), detail::Gm(block_table_gm),
      detail::Gm(indices_gm), detail::Gm(workspace),
      detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35