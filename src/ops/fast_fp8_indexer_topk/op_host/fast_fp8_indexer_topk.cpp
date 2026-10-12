// -----------------------------------------------------------------------------
// fast_fp8_indexer_topk.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "fast_fp8_indexer_topk_tiling.h"

namespace moe::kernels::arch35 {
namespace {

// Stage 1: scoring, kIndexerScoreBlockDim cores. Stage 2: selection, 1 core.
extern "C" __global__ __aicore__ void fast_fp8_indexer_score(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR dequant_scales,
    GM_ADDR block_table, GM_ADDR workspace, GM_ADDR tiling);

extern "C" __global__ __aicore__ void fast_fp8_indexer_select(
    GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

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

  // ONE staging buffer for BOTH launches. The two kernels read the same POD,
  // and the staging allocation must outlive both -- TilingStaging's teardown
  // synchronizes the stream before freeing (CLAUDE.md §3.7), so staging it
  // once here is what keeps the device-side pointer valid across the pair.
  // Staging it twice would be two mallocs and two syncs for identical bytes.
  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // Both launches go on the SAME stream with no host-side synchronization
  // between them: ACL stream ordering guarantees the score kernel has
  // completed, and its asc_stcg stores drained to GM, before the select
  // kernel starts reading the workspace. Inserting a sync here would serialize
  // the host against the device for no ordering benefit.
  //
  // The grids differ, which is the whole point of the split -- scoring is
  // key-independent and takes all 32 AIVs, selection is a total order and
  // stays on one. tiling.block_dim is the SELECT stage's grid; the score
  // grid is the shared constant both sides derive the key-distribution stride
  // from, so it is passed from the header and never spelled out here.
  fast_fp8_indexer_score<<<kIndexerScoreBlockDim,
                           IndexerScoreDynamicUbBytes(), stream>>>(
      detail::Gm(query_gm), detail::Gm(key_cache_gm),
      detail::Gm(dequant_scales_gm), detail::Gm(block_table_gm),
      detail::Gm(workspace), detail::Gm(staging.get()));

  fast_fp8_indexer_select<<<tiling.block_dim, IndexerSelectDynamicUbBytes(),
                            stream>>>(
      detail::Gm(indices_gm), detail::Gm(workspace),
      detail::Gm(staging.get()));

  // A chevron launch returns void, so neither launch reports anything at the
  // call site. FinishLaunch is what turns a failure in EITHER of them into an
  // observable Status plus a recorded LastLaunchFailure() (CLAUDE.md §3.5) --
  // it synchronizes the stream, so it covers the pair, not just the last one.
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
