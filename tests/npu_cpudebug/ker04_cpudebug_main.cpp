// KER-04 numerical acceptance under CANN's CPU interpreter.
//
// v3 serializes no indexer output, so this compares against the oracle on a
// separate fixture, as the contract requires. Two things are checked beyond
// the scores: the selected index list must match EXACTLY (it is a ranked list
// of integers, not an approximation), and the ordering must survive ties,
// which is what the kernel's composite-key construction is for.

#include "tikicpulib.h"
#include "fixture_io.h"
#include "moe/kernels/arch35/kernel_contracts.h"

#include <string>
#include <vector>

extern "C" __global__ __aicore__ void ker04_fp8_indexer_topk(
    GM_ADDR query_gm, GM_ADDR key_cache_gm, GM_ADDR dequant_scales_gm,
    GM_ADDR block_table_gm, GM_ADDR indices_gm, GM_ADDR workspace_gm,
    GM_ADDR tiling_gm);

namespace mk = moe::kernels::arch35;
using namespace arch35_test;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ker04_cpudebug <fixture-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Verdict verdict("KER-04 fused_fp8_indexer_topk");

  constexpr uint32_t kHeads = mk::kIndexerHeads;
  constexpr uint32_t kHeadDim = mk::kIndexerHeadDim;
  constexpr uint32_t kTopK = mk::kIndexerTopK;
  const uint32_t block_size = 128;

  const std::vector<unsigned char> scores_raw = ReadBytes(dir + "/expect_scores.bin");
  const uint32_t seq_len = static_cast<uint32_t>(scores_raw.size() / sizeof(float));
  const uint32_t blocks = (seq_len + block_size - 1U) / block_size;
  std::printf("  seq_len=%u block_size=%u blocks=%u\n", seq_len, block_size, blocks);

  const std::vector<uint8_t> query =
      ReadTyped<uint8_t>(dir, "query", kHeads * kHeadDim);
  const std::vector<uint8_t> key_cache = ReadTyped<uint8_t>(
      dir, "key_cache", static_cast<size_t>(blocks) * block_size * kHeadDim);
  const std::vector<float> scales =
      ReadTyped<float>(dir, "dequant_scales", kHeads + seq_len);
  const std::vector<int32_t> block_table = ReadTyped<int32_t>(dir, "block_table", blocks);
  const std::vector<float> want_scores = ReadTyped<float>(dir, "expect_scores", seq_len);
  const std::vector<int32_t> want_indices =
      ReadTyped<int32_t>(dir, "expect_indices", kTopK);

  mk::IndexerTilingData tiling{};
  const mk::Status status =
      mk::BuildIndexerTiling(kHeadDim, block_size * kHeadDim, block_size, seq_len,
                             blocks, blocks, &tiling);
  verdict.CheckTrue("host tiler accepts the geometry", status == mk::Status::success);
  mk::IndexerTilingData reject{};
  verdict.CheckTrue("host tiler rejects a short block table",
                    mk::BuildIndexerTiling(kHeadDim, block_size * kHeadDim, block_size,
                                           seq_len, blocks - 1U, blocks, &reject) !=
                        mk::Status::success);

  const size_t workspace_bytes = mk::IndexerWorkspaceBytes(seq_len);
  std::printf("  GM workspace: %zu bytes (align_up(4*seq_len, 32))\n", workspace_bytes);

  auto* query_gm = (uint8_t*)AscendC::GmAlloc(query.size());
  auto* key_gm = (uint8_t*)AscendC::GmAlloc(key_cache.size());
  auto* scales_gm = (uint8_t*)AscendC::GmAlloc(scales.size() * sizeof(float));
  auto* table_gm = (uint8_t*)AscendC::GmAlloc(block_table.size() * sizeof(int32_t));
  auto* indices_gm = (uint8_t*)AscendC::GmAlloc(kTopK * sizeof(int32_t));
  auto* workspace_gm = (uint8_t*)AscendC::GmAlloc(workspace_bytes);
  auto* tiling_gm = (uint8_t*)AscendC::GmAlloc(sizeof(tiling));

  std::memcpy(query_gm, query.data(), query.size());
  std::memcpy(key_gm, key_cache.data(), key_cache.size());
  std::memcpy(scales_gm, scales.data(), scales.size() * sizeof(float));
  std::memcpy(table_gm, block_table.data(), block_table.size() * sizeof(int32_t));
  std::memcpy(tiling_gm, &tiling, sizeof(tiling));
  std::memset(indices_gm, 0xA5, kTopK * sizeof(int32_t));
  std::memset(workspace_gm, 0xA5, workspace_bytes);

  ICPU_RUN_KF(ker04_fp8_indexer_topk, tiling.block_dim, (GM_ADDR)query_gm,
              (GM_ADDR)key_gm, (GM_ADDR)scales_gm, (GM_ADDR)table_gm,
              (GM_ADDR)indices_gm, (GM_ADDR)workspace_gm, (GM_ADDR)tiling_gm);

  // The scores land in the GM workspace, which is the declared contract.
  const auto* got_scores = reinterpret_cast<const float*>(workspace_gm);
  verdict.CheckMetrics("scores vs oracle (FP32, in workspace)",
                       Compare(got_scores, want_scores.data(), seq_len));

  const auto* got_indices = reinterpret_cast<const int32_t*>(indices_gm);
  size_t index_mismatches = 0;
  int32_t first_bad = -1;
  for (uint32_t i = 0; i < kTopK; ++i) {
    if (got_indices[i] != want_indices[i]) {
      if (first_bad < 0) {
        first_bad = static_cast<int32_t>(i);
      }
      ++index_mismatches;
    }
  }
  std::printf("  %-34s %zu/%u ranks differ%s\n", "index list vs oracle",
              index_mismatches, kTopK, index_mismatches == 0 ? "" : " <-");
  if (index_mismatches != 0) {
    std::printf("      first divergence at rank %d: got %d, want %d\n", first_bad,
                got_indices[first_bad], want_indices[first_bad]);
  }
  verdict.CheckTrue("top-512 index list matches exactly", index_mismatches == 0,
                    "ranked selection diverged from the oracle");

  // The selection must be a permutation of valid logical keys, in strictly
  // decreasing BF16 score order with ascending index on ties.
  bool in_range = true;
  bool ordered = true;
  std::vector<bool> seen(seq_len, false);
  bool unique = true;
  for (uint32_t i = 0; i < kTopK; ++i) {
    const int32_t idx = got_indices[i];
    if (idx < 0 || static_cast<uint32_t>(idx) >= seq_len) {
      in_range = false;
      continue;
    }
    if (seen[static_cast<size_t>(idx)]) {
      unique = false;
    }
    seen[static_cast<size_t>(idx)] = true;
    if (i > 0) {
      const int32_t prev = got_indices[i - 1];
      if (prev >= 0 && static_cast<uint32_t>(prev) < seq_len) {
        const float a = want_scores[static_cast<size_t>(prev)];
        const float b = want_scores[static_cast<size_t>(idx)];
        // Equal BF16 scores must come out in ascending index order.
        if (a < b || (a == b && prev > idx)) {
          ordered = false;
        }
      }
    }
  }
  verdict.CheckTrue("all selected indices in range", in_range);
  verdict.CheckTrue("no index selected twice", unique);
  verdict.CheckTrue("rank order is score-desc then index-asc", ordered,
                    "tie-break order is wrong");

  AscendC::GmFree(query_gm);
  AscendC::GmFree(key_gm);
  AscendC::GmFree(scales_gm);
  AscendC::GmFree(table_gm);
  AscendC::GmFree(indices_gm);
  AscendC::GmFree(workspace_gm);
  AscendC::GmFree(tiling_gm);
  return verdict.Finish();
}
