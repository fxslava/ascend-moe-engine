// fast_fp8_indexer_score_simt.h -- the query-decode and key-scoring SIMT VFs.
//
// Op-private (CLAUDE.md §1.2): included by bare filename from this operator's
// op_kernel/ TUs only. It exists because the same two VFs are instantiated by
// two entry points -- the multi-core `fast_fp8_indexer_score` kernel that the
// host launches, and the single-core fused `fast_fp8_indexer_topk` kernel --
// and a second transcription of a scoring loop is a second place for it to
// drift.
//
// KEY DISTRIBUTION
// Scoring is warp-per-key: one warp owns key j, its 32 lanes split the 64
// heads two apiece, and the per-key reduction is a warp butterfly with no
// cross-warp traffic. That makes the work trivially splittable at warp
// granularity, so the caller passes a (warp_base, warp_stride) pair and the
// loop is a grid-stride over GLOBAL warp ids:
//
//   warp_base   = block_idx * warps_per_core + warp      in [0, 512)
//   warp_stride = block_dim * warps_per_core             == 512 at block_dim 32
//
// With block_dim 32 and 16 warps per core that is 512 global warps, so the
// union of the per-warp sequences is exactly [0, seq_len) with every key
// visited once -- which is the whole correctness argument for the split, and
// is simulated on the host in tests/emu rather than argued from here
// (CLAUDE.md §5.5). At block_dim 1 the pair degenerates to the original
// single-core schedule, so the fused kernel keeps its verified behaviour.
//
// WHY THE SCORE STORE MUST BYPASS THE DATACACHE
// `asc_stcg` was merely tidy while this ran on one core. It is now mandatory.
// Neighbouring keys land in different warps and therefore on different cores,
// so two cores routinely write 4-byte scores inside one 64 B cache line.
// DataCache reads and writes are performed at cache-line granularity, so a
// cached store would have each core read-modify-write the whole line and
// randomly discard its neighbour's score. Storing through GM/L2 without the
// DataCache is what makes the per-key writes independent.

#ifndef MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SCORE_SIMT_H
#define MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SCORE_SIMT_H

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

// ---------------------------------------------------------------------------
// SIMT phase 0: decode the E4M3 query into FP32 (once per kernel)
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
DecodeQuerySimt(__ubuf__ const std::uint8_t* query_fp8,  // [64*128]
                __ubuf__ float* query_f32) {             // [64*128]
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t base = t * 16u;  // 8,192 bytes over 512 threads
  for (std::uint32_t q = 0; q < 16u; ++q) {
    query_f32[base + q] = DecodeFp8E4M3(query_fp8[base + q]);
  }
}

// ---------------------------------------------------------------------------
// SIMT phase 1: warp-per-key scoring, raw FP32 scores to the GM workspace
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
ScoreKeysSimt(__ubuf__ const float* query_f32,     // [64*128]
              __ubuf__ const float* head_scales,   // qw[64]
              __gm__ float* dequant_scales_gm,     // [64 | seq_len]
              __gm__ std::uint8_t* key_cache_gm,   // PA_BSND
              __gm__ std::int32_t* block_table,    // may be null
              std::uint32_t block_size, std::uint32_t seq_len,
              std::uint32_t key_block_stride,
              std::uint32_t num_physical_blocks,
              std::uint32_t warp_base,    // this core's first global warp id
              std::uint32_t warp_stride,  // total global warps
              __gm__ float* scores_ws) {
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t warp = t / kWarpSize;
  const std::uint32_t lane = t % kWarpSize;

  for (std::uint32_t j = warp_base + warp; j < seq_len; j += warp_stride) {
    const std::uint32_t logical_block = j / block_size;
    const std::uint32_t within_block = j % block_size;

    std::uint32_t physical_block = logical_block;
    if (block_table != nullptr) {
      const std::int32_t entry = block_table[logical_block];
      if (unlikely(entry < 0 ||
                   static_cast<std::uint32_t>(entry) >= num_physical_blocks)) {
        if (lane == 0u) {
          asc_stcg(scores_ws + j, -__builtin_inff());
        }
        continue;
      }
      physical_block = static_cast<std::uint32_t>(entry);
    } else if (unlikely(logical_block >= num_physical_blocks)) {
      if (lane == 0u) {
        asc_stcg(scores_ws + j, -__builtin_inff());
      }
      continue;
    }

    __gm__ std::uint8_t* key_row =
        key_cache_gm + physical_block * key_block_stride +
        within_block * kHeadDim;
    // Four E4M3 codes per load instead of one. The row base is 4-byte aligned
    // -- within_block * 128 always is, and MakeIndexerTiling rejects a
    // key_block_stride that is not a multiple of 4 precisely so this cast is
    // sound -- and the codes are consumed in ascending d, so the FP32
    // accumulation order is unchanged and the scores stay bit-identical to the
    // byte-at-a-time loop.
    const __gm__ std::uint32_t* key_words =
        reinterpret_cast<const __gm__ std::uint32_t*>(key_row);

    float acc = 0.0f;
    for (std::uint32_t h = lane; h < kHeads; h += kWarpSize) {
      const __ubuf__ float* qh = query_f32 + h * kHeadDim;
      float dot = 0.0f;
      for (std::uint32_t w = 0; w < kHeadDim / 4u; ++w) {
        const std::uint32_t packed = key_words[w];
        const std::uint32_t d = w * 4u;
        dot += qh[d + 0u] *
               DecodeFp8E4M3(static_cast<std::uint8_t>(packed & 0xFFu));
        dot += qh[d + 1u] *
               DecodeFp8E4M3(static_cast<std::uint8_t>((packed >> 8u) & 0xFFu));
        dot += qh[d + 2u] *
               DecodeFp8E4M3(static_cast<std::uint8_t>((packed >> 16u) & 0xFFu));
        dot += qh[d + 3u] *
               DecodeFp8E4M3(static_cast<std::uint8_t>((packed >> 24u) & 0xFFu));
      }
      if (dot > 0.0f) {
        acc += head_scales[h] * dot;
      }
    }
    for (std::uint32_t offset = kWarpSize / 2u; offset > 0u; offset >>= 1u) {
      acc += asc_shfl_xor(acc, static_cast<std::int32_t>(offset), kWarpSize);
    }
    if (lane == 0u) {
      const float k_scale = asc_ldcg(dequant_scales_gm + kHeads + j);
      asc_stcg(scores_ws + j, acc * k_scale);
    }
  }
}

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk

#endif  // MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SCORE_SIMT_H
