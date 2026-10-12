// fast_fp8_indexer_select_simt.h -- the tiered bitonic selection SIMT VF.
//
// Op-private (CLAUDE.md §1.2); included by bare filename from this operator's
// op_kernel/ TUs only. Shared by the single-core `fast_fp8_indexer_select`
// kernel the host launches and the fused `fast_fp8_indexer_topk` kernel, so
// the two cannot drift.
//
// The network itself is UNCHANGED from the version transcribed into
// tests/emu and compared against std::sort there (CLAUDE.md §5.5). It is
// deliberately left at one core: the 66 bitonic levels are a total order over
// all 2,048 slots, and its cross-warp tier already exchanges through UB, so
// splitting it across cores would mean a GM-resident network with inter-core
// barriers for a stage that the hardware benchmark shows costs ~0.5% of the
// operator's time. Moving it is not worth retiring its host simulation for.
//
// Its only input is scores_ws in GM, which is what lets the scoring stage run
// on a different grid in a separate launch: hardware stream ordering makes the
// preceding score kernel's GM writes visible before this one issues.

#ifndef MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SELECT_SIMT_H
#define MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SELECT_SIMT_H

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

// Re-read the GM scores, round through BF16, composite-key bitonic sort
// descending, emit the ranked indices.
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
SelectTopKSimt(__gm__ const float* scores_ws,  // [seq_len] FP32, stable copy
               __gm__ std::int32_t* indices, std::uint32_t seq_len,
               std::uint32_t top_k,
               __ubuf__ std::uint32_t* keys) {  // [2,048] cross-warp staging
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t base = t * kKeysPerThread;

  std::uint32_t key[kKeysPerThread];
  for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
    const std::uint32_t j = base + r;
    key[r] = (j < seq_len) ? MakeSelectionKey(scores_ws[j], j)
                           : kSelectionPadKey;
  }

  for (std::uint32_t k = 2u; k <= kSortLength; k <<= 1u) {
    for (std::uint32_t jj = k >> 1u; jj > 0u; jj >>= 1u) {
      if (jj < kKeysPerThread) {
        // --- tier 1: both elements are this thread's registers ------------
        for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
          const std::uint32_t partner_r = r ^ jj;
          if (partner_r > r) {
            const std::uint32_t i = base + r;
            const std::uint32_t a = key[r];
            const std::uint32_t b = key[partner_r];
            const std::uint32_t hi = a > b ? a : b;
            const std::uint32_t lo = a > b ? b : a;
            const bool want_desc = ((i & k) == 0u);
            key[r] = want_desc ? hi : lo;
            key[partner_r] = want_desc ? lo : hi;
          }
        }
      } else if (jj < kKeysPerThread * kWarpSize) {
        // --- tier 2: same warp, different lane ----------------------------
        const std::int32_t lane_mask =
            static_cast<std::int32_t>(jj / kKeysPerThread);
        for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
          const std::uint32_t i = base + r;
          const std::uint32_t a = key[r];
          const std::uint32_t b = asc_shfl_xor(a, lane_mask, kWarpSize);
          const std::uint32_t hi = a > b ? a : b;
          const std::uint32_t lo = a > b ? b : a;
          const bool want_desc = ((i & k) == 0u);
          const bool i_is_low = ((i & jj) == 0u);
          key[r] = (i_is_low == want_desc) ? hi : lo;
        }
      } else {
        // --- tier 3: cross-warp, through UB -------------------------------
        for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
          keys[base + r] = key[r];
        }
        asc_syncthreads();
        for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
          const std::uint32_t i = base + r;
          const std::uint32_t a = key[r];
          const std::uint32_t b = keys[i ^ jj];
          const std::uint32_t hi = a > b ? a : b;
          const std::uint32_t lo = a > b ? b : a;
          const bool want_desc = ((i & k) == 0u);
          const bool i_is_low = ((i & jj) == 0u);
          key[r] = (i_is_low == want_desc) ? hi : lo;
        }
        asc_syncthreads();
      }
    }
  }

  const std::uint32_t selected = top_k < seq_len ? top_k : seq_len;
  for (std::uint32_t r = 0u; r < kKeysPerThread; ++r) {
    const std::uint32_t slot = base + r;
    if (slot < top_k) {
      indices[slot] = (slot < selected)
                          ? static_cast<std::int32_t>(
                                SelectionKeyToIndex(key[r]))
                          : -1;
    }
  }
}

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk

#endif  // MOE_KERNELS_ARCH35_FAST_FP8_INDEXER_SELECT_SIMT_H
