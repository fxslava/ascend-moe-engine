// fast_fp8_indexer_select_kernel.cpp -- stage 2 of 2 for fast_fp8_indexer_topk
// (Ascend 950PR, arch35 / dav-3510, single AIV, SIMT).
//
// Reads the FP32 scores the `fast_fp8_indexer_score` kernel left in the GM
// workspace, rounds them through BF16 into the composite selection key, sorts
// 2,048 composites descending with the tiered bitonic network, and writes the
// ranked indices.
//
// Deliberately still ONE core. The network is a total order over all slots and
// its widest tier already exchanges through UB; a multi-core version would
// have to stage those exchanges in GM behind inter-core barriers, and it would
// no longer be the network that tests/emu simulates against std::sort
// (CLAUDE.md §5.5). The hardware numbers say it is ~0.5% of the operator, so
// there is nothing to buy by moving it.
//
// seq_len == 0 is a real case and is handled here rather than skipped: every
// slot takes the pad key, so the output row is all -1 and the score kernel can
// return without touching the workspace.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"
#include "fast_fp8_indexer_topk_tiling.h"
#include "fast_fp8_indexer_select_simt.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

using namespace AscendC;

class Fp8IndexerSelect {
 public:
  __aicore__ inline Fp8IndexerSelect(GM_ADDR indices, GM_ADDR workspace,
                                     const __gm__ IndexerTilingData* tiling) {
    tiling_.top_k = tiling->top_k;
    tiling_.seq_len = tiling->seq_len;

    indicesGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(indices), kTopK);
    scoresWs_ = reinterpret_cast<__gm__ float*>(workspace);
  }

  __aicore__ inline void Process() {
    // The one SIMT-visible buffer, from the dynamic region the host sized as
    // IndexerSelectDynamicUbBytes.
    LocalMemAllocator<> dynAlloc;
    LocalTensor<std::uint32_t> sortKeysBuf =
        dynAlloc.Alloc<std::uint32_t>(kSortLength);

    asc_vf_call<SelectTopKSimt>(
        dim3(kSimtThreads), scoresWs_,
        const_cast<__gm__ std::int32_t*>(
            reinterpret_cast<const __gm__ std::int32_t*>(
                indicesGm_.GetPhyAddr())),
        tiling_.seq_len, tiling_.top_k,
        reinterpret_cast<__ubuf__ std::uint32_t*>(sortKeysBuf.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::DDR>();
  }

 private:
  IndexerTilingData tiling_;

  GlobalTensor<std::int32_t> indicesGm_;
  __gm__ float* scoresWs_ = nullptr;
};

extern "C" __global__ __aicore__ void fast_fp8_indexer_select(
    GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ IndexerTilingData* td =
      reinterpret_cast<const __gm__ IndexerTilingData*>(tiling);
  // block_dim in the POD is this stage's grid, so it is checked here.
  if (unlikely(td->abi_version != kTilingAbiVersion || td->block_dim != 1 ||
               td->top_k > kTopK || td->top_k == 0 ||
               td->seq_len > kMaxSeqLen)) {
    return;
  }
  if (unlikely(td->seq_len > 0 && workspace == nullptr)) {
    return;
  }
  Fp8IndexerSelect op(indices, workspace, td);
  op.Process();
}

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk
