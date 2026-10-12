// fast_fp8_indexer_score_kernel.cpp -- stage 1 of 2 for fast_fp8_indexer_topk
// (Ascend 950PR, arch35 / dav-3510, 32 AIV, hybrid SIMD/SIMT).
//
// WHY THIS KERNEL EXISTS
// Measured on hardware, the fused single-core operator ran 2.73 / 5.36 /
// 10.63 ms at seq_len 512 / 1024 / 2048 -- clean O(seq_len) on block_dim 1,
// with scoring accounting for essentially all of it. Scoring is warp-per-key
// and every key is independent, so the only thing holding it to one core was
// that selection, which genuinely wants a total order, shared the launch.
// Splitting the two lets scoring take all 32 AIVs while selection keeps its
// single-core network and its host-simulated verification.
//
// This kernel: decode the E4M3 query into UB once per core, then score this
// core's share of the keys straight out of the GM page cache through the SIMT
// DataCache, writing FP32 scores to the GM workspace. It writes nothing else;
// `indices` is the select kernel's output.
//
// Each core decodes the full query redundantly (8 KiB in, 32 KiB out, 512
// threads, one pass). That is cheaper than staging a shared decoded copy
// through GM and synchronizing on it, and it is per-core work that overlaps
// the scoring it feeds.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"
#include "fast_fp8_indexer_topk_tiling.h"
#include "fast_fp8_indexer_score_simt.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

using namespace AscendC;

class Fp8IndexerScore {
 public:
  __aicore__ inline Fp8IndexerScore(GM_ADDR query, GM_ADDR key_cache,
                                    GM_ADDR dequant_scales,
                                    GM_ADDR block_table, GM_ADDR workspace,
                                    const __gm__ IndexerTilingData* tiling) {
    block_idx_ = GetBlockIdx();
    tiling_.seq_len = tiling->seq_len;
    tiling_.block_size = tiling->block_size;
    tiling_.key_block_stride = tiling->key_block_stride;
    tiling_.num_physical_blocks = tiling->num_physical_blocks;

    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(query),
                             kHeads * kHeadDim);
    keyCacheGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::uint8_t*>(key_cache));
    scalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dequant_scales));
    blockTableGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(block_table));
    scoresWs_ = reinterpret_cast<__gm__ float*>(workspace);
  }

  __aicore__ inline void Process() {
    const std::uint32_t seq_len = tiling_.seq_len;
    // seq_len == 0 scores nothing; the select kernel writes the all -1 row.
    if (seq_len == 0u) {
      return;
    }

    // Every buffer below is dereferenced by a SIMT VF, so all of it comes
    // from the dynamic UB region whose size the host declares at launch
    // (IndexerScoreDynamicUbBytes). Allocation order IS the layout.
    LocalMemAllocator<> dynAlloc;
    LocalTensor<std::uint8_t> queryFp8Buf =
        dynAlloc.Alloc<std::uint8_t>(kHeads * kHeadDim);
    LocalTensor<float> queryF32Buf = dynAlloc.Alloc<float>(kHeads * kHeadDim);
    LocalTensor<float> headScalesBuf = dynAlloc.Alloc<float>(kHeads);

    DataCopy(queryFp8Buf, queryGm_, kHeads * kHeadDim);
    DataCopy(headScalesBuf, scalesGm_, kHeads);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);

    DataSyncBarrier<MemDsbT::ALL>();
    asc_vf_call<DecodeQuerySimt>(
        dim3(kSimtThreads),
        reinterpret_cast<__ubuf__ std::uint8_t*>(queryFp8Buf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(queryF32Buf.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();

    // This core's first global warp id, and the stride over all of them. Both
    // halves come from constants the host shares, so the grid it launches and
    // the stride used here cannot disagree.
    const std::uint32_t warp_base =
        static_cast<std::uint32_t>(block_idx_) * kWarpsPerCore;
    asc_vf_call<ScoreKeysSimt>(
        dim3(kSimtThreads),
        reinterpret_cast<__ubuf__ float*>(queryF32Buf.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(headScalesBuf.GetPhyAddr()),
        const_cast<__gm__ float*>(
            reinterpret_cast<const __gm__ float*>(scalesGm_.GetPhyAddr())),
        const_cast<__gm__ std::uint8_t*>(
            reinterpret_cast<const __gm__ std::uint8_t*>(
                keyCacheGm_.GetPhyAddr())),
        blockTableGm_.GetPhyAddr() == nullptr
            ? nullptr
            : const_cast<__gm__ std::int32_t*>(
                  reinterpret_cast<const __gm__ std::int32_t*>(
                      blockTableGm_.GetPhyAddr())),
        tiling_.block_size, seq_len, tiling_.key_block_stride,
        tiling_.num_physical_blocks, warp_base, kScoreWarpStride, scoresWs_);
    // DDR: the scores must be in GM before the select kernel's launch reads
    // them. The stores themselves bypass the DataCache (asc_stcg), which is
    // what keeps two cores writing the same 64 B line from clobbering each
    // other; this barrier is the drain of those stores.
    DataSyncBarrier<MemDsbT::DDR>();
  }

 private:
  int32_t block_idx_;
  IndexerTilingData tiling_;

  GlobalTensor<std::uint8_t> queryGm_;
  GlobalTensor<std::uint8_t> keyCacheGm_;
  GlobalTensor<float> scalesGm_;
  GlobalTensor<std::int32_t> blockTableGm_;
  __gm__ float* scoresWs_ = nullptr;
};

extern "C" __global__ __aicore__ void fast_fp8_indexer_score(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR dequant_scales,
    GM_ADDR block_table, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ IndexerTilingData* td =
      reinterpret_cast<const __gm__ IndexerTilingData*>(tiling);
  // The same geometry gate as the fused kernel, minus top_k (not read here)
  // and minus block_dim: `block_dim` in the POD is the SELECT stage's grid,
  // which is 1; this kernel is launched on kScoreBlockDim cores instead. The
  // 4-byte key_block_stride alignment the word-wide key load needs is checked
  // host-side in MakeIndexerTiling, where a rejection is observable; repeating
  // it here would only turn a caller mistake into a silent zeroed workspace.
  if (unlikely(td->abi_version != kTilingAbiVersion ||
               td->heads != kHeads || td->head_dim != kHeadDim ||
               td->seq_len > kMaxSeqLen ||
               td->block_size == 0 ||
               td->num_physical_blocks == 0 ||
               (td->key_block_stride & 3u) != 0u ||
               static_cast<std::uint64_t>(td->key_block_stride) <
                   static_cast<std::uint64_t>(td->block_size) * kHeadDim ||
               td->block_table_len <
                   (td->seq_len + td->block_size - 1) / td->block_size)) {
    return;
  }
  if (unlikely(td->seq_len > 0 && workspace == nullptr)) {
    return;
  }
  Fp8IndexerScore op(query, key_cache, dequant_scales, block_table, workspace,
                     td);
  op.Process();
}

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk
