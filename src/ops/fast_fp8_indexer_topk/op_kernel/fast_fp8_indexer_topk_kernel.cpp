// fast_fp8_indexer_topk_kernel.cpp -- fast_fp8_indexer_topk device kernel
// (Ascend 950PR, arch35 / dav-3510, single AIV, hybrid SIMD/SIMT).
//
// THE FUSED SINGLE-CORE PATH, AND IT IS CURRENTLY UNLAUNCHED.
// LaunchFp8IndexerTopK now issues the `fast_fp8_indexer_score` (32 AIV) +
// `fast_fp8_indexer_select` (1 AIV) pair, and no host wrapper reaches this
// entry. It is compiled and linked by the device gates but nothing executes
// it, so say that plainly rather than calling it a verified fallback: the
// device suite drives LaunchFp8IndexerTopK, which means it now exercises the
// SPLIT against the tests/emu oracle, not this.
//
// It is kept for one reason: it is the only shape in which the whole operator
// is a single launch, and reinstating it needs a host wrapper, not a rewrite.
// If no caller ever wants that, this file is dead weight and should be
// deleted along with its registry and gate entries -- that is a decision for
// whoever owns the operator's API, not something to settle by leaving it here
// indefinitely.
//
// Its two SIMT phases are the same function templates the split kernels
// instantiate (fast_fp8_indexer_score_simt.h, fast_fp8_indexer_select_simt.h),
// so there is one transcription of the scoring loop and one of the selection
// network, not three. The single-core key schedule is the degenerate case of
// the grid-stride pair: warp_base 0, warp_stride kWarpsPerCore.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"
#include "fast_fp8_indexer_topk_tiling.h"
#include "fast_fp8_indexer_score_simt.h"
#include "fast_fp8_indexer_select_simt.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

using namespace AscendC;

class Fp8IndexerTopK {
 public:
  __aicore__ inline Fp8IndexerTopK(GM_ADDR query, GM_ADDR key_cache,
                                   GM_ADDR dequant_scales,
                                   GM_ADDR block_table,
                                   GM_ADDR indices, GM_ADDR workspace,
                                   const __gm__ IndexerTilingData* tiling) {
    tiling_.heads = tiling->heads;
    tiling_.head_dim = tiling->head_dim;
    tiling_.top_k = tiling->top_k;
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
    indicesGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ std::int32_t*>(indices), kTopK);
    scoresWs_ = reinterpret_cast<__gm__ float*>(workspace);
  }

  __aicore__ inline void Init() {}

  __aicore__ inline void Process() {
    const std::uint32_t seq_len = tiling_.seq_len;
    const std::uint32_t top_k = tiling_.top_k;

    // Allocate SIMT-visible buffers from dynamic UB via LocalMemAllocator
    LocalMemAllocator<> dynAlloc;
    LocalTensor<std::uint8_t> queryFp8Buf =
        dynAlloc.Alloc<std::uint8_t>(kHeads * kHeadDim);
    LocalTensor<float> queryF32Buf =
        dynAlloc.Alloc<float>(kHeads * kHeadDim);
    LocalTensor<float> headScalesBuf =
        dynAlloc.Alloc<float>(kHeads);
    LocalTensor<std::uint32_t> sortKeysBuf =
        dynAlloc.Alloc<std::uint32_t>(kSortLength);

    // Query bytes and the 64 fused head scales land in UB once.
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

    // One core: global warp id == local warp id, stride == warps per core.
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
        tiling_.num_physical_blocks, 0u, kWarpsPerCore, scoresWs_);
    DataSyncBarrier<MemDsbT::DDR>();

    asc_vf_call<SelectTopKSimt>(
        dim3(kSimtThreads), scoresWs_,
        const_cast<__gm__ std::int32_t*>(
            reinterpret_cast<const __gm__ std::int32_t*>(
                indicesGm_.GetPhyAddr())),
        seq_len, top_k,
        reinterpret_cast<__ubuf__ std::uint32_t*>(sortKeysBuf.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::DDR>();
  }

 private:
  IndexerTilingData tiling_;

  GlobalTensor<std::uint8_t> queryGm_;
  GlobalTensor<std::uint8_t> keyCacheGm_;
  GlobalTensor<float> scalesGm_;
  GlobalTensor<std::int32_t> blockTableGm_;
  GlobalTensor<std::int32_t> indicesGm_;
  __gm__ float* scoresWs_ = nullptr;
};

extern "C" __global__ __aicore__ void fast_fp8_indexer_topk(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR dequant_scales,
    GM_ADDR block_table, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ IndexerTilingData* td =
      reinterpret_cast<const __gm__ IndexerTilingData*>(tiling);
  if (unlikely(td->abi_version != kTilingAbiVersion || td->block_dim != 1 ||
               td->heads != kHeads || td->head_dim != kHeadDim ||
               td->top_k > kTopK || td->top_k == 0 ||
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
  Fp8IndexerTopK op(query, key_cache, dequant_scales, block_table, indices,
                    workspace, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::fast_fp8_indexer_topk
