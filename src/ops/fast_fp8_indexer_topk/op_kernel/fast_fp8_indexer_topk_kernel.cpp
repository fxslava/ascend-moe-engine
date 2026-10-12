// fast_fp8_indexer_topk_kernel.cpp -- fast_fp8_indexer_topk device kernel (Ascend 950PR,
// arch35 / dav-3510, single AIV, hybrid SIMD/SIMT).

#include "moe/kernels/arch35/arch35_common.hpp"
#include "fast_fp8_indexer_topk_arch35.h"
#include "fast_fp8_indexer_topk_tiling.h"

namespace moe::kernels::arch35::fast_fp8_indexer_topk {

using namespace AscendC;

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
              __gm__ float* scores_ws) {
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t warp = t / kWarpSize;
  const std::uint32_t lane = t % kWarpSize;
  const std::uint32_t num_warps = kSimtThreads / kWarpSize;  // 16

  for (std::uint32_t j = warp; j < seq_len; j += num_warps) {
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

    float acc = 0.0f;
    for (std::uint32_t h = lane; h < kHeads; h += kWarpSize) {
      const __ubuf__ float* qh = query_f32 + h * kHeadDim;
      float dot = 0.0f;
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        dot += qh[d] * DecodeFp8E4M3(key_row[d]);
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

// ---------------------------------------------------------------------------
// SIMT phase 2: re-read the GM scores, round through BF16, composite-key
// bitonic sort descending, emit the ranked indices.
// ---------------------------------------------------------------------------
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
        tiling_.num_physical_blocks, scoresWs_);
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