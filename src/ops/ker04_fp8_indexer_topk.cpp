// ker04_fp8_indexer_topk.cpp -- KER-04 device kernel (Ascend 950PR,
// arch35 / dav-3510, single AIV, hybrid SIMD/SIMT).
//
// Math (oracle src/emu/fast_fp8_indexer_topk.cpp):
//   score[j] = k_scale[j] * sum_h qw[h] * relu(q[h] . k[j])   FP8 E4M3 dot,
//                                                              ReLU PER HEAD
//                                                              BEFORE the
//                                                              weighted sum
//   selection = top-512 by BF16-rounded score descending, then logical
//               index ascending on exact BF16 ties; unused lanes get -1.
//
// Scoring runs warp-per-key: each of the 16 warps owns keys j = w, w+16,
// ..., lane l covers heads {l, l+32}; the per-key score is warp-local via
// the 5-mask butterfly. E4M3 is decoded with the exact oracle bit arithmetic
// (register-only, no table). FP32 scores land in the GM workspace
// (align_up(4*seq_len, 32) B) and are re-read for selection, as the
// contract's v1 prescribes, so the selection input is a stable GM copy.
//
// Selection builds one uint32 composite per key -- orderable BF16 bits in
// the high 16, the inverted logical index in the low 16 -- so a single
// unsigned compare realizes the golden's "score descending, index ascending"
// total order with no tie handling at all. A 2,048-key bitonic network (66
// uniform stages, asc_syncthreads between stages, single writer per pair)
// sorts descending; padding keys are exactly 0, which no real key equals
// because every inverted index is nonzero. The first min(top_k, seq_len)
// slots hold the winners in rank order; the rest of the output is -1.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "moe/kernels/arch35/ker04_ub_layout.hpp"
#include "moe/kernels/arch35/tiling_data.hpp"

namespace moe::kernels::arch35::ker04 {

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
              __gm__ float* scores_ws) {
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t warp = t / kWarpSize;
  const std::uint32_t lane = t % kWarpSize;
  const std::uint32_t num_warps = kSimtThreads / kWarpSize;  // 16

  for (std::uint32_t j = warp; j < seq_len; j += num_warps) {
    // PA_BSND with num_kv_heads == 1: row stride is head_dim, block stride
    // is block_size * head_dim; logical j resolves through the table.
    const std::uint32_t logical_block = j / block_size;
    const std::uint32_t within_block = j % block_size;
    const std::uint32_t physical_block =
        (block_table != nullptr)
            ? static_cast<std::uint32_t>(block_table[logical_block])
            : logical_block;
    __gm__ std::uint8_t* key_row =
        key_cache_gm +
        (physical_block * block_size + within_block) * kHeadDim;

    // The lane covers heads {lane, lane+32}; ReLU is per head and comes
    // BEFORE the qw weighting -- summing heads first is a different,
    // strictly larger function.
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
// bitonic sort descending, emit the ranked indices
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
SelectTopKSimt(__gm__ const float* scores_ws,  // [seq_len] FP32, stable copy
               __gm__ std::int32_t* indices, std::uint32_t seq_len,
               std::uint32_t top_k,
               __ubuf__ std::uint32_t* keys) {  // [2,048]
  const std::uint32_t t = threadIdx.x;

  // Build the composites: BF16 rounding happens here, exactly between the
  // FP32 scoring and the comparison, as the golden does.
  for (std::uint32_t j = t; j < kSortLength; j += kSimtThreads) {
    if (j < seq_len) {
      keys[j] = MakeSelectionKey(scores_ws[j], j);
    } else {
      keys[j] = kSelectionPadKey;  // exactly 0; unreachable by a real key
    }
  }
  asc_syncthreads();

  // Bitonic sort, descending overall: the (i & k) == 0 half wants the
  // larger keys. Uniform trip counts everywhere, one writer per pair.
  for (std::uint32_t k = 2u; k <= kSortLength; k <<= 1u) {
    for (std::uint32_t jj = k >> 1u; jj > 0u; jj >>= 1u) {
      for (std::uint32_t i = t; i < kSortLength; i += kSimtThreads) {
        const std::uint32_t l = i ^ jj;
        if (l > i) {
          const std::uint32_t a = keys[i];
          const std::uint32_t b = keys[l];
          const bool want_desc = ((i & k) == 0u);
          if (want_desc ? (a < b) : (a > b)) {
            keys[i] = b;
            keys[l] = a;
          }
        }
      }
      asc_syncthreads();
    }
  }

  // The winners sit in rank order at the head of the buffer; short
  // sequences leave the remaining output lanes as the golden's -1 fill.
  const std::uint32_t selected = top_k < seq_len ? top_k : seq_len;
  for (std::uint32_t slot = t; slot < top_k; slot += kSimtThreads) {
    const std::int32_t value =
        (slot < selected)
            ? static_cast<std::int32_t>(SelectionKeyToIndex(keys[slot]))
            : -1;
    indices[slot] = value;
  }
}

class Fp8IndexerTopK {
 public:
  __aicore__ inline Fp8IndexerTopK(GM_ADDR query, GM_ADDR key_cache,
                                   GM_ADDR dequant_scales,
                                   GM_ADDR block_table,
                                   GM_ADDR indices, GM_ADDR workspace,
                                   const __gm__ IndexerTilingData* tiling) {
    // CCE address spaces forbid struct copy from __gm__; field-by-field.
    tiling_.heads = tiling->heads;
    tiling_.head_dim = tiling->head_dim;
    tiling_.top_k = tiling->top_k;
    tiling_.seq_len = tiling->seq_len;
    tiling_.block_size = tiling->block_size;

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

  __aicore__ inline void Init() {
    pipe_.InitBuffer(queryFp8Buf_, kQueryFp8Bytes);
    pipe_.InitBuffer(queryF32Buf_, kQueryF32Bytes);
    pipe_.InitBuffer(headScalesBuf_, kHeadScalesBytes);
    pipe_.InitBuffer(sortKeysBuf_, kSortKeysBytes);
    pipe_.InitBuffer(scalarBuf_, kScalarStateBytes);
  }

  __aicore__ inline void Process() {
    const std::uint32_t seq_len = tiling_.seq_len;
    const std::uint32_t top_k = tiling_.top_k;

    // A zero-length sequence flows through the same SIMT selection: the
    // keys buffer holds only padding composites, every output lane is the
    // golden's -1 fill, and the workspace is never touched.

    // Query bytes and the 64 fused head scales land in UB once.
    DataCopy(queryFp8Buf_.Get<std::uint8_t>(), queryGm_, kHeads * kHeadDim);
    DataCopy(headScalesBuf_.Get<float>(), scalesGm_, kHeads);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);

    DataSyncBarrier<MemDsbT::ALL>();
    asc_vf_call<DecodeQuerySimt>(
        dim3(kSimtThreads),
        reinterpret_cast<__ubuf__ std::uint8_t*>(
            queryFp8Buf_.Get<std::uint8_t>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            queryF32Buf_.Get<float>().GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();

    asc_vf_call<ScoreKeysSimt>(
        dim3(kSimtThreads),
        reinterpret_cast<__ubuf__ float*>(
            queryF32Buf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            headScalesBuf_.Get<float>().GetPhyAddr()),
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
        tiling_.block_size, seq_len, scoresWs_);
    // The v1 contract deliberately round-trips the FP32 scores through GM
    // so the selection reads a stable copy; settle the stores first.
    DataSyncBarrier<MemDsbT::DDR>();

    asc_vf_call<SelectTopKSimt>(
        dim3(kSimtThreads), scoresWs_,
        const_cast<__gm__ std::int32_t*>(
            reinterpret_cast<const __gm__ std::int32_t*>(
                indicesGm_.GetPhyAddr())),
        seq_len, top_k,
        reinterpret_cast<__ubuf__ std::uint32_t*>(
            sortKeysBuf_.Get<std::uint32_t>().GetPhyAddr()));
    DataSyncBarrier<MemDsbT::DDR>();
  }

 private:
  IndexerTilingData tiling_;

  TPipe pipe_;
  TBuf<> queryFp8Buf_;
  TBuf<> queryF32Buf_;
  TBuf<> headScalesBuf_;
  TBuf<> sortKeysBuf_;
  TBuf<> scalarBuf_;

  GlobalTensor<std::uint8_t> queryGm_;
  GlobalTensor<std::uint8_t> keyCacheGm_;
  GlobalTensor<float> scalesGm_;
  GlobalTensor<std::int32_t> blockTableGm_;
  GlobalTensor<std::int32_t> indicesGm_;
  __gm__ float* scoresWs_ = nullptr;
};

// GM_ADDR parameter order mirrors LaunchFp8IndexerTopK. workspace must be
// non-null and at least align_up(4*seq_len, 32) bytes when seq_len > 0.
extern "C" __global__ __aicore__ void ker04_fp8_indexer_topk(
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
               td->key_block_stride < td->block_size * kHeadDim ||
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

}  // namespace moe::kernels::arch35::ker04
