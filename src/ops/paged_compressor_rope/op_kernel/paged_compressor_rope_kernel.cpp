// paged_compressor_rope.cpp -- paged_compressor_rope device kernel (Ascend 950PR,
// arch35 / dav-3510, single AIV, hybrid SIMD/SIMT), the C4A coff=2 branch
// of the paged compressor exactly as the oracle emulates it
// (src/emu/paged_compressor_decode.cpp :: paged_compressor_decode_c4a).
//
// Per decode step (step == state_meta.tokens_seen is a precondition):
//   1. kv_state   = x @ wkv^T  [1024]   stored at kvCur[slot]
//      gate_state = x @ wgate^T [1024]  stored at gateCur[slot], + ape[slot]
//   2. HOLD (step%4 != 3): write the two current regions + meta back and
//      leave cache_slot untouched; emitted = 0.
//   3. EMIT (step%4 == 3): eight-row window = previous group's four a
//      halves (kv 0 / score -inf before the first emission) followed by
//      this group's four b halves; column softmax per feature lane (FTZ
//      denormal flush, as the oracle's NPU exp emulation); weighted KV
//      reduction; RMSNorm with the BF16 gain; half-mode RoPE on lanes
//      [448,512); one BF16 [512] token to the caller-resolved PA_BSND
//      slot; emitted = 1; roll this group's a halves into the previous
//      state.
//
// The projection and the emit phase are SIMT launches (512 threads, one
// feature lane per thread); weights and x stream from GM through the SIMT
// DataCache with BF16 decoded by bit shift in registers. The state reload
// is guarded by an explicit MTE3_MTE2 pairing and the writeback by an
// MTE2_MTE3 WAR pairing. GM scratch: 0 bytes.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "paged_compressor_rope_arch35.h"
#include "paged_compressor_rope_tiling.h"

namespace moe::kernels::arch35::paged_compressor_rope {

using namespace AscendC;

// BF16 -> FP32 is exact: shift the 16 bits into the top of an FP32.
__simt_callee__ __aicore__ inline float Bf16BitsToFp32(std::uint16_t bits) {
  return BitsToFloat(static_cast<std::uint32_t>(bits) << 16u);
}

__simt_callee__ __aicore__ inline float NegativeInf() {
  return BitsToFloat(0xFF800000u);
}

// ---------------------------------------------------------------------------
// SIMT projection: kv[j] = wkv[j,:] . x, gate[j] = wgate[j,:] . x + ape
// ---------------------------------------------------------------------------
// 512 threads cover the 1,024-state-width features twice (j = t, t+512),
// interleaving the wkv and wgate features so each thread owns four 4,096-MAC
// row dots. Weight bytes stream once (asc_ldcg); x (8 KiB) is re-read from
// the DataCache (asc_ldca) by every row.
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
CompressorProjectSimt(__gm__ std::uint16_t* x_gm,
                      __gm__ std::uint16_t* wkv_gm,
                      __gm__ std::uint16_t* wgate_gm,
                      __ubuf__ const float* ape,  // [4 * 1024]
                      std::uint32_t slot,
                      __ubuf__ float* kv_cur,     // [4 * 1024]
                      __ubuf__ float* gate_cur) {  // [4 * 1024]
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t ape_base = slot * kStateWidth;
  for (std::uint32_t k = 0; k < 2; ++k) {
    const std::uint32_t j = t + k * kSimtThreads;  // state feature in [0,1024)

    float kv_dot = 0.0f;
    const __gm__ std::uint16_t* kv_row = wkv_gm + j * kHidden;
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      kv_dot += Bf16BitsToFp32(kv_row[d]) * Bf16BitsToFp32(x_gm[d]);
    }
    kv_cur[slot * kStateWidth + j] = kv_dot;

    float gate_dot = 0.0f;
    const __gm__ std::uint16_t* gate_row = wgate_gm + j * kHidden;
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      gate_dot += Bf16BitsToFp32(gate_row[d]) * Bf16BitsToFp32(x_gm[d]);
    }
    // ape is [ratio, coff*D], indexed at the token's position in the group.
    gate_cur[slot * kStateWidth + j] = gate_dot + ape[ape_base + j];
  }
}

// ---------------------------------------------------------------------------
// SIMT emit: window -> column softmax -> weighted reduce -> RMSNorm -> RoPE
// ---------------------------------------------------------------------------
// One feature lane per thread; the eight window rows live in registers.
// Ends by rolling this group's a halves into the previous-a state, exactly
// as the oracle does AFTER the reduction.
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
CompressorEmitSimt(__ubuf__ const float* kv_cur, __ubuf__ const float* gate_cur,
                   __ubuf__ float* prev_kv_a, __ubuf__ float* prev_gate_a,
                   std::int32_t have_previous, __ubuf__ const float* norm_w,
                   __ubuf__ const float* rope_cos,  // [64]
                   __ubuf__ const float* rope_sin,  // [64]
                   float eps, __ubuf__ float* out,  // [512]
                   __ubuf__ float* partials) {      // [16] + [16] inv-rms
  const std::uint32_t i = threadIdx.x;  // feature lane
  const std::uint32_t warp = i / 32u;
  const std::uint32_t lane = i % 32u;

  // Assemble the eight-row window for this lane: rows 0..3 previous a
  // halves, rows 4..7 current b halves (upper half of each state row).
  float kv[kReduceRows];
  float sc[kReduceRows];
  for (std::uint32_t r = 0; r < kRatio; ++r) {
    if (have_previous != 0) {
      kv[r] = prev_kv_a[r * kHeadDim + i];
      sc[r] = prev_gate_a[r * kHeadDim + i];
    } else {
      kv[r] = 0.0f;  // the golden's first-group initialization
      sc[r] = NegativeInf();
    }
    kv[kRatio + r] = kv_cur[r * kStateWidth + kHeadDim + i];
    sc[kRatio + r] = gate_cur[r * kStateWidth + kHeadDim + i];
  }

  // Column softmax down the eight rows, max-subtracted, with the oracle's
  // denormal flush (the NPU exp is FTZ).
  float m = sc[0];
  for (std::uint32_t r = 1; r < kReduceRows; ++r) {
    m = fmaxf(m, sc[r]);
  }
  float w[kReduceRows];
  float sum = 0.0f;
  if (isfinite(m)) {
    for (std::uint32_t r = 0; r < kReduceRows; ++r) {
      float e = expf(sc[r] - m);
      if (e < 1.17549435e-38f) {  // flush subnormal exp results, as the oracle
        e = 0.0f;
      }
      w[r] = e;
      sum += e;
    }
    // sum >= 1 because the max-subtracted row contributes exp(0) == 1.
    const float inv = 1.0f / sum;
    for (std::uint32_t r = 0; r < kReduceRows; ++r) {
      w[r] *= inv;
    }
  } else {
    for (std::uint32_t r = 0; r < kReduceRows; ++r) {
      w[r] = 0.0f;  // an all--inf lane: zero is the only defensible answer
    }
  }

  float acc = 0.0f;
  for (std::uint32_t r = 0; r < kReduceRows; ++r) {
    acc += w[r] * kv[r];
  }

  // Block RMSNorm reduce: warp butterfly, one partial per warp, warp 0
  // folds the 16 partials, and the reciprocal RMS is broadcast via UB.
  float sq = acc * acc;
  for (std::uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
    sq += asc_shfl_xor(sq, static_cast<std::int32_t>(offset), 32);
  }
  if (lane == 0u) {
    partials[warp] = sq;
  }
  asc_syncthreads();
  if (warp == 0u) {
    float total = (lane < kSimtThreads / 32u) ? partials[lane] : 0.0f;
    for (std::uint32_t offset = 8u; offset > 0u; offset >>= 1u) {
      total += asc_shfl_xor(total, static_cast<std::int32_t>(offset), 32);
    }
    if (lane == 0u) {
      const float mean = total / static_cast<float>(kHeadDim);
      partials[kSimtThreads / 32u] = 1.0f / sqrtf(mean + eps);
    }
  }
  asc_syncthreads();
  const float inv_rms = partials[kSimtThreads / 32u];

  // out = (gain * acc) * inv_rms, the oracle's multiply order.
  const float token = (norm_w[i] * acc) * inv_rms;
  out[i] = token;
  asc_syncthreads();

  // Half-mode RoPE on the LAST 64 lanes; the second line reads the tables
  // at half+i, not i (vf_rope.h indexing, kept for non-duplicated tables).
  if (i < kRopeHeadDim / 2u) {
    const float lo = out[kRopeBaseOffset + i];
    const float hi = out[kRopeBaseOffset + kRopeHeadDim / 2u + i];
    out[kRopeBaseOffset + i] =
        rope_cos[i] * lo - rope_sin[i] * hi;
    out[kRopeBaseOffset + kRopeHeadDim / 2u + i] =
        rope_sin[kRopeHeadDim / 2u + i] * lo +
        rope_cos[kRopeHeadDim / 2u + i] * hi;
  }
  asc_syncthreads();

  // Roll: this group's a halves become the next group's previous rows.
  for (std::uint32_t idx = i; idx < kRatio * kHeadDim; idx += kSimtThreads) {
    const std::uint32_t r = idx / kHeadDim;
    const std::uint32_t l = idx % kHeadDim;
    prev_kv_a[idx] = kv_cur[r * kStateWidth + l];
    prev_gate_a[idx] = gate_cur[r * kStateWidth + l];
  }
}

class PagedCompressorRope {
 public:
  __aicore__ inline PagedCompressorRope(
      GM_ADDR x, GM_ADDR wkv, GM_ADDR wgate, GM_ADDR ape, GM_ADDR norm_weight,
      GM_ADDR rope_cos, GM_ADDR rope_sin, GM_ADDR state, GM_ADDR state_meta,
      GM_ADDR cache_slot, GM_ADDR emitted, std::uint32_t step,
      const __gm__ CompressorTilingData* tiling)
      : step_(step) {
    // Only the norm epsilon survives past the launch guard; geometry is
    // validated before construction and the projection walks GM directly.
    norm_eps_ = tiling->norm_eps;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), kHidden);
    wkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wkv));
    wgateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(wgate));
    apeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ape));
    normWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(norm_weight),
                             kHeadDim);
    ropeCosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rope_cos),
                               kRopeHeadDim);
    ropeSinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rope_sin),
                               kRopeHeadDim);
    // The UB regions mirror the GM ABI offsets one-to-one.
    stateKvGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + 0,
        kKvCurRegionFloats);
    stateGateGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + 2 * kKvCurRegionFloats,
        kGateCurRegionFloats);
    statePrevKvGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + 2 * kKvCurRegionFloats,
        kPrevARegionFloats);
    statePrevGateGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + 2 * kKvCurRegionFloats +
            kPrevARegionFloats,
        kPrevARegionFloats);
    metaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(state_meta),
                            4);
    cacheSlotGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t*>(cache_slot), kHeadDim);
    emittedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(emitted),
                               1);
  }

  __aicore__ inline void Init() {
    pipe_.InitBuffer(kvCurBuf_, kKvCurRegionFloats * 4);
    pipe_.InitBuffer(gateCurBuf_, kGateCurRegionFloats * 4);
    pipe_.InitBuffer(prevKvABuf_, kPrevARegionFloats * 4);
    pipe_.InitBuffer(prevGateABuf_, kPrevARegionFloats * 4);
    pipe_.InitBuffer(apeBuf_, kApeBytes);
    pipe_.InitBuffer(outF32Buf_, kOutF32Bytes);
    pipe_.InitBuffer(normWBf16Buf_, kNormWBytes);
    pipe_.InitBuffer(normWF32Buf_, kNormWF32Bytes);
    pipe_.InitBuffer(ropeCosBuf_, kRopeTableBytes / 2);
    pipe_.InitBuffer(ropeSinBuf_, kRopeTableBytes / 2);
    pipe_.InitBuffer(outBf16Buf_, kOutBf16Bytes);
    pipe_.InitBuffer(reduceBuf_, kReduceSlackBytes);
    pipe_.InitBuffer(scalarBuf_, kScalarStateBytes);
  }

  __aicore__ inline void Process() {
    // Cadence validation: the caller must advance step == tokens_seen on the
    // same stream for a given state. On a violation the kernel refuses to
    // touch the state and reports a non-emission.
    const std::uint32_t tokens_seen = metaGm_.GetValue(1);
    if (step_ != tokens_seen) {
      emittedGm_.SetValue(0, 0u);
      return;
    }
    const std::int32_t have_previous =
        static_cast<std::int32_t>(metaGm_.GetValue(0) & 1u);
    const std::uint32_t slot = step_ % kRatio;

    LoadStateAndTables();

    // Projection: 2,048 row dots against the resident token, straight into
    // the current-group regions of the state copy.
    DataSyncBarrier<MemDsbT::ALL>();
    asc_vf_call<CompressorProjectSimt>(
        dim3(kSimtThreads),
        const_cast<__gm__ std::uint16_t*>(
            reinterpret_cast<const __gm__ std::uint16_t*>(xGm_.GetPhyAddr())),
        const_cast<__gm__ std::uint16_t*>(
            reinterpret_cast<const __gm__ std::uint16_t*>(wkvGm_.GetPhyAddr())),
        const_cast<__gm__ std::uint16_t*>(
            reinterpret_cast<const __gm__ std::uint16_t*>(
                wgateGm_.GetPhyAddr())),
        reinterpret_cast<__ubuf__ float*>(apeBuf_.Get<float>().GetPhyAddr()),
        slot,
        reinterpret_cast<__ubuf__ float*>(kvCurBuf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            gateCurBuf_.Get<float>().GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();

    if (slot != kRatio - 1u) {
      // HOLD: write the two current regions + meta; the slot is untouched.
      WritebackRegions(false);
      metaGm_.SetValue(1, step_ + 1u);
      emittedGm_.SetValue(0, 0u);
      return;
    }

    // EMIT: the eight-row window, softmax, reduction, norm, RoPE, roll.
    asc_vf_call<CompressorEmitSimt>(
        dim3(kSimtThreads),
        reinterpret_cast<__ubuf__ float*>(kvCurBuf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            gateCurBuf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            prevKvABuf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            prevGateABuf_.Get<float>().GetPhyAddr()),
        have_previous,
        reinterpret_cast<__ubuf__ float*>(
            normWF32Buf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            ropeCosBuf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(
            ropeSinBuf_.Get<float>().GetPhyAddr()),
        norm_eps_,
        reinterpret_cast<__ubuf__ float*>(outF32Buf_.Get<float>().GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(reduceBuf_.Get<float>().GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();

    auto outF32 = outF32Buf_.Get<float>();
    auto outBf16 = outBf16Buf_.Get<bfloat16_t>();
    Cast(outBf16, outF32, RoundMode::CAST_RINT, kHeadDim);
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(cacheSlotGm_, outBf16, kHeadDim);

    WritebackRegions(true);
    metaGm_.SetValue(0, 1u);              // have_previous
    metaGm_.SetValue(1, step_ + 1u);      // tokens_seen
    emittedGm_.SetValue(0, 1u);
  }

 private:
  __aicore__ inline void LoadStateAndTables() {
    // Directive guard: order the state MTE2 reload after any in-flight MTE3
    // store of the same buffer on this core.
    SetFlag<HardEvent::MTE3_MTE2>(kEvMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(kEvMte3ToMte2);

    DataCopy(kvCurBuf_.Get<float>(), stateKvGm_, kKvCurRegionFloats);
    DataCopy(gateCurBuf_.Get<float>(), stateGateGm_, kGateCurRegionFloats);
    DataCopy(prevKvABuf_.Get<float>(), statePrevKvGm_, kPrevARegionFloats);
    DataCopy(prevGateABuf_.Get<float>(), statePrevGateGm_, kPrevARegionFloats);
    DataCopy(apeBuf_.Get<float>(), apeGm_,
             kRatio * kStateWidth);  // the [4,1024] ape block, resident
    DataCopy(normWBf16Buf_.Get<bfloat16_t>(), normWGm_, kHeadDim);
    DataCopy(ropeCosBuf_.Get<float>(), ropeCosGm_, kRopeHeadDim);
    DataCopy(ropeSinBuf_.Get<float>(), ropeSinGm_, kRopeHeadDim);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    // The widened norm gain is consumed by the emit launch.
    Cast(normWF32Buf_.Get<float>(), normWBf16Buf_.Get<bfloat16_t>(),
         RoundMode::CAST_NONE, kHeadDim);
  }

  __aicore__ inline void WritebackRegions(bool with_previous) {
    // WAR guard: every state MTE2 read must drain before the MTE3 writeback.
    SetFlag<HardEvent::MTE2_MTE3>(kEvMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(kEvMte2ToMte3);
    DataCopy(stateKvGm_, kvCurBuf_.Get<float>(), kKvCurRegionFloats);
    DataCopy(stateGateGm_, gateCurBuf_.Get<float>(), kGateCurRegionFloats);
    if (with_previous) {
      DataCopy(statePrevKvGm_, prevKvABuf_.Get<float>(), kPrevARegionFloats);
      DataCopy(statePrevGateGm_, prevGateABuf_.Get<float>(),
               kPrevARegionFloats);
    }
  }

  std::uint32_t step_;
  float norm_eps_;

  TPipe pipe_;
  TBuf<> kvCurBuf_;
  TBuf<> gateCurBuf_;
  TBuf<> prevKvABuf_;
  TBuf<> prevGateABuf_;
  TBuf<> apeBuf_;
  TBuf<> outF32Buf_;
  TBuf<> normWBf16Buf_;
  TBuf<> normWF32Buf_;
  TBuf<> ropeCosBuf_;
  TBuf<> ropeSinBuf_;
  TBuf<> outBf16Buf_;
  TBuf<> reduceBuf_;
  TBuf<> scalarBuf_;

  GlobalTensor<bfloat16_t> xGm_;
  GlobalTensor<bfloat16_t> wkvGm_;
  GlobalTensor<bfloat16_t> wgateGm_;
  GlobalTensor<float> apeGm_;
  GlobalTensor<bfloat16_t> normWGm_;
  GlobalTensor<float> ropeCosGm_;
  GlobalTensor<float> ropeSinGm_;
  GlobalTensor<float> stateKvGm_;
  GlobalTensor<float> stateGateGm_;
  GlobalTensor<float> statePrevKvGm_;
  GlobalTensor<float> statePrevGateGm_;
  GlobalTensor<std::uint32_t> metaGm_;
  GlobalTensor<bfloat16_t> cacheSlotGm_;
  GlobalTensor<std::uint32_t> emittedGm_;
};

// GM_ADDR parameter order mirrors LaunchPagedCompressorRope (workspace
// unused: GM scratch is 0 bytes; the 49,168 B state is an owned allocation).
extern "C" __global__ __aicore__ void paged_compressor_rope(
    GM_ADDR x, GM_ADDR wkv, GM_ADDR wgate, GM_ADDR ape, GM_ADDR norm_weight,
    GM_ADDR rope_cos, GM_ADDR rope_sin, GM_ADDR state, GM_ADDR state_meta,
    GM_ADDR cache_slot, GM_ADDR emitted, std::uint32_t step,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  (void)workspace;
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ CompressorTilingData* td =
      reinterpret_cast<const __gm__ CompressorTilingData*>(tiling);
  if (unlikely(td->abi_version != kTilingAbiVersion || td->block_dim != 1 ||
               td->hidden != kHidden || td->head_dim != kHeadDim ||
               td->coff != kCoff || td->ratio != kRatio ||
               td->reduction_rows != kReduceRows ||
               td->rope_head_dim != kRopeHeadDim ||
               td->rotary_mode != 1 || td->reserved != 0 ||
               td->wkv_row_stride != kHidden ||
               td->wgate_row_stride != kHidden ||
               td->ape_row_stride != kStateWidth)) {
    return;
  }
  PagedCompressorRope op(x, wkv, wgate, ape, norm_weight, rope_cos, rope_sin,
                         state, state_meta, cache_slot, emitted, step, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::paged_compressor_rope
