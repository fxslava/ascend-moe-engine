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
    // The UB regions mirror the GM ABI offsets one-to-one. The ABI is four
    // consecutive regions in this order -- kv_cur | gate_cur | prev_kv_a |
    // prev_gate_a -- so gate_cur begins one kv-region in, at float 4,096.
    //
    // It was previously set to 2 * kKvCurRegionFloats (float 8,192), which is
    // where prev_kv_a starts: gate_cur then covered exactly the union of the
    // two prev-a regions and the real gate slot [4,096, 8,192) was never
    // touched. Nothing read out of bounds, so it could not fault -- it
    // silently aliased. On an EMIT the prev-a writeback landed after the gate
    // writeback and clobbered it, and the following HOLD's gate writeback
    // clobbered prev-a in turn, so the first group emitted correctly and
    // every later group reduced against corrupted previous rows.
    stateKvGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + 0,
        kKvCurRegionFloats);
    stateGateGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(state) + kKvCurRegionFloats,
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
    // ONE ALLOCATION MECHANISM. There is no TPipe in this kernel, and that is
    // load-bearing rather than tidiness.
    //
    // TPipe/TBuf and LocalMemAllocator both bump a head pointer from the SAME
    // UB base, so a TBuf and the allocator's first segment occupy the same
    // bytes. This kernel used to hold two BF16 staging TBufs -- a 1 KiB norm
    // gain and a 1 KiB output token -- alongside the allocator, and they
    // aliased kvCur_ row 0: normWBf16 over lanes [0,256), outBf16 over
    // [256,512).
    //
    // It presented as a group-boundary bug and it is not one. Every step's
    // LoadStateAndTables copies the norm gain into that TBuf, which clobbers
    // kv_cur row 0's lower lanes; the whole-region writeback then persists the
    // damage. On step 0 the projection's slot is 0, so it rewrites row 0
    // afterwards and the corruption is invisible -- which is why step 0 passed
    // and step 1 reported kv_cur row 0 lane 83 holding 1.486311, a float whose
    // top 16 bits are the BF16 norm gain 1.484375. Row 0's a half is also what
    // the EMIT rolls into prev_kv_a, so group 1 then reduced a partly
    // overwritten previous row (cos 0.8848) while group 0, which reads only b
    // halves at lanes >= 512, stayed clean.
    //
    // Note a row-scoped writeback would NOT have fixed this: the EMIT roll
    // reads row 0's a half straight out of the aliased UB, so prev_kv_a would
    // still be corrupt. The aliasing itself had to go.
    //
    // Alloc() takes ELEMENTS, not bytes, and bumps a head pointer -- so this
    // call order fixes the offsets and must match the table in the tiling
    // header, which is where the host gets the byte count it declares at
    // launch. The two BF16 buffers are appended last so every pre-existing
    // offset is unchanged.
    //
    // The allocator is a function local, not a member: as a member its
    // inline constructor must be emitted from the enclosing class's ctor,
    // and on CANN 9.2.0-beta.2 that constructor has no definition the normal
    // include path reaches (-Wundefined-inline). head_ starts at 0 per
    // instance, and every allocation happens here, so a local is equivalent.
    LocalMemAllocator<> alloc;
    kvCur_ = alloc.Alloc<float>(kKvCurRegionFloats);
    gateCur_ = alloc.Alloc<float>(kGateCurRegionFloats);
    prevKvA_ = alloc.Alloc<float>(kPrevARegionFloats);
    prevGateA_ = alloc.Alloc<float>(kPrevARegionFloats);
    apeF32_ = alloc.Alloc<float>(kRatio * kStateWidth);
    normWF32_ = alloc.Alloc<float>(kHeadDim);
    ropeCos_ = alloc.Alloc<float>(kRopeHeadDim);
    ropeSin_ = alloc.Alloc<float>(kRopeHeadDim);
    outF32_ = alloc.Alloc<float>(kHeadDim);
    reduce_ = alloc.Alloc<float>(kReduceSlackFloats);
    // The two BF16 staging buffers, formerly TBufs. No SIMT VF dereferences
    // either -- normWBf16_ is cast into normWF32_ and outBf16_ is an MTE3
    // source -- but they come from the allocator anyway, because one
    // mechanism per kernel is the rule that makes the offsets above mean
    // anything.
    normWBf16_ = alloc.Alloc<bfloat16_t>(kHeadDim);
    outBf16_ = alloc.Alloc<bfloat16_t>(kHeadDim);
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
        reinterpret_cast<__ubuf__ float*>(apeF32_.GetPhyAddr()), slot,
        reinterpret_cast<__ubuf__ float*>(kvCur_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(gateCur_.GetPhyAddr()));
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
        reinterpret_cast<__ubuf__ float*>(kvCur_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(gateCur_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(prevKvA_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(prevGateA_.GetPhyAddr()),
        have_previous,
        reinterpret_cast<__ubuf__ float*>(normWF32_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(ropeCos_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(ropeSin_.GetPhyAddr()),
        norm_eps_,
        reinterpret_cast<__ubuf__ float*>(outF32_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(reduce_.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();

    Cast(outBf16_, outF32_, RoundMode::CAST_RINT, kHeadDim);
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(cacheSlotGm_, outBf16_, kHeadDim);

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

    // A LocalMemAllocator LocalTensor is an ordinary UB tensor as far as the
    // SIMD side is concerned, so one allocation serves both the DataCopy here
    // and the SIMT VF that dereferences it -- no separate staging copy.
    DataCopy(kvCur_, stateKvGm_, kKvCurRegionFloats);
    DataCopy(gateCur_, stateGateGm_, kGateCurRegionFloats);
    DataCopy(prevKvA_, statePrevKvGm_, kPrevARegionFloats);
    DataCopy(prevGateA_, statePrevGateGm_, kPrevARegionFloats);
    DataCopy(apeF32_, apeGm_,
             kRatio * kStateWidth);  // the [4,1024] ape block, resident
    DataCopy(normWBf16_, normWGm_, kHeadDim);
    DataCopy(ropeCos_, ropeCosGm_, kRopeHeadDim);
    DataCopy(ropeSin_, ropeSinGm_, kRopeHeadDim);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    // The widened norm gain is consumed by the emit launch.
    Cast(normWF32_, normWBf16_, RoundMode::CAST_NONE, kHeadDim);
  }

  __aicore__ inline void WritebackRegions(bool with_previous) {
    // RAW guard: the four state regions were last written by a SIMT VF, which
    // issues on the Vector pipeline, and MTE3 is a different queue. The
    // canonical choreography is "MTE3 waits V_MTE3 before draining a vector
    // result to GM" -- DataSyncBarrier<ALL> is scope-limited data visibility
    // and is explicitly not a compute fence, so it does not stand in for this
    // pairing. Both hybrid kernels that run correctly on hardware
    // (mxfp4_expert_gemv, fused_mhc_pre_norm_sinkhorn) pair the barrier with
    // this flag before DMA-ing VF-written UB out; this one did not.
    //
    // The HOLD path is where that mattered. Steps 0..2 of a group project into
    // kvCur_/gateCur_ inside CompressorProjectSimt and then drain both regions
    // here with no V->MTE3 dependency, so the writeback could read UB the VF
    // had not finished writing and persist a stale row. The EMIT at step 3
    // reloads exactly those rows as window rows 4..6, so a cold-start group 0
    // reduced partly-unwritten b halves -- a direction that is correlated with
    // the golden but wrong (the observed cos 0.576) rather than garbage. The
    // roll then carried the same damage into prev_kv_a / prev_gate_a, which is
    // why group 1 was worse (0.379) despite taking the identical code path.
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    // WAR guard: every state MTE2 read must drain before the MTE3 writeback.
    SetFlag<HardEvent::MTE2_MTE3>(kEvMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(kEvMte2ToMte3);
    DataCopy(stateKvGm_, kvCur_, kKvCurRegionFloats);
    DataCopy(stateGateGm_, gateCur_, kGateCurRegionFloats);
    if (with_previous) {
      DataCopy(statePrevKvGm_, prevKvA_, kPrevARegionFloats);
      DataCopy(statePrevGateGm_, prevGateA_, kPrevARegionFloats);
    }
  }

  std::uint32_t step_;
  float norm_eps_;

  // Dynamic region: every buffer this kernel has. No TPipe -- see Init() for
  // why that is a correctness requirement and not a style choice. Allocated
  // in Init() in the order the tiling header documents.
  LocalTensor<float> kvCur_;
  LocalTensor<float> gateCur_;
  LocalTensor<float> prevKvA_;
  LocalTensor<float> prevGateA_;
  LocalTensor<float> apeF32_;
  LocalTensor<float> normWF32_;
  LocalTensor<float> ropeCos_;
  LocalTensor<float> ropeSin_;
  LocalTensor<float> outF32_;
  LocalTensor<float> reduce_;
  LocalTensor<bfloat16_t> normWBf16_;
  LocalTensor<bfloat16_t> outBf16_;

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
