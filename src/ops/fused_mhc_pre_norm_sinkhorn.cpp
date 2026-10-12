// fused_mhc_pre_norm_sinkhorn.cpp -- fused_mhc_pre_norm_sinkhorn device kernel (Ascend 950PR,
// arch35 / dav-3510, single AIV, hybrid SIMD/SIMT).
//
// Math (docs/kernel_contracts_spec.md, oracle src/emu/mhc_fused_sinkhorn_gemv.cpp):
//   r        = rsqrt(mean(flatten(x)^2) + 1e-6)          over all 16,384 lanes
//   u[row]   = (phi @ flatten(x))[row] * r               24 rows
//   pre[i]   = sigmoid(u[i]     * alpha[0] + bias[i])     + hc_eps   i < 4
//   post[i]  = 2 * sigmoid(u[4+i] * alpha[1] + bias[4+i])           i < 4
//   logits   = u[8+t] * alpha[2] + bias[8+t]              t < 16
//   B        = sinkhorn_knopp_4x4(logits, 20 iters, hc_eps)
//   h_raw[d] = sum_i pre[i] * x[i,d]
//   h_norm   = BF16(gamma * h_raw * rsqrt(mean(h_raw^2) + 1e-6))
//
// The Sinkhorn cadence is the golden's exactly: one additive hc_eps after
// the initial row softmax, then hc_eps appears only as the division guard
// /(sum + hc_eps); 20 row steps and 20 column steps, last op a column one.
// The 4x4 matrix never leaves registers: 16 SIMT threads hold one element
// each and exchange through asc_shfl_xor with row butterfly masks 1,2 and
// column masks 4,8 across all 20 iterations. The three gate segments (the
// two clamped sigmoids and the logits) are folded into the same 16-thread
// launch so every transcendental runs SIMT-side; the SIMD phases see only
// vector intrinsics (Mul/Add/Muls/Cast/ReduceSum/Sqrt).
//
// GM scratch workspace: 0 bytes (the workspace argument is unused).

#include "moe/kernels/arch35/arch35_common.hpp"
#include "moe/kernels/arch35/fused_mhc_pre_norm_sinkhorn_ub_layout.hpp"
#include "moe/kernels/arch35/tiling_data.hpp"

namespace moe::kernels::arch35::fused_mhc_pre_norm_sinkhorn {

using namespace AscendC;

// The oracle's clamped sigmoid (np.clip(-80, 80) before exp). SIMT-only.
__simt_callee__ __aicore__ inline float SigmoidClamp(float value) {
  float clamped = value;
  if (clamped > 80.0f) {
    clamped = 80.0f;
  }
  if (clamped < -80.0f) {
    clamped = -80.0f;
  }
  return 1.0f / (1.0f + expf(-clamped));
}

// ---------------------------------------------------------------------------
// SIMT phase: gate segments + register Sinkhorn, 16 threads
// ---------------------------------------------------------------------------
// Threads 0..3 produce pre[], threads 4..7 produce post[] (stored straight
// to GM), every thread derives its own logits[t] in a register, and the
// whole 4x4 Sinkhorn solve stays in registers: thread t holds B[t/4][t%4];
// row groups are the aligned 4-lane runs (xor masks 1,2), column groups are
// the strided lanes {r, r+4, r+8, r+12} (xor masks 4,8); width 16 keeps
// every exchange inside this thread block.
__simt_vf__ __aicore__ __launch_bounds__(16) inline void GatesAndSinkhorn16(
    __ubuf__ const float* u,                 // 24 normalized projections
    __gm__ const float* alpha,               // [3]
    __gm__ const float* bias,                // [24]
    float eps, std::int32_t num_iters,
    __ubuf__ float* pre_out,                 // [4], consumed by the epilogue
    __gm__ float* post_out,                  // [4]
    __gm__ float* birkhoff_out) {            // [16] row-major
  const std::uint32_t t = threadIdx.x;

  const float alpha0 = asc_ldcg(const_cast<__gm__ float*>(alpha) + 0);
  const float alpha1 = asc_ldcg(const_cast<__gm__ float*>(alpha) + 1);
  const float alpha2 = asc_ldcg(const_cast<__gm__ float*>(alpha) + 2);
  // The 24 bias rows are three segments; t < 8 indexes its own gate row,
  // every thread indexes the logits segment at 8 + t.
  const float bias_gate = asc_ldcg(const_cast<__gm__ float*>(bias) + t);
  const float bias_logit = asc_ldcg(const_cast<__gm__ float*>(bias) +
                                    2 * kStreams + t);

  if (t < kStreams) {
    // pre[i] = sigmoid(u[i] * alpha0 + bias[i]) + eps
    pre_out[t] = SigmoidClamp(u[t] * alpha0 + bias_gate) + eps;
  } else if (t < 2 * kStreams) {
    // post[i] = 2 * sigmoid(u[4+i] * alpha1 + bias[4+i]), i = t - 4
    const float value =
        2.0f * SigmoidClamp(u[t] * alpha1 + bias_gate);
    asc_stcg(post_out + (t - kStreams), value);
  }

  // logits[t] = u[8+t] * alpha2 + bias[8+t], one per thread, in registers.
  float b = u[2 * kStreams + t] * alpha2 + bias_logit;

  // Initial row softmax, max-subtracted. The max entry contributes exp(0)=1,
  // so the sum guard only matters for NaN-bearing input, as in the oracle.
  float m = b;
  m = fmaxf(m, asc_shfl_xor(m, 1, 16));
  m = fmaxf(m, asc_shfl_xor(m, 2, 16));
  const float e = expf(b - m);
  float s = e;
  s += asc_shfl_xor(s, 1, 16);
  s += asc_shfl_xor(s, 2, 16);
  // The oracle's exact form: b = e * (1/sum) + eps, with the reciprocal
  // zeroed for a non-finite sum (NaN input), never a division by zero.
  const float inv = ((s > 0.0f) && isfinite(s)) ? (1.0f / s) : 0.0f;
  b = e * inv + eps;  // the ONE additive hc_eps, right after the softmax

  // First column normalization.
  float c = b;
  c += asc_shfl_xor(c, 4, 16);
  c += asc_shfl_xor(c, 8, 16);
  b /= (c + eps);

  // Nineteen more row/column pairs; every denominator is (sum + eps).
  for (std::int32_t it = 1; it < num_iters; ++it) {
    float r = b;
    r += asc_shfl_xor(r, 1, 16);
    r += asc_shfl_xor(r, 2, 16);
    b /= (r + eps);
    c = b;
    c += asc_shfl_xor(c, 4, 16);
    c += asc_shfl_xor(c, 8, 16);
    b /= (c + eps);
  }

  asc_stcg(birkhoff_out + t, b);  // row-major B[t/4][t%4]
}

class MhcPreNormSinkhorn {
 public:
  __aicore__ inline MhcPreNormSinkhorn(
      GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
      GM_ADDR h_raw, GM_ADDR h_norm, GM_ADDR post, GM_ADDR birkhoff,
      const __gm__ MhcPreNormTilingData* tiling) {
    // CCE address spaces forbid struct copy from __gm__; field-by-field.
    tiling_.abi_version = tiling->abi_version;
    tiling_.block_dim = tiling->block_dim;
    tiling_.hidden = tiling->hidden;
    tiling_.streams = tiling->streams;
    tiling_.projection_rows = tiling->projection_rows;
    tiling_.projection_tile = tiling->projection_tile;
    tiling_.projection_chunks = tiling->projection_chunks;
    tiling_.queue_depth = tiling->queue_depth;
    tiling_.sinkhorn_iterations = tiling->sinkhorn_iterations;
    tiling_.x_stream_stride = tiling->x_stream_stride;
    tiling_.phi_row_stride = tiling->phi_row_stride;
    tiling_.reserved = tiling->reserved;
    tiling_.hc_eps = tiling->hc_eps;
    tiling_.bundle_norm_eps = tiling->bundle_norm_eps;
    tiling_.activation_norm_eps = tiling->activation_norm_eps;
    tiling_.reserved_fp32 = tiling->reserved_fp32;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(phi));
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha), 3);
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias),
                            kProjectionRows);
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gamma),
                             kHidden);
    hRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_raw), kHidden);
    hNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(h_norm),
                             kHidden);
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(post), kStreams);
    birkhoffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(birkhoff), 16);
  }

  __aicore__ inline void Init() {
    pipe_.InitBuffer(xTileBf16Buf_, kXStreamTileBytes);
    for (std::uint32_t i = 0; i < kStreams; ++i) {
      pipe_.InitBuffer(xF32Bufs_[i], kXF32StreamBytes);
    }
    for (std::uint32_t i = 0; i < kQueueDepth; ++i) {
      pipe_.InitBuffer(phiTileBufs_[i], kPhiTileBytes);
    }
    pipe_.InitBuffer(productBuf_, kProductBytes);
    pipe_.InitBuffer(hRawBuf_, kHRawBytes);
    pipe_.InitBuffer(normBuf_, kNormBytes);
    pipe_.InitBuffer(gammaTileBuf_, kGammaTileBytes);
    pipe_.InitBuffer(normOutBuf_, kNormOutTileBytes);
    pipe_.InitBuffer(reduceBuf_, kReduceScratchBytes);
    pipe_.InitBuffer(scalarBuf_, kScalarStateBytes);
  }

  __aicore__ inline void Process() {
    LoadAndWidenX();
    const float inv_rms = BundleInvRms();
    float dots[kProjectionRows];
    ProjectionDots(dots);
    // u = (phi @ flat_x) * inv_rms, staged in UB for the SIMT phase.
    auto u = scalarBuf_.Get<float>();
    for (std::uint32_t row = 0; row < kProjectionRows; ++row) {
      u.SetValue(row, dots[row] * inv_rms);
    }
    RunSimtGatesAndSinkhorn(u);
    EmitNormedActivation();
  }

 private:
  // --- load the four BF16 streams and widen once ---------------------------
  __aicore__ inline void LoadAndWidenX() {
    auto xTile = xTileBf16Buf_.Get<bfloat16_t>();
    for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
      if (stream != 0) {
        // WAR: the MTE2 refill of xTile must wait for the previous Cast read.
        SetFlag<HardEvent::V_MTE2>(kEvVToMte2);
        WaitFlag<HardEvent::V_MTE2>(kEvVToMte2);
      }
      DataCopy(xTile, xGm_[stream * tiling_.x_stream_stride], kHidden);
      SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
      auto dst = xF32Bufs_[stream].Get<float>();
      Cast(dst, xTile, RoundMode::CAST_NONE, kHidden);
    }
  }

  // --- bundle RMS reciprocal over the flattened 16,384 lanes ---------------
  // A 1-lane IEEE vector Sqrt plus a plain scalar divide realizes the
  // oracle's exact 1/sqrt(mean + eps) without the approximate vector Rsqrt
  // and without the shadowed scalar libm names.
  __aicore__ inline float BundleInvRms() {
    auto product = productBuf_.Get<float>();
    auto reduceDst = reduceBuf_.Get<float>();
    float sum_squares = 0.0f;
    for (std::uint32_t stream = 0; stream < kStreams; ++stream) {
      auto src = xF32Bufs_[stream].Get<float>();
      Mul(product, src, src, kHidden);
      ReduceSum(reduceDst, product, reduceDst, kHidden);
      SetFlag<HardEvent::V_S>(kEvVToS);
      WaitFlag<HardEvent::V_S>(kEvVToS);
      sum_squares += reduceDst.GetValue(0);
    }
    const float mean_square = sum_squares / static_cast<float>(kFlat);
    Duplicate(product, mean_square + tiling_.bundle_norm_eps, 1);
    Sqrt(product, product, 1);
    SetFlag<HardEvent::V_S>(kEvVToS);
    WaitFlag<HardEvent::V_S>(kEvVToS);
    return 1.0f / product.GetValue(0);
  }

  // --- issue one 4,096-element phi chunk into its alternating tile ---------
  __aicore__ inline std::uint32_t IssuePhiChunk(std::uint32_t row,
                                                std::uint32_t chunk,
                                                std::uint32_t issued) {
    const std::uint32_t slot = issued % kQueueDepth;
    if (issued != 0) {
      // WAR guard: the oldest tile may still be under the vector unit's eyes.
      SetFlag<HardEvent::V_MTE2>(kEvVToMte2);
      WaitFlag<HardEvent::V_MTE2>(kEvVToMte2);
    }
    auto tile = phiTileBufs_[slot].Get<float>();
    DataCopy(tile,
             phiGm_[row * tiling_.phi_row_stride + chunk * kProjectionTile],
             kProjectionTile);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    return slot;
  }

  // --- 24 rows x 16,384 dot products, streamed in 4,096-element chunks -----
  // Chunk c of row r is phi[r][c*4096, (c+1)*4096) against stream c (the
  // flattened bundle is exactly four 4,096-lane streams), so the whole phi
  // matrix is read exactly once, 16 KiB at a time through two alternating
  // tiles driven one-at-a-time.
  __aicore__ inline void ProjectionDots(float* dots) {
    auto product = productBuf_.Get<float>();
    auto reduceDst = reduceBuf_.Get<float>();
    for (std::uint32_t row = 0; row < kProjectionRows; ++row) {
      dots[row] = 0.0f;
    }
    std::uint32_t issued = 0;
    for (std::uint32_t chunk = 0; chunk < kProjectionChunks; ++chunk) {
      auto xChunk = xF32Bufs_[chunk].Get<float>();
      for (std::uint32_t row = 0; row < kProjectionRows; ++row) {
        const std::uint32_t slot = IssuePhiChunk(row, chunk, issued);
        ++issued;
        WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);  // pair for the issue above
        auto phiTile = phiTileBufs_[slot].Get<float>();
        Mul(product, phiTile, xChunk, kProjectionTile);
        ReduceSum(reduceDst, product, reduceDst, kProjectionTile);
        SetFlag<HardEvent::V_S>(kEvVToS);
        WaitFlag<HardEvent::V_S>(kEvVToS);
        dots[row] += reduceDst.GetValue(0);
      }
    }
  }

  // --- SIMT launch: pre/post gates + the register Sinkhorn -----------------
  __aicore__ inline void RunSimtGatesAndSinkhorn(LocalTensor<float>& u) {
    // Scalar-produced factors consumed by SIMT: settle UB before the launch.
    DataSyncBarrier<MemDsbT::UB>();
    __ubuf__ float* uPtr = reinterpret_cast<__ubuf__ float*>(u.GetPhyAddr());
    // pre[4] lands right behind the 24 u slots in scalarState.
    __ubuf__ float* prePtr = uPtr + kProjectionRows;
    asc_vf_call<GatesAndSinkhorn16>(
        dim3(kStreams * kStreams), uPtr,
        const_cast<__gm__ float*>(
            reinterpret_cast<const __gm__ float*>(alphaGm_.GetPhyAddr())),
        const_cast<__gm__ float*>(
            reinterpret_cast<const __gm__ float*>(biasGm_.GetPhyAddr())),
        tiling_.hc_eps,
        static_cast<std::int32_t>(tiling_.sinkhorn_iterations), prePtr,
        const_cast<__gm__ float*>(
            reinterpret_cast<const __gm__ float*>(postGm_.GetPhyAddr())),
        const_cast<__gm__ float*>(
            reinterpret_cast<const __gm__ float*>(birkhoffGm_.GetPhyAddr())));
    DataSyncBarrier<MemDsbT::ALL>();
    pre_[0] = prePtr[0];  // plain UB reads on the scalar side of the core
    pre_[1] = prePtr[1];
    pre_[2] = prePtr[2];
    pre_[3] = prePtr[3];
    DataSyncBarrier<MemDsbT::UB>();
  }

  // --- h_raw, then the fused activation RMSNorm with gamma -----------------
  __aicore__ inline void EmitNormedActivation() {
    auto hRaw = hRawBuf_.Get<float>();
    auto product = productBuf_.Get<float>();
    // h_raw = sum_i pre[i] * x[i,:], accumulation order as the oracle's.
    Muls(hRaw, xF32Bufs_[0].Get<float>(), pre_[0], kHidden);
    for (std::uint32_t i = 1; i < kStreams; ++i) {
      Muls(product, xF32Bufs_[i].Get<float>(), pre_[i], kHidden);
      Add(hRaw, hRaw, product, kHidden);
    }
    // FP32 diagnostic copy of the raw sub-block input.
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(hRawGm_, hRaw, kHidden);

    // Activation RMSNorm over h_raw, then the gamma-weighted BF16 store.
    Mul(product, hRaw, hRaw, kHidden);
    auto reduceDst = reduceBuf_.Get<float>();
    ReduceSum(reduceDst, product, reduceDst, kHidden);
    SetFlag<HardEvent::V_S>(kEvVToS);
    WaitFlag<HardEvent::V_S>(kEvVToS);
    const float mean_square =
        reduceDst.GetValue(0) / static_cast<float>(kHidden);
    Duplicate(product, mean_square + tiling_.activation_norm_eps, 1);
    Sqrt(product, product, 1);
    SetFlag<HardEvent::V_S>(kEvVToS);
    WaitFlag<HardEvent::V_S>(kEvVToS);
    const float inv_rms = 1.0f / product.GetValue(0);

    auto gammaTile = gammaTileBuf_.Get<bfloat16_t>();
    DataCopy(gammaTile, gammaGm_, kHidden);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    Cast(product, gammaTile, RoundMode::CAST_NONE, kHidden);  // gamma FP32

    auto norm = normBuf_.Get<float>();
    auto normOut = normOutBuf_.Get<bfloat16_t>();
    Mul(norm, product, hRaw, kHidden);               // (gamma * h_raw) ...
    Muls(norm, norm, inv_rms, kHidden);              // ... * rsqrt(mean+eps)
    Cast(normOut, norm, RoundMode::CAST_RINT, kHidden);  // BF16, round-to-even
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(hNormGm_, normOut, kHidden);
  }

  MhcPreNormTilingData tiling_;
  float pre_[kStreams];

  TPipe pipe_;
  TBuf<> xTileBf16Buf_;
  TBuf<> xF32Bufs_[kStreams];
  TBuf<> phiTileBufs_[kQueueDepth];
  TBuf<> productBuf_;
  TBuf<> hRawBuf_;
  TBuf<> normBuf_;
  TBuf<> gammaTileBuf_;
  TBuf<> normOutBuf_;
  TBuf<> reduceBuf_;
  TBuf<> scalarBuf_;

  GlobalTensor<bfloat16_t> xGm_;
  GlobalTensor<float> phiGm_;
  GlobalTensor<float> alphaGm_;
  GlobalTensor<float> biasGm_;
  GlobalTensor<bfloat16_t> gammaGm_;
  GlobalTensor<float> hRawGm_;
  GlobalTensor<bfloat16_t> hNormGm_;
  GlobalTensor<float> postGm_;
  GlobalTensor<float> birkhoffGm_;
};

// GM_ADDR parameter order mirrors LaunchMhcPreNormSinkhorn (workspace unused:
// the contract's GM scratch for this kernel is 0 bytes).
extern "C" __global__ __aicore__ void fused_mhc_pre_norm_sinkhorn(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
    GM_ADDR h_raw, GM_ADDR h_norm, GM_ADDR post, GM_ADDR birkhoff,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  (void)workspace;
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ MhcPreNormTilingData* td =
      reinterpret_cast<const __gm__ MhcPreNormTilingData*>(tiling);
  // Reject unknown ABI / geometry rather than silently computing a wrong
  // shape; the host owns full validation before enqueue.
  if (unlikely(td->abi_version != kTilingAbiVersion ||
               td->hidden != kHidden || td->streams != kStreams ||
               td->projection_rows != kProjectionRows ||
               td->projection_tile != kProjectionTile ||
               td->projection_chunks != kProjectionChunks ||
               td->block_dim != 1 || td->reserved != 0)) {
    return;
  }
  MhcPreNormSinkhorn op(x, phi, alpha, bias, gamma, h_raw, h_norm, post,
                        birkhoff, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::fused_mhc_pre_norm_sinkhorn
