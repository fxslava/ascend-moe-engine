/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// KER-01, fused_mhc_pre_norm_sinkhorn, Ascend 950PR / arch35, AIV only.
//
// MATH (docs/kernel_contracts_spec.md KER-01; oracle mhc_pre_sinkhorn_cpu
// plus the caller's second RMSNorm in src/emu/semi_e2e_pipeline.cpp)
//
//   r            = rsqrt(mean(flatten(x)^2) + bundle_norm_eps)
//   u[k]         = (phi[k] . flatten(x)) * r                      k in [0, 24)
//   pre[i]       = sigmoid(u[i]     * alpha[0] + bias[i])     + hc_eps
//   post[i]      = 2 * sigmoid(u[4+i] * alpha[1] + bias[4+i])
//   logits[i, j] = u[8+4i+j] * alpha[2] + bias[8+4i+j]
//   B            = sinkhorn_knopp_4x4(logits, hc_eps, 20)
//   h_raw[d]     = sum_i pre[i] * x[i, d]
//   h_norm[d]    = BF16(gamma[d] * h_raw[d] * rsqrt(mean(h_raw^2) + act_eps))
//
// TWO THINGS THAT ARE EASY TO GET WRONG AND SILENT IF WRONG
//
//  1. THE RMS SCALE IS APPLIED TO THE PROJECTION, NOT TO x. Scaling x first
//     and projecting after gives the same number only because phi is linear
//     and r is a scalar, but the golden's order is this one and it is the
//     order that keeps the pre-norm projection available as a trace output.
//
//  2. THERE ARE TWO DISTINCT RMSNorms. The bundle norm above feeds the gates;
//     the activation norm with `gamma` is fused onto the END, after the
//     four-stream h_raw reduction. Hoisting the second one ahead of that
//     reduction is a different function and nothing downstream flags it.
//
// SINKHORN EPSILON CADENCE: hc_eps is added ONCE, to every entry right after
// the initial row softmax. From then on it appears only as a `/(sum + eps)`
// division guard. Adding it after every step instead drifts the row sums by
// num_iters * n * eps and breaks the Birkhoff bound. See SinkhornKnopp4x4.
//
// ARITHMETIC PLACEMENT
//   Only the 4096-lane streams run on the Vector unit. The 24 projection
//   results, the gate arithmetic and all 40 Sinkhorn normalizations are
//   scalar: they are a handful of values, scalar FP32 add/multiply/divide is
//   exact, and it reproduces the oracle's operation order exactly. The one
//   thing scalar code on this device CANNOT do is exp() -- there is no scalar
//   libm -- so each exponential is staged through a short vector Exp.
//
//   The small GM operands (alpha 12 B, post 16 B, B 64 B) are read and
//   written with GlobalTensor Get/SetValue rather than DataCopy: they are not
//   multiples of the 32-byte access block, so a DataCopy would read or write
//   past their declared extents.
//
// GM scratch workspace = 0 bytes. One AIV, no core barrier, no GM partials.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "arch35_ub_layout.h"
#include "moe/kernels/arch35/kernel_contracts.h"

// >>> BEGIN GENERATED UB LAYOUT ANNOTATIONS -- scripts/gen_arch35_ub_layout.py
// Ker01 static UB layout: 150112 B of 221184 B usable (67.9%).
// These restore the analyzer's UB capacity, alignment, aliasing and bank
// checks, which it otherwise skips for static-mode LocalTensor offsets.
// @ascend-layout: name=xf32 pos=VECCALC offset=0 count=16384 dtype=float
// @ascend-layout: name=phi_ping pos=VECCALC offset=65568 count=4096 dtype=float
// @ascend-layout: name=prod pos=VECCALC offset=81984 count=4096 dtype=float
// @ascend-layout: name=phi_pong pos=VECCALC offset=98400 count=4096 dtype=float
// @ascend-layout: name=h_raw pos=VECCALC offset=114816 count=4096 dtype=float
// @ascend-layout: name=x_bf16 pos=VECCALC offset=131232 count=4096 dtype=bfloat16_t
// @ascend-layout: name=h_norm_bf pos=VECCALC offset=139456 count=4096 dtype=bfloat16_t
// @ascend-layout: name=red_tmp pos=VECCALC offset=147680 count=512 dtype=float
// @ascend-layout: name=red_dst pos=VECCALC offset=149760 count=8 dtype=float
// @ascend-layout: name=u_all pos=VECCALC offset=149792 count=32 dtype=float
// @ascend-layout: name=pre pos=VECCALC offset=150080 count=8 dtype=float
// <<< END GENERATED UB LAYOUT ANNOTATIONS

using namespace AscendC;

namespace {

namespace ub = moe::kernels::arch35::ker01_ub;

constexpr uint32_t kStreams = moe::kernels::arch35::kHcStreams;            // 4
constexpr uint32_t kHidden = moe::kernels::arch35::kHiddenSize;            // 4096
constexpr uint32_t kRows = moe::kernels::arch35::kHcProjectionRows;        // 24
constexpr uint32_t kIters = moe::kernels::arch35::kHcSinkhornIters;        // 20
constexpr uint32_t kChunk = 4096U;                                        // phi tile
constexpr uint32_t kChunksPerRow = moe::kernels::arch35::kHcFlatWidth / kChunk;  // 4
constexpr uint32_t kTotalChunks = kRows * kChunksPerRow;                   // 96
constexpr float kSigmoidClamp = 80.0F;  // the golden clips before exp()

// One SIMT thread per Birkhoff entry: 4x4 = 16 lanes, tid = 4*row + col.
constexpr uint32_t kSinkhornLanes = kStreams * kStreams;  // 16

}  // namespace

// ---------------------------------------------------------------------------
// SIMT: the three gate segments and the whole Sinkhorn solve, in registers.
//
// 16 threads, thread tid owning Birkhoff entry B[tid/4][tid%4]. The solve is
// 40 normalizations over 16 values; in SIMD that is 40 horizontal reductions
// of a 4x4 tile with a PipeBarrier between each, and in scalar code it is
// 640 single-element UB accesses. As a warp it is a butterfly:
//
//   a ROW    is the 4 lanes that differ only in bits 0..1  -> xor 1, 2
//   a COLUMN is the 4 lanes that differ only in bits 2..3  -> xor 4, 8
//
// so a two-step shuffle leaves every lane holding its row's or column's full
// sum, and the value never leaves the register file.
//
// THE EPSILON CADENCE IS THE ORACLE'S, INCLUDING ITS ASYMMETRY
//   iteration 0        row softmax (max-subtracted), then `+ hc_eps` on every
//                      entry ONCE, then a single column normalization
//   iterations 1..19   a row normalization, then a column one
// From the initial `+ hc_eps` onward epsilon appears only as a `/(sum + eps)`
// division guard. Adding it after every step drifts the row sums by
// num_iters * n * eps and breaks the Birkhoff bound.
//
// The last operation is always a COLUMN normalization, which is why columns
// sum to 1 tightly and rows only approximately.
// ---------------------------------------------------------------------------
__simt_vf__ inline void MhcGateSinkhornVf(
    __ubuf__ float* u_all,        // [24] projection results, already RMS-scaled
    __gm__ float* alpha,          // [3]
    __gm__ float* bias,           // [24]
    __ubuf__ float* pre_out,      // [4] h_pre gates
    __gm__ float* post_out,       // [4] h_post gains
    __gm__ float* birkhoff_out,   // [16] row-major B
    float hc_eps) {
  const uint32_t tid = threadIdx.x;
  if (tid >= kSinkhornLanes) {
    return;
  }

  // --- the 4x4 logit this lane owns -------------------------------------
  const float alpha2 = alpha[2];
  const uint32_t logit_row = 2U * kStreams + tid;  // rows [8, 24) of the projection
  const float logit = u_all[logit_row] * alpha2 + bias[logit_row];

  // --- initial row softmax, max-subtracted ------------------------------
  float maximum = logit;
  float partner = asc_shfl_xor(maximum, 1, static_cast<int32_t>(kSinkhornLanes));
  maximum = (partner > maximum) ? partner : maximum;
  partner = asc_shfl_xor(maximum, 2, static_cast<int32_t>(kSinkhornLanes));
  maximum = (partner > maximum) ? partner : maximum;

  float value = expf(logit - maximum);
  float row_sum = value;
  row_sum += asc_shfl_xor(row_sum, 1, static_cast<int32_t>(kSinkhornLanes));
  row_sum += asc_shfl_xor(row_sum, 2, static_cast<int32_t>(kSinkhornLanes));
  // The max subtraction guarantees one lane per row is exp(0) == 1, so
  // row_sum >= 1 and this division needs no guard. This is the ONE additive
  // epsilon of the whole solve.
  value = value / row_sum + hc_eps;

  // --- the first column normalization -----------------------------------
  float col_sum = value;
  col_sum += asc_shfl_xor(col_sum, 4, static_cast<int32_t>(kSinkhornLanes));
  col_sum += asc_shfl_xor(col_sum, 8, static_cast<int32_t>(kSinkhornLanes));
  value = value / (col_sum + hc_eps);

  // --- the remaining 19 row/column pairs --------------------------------
  for (uint32_t iteration = 1U; iteration < kIters; ++iteration) {
    float r = value;
    r += asc_shfl_xor(r, 1, static_cast<int32_t>(kSinkhornLanes));
    r += asc_shfl_xor(r, 2, static_cast<int32_t>(kSinkhornLanes));
    value = value / (r + hc_eps);

    float c = value;
    c += asc_shfl_xor(c, 4, static_cast<int32_t>(kSinkhornLanes));
    c += asc_shfl_xor(c, 8, static_cast<int32_t>(kSinkhornLanes));
    value = value / (c + hc_eps);
  }
  birkhoff_out[tid] = value;

  // --- the two 4-wide gates ride along on lanes 0..3 --------------------
  //
  //   pre[i]  = sigmoid(u[i]   * alpha[0] + bias[i])     + hc_eps
  //   post[i] = 2 * sigmoid(u[4+i] * alpha[1] + bias[4+i])
  //
  // The sigmoids and the factor 2 are not decoration: pre is a gate in
  // (0, 1] + eps and post is a gain in (0, 2). Using the raw projection rows
  // instead gives an unbounded residual map that nothing downstream flags.
  if (tid < kStreams) {
    const float alpha0 = alpha[0];
    const float alpha1 = alpha[1];
    float pre_arg = u_all[tid] * alpha0 + bias[tid];
    float post_arg = u_all[kStreams + tid] * alpha1 + bias[kStreams + tid];
    // The oracle clips before exp() so a large negative logit cannot become
    // inf/0; the clamp saturates, matching numpy.
    pre_arg = (pre_arg > kSigmoidClamp) ? kSigmoidClamp
                                        : ((pre_arg < -kSigmoidClamp) ? -kSigmoidClamp : pre_arg);
    post_arg = (post_arg > kSigmoidClamp) ? kSigmoidClamp
                                          : ((post_arg < -kSigmoidClamp) ? -kSigmoidClamp : post_arg);
    pre_out[tid] = 1.0F / (1.0F + expf(-pre_arg)) + hc_eps;
    post_out[tid] = 2.0F / (1.0F + expf(-post_arg));
  }
}

extern "C" __global__ __aicore__ void ker01_mhc_pre_norm_sinkhorn(
    GM_ADDR x_gm, GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm,
    GM_ADDR gamma_gm, GM_ADDR h_raw_gm, GM_ADDR h_norm_gm, GM_ADDR post_gm,
    GM_ADDR birkhoff_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  // ReduceSum's count form on dav-3510 allocates an internal V_S event through
  // GetTPipePtr(), so a TPipe must exist even though this kernel uses the
  // static tensor model and never allocates a TQue from it. Without it
  // GetTPipePtr() is null and the reduction segfaults -- which bisheng does
  // not diagnose, because the dereference happens inside the library.
  TPipe pipe;
  (void)pipe;
  (void)workspace_gm;  // GM scratch contract is 0 bytes

  const __gm__ moe::kernels::arch35::MhcPreNormTilingData* tiling =
      reinterpret_cast<const __gm__ moe::kernels::arch35::MhcPreNormTilingData*>(
          tiling_gm);
  const uint32_t x_stream_stride = tiling->x_stream_stride;
  const uint32_t phi_row_stride = tiling->phi_row_stride;
  const float hc_eps = tiling->hc_eps;
  const float bundle_norm_eps = tiling->bundle_norm_eps;
  const float activation_norm_eps = tiling->activation_norm_eps;

  GlobalTensor<bfloat16_t> x_global;
  GlobalTensor<bfloat16_t> gamma_global;
  GlobalTensor<bfloat16_t> h_norm_global;
  GlobalTensor<float> phi_global;
  GlobalTensor<float> alpha_global;
  GlobalTensor<float> bias_global;
  GlobalTensor<float> h_raw_global;
  GlobalTensor<float> post_global;
  GlobalTensor<float> birkhoff_global;
  x_global.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x_gm));
  gamma_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(gamma_gm));
  h_norm_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(h_norm_gm));
  phi_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(phi_gm));
  alpha_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha_gm));
  bias_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias_gm));
  h_raw_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_raw_gm));
  post_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(post_gm));
  birkhoff_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(birkhoff_gm));

  LocalTensor<float> xf32(TPosition::VECCALC, ub::kXf32Off, ub::kXf32Count);
  LocalTensor<float> phi_ping(TPosition::VECCALC, ub::kPhiPingOff, ub::kPhiPingCount);
  LocalTensor<float> prod(TPosition::VECCALC, ub::kProdOff, ub::kProdCount);
  LocalTensor<float> phi_pong(TPosition::VECCALC, ub::kPhiPongOff, ub::kPhiPongCount);
  LocalTensor<float> h_raw(TPosition::VECCALC, ub::kHrawOff, ub::kHrawCount);
  LocalTensor<bfloat16_t> x_bf16(
      TPosition::VECCALC, ub::kXbf16Off, ub::kXbf16Count);
  LocalTensor<bfloat16_t> h_norm_bf(
      TPosition::VECCALC, ub::kHnormBfOff, ub::kHnormBfCount);
  LocalTensor<float> u_all(TPosition::VECCALC, ub::kUAllOff, ub::kUAllCount);
  LocalTensor<float> pre(TPosition::VECCALC, ub::kPreOff, ub::kPreCount);
  LocalTensor<float> red_tmp(TPosition::VECCALC, ub::kRedtmpOff, ub::kRedtmpCount);
  LocalTensor<float> red_dst(TPosition::VECCALC, ub::kRedDstOff, ub::kRedDstCount);

  // =========================================================================
  // Phase A -- widen x once. Every later stage reads it at least twice, and
  // widening once is the only way to keep those reads bit-identical.
  // =========================================================================
  SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
  for (uint32_t stream = 0U; stream < kStreams; ++stream) {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    DataCopy(x_bf16, x_global[stream * x_stream_stride], kHidden);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(xf32[stream * kHidden], x_bf16, RoundMode::CAST_NONE, kHidden);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

  // =========================================================================
  // Phase B -- the bundle inverse RMS over the FLATTENED stream.
  // ReduceSum is a tree reduction, so the 16384-term sum keeps far more
  // precision than a sequential FP32 accumulation would.
  // =========================================================================
  float sum_squares = 0.0F;
  for (uint32_t chunk = 0U; chunk < kChunksPerRow; ++chunk) {
    Mul(prod, xf32[chunk * kChunk], xf32[chunk * kChunk], kChunk);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red_dst, prod, red_tmp, kChunk);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    sum_squares += red_dst.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);
  }
  const float bundle_mean =
      sum_squares / static_cast<float>(moe::kernels::arch35::kHcFlatWidth);
  const float inv_rms = 1.0F / sqrt(bundle_mean + bundle_norm_eps);

  // =========================================================================
  // Phase C -- phi @ flatten(x), 24 rows of 4 chunks, two-stage pipelined.
  //
  // Each half-iteration issues the NEXT chunk's load before consuming the
  // current one, so MTE2 and Vector overlap. The ping/pong selection is
  // UNROLLED -- two chunks per loop iteration, each with a literal event id --
  // because a runtime-selected event id defeats static pairing analysis
  // (AKA3003) even when the pairing is in fact balanced.
  //
  // The last half-iteration re-reads chunk 0 into the spare buffer. That one
  // redundant 16 KiB transfer keeps every iteration's flag counts identical,
  // which is what the loop-token balance check (AKA2006) requires, and costs
  // 16 KiB against the 1.5 MiB this phase already streams.
  // =========================================================================
  SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID3);

  WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
  DataCopy(phi_ping, phi_global[0], kChunk);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

  float row_dot = 0.0F;
  for (uint32_t pair = 0U; pair < (kTotalChunks / 2U); ++pair) {
    const uint32_t k_ping = pair * 2U;
    const uint32_t k_pong = k_ping + 1U;

    // --- prefetch k_pong into pong, then consume ping ---------------------
    const uint32_t pong_row = k_pong / kChunksPerRow;
    const uint32_t pong_chunk = k_pong % kChunksPerRow;
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    DataCopy(phi_pong, phi_global[pong_row * phi_row_stride + pong_chunk * kChunk],
             kChunk);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    Mul(prod, phi_ping, xf32[(k_ping % kChunksPerRow) * kChunk], kChunk);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red_dst, prod, red_tmp, kChunk);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    row_dot += red_dst.GetValue(0);
    if ((k_ping % kChunksPerRow) == (kChunksPerRow - 1U)) {
      u_all.SetValue(k_ping / kChunksPerRow, row_dot * inv_rms);
      row_dot = 0.0F;
    }
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);

    // --- prefetch the next pair's ping chunk, then consume pong -----------
    const uint32_t k_next = (k_pong + 1U) % kTotalChunks;
    const uint32_t next_row = k_next / kChunksPerRow;
    const uint32_t next_chunk = k_next % kChunksPerRow;
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    DataCopy(phi_ping, phi_global[next_row * phi_row_stride + next_chunk * kChunk],
             kChunk);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    Mul(prod, phi_pong, xf32[pong_chunk * kChunk], kChunk);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red_dst, prod, red_tmp, kChunk);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);

    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    row_dot += red_dst.GetValue(0);
    if (pong_chunk == (kChunksPerRow - 1U)) {
      u_all.SetValue(pong_row, row_dot * inv_rms);
      row_dot = 0.0F;
    }
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);
  }
  // The final half-iteration prefetched the wrap-around chunk into ping and
  // released pong, so exactly one token is outstanding on each.
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);

  // =========================================================================
  // Phase D/E -- gates and Sinkhorn, as one 16-thread SIMT launch.
  //
  // The Vector ALU is physically shared between the SIMD and SIMT paths and a
  // mode switch drains the active vector pipeline, so the call is fenced on
  // both sides. One switch covers all 24 gate values and all 40 Sinkhorn
  // normalizations, which is the opposite of the sub-tile alternation that
  // makes VF switching a loss.
  //
  // alpha and bias are read straight from GM by the warp: they are 12 and 96
  // bytes, which is below the 32-byte-block DMA granularity for alpha, and a
  // SIMT thread can address GM directly through the Data Cache.
  // =========================================================================
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);
  PipeBarrier<PIPE_V>();
  asc_vf_call<MhcGateSinkhornVf>(
      dim3(kSinkhornLanes, 1, 1),
      reinterpret_cast<__ubuf__ float*>(u_all.GetPhyAddr()),
      reinterpret_cast<__gm__ float*>(alpha_gm),
      reinterpret_cast<__gm__ float*>(bias_gm),
      reinterpret_cast<__ubuf__ float*>(pre.GetPhyAddr()),
      reinterpret_cast<__gm__ float*>(post_gm),
      reinterpret_cast<__gm__ float*>(birkhoff_gm), hc_eps);
    // ascend_analyzer has no asc_vf_call in its AST, so it cannot see the SIMT
  // launch between these two fences and reports the second as redundant.
  // Both are required: the Vector ALU is shared between the SIMD and SIMT
  // paths, so the SIMD pipeline must be drained on entry to and exit from
  // the VF. See ISSUES_AND_GAPS.md.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();

  // The four h_pre gates come back through UB because the stream reduction
  // below needs them as Muls scalars.
  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);
  float pre_gate[kStreams];
  for (uint32_t i = 0U; i < kStreams; ++i) {
    pre_gate[i] = pre.GetValue(i);
  }

  // =========================================================================
  // Phase F -- h_raw, the pre-gated sum of the four streams.
  // =========================================================================
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);
  Muls(h_raw, xf32[0], pre_gate[0], kHidden);
  PipeBarrier<PIPE_V>();
  for (uint32_t stream = 1U; stream < kStreams; ++stream) {
    Muls(prod, xf32[stream * kHidden], pre_gate[stream], kHidden);
    PipeBarrier<PIPE_V>();
    Add(h_raw, h_raw, prod, kHidden);
    PipeBarrier<PIPE_V>();
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(h_raw_global, h_raw, kHidden);

  // =========================================================================
  // Phase G -- the activation RMSNorm, fused after the reduction, with gamma.
  // =========================================================================
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  Mul(prod, h_raw, h_raw, kHidden);
  PipeBarrier<PIPE_V>();
  ReduceSum<float>(red_dst, prod, red_tmp, kHidden);
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);
  const float act_mean = red_dst.GetValue(0) / static_cast<float>(kHidden);
  const float act_inv_rms = 1.0F / sqrt(act_mean + activation_norm_eps);
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);

  // gamma reuses the BF16 load tile; `prod` is free again after the reduction.
  SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
  DataCopy(x_bf16, gamma_global, kHidden);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Cast(prod, x_bf16, RoundMode::CAST_NONE, kHidden);
  PipeBarrier<PIPE_V>();
  Mul(prod, prod, h_raw, kHidden);
  PipeBarrier<PIPE_V>();
  Muls(prod, prod, act_inv_rms, kHidden);
  PipeBarrier<PIPE_V>();
  // CAST_RINT is round-to-nearest-even, the declared BF16 store rule.
  Cast(h_norm_bf, prod, RoundMode::CAST_RINT, kHidden);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(h_norm_global, h_norm_bf, kHidden);
}
