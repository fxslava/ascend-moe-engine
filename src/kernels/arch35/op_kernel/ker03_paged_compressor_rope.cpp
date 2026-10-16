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

// KER-03, fused_paged_compressor_rope (C4A), Ascend 950PR / arch35, AIV only.
//
// MATH (docs/kernel_contracts_spec.md KER-03; oracle
// paged_compressor_decode_c4a, the coff == 2 branch)
//
//   kv_state[slot]    = x @ wkv^T                  [coff*D] = [1024]
//   score_state[slot] = x @ wgate^T + ape[slot]    [coff*D] = [1024]
//   slot              = step % ratio
//
// Steps with slot != ratio-1 are HOLD: they update state and write
// emitted = 0, and they must leave every byte of cache_slot_gm untouched.
//
// On slot == ratio-1 an [8, 512] window is assembled as
//   rows 0..3  the PREVIOUS group's a-halves  (kv 0 and score -inf before the
//              first emission, so exp() gives those rows exactly zero weight)
//   rows 4..7  THIS group's b-halves, i.e. the UPPER half of each 1024 row
// then column-softmaxed down the 8 rows, used to weight the 8 KV rows, summed
// to 512, weighted-RMSNormed, and half-mode RoPEd over the last 64 lanes.
//
// A FOUR-ROW-ONLY WINDOW IS WRONG AFTER THE FIRST EMISSION. That is the single
// most likely way to get a plausible-looking but incorrect compressed cache:
// the result stays finite and normalized, it just ignores half its context.
//
// THE a/b SPLIT: the a half is lanes [0, 512) of each 1024-wide state row and
// the b half is lanes [512, 1024). Only b feeds the current window; a is
// rolled forward to become the next group's rows 0..3, and that roll happens
// AFTER the reduction so a caller inspecting state mid-step sees the rows the
// reduction actually used.
//
// RoPE IS HALF MODE OVER THE LAST 64 LANES, i.e. base = head_dim - 64 = 448,
// and the second output line reads the tables at half+i, not i. Rotating the
// FIRST 64 lanes instead is the other classic silent divergence here.
//
// GM scratch workspace = 0 bytes. The 49,168 B persistent state is an
// input/output allocation, not scratch. One AIV, no core barrier.

#include "kernel_operator.h"

#include "arch35_ub_layout.h"
#include "moe/kernels/arch35/kernel_contracts.h"

// >>> BEGIN GENERATED UB LAYOUT ANNOTATIONS -- scripts/gen_arch35_ub_layout.py
// Ker03 static UB layout: 125632 B of 221184 B usable (56.8%).
// These restore the analyzer's UB capacity, alignment, aliasing and bank
// checks, which it otherwise skips for static-mode LocalTensor offsets.
// @ascend-layout: name=xf32 pos=VECCALC offset=0 count=4096 dtype=float
// @ascend-layout: name=wrow_ping pos=VECCALC offset=16416 count=4096 dtype=bfloat16_t
// @ascend-layout: name=wrow_f32 pos=VECCALC offset=24640 count=4096 dtype=float
// @ascend-layout: name=wrow_pong pos=VECCALC offset=41056 count=4096 dtype=bfloat16_t
// @ascend-layout: name=proj pos=VECCALC offset=49280 count=1024 dtype=float
// @ascend-layout: name=x_bf16 pos=VECCALC offset=53408 count=4096 dtype=bfloat16_t
// @ascend-layout: name=win_kv pos=VECCALC offset=61632 count=4160 dtype=float
// @ascend-layout: name=win_sc pos=VECCALC offset=78304 count=4160 dtype=float
// @ascend-layout: name=weights pos=VECCALC offset=94976 count=4160 dtype=float
// @ascend-layout: name=col_max pos=VECCALC offset=111648 count=512 dtype=float
// @ascend-layout: name=col_sum pos=VECCALC offset=113728 count=512 dtype=float
// @ascend-layout: name=comp pos=VECCALC offset=115808 count=512 dtype=float
// @ascend-layout: name=comp_bf pos=VECCALC offset=117888 count=512 dtype=bfloat16_t
// @ascend-layout: name=norm_w_f32 pos=VECCALC offset=118944 count=512 dtype=float
// @ascend-layout: name=norm_w_bf pos=VECCALC offset=121024 count=512 dtype=bfloat16_t
// @ascend-layout: name=mask pos=VECCALC offset=122080 count=256 dtype=uint8_t
// @ascend-layout: name=rope_lo pos=VECCALC offset=122368 count=64 dtype=float
// @ascend-layout: name=rope_hi pos=VECCALC offset=122656 count=64 dtype=float
// @ascend-layout: name=rope_cos pos=VECCALC offset=122944 count=64 dtype=float
// @ascend-layout: name=rope_sin pos=VECCALC offset=123232 count=64 dtype=float
// @ascend-layout: name=red_tmp pos=VECCALC offset=123520 count=512 dtype=float
// @ascend-layout: name=red_dst pos=VECCALC offset=125600 count=8 dtype=float
// <<< END GENERATED UB LAYOUT ANNOTATIONS

using namespace AscendC;

namespace {

namespace ub = moe::kernels::arch35::ker03_ub;

constexpr uint32_t kHidden = moe::kernels::arch35::kHiddenSize;        // 4096
constexpr uint32_t kHeadDim = moe::kernels::arch35::kCompressorHeadDim;  // 512
constexpr uint32_t kCoff = moe::kernels::arch35::kCompressorCoff;      // 2
constexpr uint32_t kRatio = moe::kernels::arch35::kCompressorRatio;    // 4
constexpr uint32_t kWindowRows = moe::kernels::arch35::kCompressorRows;  // 8
constexpr uint32_t kRopeDim = moe::kernels::arch35::kRopeHeadDim;      // 64
constexpr uint32_t kStateWidth = kCoff * kHeadDim;                     // 1024

// Skewed window row stride; see the layout generator for why 512 is wrong.
constexpr uint32_t kWinStride = 520U;

// State GM map, in FP32 elements (49,152 B total).
constexpr uint32_t kStateKvBase = 0U;              // [0, 4096)   current KV [4, 1024]
constexpr uint32_t kStateScoreBase = 4096U;        // [4096, 8192) current gate
constexpr uint32_t kStatePrevKvBase = 8192U;       // [8192, 10240) previous a KV [4, 512]
constexpr uint32_t kStatePrevScoreBase = 10240U;   // [10240, 12288) previous a gate

// FP32's smallest normal. The golden zeroes any exp() result below it to match
// the device's flush-to-zero exponential; keeping the subnormal instead moves
// the softmax denominator in its last bits.
constexpr float kFp32Tiny = 1.17549435e-38F;

// A large negative stand-in for -inf on the pre-first-group score rows.
// exp(x - max) underflows to exactly zero for any finite max, which is the
// same answer -inf gives, without risking inf-inf = NaN inside the Sub.
constexpr float kNegInfProxy = -3.0e38F;

}  // namespace

extern "C" __global__ __aicore__ void ker03_paged_compressor_rope(
    GM_ADDR x_gm, GM_ADDR wkv_gm, GM_ADDR wgate_gm, GM_ADDR ape_gm,
    GM_ADDR norm_weight_gm, GM_ADDR rope_cos_gm, GM_ADDR rope_sin_gm,
    GM_ADDR state_gm, GM_ADDR state_meta_gm, GM_ADDR cache_slot_gm,
    GM_ADDR emitted_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm) {
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

  const __gm__ moe::kernels::arch35::CompressorTilingData* tiling =
      reinterpret_cast<const __gm__ moe::kernels::arch35::CompressorTilingData*>(
          tiling_gm);
  const uint32_t wkv_row_stride = tiling->wkv_row_stride;
  const uint32_t wgate_row_stride = tiling->wgate_row_stride;
  const uint32_t ape_row_stride = tiling->ape_row_stride;
  const float norm_eps = tiling->norm_eps;

  GlobalTensor<bfloat16_t> x_global;
  GlobalTensor<bfloat16_t> wkv_global;
  GlobalTensor<bfloat16_t> wgate_global;
  GlobalTensor<bfloat16_t> norm_weight_global;
  GlobalTensor<bfloat16_t> cache_slot_global;
  GlobalTensor<float> ape_global;
  GlobalTensor<float> rope_cos_global;
  GlobalTensor<float> rope_sin_global;
  GlobalTensor<float> state_global;
  GlobalTensor<uint32_t> state_meta_global;
  GlobalTensor<uint32_t> emitted_global;
  x_global.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x_gm));
  wkv_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(wkv_gm));
  wgate_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(wgate_gm));
  norm_weight_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(norm_weight_gm));
  cache_slot_global.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(cache_slot_gm));
  ape_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ape_gm));
  rope_cos_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rope_cos_gm));
  rope_sin_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rope_sin_gm));
  state_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(state_gm));
  state_meta_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(state_meta_gm));
  emitted_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(emitted_gm));

  LocalTensor<float> xf32(TPosition::VECCALC, ub::kXf32Off, ub::kXf32Count);
  LocalTensor<bfloat16_t> wrow_ping(
      TPosition::VECCALC, ub::kWrowPingOff, ub::kWrowPingCount);
  LocalTensor<float> wrow_f32(TPosition::VECCALC, ub::kWrowF32Off, ub::kWrowF32Count);
  LocalTensor<bfloat16_t> wrow_pong(
      TPosition::VECCALC, ub::kWrowPongOff, ub::kWrowPongCount);
  LocalTensor<float> proj(TPosition::VECCALC, ub::kProjOff, ub::kProjCount);
  LocalTensor<bfloat16_t> x_bf16(
      TPosition::VECCALC, ub::kXbf16Off, ub::kXbf16Count);
  LocalTensor<float> win_kv(TPosition::VECCALC, ub::kWinKvOff, ub::kWinKvCount);
  LocalTensor<float> win_sc(TPosition::VECCALC, ub::kWinScOff, ub::kWinScCount);
  LocalTensor<float> weights(TPosition::VECCALC, ub::kWeightsOff, ub::kWeightsCount);
  LocalTensor<float> col_max(TPosition::VECCALC, ub::kColMaxOff, ub::kColMaxCount);
  LocalTensor<float> col_sum(TPosition::VECCALC, ub::kColSumOff, ub::kColSumCount);
  LocalTensor<float> comp(TPosition::VECCALC, ub::kCompOff, ub::kCompCount);
  LocalTensor<bfloat16_t> comp_bf(
      TPosition::VECCALC, ub::kCompBfOff, ub::kCompBfCount);
  LocalTensor<float> norm_w_f32(TPosition::VECCALC, ub::kNormwF32Off,
                                ub::kNormwF32Count);
  LocalTensor<bfloat16_t> norm_w_bf(
      TPosition::VECCALC, ub::kNormwBfOff, ub::kNormwBfCount);
  LocalTensor<uint8_t> mask(TPosition::VECCALC, ub::kMaskOff, ub::kMaskCount);
  LocalTensor<float> rope_lo(TPosition::VECCALC, ub::kRopeLoOff, ub::kRopeLoCount);
  LocalTensor<float> rope_hi(TPosition::VECCALC, ub::kRopeHiOff, ub::kRopeHiCount);
  LocalTensor<float> rope_cos(TPosition::VECCALC, ub::kRopeCosOff, ub::kRopeCosCount);
  LocalTensor<float> rope_sin(TPosition::VECCALC, ub::kRopeSinOff, ub::kRopeSinCount);
  LocalTensor<float> red_tmp(TPosition::VECCALC, ub::kRedtmpOff, ub::kRedtmpCount);
  LocalTensor<float> red_dst(TPosition::VECCALC, ub::kRedDstOff, ub::kRedDstCount);

  // The step is the sequence's token counter; the host rejects a launch whose
  // declared step disagrees with it, so this is the single source of truth.
  const uint32_t step = state_meta_global.GetValue(1);
  const uint32_t have_previous = state_meta_global.GetValue(0);
  const uint32_t slot = step % kRatio;

  // =========================================================================
  // Phase A -- widen x once; every projection row reads it.
  // =========================================================================
  DataCopy(x_bf16, x_global, kHidden);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Cast(xf32, x_bf16, RoundMode::CAST_NONE, kHidden);
  PipeBarrier<PIPE_V>();

  // =========================================================================
  // Phase B -- the two projections, one BF16 weight row at a time.
  //
  // 2048 rows of 4096 BF16 stream through a ping/pong pair; the ping/pong
  // selection is unrolled so every event id is a literal (AKA3003). Each row
  // reduces to one scalar, which is why there is a V->S handshake per row:
  // the destination index varies per row and ReduceSum only ever writes
  // element 0. This is the correctness-first shape the contract specifies.
  // =========================================================================
  for (uint32_t matrix = 0U; matrix < 2U; ++matrix) {
    const uint32_t row_stride = (matrix == 0U) ? wkv_row_stride : wgate_row_stride;

    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    if (matrix == 0U) {
      DataCopy(wrow_ping, wkv_global[0], kHidden);
    } else {
      DataCopy(wrow_ping, wgate_global[0], kHidden);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    for (uint32_t pair = 0U; pair < (kStateWidth / 2U); ++pair) {
      const uint32_t row_ping = pair * 2U;
      const uint32_t row_pong = row_ping + 1U;

      // --- prefetch the pong row, then reduce the ping row ----------------
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
      if (matrix == 0U) {
        DataCopy(wrow_pong, wkv_global[row_pong * row_stride], kHidden);
      } else {
        DataCopy(wrow_pong, wgate_global[row_pong * row_stride], kHidden);
      }
      SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

      WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
      Cast(wrow_f32, wrow_ping, RoundMode::CAST_NONE, kHidden);
      PipeBarrier<PIPE_V>();
      Mul(wrow_f32, wrow_f32, xf32, kHidden);
      PipeBarrier<PIPE_V>();
      ReduceSum<float>(red_dst, wrow_f32, red_tmp, kHidden);
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      const float dot_ping = red_dst.GetValue(0);
      proj.SetValue(row_ping, dot_ping);
      SetFlag<HardEvent::S_V>(EVENT_ID1);
      WaitFlag<HardEvent::S_V>(EVENT_ID1);

      // --- prefetch the next pair's ping row, then reduce the pong row ----
      const uint32_t row_next = (row_pong + 1U) % kStateWidth;
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
      if (matrix == 0U) {
        DataCopy(wrow_ping, wkv_global[row_next * row_stride], kHidden);
      } else {
        DataCopy(wrow_ping, wgate_global[row_next * row_stride], kHidden);
      }
      SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

      WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
      Cast(wrow_f32, wrow_pong, RoundMode::CAST_NONE, kHidden);
      PipeBarrier<PIPE_V>();
      Mul(wrow_f32, wrow_f32, xf32, kHidden);
      PipeBarrier<PIPE_V>();
      ReduceSum<float>(red_dst, wrow_f32, red_tmp, kHidden);
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      const float dot_pong = red_dst.GetValue(0);
      proj.SetValue(row_pong, dot_pong);
      SetFlag<HardEvent::S_V>(EVENT_ID1);
      WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }
    // kStateWidth is even, so one token is outstanding on each phi event.
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);

    // ape is added to the GATE projection only, at the token's position
    // WITHIN the group, so it is a function of step % ratio and not of step.
    if (matrix == 1U) {
      SetFlag<HardEvent::S_V>(EVENT_ID1);
      WaitFlag<HardEvent::S_V>(EVENT_ID1);
      DataCopy(wrow_f32, ape_global[slot * ape_row_stride], kStateWidth);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Add(proj, proj, wrow_f32, kStateWidth);
      PipeBarrier<PIPE_V>();
    } else {
      SetFlag<HardEvent::S_V>(EVENT_ID1);
      WaitFlag<HardEvent::S_V>(EVENT_ID1);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const uint32_t state_base = (matrix == 0U) ? kStateKvBase : kStateScoreBase;
    DataCopy(state_global[state_base + slot * kStateWidth], proj, kStateWidth);
    // Two distinct hazards on this store, and they need different channels:
    // MTE3->V so the next matrix's Vector writes cannot clobber `proj` while
    // it is still draining, and MTE3->MTE2 because the window assembly below
    // READS these same state bytes back on the MTE2 pipe. MTE2 and MTE3 run
    // concurrently, so without the second pair the read races the write.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  }

  // Advance the token counter. Both projections are on GM at this point.
  state_meta_global.SetValue(1, step + 1U);

  // =========================================================================
  // HOLD -- slots 0..ratio-2 write emitted = 0 and must not touch the slot.
  // =========================================================================
  if (slot != (kRatio - 1U)) {
    emitted_global.SetValue(0, 0U);
    return;
  }

  // =========================================================================
  // Phase C -- assemble the [8, 512] window.
  //   rows 0..3  previous group's a-halves (or zero / -inf before the first)
  //   rows 4..7  this group's b-halves, lanes [512, 1024) of each state row
  // =========================================================================
  emitted_global.SetValue(0, 1U);
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);

  for (uint32_t r = 0U; r < kRatio; ++r) {
    if (have_previous != 0U) {
      DataCopy(win_kv[r * kWinStride], state_global[kStatePrevKvBase + r * kHeadDim],
               kHeadDim);
      DataCopy(win_sc[r * kWinStride],
               state_global[kStatePrevScoreBase + r * kHeadDim], kHeadDim);
    }
    DataCopy(win_kv[(kRatio + r) * kWinStride],
             state_global[kStateKvBase + r * kStateWidth + kHeadDim], kHeadDim);
    DataCopy(win_sc[(kRatio + r) * kWinStride],
             state_global[kStateScoreBase + r * kStateWidth + kHeadDim], kHeadDim);
  }
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

  if (have_previous == 0U) {
    // The golden's first-group initialization: KV zero, score -inf.
    for (uint32_t r = 0U; r < kRatio; ++r) {
      Duplicate(win_kv[r * kWinStride], 0.0F, kHeadDim);
      Duplicate(win_sc[r * kWinStride], kNegInfProxy, kHeadDim);
    }
    PipeBarrier<PIPE_V>();
  }

  // =========================================================================
  // Phase D -- column softmax DOWN the 8 rows, then the weighted reduction.
  // =========================================================================
  Adds(col_max, win_sc[0], 0.0F, kHeadDim);
  PipeBarrier<PIPE_V>();
  for (uint32_t r = 1U; r < kWindowRows; ++r) {
    Max(col_max, col_max, win_sc[r * kWinStride], kHeadDim);
    PipeBarrier<PIPE_V>();
  }
  Duplicate(col_sum, 0.0F, kHeadDim);
  PipeBarrier<PIPE_V>();
  for (uint32_t r = 0U; r < kWindowRows; ++r) {
    Sub(weights[r * kWinStride], win_sc[r * kWinStride], col_max, kHeadDim);
    PipeBarrier<PIPE_V>();
    Exp(weights[r * kWinStride], weights[r * kWinStride], kHeadDim);
    PipeBarrier<PIPE_V>();
    // Flush results below FP32's smallest normal to exactly zero, matching the
    // golden's `exp_z[exp_z < tiny] = 0.0` and the device's FTZ exponential.
    CompareScalar(mask, weights[r * kWinStride], kFp32Tiny, CMPMODE::LT, kHeadDim);
    PipeBarrier<PIPE_V>();
    Select(weights[r * kWinStride], mask, weights[r * kWinStride], 0.0F,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, kHeadDim);
    PipeBarrier<PIPE_V>();
    Add(col_sum, col_sum, weights[r * kWinStride], kHeadDim);
    PipeBarrier<PIPE_V>();
  }
  Duplicate(comp, 0.0F, kHeadDim);
  PipeBarrier<PIPE_V>();
  for (uint32_t r = 0U; r < kWindowRows; ++r) {
    // The max-subtracted row contributes exp(0) == 1, so col_sum >= 1.
    Div(weights[r * kWinStride], weights[r * kWinStride], col_sum, kHeadDim);
    PipeBarrier<PIPE_V>();
    Mul(win_kv[r * kWinStride], win_kv[r * kWinStride], weights[r * kWinStride],
        kHeadDim);
    PipeBarrier<PIPE_V>();
    Add(comp, comp, win_kv[r * kWinStride], kHeadDim);
    PipeBarrier<PIPE_V>();
  }

  // =========================================================================
  // Phase E -- weighted RMSNorm over the 512-wide reduced token.
  // =========================================================================
  Mul(col_sum, comp, comp, kHeadDim);
  PipeBarrier<PIPE_V>();
  ReduceSum<float>(red_dst, col_sum, red_tmp, kHeadDim);
  PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);
  const float mean_square = red_dst.GetValue(0) / static_cast<float>(kHeadDim);
  const float inv_rms = 1.0F / sqrt(mean_square + norm_eps);
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);

  DataCopy(norm_w_bf, norm_weight_global, kHeadDim);
  DataCopy(rope_cos, rope_cos_global, kRopeDim);
  DataCopy(rope_sin, rope_sin_global, kRopeDim);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Cast(norm_w_f32, norm_w_bf, RoundMode::CAST_NONE, kHeadDim);
  PipeBarrier<PIPE_V>();
  Mul(comp, comp, norm_w_f32, kHeadDim);
  PipeBarrier<PIPE_V>();
  Muls(comp, comp, inv_rms, kHeadDim);
  PipeBarrier<PIPE_V>();

  // =========================================================================
  // Phase F -- half-mode RoPE over lanes [base, head_dim).
  //
  //   out[base + i]        = cos[i]        * lo - sin[i]        * hi
  //   out[base + half + i] = sin[half + i] * lo + cos[half + i] * hi
  //
  // The second line reads the tables at half + i, not i. For the usual table
  // whose halves are duplicated these are equal; honouring the kernel's own
  // indexing keeps a non-duplicated table correct too.
  // =========================================================================
  {
    constexpr uint32_t kBase = kHeadDim - kRopeDim;  // 448
    constexpr uint32_t kHalf = kRopeDim / 2U;        // 32
    // rope_lo[0:32] = cos[i]*lo, rope_lo[32:64] = sin[i]*hi
    Mul(rope_lo[0], rope_cos[0], comp[kBase], kHalf);
    PipeBarrier<PIPE_V>();
    Mul(rope_lo[kHalf], rope_sin[0], comp[kBase + kHalf], kHalf);
    PipeBarrier<PIPE_V>();
    // rope_hi[0:32] = sin[half+i]*lo, rope_hi[32:64] = cos[half+i]*hi
    Mul(rope_hi[0], rope_sin[kHalf], comp[kBase], kHalf);
    PipeBarrier<PIPE_V>();
    Mul(rope_hi[kHalf], rope_cos[kHalf], comp[kBase + kHalf], kHalf);
    PipeBarrier<PIPE_V>();
    Sub(rope_lo[0], rope_lo[0], rope_lo[kHalf], kHalf);
    PipeBarrier<PIPE_V>();
    Add(rope_hi[0], rope_hi[0], rope_hi[kHalf], kHalf);
    PipeBarrier<PIPE_V>();
    // Both halves are now final, so writing them back cannot clobber a source.
    Adds(comp[kBase], rope_lo[0], 0.0F, kHalf);
    PipeBarrier<PIPE_V>();
    Adds(comp[kBase + kHalf], rope_hi[0], 0.0F, kHalf);
    PipeBarrier<PIPE_V>();
  }

  // --- emit the BF16 token -------------------------------------------------
  Cast(comp_bf, comp, RoundMode::CAST_RINT, kHeadDim);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(cache_slot_global, comp_bf, kHeadDim);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

  // =========================================================================
  // Phase G -- roll the window: this group's a-halves become the next
  // group's rows 0..3. Done after the reduction, so a caller inspecting state
  // mid-step still sees the rows the reduction used.
  // =========================================================================
  for (uint32_t r = 0U; r < kRatio; ++r) {
    DataCopy(win_sc[r * kWinStride], state_global[kStateKvBase + r * kStateWidth],
             kHeadDim);
    DataCopy(win_sc[(kRatio + r) * kWinStride],
             state_global[kStateScoreBase + r * kStateWidth], kHeadDim);
  }
  SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
  for (uint32_t r = 0U; r < kRatio; ++r) {
    DataCopy(state_global[kStatePrevKvBase + r * kHeadDim], win_sc[r * kWinStride],
             kHeadDim);
    DataCopy(state_global[kStatePrevScoreBase + r * kHeadDim],
             win_sc[(kRatio + r) * kWinStride], kHeadDim);
  }
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  state_meta_global.SetValue(0, 1U);
}
