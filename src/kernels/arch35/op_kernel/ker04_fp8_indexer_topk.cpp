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

// KER-04, fused_fp8_indexer_topk, Ascend 950PR / arch35, AIV only.
//
// MATH (docs/kernel_contracts_spec.md KER-04; oracle
// fast_fp8_indexer_scores_cpu + fast_fp8_indexer_topk_cpu)
//
//   physical_block = block_table[j / block_size]
//   key_row        = (physical_block * block_size + j % block_size) * 128
//   score[j]       = key_scale[j] * sum_h head_scale[h] * max(dot(q[h], key[j]), 0)
//
// ReLU IS PER HEAD AND BEFORE THE WEIGHTED SUM. Folding it after the sum is a
// different function, and because head_scale is non-negative the result still
// looks like a plausible relevance score.
//
// SELECTION ORDER: scores are rounded through BF16 BEFORE comparison, then
// sorted by score descending with ties broken by ascending logical index.
//
// HOW THE TIE-BREAK IS MADE EXACT
//   Relying on a hardware sort's behaviour for equal keys would be a guess.
//   Instead every key is made UNIQUE before sorting, so the sort's tie
//   behaviour cannot matter:
//
//     composite[j] = bitcast_f32( bitcast_u32(bf16_round(score[j])) + (L-1-j) )
//
//   Rounding through BF16 leaves the low 16 mantissa bits of the FP32 pattern
//   zero, and (L-1-j) < 65536 fits entirely inside them, so the addition
//   cannot disturb the exponent or carry into the BF16 value. Ordering
//   composites therefore orders first by BF16 score and then by DESCENDING
//   (L-1-j), i.e. ascending j -- exactly the oracle's `Beats`. This relies on
//   score >= 0, which ReLU plus non-negative scales guarantee, and which the
//   host tiler documents; for non-negative floats the IEEE bit pattern orders
//   identically to the value.
//
// KEY REPLICATION: each 128-lane key is broadcast across the 64 head slots
// with 64 Adds of 0, so one Mul covers the whole 8192-lane block and three
// reduction stages collapse it to 64 head dots. 64 Adds of 128 lanes and one
// Gather of 8192 lanes cost the same 128 vector repeats, and the Adds need no
// 32 KiB index tensor.
//
// GM scratch workspace = align_up(4 * seq_len, 32) bytes of FP32 scores.
// One AIV, so there is no cross-core top-k merge and no barrier; every score
// is complete before selection begins.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "arch35_ub_layout.h"
#include "moe/kernels/arch35/kernel_contracts.h"

// >>> BEGIN GENERATED UB LAYOUT ANNOTATIONS -- scripts/gen_arch35_ub_layout.py
// Ker04 static UB layout: 179008 B of 221184 B usable (80.9%).
// These restore the analyzer's UB capacity, alignment, aliasing and bank
// checks, which it otherwise skips for static-mode LocalTensor offsets.
// @ascend-layout: name=q_f32 pos=VECCALC offset=0 count=8192 dtype=float
// @ascend-layout: name=key_rep pos=VECCALC offset=32800 count=8192 dtype=float
// @ascend-layout: name=k_f32 pos=VECCALC offset=65600 count=128 dtype=float
// @ascend-layout: name=q_fp8 pos=VECCALC offset=66144 count=8192 dtype=uint8_t
// @ascend-layout: name=chunk_half pos=VECCALC offset=74368 count=1024 dtype=half
// @ascend-layout: name=chunk_i32 pos=VECCALC offset=76448 count=1024 dtype=int32_t
// @ascend-layout: name=fp8_lut pos=VECCALC offset=80576 count=256 dtype=float
// @ascend-layout: name=k_fp8 pos=VECCALC offset=81632 count=256 dtype=uint8_t
// @ascend-layout: name=k_half pos=VECCALC offset=81920 count=256 dtype=half
// @ascend-layout: name=k_i32 pos=VECCALC offset=82464 count=256 dtype=int32_t
// @ascend-layout: name=red1 pos=VECCALC offset=83520 count=1152 dtype=float
// @ascend-layout: name=red2 pos=VECCALC offset=88160 count=256 dtype=float
// @ascend-layout: name=head_dot pos=VECCALC offset=89216 count=256 dtype=float
// @ascend-layout: name=head_scale pos=VECCALC offset=90272 count=64 dtype=float
// @ascend-layout: name=score pos=VECCALC offset=90560 count=2048 dtype=float
// @ascend-layout: name=score_bf pos=VECCALC offset=98784 count=2048 dtype=bfloat16_t
// @ascend-layout: name=composite pos=VECCALC offset=102912 count=2048 dtype=float
// @ascend-layout: name=tie pos=VECCALC offset=111136 count=2048 dtype=int32_t
// @ascend-layout: name=sort_idx pos=VECCALC offset=119360 count=2048 dtype=int32_t
// @ascend-layout: name=sorted pos=VECCALC offset=127584 count=4096 dtype=float
// @ascend-layout: name=sort_tmp pos=VECCALC offset=144000 count=4096 dtype=float
// @ascend-layout: name=sval pos=VECCALC offset=160416 count=2048 dtype=float
// @ascend-layout: name=sidx pos=VECCALC offset=168640 count=2048 dtype=int32_t
// @ascend-layout: name=red_tmp pos=VECCALC offset=176864 count=512 dtype=float
// @ascend-layout: name=red_dst pos=VECCALC offset=178944 count=8 dtype=float
// @ascend-layout: name=nan_box pos=VECCALC offset=178976 count=8 dtype=int32_t
// SIMT raw-pointer aliases of the tensors above, declared so the
// analyzer can still verify their extents, and grouped so the
// deliberate overlap is not reported as accidental aliasing.
// @ascend-layout: name=fp8_lut_ub pos=VECCALC offset=80576 count=256 dtype=float
// @ascend-reuse-group: group=simt_fp8_lut_ub names=fp8_lut,fp8_lut_ub
// <<< END GENERATED UB LAYOUT ANNOTATIONS

using namespace AscendC;

namespace {

namespace ub = moe::kernels::arch35::ker04_ub;

constexpr uint32_t kHeads = moe::kernels::arch35::kIndexerHeads;        // 64
constexpr uint32_t kHeadDim = moe::kernels::arch35::kIndexerHeadDim;    // 128
constexpr uint32_t kTopK = moe::kernels::arch35::kIndexerTopK;          // 512
constexpr uint32_t kQueryLen = kHeads * kHeadDim;                       // 8192
constexpr uint32_t kChunk = 1024U;                                      // query dequant tile
constexpr uint32_t kSortGroup = 32U;                                    // Sort granularity
constexpr int32_t kUnusedLane = -1;                                     // golden fill value
constexpr int32_t kQuietNanBits = 0x7FC00000;

}  // namespace

// Build the 256-code E4M3FN table in thread registers. Each SIMT lane owns
// one code, so no scalar loop writes a bulk tensor buffer.
__simt_vf__ inline void BuildE4M3LutVf(__ubuf__ float* lut, float quiet_nan) {
  const uint32_t code = threadIdx.x;
  if (code >= 256U) return;
  const uint32_t exponent = (code >> 3U) & 0x0FU;
  const uint32_t mantissa = code & 0x07U;
  const bool negative = (code & 0x80U) != 0U;
  float magnitude = exponent == 0U
      ? static_cast<float>(mantissa) * (1.0F / 512.0F)
      : ldexpf(1.0F + static_cast<float>(mantissa) * 0.125F,
               static_cast<int32_t>(exponent) - 7);
  if (exponent == 0x0FU && mantissa == 0x07U) magnitude = quiet_nan;
  lut[code] = negative ? -magnitude : magnitude;
}
extern "C" __global__ __aicore__ void ker04_fp8_indexer_topk(
    GM_ADDR query_gm, GM_ADDR key_cache_gm, GM_ADDR dequant_scales_gm,
    GM_ADDR block_table_gm, GM_ADDR indices_gm, GM_ADDR workspace_gm,
    GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  // ReduceSum's count form on dav-3510 allocates an internal V_S event through
  // GetTPipePtr(), so a TPipe must exist even though this kernel uses the
  // static tensor model and never allocates a TQue from it. Without it
  // GetTPipePtr() is null and the reduction segfaults -- which bisheng does
  // not diagnose, because the dereference happens inside the library.
  TPipe pipe;
  (void)pipe;

  const __gm__ moe::kernels::arch35::IndexerTilingData* tiling =
      reinterpret_cast<const __gm__ moe::kernels::arch35::IndexerTilingData*>(
          tiling_gm);
  const uint32_t query_head_stride = tiling->query_head_stride;
  const uint32_t key_block_stride = tiling->key_block_stride;
  const uint32_t block_size = tiling->block_size;
  const uint32_t seq_len = tiling->seq_len;

  GlobalTensor<uint8_t> query_global;
  GlobalTensor<uint8_t> key_cache_global;
  GlobalTensor<float> dequant_scales_global;
  GlobalTensor<int32_t> block_table_global;
  GlobalTensor<int32_t> indices_global;
  GlobalTensor<float> workspace_global;
  query_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(query_gm));
  key_cache_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache_gm));
  dequant_scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dequant_scales_gm));
  block_table_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(block_table_gm));
  indices_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices_gm));
  workspace_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace_gm));

  LocalTensor<float> q_f32(TPosition::VECCALC, ub::kQF32Off, ub::kQF32Count);
  LocalTensor<float> key_rep(TPosition::VECCALC, ub::kKeyRepOff, ub::kKeyRepCount);
  LocalTensor<float> k_f32(TPosition::VECCALC, ub::kKF32Off, ub::kKF32Count);
  LocalTensor<uint8_t> q_fp8(TPosition::VECCALC, ub::kQFp8Off, ub::kQFp8Count);
  LocalTensor<half> chunk_half(TPosition::VECCALC, ub::kChunkHalfOff,
                               ub::kChunkHalfCount);
  LocalTensor<int32_t> chunk_i32(TPosition::VECCALC, ub::kChunkI32Off,
                                 ub::kChunkI32Count);
  LocalTensor<float> fp8_lut(TPosition::VECCALC, ub::kFp8LutOff, ub::kFp8LutCount);
  LocalTensor<uint8_t> k_fp8(TPosition::VECCALC, ub::kKFp8Off, ub::kKFp8Count);
  LocalTensor<half> k_half(TPosition::VECCALC, ub::kKHalfOff, ub::kKHalfCount);
  LocalTensor<int32_t> k_i32(TPosition::VECCALC, ub::kKI32Off, ub::kKI32Count);
  LocalTensor<float> red1(TPosition::VECCALC, ub::kRed1Off, ub::kRed1Count);
  LocalTensor<float> red2(TPosition::VECCALC, ub::kRed2Off, ub::kRed2Count);
  LocalTensor<float> head_dot(TPosition::VECCALC, ub::kHeadDotOff, ub::kHeadDotCount);
  LocalTensor<float> head_scale(TPosition::VECCALC, ub::kHeadScaleOff,
                                ub::kHeadScaleCount);
  LocalTensor<float> score(TPosition::VECCALC, ub::kScoreOff, ub::kScoreCount);
  LocalTensor<bfloat16_t> score_bf(
      TPosition::VECCALC, ub::kScoreBfOff, ub::kScoreBfCount);
  LocalTensor<float> composite(TPosition::VECCALC, ub::kCompositeOff,
                               ub::kCompositeCount);
  LocalTensor<int32_t> tie(TPosition::VECCALC, ub::kTieOff, ub::kTieCount);
  LocalTensor<int32_t> sort_idx(TPosition::VECCALC, ub::kSortIdxOff, ub::kSortIdxCount);
  LocalTensor<float> sorted(TPosition::VECCALC, ub::kSortedOff, ub::kSortedCount);
  LocalTensor<float> sort_tmp(TPosition::VECCALC, ub::kSortTmpOff, ub::kSortTmpCount);
  LocalTensor<float> sval(TPosition::VECCALC, ub::kSvalOff, ub::kSvalCount);
  LocalTensor<int32_t> sidx(TPosition::VECCALC, ub::kSidxOff, ub::kSidxCount);
  LocalTensor<float> red_tmp(TPosition::VECCALC, ub::kRedtmpOff, ub::kRedtmpCount);
  LocalTensor<float> red_dst(TPosition::VECCALC, ub::kRedDstOff, ub::kRedDstCount);
  LocalTensor<int32_t> nan_box(TPosition::VECCALC, ub::kNanboxOff, ub::kNanboxCount);

  // =========================================================================
  // seq_len == 0 is legal: every output lane is padding.
  // =========================================================================
  if (seq_len == 0U) {
    Duplicate(sidx, kUnusedLane, kTopK);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(indices_global, sidx, kTopK);
    return;
  }

  // =========================================================================
  // Phase A -- register-parallel E4M3FN decode table construction.
  // =========================================================================
  nan_box.SetValue(0, kQuietNanBits);
  const float quiet_nan = nan_box.ReinterpretCast<float>().GetValue(0);
  __ubuf__ float* fp8_lut_ub =
      reinterpret_cast<__ubuf__ float*>(fp8_lut.GetPhyAddr());
  asc_vf_call<BuildE4M3LutVf>(dim3(256, 1, 1), fp8_lut_ub, quiet_nan);
  // The SIMT VF writes the table consumed by SIMD Gather below.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();
  // =========================================================================
  // Phase B -- dequantize the 64x128 query once, 1024 codes at a time, and
  // load the 64 head scales. head_scale already folds the query dequant and
  // the per-head gate.
  // =========================================================================
  DataCopy(q_fp8, query_global, kQueryLen);
  DataCopy(head_scale, dequant_scales_global, kHeads);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  SetFlag<HardEvent::S_V>(EVENT_ID1);
  WaitFlag<HardEvent::S_V>(EVENT_ID1);

  for (uint32_t base = 0U; base < kQueryLen; base += kChunk) {
    Cast(chunk_half, q_fp8[base], RoundMode::CAST_NONE, kChunk);
    PipeBarrier<PIPE_V>();
    Cast(chunk_i32, chunk_half, RoundMode::CAST_RINT, kChunk);
    PipeBarrier<PIPE_V>();
    ShiftLeft(chunk_i32, chunk_i32, 2, kChunk);  // code * sizeof(float)
    PipeBarrier<PIPE_V>();
    Gather(q_f32[base], fp8_lut, chunk_i32.ReinterpretCast<uint32_t>(), 0U, kChunk);
    // ascend_analyzer has no Gather in its intrinsic table, so it cannot see
    // that a vector op sits between these two barriers and reports the second
    // as redundant. The barrier is required: Gather reads an index tensor the
    // previous op wrote and writes the operand the next op reads. See
    // ISSUES_AND_GAPS.md.
    // @ascend-ignore: AKA3005
    PipeBarrier<PIPE_V>();
  }

  // =========================================================================
  // Phase C -- score every key, writing FP32 scores to the GM workspace.
  //
  // The paged address resolution is a scalar pointer chase: the block table
  // lives in GM and is read one entry per key.
  // =========================================================================
  for (uint32_t j = 0U; j < seq_len; ++j) {
    const uint32_t table_slot = j / block_size;
    const uint32_t physical_block =
        static_cast<uint32_t>(block_table_global.GetValue(table_slot));
    const uint32_t key_row = physical_block * key_block_stride +
                             (j % block_size) * kHeadDim;

    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopy(k_fp8, key_cache_global[key_row], kHeadDim);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    // --- dequantize this key row through the same E4M3 table --------------
    Cast(k_half, k_fp8, RoundMode::CAST_NONE, kHeadDim);
    PipeBarrier<PIPE_V>();
    Cast(k_i32, k_half, RoundMode::CAST_RINT, kHeadDim);
    PipeBarrier<PIPE_V>();
    ShiftLeft(k_i32, k_i32, 2, kHeadDim);
    PipeBarrier<PIPE_V>();
    Gather(k_f32, fp8_lut, k_i32.ReinterpretCast<uint32_t>(), 0U, kHeadDim);
    // @ascend-ignore: AKA3005
    PipeBarrier<PIPE_V>();

    // --- broadcast the key across the 64 head slots -----------------------
    for (uint32_t head = 0U; head < kHeads; ++head) {
      Adds(key_rep[head * kHeadDim], k_f32, 0.0F, kHeadDim);
    }
    PipeBarrier<PIPE_V>();
    Mul(key_rep, key_rep, q_f32, kQueryLen);
    PipeBarrier<PIPE_V>();

    // --- three reduction stages: 8192 -> 1024 -> 128 -> 64 head dots ------
    // Each head spans 128 lanes = 16 eight-lane blocks, so stage 1 leaves 16
    // partials per head, stage 2 leaves 2, and the pairwise stage leaves 1.
    ReduceDataBlock<ReduceType::SUM, float, float>(
        red1, key_rep, 64, static_cast<int32_t>(kQueryLen / 64U), 1, 1, 8);
    PipeBarrier<PIPE_V>();
    ReduceDataBlock<ReduceType::SUM, float, float>(
        red2, red1, 64, static_cast<int32_t>((kQueryLen / 8U) / 64U), 1, 1, 8);
    PipeBarrier<PIPE_V>();
    ReducePairElem<ReduceType::SUM, float, float>(
        head_dot, red2, 64, static_cast<int32_t>((kQueryLen / 64U) / 64U), 4, 1, 8);
    PipeBarrier<PIPE_V>();

    // --- ReLU per head, THEN the head-weighted sum ------------------------
    Relu(head_dot, head_dot, kHeads);
    PipeBarrier<PIPE_V>();
    Mul(head_dot, head_dot, head_scale, kHeads);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(red_dst, head_dot, red_tmp, kHeads);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    const float key_scale = dequant_scales_global.GetValue(kHeads + j);
    workspace_global.SetValue(j, red_dst.GetValue(0) * key_scale);
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
  }

  // =========================================================================
  // Phase D -- deterministic Top-K.
  //
  // Scores come back from the workspace, round through BF16, and become unique
  // composite keys (see the header). Because no two composites are equal, the
  // hardware sort's behaviour on equal keys cannot affect the result.
  // =========================================================================
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
  DataCopy(score, workspace_global, seq_len);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

  Cast(score_bf, score, RoundMode::CAST_RINT, seq_len);
  PipeBarrier<PIPE_V>();
  Cast(composite, score_bf, RoundMode::CAST_NONE, seq_len);
  PipeBarrier<PIPE_V>();

  // tie[j] = seq_len - 1 - j, which is < 65536 and so lands entirely in the
  // 16 mantissa bits that the BF16 round-trip left zero.
  CreateVecIndex(tie, 0, seq_len);
  // Required barrier. ascend_analyzer's intrinsic table has no
  // CreateVecIndex/Sort/Extract (nor Gather), so it does not see the vector
  // op between these barriers and calls the second redundant. Deleting it
  // would drop a real RAW edge. See ISSUES_AND_GAPS.md.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();
  Muls(tie, tie, -1, seq_len);
  PipeBarrier<PIPE_V>();
  Adds(tie, tie, static_cast<int32_t>(seq_len - 1U), seq_len);
  PipeBarrier<PIPE_V>();
  Add(composite.ReinterpretCast<int32_t>(), composite.ReinterpretCast<int32_t>(), tie,
      seq_len);
  PipeBarrier<PIPE_V>();

  CreateVecIndex(sort_idx, 0, seq_len);
  // Required barrier. ascend_analyzer's intrinsic table has no
  // CreateVecIndex/Sort/Extract (nor Gather), so it does not see the vector
  // op between these barriers and calls the second redundant. Deleting it
  // would drop a real RAW edge. See ISSUES_AND_GAPS.md.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();
  Sort<float, true>(sorted, composite, sort_idx.ReinterpretCast<uint32_t>(), sort_tmp,
                    static_cast<int32_t>(seq_len / kSortGroup));
  // Required barrier. ascend_analyzer's intrinsic table has no
  // CreateVecIndex/Sort/Extract (nor Gather), so it does not see the vector
  // op between these barriers and calls the second redundant. Deleting it
  // would drop a real RAW edge. See ISSUES_AND_GAPS.md.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();
  Extract(sval, sidx.ReinterpretCast<uint32_t>(), sorted,
          static_cast<int32_t>(seq_len / kSortGroup));
  // Required barrier. ascend_analyzer's intrinsic table has no
  // CreateVecIndex/Sort/Extract (nor Gather), so it does not see the vector
  // op between these barriers and calls the second redundant. Deleting it
  // would drop a real RAW edge. See ISSUES_AND_GAPS.md.
  // @ascend-ignore: AKA3005
  PipeBarrier<PIPE_V>();

  // Lanes [seq_len, top_k) are padding when the sequence is shorter than K.
  if (seq_len < kTopK) {
    Duplicate(sidx[seq_len], kUnusedLane, kTopK - seq_len);
    PipeBarrier<PIPE_V>();
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(indices_global, sidx, kTopK);
}
