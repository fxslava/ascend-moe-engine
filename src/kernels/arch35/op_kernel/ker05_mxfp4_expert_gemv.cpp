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

// KER-05, LaunchMxfp4ExpertGemv, Ascend 950PR / arch35, SIMT/SIMD hybrid.
//
// MATH (docs/kernel_contracts_spec.md KER-05; oracle mxfp4_gemv_cpu)
//
//   y[row] = sum_col e2m1(nibble at [row, col]) * 2^(e8m0[row, col/32] - 127)
//                  * x[col]
//
// NIBBLE ORDER IS THE WHOLE POINT OF THIS KERNEL
//   packed[row, col/2] holds logical EVEN column `col` in bits 0..3 and odd
//   column `col+1` in bits 4..7 (kMxfp4LowNibbleFirst == 1; the host tiler
//   rejects 0). Swapping the two within a byte leaves every norm plausible,
//   and an all-ones probe vector cannot see the difference.
//
// WHY THIS IS A SIMT KERNEL
//   Unpacking 4-bit weights in SIMD costs a chain of widening casts, mask and
//   shift arithmetic and a gather through a decode table, and it has to
//   materialize the raw nibbles, the split streams and the decoded weights in
//   Unified Buffer -- tens of KiB of transient data that is read exactly once.
//
//   In SIMT each thread owns ONE logical column: it reads its packed byte
//   straight from global memory through the hardware Data Cache, picks its
//   nibble, decodes OCP E2M1 in registers, applies its group's E8M0 scale and
//   multiplies by x, writing only the finished product to UB. No DMA
//   descriptor, no transient nibble buffers and no decode table: UB holds just
//   x and the product vector, which is why this kernel's static footprint is
//   34 KiB rather than 78 KiB.
//
//   The dense 4096-lane reduction that follows stays in SIMD, where the
//   hardware reduction tree belongs. One VF switch per row keeps the
//   transition cost amortized over 4096 elements, well clear of the
//   sub-512-element granularity that makes mode switching a loss.
//
// E2M1 and E8M0 are decoded ARITHMETICALLY, in register:
//   E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} are reproduced branch-free
//   from the 3 magnitude bits; the sign is bit 3.
//   E8M0 code e is 2^(e-127), which is exactly ldexpf(1, e-127) for e<255.
//   Code 0 is 2^-127; code 255 is the OCP NaN marker and is emitted as NaN.
//
// ACCUMULATION ORDER: the scale is folded per column before the reduction,
// where the oracle sums a group's 32 columns and scales the partial once.
// The two are equal by distributivity and differ only in FP32 rounding.
//
// GM scratch workspace = 0 bytes. Each core owns disjoint 32-row tiles with a
// grid-stride loop, so there is no inter-core synchronization.

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#include "arch35_ub_layout.h"
#include "moe/kernels/arch35/kernel_contracts.h"

// >>> BEGIN GENERATED UB LAYOUT ANNOTATIONS -- scripts/gen_arch35_ub_layout.py
// Ker05 static UB layout: 34944 B of 221184 B usable (15.8%).
// These restore the analyzer's UB capacity, alignment, aliasing and bank
// checks, which it otherwise skips for static-mode LocalTensor offsets.
// @ascend-layout: name=xf32 pos=VECCALC offset=0 count=4096 dtype=float
// @ascend-layout: name=prod pos=VECCALC offset=16416 count=4096 dtype=float
// @ascend-layout: name=red_tmp pos=VECCALC offset=32832 count=512 dtype=float
// @ascend-layout: name=red_dst pos=VECCALC offset=34912 count=8 dtype=float
// SIMT raw-pointer aliases of the tensors above, declared so the
// analyzer can still verify their extents, and grouped so the
// deliberate overlap is not reported as accidental aliasing.
// @ascend-layout: name=x_ub pos=VECCALC offset=0 count=4096 dtype=float
// @ascend-reuse-group: group=simt_x_ub names=xf32,x_ub
// @ascend-layout: name=prod_ub pos=VECCALC offset=16416 count=4096 dtype=float
// @ascend-reuse-group: group=simt_prod_ub names=prod,prod_ub
// <<< END GENERATED UB LAYOUT ANNOTATIONS

using namespace AscendC;

namespace {

namespace ub = moe::kernels::arch35::ker05_ub;

// One VF launch per row. 1024 threads stays inside the 2048-thread ceiling of
// a Vector core and covers cols=4096 in four strided steps.
constexpr uint32_t kVfThreads = 1024U;

}  // namespace

// ---------------------------------------------------------------------------
// SIMT: one thread per logical column.
// ---------------------------------------------------------------------------
__simt_vf__ inline void Mxfp4RowProductVf(
    __gm__ uint8_t* packed_row,  // this row's cols/2 packed bytes
    __gm__ uint8_t* scale_row,   // this row's cols/32 E8M0 codes
    __ubuf__ float* x,           // resident FP32 x[cols]
    __ubuf__ float* product,     // out: w[col] * scale * x[col]
    uint32_t cols,
    uint32_t thread_count) {
  for (uint32_t col = threadIdx.x; col < cols; col += thread_count) {
    // Straight from GM through the Data Cache. Two threads share each byte,
    // so the second one hits the line the first one pulled in.
    const uint8_t byte = packed_row[col >> 1U];
    // Low nibble is the EVEN logical column; this is kMxfp4LowNibbleFirst.
    const uint32_t code =
        ((col & 1U) != 0U) ? static_cast<uint32_t>(byte >> 4U)
                           : static_cast<uint32_t>(byte & 0x0FU);

    // OCP E2M1, branch-free from the 3 magnitude bits:
    //   m 0..3 -> m * 0.5            (0, 0.5, 1, 1.5)
    //   m 4..7 -> (1 + (m&1)/2) * 2^(m>=6 ? 2 : 1)   (2, 3, 4, 6)
    const uint32_t m = code & 7U;
    const float mantissa = ((m & 1U) != 0U) ? 1.5F : 1.0F;
    const float exponent = (m <= 5U) ? 2.0F : 4.0F;
    const float magnitude =
        (m <= 3U) ? (0.5F * static_cast<float>(m)) : (mantissa * exponent);
    const float weight = ((code & 8U) != 0U) ? -magnitude : magnitude;

    // E8M0: one scale per 32 logical columns, value 2^(code - 127).
    const uint8_t scale_code = scale_row[col >> 5U];
    const float scale = (scale_code == 255U)
        ? __builtin_nanf("")
        : ldexpf(1.0F, static_cast<int32_t>(scale_code) - 127);

    product[col] = weight * scale * x[col];
  }
}

extern "C" __global__ __aicore__ void ker05_mxfp4_expert_gemv(
    GM_ADDR packed_gm, GM_ADDR scale_gm, GM_ADDR x_gm, GM_ADDR y_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  InitSocState();
  // ReduceSum's count form on dav-3510 allocates an internal V_S event through
  // GetTPipePtr(), so a TPipe must exist even though this kernel uses the
  // static tensor model and never allocates a TQue from it.
  TPipe pipe;
  (void)pipe;
  (void)workspace_gm;  // GM scratch contract is 0 bytes

  const __gm__ moe::kernels::arch35::Mxfp4GemvTilingData* tiling =
      reinterpret_cast<const __gm__ moe::kernels::arch35::Mxfp4GemvTilingData*>(
          tiling_gm);
  const uint32_t rows = tiling->rows;
  const uint32_t cols = tiling->cols;
  const uint32_t rows_per_block = tiling->rows_per_block;
  const uint32_t packed_row_stride = tiling->packed_row_stride;  // cols/2 bytes
  const uint32_t scale_row_stride = tiling->scale_row_stride;    // cols/32 bytes
  const uint32_t block_dim = tiling->block_dim;
  const uint32_t block = static_cast<uint32_t>(GetBlockIdx());

  GlobalTensor<float> x_global;
  GlobalTensor<float> y_global;
  x_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_gm));
  y_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_gm));

  LocalTensor<float> xf32(TPosition::VECCALC, ub::kXf32Off, ub::kXf32Count);
  LocalTensor<float> prod(TPosition::VECCALC, ub::kProdOff, ub::kProdCount);
  LocalTensor<float> red_tmp(TPosition::VECCALC, ub::kRedtmpOff, ub::kRedtmpCount);
  LocalTensor<float> red_dst(TPosition::VECCALC, ub::kRedDstOff, ub::kRedDstCount);

  // --- x is read by every row's VF, so it is staged once -------------------
  DataCopy(xf32, x_global, cols);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

  __gm__ uint8_t* packed_base = reinterpret_cast<__gm__ uint8_t*>(packed_gm);
  __gm__ uint8_t* scale_base = reinterpret_cast<__gm__ uint8_t*>(scale_gm);
  __ubuf__ float* x_ub = reinterpret_cast<__ubuf__ float*>(xf32.GetPhyAddr());
  __ubuf__ float* prod_ub = reinterpret_cast<__ubuf__ float*>(prod.GetPhyAddr());

  const uint32_t tiles = (rows + rows_per_block - 1U) / rows_per_block;
  for (uint32_t tile = block; tile < tiles; tile += block_dim) {
    const uint32_t row_begin = tile * rows_per_block;
    for (uint32_t local = 0U; local < rows_per_block; ++local) {
      const uint32_t row = row_begin + local;
      if (row >= rows) {
        break;
      }

      // --- SIMT: unpack, decode, scale and multiply, one column per thread
      // The Vector ALU is shared, so the mode switch drains the SIMD pipe
      // either side of the call.
      PipeBarrier<PIPE_V>();
      asc_vf_call<Mxfp4RowProductVf>(
          dim3(kVfThreads, 1, 1), packed_base + row * packed_row_stride,
          scale_base + row * scale_row_stride, x_ub, prod_ub, cols, kVfThreads);
            // ascend_analyzer has no asc_vf_call in its AST, so it cannot see the SIMT
      // launch between these two fences and reports the second as redundant.
      // Both are required: the Vector ALU is shared between the SIMD and SIMT
      // paths, so the SIMD pipeline must be drained on entry to and exit from
      // the VF. See ISSUES_AND_GAPS.md.
      // @ascend-ignore: AKA3005
      PipeBarrier<PIPE_V>();

      // --- SIMD: the dense reduction over all cols ----------------------
      ReduceSum<float>(red_dst, prod, red_tmp, cols);
      PipeBarrier<PIPE_V>();

      // One FP32 per row: a scalar GM store, because a 4-byte DataCopy would
      // be below the 32-byte access block and rows are not block-aligned.
      SetFlag<HardEvent::V_S>(EVENT_ID1);
      WaitFlag<HardEvent::V_S>(EVENT_ID1);
      y_global.SetValue(row, red_dst.GetValue(0));
      SetFlag<HardEvent::S_V>(EVENT_ID2);
      WaitFlag<HardEvent::S_V>(EVENT_ID2);
    }
  }
}
