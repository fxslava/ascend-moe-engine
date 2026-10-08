/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor.h
 * \brief Compressor L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_COMPRESSOR_OP_H
#define OP_API_INC_LEVEL0_OP_COMPRESSOR_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Compressor L0 kernel interface (token-level KV compressor).
 *
 * Projects a token window through wkv/wgate, adds the positional bias ape,
 * pools it with a softmax-weighted reduction over cmpRatio tokens, applies
 * RMSNorm with normWeight and the partial RoPE carried by ropeSin/ropeCos,
 * and writes the recurrent pooling state back into stateCache.
 *
 * stateCache is a REF parameter: it is both an OP_INPUT and an OP_OUTPUT and
 * the kernel updates it in place, so the L2 layer passes it through untouched.
 * cmpKv is a genuine output and is allocated by this builder.
 *
 * @param [in] x Token states, [T, H] or [B, S, H], BF16/FP16.
 * @param [in] wkv Compressed-KV projection weight, dtype of x.
 * @param [in] wgate Pooling-gate projection weight, dtype of x.
 * @param [in] stateCache Recurrent pooling state, [blocks, blockSize, D],
 * FP32; updated in place.
 * @param [in] ape Absolute positional bias, FP32.
 * @param [in] normWeight RMSNorm gain, FP32 (BF16/FP16 on the 910 configs).
 * @param [in] ropeSin RoPE sin table.
 * @param [in] ropeCos RoPE cos table.
 * @param [in] stateBlockTableOptional Paged state-cache block map, INT32.
 * @param [in] cuSeqlensOptional Cumulative sequence lengths, INT32.
 * @param [in] sequsedOptional Used length per batch, INT32.
 * @param [in] startPosOptional Window start position per batch, INT32.
 * @param [in] ropeHeadDim RoPE head dimension (64).
 * @param [in] cmpRatio Compression ratio; 4 for CSA, 128 for HCA.
 * @param [in] coff Output-channel multiplier.
 * @param [in] normEps RMSNorm epsilon.
 * @param [in] rotaryMode RoPE interleaving mode.
 * @param [in] cacheMode State-cache addressing mode.
 * @param [in] stateCacheStrideDim0 Element stride of the state-cache axis 0.
 * @param [in] cmpKvRef The caller output descriptor whose shape, dtype and
 * format the allocated kernel output copies.
 * @param [in] executor Op executor.
 * @return The freshly allocated cmpKv tensor, or nullptr on failure.
 */
const aclTensor *Compressor(const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
                            const aclTensor *stateCache, const aclTensor *ape, const aclTensor *normWeight,
                            const aclTensor *ropeSin, const aclTensor *ropeCos,
                            const aclTensor *stateBlockTableOptional, const aclTensor *cuSeqlensOptional,
                            const aclTensor *sequsedOptional, const aclTensor *startPosOptional,
                            int64_t ropeHeadDim, int64_t cmpRatio, int64_t coff, double normEps, int64_t rotaryMode,
                            int64_t cacheMode, int64_t stateCacheStrideDim0, const aclTensor *cmpKvRef,
                            aclOpExecutor *executor);

}  // namespace l0op

#endif
