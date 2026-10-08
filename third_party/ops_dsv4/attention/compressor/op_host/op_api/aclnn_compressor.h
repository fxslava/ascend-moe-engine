/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_COMPRESSOR_H
#define OP_API_INC_ACLNN_COMPRESSOR_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnCompressorGetWorkspaceSize
 *
 * The token-level KV compressor: multi-token windows are projected through
 * wkv/wgate, biased by ape, pooled by a softmax-weighted reduction over
 * cmpRatio tokens, RMSNorm-ed with normWeight and rotated by the partial RoPE
 * in ropeSin/ropeCos. cmpRatio 4 selects the CSA path and 128 the HCA path.
 *
 * parameters:
 *  x                       : required, token states [T, H] or [B, S, H],
 *                            BF16/FP16.
 *  wkv                     : required, compressed-KV projection, dtype of x.
 *  wgate                   : required, pooling-gate projection, dtype of x.
 *  stateCacheRef           : required, recurrent pooling state
 *                            [blocks, blockSize, D], FP32 -- UPDATED IN
 *                            PLACE. May be non-contiguous on axis 0;
 *                            stateCacheStrideDim0 carries that stride.
 *  ape                     : required, absolute positional bias, FP32.
 *  normWeight              : required, RMSNorm gain, 1-D.
 *  ropeSin                 : required, RoPE sin table.
 *  ropeCos                 : required, RoPE cos table.
 *  stateBlockTableOptional : optional, paged state-cache block map, INT32.
 *  cuSeqlensOptional       : optional, cumulative sequence lengths, INT32.
 *  sequsedOptional         : optional, used length per batch, INT32.
 *  startPosOptional        : optional, window start per batch, INT32.
 *  ropeHeadDim             : required attr, RoPE head dim, default 64.
 *  cmpRatio                : required attr, compression ratio (4 or 128).
 *  coff                    : optional attr, output-channel multiplier.
 *  normEps                 : optional attr, RMSNorm epsilon, default 1e-6.
 *  rotaryMode              : optional attr, RoPE interleaving mode.
 *  cacheMode               : optional attr, state-cache addressing mode.
 *  stateCacheStrideDim0    : optional attr, element stride of state-cache
 *                            axis 0.
 *  cmpKvOut                : required output, compressed KV rows, dtype of x.
 *  workspaceSize           : output, size of the device workspace in bytes.
 *  executor                : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: stateCacheRef is a REF parameter and receives no ViewCopy. cmpKvOut is
 * written through a distinct executor-owned tensor, so its trailing ViewCopy
 * always has src != dst and the executor stays reusable under
 * operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate, aclTensor *stateCacheRef, const aclTensor *ape,
    const aclTensor *normWeight, const aclTensor *ropeSin, const aclTensor *ropeCos,
    const aclTensor *stateBlockTableOptional, const aclTensor *cuSeqlensOptional, const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional, int64_t ropeHeadDim, int64_t cmpRatio, int64_t coff, double normEps,
    int64_t rotaryMode, int64_t cacheMode, int64_t stateCacheStrideDim0, const aclTensor *cmpKvOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnCompressor
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value aclnnCompressorGetWorkspaceSize returned.
 *  executor      : the plan aclnnCompressorGetWorkspaceSize returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnCompressor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
