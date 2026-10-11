/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_KV_QUANT_SPARSE_ATTN_SHAREDKV_H
#define OP_API_INC_ACLNN_KV_QUANT_SPARSE_ATTN_SHAREDKV_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize
 *
 * The shared-KV sparse attention core: a dedicated arch35 MQA kernel in which
 * one BF16 query stream attends to BOTH halves of the quantized hybrid KV
 * cache -- the uncompressed (ori) stream and the compressor-produced
 * compressed (cmp) stream -- each at its own sparse indices, block table and
 * mask mode, with optional attention sinks folded into the softmax
 * denominator.
 *
 * parameters:
 *  q                           : required, BF16 query; layout per
 *                                layoutQOptional.
 *  oriKvOptional               : optional, uncompressed KV cache, FP8 E4M3.
 *                                Passed through uncontiguized; its axis-0
 *                                stride travels as oriKvStride0.
 *  cmpKvOptional               : optional, compressed KV cache, FP8 E4M3.
 *                                Passed through uncontiguized; its axis-0
 *                                stride travels as cmpKvStride0.
 *  oriSparseIndicesOptional    : optional, INT32 indices into the ori stream.
 *  cmpSparseIndicesOptional    : optional, INT32 indices into the cmp stream.
 *  oriBlockTableOptional       : optional, paged block map for ori, INT32.
 *  cmpBlockTableOptional       : optional, paged block map for cmp, INT32.
 *  cuSeqlensQOptional          : optional, cumulative query lengths, INT32.
 *  cuSeqlensOriKvOptional      : optional, cumulative ori KV lengths, INT32.
 *  cuSeqlensCmpKvOptional      : optional, cumulative cmp KV lengths, INT32.
 *  sequsedQOptional            : optional, used query length per batch, INT32.
 *  sequsedKvOptional           : optional, used KV length per batch, INT32.
 *  sinksOptional               : optional, attention sink logits, FP32.
 *  metadataOptional            : optional, precomputed scheduling metadata,
 *                                INT32.
 *  kvQuantMode                 : required attr, KV quantization mode.
 *  tileSize                    : optional attr, KV tile, default 64.
 *  ropeHeadDim                 : optional attr, RoPE head dim, default 64.
 *  softmaxScale                : required attr, logit scale.
 *  cmpRatio                    : required attr, compression ratio of the cmp
 *                                stream (4 for CSA, 128 for HCA).
 *  oriMaskMode                 : required attr, mask mode for ori, default 4.
 *  cmpMaskMode                 : required attr, mask mode for cmp, default 3.
 *  oriWinLeft                  : optional attr, sliding window left bound.
 *  oriWinRight                 : optional attr, sliding window right bound.
 *  layoutQOptional             : optional attr, query layout, default "BSND".
 *  layoutKvOptional            : optional attr, KV layout, default "PA_ND".
 *  oriKvStride0                : optional attr, ori cache axis-0 stride.
 *  cmpKvStride0                : optional attr, cmp cache axis-0 stride.
 *  returnSoftmaxLse            : optional attr, true to fill softmaxLseOut.
 *  attnOut                     : required output, BF16, shape of q.
 *  softmaxLseOut               : required output, FP32; pass a [0] tensor when
 *                                returnSoftmaxLse is false.
 *  workspaceSize               : output, device workspace bytes.
 *  executor                    : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: at least one of oriKvOptional / cmpKvOptional must be present. Both
 * outputs are written through distinct executor-owned tensors, so their
 * trailing ViewCopy always has src != dst and the returned executor is
 * reusable under operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *metadataOptional, int64_t kvQuantMode, int64_t tileSize,
    int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, int64_t oriKvStride0,
    int64_t cmpKvStride0, bool returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnKvQuantSparseAttnSharedkv
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value the plan phase returned.
 *  executor      : the plan the plan phase returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnKvQuantSparseAttnSharedkv(void *workspace, uint64_t workspaceSize,
                                                     aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
