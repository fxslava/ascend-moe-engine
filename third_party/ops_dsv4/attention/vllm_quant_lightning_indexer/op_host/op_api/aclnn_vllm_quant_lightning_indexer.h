/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_VLLM_QUANT_LIGHTNING_INDEXER_H
#define OP_API_INC_ACLNN_VLLM_QUANT_LIGHTNING_INDEXER_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnVllmQuantLightningIndexerGetWorkspaceSize
 *
 * The quantized lightning indexer: a mixed Cube/Vector arch35 kernel that
 * scores FP8 E4M3 / HiFloat8 query against key under their dequant scales and
 * emits the top-sparseCount block indices.
 *
 * Against the ops-transformer aclnnQuantLightningIndexer this adds the
 * metadata input, the cmpRatio / returnValues attributes, the explicit
 * stride / scaleStride attributes and the second sparseValues output. Both
 * operators are registered; this is the compressed-KV (shared-KV) variant.
 *
 * parameters:
 *  query                           : required, [B,S1,N1,D] or [T1,N1,D];
 *                                    D=128, N1 in {16,24,32,64} on 950PR.
 *  key                             : required, layout per layoutKeyOptional,
 *                                    N2=1, D=128, dtype of query. Passed
 *                                    through uncontiguized -- its axis-0
 *                                    stride travels as the stride attribute.
 *  weights                         : required, [B,S1,N1] or [T,N1], BF16/FP16.
 *  queryDequantScale               : required, [B,S1,N1] or [T,N1].
 *  keyDequantScale                 : required, layout of key without D; its
 *                                    axis-0 stride travels as scaleStride.
 *  actualSeqLengthsQueryOptional   : optional, INT32; required for TND.
 *  actualSeqLengthsKeyOptional     : optional, INT32; required for TND and
 *                                    PA_BSND.
 *  blockTableOptional              : optional, paged-KV block map, INT32.
 *  metadataOptional                : optional, precomputed scheduling
 *                                    metadata, INT32.
 *  queryQuantMode                  : required attr; 0 = per-token-head.
 *  keyQuantMode                    : required attr; 0 = per-token-head.
 *  layoutQueryOptional             : optional attr, "BSND" or "TND".
 *  layoutKeyOptional               : optional attr, "BSND", "TND", "PA_BSND".
 *  sparseCount                     : optional attr, top-k in [1, 2048].
 *  sparseMode                      : optional attr, 0 dense, 3 right-down
 *                                    causal.
 *  preTokens                       : optional attr, INT64_MAX = unrestricted.
 *  nextTokens                      : optional attr, INT64_MAX = unrestricted.
 *  cmpRatio                        : optional attr, compression ratio of the
 *                                    key stream.
 *  returnValues                    : optional attr, true to fill
 *                                    sparseValuesOut.
 *  stride                          : optional attr, key axis-0 element stride.
 *  scaleStride                     : optional attr, keyDequantScale axis-0
 *                                    element stride.
 *  sparseIndicesOut                : required output, INT32,
 *                                    [B,S1,N2,k] or [T,N2,k].
 *  sparseValuesOut                 : required output, FP32, shape of
 *                                    sparseIndicesOut; pass a [0] tensor when
 *                                    returnValues is false.
 *  workspaceSize                   : output, device workspace bytes.
 *  executor                        : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: both outputs are written through distinct executor-owned tensors, so
 * their trailing ViewCopy always has src != dst and the returned executor is
 * reusable under operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnVllmQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional,
    const aclTensor *metadataOptional, int64_t queryQuantMode, int64_t keyQuantMode, char *layoutQueryOptional,
    char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, bool returnValues, int64_t stride, int64_t scaleStride, const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnVllmQuantLightningIndexer
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value the plan phase returned.
 *  executor      : the plan the plan phase returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnVllmQuantLightningIndexer(void *workspace, uint64_t workspaceSize,
                                                     aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
