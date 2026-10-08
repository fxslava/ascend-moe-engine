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
 * \file vllm_quant_lightning_indexer.cpp
 * \brief VllmQuantLightningIndexer L0 API implementation
 */

#include "vllm_quant_lightning_indexer.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(VllmQuantLightningIndexer);

aclnnStatus VllmQuantLightningIndexer(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional,
    const aclTensor *metadataOptional, int64_t queryQuantMode, int64_t keyQuantMode, const char *layoutQueryOptional,
    const char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, bool returnValues, int64_t stride, int64_t scaleStride, const aclTensor *sparseIndicesRef,
    const aclTensor *sparseValuesRef, std::array<const aclTensor *, 2> &outputs, aclOpExecutor *executor)
{
    L0_DFX(VllmQuantLightningIndexer, query, key, weights, queryDequantScale, keyDequantScale,
           actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, blockTableOptional, metadataOptional,
           queryQuantMode, keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens,
           nextTokens, cmpRatio, returnValues, stride, scaleStride);

    // Both outputs mirror the caller descriptors the engine already sized, so
    // no INFER_SHAPE round trip through the deployed OPP proto is needed. They
    // are distinct tensors on purpose: the L2 ViewCopy then always has
    // src != dst, which keeps the executor reusable under manual 4.31.
    auto sparseIndices = executor->AllocTensor(sparseIndicesRef->GetViewShape(), sparseIndicesRef->GetDataType(),
                                               sparseIndicesRef->GetViewFormat());
    if (sparseIndices == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "VllmQuantLightningIndexer failed to allocate sparseIndices.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto sparseValues = executor->AllocTensor(sparseValuesRef->GetViewShape(), sparseValuesRef->GetDataType(),
                                              sparseValuesRef->GetViewFormat());
    if (sparseValues == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "VllmQuantLightningIndexer failed to allocate sparseValues.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        VllmQuantLightningIndexer,
        OP_INPUT(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
                 actualSeqLengthsKeyOptional, blockTableOptional, metadataOptional),
        OP_OUTPUT(sparseIndices, sparseValues),
        OP_ATTR(queryQuantMode, keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode,
                preTokens, nextTokens, cmpRatio, returnValues, stride, scaleStride));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "VllmQuantLightningIndexer ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    outputs[0] = sparseIndices;
    outputs[1] = sparseValues;
    return ACLNN_SUCCESS;
}

}  // namespace l0op
