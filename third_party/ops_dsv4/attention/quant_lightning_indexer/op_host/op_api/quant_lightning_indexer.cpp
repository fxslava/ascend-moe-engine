/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantLightningIndexer);

const aclTensor *QuantLightningIndexer(
        const aclTensor *query,
        const aclTensor *key,
        const aclTensor *weights,
        const aclTensor *queryDequantScale,
        const aclTensor *keyDequantScale,
        const aclTensor *actualSeqLengthsQueryOptional,
        const aclTensor *actualSeqLengthsKeyOptional,
        const aclTensor *blockTableOptional,
        int64_t queryQuantMode,
        int64_t keyQuantMode,
        const char *layoutQueryOptional,
        const char *layoutKeyOptional,
        int64_t sparseCount,
        int64_t sparseMode,
        int64_t preTokens,
        int64_t nextTokens,
        aclOpExecutor *executor)
{
    int64_t keyBlockStride = 0;
    int64_t keyDequantScaleBlockStride = 0;
    auto keyStride = key->GetViewStrides();
    keyBlockStride = keyStride[0];
    auto keyScaleStride = keyDequantScale->GetViewStrides();
    keyDequantScaleBlockStride = keyScaleStride[0];

    // L0接口时延统计以及入参打印
    L0_DFX(QuantLightningIndexer, query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
        actualSeqLengthsKeyOptional, blockTableOptional, queryQuantMode, keyQuantMode, layoutQueryOptional,
        layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, keyBlockStride, keyDequantScaleBlockStride);

    // 构造输出
    auto output = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);

    // 调用inferShape
    auto ret = INFER_SHAPE(QuantLightningIndexer,
                            OP_INPUT(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
                                    actualSeqLengthsKeyOptional, blockTableOptional),
                            OP_OUTPUT(output),
                            OP_ATTR(queryQuantMode, keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount,
                                    sparseMode, preTokens, nextTokens, keyBlockStride, keyDequantScaleBlockStride));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantLightningIndexer InferShape failed.");
        return nullptr;
    }

    // 发起aicore任务
    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantLightningIndexer,
                            OP_INPUT(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
                                    actualSeqLengthsKeyOptional, blockTableOptional),
                            OP_OUTPUT(output),
                            OP_ATTR(queryQuantMode, keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount,
                                    sparseMode, preTokens, nextTokens, keyBlockStride, keyDequantScaleBlockStride));
    
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return output;
}

} // namespace l0op
