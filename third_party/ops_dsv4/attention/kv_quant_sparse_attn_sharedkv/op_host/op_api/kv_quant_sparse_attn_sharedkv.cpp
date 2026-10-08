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
 * \file kv_quant_sparse_attn_sharedkv.cpp
 * \brief KvQuantSparseAttnSharedkv L0 API implementation
 */

#include "kv_quant_sparse_attn_sharedkv.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(KvQuantSparseAttnSharedkv);

aclnnStatus KvQuantSparseAttnSharedkv(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *metadataOptional, int64_t kvQuantMode, int64_t tileSize,
    int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional, const char *layoutKvOptional,
    int64_t oriKvStride0, int64_t cmpKvStride0, bool returnSoftmaxLse, const aclTensor *attnOutRef,
    const aclTensor *softmaxLseRef, std::array<const aclTensor *, 2> &outputs, aclOpExecutor *executor)
{
    L0_DFX(KvQuantSparseAttnSharedkv, q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional,
           cmpSparseIndicesOptional, oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional,
           cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional,
           metadataOptional, kvQuantMode, tileSize, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
           oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, oriKvStride0, cmpKvStride0,
           returnSoftmaxLse);

    // Both outputs mirror the caller descriptors the engine already sized, so
    // no INFER_SHAPE round trip through the deployed OPP proto is needed, and
    // they are distinct tensors so the L2 ViewCopy always has src != dst
    // (manual 4.31).
    auto attnOut =
        executor->AllocTensor(attnOutRef->GetViewShape(), attnOutRef->GetDataType(), attnOutRef->GetViewFormat());
    if (attnOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KvQuantSparseAttnSharedkv failed to allocate attnOut.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto softmaxLse = executor->AllocTensor(softmaxLseRef->GetViewShape(), softmaxLseRef->GetDataType(),
                                            softmaxLseRef->GetViewFormat());
    if (softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KvQuantSparseAttnSharedkv failed to allocate softmaxLse.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        KvQuantSparseAttnSharedkv,
        OP_INPUT(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                 oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                 cuSeqlensCmpKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional, metadataOptional),
        OP_OUTPUT(attnOut, softmaxLse),
        OP_ATTR(kvQuantMode, tileSize, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
                oriWinRight, layoutQOptional, layoutKvOptional, oriKvStride0, cmpKvStride0, returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KvQuantSparseAttnSharedkv ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    outputs[0] = attnOut;
    outputs[1] = softmaxLse;
    return ACLNN_SUCCESS;
}

}  // namespace l0op
