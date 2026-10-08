/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kv_quant_sparse_attn_sharedkv.h"

#include <array>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "kv_quant_sparse_attn_sharedkv.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

// The file-scope unnamed namespace must sit OUTSIDE the extern "C" block:
// a function declared inside extern "C" gets C language linkage, which is
// EXTERNAL even within an unnamed namespace, so helpers named CheckParams or
// StageContiguous here would collide at link time with the identically named
// helpers in the other operators' op_api files.
namespace {

constexpr int64_t kCmpRatioCsa = 4;
constexpr int64_t kCmpRatioHca = 128;

aclnnStatus CheckParams(const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
                        const aclTensor *sinksOptional, int64_t tileSize, double softmaxScale, int64_t cmpRatio,
                        const aclTensor *attnOut, const aclTensor *softmaxLseOut)
{
    OP_CHECK_NULL(q, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(attnOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(softmaxLseOut, return ACLNN_ERR_PARAM_NULLPTR);

    if (q->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (attnOut->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "attnOut must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (softmaxLseOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxLseOut must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // The whole point of this operator is the hybrid cache: it is meaningless
    // with neither half bound.
    if (oriKvOptional == nullptr && cmpKvOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "at least one of oriKvOptional and cmpKvOptional must be provided.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (oriKvOptional != nullptr && oriKvOptional->GetDataType() != DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "oriKvOptional must be FP8 E4M3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cmpKvOptional != nullptr && cmpKvOptional->GetDataType() != DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpKvOptional must be FP8 E4M3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sinksOptional != nullptr && sinksOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sinksOptional must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (tileSize <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tileSize must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(softmaxScale > 0.0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxScale must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmpRatio only constrains the compressed half; with no cmp stream bound
    // the attribute is unused and 1 is the definition default.
    if (cmpKvOptional != nullptr && cmpRatio != kCmpRatioCsa && cmpRatio != kCmpRatioHca) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be 4 (CSA) or 128 (HCA) when cmpKvOptional is bound.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus StageContiguous(const aclTensor *&tensor, aclOpExecutor *executor, bool optional)
{
    if (tensor == nullptr) {
        return optional ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_NULLPTR;
    }
    tensor = l0op::Contiguous(tensor, executor);
    return tensor != nullptr ? ACLNN_SUCCESS : ACLNN_ERR_INNER_NULLPTR;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *metadataOptional, int64_t kvQuantMode, int64_t tileSize,
    int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, int64_t oriKvStride0,
    int64_t cmpKvStride0, bool returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKvQuantSparseAttnSharedkv,
                   DFX_IN(q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional,
                          oriBlockTableOptional, cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional,
                          cuSeqlensCmpKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional,
                          metadataOptional, kvQuantMode, tileSize, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode,
                          cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, oriKvStride0,
                          cmpKvStride0, returnSoftmaxLse),
                   DFX_OUT(attnOut, softmaxLseOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(q, oriKvOptional, cmpKvOptional, sinksOptional, tileSize, softmaxScale, cmpRatio, attnOut,
                           softmaxLseOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (q->IsEmpty() || attnOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // oriKv and cmpKv are deliberately NOT staged: they are IgnoreContiguous
    // paged caches, contiguous on every axis except axis 0, and that stride is
    // what oriKvStride0 / cmpKvStride0 carry to the kernel. Contiguous would
    // flatten the paged view and lose the block mapping.
    CHECK_RET(StageContiguous(q, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(oriSparseIndicesOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cmpSparseIndicesOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(oriBlockTableOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cmpBlockTableOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cuSeqlensQOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cuSeqlensOriKvOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cuSeqlensCmpKvOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(sequsedQOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(sequsedKvOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(sinksOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(metadataOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    std::array<const aclTensor *, 2> kernelOuts = {nullptr, nullptr};
    ret = l0op::KvQuantSparseAttnSharedkv(
        q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional, oriBlockTableOptional,
        cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
        sequsedKvOptional, sinksOptional, metadataOptional, kvQuantMode, tileSize, ropeHeadDim, softmaxScale,
        cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, oriKvStride0,
        cmpKvStride0, returnSoftmaxLse, attnOut, softmaxLseOut, kernelOuts, l0Executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Distinct executor-owned sources, so neither copy is a same-address
    // self-copy (4.31). softmaxLse is only copied back when the caller asked
    // for it; otherwise softmaxLseOut is the [0] placeholder tensor.
    auto attnCopy = l0op::ViewCopy(kernelOuts[0], attnOut, l0Executor);
    CHECK_RET(attnCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (returnSoftmaxLse && !softmaxLseOut->IsEmpty()) {
        auto lseCopy = l0op::ViewCopy(kernelOuts[1], softmaxLseOut, l0Executor);
        CHECK_RET(lseCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvQuantSparseAttnSharedkv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvQuantSparseAttnSharedkv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
