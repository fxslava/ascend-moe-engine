/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hc_post.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hc_post.h"
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

constexpr size_t kXDim = 3;         // x    [b, s, d]
constexpr size_t kResidualDim = 4;  // residual / comb / y
constexpr size_t kPostDim = 3;      // post [b, s, hc]

bool IsFloatDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT || dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16;
}

aclnnStatus CheckNotNull(const aclTensor *x, const aclTensor *residual, const aclTensor *post, const aclTensor *comb,
                         const aclTensor *yOut)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(residual, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(post, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(comb, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *residual, const aclTensor *post, const aclTensor *comb,
                        const aclTensor *yOut)
{
    auto ret = CheckNotNull(x, residual, post, comb, yOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const auto xDtype = x->GetDataType();
    if (!IsFloatDtype(xDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be FP32, FP16 or BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (residual->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "residual must share the dtype of x.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yOut->GetDataType() != residual->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut must share the dtype of residual.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto postDtype = post->GetDataType();
    if (!IsFloatDtype(postDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "post must be FP32, FP16 or BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (comb->GetDataType() != postDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "comb must share the dtype of post.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto xShape = x->GetViewShape();
    if (xShape.GetDimNum() != kXDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be 3-D [b, s, d].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t batch = xShape.GetDim(0);
    const int64_t sequence = xShape.GetDim(1);
    const int64_t d = xShape.GetDim(2);
    if (batch <= 0 || sequence <= 0 || d <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dimensions must all be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto residualShape = residual->GetViewShape();
    if (residualShape.GetDimNum() != kResidualDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "residual must be 4-D [b, s, hc, d].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t hc = residualShape.GetDim(2);
    if (hc <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "residual's hc must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (residualShape.GetDim(0) != batch || residualShape.GetDim(1) != sequence || residualShape.GetDim(3) != d) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "residual must be [b, s, hc, d] over x's b, s and d.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto postShape = post->GetViewShape();
    if (postShape.GetDimNum() != kPostDim || postShape.GetDim(0) != batch || postShape.GetDim(1) != sequence ||
        postShape.GetDim(2) != hc) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "post must be [b, s, hc].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto combShape = comb->GetViewShape();
    if (combShape.GetDimNum() != kResidualDim || combShape.GetDim(0) != batch || combShape.GetDim(1) != sequence ||
        combShape.GetDim(2) != hc || combShape.GetDim(3) != hc) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "comb must be [b, s, hc, hc].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // The output mirrors residual: same four axes, same dtype.
    OP_CHECK_SHAPE_NOT_EQUAL(yOut, residual, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// Stage one AutoContiguous input.
aclnnStatus StageContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    return tensor != nullptr ? ACLNN_SUCCESS : ACLNN_ERR_INNER_NULLPTR;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnHcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *post,
                                        const aclTensor *comb, const aclTensor *yOut, uint64_t *workspaceSize,
                                        aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnHcPost, DFX_IN(x, residual, post, comb), DFX_OUT(yOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, residual, post, comb, yOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || yOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(StageContiguous(x, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(residual, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(post, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(comb, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelY = l0op::HcPost(x, residual, post, comb, yOut, l0Executor);
    CHECK_RET(kernelY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // kernelY is an executor-owned tensor distinct from yOut, so this copy is
    // never a same-address self-copy (4.31).
    auto viewCopyResult = l0op::ViewCopy(kernelY, yOut, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHcPost);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
