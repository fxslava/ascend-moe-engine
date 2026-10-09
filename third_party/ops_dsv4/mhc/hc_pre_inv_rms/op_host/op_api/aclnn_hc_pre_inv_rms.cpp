/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hc_pre_inv_rms.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hc_pre_inv_rms.h"
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

constexpr size_t kXDimMin = 2;

aclnnStatus CheckParams(const aclTensor *x, double epsilon, const aclTensor *yOut)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);

    const auto xDtype = x->GetDataType();
    if (xDtype != DataType::DT_FLOAT && xDtype != DataType::DT_FLOAT16 && xDtype != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be FP32, FP16 or BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(epsilon >= 0.0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "epsilon must not be negative.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // The reduction consumes the last two axes and leaves a trailing 1, so the
    // output rank is the input rank minus one: x [bs, hc, d] -> y [bs, 1],
    // x [b, s, hc, d] -> y [b, s, 1].
    const auto xShape = x->GetViewShape();
    const size_t xDim = xShape.GetDimNum();
    if (xDim < kXDimMin) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must have at least 2 dimensions to reduce over.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto yShape = yOut->GetViewShape();
    if (yShape.GetDimNum() != xDim - 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut must carry the rank of x minus one.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yShape.GetDim(static_cast<int64_t>(xDim) - 2) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut's trailing dimension must be 1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t index = 0; index + 2 < xDim; ++index) {
        if (yShape.GetDim(static_cast<int64_t>(index)) != xShape.GetDim(static_cast<int64_t>(index))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut's leading dimensions must mirror x's.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnHcPreInvRmsGetWorkspaceSize(const aclTensor *x, double epsilon, const aclTensor *yOut,
                                             uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnHcPreInvRms, DFX_IN(x, epsilon), DFX_OUT(yOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, epsilon, yOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || yOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    const aclTensor *xContiguous = l0op::Contiguous(x, l0Executor);
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelY = l0op::HcPreInvRms(xContiguous, epsilon, yOut, l0Executor);
    CHECK_RET(kernelY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // kernelY is an executor-owned tensor distinct from yOut, so this copy is
    // never a same-address self-copy (4.31).
    auto viewCopyResult = l0op::ViewCopy(kernelY, yOut, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHcPreInvRms(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHcPreInvRms);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
