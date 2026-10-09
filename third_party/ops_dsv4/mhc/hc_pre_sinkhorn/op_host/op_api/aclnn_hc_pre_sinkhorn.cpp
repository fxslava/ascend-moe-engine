/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hc_pre_sinkhorn.h"

#include <array>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hc_pre_sinkhorn.h"
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

constexpr size_t kXDimMin = 3;            // [bs, hc, d]
constexpr size_t kXDimMax = 4;            // [b, s, hc, d]
constexpr int64_t kHcMultOnly = 4;        // HC_PRE_HC_LIMIT upstream
constexpr int64_t kHcScaleSize = 3;       // HC_SCALE_SIZE upstream
constexpr int64_t kDSupported = 4096;     // HC_PRE_D_LIMIT
constexpr int64_t kDSupportedExt = 7168;  // HC_PRE_D_LIMIT_EXTEND
constexpr int64_t kItersMin = 1;
constexpr int64_t kItersMax = 100;

// Every tensor in this operator is shaped over the same leading axes as x --
// the [bs] or [b, s] prefix left once x's trailing hc and d axes are dropped.
bool LeadingAxesMatch(const Shape &shape, const Shape &xShape, size_t leading)
{
    for (size_t index = 0; index < leading; ++index) {
        if (shape.GetDim(static_cast<int64_t>(index)) != xShape.GetDim(static_cast<int64_t>(index))) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckNotNull(const aclTensor *mixes, const aclTensor *rsqrt, const aclTensor *hcScale,
                         const aclTensor *hcBase, const aclTensor *x, const aclTensor *yOut,
                         const aclTensor *postOut, const aclTensor *combFragOut)
{
    OP_CHECK_NULL(mixes, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(rsqrt, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(hcScale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(hcBase, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(yOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(postOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(combFragOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypes(const aclTensor *mixes, const aclTensor *rsqrt, const aclTensor *hcScale,
                        const aclTensor *hcBase, const aclTensor *x, const aclTensor *yOut, const aclTensor *postOut,
                        const aclTensor *combFragOut)
{
    if (x->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yOut->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const std::array<const aclTensor *, 6> fp32 = {mixes, rsqrt, hcScale, hcBase, postOut, combFragOut};
    for (const aclTensor *tensor : fp32) {
        if (tensor->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mixes, rsqrt, hcScale, hcBase, postOut and combFragOut must all be FP32.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *mixes, const aclTensor *rsqrt, const aclTensor *hcScale,
                        const aclTensor *hcBase, const aclTensor *x, int64_t hcMult, int64_t hcSinkhornIters,
                        double hcEps, const aclTensor *yOut, const aclTensor *postOut, const aclTensor *combFragOut)
{
    auto ret = CheckNotNull(mixes, rsqrt, hcScale, hcBase, x, yOut, postOut, combFragOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckDtypes(mixes, rsqrt, hcScale, hcBase, x, yOut, postOut, combFragOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (hcMult != kHcMultOnly) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcMult only supports %ld, got %ld.", kHcMultOnly, hcMult);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hcSinkhornIters < kItersMin || hcSinkhornIters > kItersMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcSinkhornIters must be in [%ld, %ld], got %ld.", kItersMin, kItersMax,
                hcSinkhornIters);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(hcEps > 0.0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcEps must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto xShape = x->GetViewShape();
    const size_t xDim = xShape.GetDimNum();
    if (xDim < kXDimMin || xDim > kXDimMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be 3-D [bs, hc, d] or 4-D [b, s, hc, d].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t leading = xDim - 2;  // the [bs] or [b, s] prefix
    const int64_t hc = xShape.GetDim(static_cast<int64_t>(xDim) - 2);
    const int64_t d = xShape.GetDim(static_cast<int64_t>(xDim) - 1);
    if (hc != hcMult) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's hc axis must equal hcMult (%ld), got %ld.", hcMult, hc);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (d != kDSupported && d != kDSupportedExt) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's d only supports %ld or %ld, got %ld.", kDSupported, kDSupportedExt, d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t mixRows = hcMult * hcMult + 2 * hcMult;  // 24 at hcMult 4
    if (hcScale->GetViewShape().GetDimNum() != 1 || hcScale->GetViewShape().GetDim(0) != kHcScaleSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcScale must be [%ld].", kHcScaleSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hcBase->GetViewShape().GetDimNum() != 1 || hcBase->GetViewShape().GetDim(0) != mixRows) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcBase must be [%ld] (hcMult^2 + 2*hcMult).", mixRows);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // mixes, rsqrt, yOut and postOut all carry x's leading axes plus one
    // trailing axis; combFragOut carries them plus the [hcMult, hcMult] pair.
    const auto mixesShape = mixes->GetViewShape();
    if (mixesShape.GetDimNum() != leading + 1 || mixesShape.GetDim(static_cast<int64_t>(leading)) != mixRows ||
        !LeadingAxesMatch(mixesShape, xShape, leading)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixes must carry x's leading axes and a trailing %ld.", mixRows);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto rsqrtShape = rsqrt->GetViewShape();
    if (rsqrtShape.GetDimNum() != leading + 1 || rsqrtShape.GetDim(static_cast<int64_t>(leading)) != 1 ||
        !LeadingAxesMatch(rsqrtShape, xShape, leading)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rsqrt must carry x's leading axes and a trailing 1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto yShape = yOut->GetViewShape();
    if (yShape.GetDimNum() != leading + 1 || yShape.GetDim(static_cast<int64_t>(leading)) != d ||
        !LeadingAxesMatch(yShape, xShape, leading)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOut must carry x's leading axes and a trailing d (%ld).", d);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto postShape = postOut->GetViewShape();
    if (postShape.GetDimNum() != leading + 1 || postShape.GetDim(static_cast<int64_t>(leading)) != hcMult ||
        !LeadingAxesMatch(postShape, xShape, leading)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "postOut must carry x's leading axes and a trailing hcMult (%ld).", hcMult);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto combShape = combFragOut->GetViewShape();
    if (combShape.GetDimNum() != leading + 2 || combShape.GetDim(static_cast<int64_t>(leading)) != hcMult ||
        combShape.GetDim(static_cast<int64_t>(leading) + 1) != hcMult ||
        !LeadingAxesMatch(combShape, xShape, leading)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "combFragOut must carry x's leading axes and a trailing [hcMult, hcMult] (%ld x %ld).", hcMult,
                hcMult);
        return ACLNN_ERR_PARAM_INVALID;
    }
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

aclnnStatus aclnnHcPreSinkhornGetWorkspaceSize(const aclTensor *mixes, const aclTensor *rsqrt,
                                               const aclTensor *hcScale, const aclTensor *hcBase, const aclTensor *x,
                                               int64_t hcMult, int64_t hcSinkhornIters, double hcEps,
                                               const aclTensor *yOut, const aclTensor *postOut,
                                               const aclTensor *combFragOut, uint64_t *workspaceSize,
                                               aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnHcPreSinkhorn, DFX_IN(mixes, rsqrt, hcScale, hcBase, x, hcMult, hcSinkhornIters, hcEps),
                   DFX_OUT(yOut, postOut, combFragOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret =
        CheckParams(mixes, rsqrt, hcScale, hcBase, x, hcMult, hcSinkhornIters, hcEps, yOut, postOut, combFragOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || yOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(StageContiguous(mixes, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(rsqrt, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(hcScale, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(hcBase, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(x, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    std::array<const aclTensor *, 3> kernelOuts = {nullptr, nullptr, nullptr};
    ret = l0op::HcPreSinkhorn(mixes, rsqrt, hcScale, hcBase, x, hcMult, hcSinkhornIters, hcEps, yOut, postOut,
                              combFragOut, kernelOuts, l0Executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Three distinct executor-owned sources, so none of these copies is a
    // same-address self-copy (4.31). Note what is deliberately NOT here: a
    // fourth copy for a normalized Sinkhorn matrix. The fusion keeps that
    // tensor inside the kernel, which removes the mhc_sinkhorn hazard by
    // construction rather than eliding it.
    auto yCopy = l0op::ViewCopy(kernelOuts[0], yOut, l0Executor);
    CHECK_RET(yCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto postCopy = l0op::ViewCopy(kernelOuts[1], postOut, l0Executor);
    CHECK_RET(postCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto combCopy = l0op::ViewCopy(kernelOuts[2], combFragOut, l0Executor);
    CHECK_RET(combCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHcPreSinkhorn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
