/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_compressor.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "compressor.h"
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
constexpr size_t kXDimMax = 3;
constexpr size_t kStateCacheDim = 3;
constexpr size_t kNormWeightDim = 1;
constexpr int64_t kCmpRatioCsa = 4;
constexpr int64_t kCmpRatioHca = 128;

aclnnStatus CheckNotNull(const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
                         const aclTensor *stateCacheRef, const aclTensor *ape, const aclTensor *normWeight,
                         const aclTensor *ropeSin, const aclTensor *ropeCos, const aclTensor *cmpKvOut)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(wkv, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(wgate, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(stateCacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(ape, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(normWeight, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(ropeSin, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(ropeCos, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(cmpKvOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
                        const aclTensor *stateCacheRef, const aclTensor *ape, const aclTensor *normWeight,
                        const aclTensor *ropeSin, const aclTensor *ropeCos, int64_t ropeHeadDim, int64_t cmpRatio,
                        double normEps, const aclTensor *cmpKvOut)
{
    auto ret = CheckNotNull(x, wkv, wgate, stateCacheRef, ape, normWeight, ropeSin, ropeCos, cmpKvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const auto xDtype = x->GetDataType();
    if (xDtype != DataType::DT_BF16 && xDtype != DataType::DT_FLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be BF16 or FP16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wkv->GetDataType() != xDtype || wgate->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "wkv and wgate must share the dtype of x.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cmpKvOut->GetDataType() != xDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpKvOut must share the dtype of x.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stateCacheRef->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stateCacheRef must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (ape->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ape must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const size_t xDim = x->GetViewShape().GetDimNum();
    if (xDim < kXDimMin || xDim > kXDimMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be 2-D [T, H] or 3-D [B, S, H].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (ropeSin->GetViewShape().GetDimNum() != xDim || ropeCos->GetViewShape().GetDimNum() != xDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ropeSin and ropeCos must carry the rank of x.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (normWeight->GetViewShape().GetDimNum() != kNormWeightDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "normWeight must be 1-D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stateCacheRef->GetViewShape().GetDimNum() != kStateCacheDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stateCacheRef must be 3-D.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (cmpRatio != kCmpRatioCsa && cmpRatio != kCmpRatioHca) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmpRatio must be 4 (CSA) or 128 (HCA).");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (ropeHeadDim <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ropeHeadDim must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(normEps > 0.0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "normEps must be positive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// Stage one AutoContiguous input. A null optional stays null.
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

aclnnStatus aclnnCompressorGetWorkspaceSize(const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
                                            aclTensor *stateCacheRef, const aclTensor *ape,
                                            const aclTensor *normWeight, const aclTensor *ropeSin,
                                            const aclTensor *ropeCos, const aclTensor *stateBlockTableOptional,
                                            const aclTensor *cuSeqlensOptional, const aclTensor *sequsedOptional,
                                            const aclTensor *startPosOptional, int64_t ropeHeadDim, int64_t cmpRatio,
                                            int64_t coff, double normEps, int64_t rotaryMode, int64_t cacheMode,
                                            int64_t stateCacheStrideDim0, const aclTensor *cmpKvOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCompressor,
                   DFX_IN(x, wkv, wgate, stateCacheRef, ape, normWeight, ropeSin, ropeCos, stateBlockTableOptional,
                          cuSeqlensOptional, sequsedOptional, startPosOptional, ropeHeadDim, cmpRatio, coff, normEps,
                          rotaryMode, cacheMode, stateCacheStrideDim0),
                   DFX_OUT(cmpKvOut, stateCacheRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, wkv, wgate, stateCacheRef, ape, normWeight, ropeSin, ropeCos, ropeHeadDim, cmpRatio,
                           normEps, cmpKvOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || cmpKvOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // Every AutoContiguous input is staged; stateCacheRef is IgnoreContiguous
    // and a REF output, so it is handed to the kernel exactly as the caller
    // owns it and its axis-0 stride travels as stateCacheStrideDim0.
    CHECK_RET(StageContiguous(x, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(wkv, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(wgate, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(ape, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(normWeight, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(ropeSin, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(ropeCos, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(stateBlockTableOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(cuSeqlensOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(sequsedOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(startPosOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelCmpKv = l0op::Compressor(
        x, wkv, wgate, stateCacheRef, ape, normWeight, ropeSin, ropeCos, stateBlockTableOptional, cuSeqlensOptional,
        sequsedOptional, startPosOptional, ropeHeadDim, cmpRatio, coff, normEps, rotaryMode, cacheMode,
        stateCacheStrideDim0, cmpKvOut, l0Executor);
    CHECK_RET(kernelCmpKv != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // kernelCmpKv is an executor-owned tensor distinct from cmpKvOut, so this
    // copy is never a same-address self-copy (4.31). stateCacheRef gets no
    // copy at all -- the kernel already updated it in place.
    auto viewCopyResult = l0op::ViewCopy(kernelCmpKv, cmpKvOut, l0Executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCompressor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCompressor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
