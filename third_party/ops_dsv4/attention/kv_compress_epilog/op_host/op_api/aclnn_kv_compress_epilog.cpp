/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kv_compress_epilog.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "kv_compress_epilog.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

// The file-scope unnamed namespace must sit OUTSIDE the extern "C" block:
// a function declared inside extern "C" gets C language linkage, which is
// EXTERNAL even within an unnamed namespace, so helpers named CheckParams or
// StageContiguous here would collide at link time with the identically named
// helpers in the other operators' op_api files.
namespace {

constexpr size_t kCacheDimMin = 3;
constexpr size_t kCacheDimMax = 4;

aclnnStatus CheckParams(const aclTensor *kvCompressCacheRef, const aclTensor *x, const aclTensor *slotMapping)
{
    OP_CHECK_NULL(kvCompressCacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(slotMapping, return ACLNN_ERR_PARAM_NULLPTR);

    const auto cacheDtype = kvCompressCacheRef->GetDataType();
    if (cacheDtype != DataType::DT_FLOAT8_E5M2 && cacheDtype != DataType::DT_FLOAT8_E4M3FN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvCompressCacheRef must be FP8 E5M2 or FP8 E4M3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x->GetDataType() != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto slotDtype = slotMapping->GetDataType();
    if (slotDtype != DataType::DT_INT32 && slotDtype != DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "slotMapping must be INT32 or INT64.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t cacheDim = kvCompressCacheRef->GetViewShape().GetDimNum();
    if (cacheDim < kCacheDimMin || cacheDim > kCacheDimMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvCompressCacheRef must be 3D or 4D.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(aclTensor *kvCompressCacheRef, const aclTensor *x,
                                                  const aclTensor *slotMapping, int64_t quantGroupSize,
                                                  int64_t quantMode, int64_t roundScale, int64_t layout,
                                                  int64_t blockStride, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKvCompressEpilog,
                   DFX_IN(kvCompressCacheRef, x, slotMapping, quantGroupSize, quantMode, roundScale, layout,
                          blockStride),
                   DFX_OUT(kvCompressCacheRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(kvCompressCacheRef, x, slotMapping);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || slotMapping->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x and slotMapping are AutoContiguous in the operator definition, so they
    // are staged here. kvCompressCacheRef is IgnoreContiguous and a REF
    // output: it is handed to the kernel exactly as the caller owns it, and
    // its axis-0 stride travels as the blockStride attribute instead.
    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *slotContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut = l0op::KvCompressEpilog(kvCompressCacheRef, xContiguous, slotContiguous,
                                                        quantGroupSize, quantMode, roundScale, layout, blockStride,
                                                        uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // No trailing ViewCopy: the cache IS the output. See the 4.31 note in the
    // header -- a copy here would be a same-address self-copy and would make
    // the returned executor non-reusable.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvCompressEpilog(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKvCompressEpilog);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
