/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_indexer_compress_epilog_v2.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "indexer_compress_epilog_v2.h"
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

aclnnStatus CheckParams(const aclTensor *cacheRef, const aclTensor *x, const aclTensor *slotMapping)
{
    OP_CHECK_NULL(cacheRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(slotMapping, return ACLNN_ERR_PARAM_NULLPTR);

    if (cacheRef->GetDataType() != DataType::DT_UINT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indexerCompressCacheRef must be UINT8.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto xDtype = x->GetDataType();
    if (xDtype != DataType::DT_FLOAT16 && xDtype != DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be FP16 or BF16.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (slotMapping->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "slotMapping must be INT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnIndexerCompressEpilogV2GetWorkspaceSize(aclTensor *indexerCompressCacheRef, const aclTensor *x,
                                                         const aclTensor *slotMapping, int64_t layout,
                                                         int64_t blockStride, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnIndexerCompressEpilogV2,
                   DFX_IN(indexerCompressCacheRef, x, slotMapping, layout, blockStride),
                   DFX_OUT(indexerCompressCacheRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(indexerCompressCacheRef, x, slotMapping);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || slotMapping->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // The cache is IgnoreContiguous and a REF output, so it goes to the kernel
    // untouched and its axis-0 stride travels as blockStride. x and
    // slotMapping are AutoContiguous and are staged here.
    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *slotContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut = l0op::IndexerCompressEpilogV2(indexerCompressCacheRef, xContiguous, slotContiguous,
                                                               layout, blockStride, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // No trailing ViewCopy -- the cache IS the output (4.31, see the header).
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexerCompressEpilogV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndexerCompressEpilogV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
