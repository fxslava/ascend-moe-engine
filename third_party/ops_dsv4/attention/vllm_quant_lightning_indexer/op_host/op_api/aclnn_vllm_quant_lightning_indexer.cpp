/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_vllm_quant_lightning_indexer.h"

#include <array>
#include <cstring>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "vllm_quant_lightning_indexer.h"

using namespace op;

// The file-scope unnamed namespace must sit OUTSIDE the extern "C" block:
// a function declared inside extern "C" gets C language linkage, which is
// EXTERNAL even within an unnamed namespace, so helpers named CheckParams or
// StageContiguous here would collide at link time with the identically named
// helpers in the other operators' op_api files.
namespace {

constexpr int64_t kIndexerHeadDim = 128;
constexpr int64_t kSparseCountMin = 1;
constexpr int64_t kSparseCountMax = 2048;
constexpr size_t kQueryDimTnd = 3;
constexpr size_t kQueryDimBsnd = 4;

bool LayoutIs(const char *layout, const char *expected)
{
    return layout != nullptr && std::strcmp(layout, expected) == 0;
}

aclnnStatus CheckNotNull(const aclTensor *query, const aclTensor *key, const aclTensor *weights,
                         const aclTensor *queryDequantScale, const aclTensor *keyDequantScale,
                         const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut)
{
    OP_CHECK_NULL(query, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(key, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(weights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(queryDequantScale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(keyDequantScale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sparseIndicesOut, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sparseValuesOut, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *query, const aclTensor *key, const aclTensor *weights,
                        const aclTensor *queryDequantScale, const aclTensor *keyDequantScale,
                        const aclTensor *actualSeqLengthsQueryOptional,
                        const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional,
                        const char *layoutQueryOptional, const char *layoutKeyOptional, int64_t sparseCount,
                        const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut)
{
    auto ret = CheckNotNull(query, key, weights, queryDequantScale, keyDequantScale, sparseIndicesOut,
                            sparseValuesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const auto queryDtype = query->GetDataType();
    if (queryDtype != DataType::DT_FLOAT8_E4M3FN && queryDtype != DataType::DT_HIFLOAT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query must be FP8 E4M3 or HiFloat8.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (key->GetDataType() != queryDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key must share the dtype of query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseIndicesOut->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseIndicesOut must be INT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseValuesOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseValuesOut must be FP32.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const size_t queryDim = query->GetViewShape().GetDimNum();
    if (queryDim != kQueryDimTnd && queryDim != kQueryDimBsnd) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query must be TND [T,N1,D] or BSND [B,S,N1,D].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (query->GetViewShape().GetDim(queryDim - 1) != kIndexerHeadDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the indexer head dim is 128.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (sparseCount < kSparseCountMin || sparseCount > kSparseCountMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseCount must be in [1, 2048].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // TND needs both cumulative-length vectors; a paged key needs the key
    // lengths and the block table.
    if (LayoutIs(layoutQueryOptional, "TND") && actualSeqLengthsQueryOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a TND query layout requires actualSeqLengthsQueryOptional.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((LayoutIs(layoutKeyOptional, "TND") || LayoutIs(layoutKeyOptional, "PA_BSND")) &&
        actualSeqLengthsKeyOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a TND or PA_BSND key layout requires actualSeqLengthsKeyOptional.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (LayoutIs(layoutKeyOptional, "PA_BSND") && blockTableOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a PA_BSND key layout requires blockTableOptional.");
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

aclnnStatus aclnnVllmQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional,
    const aclTensor *metadataOptional, int64_t queryQuantMode, int64_t keyQuantMode, char *layoutQueryOptional,
    char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, bool returnValues, int64_t stride, int64_t scaleStride, const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnVllmQuantLightningIndexer,
                   DFX_IN(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
                          actualSeqLengthsKeyOptional, blockTableOptional, metadataOptional, queryQuantMode,
                          keyQuantMode, layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens,
                          nextTokens, cmpRatio, returnValues, stride, scaleStride),
                   DFX_OUT(sparseIndicesOut, sparseValuesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
                           actualSeqLengthsKeyOptional, blockTableOptional, layoutQueryOptional, layoutKeyOptional,
                           sparseCount, sparseIndicesOut, sparseValuesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (sparseIndicesOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    // key and keyDequantScale are deliberately NOT staged: under PA_BSND they
    // are contiguous on every axis except axis 0, and that axis-0 stride is
    // what the stride / scaleStride attributes carry to the kernel. Running
    // Contiguous on them would flatten the paged view and lose the mapping.
    CHECK_RET(StageContiguous(query, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(weights, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(queryDequantScale, l0Executor, false) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(actualSeqLengthsQueryOptional, l0Executor, true) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(actualSeqLengthsKeyOptional, l0Executor, true) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(blockTableOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(StageContiguous(metadataOptional, l0Executor, true) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    std::array<const aclTensor *, 2> kernelOuts = {nullptr, nullptr};
    ret = l0op::VllmQuantLightningIndexer(
        query, key, weights, queryDequantScale, keyDequantScale, actualSeqLengthsQueryOptional,
        actualSeqLengthsKeyOptional, blockTableOptional, metadataOptional, queryQuantMode, keyQuantMode,
        layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, cmpRatio,
        returnValues, stride, scaleStride, sparseIndicesOut, sparseValuesOut, kernelOuts, l0Executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Distinct executor-owned sources, so neither copy is a same-address
    // self-copy (4.31). sparseValues is only copied back when the caller asked
    // for it; otherwise sparseValuesOut is the [0] placeholder tensor.
    auto indicesCopy = l0op::ViewCopy(kernelOuts[0], sparseIndicesOut, l0Executor);
    CHECK_RET(indicesCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (returnValues && !sparseValuesOut->IsEmpty()) {
        auto valuesCopy = l0op::ViewCopy(kernelOuts[1], sparseValuesOut, l0Executor);
        CHECK_RET(valuesCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnVllmQuantLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnVllmQuantLightningIndexer);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
