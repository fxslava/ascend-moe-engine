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
 * \file compressor.cpp
 * \brief Compressor L0 API implementation
 */

#include "compressor.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Compressor);

const aclTensor *Compressor(const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
                            const aclTensor *stateCache, const aclTensor *ape, const aclTensor *normWeight,
                            const aclTensor *ropeSin, const aclTensor *ropeCos,
                            const aclTensor *stateBlockTableOptional, const aclTensor *cuSeqlensOptional,
                            const aclTensor *sequsedOptional, const aclTensor *startPosOptional,
                            int64_t ropeHeadDim, int64_t cmpRatio, int64_t coff, double normEps, int64_t rotaryMode,
                            int64_t cacheMode, int64_t stateCacheStrideDim0, const aclTensor *cmpKvRef,
                            aclOpExecutor *executor)
{
    L0_DFX(Compressor, x, wkv, wgate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTableOptional,
           cuSeqlensOptional, sequsedOptional, startPosOptional, ropeHeadDim, cmpRatio, coff, normEps, rotaryMode,
           cacheMode, stateCacheStrideDim0);

    // cmpKv is shaped by the caller (the engine sizes its own arena slots), so
    // the kernel output mirrors the caller descriptor rather than going
    // through INFER_SHAPE -- which would need the operator proto registered in
    // the deployed OPP vendor package just to size a tensor the caller already
    // knows. A distinct tensor is allocated on purpose: the L2 layer then
    // ViewCopies into the caller output with src != dst, which keeps the
    // executor reusable under operator-library manual 4.31.
    auto cmpKv = executor->AllocTensor(cmpKvRef->GetViewShape(), cmpKvRef->GetDataType(), cmpKvRef->GetViewFormat());
    if (cmpKv == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Compressor failed to allocate the cmpKv output tensor.");
        return nullptr;
    }

    // stateCache is listed as both an input and an output: the kernel updates
    // the recurrent pooling state in place.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Compressor,
        OP_INPUT(x, wkv, wgate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTableOptional,
                 cuSeqlensOptional, sequsedOptional, startPosOptional),
        OP_OUTPUT(cmpKv, stateCache),
        OP_ATTR(ropeHeadDim, cmpRatio, coff, normEps, rotaryMode, cacheMode, stateCacheStrideDim0));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Compressor ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return cmpKv;
}

}  // namespace l0op
