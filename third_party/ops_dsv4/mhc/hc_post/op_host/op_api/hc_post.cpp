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
 * \file hc_post.cpp
 * \brief HcPost L0 API implementation
 */

#include "hc_post.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(HcPost);

const aclTensor *HcPost(const aclTensor *x, const aclTensor *residual, const aclTensor *post, const aclTensor *comb,
                        const aclTensor *yRef, aclOpExecutor *executor)
{
    L0_DFX(HcPost, x, residual, post, comb);

    // y mirrors the caller descriptor the engine already sized rather than
    // going through INFER_SHAPE, which would need the operator proto
    // registered in the deployed OPP vendor package just to size a tensor the
    // caller already knows. A distinct tensor is allocated on purpose: the L2
    // layer then ViewCopies into the caller output with src != dst, which keeps
    // the executor reusable under operator-library manual 4.31.
    auto y = executor->AllocTensor(yRef->GetViewShape(), yRef->GetDataType(), yRef->GetViewFormat());
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPost failed to allocate the y output tensor.");
        return nullptr;
    }

    // HcPost declares no attributes, so OP_ATTR() is empty.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HcPost, OP_INPUT(x, residual, post, comb), OP_OUTPUT(y));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPost ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return y;
}

}  // namespace l0op
