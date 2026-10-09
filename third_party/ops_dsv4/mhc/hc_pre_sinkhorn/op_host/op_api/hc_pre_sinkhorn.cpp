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
 * \file hc_pre_sinkhorn.cpp
 * \brief HcPreSinkhorn L0 API implementation
 */

#include "hc_pre_sinkhorn.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(HcPreSinkhorn);

aclnnStatus HcPreSinkhorn(const aclTensor *mixes, const aclTensor *rsqrt, const aclTensor *hcScale,
                          const aclTensor *hcBase, const aclTensor *x, int64_t hcMult, int64_t hcSinkhornIters,
                          double hcEps, const aclTensor *yRef, const aclTensor *postRef,
                          const aclTensor *combFragRef, std::array<const aclTensor *, 3> &outputs,
                          aclOpExecutor *executor)
{
    L0_DFX(HcPreSinkhorn, mixes, rsqrt, hcScale, hcBase, x, hcMult, hcSinkhornIters, hcEps);

    // All three outputs mirror the caller descriptors the engine already
    // sized, so no INFER_SHAPE round trip through the deployed OPP proto is
    // needed, and they are distinct tensors so every L2 ViewCopy has
    // src != dst (manual 4.31). Note what is NOT here: the operator fuses the
    // mixing projection with the Sinkhorn normalization, so there is no
    // in-place square matrix to normalize and therefore no copy stage whose
    // source and destination could coincide.
    auto y = executor->AllocTensor(yRef->GetViewShape(), yRef->GetDataType(), yRef->GetViewFormat());
    if (y == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPreSinkhorn failed to allocate the y output tensor.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto post = executor->AllocTensor(postRef->GetViewShape(), postRef->GetDataType(), postRef->GetViewFormat());
    if (post == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPreSinkhorn failed to allocate the post output tensor.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto combFrag =
        executor->AllocTensor(combFragRef->GetViewShape(), combFragRef->GetDataType(), combFragRef->GetViewFormat());
    if (combFrag == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPreSinkhorn failed to allocate the combFrag output tensor.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(HcPreSinkhorn, OP_INPUT(mixes, rsqrt, hcScale, hcBase, x),
                                           OP_OUTPUT(y, post, combFrag),
                                           OP_ATTR(hcMult, hcSinkhornIters, hcEps));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HcPreSinkhorn ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    outputs[0] = y;
    outputs[1] = post;
    outputs[2] = combFrag;
    return ACLNN_SUCCESS;
}

}  // namespace l0op
