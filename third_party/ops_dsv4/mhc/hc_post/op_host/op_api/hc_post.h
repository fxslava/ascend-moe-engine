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
 * \file hc_post.h
 * \brief HcPost L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_HC_POST_OP_H
#define OP_API_INC_LEVEL0_OP_HC_POST_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief HcPost L0 kernel interface (the fused mHC residual combine).
 *
 * Combines the layer output x [b, s, d] back into the hyper-connection
 * residual streams: y = comb^T @ residual + x * post, with residual and y
 * [b, s, hc, d], post [b, s, hc] and comb [b, s, hc, hc]. This is the BSHD
 * counterpart of the ops-transformer MhcPost, which takes the same algebra in
 * the TND layout.
 *
 * The operator declares no attributes and has no REF parameter: y is a genuine
 * output allocated here from the caller descriptor, so the L2 layer always
 * ViewCopies with src != dst (operator-library manual 4.31).
 *
 * @param [in] x Layer output, [b, s, d], FP32/FP16/BF16.
 * @param [in] residual Hyper-connection residual streams, [b, s, hc, d], dtype
 * of x.
 * @param [in] post Post-mapping state, [b, s, hc], FP32/FP16/BF16.
 * @param [in] comb Combine matrix, [b, s, hc, hc], dtype of post.
 * @param [in] yRef The caller output descriptor whose shape, dtype and format
 * the allocated kernel output copies.
 * @param [in] executor Op executor.
 * @return The freshly allocated y tensor, or nullptr on failure.
 */
const aclTensor *HcPost(const aclTensor *x, const aclTensor *residual, const aclTensor *post, const aclTensor *comb,
                        const aclTensor *yRef, aclOpExecutor *executor);

}  // namespace l0op

#endif
