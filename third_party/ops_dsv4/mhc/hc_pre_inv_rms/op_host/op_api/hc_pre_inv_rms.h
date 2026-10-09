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
 * \file hc_pre_inv_rms.h
 * \brief HcPreInvRms L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_HC_PRE_INV_RMS_OP_H
#define OP_API_INC_LEVEL0_OP_HC_PRE_INV_RMS_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief HcPreInvRms L0 kernel interface (reciprocal RMS over the mHC states).
 *
 * Reduces the stacked mHC states x over their last two axes into the
 * reciprocal RMS scale rsqrt = 1 / sqrt(mean(x^2) + epsilon), the FP32
 * normalizer HcPreSinkhorn consumes. The output drops the two reduced axes and
 * carries a trailing 1: x [bs, hc, d] -> y [bs, 1], x [b, s, hc, d] ->
 * y [b, s, 1].
 *
 * The operator has no REF parameter -- y is a genuine output allocated here
 * from the caller descriptor, so the L2 layer always ViewCopies with
 * src != dst (operator-library manual 4.31).
 *
 * @param [in] x Stacked mHC states, FP32/FP16/BF16.
 * @param [in] epsilon Variance floor added before the reciprocal square root.
 * @param [in] yRef The caller output descriptor whose shape, dtype and format
 * the allocated kernel output copies.
 * @param [in] executor Op executor.
 * @return The freshly allocated y tensor, or nullptr on failure.
 */
const aclTensor *HcPreInvRms(const aclTensor *x, double epsilon, const aclTensor *yRef, aclOpExecutor *executor);

}  // namespace l0op

#endif
