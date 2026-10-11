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
 * \file hc_pre.h
 * \brief HcPre L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_HC_PRE_OP_H
#define OP_API_INC_LEVEL0_OP_HC_PRE_OP_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief HcPre L0 kernel interface (the fully fused mHC pre-mapping).
 *
 * The whole prologue in one launch: the reciprocal-RMS reduction that
 * HcPreInvRms does on its own, the mixing projection x.flatten(-2) @ hcFn^T
 * that upstream reaches through at::linear, and the Sinkhorn normalization that
 * HcPreSinkhorn does -- all internal. It consumes only the stacked states x and
 * the mixing weights hcFn (plus the gain triple hcScale and the bias hcBase)
 * and emits the layer input y, the post-mapping state post and the
 * doubly-stochastic combine fragment combFrag.
 *
 * No intermediate leaves the kernel, so neither the rsqrt tensor nor the mixes
 * tensor nor a normalized Sinkhorn matrix exists at this layer. All three
 * outputs are genuine outputs allocated here from the caller descriptors, so
 * every L2 copy has src != dst (operator-library manual 4.31).
 *
 * @param [in] x Stacked mHC states, [bs, hc, d] or [b, s, hc, d], BF16.
 * @param [in] hcFn Mixing weights, [hcMult^2 + 2*hcMult, hc*d], FP32.
 * @param [in] hcScale Gain triple, [3], FP32.
 * @param [in] hcBase Mixing bias, [hcMult^2 + 2*hcMult], FP32.
 * @param [in] hcMult Hyper-connection multiplicity (4).
 * @param [in] hcSinkhornIters Sinkhorn iteration count.
 * @param [in] hcEps Sinkhorn division guard.
 * @param [in] normEps RMS normalization epsilon.
 * @param [in] yRef Caller descriptor for the layer input output.
 * @param [in] postRef Caller descriptor for the post-mapping state output.
 * @param [in] combFragRef Caller descriptor for the combine-fragment output.
 * @param [out] outputs Index 0 receives the allocated y tensor, index 1 post
 * and index 2 combFrag.
 * @param [in] executor Op executor.
 * @return ACLNN_SUCCESS, or the first failing status.
 */
aclnnStatus HcPre(const aclTensor *x, const aclTensor *hcFn, const aclTensor *hcScale, const aclTensor *hcBase,
                  int64_t hcMult, int64_t hcSinkhornIters, double hcEps, double normEps, const aclTensor *yRef,
                  const aclTensor *postRef, const aclTensor *combFragRef, std::array<const aclTensor *, 3> &outputs,
                  aclOpExecutor *executor);

}  // namespace l0op

#endif
