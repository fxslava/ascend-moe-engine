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
 * \file hc_pre_sinkhorn.h
 * \brief HcPreSinkhorn L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_HC_PRE_SINKHORN_OP_H
#define OP_API_INC_LEVEL0_OP_HC_PRE_SINKHORN_OP_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief HcPreSinkhorn L0 kernel interface (mixing projection fused with the
 * Sinkhorn normalization).
 *
 * Takes the already-projected mixing state `mixes` [.., hcMult^2 + 2*hcMult],
 * the reciprocal RMS scale `rsqrt` from HcPreInvRms, the gain triple
 * `hcScale`, the mixing bias `hcBase` and the stacked states `x`, and produces
 * the layer input `y`, the post-mapping state `post` and the doubly-stochastic
 * combine fragment `combFrag` in ONE launch.
 *
 * This is the structural difference from the ops-transformer MhcSinkhorn, which
 * normalizes a square matrix in place: there is no separate Sinkhorn
 * input/output tensor pair here at all, so the same-address ViewCopy that
 * MhcSinkhorn has to elide (operator-library manual 4.31) cannot arise. All
 * three outputs are genuine outputs allocated here from the caller
 * descriptors, so every L2 copy has src != dst.
 *
 * @param [in] mixes Mixing state, FP32, last axis hcMult^2 + 2*hcMult.
 * @param [in] rsqrt Reciprocal RMS scale, FP32, trailing axis 1.
 * @param [in] hcScale Gain triple, [3], FP32.
 * @param [in] hcBase Mixing bias, [hcMult^2 + 2*hcMult], FP32.
 * @param [in] x Stacked mHC states, [.., hc, d], BF16.
 * @param [in] hcMult Hyper-connection multiplicity (4).
 * @param [in] hcSinkhornIters Sinkhorn iteration count.
 * @param [in] hcEps Sinkhorn division guard.
 * @param [in] yRef Caller descriptor for the layer input output.
 * @param [in] postRef Caller descriptor for the post-mapping state output.
 * @param [in] combFragRef Caller descriptor for the combine-fragment output.
 * @param [out] outputs Index 0 receives the allocated y tensor, index 1 post
 * and index 2 combFrag.
 * @param [in] executor Op executor.
 * @return ACLNN_SUCCESS, or the first failing status.
 */
aclnnStatus HcPreSinkhorn(const aclTensor *mixes, const aclTensor *rsqrt, const aclTensor *hcScale,
                          const aclTensor *hcBase, const aclTensor *x, int64_t hcMult, int64_t hcSinkhornIters,
                          double hcEps, const aclTensor *yRef, const aclTensor *postRef,
                          const aclTensor *combFragRef, std::array<const aclTensor *, 3> &outputs,
                          aclOpExecutor *executor);

}  // namespace l0op

#endif
