/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_HC_PRE_SINKHORN_H
#define OP_API_INC_ACLNN_HC_PRE_SINKHORN_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnHcPreSinkhornGetWorkspaceSize
 *
 * The mHC pre-mapping with the mixing projection FUSED into the Sinkhorn
 * normalization. Consumes the projected mixing state `mixes`, the reciprocal
 * RMS scale `rsqrt` that aclnnHcPreInvRms produces, the gain triple `hcScale`,
 * the mixing bias `hcBase` and the stacked states `x`, and emits the layer
 * input `y`, the post-mapping state `post` and the doubly-stochastic combine
 * fragment `combFrag` in ONE launch.
 *
 * This is NOT a rename of aclnnMhcSinkhorn. MhcSinkhorn normalizes a square
 * matrix in place -- it takes the residual mapping hRes that MhcPre already
 * produced and writes the normalized matrix back -- and that in-place shape is
 * exactly what makes its trailing ViewCopy a potential same-address self-copy
 * under operator-library manual section 4.31. HcPreSinkhorn has no such tensor:
 * the normalization is interior to the kernel and only the final combine
 * fragment leaves it, so the copy stage the mhc_sinkhorn wrapper has to elide
 * does not exist here at all.
 *
 * The signature is transcribed from op_host/hc_pre_sinkhorn_def.cpp (inputs
 * mixes, rsqrt, hc_scale, hc_base, x; outputs y, post, comb_frag; OPTIONAL
 * attrs hc_mult Int 4, hc_sinkhorn_iters Int 20, hc_eps Float 1e-6) and
 * confirmed against the two
 * EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x,
 * hc_mult, hc_sinkhorn_iters, hc_eps, y, post, comb_frag) call sites in
 * vllm-ascend csrc/torch_binding.cpp, which place the attrs between the inputs
 * and the outputs.
 *
 * parameters:
 *  mixes           : required, mixing state, FP32, last axis
 *                    hcMult^2 + 2*hcMult (24 at hcMult 4). Upstream produces it
 *                    with at::linear(x.flatten(-2), hcFn); the engine produces
 *                    the same rows with aclnnMatmul under its [K, N] contract.
 *  rsqrt           : required, reciprocal RMS scale from aclnnHcPreInvRms,
 *                    FP32, trailing axis 1.
 *  hcScale         : required, gain triple, [3], FP32.
 *  hcBase          : required, mixing bias, [hcMult^2 + 2*hcMult], FP32.
 *  x               : required, stacked mHC states, [bs, hc, d] or
 *                    [b, s, hc, d], BF16; hc = hcMult = 4 and d is 4096 or
 *                    7168 on 950PR.
 *  hcMult          : optional attr, hyper-connection multiplicity, default 4.
 *  hcSinkhornIters : optional attr, Sinkhorn iteration count, default 20.
 *  hcEps           : optional attr, Sinkhorn division guard, default 1e-6.
 *                    Declared .Float() in the OpDef, which maps to double here.
 *  yOut            : required output, layer input, [bs, d] or [b, s, d], BF16.
 *  postOut         : required output, post-mapping state, [bs, hcMult] or
 *                    [b, s, hcMult], FP32.
 *  combFragOut     : required output, combine fragment,
 *                    [bs, hcMult, hcMult] or [b, s, hcMult, hcMult], FP32.
 *  workspaceSize   : output, size of the device workspace in bytes.
 *  executor        : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: the operator has no REF parameter -- no output name matches an input
 * name. All three outputs are written through distinct executor-owned tensors,
 * so every trailing ViewCopy has src != dst and the executor stays reusable
 * with aclSetAclOpExecutorRepeatable + aclSetTensorAddr.
 */
ACLNN_API aclnnStatus aclnnHcPreSinkhornGetWorkspaceSize(const aclTensor *mixes, const aclTensor *rsqrt,
                                                         const aclTensor *hcScale, const aclTensor *hcBase,
                                                         const aclTensor *x, int64_t hcMult,
                                                         int64_t hcSinkhornIters, double hcEps,
                                                         const aclTensor *yOut, const aclTensor *postOut,
                                                         const aclTensor *combFragOut, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor);

/* function: aclnnHcPreSinkhorn
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value aclnnHcPreSinkhornGetWorkspaceSize returned.
 *  executor      : the plan aclnnHcPreSinkhornGetWorkspaceSize returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnHcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
