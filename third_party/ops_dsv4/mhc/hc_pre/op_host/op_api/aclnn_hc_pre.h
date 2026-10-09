/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_HC_PRE_H
#define OP_API_INC_ACLNN_HC_PRE_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnHcPreGetWorkspaceSize
 *
 * The FULLY fused mHC pre-mapping: one launch that performs the
 * reciprocal-RMS reduction, the mixing projection and the Sinkhorn
 * normalization, consuming only the stacked states x and the mixing weights
 * hcFn (plus the gain triple hcScale and the bias hcBase) and emitting the
 * layer input y, the post-mapping state post and the doubly-stochastic combine
 * fragment combFrag.
 *
 * This is the fusion end of a three-way decomposition upstream ships:
 *   aclnnHcPre                                       one launch   (this op)
 *   aclnnHcPreInvRms + a mixing GEMM + aclnnHcPreSinkhorn  three launches
 *   aclnnMhcPre + aclnnMhcSinkhorn (ops-transformer)       two launches,
 *                                                         plus the in-place
 *                                                         Sinkhorn hazard
 * vllm-ascend's npu_hc_pre_npu takes the composite path and npu_hc_pre_v2_npu
 * takes this one; both are exercised upstream, so the two are numerically
 * interchangeable configurations of the same mapping.
 *
 * The signature is transcribed from op_host/hc_pre_def.cpp (inputs x, hc_fn,
 * hc_scale, hc_base; outputs y, post, comb_frag; OPTIONAL attrs hc_mult Int 4,
 * hc_sinkhorn_iters Int 20, hc_eps Float 1e-6, norm_eps Float 1e-6) and
 * confirmed against the
 * EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult,
 * hc_sinkhorn_iters, hc_eps, norm_eps, y, post, comb_frag) call site in
 * vllm-ascend csrc/torch_binding.cpp, which places the four attrs between the
 * inputs and the outputs. Note the attr ORDER: hc_eps precedes norm_eps, the
 * reverse of the order the two epsilons appear in aclnnMhcPre.
 *
 * parameters:
 *  x               : required, stacked mHC states, [bs, hc, d] or
 *                    [b, s, hc, d], BF16; hc = hcMult = 4 and d is 4096 or
 *                    7168 on 950PR.
 *  hcFn            : required, mixing weights,
 *                    [hcMult^2 + 2*hcMult, hc*d] = [24, 16384] at d 4096, FP32.
 *  hcScale         : required, gain triple, [3], FP32.
 *  hcBase          : required, mixing bias, [hcMult^2 + 2*hcMult], FP32.
 *  hcMult          : optional attr, hyper-connection multiplicity, default 4.
 *  hcSinkhornIters : optional attr, Sinkhorn iteration count, default 20.
 *  hcEps           : optional attr, Sinkhorn division guard, default 1e-6.
 *                    Declared .Float() in the OpDef, which maps to double here.
 *  normEps         : optional attr, RMS normalization epsilon, default 1e-6.
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
ACLNN_API aclnnStatus aclnnHcPreGetWorkspaceSize(const aclTensor *x, const aclTensor *hcFn, const aclTensor *hcScale,
                                                 const aclTensor *hcBase, int64_t hcMult, int64_t hcSinkhornIters,
                                                 double hcEps, double normEps, const aclTensor *yOut,
                                                 const aclTensor *postOut, const aclTensor *combFragOut,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnHcPre
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value aclnnHcPreGetWorkspaceSize returned.
 *  executor      : the plan aclnnHcPreGetWorkspaceSize returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnHcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
