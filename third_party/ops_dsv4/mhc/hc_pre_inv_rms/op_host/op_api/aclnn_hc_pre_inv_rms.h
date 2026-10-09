/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_HC_PRE_INV_RMS_H
#define OP_API_INC_ACLNN_HC_PRE_INV_RMS_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnHcPreInvRmsGetWorkspaceSize
 *
 * The prologue of the fused mHC pre-mapping: reduce the stacked mHC states
 * over their last two axes into the reciprocal RMS scale
 * 1 / sqrt(mean(x^2) + epsilon), which aclnnHcPreSinkhorn consumes as its
 * `rsqrt` input. The output drops the two reduced axes and keeps a trailing 1,
 * so x [bs, hc, d] gives y [bs, 1] and x [b, s, hc, d] gives y [b, s, 1].
 *
 * The signature is transcribed from op_host/hc_pre_inv_rms_def.cpp (one input
 * x, one output y, one OPTIONAL Float attr epsilon defaulting to 1e-6) and
 * confirmed against the two EXEC_NPU_CMD(aclnnHcPreInvRms, x, <eps>, yOut)
 * call sites in vllm-ascend csrc/torch_binding.cpp, which place the attr
 * between the input and the output.
 *
 * parameters:
 *  x             : required, stacked mHC states, FP32/FP16/BF16.
 *  epsilon       : optional attr, variance floor, default 1e-6. Declared
 *                  .Float() in the OpDef, which maps to double here.
 *  yOut          : required output, reciprocal RMS scale, FP32.
 *  workspaceSize : output, size of the device workspace in bytes.
 *  executor      : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: the operator has no REF parameter. yOut is written through a distinct
 * executor-owned tensor, so its trailing ViewCopy always has src != dst and
 * the executor stays reusable under operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnHcPreInvRmsGetWorkspaceSize(const aclTensor *x, double epsilon, const aclTensor *yOut,
                                                       uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnHcPreInvRms
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value aclnnHcPreInvRmsGetWorkspaceSize returned.
 *  executor      : the plan aclnnHcPreInvRmsGetWorkspaceSize returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnHcPreInvRms(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
