/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_HC_POST_H
#define OP_API_INC_ACLNN_HC_POST_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnHcPostGetWorkspaceSize
 *
 * The fused mHC residual combine: fold the layer output x back into the
 * hyper-connection residual streams as y = comb^T @ residual + x * post. It is
 * the BSHD counterpart of the ops-transformer aclnnMhcPost, which carries the
 * same algebra in the TND layout.
 *
 * The signature is transcribed from op_host/hc_post_def.cpp (four inputs
 * x, residual, post, comb; one output y; NO attributes) and confirmed against
 * the EXEC_NPU_CMD(aclnnHcPost, x, residual, post, comb, out) call site in
 * vllm-ascend csrc/torch_binding.cpp.
 *
 * parameters:
 *  x             : required, layer output, [b, s, d], FP32/FP16/BF16.
 *  residual      : required, residual streams, [b, s, hc, d], dtype of x.
 *  post          : required, post-mapping state, [b, s, hc], FP32/FP16/BF16.
 *  comb          : required, combine matrix, [b, s, hc, hc], dtype of post.
 *  yOut          : required output, [b, s, hc, d], dtype of residual.
 *  workspaceSize : output, size of the device workspace in bytes.
 *  executor      : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: the operator has no REF parameter -- the output name `y` matches no
 * input name. yOut is written through a distinct executor-owned tensor, so its
 * trailing ViewCopy always has src != dst and the executor stays reusable under
 * operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnHcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *residual,
                                                  const aclTensor *post, const aclTensor *comb,
                                                  const aclTensor *yOut, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor);

/* function: aclnnHcPost
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value aclnnHcPostGetWorkspaceSize returned.
 *  executor      : the plan aclnnHcPostGetWorkspaceSize returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnHcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
