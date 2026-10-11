/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ACLNN_INDEXER_COMPRESS_EPILOG_V2_H
#define OP_API_INC_ACLNN_INDEXER_COMPRESS_EPILOG_V2_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnIndexerCompressEpilogV2GetWorkspaceSize
 * parameters:
 *  indexerCompressCacheRef : required, the paged indexer cache, UINT8 --
 *                            WRITTEN IN PLACE. May be non-contiguous on
 *                            axis 0; blockStride carries that stride.
 *  x                       : required, rows to store, FP16 or BF16.
 *  slotMapping             : required, destination slot per row, INT32.
 *  layout                  : optional attr, layout selector, default 2.
 *  blockStride             : optional attr, element stride of the block axis.
 *  workspaceSize           : output, size of the device workspace in bytes.
 *  executor                : output, the op execution plan.
 *
 * return: 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 *         161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 *
 * note: indexerCompressCacheRef is a REF parameter -- it is both the input and
 * the output of the operator. No ViewCopy is issued on it, so the executor
 * this call returns is reusable under operator-library manual section 4.31.
 */
ACLNN_API aclnnStatus aclnnIndexerCompressEpilogV2GetWorkspaceSize(aclTensor *indexerCompressCacheRef,
                                                                   const aclTensor *x, const aclTensor *slotMapping,
                                                                   int64_t layout, int64_t blockStride,
                                                                   uint64_t *workspaceSize,
                                                                   aclOpExecutor **executor);

/* function: aclnnIndexerCompressEpilogV2
 * parameters:
 *  workspace     : device memory of at least workspaceSize bytes, or nullptr
 *                  when workspaceSize is 0.
 *  workspaceSize : the value the plan phase returned.
 *  executor      : the plan the plan phase returned.
 *  stream        : an initialized ACL stream.
 */
ACLNN_API aclnnStatus aclnnIndexerCompressEpilogV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
