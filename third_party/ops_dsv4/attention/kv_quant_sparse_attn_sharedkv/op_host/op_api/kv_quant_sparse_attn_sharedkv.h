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
 * \file kv_quant_sparse_attn_sharedkv.h
 * \brief KvQuantSparseAttnSharedkv L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_KV_QUANT_SPARSE_ATTN_SHAREDKV_OP_H
#define OP_API_INC_LEVEL0_OP_KV_QUANT_SPARSE_ATTN_SHAREDKV_OP_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief KvQuantSparseAttnSharedkv L0 kernel interface.
 *
 * The dedicated arch35 MQA core attention over the quantized hybrid KV cache:
 * one BF16 query stream attends to both the uncompressed (ori) and the
 * compressed (cmp) FP8 E4M3 KV streams at their own sparse indices and mask
 * modes, with optional attention sinks folded into the softmax denominator.
 *
 * Both outputs are allocated here from the caller descriptors, so the L2 layer
 * always ViewCopies with src != dst (operator-library manual 4.31).
 *
 * @param [out] outputs Index 0 receives the allocated attnOut tensor and index
 * 1 the allocated softmaxLse tensor.
 * @return ACLNN_SUCCESS, or the first failing status.
 */
aclnnStatus KvQuantSparseAttnSharedkv(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional, const aclTensor *metadataOptional, int64_t kvQuantMode, int64_t tileSize,
    int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional, const char *layoutKvOptional,
    int64_t oriKvStride0, int64_t cmpKvStride0, bool returnSoftmaxLse, const aclTensor *attnOutRef,
    const aclTensor *softmaxLseRef, std::array<const aclTensor *, 2> &outputs, aclOpExecutor *executor);

}  // namespace l0op

#endif
