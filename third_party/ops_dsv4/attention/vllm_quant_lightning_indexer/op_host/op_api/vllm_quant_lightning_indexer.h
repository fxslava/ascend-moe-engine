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
 * \file vllm_quant_lightning_indexer.h
 * \brief VllmQuantLightningIndexer L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_VLLM_QUANT_LIGHTNING_INDEXER_OP_H
#define OP_API_INC_LEVEL0_OP_VLLM_QUANT_LIGHTNING_INDEXER_OP_H

#include <array>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief VllmQuantLightningIndexer L0 kernel interface.
 *
 * The mixed Cube/Vector arch35 sparse-attention front end: FP8 E4M3 or
 * HiFloat8 query/key are correlated under their dequant scales, the scores are
 * weighted by `weights`, and the top-`sparseCount` blocks are emitted as INT32
 * indices (plus optional FP32 values).
 *
 * This differs from the ops-transformer QuantLightningIndexer by the extra
 * `metadata` input, the `cmpRatio` / `returnValues` attributes and the second
 * `sparseValues` output.
 *
 * Both outputs are allocated here from the caller descriptors, so the L2 layer
 * always ViewCopies with src != dst (operator-library manual 4.31).
 *
 * @param [out] outputs Index 0 receives the allocated sparseIndices tensor and
 * index 1 the allocated sparseValues tensor.
 * @return ACLNN_SUCCESS, or the first failing status.
 */
aclnnStatus VllmQuantLightningIndexer(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights, const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional, const aclTensor *blockTableOptional,
    const aclTensor *metadataOptional, int64_t queryQuantMode, int64_t keyQuantMode, const char *layoutQueryOptional,
    const char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, bool returnValues, int64_t stride, int64_t scaleStride, const aclTensor *sparseIndicesRef,
    const aclTensor *sparseValuesRef, std::array<const aclTensor *, 2> &outputs, aclOpExecutor *executor);

}  // namespace l0op

#endif
