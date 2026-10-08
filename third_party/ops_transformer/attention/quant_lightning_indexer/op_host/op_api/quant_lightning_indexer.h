/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_QUANT_LIGHTNING_INDEXER_OP_H_
#define OP_API_INC_LEVEL0_OP_QUANT_LIGHTNING_INDEXER_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
    const aclTensor *QuantLightningIndexer(
            const aclTensor *query,
            const aclTensor *key,
            const aclTensor *weights,
            const aclTensor *queryDequantScale,
            const aclTensor *keyDequantScale,
            const aclTensor *actualSeqLengthsQueryOptional,
            const aclTensor *actualSeqLengthsKeyOptional,
            const aclTensor *blockTableOptional,
            int64_t queryQuantMode,
            int64_t keyQuantMode,
            const char *layoutQueryOptional,
            const char *layoutKeyOptional,
            int64_t sparseCount,
            int64_t sparseMode,
            int64_t preTokens,
            int64_t nextTokens,
            aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_OP_QUANT_LIGHTNING_INDEXER_OP_H_
