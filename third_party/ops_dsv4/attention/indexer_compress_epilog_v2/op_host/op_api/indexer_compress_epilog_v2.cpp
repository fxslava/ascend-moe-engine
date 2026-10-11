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
 * \file indexer_compress_epilog_v2.cpp
 * \brief IndexerCompressEpilogV2 L0 API implementation
 */

#include "indexer_compress_epilog_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(IndexerCompressEpilogV2);

const aclTensor *IndexerCompressEpilogV2(const aclTensor *indexerCompressCache, const aclTensor *x,
                                         const aclTensor *slotMapping, int64_t layout, int64_t blockStride,
                                         aclOpExecutor *executor)
{
    L0_DFX(IndexerCompressEpilogV2, indexerCompressCache, x, slotMapping, layout, blockStride);

    // REF parameter: the cache is both the input and the output and the kernel
    // scatters into it in place. Allocating an output and copying back would
    // be the same-address ViewCopy that operator-library manual 4.31 makes
    // non-repeatable.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexerCompressEpilogV2, OP_INPUT(indexerCompressCache, x, slotMapping),
                                           OP_OUTPUT(indexerCompressCache), OP_ATTR(layout, blockStride));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexerCompressEpilogV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return indexerCompressCache;
}

}  // namespace l0op
