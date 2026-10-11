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
 * \file kv_compress_epilog.cpp
 * \brief KvCompressEpilog L0 API implementation
 */

#include "kv_compress_epilog.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(KvCompressEpilog);

const aclTensor *KvCompressEpilog(const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping,
                                  int64_t quantGroupSize, int64_t quantMode, int64_t roundScale, int64_t layout,
                                  int64_t blockStride, aclOpExecutor *executor)
{
    L0_DFX(KvCompressEpilog, kvCompressCache, x, slotMapping, quantGroupSize, quantMode, roundScale, layout,
           blockStride);

    // The cache is a REF parameter: the same tensor is the input and the
    // output, the kernel scatters into it in place, and no output tensor is
    // allocated. Allocating one and copying back would be the same-address
    // ViewCopy that operator-library manual 4.31 makes non-repeatable.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(KvCompressEpilog, OP_INPUT(kvCompressCache, x, slotMapping),
                                           OP_OUTPUT(kvCompressCache),
                                           OP_ATTR(quantGroupSize, quantMode, roundScale, layout, blockStride));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "KvCompressEpilog ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return kvCompressCache;
}

}  // namespace l0op
