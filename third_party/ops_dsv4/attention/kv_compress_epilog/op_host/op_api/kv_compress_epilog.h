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
 * \file kv_compress_epilog.h
 * \brief KvCompressEpilog L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_KV_COMPRESS_EPILOG_OP_H
#define OP_API_INC_LEVEL0_OP_KV_COMPRESS_EPILOG_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief KvCompressEpilog L0 kernel interface.
 *
 * Quantizes the compressed-KV block `x` and scatters it into the paged FP8
 * cache at the slots `slotMapping` names. The cache is a REF parameter: the
 * kernel writes it in place, so the tensor is both OP_INPUT and OP_OUTPUT and
 * the L2 layer must not stage it through Contiguous or ViewCopy.
 *
 * @param [in] kvCompressCache Paged FP8 E5M2/E4M3 cache; written in place.
 * @param [in] x Compressed KV block to quantize and store, BF16.
 * @param [in] slotMapping Destination slot per row, INT32/INT64.
 * @param [in] quantGroupSize Elements per quant group (128 on 950PR).
 * @param [in] quantMode Quantization mode.
 * @param [in] roundScale 1 to round the scale to a power of two.
 * @param [in] layout Cache layout selector.
 * @param [in] blockStride Element stride of the cache's block axis.
 * @param [in] executor Op executor.
 * @return The cache tensor on success, nullptr on launch failure.
 */
const aclTensor *KvCompressEpilog(const aclTensor *kvCompressCache, const aclTensor *x, const aclTensor *slotMapping,
                                  int64_t quantGroupSize, int64_t quantMode, int64_t roundScale, int64_t layout,
                                  int64_t blockStride, aclOpExecutor *executor);

}  // namespace l0op

#endif
