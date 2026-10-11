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
 * \file indexer_compress_epilog_v2.h
 * \brief IndexerCompressEpilogV2 L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_INDEXER_COMPRESS_EPILOG_V2_OP_H
#define OP_API_INC_LEVEL0_OP_INDEXER_COMPRESS_EPILOG_V2_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief IndexerCompressEpilogV2 L0 kernel interface.
 *
 * Scatters the indexer-side compressed rows x into the paged UINT8 indexer
 * cache at the slots slotMapping names. The cache is a REF parameter: the
 * kernel writes it in place, so the tensor is both OP_INPUT and OP_OUTPUT and
 * the L2 layer must not stage it through Contiguous or ViewCopy.
 *
 * @param [in] indexerCompressCache Paged UINT8 indexer cache; written in place.
 * @param [in] x Rows to store, FP16 or BF16.
 * @param [in] slotMapping Destination slot per row, INT32.
 * @param [in] layout Cache layout selector.
 * @param [in] blockStride Element stride of the cache block axis.
 * @param [in] executor Op executor.
 * @return The cache tensor on success, nullptr on launch failure.
 */
const aclTensor *IndexerCompressEpilogV2(const aclTensor *indexerCompressCache, const aclTensor *x,
                                         const aclTensor *slotMapping, int64_t layout, int64_t blockStride,
                                         aclOpExecutor *executor);

}  // namespace l0op

#endif
