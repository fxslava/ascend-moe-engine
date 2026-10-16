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
 * \file mhc_pre_utils.h
 * \brief
 */
#ifndef ASCENDC_MHC_PRE_UTILS_H
#define ASCENDC_MHC_PRE_UTILS_H

namespace MhcPreUtils {

constexpr uint32_t UB_ALIGN_SIZE = 32;

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6; 
constexpr uint16_t VEC0_FLAG_ID_OFFSET = 0;
constexpr uint16_t VEC1_FLAG_ID_OFFSET = FLAG_ID_MAX;

template <typename T>
__aicore__ inline T Max(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b) {
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b) {
    return CeilDiv(a, b) * b;
}
/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
}

#endif  // ASCENDC_MHC_PRE_UTILS_H
