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
 * \file mhc_sinkhorn.h
 * \brief mhc_sinkhorn
 */

#ifndef ASCENDC_MHC_SINKHORN_H
#define ASCENDC_MHC_SINKHORN_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_sinkhorn_struct.h"
#include "mhc_sinkhorn_tiling_key.h"

namespace MhcSinkhorn {
using namespace AscendC;

constexpr int64_t MASK_BUFFER_SIZE = 64;
constexpr int64_t MAX_BUFFER_SIZE = 256;
constexpr uint32_t MASK_4 = 0b00000000000000001111111111111111;
constexpr uint32_t MASK_6 = 0b00000000111111111111111111111111;
constexpr uint32_t MASK_8 = 0b11111111111111111111111111111111;
constexpr int64_t MASK_NUM = 8;
constexpr int64_t INDEX_BLOCK_LEN = 8;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
static const int64_t N_VALID_4 = 4;
static const int64_t N_VALID_6 = 6;
static const int64_t N_VALID_8 = 8;

class MhcSinkhornSimd {
public:
    __aicore__ inline MhcSinkhornSimd(TPipe &pipe, const MhcSinkhornTilingData &tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcSoftmax(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                       __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr, uint32_t dataLen,
                                       uint32_t n, float eps);
    __aicore__ inline void CalcCol(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                   __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr, uint32_t dataLen,
                                   uint32_t n, float eps);
    __aicore__ inline void CalcRow(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                   __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr, uint32_t dataLen,
                                   uint32_t n, float eps);

    GlobalTensor<float> hRes_;
    GlobalTensor<float> y_;
    TPipe &pipe_;
    const MhcSinkhornTilingData &tilingData_;
    int64_t blockIdx_;
    int64_t loop_;
    int64_t tailLoopSize_;
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> maxBuffer_;
};

__aicore__ inline void MhcSinkhornSimd::Init(GM_ADDR h_res, GM_ADDR y, GM_ADDR norm_out, GM_ADDR sum_out,
                                             GM_ADDR tiling)
{
    blockIdx_ = GetBlockIdx();
    hRes_.SetGlobalBuffer((__gm__ float *)(h_res));
    y_.SetGlobalBuffer((__gm__ float *)(y));
    pipe_.InitBuffer(maskBuffer_, MASK_BUFFER_SIZE);
    pipe_.InitBuffer(maxBuffer_, MAX_BUFFER_SIZE);
    pipe_.InitBuffer(inputQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * sizeof(float));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.tUbFactor * sizeof(float));
    loop_ = tilingData_.tNormCoreLoop;
    tailLoopSize_ = tilingData_.tUbFactorTail;      // 尾循环处理的数量
    if (blockIdx_ == tilingData_.usedCoreNum - 1) { //  尾核
        loop_ = tilingData_.tTailCoreLoop;
        tailLoopSize_ = tilingData_.tUbTailTail;
    }
}

// CalcSoftmax
__aicore__ inline void MhcSinkhornSimd::CalcSoftmax(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                                    __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                                    uint32_t dataLen, uint32_t n, float eps)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gatherReg;
        MicroAPI::RegTensor<float> maxReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> softmaxReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

        MicroAPI::Duplicate(duplicateReg1, static_cast<int32_t>(INDEX_BLOCK_LEN));
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n));
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg);
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg);
        MicroAPI::DataCopyGather(gatherReg, inputAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);

        MicroAPI::ReduceMaxWithDataBlock(maxReg, gatherReg, maskReg);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, maxReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(maxReg, maxAddr);

        MicroAPI::Sub(softmaxReg, gatherReg, maxReg, maskReg);
        MicroAPI::Exp(softmaxReg, softmaxReg, maskReg);

        MicroAPI::ReduceSumWithDataBlock(sumReg, softmaxReg, maskReg);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, sumReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, maxAddr);
        MicroAPI::Div(softmaxReg, softmaxReg, sumReg, maskReg);
        MicroAPI::Adds(softmaxReg, softmaxReg, eps, maskReg);

        MicroAPI::DataCopyScatter(outputAddr, softmaxReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornSimd::CalcCol(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                                __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                                uint32_t dataLen, uint32_t n, float eps)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gatherReg;
        MicroAPI::RegTensor<float> resReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

        MicroAPI::Duplicate(duplicateReg1, INDEX_BLOCK_LEN);
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n));
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg);
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        MicroAPI::Muls(tmpReg2, tmpReg2, static_cast<int32_t>(n), maskReg);
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg);
        MicroAPI::DataCopyGather(gatherReg, inputAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);

        MicroAPI::ReduceSumWithDataBlock(sumReg, gatherReg, maskReg);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, sumReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, maxAddr);
        MicroAPI::Adds(sumReg, sumReg, eps, maskReg);
        MicroAPI::Div(resReg, gatherReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(outputAddr, resReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornSimd::CalcRow(__local_mem__ float *inputAddr, __local_mem__ float *outputAddr,
                                                __local_mem__ uint32_t *maskAddr, __local_mem__ float *maxAddr,
                                                uint32_t dataLen, uint32_t n, float eps)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<uint32_t>();
        MicroAPI::LoadAlign(maskReg, maskAddr);

        MicroAPI::RegTensor<float> gatherReg;
        MicroAPI::RegTensor<float> resReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<int32_t> indexReg;
        MicroAPI::RegTensor<int32_t> orderReg;
        MicroAPI::RegTensor<int32_t> duplicateReg1;
        MicroAPI::RegTensor<int32_t> duplicateReg2;
        MicroAPI::RegTensor<int32_t> tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpReg2;
        MicroAPI::RegTensor<int32_t> tmpReg3;
        MicroAPI::MaskReg dataCopyMaskReg = MicroAPI::UpdateMask<uint32_t>(dataLen);

        MicroAPI::Duplicate(duplicateReg1, INDEX_BLOCK_LEN);
        MicroAPI::Duplicate(duplicateReg2, static_cast<int32_t>(n * n));
        MicroAPI::Arange(orderReg, static_cast<int32_t>(0));
        MicroAPI::Div(tmpReg1, orderReg, duplicateReg1, maskReg);
        MicroAPI::Mul(tmpReg2, tmpReg1, duplicateReg1, maskReg);
        MicroAPI::Sub(tmpReg2, orderReg, tmpReg2, maskReg);
        MicroAPI::Mul(tmpReg3, tmpReg1, duplicateReg2, maskReg);
        MicroAPI::Add(indexReg, tmpReg2, tmpReg3, maskReg);
        MicroAPI::DataCopyGather(gatherReg, inputAddr, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);

        MicroAPI::ReduceSumWithDataBlock(sumReg, gatherReg, maskReg);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(maxAddr, sumReg, dataCopyMaskReg);
        MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_E2B_B32>(sumReg, maxAddr);
        MicroAPI::Adds(sumReg, sumReg, eps, maskReg);
        MicroAPI::Div(resReg, gatherReg, sumReg, maskReg);

        MicroAPI::DataCopyScatter(outputAddr, resReg, (MicroAPI::RegTensor<uint32_t> &)(indexReg), maskReg);
    }
}

__aicore__ inline void MhcSinkhornSimd::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    LocalTensor<uint32_t> maskLocal = maskBuffer_.Get<uint32_t>();
    LocalTensor<float> maxLocal = maxBuffer_.Get<float>();

    uint32_t mask = 0;
    if (tilingData_.n == N_VALID_4) {
        mask = MASK_4;
    } else if (tilingData_.n == N_VALID_6) {
        mask = MASK_6;
    } else if (tilingData_.n == N_VALID_8) {
        mask = MASK_8;
    }
    Duplicate(maskLocal, mask, MASK_NUM);

    for (int64_t i = 0; i < loop_; i++) {
        LocalTensor<float> inputLocal = inputQue_.AllocTensor<float>();
        LocalTensor<float> outputLocal = outputQue_.AllocTensor<float>();

        int64_t inputOffset =
            (blockIdx_ * tilingData_.tNormCore * tilingData_.n * tilingData_.n + i * tilingData_.tUbFactor);
        uint32_t loopSize = (i == loop_ - 1) ? tailLoopSize_ : tilingData_.tUbFactor;
        loopSize = loopSize / (tilingData_.n * tilingData_.n) * (tilingData_.n * tilingData_.n);

        //  copyin
        DataCopyPadExtParams<float> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(loopSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(inputLocal, hRes_[inputOffset], dataCopyExtParams, dataCopyPadExtParams);

        auto MTE2ToVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(MTE2ToVEventID);
        WaitFlag<HardEvent::MTE2_V>(MTE2ToVEventID);

        uint32_t repeatSize = Ops::Base::GetVRegSize() / BLOCK_SIZE * tilingData_.n * tilingData_.n;
        uint16_t repeatTimes = Ops::Base::CeilDiv(loopSize, repeatSize);
        __local_mem__ float *inputAddr = (__local_mem__ float *)inputLocal.GetPhyAddr();
        __local_mem__ float *outputAddr = (__local_mem__ float *)outputLocal.GetPhyAddr();
        __local_mem__ uint32_t *maskAddr = (__local_mem__ uint32_t *)maskLocal.GetPhyAddr();
        __local_mem__ float *maxAddr = (__local_mem__ float *)maxLocal.GetPhyAddr();

        auto dataLen = loopSize / tilingData_.n / tilingData_.n;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.n; j++) {
                auto curInputAddr = inputAddr + i * repeatSize + j * tilingData_.n;
                auto curOutputAddr = outputAddr + i * repeatSize + j * tilingData_.n;
                CalcSoftmax(curInputAddr, curOutputAddr, maskAddr, maxAddr, static_cast<uint32_t>(dataLen),
                            static_cast<uint32_t>(tilingData_.n), tilingData_.eps);
            }
        }
        inputQue_.FreeTensor<float>(inputLocal);
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }


        for (uint16_t i = 0; i < repeatTimes; i++) {
            for (uint16_t j = 0; j < tilingData_.n; j++) {
                auto curAddrCol = outputAddr + i * repeatSize + j;
                CalcCol(curAddrCol, curAddrCol, maskAddr, maxAddr, static_cast<uint32_t>(dataLen),
                        static_cast<uint32_t>(tilingData_.n), tilingData_.eps);
            }
        }
        __VEC_SCOPE__
        {
            MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }


        for (int64_t iter = 0; iter < tilingData_.num_iters - 1; iter++) {
            for (uint16_t i = 0; i < repeatTimes; i++) {
                for (uint16_t j = 0; j < tilingData_.n; j++) {
                    auto curAddr = outputAddr + i * repeatSize + j * tilingData_.n;
                    CalcRow(curAddr, curAddr, maskAddr, maxAddr, static_cast<uint32_t>(dataLen),
                            static_cast<uint32_t>(tilingData_.n), tilingData_.eps);
                }
                __VEC_SCOPE__
                {
                    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                          AscendC::MicroAPI::MemType::VEC_LOAD>();
                }

                for (uint16_t j = 0; j < tilingData_.n; j++) {
                    auto curAddr = outputAddr + i * repeatSize + j;
                    CalcCol(curAddr, curAddr, maskAddr, maxAddr, static_cast<uint32_t>(dataLen),
                            static_cast<uint32_t>(tilingData_.n), tilingData_.eps);
                }
            }
            __VEC_SCOPE__
            {
                MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
            }
        }

        auto VToMTE3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(VToMTE3EventID);
        WaitFlag<HardEvent::V_MTE3>(VToMTE3EventID);

        DataCopyPad(y_[inputOffset], outputLocal, dataCopyExtParams);
        outputQue_.FreeTensor<float>(outputLocal);
    }
}
} // namespace MhcSinkhorn

#endif