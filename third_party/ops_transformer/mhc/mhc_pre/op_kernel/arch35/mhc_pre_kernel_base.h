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
 * \file mhc_pre_kernel_base.h
 * \brief Base class for MHC pre kernels
 */

#ifndef MHC_PRE_KERNEL_BASE_H_
#define MHC_PRE_KERNEL_BASE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

struct InitParams {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR bias;
    GM_ADDR gamma;
    GM_ADDR hin;
    GM_ADDR h_post;
    GM_ADDR h_res;
    GM_ADDR inv_rms;
    GM_ADDR h_mix;
    GM_ADDR h_pre;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    MhcPreTilingData *tilingData;
};

#ifndef MHC_PRE_COMMON_DEFINED
#define MHC_PRE_COMMON_DEFINED
constexpr MicroAPI::CastTrait ctFp32To16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait ctHalf2Fp32Zero = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::DivSpecificMode divMode = {MicroAPI::MaskMergeMode::ZEROING, true};

using aT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT = matmul::MatmulImpl<aT, bT, cT>;
#endif  // MHC_PRE_COMMON_DEFINED

struct MatrixInfo {
    uint32_t totalLength = 0;
    uint32_t nD = 0;
    uint32_t fusionSize = 0;
    float normEps = 0.0f;
    float hcEps = 0.0f;
};

struct VectorOffsetParams {
    uint64_t globalOffsetM = 0;
    uint64_t singleCoreM = 0;
    uint64_t offsetMStart = 0;
    uint64_t offsetMEnd = 0;
};

struct MNConfig {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t singleCoreK;
    uint64_t curSingleCoreM;
    uint64_t curSingleCoreN;
    uint64_t curSingleCoreK;
};

#define PROCESS_HIN_IDX(nIdx, lenD, dIdx, eleNumPerVf_) \
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xInAddr + nIdx * lenD + dIdx * eleNumPerVf_); \
    MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, mask); \
    MicroAPI::Muls(xFp32Reg, xFp32Reg, hPreValue##nIdx, mask); \
    MicroAPI::Add(accFp32Reg, accFp32Reg, xFp32Reg, mask)

template <class T, class P>
class MhcPreKernelBase {
public:
    static constexpr uint32_t kDoubleBufferCount = 2;
    static constexpr uint32_t kSingleBufferCount = 1;
    static constexpr uint32_t kAlphaPreIndex = 0;
    static constexpr uint32_t kAlphaPostIndex = 1;
    static constexpr uint32_t kAlphaCombIndex = 2;
    static constexpr uint32_t kAlphaCombBaseOffset = 2;
    static constexpr uint32_t kSupportedN4 = 4;
    static constexpr uint32_t kSupportedN6 = 6;
    static constexpr uint32_t kSupportedN8 = 8;
    static constexpr uint32_t kBlockLenSingle = 1;
    static constexpr uint32_t kAlignmentBytes = 32;
    static constexpr uint32_t kHalfSplitDivisor = 2;
    static constexpr float kOneValue = 1.0f;
    static constexpr float kTwoValue = 2.0f;

    __aicore__ inline MhcPreKernelBase(MT &matmul) : mm(matmul) {}

    __aicore__ inline void InitPipeAndCoreIdx(TPipe *pipe)
    {
        pipe_ = pipe;
        coreIdx_ = GetBlockIdx();
        subBlockIdx_ = GetSubBlockIdx();
    }

    __aicore__ inline void BindGlobalTensors(InitParams initParams)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
        phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
        hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
        hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
        hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));
        xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));
    }

    __aicore__ inline void InitFromTilingData(const MhcPreTilingData *tilingData)
    {
        tiling_ = tilingData;
        matrixInfo_.totalLength = tiling_->totalLength;
        matrixInfo_.nD = tiling_->nD;
        N_ = tiling_->N;
        D_ = tiling_->D;
        matrixInfo_.fusionSize = tiling_->fusionSize;
        matrixInfo_.normEps = tiling_->normEps;
        matrixInfo_.hcEps = tiling_->hcEps;
        coreNum_ = tiling_->coreNum;
        totalLength_ = tiling_->totalLength;
        outFlag_ = (tiling_->outFlag != 0);
        scaleMean_ = tiling_->scaleMean;
        chunkTSize_ = tiling_->chunkTSize;
        v1ChunkDSize_ = tiling_->v1ChunkDSize;
        hasGamma_ = (tiling_->hasGamma != 0);
        eleNumPerVf_ = MhcPreUtils::GetVRegSize() / sizeof(P);
    }

    __aicore__ inline void InitMNConfig()
    {
        mnConfig_.m = matrixInfo_.totalLength;
        mnConfig_.n = matrixInfo_.fusionSize;
        mnConfig_.k = matrixInfo_.nD;
        mnConfig_.singleCoreM = chunkTSize_;
        mnConfig_.singleCoreN = mnConfig_.n;
        mnConfig_.singleCoreK = ND_LENGTH;
        mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
        mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
        mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
        curSingleT_ = chunkTSize_;
    }

    __aicore__ inline void AIV1GetHSliceOffset()
    {
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        uint32_t offset3 = 0;
        uint32_t curOffset = 0;
        uint32_t nSquare = N_ * N_;
        for (uint32_t i = 0; i < V1_BASE_T; i++) {
            for (uint32_t j = 0; j < N_; j++) {
                preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
                curOffset++;
            }
            for (uint32_t j = 0; j < N_; j++) {
                postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
                curOffset++;
            }
            for (uint32_t j = 0; j < nSquare; j++) {
                resOffsetBuf_.SetValue(offset3++, curOffset * sizeof(P));
                curOffset++;
            }
        }
    }

    __aicore__ inline void VectorComputeOffset()
    {
        uint64_t alignSingleM = Ceil(curSingleT_ / 2, 8) * 8;
        vectorOffset_.singleCoreM = alignSingleM < curSingleT_ ? alignSingleM : curSingleT_;
        if (subBlockIdx_ == 0) {
            vectorOffset_.offsetMStart = 0;
            vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
        } else {
            vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
            vectorOffset_.singleCoreM = curSingleT_ - vectorOffset_.singleCoreM;
            vectorOffset_.offsetMEnd = curSingleT_;
        }
    }

    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXIn(__ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn,
                                            __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize)
    {
        uint32_t nSrcUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(T)));
        uint32_t nDstUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
            for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
                uint32_t elementNum = nSize;
                MicroAPI::RegTensor<P> sumReg;
                if constexpr (isFirstND) {
                    MicroAPI::Duplicate(sumReg, 0);
                } else {
                    MicroAPI::Load(sumReg, invRmsDst + mIdx);
                }
                for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                    MicroAPI::RegTensor<T> xInReg;
                    MicroAPI::RegTensor<P> gammaReg;
                    MicroAPI::RegTensor<P> xFp32Reg, xMulReg, xSquaReg;
                    MicroAPI::RegTensor<P> tmpSumReg;

                    uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf_;
                    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                    MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                    MicroAPI::Cast<float, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, maskN4B32);
                    if constexpr (hasGamma) {
                        MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf_);
                        MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, maskN4B32);
                    } else {
                        xMulReg = xFp32Reg;
                    }
                    uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf_;
                    MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, maskN4B32);

                    MicroAPI::Mul(xSquaReg, xFp32Reg, xFp32Reg, maskN4B32);
                    MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(tmpSumReg, xSquaReg, maskN4B32);
                    MicroAPI::Add(sumReg, sumReg, tmpSumReg, maskN4B32);
                }

                MicroAPI::Store(invRmsDst + mIdx, sumReg, 1);
            }
        }
    }

    __aicore__ inline void VFDoV0ProcessInvRms(__ubuf__ P *invRms, uint16_t nSize, float scaleMean, float normEps)
    {
        uint32_t nUbAligned = MhcPreUtils::Align(nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
        uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf_);
        __VEC_SCOPE__
        {
            uint32_t elementNum = nSize;
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<P>();
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg maskN4B32 = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::RegTensor<P> invrmsReg, onesReg;

                MicroAPI::LoadAlign(invrmsReg, invRms + vfBlockIdx * eleNumPerVf_);
                MicroAPI::Muls(invrmsReg, invrmsReg, scaleMean, maskN4B32);
                MicroAPI::Adds(invrmsReg, invrmsReg, normEps, maskN4B32);
                MicroAPI::Sqrt(invrmsReg, invrmsReg, maskN4B32);
                MicroAPI::Duplicate(onesReg, 1);
                MicroAPI::Div(invrmsReg, onesReg, invrmsReg, maskN4B32);
                MicroAPI::StoreAlign(invRms + vfBlockIdx * eleNumPerVf_, invrmsReg, maskN4B32);
            }
        }
    }

    __aicore__ inline void DataCopyX(uint32_t curMLen, uint32_t curNdLen, uint32_t offsetM, uint32_t offsetNd)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curMLen);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(T));
        copyParams.srcStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(T));
        copyParams.dstStride = uint32_t(0);

        uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
        DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

        uint64_t offset = globalOffsetM_ * matrixInfo_.nD + offsetM * matrixInfo_.nD + offsetNd;
        DataCopyPad(xLocal_, xGm_[offset], copyParams, padParams);
        xInQueue_.EnQue<T>(xLocal_);
    }

    __aicore__ inline void DataCopyGamma(uint32_t curNdLen, uint32_t offsetNd)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint32_t rightPadNum = Ceil(curNdLen, 16) * 16 - curNdLen;
        DataCopyPadExtParams<P> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};

        uint64_t offset = offsetNd;

        DataCopyPad(gammaUb_, gammaGm_[offset], copyParams, padParams);
        gammaInQueue_.EnQue<P>(gammaUb_);
    }

    __aicore__ inline void HMixCopyIn(uint64_t offset, uint64_t lenT)
    {
        LocalTensor<P> hMixLocal = xInQueue_.AllocTensor<P>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * mnConfig_.n * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

        DataCopyPad(hMixLocal, hMixGm_[offset], copyParams, copyPadParams);

        xInQueue_.EnQue(hMixLocal);
    }

    __aicore__ inline void DataCopyOutInvRmsUb(uint32_t curMLen, uint32_t offsetM)
    {
        invRmsOutQueue_.EnQue<P>(invRmsUb_);
        invRmsUb_ = invRmsOutQueue_.DeQue<P>();

        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(curMLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint64_t offset = globalOffsetM_ + offsetM;

        DataCopyPad(invRmsGm_[offset], invRmsUb_, copyParams);
    }

    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen,
                                                   uint32_t curNdLen, uint32_t offsetM,
                                                   uint32_t offsetNd)
    {
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(curMLen);
        copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);

        uint64_t offset =
            chunkTSize_ * ND_LENGTH * (coreIdx_ + (vectorCount_ % PARALLEL_NUM) * coreNum_) + offsetM * curNdLen;
        DataCopyPad(xFloatGm_[offset], x, copyParams);
    }

    __aicore__ inline void DataCopyOutHPre(uint64_t offset, uint32_t totalElem)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(totalElem * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopyPad(hPreGm_[offset * N_], hPreBuff_, copyParams);
    }

    __aicore__ inline void AIVPreLoad()
    {
        invRmsUb_ = invRmsOutQueue_.AllocTensor<P>();
        AIV1GetHSliceOffset();

        float alphaPre = alphaGm_.GetValue(kAlphaPreIndex);
        float alphaPost = alphaGm_.GetValue(kAlphaPostIndex);
        float alphaComb = alphaGm_.GetValue(kAlphaCombIndex);
        for (uint64_t i = 0; i < N_; ++i) {
            alphaInUb_.SetValue(i, alphaPre);
            alphaInUb_.SetValue(i + N_, alphaPost);
            for (uint64_t j = 0; j < N_; ++j) {
                alphaInUb_.SetValue((kAlphaCombBaseOffset + i) * N_ + j, alphaComb);
            }
        }
        BiasCopyIn();
        biasInUb_ = biasInQue_.DeQue<P>();
    }

    __aicore__ inline void BiasCopyIn()
    {
        LocalTensor<P> biasLocal = biasInQue_.AllocTensor<P>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(matrixInfo_.fusionSize * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};

        DataCopyPad(biasLocal, biasGm_, copyParams, copyPadParams);
        biasInQue_.EnQue(biasLocal);
    }

    __aicore__ inline void AIV1ProcessHPre(uint64_t offsetT, uint64_t lenT)
    {
        __ubuf__ P *hPreBuffAddr = (__ubuf__ P *)hPreBuff_.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint16_t nLoopCnt = Ceil(totalElem, eleNumPerVf_);
        uint32_t curElemCnt = totalElem;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> hPreReg;
            MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
                MicroAPI::LoadAlign(hPreReg, hPreBuffAddr + vfBlockIdx * eleNumPerVf_);
                MicroAPI::Neg(negReg, hPreReg, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addOneReg, expReg, static_cast<P>(kOneValue), mask);
                MicroAPI::Duplicate(oneReg, static_cast<P>(kOneValue), mask);
                MicroAPI::Div<P, &divMode>(sigmoidReg, oneReg, addOneReg, mask);

                MicroAPI::Adds(resultReg, sigmoidReg, matrixInfo_.hcEps, mask);
                MicroAPI::StoreAlign(hPreBuffAddr + vfBlockIdx * eleNumPerVf_, resultReg, mask);
            }
        }

        if (outFlag_) {
            uint64_t offset = globalOffsetM_ + offsetT;
            DataCopyOutHPre(offset, totalElem);
        }
    }

    __aicore__ inline void AIV1ProcessHPost(uint64_t offsetT, uint64_t lenT)
    {
        uint64_t offset = globalOffsetM_ + offsetT;
        LocalTensor<P> hPostOutLocal = outQueue_.AllocTensor<P>();
        __ubuf__ P *hPostBuffAddr = (__ubuf__ P *)hPostBuff_.GetPhyAddr();
        __ubuf__ P *hPostOutAddr = (__ubuf__ P *)hPostOutLocal.GetPhyAddr();
        uint32_t totalElem = lenT * N_;
        uint32_t regCapacityFp32 = 64;
        uint16_t nLoopCnt = Ceil(totalElem, regCapacityFp32);
        float scalarValue = kTwoValue;
        uint32_t curElemCnt = totalElem;

        __VEC_SCOPE__
        {
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
                uint32_t elemOffset = vfBlockIdx * regCapacityFp32;
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curElemCnt);
                MicroAPI::RegTensor<P> hPostReg;
                MicroAPI::RegTensor<P> negReg, expReg, addOneReg, sigmoidReg, resultReg, oneReg;

                MicroAPI::LoadAlign(hPostReg, hPostBuffAddr + elemOffset);
                MicroAPI::Neg(negReg, hPostReg, mask);
                MicroAPI::Exp(expReg, negReg, mask);
                MicroAPI::Adds(addOneReg, expReg, kOneValue, mask);
                MicroAPI::Duplicate(oneReg, kOneValue, mask);
                MicroAPI::Div<P, &divMode>(sigmoidReg, oneReg, addOneReg, mask);

                MicroAPI::Muls(resultReg, sigmoidReg, scalarValue, mask);
                MicroAPI::StoreAlign(hPostOutAddr + elemOffset, resultReg, mask);
            }
        }
        outQueue_.EnQue(hPostOutLocal);
        hPostOutLocal = outQueue_.DeQue<P>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * N_ * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPad(hPostGm_[offset * N_], hPostOutLocal, copyParams);
        outQueue_.FreeTensor(hPostOutLocal);
    }

    __aicore__ inline void AIV1ProcessHIn(uint64_t offsetT, uint64_t lenT, uint64_t lenD)
    {
        auto hInDealBuf = inputBuff_;
        for (uint32_t tIdx = 0; tIdx < lenT; tIdx++) {
            for (int offsetD = 0; offsetD < D_; offsetD += v1ChunkDSize_) {
                lenD = v1ChunkDSize_ < D_ - offsetD ? v1ChunkDSize_ : D_ - offsetD;
                LocalTensor<T> xIn = xInQueue_.AllocTensor<T>();
                DataCopyExtParams xInCopyParams;
                xInCopyParams.blockCount = static_cast<uint16_t>(N_);
                xInCopyParams.blockLen = uint32_t(lenD * sizeof(T));
                xInCopyParams.srcStride = uint32_t((D_ - lenD) * sizeof(T));
                xInCopyParams.dstStride = 0;

                uint32_t rightPadNum = Ceil(lenD, 16) * 16 - lenD;
                DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadNum), 0};
                uint64_t xGmOffset = (globalOffsetM_ + offsetT + tIdx) * N_ * D_ + offsetD;
                DataCopyPad(xIn, xGm_[xGmOffset], xInCopyParams, padParams);

                xInQueue_.EnQue(xIn);
                xIn = xInQueue_.DeQue<T>();
                LocalTensor<T> hinOut = outQueue_.AllocTensor<T>();
                __ubuf__ T *xInAddr = (__ubuf__ T *)xIn.GetPhyAddr();
                __ubuf__ T *hinOutAddr = (__ubuf__ T *)hinOut.GetPhyAddr();
                switch (N_) {
                    case kSupportedN4:
                        VFDoV1ProcessHinForN4(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                    case kSupportedN6:
                        VFDoV1ProcessHinForN6(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                    case kSupportedN8:
                        VFDoV1ProcessHinForN8(xInAddr, hinOutAddr, lenD, tIdx);
                        break;
                    default:
                        break;
                }

                outQueue_.EnQue(hinOut);
                hinOut = outQueue_.DeQue<T>();
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = uint32_t(lenD * sizeof(T));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;

                DataCopyPad(hinGm_[(globalOffsetM_ + offsetT + tIdx) * D_ + offsetD], hinOut, copyParams);

                xInQueue_.FreeTensor(xIn);
                outQueue_.FreeTensor(hinOut);
            }
        }
    }

    __aicore__ inline void VFDoV1ProcessHinForN4(__ubuf__ T *xInAddr, __ubuf__ T *hinOutAddr,
                                                 uint32_t lenD, uint32_t tIdx)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> xFp32Reg;
            MicroAPI::RegTensor<T> xInReg;
            MicroAPI::RegTensor<P> accFp32Reg;
            MicroAPI::RegTensor<T> outB16Reg;
            uint32_t curLenD = lenD;

            P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
            P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
            P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
            P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
            for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
                MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

                PROCESS_HIN_IDX(0, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(1, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(2, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(3, lenD, dIdx, eleNumPerVf_);

                MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf_, outB16Reg,
                                                                            mask);
            }
        }
    }

    __aicore__ inline void VFDoV1ProcessHinForN6(__ubuf__ T *xInAddr, __ubuf__ T *hinOutAddr,
                                                 uint32_t lenD, uint32_t tIdx)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> xFp32Reg;
            MicroAPI::RegTensor<T> xInReg;
            MicroAPI::RegTensor<P> accFp32Reg;
            MicroAPI::RegTensor<T> outB16Reg;
            uint32_t curLenD = lenD;

            P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
            P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
            P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
            P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
            P hPreValue4 = hPreBuff_.GetValue(tIdx * N_ + 4);
            P hPreValue5 = hPreBuff_.GetValue(tIdx * N_ + 5);
            for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
                MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

                PROCESS_HIN_IDX(0, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(1, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(2, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(3, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(4, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(5, lenD, dIdx, eleNumPerVf_);

                MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf_, outB16Reg,
                                                                            mask);
            }
        }
    }

    __aicore__ inline void VFDoV1ProcessHinForN8(__ubuf__ T *xInAddr, __ubuf__ T *hinOutAddr,
                                                 uint32_t lenD, uint32_t tIdx)
    {
        uint16_t dLoopCnt = (lenD + eleNumPerVf_ - 1) / eleNumPerVf_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<P> xFp32Reg;
            MicroAPI::RegTensor<T> xInReg;
            MicroAPI::RegTensor<P> accFp32Reg;
            MicroAPI::RegTensor<T> outB16Reg;
            uint32_t curLenD = lenD;

            P hPreValue0 = hPreBuff_.GetValue(tIdx * N_ + 0);
            P hPreValue1 = hPreBuff_.GetValue(tIdx * N_ + 1);
            P hPreValue2 = hPreBuff_.GetValue(tIdx * N_ + 2);
            P hPreValue3 = hPreBuff_.GetValue(tIdx * N_ + 3);
            P hPreValue4 = hPreBuff_.GetValue(tIdx * N_ + 4);
            P hPreValue5 = hPreBuff_.GetValue(tIdx * N_ + 5);
            P hPreValue6 = hPreBuff_.GetValue(tIdx * N_ + 6);
            P hPreValue7 = hPreBuff_.GetValue(tIdx * N_ + 7);
            for (uint16_t dIdx = 0; dIdx < dLoopCnt; dIdx++) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<P>(curLenD);
                MicroAPI::Duplicate<P>(accFp32Reg, static_cast<P>(0.0f), mask);

                PROCESS_HIN_IDX(0, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(1, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(2, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(3, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(4, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(5, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(6, lenD, dIdx, eleNumPerVf_);
                PROCESS_HIN_IDX(7, lenD, dIdx, eleNumPerVf_);

                MicroAPI::Cast<T, P, ctFp32To16>(outB16Reg, accFp32Reg, mask);
                MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(hinOutAddr + dIdx * eleNumPerVf_, outB16Reg, mask);
            }
        }
    }

protected:
    MT &mm;
    MNConfig mnConfig_;
    uint32_t coreNum_;
    uint64_t totalLength_;
    uint64_t N_;
    uint64_t D_;
    bool outFlag_;
    bool hasGamma_;

    GlobalTensor<T> xGm_;
    GlobalTensor<P> phiGm_;
    GlobalTensor<P> alphaGm_;
    GlobalTensor<P> biasGm_;
    GlobalTensor<P> gammaGm_;
    GlobalTensor<T> hinGm_;
    GlobalTensor<P> hPostGm_;
    GlobalTensor<P> hResGm_;
    GlobalTensor<P> invRmsGm_;
    GlobalTensor<P> hMixGm_;
    GlobalTensor<P> hPreGm_;
    GlobalTensor<P> xFloatGm_;

    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> gammaInQueue_;
    TQue<QuePosition::VECOUT, 1> invRmsOutQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECIN, 1> biasInQue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::VECCALC> alphaBuf_;

    LocalTensor<P> inputBuff_;
    LocalTensor<P> hPreBuff_;
    LocalTensor<P> hPostBuff_;
    LocalTensor<P> hResBuff_;

    LocalTensor<P> alphaInUb_;
    LocalTensor<P> biasInUb_;
    LocalTensor<P> matmulRes_;

    LocalTensor<T> xLocal_;
    LocalTensor<P> invRmsUb_;
    LocalTensor<P> gammaUb_;

    LocalTensor<uint32_t> preOffsetBuf_;
    LocalTensor<uint32_t> postOffsetBuf_;
    LocalTensor<uint32_t> resOffsetBuf_;

    TPipe *pipe_;
    MatrixInfo matrixInfo_;
    VectorOffsetParams vectorOffset_;

    const MhcPreTilingData *tiling_;

    static constexpr uint32_t PARALLEL_NUM = 2;
    static constexpr uint32_t ND_LENGTH = 256;

    uint32_t chunkTSize_ = 64;
    uint32_t v1ChunkDSize_ = 5120;
    uint32_t curSingleT_ = 64;
    uint32_t coreIdx_ = 0;
    uint32_t subBlockIdx_ = 0;
    float scaleMean_ = 0.0f;
    uint64_t globalOffsetM_ = 0;
    uint32_t vectorCount_ = 0;
    uint32_t cubeCount_ = 0;
    uint64_t mmCount_ = 0;
    uint64_t vec1Count_ = 0;
    uint16_t eleNumPerVf_ = 0;

    static constexpr uint32_t V0_BASE_T = 16;
    static constexpr uint64_t V1_BASE_T = 16;
    static constexpr uint64_t V1_BASE_D = 32;
};

} // namespace MhcPre

#endif // MHC_PRE_KERNEL_BASE_H_
