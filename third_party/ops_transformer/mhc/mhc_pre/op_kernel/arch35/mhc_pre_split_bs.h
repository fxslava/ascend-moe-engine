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
 * \file mhc_pre_split_bs.h
 * \brief MHC Pre kernel for batch split mode
 */

#ifndef MHC_PRE_SPLIT_BS_H_
#define MHC_PRE_SPLIT_BS_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"
#include "mhc_pre_kernel_base.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

static constexpr uint64_t SYNC_V2C = 0x1;
static constexpr uint64_t SYNC_C2V = 0x2;
static constexpr uint64_t SYNC_C2V1 = 0x3;

static constexpr uint32_t DEFAULT_CHUNK_SIZE = 64;
static constexpr uint32_t DEFAULT_V1_CHUNK_D_SIZE = 5120;

template <class T, class P>
class MhcPreKernelSplitBS : public MhcPreKernelBase<T, P> {
public:
    using Base = MhcPreKernelBase<T, P>;
    using Base::mm;
    using Base::mnConfig_;
    using Base::coreNum_;
    using Base::totalLength_;
    using Base::N_;
    using Base::D_;
    using Base::outFlag_;
    using Base::hasGamma_;
    using Base::xGm_;
    using Base::phiGm_;
    using Base::alphaGm_;
    using Base::biasGm_;
    using Base::gammaGm_;
    using Base::hinGm_;
    using Base::hPostGm_;
    using Base::hResGm_;
    using Base::invRmsGm_;
    using Base::hMixGm_;
    using Base::hPreGm_;
    using Base::xFloatGm_;
    using Base::xInQueue_;
    using Base::gammaInQueue_;
    using Base::invRmsOutQueue_;
    using Base::outQueue_;
    using Base::biasInQue_;
    using Base::tmpBuff_;
    using Base::alphaBuf_;
    using Base::inputBuff_;
    using Base::hPreBuff_;
    using Base::hPostBuff_;
    using Base::hResBuff_;
    using Base::alphaInUb_;
    using Base::biasInUb_;
    using Base::matmulRes_;
    using Base::xLocal_;
    using Base::invRmsUb_;
    using Base::gammaUb_;
    using Base::preOffsetBuf_;
    using Base::postOffsetBuf_;
    using Base::resOffsetBuf_;
    using Base::pipe_;
    using Base::matrixInfo_;
    using Base::vectorOffset_;
    using Base::tiling_;
    using Base::chunkTSize_;
    using Base::v1ChunkDSize_;
    using Base::curSingleT_;
    using Base::coreIdx_;
    using Base::subBlockIdx_;
    using Base::scaleMean_;
    using Base::globalOffsetM_;
    using Base::vectorCount_;
    using Base::cubeCount_;
    using Base::mmCount_;
    using Base::vec1Count_;
    using Base::eleNumPerVf_;
    using Base::kDoubleBufferCount;
    using Base::kSingleBufferCount;
    using Base::kHalfSplitDivisor;
    using Base::ND_LENGTH;
    using Base::PARALLEL_NUM;
    using Base::V0_BASE_T;
    using Base::V1_BASE_T;

    __aicore__ inline MhcPreKernelSplitBS(MT &matmul) : Base(matmul) {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void InitHMixBuffer(InitParams initParams);
    __aicore__ inline void InitUbBuffers();
    __aicore__ inline void Process();
    __aicore__ inline void InitBlockParams(uint64_t curblock, uint32_t tBlockNum);
    __aicore__ inline void AICProcess(uint32_t offsetNd, uint32_t outOffset);
    __aicore__ inline void V0PostProcess(uint32_t curblock, uint32_t tBlockNum);
    __aicore__ inline void V0Prologue(uint32_t curNdLen, uint32_t offsetNd);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset);
};

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::Init(InitParams initParams)
{
    this->BindGlobalTensors(initParams);
    this->InitFromTilingData(initParams.tilingData);
    this->InitMNConfig();
    InitHMixBuffer(initParams);
    this->InitPipeAndCoreIdx(initParams.tPipeIn);
    InitUbBuffers();
    SyncAll<false>();

    if ASCEND_IS_AIV {
        coreIdx_ = GetBlockIdx() / kDoubleBufferCount;
        this->AIVPreLoad();
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::InitHMixBuffer(InitParams initParams)
{
    if (outFlag_) {
        hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    } else {
        hMixGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + mnConfig_.singleCoreM * mnConfig_.singleCoreK *
                                                                    PARALLEL_NUM * sizeof(P) * coreNum_));
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::InitUbBuffers()
{
    if ASCEND_IS_NOT_AIV {
        return;
    }

    static constexpr uint32_t kXInQueueBufferBytes = 80 * 1024;
    static constexpr uint32_t kOutQueueBufferBytes = 20 * 1024;
    static constexpr uint32_t kTmpBufferBytes = 40 * 1024;

    pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
    pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
    pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, (curSingleT_ / kHalfSplitDivisor) * sizeof(P));

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
    }

    pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);

    pipe_->InitBuffer(biasInQue_, kSingleBufferCount, mnConfig_.n * sizeof(P));
    pipe_->InitBuffer(alphaBuf_, mnConfig_.n * sizeof(P));
    alphaInUb_ = alphaBuf_.template Get<P>();

    preOffsetBuf_ = tmpBuff_.template GetWithOffset<uint32_t>(uint32_t(mnConfig_.n * V1_BASE_T), 0);
    postOffsetBuf_ = preOffsetBuf_[N_ * V1_BASE_T];
    resOffsetBuf_ = postOffsetBuf_[N_ * V1_BASE_T];
    uint64_t buffOffset = 0;

    buffOffset = mnConfig_.n * V1_BASE_T * sizeof(uint32_t) + (curSingleT_ / 2) * sizeof(uint32_t);
    hPreBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);

    hPostBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::Process()
{
    uint32_t tBlockNum = Ceil(totalLength_, chunkTSize_);

    for (uint64_t curblock = coreIdx_; curblock < tBlockNum; curblock += coreNum_) {
        InitBlockParams(curblock, tBlockNum);

        uint64_t outOffset = 0;
        if ASCEND_IS_AIC {
            if (outFlag_) {
                outOffset = globalOffsetM_ * mnConfig_.n;
            } else {
                outOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((mmCount_ % 2) * coreNum_ + coreIdx_);
            }
            mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
        }

        for (uint32_t offsetNd = 0; offsetNd < matrixInfo_.nD; offsetNd += ND_LENGTH) {
            if ASCEND_IS_AIV {
                uint32_t curNdLen = MhcPreUtils::Min(ND_LENGTH, matrixInfo_.nD - offsetNd);
                if (vectorCount_ >= 2) {
                    AscendC::CrossCoreWaitFlag(SYNC_C2V);
                }
                V0Prologue(curNdLen, offsetNd);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V2C);
                vectorCount_++;
            }

            if ASCEND_IS_AIC {
                AscendC::CrossCoreWaitFlag(SYNC_V2C);
                AICProcess(offsetNd, outOffset);
                CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_C2V);
            }
        }
        mmCount_++;
        CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_C2V1);

        if ASCEND_IS_AIV {
            V0PostProcess(curblock, tBlockNum);
            AIV1Process(curblock, tBlockNum);
            vec1Count_++;
        }
    }
    if ASCEND_IS_AIV {
        invRmsOutQueue_.FreeTensor(invRmsUb_);
        biasInQue_.FreeTensor(biasInUb_);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::InitBlockParams(uint64_t curblock, uint32_t tBlockNum)
{
    globalOffsetM_ = curblock * chunkTSize_;
    curSingleT_ = chunkTSize_;
    if (curblock == tBlockNum - 1) {
        mnConfig_.curSingleCoreM = totalLength_ - globalOffsetM_;
        curSingleT_ = matrixInfo_.totalLength - curblock * chunkTSize_;
    }
    if ASCEND_IS_AIV {
        this->VectorComputeOffset();
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::AICProcess(uint32_t offsetNd, uint32_t outOffset)
{
    if (offsetNd + mnConfig_.singleCoreK > mnConfig_.k) {
        mnConfig_.curSingleCoreK = mnConfig_.k - offsetNd;
    }

    uint64_t xOffset = chunkTSize_ * ND_LENGTH * (coreIdx_ + (cubeCount_ % PARALLEL_NUM) * coreNum_);

    mm.SetOrgShape(mnConfig_.curSingleCoreM, mnConfig_.curSingleCoreN, mnConfig_.curSingleCoreK, mnConfig_.k);
    mm.SetSingleShape(mnConfig_.curSingleCoreM, mnConfig_.curSingleCoreN, mnConfig_.curSingleCoreK);
    mm.SetTensorA(xFloatGm_[xOffset]);
    mm.SetTensorB(phiGm_[offsetNd], true);
    mm.IterateAll(hMixGm_[outOffset], offsetNd == 0 ? 0 : 1);
    mm.End();
    cubeCount_++;
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::V0Prologue(uint32_t curNdLen, uint32_t offsetNd)
{
    for (uint32_t offsetM = vectorOffset_.offsetMStart; offsetM < vectorOffset_.offsetMEnd; offsetM += V0_BASE_T) {
        uint32_t curMLen = V0_BASE_T;
        if (offsetM + V0_BASE_T >= vectorOffset_.offsetMEnd) {
            curMLen = vectorOffset_.offsetMEnd - offsetM;
        }
        uint64_t invRmsOffset = offsetM - vectorOffset_.offsetMStart;
        LocalTensor<P> invRmsUb = invRmsUb_[invRmsOffset];
        xLocal_ = xInQueue_.template AllocTensor<T>();
        this->DataCopyX(curMLen, curNdLen, offsetM, offsetNd);
        xLocal_ = xInQueue_.template DeQue<T>();

        LocalTensor<P> aL1Ub = outQueue_.template AllocTensor<P>();
        if (hasGamma_) {
            gammaUb_ = gammaInQueue_.template AllocTensor<P>();
            this->DataCopyGamma(curNdLen, offsetNd);
            gammaUb_ = gammaInQueue_.template DeQue<P>();

            if (offsetNd == 0) {
                this->template VFDoV0ProcessXIn<true, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                             (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                                             curMLen, curNdLen);
            } else {
                this->template VFDoV0ProcessXIn<true, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                              (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                                              curMLen, curNdLen);
            }
            gammaInQueue_.FreeTensor(gammaUb_);
        } else {
            if (offsetNd == 0) {
                this->template VFDoV0ProcessXIn<false, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                              (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            } else {
                this->template VFDoV0ProcessXIn<false, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                                               (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            }
        }
        outQueue_.template EnQue<P>(aL1Ub);
        aL1Ub = outQueue_.template DeQue<P>();
        this->DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);

        xInQueue_.FreeTensor(xLocal_);
        outQueue_.FreeTensor(aL1Ub);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::V0PostProcess(uint32_t curblock, uint32_t tBlockNum)
{
    this->VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_,
                        matrixInfo_.normEps);
    this->DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    if (vectorOffset_.singleCoreM <= 0) {
        return;
    }
    AscendC::CrossCoreWaitFlag(SYNC_C2V1);
    uint64_t lenT = 0;
    uint64_t lenD = 0;
    uint64_t singleCoreOffset = 0;
    for (int offsetT = vectorOffset_.offsetMStart; offsetT < vectorOffset_.offsetMEnd; offsetT += V1_BASE_T) {
        lenT = V1_BASE_T < vectorOffset_.offsetMEnd - offsetT ? V1_BASE_T : vectorOffset_.offsetMEnd - offsetT;
        lenD = v1ChunkDSize_;
        AIV1Prologue(offsetT, lenT, singleCoreOffset);

        this->AIV1ProcessHPost(offsetT, lenT);

        this->AIV1ProcessHPre(offsetT, lenT);
        this->AIV1ProcessHIn(offsetT, lenT, lenD);

        singleCoreOffset += V1_BASE_T;
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitBS<T, P>::AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    uint64_t HMixOffset = 0;
    LocalTensor<P> hResOutLocal = outQueue_.template AllocTensor<P>();
    if (outFlag_) {
        HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;
    } else {
        HMixOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((vec1Count_ % 2) * coreNum_ + coreIdx_) +
                     offsetT * mnConfig_.n;
    }
    this->HMixCopyIn(HMixOffset, lenT);
    matmulRes_ = xInQueue_.template DeQue<P>();

    __ubuf__ P *matmulPtr = (__ubuf__ P *)matmulRes_.GetPhyAddr();
    __ubuf__ P *invRmsPtr = (__ubuf__ P *)invRmsUb_.GetPhyAddr();
    __ubuf__ P *biasInPtr = (__ubuf__ P *)biasInUb_.GetPhyAddr();
    __ubuf__ P *alphaInPtr = (__ubuf__ P *)alphaInUb_.GetPhyAddr();
    __ubuf__ P *hPreBuffPtr = (__ubuf__ P *)hPreBuff_.GetPhyAddr();
    __ubuf__ P *hPostBuffPtr = (__ubuf__ P *)hPostBuff_.GetPhyAddr();
    __ubuf__ P *hResOutLocalPtr = (__ubuf__ P *)hResOutLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<P> matmulResReg;
        MicroAPI::RegTensor<P> invRmsReg;
        MicroAPI::RegTensor<P> invRmsBroadReg;
        MicroAPI::RegTensor<P> biasInReg;
        MicroAPI::RegTensor<P> alphaInReg;

        for (uint16_t tIdx = 0; tIdx < static_cast<uint16_t>(lenT); tIdx++) {
            MicroAPI::Load<P>(invRmsReg, invRmsPtr + singleCoreOffset + tIdx);

            uint32_t loopCntPerN = (mnConfig_.n + eleNumPerVf_ - 1) / eleNumPerVf_;
            uint32_t curLen = mnConfig_.n;
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < (uint16_t)loopCntPerN; vfBlockIdx++) {
                uint32_t maskOffset = tIdx * mnConfig_.n + vfBlockIdx * eleNumPerVf_;
                uint32_t alphaOffset = vfBlockIdx * eleNumPerVf_;
                uint32_t biasOffset = vfBlockIdx * eleNumPerVf_;
                MicroAPI::MaskReg curMask = MicroAPI::UpdateMask<P>(curLen);

                MicroAPI::LoadAlign<P>(matmulResReg, matmulPtr + maskOffset);
                MicroAPI::LoadAlign<P>(biasInReg, biasInPtr + biasOffset);
                MicroAPI::LoadAlign<P>(alphaInReg, alphaInPtr + alphaOffset);

                MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, curMask);
                MicroAPI::Mul(matmulResReg, matmulResReg, invRmsBroadReg, curMask);

                MicroAPI::Mul(matmulResReg, matmulResReg, alphaInReg, curMask);

                MicroAPI::Add(matmulResReg, matmulResReg, biasInReg, curMask);

                MicroAPI::StoreAlign<P>(matmulPtr + maskOffset, matmulResReg, curMask);
            }
        }
    }

    Gather(hPreBuff_, matmulRes_, preOffsetBuf_, uint32_t(0), lenT * N_);
    Gather(hPostBuff_, matmulRes_, postOffsetBuf_, uint32_t(0), lenT * N_);
    Gather(hResOutLocal, matmulRes_, resOffsetBuf_, uint32_t(0), lenT * N_ * N_);
    PipeBarrier<PIPE_V>();

    outQueue_.EnQue(hResOutLocal);
    hResOutLocal = outQueue_.template DeQue<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(1);
    copyParams.blockLen = uint32_t(lenT * N_ * N_ * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t(0);
    DataCopyPad(hResGm_[offset * N_ * N_], hResOutLocal, copyParams);
    outQueue_.FreeTensor(hResOutLocal);
    xInQueue_.FreeTensor(matmulRes_);
}

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_KERNEL_H_
