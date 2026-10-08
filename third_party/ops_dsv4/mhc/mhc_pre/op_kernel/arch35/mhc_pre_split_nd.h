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
 * \file mhc_pre_split_nd.h
 * \brief MHC Pre kernel for ND split mode
 */

#ifndef MHC_PRE_SPLIT_ND_H_
#define MHC_PRE_SPLIT_ND_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"
#include "mhc_pre_kernel_base.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

struct InitParamsDecode {
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

static constexpr uint64_t SYNC_V0toV0 = 0x1;
static constexpr uint64_t SYNC_V0toC = 0x2;
static constexpr uint64_t SYNC_CtoC = 0x3;
static constexpr uint64_t SYNC_CtoV1 = 0x4;

template <class T, class P>
class MhcPreKernelSplitND : public MhcPreKernelBase<T, P> {
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
    using Base::coreIdx_;
    using Base::subBlockIdx_;
    using Base::scaleMean_;
    using Base::globalOffsetM_;
    using Base::vectorCount_;
    using Base::cubeCount_;
    using Base::mmCount_;
    using Base::vec1Count_;
    using Base::eleNumPerVf_;
    using Base::kAlphaCombBaseOffset;
    using Base::kAlphaCombIndex;
    using Base::kAlphaPostIndex;
    using Base::kAlphaPreIndex;
    using Base::kDoubleBufferCount;
    using Base::kSingleBufferCount;
    using Base::kSupportedN4;
    using Base::kSupportedN6;

    __aicore__ inline MhcPreKernelSplitND(MT &matmul) : Base(matmul) {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess();
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void VectorComputeOffset();
    __aicore__ inline void AIVPreLoad();
    __aicore__ inline void AIV1GetHSliceOffset();
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen,
                                                  uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void V0Prologue(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset);
    __aicore__ inline void HMixProcess(uint64_t offsetT, uint64_t lenT);

    GlobalTensor<P> mmResGm_;
    GlobalTensor<P> tempMMResGm_;

    static constexpr uint32_t ND_LENGTH = 2048;
    static constexpr uint64_t V1_BASE_T = 8;

private:
    static constexpr uint32_t kXInQueueBufferBytes = 80 * 1024;
    static constexpr uint32_t kOutQueueBufferBytes = 32 * 1024;
    static constexpr uint32_t kTmpBufferBytes = 20 * 1024;
    static constexpr uint32_t kMinTForN4 = 2;
    static constexpr uint32_t kMinTForN6 = 4;
    static constexpr uint32_t kDefaultCurSingleM = 2;
    static constexpr uint32_t kDefaultMinT = 1;
    static constexpr uint32_t kDefaultvectorCoreNum_ = 2;
    static constexpr uint32_t kDefaultV0BaseT = 1;

    uint32_t chunNDSize_ = 320;
    uint32_t curSingleM_ = kDefaultCurSingleM;
    uint32_t minT_ = kDefaultMinT;
    uint32_t vectorCoreNum_ = kDefaultvectorCoreNum_;
    uint32_t V0_BASE_T = kDefaultV0BaseT;
};

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::Init(InitParams initParams)
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

    this->InitFromTilingData(initParams.tilingData);

    if (N_ == kSupportedN4) {
        minT_ = kMinTForN4;
    }
    if (N_ == kSupportedN6) {
        minT_ = kMinTForN6;
    }

    chunNDSize_ = Ceil(matrixInfo_.nD, coreNum_);
    chunkTSize_ = Ceil(totalLength_, coreNum_);

    mnConfig_.m = matrixInfo_.totalLength;
    mnConfig_.n = matrixInfo_.fusionSize;
    mnConfig_.k = matrixInfo_.nD;
    mnConfig_.singleCoreM = totalLength_;
    mnConfig_.singleCoreN = mnConfig_.n;
    mnConfig_.singleCoreK = chunNDSize_;
    mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
    mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
    mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
    curSingleM_ = chunkTSize_;

    constexpr uint64_t kWorkspaceAlignBytes = 32UL;
    uint64_t xFloatWorkspaceBytes = totalLength_ * matrixInfo_.nD * sizeof(P);
    xFloatWorkspaceBytes = (xFloatWorkspaceBytes + kWorkspaceAlignBytes - 1UL) /
                        kWorkspaceAlignBytes * kWorkspaceAlignBytes;
    if (outFlag_) {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    } else {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));
    }
    tempMMResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));

    pipe_ = initParams.tPipeIn;
    coreIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();

    if ASCEND_IS_AIV {
        InitLocalBuffers();
    }

    SyncAll<false>();
}


template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::InitLocalBuffers()
{
    pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
    pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
    pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, Ceil(curSingleM_, kDoubleBufferCount) * sizeof(P));

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
    }
    
    pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);

    pipe_->InitBuffer(biasInQue_, kSingleBufferCount, mnConfig_.n * sizeof(P));
    pipe_->InitBuffer(alphaBuf_, mnConfig_.n * sizeof(P));
    alphaInUb_ = alphaBuf_.template Get<P>();

    uint64_t buffOffset = 0;
    preOffsetBuf_ = tmpBuff_.template GetWithOffset<uint32_t>(uint32_t(N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    postOffsetBuf_ = tmpBuff_.template GetWithOffset<uint32_t>(uint32_t(N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    resOffsetBuf_ = tmpBuff_.template GetWithOffset<uint32_t>(uint32_t(N_ * N_ * V1_BASE_T), buffOffset);
    buffOffset += N_ * N_ * V1_BASE_T * sizeof(uint32_t);
    buffOffset = Ceil(buffOffset, 32) * 32;

    hPreBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;

    hPostBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;

    hResBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_ * N_), buffOffset);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::Process()
{
    if ASCEND_IS_AIV {
        coreIdx_ = GetBlockIdx() / kDoubleBufferCount;
        this->AIVPreLoad();

        uint32_t tBlockNum = Ceil(totalLength_, chunkTSize_);
        if (coreIdx_ < tBlockNum) {
            globalOffsetM_ = coreIdx_ * chunkTSize_;
            V0Prologue(coreIdx_, tBlockNum);
            AIV1Process(coreIdx_, tBlockNum);
        } else {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
            AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        }
    }

    if ASCEND_IS_AIC {
        uint32_t ndBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);
        if (coreIdx_ < ndBlockNum) {
            AICProcess();
        }
    }

    if ASCEND_IS_AIV {
        invRmsOutQueue_.FreeTensor(invRmsUb_);
        biasInQue_.FreeTensor(biasInUb_);
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::AIVPreLoad()
{
    invRmsUb_ = invRmsOutQueue_.template AllocTensor<P>();
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
    this->BiasCopyIn();
    biasInUb_ = biasInQue_.template DeQue<P>();
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::AIV1GetHSliceOffset()
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

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::AICProcess()
{
    AscendC::CrossCoreWaitFlag(SYNC_V0toC);
    
    uint64_t offset = chunNDSize_ * coreIdx_;
    uint64_t outOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * coreIdx_;
    if (offset + mnConfig_.singleCoreK > mnConfig_.k) {
        mnConfig_.curSingleCoreK = mnConfig_.k - offset;
    }

    mm.SetOrgShape(mnConfig_.singleCoreM, mnConfig_.singleCoreN, mnConfig_.k);
    mm.SetSingleShape(mnConfig_.singleCoreM, mnConfig_.singleCoreN, mnConfig_.curSingleCoreK);
    mm.SetTensorA(xFloatGm_[offset]);
    mm.SetTensorB(phiGm_[offset], true);
    mm.IterateAll(tempMMResGm_[outOffset], 0);
    mm.End();
    AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_CtoC);
    AscendC::CrossCoreWaitFlag(SYNC_CtoC);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_CtoV1);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::VectorComputeOffset()
{
    uint64_t alignSingleM = Ceil(curSingleM_, 2);
    vectorOffset_.singleCoreM = alignSingleM <  curSingleM_ ? alignSingleM : curSingleM_;
    if (subBlockIdx_ == 0) {
        vectorOffset_.offsetMStart = 0;
        vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
    } else {
        vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
        vectorOffset_.singleCoreM = curSingleM_ - vectorOffset_.singleCoreM;
        vectorOffset_.offsetMEnd = curSingleM_;
    }
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen,
                                                                          uint32_t curNdLen, uint32_t offsetM,
                                                                          uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(P));

    uint64_t offset = (globalOffsetM_ + offsetM) * matrixInfo_.nD + offsetNd;
    DataCopyPad(xFloatGm_[offset], x, copyParams);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::V0Prologue(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = matrixInfo_.totalLength - curBlock * chunkTSize_;
    }

    VectorComputeOffset();
    if (vectorOffset_.singleCoreM == 0) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
        AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        return;
    }

    for (uint32_t offsetNd = 0; offsetNd < matrixInfo_.nD; offsetNd += ND_LENGTH) {
        uint32_t curNdLen = ND_LENGTH;
        if (offsetNd + ND_LENGTH >= matrixInfo_.nD) {
            curNdLen = matrixInfo_.nD - offsetNd;
        }

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
                    this->template VFDoV0ProcessXIn<true, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXIn<true, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                }
                gammaInQueue_.FreeTensor(gammaUb_);
            } else {
                if (offsetNd == 0) {
                    this->template VFDoV0ProcessXIn<false, true>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXIn<false, false>((__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(), (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                }
            }

            outQueue_.template EnQue<P>(aL1Ub);
            aL1Ub = outQueue_.template DeQue<P>();
            this->DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);

            xInQueue_.FreeTensor(xLocal_);
            outQueue_.FreeTensor(aL1Ub);
        }
    }
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
    AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);

    this->VFDoV0ProcessInvRms((__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_, matrixInfo_.normEps);
    this->DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = matrixInfo_.totalLength - curBlock * chunkTSize_;
    }
    VectorComputeOffset();
    if (vectorOffset_.singleCoreM == 0) { // uint64_t类型永远不会小于0
        return;
    }
    AscendC::CrossCoreWaitFlag(SYNC_CtoV1);
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
__aicore__ inline void MhcPreKernelSplitND<T, P>::AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    uint64_t HMixOffset = 0;
    
    if (outFlag_) {
        HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;
    } else {
        HMixOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * ((vec1Count_ % 2) * coreNum_ + coreIdx_) + offsetT * mnConfig_.n;
    }
    HMixProcess(offsetT, lenT);

    matmulRes_ = xInQueue_.template DeQue<P>(); 
    LocalTensor<P> hResOutLocal = outQueue_.template AllocTensor<P>();

    __ubuf__ P* matmulPtr = (__ubuf__ P*)matmulRes_.GetPhyAddr();
    __ubuf__ P* invRmsPtr = (__ubuf__ P*)invRmsUb_.GetPhyAddr();
    __ubuf__ P* biasInPtr = (__ubuf__ P*)biasInUb_.GetPhyAddr();
    __ubuf__ P* alphaInPtr = (__ubuf__ P*)alphaInUb_.GetPhyAddr();
    __ubuf__ P* hPreBuffPtr = (__ubuf__ P*)hPreBuff_.GetPhyAddr();
    __ubuf__ P* hPostBuffPtr = (__ubuf__ P*)hPostBuff_.GetPhyAddr();
    __ubuf__ P* hResOutLocalPtr = (__ubuf__ P*)hResOutLocal.GetPhyAddr();

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

template <class T, class P>
__aicore__ inline void MhcPreKernelSplitND<T, P>::HMixProcess(uint64_t offsetT, uint64_t lenT)
{
    uint32_t mmResGmBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);
    uint32_t computeLen = lenT * mnConfig_.n;
    uint64_t HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;

    LocalTensor<P> hMixLocal = xInQueue_.template AllocTensor<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmResGmBlockNum);
    copyParams.blockLen = uint32_t(computeLen * sizeof(P));
    copyParams.srcStride = uint32_t((mnConfig_.curSingleCoreM * mnConfig_.curSingleCoreN - computeLen) * sizeof(P));
    copyParams.dstStride = uint32_t(0);
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};
    DataCopyPad(hMixLocal, tempMMResGm_[HMixOffset], copyParams, copyPadParams);
    xInQueue_.EnQue(hMixLocal);
    hMixLocal = xInQueue_.template DeQue<P>();

    uint64_t addOffset = 0;
    for(uint32_t mmResGmBlockIdx = 1; mmResGmBlockIdx < mmResGmBlockNum; mmResGmBlockIdx++)
    {
        addOffset += computeLen;
        Add(hMixLocal, hMixLocal, hMixLocal[addOffset], computeLen);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    if(outFlag_)
    {
        DataCopyExtParams MixOutCopyParams;
        MixOutCopyParams.blockCount = static_cast<uint16_t>(1);
        MixOutCopyParams.blockLen = uint32_t(computeLen * sizeof(P));
        MixOutCopyParams.srcStride = uint32_t(0);
        MixOutCopyParams.dstStride = uint32_t(0);
        DataCopyPad(mmResGm_[HMixOffset], hMixLocal, MixOutCopyParams);
    }
    xInQueue_.EnQue(hMixLocal);
}

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_ND_H_
