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
 * \file mhc_sinkhorn_tiling.cpp
 * \brief mhc_sinkhorn_tiling
 */

#include <vector>
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "mhc_sinkhorn_tiling.h"

using namespace AscendC;
namespace optiling {
constexpr int64_t X_IDX = 0;
constexpr int64_t Y_IDX = 0;
constexpr int64_t ATTR_EPS_IDX = 0;
constexpr int64_t ATTR_NUM_ITERS_IDX = 1;
constexpr int64_t ATTR_OUT_FLAG_IDX = 2;

constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t DIM_ZERO = 0;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t NUM_ONE_HUNDRED = 100;

constexpr int64_t N_NUM_4 = 4;
constexpr int64_t N_NUM_6 = 6;
constexpr int64_t N_NUM_8 = 8;

constexpr int64_t SIMD_RESERVED_SIZE = static_cast<uint64_t>(8) * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 0;
constexpr int64_t MASK_BUFFER = 64;
constexpr int64_t MAX_BUFFER = 256;

static const std::set<ge::DataType> X_DTYPE = {ge::DT_FLOAT};

bool MhcSinkhornTiling::IsCapable()
{
    return true;
}

ge::graphStatus MhcSinkhornTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const MhcSinkhornCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size less than 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling GetShapeAttrsInfo.");
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto epsPtr = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, epsPtr);
    eps_ = static_cast<float>(*epsPtr);
    auto numItersPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ITERS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numItersPtr);
    num_iters_ = static_cast<int64_t>(*numItersPtr);
    OP_CHECK_IF(
        (num_iters_ < NUM_ONE || num_iters_ > NUM_ONE_HUNDRED),
        OP_LOGE(opName_, "num_iters_ must be greater than 0 and less than or equal to 100, but got %d .", num_iters_),
        return ge::GRAPH_FAILED);
    auto outFlagPtr = attrs->GetAttrPointer<int64_t>(ATTR_OUT_FLAG_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outFlagPtr);
    out_flag_ = static_cast<int64_t>(*outFlagPtr);
    OP_CHECK_IF((out_flag_ != NUM_ZERO && out_flag_ != NUM_ONE),
                OP_LOGE(opName_, "outFlag value error, outFlag must be 0 or 1, but got outFlag = %d .", out_flag_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "input shape check failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputDtype()
{
    auto xPtr = context_->GetInputDesc(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xPtr);
    xDtype_ = xPtr->GetDataType();
    OP_CHECK_IF((X_DTYPE.find(xDtype_) == X_DTYPE.end()),
                OP_LOGE(opName_, "indices dtype only support float32 currently, please check."),
                return ge::GRAPH_FAILED);

    auto outputPtr = context_->GetOutputDesc(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputPtr);
    auto outputDtype = outputPtr->GetDataType();
    OP_CHECK_IF(outputDtype != xDtype_, OP_LOGE(opName_, "expected output dtype to be equal to xdtype, please check."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::CheckInputShape()
{
    auto xShapePtr = context_->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    OP_CHECK_IF((xShape.GetShapeSize() == 0),
                OP_LOGE(opName_, "Input x must not be empty tensor"), return ge::GRAPH_FAILED);
    xDimNum_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF((xDimNum_ != DIM_NUM_3 && xDimNum_ != DIM_NUM_4),
                OP_LOGE(opName_, "xDimNum must be 3 or 4, but got %d .", xDimNum_), return ge::GRAPH_FAILED);
    n_ = xShape.GetDim(DIM_TWO);
    OP_CHECK_IF((n_ != N_NUM_4 && n_ != N_NUM_6 && n_ != N_NUM_8),
                OP_LOGE(opName_, "the nDim of x must be 4 or 6 or 8, but got %d .", n_), return ge::GRAPH_FAILED);
    if (xDimNum_ == 3) {
        T_ = xShape.GetDim(DIM_ZERO);
    } else {
        T_ = xShape.GetDim(DIM_ZERO) * xShape.GetDim(DIM_ONE);
    }

    auto yShapePtr = context_->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    yDimNum_ = static_cast<int64_t>(yShape.GetDimNum());
    OP_CHECK_IF((yDimNum_ != DIM_NUM_3 && yDimNum_ != DIM_NUM_4),
                OP_LOGE(opName_, "yDimNum must be 3 or 4, but got %d .", yDimNum_), return ge::GRAPH_FAILED);
    int64_t n = yShape.GetDim(DIM_TWO);
    OP_CHECK_IF((n != 4 && n != 6 && n != 8), OP_LOGE(opName_, "the nDim of y must be 4 or 6 or 8, but got %d .", n),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yDimNum_ != xDimNum_), OP_LOGE(opName_, "yDimNum must be equal xDimNum"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::SplitByCoreNum(int64_t tCoreNum, int64_t ubBlockX, int64_t xDtypeSize, int64_t &tUbFactor,
                                       int64_t &tCoreLoop, int64_t &tUbFactorTail)
{
    int64_t blockFactorAlignX = Ops::Base::CeilAlign(tCoreNum * n_ * n_, ubBlockX);
    if (blockFactorAlignX * 4 <= Ops::Base::FloorDiv(ubSizeUsed_, xDtypeSize)) {
        tUbFactor = blockFactorAlignX;
    } else {
        tUbFactor =
            Ops::Base::FloorAlign(Ops::Base::FloorDiv(ubSizeUsed_ / static_cast<int64_t>(4), xDtypeSize), ubBlockX);
        tUbFactor = Ops::Base::FloorAlign(tUbFactor, n_ * n_);
    }
    tCoreLoop = Ops::Base::CeilDiv(blockFactorAlignX, tUbFactor);
    tUbFactorTail = blockFactorAlignX - (tCoreLoop - 1) * tUbFactor;
}

ge::graphStatus MhcSinkhornTiling::DoOpTiling()
{
    OP_LOGD(opName_, "MhcSinkhorn tiling DoOpTiling.");

    tNormCore_ = Ops::Base::CeilDiv(T_, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(T_, tNormCore_);
    tTailCore_ = T_ - tNormCore_ * (usedCoreNum_ - 1);

    ubSizeUsed_ = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    int64_t ubBlockX = Ops::Base::FloorDiv(ubBlock, xDtypeSize);
    int64_t tTailUbFactor = 0;

    SplitByCoreNum(tNormCore_, ubBlockX, xDtypeSize, tUbFactor_, tNormCoreLoop_, tUbFactorTail_);
    SplitByCoreNum(tTailCore_, ubBlockX, xDtypeSize, tTailUbFactor, tTailCoreLoop_, tUbTailTail_);

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::SetTilingData()
{
    MhcSinkhornTilingData *tilingData = context_->GetTilingData<MhcSinkhornTilingData>();
    tilingData->eps = eps_;
    tilingData->num_iters = num_iters_;
    tilingData->out_flag = out_flag_;
    tilingData->n = n_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->tNormCoreLoop = tNormCoreLoop_;
    tilingData->tUbFactor = tUbFactor_;
    tilingData->tUbFactorTail = tUbFactorTail_;
    tilingData->tTailCoreLoop = tTailCoreLoop_;
    tilingData->tUbTailTail = tUbTailTail_;
    tilingData->tNormCore = tNormCore_;
}

ge::graphStatus MhcSinkhornTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MhcSinkhornTiling::GetTilingKey() const
{
    int64_t tilingKey = 1;
    return GET_TPL_TILING_KEY(tilingKey);
}

ge::graphStatus MhcSinkhornTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcSinkhornTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MhcSinkhornTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "\n eps: " << eps_;
    info << "\n numIters: " << num_iters_;
    info << "\n outFlag: " << out_flag_;
    info << "\n T: " << T_;
    info << "\n n: " << n_;
    info << "\n tilingKey: " << GetTilingKey();
    info << "\n usedCoreNum: " << usedCoreNum_;
    info << "\n tNormCoreLoop: " << tNormCoreLoop_;
    info << "\n tUbFactor: " << tUbFactor_;
    info << "\n tUbFactorTail: " << tUbFactorTail_;
    info << "\n tTailCoreLoop: " << tTailCoreLoop_;
    info << "\n tUbTailTail: " << tUbTailTail_;
    info << "\n tNormCore: " << tNormCore_;
    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingForMhcSinkhorn(gert::TilingContext *context)
{
    MhcSinkhornTiling tiling(context);
    auto ret = tiling.DoTiling();
    return ret;
}

static ge::graphStatus TilingPrepareForMhcSinkhorn([[maybe_unused]] gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepareForMhcSinkhorn entering.");
    auto compileInfo = context->GetCompiledInfo<MhcSinkhornCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcSinkhorn)
    .Tiling(TilingForMhcSinkhorn)
    .TilingParse<MhcSinkhornCompileInfo>(TilingPrepareForMhcSinkhorn);
} // namespace optiling
