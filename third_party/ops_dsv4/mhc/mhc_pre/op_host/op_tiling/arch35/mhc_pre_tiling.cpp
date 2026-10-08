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
 * \file mhc_pre_tiling.cpp
 * \brief
 */

#include "mhc_pre_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "err/ops_err.h"
#include "../../../op_kernel/arch35/mhc_pre_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr uint32_t X_INDEX = 0;
const constexpr uint32_t PHI_INDEX = 1;
const constexpr uint32_t ALPHA_INDEX = 2;
const constexpr uint32_t BIAS_INDEX = 3;
const constexpr uint32_t GAMMA_INDEX = 4;

const constexpr uint32_t H_IN_INDEX = 0;
const constexpr uint32_t H_POST_INDEX = 1;
const constexpr uint32_t H_RES_INDEX = 2;
const constexpr uint32_t INV_RMS_INDEX = 3;
const constexpr uint32_t H_MIX_INDEX = 4;
const constexpr uint32_t H_PRE_INDEX = 5;

const constexpr int64_t INDEX_B_BSND = 0;
const constexpr int64_t INDEX_S_BSND = 1;
const constexpr int64_t INDEX_N_BSND = 2;
const constexpr int64_t INDEX_D_BSND = 3;

const constexpr int64_t INDEX_T_TND = 0;
const constexpr int64_t INDEX_N_TND = 1;
const constexpr int64_t INDEX_D_TND = 2;

const constexpr uint32_t N_VALID_VALUES[] = {4, 6, 8};
const constexpr uint32_t D_ALIGNMENT = 16;
const constexpr uint32_t CHUNK_T_MAX = 128;
const constexpr uint32_t V1_CHUNK_D_SIZE = 5120;
const constexpr uint32_t CHUNK_T_CALC_FACTOR = 32;
const constexpr uint64_t TOTAL_LENGTH_MIN = 1;
const constexpr uint64_t TOTAL_LENGTH_MAX = 65536;
const constexpr uint64_t D_MIN = 512;
const constexpr uint64_t D_MAX = 16384;
const constexpr uint32_t L0_B_SIZE = 8 * 1024;
const constexpr uint32_t FLOAT_ELE_SIZE = 8;
const constexpr uint32_t KERNEL_WIDTH = 8;

const constexpr uint32_t DB_L0C = 2;
const constexpr uint32_t DB_L0A = 2;
const constexpr uint32_t DB_L0B = 2;
const constexpr uint32_t STEP_K = 1;
const constexpr uint32_t DEPTH_K = 2;
const constexpr uint32_t STEP_MN = 1;

const constexpr size_t WORKSPACE_MULT_A = 2;
const constexpr size_t WORKSPACE_MULT_B = 192;
const constexpr size_t WORKSPACE_DIM_M = 256;
const constexpr size_t WORKSPACE_ELEMENTS = 8;
const constexpr size_t SYSTEM_WORKSPACE = 20 * 1024 * 1024;

const constexpr uint32_t SCHEDULE_MODE = 1;

const constexpr float DEFAULT_NORM_EPS = 1e-6f;
const constexpr float DEFAULT_HC_EPS = 1e-6f;

const constexpr uint64_t DECODE_BS_THRESHOLD = 512;
const constexpr uint32_t DECODE_CHUNK_T_SIZE = 2;
const constexpr uint32_t DECODE_ALIGN_16 = 16;
const constexpr size_t DECODE_WORKSPACE_ALIGN = 32;

static constexpr size_t OUT_H_IN_INDEX = 0;
static constexpr size_t OUT_H_POST_INDEX = 1;
static constexpr size_t OUT_H_RES_INDEX = 2;
static constexpr size_t OUT_INV_RMS_INDEX = 3;
static constexpr size_t OUT_MM_RES_INDEX = 4;
static constexpr size_t OUT_H_PRE_INDEX = 5;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;
static constexpr uint32_t HAS_GAMMA_TRUE = 1;

REGISTER_OPS_TILING_TEMPLATE(MhcPre, MhcPreBaseTiling, 1000);

ge::graphStatus MhcPreBaseTiling::GetInputShape()
{
    auto xTensor = context_->GetDynamicInputTensor(X_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, phiTensor);
    auto alphaTensor = context_->GetDynamicInputTensor(ALPHA_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, alphaTensor);
    auto biasTensor = context_->GetDynamicInputTensor(BIAS_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasTensor);

    auto gammaTensor = context_->GetDynamicInputTensor(GAMMA_INDEX, 0);
    hasGamma_ = (gammaTensor == nullptr) ? 0 : 1;

    auto xDims = xTensor->GetStorageShape().GetDimNum();
    if (xDims == BSND_DIM_NUM) {
        return ParseBsndFormat(xTensor);
    } else if (xDims == TND_DIM_NUM) {
        return ParseTndFormat(xTensor);
    }

    OP_LOGE(context_->GetNodeName(), "X dimNum must be %ld or %ld, got %zu", TND_DIM_NUM, BSND_DIM_NUM, xDims);
    return ge::GRAPH_FAILED;
}

ge::graphStatus MhcPreBaseTiling::CheckDescAndShape()
{
    for (size_t i = 0; i <= GAMMA_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "Input %zu desc is nullptr", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_->GetNodeName(), "Input %zu shape is nullptr", i),
                    return ge::GRAPH_FAILED);
    }

    for (size_t i = 0; i <= OUT_H_PRE_INDEX; i++) {
        auto desc = context_->GetOutputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context_->GetNodeName(), "Output %zu desc is nullptr", i),
                    return ge::GRAPH_FAILED);
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(shape == nullptr, OP_LOGE(context_->GetNodeName(), "Output %zu shape is nullptr", i),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckShapePositive()
{
    for (size_t i = 0; i <= GAMMA_INDEX; i++) {
        auto shape = context_->GetInputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            OP_CHECK_IF(
                shape.GetDim(j) <= 0,
                OP_LOGE(context_->GetNodeName(), "Input %zu dim %zu should be > 0, got %ld", i, j, shape.GetDim(j)),
                return ge::GRAPH_FAILED);
        }
    }

    for (size_t i = 0; i <= OUT_H_PRE_INDEX; i++) {
        auto shape = context_->GetOutputShape(i)->GetStorageShape();
        for (size_t j = 0; j < shape.GetDimNum(); j++) {
            OP_CHECK_IF(
                shape.GetDim(j) <= 0,
                OP_LOGE(context_->GetNodeName(), "Output %zu dim %zu should be > 0, got %ld", i, j, shape.GetDim(j)),
                return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckDataType()
{
    auto xDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
                OP_LOGE(context_->GetNodeName(), "X dtype should be BF16 or FP16, got %s",
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto phiDtype = context_->GetInputDesc(PHI_INDEX)->GetDataType();
    OP_CHECK_IF(phiDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "Phi dtype should be FLOAT32, got %s",
                        ge::TypeUtils::DataTypeToSerialString(phiDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto alphaDtype = context_->GetInputDesc(ALPHA_INDEX)->GetDataType();
    OP_CHECK_IF(alphaDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "Alpha dtype should be FLOAT32, got %s",
                        ge::TypeUtils::DataTypeToSerialString(alphaDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto biasDtype = context_->GetInputDesc(BIAS_INDEX)->GetDataType();
    OP_CHECK_IF(biasDtype != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "Bias dtype should be FLOAT32, got %s",
                        ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str()),
                return ge::GRAPH_FAILED);

    if (hasGamma_ == HAS_GAMMA_TRUE) {
        auto gammaDtype = context_->GetInputDesc(GAMMA_INDEX)->GetDataType();
        OP_CHECK_IF(gammaDtype != ge::DT_FLOAT,
                    OP_LOGE(context_->GetNodeName(), "Gamma dtype should be FLOAT32, got %s",
                            ge::TypeUtils::DataTypeToSerialString(gammaDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    auto outHinDtype = context_->GetOutputDesc(OUT_H_IN_INDEX)->GetDataType();
    OP_CHECK_IF(outHinDtype != xDtype,
                OP_LOGE(context_->GetNodeName(), "OutHin dtype should be %s, got %s",
                        ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(outHinDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckOutputShapeConsistency()
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    auto xShape = &xShapePtr->GetStorageShape();
    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum == BSND_DIM_NUM) {
        uint64_t b = xShape->GetDim(DIM_0);
        uint64_t s = xShape->GetDim(DIM_1);
        uint64_t n = xShape->GetDim(DIM_2);
        uint64_t d = xShape->GetDim(DIM_3);
        OP_CHECK_IF(CheckBsndOutputShape(b, s, n, d) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CheckBsndOutputShape failed"), return ge::GRAPH_FAILED);
    } else {
        uint64_t t = xShape->GetDim(DIM_0);
        uint64_t n = xShape->GetDim(DIM_1);
        uint64_t d = xShape->GetDim(DIM_2);
        OP_CHECK_IF(CheckTndOutputShape(t, n, d) != ge::GRAPH_SUCCESS,
            OP_LOGE(context_->GetNodeName(), "CheckTndOutputShape failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckBsndOutputShape(uint64_t b, uint64_t s, uint64_t n, uint64_t d)
{
    auto outHinShapePtr = context_->GetOutputShape(OUT_H_IN_INDEX);
    auto outHinShape = &outHinShapePtr->GetStorageShape();
    OP_CHECK_IF(outHinShape->GetDimNum() != DIM_NUM_3 || outHinShape->GetDim(DIM_0) != b ||
                outHinShape->GetDim(DIM_1) != s || outHinShape->GetDim(DIM_2) != d,
        OP_LOGE(context_->GetNodeName(), "OutHin shape (B,S,D) mismatch with X (B,S,N,D)"), return ge::GRAPH_FAILED);

    auto outHpostShapePtr = context_->GetOutputShape(OUT_H_POST_INDEX);
    auto outHpostShape = &outHpostShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpostShape->GetDimNum() != DIM_NUM_3 || outHpostShape->GetDim(DIM_0) != b ||
                outHpostShape->GetDim(DIM_1) != s || outHpostShape->GetDim(DIM_2) != n,
        OP_LOGE(context_->GetNodeName(), "OutHpost shape (B,S,N) mismatch with X"), return ge::GRAPH_FAILED);

    auto outHresShapePtr = context_->GetOutputShape(OUT_H_RES_INDEX);
    auto outHresShape = &outHresShapePtr->GetStorageShape();
    OP_CHECK_IF(outHresShape->GetDimNum() != DIM_NUM_4 || outHresShape->GetDim(DIM_0) != b ||
                outHresShape->GetDim(DIM_1) != s || outHresShape->GetDim(DIM_2) != n || outHresShape->GetDim(DIM_3) != n,
        OP_LOGE(context_->GetNodeName(), "OutHres shape (B,S,N,N) mismatch with X"), return ge::GRAPH_FAILED);

    auto outInvRmsShapePtr = context_->GetOutputShape(OUT_INV_RMS_INDEX);
    auto outInvRmsShape = &outInvRmsShapePtr->GetStorageShape();
    OP_CHECK_IF(outInvRmsShape->GetDimNum() != DIM_NUM_2 || outInvRmsShape->GetDim(DIM_0) != b ||
                outInvRmsShape->GetDim(DIM_1) != s,
        OP_LOGE(context_->GetNodeName(), "OutInvRms shape (B,S) mismatch with X"), return ge::GRAPH_FAILED);

    auto outMmresShapePtr = context_->GetOutputShape(OUT_MM_RES_INDEX);
    auto outMmresShape = &outMmresShapePtr->GetStorageShape();
    OP_CHECK_IF(outMmresShape->GetDimNum() != DIM_NUM_3 || outMmresShape->GetDim(DIM_0) != b ||
                outMmresShape->GetDim(DIM_1) != s,
        OP_LOGE(context_->GetNodeName(), "OutMmRes shape (B,S,matK) mismatch with X"), return ge::GRAPH_FAILED);

    auto outHpreShapePtr = context_->GetOutputShape(OUT_H_PRE_INDEX);
    auto outHpreShape = &outHpreShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpreShape->GetDimNum() != DIM_NUM_3 || outHpreShape->GetDim(DIM_0) != b ||
                outHpreShape->GetDim(DIM_1) != s || outHpreShape->GetDim(DIM_2) != n,
        OP_LOGE(context_->GetNodeName(), "OutHpre shape (B,S,N) mismatch with X"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckTndOutputShape(uint64_t t, uint64_t n, uint64_t d)
{
    auto outHinShapePtr = context_->GetOutputShape(OUT_H_IN_INDEX);
    auto outHinShape = &outHinShapePtr->GetStorageShape();
    OP_CHECK_IF(outHinShape->GetDimNum() != DIM_NUM_2 || outHinShape->GetDim(DIM_0) != t ||
                outHinShape->GetDim(DIM_1) != d,
        OP_LOGE(context_->GetNodeName(), "OutHin shape (T,D) mismatch with X (T,N,D)"), return ge::GRAPH_FAILED);

    auto outHpostShapePtr = context_->GetOutputShape(OUT_H_POST_INDEX);
    auto outHpostShape = &outHpostShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpostShape->GetDimNum() != DIM_NUM_2 || outHpostShape->GetDim(DIM_0) != t ||
                outHpostShape->GetDim(DIM_1) != n,
        OP_LOGE(context_->GetNodeName(), "OutHpost shape (T,N) mismatch with X"), return ge::GRAPH_FAILED);

    auto outHresShapePtr = context_->GetOutputShape(OUT_H_RES_INDEX);
    auto outHresShape = &outHresShapePtr->GetStorageShape();
    OP_CHECK_IF(outHresShape->GetDimNum() != DIM_NUM_3 || outHresShape->GetDim(DIM_0) != t ||
                outHresShape->GetDim(DIM_1) != n || outHresShape->GetDim(DIM_2) != n,
        OP_LOGE(context_->GetNodeName(), "OutHres shape (T,N,N) mismatch with X"), return ge::GRAPH_FAILED);

    auto outInvRmsShapePtr = context_->GetOutputShape(OUT_INV_RMS_INDEX);
    auto outInvRmsShape = &outInvRmsShapePtr->GetStorageShape();
    OP_CHECK_IF(outInvRmsShape->GetDimNum() != 1 || outInvRmsShape->GetDim(DIM_0) != t,
        OP_LOGE(context_->GetNodeName(), "OutInvRms shape (T) mismatch with X"), return ge::GRAPH_FAILED);

    auto outMmresShapePtr = context_->GetOutputShape(OUT_MM_RES_INDEX);
    auto outMmresShape = &outMmresShapePtr->GetStorageShape();
    OP_CHECK_IF(outMmresShape->GetDimNum() != DIM_NUM_2 || outMmresShape->GetDim(DIM_0) != t,
        OP_LOGE(context_->GetNodeName(), "OutMmRes shape (T,matK) mismatch with X"), return ge::GRAPH_FAILED);

    auto outHpreShapePtr = context_->GetOutputShape(OUT_H_PRE_INDEX);
    auto outHpreShape = &outHpreShapePtr->GetStorageShape();
    OP_CHECK_IF(outHpreShape->GetDimNum() != DIM_NUM_2 || outHpreShape->GetDim(DIM_0) != t ||
                outHpreShape->GetDim(DIM_1) != n,
        OP_LOGE(context_->GetNodeName(), "OutHpre shape (T,N) mismatch with X"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckDataRange()
{
    OP_CHECK_IF(totalLength_ < TOTAL_LENGTH_MIN || totalLength_ > TOTAL_LENGTH_MAX,
                OP_LOGE(context_->GetNodeName(), "TotalLength must be in [%lu, %lu], got %lu", 
                        TOTAL_LENGTH_MIN, TOTAL_LENGTH_MAX, totalLength_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(D_ < D_MIN || D_ > D_MAX, 
                OP_LOGE(context_->GetNodeName(), "D must be in [%lu, %lu], got %lu", D_MIN, D_MAX, D_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseBsndFormat(const gert::Tensor *xTensor)
{
    uint64_t batch = xTensor->GetStorageShape().GetDim(INDEX_B_BSND);
    uint64_t sequence = xTensor->GetStorageShape().GetDim(INDEX_S_BSND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_BSND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_BSND);

    totalLength_ = batch * sequence;
    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ParseTndFormat(const gert::Tensor *xTensor)
{
    totalLength_ = xTensor->GetStorageShape().GetDim(INDEX_T_TND);
    uint64_t numsResidual = xTensor->GetStorageShape().GetDim(INDEX_N_TND);
    uint64_t dimens = xTensor->GetStorageShape().GetDim(INDEX_D_TND);

    matK_ = numsResidual * dimens;
    N_ = numsResidual;
    D_ = dimens;

    return ValidateAndSetTilingParams(xTensor);
}

ge::graphStatus MhcPreBaseTiling::ValidateAndSetTilingParams(const gert::Tensor *xTensor)
{
    auto phiTensor = context_->GetDynamicInputTensor(PHI_INDEX, 0);
    auto phiDims = phiTensor->GetStorageShape().GetDimNum();
    if (phiDims != 2) {
        OP_LOGE(context_->GetNodeName(), "Phi dims must be 2, but got %u", phiDims);
        return ge::GRAPH_FAILED;
    }

    bool isValidN = false;
    for (auto validN : N_VALID_VALUES) {
        if (N_ == validN) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE(context_->GetNodeName(), "N must be 4/6/8, but got %lu", N_);
        return ge::GRAPH_FAILED;
    }

    if (D_ % D_ALIGNMENT != 0) {
        OP_LOGE(context_->GetNodeName(),
                "D must be aligned to %u elements, but got %lu", D_ALIGNMENT, D_);
        return ge::GRAPH_FAILED;
    }

    matM_ = totalLength_;
    matN_ = phiTensor->GetStorageShape().GetDim(0);

    uint64_t phiSecondDim = phiTensor->GetStorageShape().GetDim(1);
    if (phiSecondDim != matK_) {
        OP_LOGE(context_->GetNodeName(), "Phi dim[1]=%lu must equal matK=%lu", phiSecondDim, matK_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseInputAndAttr()
{
    if (InitPlatformMemory() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    
    OP_CHECK_IF(CheckDescAndShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckDescAndShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapePositive() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckShapePositive failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDataType() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckDataType failed"),
                return ge::GRAPH_FAILED);

    if (GetInputShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "GetInputShape failed");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckDataRange() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CheckDataRange failed"),
                return ge::GRAPH_FAILED);

    if (CheckOutputShapeConsistency() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "CheckOutputShapeConsistency failed");
        return ge::GRAPH_FAILED;
    }

    if (ParseOutputFlags() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ParseEpsAttributes();
}

ge::graphStatus MhcPreBaseTiling::InitPlatformMemory()
{
    uint64_t ubSize, l1Size, l0CSize;

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    ubSize_ = ubSize;  // 保存UB大小供后续校验使用

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseOutputFlags()
{
    auto invRmsDesc = context_->GetOutputDesc(INV_RMS_INDEX);
    auto hMixDesc = context_->GetOutputDesc(H_MIX_INDEX);
    auto hPreDesc = context_->GetOutputDesc(H_PRE_INDEX);

    outFlag_ = (invRmsDesc != nullptr && hMixDesc != nullptr && hPreDesc != nullptr);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::ParseEpsAttributes()
{
    auto attrs = context_->GetAttrs();

    auto normEpsPtr = attrs->GetAttrPointer<double>(1);
    normEps_ = (normEpsPtr != nullptr) ? static_cast<float>(*normEpsPtr) : DEFAULT_NORM_EPS;

    auto hcEpsPtr = attrs->GetAttrPointer<double>(2);
    hcEps_ = (hcEpsPtr != nullptr) ? static_cast<float>(*hcEpsPtr) : DEFAULT_HC_EPS;

    return ge::GRAPH_SUCCESS;
}


void MhcPreBaseTiling::FillTilingData()
{
    tilingData_.matmulTiling.set_dbL0C(DB_L0C);
    tilingData_.matmulTiling.set_dbL0A(DB_L0A);
    tilingData_.matmulTiling.set_dbL0B(DB_L0B);
    tilingData_.matmulTiling.set_stepKa(STEP_K);
    tilingData_.matmulTiling.set_stepKb(STEP_K);
    tilingData_.matmulTiling.set_depthA1(DEPTH_K);
    tilingData_.matmulTiling.set_depthB1(DEPTH_K);
    tilingData_.matmulTiling.set_stepM(STEP_MN);
    tilingData_.matmulTiling.set_stepN(STEP_MN);

    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;

    if (tilingMode_ == TilingMode::SPLIT_ND) {
        baseN = (matN_ + DECODE_ALIGN_16 - 1) / DECODE_ALIGN_16 * DECODE_ALIGN_16;
        baseK = L0_B_SIZE / baseN / FLOAT_ELE_SIZE * KERNEL_WIDTH;
        baseM = L0_B_SIZE / baseK / DECODE_ALIGN_16 * DECODE_ALIGN_16;
    } else {
        baseM = chunkTSize_;
        baseN = baseM;
        baseK = L0_B_SIZE / baseN / FLOAT_ELE_SIZE * KERNEL_WIDTH;
    }

    tilingData_.matmulTiling.set_baseM(baseM);
    tilingData_.matmulTiling.set_baseN(baseN);
    tilingData_.matmulTiling.set_baseK(baseK);

    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_nD(matK_);
    tilingData_.set_fusionSize(matN_);
    tilingData_.set_N(N_);
    tilingData_.set_D(D_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_chunkTSize(chunkTSize_);
    tilingData_.set_v1ChunkDSize(v1ChunkDSize_);
    tilingData_.set_outFlag(outFlag_);
    tilingData_.set_hasGamma(hasGamma_);

    float scaleMean = 1.0f / static_cast<float>(matK_);
    tilingData_.set_scaleMean(scaleMean);
}

ge::graphStatus MhcPreBaseTiling::TilingProcess()
{
    tilingMode_ = (totalLength_ <= DECODE_BS_THRESHOLD) ? TilingMode::SPLIT_ND : TilingMode::SPLIT_BS;

    if (tilingMode_ == TilingMode::SPLIT_ND) {
        chunkTSize_ = DECODE_CHUNK_T_SIZE;
        v1ChunkDSize_ = V1_CHUNK_D_SIZE;
    } else {
        chunkTSize_ = (((totalLength_ + blockDim_ - 1) / blockDim_) + CHUNK_T_CALC_FACTOR - 1) / CHUNK_T_CALC_FACTOR *
                      CHUNK_T_CALC_FACTOR;
        if (chunkTSize_ > CHUNK_T_MAX) {
            chunkTSize_ = CHUNK_T_MAX;
        }
        v1ChunkDSize_ = V1_CHUNK_D_SIZE;
    }

    size_t userWorkspaceSize;
    size_t systemWorkspaceSize = SYSTEM_WORKSPACE;

    if (tilingMode_ == TilingMode::SPLIT_ND) {
        size_t xFloatWorkspaceSizeRaw = static_cast<size_t>(matM_) * static_cast<size_t>(matK_) * sizeof(float);
        size_t xFloatWorkspaceSize =
            (xFloatWorkspaceSizeRaw + DECODE_WORKSPACE_ALIGN - 1U) / DECODE_WORKSPACE_ALIGN * DECODE_WORKSPACE_ALIGN;
        size_t mmResWorkspaceSize = 0;

        uint64_t chunkNd = (matK_ + blockDim_ - 1) / blockDim_;
        uint64_t mmResBlockNum = (matK_ + chunkNd - 1) / chunkNd;
        mmResWorkspaceSize = static_cast<size_t>(mmResBlockNum) * static_cast<size_t>(matM_) *
                             static_cast<size_t>(matN_) * sizeof(float);

        userWorkspaceSize = xFloatWorkspaceSize + mmResWorkspaceSize;
    } else {
        userWorkspaceSize =
            (WORKSPACE_MULT_A * WORKSPACE_MULT_B * WORKSPACE_DIM_M +
             WORKSPACE_MULT_A * WORKSPACE_MULT_B * (KERNEL_WIDTH * KERNEL_WIDTH + WORKSPACE_MULT_A * KERNEL_WIDTH)) *
            sizeof(float) * blockDim_;
    }
    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    if (CheckUbBufferSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm_.SetBias(false);
    mm_.SetDim(1);
    mm_.SetShape(matM_, matN_, matK_);
    mm_.SetOrgShape(matM_, matN_, matK_);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "MhcPre tiling get failed, batch: %lu, M: %lu", totalLength_, matM_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreBaseTiling::CheckUbBufferSize()
{
    size_t fixedBufferSize = 0;
    size_t dynamicBufferSize = 0;
    const size_t floatSize = sizeof(float);

    if (tilingMode_ == TilingMode::SPLIT_BS) {
        fixedBufferSize = 80 * 1024 + 20 * 1024 + 40 * 1024;
    } else {
        fixedBufferSize = 80 * 1024 + 32 * 1024 + 20 * 1024;
    }

    dynamicBufferSize = static_cast<size_t>(matN_) * floatSize * 2;

    if (outFlag_) {
        size_t invRmsSize = (tilingMode_ == TilingMode::SPLIT_BS)
            ? (chunkTSize_ / 2) * floatSize
            : ((chunkTSize_ + 1) / 2) * floatSize;
        dynamicBufferSize += invRmsSize;
    }

    if (hasGamma_) {
        dynamicBufferSize += static_cast<size_t>(D_) * floatSize;
    }

    size_t totalUbRequired = fixedBufferSize + dynamicBufferSize;
    if (totalUbRequired > ubSize_) {
        OP_LOGE(context_->GetNodeName(),
                "UB buffer require %zu bytes exceeds ubSize %lu bytes",
                totalUbRequired, ubSize_);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MhcPreBaseTiling::DoOpTiling()
{
    auto inputXDesc = context_->GetInputDesc(0);
    if (inputXDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "X input pointer is null");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();

    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void MhcPreBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "BlockDim: [%u]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "TotalLength: [%lu]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "ND: [%lu]", tilingData_.get_nD());
    OP_LOGD(context_->GetNodeName(), "FusionSize: [%lu]", tilingData_.get_fusionSize());
    OP_LOGD(context_->GetNodeName(), "N: [%lu]", tilingData_.get_N());
    OP_LOGD(context_->GetNodeName(), "D: [%lu]", tilingData_.get_D());
    OP_LOGD(context_->GetNodeName(), "NormEps: [%e]", tilingData_.get_normEps());
    OP_LOGD(context_->GetNodeName(), "HcEps: [%e]", tilingData_.get_hcEps());
    OP_LOGD(context_->GetNodeName(), "OutFlag: [%d]", tilingData_.get_outFlag());
    OP_LOGD(context_->GetNodeName(), "HasGamma: [%u]", tilingData_.get_hasGamma());
    OP_LOGD(context_->GetNodeName(), "ChunkTSize: [%u]", tilingData_.get_chunkTSize());
    OP_LOGD(context_->GetNodeName(), "V1ChunkDSize: [%u]", tilingData_.get_v1ChunkDSize());
}

uint64_t MhcPreBaseTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(tilingMode_));
}

ge::graphStatus MhcPreBaseTiling::PostTiling()
{
    OP_CHECK_IF(
        tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "TilingData size not aligned to 8, got %zu", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(SCHEDULE_MODE);

    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "Workspaces is null"),
                return ge::GRAPH_FAILED);

    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc4mHCPre(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[mHCPreTilingTilingFunc]", "Context is null"),
                return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}


static ge::graphStatus TilingPrepare4mHCPre(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_CUBE_INNER_ERR("[TilingPrepare4mHC]", "Context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "PlatformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MhcPreCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "CompileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(), "ParseCompileInfo success, L1Size: %lu, L2Size: %lu, CoreNum: %lu",
            compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MhcPre).Tiling(TilingFunc4mHCPre).TilingParse<MhcPreCompileInfo>(TilingPrepare4mHCPre);
} // namespace optiling
