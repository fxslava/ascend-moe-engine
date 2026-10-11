/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mhc_pre.h"
#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr size_t DIM_NUM_1 = 1UL;
constexpr size_t DIM_NUM_2 = 2UL;
constexpr size_t DIM_NUM_3 = 3UL;
constexpr size_t DIM_NUM_4 = 4UL;

constexpr int64_t N_VALID_VALUES[] = {4, 6, 8};
constexpr int64_t D_ALIGNMENT = 16;
constexpr int64_t ALPHA_DIM_SIZE = 3;
constexpr int64_t PHI_DIM_OFFSET = 2;

bool CheckAlphaShape(const aclTensor *alphaTensor);
bool ValidateNDParams(int64_t n, int64_t d);
bool CheckPhiShape(const aclTensor *phiTensor, int64_t n2Plus2n, int64_t nD);
bool CheckBiasShape(const aclTensor *biasTensor, int64_t n2Plus2n);
bool CheckGammaShape(const aclTensor *gammaOptional, int64_t n, int64_t d);

struct MhcParamsBase {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *gammaOptional = nullptr;
    double normEps;
    double hcEps;
    aclTensor *hIn = nullptr;
    aclTensor *hPost = nullptr;
    aclTensor *hRes = nullptr;
    aclTensor *invRmsOptional = nullptr;
    aclTensor *hMixOptional = nullptr;
    aclTensor *hPreOptional = nullptr;

    // 用于存储转换后的连续tensor（在ConvertDataContiguous中使用）
    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *biasContiguous = nullptr;
    const aclTensor *gammaOptionalContiguous = nullptr;
};

class MhcBuilder {
public:
    static MhcBuilder Create()
    {
        MhcBuilder obj;

        return obj;
    }

    MhcBuilder &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                         const aclTensor *gammaOptional)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        obj_.gammaOptional = gammaOptional;
        return *this;
    }

    MhcBuilder &SetAttr(double normEps, double hcEps)
    {
        obj_.normEps = normEps;
        obj_.hcEps = hcEps;
        return *this;
    }

    MhcBuilder &SetOutput(aclTensor *hIn, aclTensor *hPost, aclTensor *hRes)
    {
        obj_.hIn = hIn;
        obj_.hPost = hPost;
        obj_.hRes = hRes;
        return *this;
    }

    MhcBuilder &SetOptionalOutput(aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional)
    {
        obj_.invRmsOptional = invRmsOptional;
        obj_.hMixOptional = hMixOptional;
        obj_.hPreOptional = hPreOptional;

        return *this;
    }

    MhcParamsBase Build() const
    {
        return obj_;
    }

private:
    MhcParamsBase obj_;
};

bool CheckNotNull(const MhcParamsBase &params)
{
    if (params.x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "X tensor is nullptr");
        return false;
    }
    if (params.phi == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Phi tensor is nullptr");
        return false;
    }
    if (params.alpha == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Alpha tensor is nullptr");
        return false;
    }
    if (params.bias == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Bias tensor is nullptr");
        return false;
    }
    if (params.hIn == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HIn tensor is nullptr");
        return false;
    }
    if (params.hPost == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HPost tensor is nullptr");
        return false;
    }
    if (params.hRes == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "HRes tensor is nullptr");
        return false;
    }
    return true;
}

bool CheckEmptyTensor(const MhcParamsBase &params)
{
    if (params.x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor is empty");
        return false;
    }
    if (params.phi->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor is empty");
        return false;
    }
    if (params.alpha->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor is empty");
        return false;
    }
    if (params.bias->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor is empty");
        return false;
    }
    return true;
}

bool CheckInputOutDims(const MhcParamsBase &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor dim num must be 3 or 4, but got %zu", xDimNum);
        return false;
    }

    auto phiDimNum = params.phi->GetViewShape().GetDimNum();
    if (phiDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor dim num must be 2, but got %zu", phiDimNum);
        return false;
    }

    auto alphaDimNum = params.alpha->GetViewShape().GetDimNum();
    if (alphaDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor dim num must be 1, but got %zu", alphaDimNum);
        return false;
    }

    auto biasDimNum = params.bias->GetViewShape().GetDimNum();
    if (biasDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim num must be 1, but got %zu", biasDimNum);
        return false;
    }

    if (params.gammaOptional != nullptr) {
        auto gammaDimNum = params.gammaOptional->GetViewShape().GetDimNum();
        if (gammaDimNum != DIM_NUM_2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor dim num must be 2, but got %zu", gammaDimNum);
            return false;
        }
    }

    return true;
}

bool CheckInputOutShape(const MhcParamsBase &params)
{
    auto xShape = params.x->GetViewShape();

    if (!CheckAlphaShape(params.alpha)) {
        return false;
    }

    int64_t n = 0;
    int64_t d = 0;
    int64_t nD = 0;
    auto xDimNum = xShape.GetDimNum();
    if (xDimNum == DIM_NUM_4) {
        n = xShape.GetDim(2);
        d = xShape.GetDim(3);
        nD = n * d;
    } else if (xDimNum == DIM_NUM_3) {
        n = xShape.GetDim(1);
        d = xShape.GetDim(2);
        nD = n * d;
    }

    if (!ValidateNDParams(n, d)) {
        return false;
    }

    int64_t n2Plus2n = n * n + 2 * n;

    if (!CheckPhiShape(params.phi, n2Plus2n, nD)) {
        return false;
    }

    if (!CheckBiasShape(params.bias, n2Plus2n)) {
        return false;
    }

    if (!CheckGammaShape(params.gammaOptional, n, d)) {
        return false;
    }

    return true;
}

bool CheckAlphaShape(const aclTensor *alphaTensor)
{
    auto alphaShape = alphaTensor->GetViewShape();
    if (alphaShape.GetDim(0) != ALPHA_DIM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor shape must be (3), but got (%ld)", alphaShape.GetDim(0));
        return false;
    }
    return true;
}

bool ValidateNDParams(int64_t n, int64_t d)
{
    if (n <= 0 || d <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid X tensor shape: n=%ld, d=%ld", n, d);
        return false;
    }

    bool isValidN = false;
    for (size_t i = 0; i < sizeof(N_VALID_VALUES) / sizeof(N_VALID_VALUES[0]); ++i) {
        if (n == N_VALID_VALUES[i]) {
            isValidN = true;
            break;
        }
    }
    if (!isValidN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N must be 4/6/8, but got n=%ld", n);
        return false;
    }

    if (d % D_ALIGNMENT != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "D must be aligned to %u elements, but got d=%ld", D_ALIGNMENT, d);
        return false;
    }

    return true;
}

bool CheckPhiShape(const aclTensor *phiTensor, int64_t n2Plus2n, int64_t nD)
{
    auto phiShape = phiTensor->GetViewShape();
    if (phiShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor first dim must be n^2+2n=%ld, but got %ld", n2Plus2n,
                phiShape.GetDim(0));
        return false;
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }
    return true;
}

bool CheckBiasShape(const aclTensor *biasTensor, int64_t n2Plus2n)
{
    auto biasShape = biasTensor->GetViewShape();
    if (biasShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dim must be n^2+2n=%ld, but got %ld", n2Plus2n,
                biasShape.GetDim(0));
        return false;
    }
    return true;
}

bool CheckGammaShape(const aclTensor *gammaOptional, int64_t n, int64_t d)
{
    if (gammaOptional != nullptr) {
        auto gammaShape = gammaOptional->GetViewShape();
        if (gammaShape.GetDim(0) != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor first dim must be n=%ld, but got %ld", n,
                    gammaShape.GetDim(0));
            return false;
        }
        if (gammaShape.GetDim(1) != d) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor second dim must be D=%ld, but got %ld", d,
                    gammaShape.GetDim(1));
            return false;
        }
    }
    return true;
}

bool CheckDtypeValid(const MhcParamsBase &params)
{
    const std::initializer_list<DataType> X_SUPPORT_DTYPE_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16};

    auto xDtype = params.x->GetDataType();
    bool xDtypeValid = false;
    for (const auto &dtype : X_SUPPORT_DTYPE_LIST) {
        if (xDtype == dtype) {
            xDtypeValid = true;
            break;
        }
    }
    if (!xDtypeValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor dtype must be BF16 or FP16");
        return false;
    }

    if (params.phi->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor dtype must be FP32");
        return false;
    }
    if (params.alpha->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor dtype must be FP32");
        return false;
    }
    if (params.bias->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor dtype must be FP32");
        return false;
    }
    if (params.gammaOptional != nullptr) {
        if (params.gammaOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor dtype must be FP32");
            return false;
        }
    }

    return true;
}

static bool IsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

bool CheckFormat(const MhcParamsBase &params)
{
    if (IsPrivateFormat(params.x->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "X tensor format must be ND");
        return false;
    }

    if (IsPrivateFormat(params.phi->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Phi tensor format must be ND");
        return false;
    }

    if (IsPrivateFormat(params.alpha->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Alpha tensor format must be ND");
        return false;
    }

    if (IsPrivateFormat(params.bias->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias tensor format must be ND");
        return false;
    }

    if (params.gammaOptional != nullptr) {
        if (IsPrivateFormat(params.gammaOptional->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GammaOptional tensor format must be ND");
            return false;
        }
    }

    return true;
}

aclnnStatus CheckParams(const MhcParamsBase &params)
{
    // 1. 检查参数是否为空指针、空tensor
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);

    // 2. 校验输入、输出参数维度
    CHECK_RET(CheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(CheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据形状是否支持
    CHECK_RET(CheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ConvertDataContiguous(MhcParamsBase &params, aclOpExecutor *executor)
{
    // 将输入tensor转换为连续格式
    params.xContiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.phiContiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.alphaContiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.biasContiguous = l0op::Contiguous(params.bias, executor);
    CHECK_RET(params.biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gammaOptional != nullptr) {
        params.gammaOptionalContiguous = l0op::Contiguous(params.gammaOptional, executor);
        CHECK_RET(params.gammaOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus mHCPreCommonProcess(MhcParamsBase &params, aclOpExecutor *executor)
{
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = ConvertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t outFlag =
        (params.invRmsOptional != nullptr && params.hMixOptional != nullptr && params.hPreOptional != nullptr) ? 1 : 0;
    auto outParams = l0op::MhcPre(params.xContiguous, params.phiContiguous, params.alphaContiguous,
                                  params.biasContiguous, params.gammaOptionalContiguous, outFlag, params.normEps,
                                  params.hcEps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    auto out0 = std::get<0>(outParams);
    auto ret0 = l0op::ViewCopy(out0, params.hIn, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out1 = std::get<1>(outParams);
    auto ret1 = l0op::ViewCopy(out1, params.hPost, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out2 = std::get<2>(outParams);
    auto retView = l0op::ViewCopy(out2, params.hRes, executor);
    CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.invRmsOptional != nullptr) {
        auto out3 = std::get<3>(outParams);
        retView = l0op::ViewCopy(out3, params.invRmsOptional, executor);
        CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (params.hMixOptional != nullptr) {
        auto out4 = std::get<4>(outParams);
        retView = l0op::ViewCopy(out4, params.hMixOptional, executor);
        CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (params.hPreOptional != nullptr) {
        auto out5 = std::get<5>(outParams);
        retView = l0op::ViewCopy(out5, params.hPreOptional, executor);
        CHECK_RET(retView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                        const aclTensor *bias, const aclTensor *gammaOptional, double normEps,
                                        double hcEps, aclTensor *hIn, aclTensor *hPost, aclTensor *hRes,
                                        aclTensor *invRmsOptional, aclTensor *hMixOptional, aclTensor *hPreOptional,
                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPre, DFX_IN(x, phi, alpha, bias, gammaOptional, normEps, hcEps),
                   DFX_OUT(hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional));
    auto uniqueExecutor = CREATE_EXECUTOR();

    MhcParamsBase params = MhcBuilder::Create()
                               .SetInput(x, phi, alpha, bias, gammaOptional)
                               .SetAttr(normEps, hcEps)
                               .SetOutput(hIn, hPost, hRes)
                               .SetOptionalOutput(invRmsOptional, hMixOptional, hPreOptional)
                               .Build();

    auto ret = mHCPreCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPre);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace
#ifdef __cplusplus
}
#endif
