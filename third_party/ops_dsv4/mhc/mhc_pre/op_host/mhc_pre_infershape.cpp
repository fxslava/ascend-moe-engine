/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_pre_infershape.cpp
 * \brief
 */
#include <map>
#include <string>
#include <sstream>
#include <initializer_list>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t X_INDEX = 0;
const constexpr int64_t PHI_INDEX = 1;

const constexpr int64_t OUT_H_IN_INDEX = 0;
const constexpr int64_t OUT_H_POST_INDEX = 1;
const constexpr int64_t OUT_H_RES_INDEX = 2;
const constexpr int64_t OUT_INV_RMS_INDEX = 3;
const constexpr int64_t OUT_MM_RES_INDEX = 4;
const constexpr int64_t OUT_H_PRE_INDEX = 5;

const constexpr int64_t BSND_DIM_NUM = 4;
const constexpr int64_t TND_DIM_NUM = 3;
const constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

static void SetShape3D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2)
{
    shape->SetDimNum(3);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
}

static void SetShape2D(gert::Shape *shape, uint64_t d0, uint64_t d1)
{
    shape->SetDimNum(2);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
}

static void SetShape4D(gert::Shape *shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
{
    shape->SetDimNum(4);
    shape->SetDim(0, d0);
    shape->SetDim(1, d1);
    shape->SetDim(2, d2);
    shape->SetDim(3, d3);
}

static void SetShape1D(gert::Shape *shape, uint64_t d0)
{
    shape->SetDimNum(1);
    shape->SetDim(0, d0);
}

static void SetShapeFromX(gert::Shape *dst, const gert::Shape *src)
{
    dst->SetDimNum(src->GetDimNum());
    for (int64_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

static ge::graphStatus InferShape4MhcPre(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape MhcPre");
    const gert::Shape *xShape = context->GetDynamicInputShape(X_INDEX, 0);
    const gert::Shape *phiShape = context->GetDynamicInputShape(PHI_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, phiShape);

    int64_t phiDim = phiShape->GetDimNum();
    int64_t xDim = xShape->GetDimNum();
    OP_CHECK_IF(phiDim != 2, OP_LOGE(context->GetNodeName(), "phiShapeDim should be 2, but got %ld", phiDim),
                return GRAPH_FAILED);
    OP_CHECK_IF(
        xDim != BSND_DIM_NUM && xDim != TND_DIM_NUM,
        OP_LOGE(context->GetNodeName(), "xShapeDim should be %ld or %ld, but got %ld", BSND_DIM_NUM, TND_DIM_NUM, xDim),
        return GRAPH_FAILED);
    uint64_t matK = phiShape->GetDim(0);
    gert::Shape *outShapes[6] = {context->GetOutputShape(OUT_H_IN_INDEX),   context->GetOutputShape(OUT_H_POST_INDEX),
                                 context->GetOutputShape(OUT_H_RES_INDEX),  context->GetOutputShape(OUT_INV_RMS_INDEX),
                                 context->GetOutputShape(OUT_MM_RES_INDEX), context->GetOutputShape(OUT_H_PRE_INDEX)};
    if (xDim == BSND_DIM_NUM) {
        uint64_t b = xShape->GetDim(0), s = xShape->GetDim(1), n = xShape->GetDim(2), d = xShape->GetDim(3);
        SetShape3D(outShapes[0], b, s, d);
        SetShape3D(outShapes[1], b, s, n);
        SetShape4D(outShapes[2], b, s, n, n);
        SetShape2D(outShapes[3], b, s);
        SetShape3D(outShapes[4], b, s, matK);
        SetShape3D(outShapes[5], b, s, n);
    } else {
        uint64_t t = xShape->GetDim(0), n = xShape->GetDim(1), d = xShape->GetDim(2);
        SetShape2D(outShapes[0], t, d);
        SetShape2D(outShapes[1], t, n);
        SetShape3D(outShapes[2], t, n, n);
        SetShape1D(outShapes[3], t);
        SetShape2D(outShapes[4], t, matK);
        SetShape2D(outShapes[5], t, n);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape MhcPre");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4MhcPre(gert::InferDataTypeContext *context)
{
    const auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(OUT_H_IN_INDEX, xDtype);
    context->SetOutputDataType(OUT_H_POST_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_RES_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_INV_RMS_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_MM_RES_INDEX, DataType::DT_FLOAT);
    context->SetOutputDataType(OUT_H_PRE_INDEX, DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcPre).InferShape(InferShape4MhcPre).InferDataType(InferDataType4MhcPre);
} // namespace ops
