/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Compatibility definition (toolkit stub) for Ops::Base::ToString(const gert::Shape&).
 *
 * op_common/log/log.h declares this helper OPBASE_API and the vendored op_host
 * sources reach it through logging inlines, but this toolkit build defines it
 * in no library (the 9.2 host toolkit exports it from libopapi.so; this
 * 9.1.0-950 image does not export it at all), so any probe library that
 * references it cannot be dlopened by op_build. This file is that definition,
 * formatted the way the operator logs spell shapes.
 */

#include <string>

#include "exe_graph/runtime/shape.h"
#include "log/log.h"

namespace Ops {
namespace Base {
std::string ToString(const gert::Shape &shape)
{
    std::string out = "[";
    const size_t dimNum = shape.GetDimNum();
    for (size_t index = 0; index < dimNum; ++index) {
        if (index != 0) {
            out += ", ";
        }
        out += std::to_string(shape.GetDim(index));
    }
    out += "]";
    return out;
}
}  // namespace Base
}  // namespace Ops
