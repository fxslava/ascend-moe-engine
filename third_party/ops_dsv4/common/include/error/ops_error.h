/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Local stand-in for the CANN operator-dev header `error/ops_error.h`, which
 * the upstream operator repositories assume on their include path but which
 * ships with neither the CANN toolkit nor the vendored trees. Implements
 * exactly the surface this repository's op_host sources call, on top of the
 * toolkit's own op_common reporting (the same primitives the vendored
 * err/ops_err.h defines). Call-site semantics match the upstream macros; the
 * error codes differ.
 */

#ifndef DSV4_SHIM_OPS_ERROR_H
#define DSV4_SHIM_OPS_ERROR_H

#include "log/ops_log.h"

// if (cond) { log; ret; } -- the vendored call sites pass an OP_LOGE /
// OPS_REPORT_* expression as the middle argument.
#define OPS_ERR_IF(COND, LOG, RET)       \
    do {                                 \
        if (COND) {                      \
            LOG;                         \
            RET;                         \
        }                                \
    } while (0)

// OPS_REPORT_VECTOR_INNER_ERR / OPS_REPORT_CUBE_INNER_ERR are defined in
// log/ops_log.h above -- some vendored sources include only this header and
// still call them, some include only ops_log.h.

#endif  // DSV4_SHIM_OPS_ERROR_H
