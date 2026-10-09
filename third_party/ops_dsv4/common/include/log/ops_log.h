/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Local stand-in for the CANN operator-dev header `log/ops_log.h`, which the
 * upstream operator repositories assume on their include path but which ships
 * with neither the CANN toolkit nor the vendored trees. Implements exactly the
 * surface this repository's op_host sources call, on top of the toolkit's own
 * op_common reporting (the same primitive the vendored err/ops_err.h uses).
 *
 * The upstream macros accept an arbitrary "description" first argument -- the
 * call sites here pass either an op name or a gert::TilingContext* -- so the
 * shim deliberately routes everything through REPORT_INNER_ERR_MSG and does
 * not touch the char*-typed logging helpers. Errors keep full message text;
 * info-level logs are diagnostic-only and become no-ops.
 */

#ifndef DSV4_SHIM_OPS_LOG_H
#define DSV4_SHIM_OPS_LOG_H

#include "log/log.h"

#define DSV4_SHIM_OPS_LOG_MACROS

#define OPS_LOG_I(OPS_DESC, ...)                       \
    do {                                               \
        (void)(OPS_DESC); /* info: diagnostic only */  \
    } while (0)

#define OPS_LOG_D(OPS_DESC, ...)                       \
    do {                                               \
        (void)(OPS_DESC); /* debug: diagnostic only */ \
    } while (0)

#define OPS_LOG_W(OPS_DESC, ...)                       \
    do {                                               \
        (void)(OPS_DESC); /* warn: diagnostic only */  \
    } while (0)

#define OPS_LOG_E(OPS_DESC, ...)                       \
    do {                                               \
        (void)(OPS_DESC);                              \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)

// The upstream error/ops_error.h carries these two, but several vendored
// sources include only log/ops_log.h and still call them, so they live here.
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...)     \
    do {                                               \
        (void)(OPS_DESC);                              \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)

#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...)       \
    do {                                               \
        (void)(OPS_DESC);                              \
        REPORT_INNER_ERR_MSG("EZ9999", ##__VA_ARGS__); \
    } while (0)

// if (ptr == nullptr) { log; ret; } -- deliberately a BARE if, not
// do-while(0): several vendored call sites omit the trailing semicolon, which
// only parses when the macro expands to a lone if statement.
#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, RET)          \
    if ((PTR) == nullptr) {                            \
        OPS_LOG_E(OPS_DESC, "%s", "input is null");    \
        RET;                                           \
    }

// if (cond) { OPS_LOG_E(desc, msg); ret; }
#define OPS_LOG_E_IF(COND, OPS_DESC, RET, ...)         \
    do {                                               \
        if (COND) {                                    \
            OPS_LOG_E(OPS_DESC, ##__VA_ARGS__);        \
            RET;                                       \
        }                                              \
    } while (0)

#endif  // DSV4_SHIM_OPS_LOG_H
