/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

// =================================================================================================
// test_dequant_micro_kernel.cpp -- Step 1 micro-kernel: DequantSubTileVector in isolation
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2)
// Kernel Task Mode    : vector only -- no Cube, no KfcServer, no Matmul object
// =================================================================================================
//
// 1. WHAT THIS ISOLATES, AND WHY ISOLATION IS THE POINT
// -------------------------------------------------------------------------------------------------
// The end-to-end `dsv4_moe_prefill_cube` gate has never completed a trip: under
// KERNEL_TYPE_MIX_AIC_1_2 the CAModel floods `pem_lsu.cc invalid ldst addr` with a repeating cycle
// of small offsets and never terminates. That symptom is produced somewhere inside a schedule that
// stands up a KfcServer, two Matmul objects, an L1 staging ring and a UB activation plane all at
// once, so it cannot attribute the fault to any one of them.
//
// This translation unit executes exactly ONE routine of that schedule -- the production
// `DequantSubTileVector`, textually included and unmodified -- on a single vector core with no
// Cube participation whatsoever. If the MXFP4/E8M0 conversion faults, it faults here, with nothing
// else running that could be blamed. If it does not, the whole class is eliminated from the
// end-to-end triage, which is worth more than a passing number.
//
// The body under test is the shipping body. The production TU is `#include`d rather than copied,
// so this gate cannot drift from the operator it verifies; only the trip schedule (one sub-tile
// instead of the full r0/k0 lattice) differs.
//
// 2. WHY `if ASCEND_IS_AIV`, GIVEN THAT THE TASK-TYPE MACRO IS INERT HERE
// -------------------------------------------------------------------------------------------------
// `KERNEL_TASK_TYPE_DEFAULT` expands to `ENABLE_FEATURE_FOR_COMPILE`, which is EMPTY unless
// `__CHECK_FEATURE_AT_PRECOMPILE` is defined -- and the direct `bisheng --asc-aicore-lang` flow
// this repository uses never defines it. So the macro declares intent and enforces nothing, and
// whether this entry is scheduled onto a vector core alone or onto both cores of a MIX subsystem
// is not something the macro settles.
//
// Rather than depend on that, the body is guarded. `ASCEND_IS_AIV` is a compile-time predicate on
// `g_coreType`, so under a MIX compilation the cube variant of this function is empty and returns
// immediately, and under a vector-only compilation the guard is trivially true. Either way exactly
// one core touches UB -- which is the property the isolation claim rests on, and it is cheap enough
// that assuming it instead would be a false economy.
//
// 3. DIAGNOSTIC CONTRACT
// -------------------------------------------------------------------------------------------------
// A kernel that rejects bad geometry by returning early produces a ZEROED output on device, which
// is indistinguishable from a wrong answer (CLAUDE.md section 2.5). This entry therefore reports
// through a status word instead: every rejection path writes a distinct `MicroStatus` code and the
// observed operands beside it, so a bail-out reads as a named refusal on the host rather than as a
// silent zero. The bus predicates are checked at the issue site, before the first burst, so an
// illegal transfer is a reported refusal rather than an opaque simulator fault 200 s in.
//
// A second UB buffer is allocated immediately after `scale_temporary` and carries a host-poisoned
// sentinel through the routine untouched. `scale_temporary` is 32 B -- exactly the 16 `uint16_t`
// the per-row `Cast` writes, with no slack -- so a conversion that rounded its count up to a
// vector repeat would overrun it. The sentinel is the cheapest way to see that happen; TPipe
// adjacency is a bump-allocator property rather than a documented one, so this is a probe and not
// a proof.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "launch_common.hpp"

#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

namespace {

// Mirrors `moe::sim::prefill_cube` on the host side. The host states these numbers independently
// because the operator's op_kernel header is private to the operator (CLAUDE.md section 1.2), and
// this TU is the one place that can see both -- so this is where a drift becomes a compile error.
inline constexpr std::uint32_t kMicroGroupSize      = 32;
inline constexpr std::uint32_t kMicroScaleRowStride = 32;
static_assert(kMicroGroupSize == kCubeGroup,
              "the host golden MXFP4 group size has drifted from the operator");
static_assert(kMicroScaleRowStride == kCubeScaleRowStride,
              "the host golden scale row stride has drifted from the operator");

// Bytes of host-poisoned sentinel carried through the routine to detect a UB overrun past
// `scale_temporary`. One burst block times eight, so the probe covers more than any plausible
// single-repeat overshoot.
inline constexpr std::uint32_t kMicroUbSentinelBytes = 256;

inline constexpr std::uint32_t kMicroStatusWords = 8;

enum MicroStatus : std::uint32_t {
    kMicroOk                 = 0,  // the routine ran and the output tile is valid
    kMicroNullOperand        = 1,  // a GM operand was null
    kMicroShapeRejected      = 2,  // rows/cols empty, unaligned, or scales exceed the row stride
    kMicroBudgetExceeded     = 3,  // the sub-tile does not fit the operator's own UB staging
    kMicroBurstMisaligned    = 4,  // an MTE burst would violate the 32-byte rule
    kMicroStatusNotWritten   = 0xFFFFFFFFU,  // host poison; still present means the kernel never ran
};

__aicore__ inline void ReportStatus(GM_ADDR status, std::uint32_t code, std::uint32_t observed_rows,
                                    std::uint32_t observed_cols, std::uint32_t observed_bytes,
                                    const LocalTensor<std::uint32_t>& staging) {
    staging.SetValue(0, code);
    staging.SetValue(1, observed_rows);
    staging.SetValue(2, observed_cols);
    staging.SetValue(3, observed_bytes);
    staging.SetValue(4, kCubeWeightSubElems);
    staging.SetValue(5, kCubeScaleRowsMax);
    staging.SetValue(6, kCubeScaleRowStride);
    staging.SetValue(7, kTilingAbiVersion);

    GlobalTensor<std::uint32_t> status_gm;
    status_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(status), kMicroStatusWords);
    PipeBarrier<PIPE_ALL>();
    DataCopy(status_gm, staging, kMicroStatusWords);
}

}  // namespace

// `shape` carries {rows, cols, scale_row_stride, reserved}; `status` receives `MicroStatus` plus
// the operands the kernel actually observed. `ub_sentinel` is poisoned by the host, copied into UB
// before the routine and back out after it.
extern "C" __global__ __aicore__ void dsv4_prefill_cube_dequant_micro(
    GM_ADDR packed, GM_ADDR scales, GM_ADDR output, GM_ADDR shape, GM_ADDR status,
    GM_ADDR ub_sentinel) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }

        TPipe pipe;
        TQue<TPosition::VECCALC, 1> q_nibbles;
        TQue<TPosition::VECCALC, 1> q_expanded;
        TQue<TPosition::VECCALC, 1> q_scales;
        TQue<TPosition::VECCALC, 1> q_plane;
        TQue<TPosition::VECCALC, 1> q_temporary;
        TQue<TPosition::VECCALC, 1> q_ub_sentinel;
        TQue<TPosition::VECCALC, 1> q_status;

        // Allocation order matters for the sentinel probe only: it must follow `q_temporary` so
        // that a write past the 32-byte temporary lands in it. Every other size is the operator's
        // own, taken from the private header rather than restated, so this gate exercises the
        // production UB footprint and not a roomier one.
        pipe.InitBuffer(q_nibbles,     1, kCubeNibBytes);
        pipe.InitBuffer(q_expanded,    1, kCubeWideBytes);
        pipe.InitBuffer(q_scales,      1, kCubeScaleBytes);
        pipe.InitBuffer(q_plane,       1, kCubeWideBytes);
        pipe.InitBuffer(q_temporary,   1, 16 * sizeof(std::uint16_t));
        pipe.InitBuffer(q_ub_sentinel, 1, kMicroUbSentinelBytes);
        pipe.InitBuffer(q_status,      1, kMicroStatusWords * sizeof(std::uint32_t));

        LocalTensor<std::uint32_t> status_staging = q_status.AllocTensor<std::uint32_t>();

        if (unlikely(packed == nullptr || scales == nullptr || output == nullptr ||
                     shape == nullptr || status == nullptr || ub_sentinel == nullptr)) {
            ReportStatus(status, kMicroNullOperand, 0, 0, 0, status_staging);
            return;
        }

        const __gm__ std::uint32_t* dimensions = reinterpret_cast<const __gm__ std::uint32_t*>(shape);
        const std::uint32_t         rows               = dimensions[0];
        const std::uint32_t         cols               = dimensions[1];
        const std::uint32_t         scale_row_stride   = dimensions[2];

        if (unlikely(rows == 0u || cols == 0u || cols % kMicroGroupSize != 0u ||
                     scale_row_stride != kCubeScaleRowStride ||
                     cols / kMicroGroupSize > scale_row_stride)) {
            ReportStatus(status, kMicroShapeRejected, rows, cols, scale_row_stride, status_staging);
            return;
        }

        // Widened before multiplying a caller-supplied extent (CLAUDE.md section 2.7): at the
        // production sub-tile widths `rows * cols` is already 16 K, and a caller that asked for a
        // sub-tile larger than the operator stages must be refused rather than allowed to walk off
        // the end of `weight_expanded`.
        const std::uint64_t total_elements = static_cast<std::uint64_t>(rows) * cols;
        if (unlikely(total_elements > kCubeWeightSubElems || rows > kCubeScaleRowsMax)) {
            ReportStatus(status, kMicroBudgetExceeded, rows, cols,
                         static_cast<std::uint32_t>(total_elements), status_staging);
            return;
        }

        const std::uint64_t nibble_bytes = total_elements / 2u;
        const std::uint64_t scale_bytes  = static_cast<std::uint64_t>(rows) * scale_row_stride;
        const std::uint64_t output_bytes = total_elements * sizeof(bfloat16_t);
        if (unlikely(nibble_bytes % 32u != 0u || scale_bytes % 32u != 0u ||
                     output_bytes % 32u != 0u)) {
            ReportStatus(status, kMicroBurstMisaligned, rows, cols,
                         static_cast<std::uint32_t>(nibble_bytes), status_staging);
            return;
        }

        Buffers buf;
        buf.weight_nibbles  = q_nibbles.AllocTensor<fp4x2_e2m1_t>();
        buf.weight_expanded = q_expanded.AllocTensor<bfloat16_t>();
        buf.weight_scales   = q_scales.AllocTensor<std::uint8_t>();
        buf.scale_plane     = q_plane.AllocTensor<bfloat16_t>();
        buf.scale_temporary = q_temporary.AllocTensor<std::uint16_t>();

        LocalTensor<std::uint8_t> sentinel_resident = q_ub_sentinel.AllocTensor<std::uint8_t>();

        GlobalTensor<std::uint8_t> packed_gm;
        GlobalTensor<std::uint8_t> scale_gm;
        GlobalTensor<std::uint8_t> sentinel_gm;
        packed_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(packed), nibble_bytes);
        scale_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(scales), scale_bytes);
        sentinel_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(ub_sentinel),
                                    kMicroUbSentinelBytes);

        DataCopy(buf.weight_nibbles.ReinterpretCast<std::uint8_t>(), packed_gm,
                 static_cast<std::uint32_t>(nibble_bytes));
        DataCopy(buf.weight_scales, scale_gm, static_cast<std::uint32_t>(scale_bytes));
        DataCopy(sentinel_resident, sentinel_gm, kMicroUbSentinelBytes);
        SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);

        DequantSubTileVector(rows, cols, buf);

        GlobalTensor<bfloat16_t> result_gm;
        result_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output), total_elements);
        SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        DataCopy(result_gm, buf.weight_expanded, static_cast<std::uint32_t>(total_elements));
        DataCopy(sentinel_gm, sentinel_resident, kMicroUbSentinelBytes);

        ReportStatus(status, kMicroOk, rows, cols, static_cast<std::uint32_t>(total_elements),
                     status_staging);

        q_ub_sentinel.FreeTensor(sentinel_resident);
        q_temporary.FreeTensor(buf.scale_temporary);
        q_plane.FreeTensor(buf.scale_plane);
        q_scales.FreeTensor(buf.weight_scales);
        q_expanded.FreeTensor(buf.weight_expanded);
        q_nibbles.FreeTensor(buf.weight_nibbles);
        q_status.FreeTensor(status_staging);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube

extern "C" void LaunchDsv4PrefillCubeDequantMicro(void* packed, void* scales, void* output,
                                                   void* shape, void* status, void* ub_sentinel,
                                                   aclrtStream stream) {
    moe::kernels::arch35::dsv4_moe_prefill_cube::dsv4_prefill_cube_dequant_micro<<<1, 0, stream>>>(
        moe::kernels::arch35::detail::Gm(packed), moe::kernels::arch35::detail::Gm(scales),
        moe::kernels::arch35::detail::Gm(output), moe::kernels::arch35::detail::Gm(shape),
        moe::kernels::arch35::detail::Gm(status),
        moe::kernels::arch35::detail::Gm(ub_sentinel));
}
