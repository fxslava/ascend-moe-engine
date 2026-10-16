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

// =================================================================================================
// STEP 2 -- the two-slot token prefetch ring, in isolation
// =================================================================================================
//
// WHAT IS UNDER TEST, AND WHAT THE HARNESS MUST NOT DO TO IT
// -------------------------------------------------------------------------------------------------
// `AsyncLoadTokensChunk` and `PrepareTokensChunk` are called verbatim, in the production prologue /
// loop / epilogue shape, including the detail that matters: the loop issues the load for chunk
// k+1 BEFORE converting chunk k, which is the whole reason the ring has two slots.
//
// The harness has to get chunk data out to GM so the host can check it, and the obvious way to do
// that -- cast, then immediately store -- would QUIETLY INVALIDATE THE GATE. A store needs a
// `V_MTE3` fence, and that fence forces the cast of chunk k to retire before anything after it
// issues, including the next iteration's MTE2 load. The production loop has no `V_MTE2` edge
// telling MTE2 that the Vector has finished reading a slot, so an immediate store would supply, by
// accident, exactly the ordering whose absence is worth examining.
//
// So the store is DEFERRED BY ONE ITERATION. Chunk k is stored during iteration k+1, after that
// iteration has already issued its load, which keeps the Vector-read-to-MTE2-refill window open.
// The bf16 slots ping-pong, so chunk k's converted output is still intact when iteration k+1
// stores it. Token accounting for the two harness events stays 1:1 per iteration.
//
// This entry therefore reports on the ring's VALUES and its ROTATION, which is the correctness
// content of double buffering. It does not measure overlap in time -- see the host harness header.

namespace {

inline constexpr std::uint32_t kMicroMaxChunks = 32;

// Harness-only events, on the two ids the routines under test do not touch. The token ring owns
// ids 2 and 3; the dequant path owns 0 and 1 but is not running in this entry, so borrowing them
// here cannot collide. That there are exactly four ids on arch35 and this operator already spends
// all four is the reason a harness event has to be accounted for at all.
inline constexpr std::int32_t kEvMicroVToMte3 = kEvCubeVToMte3;
inline constexpr std::int32_t kEvMicroMte3ToV = kEvCubeMte2ToV;

static_assert(kEvTokensPing == 2 && kEvTokensPong == 3,
              "the host token golden assumes the ring uses event ids 2 and 3");
static_assert(kEvMicroVToMte3 != kEvTokensPing && kEvMicroVToMte3 != kEvTokensPong &&
                  kEvMicroMte3ToV != kEvTokensPing && kEvMicroMte3ToV != kEvTokensPong,
              "a harness event id must not collide with the ring under test");

// Records, per chunk, the UB address the concurrent load targeted and the UB address the cast read
// from. Taken from the tensors the production routines were actually handed, so a ring that failed
// to rotate shows up as two equal addresses rather than as a host-side restatement of `is_ping`.
__aicore__ inline void RecordChunkSlots(const LocalTensor<std::uint32_t>& trace,
                                        std::uint32_t chunk, std::uint64_t load_destination,
                                        std::uint64_t cast_source) {
    if (chunk >= kMicroMaxChunks) return;
    trace.SetValue(2u * chunk, static_cast<std::uint32_t>(load_destination));
    trace.SetValue(2u * chunk + 1u, static_cast<std::uint32_t>(cast_source));
}

__aicore__ inline void StoreChunkDeferred(const GlobalTensor<bfloat16_t>& out_gm,
                                          const LocalTensor<bfloat16_t>& chunk_bf16,
                                          std::uint32_t chunk, std::uint32_t chunk_elements) {
    WaitFlag<HardEvent::V_MTE3>(kEvMicroVToMte3);
    DataCopy(out_gm[static_cast<std::uint64_t>(chunk) * chunk_elements], chunk_bf16, chunk_elements);
    SetFlag<HardEvent::MTE3_V>(kEvMicroMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMicroMte3ToV);
}

}  // namespace

// `shape` carries {rows, hidden, chunk_width, reserved}; `trace` receives two UB addresses per
// chunk. `ub_sentinel` is poisoned by the host and carried through untouched.
extern "C" __global__ __aicore__ void dsv4_prefill_cube_tokens_micro(
    GM_ADDR tokens_in, GM_ADDR output, GM_ADDR shape, GM_ADDR status, GM_ADDR trace,
    GM_ADDR ub_sentinel) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }

        TPipe pipe;
        TQue<TPosition::VECCALC, 1> q_tokens_f32_ping;
        TQue<TPosition::VECCALC, 1> q_tokens_f32_pong;
        TQue<TPosition::VECOUT,  1> q_tokens_bf16_ping;
        TQue<TPosition::VECOUT,  1> q_tokens_bf16_pong;
        TQue<TPosition::VECCALC, 1> q_ub_sentinel;
        TQue<TPosition::VECCALC, 1> q_trace;
        TQue<TPosition::VECCALC, 1> q_status;

        // The operator's own sizes, from the private header. The sentinel follows the last ring
        // slot so a burst that overran a slot lands in it.
        pipe.InitBuffer(q_tokens_f32_ping,  1, kCubeXF32Bytes);
        pipe.InitBuffer(q_tokens_f32_pong,  1, kCubeXF32Bytes);
        pipe.InitBuffer(q_tokens_bf16_ping, 1, kCubeXSubBytes);
        pipe.InitBuffer(q_tokens_bf16_pong, 1, kCubeXSubBytes);
        pipe.InitBuffer(q_ub_sentinel,      1, kMicroUbSentinelBytes);
        pipe.InitBuffer(q_trace,            1, 2u * kMicroMaxChunks * sizeof(std::uint32_t));
        pipe.InitBuffer(q_status,           1, kMicroStatusWords * sizeof(std::uint32_t));

        LocalTensor<std::uint32_t> status_staging = q_status.AllocTensor<std::uint32_t>();
        LocalTensor<std::uint32_t> trace_staging  = q_trace.AllocTensor<std::uint32_t>();
        for (std::uint32_t i = 0; i < 2u * kMicroMaxChunks; ++i) {
            trace_staging.SetValue(i, 0xFFFFFFFFu);
        }

        if (unlikely(tokens_in == nullptr || output == nullptr || shape == nullptr ||
                     status == nullptr || trace == nullptr || ub_sentinel == nullptr)) {
            ReportStatus(status, kMicroNullOperand, 0, 0, 0, status_staging);
            return;
        }

        const __gm__ std::uint32_t* dimensions = reinterpret_cast<const __gm__ std::uint32_t*>(shape);
        const std::uint32_t         rows         = dimensions[0];
        const std::uint32_t         hidden       = dimensions[1];
        const std::uint32_t         chunk_width  = dimensions[2];

        if (unlikely(rows == 0u || rows > kCubeMTile || chunk_width != kCubeKSub ||
                     hidden == 0u || hidden % chunk_width != 0u || hidden > kCubeHiddenMax)) {
            ReportStatus(status, kMicroShapeRejected, rows, hidden, chunk_width, status_staging);
            return;
        }

        const std::uint32_t chunks = hidden / chunk_width;
        if (unlikely(chunks > kMicroMaxChunks)) {
            ReportStatus(status, kMicroBudgetExceeded, rows, hidden, chunks, status_staging);
            return;
        }

        // The burst predicates, at the issue site. `blockLen` and `srcStride` are 16-bit fields in
        // DataCopyParams, and `srcStride` grows with the FULL hidden extent -- so a wide token
        // plane is the operand that overflows them, and silently, since the narrowing is implicit
        // in the production routine's casts.
        const std::uint64_t block_length = (static_cast<std::uint64_t>(chunk_width) * 4u) / 32u;
        const std::uint64_t source_stride =
            (static_cast<std::uint64_t>(hidden - chunk_width) * 4u) / 32u;
        const std::uint64_t chunk_bytes = static_cast<std::uint64_t>(rows) * chunk_width * 2u;
        if (unlikely((static_cast<std::uint64_t>(chunk_width) * 4u) % 32u != 0u ||
                     ((static_cast<std::uint64_t>(hidden) - chunk_width) * 4u) % 32u != 0u ||
                     chunk_bytes % 32u != 0u || block_length > 0xFFFFu ||
                     source_stride > 0xFFFFu)) {
            ReportStatus(status, kMicroBurstMisaligned, rows, hidden,
                         static_cast<std::uint32_t>(source_stride), status_staging);
            return;
        }

        Geometry geom;
        geom.block_dim             = 1u;
        geom.hidden                = hidden;
        geom.inter                 = kCubeInterMax;
        geom.num_experts           = 1u;
        geom.total_tokens          = rows;
        geom.token_tile            = kCubeMTile;
        geom.inter_sub_fwd         = kCubeRSub;
        geom.hidden_sub_fwd        = chunk_width;
        geom.hidden_sub_down       = kCubeHSub;
        geom.inter_sub_down        = kCubeRSub2;
        geom.gate_up_packed_stride = 0u;
        geom.gate_up_scale_stride  = 0u;
        geom.down_packed_stride    = 0u;
        geom.down_scale_stride     = 0u;
        geom.swiglu_limit          = 0.0F;

        Buffers buf;
        buf.tokens_f32_ping      = q_tokens_f32_ping.AllocTensor<float>();
        buf.tokens_f32_pong      = q_tokens_f32_pong.AllocTensor<float>();
        buf.tokens_sub_bf16_ping = q_tokens_bf16_ping.AllocTensor<bfloat16_t>();
        buf.tokens_sub_bf16_pong = q_tokens_bf16_pong.AllocTensor<bfloat16_t>();

        LocalTensor<std::uint8_t> sentinel_resident = q_ub_sentinel.AllocTensor<std::uint8_t>();

        GlobalTensor<std::uint8_t> sentinel_gm;
        sentinel_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(ub_sentinel),
                                    kMicroUbSentinelBytes);
        DataCopy(sentinel_resident, sentinel_gm, kMicroUbSentinelBytes);

        GlobalTensor<float> tokens_in_gm;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in),
                                     static_cast<std::uint64_t>(rows) * hidden);
        GlobalTensor<bfloat16_t> out_gm;
        out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output),
                               static_cast<std::uint64_t>(chunks) * rows * chunk_width);

        const std::uint32_t chunk_elements = rows * chunk_width;
        std::uint32_t       hidden_base    = 0;
        bool                is_ping        = true;

        // Unguarded, like the production prologue: the free edge starts empty and the first fill
        // of each slot has no token to wait for.
        AsyncLoadTokensChunk(tokens_in_gm, 0u, hidden_base, rows, geom, buf.tokens_f32_ping,
                             kEvTokensPing, false);

        for (; hidden_base + chunk_width < hidden; hidden_base += chunk_width) {
            const std::uint32_t next_hidden_base = hidden_base + chunk_width;
            const std::uint32_t chunk            = hidden_base / chunk_width;

            auto& curr_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
            auto& next_tokens_f32  = is_ping ? buf.tokens_f32_pong      : buf.tokens_f32_ping;
            auto& curr_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
            auto& prev_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_pong : buf.tokens_sub_bf16_ping;
            const int curr_event   = is_ping ? kEvTokensPing            : kEvTokensPong;
            const int next_event   = is_ping ? kEvTokensPong            : kEvTokensPing;

            AsyncLoadTokensChunk(tokens_in_gm, 0u, next_hidden_base, rows, geom, next_tokens_f32,
                                 next_event, chunk + 1u >= kTokenRingSlots);

            // The deferred store sits HERE, after this iteration has issued its load, so the
            // fence it carries cannot stand in for the missing V-to-MTE2 edge.
            if (chunk > 0u) {
                StoreChunkDeferred(out_gm, prev_tokens_bf16, chunk - 1u, chunk_elements);
            }

            PrepareTokensChunk(curr_tokens_bf16, curr_tokens_f32, rows, chunk_width, curr_event);
            SetFlag<HardEvent::V_MTE3>(kEvMicroVToMte3);

            RecordChunkSlots(trace_staging, chunk, next_tokens_f32.GetPhyAddr(),
                             curr_tokens_f32.GetPhyAddr());

            is_ping = !is_ping;
        }

        auto& final_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
        auto& final_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
        auto& penultimate_bf16  = is_ping ? buf.tokens_sub_bf16_pong : buf.tokens_sub_bf16_ping;
        const int final_event   = is_ping ? kEvTokensPing            : kEvTokensPong;
        const std::uint32_t final_chunk = hidden_base / chunk_width;

        if (final_chunk > 0u) {
            StoreChunkDeferred(out_gm, penultimate_bf16, final_chunk - 1u, chunk_elements);
        }

        PrepareTokensChunk(final_tokens_bf16, final_tokens_f32, rows, chunk_width, final_event);
        SetFlag<HardEvent::V_MTE3>(kEvMicroVToMte3);
        RecordChunkSlots(trace_staging, final_chunk, 0xFFFFFFFFu, final_tokens_f32.GetPhyAddr());
        StoreChunkDeferred(out_gm, final_tokens_bf16, final_chunk, chunk_elements);

        // The ring leaves one buffer-free token per occupied slot. Draining it here is not harness
        // tidiness: a leaked flag outlives the launch, so leaving it would make the NEXT launch in
        // this binary fail for a reason that has nothing to do with the routine it tests.
        DrainTokenRing(chunks);

        DataCopy(sentinel_gm, sentinel_resident, kMicroUbSentinelBytes);

        GlobalTensor<std::uint32_t> trace_gm;
        trace_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint32_t*>(trace),
                                 2u * kMicroMaxChunks);
        PipeBarrier<PIPE_ALL>();
        DataCopy(trace_gm, trace_staging, 2u * kMicroMaxChunks);

        ReportStatus(status, kMicroOk, rows, hidden, chunks, status_staging);

        q_ub_sentinel.FreeTensor(sentinel_resident);
        q_tokens_bf16_pong.FreeTensor(buf.tokens_sub_bf16_pong);
        q_tokens_bf16_ping.FreeTensor(buf.tokens_sub_bf16_ping);
        q_tokens_f32_pong.FreeTensor(buf.tokens_f32_pong);
        q_tokens_f32_ping.FreeTensor(buf.tokens_f32_ping);
        q_trace.FreeTensor(trace_staging);
        q_status.FreeTensor(status_staging);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube

extern "C" void LaunchDsv4PrefillCubeTokensMicro(void* tokens_in, void* output, void* shape,
                                                  void* status, void* trace, void* ub_sentinel,
                                                  aclrtStream stream) {
    moe::kernels::arch35::dsv4_moe_prefill_cube::dsv4_prefill_cube_tokens_micro<<<1, 0, stream>>>(
        moe::kernels::arch35::detail::Gm(tokens_in), moe::kernels::arch35::detail::Gm(output),
        moe::kernels::arch35::detail::Gm(shape), moe::kernels::arch35::detail::Gm(status),
        moe::kernels::arch35::detail::Gm(trace),
        moe::kernels::arch35::detail::Gm(ub_sentinel));
}

extern "C" void LaunchDsv4PrefillCubeDequantMicro(void* packed, void* scales, void* output,
                                                   void* shape, void* status, void* ub_sentinel,
                                                   aclrtStream stream) {
    moe::kernels::arch35::dsv4_moe_prefill_cube::dsv4_prefill_cube_dequant_micro<<<1, 0, stream>>>(
        moe::kernels::arch35::detail::Gm(packed), moe::kernels::arch35::detail::Gm(scales),
        moe::kernels::arch35::detail::Gm(output), moe::kernels::arch35::detail::Gm(shape),
        moe::kernels::arch35::detail::Gm(status),
        moe::kernels::arch35::detail::Gm(ub_sentinel));
}
