/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY. Pure payloads for the dsv4_moe_prefill_cube trace probe.
// Invariant: no CrossCore*, PipeBarrier, SetFlag, WaitFlag, kCubeFlag*, HardEvent.
// Must be included AFTER op_kernel/dsv4_moe_prefill_cube_kernel.cpp: the production
// types live in that file's anonymous namespace, reached by reopening the namespace.

#pragma once

#include "kernel_operator.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

// ---------------------------------------------------------------------------
// Microstep schedule
// ---------------------------------------------------------------------------

enum class StepStage : std::uint8_t {
    Prologue,   // Warmup: DMA in flight, no Dequant, no Cube Mmad
    Steady,     // Full pipeline: DMA(s), Dequant(s-1), Mmad(s-1)
    Epilogue    // Drain: DMA done, final Dequant and Mmad retire
};

/// Every buffer index and predicate of microstep `s`, derived from `s` alone.
struct StepSlotMapping {
    std::uint32_t step_no;
    std::uint32_t slot_dma;      // s % 2
    std::uint32_t slot_dequant;  // (s > 0) ? ((s - 1) % 2) : 0
    std::uint32_t slot_l1;       // (s > 0) ? ((s - 1) % 2) : 0
    std::uint32_t tile_dma;      // s
    std::uint32_t tile_mmad;     // (s > 0) ? (s - 1) : 0

    bool dma_active;
    bool dequant_active;
    bool mmad_active;
    bool is_first_mmad;
    bool is_last_mmad;

    StepStage stage;             // narration only; classifies the three predicates
};

__aicore__ inline StepSlotMapping ResolveStepSlots(std::uint32_t s, std::uint32_t total_steps) {
    StepSlotMapping m{};
    m.step_no       = s;
    m.slot_dma      = s % kCubeRingSlots;
    m.slot_dequant  = (s > 0u) ? ((s - 1u) % kCubeRingSlots) : 0u;
    m.slot_l1       = m.slot_dequant;
    m.tile_dma      = s;
    m.tile_mmad     = (s > 0u) ? (s - 1u) : 0u;

    m.dma_active     = (s < total_steps);
    m.dequant_active = (s >= 1u && s <= total_steps);
    m.mmad_active    = m.dequant_active;
    m.is_first_mmad  = (s == 1u);
    m.is_last_mmad   = (s == total_steps);

    m.stage = !m.dequant_active ? StepStage::Prologue
            : (m.dma_active ? StepStage::Steady : StepStage::Epilogue);
    return m;
}

/// First microstep at which an L1 slot is REUSED, so the first needing a WAR guard.
/// Slot v is first written at s = v + 1, so the second write of any slot is here.
/// Guarding below this waits for a release the Cube only issues later in the SAME
/// microstep, against a Cube waiting on the StepReady this AIV has not reached: deadlock.
inline constexpr std::uint32_t kCubeFirstL1Reuse = 1u + kCubeRingSlots;

// ---------------------------------------------------------------------------
// Per-core contexts
// ---------------------------------------------------------------------------

/// AIV state. Reference members alias the caller's buffers; GM bases are unoffset
/// and the accessors apply the per-expert stride, widening first (CLAUDE.md 2.7).
struct AivStepContext {
    const Geometry&            geom;
    const AivBuffers&          buf;
    AivRingState&              ring;
    const GlobalTensor<float>& tokens_in;
    const GlobalTensor<float>& tokens_out;
    TBuf<TPosition::A1> (&l1a)[kCubeRingSlots];
    TBuf<TPosition::B1> (&l1w)[kCubeRingSlots][kCubeWeightStreams];
    TBuf<TPosition::A1> (&l1a2)[kCubeRingSlots];
    TBuf<TPosition::B1> (&l1w2)[kCubeRingSlots];
    GM_ADDR gate_packed_base;
    GM_ADDR gate_scales_base;
    GM_ADDR up_packed_base;
    GM_ADDR up_scales_base;
    GM_ADDR down_packed_base;
    GM_ADDR down_scales_base;
    GM_ADDR lut;
    std::uint32_t sub_idx;
    std::uint32_t row_base;          // sub_idx * kCubeMHalf
    std::uint32_t token_row_start;   // per admitted tile
    std::uint32_t valid_rows;
    std::uint32_t store_rows;
    std::uint32_t expert_id;
    std::uint32_t step_no;           // monotonic across blocks and phases

    __aicore__ inline __gm__ std::uint8_t* GatePacked() const {
        return Bytes(gate_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* GateScales() const {
        return Bytes(gate_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_scale_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* UpPacked() const {
        return Bytes(up_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* UpScales() const {
        return Bytes(up_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_scale_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* DownPacked() const {
        return Bytes(down_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.down_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* DownScales() const {
        return Bytes(down_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.down_scale_stride;
    }
};

/// Cube state. L0 buffers are address-space-qualified POINTERS, never dereferenced
/// here; every use goes through Te::MakeMemPtr inside the production helpers.
struct CubeStepContext {
    const Geometry& geom;
    TBuf<TPosition::A1> (&l1a)[kCubeRingSlots];
    TBuf<TPosition::B1> (&l1w)[kCubeRingSlots][kCubeWeightStreams];
    TBuf<TPosition::A1> (&l1a2)[kCubeRingSlots];
    TBuf<TPosition::B1> (&l1w2)[kCubeRingSlots];
    __ca__ bfloat16_t* l0a;
    __cb__ bfloat16_t* l0b_gate;
    __cb__ bfloat16_t* l0b_up;
    __cc__ float*      l0c_gate;
    __cc__ float*      l0c_up;
    GM_ADDR lut;
    std::uint32_t expert_id;
    std::uint32_t step_no;
};

/// Refresh the per-tile fields, or report the tile is not this expert's.
__aicore__ inline bool AivAdmitTile(AivStepContext& ctx, std::uint32_t tile_idx) {
    const std::uint32_t token_row_start = tile_idx * ctx.geom.token_tile;
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, token_row_start, ctx.geom.num_experts, expert_id)) {
        return false;
    }
    const std::uint32_t remaining  = ctx.geom.total_tokens - token_row_start;
    const std::uint32_t valid_rows = remaining < ctx.geom.token_tile
                                         ? remaining : ctx.geom.token_tile;
    const std::uint32_t owned = valid_rows > ctx.row_base ? (valid_rows - ctx.row_base) : 0u;

    ctx.token_row_start = token_row_start;
    ctx.valid_rows      = valid_rows;
    ctx.store_rows      = owned < kCubeMHalf ? owned : kCubeMHalf;
    ctx.expert_id       = expert_id;
    return true;
}

__aicore__ inline bool CubeAdmitTile(CubeStepContext& ctx, std::uint32_t tile_idx) {
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, tile_idx * ctx.geom.token_tile,
                      ctx.geom.num_experts, expert_id)) {
        return false;
    }
    ctx.expert_id = expert_id;
    return true;
}

// ---------------------------------------------------------------------------
// AIV payloads, phase 1
// ---------------------------------------------------------------------------

/// GM -> UB slot_dma: token half plus both weight halves of tile_dma.
/// Leaves ONE MTE2_V token outstanding (the id is hard-coded in op_kernel/); the
/// caller consumes it one microstep later.
__aicore__ inline void PayloadAiv_DmaInFwd(
    const AivStepContext& ctx, const StepSlotMapping& m, std::uint32_t inter_base) {
    AivPrefetchFwdStep(ctx.tokens_in, ctx.token_row_start,
                       ctx.GatePacked(), ctx.GateScales(), ctx.UpPacked(), ctx.UpScales(),
                       inter_base, m.tile_dma * kCubeKSub, ctx.valid_rows,
                       ctx.sub_idx, ctx.geom, ctx.buf, m.slot_dma, ctx.ring);
}

/// FP4 -> BF16 on both weight streams, then the token half F32 -> BF16.
/// event_id = -1 on both: the slot's token is consumed by the caller, not here.
__aicore__ inline void PayloadAiv_DequantAndCast(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[m.slot_dequant][0], -1);
    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[m.slot_dequant][1], -1);
    Cast(ctx.buf.x_bf16[m.slot_dequant], ctx.buf.x_f32[m.slot_dequant],
         RoundMode::CAST_RINT, kCubeMTile * kCubeKHalf);
}

/// UB -> L1 slot_l1: A, Wg, Wu. col_base is this AIV's contraction half.
__aicore__ inline void PayloadAiv_PushL1(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    const std::uint32_t col_base = ctx.sub_idx * kCubeKHalf;
    PushUbTileToL1Nz(ctx.l1a[m.slot_l1].Get<bfloat16_t>(), ctx.buf.x_bf16[m.slot_dequant],
                     kCubeMTile, kCubeKHalf, kCubeKHalf, kCubeMTile, col_base, 0);
    PushUbTileToL1Nz(ctx.l1w[m.slot_l1][0].Get<bfloat16_t>(),
                     ctx.buf.w[m.slot_dequant][0].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
    PushUbTileToL1Nz(ctx.l1w[m.slot_l1][1].Get<bfloat16_t>(),
                     ctx.buf.w[m.slot_dequant][1].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
}

/// Gate pre-activation out of the landing zone, clamped ABOVE only.
__aicore__ inline void PayloadAiv_EpilogueGate(const AivStepContext& ctx) {
    Mins(ctx.buf.gate_acc, ctx.buf.landing, ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// Up pre-activation, lower clamp; this is also the move OUT of the landing zone.
__aicore__ inline void PayloadAiv_EpilogueUp(const AivStepContext& ctx) {
    Maxs(ctx.buf.up_acc, ctx.buf.landing, -ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// Up pre-activation, upper clamp; reads up_acc, not the landing zone.
__aicore__ inline void PayloadAiv_EpilogueUpClamp(const AivStepContext& ctx) {
    Mins(ctx.buf.up_acc, ctx.buf.up_acc, ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// y = Silu(gate) * up -> BF16 -> this AIV's half of the resident plane.
__aicore__ inline void PayloadAiv_ApplySwiGLU(
    const AivStepContext& ctx, std::uint32_t inter_base) {
    AivApplySwiGLU(ctx.buf, inter_base, ctx.geom);
}

// ---------------------------------------------------------------------------
// AIV payloads, phase 2
// ---------------------------------------------------------------------------

/// GM -> UB slot_dma: one weight stream. No token burst; A comes from the plane.
__aicore__ inline void PayloadAiv_DmaInDown(
    const AivStepContext& ctx, const StepSlotMapping& m, std::uint32_t hidden_base) {
    AivPrefetchDownStep(ctx.DownPacked(), ctx.DownScales(),
                        hidden_base, m.tile_dma * kCubeRSub2,
                        ctx.sub_idx, ctx.geom, ctx.buf, m.slot_dma, ctx.ring);
}

__aicore__ inline void PayloadAiv_DequantDown(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    DequantSubTileVector(kCubeHSub, kCubeRHalf2, ctx.buf.w[m.slot_dequant][0], -1);
}

/// UB -> L1 slot_l1. The two pushes split on DIFFERENT axes: the weight half is a
/// contraction half (C0-block range), the token half a token-row range of every block.
__aicore__ inline void PayloadAiv_PushL1Down(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    PushUbTileToL1Nz(ctx.l1w2[m.slot_l1].Get<bfloat16_t>(),
                     ctx.buf.w[m.slot_dequant][0].weight_expanded,
                     kCubeHSub, kCubeRHalf2, kCubeRHalf2,
                     kCubeHSub, ctx.sub_idx * kCubeRHalf2, 0);
    PushUbTileToL1Nz(ctx.l1a2[m.slot_l1].Get<bfloat16_t>(),
                     ctx.buf.act[m.tile_mmad * kCubeRSub2],
                     kCubeMHalf, kCubeRSub2, ctx.geom.inter,
                     kCubeMTile, 0, ctx.sub_idx * kCubeMHalf);
}

/// This AIV's M/2 rows of one hidden block, landing zone -> GM. store_rows == 0 when
/// the tile's tail does not reach this half, and a zero A row produced a zero C row.
__aicore__ inline void PayloadAiv_StoreLandingPlane(
    const AivStepContext& ctx, std::uint32_t hidden_base) {
    if (ctx.store_rows == 0u) {
        return;
    }
    DataCopyParams out_burst;
    out_burst.blockCount = static_cast<std::uint16_t>(ctx.store_rows);
    out_burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
    out_burst.srcStride  = 0;
    out_burst.dstStride  = static_cast<std::uint16_t>(((ctx.geom.hidden - kCubeHSub) * 4u) / 32u);
    DataCopy(ctx.tokens_out[(static_cast<std::uint64_t>(ctx.token_row_start + ctx.row_base) *
                             ctx.geom.hidden) + hidden_base],
             ctx.buf.landing, out_burst);
}

// ---------------------------------------------------------------------------
// Cube payloads
// ---------------------------------------------------------------------------

__aicore__ inline void PayloadCube_LoadL0(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1a[m.slot_l1].Get<bfloat16_t>(), kCubeMTile, kCubeKSub);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1w[m.slot_l1][0].Get<bfloat16_t>(),
                         kCubeKSub, kCubeRSub);
    CubeLoadWeightsToL0B(ctx.l0b_up, ctx.l1w[m.slot_l1][1].Get<bfloat16_t>(),
                         kCubeKSub, kCubeRSub);
}

__aicore__ inline void PayloadCube_LoadL0Down(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1a2[m.slot_l1].Get<bfloat16_t>(), kCubeMTile, kCubeRSub2);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1w2[m.slot_l1].Get<bfloat16_t>(),
                         kCubeRSub2, kCubeHSub);
}

/// No SetFlag/WaitFlag<MTE1_M>: the caller's PipeBarrier<PIPE_ALL> after the L0 load
/// carries that RAW edge. CubeCloseAccumulationStep keeps the K-axis interlock, which
/// is a hardware contract below (m/16)*(n/16) < 10 and is NOT subsumed by that barrier.
__aicore__ inline void PayloadCube_MmadAccumulate(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeRSub, kCubeKSub, m.is_first_mmad);
    CubeMmadAccumulate(ctx.l0c_up, ctx.l0a, ctx.l0b_up,
                       kCubeMTile, kCubeRSub, kCubeKSub, m.is_first_mmad);
    CubeCloseAccumulationStep();
}

__aicore__ inline void PayloadCube_MmadDown(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeHSub, kCubeRSub2, m.is_first_mmad);
    CubeCloseAccumulationStep();
}

/// L0C -> UB offset 0 in EACH AIV's own address space, full M (CubeL0c2UbTraitDualDstM).
__aicore__ inline void PayloadCube_FixpipeMoveOut(
    __cc__ float* l0c_mem, std::uint32_t extent_m, std::uint32_t extent_n) {
    auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(extent_m, extent_n));
    auto ub  = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kCubeLandingUbOffset)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn>(extent_m, extent_n));
    Te::FixpipeParams fixParams;
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(fixParams), ub, l0C);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
