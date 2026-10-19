/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY. Progress-trace probe for dsv4_moe_prefill_cube: a lockstep microstep FSM
// over the pure payloads in dsv4_moe_prefill_cube_common.h, with a device printf at
// every cross-core edge. All synchronization lives here; op_kernel/ is untouched.
//
// One counter s = 0..N drives a one-deep pipeline: DMA tile s into UB slot s%2 while
// the Vector/MTE3/Cube work tile s-1 out of slot (s-1)%2. Every microstep ends in
// PipeBarrier<PIPE_ALL>, so the overlap is in buffer ownership only -- which is what
// keeps the records orderable and is what the production kernel must NOT do.
//
// Flag budget per block, N = total_steps:
//   MTE2_V kEvCubeSlotFull[slot]  N raised by DMA (s=0..N-1), N consumed by the
//                                 dequant (s=1..N), each in the next microstep.
//   StepReady + slot (mode 2)     N set per AIV vs N AIC waits (mode 2 counts both
//                                 AIVs setting as one).
//   StepFree + slot (mode 4)      N released; max(0, N-kCubeRingSlots) taken by the
//                                 WAR guard, the rest by SyncAiv_DrainStepFreeTail.
//   LandingFree / Product         phase 1: 3 sets per AIV vs 3 AIC waits, 2 Product
//                                 releases vs 2 waits per AIV. Phase 2: one of each.
//
// The host driver prints PRODUCTION's conditional StepFree count; it will not match
// this probe's N-per-block and asserts nothing. Never run on hardware (CLAUDE.md 5.7).

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils/debug/asc_printf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

// Order is load-bearing: the payload header is a textual continuation of this .cpp.
#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"
#include "dsv4_moe_prefill_cube_common.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

namespace {

// Per-block GM ring buffers are flushed independently, so stream order is not time
// order. GetSystemCycle() stamps every record so the three cores can be merged and
// sorted; the low 32 bits suffice for a ~1e6-cycle run.
#define DSV4_TRACE(fmt, ...)                                                   \
    __asc_aicore::printf("@%u " fmt,                                           \
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(GetSystemCycle())\
                                   & 0xFFFFFFFFull),                           \
        ##__VA_ARGS__)

}  // namespace

// =================================================================================
// Sync primitives. Every cross-core edge in this file is one of these.
// A mode 2/4 wait blocks ONLY the pipe it names, so each is drained to PIPE_ALL --
// otherwise the scalar unit walks past it and the "GOT" record is a lie.
// =================================================================================

/// RAW on L1, AIV -> AIC, mode 2. Raised when MTE3 drains this AIV's pushes.
__aicore__ inline void SyncAiv_SignalL1Ready(std::uint32_t s, std::uint32_t slot) {
    CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
        static_cast<uint16_t>(kCubeFlagStepReady + slot));
    DSV4_TRACE("V s=%u SET StepReady slot=%u\n", s, slot);
}

/// RAW on L1, AIC side. ONE wait for both AIVs: mode 2 counts them as one setter.
__aicore__ inline void SyncCube_WaitL1Ready(std::uint32_t s, std::uint32_t slot) {
    DSV4_TRACE("C s=%u WAIT StepReady slot=%u\n", s, slot);
    CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
        static_cast<uint16_t>(kCubeFlagStepReady + slot));
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C s=%u GOT StepReady slot=%u\n", s, slot);
}

/// WAR on L1, AIC -> AIV, mode 4: one dedicated channel per AIV (base + sub * stride).
__aicore__ inline void SyncCube_ReleaseL1(std::uint32_t s, std::uint32_t slot) {
    CubeReleaseToAllAivs<PIPE_MTE1>(static_cast<uint16_t>(kCubeFlagStepFree + slot));
    DSV4_TRACE("C s=%u SET StepFree slot=%u\n", s, slot);
}

/// WAR on L1, AIV side. Scoped to PIPE_MTE3: what it gates is this AIV's push.
__aicore__ inline void SyncAiv_WaitL1Free(std::uint32_t s, std::uint32_t slot) {
    DSV4_TRACE("V s=%u WAIT StepFree slot=%u\n", s, slot);
    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
        static_cast<uint16_t>(kCubeFlagStepFree + slot));
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s=%u GOT StepFree slot=%u\n", s, slot);
}

/// CONSUME, not fence: the previous microstep's barrier already ordered MTE2 vs V.
/// Mandatory anyway -- an intra-core flag is BINARY, so a token left raised IS the
/// `execute_set_flag already has same set_flag!` on the next lap through this slot.
__aicore__ inline void SyncAiv_ConsumeDmaToken(std::uint32_t s, std::uint32_t slot) {
    WaitFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[slot]);
    DSV4_TRACE("V s=%u took SlotFull slot=%u\n", s, slot);
}

/// Drain the StepFree releases the in-loop WAR guard could not reach, so no counter
/// drifts into the next block. Release j (j = 1..N) carries slot (j-1) % slots; the
/// guard consumed j = 1..N-slots, so drain j = max(1, N-slots+1)..N.
__aicore__ inline void SyncAiv_DrainStepFreeTail(std::uint32_t total_steps) {
    const std::uint32_t first =
        total_steps > kCubeRingSlots ? (total_steps - kCubeRingSlots + 1u) : 1u;
    for (std::uint32_t j = first; j <= total_steps; ++j) {
        const std::uint32_t slot = (j - 1u) % kCubeRingSlots;
        DSV4_TRACE("V tail WAIT StepFree j=%u slot=%u\n", j, slot);
        CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
            static_cast<uint16_t>(kCubeFlagStepFree + slot));
        PipeBarrier<PIPE_ALL>();
        DSV4_TRACE("V tail GOT StepFree j=%u slot=%u\n", j, slot);
    }
}

/// L0C -> both AIVs' UBs, then hand over and wait for the acknowledgement. Inlined
/// rather than one helper so the trace can print across all three edges.
__aicore__ inline void SyncCube_FixpipeHandover(
    __cc__ float* l0c_mem, std::uint32_t extent_m, std::uint32_t extent_n,
    std::uint32_t tag) {
    DSV4_TRACE("C fix%u begin n=%u\n", tag, extent_n);
    PayloadCube_FixpipeMoveOut(l0c_mem, extent_m, extent_n);
    PipeBarrier<PIPE_FIX>();   // drain before Product releases the AIVs onto these rows
    PipeBarrier<PIPE_ALL>();
    CubeReleaseToAllAivs<PIPE_FIX>(kCubeFlagProduct);
    DSV4_TRACE("C fix%u SET Product x%u\n", tag, (unsigned)kCubeAivPerAic);
    // PIPE_FIX, because what this gates is the NEXT Fixpipe write into the landing zone.
    CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C fix%u GOT LandingFree\n", tag);
}

// =================================================================================
// Unified microstep dispatcher. Both contexts, one per core; ASCEND_IS_* discards
// the other core's half at compile time, so the caller passes nullptr for it.
// =================================================================================

/// Phase 1 microstep: AIV guard/DMA/dequant/push/signal, AIC wait/load/release/Mmad.
__aicore__ inline void ExecuteMicroStepLockstep(
    AivStepContext* aiv, CubeStepContext* cube,
    std::uint32_t s, std::uint32_t total_steps, std::uint32_t inter_base) {
    const StepSlotMapping m = ResolveStepSlots(s, total_steps);

    // --- 1. AIV WAR guard on L1 (mode 4, AIC -> AIV) -----------------------------
    if ASCEND_IS_AIV {
        ++aiv->step_no;
        DSV4_TRACE("[AIV%u] Step %u s=%u ib=%u st=%u dma=T%u/%u dq=T%u/%u l1=%u\n",
                   aiv->sub_idx, aiv->step_no, s, inter_base, (unsigned)m.stage,
                   m.tile_dma, m.slot_dma, m.tile_mmad, m.slot_dequant, m.slot_l1);
        if (m.dequant_active && s >= kCubeFirstL1Reuse) {
            SyncAiv_WaitL1Free(s, m.slot_l1);
        }
    }

    // --- 2. Action payloads ------------------------------------------------------
    if ASCEND_IS_AIV {
        if (m.dma_active) {
            PayloadAiv_DmaInFwd(*aiv, m, inter_base);
        }
        if (m.dequant_active) {
            SyncAiv_ConsumeDmaToken(s, m.slot_dequant);
            PayloadAiv_DequantAndCast(*aiv, m);
            PipeBarrier<PIPE_ALL>();
            PayloadAiv_PushL1(*aiv, m);
            PipeBarrier<PIPE_ALL>();
        }
    }

    // --- 3. RAW handshake on L1 (mode 2, AIV -> AIC) -----------------------------
    if ASCEND_IS_AIV {
        if (m.dequant_active) {
            SyncAiv_SignalL1Ready(s, m.slot_l1);
        }
    }
    if ASCEND_IS_AIC {
        ++cube->step_no;
        if (m.mmad_active) {
            DSV4_TRACE("[STEP%u Begin]\n", cube->step_no);
            DSV4_TRACE("[AIC0] Step %u s=%u ib=%u T%u/%u gemm(x*Wg,x*Wu) init=%u\n",
                       cube->step_no, s, inter_base, m.tile_mmad, m.slot_l1,
                       (unsigned)m.is_first_mmad);
            SyncCube_WaitL1Ready(s, m.slot_l1);
        }
    }

    // --- 4. Cube load, L1 release (mode 4), Mmad ---------------------------------
    if ASCEND_IS_AIC {
        if (m.mmad_active) {
            PayloadCube_LoadL0(*cube, m);
            PipeBarrier<PIPE_ALL>();   // RAW MTE1 -> M, in place of SetFlag<MTE1_M>
            SyncCube_ReleaseL1(s, m.slot_l1);
            PayloadCube_MmadAccumulate(*cube, m);
            PipeBarrier<PIPE_ALL>();
            DSV4_TRACE("[STEP%u Completed]\n", cube->step_no);
        }
    }

    // Whole-core drain: makes the deferred MTE2_V token safe (raised at s, consumed
    // at s+1, raised again at s+2) and every record above true of the whole core.
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
    }
}

/// Phase 2 microstep: one weight stream, A read out of the resident plane.
__aicore__ inline void ExecuteMicroStepLockstepDown(
    AivStepContext* aiv, CubeStepContext* cube,
    std::uint32_t s, std::uint32_t total_steps, std::uint32_t hidden_base) {
    const StepSlotMapping m = ResolveStepSlots(s, total_steps);

    if ASCEND_IS_AIV {
        ++aiv->step_no;
        DSV4_TRACE("[AIV%u] Step %u s=%u hb=%u st=%u dma=T%u/%u dq=T%u/%u l1=%u\n",
                   aiv->sub_idx, aiv->step_no, s, hidden_base, (unsigned)m.stage,
                   m.tile_dma, m.slot_dma, m.tile_mmad, m.slot_dequant, m.slot_l1);
        if (m.dequant_active && s >= kCubeFirstL1Reuse) {
            SyncAiv_WaitL1Free(s, m.slot_l1);
        }
    }

    if ASCEND_IS_AIV {
        if (m.dma_active) {
            PayloadAiv_DmaInDown(*aiv, m, hidden_base);
        }
        if (m.dequant_active) {
            SyncAiv_ConsumeDmaToken(s, m.slot_dequant);
            PayloadAiv_DequantDown(*aiv, m);
            PipeBarrier<PIPE_ALL>();
            PayloadAiv_PushL1Down(*aiv, m);
            PipeBarrier<PIPE_ALL>();
        }
    }

    if ASCEND_IS_AIV {
        if (m.dequant_active) {
            SyncAiv_SignalL1Ready(s, m.slot_l1);
        }
    }
    if ASCEND_IS_AIC {
        ++cube->step_no;
        if (m.mmad_active) {
            DSV4_TRACE("[STEP%u Begin]\n", cube->step_no);
            DSV4_TRACE("[AIC0] Step %u s=%u hb=%u T%u/%u gemm(act*Wd) init=%u\n",
                       cube->step_no, s, hidden_base, m.tile_mmad, m.slot_l1,
                       (unsigned)m.is_first_mmad);
            SyncCube_WaitL1Ready(s, m.slot_l1);
        }
    }

    if ASCEND_IS_AIC {
        if (m.mmad_active) {
            PayloadCube_LoadL0Down(*cube, m);
            PipeBarrier<PIPE_ALL>();
            SyncCube_ReleaseL1(s, m.slot_l1);
            PayloadCube_MmadDown(*cube, m);
            PipeBarrier<PIPE_ALL>();
            DSV4_TRACE("[STEP%u Completed]\n", cube->step_no);
        }
    }

    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
    }
}

// =================================================================================
// Block epilogues and the phase rendezvous.
// Phase 1 orders the two handovers (gate then up) so the AIVs consume the two
// Products in that order; swapping them is a wrong answer, not a hang. The THIRD
// LandingFree is the rendezvous that stops the Cube leaving the block early.
// =================================================================================

__aicore__ inline void ExecutePhase1Epilogue(
    AivStepContext* aiv, CubeStepContext* cube, std::uint32_t inter_base) {
    if ASCEND_IS_AIV {
        DSV4_TRACE("V p1 WAIT Product gate ib=%u\n", inter_base);
        CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
        PipeBarrier<PIPE_ALL>();
        PayloadAiv_EpilogueGate(*aiv);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
        DSV4_TRACE("V p1 SET LandingFree gate\n");

        DSV4_TRACE("V p1 WAIT Product up ib=%u\n", inter_base);
        CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
        PipeBarrier<PIPE_ALL>();
        PayloadAiv_EpilogueUp(*aiv);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
        DSV4_TRACE("V p1 SET LandingFree up\n");
        PayloadAiv_EpilogueUpClamp(*aiv);
        PipeBarrier<PIPE_ALL>();

        PayloadAiv_ApplySwiGLU(*aiv, inter_base);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
        DSV4_TRACE("V p1 SET BlockDone ib=%u\n", inter_base);
    }

    if ASCEND_IS_AIC {
        SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);   // contraction done, Fixpipe may read L0C
        WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);
        PipeBarrier<PIPE_ALL>();
        SyncCube_FixpipeHandover(cube->l0c_gate, kCubeMTile, kCubeRSub, 1u);
        SyncCube_FixpipeHandover(cube->l0c_up,   kCubeMTile, kCubeRSub, 2u);
        DSV4_TRACE("C p1 WAIT BlockDone ib=%u\n", inter_base);
        CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(kCubeFlagLandingFree);
        PipeBarrier<PIPE_ALL>();
        DSV4_TRACE("C p1 GOT BlockDone ib=%u\n", inter_base);
    }
}

/// Phase 2 needs no extra rendezvous flag: the GM store is the AIV's last act, so
/// draining it before the single LandingFree makes that flag the rendezvous.
__aicore__ inline void ExecutePhase2Epilogue(
    AivStepContext* aiv, CubeStepContext* cube, std::uint32_t hidden_base) {
    if ASCEND_IS_AIV {
        DSV4_TRACE("V p2 WAIT Product hb=%u\n", hidden_base);
        CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStore>(kCubeFlagProduct);
        PipeBarrier<PIPE_ALL>();
        PayloadAiv_StoreLandingPlane(*aiv, hidden_base);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
        DSV4_TRACE("V p2 SET BlockDone hb=%u rows=%u\n", hidden_base, aiv->store_rows);
    }

    if ASCEND_IS_AIC {
        SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
        WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);
        PipeBarrier<PIPE_ALL>();
        SyncCube_FixpipeHandover(cube->l0c_gate, kCubeMTile, kCubeHSub, 3u);
    }
}

// =================================================================================
// Root pipelines. AivRingState is DEAD: the two production prefetch helpers consume
// ring.stage_live[slot] but only AivConvertAndPush*, which this file does not call,
// raises it -- so one AivResetRing leaves it false and AivDrainRing is never needed.
// =================================================================================

__aicore__ inline void TraceRunVectorCore(
    const Geometry& geom, std::uint32_t sub_idx,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut,
    const GlobalTensor<float>& tokens_in_gm, const GlobalTensor<float>& tokens_out_gm,
    const AivBuffers& buf,
    TBuf<TPosition::A1> (&l1a_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1w_buf)[kCubeRingSlots][kCubeWeightStreams],
    TBuf<TPosition::A1> (&l1a2_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1w2_buf)[kCubeRingSlots]) {
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;
    // An AIV's GetBlockIdx() counts AIVs, not AI Core groups; the census established it.
    const std::uint32_t core_idx =
        static_cast<std::uint32_t>(GetBlockIdx()) / static_cast<std::uint32_t>(GetSubBlockNum());
    const std::uint32_t fwd_steps  = geom.hidden / kCubeKSub;
    const std::uint32_t down_steps = geom.inter / kCubeRSub2;

    AivRingState ring;
    AivResetRing(ring);

    AivStepContext ctx{
        geom, buf, ring, tokens_in_gm, tokens_out_gm,
        l1a_buf, l1w_buf, l1a2_buf, l1w2_buf,
        gate_weights_packed, gate_weight_scales,
        up_weights_packed,   up_weight_scales,
        down_weights_packed, down_weight_scales,
        token_to_expert_lut,
        sub_idx, sub_idx * kCubeMHalf,
        /*token_row_start=*/0u, /*valid_rows=*/0u, /*store_rows=*/0u, /*expert_id=*/0u,
        /*step_no=*/0u,
    };

    DSV4_TRACE("V enter sub=%u tiles=%u h=%u i=%u fwd=%u down=%u\n",
               sub_idx, total_tiles, geom.hidden, geom.inter, fwd_steps, down_steps);

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        if (!AivAdmitTile(ctx, tile_idx)) {
            DSV4_TRACE("V tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("V tile=%u expert=%u rows=%u\n", tile_idx, ctx.expert_id, ctx.valid_rows);

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("V p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            for (std::uint32_t s = 0; s <= fwd_steps; ++s) {
                ExecuteMicroStepLockstep(&ctx, nullptr, s, fwd_steps, inter_base);
            }
            SyncAiv_DrainStepFreeTail(fwd_steps);
            ExecutePhase1Epilogue(&ctx, nullptr, inter_base);
        }

        DSV4_TRACE("V p2 enter steps=%u store_rows=%u\n", down_steps, ctx.store_rows);
        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("V p2 blk hb=%u\n", hidden_base);
            for (std::uint32_t s = 0; s <= down_steps; ++s) {
                ExecuteMicroStepLockstepDown(&ctx, nullptr, s, down_steps, hidden_base);
            }
            SyncAiv_DrainStepFreeTail(down_steps);
            ExecutePhase2Epilogue(&ctx, nullptr, hidden_base);
        }
        DSV4_TRACE("V tile=%u DONE\n", tile_idx);
    }
    DSV4_TRACE("V exit\n");
}

__aicore__ inline void TraceRunCubeCore(
    const Geometry& geom,
    GM_ADDR token_to_expert_lut,
    TBuf<TPosition::A1> (&l1a_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1w_buf)[kCubeRingSlots][kCubeWeightStreams],
    TBuf<TPosition::A1> (&l1a2_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1w2_buf)[kCubeRingSlots]) {
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;
    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx());
    const std::uint32_t fwd_steps   = geom.hidden / kCubeKSub;
    const std::uint32_t down_steps  = geom.inter / kCubeRSub2;

    __ca__ bfloat16_t l0a[kCubeL0AElems];
    __cb__ bfloat16_t l0b_gate[kCubeL0BElems];
    __cb__ bfloat16_t l0b_up[kCubeL0BElems];
    __cc__ float      l0c_gate[kCubeL0CElems];
    __cc__ float      l0c_up[kCubeL0CElems];

    CubeStepContext ctx{
        geom,
        l1a_buf, l1w_buf, l1a2_buf, l1w2_buf,
        l0a, l0b_gate, l0b_up, l0c_gate, l0c_up,
        token_to_expert_lut,
        /*expert_id=*/0u, /*step_no=*/0u,
    };

    DSV4_TRACE("C enter tiles=%u h=%u i=%u fwd=%u down=%u\n",
               total_tiles, geom.hidden, geom.inter, fwd_steps, down_steps);

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        if (!CubeAdmitTile(ctx, tile_idx)) {
            DSV4_TRACE("C tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("C tile=%u expert=%u\n", tile_idx, ctx.expert_id);

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("C p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            for (std::uint32_t s = 0; s <= fwd_steps; ++s) {
                ExecuteMicroStepLockstep(nullptr, &ctx, s, fwd_steps, inter_base);
            }
            ExecutePhase1Epilogue(nullptr, &ctx, inter_base);
            DSV4_TRACE("C p1 blk ib=%u DONE\n", inter_base);
        }

        DSV4_TRACE("C p2 enter steps=%u\n", down_steps);
        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("C p2 blk hb=%u\n", hidden_base);
            for (std::uint32_t s = 0; s <= down_steps; ++s) {
                ExecuteMicroStepLockstepDown(nullptr, &ctx, s, down_steps, hidden_base);
            }
            ExecutePhase2Epilogue(nullptr, &ctx, hidden_base);
            DSV4_TRACE("C p2 blk hb=%u DONE\n", hidden_base);
        }
        DSV4_TRACE("C tile=%u DONE\n", tile_idx);
    }
    DSV4_TRACE("C exit\n");
}

// =================================================================================
// Entry point. UB/L1 allocation, the declaration ORDER that is the Fixpipe ABI, the
// block-index census and the production V_MTE2 priming pair are all unchanged.
// =================================================================================

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_trace_sim(
    GM_ADDR tokens_in,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut, GM_ADDR tokens_out,
    GM_ADDR shared_workspace,    GM_ADDR tiling_config) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (unlikely(tiling_config == nullptr || shared_workspace == nullptr)) {
        return;
    }
    if (unlikely(tokens_in == nullptr || tokens_out == nullptr || token_to_expert_lut == nullptr)) {
        return;
    }
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data =
        reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
    if (unlikely(TilingRejected(tiling_data))) {
        return;
    }
    const Geometry geom = LoadGeometry(tiling_data);

    TPipe pipe;
    TBuf<TPosition::A1> l1a_buf[kCubeRingSlots];
    TBuf<TPosition::B1> l1w_buf[kCubeRingSlots][kCubeWeightStreams];
    TBuf<TPosition::A1> l1a2_buf[kCubeRingSlots];
    TBuf<TPosition::B1> l1w2_buf[kCubeRingSlots];
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        pipe.InitBuffer(l1a_buf[s], kCubeL1ASlotBytes);
    }
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
            pipe.InitBuffer(l1w_buf[s][w], kCubeL1WSlotBytes);
        }
    }
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        pipe.InitBuffer(l1a2_buf[s], kCubeL1A2SlotBytes);
    }
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        pipe.InitBuffer(l1w2_buf[s], kCubeL1W2SlotBytes);
    }

    // Block-index census, printed before any flag is touched. An AIV's GetBlockIdx()
    // counts AIVs, not AI Core groups, so `cand` = blk / sub_num is the group index
    // the tile loop and the AIC must agree on; `derived` is what production uses.
    {
        const std::uint32_t blk     = static_cast<std::uint32_t>(GetBlockIdx());
        const std::uint32_t blk_num = static_cast<std::uint32_t>(GetBlockNum());
        const std::uint32_t sub     = static_cast<std::uint32_t>(GetSubBlockIdx());
        const std::uint32_t sub_num = static_cast<std::uint32_t>(GetSubBlockNum());
        const std::uint32_t cand    = sub_num != 0u ? (blk / sub_num) : blk;
        if ASCEND_IS_AIC {
            DSV4_TRACE("IDS AIC blk=%u blk_num=%u sub=%u sub_num=%u derived=%u cand=%u bd=%u\n",
                       blk, blk_num, sub, sub_num, blk, cand, geom.block_dim);
        }
        if ASCEND_IS_AIV {
            DSV4_TRACE("IDS AIV blk=%u blk_num=%u sub=%u sub_num=%u derived=%u cand=%u bd=%u\n",
                       blk, blk_num, sub, sub_num, blk, cand, geom.block_dim);
        }
    }

    if ASCEND_IS_AIC {
        TraceRunCubeCore(geom, token_to_expert_lut, l1a_buf, l1w_buf, l1a2_buf, l1w2_buf);
    }

    if ASCEND_IS_AIV {
        GlobalTensor<float> tokens_in_gm;
        GlobalTensor<float> tokens_out_gm;
        const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
        tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

        // Declaration ORDER is the ABI with the Cube core (q_landing at UB offset 0).
        TQue<TPosition::VECIN,   1> q_landing;
        TQue<TPosition::VECCALC, 1> q_act;
        TQue<TPosition::VECCALC, 1> q_gate_acc;
        TQue<TPosition::VECCALC, 1> q_up_acc;
        TQue<TPosition::VECOUT,  1> q_act_sub;
        TQue<TPosition::VECCALC, 1> q_x_f32[kCubeRingSlots];
        TQue<TPosition::VECOUT,  1> q_x_bf16[kCubeRingSlots];
        TQue<TPosition::VECCALC, 1> q_nib[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_scale[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_wide[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_scale_plane;
        TQue<TPosition::VECCALC, 1> q_scale_temp;

        pipe.InitBuffer(q_landing,  1, kCubeLandingBytes);
        pipe.InitBuffer(q_act,      1, kCubeActPlaneBytes);
        pipe.InitBuffer(q_gate_acc, 1, kCubeAccBytes);
        pipe.InitBuffer(q_up_acc,   1, kCubeAccBytes);
        pipe.InitBuffer(q_act_sub,  1, kCubeActSubBytes);
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            pipe.InitBuffer(q_x_f32[s], 1, kCubeTokenF32Bytes);
        }
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            pipe.InitBuffer(q_x_bf16[s], 1, kCubeTokenSubBytes);
        }
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
                pipe.InitBuffer(q_nib[s][w], 1, kCubeNibHalfBytes);
            }
        }
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
                pipe.InitBuffer(q_scale[s][w], 1, kCubeScaleBytes);
            }
        }
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
                pipe.InitBuffer(q_wide[s][w], 1, kCubeWideHalfBytes);
            }
        }
        pipe.InitBuffer(q_scale_plane, 1, kCubeScaleRowBytes);
        pipe.InitBuffer(q_scale_temp,  1, kCubeScaleTempBytes);

        AivBuffers buf;
        buf.landing  = q_landing.AllocTensor<float>();
        buf.act      = q_act.AllocTensor<bfloat16_t>();
        buf.gate_acc = q_gate_acc.AllocTensor<float>();
        buf.up_acc   = q_up_acc.AllocTensor<float>();
        buf.act_sub  = q_act_sub.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> scale_plane = q_scale_plane.AllocTensor<bfloat16_t>();
        LocalTensor<uint16_t>   scale_temp  = q_scale_temp.AllocTensor<uint16_t>();
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            buf.x_f32[s]  = q_x_f32[s].AllocTensor<float>();
            buf.x_bf16[s] = q_x_bf16[s].AllocTensor<bfloat16_t>();
            for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
                buf.w[s][w].weight_nibbles  = q_nib[s][w].AllocTensor<fp4x2_e2m1_t>();
                buf.w[s][w].weight_scales   = q_scale[s][w].AllocTensor<uint8_t>();
                buf.w[s][w].weight_expanded = q_wide[s][w].AllocTensor<bfloat16_t>();
                buf.w[s][w].scale_plane     = scale_plane;
                buf.w[s][w].scale_temporary = scale_temp;
            }
        }

        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            Duplicate(buf.x_f32[s], 0.0F, kCubeMTile * kCubeKHalf);
        }
        // Production priming pair, kept: balanced on its own, and the no-prime
        // variant is a separate probe so the two deltas stay uncombined.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
            WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
        }

        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());

        TraceRunVectorCore(geom, sub_idx,
                           gate_weights_packed, gate_weight_scales,
                           up_weights_packed,   up_weight_scales,
                           down_weights_packed, down_weight_scales,
                           token_to_expert_lut,
                           tokens_in_gm, tokens_out_gm, buf,
                           l1a_buf, l1w_buf, l1a2_buf, l1w2_buf);

        for (std::uint32_t s = kCubeRingSlots; s-- > 0;) {
            for (std::uint32_t w = kCubeWeightStreams; w-- > 0;) {
                q_wide[s][w].FreeTensor(buf.w[s][w].weight_expanded);
                q_scale[s][w].FreeTensor(buf.w[s][w].weight_scales);
                q_nib[s][w].FreeTensor(buf.w[s][w].weight_nibbles);
            }
            q_x_bf16[s].FreeTensor(buf.x_bf16[s]);
            q_x_f32[s].FreeTensor(buf.x_f32[s]);
        }
        q_scale_temp.FreeTensor(scale_temp);
        q_scale_plane.FreeTensor(scale_plane);
        q_act_sub.FreeTensor(buf.act_sub);
        q_up_acc.FreeTensor(buf.up_acc);
        q_gate_acc.FreeTensor(buf.gate_acc);
        q_act.FreeTensor(buf.act);
        q_landing.FreeTensor(buf.landing);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
