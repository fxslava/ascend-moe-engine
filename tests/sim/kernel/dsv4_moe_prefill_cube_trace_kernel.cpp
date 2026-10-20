/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY, never run on hardware (CLAUDE.md 5.7). All synchronization lives here.
// Counter s = 0..N: DMA tile s into slot s%2 while Vector/MTE3/Cube work tile s-1.

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

// Stream order is not time order, so every record carries a cycle stamp for merging.
#define DSV4_TRACE(fmt, ...)                                                   \
    __asc_aicore::printf("@%u " fmt,                                           \
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(GetSystemCycle())\
                                   & 0xFFFFFFFFull),                           \
        ##__VA_ARGS__)

}  // namespace

// --- sync primitives: a mode 2/4 wait blocks only its own pipe, so each is drained ---

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

/// Mandatory: an intra-core flag is BINARY, so a token left raised double-sets next lap.
__aicore__ inline void SyncAiv_ConsumeDmaToken(std::uint32_t s, std::uint32_t slot) {
    WaitFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[slot]);
    DSV4_TRACE("V s=%u took SlotFull slot=%u\n", s, slot);
}

/// Drain the releases the in-loop guard could not reach, so no counter crosses a block.
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

// --- the four loop steps: ALL core differentiation for the body is here ---

/// Loop step 2, wait: the previous microstep's async work, and nothing else. Issues nothing.
__aicore__ inline void SyncAcquireDependencies(const MicrostepIter& it) {
    const StepSlotMapping& m = it.slots;
    if ASCEND_IS_AIV {
        DSV4_TRACE("[AIV%u] s=%u st=%u ep=%u/%u dma=T%u/%u dq=T%u/%u l1=%u\n",
                   static_cast<std::uint32_t>(GetSubBlockIdx()), it.step, (unsigned)m.stage,
                   (unsigned)it.epilogue, it.stage,
                   m.tile_dma, m.slot_dma, m.tile_mmad, m.slot_dequant, m.slot_l1);
        // WAR: the Cube must have freed this L1 slot before the lap that rewrites it.
        if (m.dequant_active && it.step >= kCubeFirstL1Reuse) {
            SyncAiv_WaitL1Free(it.step, m.slot_l1);
        }
        // RAW: take the DMA token this slot's burst raised one microstep ago.
        if (m.dequant_active) {
            SyncAiv_ConsumeDmaToken(it.step, m.slot_dequant);
        }
        // Reconcile the block's StepFree releases the in-loop guard could not reach.
        if (it.drain_step_free) {
            SyncAiv_DrainStepFreeTail(it.total_steps);
        }
        if (it.aiv_wait_product) {
            DSV4_TRACE("V WAIT Product s=%u base=%u\n", it.step, it.base);
            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            PipeBarrier<PIPE_ALL>();
            DSV4_TRACE("V GOT Product s=%u\n", it.step);
        }
    }
    if ASCEND_IS_AIC {
        // RAW on L1 plus, via its drain, the WAR that retires the previous step's Mmad
        // before this step's MTE1 overwrites L0A/L0B.
        if (m.mmad_active) {
            DSV4_TRACE("[STEP%u Begin]\n", it.step);
            SyncCube_WaitL1Ready(it.step, m.slot_l1);
        }
        // Contraction done: the Mmads must retire before Fixpipe reads L0C.
        if (it.aic_retire_mmad) {
            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            PipeBarrier<PIPE_ALL>();
            DSV4_TRACE("C retired mmad s=%u base=%u\n", it.step, it.base);
        }
        // Scoped to what the wait actually gates: the next Fixpipe write into the landing
        // zone, or, on the last stage, the next block's L1 -> L0 loads.
        if (it.aic_wait_landing_free) {
            DSV4_TRACE("C WAIT LandingFree s=%u rv=%u\n",
                       it.step, (unsigned)it.aic_wait_rendezvous);
            if (it.aic_wait_rendezvous) {
                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(kCubeFlagLandingFree);
            } else {
                CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
            }
            PipeBarrier<PIPE_ALL>();
            DSV4_TRACE("C GOT LandingFree s=%u\n", it.step);
        }
    }
}

/// Loop step 3, process: issue this microstep's async work. No cross-core flag, no drain,
/// no token. The two edges left are intra-step RAW within ONE core and cannot be hoisted
/// into step 2 or 4: at ring depth 2 the push must read what this same step dequanted, and
/// L0B is exactly full so the Mmad must read what this same step loaded.
__aicore__ inline void ProcessAsync(
    AivStepContext* aiv, CubeStepContext* cube, const MicrostepIter& it) {
    const StepSlotMapping& m = it.slots;
    if ASCEND_IS_AIV {
        if (m.dma_active) {
            if (it.forward) {
                PayloadAiv_DmaInFwd(*aiv, m, it.base);
            } else {
                PayloadAiv_DmaInDown(*aiv, m, it.base);
            }
        }
        if (m.dequant_active) {
            if (it.forward) {
                PayloadAiv_DequantAndCast(*aiv, m);
            } else {
                PayloadAiv_DequantDown(*aiv, m);
            }
            // V -> MTE3 on this slot only, so the DMA above keeps running underneath.
            SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
            WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
            if (it.forward) {
                PayloadAiv_PushL1(*aiv, m);
            } else {
                PayloadAiv_PushL1Down(*aiv, m);
            }
        }
        if (it.aiv_gate) {
            PayloadAiv_EpilogueGate(*aiv);
        }
        if (it.aiv_up) {
            PayloadAiv_EpilogueUp(*aiv);
        }
        if (it.aiv_swiglu) {
            // The upper clamp reads up_acc, not the landing zone, so it is free to run
            // here rather than against the Fixpipe that has already been released.
            PayloadAiv_EpilogueUpClamp(*aiv);
            PayloadAiv_ApplySwiGLU(*aiv, it.base);
        }
        if (it.aiv_store) {
            PayloadAiv_StoreLandingPlane(*aiv, it.base);
        }
    }
    if ASCEND_IS_AIC {
        if (m.mmad_active) {
            if (it.forward) {
                PayloadCube_LoadL0(*cube, m);
            } else {
                PayloadCube_LoadL0Down(*cube, m);
            }
            PipeBarrier<PIPE_ALL>();   // MTE1 -> M; MTE1_M events are out of scope here
            if (it.forward) {
                PayloadCube_MmadAccumulate(*cube, m);
            } else {
                PayloadCube_MmadDown(*cube, m);
            }
        }
        if (it.aic_fixpipe) {
            const std::uint32_t extent_n = it.forward ? kCubeRSub : kCubeHSub;
            DSV4_TRACE("C fixpipe s=%u up=%u n=%u\n", it.step, (unsigned)it.aic_fixpipe_up,
                       extent_n);
            PayloadCube_FixpipeMoveOut(
                it.aic_fixpipe_up ? cube->l0c_up : cube->l0c_gate, kCubeMTile, extent_n);
        }
    }
}

/// Loop step 4, flag: publish what step 3 issued, each flag scoped to the pipe that did
/// the work, so the scope IS the drain. The AIV PIPE_ALL is this probe's lockstep.
__aicore__ inline void SyncPublishCompletions(const MicrostepIter& it) {
    const StepSlotMapping& m = it.slots;
    if ASCEND_IS_AIV {
        if (m.dequant_active) {
            SyncAiv_SignalL1Ready(it.step, m.slot_l1);
        }
        if (it.aiv_set_landing_free) {
            if (it.aiv_flag_after_store) {
                PipeBarrier<PIPE_ALL>();
                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
            } else {
                CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
            }
            DSV4_TRACE("V SET LandingFree s=%u store=%u\n",
                       it.step, (unsigned)it.aiv_flag_after_store);
        }
        PipeBarrier<PIPE_ALL>();
    }
    if ASCEND_IS_AIC {
        if (m.mmad_active) {
            SyncCube_ReleaseL1(it.step, m.slot_l1);
            DSV4_TRACE("[STEP%u Completed]\n", it.step);
        }
        // PIPE_FIX scoping is the drain: the release fires once the Fixpipe write lands.
        if (it.aic_set_product) {
            CubeReleaseToAllAivs<PIPE_FIX>(kCubeFlagProduct);
            DSV4_TRACE("C SET Product s=%u x%u\n", it.step, (unsigned)kCubeAivPerAic);
        }
    }
}

// --- the microstep loop: ONE definition, contraction and epilogue, both cores ---

__aicore__ inline void ExecuteMicrostepLoop(
    AivStepContext* aiv, CubeStepContext* cube, const MicrostepPlan& plan) {
    for (std::uint32_t s = 0; s <= plan.LastStep(); ++s) {
        const MicrostepIter it = PlanMicrostep(plan, s);   // 1. plan slots and direction
        SyncAcquireDependencies(it);                       // 2. wait on the previous step
        ProcessAsync(aiv, cube, it);                       // 3. issue, no sync inside
        SyncPublishCompletions(it);                        // 4. flag what was issued
    }
}

// --- trace primitives: the per-core prefix and lookups, so the pipeline has none ---

/// An AIV's GetBlockIdx() counts AIVs, not AI Core groups; the census established it.
__aicore__ inline std::uint32_t TraceCoreIndex() {
    if ASCEND_IS_AIV {
        return static_cast<std::uint32_t>(GetBlockIdx()) /
               static_cast<std::uint32_t>(GetSubBlockNum());
    }
    return static_cast<std::uint32_t>(GetBlockIdx());
}

__aicore__ inline bool TraceAdmitTile(
    AivStepContext* aiv, CubeStepContext* cube, std::uint32_t tile_idx) {
    if ASCEND_IS_AIV {
        return AivAdmitTile(*aiv, tile_idx);
    }
    return CubeAdmitTile(*cube, tile_idx);
}

__aicore__ inline void TraceEnter(
    const Geometry& geom, std::uint32_t total_tiles,
    std::uint32_t fwd_steps, std::uint32_t down_steps) {
    if ASCEND_IS_AIV {
        DSV4_TRACE("V enter sub=%u tiles=%u h=%u i=%u fwd=%u down=%u\n",
                   static_cast<std::uint32_t>(GetSubBlockIdx()), total_tiles,
                   geom.hidden, geom.inter, fwd_steps, down_steps);
    }
    if ASCEND_IS_AIC {
        DSV4_TRACE("C enter tiles=%u h=%u i=%u fwd=%u down=%u\n",
                   total_tiles, geom.hidden, geom.inter, fwd_steps, down_steps);
    }
}

__aicore__ inline void TraceTile(
    AivStepContext* aiv, CubeStepContext* cube, std::uint32_t tile_idx, bool admitted) {
    if ASCEND_IS_AIV {
        if (admitted) {
            DSV4_TRACE("V tile=%u expert=%u rows=%u\n",
                       tile_idx, aiv->expert_id, aiv->valid_rows);
        } else {
            DSV4_TRACE("V tile=%u SKIP\n", tile_idx);
        }
    }
    if ASCEND_IS_AIC {
        if (admitted) {
            DSV4_TRACE("C tile=%u expert=%u\n", tile_idx, cube->expert_id);
        } else {
            DSV4_TRACE("C tile=%u SKIP\n", tile_idx);
        }
    }
}

__aicore__ inline void TraceBlock(const MicrostepPlan& plan, bool done) {
    const std::uint32_t phase = plan.Forward() ? 1u : 2u;
    if ASCEND_IS_AIV {
        DSV4_TRACE("V p%u blk base=%u steps=%u done=%u\n",
                   phase, plan.base, plan.total_steps, (unsigned)done);
    }
    if ASCEND_IS_AIC {
        DSV4_TRACE("C p%u blk base=%u steps=%u done=%u\n",
                   phase, plan.base, plan.total_steps, (unsigned)done);
    }
}

__aicore__ inline void TraceTileDone(std::uint32_t tile_idx) {
    if ASCEND_IS_AIV {
        DSV4_TRACE("V tile=%u DONE\n", tile_idx);
    }
    if ASCEND_IS_AIC {
        DSV4_TRACE("C tile=%u DONE\n", tile_idx);
    }
}

__aicore__ inline void TraceExit() {
    if ASCEND_IS_AIV {
        DSV4_TRACE("V exit\n");
    }
    if ASCEND_IS_AIC {
        DSV4_TRACE("C exit\n");
    }
}

/// One block: contraction and epilogue are the same loop, so this only brackets it.
__aicore__ inline void ExecuteBlock(
    AivStepContext* aiv, CubeStepContext* cube, const MicrostepPlan& plan) {
    TraceBlock(plan, /*done=*/false);
    ExecuteMicrostepLoop(aiv, cube, plan);
    TraceBlock(plan, /*done=*/true);
}

// --- THE root pipeline: one definition, both cores call it with the context they own ---
// AivRingState is dead here: nothing this file calls ever raises ring.stage_live[slot].

__aicore__ inline void ExecuteTracePipeline(
    AivStepContext* aiv, CubeStepContext* cube, const Geometry& geom) {
    const std::uint32_t total_tiles =
        (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;
    const std::uint32_t fwd_steps  = geom.hidden / kCubeKSub;
    const std::uint32_t down_steps = geom.inter / kCubeRSub2;

    TraceEnter(geom, total_tiles, fwd_steps, down_steps);

    for (std::uint32_t tile_idx = TraceCoreIndex(); tile_idx < total_tiles;
         tile_idx += geom.block_dim) {
        if (!TraceAdmitTile(aiv, cube, tile_idx)) {
            TraceTile(aiv, cube, tile_idx, /*admitted=*/false);
            continue;
        }
        TraceTile(aiv, cube, tile_idx, /*admitted=*/true);

        for (std::uint32_t base = 0; base < geom.inter; base += kCubeRSub) {
            ExecuteBlock(aiv, cube, {MicrostepPhase::Forward, base, fwd_steps});
        }

        for (std::uint32_t base = 0; base < geom.hidden; base += kCubeHSub) {
            ExecuteBlock(aiv, cube, {MicrostepPhase::Down, base, down_steps});
        }
        TraceTileDone(tile_idx);
    }
    TraceExit();
}

// --- entry point: UB and L1 carved by hand from one reservation per pool ---
// TPipe is the arena reservation ONLY, and not optional: Broadcast inside the untouched
// DequantSubTileVector pops its tmp buffer from the pool end, which this moves above us.

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

    // Identical on both cores, before any branch: the only thing making addresses agree.
    TBuf<TPosition::A1> l1_arena;
    pipe.InitBuffer(l1_arena, kL1ArenaBytes);
    L1Operands l1;
    PlaceL1Operands(l1, static_cast<std::uint32_t>(l1_arena.Get<bfloat16_t>().GetPhyAddr()));

    // Census before any flag is touched: cand = blk / sub_num, derived is what ships.
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

    // Per-core setup only. The pipeline itself is ONE function, below each branch.
    if ASCEND_IS_AIC {
        // L0 is Cube-private; the AIV has no such memory, so these stay in this branch.
        __ca__ bfloat16_t l0a[kCubeL0AElems];
        __cb__ bfloat16_t l0b_gate[kCubeL0BElems];
        __cb__ bfloat16_t l0b_up[kCubeL0BElems];
        __cc__ float      l0c_gate[kCubeL0CElems];
        __cc__ float      l0c_up[kCubeL0CElems];

        CubeStepContext ctx{
            geom, l1,
            l0a, l0b_gate, l0b_up, l0c_gate, l0c_up,
            token_to_expert_lut,
            /*expert_id=*/0u,
        };
        ExecuteTracePipeline(nullptr, &ctx, geom);
    }

    if ASCEND_IS_AIV {
        GlobalTensor<float> tokens_in_gm;
        GlobalTensor<float> tokens_out_gm;
        const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
        tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

        // Each AIV owns its own 256 KiB UB, so both run this table over their own space.
        TBuf<TPosition::VECCALC> ub_arena;
        pipe.InitBuffer(ub_arena, kUbArenaBytes);
        const std::uint32_t ub_base =
            static_cast<std::uint32_t>(ub_arena.Get<float>().GetPhyAddr());
        // A nonzero base means the AIC's Fixpipe target no longer names the landing zone.
        DSV4_TRACE("UB arena base=%u bytes=%u fixpipe_target=%u\n",
                   ub_base, (unsigned)kUbArenaBytes, (unsigned)kUbOffLanding);

        AivBuffers buf;
        PlaceAivBuffers(buf, ub_base);

        // Rows past valid_rows keep these zeros: that is what makes full-M contraction right.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            Duplicate(buf.x_f32[s], 0.0F, kCubeMTile * kCubeKHalf);
        }
        // Production priming pair, kept: balanced on its own.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
            WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
        }

        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());

        AivRingState ring;
        AivResetRing(ring);

        AivStepContext ctx{
            geom, buf, ring, tokens_in_gm, tokens_out_gm, l1,
            gate_weights_packed, gate_weight_scales,
            up_weights_packed,   up_weight_scales,
            down_weights_packed, down_weight_scales,
            token_to_expert_lut,
            sub_idx, sub_idx * kCubeMHalf,
            /*token_row_start=*/0u, /*valid_rows=*/0u, /*store_rows=*/0u, /*expert_id=*/0u,
        };
        ExecuteTracePipeline(&ctx, nullptr, geom);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
