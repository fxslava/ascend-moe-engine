/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- BISECTION PROBE: dsv4_moe_prefill_cube with the AIV's V_MTE2
// priming pair removed, and NOTHING ELSE CHANGED.
//
// ===========================================================================
// THE QUESTION THIS ANSWERS
// ===========================================================================
// The first CAModel execution of the production entry point (see
// tests/sim/test_dsv4_moe_prefill_cube_sim.cpp) does not retire. MEASURED,
// 2026-10-19, at hidden 128 / inter 256 -- four contraction steps, the
// cheapest admissible shape:
//
//   * the MIX_AIC_1_2 bring-up completes and the launch starts (one
//     [TASK_BEGIN], no [TASK_END])
//   * 320,000 simulated cycles in 900 s of host time (~355 cycles/s), then
//     the driver's watchdog ended the process
//   * ZERO memory faults: no pem_lsu, no judge_valid_addr, no
//     ub_addr_overflow, no abort. The inductive gate's
//     `pem_lsu.cc:484 invalid ldst addr` flood does NOT reproduce here.
//   * EXACTLY FOUR `pem_ccu.cc:2277 execute_set_flag already has same
//     set_flag!`, at two PCs (0x9000d0d724 and 0x9000d0d868) twice each, all
//     inside the first ~20,000 cycles and never recurring.
//
// An advancing cycle counter proves nothing about forward progress -- a
// CAModel keeps clocking while every core sits parked on a flag -- so "slow"
// and "spinlocked" are both still live. What is NOT ambiguous is the meaning
// of the four reports. dsv4_moe_prefill_cube_arch35.h states the rule:
//
//     "An intra-core flag is BINARY, not a counter: a second SetFlag on the
//      same (srcPipe, dstPipe, eventID) triple before the first is consumed
//      is undefined behaviour ... and a later PipeBarrier<PIPE_ALL> then
//      hangs."
//
// A swallowed set means one token fewer than the matching waits expect, which
// is a wait that never satisfies -- a spinlock, with the cycle counter still
// running. So the four reports are a sufficient mechanism for the observed
// symptom, and they have to be attributed before anything else is believed.
//
// TWO EXPLANATIONS WERE ELIMINATED ON THE HOST FIRST, where it costs nothing:
//
//   1. Event id aliasing. Ruled out: a flag is keyed by the full triple, and
//      kEvCubeSlotFull / kEvCubeSlotStage / kEvCubeSlotPushed are MTE2_V,
//      V_MTE2 and MTE3_V respectively, so their shared {0, 1} values are six
//      distinct hardware flags. kEvCubeWideReady is 2. Nothing collides.
//   2. A ring drain leak. Ruled out by walking the token balance at
//      fwd_steps == 1 and down_steps == 1, the degenerate shape the failing
//      run drives: AivResetRing clears both slots, the prologue prefetch
//      finds stage_live[0] false and waits for nothing, k == 0 skips both the
//      next-step prefetch and the StepFree wait, AivConvertAndPushFwdStep
//      raises stage[0] and pushed[0] with the kEvCubeWideReady pair immediate,
//      and AivDrainRing consumes exactly stage[0] and pushed[0] while finding
//      slot 1 clean. Balanced. Phase 2 is the same shape.
//
// PC attribution would settle it directly and IS NOT AVAILABLE: the device
// code lives in a nested `.aicore_binary` ELF of format elf64-hiipu, and the
// toolkit's llvm-objdump has no disassembler for that target, so every
// instruction decodes as `<not available>`.
//
// ===========================================================================
// THE ONE DELTA
// ===========================================================================
// The production entry primes the ring's WAR channel per slot:
//
//     for (s) Duplicate(buf.x_f32[s], 0.0F, ...);
//     for (s) { SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
//               WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]); }
//
// on ids {0, 1} -- the SAME ids the ring's own deferred V_MTE2 token uses.
// This probe deletes those flags entirely and orders the zeroing against the
// first MTE2 burst with a barrier instead:
//
//     for (s) Duplicate(buf.x_f32[s], 0.0F, ...);
//     PipeBarrier<PIPE_ALL>();
//
// WHY DELETION RATHER THAN RE-IDENTIFICATION. Moving the priming to unused
// ids would only test one hypothesis (that the priming collides with the
// ring). Deleting the two SetFlag sites tests the stronger and more useful
// proposition: IF the two reported PCs are the priming sets, the reports must
// disappear. The experiment is therefore informative in all three outcomes:
//
//   reports gone AND the launch retires   -> the priming pair is the defect
//   reports gone AND it still hangs       -> the priming was the double-set
//                                            source but not the hang; the
//                                            hang is elsewhere and now
//                                            separable
//   reports persist                       -> the PCs are NOT the priming
//                                            sets, which eliminates two
//                                            candidate sites and points at
//                                            the ring's own deferred sets
//
// THE BARRIER IS SOUND FOR THE SEMANTICS BEING REPLACED, and weaker than what
// it replaces only in cost. The priming existed to stop the first token burst
// racing the Duplicate; PipeBarrier<PIPE_ALL> orders every pipe at that point,
// which subsumes V -> MTE2. It is a drain and would be wrong in a hot loop --
// it runs ONCE per launch, before the tile loop, so it costs one drain.
//
// NOT A CANDIDATE FIX. If this probe implicates the priming pair, the repair
// belongs in the operator and must preserve the overlap; a PIPE_ALL drain in
// the production entry is acceptable only because this one sits outside every
// loop, and that reasoning has to be redone at the real fix site.
//
// Everything below the delta is the production schedule, reached through the
// production RunVectorCore and RunCubeCore by textual inclusion. The entry
// body is transcribed rather than shared because the priming sits INSIDE the
// production __global__ function, which has no seam to override -- the same
// reason dsv4_moe_prefill_cube_inductive_kernel.cpp transcribes its own.
// op_kernel/ stays clean (CLAUDE.md section 1.1): no verification branch and
// no #ifdef SIM reaches the shipping kernel.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

// The production TU, verbatim. Brings RunVectorCore, RunCubeCore, AivBuffers,
// LoadGeometry, TilingRejected and every UB/L1 constant into THIS translation
// unit, including the ones in its anonymous namespace, which is why the probe
// can call them at all.
#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_noprime_sim(
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

    // L1, allocated IDENTICALLY ON BOTH CORES and in the same order, outside
    // the ASCEND_IS_* branches: the AIVs write these slots and the AIC reads
    // them, so the three cores have to land on the same L1 offsets and nothing
    // in the toolchain checks that they do.
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

    // UNCHANGED. The AIC side of this probe is the production Cube core, flag
    // for flag -- the delta is AIV-side only, so an AIC difference could not
    // be a confound.
    if ASCEND_IS_AIC {
        RunCubeCore(geom, token_to_expert_lut, l1a_buf, l1w_buf, l1a2_buf, l1w2_buf);
    }

    if ASCEND_IS_AIV {
        GlobalTensor<float> tokens_in_gm;
        GlobalTensor<float> tokens_out_gm;
        const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
        tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

        // q_landing IS FIRST, AND THAT IS THE CONTRACT WITH THE AIC: the
        // Fixpipe names UB offset 0 in each AIV's own address space and the
        // TPipe dynamic pool starts at 0 on this toolchain. The declaration
        // ORDER below is therefore part of the ABI with RunCubeCore and is
        // transcribed exactly, not tidied.
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

        // Zero the token staging once, exactly as production does: the Cube
        // always contracts the full M = 16, so rows past valid_rows must be
        // ZERO rather than absent.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            Duplicate(buf.x_f32[s], 0.0F, kCubeMTile * kCubeKHalf);
        }

        // ===================================================================
        // THE DELTA. Production has here:
        //
        //     for (s) { SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
        //               WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]); }
        //
        // two SetFlags on the same V_MTE2 ids {0, 1} that the ring's own
        // deferred WAR token later reuses. This probe raises no flag at all
        // and orders the Duplicate against the first MTE2 burst with a full
        // pipe barrier. One drain, once per launch, outside every loop.
        // ===================================================================
        PipeBarrier<PIPE_ALL>();

        // `% GetSubBlockNum()` IS LOAD-BEARING: GetSubBlockIdx() is the AIV's
        // logical id across ALL AI Core groups, while the id WITHIN the group
        // is what "am I this AIC's first vector core" needs.
        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());

        RunVectorCore(geom, sub_idx,
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
