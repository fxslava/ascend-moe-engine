/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- PROGRESS TRACE PROBE for dsv4_moe_prefill_cube.
//
// ===========================================================================
// WHAT THIS IS FOR
// ===========================================================================
// The production entry point does not retire on the dav-3510 CAModel, and
// three rounds of inference from the simulator log have not localized it:
//
//   * 320,000 cycles in 900 s at the 4-step shape, one [TASK_BEGIN], no
//     [TASK_END], and ZERO memory faults (no pem_lsu, no judge_valid_addr,
//     no ub_addr_overflow, no abort).
//   * EXACTLY FOUR `pem_ccu.cc:2277 execute_set_flag already has same
//     set_flag!`, at two sites 0x144 apart, two hits each, inside the first
//     ~20,000 cycles and never recurring.
//   * The no-priming probe (dsv4_moe_prefill_cube_noprime_kernel.cpp) moved
//     both PCs by 0xAFC and left the signature IDENTICAL -- same two sites,
//     same 0x144 separation, same multiplicity -- which FALSIFIED the
//     hypothesis that the entry point's V_MTE2 priming pair raises them.
//
// An advancing cycle counter cannot tell a slow simulation from a spinlock,
// and PC attribution is unavailable: the device code is a nested elf64-hiipu
// ELF and the toolkit's llvm-objdump has no disassembler for that target, so
// every instruction decodes as `<not available>`.
//
// So stop inferring. This probe makes the three cores SAY where they are. It
// turns "it stops somewhere" into "the AIV reached phase 1 block 0 step 0 and
// posted StepReady; the AIC never returned from its StepReady wait" -- which
// names the edge instead of guessing the flag.
//
// ===========================================================================
// THE MECHANISM, AND WHY IT IS SAFE TO ATTEMPT
// ===========================================================================
// __asc_aicore::printf, which is available for __NPU_ARCH__ == 3510. It writes
// a TLV into a GM debug ring buffer that the runtime provisions, and
// scalar_printf_impl's FIRST action is
//
//     __gm__ DebugBlockHeadInfo* blockInfo = get_block_info();
//     if (blockInfo == nullptr) { return; }
//
// so if the chevron launch path does not provision that buffer the calls are a
// silent no-op rather than a GM fault -- the failure mode is "no output", not
// "a second bug to debug". check_ringbuf_space() likewise DROPS a record that
// does not fit instead of overrunning, so the trace cannot corrupt anything.
// printf_impl_common also emits a `CANN Version: ...` header on the first
// call, which is a positive signal that the mechanism is live: if that line
// appears, absence of later lines is real information.
//
// Each record is prefixed by the toolkit with `[AIV Block %u/%u]` or
// `[AIC Block %u/%u]`, so the three cores are told apart without doing it here.
//
// THE TRACE PERTURBS TIMING, AND THAT IS A REAL LIMITATION. Every record is a
// GM write plus a dcci, on the scalar unit, between pipeline stages. A race
// whose window is a few cycles can be closed by that, so this probe can
// localize a HARD STOP and cannot be trusted to reproduce a marginal one. If
// the kernel retires here but not in the gate, that asymmetry is itself the
// finding and points at a race rather than a deadlock.
//
// ===========================================================================
// WHY THE TWO CORE ROUTINES ARE TRANSCRIBED, AND WHAT THE DELTA IS NOW
// ===========================================================================
// The trace points have to sit BETWEEN the stages inside RunVectorCore and
// RunCubeCore, which are production functions. op_kernel/ stays clean
// (CLAUDE.md section 1.1) -- no probe, no branch and no #ifdef SIM reaches the
// shipping kernel -- so the two routines are transcribed here.
//
// THE DELTA IS NO LONGER "TRACE CALLS ONLY". The cycle-stamped trace showed
// EXTREME INTER-CORE PHASE DRIFT: the AIC ran ahead through contraction steps
// and whole inter blocks while AIV0 and AIV1 were still inside phase 1, so a
// StepReady/StepFree handshake the log attributes to step k was in fact being
// satisfied by a token raised for a different k. A ghost handshake, and then a
// stop. With the three cores free to slip whole laps apart, the log cannot
// name an edge, because the edge the log names is not the edge that fired.
//
// So this probe is now a LOCKSTEP transcription. The production schedule's
// three-stage ring overlap is REMOVED here, deliberately:
//
//   * The deferred ring tokens are gone. `AivRingState::stage_live` and
//     `pushed_live` -- the V_MTE2 and MTE3_V fences that are raised on one lap
//     and consumed on the next -- are never raised, so nothing on the AIV can
//     be a lap out of step with anything else on it.
//   * Every intra-core stage boundary is a `PipeBarrier<PIPE_ALL>()`. That is
//     exactly what the production kernel must NOT do (it drains every pipe and
//     throws the overlap away), and exactly what a probe wants: a record
//     printed after the barrier is true of the whole core, not of the scalar
//     unit running ahead of it.
//   * Every cross-core wait is followed by a `PipeBarrier<PIPE_ALL>()`. A mode
//     2/4 wait is EFFECTIVE on 950PR and blocks ONLY the pipeline it names, so
//     without the barrier the scalar unit walks straight past the wait and the
//     "GOT" record is a lie. The barrier converts a pipe-scoped wait into a
//     whole-core park, which is what makes the trace orderable.
//   * StepFree is released and awaited ONCE PER STEP, unconditionally. The
//     production code releases it only for `k + kCubeRingSlots < fwd_steps`
//     and awaits it only for `k >= kCubeRingSlots` -- textually balanced, but
//     it means the token for step k is consumed at step k + 2. Here it is 1:1
//     within the step, so the AIC cannot enter step k+1 until both AIVs have
//     left step k.
//   * Each block ends in an explicit rendezvous. In phase 1 the AIVs raise a
//     third LandingFree after the SwiGLU plane store and the AIC waits for it
//     before advancing `inter_base` or entering phase 2, so the Cube cannot be
//     in a later block than the vector cores. In phase 2 the single
//     LandingFree already sits after the GM store and is that rendezvous.
//
// EVERY FLAG AND EVERY EVENT ID STAYS BALANCED, which is the one property this
// rewrite could have broken. Per phase 1 contraction step: one MTE2_V
// (kEvCubeSlotFull[slot]) raised by the prefetch and consumed by the first
// dequant; one StepReady set per AIV and one AIC wait; one StepFree release
// per AIV and one wait on each. Per inter block: three LandingFree sets per
// AIV against three AIC waits (gate, up, rendezvous) and two Product releases
// against two waits on each AIV. Balance is necessary and not sufficient --
// the note in RunCubeCore says why -- but here nothing is deferred at all, so
// the sufficiency argument is trivial.
//
// Every helper the two routines call (AivPrefetchFwdStep, AivPrefetchDownStep,
// DequantSubTileVector, PushUbTileToL1Nz, AivApplySwiGLU, CubeLoadTokensToL0A,
// CubeLoadWeightsToL0B, CubeMmadAccumulate, CubeCloseAccumulationStep,
// CubeReleaseToAllAivs) is still the production one, reached by textual
// inclusion. AivConvertAndPushFwdStep and AivConvertAndPushDownStep are the
// two that are NOT called: their bodies are the deferred-token logic being
// removed, so their dequant/Cast/push sequences are inlined below without it.
// CubeMoveOutAndHandOver is inlined for the original reason -- the cross-core
// handover it performs is a prime suspect and a trace that could not print
// between its Fixpipe, its Product release and its LandingFree wait would be
// blind to exactly the edge in question.
//
// WHAT THIS COSTS. A lockstep probe cannot reproduce a race whose window the
// overlap opens, and it is slower per step. It answers one question: do the
// three cores, when they CANNOT drift, still stop -- and if so, at which step
// number on which core. If it retires here and hangs in the gate, the drift
// is the finding.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "utils/debug/asc_printf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

namespace {

// One place to turn the trace off, and one place to change what a record looks
// like. Terse on purpose: the ring buffer is finite and drops what will not
// fit, so a 4-step run must not spend it on prose.
//
// EVERY RECORD CARRIES A CYCLE STAMP, AND THAT IS WHAT MAKES THE TRACE
// ORDERABLE AT ALL.
//
// Records reach stdout through PER-BLOCK GM ring buffers that are flushed
// independently, so their order IN THE STREAM is not their order in time: the
// AIC's records can appear before AIV records that happened earlier. The first
// traced run showed the AIC clearing its StepReady wait seemingly before either
// AIV had entered RunVectorCore, which would be a serious finding -- and is
// exactly the kind of claim the stream order cannot support.
//
// GetSystemCycle() is a device-wide counter with a dav_3510 implementation
// (asc/impl/basic_api/dav_3510/kernel_operator_sys_var_impl.h), so stamping
// each record lets the three cores' streams be MERGED AND SORTED after the run.
// The stamp is what turns "these lines are in this order in the log" into "this
// happened before that".
//
// The low 32 bits are printed rather than the full 64: the counter's absolute
// base is irrelevant here and these runs are ~1e6 cycles, so a 4.3e9-cycle
// wrap cannot occur inside one. Printed via a uint32 because the toolkit's
// scalar marshalling takes a fixed 64-bit slot per argument and %u on a
// 32-bit value is the form already exercised by every other record here.
#define DSV4_TRACE(fmt, ...)                                                   \
    __asc_aicore::printf("@%u " fmt,                                           \
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(GetSystemCycle())\
                                   & 0xFFFFFFFFull),                           \
        ##__VA_ARGS__)

}  // namespace

// =================================================================================================
// HOW THIS FILE IS LAID OUT
// =================================================================================================
// Three layers, bottom up:
//
//   1. TWO CONTEXT STRUCTS, AivStepContext and CubeStepContext, which carry
//      everything a step needs that is not the step number: the geometry, the
//      UB and L1 buffers, the L0 buffers, the GM operand bases and the handful
//      of per-tile values derived from the expert LUT. They are plain
//      aggregates with reference members, built once per core and refreshed
//      per admitted tile by AivAdmitTile / CubeAdmitTile.
//
//   2. ONE `Step*_` HELPER PER DISCRETE PHASE, each owning its own bounding
//      `PipeBarrier<PIPE_ALL>()` calls and its own trace records. A barrier
//      belongs to the step whose effects it publishes, so no caller has to
//      remember to fence anything -- which is the property that made the
//      previous flat transcription hard to audit.
//
//   3. THE TWO ROOT FUNCTIONS, which are nothing but loop nests over those
//      helpers. They are meant to be read SIDE BY SIDE: phase 1 step k on the
//      AIV is Prefetch/Push/Wait and on the AIC is Wait/Mmad/Release, so the
//      lockstep is visible as two three-line bodies rather than inferred from
//      two hundred lines of interleaved indexing.
//
// The stage numbering the records use, unchanged:
//
//   STAGE 1  prefetch (MTE2), barrier, dequant + Cast (V), barrier
//   STAGE 2  push UB -> L1 (MTE3), barrier, set StepReady
//   STAGE 3  (AIC) wait StepReady, L1 -> L0A/L0B, Mmad, barrier, release StepFree
//   STAGE 4  wait StepFree, barrier -- after which all three cores are at step k
//
// NOTHING ABOUT THE SCHEDULE CHANGES HERE. Every instruction, every flag, every
// barrier and every trace record is the one the flat lockstep transcription
// already carried; only the nesting is different. The flag and event budget is
// therefore the same, and the argument for it still stands: per contraction
// step one MTE2_V token raised by the prefetch and consumed by the first
// dequant, one StepReady set per AIV against one AIC wait, one StepFree release
// per AIV against one wait on each; per inter block three LandingFree sets per
// AIV against three AIC waits and two Product releases against two waits.

// =================================================================================================
// THE AIV's CONTEXT
// =================================================================================================
// A plain aggregate, so it is initialized once with a brace list and never
// copied. The reference members are what keep it free: `geom`, `buf` and the
// four L1 buffer arrays are the caller's, and a reference member reached
// through a `const AivStepContext&` is still a non-const reference to its
// target, which is why the step helpers can take the context by const
// reference and still push into L1 and still hand `ring` to a production
// helper that wants it mutable.
//
// THE OPERAND BASES ARE STORED UNOFFSET and the per-expert offset is applied
// by the six accessors below. Keeping `expert_id` rather than a byte offset is
// what lets the trace print it, and the accessors widen before multiplying
// (CLAUDE.md section 2.7) so a large expert index cannot wrap a stride.
struct AivStepContext {
    // --- invariant for the whole launch ---
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
    std::uint32_t row_base;   // sub_idx * kCubeMHalf -- this AIV's token rows

    // --- refreshed per admitted tile by AivAdmitTile ---
    std::uint32_t token_row_start;
    std::uint32_t valid_rows;
    std::uint32_t store_rows;
    std::uint32_t expert_id;

    // THE STEP COUNTER IS WHAT THE NARRATION IS FOR. It runs monotonically
    // across blocks AND across both phases, and all three cores walk identical
    // loop bounds -- same geometry, same admitted tiles, same block and step
    // counts -- so "Step 7" means the same step on AIV0, AIV1 and the AIC. That
    // is the alignment this probe exists to check, so a merged log in which the
    // three Step 7 lines do NOT sit together is itself the finding.
    std::uint32_t step_no;

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

// Refresh the per-tile fields, or report the tile is not this expert's.
// `store_rows` is derived here rather than in phase 2 so the whole of the
// tile's geometry is settled in one place.
__aicore__ inline bool AivAdmitTile(AivStepContext& ctx, std::uint32_t tile_idx) {
    const std::uint32_t token_row_start = tile_idx * ctx.geom.token_tile;
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, token_row_start, ctx.geom.num_experts, expert_id)) {
        return false;
    }
    const std::uint32_t remaining = ctx.geom.total_tokens - token_row_start;
    const std::uint32_t valid_rows =
        remaining < ctx.geom.token_tile ? remaining : ctx.geom.token_tile;
    const std::uint32_t owned = valid_rows > ctx.row_base ? (valid_rows - ctx.row_base) : 0u;

    ctx.token_row_start = token_row_start;
    ctx.valid_rows      = valid_rows;
    ctx.store_rows      = owned < kCubeMHalf ? owned : kCubeMHalf;
    ctx.expert_id       = expert_id;
    return true;
}

// =================================================================================================
// THE AIV's STEPS
// =================================================================================================

// STAGE 1, phase 1. The FIRST dequant consumes the MTE2_V token the prefetch's
// last burst raised; the second passes -1 because one token covered both weight
// bursts. The barrier above it has already ordered MTE2 against V, so that
// WaitFlag is redundant as a fence -- it is there to CONSUME the token, which is
// not optional: an unconsumed binary flag is the double-set on the next lap
// through this slot.
__aicore__ inline void StepAiv_PrefetchAndDequant(
    const AivStepContext& ctx, std::uint32_t slot,
    std::uint32_t inter_base, std::uint32_t k) {
    AivPrefetchFwdStep(ctx.tokens_in, ctx.token_row_start,
                       ctx.GatePacked(), ctx.GateScales(), ctx.UpPacked(), ctx.UpScales(),
                       inter_base, k * kCubeKSub, ctx.valid_rows,
                       ctx.sub_idx, ctx.geom, ctx.buf, slot, ctx.ring);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s1 k=%u prefetched slot=%u\n", k, slot);

    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[slot][0], kEvCubeSlotFull[slot]);
    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[slot][1], -1);
    Cast(ctx.buf.x_bf16[slot], ctx.buf.x_f32[slot], RoundMode::CAST_RINT,
         kCubeMTile * kCubeKHalf);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s1 k=%u dequant+cast done\n", k);
}

// STAGE 2, phase 1. AivConvertAndPushFwdStep's push sequence, instruction for
// instruction, minus its deferred SetFlag<V_MTE2> and SetFlag<MTE3_V> and minus
// its immediate V_MTE3 pair, all three of which the barriers subsume.
__aicore__ inline void StepAiv_PushL1AndSignalReady(
    const AivStepContext& ctx, std::uint32_t slot,
    std::uint32_t col_base, std::uint32_t k) {
    PushUbTileToL1Nz(ctx.l1a[slot].Get<bfloat16_t>(), ctx.buf.x_bf16[slot],
                     kCubeMTile, kCubeKHalf, kCubeKHalf, kCubeMTile, col_base, 0);
    PushUbTileToL1Nz(ctx.l1w[slot][0].Get<bfloat16_t>(), ctx.buf.w[slot][0].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
    PushUbTileToL1Nz(ctx.l1w[slot][1].Get<bfloat16_t>(), ctx.buf.w[slot][1].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s2 k=%u pushed slot=%u\n", k, slot);

    CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
        static_cast<uint16_t>(kCubeFlagStepReady + slot));
    DSV4_TRACE("V s2 k=%u SET StepReady slot=%u\n", k, slot);
}

// STAGE 4, both phases. Unconditional, 1:1 against the AIC's unconditional
// release. The wait is scoped to PIPE_MTE3 -- what StepFree protects is this
// AIV's push INTO the slot -- and a scoped wait on 950PR blocks only that pipe,
// so the barrier after it is what makes the whole core park, and what makes the
// record below true rather than a claim about the scalar unit.
__aicore__ inline void StepAiv_WaitStepComplete(
    const AivStepContext& ctx, std::uint32_t slot, std::uint32_t k) {
    (void)ctx;  // present for signature symmetry with the other StepAiv_ steps
    DSV4_TRACE("V s4 k=%u WAIT StepFree slot=%u\n", k, slot);
    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
        static_cast<uint16_t>(kCubeFlagStepFree + slot));
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s4 k=%u GOT StepFree slot=%u\n", k, slot);
}

// PHASE 1 EPILOGUE, first half: the gate pre-activation arrives in the landing
// zone, is clamped ABOVE by the limit only, and the landing zone is released.
__aicore__ inline void StepAiv_EpilogueGate(
    const AivStepContext& ctx, std::uint32_t inter_base) {
    const std::uint32_t elems = kCubeMHalf * kCubeRSub;
    DSV4_TRACE("V p1 e1 WAIT Product gate ib=%u\n", inter_base);
    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V p1 e1 GOT Product gate\n");

    Mins(ctx.buf.gate_acc, ctx.buf.landing, ctx.geom.swiglu_limit, elems);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
    DSV4_TRACE("V p1 e1 SET LandingFree gate\n");
}

// PHASE 1 EPILOGUE, second half: the up pre-activation, clamped to
// [-limit, limit]. The LOWER clamp is the operation that also moves it out of
// the landing zone, so it comes before the release; the UPPER clamp reads
// `up_acc` and not the landing zone, so it comes after, and holding the Cube
// for it would be a stall for nothing.
__aicore__ inline void StepAiv_EpilogueUp(
    const AivStepContext& ctx, std::uint32_t inter_base) {
    const std::uint32_t elems = kCubeMHalf * kCubeRSub;
    DSV4_TRACE("V p1 e2 WAIT Product up ib=%u\n", inter_base);
    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V p1 e2 GOT Product up\n");

    Maxs(ctx.buf.up_acc, ctx.buf.landing, -ctx.geom.swiglu_limit, elems);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
    DSV4_TRACE("V p1 e2 SET LandingFree up\n");

    Mins(ctx.buf.up_acc, ctx.buf.up_acc, ctx.geom.swiglu_limit, elems);
    PipeBarrier<PIPE_ALL>();
}

// PHASE 1 BLOCK RENDEZVOUS. A THIRD LandingFree, raised only once SwiGLU and
// its plane store have completely finished, against a third AIC wait. This is
// what stops the Cube advancing to the next inter block -- or into phase 2 --
// while an AIV is still in this one. LandingFree is reused rather than a new
// flag id because ids 9..15 have never been exercised green by any probe (see
// the allocation note in the arch35 header) and because the AIC is already
// parked in its second LandingFree wait long before this set can be issued: it
// enters that wait immediately after releasing Product, which is the token this
// AIV needed before it could even run Maxs.
__aicore__ inline void StepAiv_BlockRendezvous(
    const AivStepContext& ctx, std::uint32_t inter_base) {
    AivApplySwiGLU(ctx.buf, inter_base, ctx.geom);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V p1 swiglu done ib=%u\n", inter_base);

    CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
    DSV4_TRACE("V p1 SET BlockDone ib=%u\n", inter_base);
}

// STAGE 1, phase 2. One weight stream and no token burst -- phase 2 reads its A
// operand out of the resident activation plane.
__aicore__ inline void StepAiv_PrefetchDown(
    const AivStepContext& ctx, std::uint32_t slot,
    std::uint32_t hidden_base, std::uint32_t r) {
    AivPrefetchDownStep(ctx.DownPacked(), ctx.DownScales(), hidden_base, r * kCubeRSub2,
                        ctx.sub_idx, ctx.geom, ctx.buf, slot, ctx.ring);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s1 r=%u prefetched slot=%u\n", r, slot);

    DequantSubTileVector(kCubeHSub, kCubeRHalf2, ctx.buf.w[slot][0], kEvCubeSlotFull[slot]);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s1 r=%u dequant done\n", r);
}

// STAGE 2, phase 2. AivConvertAndPushDownStep's two pushes, same order, same
// split -- and the split is the one place the two bisections meet:
//   the weight half is a CONTRACTION half -> a C0-block range, row_base 0
//   the token half   is a TOKEN-ROW half  -> a row range of every block, col_base 0
__aicore__ inline void StepAiv_PushDownAndSignalReady(
    const AivStepContext& ctx, std::uint32_t slot, std::uint32_t r) {
    PushUbTileToL1Nz(ctx.l1w2[slot].Get<bfloat16_t>(), ctx.buf.w[slot][0].weight_expanded,
                     kCubeHSub, kCubeRHalf2, kCubeRHalf2,
                     kCubeHSub, ctx.sub_idx * kCubeRHalf2, 0);
    PushUbTileToL1Nz(ctx.l1a2[slot].Get<bfloat16_t>(), ctx.buf.act[r * kCubeRSub2],
                     kCubeMHalf, kCubeRSub2, ctx.geom.inter,
                     kCubeMTile, 0, ctx.sub_idx * kCubeMHalf);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V s2 r=%u pushed slot=%u\n", r, slot);

    CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
        static_cast<uint16_t>(kCubeFlagStepReady + slot));
    DSV4_TRACE("V s2 r=%u SET StepReady slot=%u\n", r, slot);
}

// This AIV's M/2 rows of one hidden block, out of the landing zone and into GM.
// `store_rows` is zero when the tile's tail does not reach this AIV's half, and
// then nothing is stored -- the rows past the tail carry the zeros the setup
// Duplicate left, and a zero A row produces a zero C row.
__aicore__ inline void StepAiv_StoreLandingPlane(
    const AivStepContext& ctx, std::uint32_t hidden_base) {
    if (ctx.store_rows == 0u) {
        return;
    }
    DataCopyParams out_burst;
    out_burst.blockCount = static_cast<std::uint16_t>(ctx.store_rows);
    out_burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
    out_burst.srcStride  = 0;
    out_burst.dstStride  =
        static_cast<std::uint16_t>(((ctx.geom.hidden - kCubeHSub) * 4u) / 32u);
    DataCopy(ctx.tokens_out[(static_cast<std::uint64_t>(ctx.token_row_start + ctx.row_base) *
                             ctx.geom.hidden) + hidden_base],
             ctx.buf.landing, out_burst);
}

// PHASE 2 BLOCK RENDEZVOUS. No separate flag is needed here: the GM store is
// the AIV's last act in the block, so draining it before the single LandingFree
// set makes that flag itself the rendezvous -- the AIC's fix3 wait cannot clear
// until both AIVs have finished storing, and it is the last thing the AIC does
// in the block.
__aicore__ inline void StepAiv_StoreGMAndBlockDone(
    const AivStepContext& ctx, std::uint32_t hidden_base) {
    DSV4_TRACE("V p2 WAIT Product hb=%u\n", hidden_base);
    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStore>(kCubeFlagProduct);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V p2 GOT Product\n");

    StepAiv_StoreLandingPlane(ctx, hidden_base);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("V p2 stored hb=%u rows=%u\n", hidden_base, ctx.store_rows);

    CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
    DSV4_TRACE("V p2 SET BlockDone hb=%u\n", hidden_base);
}

// =================================================================================================
// THE CUBE's CONTEXT
// =================================================================================================
// The L0 buffers are held as address-space-qualified POINTERS. BiSheng permits
// that and permits passing them on; what it rejects is DEREFERENCING a
// `__ca__` / `__cb__` / `__cc__` pointer ("only __ubuf__ and __gm__ and local
// memory pointer can be dereferenced"), which nothing here does -- every use
// goes through Te::MakeMemPtr inside the production load/Mmad/Fixpipe helpers.
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

    // --- refreshed per admitted tile by CubeAdmitTile ---
    std::uint32_t expert_id;
    // Monotonic across blocks and phases; see AivStepContext::step_no.
    std::uint32_t step_no;
};

__aicore__ inline bool CubeAdmitTile(CubeStepContext& ctx, std::uint32_t tile_idx) {
    const std::uint32_t token_row_start = tile_idx * ctx.geom.token_tile;
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, token_row_start, ctx.geom.num_experts, expert_id)) {
        return false;
    }
    ctx.expert_id = expert_id;
    return true;
}

// =================================================================================================
// THE CUBE's STEPS
// =================================================================================================

// STAGE 3 head, both phases. ONE wait, not one per AIV: mode 2's AIV -> AIC
// direction increments the AIC's counter once BOTH AIVs have set. Scoped to
// PIPE_MTE1, the pipe that reads the slot, then drained so the record is true.
__aicore__ inline void StepCube_WaitAivReady(
    const CubeStepContext& ctx, std::uint32_t slot, std::uint32_t k) {
    (void)ctx;  // present for signature symmetry with the other StepCube_ steps
    DSV4_TRACE("C s3 k=%u WAIT StepReady slot=%u\n", k, slot);
    CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
        static_cast<uint16_t>(kCubeFlagStepReady + slot));
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C s3 k=%u GOT StepReady slot=%u\n", k, slot);
}

// STAGE 3 body, phase 1: both weight streams against the one token tile.
// CubeCloseAccumulationStep carries the K-axis accumulation interlock, which is
// a documented hardware contract below (m/16)*(n/16) < 10 and is NOT subsumed
// by the PIPE_ALL barrier after it -- it must sit between these Mmads' L0C write
// and the next step's read.
__aicore__ inline void StepCube_MmadAccumulate(
    const CubeStepContext& ctx, std::uint32_t slot, std::uint32_t k, bool is_first) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1a[slot].Get<bfloat16_t>(), kCubeMTile, kCubeKSub);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1w[slot][0].Get<bfloat16_t>(), kCubeKSub, kCubeRSub);
    CubeLoadWeightsToL0B(ctx.l0b_up, ctx.l1w[slot][1].Get<bfloat16_t>(), kCubeKSub, kCubeRSub);
    DSV4_TRACE("C s3 k=%u loaded L0A/L0B\n", k);

    SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
    WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeRSub, kCubeKSub, is_first);
    CubeMmadAccumulate(ctx.l0c_up, ctx.l0a, ctx.l0b_up,
                       kCubeMTile, kCubeRSub, kCubeKSub, is_first);
    DSV4_TRACE("C s3 k=%u mmad init=%u\n", k, (unsigned)is_first);

    CubeCloseAccumulationStep();
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C s3 k=%u closed\n", k);
}

// STAGE 3 body, phase 2: one weight stream, and the A operand is the token tile
// the AIVs pushed out of the resident activation plane.
__aicore__ inline void StepCube_MmadDown(
    const CubeStepContext& ctx, std::uint32_t slot, std::uint32_t r, bool is_first) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1a2[slot].Get<bfloat16_t>(), kCubeMTile, kCubeRSub2);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1w2[slot].Get<bfloat16_t>(), kCubeRSub2, kCubeHSub);
    DSV4_TRACE("C s3 r=%u loaded L0A/L0B\n", r);

    SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
    WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeHSub, kCubeRSub2, is_first);
    DSV4_TRACE("C s3 r=%u mmad init=%u\n", r, (unsigned)is_first);

    CubeCloseAccumulationStep();
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C s3 r=%u closed\n", r);
}

// STAGE 3 tail, both phases. Production gates this on
// `k + kCubeRingSlots < fwd_steps`, which hands the token for step k to step
// k + 2. Here it is unconditional and matches the AIVs' unconditional stage 4
// wait one for one, so neither side can be a lap ahead.
__aicore__ inline void StepCube_ReleaseAiv(
    const CubeStepContext& ctx, std::uint32_t slot, std::uint32_t k) {
    (void)ctx;  // present for signature symmetry with the other StepCube_ steps
    CubeReleaseToAllAivs<PIPE_MTE1>(static_cast<uint16_t>(kCubeFlagStepFree + slot));
    DSV4_TRACE("C s3 k=%u SET StepFree slot=%u\n", k, slot);
}

// The contraction is complete; let the Fixpipe read L0C.
__aicore__ inline void StepCube_DrainToFixpipe(
    const CubeStepContext& ctx, std::uint32_t blk) {
    (void)ctx;  // present for signature symmetry with the other StepCube_ steps
    SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
    WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C M_FIX done blk=%u\n", blk);
}

// L0C -> both AIVs' UBs, then the handover and its acknowledgement.
// CubeMoveOutAndHandOver is INLINED here rather than called, so the trace can
// print between the Fixpipe, the Product release and the LandingFree wait: the
// cross-core handover is a prime suspect and a probe that could not print
// across those three edges would be blind to exactly the edge in question.
// The destination is UB offset 0 in EACH AIV's own address space and the layout
// carries the FULL M -- see CubeL0c2UbTraitDualDstM.
__aicore__ inline void StepCube_FixpipeHandover(
    const CubeStepContext& ctx, __cc__ float* l0c_mem,
    std::uint32_t extent_m, std::uint32_t extent_n, std::uint32_t tag) {
    (void)ctx;  // present for signature symmetry with the other StepCube_ steps
    auto l0C = Te::MakeTensor(
        Te::MakeMemPtr(l0c_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(extent_m, extent_n));
    auto ub = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kCubeLandingUbOffset)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn>(extent_m, extent_n));
    Te::FixpipeParams fixParams;
    DSV4_TRACE("C fix%u begin n=%u\n", tag, extent_n);
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(fixParams),
             ub, l0C);
    PipeBarrier<PIPE_FIX>();
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C fix%u drained\n", tag);

    CubeReleaseToAllAivs<PIPE_FIX>(kCubeFlagProduct);
    DSV4_TRACE("C fix%u SET Product x%u\n", tag, (unsigned)kCubeAivPerAic);
    DSV4_TRACE("C fix%u WAIT LandingFree\n", tag);
    CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C fix%u GOT LandingFree\n", tag);
}

// The AIVs' third LandingFree, raised only after SwiGLU and its plane store have
// completely finished. Until it arrives the Cube does not advance `inter_base`
// and does not enter phase 2, so the activation plane phase 2 reads is complete
// on both AIVs.
__aicore__ inline void StepCube_WaitBlockDone(
    const CubeStepContext& ctx, std::uint32_t inter_base) {
    (void)ctx;  // present for signature symmetry with the other StepCube_ steps
    DSV4_TRACE("C p1 WAIT BlockDone ib=%u\n", inter_base);
    CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(kCubeFlagLandingFree);
    PipeBarrier<PIPE_ALL>();
    DSV4_TRACE("C p1 GOT BlockDone ib=%u\n", inter_base);
}

// =================================================================================================
// THE VECTOR CORE: the root pipeline
// =================================================================================================
// TWO THINGS THE LOOP NEST BELOW DOES NOT SAY, BECAUSE SAYING THEM THERE WOULD
// COST THE READABILITY THE NEST EXISTS FOR.
//
// THE RING STATE IS DEAD, AND IS KEPT ONLY TO SATISFY THE TWO PRODUCTION
// PREFETCH SIGNATURES. AivPrefetchFwdStep and AivPrefetchDownStep CONSUME
// `stage_live[slot]` -- a deferred WaitFlag<V_MTE2> -- but never raise it; the
// raise lives in AivConvertAndPushFwdStep, which this lockstep transcription
// does not call. So one AivResetRing leaves both arrays false for the whole run
// and the two helpers take no deferred fence, which is what "the ring overlap is
// removed" means mechanically. AivDrainRing is correspondingly never called:
// there is nothing outstanding for it to drain, and waiting on an unraised token
// is the one thing in this file that would hang on its own.
//
// BOTH Tile[] INDICES IN THE NARRATION ARE k, AND THAT IS THE POINT. In the
// production pipeline the copy runs a tile AHEAD of the dequant -- Tile[k+1]
// copied while Tile[k] converts -- and this lockstep probe removed that overlap,
// so one step copies and converts the SAME tile. Printing a staggered pair would
// describe a schedule this file does not run, which is exactly the class of
// mistake the probe was rewritten to stop making.
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
    // Kept in step with production's RunVectorCore: an AIV's GetBlockIdx()
    // counts AIVs, not AI Core groups, so the group index needs the divide.
    // The id census printed by the entry point is what established that.
    const std::uint32_t core_idx =
        static_cast<std::uint32_t>(GetBlockIdx()) / static_cast<std::uint32_t>(GetSubBlockNum());
    const std::uint32_t col_base   = sub_idx * kCubeKHalf;
    const std::uint32_t fwd_steps  = geom.hidden / kCubeKSub;
    const std::uint32_t down_steps = geom.inter / kCubeRSub2;

    AivRingState ring;  // dead; see the banner above
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

    DSV4_TRACE("V enter sub=%u tiles=%u h=%u i=%u\n", sub_idx, total_tiles, geom.hidden, geom.inter);

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        if (!AivAdmitTile(ctx, tile_idx)) {
            DSV4_TRACE("V tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("V tile=%u expert=%u rows=%u\n", tile_idx, ctx.expert_id, ctx.valid_rows);

        // --- PHASE 1: x @ W_gate and x @ W_up, then SwiGLU into the plane ----------------------
        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("V p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot = k % kCubeRingSlots;
                ++ctx.step_no;
                DSV4_TRACE("[AIV%u] Step %u blk=%u. Tile[%u]: copy (x,Wg,Wu) GM->UB; "
                           "Tile[%u]: dequant UB->UB\n",
                           sub_idx, ctx.step_no, inter_base, k, k);
                StepAiv_PrefetchAndDequant  (ctx, slot, inter_base, k);
                StepAiv_PushL1AndSignalReady(ctx, slot, col_base, k);
                StepAiv_WaitStepComplete    (ctx, slot, k);
            }
            StepAiv_EpilogueGate   (ctx, inter_base);
            StepAiv_EpilogueUp     (ctx, inter_base);
            StepAiv_BlockRendezvous(ctx, inter_base);
        }

        // --- PHASE 2: act @ W_down, out to GM -------------------------------------------------
        DSV4_TRACE("V p2 enter steps=%u store_rows=%u\n", down_steps, ctx.store_rows);
        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("V p2 blk hb=%u\n", hidden_base);
            for (std::uint32_t r = 0; r < down_steps; ++r) {
                const std::uint32_t slot = r % kCubeRingSlots;
                ++ctx.step_no;
                DSV4_TRACE("[AIV%u] Step %u blk=%u. Tile[%u]: copy (Wd) GM->UB; "
                           "Tile[%u]: dequant UB->UB\n",
                           sub_idx, ctx.step_no, hidden_base, r, r);
                StepAiv_PrefetchDown          (ctx, slot, hidden_base, r);
                StepAiv_PushDownAndSignalReady(ctx, slot, r);
                StepAiv_WaitStepComplete      (ctx, slot, r);
            }
            StepAiv_StoreGMAndBlockDone(ctx, hidden_base);
        }
        DSV4_TRACE("V tile=%u DONE\n", tile_idx);
    }
    DSV4_TRACE("V exit\n");
}

// =================================================================================================
// THE CUBE CORE: the root pipeline
// =================================================================================================
// Read this against TraceRunVectorCore line by line. Phase 1 step k is
// Prefetch / Push / Wait there and Wait / Mmad / Release here, and the two
// epilogues pair off the same way: three AIV LandingFree sets against the two
// Fixpipe handovers plus the block rendezvous.
//
// THE AIC BRACKETS THE STEP, which is why [STEP n Begin] and [STEP n Completed]
// are printed here and on no other core: the AIC is the one core there is
// exactly one of per group, and the one that ENDS the step -- the AIVs leave
// step k only when its StepFree release reaches them. The next iteration's
// Begin is the (n+1) line.
//
// THE TWO PHASE 1 HANDOVERS ARE ORDERED, fix1 (gate) then fix2 (up), each
// closed by its own full drain, so the order the two Products reach the AIVs is
// fixed rather than whatever the Fixpipe queue happens to retire first. The
// AIVs consume them in exactly that order -- Mins on gate, then Maxs on up --
// and getting them the other way round is a wrong answer, not a hang. Phase 2's
// single handover needs no such ordering, and its LandingFree wait IS the block
// rendezvous: the AIVs raise it only once their GM store has drained, so the
// Cube leaves the block no earlier than they do.
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

    DSV4_TRACE("C enter tiles=%u h=%u i=%u\n", total_tiles, geom.hidden, geom.inter);

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        if (!CubeAdmitTile(ctx, tile_idx)) {
            DSV4_TRACE("C tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("C tile=%u expert=%u\n", tile_idx, ctx.expert_id);

        // --- PHASE 1 --------------------------------------------------------------------------
        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("C p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot = k % kCubeRingSlots;
                ++ctx.step_no;
                DSV4_TRACE("[STEP%u Begin]\n", ctx.step_no);
                DSV4_TRACE("[AIC0] Step %u blk=%u. Tile[%u]: gemm (x*Wg, x*Wu)\n",
                           ctx.step_no, inter_base, k);
                StepCube_WaitAivReady  (ctx, slot, k);
                StepCube_MmadAccumulate(ctx, slot, k, k == 0u);
                StepCube_ReleaseAiv    (ctx, slot, k);
                DSV4_TRACE("[STEP%u Completed]\n", ctx.step_no);
            }
            StepCube_DrainToFixpipe(ctx, inter_base);
            StepCube_FixpipeHandover(ctx, l0c_gate, kCubeMTile, kCubeRSub, 1u);
            StepCube_FixpipeHandover(ctx, l0c_up,   kCubeMTile, kCubeRSub, 2u);
            StepCube_WaitBlockDone  (ctx, inter_base);
            DSV4_TRACE("C p1 blk ib=%u DONE\n", inter_base);
        }

        // --- PHASE 2 --------------------------------------------------------------------------
        DSV4_TRACE("C p2 enter steps=%u\n", down_steps);
        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("C p2 blk hb=%u\n", hidden_base);
            for (std::uint32_t r = 0; r < down_steps; ++r) {
                const std::uint32_t slot = r % kCubeRingSlots;
                ++ctx.step_no;
                DSV4_TRACE("[STEP%u Begin]\n", ctx.step_no);
                DSV4_TRACE("[AIC0] Step %u blk=%u. Tile[%u]: gemm (act*Wd)\n",
                           ctx.step_no, hidden_base, r);
                StepCube_WaitAivReady(ctx, slot, r);
                StepCube_MmadDown    (ctx, slot, r, r == 0u);
                StepCube_ReleaseAiv  (ctx, slot, r);
                DSV4_TRACE("[STEP%u Completed]\n", ctx.step_no);
            }
            StepCube_DrainToFixpipe(ctx, hidden_base);
            StepCube_FixpipeHandover(ctx, l0c_gate, kCubeMTile, kCubeHSub, 3u);
            DSV4_TRACE("C p2 blk hb=%u DONE\n", hidden_base);
        }
        DSV4_TRACE("C tile=%u DONE\n", tile_idx);
    }
    DSV4_TRACE("C exit\n");
}

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

    // ===================================================================
    // THE BLOCK-INDEX CENSUS, printed before any flag is touched
    // ===================================================================
    // The first trace run showed `[AIV Block 1/2] V enter` followed straight
    // by `V exit`: the second AIV's tile loop
    //
    //     for (tile_idx = GetBlockIdx(); tile_idx < total_tiles; ...)
    //
    // never executed at total_tiles == 1, which is only possible if
    // GetBlockIdx() returned a nonzero value there. That contradicts the
    // assumption the schedule is built on, written down in
    // dsv4_moe_prefill_cube_inductive_kernel.cpp as "one AIC is paired with
    // TWO AIVs, and both of them share a GetBlockIdx()".
    //
    // That was an INFERENCE from a loop not executing. These prints make it a
    // MEASUREMENT, and they also decide the repair: if the AIV's GetBlockIdx()
    // counts AIVs rather than AI Core groups, the group index is
    // GetBlockIdx() / GetSubBlockNum() -- `cand` below -- and that is what the
    // tile loop and the AIC must agree on. `derived` is what the production
    // code uses today, printed beside it so the two are comparable at a
    // glance.
    //
    // Printed from BOTH branches: the AIC's own indices are half the contract
    // and a census of the AIVs alone could not show a mismatch.
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

        // Declaration ORDER is the ABI with the Cube core (q_landing at UB
        // offset 0); transcribed exactly, not tidied.
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
        // The PRODUCTION priming pair, kept. It is a SetFlag immediately
        // followed by its WaitFlag, so it is balanced on its own and is not
        // part of the ring overlap this transcription removes -- it primes the
        // V_MTE2 token the production AivConvertAndPushFwdStep would consume,
        // and the lockstep path simply never consumes one. The no-prime
        // variant is a separate probe and the two deltas are deliberately not
        // combined.
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
