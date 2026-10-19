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
// WHY THE TWO CORE ROUTINES ARE TRANSCRIBED
// ===========================================================================
// The trace points have to sit BETWEEN the stages inside RunVectorCore and
// RunCubeCore, which are production functions. op_kernel/ stays clean
// (CLAUDE.md section 1.1) -- no probe, no branch and no #ifdef SIM reaches the
// shipping kernel -- so the two routines are transcribed here with trace calls
// interleaved and NOTHING ELSE CHANGED. Every helper they call
// (AivPrefetchFwdStep, AivConvertAndPushFwdStep, AivDrainRing, AivApplySwiGLU,
// CubeLoadTokensToL0A, CubeLoadWeightsToL0B, CubeMmadAccumulate,
// CubeCloseAccumulationStep, CubeReleaseToAllAivs) is the production one,
// reached by textual inclusion, including the ones in its anonymous namespace.
//
// CubeMoveOutAndHandOver is the ONE helper whose body is also inlined here
// rather than called: the cross-core handover it performs is a prime suspect,
// and a trace that could not print between its Fixpipe, its Product release
// and its LandingFree wait would be blind to exactly the edge in question. The
// instruction sequence is copied verbatim.
//
// A TRANSCRIPTION CAN DRIFT FROM ITS ORIGINAL, and that is the cost being
// accepted here. It is bounded: this probe answers one question and is not a
// gate, and the delta it is allowed to carry is "trace calls only".

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
// THE VECTOR CORE, transcribed with trace points
// =================================================================================================
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
    const std::uint32_t row_base    = sub_idx * kCubeMHalf;

    DSV4_TRACE("V enter sub=%u tiles=%u h=%u i=%u\n", sub_idx, total_tiles, geom.hidden, geom.inter);

    AivRingState ring;

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;
        const std::uint32_t remaining       = geom.total_tokens - token_row_start;
        const std::uint32_t valid_rows      = remaining < geom.token_tile ? remaining : geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            DSV4_TRACE("V tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("V tile=%u expert=%u rows=%u\n", tile_idx, expert_id, valid_rows);
        const std::uint64_t expert_offset = static_cast<std::uint64_t>(expert_id);

        __gm__ std::uint8_t* gate_packed = Bytes(gate_weights_packed) + expert_offset * geom.gate_up_packed_stride;
        __gm__ std::uint8_t* gate_scales = Bytes(gate_weight_scales)  + expert_offset * geom.gate_up_scale_stride;
        __gm__ std::uint8_t* up_packed   = Bytes(up_weights_packed)   + expert_offset * geom.gate_up_packed_stride;
        __gm__ std::uint8_t* up_scales   = Bytes(up_weight_scales)    + expert_offset * geom.gate_up_scale_stride;
        __gm__ std::uint8_t* down_packed = Bytes(down_weights_packed) + expert_offset * geom.down_packed_stride;
        __gm__ std::uint8_t* down_scales = Bytes(down_weight_scales)  + expert_offset * geom.down_scale_stride;

        // --- PHASE 1 ---------------------------------------------------------------------------
        const std::uint32_t fwd_steps = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("V p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            AivResetRing(ring);

            AivPrefetchFwdStep(tokens_in_gm, token_row_start, gate_packed, gate_scales,
                               up_packed, up_scales, inter_base, 0u, valid_rows,
                               sub_idx, geom, buf, 0u, ring);
            DSV4_TRACE("V p1 prologue prefetch done\n");

            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot      = k % kCubeRingSlots;
                const std::uint32_t next_slot = (k + 1u) % kCubeRingSlots;

                if (k + 1u < fwd_steps) {
                    AivPrefetchFwdStep(tokens_in_gm, token_row_start, gate_packed, gate_scales,
                                       up_packed, up_scales, inter_base,
                                       (k + 1u) * kCubeKSub, valid_rows,
                                       sub_idx, geom, buf, next_slot, ring);
                    DSV4_TRACE("V p1 k=%u prefetched %u\n", k, next_slot);
                }

                if (k >= kCubeRingSlots) {
                    DSV4_TRACE("V p1 k=%u WAIT StepFree slot=%u\n", k, slot);
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                    DSV4_TRACE("V p1 k=%u GOT StepFree\n", k);
                }

                AivConvertAndPushFwdStep(l1a_buf[slot].Get<bfloat16_t>(),
                                         l1w_buf[slot][0].Get<bfloat16_t>(),
                                         l1w_buf[slot][1].Get<bfloat16_t>(),
                                         sub_idx, buf, slot, ring);
                DSV4_TRACE("V p1 k=%u pushed slot=%u\n", k, slot);

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
                DSV4_TRACE("V p1 k=%u SET StepReady slot=%u\n", k, slot);
            }

            AivDrainRing(ring);
            DSV4_TRACE("V p1 drained ib=%u\n", inter_base);

            const std::uint32_t elems = kCubeMHalf * kCubeRSub;

            DSV4_TRACE("V p1 WAIT Product (gate)\n");
            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            DSV4_TRACE("V p1 GOT Product (gate)\n");
            Mins(buf.gate_acc, buf.landing, geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
            DSV4_TRACE("V p1 SET LandingFree (1st)\n");

            DSV4_TRACE("V p1 WAIT Product (up)\n");
            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            DSV4_TRACE("V p1 GOT Product (up)\n");
            Maxs(buf.up_acc, buf.landing, -geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
            DSV4_TRACE("V p1 SET LandingFree (2nd)\n");
            Mins(buf.up_acc, buf.up_acc, geom.swiglu_limit, elems);

            AivApplySwiGLU(buf, inter_base, geom);
            DSV4_TRACE("V p1 swiglu done ib=%u\n", inter_base);
        }

        // --- PHASE 2 ---------------------------------------------------------------------------
        const std::uint32_t down_steps = geom.inter / kCubeRSub2;
        const std::uint32_t owned      = valid_rows > row_base ? (valid_rows - row_base) : 0u;
        const std::uint32_t store_rows = owned < kCubeMHalf ? owned : kCubeMHalf;

        DSV4_TRACE("V p2 enter steps=%u store_rows=%u\n", down_steps, store_rows);

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("V p2 blk hb=%u\n", hidden_base);
            AivResetRing(ring);

            AivPrefetchDownStep(down_packed, down_scales, hidden_base, 0u, sub_idx,
                                geom, buf, 0u, ring);

            for (std::uint32_t r = 0; r < down_steps; ++r) {
                const std::uint32_t slot      = r % kCubeRingSlots;
                const std::uint32_t next_slot = (r + 1u) % kCubeRingSlots;

                if (r + 1u < down_steps) {
                    AivPrefetchDownStep(down_packed, down_scales, hidden_base,
                                        (r + 1u) * kCubeRSub2, sub_idx,
                                        geom, buf, next_slot, ring);
                }

                if (r >= kCubeRingSlots) {
                    DSV4_TRACE("V p2 r=%u WAIT StepFree slot=%u\n", r, slot);
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                    DSV4_TRACE("V p2 r=%u GOT StepFree\n", r);
                }

                AivConvertAndPushDownStep(l1a2_buf[slot].Get<bfloat16_t>(),
                                          l1w2_buf[slot].Get<bfloat16_t>(),
                                          r * kCubeRSub2, sub_idx, geom, buf, slot, ring);

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
                DSV4_TRACE("V p2 r=%u SET StepReady slot=%u\n", r, slot);
            }

            AivDrainRing(ring);
            DSV4_TRACE("V p2 drained hb=%u\n", hidden_base);

            DSV4_TRACE("V p2 WAIT Product\n");
            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStore>(kCubeFlagProduct);
            DSV4_TRACE("V p2 GOT Product\n");

            if (store_rows > 0u) {
                DataCopyParams out_burst;
                out_burst.blockCount = static_cast<std::uint16_t>(store_rows);
                out_burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
                out_burst.srcStride  = 0;
                out_burst.dstStride  = static_cast<std::uint16_t>(((geom.hidden - kCubeHSub) * 4u) / 32u);
                DataCopy(tokens_out_gm[(static_cast<std::uint64_t>(token_row_start + row_base) *
                                        geom.hidden) + hidden_base],
                         buf.landing, out_burst);
                DSV4_TRACE("V p2 stored hb=%u\n", hidden_base);
            }

            CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
            DSV4_TRACE("V p2 SET LandingFree hb=%u\n", hidden_base);
        }
        DSV4_TRACE("V tile=%u DONE\n", tile_idx);
    }
    DSV4_TRACE("V exit\n");
}

// =================================================================================================
// THE CUBE CORE, transcribed with trace points
// =================================================================================================
// CubeMoveOutAndHandOver is INLINED below rather than called, so the trace can
// print between the Fixpipe, the Product release and the LandingFree wait.
__aicore__ inline void TraceCubeMoveOutAndHandOver(
    __cc__ float* l0c_mem, std::uint32_t m_extent, std::uint32_t n_extent, std::uint32_t tag) {
    auto l0C = Te::MakeTensor(
        Te::MakeMemPtr(l0c_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(m_extent, n_extent));
    auto ub = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kCubeLandingUbOffset)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn>(m_extent, n_extent));
    Te::FixpipeParams fixParams;
    DSV4_TRACE("C fix%u begin n=%u\n", tag, n_extent);
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(fixParams),
             ub, l0C);
    PipeBarrier<PIPE_FIX>();
    DSV4_TRACE("C fix%u drained\n", tag);
    CubeReleaseToAllAivs<PIPE_FIX>(kCubeFlagProduct);
    DSV4_TRACE("C fix%u SET Product x%u\n", tag, (unsigned)kCubeAivPerAic);
    DSV4_TRACE("C fix%u WAIT LandingFree\n", tag);
    CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
    DSV4_TRACE("C fix%u GOT LandingFree\n", tag);
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

    __ca__ bfloat16_t l0a[kCubeL0AElems];
    __cb__ bfloat16_t l0b_gate[kCubeL0BElems];
    __cb__ bfloat16_t l0b_up[kCubeL0BElems];
    __cc__ float      l0c_gate[kCubeL0CElems];
    __cc__ float      l0c_up[kCubeL0CElems];

    DSV4_TRACE("C enter tiles=%u h=%u i=%u\n", total_tiles, geom.hidden, geom.inter);

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            DSV4_TRACE("C tile=%u SKIP\n", tile_idx);
            continue;
        }
        DSV4_TRACE("C tile=%u expert=%u\n", tile_idx, expert_id);

        // --- PHASE 1 ---------------------------------------------------------------------------
        const std::uint32_t fwd_steps = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            DSV4_TRACE("C p1 blk ib=%u steps=%u\n", inter_base, fwd_steps);
            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot = k % kCubeRingSlots;

                DSV4_TRACE("C p1 k=%u WAIT StepReady slot=%u\n", k, slot);
                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
                DSV4_TRACE("C p1 k=%u GOT StepReady\n", k);

                CubeLoadTokensToL0A(l0a, l1a_buf[slot].Get<bfloat16_t>(), kCubeMTile, kCubeKSub);
                CubeLoadWeightsToL0B(l0b_gate, l1w_buf[slot][0].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);
                CubeLoadWeightsToL0B(l0b_up, l1w_buf[slot][1].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);
                DSV4_TRACE("C p1 k=%u loaded L0A/L0B\n", k);

                if (k + kCubeRingSlots < fwd_steps) {
                    CubeReleaseToAllAivs<PIPE_MTE1>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                    DSV4_TRACE("C p1 k=%u SET StepFree slot=%u\n", k, slot);
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate,
                                   kCubeMTile, kCubeRSub, kCubeKSub, k == 0u);
                CubeMmadAccumulate(l0c_up, l0a, l0b_up,
                                   kCubeMTile, kCubeRSub, kCubeKSub, k == 0u);
                DSV4_TRACE("C p1 k=%u mmad init=%u\n", k, (unsigned)(k == 0u));

                CubeCloseAccumulationStep();
                DSV4_TRACE("C p1 k=%u closed\n", k);
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            DSV4_TRACE("C p1 M_FIX done ib=%u\n", inter_base);

            TraceCubeMoveOutAndHandOver(l0c_gate, kCubeMTile, kCubeRSub, 1u);
            TraceCubeMoveOutAndHandOver(l0c_up, kCubeMTile, kCubeRSub, 2u);
            DSV4_TRACE("C p1 blk ib=%u DONE\n", inter_base);
        }

        // --- PHASE 2 ---------------------------------------------------------------------------
        const std::uint32_t down_steps = geom.inter / kCubeRSub2;
        DSV4_TRACE("C p2 enter steps=%u\n", down_steps);

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            DSV4_TRACE("C p2 blk hb=%u\n", hidden_base);
            for (std::uint32_t r = 0; r < down_steps; ++r) {
                const std::uint32_t slot = r % kCubeRingSlots;

                DSV4_TRACE("C p2 r=%u WAIT StepReady slot=%u\n", r, slot);
                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
                DSV4_TRACE("C p2 r=%u GOT StepReady\n", r);

                CubeLoadTokensToL0A(l0a, l1a2_buf[slot].Get<bfloat16_t>(),
                                    kCubeMTile, kCubeRSub2);
                CubeLoadWeightsToL0B(l0b_gate, l1w2_buf[slot].Get<bfloat16_t>(),
                                     kCubeRSub2, kCubeHSub);

                if (r + kCubeRingSlots < down_steps) {
                    CubeReleaseToAllAivs<PIPE_MTE1>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                    DSV4_TRACE("C p2 r=%u SET StepFree slot=%u\n", r, slot);
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate,
                                   kCubeMTile, kCubeHSub, kCubeRSub2, r == 0u);
                CubeCloseAccumulationStep();
                DSV4_TRACE("C p2 r=%u closed\n", r);
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);

            TraceCubeMoveOutAndHandOver(l0c_gate, kCubeMTile, kCubeHSub, 3u);
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
        // The PRODUCTION priming pair, kept: this probe changes only what it
        // PRINTS. The no-prime variant is a separate probe and the two deltas
        // are deliberately not combined.
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
