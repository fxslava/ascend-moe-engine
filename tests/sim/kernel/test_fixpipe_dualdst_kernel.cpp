/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- STEP 1 of the Fixpipe dual-destination refactor: one isolated
// low-level GEMM, built entirely from Tensor API primitives, with NO Matmul
// framework anywhere in it.
//
// The reason it cannot use the framework, and the data path it builds
// instead, are in kernel/dsv4_prefill_cube_dualdst.h. This file answers one
// question only: does L0C -> UB with DUAL_DST_SPLIT_M and an in-line
// float -> bfloat16_t cast actually deliver rows 0..7 to AIV0's UB and rows
// 8..15 to AIV1's UB. It says nothing about the production schedule, and the
// MXFP4 dequantization path is deliberately untouched -- B arrives already in
// BF16 from the host.
//
// WHO RUNS WHAT
//
//   AIC   the whole Cube chain: GM -> L1 -> L0A/L0B -> Mmad -> L0C, then the
//         Fixpipe move-out. Issued MONOLITHICALLY -- there is no
//         GetSubBlockIdx() gating on this core, because GetSubBlockIdx()
//         returns 0 on the AIC and gating on it would be a test that is
//         always true. The dual-destination split is the HARDWARE's job, not
//         a loop over sub-blocks.
//
//   AIV   waits on its own cross-core flag, then copies its half of the
//         landing zone out to GM. Both AIVs run; neither touches the Cube.
//         This is the one place GetSubBlockIdx() is read, which is the only
//         place it means anything.
//
// CopyL0C2UB::Copy is itself `if ASCEND_IS_AIV { return; }` internally
// (impl/tensor_api/arch/cube/l0c_to_ub/copy.h), so the move-out is AIC-only
// whether or not the caller guards it. It is guarded here anyway, because the
// L1/L0 buffers it reads from only exist on the Cube core.

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "dsv4_prefill_cube_dualdst.h"

namespace moe::kernels::arch35::dualdst {

using namespace AscendC;

// NOT `using namespace AscendC::Te`. The Tensor API redeclares RoundMode,
// MmadParams and Mmad in its own namespace, and pulling both in unqualified
// makes every one of them ambiguous. Te symbols are spelled out.
namespace Te = AscendC::Te;

namespace {

// The trait the whole probe exists to exercise.
constexpr Te::CopyL0C2UBTrait kL0c2UbTrait = {
    Te::RoundMode::DEFAULT,
    false,             // enableRelu
    false,             // enableChannelSplit
    Te::DUAL_DST_SPLIT_M   // bisect M = 16 -> 8 rows to SUB BLOCK0, 8 to SUB BLOCK1
};

struct CopyL0C2UBTraitDualDstM {
    using TraitType = Te::CopyL0C2UBTrait;
    static constexpr const TraitType value = kL0c2UbTrait;
};

// The pipe the AIV's CrossCoreWaitFlag scopes its block to: the pipeline that
// CONSUMES what the flag protects. See dsv4_prefill_cube_dualdst.h.
inline constexpr pipe_t kDualDstWaitPipeAiv = PIPE_MTE3;  // DataCopy UB -> GM

// `region` picks the writer's own cache line; `slot` is the word inside it.
// See the header for why a shared line loses writes.
__aicore__ inline void WriteStatus(GM_ADDR status, std::uint32_t region,
                                   std::uint32_t slot, std::uint32_t value) {
    GlobalTensor<uint32_t> s;
    s.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status), kDualDstStatusWords);
    s.SetValue(region * kDualDstStatusStride + slot, value);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
                             DcciDst::CACHELINE_OUT>(s);
}

}  // namespace

extern "C" __global__ __aicore__ void test_fixpipe_dualdst_kernel(
    GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR out_gm, GM_ADDR status_gm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (unlikely(a_gm == nullptr || b_gm == nullptr || out_gm == nullptr ||
                 status_gm == nullptr)) {
        return;
    }

    const std::uint32_t stage = *reinterpret_cast<const __gm__ std::uint32_t*>(
        reinterpret_cast<__gm__ std::uint8_t*>(status_gm) +
        kDualDstStageWord * sizeof(std::uint32_t));

    if ASCEND_IS_AIC {
        // STAGE 4 ONLY: the rendezvous the reference opens with. Both AIVs
        // raise kDualDstFlagOperands before the Cube starts, and mode 2's
        // AIV -> AIC direction only releases once BOTH have, so this is one
        // wait for two setters.
        if (stage >= kDualDstStageRendezvous) {
            // Mode 2's AIV -> AIC direction releases only once BOTH AIVs have
            // set, so this is one wait for two setters.
            //
            // THE BARE SPELLING, because it is the one the passing
            // configuration carries. It briefly became
            // `CrossCoreWaitFlag<kDualDstSyncMode, PIPE_MTE2>` as part of
            // naming the mode and the pipe at every site, and that version
            // hung -- but the bisection in dsv4_prefill_cube_dualdst.h shows
            // the hang was the flagIds moving to 8/9 at the same time, NOT
            // this spelling. The explicit form here is therefore untested
            // rather than wrong; it is simply not what has been measured
            // green, so it is not what ships in the probe.
            CrossCoreWaitFlag(kDualDstFlagOperands);
        }

        __cbuf__ bfloat16_t l1a[kDualDstM * kDualDstK];
        __cbuf__ bfloat16_t l1b[kDualDstK * kDualDstN];
        __ca__   bfloat16_t l0a[kDualDstM * kDualDstK];
        __cb__   bfloat16_t l0b[kDualDstK * kDualDstN];
        __cc__   float      l0c[kDualDstM * kDualDstN];

        // --- GM -> L1. A becomes NZ, B becomes ZN, both by the inline
        //     ND2NZ / ND2ZN conversion this path derives from the layouts.
        auto gmA = Te::MakeTensor(Te::MakeMemPtr(reinterpret_cast<__gm__ bfloat16_t*>(a_gm)),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn>(kDualDstM, kDualDstK));
        auto l1A = Te::MakeTensor(Te::MakeMemPtr(l1a),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(kDualDstM, kDualDstK));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}, Te::CopyGM2L1TraitDefault{}), l1A, gmA);

        auto gmB = Te::MakeTensor(Te::MakeMemPtr(reinterpret_cast<__gm__ bfloat16_t*>(b_gm)),
                                  Te::MakeFrameLayout<Te::NDExtLayoutPtn>(kDualDstK, kDualDstN));
        auto l1B = Te::MakeTensor(Te::MakeMemPtr(l1b),
                                  Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(kDualDstK, kDualDstN));
        Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}, Te::CopyGM2L1TraitDefault{}), l1B, gmB);

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        // --- L1 -> L0A / L0B, both non-transpose (NZ->NZ and ZN->ZN).
        auto l0A = Te::MakeTensor(Te::MakeMemPtr(l0a),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(kDualDstM, kDualDstK));
        Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}, Te::CopyL12L0ATraitDefault{}), l0A, l1A);

        auto l0B = Te::MakeTensor(Te::MakeMemPtr(l0b),
                                  Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(kDualDstK, kDualDstN));
        Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}, Te::CopyL12L0BTraitDefault{}), l0B, l1B);

        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        // --- Mmad. ccubeInitVal = true because this is the first (and only)
        //     K block: with false, C would start on whatever L0C held.
        auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(kDualDstM, kDualDstN));
        Te::MmadParams mmadParams(static_cast<uint16_t>(kDualDstM),
                              static_cast<uint16_t>(kDualDstN),
                              static_cast<uint16_t>(kDualDstK),
                              0, true);
        Te::Mmad(Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{}).with(mmadParams),
                 l0C, l0A, l0B);

        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

        // --- L0C -> UB, DUAL_DST_SPLIT_M, float -> bfloat16_t in-line.
        //
        // THE DESTINATION LAYOUT CARRIES THE FULL M, NOT M/2 -- even though
        // each UB only receives M/2 rows and the landing buffer is only
        // M/2 * N elements.
        //
        // The dual-destination doc describes the raw instruction field as
        // "the size along the M dimension RECEIVED BY EACH destination
        // Unified Buffer", which reads as M/2. The Tensor API does not pass
        // the layout's M through to that field unchanged: MEASURED, a dst
        // layout of M/2 = 8 delivered FOUR rows to each UB -- rows 0..3 to
        // sub-block 0 and rows 4..7 to sub-block 1, the rest zero. It had
        // halved the 8 again. With the full M = 16 each side gets its 8.
        //
        // This matches the shipping turboquant kernel, which passes the full
        // (even-padded) M as the Fixpipe mSize while allocating only
        // kHalfRows = M/2 rows of destination UB.
        //
        // No `quant` argument: float -> bfloat16_t with no quant parameter is
        // the documented DIRECT CAST path (quant_pre.md -- "When quant is not
        // passed, only non-quantized output or direct cast output from float
        // to half/bfloat16_t is supported"). That is the in-line cast, and it
        // is why no FP32 C ever lands in UB.
        auto ub = Te::MakeTensor(
            Te::MakeMemPtr(reinterpret_cast<__ubuf__ bfloat16_t*>(kDualDstUbOffset)),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(kDualDstM, kDualDstN));
        Te::FixpipeParams fixParams;
        Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CopyL0C2UBTraitDualDstM{}).with(fixParams),
                 ub, l0C);

        // Drain the Fixpipe before the ready flag releases the vector cores
        // onto the rows it is still writing.
        PipeBarrier<PIPE_FIX>();

        // ONE flag for BOTH AIVs: mode 2 is "the AIC and all AIVs of this AI
        // Core". See the header for why the wait side takes no template
        // arguments.
        CrossCoreSetFlag<kDualDstSyncMode, PIPE_FIX>(kDualDstFlagReady);

        WriteStatus(status_gm, kDualDstAicRegion, kDualDstSlotLive, kDualDstAicDone);
    }

    if ASCEND_IS_AIV {
        // The landing zone, claimed through TPipe so its address is the
        // pool base -- which this probe then REPORTS rather than trusts.
        TPipe pipe;
        TQue<TPosition::VECIN, 1> q_landing;
        pipe.InitBuffer(q_landing, 1, kDualDstHalfElems * sizeof(bfloat16_t));
        LocalTensor<bfloat16_t> landing = q_landing.AllocTensor<bfloat16_t>();

        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());

        const std::uint32_t region = kDualDstAivRegion + sub_idx;
        WriteStatus(status_gm, region, kDualDstSlotAddr,
                    static_cast<std::uint32_t>(landing.GetPhyAddr()));
        WriteStatus(status_gm, region, kDualDstSlotSubs,
                    static_cast<std::uint32_t>(GetSubBlockNum()));

        // STAGE 1 stops here. The AIC still runs its whole Cube chain and
        // still raises both flags; what stage 1 removes is the WAIT, so a
        // run that retires at 1 and hangs at 2 indicts the handshake and not
        // the Cube. The flags it leaves unconsumed do not outlive the launch,
        // because nothing else in this probe waits on them.
        if (stage >= kDualDstStageRendezvous) {
            CrossCoreSetFlag<kDualDstSyncMode, PIPE_MTE3>(kDualDstFlagOperands);
        }

        if (stage >= kDualDstStageFull) {
            // Stage 4 names the mode and the consuming pipe; stage 3 keeps
            // the bare default (<modeId 0, PIPE_S>) as the known-bad rung.
            // The block has to land on PIPE_MTE3 because the instruction it
            // must gate is the DataCopy below -- a PIPE_V scope would gate
            // nothing here and let the copy-out read the landing zone before
            // the Fixpipe wrote it.
            if (stage >= kDualDstStageRendezvous) {
                CrossCoreWaitFlag<kDualDstSyncMode, kDualDstWaitPipeAiv>(
                    kDualDstFlagReady);
            } else {
                CrossCoreWaitFlag(kDualDstFlagReady);
            }

            GlobalTensor<bfloat16_t> out;
            out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out_gm), kDualDstOutElems);
            DataCopy(out[sub_idx * kDualDstHalfElems], landing, kDualDstHalfElems);

            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }

        WriteStatus(status_gm, region, kDualDstSlotLive, kDualDstAivDone);
        q_landing.FreeTensor(landing);
    }
}

}  // namespace moe::kernels::arch35::dualdst
