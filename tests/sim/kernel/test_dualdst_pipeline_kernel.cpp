/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- STEP 2: the three-stage asynchronous pipeline, built on the
// dual-destination Fixpipe epilogue STEP 1 proved.
//
// kernel/dsv4_prefill_cube_pipeline.h has the shapes, the flag plan and the
// reason the accumulation stays in L0C. This file is the schedule.
//
//   AIV (both, each owning HALF the weight rows)
//       MTE2  GM -> UB   MXFP4 nibbles + E8M0 scales for tile k+1
//       V     UB -> UB   dequantize tile k to BF16   (production routine,
//                        reused verbatim -- this probe does not reimplement
//                        the FP4 -> BF16 conversion)
//       MTE3  UB -> L1   tile k, with an inline ND2NZ conversion, into its
//                        own row half of the shared ring slot
//
//   AIC
//       MTE2  GM -> L1   A tile k
//       MTE1  L1 -> L0A  A tile k          (NZ -> NZ, no transpose)
//             L1 -> L0B  W tile k          (NZ -> ZN, TRANSPOSED: L1 holds
//                                           W as [N, K] and the Cube needs
//                                           B = W^T as [K, N])
//       M     Mmad into L0C, ccubeInitVal only on k == 0
//       FIX   once, after the last tile: L0C -> both UBs, bf16, split M
//
// L1 IS ALLOCATED THROUGH TPipe ON BOTH CORES, in the same order, OUTSIDE the
// ASCEND_IS_* branches. That is not tidiness: the AIV writes the ring slot
// and the AIC reads it, so the two cores have to land on the same L1 offset,
// and nothing in the toolchain checks that they do. Allocating identically on
// both is the only thing that makes the addresses agree.
//
// L0A/L0B/L0C stay raw __ca__/__cb__/__cc__ arrays inside the AIC branch:
// they are Cube-private, the AIV has no such memory, and TPipe would have to
// be told so.

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include "dsv4_prefill_cube_pipeline.h"

#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

// NOT `using namespace AscendC::Te`: it redeclares RoundMode, MmadParams and
// Mmad, and pulling both in unqualified makes every one ambiguous.
namespace Te = AscendC::Te;

namespace {

constexpr Te::CopyL0C2UBTrait kPipeL0c2UbTrait = {
    Te::RoundMode::DEFAULT,
    false,                  // enableRelu
    false,                  // enableChannelSplit
    Te::DUAL_DST_SPLIT_M    // M = 16 -> 8 rows per AIV
};

struct PipeL0c2UbTraitDualDstM {
    using TraitType = Te::CopyL0C2UBTrait;
    static constexpr const TraitType value = kPipeL0c2UbTrait;
};

__aicore__ inline void PipeStatus(GM_ADDR status, std::uint32_t region,
                                  std::uint32_t slot, std::uint32_t value) {
    GlobalTensor<uint32_t> s;
    s.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status), kPipeStatusWords);
    s.SetValue(region * kPipeStatusStride + slot, value);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
                             DcciDst::CACHELINE_OUT>(s);
}

}  // namespace

extern "C" __global__ __aicore__ void test_dualdst_pipeline_kernel(
    GM_ADDR a_gm, GM_ADDR w_packed_gm, GM_ADDR w_scale_gm,
    GM_ADDR out_gm, GM_ADDR status_gm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (unlikely(a_gm == nullptr || w_packed_gm == nullptr || w_scale_gm == nullptr ||
                 out_gm == nullptr || status_gm == nullptr)) {
        return;
    }

    const std::uint32_t stage = *reinterpret_cast<const __gm__ std::uint32_t*>(
        reinterpret_cast<__gm__ std::uint8_t*>(status_gm) +
        kPipeStageWord * sizeof(std::uint32_t));
    const std::uint32_t tiles = (stage == kPipeStageOneTile) ? 1u : kPipeTiles;
    const bool prefetch = (stage >= kPipeStagePrefetch);

    // --- L1, allocated identically on BOTH cores ---------------------------
    TPipe pipe;
    TBuf<TPosition::A1> l1a_buf;
    TBuf<TPosition::B1> l1w_buf[kPipeRingSlots];
    pipe.InitBuffer(l1a_buf, kPipeM * kPipeKSub * sizeof(bfloat16_t));
    for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
        pipe.InitBuffer(l1w_buf[s], kPipeN * kPipeKSub * sizeof(bfloat16_t));
    }

    // The pipeline each CrossCoreWaitFlag scopes its block to: the one that
    // CONSUMES what the flag protects. See dsv4_prefill_cube_pipeline.h.
    constexpr pipe_t kPipeWaitAicSlot  = PIPE_MTE1;  // CopyL12L0B from the slot
    constexpr pipe_t kPipeWaitAivSlot  = PIPE_MTE3;  // ND2NZ push into the slot
    constexpr pipe_t kPipeWaitAivFinal = PIPE_MTE3;  // DataCopy UB -> GM

    if ASCEND_IS_AIC {
        __ca__ bfloat16_t l0a[kPipeM * kPipeKSub];
        __cb__ bfloat16_t l0b[kPipeKSub * kPipeN];
        __cc__ float      l0c[kPipeM * kPipeN];

        auto l0A = Te::MakeTensor(Te::MakeMemPtr(l0a),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(kPipeM, kPipeKSub));
        auto l0B = Te::MakeTensor(Te::MakeMemPtr(l0b),
                                  Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(kPipeKSub, kPipeN));
        auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c),
                                  Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(kPipeM, kPipeN));

        LocalTensor<bfloat16_t> l1a_t = l1a_buf.Get<bfloat16_t>();
        auto l1A = Te::MakeTensor(
            Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1a_t.GetPhyAddr())),
            Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(kPipeM, kPipeKSub));

        // WHY EVERY FENCE IN THIS LOOP IS AN IMMEDIATE SetFlag/WaitFlag PAIR
        // AND NOT A DEFERRED TOKEN -- MEASURED, dav-3510 CAModel, 2026-10-18.
        //
        // This loop briefly carried both WAR fences as tokens raised on one
        // lap and consumed on the next, with `bool ..._war_live` bookkeeping,
        // so that the Cube could run behind the loader. The textual balance
        // was 1:1 and the drain after the loop consumed exactly what the loop
        // raised. IT STILL BROKE, in two ways at once:
        //
        //   * 4 x `pem_ccu.cc execute_set_flag already has same set_flag!`.
        //     A deferred fence's SetFlag and WaitFlag sit on DIFFERENT PIPES
        //     -- M and MTE1 for the L0 fence, MTE1 and MTE2 for the l1A one --
        //     and the scalar unit issues the whole loop body without blocking.
        //     Program order therefore does NOT order the set against the wait:
        //     the producing pipe runs ahead and raises lap k+1's token before
        //     the consuming pipe has reached lap k's wait, which lands a
        //     SetFlag on an already-set binary flag. An IMMEDIATE pair cannot
        //     do this, because the wait is on the consuming pipe and the
        //     scalar unit cannot get past it with a token outstanding.
        //   * the 4-tile accumulation came back as ONE tile's product
        //     (`got 1 want 4`, 256 of 2048 elements wrong). Removing the
        //     `PipeBarrier<PIPE_M>` below is what did that -- see the note on
        //     it at the Mmad.
        //
        // So: balance is necessary and NOT sufficient. A deferred intra-core
        // fence also needs the two pipes to be ordered against each other,
        // which nothing here provides.
        for (std::uint32_t k = 0; k < tiles; ++k) {
            const std::uint32_t slot = k % kPipeRingSlots;

            // A tile k: contiguous in GM because A is tile-major.
            auto gmA = Te::MakeTensor(
                Te::MakeMemPtr(reinterpret_cast<__gm__ bfloat16_t*>(a_gm) +
                               k * kPipeM * kPipeKSub),
                Te::MakeFrameLayout<Te::NDExtLayoutPtn>(kPipeM, kPipeKSub));
            Te::Copy(Te::MakeCopy(Te::CopyGM2L1{}, Te::CopyGM2L1TraitDefault{}), l1A, gmA);

            // ONE wait, not one per AIV. Mode 2's AIV -> AIC direction
            // increments the AIC's counter once BOTH AIVs have set, exactly
            // as documented -- see kPipeAivPerAic for the measurement that
            // settles it.
            //
            // Scoped to PIPE_MTE1, the pipe that reads the slot. The MTE2
            // burst above is deliberately issued BEFORE this wait and is not
            // gated by it: A tile k does not depend on the weight slot, so
            // the two overlap.
            CrossCoreWaitFlag<kPipeSyncMode, kPipeWaitAicSlot>(
                static_cast<uint16_t>(kPipeFlagSlotReady + slot));

            LocalTensor<bfloat16_t> l1w_t = l1w_buf[slot].Get<bfloat16_t>();
            auto l1W = Te::MakeTensor(
                Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1w_t.GetPhyAddr())),
                Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(kPipeKSub, kPipeN));

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

            Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}, Te::CopyL12L0ATraitDefault{}), l0A, l1A);
            // NZ -> ZN converts STORAGE for the same logical [K, N] matrix.
            // It is NOT a logical transpose -- see the header -- which is
            // why L1 already holds the contraction-major tile.
            Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}, Te::CopyL12L0BTraitDefault{}), l0B, l1W);

            // WAR ON l1A, AND IT WAS A LATENT RACE BEFORE THIS FENCE EXISTED.
            // l1A is a SINGLE buffer rewritten every tile, so tile k+1's
            // GM -> L1 burst (MTE2) targets the same bytes tile k's
            // CopyL12L0A (MTE1) has just read. The loop had an MTE2_MTE1
            // fence, which is the RAW direction only; nothing held MTE2 back.
            // It did not corrupt the 4-tile sum, but only because the burst is
            // slower than the L1 -> L0 read it was racing, which is a timing
            // accident and not a guarantee.
            //
            // IMMEDIATE, not deferred: the wait blocks MTE2, which is the pipe
            // that would do the overwriting, and it is placed after the MTE1
            // reads it protects.
            SetFlag<HardEvent::MTE1_MTE2>(kEvPipeMte1ToMte2);
            WaitFlag<HardEvent::MTE1_MTE2>(kEvPipeMte1ToMte2);

            // RELEASE THE SLOT HERE, NOT AFTER THE MMAD. The AIV may refill
            // it as soon as MTE1 has drained it, which this set waits for by
            // construction; holding it until after the Mmad made the AIV wait
            // a full Cube step for no dependency.
            if (k + kPipeRingSlots < tiles) {
                CrossCoreSetFlag<kPipeSyncMode, PIPE_MTE1>(
                    static_cast<uint16_t>(kPipeFlagSlotFree + slot));
            }

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

            // THE ACCUMULATION. ccubeInitVal only on the first tile; every
            // later Mmad adds into the same L0C.
            //
            // THIS CONFIGURATION IS CORRECT AND IS NOT THE BUG, which took
            // five hypotheses to establish, so the evidence is recorded here
            // rather than re-derived. CANN's own Matmul framework computes the
            // same two flags for its K loop
            // (scheduler/batch/batch_scheduler_base.h,
            // UpdateMmadComputeParams):
            //
            //     cmatrixSource  = sL0CInit && IsBias();
            //     cmatrixInitVal = sL0CInit && !IsBias();
            //
            // so for every K step after the first (sL0CInit == false) it
            // passes (source = false, initVal = false) through the PLAIN
            // three-operand Mmad -- exactly what this call does. That is the
            // canonical in-place `C += A * B` on this part, demonstrated by a
            // shipping implementation.
            //
            // ALSO TRIED AND EQUIVALENT: the four-operand bias overload,
            // `Te::Mmad(op, l0C, l0A, l0B, l0C)`. A bias in L0C routes to
            // MmadWithBias, which sets cmatrixSource = false and hardcodes
            // cmatrixInitVal = false -- the same pair. It compiles and it
            // produced a BIT-IDENTICAL wrong answer on the CAModel. Do not
            // spend another run on it.
            Te::MmadParams mmadParams(static_cast<uint16_t>(kPipeM),
                                      static_cast<uint16_t>(kPipeN),
                                      static_cast<uint16_t>(kPipeKSub),
                                      0, k == 0u);
            Te::Mmad(Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{}).with(mmadParams),
                     l0C, l0A, l0B);

            // MEASURED: AN Mmad WITH ccubeInitVal = false DOES ACCUMULATE IN
            // PLACE. A diagnostic that issued a SECOND Mmad over the same
            // L0A/L0B with init = false, run at stage 1, returned `got 2 want
            // 1` / `got 4 want 2` / `got 6 want 3` -- exactly 2x on all 2048
            // elements, dav-3510 CAModel, 2026-10-18. So the Cube's
            // accumulation path is sound and the multi-tile shortfall this
            // probe used to report was never about these flags.
            // THIS BARRIER IS A DOCUMENTED HARDWARE CONTRACT AT THIS SHAPE.
            //
            // It was removed once, on the reasoning that successive Mmads into
            // one L0C are ordered by PIPE_M against itself. Mmad.md's
            // "Synchronization optimization description" says otherwise: a
            // PipeBarrier<PIPE_M> is required between consecutive Mmads
            // accumulating into the same L0C unless the shape clears
            //
            //     (m / 16) * (n / 16) < 10
            //
            // and this probe is m = 16, n = 128 -> 1 * 8 = 8, UNDER the
            // threshold, so the barrier is mandatory. Above the threshold the
            // hardware scheduler handles the write-read dependency itself,
            // which is the regime the removal was reasoning from.
            PipeBarrier<PIPE_M>();

            // WAR ON L0A/L0B: tile k+1's MTE1 writes the buffers tile k's
            // Mmad reads. IMMEDIATE, for the pipe-ordering reason at the top
            // of the loop. The barrier above has already retired the Mmad, so
            // this pair only has to hold MTE1 back, not wait on the Cube.
            SetFlag<HardEvent::M_MTE1>(kEvPipeMToMte1);
            WaitFlag<HardEvent::M_MTE1>(kEvPipeMToMte1);
        }

        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

        // ONE Fixpipe, on the completed contraction. The destination layout
        // carries the FULL M -- the hardware halves it -- while each AIV's
        // landing buffer is only M/2 rows. STEP 1 measured that passing M/2
        // here delivers M/4 rows per core.
        auto ub = Te::MakeTensor(
            Te::MakeMemPtr(reinterpret_cast<__ubuf__ bfloat16_t*>(0)),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(kPipeM, kPipeN));
        Te::FixpipeParams fixParams;
        Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, PipeL0c2UbTraitDualDstM{}).with(fixParams),
                 ub, l0C);
        PipeBarrier<PIPE_FIX>();

        CrossCoreSetFlag<kPipeSyncMode, PIPE_FIX>(kPipeFlagProduct);
        PipeStatus(status_gm, kPipeAicRegion, kPipeSlotLive, kPipeAicDone);
        PipeStatus(status_gm, kPipeAicRegion, kPipeSlotTiles, tiles);
    }

    if ASCEND_IS_AIV {
        // The landing zone is the FIRST UB allocation, so it sits at offset
        // 0 -- the address the AIC's Fixpipe names. Everything else follows.
        TQue<TPosition::VECIN, 1> q_landing;
        pipe.InitBuffer(q_landing, 1, kPipeHalfElems * sizeof(bfloat16_t));
        LocalTensor<bfloat16_t> landing = q_landing.AllocTensor<bfloat16_t>();

        // Ping/pong nibble and scale staging, so the MTE2 for tile k+1 can be
        // in flight while the Vector unit dequantizes tile k. weight_expanded
        // and the scale scratch are SHARED: the dequants are serialized on
        // the Vector pipe anyway, and the UB -> L1 push of tile k is fenced
        // against the dequant of k+1 below.
        TQue<TPosition::VECCALC, 1> q_nib[kPipeRingSlots];
        TQue<TPosition::VECCALC, 1> q_scale[kPipeRingSlots];
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            pipe.InitBuffer(q_nib[s], 1, kPipeKSub * kPipeNHalf / 2u);
            pipe.InitBuffer(q_scale[s], 1, kPipeKSub * kCubeScaleRowStride);
        }
        // `wide` IS NOW PING-PONGED, and that is the change that buys the
        // Vector/MTE3 overlap. With one shared wide buffer the ND2NZ push of
        // tile k and the dequant of tile k+1 contend for the same bytes, so
        // the loop had to fence MTE3 -> V immediately after every push, which
        // blocked the Vector unit until the push drained. Two buffers make
        // that fence a full ring lap long, so dequant(k+1) overlaps push(k).
        //
        // Costs kPipeKSub * kPipeNHalf * 2 = 16,384 B of extra UB.
        //
        // scale_plane and scale_temporary stay SHARED on purpose: they are
        // scratch consumed inside DequantSubTileVector, and there is one
        // Vector unit, so two dequants never overlap each other.
        TQue<TPosition::VECCALC, 1> q_wide[kPipeRingSlots];
        TQue<TPosition::VECCALC, 1> q_plane, q_temp;
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            pipe.InitBuffer(q_wide[s], 1, kPipeKSub * kPipeNHalf * sizeof(bfloat16_t));
        }
        pipe.InitBuffer(q_plane, 1, kCubeScaleRowBytes);
        pipe.InitBuffer(q_temp,  1, kCubeScaleTempBytes);

        Buffers buf[kPipeRingSlots];
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            buf[s].weight_nibbles  = q_nib[s].AllocTensor<fp4x2_e2m1_t>();
            buf[s].weight_scales   = q_scale[s].AllocTensor<uint8_t>();
        }
        LocalTensor<bfloat16_t> wide[kPipeRingSlots];
        LocalTensor<bfloat16_t> plane = q_plane.AllocTensor<bfloat16_t>();
        LocalTensor<uint16_t>   temp  = q_temp.AllocTensor<uint16_t>();
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            wide[s] = q_wide[s].AllocTensor<bfloat16_t>();
            buf[s].weight_expanded  = wide[s];
            buf[s].scale_plane      = plane;
            buf[s].scale_temporary  = temp;
        }

        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());
        // This AIV's COLUMN half of the [K, N] weight tile.
        const std::uint32_t col_start = sub_idx * kPipeNHalf;
        const std::uint32_t region = kPipeAivRegion + sub_idx;

        PipeStatus(status_gm, region, kPipeSlotAddr,
                   static_cast<std::uint32_t>(landing.GetPhyAddr()));

        __gm__ std::uint8_t* packed = reinterpret_cast<__gm__ std::uint8_t*>(w_packed_gm);
        __gm__ std::uint8_t* scales = reinterpret_cast<__gm__ std::uint8_t*>(w_scale_gm);

        // Prologue: with prefetch on, tile 0's MTE2 is issued before the loop
        // so that iteration 0 can issue tile 1's load before dequantizing 0.
        // No V_MTE2 WAR wait on this one -- nothing has read slot 0 yet.
        if (prefetch) {
            AsyncLoadWeightsSubTile(packed, scales, 0u, col_start,
                                    kPipeKSub, kPipeNHalf, kPipeWCols, buf[0],
                                    kEvPipeMte2ToV[0]);
        }

        // One outstanding deferred token per ring slot, tracked explicitly so
        // the drain after the loop consumes exactly what the loop raised.
        bool wide_war_live[kPipeRingSlots] = {false, false};
        bool nib_war_live[kPipeRingSlots]  = {false, false};

        for (std::uint32_t k = 0; k < tiles; ++k) {
            const std::uint32_t slot      = k % kPipeRingSlots;
            const std::uint32_t next_slot = (k + 1u) % kPipeRingSlots;

            if (!prefetch) {
                // The WAR event is passed here too. Without it slot `slot`'s
                // V_MTE2 token from lap k-2 is never consumed, and the
                // SetFlag below then lands on an already-set flag -- the
                // exact undefined behaviour this refactor exists to remove.
                AsyncLoadWeightsSubTile(packed, scales, k * kPipeKSub, col_start,
                                        kPipeKSub, kPipeNHalf, kPipeWCols, buf[slot],
                                        kEvPipeMte2ToV[slot],
                                        nib_war_live[slot] ? kEvPipeNibFree[slot] : -1);
                nib_war_live[slot] = false;
            }

            // THE PREFETCH NOW PRECEDES THE DEQUANT, which is what makes
            // tile k+1's MTE2 burst run against tile k's Vector work rather
            // than merely against the Cube.
            //
            // Two things had to change for this to be legal. An intra-core
            // flag is BINARY, so one shared MTE2_V id allowed only one burst
            // in flight -- hence kEvPipeMte2ToV[slot], one id per slot. And
            // the burst now overwrites a staging buffer whose previous tenant
            // the Vector unit may still be reading, so the loader takes a
            // V_MTE2 WAR event as well. Both are defaulted parameters on the
            // production routine, so its own callers are untouched.
            if (prefetch && k + 1u < tiles) {
                AsyncLoadWeightsSubTile(packed, scales, (k + 1u) * kPipeKSub, col_start,
                                        kPipeKSub, kPipeNHalf, kPipeWCols,
                                        buf[next_slot], kEvPipeMte2ToV[next_slot],
                                        nib_war_live[next_slot]
                                            ? kEvPipeNibFree[next_slot] : -1);
                nib_war_live[next_slot] = false;
            }

            // Scoped to PIPE_MTE3: what this flag protects is the AIV's own
            // ND2NZ push INTO the L1 slot, and MTE3 is what performs it.
            // Leaving the Vector pipe unblocked here is deliberate -- the
            // dequant below does not touch L1.
            if (k >= kPipeRingSlots) {
                CrossCoreWaitFlag<kPipeSyncMode, kPipeWaitAivSlot>(
                    static_cast<uint16_t>(kPipeFlagSlotFree + slot));
            }

            // The production FP4 -> BF16 conversion, verbatim. Its first act
            // is WaitFlag<MTE2_V>, which CONSUMES the token this slot's load
            // raised.
            // WAR on this slot's `wide`: its push, a full ring lap ago, must
            // have drained before the dequant overwrites it.
            if (wide_war_live[slot]) {
                WaitFlag<HardEvent::MTE3_V>(kEvPipeWideFree[slot]);
                wide_war_live[slot] = false;
            }

            DequantSubTileVector(kPipeKSub, kPipeNHalf, buf[slot],
                                 kEvPipeMte2ToV[slot]);

            // The nibble staging is now readable-no-more by anyone until the
            // next burst; hand the WAR token to that burst.
            SetFlag<HardEvent::V_MTE2>(kEvPipeNibFree[slot]);
            nib_war_live[slot] = true;

            // UB -> L1 with an inline ND2NZ conversion. The UB holds
            // ND(K_SUB, N/2) -- this AIV's column half of the tile -- and
            // the destination is NZ(K_SUB, N). NZ splits the COLUMN axis
            // into C0 blocks, so this AIV owns whole blocks
            // [col_start/C0, +N/2/C0) and its base is simply
            // col_start * K_SUB elements in. The two AIVs therefore write
            // disjoint block ranges, which the ND2NZ contract requires --
            // it does not order overlapping writes.
            // THE PUSH IS THE SHIPPING ROUTINE, NOT THE Nd2NzParams OVERLOAD.
            //
            // `PushUbTileToL1Nz` comes from the operator's own kernel source,
            // which this TU already includes -- so this probe now exercises
            // the primitive production actually uses, which is the point of
            // the probe. It bursts straight from wide[slot] into the slot's C0
            // blocks, one strided DataCopy per block, with NO intermediate
            // buffer.
            //
            // WHAT THAT REMOVES. The Nd2NzParams overload resolves to
            // DataCopyUB2L1ND2NZImpl, which runs `TransND2NZ` into a SHARED
            // PopStackBuffer and bursts from there. That buffer is rewritten
            // by every push, and the implementation fences it V -> MTE3 but
            // not MTE3 -> V, so the write-after-read is the caller's to close.
            // Leaving it open corrupted the FIRST C0 BLOCK of every tile but
            // the last -- MEASURED as 256 of 2048 elements wrong, which is 16
            // columns x 16 rows, with columns 16..127 correct. An immediate
            // MTE3 -> V fence also closes it, and did; bursting from
            // wide[slot] means there is no shared buffer to close over at all.
            //
            // AND IT MAKES THE DEFERRED FENCE SOUND AGAIN. See the MTE3_V set
            // below: `wide` is per-slot, so a token consumed one ring lap
            // later is the right scope for it -- which was the original
            // reasoning, correct about `wide` and wrong only because MTE3 was
            // not actually reading `wide`.
            //
            // RAW: the dequant wrote wide[slot] on the Vector pipe and MTE3
            // reads it DIRECTLY now, so this fence is REQUIRED. It was
            // redundant under the Nd2NzParams overload -- there TransND2NZ
            // read `wide` on V, making the edge V -> V -- and dropping it was
            // correct then and would be a silent wrong answer now.
            SetFlag<HardEvent::V_MTE3>(kEvPipeWideReady);
            WaitFlag<HardEvent::V_MTE3>(kEvPipeWideReady);

            PushUbTileToL1Nz(l1w_buf[slot].Get<bfloat16_t>(), wide[slot],
                             /*rows=*/kPipeKSub, /*cols=*/kPipeNHalf,
                             /*src_row_stride=*/kPipeNHalf,
                             /*nz_rows=*/kPipeKSub,
                             /*col_base=*/col_start, /*row_base=*/0u);

            // WAR on wide[slot], DEFERRED ONE RING LAP. This is the overlap
            // the three-stage schedule exists for: an immediate wait here
            // would block PIPE_V until the push drained, so the next dequant
            // could not start. It is sound because wide is per-slot and
            // nothing overwrites wide[slot] until the lap that reuses it.
            SetFlag<HardEvent::MTE3_V>(kEvPipeWideFree[slot]);
            wide_war_live[slot] = true;

            CrossCoreSetFlag<kPipeSyncMode, PIPE_MTE3>(
                static_cast<uint16_t>(kPipeFlagSlotReady + slot));
        }

        // DRAIN EVERY DEFERRED TOKEN. A flag left raised is not harmless: the
        // next SetFlag on the same (srcPipe, dstPipe, eventID) triple lands on
        // an already-set flag, which is undefined behaviour and what the
        // simulator reports as `execute_set_flag already has same set_flag!`.
        // Since the host relaunches this kernel per stage, a leak would show
        // up as a hang in a LATER run rather than this one.
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            if (wide_war_live[s]) {
                WaitFlag<HardEvent::MTE3_V>(kEvPipeWideFree[s]);
                wide_war_live[s] = false;
            }
            if (nib_war_live[s]) {
                WaitFlag<HardEvent::V_MTE2>(kEvPipeNibFree[s]);
                nib_war_live[s] = false;
            }
        }

        // Scoped to PIPE_MTE3: the DataCopy below is what must not run until
        // the Fixpipe has landed the product in this AIV's UB.
        CrossCoreWaitFlag<kPipeSyncMode, kPipeWaitAivFinal>(kPipeFlagProduct);

        GlobalTensor<bfloat16_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out_gm), kPipeOutElems);
        DataCopy(out[sub_idx * kPipeHalfElems], landing, kPipeHalfElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        PipeStatus(status_gm, region, kPipeSlotTiles, tiles);
        PipeStatus(status_gm, region, kPipeSlotLive, kPipeAivDone);

        q_temp.FreeTensor(temp);
        q_plane.FreeTensor(plane);
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            q_wide[s].FreeTensor(wide[s]);
        }
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            q_scale[s].FreeTensor(buf[s].weight_scales);
            q_nib[s].FreeTensor(buf[s].weight_nibbles);
        }
        q_landing.FreeTensor(landing);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
