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
            CrossCoreWaitFlag(static_cast<uint16_t>(kPipeFlagSlotReady + slot));

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

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

            // THE ACCUMULATION. ccubeInitVal only on the first tile; every
            // later Mmad adds into the same L0C.
            Te::MmadParams mmadParams(static_cast<uint16_t>(kPipeM),
                                      static_cast<uint16_t>(kPipeN),
                                      static_cast<uint16_t>(kPipeKSub),
                                      0, k == 0u);
            Te::Mmad(Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{}).with(mmadParams),
                     l0C, l0A, l0B);
            PipeBarrier<PIPE_M>();

            if (k + kPipeRingSlots < tiles) {
                CrossCoreSetFlag<kPipeSyncMode, PIPE_MTE1>(
                    static_cast<uint16_t>(kPipeFlagSlotFree + slot));
            }
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
        TQue<TPosition::VECCALC, 1> q_wide, q_plane, q_temp;
        pipe.InitBuffer(q_wide,  1, kPipeKSub * kPipeNHalf * sizeof(bfloat16_t));
        pipe.InitBuffer(q_plane, 1, kCubeScaleRowBytes);
        pipe.InitBuffer(q_temp,  1, kCubeScaleTempBytes);

        Buffers buf[kPipeRingSlots];
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            buf[s].weight_nibbles  = q_nib[s].AllocTensor<fp4x2_e2m1_t>();
            buf[s].weight_scales   = q_scale[s].AllocTensor<uint8_t>();
        }
        LocalTensor<bfloat16_t> wide  = q_wide.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> plane = q_plane.AllocTensor<bfloat16_t>();
        LocalTensor<uint16_t>   temp  = q_temp.AllocTensor<uint16_t>();
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            buf[s].weight_expanded  = wide;
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
        if (prefetch) {
            AsyncLoadWeightsSubTile(packed, scales, 0u, col_start,
                                    kPipeKSub, kPipeNHalf, kPipeWCols, buf[0]);
        }

        for (std::uint32_t k = 0; k < tiles; ++k) {
            const std::uint32_t slot = k % kPipeRingSlots;

            if (!prefetch) {
                AsyncLoadWeightsSubTile(packed, scales, k * kPipeKSub, col_start,
                                        kPipeKSub, kPipeNHalf, kPipeWCols, buf[slot]);
            }

            if (k >= kPipeRingSlots) {
                CrossCoreWaitFlag(static_cast<uint16_t>(kPipeFlagSlotFree + slot));
            }

            // The production FP4 -> BF16 conversion, verbatim. Its first act
            // is WaitFlag<MTE2_V>, which CONSUMES the token the load above
            // (or the previous iteration's prefetch) raised.
            DequantSubTileVector(kPipeKSub, kPipeNHalf, buf[slot]);

            // THE PREFETCH GOES HERE, NOT BEFORE THE DEQUANT.
            //
            // Intra-core SetFlag is NOT a counting semaphore: raising
            // MTE2_V for tile k+1 while tile k's token is still outstanding
            // is a hardware error, and the simulator says so --
            // `pem_ccu.cc:2277 execute_set_flag already has same set_flag!`.
            // AsyncLoadWeightsSubTile raises kEvCubeMte2ToV unconditionally,
            // so exactly one of its tokens may be in flight at a time.
            //
            // Issuing the load after the dequant has consumed the token
            // still overlaps three stages -- tile k+1's MTE2 runs against
            // tile k's UB -> L1 push and against the Cube's consumption of
            // tile k -- it just does not also overlap the vector dequant
            // with its own prefetch. Recovering that last overlap needs
            // ping/pong event ids inside the load helper, which would mean
            // editing a production routine this probe reuses verbatim.
            if (prefetch && k + 1u < tiles) {
                AsyncLoadWeightsSubTile(packed, scales, (k + 1u) * kPipeKSub, col_start,
                                        kPipeKSub, kPipeNHalf, kPipeWCols,
                                        buf[(k + 1u) % kPipeRingSlots]);
            }

            // UB -> L1 with an inline ND2NZ conversion. The UB holds
            // ND(K_SUB, N/2) -- this AIV's column half of the tile -- and
            // the destination is NZ(K_SUB, N). NZ splits the COLUMN axis
            // into C0 blocks, so this AIV owns whole blocks
            // [col_start/C0, +N/2/C0) and its base is simply
            // col_start * K_SUB elements in. The two AIVs therefore write
            // disjoint block ranges, which the ND2NZ contract requires --
            // it does not order overlapping writes.
            LocalTensor<bfloat16_t> l1w_t = l1w_buf[slot].Get<bfloat16_t>();
            SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
            WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
            Nd2NzParams nd2nz;
            nd2nz.ndNum             = 1;
            nd2nz.nValue            = static_cast<uint16_t>(kPipeKSub);
            nd2nz.dValue            = kPipeNHalf;
            nd2nz.srcNdMatrixStride = 0;
            nd2nz.srcDValue         = kPipeNHalf;
            nd2nz.dstNzC0Stride     = static_cast<uint16_t>(kPipeKSub);
            nd2nz.dstNzNStride      = 1;
            nd2nz.dstNzMatrixStride = 0;
            DataCopy(l1w_t[col_start * kPipeKSub], wide, nd2nz);
            SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);

            CrossCoreSetFlag<kPipeSyncMode, PIPE_MTE3>(
                static_cast<uint16_t>(kPipeFlagSlotReady + slot));
        }

        CrossCoreWaitFlag(kPipeFlagProduct);

        GlobalTensor<bfloat16_t> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out_gm), kPipeOutElems);
        DataCopy(out[sub_idx * kPipeHalfElems], landing, kPipeHalfElems);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

        PipeStatus(status_gm, region, kPipeSlotTiles, tiles);
        PipeStatus(status_gm, region, kPipeSlotLive, kPipeAivDone);

        q_temp.FreeTensor(temp);
        q_plane.FreeTensor(plane);
        q_wide.FreeTensor(wide);
        for (std::uint32_t s = 0; s < kPipeRingSlots; ++s) {
            q_scale[s].FreeTensor(buf[s].weight_scales);
            q_nib[s].FreeTensor(buf[s].weight_nibbles);
        }
        q_landing.FreeTensor(landing);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
