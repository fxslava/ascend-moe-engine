/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- VARIANT (b) of the dsv4_moe_prefill_cube bisection, STAGED.
//
// Variant (a) (test_kfc_lifecycle_kernel.cpp) retires in 41 s with no compute.
// The full kernel never retires at any shape. This walks the delta between
// them ONE STAGE AT A TIME, so the run that first fails to retire names the
// routine rather than the region.
//
// The production routines are reached by TEXTUALLY INCLUDING the production TU,
// exactly as the inductive gate does. Nothing is transcribed: AsyncLoadWeights-
// SubTile, DequantSubTileVector and DequantSubTileToL1 here ARE the shipping
// ones, so a stage that hangs indicts production code and not a copy of it.
//
// STAGES, each a strict superset of the one before:
//
//   1  REGIST_MATMUL_OBJ + the production UB plane            (== variant (a))
//   2  + AsyncLoadWeightsSubTile      GM -> UB, MTE2, SetFlag<MTE2_V>
//   3  + DequantSubTileVector         WaitFlag<MTE2_V>, Cast/ShiftLeft/
//                                     Broadcast/Mul in UB
//   4  + DequantSubTileToL1           as 3, plus V_MTE3, the UB -> L1 (TSCM)
//                                     DataCopy and MTE3_V
//   5  + EnQue/DeQue and the Matmul operand set-up (SetTensorA, SetTensorB,
//        SetOrgShape, SetTail) -- no Cube work issued yet
//   6  + Iterate / GetTensorC         the Cube step and the Fixpipe -> VECIN
//                                     delivery
//   7  + End() and the FP32 accumulate
//
// Stages 3 and 4 are MUTUALLY EXCLUSIVE paths, not nested calls: stage 4 runs
// DequantSubTileToL1, which itself calls DequantSubTileVector. Calling both
// would issue two WaitFlag<MTE2_V> against one SetFlag and hang for a reason
// that is this harness's fault rather than the kernel's.
//
// A is a CONSTANT operand (Duplicate 1.0) rather than real tokens. This probe
// answers "does it retire", never "is the answer right", so the token prefetch
// ring is deliberately out of the picture -- it was audited separately and
// balances. Keeping it out means a hang here cannot be the ring.
//
// EVERY STAGE ALSO DUMPS A UB PHYSICAL ADDRESS CENSUS into the status buffer
// before it runs. See kernel/dsv4_prefill_cube_ub_census.h: the simulator's
// ub_addr_overflow names an address, and the census is what turns that into a
// buffer name. It is how the pool base was found to be 0 rather than the
// ~131,616 the DataCache story predicted, and how the TSCM weight ring was
// found to sit in UB rather than L1.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include "dsv4_prefill_cube_ub_census.h"

#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

namespace {

// --- UB PHYSICAL ADDRESS CENSUS -------------------------------------------
//
// One entry per InitBuffer, in allocation order, written as two uint32 words
// into the probe's status buffer. The wire format and the reason it exists
// are in kernel/dsv4_prefill_cube_ub_census.h.
//
// GetPhyAddr() is a pure read of the address TPipe assigned at InitBuffer
// time, so it is valid even when the layout over-commits -- which is the case
// this is for. It costs scalar stores and no UB traffic, so it cannot itself
// perturb the thing being measured.
__aicore__ inline void WriteCensusEntry(GlobalTensor<uint32_t>& census,
                                        std::uint32_t slot, std::uint64_t addr) {
    census.SetValue(kUbCensusHeaderWords + 2u * slot,
                    static_cast<uint32_t>(addr & 0xFFFFFFFFull));
    census.SetValue(kUbCensusHeaderWords + 2u * slot + 1u,
                    static_cast<uint32_t>(addr >> 32));
}

__aicore__ inline GlobalTensor<uint32_t> CensusTensor(GM_ADDR status_out) {
    GlobalTensor<uint32_t> census;
    census.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status_out),
                           kUbCensusStatusWords);
    return census;
}

// Written BEFORE any stage work, because the addresses are fixed the moment
// InitBuffer returns and a stage that does not retire would otherwise take the
// census down with it. The magic word goes last so a torn census is
// distinguishable from no census.
__aicore__ inline void DumpUbCensus(const Buffers& buf, GM_ADDR status_out) {
    GlobalTensor<uint32_t> census = CensusTensor(status_out);
    WriteCensusEntry(census,  0, buf.activation_resident.GetPhyAddr());
    WriteCensusEntry(census,  1, buf.gate_accumulator.GetPhyAddr());
    WriteCensusEntry(census,  2, buf.up_accumulator.GetPhyAddr());
    WriteCensusEntry(census,  3, buf.subgemm_fwd.GetPhyAddr());
    WriteCensusEntry(census,  4, buf.tokens_f32_ping.GetPhyAddr());
    WriteCensusEntry(census,  5, buf.tokens_f32_pong.GetPhyAddr());
    WriteCensusEntry(census,  6, buf.tokens_sub_bf16_ping.GetPhyAddr());
    WriteCensusEntry(census,  7, buf.tokens_sub_bf16_pong.GetPhyAddr());
    WriteCensusEntry(census,  8, buf.down_accumulator.GetPhyAddr());
    WriteCensusEntry(census,  9, buf.subgemm_down.GetPhyAddr());
    WriteCensusEntry(census, 10, buf.activation_sub_bf16.GetPhyAddr());
    WriteCensusEntry(census, 11, buf.weight_nibbles.GetPhyAddr());
    WriteCensusEntry(census, 12, buf.weight_expanded.GetPhyAddr());
    WriteCensusEntry(census, 13, buf.weight_scales.GetPhyAddr());
    WriteCensusEntry(census, 14, buf.scale_plane.GetPhyAddr());
    WriteCensusEntry(census, 15, buf.scale_temporary.GetPhyAddr());
    census.SetValue(2u, kUbCensusTotalEntries);
    census.SetValue(3u, 0u);
    census.SetValue(1u, kUbCensusMagic);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
                             DcciDst::CACHELINE_OUT>(census);
}

}  // namespace

extern "C" __global__ __aicore__ void test_subgemm_stage_kernel(
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR status_out, GM_ADDR shared_workspace, GM_ADDR tiling_config) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (unlikely(tiling_config == nullptr || shared_workspace == nullptr)) {
        return;
    }
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data =
        reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
    if (unlikely(TilingRejected(tiling_data))) {
        return;
    }
    const Geometry geom = LoadGeometry(tiling_data);

    // The stage word sits behind both staged TCubeTilings, for the reason the
    // sim sample cap does: offset 72 is the gate tiling's first field.
    const std::uint32_t stage = *reinterpret_cast<const __gm__ std::uint32_t*>(
        Bytes(tiling_config) + kPrefillCubeStagingBytes);

    TPipe pipe;
    GateUpMatmul matmul_gate_up;
    DownMatmul   matmul_down;
    const __gm__ TCubeTiling* gate_up_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeGateTilingOffset);
    const __gm__ TCubeTiling* down_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeDownTilingOffset);
    REGIST_MATMUL_OBJ(&pipe, shared_workspace,
                      matmul_gate_up, gate_up_cube_tiling,
                      matmul_down,    down_cube_tiling);

    TQue<TPosition::VECCALC, 1> q_activation_resident;
    TQue<TPosition::VECCALC, 1> q_gate_accumulator;
    TQue<TPosition::VECCALC, 1> q_up_accumulator;
    TQue<TPosition::VECIN,   1> q_subgemm_fwd;
    TQue<TPosition::VECCALC, 1> q_tokens_f32_ping;
    TQue<TPosition::VECCALC, 1> q_tokens_f32_pong;
    TQue<TPosition::VECOUT,  1> q_tokens_sub_bf16_ping;
    TQue<TPosition::VECOUT,  1> q_tokens_sub_bf16_pong;
    TQue<TPosition::VECCALC, 1> q_down_accumulator;
    TQue<TPosition::VECIN,   1> q_subgemm_down;
    TQue<TPosition::VECOUT,  1> q_activation_sub_bf16;
    TQue<TPosition::VECCALC, 1> q_weight_nibbles;
    TQue<TPosition::VECCALC, 1> q_weight_expanded;
    TQue<TPosition::VECCALC, 1> q_weight_scales;
    TQue<TPosition::VECCALC, 1> q_scale_plane;
    TQue<TPosition::VECCALC, 1> q_scale_temporary;

    pipe.InitBuffer(q_activation_resident,   1, kCubeActBytes);
    pipe.InitBuffer(q_gate_accumulator,      1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_up_accumulator,        1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_subgemm_fwd,           1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_tokens_f32_ping,       1, kCubeXF32Bytes);
    pipe.InitBuffer(q_tokens_f32_pong,       1, kCubeXF32Bytes);
    pipe.InitBuffer(q_tokens_sub_bf16_ping,  1, kCubeXSubBytes);
    pipe.InitBuffer(q_tokens_sub_bf16_pong,  1, kCubeXSubBytes);
    pipe.InitBuffer(q_down_accumulator,      1, kCubeAcc2Bytes);
    pipe.InitBuffer(q_subgemm_down,          1, kCubeAcc2Bytes);
    pipe.InitBuffer(q_activation_sub_bf16,   1, kCubeASubBytes);
    pipe.InitBuffer(q_weight_nibbles,        1, kCubeNibBytes);
    pipe.InitBuffer(q_weight_expanded,       1, kCubeWideBytes);
    pipe.InitBuffer(q_weight_scales,         1, kCubeScaleBytes);
    pipe.InitBuffer(q_scale_plane,           1, kCubeScaleRowBytes);
    pipe.InitBuffer(q_scale_temporary,       1, kCubeScaleTempBytes);

    TQue<TPosition::TSCM, 2> q_weight_fwd;
    TQue<TPosition::TSCM, 2> q_weight_down;
    pipe.InitBuffer(q_weight_fwd,  2, kCubeB1SlotBytes);
    pipe.InitBuffer(q_weight_down, 2, kCubeB2SlotBytes);

    Buffers buf;
    buf.activation_resident   = q_activation_resident.AllocTensor<bfloat16_t>();
    buf.gate_accumulator      = q_gate_accumulator.AllocTensor<float>();
    buf.up_accumulator        = q_up_accumulator.AllocTensor<float>();
    buf.subgemm_fwd           = q_subgemm_fwd.AllocTensor<float>();
    buf.tokens_f32_ping       = q_tokens_f32_ping.AllocTensor<float>();
    buf.tokens_f32_pong       = q_tokens_f32_pong.AllocTensor<float>();
    buf.tokens_sub_bf16_ping  = q_tokens_sub_bf16_ping.AllocTensor<bfloat16_t>();
    buf.tokens_sub_bf16_pong  = q_tokens_sub_bf16_pong.AllocTensor<bfloat16_t>();
    buf.down_accumulator      = q_down_accumulator.AllocTensor<float>();
    buf.subgemm_down          = q_subgemm_down.AllocTensor<float>();
    buf.activation_sub_bf16   = q_activation_sub_bf16.AllocTensor<bfloat16_t>();
    buf.weight_nibbles        = q_weight_nibbles.AllocTensor<fp4x2_e2m1_t>();
    buf.weight_expanded       = q_weight_expanded.AllocTensor<bfloat16_t>();
    buf.weight_scales         = q_weight_scales.AllocTensor<uint8_t>();
    buf.scale_plane           = q_scale_plane.AllocTensor<bfloat16_t>();
    buf.scale_temporary       = q_scale_temporary.AllocTensor<uint16_t>();

    const bool drives_cube = (GetSubBlockIdx() % GetSubBlockNum()) == 0;

    // One writer only: both paired AIVs run this body and both would target
    // the same GM words. Their layouts are identical -- the allocation is
    // static and neither sub-core's InitBuffer sequence differs -- so the
    // second writer would add nothing but a race.
    if (drives_cube && status_out != nullptr) {
        DumpUbCensus(buf, status_out);

        // THE RING SLOTS, ALL FOUR, BEFORE ANY STAGE WORK.
        //
        // Both slots of each queue have to be live at once to see both
        // addresses, which is what a depth-2 TQue is for, so this is an
        // ordinary double-buffer alloc and not an abuse of the ring. The
        // Alloc/Free pairs balance and nothing is EnQued, which is exactly
        // the shape the `effective_stage < 5` path below already uses -- so
        // the ring is handed back in the state it was found in. The stage
        // loop re-reads slot 0 into the same census entry; if the two
        // readings disagree, this probe moved the ring and the measurement,
        // not the kernel, is what to fix.
        GlobalTensor<uint32_t> census = CensusTensor(status_out);
        LocalTensor<bfloat16_t> fwd0  = q_weight_fwd.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> fwd1  = q_weight_fwd.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> down0 = q_weight_down.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> down1 = q_weight_down.AllocTensor<bfloat16_t>();
        WriteCensusEntry(census, kUbCensusTscmEntry + 0u, fwd0.GetPhyAddr());
        WriteCensusEntry(census, kUbCensusTscmEntry + 1u, fwd1.GetPhyAddr());
        WriteCensusEntry(census, kUbCensusTscmEntry + 2u, down0.GetPhyAddr());
        WriteCensusEntry(census, kUbCensusTscmEntry + 3u, down1.GetPhyAddr());
        q_weight_down.FreeTensor(down1);
        q_weight_down.FreeTensor(down0);
        q_weight_fwd.FreeTensor(fwd1);
        q_weight_fwd.FreeTensor(fwd0);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(census);
    }

    // STAGES 8 AND 9 ARE STAGE 7, REPLAYED.
    //
    // Stage 7 -- one complete sub-GEMM -- retires in 94 s, and six of them
    // should therefore cost about 365 s against a 900 s box the full schedule
    // still blew through. So the cost is not the trip count and the defect is
    // not inside a trip: it is in what REPEATING one adds. Three things only:
    // the TQue<TPosition::TSCM, 2> weight ring hands back a REUSED slot from
    // the second trip on, End() is called once per trip against one KfcServer,
    // and the FP32 accumulator is read-modify-written across trips.
    //
    // 2 reps is the first trip that reuses a ring slot; 4 reps is the first
    // that wraps it twice. If 2 hangs, the sub-GEMM is not re-enterable and
    // the search collapses to those three mechanisms.
    const std::uint32_t effective_stage = (stage >= 8u) ? 7u : stage;
    const std::uint32_t reps = (stage == 8u) ? 2u : ((stage >= 9u) ? 4u : 1u);

    for (std::uint32_t rep = 0; drives_cube && effective_stage >= 2u && rep < reps; ++rep) {
        const std::uint32_t rows  = geom.token_tile;
        const std::uint32_t r_sub  = geom.inter_sub_fwd;
        const std::uint32_t k_sub  = geom.hidden_sub_fwd;

        LocalTensor<bfloat16_t> slot_weight = q_weight_fwd.AllocTensor<bfloat16_t>();
        if (rep == 0u && status_out != nullptr) {
            GlobalTensor<uint32_t> census = CensusTensor(status_out);
            WriteCensusEntry(census, kUbCensusTscmEntry, slot_weight.GetPhyAddr());
            DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(census);
        }

        AsyncLoadWeightsSubTile(
            reinterpret_cast<__gm__ std::uint8_t*>(gate_weights_packed),
            reinterpret_cast<__gm__ std::uint8_t*>(gate_weight_scales),
            0u, 0u, r_sub, k_sub, geom.hidden, buf);

        if (effective_stage == 3u) {
            // Vector dequant only. The MTE2_V token raised above is consumed
            // here, so this path is self-contained.
            DequantSubTileVector(r_sub, k_sub, buf);
        } else if (effective_stage >= 4u) {
            // Full dequant including the UB -> L1 push. Calls
            // DequantSubTileVector itself, which is why stage 3 is exclusive.
            DequantSubTileToL1(r_sub, k_sub, buf, slot_weight);
        }

        if (effective_stage >= 5u) {
            // A constant A operand: this probe measures retirement, not values.
            Duplicate(buf.tokens_sub_bf16_ping, static_cast<bfloat16_t>(1.0F),
                      rows * k_sub);
            PipeBarrier<PIPE_V>();

            q_weight_fwd.EnQue(slot_weight);
            LocalTensor<bfloat16_t> staged_weight = q_weight_fwd.DeQue<bfloat16_t>();

            matmul_gate_up.SetTensorA(buf.tokens_sub_bf16_ping);
            matmul_gate_up.SetTensorB(staged_weight, true);
            matmul_gate_up.SetOrgShape(rows, r_sub, k_sub);
            matmul_gate_up.SetTail(rows, r_sub, k_sub);

            if (effective_stage >= 6u) {
                while (matmul_gate_up.Iterate()) {
                    matmul_gate_up.GetTensorC(buf.subgemm_fwd);
                }
            }
            if (effective_stage >= 7u) {
                matmul_gate_up.End();
                Add(buf.gate_accumulator, buf.gate_accumulator, buf.subgemm_fwd,
                    rows * r_sub);
                PipeBarrier<PIPE_V>();
            }
            q_weight_fwd.FreeTensor(staged_weight);
        } else {
            q_weight_fwd.FreeTensor(slot_weight);
        }
    }

    // Liveness: reached only if every stage above returned.
    if (drives_cube && status_out != nullptr) {
        GlobalTensor<int32_t> status_gm;
        status_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status_out),
                                  kUbCensusStatusWords);
        status_gm.SetValue(0, static_cast<int32_t>(stage));
        // ENTIRE_DATA_CACHE rather than the single line word 0 sits in: the
        // census shares this buffer, and the TSCM entry was written from
        // inside the loop after the first flush.
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(status_gm);
    }

    q_scale_temporary.FreeTensor(buf.scale_temporary);
    q_scale_plane.FreeTensor(buf.scale_plane);
    q_weight_scales.FreeTensor(buf.weight_scales);
    q_weight_expanded.FreeTensor(buf.weight_expanded);
    q_weight_nibbles.FreeTensor(buf.weight_nibbles);
    q_activation_sub_bf16.FreeTensor(buf.activation_sub_bf16);
    q_subgemm_down.FreeTensor(buf.subgemm_down);
    q_down_accumulator.FreeTensor(buf.down_accumulator);
    q_tokens_sub_bf16_pong.FreeTensor(buf.tokens_sub_bf16_pong);
    q_tokens_sub_bf16_ping.FreeTensor(buf.tokens_sub_bf16_ping);
    q_tokens_f32_pong.FreeTensor(buf.tokens_f32_pong);
    q_tokens_f32_ping.FreeTensor(buf.tokens_f32_ping);
    q_subgemm_fwd.FreeTensor(buf.subgemm_fwd);
    q_up_accumulator.FreeTensor(buf.up_accumulator);
    q_gate_accumulator.FreeTensor(buf.gate_accumulator);
    q_activation_resident.FreeTensor(buf.activation_resident);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
