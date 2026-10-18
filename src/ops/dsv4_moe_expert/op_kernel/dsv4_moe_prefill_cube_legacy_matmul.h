// dsv4_moe_prefill_cube_legacy_matmul.h -- QUARANTINE.
//
// This header holds the Matmul-framework schedule `dsv4_moe_prefill_cube` used to run, and no
// longer does. NOTHING IN THE SHIPPING KERNEL CALLS ANYTHING IN HERE. It is included, at the very
// end of op_kernel/dsv4_moe_prefill_cube_kernel.cpp and inside that file's namespace, for exactly
// one reason:
//
//   FIVE SIM BISECTION PROBES TEXTUALLY INCLUDE THE OPERATOR'S KERNEL SOURCE so that the bodies
//   they exercise are verbatim production code, and they call these routines:
//
//     tests/sim/kernel/dsv4_moe_prefill_cube_dequant_unit.cpp     -> Dsv4PrefillCube dequant unit
//     tests/sim/kernel/dsv4_moe_prefill_cube_inductive_kernel.cpp -> Dsv4MoePrefillCubeInductiveSim
//     tests/sim/kernel/test_subgemm_stage_kernel.cpp              -> test_subgemm_stage_sim
//     tests/sim/kernel/test_dequant_micro_kernel.cpp              -> Dsv4PrefillCubeDequantMicroSim
//     tests/sim/kernel/test_kfc_lifecycle_kernel.cpp              -> KfcLifecycleSim
//                                                 (this one needs the constants only, not the code)
//   plus the host-side gate Dsv4MoePrefillCubeSingleTileSim, which drives the operator's launch
//   wrapper over one ProcessTokenTile trip.
//
// WHY IT IS NOT SIMPLY DELETED. Deleting it retires those gates, and that is a decision about
// verification coverage rather than a tidy-up. The two probes that still test LIVE code -- the
// MXFP4/E8M0 dequantizer, which the new path calls unchanged -- want REWORKING onto the new
// schedule, not deleting; the three that bisected the framework (the KFC lifecycle, the null
// TCubeTiling fault, the TSCM ring's UB residency) answered questions that no longer have a
// subject and can go. That is a separate change.
//
// WHY IT COSTS THE SHIPPING KERNEL NOTHING. UB is charged by InitBuffer calls, and the entry point
// makes none of these. The two Matmul TYPES below reserve flagIds [0, 2N-1] only for N DEFINED
// OBJECTS, and the entry point defines none and never calls REGIST_MATMUL_OBJ -- so no KfcServer
// is stood up, no message queue is placed in the workspace, and the live schedule's flagIds (4-9,
// see the UB layout header) were kept clear of [0, 3] anyway.
//
// DO NOT EXTEND THIS, add features to it, or size a new buffer from its constants. The constant
// names that differ from the live ones by one word -- kCubeActBytes vs kCubeActPlaneBytes,
// kCubeWideBytes vs kCubeWideHalfBytes, kCubeXF32Bytes vs kCubeTokenF32Bytes -- differ because the
// live path BISECTS what this one did whole, and crossing them silently halves or doubles a buffer.
//
// The commentary inside each routine is the ORIGINAL commentary, kept verbatim: it records
// measurements (the MTE3_MTE1 deadlock, the Iterate-vs-IterateAll contract for a VECIN C, the
// token ring's missing V_MTE2 edge) that were paid for on the CAModel and that remain true about
// the code they describe.

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_LEGACY_MATMUL_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_LEGACY_MATMUL_HPP

// `lib/matmul_intf.h` is included by the kernel TU at FILE SCOPE, not here: this header is
// included inside the operator's namespace, and matmul_intf's sources name `AscendC::`
// explicitly, so including it from here resolves `AscendC` to a nested namespace that does not
// exist. Removing the framework means deleting this header AND that include together.

using AGateType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t>;
using ADownType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t>;
using WType     = MatmulType<TPosition::TSCM,   CubeFormat::ND, bfloat16_t, true>;
using AccType   = MatmulType<TPosition::VECIN,  CubeFormat::ND, float>;
using BiasType  = MatmulType<TPosition::GM,     CubeFormat::ND, float>;

constexpr MatmulConfig kCubeCfg    = GetNormalConfig();
constexpr auto         kGateTiling = GetMatmulApiTiling<AGateType, WType, AccType, BiasType>(kCubeCfg);
constexpr auto         kDownTiling = GetMatmulApiTiling<ADownType, WType, AccType, BiasType>(kCubeCfg);

using GateUpMatmul = Matmul<AGateType, WType, AccType, BiasType, kGateTiling>;
using DownMatmul   = Matmul<ADownType, WType, AccType, BiasType, kDownTiling>;

// The full staging aggregate the probes declare and populate. It DERIVES from the live
// `WeightStage`, so one definition of AsyncLoadWeightsSubTile / DequantSubTileVector serves both
// paths and a `Buffers` still converts to a `const WeightStage&`.
struct Buffers : WeightStage {
    LocalTensor<bfloat16_t>  activation_resident;
    LocalTensor<float>        gate_accumulator;
    LocalTensor<float>        up_accumulator;
    LocalTensor<float>        subgemm_fwd;
    LocalTensor<float>        tokens_f32_ping;
    LocalTensor<float>        tokens_f32_pong;
    LocalTensor<bfloat16_t>  tokens_sub_bf16_ping;
    LocalTensor<bfloat16_t>  tokens_sub_bf16_pong;
    LocalTensor<float>        down_accumulator;
    LocalTensor<float>        subgemm_down;
    LocalTensor<bfloat16_t>  activation_sub_bf16;
};

// The legacy token prefetch ring. Two slots, two event ids -- and each id carries TWO flags,
// because a flag is keyed by the (srcPipe, dstPipe, eventID) triple: `MTE2_V` id 2 ("ping is
// full") and `V_MTE2` id 2 ("ping is free") are distinct hardware flags. That is what let a
// correct two-way handshake fit the four event ids arch35_common.hpp admits.
constexpr std::uint32_t kTokenRingSlots = 2;
constexpr int           kEvTokensPing   = 2;
constexpr int           kEvTokensPong   = 3;

__aicore__ inline void DequantSubTileToL1(
    std::uint32_t row_count, std::uint32_t col_count,
    const Buffers& buf, const LocalTensor<bfloat16_t>& l1_weight_slot) {
    DequantSubTileVector(row_count, col_count, buf);
    const std::uint32_t total_elements = row_count * col_count;
    SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    DataCopy(l1_weight_slot, buf.weight_expanded, total_elements);
    // NO MTE3_MTE1 HERE. It used to be, and it was the deadlock.
    //
    // MTE1 (L1 -> L0A/L0B) is an AIC-ONLY pipeline. The Ascend C intra-core
    // synchronization tables list MTE3_MTE1 among the legal combinations for
    // the AIC and NOT for the AIV, whose only pipelines are S, V, MTE2 and
    // MTE3. This function runs on the AIV, so `WaitFlag<MTE3_MTE1>` here
    // blocked on a pipe event that NO transition on this core can produce --
    // a directed circuit with zero tokens and no live producer. It compiled,
    // and it never completed.
    //
    // It was not a BALANCE error, which is why counting SetFlag against
    // WaitFlag did not find it: the pair was textually 1:1. What was wrong is
    // that the pipe pair is not legal on this core.
    //
    // The dependency it was reaching for -- this AIV's UB -> L1 write must
    // land before the Cube's MTE1 reads that L1 slot -- is CROSS-CORE, and
    // intra-core SetFlag/WaitFlag cannot express a cross-core edge at all;
    // that would need CrossCoreSetFlag/CrossCoreWaitFlag. It also does not
    // need to be expressed here: the caller EnQues this slot on the
    // TQue<TPosition::TSCM, 2> and DeQues it before SetTensorB, and the
    // Matmul framework's KFC message at IterateAll is what orders the Cube
    // against it.
    //
    // MEASURED: with this pair present, no launch retired at any shape -- 48
    // sub-GEMMs for 2700 s, and 6 sub-GEMMs for 900 s, both ending on the
    // host watchdog with zero pem_lsu diagnostics and the cycle count still
    // climbing. The lifecycle-only variant, which never reaches this
    // function, retired in 41 s.
    SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
}

// THE TOKEN RING NEEDS EDGES IN BOTH DIRECTIONS.
//
// `MTE2_V` says "the slot is full, the Vector may read it". That alone is not a double buffer: it
// leaves the other direction unstated, and the other direction is what stops MTE2 from refilling a
// slot the Vector has not finished reading. With two slots and three or more chunks, chunk k+1 is
// loaded into the slot chunk k-1 is still being cast out of, and nothing orders the two.
//
// OBSERVED, 2026-10-16, dav-3510 CAModel, `Dsv4PrefillCubeDequantMicroSim` at hidden = 512: chunk 0
// came back holding CHUNK 2's values, which is exactly that race resolving the wrong way -- the
// MTE2 burst for chunk 2 landed in the ping slot before the Cast of chunk 0 had read it. Four
// chunks is the smallest grid that can show it (a two-chunk run reuses no slot and passed), and the
// corruption is silent: no fault, no flag, plausible numbers from the wrong chunk.
//
// So `V_MTE2` carries the matching buffer-free token: `PrepareTokensChunk` sets it once it has
// consumed a slot, and `AsyncLoadTokensChunk` waits for it before refilling one. The event id is
// the slot's own -- flags are keyed by the (srcPipe, dstPipe, eventID) triple, so `V_MTE2` id 2 and
// `MTE2_V` id 2 are different flags and the ring still costs only the two ids it already spent.
//
// `slot_was_consumed` is the caller's statement that this slot has been cast out of before, so
// there is a token to wait for. It is false for the first load into each slot, where the edge
// starts empty; waiting there would hang rather than synchronize. The caller must then DRAIN the
// tokens the last chunks leave outstanding -- see `DrainTokenRing` and its use per `inter_base`,
// because a leaked flag outlives the launch and corrupts the next one.
__aicore__ inline void AsyncLoadTokensChunk(
    const GlobalTensor<float>& tokens_in_gm, std::uint32_t token_row_start, std::uint32_t hidden_base,
    std::uint32_t valid_rows, const Geometry& geom,
    const LocalTensor<float>& dst_tensor, int event_id, bool slot_was_consumed) {
    if (slot_was_consumed) {
        WaitFlag<HardEvent::V_MTE2>(event_id);
    }
    DataCopyParams burst_params;
    burst_params.blockCount = static_cast<std::uint16_t>(valid_rows);
    burst_params.blockLen   = static_cast<std::uint16_t>((geom.hidden_sub_fwd * 4u) / 32u);
    burst_params.srcStride  = static_cast<std::uint16_t>(((geom.hidden - geom.hidden_sub_fwd) * 4u) / 32u);
    burst_params.dstStride  = 0;
    DataCopy(dst_tensor, tokens_in_gm[(static_cast<std::uint64_t>(token_row_start) * geom.hidden) + hidden_base], burst_params);
    SetFlag<HardEvent::MTE2_V>(event_id);
}

__aicore__ inline void PrepareTokensChunk(
    const LocalTensor<bfloat16_t>& tokens_sub_bf16, const LocalTensor<float>& tokens_f32,
    std::uint32_t valid_rows, std::uint32_t hidden_sub_fwd, int event_id) {
    WaitFlag<HardEvent::MTE2_V>(event_id);
    Cast(tokens_sub_bf16, tokens_f32, RoundMode::CAST_RINT, valid_rows * hidden_sub_fwd);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event_id);
}

// Consumes the buffer-free tokens the last chunks leave outstanding, so none survives into the next
// `inter_base` pass or the next launch.
//
// The count is exactly one per slot that was cast out of, and it does NOT depend on the parity of
// the chunk count. Per slot: a token is set once per chunk that occupied it, and consumed once per
// REFILL of it, and a slot is refilled for every chunk it holds except the first -- so each slot
// ends one token up, whichever slot the last chunk happened to land in. Clamped to the chunk count
// because a single-chunk pass never occupies the second slot, and draining it would wait forever on
// a token nothing set.
__aicore__ inline void DrainTokenRing(std::uint32_t chunk_count) {
    if (chunk_count >= 1u) {
        WaitFlag<HardEvent::V_MTE2>(kEvTokensPing);
    }
    if (chunk_count >= 2u) {
        WaitFlag<HardEvent::V_MTE2>(kEvTokensPong);
    }
}

__aicore__ inline void RunSubGemmAccumulate(
    GateUpMatmul& matmul_gate_up, const LocalTensor<bfloat16_t>& tokens_sub_bf16,
    __gm__ std::uint8_t* weights_packed, __gm__ std::uint8_t* weight_scales,
    std::uint32_t inter_base, std::uint32_t hidden_base, std::uint32_t valid_rows,
    const Geometry& geom, const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_fwd,
    const LocalTensor<float>& dst_accumulator, std::uint32_t accum_elements) {
    LocalTensor<bfloat16_t> slot_weight = q_weight_fwd.AllocTensor<bfloat16_t>();
    AsyncLoadWeightsSubTile(weights_packed, weight_scales, inter_base, hidden_base, geom.inter_sub_fwd, geom.hidden_sub_fwd, geom.hidden, buf);
    DequantSubTileToL1(geom.inter_sub_fwd, geom.hidden_sub_fwd, buf, slot_weight);
    q_weight_fwd.EnQue(slot_weight);
    LocalTensor<bfloat16_t> staged_weight = q_weight_fwd.DeQue<bfloat16_t>();

    matmul_gate_up.SetTensorA(tokens_sub_bf16);
    matmul_gate_up.SetTensorB(staged_weight, true);
    matmul_gate_up.SetOrgShape(valid_rows, geom.inter_sub_fwd, geom.hidden_sub_fwd);
    matmul_gate_up.SetTail(valid_rows, geom.inter_sub_fwd, geom.hidden_sub_fwd);
    // VECIN OUTPUT GOES THROUGH Iterate + GetTensorC, NOT IterateAll.
    //
    // This used to be `IterateAll(buf.subgemm_fwd)`. The Ascend C reference for
    // IterateAll's LocalTensor overload says its supported TPositions are TSCM
    // and VECIN and then, for VECIN specifically: "When outputting to the
    // VECIN tensor, use the Iterate API." C here is
    // MatmulType<TPosition::VECIN, ...>, so IterateAll was the one overload
    // the contract rules out for this destination, and the Fixpipe -> UB
    // delivery it synchronously waits on is the step GetTensorC drives -- a
    // call this kernel never made.
    //
    // THE LOOP RUNS EXACTLY ONCE AT THIS TILING, and that is load-bearing:
    // every GetTensorC writes the SAME UB tensor, so with more than one block
    // the Add below would accumulate only the last one. The host tiling makes
    // base == single for both sub-GEMMs (gate/up 16x128x128, down 16x64x256,
    // verified from MakeDsv4MoePrefillCubeGemmTiling's output), so there is a
    // single block. The `while` form is kept rather than a bare pair because
    // it is the documented shape and it terminates correctly either way --
    // but if base ever stops equalling single, this accumulation has to be
    // rewritten, not just re-looped.
    while (matmul_gate_up.Iterate()) {
        matmul_gate_up.GetTensorC(buf.subgemm_fwd);
    }
    matmul_gate_up.End();
    q_weight_fwd.FreeTensor(staged_weight);

    Add(dst_accumulator, dst_accumulator, buf.subgemm_fwd, accum_elements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplySwiGLUActivation(
    const Buffers& buf, std::uint32_t valid_rows, std::uint32_t inter_base,
    const Geometry& geom, std::uint32_t accum_elements) {
    Mins(buf.gate_accumulator, buf.gate_accumulator,  geom.swiglu_limit, accum_elements);
    Maxs(buf.up_accumulator,   buf.up_accumulator,   -geom.swiglu_limit, accum_elements);
    PipeBarrier<PIPE_V>();

    Mins(buf.up_accumulator,   buf.up_accumulator,    geom.swiglu_limit, accum_elements);
    Silu(buf.gate_accumulator, buf.gate_accumulator,  accum_elements);
    PipeBarrier<PIPE_V>();

    Mul(buf.gate_accumulator,  buf.gate_accumulator,  buf.up_accumulator, accum_elements);
    PipeBarrier<PIPE_V>();

    Cast(buf.activation_sub_bf16, buf.gate_accumulator, RoundMode::CAST_RINT, accum_elements);
    PipeBarrier<PIPE_V>();

    DataCopyParams stride_copy_params;
    stride_copy_params.blockCount = static_cast<std::uint16_t>(valid_rows);
    stride_copy_params.blockLen   = static_cast<std::uint16_t>((geom.inter_sub_fwd * sizeof(bfloat16_t)) / 32u);
    stride_copy_params.srcStride  = 0;
    stride_copy_params.dstStride  = static_cast<std::uint16_t>(((geom.inter - geom.inter_sub_fwd) * sizeof(bfloat16_t)) / 32u);

    DataCopy(buf.activation_resident[inter_base], buf.activation_sub_bf16, stride_copy_params);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunDownProjectionStep(
    DownMatmul& matmul_down, __gm__ std::uint8_t* down_packed_ptr, __gm__ std::uint8_t* down_scales_ptr,
    std::uint32_t hidden_base, std::uint32_t inter_base, std::uint32_t valid_rows,
    const Geometry& geom, const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_down,
    std::uint32_t down_accum_elements) {
    for (std::uint32_t r_idx = 0; r_idx < valid_rows; ++r_idx) {
        DataCopy(buf.activation_sub_bf16[r_idx * geom.inter_sub_down], buf.activation_resident[static_cast<std::uint32_t>(r_idx) * geom.inter + inter_base], geom.inter_sub_down);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<bfloat16_t> slot_down   = q_weight_down.AllocTensor<bfloat16_t>();
    AsyncLoadWeightsSubTile(down_packed_ptr, down_scales_ptr, hidden_base, inter_base, geom.hidden_sub_down, geom.inter_sub_down, geom.inter, buf);
    DequantSubTileToL1(geom.hidden_sub_down, geom.inter_sub_down, buf, slot_down);
    q_weight_down.EnQue(slot_down);
    LocalTensor<bfloat16_t> staged_down = q_weight_down.DeQue<bfloat16_t>();

    matmul_down.SetTensorA(buf.activation_sub_bf16);
    matmul_down.SetTensorB(staged_down, true);
    matmul_down.SetOrgShape(valid_rows, geom.hidden_sub_down, geom.inter_sub_down);
    matmul_down.SetTail(valid_rows, geom.hidden_sub_down, geom.inter_sub_down);
    // VECIN OUTPUT GOES THROUGH Iterate + GetTensorC, NOT IterateAll.
    //
    // This used to be `IterateAll(buf.subgemm_down)`. The Ascend C reference for
    // IterateAll's LocalTensor overload says its supported TPositions are TSCM
    // and VECIN and then, for VECIN specifically: "When outputting to the
    // VECIN tensor, use the Iterate API." C here is
    // MatmulType<TPosition::VECIN, ...>, so IterateAll was the one overload
    // the contract rules out for this destination, and the Fixpipe -> UB
    // delivery it synchronously waits on is the step GetTensorC drives -- a
    // call this kernel never made.
    //
    // THE LOOP RUNS EXACTLY ONCE AT THIS TILING, and that is load-bearing:
    // every GetTensorC writes the SAME UB tensor, so with more than one block
    // the Add below would accumulate only the last one. The host tiling makes
    // base == single for both sub-GEMMs (gate/up 16x128x128, down 16x64x256,
    // verified from MakeDsv4MoePrefillCubeGemmTiling's output), so there is a
    // single block. The `while` form is kept rather than a bare pair because
    // it is the documented shape and it terminates correctly either way --
    // but if base ever stops equalling single, this accumulation has to be
    // rewritten, not just re-looped.
    while (matmul_down.Iterate()) {
        matmul_down.GetTensorC(buf.subgemm_down);
    }
    matmul_down.End();
    q_weight_down.FreeTensor(staged_down);

    Add(buf.down_accumulator, buf.down_accumulator, buf.subgemm_down, down_accum_elements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessTokenTile(
    const Geometry& geom, std::uint32_t tile_idx,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut,
    const GlobalTensor<float>& tokens_in_gm, const GlobalTensor<float>& tokens_out_gm,
    const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_fwd, TQue<TPosition::TSCM, 2>& q_weight_down,
    GateUpMatmul& matmul_gate_up, DownMatmul& matmul_down) {
    const std::uint32_t token_row_start = tile_idx * geom.token_tile;
    const std::uint32_t valid_rows      = (geom.total_tokens - token_row_start) < geom.token_tile ? (geom.total_tokens - token_row_start) : geom.token_tile;

    const std::uint32_t expert_id = static_cast<std::uint32_t>(reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert_lut)[token_row_start]);
    if (unlikely(expert_id >= geom.num_experts)) {
        return;
    }
    const std::uint64_t expert_offset = static_cast<std::uint64_t>(expert_id);

    __gm__ std::uint8_t* gate_packed_ptr = Bytes(gate_weights_packed) + expert_offset * geom.gate_up_packed_stride;
    __gm__ std::uint8_t* gate_scales_ptr = Bytes(gate_weight_scales)  + expert_offset * geom.gate_up_scale_stride;
    __gm__ std::uint8_t* up_packed_ptr   = Bytes(up_weights_packed)   + expert_offset * geom.gate_up_packed_stride;
    __gm__ std::uint8_t* up_scales_ptr   = Bytes(up_weight_scales)    + expert_offset * geom.gate_up_scale_stride;
    __gm__ std::uint8_t* down_packed_ptr = Bytes(down_weights_packed) + expert_offset * geom.down_packed_stride;
    __gm__ std::uint8_t* down_scales_ptr = Bytes(down_weight_scales)  + expert_offset * geom.down_scale_stride;

    for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += geom.inter_sub_fwd) {
        const std::uint32_t accum_elements = valid_rows * geom.inter_sub_fwd;
        Duplicate(buf.gate_accumulator, 0.0F, accum_elements);
        Duplicate(buf.up_accumulator,   0.0F, accum_elements);
        PipeBarrier<PIPE_V>();

        const std::uint32_t chunk_count = geom.hidden / geom.hidden_sub_fwd;
        std::uint32_t hidden_base = 0;
        bool is_ping              = true;

        // The first load into each slot has no buffer-free token to wait for, so the prologue and
        // the loads for chunks below the slot count pass `false`. `kTokenRingSlots` is the ring
        // depth, stated once so the guard and the drain cannot disagree about it.
        AsyncLoadTokensChunk(tokens_in_gm, token_row_start, hidden_base, valid_rows, geom, buf.tokens_f32_ping, kEvTokensPing, false);

        for (; hidden_base + geom.hidden_sub_fwd < geom.hidden; hidden_base += geom.hidden_sub_fwd) {
            const std::uint32_t next_hidden_base = hidden_base + geom.hidden_sub_fwd;
            const std::uint32_t next_chunk       = next_hidden_base / geom.hidden_sub_fwd;
            auto& curr_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
            auto& next_tokens_f32  = is_ping ? buf.tokens_f32_pong      : buf.tokens_f32_ping;
            auto& curr_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
            const int curr_event   = is_ping ? kEvTokensPing            : kEvTokensPong;
            const int next_event   = is_ping ? kEvTokensPong            : kEvTokensPing;

            AsyncLoadTokensChunk(tokens_in_gm, token_row_start, next_hidden_base, valid_rows, geom, next_tokens_f32, next_event, next_chunk >= kTokenRingSlots);

            PrepareTokensChunk(curr_tokens_bf16, curr_tokens_f32, valid_rows, geom.hidden_sub_fwd, curr_event);
            RunSubGemmAccumulate(matmul_gate_up, curr_tokens_bf16, gate_packed_ptr, gate_scales_ptr, inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.gate_accumulator, accum_elements);
            RunSubGemmAccumulate(matmul_gate_up, curr_tokens_bf16, up_packed_ptr,   up_scales_ptr,   inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.up_accumulator,   accum_elements);

            is_ping = !is_ping;
        }

        auto& final_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
        auto& final_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
        const int final_event   = is_ping ? kEvTokensPing            : kEvTokensPong;

        PrepareTokensChunk(final_tokens_bf16, final_tokens_f32, valid_rows, geom.hidden_sub_fwd, final_event);
        RunSubGemmAccumulate(matmul_gate_up, final_tokens_bf16, gate_packed_ptr, gate_scales_ptr, inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.gate_accumulator, accum_elements);
        RunSubGemmAccumulate(matmul_gate_up, final_tokens_bf16, up_packed_ptr,   up_scales_ptr,   inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.up_accumulator,   accum_elements);

        // The ring is re-primed on every `inter_base` pass, so it must also be drained on every
        // pass. Leaving the tokens outstanding would make the next pass's prologue -- which does
        // not wait -- run against a stale free token, and the last pass's would outlive the launch.
        DrainTokenRing(chunk_count);

        ApplySwiGLUActivation(buf, valid_rows, inter_base, geom, accum_elements);
    }

    bool has_pending_output_burst = false;

    for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += geom.hidden_sub_down) {
        const std::uint32_t down_accum_elements = valid_rows * geom.hidden_sub_down;

        if (has_pending_output_burst) {
            WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
            has_pending_output_burst = false;
        }

        Duplicate(buf.down_accumulator, 0.0F, down_accum_elements);
        PipeBarrier<PIPE_V>();

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += geom.inter_sub_down) {
            RunDownProjectionStep(matmul_down, down_packed_ptr, down_scales_ptr, hidden_base, inter_base, valid_rows, geom, buf, q_weight_down, down_accum_elements);
        }

        SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);

        DataCopyParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(valid_rows);
        burst_params.blockLen   = static_cast<std::uint16_t>((geom.hidden_sub_down * 4u) / 32u);
        burst_params.srcStride  = 0;
        burst_params.dstStride  = static_cast<std::uint16_t>(((geom.hidden - geom.hidden_sub_down) * 4u) / 32u);
        DataCopy(tokens_out_gm[(static_cast<std::uint64_t>(token_row_start) * geom.hidden) + hidden_base], buf.down_accumulator, burst_params);

        SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
        has_pending_output_burst = true;
    }

    if (has_pending_output_burst) {
        WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    }
}


#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_LEGACY_MATMUL_HPP
