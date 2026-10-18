// =================================================================================================
// dsv4_moe_prefill_cube_kernel.cpp -- Batched MoE Prefill, W1/W3/W2 on the Cube in BF16
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2)
// Kernel Task Mode    : KERNEL_TYPE_MIX_AIC_1_2 (1 Cube Core + 2 Vector Cores)
// Intermediates       : ZERO-GM WORKSPACE (on-chip resident Unified Buffer activation plane)
// Cube driver         : BARE-METAL TENSOR API. No Matmul framework, no KfcServer.
// =================================================================================================
//
// op_kernel/dsv4_moe_prefill_cube_arch35.h is the design document: the UB and L1 budget, the NZ
// layout arithmetic that makes the [N][K] -> [K][N] operand transpose free, the C0-block rule that
// decides which axis each AIV owns, the flag plan and the event plan all live there with their
// derivations and their measurements. This file is the SCHEDULE.
//
// -------------------------------------------------------------------------------------------------
// THE THREE-STAGE PIPELINE
// -------------------------------------------------------------------------------------------------
// Transcribed from the STEP 2 probe (tests/sim/kernel/test_dualdst_pipeline_kernel.cpp), which is
// the configuration that retires on the dav-3510 CAModel. Per contraction step k:
//
//   AIV (both, each owning the contraction half [s*K/2, +K/2) of the step)
//       MTE2   GM -> UB   step k+1's token half (FP32) and MXFP4 nibbles + E8M0 scales
//       V      UB -> UB   dequantize step k's weight halves to BF16; cast step k's tokens to BF16
//       MTE3   UB -> L1   step k's operands, written as a disjoint range of whole NZ C0 blocks
//
//   AIC
//       MTE1   L1 -> L0A  the token tile      (NZ -> NZ, no transpose)
//              L1 -> L0B  the weight tile(s)  (ZN -> ZN, no transpose -- the transpose is the
//                                              REINTERPRETATION of the AIV's NZ(N, K) write as
//                                              ZN(K, N); see the header)
//       M      Mmad into L0C, ACCUMULATING -- ccubeInitVal is true only on step 0
//       FIX    once, on the completed contraction: L0C -> both UBs, DUAL_DST_SPLIT_M
//
// THE ACCUMULATION INVARIANCE IS THE POINT. Every partial sum stays in L0C in FP32 for the whole
// contraction and the move-out runs EXACTLY ONCE. The framework path could not express that -- it
// owns L0C and retires it per sub-GEMM -- so it had to evict each K block to a UB FP32 accumulator
// and Add there, which injects a rounding per block and spends two accumulators per sub-GEMM.
//
// -------------------------------------------------------------------------------------------------
// WHY GATE AND UP SHARE ONE CONTRACTION LOOP
// -------------------------------------------------------------------------------------------------
// W1 (gate) and W3 (up) reduce against the SAME token tile, so running them in one loop with two
// L0C accumulators and two L0B operands reads the token tile once instead of twice. That is the
// reuse the k-inner loop order exists for, and the framework path never realised it: it serialized
// dequant-W1 / GEMM / free / dequant-W3 / GEMM, so one weight ring slot was live at a time and the
// second was headroom.
//
// -------------------------------------------------------------------------------------------------
// NUMERICS ARE UNCHANGED, DELIBERATELY
// -------------------------------------------------------------------------------------------------
// The Fixpipe moves FP32 out to FP32 UB (CheckL0C2UbDataType's NoQuant case,
// `__ubuf__ float <- __cc__ float`), NOT FP32 -> BF16. SwiGLU therefore still runs in FP32 on
// FP32 pre-activations, exactly as the framework path did, and the only BF16 rounding on the
// activation is the one the resident plane always had. A BF16 Fixpipe would have been one
// instruction cheaper and would have put a 2^-9 relative error on every pre-activation -- a change
// to what this operator computes, which is not what an integration of a schedule should smuggle in.
// The emu golden therefore applies verbatim; the arithmetic below is op-for-op the arithmetic the
// framework path performed, in the same order within each row.
//
// -------------------------------------------------------------------------------------------------
// WHAT IS VERIFIED AND WHAT IS NOT
// -------------------------------------------------------------------------------------------------
// COMPILED AND LINKED FOR dav-3510; NEVER EXECUTED. The STEP 1 and STEP 2 probes measured the
// Fixpipe dual-destination split, the mode 2 handshake and the three-stage overlap on the CAModel,
// and this schedule is built from those primitives unchanged. Two things in it are NOT covered by
// those measurements and are called out at their definition sites:
//
//   * the NZ(N, K) == ZN(K, N) reinterpretation that supplies B = W^T (header, "THE WEIGHTS ARE
//     [N][K]"). Derived from the toolkit's layout arithmetic. The STEP 2 history says to settle
//     any doubt with a SELECTOR A, not by reading addresses again.
//   * `PushUbTileToL1Nz`, which writes the NZ slots with plain strided bursts instead of the
//     Nd2NzParams overload (header, "THE C0-BLOCK RULE"). It emits what the toolkit's own
//     DataCopyUB2L1ND2NZImplV2 emits, and it is what lets one AIV write a ROW RANGE of a slot --
//     which the Nd2NzParams path provably cannot.
// =================================================================================================

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
// THE FRAMEWORK, AND IT IS NOT USED BY ANYTHING BELOW THE QUARANTINE LINE.
//
// `lib/matmul_intf.h` has to be included HERE, at file scope, rather than from
// dsv4_moe_prefill_cube_legacy_matmul.h which is what actually needs it: that header is included
// inside this file's namespace, and matmul_intf's own sources name `AscendC::` explicitly, so
// including it from there resolves `AscendC` to a nested namespace that does not exist. No Matmul
// object is instantiated and REGIST_MATMUL_OBJ is never called, so no KfcServer is stood up --
// deleting the quarantined header and this line together is all it takes to remove the framework
// from the operator.
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

// NOT `using namespace AscendC::Te`: the Tensor API redeclares RoundMode, MmadParams and Mmad in
// its own namespace, and pulling both in unqualified makes every one of them ambiguous.
namespace Te = AscendC::Te;

static_assert(kCubeMTile == kPrefillCubeMTile,     "token tile disagrees between the UB layout and the tiling header the host validates from");
static_assert(kCubeRSub  == kPrefillCubeRSub,      "phase 1 inter output sub-tile disagrees");
static_assert(kCubeKSub  == kPrefillCubeKSub,      "phase 1 hidden contraction sub-tile disagrees");
static_assert(kCubeHSub  == kPrefillCubeHSub,      "phase 2 hidden output sub-tile disagrees");
static_assert(kCubeRSub2 == kPrefillCubeRSub2,     "phase 2 inter contraction sub-tile disagrees");
static_assert(kCubeGroup == kPrefillCubeGroupSize, "MXFP4 group size disagrees");
// The act plane's capacity. The HOST copy is authoritative because the host admissibility
// predicate is what rejects `inter > cap`; this one only sizes the buffer. A drift between them is
// a plane that is smaller than the geometry the host admitted, which is a silent write past its
// end, so it is a compile error.
static_assert(kCubeActInterCap == kPrefillCubeInterCap, "act plane capacity disagrees between the UB layout and the tiling header the host validates from");

namespace {

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR ptr) {
    return reinterpret_cast<__gm__ std::uint8_t*>(ptr);
}

struct Geometry {
    std::uint32_t block_dim;
    std::uint32_t hidden;
    std::uint32_t inter;
    std::uint32_t num_experts;
    std::uint32_t total_tokens;
    std::uint32_t token_tile;
    // Read only by the quarantined legacy routines; the live path takes every sub-tile extent
    // from the compile-time constants, which TilingRejected has already checked these against.
    std::uint32_t inter_sub_fwd;
    std::uint32_t hidden_sub_fwd;
    std::uint32_t hidden_sub_down;
    std::uint32_t inter_sub_down;
    std::uint32_t gate_up_packed_stride;
    std::uint32_t gate_up_scale_stride;
    std::uint32_t down_packed_stride;
    std::uint32_t down_scale_stride;
    float         swiglu_limit;
};

__aicore__ inline Geometry LoadGeometry(const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data) {
    Geometry geom;
    geom.block_dim              = tiling_data->block_dim;
    geom.hidden                 = tiling_data->hidden;
    geom.inter                  = tiling_data->inter;
    geom.num_experts            = tiling_data->num_experts;
    geom.total_tokens           = tiling_data->total_tokens;
    geom.token_tile             = tiling_data->m_tile;
    geom.inter_sub_fwd          = tiling_data->r_sub;
    geom.hidden_sub_fwd         = tiling_data->k_sub;
    geom.hidden_sub_down        = tiling_data->h_sub;
    geom.inter_sub_down         = tiling_data->r_sub2;
    geom.gate_up_packed_stride  = tiling_data->w13_packed_expert_stride;
    geom.gate_up_scale_stride   = tiling_data->w13_scale_expert_stride;
    geom.down_packed_stride     = tiling_data->w2_packed_expert_stride;
    geom.down_scale_stride      = tiling_data->w2_scale_expert_stride;
    geom.swiglu_limit           = tiling_data->swiglu_limit;
    return geom;
}

__aicore__ inline bool TilingRejected(const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data) {
    const std::uint32_t hidden = tiling_data->hidden;
    const std::uint32_t inter  = tiling_data->inter;
    return tiling_data->abi_version      != kTilingAbiVersion ||
           tiling_data->group_size       != kCubeGroup        ||
           tiling_data->low_nibble_first != 1u                ||
           tiling_data->num_experts      == 0u                ||
           tiling_data->total_tokens     == 0u                ||
           tiling_data->m_tile           != kCubeMTile        ||
           tiling_data->r_sub            != kCubeRSub         ||
           tiling_data->k_sub            != kCubeKSub         ||
           tiling_data->h_sub            != kCubeHSub         ||
           tiling_data->r_sub2           != kCubeRSub2        ||
           !Dsv4MoePrefillCubeAdmissible(hidden, inter);
}

// THE EXPERT LOOKUP IS READ ON ALL THREE CORES, AND THAT IS NOT REDUNDANT WORK.
//
// The AIC and its two AIVs are a lockstep triple: every cross-core flag is a rendezvous, so a core
// that SKIPS a token tile the others process deadlocks the pair. The framework path could return
// early from the AIV because the AIC was a passive server; here the AIC is a participant. So the
// admission decision has to be a pure function of GM state that every core evaluates identically,
// and it is this one.
__aicore__ inline bool TileAdmitted(GM_ADDR token_to_expert_lut, std::uint32_t token_row_start,
                                    std::uint32_t num_experts, std::uint32_t& expert_id_out) {
    const std::uint32_t expert_id = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert_lut)[token_row_start]);
    expert_id_out = expert_id;
    return expert_id < num_experts;
}

}  // namespace

// Per-weight-stream, per-ring-slot dequantization staging: everything
// AsyncLoadWeightsSubTile writes and DequantSubTileVector reads, and nothing else.
//
// THE MEMBER NAMES ARE LOAD-BEARING. Five sim probes textually include this file and populate
// exactly these fields to reuse the two routines below verbatim, and
// tests/sim/kernel/test_dualdst_pipeline_kernel.cpp does so through a `Buffers`, which the
// quarantined legacy header derives from this struct. A `Buffers` therefore still converts to a
// `const WeightStage&` and both callers compile against one definition of the dequantizer.
struct WeightStage {
    LocalTensor<fp4x2_e2m1_t> weight_nibbles;
    LocalTensor<bfloat16_t>   weight_expanded;
    LocalTensor<uint8_t>      weight_scales;
    LocalTensor<bfloat16_t>   scale_plane;
    LocalTensor<uint16_t>     scale_temporary;
};

__aicore__ inline void AsyncLoadWeightsSubTile(
    __gm__ std::uint8_t* packed_weights, __gm__ std::uint8_t* scale_factors,
    std::uint32_t row_start, std::uint32_t col_start, std::uint32_t row_count, std::uint32_t col_count, std::uint32_t matrix_cols,
    const WeightStage& buf, int mte2_to_v_event = kEvCubeMte2ToV,
    int v_to_mte2_event = -1) {
    // `v_to_mte2_event` >= 0 adds the WAR half of the double buffer: it waits
    // for the Vector unit to have finished READING this slot's nibble/scale
    // staging before the burst overwrites it. DEFAULTS TO -1 (no wait), so
    // existing callers -- which reuse one staging buffer and are serialized
    // by their own MTE2_V token anyway -- are unchanged.
    if (v_to_mte2_event >= 0) {
        WaitFlag<HardEvent::V_MTE2>(v_to_mte2_event);
    }
    // `mte2_to_v_event` is a parameter because an intra-core flag is a BINARY
    // flag and not a counting semaphore: raising MTE2_V for tile k+1 while
    // tile k's token is still outstanding is undefined behaviour, and the
    // simulator reports it as
    // `pem_ccu.cc execute_set_flag already has same set_flag!`. A caller that
    // wants tile k+1's burst IN FLIGHT while the Vector unit is still
    // dequantizing tile k therefore needs one id per ring slot, and must pass
    // the SAME id to the matching DequantSubTileVector.
    //
    // PASSING -1 SUPPRESSES THE SetFlag ENTIRELY, which is how a caller issues
    // SEVERAL bursts under ONE token. MTE2 is in order, so a single flag
    // raised after the LAST burst of a ring slot covers every burst into it --
    // and that is what keeps this operator inside the six usable event ids
    // while loading a token half and two weight halves per step.
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t row_groups     = matrix_cols / kCubeGroup;

    {
        const std::uint64_t byte_offset = static_cast<std::uint64_t>(row_start) * (matrix_cols / 2u) + (col_start / 2u);
        GlobalTensor<fp4x2_e2m1_t> packed_gm;
        packed_gm.SetGlobalBuffer(reinterpret_cast<__gm__ fp4x2_e2m1_t*>(packed_weights + byte_offset), static_cast<std::uint64_t>(row_count) * col_count);
        DataCopyParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(row_count);
        burst_params.blockLen   = static_cast<std::uint16_t>((col_count / 2u) / 32u);
        burst_params.srcStride  = static_cast<std::uint16_t>(((matrix_cols - col_count) / 2u) / 32u);
        burst_params.dstStride  = 0;
        DataCopy(buf.weight_nibbles, packed_gm, burst_params);
    }

    {
        const std::uint64_t scale_offset = static_cast<std::uint64_t>(row_start) * row_groups + (col_start / kCubeGroup);
        GlobalTensor<uint8_t> scale_gm;
        scale_gm.SetGlobalBuffer(scale_factors + scale_offset, static_cast<std::uint64_t>(row_count) * groups_per_row);
        DataCopyExtParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(row_count);
        burst_params.blockLen   = groups_per_row;
        burst_params.srcStride  = row_groups - groups_per_row;
        burst_params.dstStride  = 0;
        burst_params.rsv        = 0;
        DataCopyPadExtParams<uint8_t> pad_params;
        pad_params.isPad        = false;
        pad_params.leftPadding  = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;
        DataCopyPad(buf.weight_scales, scale_gm, burst_params, pad_params);
    }

    if (mte2_to_v_event >= 0) {
        SetFlag<HardEvent::MTE2_V>(mte2_to_v_event);
    }
}

__aicore__ inline void DequantSubTileVector(
    std::uint32_t row_count, std::uint32_t col_count,
    const WeightStage& buf, int mte2_to_v_event = kEvCubeMte2ToV) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t total_elements = row_count * col_count;

    // Must be the id the matching AsyncLoadWeightsSubTile raised -- SetFlag
    // and WaitFlag are paired on the (srcPipe, dstPipe, eventID) triple, and
    // a mismatched id does not fail, it times out. -1 means "the token for
    // this burst was already consumed by an earlier dequant in the same step",
    // which is the other half of the one-flag-per-slot economy above.
    if (mte2_to_v_event >= 0) {
        WaitFlag<HardEvent::MTE2_V>(mte2_to_v_event);
    }

    Cast(buf.weight_expanded, buf.weight_nibbles, RoundMode::CAST_NONE, total_elements);

    LocalTensor<uint16_t>   row_scales_u16  = buf.scale_temporary;
    LocalTensor<bfloat16_t> row_scales_bf16 = buf.scale_temporary.ReinterpretCast<bfloat16_t>();

    const uint32_t dst_shape[2] = {groups_per_row, kCubeGroup};
    const uint32_t src_shape[2] = {groups_per_row, 1};

    // ONE ROW AT A TIME, with the Mul INSIDE the loop.
    //
    // `buf.scale_plane` is one row wide, not a plane: a row's scales are used only by that row's
    // multiply. WHY IT MATTERED: the old Broadcast into a plane was the instruction the simulator
    // reported as `ub_addr_overflow`, 1,280 times per call, because the UB layout over-committed
    // and that was the buffer sitting past the end of it.
    //
    // The arithmetic is elementwise multiply, same operands, same order within a row, so the emu
    // golden is unaffected.
    for (std::uint32_t r_idx = 0; r_idx < row_count; ++r_idx) {
        const std::uint32_t scale_base = r_idx * kCubeScaleRowStride;
        const std::uint32_t out_base   = r_idx * col_count;

        Cast(row_scales_u16, buf.weight_scales[scale_base], RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        ShiftLeft(row_scales_u16, row_scales_u16, static_cast<uint16_t>(7), 16);
        PipeBarrier<PIPE_V>();

        Broadcast<bfloat16_t, 2, 1>(buf.scale_plane, row_scales_bf16, dst_shape, src_shape);
        PipeBarrier<PIPE_V>();

        Mul(buf.weight_expanded[out_base], buf.weight_expanded[out_base],
            buf.scale_plane, col_count);
        PipeBarrier<PIPE_V>();
    }
}

namespace {

// ---------------------------------------------------------------------------------------------
// UB -> L1, writing a dense ND tile into a range of an NZ slot's C0 blocks
// ---------------------------------------------------------------------------------------------
// An NZ(nz_rows, C) slot is C/16 blocks, each a dense [nz_rows][16] row-major run, blocks strided
// 16*nz_rows apart. (Fold the NZ address function in the header: r1*256 + r0*16 == r*16.) So one
// block is `nz_rows` consecutive 32-byte units, and this routine issues exactly one strided burst
// per block -- instruction for instruction what the toolkit's own DataCopyUB2L1ND2NZImplV2 emits.
//
// `rows` may be LESS than `nz_rows`: `row_base` then selects a row range of every block, which is
// how the two AIVs each contribute their half of the phase 2 token tile. The Nd2NzParams overload
// cannot express that -- see "THE C0-BLOCK RULE" in the header for the actualSize derivation that
// shows it would carry uninitialized stack bytes over the other core's rows.
//
// `src_row_stride` is the SOURCE row pitch in elements, which is not always `cols`: the phase 2
// token push reads a window of the activation plane, whose pitch is `inter`.
//
// Preconditions, all compile-time or geometry-checked by the host:
//   cols % 16 == 0, src_row_stride % 16 == 0, col_base % 16 == 0
//   sizeof(T) == 2, so one 32-byte burst unit is exactly 16 elements
__aicore__ inline void PushUbTileToL1Nz(
    const LocalTensor<bfloat16_t>& l1_slot, const LocalTensor<bfloat16_t>& ub_tile,
    std::uint32_t rows, std::uint32_t cols, std::uint32_t src_row_stride,
    std::uint32_t nz_rows, std::uint32_t col_base, std::uint32_t row_base) {
    DataCopyParams burst;
    burst.blockCount = static_cast<std::uint16_t>(rows);
    burst.blockLen   = 1;
    burst.srcStride  = static_cast<std::uint16_t>(src_row_stride / kCubeC0 - 1u);
    burst.dstStride  = 0;

    const std::uint32_t blocks = cols / kCubeC0;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        DataCopy(l1_slot[(col_base + b * kCubeC0) * nz_rows + row_base * kCubeC0],
                 ub_tile[b * kCubeC0], burst);
    }
}

// --------------------------------------------------------------------------------------------
// The Fixpipe trait, and the one thing about it that is not guessable
// ---------------------------------------------------------------------------------------------
// float -> float with no `quant` argument is CheckL0C2UbDataType's NoQuant case
// (`Std::tuple<__ubuf__ float, __cc__ float>`), so this is a straight move-out with no conversion.
// DUAL_DST_SPLIT_M bisects M across the two AIVs' separately-addressed UBs.
//
// THE DESTINATION LAYOUT CARRIES THE FULL M, NOT M/2, even though each UB receives only M/2 rows
// and the landing buffer is only M/2 * N elements. MEASURED in STEP 1: a dst layout of M/2 = 8
// delivered FOUR rows to each UB -- it had halved the 8 again. With the full M = 16 each side gets
// its 8. The shipping turboquant kernel does the same: full (even-padded) M as the Fixpipe mSize,
// kHalfRows = M/2 rows of destination UB.
constexpr Te::CopyL0C2UBTrait kCubeL0c2UbTrait = {
    Te::RoundMode::DEFAULT,
    false,                  // enableRelu
    false,                  // enableChannelSplit
    Te::DUAL_DST_SPLIT_M
};

struct CubeL0c2UbTraitDualDstM {
    using TraitType = Te::CopyL0C2UBTrait;
    static constexpr const TraitType value = kCubeL0c2UbTrait;
};

// The pipeline each CrossCoreWaitFlag scopes its block to: the one that CONSUMES what the flag
// protects. An effective mode 2 wait blocks ONLY the pipeline it names (950PR), and naming the
// wrong one does not fail loudly -- it stops gating the consumer, which is a silent wrong answer
// rather than a hang. Per site:
//
//   AIC waits StepReady   -> next is CopyL12L0A/B out of the L1 slot, MTE1.
//   AIC waits LandingFree -> next is the Fixpipe write into the landing zone, FIX.
//   AIV waits StepFree    -> what it gates is this AIV's push INTO that L1 slot, MTE3.
//   AIV waits Product (phase 1) -> next is Mins/Maxs reading the landing zone, V.
//   AIV waits Product (phase 2) -> next is DataCopy(out, landing), UB -> GM, MTE3.
//
// `PIPE_FIX` IS NOT SPELLED HERE, and that is a toolchain fact rather than an inconsistency.
// Including `tensor_api/tensor.h` makes the name unresolvable at NAMESPACE SCOPE -- MEASURED:
// `constexpr pipe_t x = PIPE_FIX;` at namespace scope is "use of undeclared identifier
// 'PIPE_FIX'" with tensor.h included and compiles without it, while the same name inside a
// function body compiles either way (which is why the STEP 1 probe, which also includes tensor.h,
// gets away with it). So the AIC's landing-zone wait names PIPE_FIX at its call site, in
// CubeMoveOutAndHandOver.
constexpr pipe_t kCubeWaitAicStep    = PIPE_MTE1;
constexpr pipe_t kCubeWaitAivStep    = PIPE_MTE3;
constexpr pipe_t kCubeWaitAivVector  = PIPE_V;
constexpr pipe_t kCubeWaitAivStore   = PIPE_MTE3;

// ---------------------------------------------------------------------------------------------
// AIV state
// ---------------------------------------------------------------------------------------------
struct AivBuffers {
    LocalTensor<float>      landing;   // MUST be the first InitBuffer: the AIC names UB offset 0
    LocalTensor<bfloat16_t> act;       // [M/2, inter_cap], resident across the phase boundary
    LocalTensor<float>      gate_acc;
    LocalTensor<float>      up_acc;
    LocalTensor<bfloat16_t> act_sub;
    LocalTensor<float>      x_f32[kCubeRingSlots];
    LocalTensor<bfloat16_t> x_bf16[kCubeRingSlots];
    WeightStage             w[kCubeRingSlots][kCubeWeightStreams];
};

// One outstanding deferred token per ring slot, tracked explicitly so the drain after each
// contraction consumes exactly what the contraction raised. A flag left raised is not harmless:
// the next SetFlag on the same triple lands on an already-set flag, which is the undefined
// behaviour the simulator reports as `execute_set_flag already has same set_flag!` -- and since a
// contraction runs once per inter block, a leak would surface as a hang many blocks later.
struct AivRingState {
    bool stage_live[kCubeRingSlots];   // V_MTE2  : staging may be overwritten
    bool pushed_live[kCubeRingSlots];  // MTE3_V  : `wide` may be overwritten
};

__aicore__ inline void AivResetRing(AivRingState& ring) {
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        ring.stage_live[s]  = false;
        ring.pushed_live[s] = false;
    }
}

__aicore__ inline void AivDrainRing(AivRingState& ring) {
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        if (ring.stage_live[s]) {
            WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
            ring.stage_live[s] = false;
        }
        if (ring.pushed_live[s]) {
            WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[s]);
            ring.pushed_live[s] = false;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// PHASE 1, AIV: one contraction step's MTE2 prefetch
// ---------------------------------------------------------------------------------------------
// The token half and BOTH weight halves are issued back to back under ONE MTE2_V token, raised by
// the last of the three. MTE2 is in order, so that token covers all three bursts.
__aicore__ inline void AivPrefetchFwdStep(
    const GlobalTensor<float>& tokens_in_gm, std::uint32_t token_row_start,
    __gm__ std::uint8_t* gate_packed, __gm__ std::uint8_t* gate_scales,
    __gm__ std::uint8_t* up_packed,   __gm__ std::uint8_t* up_scales,
    std::uint32_t inter_base, std::uint32_t hidden_base, std::uint32_t valid_rows,
    std::uint32_t sub_idx, const Geometry& geom,
    const AivBuffers& buf, std::uint32_t slot, AivRingState& ring) {
    // WAR on the whole slot's staging: the Vector unit must have finished READING the token half
    // and both nibble/scale pairs before this burst overwrites them. One token covers all of it
    // because they are written by one MTE2 run and read by one Vector run.
    if (ring.stage_live[slot]) {
        WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
        ring.stage_live[slot] = false;
    }

    // This AIV's contraction half of the token tile. ONLY `valid_rows` ROWS ARE READ; the rows
    // past the tail keep the zeros the setup Duplicate left, which is what makes the Cube's
    // always-full-M contraction correct without any tail arithmetic -- a zero A row produces a
    // zero C row, and that row is simply not stored.
    const std::uint32_t col_start = hidden_base + sub_idx * kCubeKHalf;
    DataCopyParams token_burst;
    token_burst.blockCount = static_cast<std::uint16_t>(valid_rows);
    token_burst.blockLen   = static_cast<std::uint16_t>((kCubeKHalf * 4u) / 32u);
    token_burst.srcStride  = static_cast<std::uint16_t>(((geom.hidden - kCubeKHalf) * 4u) / 32u);
    token_burst.dstStride  = 0;
    DataCopy(buf.x_f32[slot],
             tokens_in_gm[(static_cast<std::uint64_t>(token_row_start) * geom.hidden) + col_start],
             token_burst);

    AsyncLoadWeightsSubTile(gate_packed, gate_scales, inter_base, col_start,
                            kCubeRSub, kCubeKHalf, geom.hidden, buf.w[slot][0],
                            /*mte2_to_v_event=*/-1);
    AsyncLoadWeightsSubTile(up_packed, up_scales, inter_base, col_start,
                            kCubeRSub, kCubeKHalf, geom.hidden, buf.w[slot][1],
                            kEvCubeSlotFull[slot]);
}

// PHASE 1, AIV: one contraction step's Vector work, then its three L1 pushes.
__aicore__ inline void AivConvertAndPushFwdStep(
    const LocalTensor<bfloat16_t>& l1_tokens,
    const LocalTensor<bfloat16_t>& l1_gate,
    const LocalTensor<bfloat16_t>& l1_up,
    std::uint32_t sub_idx, const AivBuffers& buf, std::uint32_t slot, AivRingState& ring) {
    // WAR on this slot's `wide` buffers: their push, a full ring lap ago, must have drained before
    // the dequant overwrites them. Deferred to here from immediately-after-the-push, which is the
    // serialization the three-stage overlap exists to remove.
    if (ring.pushed_live[slot]) {
        WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
        ring.pushed_live[slot] = false;
    }

    // The production FP4 -> BF16 conversion, unchanged. The FIRST call consumes this slot's MTE2_V
    // token; the second must not wait again, because one token covered both bursts.
    DequantSubTileVector(kCubeRSub, kCubeKHalf, buf.w[slot][0], kEvCubeSlotFull[slot]);
    DequantSubTileVector(kCubeRSub, kCubeKHalf, buf.w[slot][1], -1);
    Cast(buf.x_bf16[slot], buf.x_f32[slot], RoundMode::CAST_RINT, kCubeMTile * kCubeKHalf);

    // The staging is readable-no-more by anyone until the next burst; hand the WAR token over.
    SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
    ring.stage_live[slot] = true;

    // RAW: the Vector unit wrote x_bf16 and both `wide` buffers, MTE3 reads them. This pair stays
    // IMMEDIATE and that is correct rather than lockstep -- a WaitFlag<V_MTE3> blocks only
    // PIPE_MTE3, so the Vector unit runs straight on into the next iteration behind it.
    SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);

    const std::uint32_t col_base = sub_idx * kCubeKHalf;
    PushUbTileToL1Nz(l1_tokens, buf.x_bf16[slot], kCubeMTile, kCubeKHalf, kCubeKHalf,
                     kCubeMTile, col_base, 0);
    PushUbTileToL1Nz(l1_gate, buf.w[slot][0].weight_expanded, kCubeRSub, kCubeKHalf, kCubeKHalf,
                     kCubeRSub, col_base, 0);
    PushUbTileToL1Nz(l1_up, buf.w[slot][1].weight_expanded, kCubeRSub, kCubeKHalf, kCubeKHalf,
                     kCubeRSub, col_base, 0);

    // DEFERRED, NO IMMEDIATE WAIT. This is the one edge that turns the loop from lockstep into an
    // overlap: an immediate WaitFlag<MTE3_V> here would block PIPE_V until the pushes had drained,
    // so the next dequant could not start. The token is consumed a full ring lap later, at the top
    // of the iteration that reuses this slot.
    SetFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
    ring.pushed_live[slot] = true;
}

// PHASE 2, AIV: one contraction step's MTE2 prefetch. One weight stream, no token burst -- phase 2
// reads its A operand out of the resident activation plane.
__aicore__ inline void AivPrefetchDownStep(
    __gm__ std::uint8_t* down_packed, __gm__ std::uint8_t* down_scales,
    std::uint32_t hidden_base, std::uint32_t inter_base, std::uint32_t sub_idx,
    const Geometry& geom, const AivBuffers& buf, std::uint32_t slot, AivRingState& ring) {
    if (ring.stage_live[slot]) {
        WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
        ring.stage_live[slot] = false;
    }
    AsyncLoadWeightsSubTile(down_packed, down_scales, hidden_base,
                            inter_base + sub_idx * kCubeRHalf2,
                            kCubeHSub, kCubeRHalf2, geom.inter, buf.w[slot][0],
                            kEvCubeSlotFull[slot]);
}

// PHASE 2, AIV: dequantize the weight half and push both operands.
//
// THE TWO PUSHES SPLIT ON DIFFERENT AXES, and that is the one place the two bisections meet.
//   the weight half is a CONTRACTION half  -> a C0-block range, row_base 0
//   the token half   is a TOKEN-ROW half   -> a row range of every block, col_base 0
// The second is only expressible because PushUbTileToL1Nz writes blocks itself; see the header.
__aicore__ inline void AivConvertAndPushDownStep(
    const LocalTensor<bfloat16_t>& l1_act,
    const LocalTensor<bfloat16_t>& l1_down,
    std::uint32_t inter_base, std::uint32_t sub_idx, const Geometry& geom,
    const AivBuffers& buf, std::uint32_t slot, AivRingState& ring) {
    if (ring.pushed_live[slot]) {
        WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
        ring.pushed_live[slot] = false;
    }

    DequantSubTileVector(kCubeHSub, kCubeRHalf2, buf.w[slot][0], kEvCubeSlotFull[slot]);

    SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
    ring.stage_live[slot] = true;

    SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);

    PushUbTileToL1Nz(l1_down, buf.w[slot][0].weight_expanded,
                     kCubeHSub, kCubeRHalf2, kCubeRHalf2,
                     kCubeHSub, sub_idx * kCubeRHalf2, 0);
    // Straight out of the resident plane: its row pitch is `inter`, not the tile width, which is
    // why PushUbTileToL1Nz takes the source pitch separately.
    PushUbTileToL1Nz(l1_act, buf.act[inter_base],
                     kCubeMHalf, kCubeRSub2, geom.inter,
                     kCubeMTile, 0, sub_idx * kCubeMHalf);

    SetFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
    ring.pushed_live[slot] = true;
}

// ---------------------------------------------------------------------------------------------
// SwiGLU, on this AIV's M/2 token rows of one inter block
// ---------------------------------------------------------------------------------------------
// OP FOR OP THE FRAMEWORK PATH'S ARITHMETIC, in the same order, on the same FP32 operands:
//   gate is clamped ABOVE by the limit only; up is clamped to [-limit, limit]
//   y = Silu(gate) * up, then cast to BF16 for the resident plane
// The restructuring is only in WHERE the two pre-activations come from: each arrives from the
// Fixpipe in the single landing zone, so the first operation on each is the one that also moves it
// out of the landing zone and frees it for the next move-out.
//
// NO PipeBarrier<PIPE_V> BETWEEN THESE. They are intra-vector read-after-writes on the same
// buffers and the compiler's automatic synchronization covers them; a manual barrier here drains
// the vector pipeline every inter block for nothing. The V -> MTE3 edge onto the plane is a
// different dependency and IS fenced, below.
__aicore__ inline void AivApplySwiGLU(const AivBuffers& buf, std::uint32_t inter_base,
                                      const Geometry& geom) {
    const std::uint32_t elems = kCubeMHalf * kCubeRSub;

    Silu(buf.gate_acc, buf.gate_acc, elems);
    Mul(buf.gate_acc, buf.gate_acc, buf.up_acc, elems);
    Cast(buf.act_sub, buf.gate_acc, RoundMode::CAST_RINT, elems);

    // V -> MTE3: the plane store below is a strided burst, not a vector op.
    SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);

    DataCopyParams plane_store;
    plane_store.blockCount = static_cast<std::uint16_t>(kCubeMHalf);
    plane_store.blockLen   = static_cast<std::uint16_t>((kCubeRSub * 2u) / 32u);
    plane_store.srcStride  = 0;
    plane_store.dstStride  = static_cast<std::uint16_t>(((geom.inter - kCubeRSub) * 2u) / 32u);
    DataCopy(buf.act[inter_base], buf.act_sub, plane_store);
}

}  // namespace

// =================================================================================================
// THE VECTOR CORE
// =================================================================================================
__aicore__ inline void RunVectorCore(
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
    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx());

    // The token rows this AIV owns, end to end: the Fixpipe's M split decides them, and the
    // activation plane, SwiGLU and the output store all follow it.
    const std::uint32_t row_base = sub_idx * kCubeMHalf;

    AivRingState ring;

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;
        const std::uint32_t remaining       = geom.total_tokens - token_row_start;
        const std::uint32_t valid_rows      = remaining < geom.token_tile ? remaining : geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            continue;  // the AIC evaluates the same predicate and skips the same tile
        }
        const std::uint64_t expert_offset = static_cast<std::uint64_t>(expert_id);

        __gm__ std::uint8_t* gate_packed = Bytes(gate_weights_packed) + expert_offset * geom.gate_up_packed_stride;
        __gm__ std::uint8_t* gate_scales = Bytes(gate_weight_scales)  + expert_offset * geom.gate_up_scale_stride;
        __gm__ std::uint8_t* up_packed   = Bytes(up_weights_packed)   + expert_offset * geom.gate_up_packed_stride;
        __gm__ std::uint8_t* up_scales   = Bytes(up_weight_scales)    + expert_offset * geom.gate_up_scale_stride;
        __gm__ std::uint8_t* down_packed = Bytes(down_weights_packed) + expert_offset * geom.down_packed_stride;
        __gm__ std::uint8_t* down_scales = Bytes(down_weight_scales)  + expert_offset * geom.down_scale_stride;

        // --- PHASE 1: gate and up, one inter block at a time ------------------------------------
        const std::uint32_t fwd_steps = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            AivResetRing(ring);

            // Prologue: step 0's burst is issued before the loop so that iteration 0 can issue
            // step 1's burst before converting step 0. No WAR wait on this one -- nothing has read
            // slot 0 yet, and AivResetRing has just declared that.
            AivPrefetchFwdStep(tokens_in_gm, token_row_start, gate_packed, gate_scales,
                               up_packed, up_scales, inter_base, 0u, valid_rows,
                               sub_idx, geom, buf, 0u, ring);

            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot      = k % kCubeRingSlots;
                const std::uint32_t next_slot = (k + 1u) % kCubeRingSlots;

                // THE PREFETCH PRECEDES THE CONVERT, which is what makes step k+1's MTE2 burst run
                // against step k's Vector work rather than merely against the Cube.
                if (k + 1u < fwd_steps) {
                    AivPrefetchFwdStep(tokens_in_gm, token_row_start, gate_packed, gate_scales,
                                       up_packed, up_scales, inter_base,
                                       (k + 1u) * kCubeKSub, valid_rows,
                                       sub_idx, geom, buf, next_slot, ring);
                }

                // Scoped to PIPE_MTE3: what this flag protects is this AIV's push INTO the L1
                // slot, and MTE3 is what performs it. Leaving the Vector pipe unblocked is
                // deliberate -- the dequant below does not touch L1.
                if (k >= kCubeRingSlots) {
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                AivConvertAndPushFwdStep(l1a_buf[slot].Get<bfloat16_t>(),
                                         l1w_buf[slot][0].Get<bfloat16_t>(),
                                         l1w_buf[slot][1].Get<bfloat16_t>(),
                                         sub_idx, buf, slot, ring);

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
            }

            AivDrainRing(ring);

            // The epilogue, twice through ONE landing zone. The first operation on each
            // pre-activation is also the one that moves it out and frees the zone.
            const std::uint32_t elems = kCubeMHalf * kCubeRSub;

            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            Mins(buf.gate_acc, buf.landing, geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);

            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            Maxs(buf.up_acc, buf.landing, -geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
            Mins(buf.up_acc, buf.up_acc, geom.swiglu_limit, elems);

            AivApplySwiGLU(buf, inter_base, geom);
        }

        // --- PHASE 2: the down projection, one hidden block at a time ---------------------------
        const std::uint32_t down_steps = geom.inter / kCubeRSub2;
        // Rows of this AIV's half that are real tokens rather than zero padding.
        const std::uint32_t owned      = valid_rows > row_base ? (valid_rows - row_base) : 0u;
        const std::uint32_t store_rows = owned < kCubeMHalf ? owned : kCubeMHalf;

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
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
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                AivConvertAndPushDownStep(l1a2_buf[slot].Get<bfloat16_t>(),
                                          l1w2_buf[slot].Get<bfloat16_t>(),
                                          r * kCubeRSub2, sub_idx, geom, buf, slot, ring);

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
            }

            AivDrainRing(ring);

            // Scoped to PIPE_MTE3: the store below is what must not run until the Fixpipe has
            // landed the product in this AIV's UB.
            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStore>(kCubeFlagProduct);

            // THE WAIT IS UNCONDITIONAL AND THE STORE IS NOT. A tail tile can leave this AIV with
            // no real rows (valid_rows <= row_base), but the flag is a rendezvous: skipping the
            // wait would leave the AIC's set unconsumed and desynchronize every later block.
            if (store_rows > 0u) {
                DataCopyParams out_burst;
                out_burst.blockCount = static_cast<std::uint16_t>(store_rows);
                out_burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
                out_burst.srcStride  = 0;
                out_burst.dstStride  = static_cast<std::uint16_t>(((geom.hidden - kCubeHSub) * 4u) / 32u);
                DataCopy(tokens_out_gm[(static_cast<std::uint64_t>(token_row_start + row_base) *
                                        geom.hidden) + hidden_base],
                         buf.landing, out_burst);
            }

            CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
        }
    }
}

// =================================================================================================
// THE CUBE CORE
// =================================================================================================
// L0A/L0B/L0C are raw __ca__/__cb__/__cc__ arrays: they are Cube-private, the AIV has no such
// memory, and TPipe would have to be told so.
//
// ONE ARRAY PER OPERAND, NOT AN ARRAY OF ARRAYS. The gate and up streams get separately named
// buffers because `l0b[w]` -- indexing a two-dimensional __cb__ array -- is rejected by BiSheng
// ("only __ubuf__ and __gm__ and local memory pointer can be dereferenced"): the inner dereference
// produces an address-space pointer the front end will not form. Passing the buffer to the small
// helpers below as a `__cb__ bfloat16_t*` is fine; indexing a 2D array of them is not.
//
// There is NO GetSubBlockIdx() gating anywhere on this core. It returns 0 on the AIC, so gating on
// it would be a test that is always true -- and the dual-destination split is the HARDWARE's job,
// not a loop over sub-blocks.

namespace {

// THE AIC -> AIV RELEASE, ONE CHANNEL PER AIV.
//
// Mode 4 is point-to-point: the AIC names sub-block s by adding
// s * kCubeIntraFlagStride to the flagId, and each AIV waits on the base id.
// Mode 2's single flag for all AIVs is a ~50% intermittent hang here -- see
// the measurement in the UB layout header. `pipe` is a template parameter
// because the two release sites drain different pipes: the ring slot is freed
// once MTE1 has read it, the product once the Fixpipe has written it.
template <pipe_t pipe>
__aicore__ inline void CubeReleaseToAllAivs(std::uint16_t base_flag) {
    for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
        CrossCoreSetFlag<kCubeIntraSyncMode, pipe>(
            static_cast<uint16_t>(base_flag + sub * kCubeIntraFlagStride));
    }
}

// L1 -> L0A. The token tile is NZ on both sides, so this is the non-transpose route.
__aicore__ inline void CubeLoadTokensToL0A(
    __ca__ bfloat16_t* l0a_mem, const LocalTensor<bfloat16_t>& l1a_t,
    std::uint32_t m_extent, std::uint32_t k_extent) {
    auto l1A = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1a_t.GetPhyAddr())),
        Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    auto l0A = Te::MakeTensor(
        Te::MakeMemPtr(l0a_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}, Te::CopyL12L0ATraitDefault{}), l0A, l1A);
}

// L1 -> L0B, AND THE OPERAND TRANSPOSE IS THE FIRST LINE OF IT.
//
// The AIVs wrote these bytes as NZ(n_extent, k_extent) -- a [N][K] weight tile, which is how the
// weights are stored and how the MXFP4 groups run -- and NZ(N, K) is bit-identical to ZN(K, N).
// So declaring the SAME MEMORY as ZN(k_extent, n_extent) IS W^T, with no instruction spent, and
// CopyL12L0B then sees ZN -> ZN and takes its noTrans / CopyMode::NORMAL route rather than the
// fractal-transposing one that cannot perform a logical transpose at all.
//
// DERIVED FROM THE TOOLKIT'S LAYOUT ARITHMETIC, NOT MEASURED. The derivation is in the UB layout
// header under "THE WEIGHTS ARE [N][K]"; the STEP 2 history says to settle any doubt with a
// selector A (A[m][k] = (k == m)), which makes C a direct readout of B.
__aicore__ inline void CubeLoadWeightsToL0B(
    __cb__ bfloat16_t* l0b_mem, const LocalTensor<bfloat16_t>& l1w_t,
    std::uint32_t k_extent, std::uint32_t n_extent) {
    auto l1W = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1w_t.GetPhyAddr())),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    auto l0B = Te::MakeTensor(
        Te::MakeMemPtr(l0b_mem),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}, Te::CopyL12L0BTraitDefault{}), l0B, l1W);
}

// THE ACCUMULATION. `init` is true only on the contraction's FIRST step; every later Mmad adds
// into the same L0C.
//
// (init, source) = (false, false) FOR THE LATER STEPS IS THE CANONICAL FORM, demonstrated by
// CANN's own Matmul framework, which computes exactly these two flags for its K loop
// (scheduler/batch/batch_scheduler_base.h, UpdateMmadComputeParams):
//
//     cmatrixSource  = sL0CInit && IsBias();
//     cmatrixInitVal = sL0CInit && !IsBias();
//
// so after the first step it passes (false, false) through the plain three-operand Mmad. The
// four-operand bias overload with the bias in L0C resolves to the same pair via MmadWithBias and
// was measured bit-identical, so it buys nothing.
__aicore__ inline void CubeMmadAccumulate(
    __cc__ float* l0c_mem, __ca__ bfloat16_t* l0a_mem, __cb__ bfloat16_t* l0b_mem,
    std::uint32_t m_extent, std::uint32_t n_extent, std::uint32_t k_extent, bool init) {
    auto l0A = Te::MakeTensor(
        Te::MakeMemPtr(l0a_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    auto l0B = Te::MakeTensor(
        Te::MakeMemPtr(l0b_mem),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    auto l0C = Te::MakeTensor(
        Te::MakeMemPtr(l0c_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(m_extent, n_extent));
    Te::MmadParams mmadParams(static_cast<uint16_t>(m_extent),
                              static_cast<uint16_t>(n_extent),
                              static_cast<uint16_t>(k_extent),
                              0, init);
    Te::Mmad(Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{}).with(mmadParams),
             l0C, l0A, l0B);
}

// THE ACCUMULATION INTERLOCK. IT IS A DOCUMENTED HARDWARE CONTRACT, NOT A PRECAUTION.
//
// Mmad.md, "Synchronization optimization description" (cube_compute_ISASI/mmad_compute): when a
// cube computation accumulates along the K axis into the same L0C, a PipeBarrier<PIPE_M> must sit
// between consecutive Mmad calls so the previous write completes before the next read. Above a
// size threshold the hardware scheduler handles the dependency itself and the barrier may be
// omitted; below it, the barrier is MANDATORY. The threshold is
//
//     (m / 16) * (n / 16) < 10   ->   PipeBarrier<PIPE_M>() required
//
// and the doc gives that `if` verbatim as the recommended guard. BOTH PHASES OF THIS OPERATOR ARE
// UNDER IT, so the barrier is unconditional here rather than guarded:
//
//     phase 1   m = 16, n = R_SUB = 128   ->  1 * 8 = 8   < 10
//     phase 2   m = 16, n = H_SUB =  64   ->  1 * 4 = 4   < 10
//
// It cost a real wrong answer to find. MEASURED on the dav-3510 CAModel, 2026-10-18, by the STEP 2
// probe with this barrier removed: a FOUR-tile contraction came back as exactly ONE tile's
// product, retiring cleanly with correct liveness on all three cores and the AIC's own status word
// reporting all four tiles processed -- a silent wrong answer, not a fault.
//
// The barrier drains the Cube pipeline per step, which is a real cost. The doc's own escape is to
// get (m/16)*(n/16) over 10, i.e. a wider n, which is a tiling decision and not a fence decision.
__aicore__ inline void CubeCloseAccumulationStep() {
    PipeBarrier<PIPE_M>();
    // WAR on L0A/L0B: the next step's MTE1 writes the buffers this step's Mmads read. IMMEDIATE,
    // never deferred -- see the note in RunCubeCore on why a deferred intra-core fence double-sets.
    // The barrier above has already retired the Mmads, so this pair only holds MTE1 back.
    SetFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
    WaitFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
}

// L0C -> both AIVs' UBs, once, on the completed contraction, then hand it over and wait for the
// acknowledgement. The destination is UB offset 0 in EACH AIV's own address space and the layout
// carries the FULL M -- see CubeL0c2UbTraitDualDstM.
__aicore__ inline void CubeMoveOutAndHandOver(
    __cc__ float* l0c_mem, std::uint32_t m_extent, std::uint32_t n_extent) {
    auto l0C = Te::MakeTensor(
        Te::MakeMemPtr(l0c_mem),
        Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(m_extent, n_extent));
    auto ub = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kCubeLandingUbOffset)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn>(m_extent, n_extent));
    Te::FixpipeParams fixParams;
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(fixParams),
             ub, l0C);
    // Drain the Fixpipe before the ready flag releases the vector cores onto the rows it is still
    // writing.
    PipeBarrier<PIPE_FIX>();
    CubeReleaseToAllAivs<PIPE_FIX>(kCubeFlagProduct);
    // Scoped to PIPE_FIX: what this gates is the NEXT Fixpipe write into the landing zone. It also
    // drains the flag, so none outlives the block. The pipe is named inline rather than through a
    // namespace-scope constant -- see the note on kCubeWaitAicStep.
    CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
}

}  // namespace

// WHY THE AIC's FENCES ARE IMMEDIATE AND THE AIV's ARE DEFERRED.
//
// A deferred intra-core fence raises a token on one lap and consumes it on the next, so the
// producing pipe can run ahead of the consuming one. Program order does NOT order a SetFlag on
// pipe X against a WaitFlag on pipe Y, and the scalar unit issues the whole loop body without
// blocking, so the producer can raise lap k+1's token before the consumer has reached lap k's
// wait -- which lands a SetFlag on an already-set binary flag. MEASURED: the STEP 2 probe did
// exactly this and the simulator reported
// `pem_ccu.cc execute_set_flag already has same set_flag!` four times per run, with the
// SetFlag/WaitFlag counts textually balanced 1:1 and the post-loop drain consuming exactly what
// the loop raised. Balance is necessary and NOT sufficient.
//
// A deferred fence is only safe when a RAW chain already orders the two pipes. On the AIV side
// each one has that: the burst for slot s cannot be issued until the Vector unit has released the
// slot's staging, and the Vector unit cannot dequantize slot s until that burst has landed, so
// MTE2 and V cannot slip a whole lap past each other. On the AIC side nothing plays that role for
// the L0 buffers, so every fence here is an immediate pair.
__aicore__ inline void RunCubeCore(
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

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            continue;  // the same predicate the AIVs evaluate, so the triple stays in step
        }

        // --- PHASE 1 ----------------------------------------------------------------------------
        // The Cube's work is IDENTICAL for every inter block: the AIVs pick the weight window and
        // the Cube only ever sees L1 slots, so `inter_base` appears here only as a trip count.
        const std::uint32_t fwd_steps = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            for (std::uint32_t k = 0; k < fwd_steps; ++k) {
                const std::uint32_t slot = k % kCubeRingSlots;

                // ONE wait, not one per AIV. Mode 2's AIV -> AIC direction increments the AIC's
                // counter once BOTH AIVs have set -- MEASURED in STEP 2: a second wait HANGS.
                // Scoped to PIPE_MTE1, the pipe that reads the slot.
                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));

                CubeLoadTokensToL0A(l0a, l1a_buf[slot].Get<bfloat16_t>(), kCubeMTile, kCubeKSub);
                CubeLoadWeightsToL0B(l0b_gate, l1w_buf[slot][0].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);
                CubeLoadWeightsToL0B(l0b_up, l1w_buf[slot][1].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);

                // RELEASE THE SLOT HERE, NOT AFTER THE MMADS. The AIVs may refill it as soon as
                // MTE1 has drained it, which this set waits for by construction; holding it until
                // after the Mmad made them wait a full Cube step for no dependency.
                if (k + kCubeRingSlots < fwd_steps) {
                    CubeReleaseToAllAivs<PIPE_MTE1>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                // ONE TOKEN TILE, TWO WEIGHT TILES, TWO ACCUMULATORS. That reuse is the reason
                // gate and up share this loop.
                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate,
                                   kCubeMTile, kCubeRSub, kCubeKSub, k == 0u);
                CubeMmadAccumulate(l0c_up, l0a, l0b_up,
                                   kCubeMTile, kCubeRSub, kCubeKSub, k == 0u);

                // ONE barrier for BOTH Mmads. Mmad.md ties the rule to "intermediate results ...
                // written to the same L0C Buffer", and gate and up target DIFFERENT L0C buffers,
                // so they do not need separating from each other -- only l0c_gate(k) from
                // l0c_gate(k+1), which this placement does. That is a reading of the doc rather
                // than a measurement, and a barrier between the two Mmads as well is the cheap
                // conservative variant if this schedule ever comes back wrong.
                CubeCloseAccumulationStep();
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);

            // TWO MOVE-OUTS THROUGH ONE LANDING ZONE. There is exactly one agreed UB offset (see
            // kCubeLandingUbOffset), so the gate product is handed over, consumed and acknowledged
            // before the up product is written. The cost is two cross-core round trips per inter
            // block, against a whole contraction's worth of Cube work.
            CubeMoveOutAndHandOver(l0c_gate, kCubeMTile, kCubeRSub);
            CubeMoveOutAndHandOver(l0c_up, kCubeMTile, kCubeRSub);
        }

        // --- PHASE 2 ----------------------------------------------------------------------------
        const std::uint32_t down_steps = geom.inter / kCubeRSub2;

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            for (std::uint32_t r = 0; r < down_steps; ++r) {
                const std::uint32_t slot = r % kCubeRingSlots;

                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));

                CubeLoadTokensToL0A(l0a, l1a2_buf[slot].Get<bfloat16_t>(),
                                    kCubeMTile, kCubeRSub2);
                // Same reinterpretation as phase 1: the AIVs wrote NZ(H_SUB, R_SUB2), which is W2
                // as stored ([hidden][inter]), so reading it as ZN(R_SUB2, H_SUB) is W2^T.
                CubeLoadWeightsToL0B(l0b_gate, l1w2_buf[slot].Get<bfloat16_t>(),
                                     kCubeRSub2, kCubeHSub);

                if (r + kCubeRingSlots < down_steps) {
                    CubeReleaseToAllAivs<PIPE_MTE1>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate,
                                   kCubeMTile, kCubeHSub, kCubeRSub2, r == 0u);

                CubeCloseAccumulationStep();
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);

            CubeMoveOutAndHandOver(l0c_gate, kCubeMTile, kCubeHSub);
        }
    }
}

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube(
    GM_ADDR tokens_in,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut, GM_ADDR tokens_out,
    GM_ADDR shared_workspace,    GM_ADDR tiling_config) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // `shared_workspace` is no longer read: it held the KfcServer's message queue, and
    // REGIST_MATMUL_OBJ is gone. The null check is kept so the operator's precondition is
    // unchanged for callers, and the parameter stays in the signature because the tiling ABI and
    // the host launch wrapper are not part of this change.
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

    // --- L1, ALLOCATED IDENTICALLY ON BOTH CORES, in the same order, OUTSIDE the ASCEND_IS_*
    //     branches. That is not tidiness: the AIVs write these slots and the AIC reads them, so
    //     the three cores have to land on the same L1 offsets, and nothing in the toolchain checks
    //     that they do. Allocating identically on all of them is the only thing that makes the
    //     addresses agree.
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

    if ASCEND_IS_AIC {
        RunCubeCore(geom, token_to_expert_lut, l1a_buf, l1w_buf, l1a2_buf, l1w2_buf);
    }

    if ASCEND_IS_AIV {
        GlobalTensor<float> tokens_in_gm;
        GlobalTensor<float> tokens_out_gm;
        const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
        tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

        // q_landing IS FIRST, AND THAT IS THE CONTRACT WITH THE AIC. The Fixpipe names UB offset 0
        // in each AIV's own address space, and the TPipe dynamic pool starts at 0 on this
        // toolchain -- MEASURED by the GetPhyAddr census in
        // tests/sim/kernel/test_subgemm_stage_kernel.cpp and re-checked by the STEP 1 probe.
        // Everything else follows.
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
                // scale_plane and scale_temporary stay SHARED on purpose: they are scratch
                // consumed inside DequantSubTileVector, and there is one Vector unit, so two
                // dequants never overlap each other.
                buf.w[s][w].scale_plane     = scale_plane;
                buf.w[s][w].scale_temporary = scale_temp;
            }
        }

        // ZERO THE TOKEN STAGING ONCE, AND THE TAIL NEVER NEEDS ANOTHER THOUGHT.
        //
        // The Cube always contracts the full M = 16 -- DUAL_DST_SPLIT_M bisects a fixed M, and
        // making it ragged would put the split point in doubt. So the rows past `valid_rows` have
        // to be ZERO rather than absent: a zero A row gives a zero C row, which costs the Cube
        // nothing it was not going to spend and is simply not stored. Every later token burst
        // writes only `valid_rows` rows and leaves these zeros in place.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            Duplicate(buf.x_f32[s], 0.0F, kCubeMTile * kCubeKHalf);
        }
        // V -> MTE2: the first token burst into each slot must not race the zeroing. The ring's
        // own WAR token covers every LATER burst; this covers the first, and there is exactly one
        // of these per slot so nothing is left raised.
        for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
            SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
            WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[s]);
        }

        // `% GetSubBlockNum()` IS LOAD-BEARING, not defensive. AscendC::GetSubBlockIdx() is the
        // AIV's logical id across ALL AI Core groups -- with 10 groups at a 1:2 ratio it returns
        // 0..19 -- while the id WITHIN the group is what "am I this AIC's first vector core"
        // needs. GetSubBlockNum() is the AIVs per logical AI Core, so the modulo recovers it at
        // any grid size.
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

// =================================================================================================
// QUARANTINED: the framework path, unreachable from the entry point above.
//
// Retained only so the five sim bisection probes that textually include this file keep compiling
// and their registered gates keep running. See the header of
// dsv4_moe_prefill_cube_legacy_matmul.h for the full list and for what deleting it would retire.
// Nothing above this line calls anything below it.
// =================================================================================================
#include "dsv4_moe_prefill_cube_legacy_matmul.h"

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
