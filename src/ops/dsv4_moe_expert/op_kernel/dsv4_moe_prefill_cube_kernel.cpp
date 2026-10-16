// dsv4_moe_prefill_cube_kernel.cpp -- batched MoE prefill with W1/W3/W2 on
// the Cube in BF16 (Ascend 950PR, arch35 / dav-3510, MIX AIC+AIV).
//
// A TWO-PHASE ON-CHIP FUSED PIPELINE with no GM intermediate workspace.
//
// Oracle: src/emu/moe_prefill_cube_2d.cpp ::
// moe_prefill_cube_flash_2d_w4a16_cpu, which models THIS schedule -- the
// 1-D token-tile grid, the resident act plane, the three-level tile
// decomposition and the FP32 accumulation at every level. See the numerics
// note on op_host/dsv4_moe_prefill_cube_tiling.h for why BF16 and not native
// Cube MXFP4, and for the proof sketch that the weight dequantization is exact.
//
//   gate[m, r] = sum_c  X_bf16[m, c] * W1_bf16[r, c]        Cube
//   up  [m, r] = sum_c  X_bf16[m, c] * W3_bf16[r, c]        Cube
//   act [m, r] = silu(min(gate, L)) * clamp(up, -L, L)      Vector
//   y   [m, h] = sum_r  act_bf16[m, r] * W2_bf16[h, r]      Cube
//
// THE SCHEDULE, which is the whole content of this file:
//
//   for each token tile (16 rows), grid-striding over cores
//     PHASE 1
//     for r0 in 0, R_SUB, ... < inter
//       gate_acc = up_acc = 0                      FP32 in UB
//       for k0 in 0, K_SUB, ... < hidden
//         X[16, K_SUB]      -> UB, cast to BF16    read ONCE, used TWICE
//         W1[R_SUB, K_SUB]  -> UB -> L1            dequantized per sub-tile
//         W3[R_SUB, K_SUB]  -> UB -> L1
//         gate_acc += X @ W1^T ; up_acc += X @ W3^T   two sub-GEMMs
//       act[:, r0 : r0+R_SUB] = clamped SwiGLU     BF16, stays resident
//     PHASE 2
//     for h0 in 0, H_SUB, ... < hidden
//       y_acc = 0                                  FP32 in UB
//       for r0 in 0, R_SUB2, ... < inter
//         act[:, r0 : r0+R_SUB2] -> a_sub          contiguous A, at VECOUT
//         W2[H_SUB, R_SUB2] -> UB -> L1
//         y_acc += a_sub @ W2^T
//       y_acc -> GM                                one strided burst
//
// WHERE THE LEVEL-2 SUBTOTAL LIVES, because it is a deviation from the
// idealized schedule and reading this file without the note would mislead: a
// `TPosition::TSCM` B operand must be the WHOLE single-core B, so splitting K
// makes each (r0, k0) step its own single-core GEMM -- and L0C does not
// survive between separately driven GEMMs, while `IterateAll`'s accumulating
// form is GM-atomic, which the zero-GM-workspace rule forbids. So the
// subtotals are summed by the Vector into an FP32 UB accumulator. The
// ARITHMETIC is identical to the L0C-resident form -- fractal, then sub-tile
// block, then axis, FP32 throughout -- which is what the emu golden models, so
// the oracle comparison is unaffected. Only the buffer differs.
//
// THE MIX EXECUTION MODEL, which dictates the shape of this file.
// REGIST_MATMUL_OBJ stands up an AscendC::KfcServer. On the AIC side the macro
// spins in server.Run() and RETURNS FROM INSIDE THE MACRO; only the AIV side
// falls through. So everything written after the macro is AIV code that drives
// the Cube through the server's message queue, and `workspace` is that queue's
// GM backing -- which is now the ONLY thing the workspace holds.
//
// THE CLAMP IS ASYMMETRIC AND THE TWO HALVES ARE NOT INTERCHANGEABLE.
// AscendSiluAndMulWithClamp bounds the GATE ABOVE ONLY and bounds UP ON BOTH
// SIDES, and the AIV prefill kernel does the same. Dropping the `up` clamp is
// not harmless: `up` is a raw hidden-wide GEMM output, so it exceeds L readily
// and the error is unbounded.
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under the dav_3510
// CAModel only (CLAUDE.md §5.7). No claim is made here about observed
// throughput, bandwidth or latency.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

static_assert(kCubeMTile == kPrefillCubeMTile,     "token tile disagrees between the UB layout and the tiling header the host validates from");
static_assert(kCubeRSub  == kPrefillCubeRSub,      "phase 1 inter output sub-tile disagrees");
static_assert(kCubeKSub  == kPrefillCubeKSub,      "phase 1 hidden contraction sub-tile disagrees");
static_assert(kCubeHSub  == kPrefillCubeHSub,      "phase 2 hidden output sub-tile disagrees");
static_assert(kCubeRSub2 == kPrefillCubeRSub2,     "phase 2 inter contraction sub-tile disagrees");
static_assert(kCubeGroup == kPrefillCubeGroupSize, "MXFP4 group size disagrees");

constexpr std::uint32_t kKfcWorkspaceSlotBytes = 64u * 1024u;
constexpr int           kEvTokensPing          = 2;
constexpr int           kEvTokensPong          = 3;

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

struct Buffers {
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
    LocalTensor<fp4x2_e2m1_t> weight_nibbles;
    LocalTensor<bfloat16_t>  weight_expanded;
    LocalTensor<uint8_t>      weight_scales;
    LocalTensor<bfloat16_t>  scale_plane;
    LocalTensor<uint16_t>     scale_temporary;
};

}  // namespace

__aicore__ inline void AsyncLoadWeightsSubTile(
    __gm__ std::uint8_t* packed_weights, __gm__ std::uint8_t* scale_factors,
    std::uint32_t row_start, std::uint32_t col_start, std::uint32_t row_count, std::uint32_t col_count, std::uint32_t matrix_cols,
    const Buffers& buf) {
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

    SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
}

__aicore__ inline void DequantSubTileToL1(
    std::uint32_t row_count, std::uint32_t col_count,
    const Buffers& buf, const LocalTensor<bfloat16_t>& l1_weight_slot) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t total_elements = row_count * col_count;

    WaitFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);

    Cast(buf.weight_expanded, buf.weight_nibbles, RoundMode::CAST_NONE, total_elements);

    LocalTensor<uint16_t>   row_scales_u16  = buf.scale_temporary;
    LocalTensor<bfloat16_t> row_scales_bf16 = buf.scale_temporary.ReinterpretCast<bfloat16_t>();

    const uint32_t dst_shape[2] = {groups_per_row, kCubeGroup};
    const uint32_t src_shape[2] = {groups_per_row, 1};

    for (std::uint32_t r_idx = 0; r_idx < row_count; ++r_idx) {
        const std::uint32_t scale_base = r_idx * kCubeScaleRowStride;
        const std::uint32_t out_base   = r_idx * col_count;

        Cast(row_scales_u16, buf.weight_scales[scale_base], RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        ShiftLeft(row_scales_u16, row_scales_u16, static_cast<uint16_t>(7), 16);
        PipeBarrier<PIPE_V>();

        Broadcast<bfloat16_t, 2>(buf.scale_plane[out_base], row_scales_bf16, dst_shape, src_shape, nullptr);
    }
    PipeBarrier<PIPE_V>();

    Mul(buf.weight_expanded, buf.weight_expanded, buf.scale_plane, total_elements);
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    DataCopy(l1_weight_slot, buf.weight_expanded, total_elements);
    SetFlag<HardEvent::MTE3_MTE1>(kEvCubeMte3ToMte1);
    WaitFlag<HardEvent::MTE3_MTE1>(kEvCubeMte3ToMte1);
    SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
}

__aicore__ inline void AsyncLoadTokensChunk(
    const GlobalTensor<float>& tokens_in_gm, std::uint32_t token_row_start, std::uint32_t hidden_base,
    std::uint32_t valid_rows, const Geometry& geom,
    const LocalTensor<float>& dst_tensor, int event_id) {
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
    matmul_gate_up.IterateAll(buf.subgemm_fwd);
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
    matmul_down.IterateAll(buf.subgemm_down);
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

        std::uint32_t hidden_base = 0;
        bool is_ping              = true;

        AsyncLoadTokensChunk(tokens_in_gm, token_row_start, hidden_base, valid_rows, geom, buf.tokens_f32_ping, kEvTokensPing);

        for (; hidden_base + geom.hidden_sub_fwd < geom.hidden; hidden_base += geom.hidden_sub_fwd) {
            const std::uint32_t next_hidden_base = hidden_base + geom.hidden_sub_fwd;
            auto& curr_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
            auto& next_tokens_f32  = is_ping ? buf.tokens_f32_pong      : buf.tokens_f32_ping;
            auto& curr_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
            const int curr_event   = is_ping ? kEvTokensPing            : kEvTokensPong;
            const int next_event   = is_ping ? kEvTokensPong            : kEvTokensPing;

            AsyncLoadTokensChunk(tokens_in_gm, token_row_start, next_hidden_base, valid_rows, geom, next_tokens_f32, next_event);

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

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube(
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
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data = reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
    if (unlikely(TilingRejected(tiling_data))) {
        return;
    }
    const Geometry geom = LoadGeometry(tiling_data);

    const std::uint32_t sub_core_idx  = static_cast<std::uint32_t>(GetSubBlockIdx());
    GM_ADDR             aiv_workspace = Bytes(shared_workspace) + sub_core_idx * kKfcWorkspaceSlotBytes;

    TPipe pipe;
    GateUpMatmul matmul_gate_up;
    DownMatmul   matmul_down;
    REGIST_MATMUL_OBJ(&pipe, aiv_workspace, matmul_gate_up, matmul_down);

    GlobalTensor<float> tokens_in_gm;
    GlobalTensor<float> tokens_out_gm;
    const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
    tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
    tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

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
    pipe.InitBuffer(q_scale_plane,           1, kCubeRSub * kCubeKSub * sizeof(bfloat16_t));
    pipe.InitBuffer(q_scale_temporary,       1, 16 * sizeof(uint16_t));

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

    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx());
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        ProcessTokenTile(
            geom, tile_idx,
            gate_weights_packed, gate_weight_scales,
            up_weights_packed,   up_weight_scales,
            down_weights_packed, down_weight_scales,
            token_to_expert_lut,
            tokens_in_gm, tokens_out_gm,
            buf, q_weight_fwd, q_weight_down,
            matmul_gate_up, matmul_down);
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