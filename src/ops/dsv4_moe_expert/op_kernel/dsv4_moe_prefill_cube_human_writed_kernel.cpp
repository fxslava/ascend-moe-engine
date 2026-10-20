/*
 * Educational 950PR MXFP4 MoE prefill kernel for MIX_AIC_1_2.
 * One AIC and two AIVs process each token_row. AIV block index / 2 selects the token_row,
 * while GetSubBlockIdx selects its private 1024-column band. Projection math,
 * weight_scale_codes_ub, accumulators, and materialized activations are BF16. FP32 UB storage
 * is limited to exact SwiGLU Exp/Div because CANN 9.2 does not accept BF16 there.
 * The first UB allocation is inter_activations_bf16 at offset zero. Phase three
 * reuses the earlier weight and scratch allocations; LocalMemAllocator has no
 * reset API. Mode 2 ready flags from both AIVs rendezvous at the AIC before
 * paired Mode 4 release flags allow down projection to begin.
 */
#include "kernel_operator.h"
#include "basic_api/kernel_operator_block_sync_intf.h"
#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
#include "utils/debug/asc_printf.h"
#define DSV4_PIPELINE_TRACE(...) __asc_aicore::printf(__VA_ARGS__)
#else
#define DSV4_PIPELINE_TRACE(...)
#endif

namespace AscendC {
namespace {
constexpr uint32_t kTokenCount = 16;
constexpr uint32_t kHiddenDimension = 4096;
constexpr uint32_t kIntermediateDimension = 2048;
constexpr uint32_t kScaleGroupSize = 32;
constexpr uint32_t kReductionTile = 32;
constexpr uint32_t kProjectionTileWidth = 1024;
constexpr uint32_t kSubBlockCount = 2;
constexpr uint32_t kIntermediateBandWidth = kIntermediateDimension / kSubBlockCount;
constexpr uint32_t kHiddenBandWidth = kHiddenDimension / kSubBlockCount;
constexpr uint32_t kWeightTileElementCount = kReductionTile * kProjectionTileWidth;
constexpr uint32_t kWeightCastHalfElementCount = kWeightTileElementCount / 2;
constexpr uint32_t kWeightCastRepeatCount = kWeightCastHalfElementCount / 128;
constexpr uint32_t kScaleGroupsPerProjectionTile = kProjectionTileWidth / kScaleGroupSize;
constexpr uint32_t kScaleTileElementCount = kReductionTile * kScaleGroupsPerProjectionTile;
constexpr uint16_t kAivReadyFlag = 0;
constexpr uint16_t kAicReleaseFlag = 2;

__aicore__ inline void TraceBf16(const __gm__ char* label, const LocalTensor<bfloat16_t>& values, const LocalTensor<float>& trace_values_fp32) {
    Cast<float, bfloat16_t>(trace_values_fp32, values, RoundMode::CAST_NONE, 16);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    DSV4_PIPELINE_TRACE("%s: %f %f %f %f %f %f %f %f\n", label, trace_values_fp32.GetValue(0), trace_values_fp32.GetValue(1), trace_values_fp32.GetValue(2), trace_values_fp32.GetValue(3), trace_values_fp32.GetValue(4), trace_values_fp32.GetValue(5), trace_values_fp32.GetValue(6), trace_values_fp32.GetValue(7));
}

__aicore__ inline void AccumulateProjectionTile(
    const LocalTensor<bfloat16_t>& projection_accumulator_bf16,
    const LocalTensor<bfloat16_t>& activation_tokens_bf16,
    const LocalTensor<uint8_t>& packed_weights_bytes_ub,
    const LocalTensor<bfloat16_t>& dequant_weights_bf16,
    const LocalTensor<bfloat16_t>& scaled_weight_product_bf16,
    const LocalTensor<uint8_t>& weight_scale_codes_ub,
    const LocalTensor<uint16_t>& scale_codes_bf16_bits,
    const GlobalTensor<uint8_t>& packed_weights_gm,
    const GlobalTensor<uint8_t>& weight_scales_gm,
    uint32_t projection_output_dimension,
    uint32_t scale_groups_per_output_row,
    uint32_t reduction_tile_start,
    uint32_t projection_output_start,
    const LocalTensor<float>& trace_values_fp32,
    bool trace_weight_strip) {
    DataCopyParams weight_params;
    weight_params.blockCount = kReductionTile;
    weight_params.blockLen   = (kProjectionTileWidth / 2) / 32;
    weight_params.srcStride  = ((projection_output_dimension - kProjectionTileWidth) / 2) / 32;
    weight_params.dstStride  = 0;
    DataCopyParams scale_params;
    scale_params.blockCount = kReductionTile;
    scale_params.blockLen   = 1;
    scale_params.srcStride  = (scale_groups_per_output_row - kScaleGroupsPerProjectionTile) / 32;
    scale_params.dstStride  = 0;
    const uint32_t weight_offset = (reduction_tile_start * projection_output_dimension + projection_output_start) / 2;
    const uint32_t scale_offset = reduction_tile_start * scale_groups_per_output_row + projection_output_start / kScaleGroupSize;
    DataCopy(packed_weights_bytes_ub, packed_weights_gm[weight_offset], weight_params);
    DataCopy(weight_scale_codes_ub, weight_scales_gm[scale_offset], scale_params);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<fp4x2_e2m1_t> packed_weights_fp4 = packed_weights_bytes_ub.ReinterpretCast<fp4x2_e2m1_t>();
    UnaryRepeatParams fp4_cast_repeat_params(1, 1, 8, 2);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16, packed_weights_fp4, RoundMode::CAST_NONE, 128ull, static_cast<uint8_t>(kWeightCastRepeatCount), fp4_cast_repeat_params);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16[kWeightCastHalfElementCount], packed_weights_fp4[kWeightCastHalfElementCount / 2], RoundMode::CAST_NONE, 128ull, static_cast<uint8_t>(kWeightCastRepeatCount), fp4_cast_repeat_params);
    Cast<uint16_t, uint8_t>(scale_codes_bf16_bits, weight_scale_codes_ub, RoundMode::CAST_NONE, kScaleTileElementCount);
    ShiftLeft(scale_codes_bf16_bits, scale_codes_bf16_bits, static_cast<uint16_t>(7), kScaleTileElementCount);
    LocalTensor<bfloat16_t> scale_bf16 = scale_codes_bf16_bits.ReinterpretCast<bfloat16_t>();
    PipeBarrier<PIPE_V>();
    for (uint32_t reduction = 0; reduction < kReductionTile; ++reduction) {
        const bfloat16_t activation_scalar = activation_tokens_bf16.GetValue(reduction_tile_start + reduction);
        for (uint32_t group = 0; group < kScaleGroupsPerProjectionTile; ++group) {
            const uint32_t weight_index = reduction * kProjectionTileWidth + group * kScaleGroupSize;
            const uint32_t scale_index = reduction * kScaleGroupsPerProjectionTile + group;
            const bfloat16_t scale_scalar = scale_bf16.GetValue(scale_index);
            Muls(scaled_weight_product_bf16[group * kScaleGroupSize], dequant_weights_bf16[weight_index], scale_scalar, kScaleGroupSize);
            if (trace_weight_strip && reduction == 0 && group == 0) {
                TraceBf16("[dsv4] dequantized weights", scaled_weight_product_bf16, trace_values_fp32);
            }
            Muls(scaled_weight_product_bf16[group * kScaleGroupSize], scaled_weight_product_bf16[group * kScaleGroupSize], activation_scalar, kScaleGroupSize);
            Add(projection_accumulator_bf16[group * kScaleGroupSize], projection_accumulator_bf16[group * kScaleGroupSize], scaled_weight_product_bf16[group * kScaleGroupSize], kScaleGroupSize);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunVectorCore(
    uint32_t token_row,
    uint32_t sub_block_idx,
    const GlobalTensor<bfloat16_t>& input_tokens_gm,
    const GlobalTensor<uint8_t>& gate_weights_gm,
    const GlobalTensor<uint8_t>& gate_scales_gm,
    const GlobalTensor<uint8_t>& up_weights_gm,
    const GlobalTensor<uint8_t>& up_scales_gm,
    const GlobalTensor<uint8_t>& down_weights_gm,
    const GlobalTensor<uint8_t>& down_scales_gm,
    const GlobalTensor<bfloat16_t>& gate_gm,
    const GlobalTensor<bfloat16_t>& up_gm,
    const GlobalTensor<bfloat16_t>& inter_gm,
    const GlobalTensor<bfloat16_t>& final_output_gm) {
    LocalMemAllocator<Hardware::UB> allocator;
    LocalTensor<bfloat16_t> inter_activations_bf16 = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<bfloat16_t> tokens_tile_ub = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kHiddenDimension);
    LocalTensor<uint8_t> packed_weights_ub = allocator.Alloc<TPosition::VECIN, uint8_t>(kWeightTileElementCount / 2);
    LocalTensor<bfloat16_t> dequant_weights_bf16 = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kWeightTileElementCount);
    LocalTensor<bfloat16_t> scaled_weight_product_bf16 = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<uint8_t> scale_codes_ub = allocator.Alloc<TPosition::VECIN, uint8_t>(kScaleTileElementCount);
    LocalTensor<uint16_t> scale_bits_ub = allocator.Alloc<TPosition::VECCALC, uint16_t>(kScaleTileElementCount);
    LocalTensor<bfloat16_t> gate_acc_bf16 = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t> up_acc_bf16 = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t> down_acc_bf16 = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t> gate_bf16 = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<bfloat16_t> up_bf16 = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<float> gate_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> up_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> negative_gate_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> exp_negative_gate_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> denominator_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> sigmoid_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> silu_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float> inter_fp32 = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);

    DataCopy(tokens_tile_ub, input_tokens_gm[token_row * kHiddenDimension], kHiddenDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] input tokens", tokens_tile_ub, gate_fp32);
    }

    const uint32_t inter_start = sub_block_idx * kIntermediateBandWidth;
    Duplicate(gate_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
    Duplicate(up_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
    for (uint32_t reduction_tile_start = 0; reduction_tile_start < kHiddenDimension; reduction_tile_start += kReductionTile) {
        AccumulateProjectionTile(gate_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, gate_weights_gm, gate_scales_gm, kIntermediateDimension, kIntermediateDimension / kScaleGroupSize, reduction_tile_start, inter_start, gate_fp32, token_row == 0 && sub_block_idx == 0);
        AccumulateProjectionTile(up_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, up_weights_gm, up_scales_gm, kIntermediateDimension, kIntermediateDimension / kScaleGroupSize, reduction_tile_start, inter_start, gate_fp32, false);
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(gate_gm[token_row * kIntermediateDimension + inter_start], gate_acc_bf16, kProjectionTileWidth);
    DataCopy(up_gm[token_row * kIntermediateDimension + inter_start], up_acc_bf16, kProjectionTileWidth);
    PipeBarrier<PIPE_ALL>();

    DataCopy(gate_bf16, gate_gm[token_row * kIntermediateDimension + inter_start], kIntermediateBandWidth);
    DataCopy(up_bf16, up_gm[token_row * kIntermediateDimension + inter_start], kIntermediateBandWidth);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast<float, bfloat16_t>(gate_fp32, gate_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    Cast<float, bfloat16_t>(up_fp32, up_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] gate GEMM", gate_bf16, gate_fp32);
    }
    Muls(negative_gate_fp32, gate_fp32, -1.0f, kIntermediateBandWidth);
    Exp(exp_negative_gate_fp32, negative_gate_fp32, kIntermediateBandWidth);
    Adds(denominator_fp32, exp_negative_gate_fp32, 1.0f, kIntermediateBandWidth);
    Duplicate(sigmoid_fp32, 1.0f, kIntermediateBandWidth);
    Div(sigmoid_fp32, sigmoid_fp32, denominator_fp32, kIntermediateBandWidth);
    Mul(silu_fp32, gate_fp32, sigmoid_fp32, kIntermediateBandWidth);
    Mul(inter_fp32, silu_fp32, up_fp32, kIntermediateBandWidth);
    Cast<bfloat16_t, float>(inter_activations_bf16, inter_fp32, RoundMode::CAST_RINT, kIntermediateBandWidth);
    PipeBarrier<PIPE_V>();
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] SwiGLU", inter_activations_bf16, gate_fp32);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(inter_gm[token_row * kIntermediateDimension + inter_start], inter_activations_bf16, kIntermediateBandWidth);
    PipeBarrier<PIPE_ALL>();

    CrossCoreSetFlag<2, PIPE_V>(static_cast<uint16_t>(kAivReadyFlag + sub_block_idx));
    CrossCoreWaitFlag<4, PIPE_FIX>(static_cast<uint16_t>(kAicReleaseFlag + sub_block_idx));
    DataCopy(tokens_tile_ub, inter_gm[token_row * kIntermediateDimension], kIntermediateDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t hidden_start = sub_block_idx * kHiddenBandWidth;
    for (uint32_t projection_output_start = hidden_start; projection_output_start < hidden_start + kHiddenBandWidth; projection_output_start += kProjectionTileWidth) {
        Duplicate(down_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
        for (uint32_t reduction_tile_start = 0; reduction_tile_start < kIntermediateDimension; reduction_tile_start += kReductionTile) {
            AccumulateProjectionTile(down_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, down_weights_gm, down_scales_gm, kHiddenDimension, kHiddenDimension / kScaleGroupSize, reduction_tile_start, projection_output_start, gate_fp32, false);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(final_output_gm[token_row * kHiddenDimension + projection_output_start], down_acc_bf16, kProjectionTileWidth);
        if (token_row == 0 && sub_block_idx == 0 && projection_output_start == hidden_start) {
            TraceBf16("[dsv4] down projection", down_acc_bf16, gate_fp32);
        }
    }
}
}

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_human_writed(
    GM_ADDR input_tokens_gm, GM_ADDR gate_packed_weights_gm, GM_ADDR gate_weight_scales_gm,
    GM_ADDR up_packed_weights_gm, GM_ADDR up_weight_scales_gm, GM_ADDR down_packed_weights_gm,
    GM_ADDR down_weight_scales_gm, GM_ADDR gate_gm, GM_ADDR up_gm, GM_ADDR inter_gm,
    GM_ADDR final_output_gm) {
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if ASCEND_IS_AIC {
        const uint32_t token_row = GetBlockIdx();
        if (token_row < kTokenCount) {
            CrossCoreWaitFlag<2, PIPE_V>(kAivReadyFlag);
            CrossCoreWaitFlag<2, PIPE_V>(kAivReadyFlag + kSubBlockCount - 1);
            CrossCoreSetFlag<4, PIPE_FIX>(kAicReleaseFlag);
            CrossCoreSetFlag<4, PIPE_FIX>(kAicReleaseFlag + kSubBlockCount - 1);
        }
    }
    if ASCEND_IS_AIV {
        const uint32_t token_row = GetBlockIdx() / kSubBlockCount;
        const uint32_t sub_block_idx = GetSubBlockIdx();
        if (token_row >= kTokenCount) {
            return;
        }
        GlobalTensor<bfloat16_t> input_tokens;
        GlobalTensor<uint8_t>    gate_weights;
        GlobalTensor<uint8_t>    gate_scales;
        GlobalTensor<uint8_t>    up_weights;
        GlobalTensor<uint8_t>    up_scales;
        GlobalTensor<uint8_t>    down_weights;
        GlobalTensor<uint8_t>    down_scales;
        GlobalTensor<bfloat16_t> gate_projection;
        GlobalTensor<bfloat16_t> up_projection;
        GlobalTensor<bfloat16_t> inter_activations;
        GlobalTensor<bfloat16_t> final_output;
        input_tokens.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input_tokens_gm));
        gate_weights.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gate_packed_weights_gm));
        gate_scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gate_weight_scales_gm));
        up_weights.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(up_packed_weights_gm));
        up_scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(up_weight_scales_gm));
        down_weights.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(down_packed_weights_gm));
        down_scales.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(down_weight_scales_gm));
        gate_projection.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gate_gm));
        up_projection.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(up_gm));
        inter_activations.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(inter_gm));
        final_output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(final_output_gm));
        RunVectorCore(token_row, sub_block_idx, input_tokens, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales, gate_projection, up_projection, inter_activations, final_output);
    }
}

}
