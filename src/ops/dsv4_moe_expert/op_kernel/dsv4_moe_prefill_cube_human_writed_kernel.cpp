/*
 * Educational 950PR MXFP4 MoE prefill kernel for MIX_AIC_1_2.
 * One AIC and two AIVs process each token_row. AIV block index / 2 selects the token_row,
 * while GetSubBlockIdx selects its private 1024-column band. Projection math,
 * weight_scale_codes_ub, accumulators, and materialized activations are BF16. FP32 UB storage
 * is limited to exact SwiGLU Exp/Div because CANN 9.2 does not accept BF16 there.
 * The first UB allocation is inter_activations_bf16 at offset zero. Phase three
 * reuses the earlier weight and scratch allocations; LocalMemAllocator has no
 * reset API. Production uses the AIC Mode 2 rendezvous; the simulator build
 * bypasses AIC and synchronizes the two AIVs directly with Mode 1.
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
constexpr uint32_t kScaleIndexElementCount = kProjectionTileWidth;
#ifndef DSV4_SIM_STEPS
#define DSV4_SIM_STEPS 2
#endif

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
    const LocalTensor<uint32_t>& scale_group_indices_ub,
    const LocalTensor<bfloat16_t>& expanded_scales_bf16,
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
    LocalTensor<fp4x2_e2m1_t> packed_weights_fp4         = packed_weights_bytes_ub.ReinterpretCast<fp4x2_e2m1_t>();
    UnaryRepeatParams fp4_cast_repeat_params(1, 1, 8, 2);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16, packed_weights_fp4, RoundMode::CAST_NONE, 128ull, static_cast<uint8_t>(kWeightCastRepeatCount), fp4_cast_repeat_params);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16[kWeightCastHalfElementCount], packed_weights_fp4[kWeightCastHalfElementCount / 2], RoundMode::CAST_NONE, 128ull, static_cast<uint8_t>(kWeightCastRepeatCount), fp4_cast_repeat_params);
    Cast<uint16_t, uint8_t>(scale_codes_bf16_bits, weight_scale_codes_ub, RoundMode::CAST_NONE, kScaleTileElementCount);
    ShiftLeft(scale_codes_bf16_bits, scale_codes_bf16_bits, static_cast<uint16_t>(7), kScaleTileElementCount);
    LocalTensor<bfloat16_t>   scale_bf16                 = scale_codes_bf16_bits.ReinterpretCast<bfloat16_t>();
    PipeBarrier<PIPE_V>();
    for (uint32_t reduction = 0; reduction < kReductionTile; ++reduction) {
        const uint32_t weight_offset = reduction * kProjectionTileWidth;
        const uint32_t scale_base_offset = reduction * kScaleGroupsPerProjectionTile;
        const bfloat16_t activation_scalar = activation_tokens_bf16.GetValue(reduction_tile_start + reduction);
        Gather(expanded_scales_bf16, scale_bf16, scale_group_indices_ub, scale_base_offset, kProjectionTileWidth);
        Mul(scaled_weight_product_bf16, dequant_weights_bf16[weight_offset], expanded_scales_bf16, kProjectionTileWidth);
        if (trace_weight_strip && reduction == 0) {
            TraceBf16("[dsv4] dequantized weights", scaled_weight_product_bf16, trace_values_fp32);
        }
        Muls(scaled_weight_product_bf16, scaled_weight_product_bf16, activation_scalar, kProjectionTileWidth);
        Add(projection_accumulator_bf16, projection_accumulator_bf16, scaled_weight_product_bf16, kProjectionTileWidth);
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
    LocalTensor<bfloat16_t>   inter_activations_bf16     = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<bfloat16_t>   tokens_tile_ub             = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kHiddenDimension);
    LocalTensor<uint8_t>      packed_weights_ub          = allocator.Alloc<TPosition::VECIN, uint8_t>(kWeightTileElementCount / 2);
    LocalTensor<bfloat16_t>   dequant_weights_bf16       = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kWeightTileElementCount);
    LocalTensor<bfloat16_t>   scaled_weight_product_bf16 = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<uint8_t>      scale_codes_ub             = allocator.Alloc<TPosition::VECIN, uint8_t>(kScaleTileElementCount);
    LocalTensor<uint16_t>     scale_bits_ub              = allocator.Alloc<TPosition::VECCALC, uint16_t>(kScaleTileElementCount);
    LocalTensor<uint32_t>     scale_group_indices_ub     = allocator.Alloc<TPosition::VECCALC, uint32_t>(kScaleIndexElementCount);
    LocalTensor<bfloat16_t>   expanded_scales_bf16       = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t>   gate_acc_bf16              = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t>   up_acc_bf16                = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t>   down_acc_bf16              = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kProjectionTileWidth);
    LocalTensor<bfloat16_t>   gate_bf16                  = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<bfloat16_t>   up_bf16                    = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kIntermediateBandWidth);
    LocalTensor<float>        scratch_acc_fp32           = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);
    LocalTensor<float>        scratch_math_fp32          = allocator.Alloc<TPosition::VECCALC, float>(kIntermediateBandWidth);

    for (uint32_t group = 0; group < kScaleGroupsPerProjectionTile; ++group) {
        Duplicate(scale_group_indices_ub[group * kScaleGroupSize], static_cast<uint32_t>(group), kScaleGroupSize);
    }

    DataCopy(tokens_tile_ub, input_tokens_gm[token_row * kHiddenDimension], kHiddenDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] input tokens", tokens_tile_ub, scratch_acc_fp32);
    }

    const uint32_t inter_start = sub_block_idx * kIntermediateBandWidth;
    Duplicate(gate_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
    Duplicate(up_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
    for (uint32_t reduction_tile_start = 0; reduction_tile_start < kHiddenDimension; reduction_tile_start += kReductionTile) {
        AccumulateProjectionTile(gate_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, scale_group_indices_ub, expanded_scales_bf16, gate_weights_gm, gate_scales_gm, kIntermediateDimension, kIntermediateDimension / kScaleGroupSize, reduction_tile_start, inter_start, scratch_acc_fp32, token_row == 0 && sub_block_idx == 0);
        AccumulateProjectionTile(up_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, scale_group_indices_ub, expanded_scales_bf16, up_weights_gm, up_scales_gm, kIntermediateDimension, kIntermediateDimension / kScaleGroupSize, reduction_tile_start, inter_start, scratch_acc_fp32, false);
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
    Cast<float, bfloat16_t>(scratch_acc_fp32, gate_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    Cast<float, bfloat16_t>(scratch_math_fp32, up_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] gate GEMM", gate_bf16, scratch_math_fp32);
    }
    Muls(scratch_acc_fp32, scratch_acc_fp32, -1.0f, kIntermediateBandWidth);
    Exp(scratch_acc_fp32, scratch_acc_fp32, kIntermediateBandWidth);
    Adds(scratch_acc_fp32, scratch_acc_fp32, 1.0f, kIntermediateBandWidth);
    Duplicate(scratch_math_fp32, 1.0f, kIntermediateBandWidth);
    Div(scratch_math_fp32, scratch_math_fp32, scratch_acc_fp32, kIntermediateBandWidth);
    Cast<float, bfloat16_t>(scratch_acc_fp32, gate_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    Mul(scratch_acc_fp32, scratch_acc_fp32, scratch_math_fp32, kIntermediateBandWidth);
    Cast<float, bfloat16_t>(scratch_math_fp32, up_bf16, RoundMode::CAST_NONE, kIntermediateBandWidth);
    Mul(scratch_acc_fp32, scratch_acc_fp32, scratch_math_fp32, kIntermediateBandWidth);
    Cast<bfloat16_t, float>(inter_activations_bf16, scratch_acc_fp32, RoundMode::CAST_RINT, kIntermediateBandWidth);
    PipeBarrier<PIPE_V>();
    if (token_row == 0 && sub_block_idx == 0) {
        TraceBf16("[dsv4] SwiGLU", inter_activations_bf16, scratch_acc_fp32);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(inter_gm[token_row * kIntermediateDimension + inter_start], inter_activations_bf16, kIntermediateBandWidth);
    PipeBarrier<PIPE_ALL>();

    CrossCoreSetFlag<2, PIPE_V>(0);
    CrossCoreWaitFlag<2, PIPE_FIX>(1);
    DataCopy(tokens_tile_ub, inter_gm[token_row * kIntermediateDimension], kIntermediateDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t hidden_start = sub_block_idx * kHiddenBandWidth;
    for (uint32_t projection_output_start = hidden_start; projection_output_start < hidden_start + kHiddenBandWidth; projection_output_start += kProjectionTileWidth) {
        Duplicate(down_acc_bf16, bfloat16_t(0.0f), kProjectionTileWidth);
        for (uint32_t reduction_tile_start = 0; reduction_tile_start < kIntermediateDimension; reduction_tile_start += kReductionTile) {
            AccumulateProjectionTile(down_acc_bf16, tokens_tile_ub, packed_weights_ub, dequant_weights_bf16, scaled_weight_product_bf16, scale_codes_ub, scale_bits_ub, scale_group_indices_ub, expanded_scales_bf16, down_weights_gm, down_scales_gm, kHiddenDimension, kHiddenDimension / kScaleGroupSize, reduction_tile_start, projection_output_start, scratch_acc_fp32, false);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(final_output_gm[token_row * kHiddenDimension + projection_output_start], down_acc_bf16, kProjectionTileWidth);
        if (token_row == 0 && sub_block_idx == 0 && projection_output_start == hidden_start) {
            TraceBf16("[dsv4] down projection", down_acc_bf16, scratch_acc_fp32);
        }
    }
}
__aicore__ inline void TraceSimBf16(
    const LocalTensor<bfloat16_t>& values,
    uint32_t value_offset,
    const LocalTensor<float>& trace_values_fp32) {
    Cast<float, bfloat16_t>(trace_values_fp32, values[value_offset], RoundMode::CAST_NONE, 8);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    DSV4_PIPELINE_TRACE("%f %f %f %f\n", trace_values_fp32.GetValue(0),
                        trace_values_fp32.GetValue(1), trace_values_fp32.GetValue(2),
                        trace_values_fp32.GetValue(3));
    DSV4_PIPELINE_TRACE("%f %f %f %f\n", trace_values_fp32.GetValue(4),
                        trace_values_fp32.GetValue(5), trace_values_fp32.GetValue(6),
                        trace_values_fp32.GetValue(7));
}

__aicore__ inline void AccumulateMicroBand(
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
    uint32_t reduction_start,
    uint32_t activation_offset,
    uint32_t sub_block_idx,
    uint32_t step,
    const LocalTensor<float>& trace_values_fp32,
    bool trace_weight_strip) {
    constexpr uint32_t kRowsPerStep = 16;
    constexpr uint32_t kPaddedProjectionWidth = 64;
    constexpr uint32_t kPackedBytesPerRow = kPaddedProjectionWidth / 2;
    constexpr uint32_t kScaleBytesPerRow = 32;
    constexpr uint32_t kScaleTileByteCount = kRowsPerStep * kScaleBytesPerRow;
    constexpr uint32_t kWeightTileElementCount = kRowsPerStep * kPaddedProjectionWidth;
    DataCopyParams weight_params;
    weight_params.blockCount = kRowsPerStep;
    weight_params.blockLen   = kPackedBytesPerRow / 32;
    weight_params.srcStride  = (projection_output_dimension / 2 - kPackedBytesPerRow) / 32;
    weight_params.dstStride  = 0;
    DataCopyParams scale_params;
    scale_params.blockCount = kRowsPerStep;
    scale_params.blockLen   = 1;
    scale_params.srcStride  = (scale_groups_per_output_row - kScaleBytesPerRow) / 32;
    scale_params.dstStride  = 0;
    const uint32_t packed_offset = reduction_start * projection_output_dimension / 2;
    const uint32_t scale_offset = reduction_start * scale_groups_per_output_row;
    DataCopy(packed_weights_bytes_ub, packed_weights_gm[packed_offset], weight_params);
    DataCopy(weight_scale_codes_ub, weight_scales_gm[scale_offset], scale_params);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LocalTensor<fp4x2_e2m1_t> packed_weights_fp4         = packed_weights_bytes_ub.ReinterpretCast<fp4x2_e2m1_t>();
    UnaryRepeatParams fp4_cast_repeat_params(1, 1, 8, 2);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16, packed_weights_fp4, RoundMode::CAST_NONE, 128ull, 4, fp4_cast_repeat_params);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16[kWeightTileElementCount / 2], packed_weights_fp4[kWeightTileElementCount / 4], RoundMode::CAST_NONE, 128ull, 4, fp4_cast_repeat_params);
    Cast<uint16_t, uint8_t>(scale_codes_bf16_bits, weight_scale_codes_ub, RoundMode::CAST_NONE, kScaleTileByteCount);
    ShiftLeft(scale_codes_bf16_bits, scale_codes_bf16_bits, static_cast<uint16_t>(7), kScaleTileByteCount);
    LocalTensor<bfloat16_t>   scale_values_bf16          = scale_codes_bf16_bits.ReinterpretCast<bfloat16_t>();
    PipeBarrier<PIPE_V>();
    const uint32_t column_offset = sub_block_idx * 16;
    for (uint32_t reduction = 0; reduction < kRowsPerStep; ++reduction) {
        const bfloat16_t activation_scalar = activation_tokens_bf16.GetValue(activation_offset + reduction);
        const bfloat16_t scale_scalar = scale_values_bf16.GetValue(reduction * 32 + column_offset / 32);
        const uint32_t tile_offset = reduction * kPaddedProjectionWidth + column_offset;
        Muls(scaled_weight_product_bf16, dequant_weights_bf16[tile_offset], scale_scalar, 16);
        if (trace_weight_strip && reduction == 0) {
            DSV4_PIPELINE_TRACE("[step %u] dequantized weights\n", step);
            TraceSimBf16(scaled_weight_product_bf16, 0, trace_values_fp32);
        }
        Muls(scaled_weight_product_bf16, scaled_weight_product_bf16, activation_scalar, 16);
        Add(projection_accumulator_bf16, projection_accumulator_bf16, scaled_weight_product_bf16, 16);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunMicroTileCore(
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
    constexpr uint32_t kRowsPerStep = 16;
    constexpr uint32_t kPaddedProjectionWidth = 64;
    constexpr uint32_t kSimOutputWidth = 16;
    constexpr uint32_t kSimOutputColumns = 32;
    constexpr uint32_t kSimStepsLocal = DSV4_SIM_STEPS;
    constexpr uint32_t kSimReductionCount = kRowsPerStep * kSimStepsLocal;
    constexpr uint32_t kScaleGroupsPerGateRow = kIntermediateDimension / 32;
    constexpr uint32_t kWeightTileElementCount = kRowsPerStep * kPaddedProjectionWidth;
    constexpr uint32_t kScaleTileElementCount = kRowsPerStep * 32;
    LocalMemAllocator<Hardware::UB> allocator;
    LocalTensor<bfloat16_t>   gate_accumulator_bf16      = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kSimOutputWidth);
    LocalTensor<bfloat16_t>   up_accumulator_bf16        = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kSimOutputWidth);
    LocalTensor<bfloat16_t>   down_accumulator_bf16      = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kSimOutputWidth);
    LocalTensor<bfloat16_t>   tokens_ub                  = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kTokenCount * kSimReductionCount);
    LocalTensor<uint8_t>      packed_weights_ub          = allocator.Alloc<TPosition::VECIN, uint8_t>(kWeightTileElementCount / 2);
    LocalTensor<bfloat16_t>   dequant_weights_ub         = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kWeightTileElementCount);
    LocalTensor<bfloat16_t>   scaled_products_ub         = allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kPaddedProjectionWidth);
    LocalTensor<uint8_t>      scale_codes_ub             = allocator.Alloc<TPosition::VECIN, uint8_t>(kScaleTileElementCount);
    LocalTensor<uint16_t>     scale_bits_ub              = allocator.Alloc<TPosition::VECCALC, uint16_t>(kScaleTileElementCount);
    LocalTensor<bfloat16_t>   gate_bf16                  = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kSimOutputWidth);
    LocalTensor<bfloat16_t>   up_bf16                    = allocator.Alloc<TPosition::VECIN, bfloat16_t>(kSimOutputWidth);
    LocalTensor<bfloat16_t>   inter_bf16                 = allocator.Alloc<TPosition::VECOUT, bfloat16_t>(kSimOutputWidth);
    LocalTensor<float>        scratch_acc_fp32           = allocator.Alloc<TPosition::VECCALC, float>(kSimOutputWidth);
    LocalTensor<float>        scratch_math_fp32          = allocator.Alloc<TPosition::VECCALC, float>(kSimOutputWidth);

    DataCopy(tokens_ub, input_tokens_gm, kTokenCount * kSimReductionCount);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (sub_block_idx == 0) {
        for (uint32_t step = 0; step < kSimStepsLocal; ++step) {
            DSV4_PIPELINE_TRACE("[step %u] raw loaded tokens\n", step);
            TraceSimBf16(tokens_ub, step * kRowsPerStep, scratch_acc_fp32);
        }
    }

    for (uint32_t token_row = 0; token_row < kTokenCount; ++token_row) {
        Duplicate(gate_accumulator_bf16, bfloat16_t(0.0f), kSimOutputWidth);
        Duplicate(up_accumulator_bf16, bfloat16_t(0.0f), kSimOutputWidth);
        for (uint32_t step = 0; step < kSimStepsLocal; ++step) {
            const uint32_t reduction_start = step * kRowsPerStep;
            const uint32_t activation_offset = token_row * kSimReductionCount + reduction_start;
            AccumulateMicroBand(gate_accumulator_bf16, tokens_ub, packed_weights_ub, dequant_weights_ub, scaled_products_ub, scale_codes_ub, scale_bits_ub, gate_weights_gm, gate_scales_gm, kIntermediateDimension, kScaleGroupsPerGateRow, reduction_start, activation_offset, sub_block_idx, step, scratch_acc_fp32, token_row == 0 && sub_block_idx == 0);
            AccumulateMicroBand(up_accumulator_bf16, tokens_ub, packed_weights_ub, dequant_weights_ub, scaled_products_ub, scale_codes_ub, scale_bits_ub, up_weights_gm, up_scales_gm, kIntermediateDimension, kScaleGroupsPerGateRow, reduction_start, activation_offset, sub_block_idx, step, scratch_acc_fp32, false);
            if (token_row == 0 && sub_block_idx == 0) {
                DSV4_PIPELINE_TRACE("[step %u] gate accumulator after Muls/Add\n", step);
                TraceSimBf16(gate_accumulator_bf16, 0, scratch_acc_fp32);
            }
        }
        const uint32_t output_offset = token_row * kSimOutputColumns + sub_block_idx * kSimOutputWidth;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(gate_gm[output_offset], gate_accumulator_bf16, kSimOutputWidth);
        DataCopy(up_gm[output_offset], up_accumulator_bf16, kSimOutputWidth);
    }
    PipeBarrier<PIPE_ALL>();

    for (uint32_t token_row = 0; token_row < kTokenCount; ++token_row) {
        const uint32_t output_offset = token_row * kSimOutputColumns + sub_block_idx * kSimOutputWidth;
        DataCopy(gate_bf16, gate_gm[output_offset], kSimOutputWidth);
        DataCopy(up_bf16, up_gm[output_offset], kSimOutputWidth);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (token_row == 0 && sub_block_idx == 0) {
            DSV4_PIPELINE_TRACE("[phase 2] gate input to SwiGLU\n");
            TraceSimBf16(gate_bf16, 0, scratch_acc_fp32);
        }
        Cast<float, bfloat16_t>(scratch_acc_fp32, gate_bf16, RoundMode::CAST_NONE, kSimOutputWidth);
        Cast<float, bfloat16_t>(scratch_math_fp32, up_bf16, RoundMode::CAST_NONE, kSimOutputWidth);
        Muls(scratch_acc_fp32, scratch_acc_fp32, -1.0f, kSimOutputWidth);
        Exp(scratch_acc_fp32, scratch_acc_fp32, kSimOutputWidth);
        Adds(scratch_acc_fp32, scratch_acc_fp32, 1.0f, kSimOutputWidth);
        Duplicate(scratch_math_fp32, 1.0f, kSimOutputWidth);
        Div(scratch_math_fp32, scratch_math_fp32, scratch_acc_fp32, kSimOutputWidth);
        Cast<float, bfloat16_t>(scratch_acc_fp32, gate_bf16, RoundMode::CAST_NONE, kSimOutputWidth);
        Mul(scratch_acc_fp32, scratch_acc_fp32, scratch_math_fp32, kSimOutputWidth);
        Cast<float, bfloat16_t>(scratch_math_fp32, up_bf16, RoundMode::CAST_NONE, kSimOutputWidth);
        Mul(scratch_acc_fp32, scratch_acc_fp32, scratch_math_fp32, kSimOutputWidth);
        Cast<bfloat16_t, float>(inter_bf16, scratch_acc_fp32, RoundMode::CAST_RINT, kSimOutputWidth);
        PipeBarrier<PIPE_V>();
        if (token_row == 0 && sub_block_idx == 0) {
            DSV4_PIPELINE_TRACE("[phase 2] intermediate SwiGLU output\n");
            TraceSimBf16(inter_bf16, 0, scratch_acc_fp32);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(inter_gm[output_offset], inter_bf16, kSimOutputWidth);
    }
    PipeBarrier<PIPE_ALL>();

    PipeBarrier<PIPE_ALL>();
    for (uint32_t token_row = 0; token_row < kTokenCount; ++token_row) {
        const uint32_t output_offset = token_row * kSimOutputColumns + sub_block_idx * kSimOutputWidth;
        DataCopy(tokens_ub, inter_gm[token_row * kSimOutputColumns], kSimReductionCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(down_accumulator_bf16, bfloat16_t(0.0f), kSimOutputWidth);
        for (uint32_t step = 0; step < kSimStepsLocal; ++step) {
            const uint32_t reduction_start = step * kRowsPerStep;
            AccumulateMicroBand(down_accumulator_bf16, tokens_ub, packed_weights_ub, dequant_weights_ub, scaled_products_ub, scale_codes_ub, scale_bits_ub, down_weights_gm, down_scales_gm, kHiddenDimension, kHiddenDimension / 32, reduction_start, reduction_start, sub_block_idx, step, scratch_acc_fp32, false);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(final_output_gm[output_offset], down_accumulator_bf16, kSimOutputWidth);
        if (token_row == 0 && sub_block_idx == 0) {
            DSV4_PIPELINE_TRACE("[phase 3] down-projection output\n");
            TraceSimBf16(down_accumulator_bf16, 0, scratch_acc_fp32);
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

#if defined(DSV4_SIM_MICROTILE)
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        const uint32_t sub_block_idx = GetSubBlockIdx();
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
        RunMicroTileCore(sub_block_idx, input_tokens, gate_weights, gate_scales, up_weights, up_scales, down_weights, down_scales, gate_projection, up_projection, inter_activations, final_output);
    }
#else
    if ASCEND_IS_AIC {
        const uint32_t token_row = GetBlockIdx();
        if (token_row < kTokenCount) {
            CrossCoreWaitFlag<2, PIPE_V>(0);
            CrossCoreSetFlag<2, PIPE_FIX>(1);
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
#endif
}

}
