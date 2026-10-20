/*
 * Educational arch35 MXFP4 MoE prefill kernel for Ascend 950PR and CANN 9.2.
 *
 * Each AIV block owns one token row. The gate and up projections traverse
 * K=4096 in 32-row reduction tiles and N=2048 in 512-column tiles. The down
 * projection uses K=2048 and N=4096 with the same tile sizes. Matrix weights
 * are row-major MXFP4 E2M1 values, with one E8M0 scale code per 32 adjacent
 * values. Packed weights and complete scale rows are copied into UB through
 * MTE2 before vector conversion and arithmetic.
 *
 * The SIMD data path converts packed FP4 to BF16 with Cast, converts scale
 * codes to powers of two with vector Cast/Add/Mul/Exp, then applies each
 * group scale and token scalar through vector Muls before vector Add
 * accumulation across K. The full SwiGLU epilogue uses vector Muls, Exp,
 * Adds, Duplicate, Div, and Mul. Intermediate projections are materialized
 * in GM so each stage boundary is explicit and easy to inspect.
 *
 * UB allocations are deterministic and 32-byte aligned through
 * LocalMemAllocator. The largest tile uses 16,384 FP4 values, 16,384 BF16
 * dequantized values, 16,384 FP32 dequantized values, and 4,096 scale codes.
 * MTE2-to-Vector copies and Vector-to-Scalar trace reads use paired hard
 * events. Stage transitions use explicit barriers after GM materialization.
 */

#include "kernel_operator.h"

#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
#include "utils/debug/asc_printf.h"
#define DSV4_PIPELINE_TRACE(...) __asc_aicore::printf(__VA_ARGS__)
#else
#define DSV4_PIPELINE_TRACE(...)
#endif

namespace AscendC {
namespace {

constexpr uint32_t kTokenCount            = 16;
constexpr uint32_t kHiddenDimension       = 4096;
constexpr uint32_t kIntermediateDimension = 2048;
constexpr uint32_t kScaleGroupSize        = 32;
constexpr uint32_t kReductionTile         = 32;
constexpr uint32_t kOutputTile            = 512;
constexpr uint32_t kWeightTileElements    = kReductionTile * kOutputTile;
constexpr uint32_t kWeightCastRepeatCount = kWeightTileElements / 128;
constexpr uint32_t kMaximumScaleRowCount  = kHiddenDimension / kScaleGroupSize;
constexpr uint32_t kScaleTileElementCount = kReductionTile * kMaximumScaleRowCount;

__aicore__ inline void TraceInput(const LocalTensor<float>& input_tokens) {
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    DSV4_PIPELINE_TRACE("[dsv4-human] input tokens: %f %f %f %f %f %f %f %f\n", input_tokens.GetValue(0), input_tokens.GetValue(1), input_tokens.GetValue(2), input_tokens.GetValue(3), input_tokens.GetValue(4), input_tokens.GetValue(5), input_tokens.GetValue(6), input_tokens.GetValue(7));
}

__aicore__ inline void TraceWeightStrip(const LocalTensor<float>& dequant_weights) {
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    DSV4_PIPELINE_TRACE("[dsv4-human] dequantized weight strip: %f %f %f %f %f %f %f %f\n", dequant_weights.GetValue(0), dequant_weights.GetValue(1), dequant_weights.GetValue(2), dequant_weights.GetValue(3), dequant_weights.GetValue(4), dequant_weights.GetValue(5), dequant_weights.GetValue(6), dequant_weights.GetValue(7));
}

__aicore__ inline void AccumulateWeightTile(
    const LocalTensor<float>& projection_accumulator,
    const LocalTensor<float>& activation_tokens,
    const LocalTensor<uint8_t>& weights_packed_bytes,
    const LocalTensor<bfloat16_t>& dequant_weights_bf16,
    const LocalTensor<float>& dequant_weights_float,
    const LocalTensor<float>& scaled_weight_product,
    const LocalTensor<uint8_t>& scale_codes,
    const LocalTensor<uint16_t>& scale_codes_as_uint16,
    const LocalTensor<float>& scale_factors,
    const GlobalTensor<uint8_t>& packed_weights_global,
    const GlobalTensor<uint8_t>& weight_scales_global,
    const uint32_t projection_output_dimension,
    const uint32_t scale_groups_per_output_row,
    const uint32_t reduction_tile_start,
    const uint32_t output_tile_start,
    const bool trace_weight_strip) {
    DataCopyParams weights_copy_params;
    weights_copy_params.blockCount = kReductionTile;
    weights_copy_params.blockLen   = (kOutputTile / 2) / 32;
    weights_copy_params.srcStride  = ((projection_output_dimension - kOutputTile) / 2) / 32;
    weights_copy_params.dstStride  = 0;

    DataCopyParams scales_copy_params;
    scales_copy_params.blockCount = kReductionTile;
    scales_copy_params.blockLen   = scale_groups_per_output_row / 32;
    scales_copy_params.srcStride  = 0;
    scales_copy_params.dstStride  = 0;

    const uint32_t packed_weight_offset = (reduction_tile_start * projection_output_dimension + output_tile_start) / 2;
    const uint32_t scale_row_offset = reduction_tile_start * scale_groups_per_output_row;
    DataCopy(weights_packed_bytes, packed_weights_global[packed_weight_offset], weights_copy_params);
    DataCopy(scale_codes, weight_scales_global[scale_row_offset], scales_copy_params);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    LocalTensor<fp4x2_e2m1_t> packed_weights_fp4 = weights_packed_bytes.ReinterpretCast<fp4x2_e2m1_t>();
    UnaryRepeatParams fp4_cast_repeat_params(1, 1, 8, 2);
    Cast<bfloat16_t, fp4x2_e2m1_t>(dequant_weights_bf16, packed_weights_fp4, RoundMode::CAST_NONE, 128ull, static_cast<uint8_t>(kWeightCastRepeatCount), fp4_cast_repeat_params);
    Cast<float, bfloat16_t>(dequant_weights_float, dequant_weights_bf16, RoundMode::CAST_NONE, kWeightTileElements);

    const uint32_t scale_tile_count = kReductionTile * scale_groups_per_output_row;
    Cast<uint16_t, uint8_t>(scale_codes_as_uint16, scale_codes, RoundMode::CAST_NONE, scale_tile_count);
    Cast<float, uint16_t>(scale_factors, scale_codes_as_uint16, RoundMode::CAST_NONE, scale_tile_count);
    Adds(scale_factors, scale_factors, -127.0f, scale_tile_count);
    Muls(scale_factors, scale_factors, 0.6931471805599453f, scale_tile_count);
    Exp(scale_factors, scale_factors, scale_tile_count);
    PipeBarrier<PIPE_V>();

    const uint32_t scale_group_start = output_tile_start / kScaleGroupSize;
    const uint32_t groups_in_output_tile = kOutputTile / kScaleGroupSize;
    for (uint32_t reduction_tile_offset = 0; reduction_tile_offset < kReductionTile; ++reduction_tile_offset) {
        const float activation_value = activation_tokens.GetValue(reduction_tile_start + reduction_tile_offset);
        for (uint32_t scale_group_offset = 0; scale_group_offset < groups_in_output_tile; ++scale_group_offset) {
            const uint32_t weight_offset = reduction_tile_offset * kOutputTile + scale_group_offset * kScaleGroupSize;
            const uint32_t scale_offset = reduction_tile_offset * scale_groups_per_output_row + scale_group_start + scale_group_offset;
            const float scale_value = scale_factors.GetValue(scale_offset);
            Muls(scaled_weight_product[scale_group_offset * kScaleGroupSize], dequant_weights_float[weight_offset], scale_value, kScaleGroupSize);
            if (trace_weight_strip && reduction_tile_offset == 0 && scale_group_offset == 0) {
                TraceWeightStrip(scaled_weight_product);
            }
            Muls(scaled_weight_product[scale_group_offset * kScaleGroupSize], scaled_weight_product[scale_group_offset * kScaleGroupSize], activation_value, kScaleGroupSize);
            Add(projection_accumulator[scale_group_offset * kScaleGroupSize], projection_accumulator[scale_group_offset * kScaleGroupSize], scaled_weight_product[scale_group_offset * kScaleGroupSize], kScaleGroupSize);
        }
    }
    PipeBarrier<PIPE_V>();
}

}

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_human_writed(
    GM_ADDR input_tokens_gm,
    GM_ADDR gate_packed_gm,
    GM_ADDR gate_scales_gm,
    GM_ADDR up_packed_gm,
    GM_ADDR up_scales_gm,
    GM_ADDR down_packed_gm,
    GM_ADDR down_scales_gm,
    GM_ADDR gate_projection_gm,
    GM_ADDR up_projection_gm,
    GM_ADDR inter_activation_gm,
    GM_ADDR final_output_gm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t token_row = GetBlockIdx();
    if (token_row >= kTokenCount) {
        return;
    }

    LocalMemAllocator<Hardware::UB> ub_allocator;
    LocalTensor<bfloat16_t> input_tokens_bf16     = ub_allocator.Alloc<TPosition::VECIN, bfloat16_t>(kHiddenDimension);
    LocalTensor<float>      input_tokens_float    = ub_allocator.Alloc<TPosition::VECCALC, float>(kHiddenDimension);
    LocalTensor<uint8_t>    weights_packed_bytes  = ub_allocator.Alloc<TPosition::VECIN, uint8_t>(kWeightTileElements / 2);
    LocalTensor<bfloat16_t> dequant_weights_bf16  = ub_allocator.Alloc<TPosition::VECCALC, bfloat16_t>(kWeightTileElements);
    LocalTensor<float>      dequant_weights_float = ub_allocator.Alloc<TPosition::VECCALC, float>(kWeightTileElements);
    LocalTensor<float>      scaled_weight_product = ub_allocator.Alloc<TPosition::VECCALC, float>(kOutputTile);
    LocalTensor<uint8_t>    scale_codes           = ub_allocator.Alloc<TPosition::VECIN, uint8_t>(kScaleTileElementCount);
    LocalTensor<uint16_t>   scale_codes_as_uint16 = ub_allocator.Alloc<TPosition::VECCALC, uint16_t>(kScaleTileElementCount);
    LocalTensor<float>      scale_factors         = ub_allocator.Alloc<TPosition::VECCALC, float>(kScaleTileElementCount);
    LocalTensor<float>      gate_accumulator      = ub_allocator.Alloc<TPosition::VECOUT, float>(kOutputTile);
    LocalTensor<float>      up_accumulator        = ub_allocator.Alloc<TPosition::VECOUT, float>(kOutputTile);
    LocalTensor<float>      down_accumulator      = ub_allocator.Alloc<TPosition::VECOUT, float>(kOutputTile);
    LocalTensor<float>      gate_activation       = ub_allocator.Alloc<TPosition::VECIN, float>(kIntermediateDimension);
    LocalTensor<float>      up_activation         = ub_allocator.Alloc<TPosition::VECIN, float>(kIntermediateDimension);
    LocalTensor<float>      inter_swiglu_ub       = ub_allocator.Alloc<TPosition::VECCALC, float>(kIntermediateDimension);
    LocalTensor<float>      negated_gate          = ub_allocator.Alloc<TPosition::VECCALC, float>(kIntermediateDimension);
    LocalTensor<float>      exp_negated_gate      = ub_allocator.Alloc<TPosition::VECCALC, float>(kIntermediateDimension);
    LocalTensor<float>      sigmoid_denominator   = ub_allocator.Alloc<TPosition::VECCALC, float>(kIntermediateDimension);
    LocalTensor<float>      sigmoid_numerator     = ub_allocator.Alloc<TPosition::VECCALC, float>(kIntermediateDimension);

    GlobalTensor<bfloat16_t>   input_global;
    GlobalTensor<uint8_t>      gate_weights_global;
    GlobalTensor<uint8_t>      gate_scales_global;
    GlobalTensor<uint8_t>      up_weights_global;
    GlobalTensor<uint8_t>      up_scales_global;
    GlobalTensor<uint8_t>      down_weights_global;
    GlobalTensor<uint8_t>      down_scales_global;
    GlobalTensor<float>        gate_output_global;
    GlobalTensor<float>        up_output_global;
    GlobalTensor<float>        inter_output_global;
    GlobalTensor<float>        output_global;
    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input_tokens_gm));
    gate_weights_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gate_packed_gm));
    gate_scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gate_scales_gm));
    up_weights_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(up_packed_gm));
    up_scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(up_scales_gm));
    down_weights_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(down_packed_gm));
    down_scales_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(down_scales_gm));
    gate_output_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gate_projection_gm));
    up_output_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(up_projection_gm));
    inter_output_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inter_activation_gm));
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(final_output_gm));

    DataCopy(input_tokens_bf16, input_global[token_row * kHiddenDimension], kHiddenDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast<float, bfloat16_t>(input_tokens_float, input_tokens_bf16, RoundMode::CAST_NONE, kHiddenDimension);
    PipeBarrier<PIPE_V>();
    if (token_row == 0) {
        TraceInput(input_tokens_float);
    }

    const uint32_t gate_scale_groups_per_row = kIntermediateDimension / kScaleGroupSize;
    for (uint32_t output_tile_start = 0; output_tile_start < kIntermediateDimension; output_tile_start += kOutputTile) {
        Duplicate(gate_accumulator, 0.0f, kOutputTile);
        Duplicate(up_accumulator, 0.0f, kOutputTile);
        for (uint32_t reduction_tile_start = 0; reduction_tile_start < kHiddenDimension; reduction_tile_start += kReductionTile) {
            AccumulateWeightTile(gate_accumulator, input_tokens_float, weights_packed_bytes, dequant_weights_bf16, dequant_weights_float, scaled_weight_product, scale_codes, scale_codes_as_uint16, scale_factors, gate_weights_global, gate_scales_global, kIntermediateDimension, gate_scale_groups_per_row, reduction_tile_start, output_tile_start, token_row == 0 && output_tile_start == 0);
            AccumulateWeightTile(up_accumulator, input_tokens_float, weights_packed_bytes, dequant_weights_bf16, dequant_weights_float, scaled_weight_product, scale_codes, scale_codes_as_uint16, scale_factors, up_weights_global, up_scales_global, kIntermediateDimension, gate_scale_groups_per_row, reduction_tile_start, output_tile_start, false);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(gate_output_global[token_row * kIntermediateDimension + output_tile_start], gate_accumulator, kOutputTile);
        DataCopy(up_output_global[token_row * kIntermediateDimension + output_tile_start], up_accumulator, kOutputTile);
        PipeBarrier<PIPE_ALL>();
        if (token_row == 0 && output_tile_start == 0) {
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            DSV4_PIPELINE_TRACE("[dsv4-human] gate gemm slice: %f %f %f %f %f %f %f %f\n", gate_accumulator.GetValue(0), gate_accumulator.GetValue(1), gate_accumulator.GetValue(2), gate_accumulator.GetValue(3), gate_accumulator.GetValue(4), gate_accumulator.GetValue(5), gate_accumulator.GetValue(6), gate_accumulator.GetValue(7));
        }
    }

    PipeBarrier<PIPE_ALL>();
    DataCopy(gate_activation, gate_output_global[token_row * kIntermediateDimension], kIntermediateDimension);
    DataCopy(up_activation, up_output_global[token_row * kIntermediateDimension], kIntermediateDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Muls(negated_gate, gate_activation, -1.0f, kIntermediateDimension);
    Exp(exp_negated_gate, negated_gate, kIntermediateDimension);
    Adds(sigmoid_denominator, exp_negated_gate, 1.0f, kIntermediateDimension);
    Duplicate(sigmoid_numerator, 1.0f, kIntermediateDimension);
    Div(sigmoid_numerator, sigmoid_numerator, sigmoid_denominator, kIntermediateDimension);
    Mul(inter_swiglu_ub, gate_activation, sigmoid_numerator, kIntermediateDimension);
    Mul(inter_swiglu_ub, inter_swiglu_ub, up_activation, kIntermediateDimension);
    PipeBarrier<PIPE_V>();
    if (token_row == 0) {
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        DSV4_PIPELINE_TRACE("[dsv4-human] SwiGLU activation: %f %f %f %f %f %f %f %f\n", inter_swiglu_ub.GetValue(0), inter_swiglu_ub.GetValue(1), inter_swiglu_ub.GetValue(2), inter_swiglu_ub.GetValue(3), inter_swiglu_ub.GetValue(4), inter_swiglu_ub.GetValue(5), inter_swiglu_ub.GetValue(6), inter_swiglu_ub.GetValue(7));
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    DataCopy(inter_output_global[token_row * kIntermediateDimension], inter_swiglu_ub, kIntermediateDimension);
    PipeBarrier<PIPE_ALL>();

    DataCopy(input_tokens_float, inter_output_global[token_row * kIntermediateDimension], kIntermediateDimension);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    const uint32_t down_scale_groups_per_row = kHiddenDimension / kScaleGroupSize;
    for (uint32_t output_tile_start = 0; output_tile_start < kHiddenDimension; output_tile_start += kOutputTile) {
        Duplicate(down_accumulator, 0.0f, kOutputTile);
        for (uint32_t reduction_tile_start = 0; reduction_tile_start < kIntermediateDimension; reduction_tile_start += kReductionTile) {
            AccumulateWeightTile(down_accumulator, input_tokens_float, weights_packed_bytes, dequant_weights_bf16, dequant_weights_float, scaled_weight_product, scale_codes, scale_codes_as_uint16, scale_factors, down_weights_global, down_scales_global, kHiddenDimension, down_scale_groups_per_row, reduction_tile_start, output_tile_start, token_row == 0 && output_tile_start == 0);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(output_global[token_row * kHiddenDimension + output_tile_start], down_accumulator, kOutputTile);
        PipeBarrier<PIPE_ALL>();
        if (token_row == 0 && output_tile_start == 0) {
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
            DSV4_PIPELINE_TRACE("[dsv4-human] final down projection: %f %f %f %f %f %f %f %f\n", down_accumulator.GetValue(0), down_accumulator.GetValue(1), down_accumulator.GetValue(2), down_accumulator.GetValue(3), down_accumulator.GetValue(4), down_accumulator.GetValue(5), down_accumulator.GetValue(6), down_accumulator.GetValue(7));
        }
    }
}

}
