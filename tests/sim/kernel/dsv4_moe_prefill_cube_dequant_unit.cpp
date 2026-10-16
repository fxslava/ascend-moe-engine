// Standalone CAModel check of the production MXFP4/E8M0 vector conversion.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "launch_common.hpp"

#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_dequant_unit(
    GM_ADDR packed, GM_ADDR scales, GM_ADDR output, GM_ADDR shape) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (GetBlockIdx() != 0) return;

    const __gm__ std::uint32_t* dimensions = reinterpret_cast<const __gm__ std::uint32_t*>(shape);
    const std::uint32_t rows = dimensions[0];
    const std::uint32_t cols = dimensions[1];

    TPipe pipe;
    TQue<TPosition::VECCALC, 1> q_nibbles, q_expanded, q_scales, q_plane, q_temporary;
    pipe.InitBuffer(q_nibbles, 1, kCubeNibBytes);
    pipe.InitBuffer(q_expanded, 1, kCubeWideBytes);
    pipe.InitBuffer(q_scales, 1, kCubeScaleBytes);
    pipe.InitBuffer(q_plane, 1, kCubeWideBytes);
    pipe.InitBuffer(q_temporary, 1, 16 * sizeof(std::uint16_t));

    Buffers buf;
    buf.weight_nibbles = q_nibbles.AllocTensor<fp4x2_e2m1_t>();
    buf.weight_expanded = q_expanded.AllocTensor<bfloat16_t>();
    buf.weight_scales = q_scales.AllocTensor<std::uint8_t>();
    buf.scale_plane = q_plane.AllocTensor<bfloat16_t>();
    buf.scale_temporary = q_temporary.AllocTensor<std::uint16_t>();

    GlobalTensor<std::uint8_t> packed_gm;
    packed_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(packed), rows * cols / 2);
    GlobalTensor<std::uint8_t> scales_gm;
    scales_gm.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(scales),
                              rows * kCubeScaleRowStride);
    DataCopy(buf.weight_nibbles.ReinterpretCast<std::uint8_t>(), packed_gm, rows * cols / 2);
    DataCopy(buf.weight_scales, scales_gm, rows * kCubeScaleRowStride);
    SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
    DequantSubTileVector(rows, cols, buf);

    GlobalTensor<bfloat16_t> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output), rows * cols);
    SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    DataCopy(result, buf.weight_expanded, rows * cols);

    q_temporary.FreeTensor(buf.scale_temporary);
    q_plane.FreeTensor(buf.scale_plane);
    q_scales.FreeTensor(buf.weight_scales);
    q_expanded.FreeTensor(buf.weight_expanded);
    q_nibbles.FreeTensor(buf.weight_nibbles);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube

extern "C" void LaunchDsv4MoePrefillCubeDequantUnit(
    void* packed, void* scales, void* output, void* shape, aclrtStream stream) {
    moe::kernels::arch35::dsv4_moe_prefill_cube::dsv4_moe_prefill_cube_dequant_unit
        <<<1, 0, stream>>>(moe::kernels::arch35::detail::Gm(packed),
                           moe::kernels::arch35::detail::Gm(scales),
                           moe::kernels::arch35::detail::Gm(output),
                           moe::kernels::arch35::detail::Gm(shape));
}
