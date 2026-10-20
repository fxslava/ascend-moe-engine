#include "kernel_operator.h"
#include "launch_common.hpp"
#include "dsv4_moe_prefill_cube_human_writed.h"

namespace moe::kernels::arch35 {
extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_human_writed(
    GM_ADDR x_gm, GM_ADDR gate_packed_gm, GM_ADDR gate_scales_gm,
    GM_ADDR up_packed_gm, GM_ADDR up_scales_gm, GM_ADDR down_packed_gm,
    GM_ADDR down_scales_gm, GM_ADDR gate_gm, GM_ADDR up_gm, GM_ADDR inter_gm,
    GM_ADDR y_gm);

void LaunchDsv4MoePrefillCubeHumanWrited(
    const uint16_t* x, const uint8_t* gate_packed, const uint8_t* gate_scales,
    const uint8_t* up_packed, const uint8_t* up_scales,
    const uint8_t* down_packed, const uint8_t* down_scales, float* gate,
    float* up, float* inter, float* y, aclrtStream stream) {
  using detail::Gm;
  dsv4_moe_prefill_cube_human_writed<<<16, 0u, stream>>>(
      Gm(x), Gm(gate_packed), Gm(gate_scales), Gm(up_packed), Gm(up_scales),
      Gm(down_packed), Gm(down_scales), Gm(gate), Gm(up), Gm(inter), Gm(y));
}
}  // namespace moe::kernels::arch35
