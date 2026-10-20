#ifndef DSV4_MOE_PREFILL_CUBE_HUMAN_WRITED_H
#define DSV4_MOE_PREFILL_CUBE_HUMAN_WRITED_H

#include <acl/acl.h>
#include <cstdint>

namespace moe::kernels::arch35 {
inline constexpr uint32_t kHumanMoeTokens = 16;
inline constexpr uint32_t kHumanMoeHidden = 4096;
inline constexpr uint32_t kHumanMoeInter = 2048;

void LaunchDsv4MoePrefillCubeHumanWrited(
    const uint16_t* x, const uint8_t* gate_packed, const uint8_t* gate_scales,
    const uint8_t* up_packed, const uint8_t* up_scales,
    const uint8_t* down_packed, const uint8_t* down_scales, float* gate,
    float* up, float* inter, float* y, aclrtStream stream);
}  // namespace moe::kernels::arch35

#endif
