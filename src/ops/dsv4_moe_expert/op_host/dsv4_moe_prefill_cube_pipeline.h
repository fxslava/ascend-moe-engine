#ifndef DSV4_MOE_PREFILL_CUBE_PIPELINE_H
#define DSV4_MOE_PREFILL_CUBE_PIPELINE_H

// =================================================================================================
// Host launch surface for the dsv4_moe_prefill_cube_pipeline entry point.
//
// Declared in op_host/ next to its wrapper, the way this operator's other op_host headers are, and
// deliberately NOT in include/moe/kernels/arch35/launch.hpp: this entry point exists to be verified
// in isolation, and a consumer that wants it links the four TUs it needs rather than the whole
// engine launch layer.
//
// Two deliberate differences from LaunchDsv4MoePrefillCube in op_host/dsv4_moe_prefill_cube.cpp:
//   * `aiv_count` is a parameter rather than detail::QueryAivCoreCount(), so a caller pins the grid
//     to a reproducible value instead of inheriting the host part's core count. In
//     KERNEL_TYPE_MIX_AIC_1_2 one AI Core group is 1 AIC + 2 AIV.
//   * The staged tiling blob is the published 72-byte POD alone. The production wrapper appends two
//     TCubeTilings because REGIST_MATMUL_OBJ needs them; this kernel instantiates no Matmul object
//     and stands up no KfcServer, so there is nothing to append.
// =================================================================================================

#include "moe/kernels/arch35/launch.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstddef>
#include <cstdint>

namespace moe::kernels::arch35 {

// The AIC spins on one 128-byte line per paired AIV until both have signalled that ring slot 0 of
// the L1 operand ring holds their operands. Those bytes must be ZERO at launch; the wrapper zeroes
// them with a SYNCHRONOUS H2D copy rather than aclrtMemsetAsync, which the ESL SDMA model does not
// implement and which aborts the dav-3510 simulator before the kernel runs.
inline constexpr std::size_t kPrefillCubePipelineStartupBytes = 256;

// What the caller must allocate for `workspace_gm`. Only the startup lines are read; the rest is
// headroom so a later rendezvous word needs no caller change.
inline constexpr std::size_t kPrefillCubePipelineWorkspaceBytes = 4096;

Status LaunchDsv4MoePrefillCubePipeline(
    const float* x_tokens_gm, const Mxfp4Pair* gate_packed_gm,
    const E8M0* gate_scale_gm, const Mxfp4Pair* up_packed_gm,
    const E8M0* up_scale_gm, const Mxfp4Pair* down_packed_gm,
    const E8M0* down_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, Dsv4MoePrefillCubeTilingData* tiling_out,
    aclrtStream stream);

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_PREFILL_CUBE_PIPELINE_H
