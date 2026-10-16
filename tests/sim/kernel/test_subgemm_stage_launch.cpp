// SIM ONLY -- host launch wrapper for the staged variant (b) bisection.
// Compiled by BiSheng (CLAUDE.md sections 3.1-3.3). Stages the production blob
// layout plus one trailing uint32 stage selector, behind both TCubeTilings for
// the reason the sim sample cap is there: offset 72 belongs to the gate tiling.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include <cstring>

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void test_subgemm_stage_kernel(
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR status_out, GM_ADDR shared_workspace, GM_ADDR tiling_config);

constexpr const char* kStageOp = "test_subgemm_stage_kernel";

}  // namespace

Status LaunchSubgemmStageProbe(const Mxfp4Pair* w_packed_gm,
                               const E8M0* w_scale_gm, std::int32_t* status_gm,
                               void* workspace_gm, std::uint32_t hidden,
                               std::uint32_t inter, std::uint32_t num_experts,
                               std::uint32_t total_tokens,
                               std::uint32_t aiv_count, std::uint32_t stage,
                               aclrtStream stream) {
  if (w_packed_gm == nullptr || w_scale_gm == nullptr || status_gm == nullptr ||
      workspace_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillCubeTiling(
      hidden, inter, num_experts, total_tokens, aiv_count, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  alignas(8) unsigned char blob[kPrefillCubeStagingBytes + sizeof(std::uint32_t)] = {};
  std::memcpy(blob, &tiling, sizeof(tiling));
  const Status gate_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::gate_up, tiling.m_tile, blob + kPrefillCubeGateTilingOffset);
  if (gate_tiling != Status::success) {
    return gate_tiling;
  }
  const Status down_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::down, tiling.m_tile, blob + kPrefillCubeDownTilingOffset);
  if (down_tiling != Status::success) {
    return down_tiling;
  }
  std::memcpy(blob + kPrefillCubeStagingBytes, &stage, sizeof(stage));

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kStageOp, blob, sizeof(blob));
  if (staged != Status::success) {
    return staged;
  }

  test_subgemm_stage_kernel<<<1u, 0u, stream>>>(
      detail::Gm(w_packed_gm), detail::Gm(w_scale_gm), detail::Gm(status_gm),
      detail::Gm(workspace_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kStageOp, stream);
}

}  // namespace moe::kernels::arch35
