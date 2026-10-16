// SIM ONLY -- host launch wrapper for variant (a) of the prefill-cube
// bisection. Compiled by BiSheng, because the triple-chevron form is the only
// dispatch and BiSheng is what lowers it (CLAUDE.md sections 3.1-3.3).
//
// It stages the SAME blob layout production stages -- the 72-byte operator POD
// at offset 0, then the gate and down TCubeTilings -- so the kernel resolves
// its tiling pointers with production's offsets. The POD's contents are
// irrelevant here (nothing reads geometry), but the two TCubeTilings are not:
// REGIST_MATMUL_OBJ copies them out of GM, so they must be real.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include <cstring>

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void test_kfc_lifecycle_kernel(
    GM_ADDR shared_workspace, GM_ADDR tiling_config, GM_ADDR status_out);

constexpr const char* kLifecycleOp = "test_kfc_lifecycle_kernel";

}  // namespace

Status LaunchKfcLifecycleProbe(void* workspace_gm, std::int32_t* status_gm,
                               std::uint32_t hidden, std::uint32_t inter,
                               std::uint32_t num_experts,
                               std::uint32_t total_tokens,
                               std::uint32_t aiv_count, aclrtStream stream) {
  if (workspace_gm == nullptr || status_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillCubeTiling(
      hidden, inter, num_experts, total_tokens, aiv_count, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  alignas(8) unsigned char staging_blob[kPrefillCubeStagingBytes] = {};
  std::memcpy(staging_blob, &tiling, sizeof(tiling));
  const Status gate_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::gate_up, tiling.m_tile,
      staging_blob + kPrefillCubeGateTilingOffset);
  if (gate_tiling != Status::success) {
    return gate_tiling;
  }
  const Status down_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::down, tiling.m_tile,
      staging_blob + kPrefillCubeDownTilingOffset);
  if (down_tiling != Status::success) {
    return down_tiling;
  }

  detail::TilingStaging staging;
  const Status staged =
      staging.Stage(kLifecycleOp, staging_blob, sizeof(staging_blob));
  if (staged != Status::success) {
    return staged;
  }

  // block_dim 1: one AIC block, which in MIX_AIC_1_2 carries two AIV
  // subblocks. Pinned rather than taken from the tiling so the topology under
  // test cannot drift with the fixture's token count.
  test_kfc_lifecycle_kernel<<<1u, 0u, stream>>>(
      detail::Gm(workspace_gm), detail::Gm(staging.get()),
      detail::Gm(status_gm));

  return detail::FinishLaunch(kLifecycleOp, stream);
}

}  // namespace moe::kernels::arch35
