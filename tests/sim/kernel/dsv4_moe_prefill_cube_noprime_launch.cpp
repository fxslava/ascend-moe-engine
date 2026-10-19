/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- host launch wrapper for the no-priming bisection probe.
//
// A DELIBERATE NEAR-COPY of LaunchDsv4MoePrefillCube in
// src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube.cpp: same tiling
// builder, same aliasing rejections, same staging blob with both Cube tilings
// behind the operator POD, same QueryAivCoreCount(), same dyn_ub_size of zero,
// same FinishLaunch. The ONLY difference is the entry symbol it dispatches to.
//
// That matters for the experiment: a shape this wrapper admits is exactly a
// shape the production wrapper admits, so any difference between this probe's
// run and the production gate's run is a property of the ONE kernel-side delta
// (the removed V_MTE2 priming pair) and not of the launch path. A wrapper that
// pinned the grid differently, or staged the tiling differently, would make
// the comparison worthless.
//
// Compiled by BiSheng as an ASC TU (CLAUDE.md sections 3.2-3.3): the
// triple-chevron form is the only dispatch and BiSheng is what lowers it.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include <cstring>

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_noprime_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR tiling);

constexpr const char* kNoPrimeOp = "dsv4_moe_prefill_cube_noprime_sim";

}  // namespace

Status LaunchDsv4MoePrefillCubeNoPrimeSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, float* y_tokens_gm, void* workspace_gm,
    aclrtStream stream) {
  if (x_tokens_gm == nullptr || w1_packed_gm == nullptr ||
      w1_scale_gm == nullptr || w3_packed_gm == nullptr ||
      w3_scale_gm == nullptr || w2_packed_gm == nullptr ||
      w2_scale_gm == nullptr || token_to_expert_gm == nullptr ||
      y_tokens_gm == nullptr || workspace_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  if (static_cast<const void*>(y_tokens_gm) ==
      static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }
  if (workspace_gm == static_cast<const void*>(x_tokens_gm) ||
      workspace_gm == static_cast<const void*>(y_tokens_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillCubeTiling(
      hidden, inter, num_experts, total_tokens, detail::QueryAivCoreCount(),
      &tiling);
  if (shape != Status::success) {
    return shape;
  }

  // The staging blob carries the operator POD at offset 0 and both Cube
  // tilings behind it, exactly as production does. The probe's kernel never
  // reads the Cube tilings -- REGIST_MATMUL_OBJ is gone from this path -- but
  // the blob is kept byte-identical so the staging allocation, its zeroing and
  // its H2D copy are the same ones the production gate exercises.
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
      staging.Stage(kNoPrimeOp, staging_blob, sizeof(staging_blob));
  if (staged != Status::success) {
    return staged;
  }

  // dyn_ub_size stays ZERO, as in production: no SIMT VF, so the UB comes from
  // TQue/TBuf in the STATIC region.
  dsv4_moe_prefill_cube_noprime_sim<<<tiling.block_dim, 0u, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(w1_packed_gm),
      detail::Gm(w1_scale_gm), detail::Gm(w3_packed_gm),
      detail::Gm(w3_scale_gm), detail::Gm(w2_packed_gm),
      detail::Gm(w2_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(workspace_gm),
      detail::Gm(staging.get()));

  return detail::FinishLaunch(kNoPrimeOp, stream);
}

}  // namespace moe::kernels::arch35
