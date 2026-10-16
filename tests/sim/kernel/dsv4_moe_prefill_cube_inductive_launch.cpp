// SIM ONLY -- host launch wrapper for the collapsed dsv4_moe_prefill_cube
// gate.
//
// Compiled by BiSheng as an ASC TU, for the reason in CLAUDE.md §3.2-3.3: the
// triple-chevron form is the only dispatch and BiSheng is what lowers it, so a
// host compiler cannot build this file.
//
// A deliberate near-copy of LaunchDsv4MoePrefillCube in
// src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube.cpp: SAME tiling
// builder, SAME aliasing rejections, SAME staging helper, SAME dyn_ub_size of
// zero, SAME FinishLaunch. That matters -- a shape this wrapper admits is
// exactly a shape the production wrapper would admit, so the collapsed grid is
// a property of the KERNEL and never of the tiling.
//
// TWO DEVIATIONS, both load-bearing for the gate:
//
//   1. `aiv_count` is an explicit parameter instead of
//      detail::QueryAivCoreCount(). Under CAModel the capability query reports
//      the simulated part, and the grid has to be pinned to a small
//      reproducible value -- a core count that moved with the host's SKU would
//      make both the run's cost and its coverage irreproducible. The
//      production wrapper queries; this one is told.
//   2. `sample_cap_gm` points at one uint32 in GM that caps how many of the
//      sampled tiles run. The CAModel cost model is MEASURED, never derived
//      (three predictions for these suites have been wrong), so the driver
//      needs to be able to run 1 and 2 tiles to measure this gate's own fixed
//      cost. It is a separate operand rather than a tiling field because
//      Dsv4MoePrefillCubeTilingData's 72-byte layout is public ABI.
//
// Everything downstream of the builder is identical, so the geometry
// validation under test is production's. The wrapper owns validation, the
// tiling staging buffer, the dispatch and its teardown; it does not allocate
// or copy operand data (§3.4), including the caller-owned workspace.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_inductive_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR sample_cap, GM_ADDR tiling);

constexpr const char* kSimOp = "dsv4_moe_prefill_cube_inductive_sim";

}  // namespace

Status LaunchDsv4MoePrefillCubeInductiveSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, const std::uint32_t* sample_cap_gm,
    Dsv4MoePrefillCubeTilingData* tiling_out, aclrtStream stream) {
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
      hidden, inter, num_experts, total_tokens, aiv_count, &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (tiling_out != nullptr) {
    *tiling_out = tiling;  // so the driver can print and mirror the grid
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kSimOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // dyn_ub_size stays ZERO, exactly as in production: this kernel has no SIMT
  // VF, so its UB comes from TQue/TBuf in the STATIC region and declaring
  // dynamic bytes nobody addresses would reserve UB the static allocator then
  // could not use.
  dsv4_moe_prefill_cube_inductive_sim<<<tiling.block_dim, 0u, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(w1_packed_gm),
      detail::Gm(w1_scale_gm), detail::Gm(w3_packed_gm),
      detail::Gm(w3_scale_gm), detail::Gm(w2_packed_gm),
      detail::Gm(w2_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(workspace_gm),
      detail::Gm(sample_cap_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kSimOp, stream);
}

}  // namespace moe::kernels::arch35
