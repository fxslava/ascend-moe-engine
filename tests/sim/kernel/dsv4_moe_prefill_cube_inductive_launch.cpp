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
//   2. A simulation-only uint32 follows the production tiling POD in its
//      staged GM allocation and caps how many sampled tiles run. The CAModel cost model is MEASURED, never derived
//      (three predictions for these suites have been wrong), so the driver
//      needs to be able to run 1 and 2 tiles to measure this gate's own fixed
//      cost. The production 72-byte tiling ABI remains unchanged.
//
// Everything downstream of the builder is identical, so the geometry
// validation under test is production's. The wrapper owns validation, the
// tiling staging buffer, the dispatch and its teardown; it does not allocate
// or copy operand data (§3.4), including the caller-owned workspace.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include <cstring>

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_inductive_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR tiling);

constexpr const char* kSimOp = "dsv4_moe_prefill_cube_inductive_sim";

}  // namespace

Status LaunchDsv4MoePrefillCubeInductiveSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, std::uint32_t sample_cap,
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

  // THE STAGING BUFFER CARRIES THE OPERATOR POD **AND** THE TWO CUBE TILINGS.
  //
  // The 72-byte Dsv4MoePrefillCubeTilingData is unchanged and still at offset
  // 0; the Matmul framework's two TCubeTilings follow it. They have to reach
  // the device somehow, and REGIST_MATMUL_OBJ needs them as
  // `const __gm__ TCubeTiling*` -- see dsv4_moe_prefill_cube_cube_tiling.h for
  // what passing no tiling at all did instead (a null-pointer read storm on
  // the Cube that never terminated).
  //
  // This is NOT an operator ABI break: the published POD keeps its size and
  // its MOE_ARCH35_ASSERT_TILING_POD. The staging buffer is wrapper-private.
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

  // The sim-only sample cap now sits BEHIND both Cube tilings. It used to be
  // at sizeof(Dsv4MoePrefillCubeTilingData) == 72, which is where the gate
  // tiling starts; leaving it there would have made the cap and the gate
  // tiling's `usedCoreNum` the same four bytes.
  alignas(8) unsigned char sim_blob[kPrefillCubeStagingBytes + sizeof(std::uint32_t)] = {};
  std::memcpy(sim_blob, staging_blob, sizeof(staging_blob));
  std::memcpy(sim_blob + kPrefillCubeStagingBytes, &sample_cap, sizeof(sample_cap));

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kSimOp, sim_blob, sizeof(sim_blob));
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
      detail::Gm(staging.get()));

  return detail::FinishLaunch(kSimOp, stream);
}

}  // namespace moe::kernels::arch35
