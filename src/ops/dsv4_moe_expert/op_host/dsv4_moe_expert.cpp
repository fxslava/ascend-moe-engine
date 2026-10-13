// -----------------------------------------------------------------------------
// dsv4_moe_expert.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
//
// TWO DISPATCHES, ONE OPERATOR. The down leg reduces over the whole
// `activated` vector while the gate/up legs are split across 32 cores by row
// tile, so no core holds all of `activated` when its own tiles finish. The
// barrier between the two phases is stream ordering, and `activated` lives in
// the caller's workspace across it. This is the same shape
// LaunchFp8IndexerTopK already uses for its score/select pair: one directory,
// one host ABI, one tiling POD, two entry points.
//
// The caller owns `workspace` as it owns every other GM buffer (§3.4); this
// wrapper allocates only the tiling staging, and does that once for both
// dispatches because the two kernels read the same POD.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_expert_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_expert_gate_up(
    GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed, GM_ADDR w3_scale,
    GM_ADDR x, GM_ADDR activated, GM_ADDR tiling);

extern "C" __global__ __aicore__ void dsv4_moe_expert_down(
    GM_ADDR w2_packed, GM_ADDR w2_scale, GM_ADDR activated, GM_ADDR y,
    GM_ADDR tiling);

constexpr const char* kOp = "dsv4_moe_expert";

}  // namespace

std::size_t Dsv4MoeExpertWorkspaceBytes(std::uint32_t inter) noexcept {
  return static_cast<std::size_t>(Dsv4ExpertWorkspaceBytes(inter));
}

Status LaunchDsv4MoeExpert(
    const Mxfp4Pair* w1_packed_gm, const E8M0* w1_scale_gm,
    const Mxfp4Pair* w3_packed_gm, const E8M0* w3_scale_gm,
    const Mxfp4Pair* w2_packed_gm, const E8M0* w2_scale_gm,
    std::uint32_t w13_packed_row_stride, std::uint32_t w13_scale_row_stride,
    std::uint32_t w2_packed_row_stride, std::uint32_t w2_scale_row_stride,
    const float* x_gm, std::uint32_t hidden, std::uint32_t inter, float* y_gm,
    void* workspace, std::size_t workspace_bytes, aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (w1_packed_gm == nullptr || w1_scale_gm == nullptr ||
      w3_packed_gm == nullptr || w3_scale_gm == nullptr ||
      w2_packed_gm == nullptr || w2_scale_gm == nullptr || x_gm == nullptr ||
      y_gm == nullptr || workspace == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // `activated` is an operand this operator owns the lifetime of but not the
  // storage for, so a short workspace is a caller mistake, not a geometry
  // this version declines to implement (§2.6).
  if (static_cast<std::uint64_t>(workspace_bytes) <
      Dsv4ExpertWorkspaceBytes(inter)) {
    return Status::invalid_argument;
  }
  // The down pass reads `activated` as a GM operand while the gate_up pass
  // writes it. Aliasing either against y or x would make the second dispatch
  // read what the first overwrote.
  if (workspace == static_cast<void*>(y_gm) ||
      workspace == static_cast<const void*>(x_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoeExpertTilingData tiling{};
  const Status shape = MakeDsv4MoeExpertTiling(
      hidden, inter, w13_packed_row_stride, w13_scale_row_stride,
      w2_packed_row_stride, w2_scale_row_stride, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  const std::uint64_t gate_up_ub = Dsv4ExpertGateUpDynUbBytes(tiling.hidden);
  const std::uint64_t down_ub = Dsv4ExpertDownDynUbBytes(tiling.inter);
  if (gate_up_ub > kDsv4ExpertDynUbCeilingBytes ||
      down_ub > kDsv4ExpertDynUbCeilingBytes) {
    // Unreachable for the one geometry MakeDsv4MoeExpertTiling admits, but
    // the ceiling is part of the launch contract, so it is checked rather
    // than assumed.
    return Status::unsupported_shape;
  }

  // One staging buffer for both dispatches: the two kernels read the same
  // POD, and it stays live until the trailing synchronize in FinishLaunch.
  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // The second chevron slot is dyn_ub_size (uint32 BYTES), not an l2ctrl
  // pointer: the 950PR SIMD/SIMT hybrid launch is
  // kernel<<<block_num, dyn_ub_size, stream>>>. Both entry points hand
  // __ubuf__ pointers to a SIMT VF, so a declared size of 0 leaves the VF
  // addressing a region the launch never reserved -- the error-341 / 507035
  // class. CLAUDE.md §3.1 prescribes nullptr for this slot; the user granted
  // an explicit exception for the dynamic-UB argument.
  dsv4_moe_expert_gate_up<<<tiling.block_dim,
                            static_cast<std::uint32_t>(gate_up_ub), stream>>>(
      detail::Gm(w1_packed_gm), detail::Gm(w1_scale_gm),
      detail::Gm(w3_packed_gm), detail::Gm(w3_scale_gm), detail::Gm(x_gm),
      detail::Gm(workspace), detail::Gm(staging.get()));

  // No explicit barrier: both dispatches are on `stream`, so the down pass is
  // ordered after every core's gate_up tile stores.
  dsv4_moe_expert_down<<<tiling.block_dim,
                         static_cast<std::uint32_t>(down_ub), stream>>>(
      detail::Gm(w2_packed_gm), detail::Gm(w2_scale_gm),
      detail::Gm(workspace), detail::Gm(y_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
