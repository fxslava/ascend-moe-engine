// -----------------------------------------------------------------------------
// fused_mhc_post_residual.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "fused_mhc_post_residual_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void fused_mhc_post_residual(
    GM_ADDR x, GM_ADDR birkhoff, GM_ADDR post, GM_ADDR h_out, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling);

constexpr const char* kOp = "fused_mhc_post_residual";

}  // namespace

Status LaunchMhcPostResidual(
    const Bf16* x_gm, std::uint32_t x_stream_stride, const float* birkhoff_gm,
    std::uint32_t b_row_stride, const float* post_gm, const Bf16* h_out_gm,
    Bf16* y_gm, std::uint32_t y_stream_stride, void* workspace,
    std::size_t workspace_bytes, aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (x_gm == nullptr || birkhoff_gm == nullptr || post_gm == nullptr ||
      h_out_gm == nullptr || y_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // "in-place x == y is forbidden because parallel output streams can
  // overwrite another block's source." Catch it here: on device this would be
  // a silent data race, not a fault.
  if (static_cast<const void*>(x_gm) == static_cast<const void*>(y_gm)) {
    return Status::invalid_argument;
  }
  if (workspace_bytes != kMhcPostWorkspaceBytes) {
    return Status::invalid_argument;
  }

  MhcPostTilingData tiling{};
  const Status shape = MakeMhcPostTiling(x_stream_stride, y_stream_stride,
                                         b_row_stride, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  fused_mhc_post_residual<<<tiling.block_dim, nullptr, stream>>>(
      detail::Gm(x_gm), detail::Gm(birkhoff_gm), detail::Gm(post_gm),
      detail::Gm(h_out_gm), detail::Gm(y_gm), detail::Gm(workspace),
      detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
