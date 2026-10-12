// -----------------------------------------------------------------------------
// mxfp4_expert_gemv.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "mxfp4_expert_gemv_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void mxfp4_expert_gemv(GM_ADDR packed,
                                                        GM_ADDR scale,
                                                        GM_ADDR x, GM_ADDR y,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tiling);

constexpr const char* kOp = "mxfp4_expert_gemv";

}  // namespace

Status LaunchMxfp4ExpertGemv(
    const Mxfp4Pair* packed_gm, std::uint32_t packed_row_stride,
    const E8M0* scale_gm, std::uint32_t scale_row_stride, const float* x_gm,
    std::uint32_t rows, std::uint32_t cols, float* y_gm, void* workspace,
    std::size_t workspace_bytes, aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (packed_gm == nullptr || scale_gm == nullptr || x_gm == nullptr ||
      y_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  if (workspace_bytes != kMxfp4GemvWorkspaceBytes) {
    return Status::invalid_argument;
  }

  Mxfp4GemvTilingData tiling{};
  const Status shape = MakeMxfp4GemvTiling(packed_row_stride, scale_row_stride,
                                           rows, cols, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  mxfp4_expert_gemv<<<tiling.block_dim, nullptr, stream>>>(
      detail::Gm(packed_gm), detail::Gm(scale_gm), detail::Gm(x_gm),
      detail::Gm(y_gm), detail::Gm(workspace), detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
