// -----------------------------------------------------------------------------
// bf16_block_quant_fp8.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "bf16_block_quant_fp8_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void bf16_block_quant_fp8(GM_ADDR x,
                                                           GM_ADDR y,
                                                           GM_ADDR scale,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tiling);

constexpr const char* kOp = "bf16_block_quant_fp8";

}  // namespace

Status LaunchBf16BlockQuantFp8(const Bf16* x_gm, std::uint32_t width,
                               Fp8E4M3* y_gm, float* scale_gm, void* workspace,
                               std::size_t workspace_bytes,
                               aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (x_gm == nullptr || y_gm == nullptr || scale_gm == nullptr ||
      stream == nullptr) {
    return Status::invalid_argument;
  }
  if (workspace_bytes != kBlockQuantWorkspaceBytes) {
    return Status::invalid_argument;
  }
  // The caller owns every operand buffer (§3.4), and this kernel writes y and
  // scale while reading x, so an alias between them is a caller mistake that
  // would silently corrupt the result mid-tile. Checked here because the
  // device cannot see it.
  if (static_cast<const void*>(x_gm) == static_cast<const void*>(y_gm) ||
      static_cast<const void*>(x_gm) == static_cast<const void*>(scale_gm) ||
      static_cast<const void*>(y_gm) == static_cast<const void*>(scale_gm)) {
    return Status::invalid_argument;
  }

  Bf16BlockQuantFp8TilingData tiling{};
  const Status shape = MakeBf16BlockQuantFp8Tiling(width, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // The second chevron slot is dyn_ub_size (uint32 BYTES), not an l2ctrl
  // pointer: the 950PR hybrid launch is
  // kernel<<<block_num, dyn_ub_size, stream>>>. Three of this kernel's four
  // UB buffers are addressed by the SIMT VF, and a declared size of 0 leaves
  // the VF addressing a region the launch never reserved -- the error-341 /
  // aclError-507035 class. CLAUDE.md §3.1 prescribes nullptr for this slot;
  // the dynamic-UB argument is the granted exception, as in the other SIMT
  // operators here.
  const std::uint64_t dyn_ub_bytes = BlockQuantDynUbBytes();
  if (dyn_ub_bytes > kBlockQuantDynUbCeilingBytes) {
    return Status::unsupported_shape;
  }

  bf16_block_quant_fp8<<<tiling.block_dim,
                         static_cast<std::uint32_t>(dyn_ub_bytes), stream>>>(
      detail::Gm(x_gm), detail::Gm(y_gm), detail::Gm(scale_gm),
      detail::Gm(workspace), detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
