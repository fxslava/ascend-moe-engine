// -----------------------------------------------------------------------------
// fused_mhc_pre_norm_sinkhorn.cpp -- host-side launch wrapper.
//
// Compiled by BiSheng as an ASC TU: it dispatches with the triple-chevron
// direct-launch syntax. See src/kernels/arch35/launch_common.hpp for why that
// form and not ACLRT_LAUNCH_KERNEL, and CLAUDE.md §3 for the binding rule.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "fused_mhc_pre_norm_sinkhorn_tiling.h"

namespace moe::kernels::arch35 {
namespace {

// The device entry point, as defined in op_kernel/.
//
// Declared with GM_ADDR (an address-space-qualified __gm__ uint8_t*) because
// that is the kernel ABI. Host-side arguments are handed to the chevron call
// as plain uint8_t*; BiSheng performs the conversion as part of lowering the
// launch, which is why no cast to GM_ADDR appears at the call site (and why
// one would not compile -- void* cannot be cast to a __gm__ pointer).
extern "C" __global__ __aicore__ void fused_mhc_pre_norm_sinkhorn(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
    GM_ADDR h_raw, GM_ADDR h_norm, GM_ADDR post, GM_ADDR birkhoff,
    GM_ADDR workspace, GM_ADDR tiling);

constexpr const char* kOp = "fused_mhc_pre_norm_sinkhorn";

}  // namespace

Status LaunchMhcPreNormSinkhorn(
    const Bf16* x_gm, std::uint32_t x_stream_stride, const float* phi_gm,
    std::uint32_t phi_row_stride, const float* alpha_gm, const float* bias_gm,
    const Bf16* gamma_gm, float* h_raw_gm, Bf16* h_norm_gm, float* post_gm,
    float* birkhoff_gm, void* workspace, std::size_t workspace_bytes,
    aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (x_gm == nullptr || phi_gm == nullptr || alpha_gm == nullptr ||
      bias_gm == nullptr || gamma_gm == nullptr || h_raw_gm == nullptr ||
      h_norm_gm == nullptr || post_gm == nullptr || birkhoff_gm == nullptr ||
      stream == nullptr) {
    return Status::invalid_argument;
  }
  // GM scratch is 0 B for this kernel, so a null workspace is correct and a
  // caller passing a buffer is not an error. What WOULD be an error is asking
  // for a nonzero scratch size this kernel does not have.
  if (workspace_bytes != kMhcPreNormWorkspaceBytes) {
    return Status::invalid_argument;
  }

  MhcPreNormTilingData tiling{};
  const Status shape =
      MakeMhcPreNormTiling(x_stream_stride, phi_row_stride, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  fused_mhc_pre_norm_sinkhorn<<<tiling.block_dim, nullptr, stream>>>(
      detail::Gm(x_gm), detail::Gm(phi_gm), detail::Gm(alpha_gm),
      detail::Gm(bias_gm), detail::Gm(gamma_gm), detail::Gm(h_raw_gm),
      detail::Gm(h_norm_gm), detail::Gm(post_gm), detail::Gm(birkhoff_gm),
      detail::Gm(workspace), detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
