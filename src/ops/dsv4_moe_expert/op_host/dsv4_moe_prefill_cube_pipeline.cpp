// =================================================================================================
// dsv4_moe_prefill_cube_pipeline.cpp -- the chevron launch wrapper for the decomposed three-stage
// Cube prefill pipeline. Compiled by BiSheng as an ASC TU, like every other op_host TU here.
//
// The triple-chevron form is the only dispatch this toolkit supports (CLAUDE.md section 3.1-3.3):
// the ASC flow passes -DGEN_ACLRT=<dir> but nothing consumes it, so no aclrtlaunch_<kernel> stub is
// emitted and a host-compiler build of this file would fail at link with an undefined symbol.
//
// The wrapper owns geometry validation, the tiling staging buffer, the cold-start rendezvous
// zeroing, the dispatch and its teardown. It does not allocate or copy operand data; every operand
// GM buffer belongs to the caller.
//
// A chevron launch returns void, so no error is observable at the call. Every ACL failure is
// recorded through detail::RecordAclFailure, which is what lets LastLaunchFailure() tell a tiling
// malloc, a rendezvous copy, a bad grid and a device-side fault apart.
// =================================================================================================

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_pipeline.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_pipeline(
    GM_ADDR x_tokens, GM_ADDR gate_packed, GM_ADDR gate_scale, GM_ADDR up_packed,
    GM_ADDR up_scale, GM_ADDR down_packed, GM_ADDR down_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace, GM_ADDR tiling);

constexpr const char* kPipelineOp = "dsv4_moe_prefill_cube_pipeline";

}  // namespace

Status LaunchDsv4MoePrefillCubePipeline(
    const float* x_tokens_gm, const Mxfp4Pair* gate_packed_gm,
    const E8M0* gate_scale_gm, const Mxfp4Pair* up_packed_gm,
    const E8M0* up_scale_gm, const Mxfp4Pair* down_packed_gm,
    const E8M0* down_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, Dsv4MoePrefillCubeTilingData* tiling_out,
    aclrtStream stream) {
  if (x_tokens_gm == nullptr || gate_packed_gm == nullptr ||
      gate_scale_gm == nullptr || up_packed_gm == nullptr ||
      up_scale_gm == nullptr || down_packed_gm == nullptr ||
      down_scale_gm == nullptr || token_to_expert_gm == nullptr ||
      y_tokens_gm == nullptr || workspace_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // Phase 2 writes y rows that a later token tile on another core has not yet read as x, with no
  // ordering between them, so an in-place launch is rejected rather than merely discouraged.
  if (static_cast<const void*>(y_tokens_gm) == static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }
  if (workspace_gm == static_cast<const void*>(x_tokens_gm) ||
      workspace_gm == static_cast<const void*>(y_tokens_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillCubeTiling(hidden, inter, num_experts,
                                                    total_tokens, aiv_count, &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (tiling_out != nullptr) {
    *tiling_out = tiling;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kPipelineOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // The cold-start rendezvous lines, zeroed with an explicit SYNCHRONOUS H2D copy.
  //
  // NOT aclrtMemsetAsync: the ESL SDMA model implements PREFETCH / WRITE_BACK / INVALIDATE / FLUSH
  // and not memset, and aborts the dav-3510 simulator before the kernel runs at all ("ESL Model
  // not support memset in sdma!", exit 134). A synchronous copy works on hardware and on the
  // simulator alike, and being synchronous it is ordered before the dispatch with no stream fence.
  //
  // The wrapper does it rather than the caller because launch.hpp has always documented this
  // workspace as needing no initialization: a caller handing over recycled GM with a nonzero byte
  // in the first 256 would otherwise release the Cube before any operand existed.
  const unsigned char zeroed_startup[kPrefillCubePipelineStartupBytes] = {};
  const aclError cleared =
      aclrtMemcpy(workspace_gm, kPrefillCubePipelineStartupBytes, zeroed_startup,
                  kPrefillCubePipelineStartupBytes, ACL_MEMCPY_HOST_TO_DEVICE);
  if (cleared != ACL_SUCCESS) {
    detail::RecordAclFailure(kPipelineOp, "aclrtMemcpy(startup rendezvous)",
                             static_cast<int>(cleared));
    return Status::launch_error;
  }

  // dyn_ub_size is ZERO: this kernel has no SIMT VF, so all of its UB comes from TPipe's STATIC
  // region, and declaring dynamic bytes nobody addresses would reserve UB the static allocator
  // then could not use.
  dsv4_moe_prefill_cube_pipeline<<<tiling.block_dim, 0u, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(gate_packed_gm), detail::Gm(gate_scale_gm),
      detail::Gm(up_packed_gm), detail::Gm(up_scale_gm), detail::Gm(down_packed_gm),
      detail::Gm(down_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(workspace_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kPipelineOp, stream);
}

}  // namespace moe::kernels::arch35
