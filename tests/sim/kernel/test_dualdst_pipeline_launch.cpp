// SIM ONLY -- host launch wrapper for the STEP 2 three-stage pipeline probe.
// Compiled by BiSheng (CLAUDE.md sections 3.1-3.3): triple-chevron only.
//
// The stage selector travels in the status buffer, planted by the host before
// the launch (kPipeStageWord), for the same reason STEP 1 does it: the shape
// is a compile-time constant and a tiling POD for one uint32 is ceremony.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_prefill_cube_pipeline.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void test_dualdst_pipeline_kernel(
    GM_ADDR a_gm, GM_ADDR w_packed_gm, GM_ADDR w_scale_gm,
    GM_ADDR out_gm, GM_ADDR status_gm);

constexpr const char* kPipeOp = "test_dualdst_pipeline_kernel";

}  // namespace

Status LaunchDualDstPipelineProbe(const void* a_gm, const void* w_packed_gm,
                                  const void* w_scale_gm, void* out_gm,
                                  void* status_gm, aclrtStream stream) {
  if (a_gm == nullptr || w_packed_gm == nullptr || w_scale_gm == nullptr ||
      out_gm == nullptr || status_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }

  // One AI Core: one AIC and the two AIVs paired with it, which is the whole
  // topology the dual-destination split and the ring handshake live in.
  test_dualdst_pipeline_kernel<<<1u, 0u, stream>>>(
      detail::Gm(a_gm), detail::Gm(w_packed_gm), detail::Gm(w_scale_gm),
      detail::Gm(out_gm), detail::Gm(status_gm));

  return detail::FinishLaunch(kPipeOp, stream);
}

}  // namespace moe::kernels::arch35
