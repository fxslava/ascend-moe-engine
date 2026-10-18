// SIM ONLY -- host launch wrapper for the STEP 1 Fixpipe dual-destination
// probe. Compiled by BiSheng (CLAUDE.md sections 3.1-3.3): the triple-chevron
// form only, no ACLRT_LAUNCH_KERNEL.
//
// There is no tiling blob. The probe's shape is a compile-time constant shared
// through kernel/dsv4_prefill_cube_dualdst.h, because a tiling POD would be
// one more thing that could be wrong while answering a question that is not
// about tiling.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_prefill_cube_dualdst.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void test_fixpipe_dualdst_kernel(
    GM_ADDR a_gm, GM_ADDR b_gm, GM_ADDR out_gm, GM_ADDR status_gm);

constexpr const char* kDualDstOp = "test_fixpipe_dualdst_kernel";

}  // namespace

Status LaunchFixpipeDualDstProbe(const void* a_gm, const void* b_gm, void* out_gm,
                                 void* status_gm, aclrtStream stream) {
  // The stage selector travels in the status buffer's last word, planted by
  // the host before the launch -- see kDualDstStageWord. It is not a separate
  // operand because a four-pointer ABI is the whole interface and a tiling
  // blob for one uint32 would be ceremony.
  if (a_gm == nullptr || b_gm == nullptr || out_gm == nullptr ||
      status_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }

  // One AI Core: one AIC and the two AIVs paired with it. That is the whole
  // topology the dual-destination split lives in, so a wider grid would add
  // cores with nothing to do and nothing to check.
  test_fixpipe_dualdst_kernel<<<1u, 0u, stream>>>(
      detail::Gm(a_gm), detail::Gm(b_gm), detail::Gm(out_gm), detail::Gm(status_gm));

  return detail::FinishLaunch(kDualDstOp, stream);
}

}  // namespace moe::kernels::arch35
