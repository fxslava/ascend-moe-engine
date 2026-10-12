// -----------------------------------------------------------------------------
// device_cases_stub.cpp -- the no-launch-layer build of RunDeviceCases().
//
// Compiled instead of device_cases.cpp when MOE_KERNEL_LAUNCH_LAYER_AVAILABLE
// is not defined, i.e. in any build without BiSheng and CANN. Reports every
// device comparison as skipped, with the reason, so an absent device path can
// never be mistaken for a pass.
// -----------------------------------------------------------------------------

#include "device_cases.hpp"

#include <string>

namespace moe::kerntest {

void RunDeviceCases(Report& rep) {
  section("device: kernel output vs in-memory golden");

  const char* why =
      "built without the arch35 device launch layer. Configure with "
      "-DENABLE_ARCH35_DEVICE_LAUNCH=ON inside a CANN 9.2 environment "
      "(BiSheng required) to compile the device cases; they then skip at "
      "runtime unless a physical NPU is present.";

  for (const char* k : {"fused_mhc_pre_norm_sinkhorn",
                        "fused_mhc_post_residual",
                        "paged_compressor_rope",
                        "fast_fp8_indexer_topk",
                        "mxfp4_expert_gemv"}) {
    rep.skip(std::string("device ") + k, why);
  }
}

}  // namespace moe::kerntest
