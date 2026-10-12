#pragma once
// Device comparison cases for test_custom_kernels_verification.
//
// Two implementations exist and exactly one is compiled in:
//   device_cases.cpp       when MOE_KERNEL_LAUNCH_LAYER_AVAILABLE is defined
//                          (requires BiSheng + CANN; links libascendcl)
//   device_cases_stub.cpp  otherwise -- reports every case as skipped
//
// Either way the harness calls RunDeviceCases() unconditionally, so the
// host-only build and the device build print the same section.

#include "kernel_test_harness.hpp"

namespace moe::kerntest {

void RunDeviceCases(Report& rep);

}  // namespace moe::kerntest
