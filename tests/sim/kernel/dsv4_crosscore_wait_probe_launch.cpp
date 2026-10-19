/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- host launch wrapper for the cross-core wait probe.
//
// Deliberately the SMALLEST launch that still produces the production
// topology: block_dim 1, which under KERNEL_TYPE_MIX_AIC_1_2 is one Cube core
// and its two paired vector cores -- the exact grid the measurement that
// motivates this probe came from. No tiling struct is staged because the kernel
// reads no geometry; the one GM argument exists only so the kernel has a
// non-null pointer to check, matching how every other entry point here starts.
//
// Compiled by BiSheng as an ASC TU (CLAUDE.md sections 3.2-3.3): the
// triple-chevron form is the only dispatch and BiSheng is what lowers it.

#include "kernel_operator.h"

#include "launch_common.hpp"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_crosscore_wait_probe(GM_ADDR scratch);

constexpr const char* kProbeOp = "dsv4_crosscore_wait_probe";

}  // namespace

Status LaunchDsv4CrossCoreWaitProbe(void* scratch_gm, aclrtStream stream) {
  if (scratch_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }

  dsv4_crosscore_wait_probe<<<1u, 0u, stream>>>(detail::Gm(scratch_gm));

  return detail::FinishLaunch(kProbeOp, stream);
}

}  // namespace moe::kernels::arch35
