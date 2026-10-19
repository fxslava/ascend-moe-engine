/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Driver for the cross-core wait probe. See
// tests/sim/kernel/dsv4_crosscore_wait_probe_kernel.cpp for the question and
// the measurement that raised it.
//
// THE CHANNEL IS TERMINATION, NOT OUTPUT. Run it at N = 1 and then N = 2
// (argv[1]); the reading is whether the kernel RETURNS:
//
//     N=1 returns, N=2 hangs   EXACTLY ONE free clear per flag id
//                              -> residual state at kernel entry
//     N=1 and N=2 both return  the mode-2 wait does not block at all
//     N=1 hangs                the wait blocks correctly, and the cycle-stamped
//                              trace that motivated this probe needs another
//                              explanation
//
// WHY NOT COUNT PRINTED RECORDS. The first version of this driver did, and it
// could not: MEASURED, at N = 4 the run emitted NOT ONE device record before
// the watchdog ended it, because the debug ring buffer is drained on fill or at
// kernel end and a four-record kernel that then blocks never flushes. Absence
// of records was indistinguishable from "the wait blocked" -- the exact
// distinction being measured. Termination needs no flush.
//
// A watchdog kill (exit 3) is therefore an EXPECTED outcome and not a defect in
// the probe. This driver asserts nothing about which outcome occurs; baking in
// an expected answer is how a probe stops being evidence.
//
// The budget is small on purpose: the kernel issues a handful of waits and
// nothing else, so all of its cost is simulator bring-up. If it has not
// resolved in two minutes, it is blocked.

#include "dsv4_moe_block_sim_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace moe::kernels::arch35 {
Status LaunchDsv4CrossCoreWaitProbe(void* scratch_gm, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

constexpr int kProbeBudgetSeconds = 120;

int RunProbe(std::uint32_t control) {
  using namespace moe::kernels::arch35;

  const bool handshake_mode = (control == 0xFFFFFFFFu);
  const std::uint32_t waits = handshake_mode ? 0u : control;

  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // Deliberately NOT quoting the kernel's own record text here. The first
  // version of this banner contained the literal string the result was grepped
  // for, and the grep matched the banner instead of a device record -- the
  // driver fed its own instructions back as data.
  std::printf(
      "dsv4_crosscore_wait_probe\n"
      "  mode: %s\n"
      "  grid: block_dim 1 (MIX_AIC_1_2 = 1 AIC + 2 AIV), the topology the "
      "measurement came from\n"
      "  THE READING IS WHETHER THIS PROCESS RETURNS (exit 0) OR BLOCKS "
      "(exit 3).\n",
      handshake_mode
          ? "GM HANDSHAKE ALONE -- both AIVs write and flush their word, the "
            "Cube spins until it observes both"
          : "UNMATCHED CROSS-CORE WAITS on the operator's own flag id, mode "
            "and pipe; nothing ever sets it");

  const DumpCensus before = ExceptionDumps();

  try {
    Runtime runtime;
    LaunchWatchdog watchdog(kProbeBudgetSeconds, "[CROSSCORE-WAIT-PROBE]");

    // One page is plenty: the kernel reads the control word from offset 0 and
    //, in handshake mode, two ready words at 256 and 384. ZEROED, which the
    // handshake depends on -- the Cube spins until both words are nonzero, so a
    // stale nonzero byte would release it immediately and the probe would
    // report success without having tested anything.
    DeviceBuffer scratch(4096);
    {
      std::vector<std::uint8_t> init(4096, 0u);
      std::memcpy(init.data(), &control, sizeof(control));
      CopyToDevice(scratch, init);
    }

    watchdog.Arm("LaunchDsv4CrossCoreWaitProbe");
    const auto started = std::chrono::steady_clock::now();
    const Status status = LaunchDsv4CrossCoreWaitProbe(scratch.get(), runtime.stream());
    const auto finished = std::chrono::steady_clock::now();
    watchdog.Disarm();
    const double seconds = std::chrono::duration<double>(finished - started).count();

    std::printf("  launch returned %s after %.1f s\n",
                ToString(status), seconds);
    if (status != Status::success) {
      // A launch-path failure IS a failure -- it means the probe never ran and
      // produced no reading at all.
      Fail(LaunchFailure("LaunchDsv4CrossCoreWaitProbe", status));
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("  excp_log*.dump: %zu before, %zu after (%zu non-empty)\n",
                before.files, after.files, after.non_empty);

    // The flag id is NOT named here: it lives in the operator-private UB
    // header, which a test may not include (CLAUDE.md section 1.2). The kernel
    // prints it on the runs that flush.
    if (handshake_mode) {
      std::printf(
          "\n[DONE] RETURNED in GM-handshake mode: the Cube OBSERVED both AIVs' "
          "GM writes. The working form is a COMPILER BARRIER around the dcci "
          "plus a volatile __gm__ read; dcci alone with GlobalTensor::GetValue "
          "deadlocks, because the load is hoisted out of the spin. Wall %.1f "
          "s.\n",
          seconds);
    } else {
      std::printf(
          "\n[DONE] RETURNED with waits = %u, so all %u mode-2 waits cleared "
          "with no matching CrossCoreSetFlag. Wall %.1f s.\n",
          waits, waits, seconds);
    }
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}

}  // namespace

// argv[1]: a wait count (MODE 0), or "gm" to run the GM handshake alone
// (MODE 1). MODE 1 answers whether DataCacheCleanAndInvalid plus GetValue makes
// an AIV's GM write visible to the Cube -- the primitive the operator's new
// startup rendezvous is built on, and the prime suspect for the deadlock that
// replaced the flag collisions.
int main(int argc, char** argv) {
  std::uint32_t control = 1;
  if (argc > 1) {
    if (std::strcmp(argv[1], "gm") == 0) {
      control = 0xFFFFFFFFu;
    } else {
      const long value = std::strtol(argv[1], nullptr, 10);
      if (value > 0) control = static_cast<std::uint32_t>(value);
    }
  }
  return RunProbe(control);
}
