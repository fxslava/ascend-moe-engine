/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// VARIANT (a) of the dsv4_moe_prefill_cube bisection: does the Matmul/KFC
// LIFECYCLE retire, with no compute at all?
//
// See tests/sim/kernel/test_kfc_lifecycle_kernel.cpp for what is kept and what
// is removed, and for what each outcome licenses. In short:
//
//   exit 0  -- REGIST_MATMUL_OBJ, the KfcServer, both KfcCommClients and the
//              QUIT handshake all complete in this build and launch
//              configuration. The hang is then in a compute stage, and the
//              bisection moves to variant (b).
//   exit 3  -- the lifecycle itself does not complete. That would explain why
//              three schedule-level fixes changed nothing, and would localize
//              the defect to the Matmul framework's cross-core machinery
//              rather than to anything this operator computes.
//
// THE WATCHDOG IS 180 s AND THAT IS DEFENSIBLE HERE, unlike on the compute
// shapes. There is no work to simulate: the dequant micro-gate measured the
// fixed simulator bring-up at ~275 s of wall time for ~37,000 simulated
// cycles, and this kernel issues no DataCopy, no Cast and no Iterate at all.
// So anything past bring-up is the handshake, and 180 s is a box around the
// handshake rather than around a workload. It still is NOT a deadlock verdict
// -- a CAModel advances cycles while cores sit blocked -- so a trip is
// reported as "did not retire", and the cycle count is printed so it can be
// compared against the compute runs' trajectories.

#include "dsv4_moe_block_sim_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>

namespace moe::kernels::arch35 {
Status LaunchKfcLifecycleProbe(void* workspace_gm, std::int32_t* status_gm,
                               std::uint32_t hidden, std::uint32_t inter,
                               std::uint32_t num_experts,
                               std::uint32_t total_tokens,
                               std::uint32_t aiv_count, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

// The same reduced shape the compute runs use, so the tiling the Cube is
// initialized with is the one under investigation.
constexpr std::uint32_t kProbeHidden = 128;
constexpr std::uint32_t kProbeInter  = 256;
constexpr std::uint32_t kProbeTokens = moe::kernels::arch35::kPrefillCubeMTile;
constexpr std::uint32_t kProbeAiv    = 2;

constexpr int kProbeBudgetSeconds = 180;

int RunProbe() {
  using namespace moe::kernels::arch35;

  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::printf(
      "dsv4_moe_prefill_cube -- VARIANT (a): KFC LIFECYCLE ONLY, no compute\n"
      "hidden %u  inter %u  experts %u  tokens %u  aiv %u  budget %d s\n",
      kProbeHidden, kProbeInter, kExperts, kProbeTokens, kProbeAiv,
      kProbeBudgetSeconds);

  const DumpCensus before = ExceptionDumps();

  try {
    Runtime runtime;
    LaunchWatchdog watchdog(kProbeBudgetSeconds, "[KFC-LIFECYCLE]");

    const std::uint64_t declared =
        Dsv4MoePrefillCubeWorkspaceBytes(kProbeTokens, kProbeHidden, kProbeInter);
    const std::uint64_t kfc_reserve = std::max<std::uint64_t>(
        declared, static_cast<std::uint64_t>(kProbeAiv) * 64ull * 1024ull);
    constexpr std::uint64_t kProbeTailBytes = 1ull << 20;
    DeviceBuffer wsDev(static_cast<std::size_t>(kfc_reserve + kProbeTailBytes));
    {
      std::vector<std::uint8_t> init(
          static_cast<std::size_t>(kfc_reserve + kProbeTailBytes), 0x5Au);
      std::fill(init.begin(),
                init.begin() + static_cast<std::ptrdiff_t>(kfc_reserve), 0u);
      CopyToDevice(wsDev, init);
    }
    std::printf("  KFC reserve %llu KiB, zeroed; %llu MiB of 0x5A behind it\n",
                static_cast<unsigned long long>(kfc_reserve >> 10),
                static_cast<unsigned long long>(kProbeTailBytes >> 20));

    // Liveness word, poisoned so "kernel body was entered" is provable rather
    // than assumed. Zero would be ambiguous with a never-written buffer.
    std::vector<std::int32_t> status_host(8, -1);
    DeviceBuffer statusDev(status_host.size() * sizeof(std::int32_t));
    CopyToDevice(statusDev, status_host);

    watchdog.Arm("LaunchKfcLifecycleProbe");
    const auto started = std::chrono::steady_clock::now();
    const Status status = LaunchKfcLifecycleProbe(
        wsDev.get(), static_cast<std::int32_t*>(statusDev.get()), kProbeHidden,
        kProbeInter, kExperts, kProbeTokens, kProbeAiv, runtime.stream());
    const auto finished = std::chrono::steady_clock::now();
    watchdog.Disarm();
    const double seconds =
        std::chrono::duration<double>(finished - started).count();

    std::printf("  RETIRED in %.1f s\n", seconds);
    if (status != Status::success) {
      Fail(LaunchFailure("LaunchKfcLifecycleProbe", status));
    }

    CopyFromDevice(status_host, statusDev);
    std::printf("  liveness word: %d (1 = the kernel body ran on the driving "
                "sub-core, -1 = never written)\n",
                status_host[0]);
    if (status_host[0] != 1) {
      Fail("the launch returned success but the kernel body never wrote its "
           "liveness word: the dispatch did not reach the driving sub-core");
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("  excp_log*.dump: %zu before, %zu after (%zu non-empty)\n",
                before.files, after.files, after.non_empty);
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf(
        "\n[PASS] the Matmul/KFC LIFECYCLE retires on the dav-3510 CAModel in "
        "MIX_AIC_1_2 at block_dim 1: REGIST_MATMUL_OBJ with two GM "
        "TCubeTilings, the KfcServer, both KfcCommClients and the QUIT "
        "handshake, in %.1f s. This says NOTHING about any computed value -- "
        "no compute instruction was issued. The hang therefore lies in a "
        "compute stage; proceed to variant (b). Never run on hardware.\n",
        seconds);
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}

}  // namespace

int main() { return RunProbe(); }
