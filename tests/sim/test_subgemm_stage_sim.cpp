/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// VARIANT (b), STAGED: walks the delta between the passing lifecycle-only
// probe and the never-retiring full kernel, one routine at a time.
//
// See tests/sim/kernel/test_subgemm_stage_kernel.cpp for the stage ladder.
// Usage: test_subgemm_stage_sim <stage>   (default 7 = everything)
//
// It reports RETIRED or trips a 180 s box. The box is defensible here for the
// same reason it was on variant (a): the lifecycle alone cost 20,000 simulated
// cycles and 41 s, and one sub-GEMM adds a single dequant and a single Cube
// step -- not the 660,000+ cycles the full schedule was still climbing through
// when its 900 s box expired. It remains a TIME BOX, not a deadlock verdict.

#include "dsv4_moe_block_sim_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>
#include <cstdlib>

namespace moe::kernels::arch35 {
Status LaunchSubgemmStageProbe(const Mxfp4Pair* w_packed_gm,
                               const E8M0* w_scale_gm, std::int32_t* status_gm,
                               void* workspace_gm, std::uint32_t hidden,
                               std::uint32_t inter, std::uint32_t num_experts,
                               std::uint32_t total_tokens,
                               std::uint32_t aiv_count, std::uint32_t stage,
                               aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

constexpr std::uint32_t kProbeHidden = 128;
constexpr std::uint32_t kProbeInter  = 256;
constexpr std::uint32_t kProbeTokens = moe::kernels::arch35::kPrefillCubeMTile;
constexpr std::uint32_t kProbeAiv    = 2;
// Budget is a PARAMETER, not a constant, because the stage ladder's cost is
// linear in the replay count and a box sized for one trip silently converts a
// slow stage into a fake hang. MEASURED: ~40 s fixed bring-up plus ~57 s per
// replayed sub-GEMM (94.1 s at 1 rep, 151.6 s at 2). Pass it explicitly when
// the stage is expected to cost more.
int g_probe_budget = 180;

const char* StageName(std::uint32_t s) {
  switch (s) {
    case 1: return "REGIST_MATMUL_OBJ + UB plane (== variant a)";
    case 2: return "+ AsyncLoadWeightsSubTile (GM->UB, SetFlag<MTE2_V>)";
    case 3: return "+ DequantSubTileVector (WaitFlag<MTE2_V>, Cast/Mul in UB)";
    case 4: return "+ DequantSubTileToL1 (V_MTE3, UB->L1 TSCM copy, MTE3_V)";
    case 5: return "+ EnQue/DeQue + SetTensorA/B + SetOrgShape/SetTail";
    case 6: return "+ Iterate / GetTensorC (Cube step, Fixpipe->VECIN)";
    case 7: return "+ End() and the FP32 accumulate";
    case 8: return "stage 7 REPLAYED 2x (first TSCM ring slot reuse, 2x End())";
    case 9: return "stage 7 REPLAYED 4x (ring wraps twice)";
    default: return "(unknown)";
  }
}

int RunStage(std::uint32_t stage) {
  using namespace moe::kernels::arch35;

  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::printf(
      "dsv4_moe_prefill_cube -- VARIANT (b) STAGE %u\n  %s\n"
      "hidden %u inter %u tokens %u aiv %u budget %d s\n",
      stage, StageName(stage), kProbeHidden, kProbeInter, kProbeTokens,
      kProbeAiv, g_probe_budget);

  const DumpCensus before = ExceptionDumps();

  try {
    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(kProbeInter, kProbeHidden, kExperts, rng);

    Runtime runtime;
    LaunchWatchdog watchdog(g_probe_budget, "[SUBGEMM-STAGE]");

    DeviceBuffer wp(w1.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(wp, w1.packed);
    DeviceBuffer ws(w1.scales.size() * sizeof(E8M0));
    CopyToDevice(ws, w1.scales);

    const std::uint64_t declared =
        Dsv4MoePrefillCubeWorkspaceBytes(kProbeTokens, kProbeHidden, kProbeInter);
    const std::uint64_t kfc = std::max<std::uint64_t>(
        declared, static_cast<std::uint64_t>(kProbeAiv) * 64ull * 1024ull);
    DeviceBuffer wsDev(static_cast<std::size_t>(kfc + (1ull << 20)));
    {
      std::vector<std::uint8_t> init(static_cast<std::size_t>(kfc + (1ull << 20)), 0x5Au);
      std::fill(init.begin(), init.begin() + static_cast<std::ptrdiff_t>(kfc), 0u);
      CopyToDevice(wsDev, init);
    }

    std::vector<std::int32_t> status(8, -1);
    DeviceBuffer statusDev(status.size() * sizeof(std::int32_t));
    CopyToDevice(statusDev, status);

    watchdog.Arm("LaunchSubgemmStageProbe");
    const auto t0 = std::chrono::steady_clock::now();
    const Status st = LaunchSubgemmStageProbe(
        static_cast<const Mxfp4Pair*>(wp.get()), static_cast<const E8M0*>(ws.get()),
        static_cast<std::int32_t*>(statusDev.get()), wsDev.get(), kProbeHidden,
        kProbeInter, kExperts, kProbeTokens, kProbeAiv, stage, runtime.stream());
    const auto t1 = std::chrono::steady_clock::now();
    watchdog.Disarm();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::printf("  RETIRED in %.1f s\n", secs);
    if (st != Status::success) {
      Fail(LaunchFailure("LaunchSubgemmStageProbe", st));
    }
    CopyFromDevice(status, statusDev);
    std::printf("  liveness word: %d (expected %u)\n", status[0], stage);
    if (status[0] != static_cast<std::int32_t>(stage)) {
      Fail("the launch returned but the kernel body did not complete its stage");
    }

    runtime.Close();
    const DumpCensus after = ExceptionDumps();
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf("\n[PASS] stage %u retires in %.1f s. Says nothing about any "
                "computed value. Never run on hardware.\n", stage, secs);
    return 0;
  } catch (const std::exception& e) {
    std::printf("\n[FAIL] %s\n", e.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t stage = 7;
  if (argc > 1) {
    stage = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  }
  if (argc > 2) {
    g_probe_budget = static_cast<int>(std::strtol(argv[2], nullptr, 10));
  }
  if (stage < 1u || stage > 9u) {
    std::printf("stage must be 1..9\n");
    return 2;
  }
  return RunStage(stage);
}
