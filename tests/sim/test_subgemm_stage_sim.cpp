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
#include "kernel/dsv4_prefill_cube_ub_census.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

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

// Decodes the UB physical address census the probe writes. See
// kernel/dsv4_prefill_cube_ub_census.h for the wire format and the reason it
// is read rather than computed.
//
// Nothing here FAILS the run: the census is a measurement, and the thing it
// measures is exactly the thing that is currently wrong. Printing it under a
// failing stage is the point.
void PrintUbCensus(const std::vector<std::int32_t>& status) {
  using namespace moe::kernels::arch35;

  const auto word = [&status](std::uint32_t i) -> std::uint32_t {
    return static_cast<std::uint32_t>(status[static_cast<std::size_t>(i)]);
  };

  if (word(1) != kUbCensusMagic) {
    std::printf("  UB census: ABSENT (word 1 = 0x%08X, expected 0x%08X)\n",
                word(1), kUbCensusMagic);
    return;
  }
  const std::uint32_t entries = word(2);
  if (entries != kUbCensusTotalEntries) {
    std::printf("  UB census: TORN (%u entries, expected %u)\n", entries,
                kUbCensusTotalEntries);
    return;
  }

  std::vector<std::uint64_t> addr(entries);
  for (std::uint32_t i = 0; i < entries; ++i) {
    addr[i] = static_cast<std::uint64_t>(word(kUbCensusHeaderWords + 2u * i)) |
              (static_cast<std::uint64_t>(word(kUbCensusHeaderWords + 2u * i + 1u))
               << 32);
  }

  const std::uint64_t base = addr[0];
  std::printf(
      "\n  UB PHYSICAL ADDRESS CENSUS (GetPhyAddr, allocation order)\n"
      "  TPipe dynamic pool base = %llu B (0x%llX)\n"
      "  physical UB             = %u B (0x%X)\n"
      "  dynamic pool available  = %lld B\n"
      "  %-24s %10s %10s %10s  %s\n",
      static_cast<unsigned long long>(base),
      static_cast<unsigned long long>(base), kUbCensusPhysicalUbBytes,
      kUbCensusPhysicalUbBytes,
      static_cast<long long>(kUbCensusPhysicalUbBytes) -
          static_cast<long long>(base),
      "buffer", "addr", "0x", "size", "");

  // Sizes are not transmitted: TPipe allocates in InitBuffer order, so each
  // UB buffer is bounded by the next one's address. The last UB entry has no
  // successor, so its size is left blank rather than guessed.
  for (std::uint32_t i = 0; i < kUbCensusUbEntries; ++i) {
    const bool has_next = (i + 1u) < kUbCensusUbEntries;
    const long long size =
        has_next ? static_cast<long long>(addr[i + 1u]) -
                       static_cast<long long>(addr[i])
                 : -1;
    const bool past_end = addr[i] >= kUbCensusPhysicalUbBytes;
    char size_text[24];
    if (size >= 0) {
      std::snprintf(size_text, sizeof(size_text), "%lld", size);
    } else {
      std::snprintf(size_text, sizeof(size_text), "%s", "-");
    }
    std::printf("  %-24s %10llu 0x%08llX %10s  %s\n", kUbCensusNames[i],
                static_cast<unsigned long long>(addr[i]),
                static_cast<unsigned long long>(addr[i]), size_text,
                past_end ? "<== PAST PHYSICAL UB" : "");
  }

  // The first address past the last UB allocation. scale_temporary is
  // 16 uint16 = 32 B, the one size the census cannot derive from a successor.
  constexpr std::uint64_t kScaleTemporaryBytes = 32;
  const std::uint64_t ub_end =
      addr[kUbCensusUbEntries - 1u] + kScaleTemporaryBytes;

  // The ring slots. Whether these are UB or L1 is the open question the
  // census exists to close, so the verdict is computed from the addresses
  // rather than asserted: contiguous from ub_end means they share the UB pool.
  bool ring_in_ub = true;
  for (std::uint32_t i = 0; i < kUbCensusTscmSlots; ++i) {
    const std::uint32_t e = kUbCensusTscmEntry + i;
    const std::uint64_t a = addr[e];
    if (a == ~static_cast<std::uint64_t>(0)) {
      std::printf("  %-24s %10s %10s %10s  %s\n", kUbCensusNames[e], "-", "-",
                  "-", "(not recorded)");
      ring_in_ub = false;
      continue;
    }
    const long long size =
        (i + 1u < kUbCensusTscmSlots &&
         addr[e + 1u] != ~static_cast<std::uint64_t>(0))
            ? static_cast<long long>(addr[e + 1u]) - static_cast<long long>(a)
            : -1;
    char size_text[24];
    if (size >= 0) {
      std::snprintf(size_text, sizeof(size_text), "%lld", size);
    } else {
      std::snprintf(size_text, sizeof(size_text), "%s", "-");
    }
    std::printf("  %-24s %10llu 0x%08llX %10s  %s\n", kUbCensusNames[e],
                static_cast<unsigned long long>(a),
                static_cast<unsigned long long>(a), size_text,
                a >= kUbCensusPhysicalUbBytes ? "<== PAST PHYSICAL UB" : "");
    if (a < ub_end) {
      ring_in_ub = false;  // below the UB high-water mark: a separate pool
    }
  }

  // THE NUMBER THAT MATTERS. If the ring shares the UB pool, the budget is
  // everything up to the last ring slot's end, not just the sixteen UB
  // buffers -- and the sixteen alone would understate it by 131,072 B.
  constexpr std::uint64_t kRingSlotBytes = 32768;
  const std::uint64_t used_end =
      ring_in_ub ? addr[kUbCensusTotalEntries - 1u] + kRingSlotBytes : ub_end;
  std::printf(
      "  16 UB buffers           = %llu B (base .. %llu)\n"
      "  TSCM ring slots         = %s\n"
      "  UB pool high-water mark = %llu B\n"
      "  headroom to physical UB = %lld B%s\n\n",
      static_cast<unsigned long long>(ub_end - base),
      static_cast<unsigned long long>(ub_end),
      ring_in_ub ? "IN THE UB POOL, contiguous from the UB end"
                 : "a separate pool (L1), as the headers say",
      static_cast<unsigned long long>(used_end),
      static_cast<long long>(kUbCensusPhysicalUbBytes) -
          static_cast<long long>(used_end),
      used_end > kUbCensusPhysicalUbBytes ? "   *** OVERFLOWS ***" : "");
}

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

    std::vector<std::int32_t> status(kUbCensusStatusWords, -1);
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
    PrintUbCensus(status);
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
