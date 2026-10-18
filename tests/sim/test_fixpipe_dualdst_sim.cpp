/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- STEP 1: does Fixpipe DUAL_DST_SPLIT_M with an in-line
// float -> bfloat16_t cast deliver rows 0..7 to AIV0's UB and rows 8..15 to
// AIV1's UB?
//
// See kernel/dsv4_prefill_cube_dualdst.h for the data path, the test matrices
// and why the Matmul framework cannot answer this question. This binary
// checks three things and fails on any of them:
//
//   1. both halves arrive, EXACTLY -- the operands are chosen so every
//      product is integral and below 256, which BF16 represents without
//      rounding, so this is an equality check and not a tolerance;
//   2. the halves are the RIGHT halves and in the right order, which is what
//      DUAL_DST_SPLIT_M is being asked to do;
//   3. the landing zone really is at UB offset 0 on both AIVs, which is the
//      assumption the AIC's Fixpipe destination address rests on.
//
// It is a device-behaviour probe under CAModel, never run on hardware, and it
// says nothing about the production schedule.

#include "dsv4_moe_block_sim_common.hpp"

#include "kernel/dsv4_prefill_cube_dualdst.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace moe::kernels::arch35 {
Status LaunchFixpipeDualDstProbe(const void* a_gm, const void* b_gm, void* out_gm,
                                 void* status_gm, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

using namespace moe::kernels::arch35;

int g_budget = 300;
std::uint32_t g_stage = kDualDstStageRendezvous;

// BF16 as the device stores it: the top 16 bits of the FP32 word. Every value
// this probe produces is an integer below 256, so the low 16 bits are zero and
// this is exact rather than a rounding -- see the header. Asserted below
// rather than assumed.
std::uint16_t Bf16(float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0xFFFFu) != 0u) {
    Fail("the probe's operands must be exactly representable in BF16");
  }
  return static_cast<std::uint16_t>(bits >> 16);
}

float FromBf16(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

// A[m][k] = (k == 0) ? (m + 1) : 0
float AValue(std::uint32_t m, std::uint32_t k) {
  return (k == 0u) ? static_cast<float>(m + 1u) : 0.0F;
}
// B[k][n] = (k == 0) ? ((n % 8) + 1) : 0
float BValue(std::uint32_t k, std::uint32_t n) {
  return (k == 0u) ? static_cast<float>((n % 8u) + 1u) : 0.0F;
}
// C[m][n] = (m + 1) * ((n % 8) + 1)
float CValue(std::uint32_t m, std::uint32_t n) {
  return static_cast<float>((m + 1u) * ((n % 8u) + 1u));
}

int Run() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::printf(
      "dsv4 Fixpipe DUAL_DST_SPLIT_M probe -- STEP 1\n"
      "  M %u (split %u/%u)  N %u  K %u   budget %d s\n"
      "  path: GM -> L1 -> L0A/L0B -> Mmad -> L0C -> UB (bf16, dual dst)\n"
      "  stage %u (%s)\n",
      kDualDstM, kDualDstMHalf, kDualDstMHalf, kDualDstN, kDualDstK, g_budget,
      g_stage,
      g_stage == kDualDstStageAicOnly    ? "AIC Cube chain only, no wait"
      : g_stage == kDualDstStageFull      ? "full, AIV waits with default template args"
      : g_stage == kDualDstStageWaitMode2 ? "full, AIV waits with explicit <2, PIPE_MTE3>"
                                          : "full, plus the AIV -> AIC rendezvous first");

  const DumpCensus before = ExceptionDumps();

  try {
    std::vector<std::uint16_t> a(kDualDstM * kDualDstK);
    for (std::uint32_t m = 0; m < kDualDstM; ++m) {
      for (std::uint32_t k = 0; k < kDualDstK; ++k) {
        a[m * kDualDstK + k] = Bf16(AValue(m, k));
      }
    }
    std::vector<std::uint16_t> b(kDualDstK * kDualDstN);
    for (std::uint32_t k = 0; k < kDualDstK; ++k) {
      for (std::uint32_t n = 0; n < kDualDstN; ++n) {
        b[k * kDualDstN + n] = Bf16(BValue(k, n));
      }
    }

    Runtime runtime;
    LaunchWatchdog watchdog(g_budget, "[FIXPIPE-DUALDST]");

    DeviceBuffer aDev(a.size() * sizeof(std::uint16_t));
    CopyToDevice(aDev, a);
    DeviceBuffer bDev(b.size() * sizeof(std::uint16_t));
    CopyToDevice(bDev, b);

    // Poisoned, so a half that is never written is a failure rather than a
    // buffer of plausible zeros.
    std::vector<std::uint16_t> out(kDualDstOutElems, 0xDEADu);
    DeviceBuffer outDev(out.size() * sizeof(std::uint16_t));
    CopyToDevice(outDev, out);

    std::vector<std::uint32_t> status(kDualDstStatusWords, 0xFFFFFFFFu);
    status[kDualDstStageWord] = g_stage;
    DeviceBuffer statusDev(status.size() * sizeof(std::uint32_t));
    CopyToDevice(statusDev, status);

    watchdog.Arm("LaunchFixpipeDualDstProbe");
    const auto t0 = std::chrono::steady_clock::now();
    const Status st = LaunchFixpipeDualDstProbe(aDev.get(), bDev.get(), outDev.get(),
                                                statusDev.get(), runtime.stream());
    const auto t1 = std::chrono::steady_clock::now();
    watchdog.Disarm();
    std::printf("  RETIRED in %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());
    if (st != Status::success) {
      Fail(LaunchFailure("LaunchFixpipeDualDstProbe", st));
    }

    CopyFromDevice(status, statusDev);
    CopyFromDevice(out, outDev);

    const auto at = [&status](std::uint32_t region, std::uint32_t slot) {
      return status[region * kDualDstStatusStride + slot];
    };
    const std::uint32_t aic = at(kDualDstAicRegion, kDualDstSlotLive);
    const std::uint32_t live0 = at(kDualDstAivRegion + 0u, kDualDstSlotLive);
    const std::uint32_t live1 = at(kDualDstAivRegion + 1u, kDualDstSlotLive);
    const std::uint32_t addr0 = at(kDualDstAivRegion + 0u, kDualDstSlotAddr);
    const std::uint32_t addr1 = at(kDualDstAivRegion + 1u, kDualDstSlotAddr);

    std::printf("  AIC liveness  0x%08X (expect 0x%08X)\n", aic, kDualDstAicDone);
    std::printf("  AIV0 liveness 0x%08X   AIV1 liveness 0x%08X (expect 0x%08X)\n",
                live0, live1, kDualDstAivDone);
    std::printf("  landing zone UB addr: AIV0 %u  AIV1 %u   GetSubBlockNum %u\n",
                addr0, addr1, at(kDualDstAivRegion, kDualDstSlotSubs));

    if (aic != kDualDstAicDone) {
      Fail("the AIC did not reach the end of its Cube chain");
    }
    if (live0 != kDualDstAivDone || live1 != kDualDstAivDone) {
      Fail("an AIV did not complete its copy-out -- check the cross-core flags");
    }
    if (addr0 != kDualDstUbOffset || addr1 != kDualDstUbOffset) {
      Fail("the AIV landing zone is not at the UB offset the AIC wrote to; "
           "the TPipe pool base moved and the Fixpipe destination is wrong");
    }

    if (g_stage == kDualDstStageAicOnly) {
      runtime.Close();
      std::printf(
          "\n[PASS] stage 1: the AIC Cube chain (GM -> L1 -> L0A/L0B -> Mmad "
          "-> L0C -> Fixpipe) retires. Says NOTHING about the delivered "
          "values -- run stage 2 for that. Never run on hardware.\n");
      return 0;
    }

    // --- the halves, exactly --------------------------------------------
    std::uint32_t bad = 0;
    for (std::uint32_t half = 0; half < 2u; ++half) {
      for (std::uint32_t r = 0; r < kDualDstMHalf; ++r) {
        const std::uint32_t m = half * kDualDstMHalf + r;
        for (std::uint32_t n = 0; n < kDualDstN; ++n) {
          const std::uint16_t got =
              out[half * kDualDstHalfElems + r * kDualDstN + n];
          const std::uint16_t want = Bf16(CValue(m, n));
          if (got != want) {
            if (bad < 8u) {
              std::printf(
                  "    MISMATCH half %u row %u (global m=%u) col %u: "
                  "got %g (0x%04X) want %g (0x%04X)\n",
                  half, r, m, n, static_cast<double>(FromBf16(got)), got,
                  static_cast<double>(FromBf16(want)), want);
            }
            ++bad;
          }
        }
      }
    }

    std::printf("  half 0 row 0 col 0 = %g (expect %g)\n",
                static_cast<double>(FromBf16(out[0])),
                static_cast<double>(CValue(0, 0)));
    std::printf("  half 1 row 0 col 0 = %g (expect %g)\n",
                static_cast<double>(FromBf16(out[kDualDstHalfElems])),
                static_cast<double>(CValue(kDualDstMHalf, 0)));

    if (bad != 0u) {
      std::printf("  %u of %u elements wrong\n", bad, kDualDstOutElems);
      Fail("the dual-destination split did not deliver the expected halves");
    }

    runtime.Close();
    const DumpCensus after = ExceptionDumps();
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf(
        "\n[PASS] DUAL_DST_SPLIT_M delivered rows 0..%u to SUB BLOCK0 and rows "
        "%u..%u to SUB BLOCK1, cast to BF16 in Fixpipe, exactly. "
        "Never run on hardware.\n",
        kDualDstMHalf - 1u, kDualDstMHalf, kDualDstM - 1u);
    return 0;
  } catch (const std::exception& e) {
    std::printf("\n[FAIL] %s\n", e.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    g_stage = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  }
  if (argc > 2) {
    g_budget = static_cast<int>(std::strtol(argv[2], nullptr, 10));
  }
  if (g_stage < kDualDstStageAicOnly || g_stage > kDualDstStageRendezvous) {
    std::printf("usage: %s [stage 1|2|3|4] [budget seconds]\n", argv[0]);
    return 2;
  }
  return Run();
}
