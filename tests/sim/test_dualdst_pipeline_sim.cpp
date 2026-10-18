/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- STEP 2: does the three-stage pipeline accumulate a whole K axis
// in L0C and move it out once, correctly, to both AIVs?
//
// See kernel/dsv4_prefill_cube_pipeline.h for the schedule. This binary
// checks that the answer is EXACT: the operands are chosen so every product
// and every partial sum is an integer below 256, which BF16 holds without
// rounding, so a single rounding anywhere -- including the per-K-block cast
// that a UB-accumulator schedule would be forced into -- shows up as a
// mismatch rather than as drift inside a tolerance.

#include "dsv4_moe_block_sim_common.hpp"

#include "kernel/dsv4_prefill_cube_pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace moe::kernels::arch35 {
Status LaunchDualDstPipelineProbe(const void* a_gm, const void* w_packed_gm,
                                  const void* w_scale_gm, void* out_gm,
                                  void* status_gm, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

using namespace moe::kernels::arch35;

int g_budget = 600;
std::uint32_t g_stage = kPipeStagePrefetch;

std::uint16_t Bf16(float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0xFFFFu) != 0u) {
    Fail("a probe operand is not exactly representable in BF16");
  }
  return static_cast<std::uint16_t>(bits >> 16);
}
float FromBf16(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

// A IS A SELECTOR, WHICH MAKES THE OUTPUT A DIRECT READOUT OF B.
//
//   A[tile][m][c] = (c == m) ? 1 : 0
//     => C[m][n] = sum_k A[m][k] * B[k][n] = sum_over_tiles B_tile[m][n]
//
// So at one tile the result IS the B operand's first M rows, and a layout
// mistake in the UB -> L1 -> L0B hand-off is readable off the numbers instead
// of being inferred. W is position-dependent in BOTH axes so an n <-> k swap
// cannot hide:
//
//   W[k][n] = ((n + 2 * k + k / K_SUB) % 4) + 1        values 1..4
//
// The multiplier on k must NOT be odd-and-congruent-to-1 mod 4: a first
// attempt used (n * 5 + k) % 4, and since 5 == 1 (mod 4) that is symmetric in
// n and k, so it matched the transposed and untransposed models identically
// and could not tell them apart.
//
//   correct   (B = W^T) : C[m][n] = W[n][m] = ((n + 2 * m) % 4) + 1
//   untransposed        : C[m][n] = W[m][n] = ((m + 2 * n) % 4) + 1
//
// 1..4 are all exact OCP E2M1 codes at scale 2^0, so the MXFP4 round trip is
// lossless and every value is exact in BF16.
float AValue(std::uint32_t m, std::uint32_t c) {
  return (c == m) ? 1.0F : 0.0F;
}
// W is [K][N], contraction-major -- see the header for why.
//
// THE `kk / kPipeKSub` TERM IS THE TILE INDEX, AND WITHOUT IT THIS PROBE
// CANNOT LOCALIZE AN ACCUMULATION FAILURE.
//
// The term was absent, leaving W[kk][n] = ((n + 2*kk) % 4) + 1. Tile t row m
// is kk = t * K_SUB + m, and 2 * t * K_SUB = 256 * t is a multiple of 4, so
// EVERY TILE CONTRIBUTED THE IDENTICAL VALUE and the 4-tile sum was just
// 4x one tile. That is enough to notice that an accumulation is wrong -- it
// reported `got 1 want 4` -- but not to say WHICH tile survived, or whether
// one survived at all versus three being skipped. A degeneracy in the operand
// that hides the thing being measured is exactly what CLAUDE.md section 4.8
// is about, and it cost several cycle-accurate runs here.
//
// With the term, tile t contributes ((n + 2*m + t) % 4) + 1, the four tiles
// contribute the four distinct residues, and the sum is 1+2+3+4 = 10 at EVERY
// (m, n). So:
//
//   got == 10         accumulated correctly
//   got == t + 1      at (m, n) = (0, 0): only tile t landed
//
// Still exact: the values are 1..4, all exact OCP E2M1 codes at scale 2^0, and
// 10 is exact in BF16.
float WValue(std::uint32_t kk, std::uint32_t n) {
  return static_cast<float>(((n + 2u * kk + kk / kPipeKSub) % 4u) + 1u);
}
// B[k][n] = W[k][n] directly, so with the selector A:
//   C[m][n] = sum over tiles of W[t * K_SUB + m][n]
float CValue(std::uint32_t m, std::uint32_t n, std::uint32_t tiles) {
  float acc = 0.0F;
  for (std::uint32_t t = 0; t < tiles; ++t) {
    acc += WValue(t * kPipeKSub + m, n);
  }
  return acc;
}
// The layout mistake STEP 2 chased down, kept as the diagnostic's control:
// if the logical matrix in L1 were [N, K] again, the readout would match
// this instead.
float CSwapped(std::uint32_t m, std::uint32_t n, std::uint32_t tiles) {
  float acc = 0.0F;
  for (std::uint32_t t = 0; t < tiles; ++t) {
    acc += WValue(t * kPipeKSub + n, m);
  }
  return acc;
}

const char* StageName(std::uint32_t s) {
  switch (s) {
    case kPipeStageOneTile:  return "one tile: plumbing and epilogue only";
    case kPipeStageAccum:    return "all tiles, ring depth 2, no MTE2 prefetch";
    case kPipeStagePrefetch: return "all tiles, full three-stage overlap";
    default: return "(unknown)";
  }
}

int Run() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::uint32_t tiles = (g_stage == kPipeStageOneTile) ? 1u : kPipeTiles;
  std::printf(
      "dsv4 three-stage dual-destination pipeline -- STEP 2\n"
      "  M %u (split %u/%u)  N %u  K_SUB %u  tiles %u   budget %d s\n"
      "  stage %u (%s)\n",
      kPipeM, kPipeMHalf, kPipeMHalf, kPipeN, kPipeKSub, tiles, g_budget,
      g_stage, StageName(g_stage));

  const DumpCensus before = ExceptionDumps();

  try {
    // A, tile-major [tiles][M][K_SUB].
    std::vector<std::uint16_t> a(kPipeAElems);
    for (std::uint32_t t = 0; t < kPipeTiles; ++t) {
      for (std::uint32_t m = 0; m < kPipeM; ++m) {
        for (std::uint32_t c = 0; c < kPipeKSub; ++c) {
          a[(t * kPipeM + m) * kPipeKSub + c] = Bf16(AValue(m, c));
        }
      }
    }

    // W as MXFP4 [N][K], scale code 127 == 2^0 so the nibble is the value.
    std::vector<Mxfp4Pair> packed(kPipeWPackedBytes);
    std::vector<E8M0> wscale(kPipeWScaleBytes, static_cast<E8M0>(127));
    for (std::uint32_t kk = 0; kk < kPipeWRows; ++kk) {
      for (std::uint32_t n = 0; n < kPipeWCols; n += 2u) {
        const std::uint8_t lo = EncodeFp4(WValue(kk, n));
        const std::uint8_t hi = EncodeFp4(WValue(kk, n + 1u));
        packed[kk * (kPipeWCols / 2u) + n / 2u] =
            static_cast<Mxfp4Pair>(lo | (hi << 4u));
      }
    }

    Runtime runtime;
    LaunchWatchdog watchdog(g_budget, "[DUALDST-PIPE]");

    DeviceBuffer aDev(a.size() * sizeof(std::uint16_t));
    CopyToDevice(aDev, a);
    DeviceBuffer wpDev(packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(wpDev, packed);
    DeviceBuffer wsDev(wscale.size() * sizeof(E8M0));
    CopyToDevice(wsDev, wscale);

    std::vector<std::uint16_t> out(kPipeOutElems, 0xDEADu);
    DeviceBuffer outDev(out.size() * sizeof(std::uint16_t));
    CopyToDevice(outDev, out);

    std::vector<std::uint32_t> status(kPipeStatusWords, 0xFFFFFFFFu);
    status[kPipeStageWord] = g_stage;
    DeviceBuffer stDev(status.size() * sizeof(std::uint32_t));
    CopyToDevice(stDev, status);

    watchdog.Arm("LaunchDualDstPipelineProbe");
    const auto t0 = std::chrono::steady_clock::now();
    const Status st = LaunchDualDstPipelineProbe(aDev.get(), wpDev.get(), wsDev.get(),
                                                 outDev.get(), stDev.get(),
                                                 runtime.stream());
    const auto t1 = std::chrono::steady_clock::now();
    watchdog.Disarm();
    std::printf("  RETIRED in %.1f s\n",
                std::chrono::duration<double>(t1 - t0).count());
    if (st != Status::success) {
      Fail(LaunchFailure("LaunchDualDstPipelineProbe", st));
    }

    CopyFromDevice(status, stDev);
    CopyFromDevice(out, outDev);

    const auto at = [&status](std::uint32_t r, std::uint32_t s) {
      return status[r * kPipeStatusStride + s];
    };
    std::printf("  AIC liveness 0x%08X (expect 0x%08X), tiles %u\n",
                at(kPipeAicRegion, kPipeSlotLive), kPipeAicDone,
                at(kPipeAicRegion, kPipeSlotTiles));
    std::printf("  AIV0 0x%08X  AIV1 0x%08X (expect 0x%08X)\n",
                at(kPipeAivRegion, kPipeSlotLive),
                at(kPipeAivRegion + 1u, kPipeSlotLive), kPipeAivDone);
    std::printf("  landing UB addr: AIV0 %u  AIV1 %u\n",
                at(kPipeAivRegion, kPipeSlotAddr),
                at(kPipeAivRegion + 1u, kPipeSlotAddr));

    if (at(kPipeAicRegion, kPipeSlotLive) != kPipeAicDone) {
      Fail("the AIC did not finish its K loop and epilogue");
    }
    if (at(kPipeAivRegion, kPipeSlotLive) != kPipeAivDone ||
        at(kPipeAivRegion + 1u, kPipeSlotLive) != kPipeAivDone) {
      Fail("an AIV did not finish -- check the ring handshake");
    }
    if (at(kPipeAivRegion, kPipeSlotAddr) != 0u ||
        at(kPipeAivRegion + 1u, kPipeSlotAddr) != 0u) {
      Fail("an AIV landing zone is not at UB offset 0");
    }

    std::uint32_t bad = 0;
    for (std::uint32_t half = 0; half < 2u; ++half) {
      for (std::uint32_t r = 0; r < kPipeMHalf; ++r) {
        const std::uint32_t m = half * kPipeMHalf + r;
        for (std::uint32_t n = 0; n < kPipeN; ++n) {
          const std::uint16_t got = out[half * kPipeHalfElems + r * kPipeN + n];
          const std::uint16_t want = Bf16(CValue(m, n, tiles));
          if (got != want) {
            if (bad < 8u) {
              std::printf("    MISMATCH half %u row %u (m=%u) col %u: got %g want %g\n",
                          half, r, m, n, static_cast<double>(FromBf16(got)),
                          static_cast<double>(FromBf16(want)));
            }
            ++bad;
          }
        }
      }
    }
    std::printf("  half 0 row 0 col 0 = %g (expect %g)\n",
                static_cast<double>(FromBf16(out[0])),
                static_cast<double>(CValue(0, 0, tiles)));
    std::printf("  half 1 row 0 col 0 = %g (expect %g)\n",
                static_cast<double>(FromBf16(out[kPipeHalfElems])),
                static_cast<double>(CValue(kPipeMHalf, 0, tiles)));
    if (bad != 0u) {
      std::printf("  %u of %u elements wrong\n", bad, kPipeOutElems);
      // WHAT IS ACTUALLY IN L0B. The selector A makes C a readout of B, so
      // this grid names the layout rather than just reporting a mismatch.
      std::printf("\n  C[m][n] readout -- got / want(B=W) / if-axes-swapped\n");
      for (std::uint32_t m = 0; m < 4u; ++m) {
        std::printf("   m=%u ", m);
        for (std::uint32_t n = 0; n < 6u; ++n) {
          std::printf(" %4g/%4g/%4g", static_cast<double>(FromBf16(out[m * kPipeN + n])),
                      static_cast<double>(CValue(m, n, tiles)),
                      static_cast<double>(CSwapped(m, n, tiles)));
        }
        std::printf("\n");
      }
      std::uint32_t untrans_hits = 0;
      for (std::uint32_t m = 0; m < kPipeMHalf; ++m) {
        for (std::uint32_t n = 0; n < kPipeN; ++n) {
          if (out[m * kPipeN + n] == Bf16(CSwapped(m, n, tiles))) ++untrans_hits;
        }
      }
      std::printf("  half 0 matches the AXES-SWAPPED model in %u of %u cells\n",
                  untrans_hits, kPipeMHalf * kPipeN);
      Fail("the pipelined accumulation did not produce the expected product");
    }

    runtime.Close();
    const DumpCensus after = ExceptionDumps();
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf(
        "\n[PASS] %u K tiles accumulated in L0C, ONE Fixpipe epilogue, rows "
        "0..%u to SUB BLOCK0 and %u..%u to SUB BLOCK1, bf16, exactly. "
        "Never run on hardware.\n",
        tiles, kPipeMHalf - 1u, kPipeMHalf, kPipeM - 1u);
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
  if (g_stage < kPipeStageOneTile || g_stage > kPipeStagePrefetch) {
    std::printf("usage: %s [stage 1|2|3] [budget seconds]\n", argv[0]);
    return 2;
  }
  return Run();
}
