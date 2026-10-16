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

// SINGLE-TILE bring-up run for dsv4_moe_prefill_cube on the dav-3510 CAModel.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS SEPARATELY FROM THE INDUCTIVE GATE
// ---------------------------------------------------------------------------
// test_dsv4_moe_prefill_cube_inductive_sim is the real gate: 1020 tokens, a
// ragged tail, |S| = 6 sampled tiles out of 64, two launches at 2 and 4 AIVs.
// It has NEVER COMPLETED A TRIP -- under KERNEL_TYPE_MIX_AIC_1_2 the simulator
// floods pem_lsu with small-offset ldst faults and does not terminate -- and
// that gate cannot say which of its moving parts is responsible, because it
// stands up a KfcServer, two Matmul objects, an L1 weight ring, a UB
// activation plane, a grid stride across clusters and a ragged epilogue at
// once.
//
// This driver removes every one of those degrees of freedom that is not the
// Cube path itself:
//
//   total_tokens = 16 = EXACTLY ONE M_TILE.  So total_tiles == 1, the sample
//       set degenerates to S = {0} with no cap needed, block_dim comes out
//       min(1, aiv) == 1, there is no grid stride, no second tile to overlap
//       with, and no ragged `rows` clamp. ProcessTokenTile is entered once,
//       for tile 0, on one AIC cluster.
//   ONE launch, at aiv_count = 2, which under MIX_AIC_1_2 is one AIC driven
//       by its two paired AIVs -- the minimum topology in which this kernel
//       can run at all, since the Cube side is reached only through the KFC
//       message queue.
//
// The geometry is otherwise the gate's: hidden 512, inter 512, 4 experts,
// which is the smallest shape MakeDsv4MoePrefillCubeTiling admits under
// DSV4_TILING_TEST_OVERRIDE (512 is the least common multiple of R_SUB 128,
// K_SUB 128, H_SUB 64 and R_SUB2 256, so every sub-GEMM loop stays square).
// Every sub-GEMM loop therefore still executes more than once -- phase 1 runs
// 4 r-steps x 4 k-steps, phase 2 runs 8 h-steps x 2 r-steps -- so this is a
// reduced TRIP COUNT, not a reduced kernel.
//
// WHAT A PASS HERE LICENSES, AND WHAT IT DOES NOT
//   It licenses: one ProcessTokenTile trip executes on the dav-3510 CAModel
//   under MIX_AIC_1_2 and its 16x512 output matches the audited two-phase
//   Cube golden. That is strictly less than the inductive gate's claim -- it
//   says nothing about trip-to-trip state, the ragged tail, or the grid
//   stride across clusters, all of which the sample set exists to cover.
//   NEVER RUN ON HARDWARE (CLAUDE.md section 5.7).
//
// Tolerance is the gate's and comes from the FORMAT (section 4.7): the only
// lossy step in a W4A16 Cube path is the FP32 -> BF16 activation cast, which
// keeps an 8-bit mantissa and so carries up to 2^-9 relative error per store;
// weights MXFP4 -> BF16 and the E8M0 power-of-two group scale are both exact.
// Accumulated over a 512-long contraction and two chained phases, 1e-2
// absolute plus 1e-2 relative is the envelope, while cosine stays strict at
// 0.999 -- rounding moves magnitude, structural bugs move direction.

#include "dsv4_moe_block_sim_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>
#include <cstdlib>

namespace moe::kernels::arch35 {
Status LaunchDsv4MoePrefillCubeInductiveSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, std::uint32_t sample_cap,
    Dsv4MoePrefillCubeTilingData* tiling_out, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

constexpr std::uint32_t kMTile = moe::kernels::arch35::kPrefillCubeMTile;

// EXACTLY ONE TILE. The static_assert is the whole point of this driver: if
// M_TILE ever moves, this is not a single-tile run any more and the file must
// be re-derived rather than silently become a two-tile one.
constexpr std::uint32_t kTotalTokens = kMTile;
static_assert(kTotalTokens == 16u,
              "this driver is the 16-row single-tile bring-up; M_TILE moved");
static_assert(kTotalTokens % kMTile == 0u,
              "the single tile must be FULL -- a ragged tail is the inductive "
              "gate's job, not this one");

// MIX_AIC_1_2: one AIC cluster holds 2 AIVs, so 2 is the minimum topology.
constexpr std::uint32_t kSimAiv = 2;

// ---------------------------------------------------------------------------
// GEOMETRY: the axes' TRUE MINIMA, not the shared fixture's 512/512
// ---------------------------------------------------------------------------
// This driver overrides kSingleTileHidden/kSingleTileInter deliberately. The shared fixture uses
// 512 on both axes, which at M_TILE = 16 is 48 sub-GEMMs for ONE token tile
// (phase 1: inter/R_SUB * hidden/K_SUB * 2 = 32, phase 2: hidden/H_SUB *
// inter/R_SUB2 = 16). MEASURED: that did not finish inside 2700 s of CAModel
// wall time, reaching 1,610,000 cycles with zero faults -- about 596
// simulated cycles per second of host time.
//
// 128 x 256 is the smallest shape the SCHEDULE admits, derived per axis:
// hidden needs lcm(K_SUB 128, H_SUB 64) = 128, inter needs lcm(R_SUB 128,
// R_SUB2 256) = 256. That is 6 sub-GEMMs -- phase 1: 2 * 1 * 2, phase 2:
// 2 * 1 -- an 8x cut in Cube work.
//
// WHAT IS NOT GIVEN UP: the arithmetic. Every contraction still runs over its
// WHOLE axis, so this is the real two-phase operator on a smaller problem and
// the audited emu golden applies verbatim. That is the difference between
// shrinking the AXES and capping the sub-GEMM LOOPS -- the latter would leave
// phase 1 holding a partial sum of `act`, which is not the operator's answer
// and could not be compared against this golden at all.
//
// WHAT IS GIVEN UP: hidden/K_SUB == 1, so the token prefetch ring never
// ping-pongs here. It instead exercises the chunk_count == 1 edge of the
// V_MTE2 handshake, where the prologue waits for nothing and DrainTokenRing
// consumes exactly one token. The multi-chunk ring stays the 512-axis gate's
// job.
constexpr std::uint32_t kSingleTileHidden = 128;
constexpr std::uint32_t kSingleTileInter  = 256;

static_assert(kSingleTileHidden % moe::kernels::arch35::kPrefillCubeKSub == 0u &&
                  kSingleTileHidden % moe::kernels::arch35::kPrefillCubeHSub == 0u &&
                  kSingleTileInter % moe::kernels::arch35::kPrefillCubeRSub == 0u &&
                  kSingleTileInter % moe::kernels::arch35::kPrefillCubeRSub2 == 0u,
              "the axes must be whole multiples of every sub-tile extent or "
              "MakeDsv4MoePrefillCubeTiling rejects the shape and this run "
              "never reaches the kernel");

// The sub-GEMM census this run is expected to drive, stated so the cost of a
// change to it is visible rather than discovered as a timeout.
constexpr std::uint32_t kPhase1SubGemms =
    (kSingleTileInter / moe::kernels::arch35::kPrefillCubeRSub) *
    (kSingleTileHidden / moe::kernels::arch35::kPrefillCubeKSub) * 2u;
constexpr std::uint32_t kPhase2SubGemms =
    (kSingleTileHidden / moe::kernels::arch35::kPrefillCubeHSub) *
    (kSingleTileInter / moe::kernels::arch35::kPrefillCubeRSub2);
static_assert(kPhase1SubGemms + kPhase2SubGemms == 6u,
              "the sub-GEMM count moved; re-derive the watchdog budget before "
              "changing this, because the 512-axis shape's 48 sub-GEMMs did "
              "not finish in 2700 s");

constexpr float  kAbsTol      = 1.0e-2F;
constexpr float  kRelTol      = 1.0e-2F;
constexpr double kCosineFloor = 0.999;

// SIZED FROM A MEASUREMENT, and it is a TIME BOX, not a deadlock detector.
//
// The 512-axis shape ran 2700 s and reached 1,610,000 cycles: ~596 simulated
// cycles per second of host time. This shape has 1/8 the Cube work, so the
// same schedule should land near 200,000 cycles, or ~340 s, on top of the
// fixed simulator bring-up the dequant micro-gate measured at ~275 s. 900 s
// therefore carries roughly a 1.5x margin over the expected total.
//
// WHAT TRIPPING IT DOES AND DOES NOT MEAN. This is host WALL-CLOCK: it cannot
// tell a deadlock from a slow simulation, because a CAModel keeps advancing
// cycles while every core sits blocked on a flag. Exceeding it means "did not
// finish in the budget" and nothing more. To call a hang, compare the cycle
// count at two trip points and show it is not advancing, or show a WaitFlag
// with no matching SetFlag -- the budget alone is not evidence of either.
constexpr int kSingleTileBudgetSeconds = 900;

struct Spread {
  double rms = 0.0, stddev = 0.0, min = 0.0, max = 0.0;
};

Spread Measure(const std::vector<float>& v) {
  Spread s;
  if (v.empty()) return s;
  double sum = 0.0, sum_sq = 0.0;
  s.min = v[0];
  s.max = v[0];
  for (const float value : v) {
    const double x = value;
    sum += x;
    sum_sq += x * x;
    s.min = std::min(s.min, x);
    s.max = std::max(s.max, x);
  }
  const double n = static_cast<double>(v.size());
  const double mean = sum / n;
  const double mean_sq = sum_sq / n;
  s.rms = std::sqrt(mean_sq);
  s.stddev = std::sqrt(std::max(0.0, mean_sq - mean * mean));
  return s;
}

struct RowDiff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double cosine  = 1.0;
  bool   within  = true;
  bool   finite  = true;
  std::size_t worst = 0;
};

RowDiff Compare16(const std::vector<float>& got,
                  const std::vector<float>& want) {
  RowDiff d;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double a = got[i];
    const double b = want[i];
    if (!std::isfinite(a) || !std::isfinite(b)) {
      d.finite = false;
      d.within = false;
      continue;
    }
    const double delta = std::fabs(a - b);
    if (delta > d.max_abs) {
      d.max_abs = delta;
      d.worst = i;
    }
    if (std::fabs(b) > 1.0e-6) {
      d.max_rel = std::max(d.max_rel, delta / std::fabs(b));
    }
    if (delta > kAbsTol + kRelTol * std::fabs(b)) {
      d.within = false;
    }
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  d.cosine = (na == 0.0 && nb == 0.0)
                 ? 1.0
                 : ((na == 0.0 || nb == 0.0)
                        ? 0.0
                        : dot / (std::sqrt(na) * std::sqrt(nb)));
  return d;
}

// Reports how much of the output tile is STILL CANARY. A kernel that returned
// early -- the failure mode CLAUDE.md section 2.5 is about -- leaves a tile
// that is either untouched or zeroed, and both read as "wrong answer" unless
// the two are told apart explicitly.
struct Occupancy {
  std::size_t canary = 0;
  std::size_t zero   = 0;
  std::size_t live   = 0;
};

Occupancy MeasureOccupancy(const std::vector<float>& y) {
  Occupancy o;
  for (const float v : y) {
    if (IsCanary(v)) {
      ++o.canary;
    } else if (v == 0.0F) {
      ++o.zero;
    } else {
      ++o.live;
    }
  }
  return o;
}

std::string CheckWorkspaceUnwritten(const std::vector<std::uint8_t>& probe) {
  for (std::size_t i = 0; i < probe.size(); ++i) {
    if (probe[i] != 0x5Au) {
      return "workspace byte " + std::to_string(i) +
             " past the KFC reserve was written (0x" +
             std::to_string(static_cast<unsigned>(probe[i])) +
             "): the kernel spilled outside the designated KFC queues into GM";
    }
  }
  return {};
}

int RunSingleTile() {
  using namespace moe::kernels::arch35;

  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  std::printf(
      "dsv4_moe_prefill_cube -- SINGLE-TILE dav-3510 CAModel bring-up "
      "(MIX_AIC_1_2)\n"
      "seed 0x%llX  hidden %u  inter %u  experts %u  tokens %u  m_tile %u  "
      "tiles 1  aiv %u\n",
      static_cast<unsigned long long>(kSeed), kSingleTileHidden, kSingleTileInter, kExperts,
      kTotalTokens, kMTile, kSimAiv);

  const DumpCensus before = ExceptionDumps();

  try {
    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(kSingleTileInter, kSingleTileHidden, kExperts, rng);
    const ExpertBank w3 = MakeBank(kSingleTileInter, kSingleTileHidden, kExperts, rng);
    const ExpertBank w2 = MakeBank(kSingleTileHidden, kSingleTileInter, kExperts, rng);

    const std::size_t y_floats =
        static_cast<std::size_t>(kTotalTokens) * kSingleTileHidden;
    std::vector<float> x(y_floats);
    std::normal_distribution<float> draw(0.0F, 1.0F);
    for (float& v : x) v = draw(rng);

    // The single tile routes to ONE expert, and deliberately not expert 0:
    // expert 2 exercises a nonzero w*_packed_expert_stride, so a stride that
    // is computed wrong cannot be masked by a base-pointer read.
    constexpr std::int32_t kExpert = 2;
    static_assert(kExpert < static_cast<std::int32_t>(kExperts),
                  "the routed expert must exist in the bank");
    const std::vector<std::int32_t> route(kTotalTokens, kExpert);

    Runtime runtime;
    LaunchWatchdog watchdog(kSingleTileBudgetSeconds, "[PREFILL-CUBE-1TILE]");

    DeviceBuffer xDev(x.size() * sizeof(float));
    CopyToDevice(xDev, x);
    DeviceBuffer w1p(w1.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w1p, w1.packed);
    DeviceBuffer w1s(w1.scales.size() * sizeof(E8M0));
    CopyToDevice(w1s, w1.scales);
    DeviceBuffer w3p(w3.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w3p, w3.packed);
    DeviceBuffer w3s(w3.scales.size() * sizeof(E8M0));
    CopyToDevice(w3s, w3.scales);
    DeviceBuffer w2p(w2.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w2p, w2.packed);
    DeviceBuffer w2s(w2.scales.size() * sizeof(E8M0));
    CopyToDevice(w2s, w2.scales);
    DeviceBuffer routeDev(route.size() * sizeof(std::int32_t));
    CopyToDevice(routeDev, route);

    // THREE-ZONE OUTPUT: [pre 4 KiB canary][output][post 4 KiB canary].
    //
    // The shared fixture's MakeGuardedOutput poisons a 256-byte tail only,
    // which catches a store that runs off the END of y. It cannot catch one
    // that lands BEFORE the base -- a negative row index, or a GM base that is
    // short by a sub-tile -- because there is nothing poisoned there to
    // notice. Phase 2 stores at `token_row_start * hidden + hidden_base`, so
    // an index computed one sub-tile low writes below y and the post-only
    // guard reports success.
    //
    // 4 KiB per band rather than 256 B: the phase 2 burst is `valid_rows`
    // blocks of H_SUB floats with a per-row stride, so a mis-strided copy can
    // skip well past a 256-byte band and land in live memory with the guard
    // intact. 4 KiB covers 16 rows x 64 floats, which is one whole phase 2
    // output sub-tile.
    constexpr std::size_t kCanaryBandBytes  = 4096;
    constexpr std::size_t kCanaryBandFloats = kCanaryBandBytes / sizeof(float);
    const std::size_t y_total_floats = y_floats + 2u * kCanaryBandFloats;
    DeviceBuffer yDev(y_total_floats * sizeof(float));
    {
      const std::vector<float> poison(y_total_floats, CanaryFloat());
      CopyToDevice(yDev, poison);
    }
    // The kernel is handed the MIDDLE zone, so the pre band sits at negative
    // offsets from its y base exactly as a real underrun would find it.
    float* const y_device = static_cast<float*>(yDev.get()) + kCanaryBandFloats;
    std::printf("  output: [pre %zu B 0x5A][%zu floats][post %zu B 0x5A]\n",
                kCanaryBandBytes, y_floats, kCanaryBandBytes);

    // KFC reserve, zeroed, then 0x5A past it. The reserve is the larger of
    // what the operator declares and 64 KiB per launched AIV, so a sub-core
    // queue cannot run off the end of the allocation; the poison band behind
    // it is what turns a spill into a named failure instead of a silent
    // corruption of whatever GM happened to follow.
    const std::uint64_t declared =
        Dsv4MoePrefillCubeWorkspaceBytes(kTotalTokens, kSingleTileHidden, kSingleTileInter);
    const std::uint64_t kfc_reserve = std::max<std::uint64_t>(
        declared, static_cast<std::uint64_t>(kSimAiv) * 64ull * 1024ull);
    constexpr std::uint64_t kProbeBytes = 8ull << 20;
    DeviceBuffer wsDev(static_cast<std::size_t>(kfc_reserve + kProbeBytes));
    {
      std::vector<std::uint8_t> init(
          static_cast<std::size_t>(kfc_reserve + kProbeBytes), 0x5Au);
      std::fill(init.begin(),
                init.begin() + static_cast<std::ptrdiff_t>(kfc_reserve), 0u);
      CopyToDevice(wsDev, init);
    }
    std::printf(
        "  KFC reserve %llu KiB (declared %llu KiB, %u AIV x 64 KiB), then "
        "%llu MiB poisoned 0x5A\n",
        static_cast<unsigned long long>(kfc_reserve >> 10),
        static_cast<unsigned long long>(declared >> 10), kSimAiv,
        static_cast<unsigned long long>(kProbeBytes >> 20));

    // ----------------------------------------------------------------- golden
    // Computed BEFORE the launch and audited, so a device result is never the
    // thing that decides what the expected answer was.
    std::vector<float> want(y_floats, CanaryFloat());
    {
      using moe::emu::CubeTileGeometry;
      CubeTileGeometry geom;
      geom.m_tile      = static_cast<int>(kMTile);
      geom.r_tile      = static_cast<int>(kPrefillCubeRSub);
      geom.k_tile      = static_cast<int>(kPrefillCubeKSub);
      geom.h_tile      = static_cast<int>(kPrefillCubeHSub);
      geom.r_tile_down = static_cast<int>(kPrefillCubeRSub2);
      geom.buffer_depth = 2;

      std::vector<moe::emu::Mxfp4Matrix> v1, v3, v2;
      for (std::uint32_t e = 0; e < kExperts; ++e) {
        v1.push_back(w1.View(e));
        v3.push_back(w3.View(e));
        v2.push_back(w2.View(e));
      }
      std::vector<float> scratch;
      moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
          v1.data(), v3.data(), v2.data(), x.data(), route.data(),
          static_cast<int>(kTotalTokens), static_cast<int>(kExperts),
          want.data(), kLimit, geom, nullptr, &scratch);

      if (!NonDegenerate(want)) {
        Fail("the golden for tile 0 is degenerate");
      }
      // "still canary" and "correct" must be DISJOINT, or the occupancy
      // census below cannot distinguish an unwritten word from a right one.
      for (const float v : want) {
        if (IsCanary(v)) {
          Fail("a golden element collides with the 0x5A5A5A5A canary; the "
               "occupancy census would be unsound");
        }
      }
      const Spread s = Measure(want);
      std::printf("  golden: rms %.4f stddev %.4f min %.4f max %.4f (%zu "
                  "floats, expert %d)\n",
                  s.rms, s.stddev, s.min, s.max, want.size(),
                  static_cast<int>(kExpert));
    }

    // ----------------------------------------------------------------- launch
    Dsv4MoePrefillCubeTilingData tiling{};
    watchdog.Arm("single-tile LaunchDsv4MoePrefillCubeInductiveSim");
    const auto started = std::chrono::steady_clock::now();
    const Status status = LaunchDsv4MoePrefillCubeInductiveSim(
        static_cast<const float*>(xDev.get()),
        static_cast<const Mxfp4Pair*>(w1p.get()),
        static_cast<const E8M0*>(w1s.get()),
        static_cast<const Mxfp4Pair*>(w3p.get()),
        static_cast<const E8M0*>(w3s.get()),
        static_cast<const Mxfp4Pair*>(w2p.get()),
        static_cast<const E8M0*>(w2s.get()),
        static_cast<const std::int32_t*>(routeDev.get()), kSingleTileHidden, kSingleTileInter,
        kExperts, kTotalTokens, kSimAiv, y_device,
        wsDev.get(), /*sample_cap=*/0u, &tiling, runtime.stream());
    const auto finished = std::chrono::steady_clock::now();
    watchdog.Disarm();
    const double seconds =
        std::chrono::duration<double>(finished - started).count();

    std::printf("  elapsed %.1f s, block_dim %u, abi %u, m_tile %u\n", seconds,
                tiling.block_dim, tiling.abi_version, tiling.m_tile);
    if (status != Status::success) {
      Fail(LaunchFailure("LaunchDsv4MoePrefillCubeInductiveSim", status));
    }
    if (tiling.block_dim != 1u) {
      Fail("block_dim came out " + std::to_string(tiling.block_dim) +
           ", not 1: this is no longer a single-tile grid");
    }

    // ------------------------------------------------------------ inspection
    // Read the WHOLE allocation once, so both bands and the payload come from
    // the same snapshot and a late store cannot slip between two copies.
    std::vector<float> y_all(y_total_floats, 0.0F);
    CopyFromDevice(y_all, yDev);

    for (std::size_t i = 0; i < kCanaryBandFloats; ++i) {
      if (!IsCanary(y_all[i])) {
        Fail("PRE-canary band corrupted at float -" +
             std::to_string(kCanaryBandFloats - i) + " below the y base: the "
             "kernel stored BELOW its output. Check the phase 2 y row index "
             "and the GM base passed to tokens_out.");
      }
    }
    for (std::size_t i = 0; i < kCanaryBandFloats; ++i) {
      const std::size_t at = kCanaryBandFloats + y_floats + i;
      if (!IsCanary(y_all[at])) {
        Fail("POST-canary band corrupted at float +" + std::to_string(i) +
             " past the " + std::to_string(y_floats) +
             "-float output: the kernel stored past its output. Check the "
             "phase 2 tile ring's y row index.");
      }
    }
    std::printf("  canary bands: both intact (%zu floats each side)\n",
                kCanaryBandFloats);

    const std::vector<float> y(y_all.begin() +
                                   static_cast<std::ptrdiff_t>(kCanaryBandFloats),
                               y_all.begin() + static_cast<std::ptrdiff_t>(
                                                   kCanaryBandFloats + y_floats));

    const Occupancy occ = MeasureOccupancy(y);
    std::printf("  occupancy: %zu live, %zu exactly zero, %zu STILL CANARY "
                "of %zu\n",
                occ.live, occ.zero, occ.canary, y.size());
    if (occ.canary != 0) {
      Fail(std::to_string(occ.canary) + " of " + std::to_string(y.size()) +
           " output words were never written (still 0x5A5A5A5A): the kernel "
           "did not cover its own tile");
    }

    {
      std::vector<std::uint8_t> probe(static_cast<std::size_t>(kProbeBytes), 0u);
      RequireAcl(aclrtMemcpy(probe.data(), probe.size(),
                             static_cast<const std::uint8_t*>(wsDev.get()) +
                                 kfc_reserve,
                             probe.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy(workspace probe past the KFC reserve)");
      const std::string ws = CheckWorkspaceUnwritten(probe);
      if (!ws.empty()) Fail(ws);
      std::printf("  workspace: all %llu MiB past the KFC reserve still 0x5A\n",
                  static_cast<unsigned long long>(kProbeBytes >> 20));
    }

    for (std::size_t i = 0; i < 4; ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &y[i], sizeof(bits));
      std::printf("  output[%zu]: actual %.9g (0x%08x), want %.9g\n", i, y[i],
                  bits, want[i]);
    }

    const RowDiff diff = Compare16(y, want);
    std::printf("  diff: max abs %.6e at [%zu] (row %zu col %zu), max rel "
                "%.6e, cosine %.8f\n",
                diff.max_abs, diff.worst, diff.worst / kSingleTileHidden,
                diff.worst % kSingleTileHidden, diff.max_rel, diff.cosine);
    if (!diff.finite) Fail("the device produced a non-finite value");
    if (!diff.within) {
      Fail("max abs " + std::to_string(diff.max_abs) + " / max rel " +
           std::to_string(diff.max_rel) + " exceeds the " +
           std::to_string(kAbsTol) + " + " + std::to_string(kRelTol) +
           "*|want| envelope");
    }
    if (diff.cosine < kCosineFloor) {
      Fail("cosine " + std::to_string(diff.cosine) + " is below the " +
           std::to_string(kCosineFloor) + " floor");
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("  excp_log*.dump: %zu before, %zu after (%zu non-empty)\n",
                before.files, after.files, after.non_empty);
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf(
        "\n[PASS] ONE ProcessTokenTile trip (tile 0, %u rows x %u, expert %d) "
        "executes on the dav-3510 CAModel in MIX_AIC_1_2 at block_dim 1 and "
        "matches the two-phase Cube golden inside %g + %g*|want| with cosine "
        "%.8f >= %g. Wall %.1f s. Never run on hardware.\n",
        kTotalTokens, kSingleTileHidden, static_cast<int>(kExpert),
        static_cast<double>(kAbsTol), static_cast<double>(kRelTol),
        diff.cosine, kCosineFloor, seconds);
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}

}  // namespace

int main() { return RunSingleTile(); }
