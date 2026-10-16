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

static_assert(kHidden % moe::kernels::arch35::kPrefillCubeKSub == 0u &&
                  kHidden % moe::kernels::arch35::kPrefillCubeHSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub2 == 0u,
              "the axes must be whole multiples of every sub-tile extent or "
              "MakeDsv4MoePrefillCubeTiling rejects the shape and this run "
              "never reaches the kernel");

constexpr float  kAbsTol      = 1.0e-2F;
constexpr float  kRelTol      = 1.0e-2F;
constexpr double kCosineFloor = 0.999;

// Generous on purpose: it converts an unbalanced handshake from an infinite
// wait into exit 3 with a name, and is not a performance threshold. The
// measured dequant micro-run needed ~275 s almost entirely in simulator
// bring-up, and this run adds a KfcServer and two Matmul objects to that.
constexpr int kSingleTileBudgetSeconds = 2700;

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
      static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
      kTotalTokens, kMTile, kSimAiv);

  const DumpCensus before = ExceptionDumps();

  try {
    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w3 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w2 = MakeBank(kHidden, kInter, kExperts, rng);

    const std::size_t y_floats =
        static_cast<std::size_t>(kTotalTokens) * kHidden;
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

    auto yDev = MakeGuardedOutput(y_floats);

    // KFC reserve, zeroed, then 0x5A past it. The reserve is the larger of
    // what the operator declares and 64 KiB per launched AIV, so a sub-core
    // queue cannot run off the end of the allocation; the poison band behind
    // it is what turns a spill into a named failure instead of a silent
    // corruption of whatever GM happened to follow.
    const std::uint64_t declared =
        Dsv4MoePrefillCubeWorkspaceBytes(kTotalTokens, kHidden, kInter);
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
        static_cast<const std::int32_t*>(routeDev.get()), kHidden, kInter,
        kExperts, kTotalTokens, kSimAiv, static_cast<float*>(yDev->get()),
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
    RequireGuardIntact(*yDev, y_floats, "single tile");
    std::printf("  guard band: intact (%zu floats past the %zu-float output)\n",
                kGuardFloats, y_floats);

    std::vector<float> y(y_floats, 0.0F);
    CopyFromDevice(y, *yDev);

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
                diff.max_abs, diff.worst, diff.worst / kHidden,
                diff.worst % kHidden, diff.max_rel, diff.cosine);
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
        kTotalTokens, kHidden, static_cast<int>(kExpert),
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
