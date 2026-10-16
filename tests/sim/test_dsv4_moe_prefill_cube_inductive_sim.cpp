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

// Collapsed boundary-grid CAModel gate for dsv4_moe_prefill_cube -- the MIX
// AIC+AIV Cube prefill path (KERNEL_TYPE_MIX_AIC_1_2).
//
// Runs tests/sim/kernel/dsv4_moe_prefill_cube_inductive_kernel.cpp: the
// production ProcessTokenTile body, over the boundary tiles of the token grid
// instead of all of them, on the dav-3510 cycle-accurate model.
//
// WHAT THIS COVERS THAT NOTHING ELSE DOES
//   * tests/emu/test_dsv4_moe_prefill_cube_emu audits the W4A16 oracle on the
//     host, in milliseconds, without executing any device code.
//   * tests/sim/test_dsv4_moe_prefill_cube_sample_set_sim audits the collapsed
//     schedule's index arithmetic, also on the host.
//   * the BiSheng dav-3510 gate answers "does it compile".
//   * THIS gate is the only thing on a workstation with no NPU that EXECUTES
//     the path: the KfcServer handshake between the AIC and paired AIVs, the
//     Matmul framework's L0A/L0B/L0C staging, C landing at TPosition::VECIN,
//     A taken from VECOUT and B taken from TPosition::TSCM -- so neither the
//     intermediate nor a dequantized weight round-trips through GM -- the
//     per-sub-tile MXFP4 -> BF16 dequantization through UB and its UB -> L1
//     push, and the `act` plane surviving the phase 1 / phase 2 boundary in
//     UB. Then it compares the result against the audited golden.
//
// ---------------------------------------------------------------------------
// THE GRID IN MIX_AIC_1_2 TOPOLOGY
// ---------------------------------------------------------------------------
// Two launches, BOTH over the full four-expert routing:
//
//   A. 1 AIC cluster (2 AIVs). Primary gate: one AIC driven by two paired
//      AIVs with sub-core KFC workspace isolation, every sampled tile verified.
//   B. 2 AIC clusters (4 AIVs). Two independent AICs with 4 total AIVs, testing
//      the grid-stride map across clusters with cores on DIFFERENT experts.

#include "dsv4_moe_block_sim_common.hpp"

#include "kernel/dsv4_moe_prefill_cube_sample_set.h"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>
#include <cstdlib>
#include <set>

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

using moe::sim::prefill_cube::BuildTileSampleSet;
using moe::sim::prefill_cube::kMaxSamples;
using moe::sim::prefill_cube::kRingDepth;

static_assert(kHidden % moe::kernels::arch35::kPrefillCubeKSub == 0u &&
                  kHidden % moe::kernels::arch35::kPrefillCubeHSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub2 == 0u,
              "the sim fixture's axes must be whole multiples of every "
              "sub-tile extent, or MakeDsv4MoePrefillCubeTiling rejects the "
              "shape and this gate never reaches the kernel");
constexpr std::uint32_t kMTile = moe::kernels::arch35::kPrefillCubeMTile;

constexpr std::uint32_t kTotalTokens = 1020;
constexpr std::uint32_t kTileCount   = (kTotalTokens + kMTile - 1u) / kMTile;
static_assert(kTotalTokens % kMTile != 0, "the last tile must be ragged");
static_assert(kTileCount == 64u, "the sample set is pinned at N = 64");

constexpr std::uint32_t kTilesPerExpert = kTileCount / 4u;

// В режиме KERNEL_TYPE_MIX_AIC_1_2 один кластер AIC содержит 2 AIV.
// Соответственно: 1 кластер = 2 AIV, 2 кластера = 4 AIV.
constexpr std::uint32_t kSimAivSingle = 2;
constexpr std::uint32_t kSimAivMulti  = 4;

constexpr float  kAbsTol      = 1.0e-2F;
constexpr float  kRelTol      = 1.0e-2F;
constexpr double kCosineFloor = 0.999;

constexpr int kPrefillLaunchBudgetSeconds = 3600;

struct Spread {
  double rms    = 0.0;
  double stddev = 0.0;
  double min    = 0.0;
  double max    = 0.0;
};

Spread MeasureSampled(const std::vector<float>& v) {
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

std::vector<std::uint32_t> SampleSet(std::uint32_t tiles) {
  std::uint32_t raw[kMaxSamples] = {};
  const std::uint32_t n = BuildTileSampleSet(tiles, kRingDepth, raw);
  return std::vector<std::uint32_t>(raw, raw + n);
}

std::string Join(const std::vector<std::uint32_t>& v) {
  std::string out = "{";
  for (std::size_t i = 0; i < v.size(); ++i) {
    out += (i ? ", " : "") + std::to_string(v[i]);
  }
  return out + "}";
}

struct RowDiff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double cosine  = 1.0;
  bool   within  = true;
  bool   finite  = true;
};

RowDiff CompareOnSamples(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         const std::vector<std::uint32_t>& samples,
                         std::uint32_t hidden, std::uint32_t m_tile,
                         std::uint32_t total_tokens) {
  RowDiff d;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (const std::uint32_t tile : samples) {
    const std::uint64_t row0 = static_cast<std::uint64_t>(tile) * m_tile;
    const std::uint32_t rows =
        (total_tokens - row0) < m_tile
            ? static_cast<std::uint32_t>(total_tokens - row0)
            : m_tile;
    for (std::uint32_t r = 0; r < rows; ++r) {
      for (std::uint32_t c = 0; c < hidden; ++c) {
        const std::size_t i =
            static_cast<std::size_t>(row0 + r) * hidden + c;
        const double got  = actual[i];
        const double want = expected[i];
        if (!std::isfinite(got) || !std::isfinite(want)) {
          d.finite = false;
          d.within = false;
          continue;
        }
        const double delta = std::fabs(got - want);
        d.max_abs = std::max(d.max_abs, delta);
        if (std::fabs(want) > 1.0e-6) {
          d.max_rel = std::max(d.max_rel, delta / std::fabs(want));
        }
        if (delta > kAbsTol + kRelTol * std::fabs(want)) {
          d.within = false;
        }
        dot += got * want;
        na += got * got;
        nb += want * want;
      }
    }
  }
  d.cosine = (na == 0.0 && nb == 0.0)
                 ? 1.0
                 : ((na == 0.0 || nb == 0.0) ? 0.0
                                             : dot / (std::sqrt(na) * std::sqrt(nb)));
  return d;
}

std::string CheckCanaryOutsideSamples(const std::vector<float>& y,
                                      const std::vector<std::uint32_t>& samples,
                                      std::uint32_t hidden,
                                      std::uint32_t m_tile,
                                      std::uint32_t total_tokens) {
  std::set<std::uint32_t> sampled(samples.begin(), samples.end());
  for (std::uint32_t row = 0; row < total_tokens; ++row) {
    if (sampled.count(row / m_tile) != 0u) {
      continue;
    }
    for (std::uint32_t c = 0; c < hidden; ++c) {
      const std::size_t i = static_cast<std::size_t>(row) * hidden + c;
      if (!IsCanary(y[i])) {
        return "y row " + std::to_string(row) + " (tile " +
               std::to_string(row / m_tile) +
               ", not in S) column " + std::to_string(c) +
               " was written: the kernel touched a tile it was not scheduled to run";
      }
    }
  }
  return {};
}

std::string CheckWorkspaceUnwritten(const std::vector<std::uint8_t>& probe) {
  for (std::size_t i = 0; i < probe.size(); ++i) {
    if (probe[i] != 0x5Au) {
      return "workspace byte " + std::to_string(i) +
             " past the KFC reserve was written (0x" +
             std::to_string(static_cast<unsigned>(probe[i])) +
             "): the kernel wrote outside designated KFC queues into GM";
    }
  }
  return {};
}

void GoldenOnSamples(const ExpertBank& w1, const ExpertBank& w3,
                     const ExpertBank& w2, const std::vector<float>& x,
                     const std::vector<std::int32_t>& route,
                     const std::vector<std::uint32_t>& samples,
                     std::uint32_t hidden, std::uint32_t m_tile,
                     std::uint32_t total_tokens, std::vector<float>& y_out) {
  using moe::emu::CubeTileGeometry;
  using moe::kernels::arch35::kPrefillCubeHSub;
  using moe::kernels::arch35::kPrefillCubeKSub;
  using moe::kernels::arch35::kPrefillCubeRSub;
  using moe::kernels::arch35::kPrefillCubeRSub2;

  CubeTileGeometry geom;
  geom.m_tile       = static_cast<int>(m_tile);
  geom.r_tile       = static_cast<int>(kPrefillCubeRSub);
  geom.k_tile       = static_cast<int>(kPrefillCubeKSub);
  geom.h_tile       = static_cast<int>(kPrefillCubeHSub);
  geom.r_tile_down  = static_cast<int>(kPrefillCubeRSub2);
  geom.buffer_depth = 2;

  std::vector<moe::emu::Mxfp4Matrix> v1, v3, v2;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    v1.push_back(w1.View(e));
    v3.push_back(w3.View(e));
    v2.push_back(w2.View(e));
  }

  std::vector<float> scratch;
  for (const std::uint32_t tile : samples) {
    const std::uint64_t row0 = static_cast<std::uint64_t>(tile) * m_tile;
    const std::uint32_t rows =
        (total_tokens - row0) < m_tile
            ? static_cast<std::uint32_t>(total_tokens - row0)
            : m_tile;
    const std::int32_t expert = route[static_cast<std::size_t>(row0)];
    const std::vector<std::int32_t> tile_route(rows, expert);
    moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
        v1.data(), v3.data(), v2.data(),
        x.data() + static_cast<std::size_t>(row0) * hidden, tile_route.data(),
        static_cast<int>(rows), static_cast<int>(kExperts),
        y_out.data() + static_cast<std::size_t>(row0) * hidden, kLimit, geom,
        nullptr, &scratch);
  }
}

struct LaunchOutcome {
  double seconds = 0.0;
  bool ok = false;
};

LaunchOutcome RunOne(const char* label, Runtime& runtime,
                     LaunchWatchdog& watchdog, const ExpertBank& w1,
                     const ExpertBank& w3, const ExpertBank& w2,
                     const std::vector<float>& x,
                     const std::vector<std::int32_t>& route,
                     std::uint32_t aiv_count, std::uint32_t sample_cap,
                     DeviceBuffer& xDev, DeviceBuffer& w1p, DeviceBuffer& w1s,
                     DeviceBuffer& w3p, DeviceBuffer& w3s, DeviceBuffer& w2p,
                     DeviceBuffer& w2s, DeviceBuffer& routeDev) {
  using namespace moe::kernels::arch35;
  LaunchOutcome outcome;

  const std::size_t y_floats = static_cast<std::size_t>(kTotalTokens) * kHidden;
  auto yDev = MakeGuardedOutput(y_floats);

  // KFC Workspace расчет под топологию MIX_AIC_1_2:
  // Резерв учитывает все запущенные AIV ядра (минимум kKfcReserve)
  const std::uint64_t ws_bytes =
      Dsv4MoePrefillCubeWorkspaceBytes(kTotalTokens, kHidden, kInter);
  const std::uint64_t actual_kfc_reserve = std::max<std::uint64_t>(
      ws_bytes, static_cast<std::uint64_t>(aiv_count) * 64ull * 1024ull);

  constexpr std::uint64_t kProbeBytes = 8ull << 20;
  DeviceBuffer wsDev(static_cast<std::size_t>(actual_kfc_reserve + kProbeBytes));
  {
    std::vector<std::uint8_t> ws_init(
        static_cast<std::size_t>(actual_kfc_reserve + kProbeBytes), 0x5Au);
    std::fill(ws_init.begin(),
              ws_init.begin() + static_cast<std::ptrdiff_t>(actual_kfc_reserve), 0u);
    CopyToDevice(wsDev, ws_init);
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  std::printf("\n--- %s ---\n", label);
  std::printf("  aiv_count %u, sample cap %u (0 = all of S)\n", aiv_count, sample_cap);

  watchdog.Arm(std::string(label) + " LaunchDsv4MoePrefillCubeInductiveSim");
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
      kExperts, kTotalTokens, aiv_count,
       static_cast<float*>(yDev->get()), wsDev.get(),
       sample_cap, &tiling,
      runtime.stream());
  const auto finished = std::chrono::steady_clock::now();
  watchdog.Disarm();
  outcome.seconds = std::chrono::duration<double>(finished - started).count();

  std::printf("  elapsed %.1f s, block_dim %u, tiles %u\n", outcome.seconds,
              tiling.block_dim, kTileCount);

  if (status != Status::success) {
    Fail(LaunchFailure("LaunchDsv4MoePrefillCubeInductiveSim", status));
  }

  std::vector<std::uint32_t> samples = SampleSet(kTileCount);
  if (sample_cap != 0u && sample_cap < samples.size()) {
    samples.resize(sample_cap);
  }
  std::printf("  S = %s  (|S| = %zu of %u tiles, %.1fx collapse)\n",
              Join(samples).c_str(), samples.size(), kTileCount,
              static_cast<double>(kTileCount) / static_cast<double>(samples.size()));

  RequireGuardIntact(*yDev, y_floats, label);
  std::printf("  guard band: intact (%zu floats past the %zu-float output)\n",
              kGuardFloats, y_floats);

  std::vector<float> y(y_floats, 0.0F);
  CopyFromDevice(y, *yDev);

  const std::string canary =
      CheckCanaryOutsideSamples(y, samples, kHidden, kMTile, kTotalTokens);
  if (!canary.empty()) {
    Fail(std::string(label) + ": " + canary);
  }
  std::printf("  output canary: every row outside S still 0x5A5A5A5A\n");

  {
    std::vector<std::uint8_t> probe(static_cast<std::size_t>(kProbeBytes), 0u);
    RequireAcl(aclrtMemcpy(probe.data(), probe.size(),
                           static_cast<const std::uint8_t*>(wsDev.get()) +
                               actual_kfc_reserve,
                           probe.size(), ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(workspace probe past the KFC reserve)");
    const std::string ws = CheckWorkspaceUnwritten(probe);
    if (!ws.empty()) {
      Fail(std::string(label) + ": " + ws);
    }
    std::printf("  workspace: all %llu MiB past the KFC reserve still 0x5A\n",
                static_cast<unsigned long long>(kProbeBytes >> 20));
  }

  std::vector<float> want(y_floats, CanaryFloat());
  GoldenOnSamples(w1, w3, w2, x, route, samples, kHidden, kMTile, kTotalTokens, want);

  {
    std::vector<float> sampled_want;
    for (const std::uint32_t tile : samples) {
      const std::size_t row0 = static_cast<std::size_t>(tile) * kMTile;
      const std::size_t row_end = std::min<std::size_t>(row0 + kMTile, kTotalTokens);
      sampled_want.insert(sampled_want.end(),
                          want.begin() + static_cast<std::ptrdiff_t>(row0 * kHidden),
                          want.begin() + static_cast<std::ptrdiff_t>(row_end * kHidden));
    }
    if (!NonDegenerate(sampled_want)) {
      Fail(std::string(label) + ": the golden on S is degenerate");
    }
    const Spread spread = MeasureSampled(sampled_want);
    std::printf("  golden on S: rms %.4f stddev %.4f min %.4f max %.4f\n",
                spread.rms, spread.stddev, spread.min, spread.max);
  }

  for (std::size_t i = 0; i < 4; ++i) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &y[i], sizeof(bits));
    std::printf("  tile 0 output[%zu]: actual %.9g (0x%08x), want %.9g\n",
                i, y[i], bits, want[i]);
  }
  const RowDiff diff = CompareOnSamples(y, want, samples, kHidden, kMTile, kTotalTokens);
  std::printf("  diff on S: max abs %.6e, max rel %.6e, cosine %.8f\n",
              diff.max_abs, diff.max_rel, diff.cosine);
  if (!diff.finite) {
    Fail(std::string(label) + ": the device produced a non-finite value on S");
  }
  if (!diff.within) {
    Fail(std::string(label) + ": max abs " + std::to_string(diff.max_abs) +
         " / max rel " + std::to_string(diff.max_rel) +
         " exceeds envelope");
  }
  if (diff.cosine < kCosineFloor) {
    Fail(std::string(label) + ": cosine " + std::to_string(diff.cosine) +
         " is below floor");
  }

  outcome.ok = true;
  return outcome;
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t cap = 0u;
  if (argc > 1) {
    cap = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  }
  std::string which = "AB";
  if (argc > 2) {
    which = argv[2];
  }
  const bool run_a = which.find('A') != std::string::npos;
  const bool run_b = which.find('B') != std::string::npos;

  std::printf(
      "dsv4_moe_prefill_cube -- collapsed boundary-grid CAModel gate (MIX_AIC_1_2)\n"
      "seed 0x%llX  hidden %u  inter %u  experts %u  tokens %u  m_tile %u\n",
      static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
      kTotalTokens, kMTile);

  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const DumpCensus before = ExceptionDumps();

  try {
    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w3 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w2 = MakeBank(kHidden, kInter, kExperts, rng);

    const std::size_t x_floats = static_cast<std::size_t>(kTotalTokens) * kHidden;
    std::vector<float> x(x_floats);
    std::normal_distribution<float> draw(0.0F, 1.0F);
    for (float& v : x) v = draw(rng);

    std::vector<std::int32_t> route_multi(kTotalTokens);
    for (std::uint32_t t = 0; t < kTotalTokens; ++t) {
      const std::uint32_t tile = t / kMTile;
      route_multi[t] = static_cast<std::int32_t>(
          std::min(tile / kTilesPerExpert, kExperts - 1u));
    }

    Runtime runtime;
    LaunchWatchdog watchdog(kPrefillLaunchBudgetSeconds, "[PREFILL-CUBE]");

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

    DeviceBuffer routeMultiDev(route_multi.size() * sizeof(std::int32_t));
    CopyToDevice(routeMultiDev, route_multi);

    LaunchOutcome a;
    if (run_a) {
      a = RunOne("A: 1 AIC cluster (2 AIVs), tokens grouped over 4 experts", runtime,
                 watchdog, w1, w3, w2, x, route_multi, kSimAivSingle, cap,
                 xDev, w1p, w1s, w3p, w3s, w2p, w2s, routeMultiDev);
    }
    LaunchOutcome b;
    if (run_b) {
      b = RunOne("B: 2 AIC clusters (4 AIVs) grid-striding over 4 experts", runtime,
                 watchdog, w1, w3, w2, x, route_multi, kSimAivMulti, cap,
                 xDev, w1p, w1s, w3p, w3s, w2p, w2s, routeMultiDev);
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("\n--- census ---\n");
    std::printf("  excp_log*.dump: %zu file(s) before, %zu after (%zu non-empty)\n",
                before.files, after.files, after.non_empty);
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }
    std::printf("  wall time: A %.1f s, B %.1f s, total %.1f s\n", a.seconds,
                b.seconds, a.seconds + b.seconds);

    std::printf(
        "\n[PASS] dsv4_moe_prefill_cube executes on the dav-3510 CAModel over "
        "trips S = %s of %u, in MIX_AIC_1_2 topology, and matches the "
        "self-audited TWO-PHASE Cube golden inside %g + %g*|want| with cosine >= %g.\n",
        Join(SampleSet(kTileCount)).c_str(), kTileCount,
        static_cast<double>(kAbsTol), static_cast<double>(kRelTol), kCosineFloor);
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}
