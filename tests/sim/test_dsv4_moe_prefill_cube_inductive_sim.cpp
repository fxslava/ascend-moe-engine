/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
// AIC+AIV Cube prefill path (KERNEL_TYPE_MIX_AIC_1_1).
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
//     the path: the KfcServer handshake between the AIC and the AIV, the
//     Matmul framework's L0A/L0B/L0C staging, C landing at TPosition::VECIN,
//     A taken from VECOUT and B taken from TPosition::TSCM -- so neither the
//     intermediate nor a dequantized weight round-trips through GM -- the
//     per-sub-tile MXFP4 -> BF16 dequantization through UB and its UB -> L1
//     push, and the `act` plane surviving the phase 1 / phase 2 boundary in
//     UB. Then it compares the result against the audited golden.
//
// WHAT IT CANNOT DO, stated up front
//   * it is not a latency measurement. A CAModel cycle count is not hardware
//     latency, and nothing here may be read as throughput (CLAUDE.md §5.7).
//   * tiles outside S never run, so an indexing bug confined to them is
//     invisible. The dense-grid backstop for this kernel does not exist yet;
//     that is a full-count soak, and the claim below says so.
//   * the kernel has NEVER RUN ON HARDWARE.
//
// ---------------------------------------------------------------------------
// THE GRID, AND A RESTRICTION THE TWO-PHASE REWRITE LIFTED
// ---------------------------------------------------------------------------
// Two launches, BOTH over the full four-expert routing:
//
//   A. block_dim = 1. The primary gate: one AIC/AIV pair, the whole compute
//      path, every sampled tile on one core.
//   B. block_dim = 2. The grid-stride map and a second, independent KfcServer
//      pair, with the two cores on DIFFERENT experts.
//
// LAUNCH B USED TO BE RESTRICTED TO A SINGLE EXPERT, and the restriction was
// real rather than cosmetic: the predecessor kernel dequantized whole BF16
// weight planes into workspace offsets that did not depend on GetBlockIdx(),
// and the workspace was sized for exactly ONE expert's worth. Two cores on
// different experts therefore dequantized into the same bytes and then read
// each other's weights, so the only safe multi-core launch was one where
// every core wrote identical values.
//
// The two-phase kernel has no weight planes -- each core stages its own
// sub-tiles through its own UB and its own L1 -- so there is no shared mutable
// GM state left to race over, and B now mixes experts exactly as A does. That
// makes the multi-core launch test the grid-stride map against a REAL
// per-expert addressing pattern, which is what it was supposed to do before
// the workspace forced the compromise.

#include "dsv4_moe_block_sim_common.hpp"

#include "kernel/dsv4_moe_prefill_cube_sample_set.h"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>
#include <cstdlib>
#include <set>

namespace moe::kernels::arch35 {
// Defined in tests/sim/kernel/dsv4_moe_prefill_cube_inductive_launch.cpp,
// which only this target links.
Status LaunchDsv4MoePrefillCubeInductiveSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, std::uint32_t aiv_count, float* y_tokens_gm,
    void* workspace_gm, const std::uint32_t* sample_cap_gm,
    Dsv4MoePrefillCubeTilingData* tiling_out, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

using moe::sim::prefill_cube::BuildTileSampleSet;
using moe::sim::prefill_cube::kMaxSamples;
using moe::sim::prefill_cube::kRingDepth;

// kHidden, kInter, kExperts, kGroup and kLimit come from the shared fixture.
// 512 is the floor: the operator's admissible geometry under
// DSV4_TILING_TEST_OVERRIDE wants hidden and inter to be nonzero multiples of
// 512, which is the smallest value every sub-tile extent divides
// (R_SUB 128, K_SUB 128, H_SUB 64, R_SUB2 256) -- this version does not drive
// a ragged final sub-tile on any of the four axes, only a ragged TOKEN tail.
static_assert(kHidden % moe::kernels::arch35::kPrefillCubeKSub == 0u &&
                  kHidden % moe::kernels::arch35::kPrefillCubeHSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub == 0u &&
                  kInter % moe::kernels::arch35::kPrefillCubeRSub2 == 0u,
              "the sim fixture's axes must be whole multiples of every "
              "sub-tile extent, or MakeDsv4MoePrefillCubeTiling rejects the "
              "shape and this gate never reaches the kernel");
constexpr std::uint32_t kMTile = moe::kernels::arch35::kPrefillCubeMTile;  // 16

// 1020 tokens -> 64 token tiles, the last of them PARTIAL (12 rows).
//
// Two decisions in one number:
//   * 64 tiles against |S| = 6 makes the collapse ratio real. A token count
//     small enough for the three sample groups to overlap would make the whole
//     mechanism vacuous, and a much larger one buys nothing -- the untouched
//     tiles cost only GM and a canary byte.
//   * 1020 and not 1024 because 1020 % 16 = 12, so the LAST tile is ragged.
//     That is the entire reason the sample set carries a tail pair: `rows`
//     goes partial, SetTail(rows, ...) has to narrow the Cube's M extent, and
//     the final store must stop at row 1019 instead of running to a tile
//     boundary. At 1024 that path would never execute and the {N-2, N-1}
//     samples would cost simulator time for nothing.
constexpr std::uint32_t kTotalTokens = 1020;
constexpr std::uint32_t kTileCount =
    (kTotalTokens + kMTile - 1u) / kMTile;  // 64
static_assert(kTotalTokens % kMTile != 0,
              "the last tile must be ragged, or the tail samples prove "
              "nothing about partial-tile geometry");
static_assert(kTileCount == 64u, "the sample set is pinned at N = 64");

// Launch A spreads the tokens over the experts in CONTIGUOUS RUNS, which is
// the dispatch order the kernel documents and relies on: it reads
// token_to_expert at the FIRST row of a tile and applies that expert to all
// m_tile rows. A run length that is a whole number of tiles is therefore part
// of the operator's contract, not a convenience here.
constexpr std::uint32_t kTilesPerExpert = kTileCount / 4u;  // 16, kExperts = 4

// Pinned grids. Not queried: a simulated core count that moved with the
// host's SKU would make both the cost and the coverage of this run
// irreproducible.
constexpr std::uint32_t kSimAivSingle = 1;
constexpr std::uint32_t kSimAivMulti = 2;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------
// THE GOLDEN NOW MODELS THIS KERNEL'S SCHEDULE, so both the BF16 rounding and
// the three-level reduction shape are common-mode and cancel. THREE terms are
// left, and the numbers below are unchanged from when the comparison was
// against the flat oracle -- deliberately. Two of the three are not derivable,
// so tightening the envelope to fit a run nobody has seen would be exactly the
// fitting §4.7 forbids.
//
//   * THE DEVICE'S Silu against std::exp. Not derivable from any format:
//     AscendC::Silu is a vector intrinsic over the hardware's exp/reciprocal.
//     This is the dominant term and the honest reason the envelope is loose.
//   * THE ORDER OF THE 16 PRODUCTS INSIDE ONE mmad. The ISA does not specify
//     it; the golden picks natural order as one admissible choice.
//   * THE MATMUL FRAMEWORK'S OWN baseK BLOCKING INSIDE a sub-GEMM. The golden
//     accumulates a whole K_SUB block of fractals before adding it to the
//     axis total; GetMatmulApiTiling may split K_SUB further into L0 micro-
//     tiles, which is one more level of FP32 reassociation the golden does not
//     model. Bounded by K_SUB * 2^-24 relative = 7.6e-6 at K_SUB = 128, so it
//     is far below the Silu term -- but it is real, and it is why the golden
//     is not expected to be bit-exact even though it shares the schedule.
//
// The envelope is also justified independently by the FORMAT: even on the
// pessimistic assumption that the device's BF16 stores round differently from
// the golden's, BF16's 8-bit significand caps each at 2^-9 and the composition
// at 4 * 2^-9 = 7.8e-3 < 1e-2. The absolute term covers outputs near zero,
// where a relative bound means nothing.
//
// COSINE IS KEPT STRICT, at 0.999. Every term above moves MAGNITUDE;
// a structural defect -- a wrong tile, a wrong expert, a wrong stride, a
// transposed operand -- moves DIRECTION, and that is what a loose pointwise
// envelope would otherwise hide.
constexpr float kAbsTol = 1.0e-2F;
constexpr float kRelTol = 1.0e-2F;
constexpr double kCosineFloor = 0.999;

// Per-launch hang budget. This converts an infinite wait into a named failure;
// it does not police speed. An unbalanced SetFlag/WaitFlag inside
// DequantSubTileToL1, a TSCM EnQue with no matching DeQue, or a KFC client
// whose server never answers all present as a hang and not as a fault.
constexpr int kPrefillLaunchBudgetSeconds = 3600;

struct Spread {
  double rms = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
};

// Measures the REFERENCE, never a vector against itself: `NonDegenerate` in
// the shared fixture is the gate, and this only supplies the numbers to print
// beside it. Accumulated in double so the metric is not the thing that loses
// precision.
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

// ---------------------------------------------------------------------------
// Metrics, on S only
// ---------------------------------------------------------------------------
struct RowDiff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double cosine = 1.0;
  bool within = true;
  bool finite = true;
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
        const double got = actual[i];
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

// Everything NOT in S must still carry the poison. An out-of-bounds store, a
// wrong row0 or a grid-stride off-by-one then fails as a named canary
// corruption with a row index instead of as a wrong number much later.
//
// THE TRAP THIS AVOIDS: the unsampled rows are LEGITIMATELY still canary, so
// the golden comparison above must run on S only. Comparing the whole buffer
// would fail every collapsed run against its own oracle.
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
               " was written: the kernel touched a tile it was not scheduled "
               "to run";
      }
    }
  }
  return {};
}

// ZERO GM INTERMEDIATE WORKSPACE, AS AN EXECUTED CHECK.
//
// This replaced a bounds check on the dequantized planes, and it is strictly
// stronger than what it replaced. The predecessor wrote three planes into the
// workspace -- X in BF16, then W1+W3, then W2 -- and this function used to
// verify that the X plane's rows outside S were untouched. The two-phase
// kernel must write NOTHING there at all: every weight sub-tile is decoded
// into UB and pushed to L1, and `act` never leaves UB.
//
// So the driver over-allocates the workspace by a probe region past the KFC
// message-queue reserve, poisons it, and requires every byte of it to survive.
// The probe is sized to cover what the OLD layout would have written at this
// fixture's geometry (X 1 MiB + W1/W3 1 MiB + W2 0.5 MiB), with room to spare,
// so a regression that reintroduced a plane lands inside it rather than past
// it.
std::string CheckWorkspaceUnwritten(const std::vector<std::uint8_t>& probe) {
  for (std::size_t i = 0; i < probe.size(); ++i) {
    if (probe[i] != 0x5Au) {
      return "workspace byte " + std::to_string(i) +
             " past the KFC reserve was written (0x" +
             std::to_string(static_cast<unsigned>(probe[i])) +
             "): the kernel is still putting something in GM, so the "
             "zero-intermediate-workspace claim is false";
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// The golden, on S only -- THE TWO-PHASE CUBE MODEL, not the monolithic oracle
// ---------------------------------------------------------------------------
// THIS COMPARES AGAINST THE SCHEDULE THE KERNEL RUNS, and that is the point of
// the change. The predecessor compared against
// moe_expert_swiglu_mxfp4_w4a16_cpu -- a flat per-token reduction -- so the
// tolerance had to absorb the WHOLE difference between a flat sum and the
// Cube's three-level one. A structural mismatch between the kernel's
// decomposition and the one it was designed to implement would have been
// absorbed along with it.
//
// moe_prefill_cube_flash_2d_w4a16_cpu models this kernel's schedule: the
// resident act plane, the (R_SUB, K_SUB) gate/up sub-tiles, the
// (H_SUB, R_SUB2) down sub-tiles, FP32 at every accumulation level, and the
// C0 = 16 fractal underneath. tests/emu/test_dsv4_moe_prefill_cube_2d_emu
// audits it against the monolithic oracle inside a bound derived from the
// formats, so the chain from this comparison back to the flat reference is
// intact and that link is checked on the host, in milliseconds, where it
// belongs.
//
// ONE TILE AT A TIME, which is exact rather than an approximation. Token tiles
// are spatially independent -- the emu suite asserts it bit-for-bit over four
// visit orders, not as a claim about the source -- so running the golden on a
// single tile's rows with total_tokens = rows gives the same answer as running
// the whole batch and reading those rows out. It also keeps the ragged tail
// ragged: that tile goes in as a 1-tile batch of 12 rows, so the golden walks
// the same partial geometry the kernel does.
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

  // The geometry is the OPERATOR'S OWN, read from the tiling header rather
  // than written out here. A literal would let the two drift, and the drift
  // would present as a tolerance failure with no indication of its cause.
  CubeTileGeometry geom;
  geom.m_tile = static_cast<int>(m_tile);
  geom.r_tile = static_cast<int>(kPrefillCubeRSub);
  geom.k_tile = static_cast<int>(kPrefillCubeKSub);
  geom.h_tile = static_cast<int>(kPrefillCubeHSub);
  geom.r_tile_down = static_cast<int>(kPrefillCubeRSub2);
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
    // The kernel reads the expert id at the tile's FIRST row and applies it to
    // the whole tile. The oracle must do the same, or a tile that straddled an
    // expert boundary would be compared against a per-token routing the device
    // never performs -- the golden IS per-token, so this is handed to it as a
    // uniform route rather than left to coincide.
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

// ---------------------------------------------------------------------------
// One launch, start to finish
// ---------------------------------------------------------------------------
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

  const std::size_t y_floats =
      static_cast<std::size_t>(kTotalTokens) * kHidden;
  auto yDev = MakeGuardedOutput(y_floats);

  // --- the workspace: ZERO the KFC reserve, POISON A PROBE PAST IT ---------
  // The KFC message queue lives at the head and the Matmul framework stands it
  // up there, so it gets zeros -- poisoning a structure the framework
  // initializes would be testing the harness, not the kernel.
  //
  // Everything past the reserve is deliberately MORE than the operator asks
  // for: Dsv4MoePrefillCubeWorkspaceBytes() is now the reserve alone, and the
  // probe is here so "no GM intermediate" is executed rather than asserted.
  // See CheckWorkspaceUnwritten.
  const std::uint64_t ws_bytes =
      Dsv4MoePrefillCubeWorkspaceBytes(kTotalTokens, kHidden, kInter);
  constexpr std::uint64_t kKfcReserve = 1ull << 20;
  if (ws_bytes != kKfcReserve) {
    Fail(std::string(label) +
         ": the public workspace helper returned " + std::to_string(ws_bytes) +
         " bytes, not the KFC reserve alone -- this gate's probe region "
         "assumes the operator asks for nothing else");
  }
  constexpr std::uint64_t kProbeBytes = 8ull << 20;
  DeviceBuffer wsDev(static_cast<std::size_t>(ws_bytes + kProbeBytes));
  {
    std::vector<std::uint8_t> ws_init(
        static_cast<std::size_t>(ws_bytes + kProbeBytes), 0x5Au);
    std::fill(ws_init.begin(),
              ws_init.begin() + static_cast<std::ptrdiff_t>(kKfcReserve), 0u);
    CopyToDevice(wsDev, ws_init);
  }

  DeviceBuffer capDev(sizeof(std::uint32_t));
  {
    const std::vector<std::uint32_t> cap{sample_cap};
    CopyToDevice(capDev, cap);
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  std::printf("\n--- %s ---\n", label);
  std::printf("  aiv_count %u, sample cap %u (0 = all of S)\n", aiv_count,
              sample_cap);

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
      static_cast<const std::uint32_t*>(capDev.get()), &tiling,
      runtime.stream());
  const auto finished = std::chrono::steady_clock::now();
  watchdog.Disarm();
  outcome.seconds =
      std::chrono::duration<double>(finished - started).count();

  std::printf("  elapsed %.1f s, block_dim %u, tiles %u\n", outcome.seconds,
              tiling.block_dim, kTileCount);

  // Past the device probe there IS a simulated device, so a non-success
  // Status is a hard FAIL carrying its code, the recorded ACL call and the
  // recent ACL message (CLAUDE.md §4.4) -- never a skip.
  if (status != Status::success) {
    Fail(LaunchFailure("LaunchDsv4MoePrefillCubeInductiveSim", status));
  }

  // The sample set the DEVICE built, mirrored here from the same header.
  std::vector<std::uint32_t> samples = SampleSet(kTileCount);
  if (sample_cap != 0u && sample_cap < samples.size()) {
    samples.resize(sample_cap);
  }
  std::printf("  S = %s  (|S| = %zu of %u tiles, %.1fx collapse)\n",
              Join(samples).c_str(), samples.size(), kTileCount,
              static_cast<double>(kTileCount) /
                  static_cast<double>(samples.size()));

  // --- guard band, then the canary, then the golden -----------------------
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

  // Everything past the KFC reserve must still be poison.
  {
    std::vector<std::uint8_t> probe(static_cast<std::size_t>(kProbeBytes), 0u);
    RequireAcl(aclrtMemcpy(probe.data(), probe.size(),
                           static_cast<const std::uint8_t*>(wsDev.get()) +
                               kKfcReserve,
                           probe.size(), ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(workspace probe past the KFC reserve)");
    const std::string ws = CheckWorkspaceUnwritten(probe);
    if (!ws.empty()) {
      Fail(std::string(label) + ": " + ws);
    }
    std::printf("  workspace: all %llu MiB past the KFC reserve still "
                "0x5A -- no GM intermediate of any kind\n",
                static_cast<unsigned long long>(kProbeBytes >> 20));
  }

  std::vector<float> want(y_floats, CanaryFloat());
  GoldenOnSamples(w1, w3, w2, x, route, samples, kHidden, kMTile, kTotalTokens,
                  want);

  // The golden must itself carry magnitude and variation before any tolerance
  // on it means anything (§4.8): a nonzero norm and real variance, never
  // CosineSimilarity(want, want) == 1.0.
  {
    std::vector<float> sampled_want;
    for (const std::uint32_t tile : samples) {
      const std::size_t row0 = static_cast<std::size_t>(tile) * kMTile;
      // CLAMPED to the token count: the last tile is ragged by construction,
      // so row0 + m_tile runs past the end of the buffer there.
      const std::size_t row_end = std::min<std::size_t>(row0 + kMTile,
                                                        kTotalTokens);
      sampled_want.insert(sampled_want.end(),
                          want.begin() + static_cast<std::ptrdiff_t>(
                                             row0 * kHidden),
                          want.begin() + static_cast<std::ptrdiff_t>(
                                             row_end * kHidden));
    }
    if (!NonDegenerate(sampled_want)) {
      Fail(std::string(label) +
           ": the golden on S is degenerate (zero norm or zero variance), so "
           "no comparison against it would mean anything");
    }
    const Spread spread = MeasureSampled(sampled_want);
    std::printf("  golden on S: rms %.4f stddev %.4f min %.4f max %.4f\n",
                spread.rms, spread.stddev, spread.min, spread.max);
  }

  const RowDiff diff = CompareOnSamples(y, want, samples, kHidden, kMTile,
                                        kTotalTokens);
  std::printf("  diff on S: max abs %.6e, max rel %.6e, cosine %.8f\n",
              diff.max_abs, diff.max_rel, diff.cosine);
  if (!diff.finite) {
    Fail(std::string(label) + ": the device produced a non-finite value on S");
  }
  if (!diff.within) {
    Fail(std::string(label) + ": max abs " + std::to_string(diff.max_abs) +
         " / max rel " + std::to_string(diff.max_rel) +
         " exceeds the " + std::to_string(kAbsTol) + " + " +
         std::to_string(kRelTol) + "*|want| envelope");
  }
  if (diff.cosine < kCosineFloor) {
    Fail(std::string(label) + ": cosine " + std::to_string(diff.cosine) +
         " is below the strict " + std::to_string(kCosineFloor) +
         " floor -- the direction moved, which rounding does not do");
  }

  outcome.ok = true;
  return outcome;
}

}  // namespace

int main(int argc, char** argv) {
  // argv[1] -- SAMPLE CAP. Exists so this gate's own cost can be MEASURED
  // rather than extrapolated from a sibling suite: every closed-form CAModel
  // prediction for these kernels has been wrong, and this kernel turned out to
  // be far more expensive per trip than any of them. 0 (the default, and what
  // ctest passes) runs the whole of S.
  //
  // argv[2] -- WHICH LAUNCHES: "A", "B" or "AB" (default). Separable because
  // the two launches differ in grid and in routing, so timing them together
  // gives one number for two different shapes -- which is the mistake the
  // cost-model notes on the sibling gates keep warning about.
  //
  // Both knobs are for measurement and triage. The registered ctest passes
  // neither, so what CI runs is the full S on both grids.
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
      "dsv4_moe_prefill_cube -- collapsed boundary-grid CAModel gate\n"
      "seed 0x%llX  hidden %u  inter %u  experts %u  tokens %u  m_tile %u\n"
      "ring depth D = %u (the TSCM slot count of 2 is W1 and W3 of one "
      "(r0, k0) step, not a ping-pong rotation)\n",
      static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
      kTotalTokens, kMTile, kRingDepth);

  // Line-buffer stdout. These runs are long enough that a block-buffered
  // stream hides the per-launch timing until the process exits -- and if it is
  // killed by a timeout first, the measurement is simply lost.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const DumpCensus before = ExceptionDumps();

  try {
    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w3 = MakeBank(kInter, kHidden, kExperts, rng);
    const ExpertBank w2 = MakeBank(kHidden, kInter, kExperts, rng);

    const std::size_t x_floats =
        static_cast<std::size_t>(kTotalTokens) * kHidden;
    std::vector<float> x(x_floats);
    std::normal_distribution<float> draw(0.0F, 1.0F);
    for (float& v : x) v = draw(rng);

    // Launch A's routing: contiguous runs of whole tiles, which is the
    // grouped-by-expert dispatch the operator documents.
    std::vector<std::int32_t> route_multi(kTotalTokens);
    for (std::uint32_t t = 0; t < kTotalTokens; ++t) {
      const std::uint32_t tile = t / kMTile;
      route_multi[t] = static_cast<std::int32_t>(
          std::min(tile / kTilesPerExpert, kExperts - 1u));
    }
    // Launch B uses THE SAME four-expert routing as A. See the header note:
    // the single-expert restriction existed because the predecessor's
    // dequantized weight planes were shared GM state, and the two-phase
    // kernel has no such state. Reusing A's routing is the point -- two cores
    // grid-striding over tiles that belong to different experts is the
    // addressing pattern production performs.

    Runtime runtime;
    LaunchWatchdog watchdog(kPrefillLaunchBudgetSeconds, "[PREFILL-CUBE]");

    // Operands are uploaded once and shared by both launches: neither writes
    // them, and the wrapper does not own operand data (§3.4).
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
      a = RunOne("A: one core, tokens grouped over 4 experts", runtime,
                 watchdog, w1, w3, w2, x, route_multi, kSimAivSingle, cap,
                 xDev, w1p, w1s, w3p, w3s, w2p, w2s, routeMultiDev);
    }
    LaunchOutcome b;
    if (run_b) {
      b = RunOne("B: two cores grid-striding over 4 experts", runtime,
                 watchdog, w1, w3, w2, x, route_multi, kSimAivMulti, cap,
                 xDev, w1p, w1s, w3p, w3s, w2p, w2s, routeMultiDev);
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("\n--- census ---\n");
    std::printf("  excp_log*.dump: %zu file(s) before, %zu after (%zu "
                "non-empty)\n",
                before.files, after.files, after.non_empty);
    // A launch can return ACL_SUCCESS and still have faulted on device,
    // leaving a dump behind. Counting them keeps that case from reading as a
    // pass.
    if (after.non_empty > before.non_empty) {
      Fail("the run left " +
           std::to_string(after.non_empty - before.non_empty) +
           " new non-empty excp_log*.dump file(s): a core faulted even though "
           "the launches reported success");
    }
    std::printf("  wall time: A %.1f s, B %.1f s, total %.1f s\n", a.seconds,
                b.seconds, a.seconds + b.seconds);

    std::printf(
        "\n[PASS] dsv4_moe_prefill_cube executes on the dav-3510 CAModel over "
        "trips S = %s of %u, on a 1-core and a 2-core grid, and matches the "
        "self-audited TWO-PHASE Cube golden -- the same schedule, not a flat "
        "oracle -- inside %g + %g*|want| with cosine >= %g; seed 0x%llX. "
        "Guard bands intact, no canary corruption, nothing written to GM past "
        "the KFC reserve, no exception dumps. NEVER RUN ON HARDWARE, and the "
        "tiles outside S have never executed.\n",
        Join(SampleSet(kTileCount)).c_str(), kTileCount,
        static_cast<double>(kAbsTol), static_cast<double>(kRelTol),
        kCosineFloor, static_cast<unsigned long long>(kSeed));
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}
