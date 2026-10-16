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

// Collapsed, MULTI-CORE CAModel gate for the GANG DECODE path of
// dsv4_moe_block.
//
// Runs tests/sim/kernel/dsv4_moe_block_decode_inductive_kernel.cpp: the
// production compute bodies and the production control flow around
// AscendC::SyncAll<true>(), on a collapsed per-lane tile grid, over a real
// gang of gang_size * top_k cores.
//
// WHAT THIS COVERS THAT NOTHING ELSE DOES
//   * tests/emu/test_dsv4_moe_block_gang_emu proves the lane map partitions
//     the tiles exactly once -- on the host, in milliseconds, without
//     executing anything.
//   * Dsv4MoeBlockInductiveSim executes the PREFILL path on ONE core. A
//     single-core run satisfies SyncAll trivially, so it says nothing about
//     the barrier.
//   * this gate is the only one that EXECUTES the cross-core handshake: every
//     launched core reaching the barrier once, `activated` crossing cores
//     through GM, each lane reducing over the whole plane, and top_k slots
//     atomically accumulating into the same `y` rows.
//
// WHY A COLLAPSED GRID IS STILL A FAITHFUL COMPARISON
// Phase 2 is a LINEAR reduction over `activated`:
//
//     y[r] = sum_slots route_weight[s] * sum_c w2[e(s)][r][c] * activated[s][c]
//
// so the oracle does not need `activated` to hold the true gate/up output --
// it only needs to KNOW what it holds. The HOST pre-fills the caller-owned
// `activated_ws` plane with a deterministic seed, phase 1 overwrites only the
// sampled tiles, and the oracle below reproduces exactly that vector before
// running the down leg. The arithmetic under test is the real thing, at FULL
// reduction length, on the sampled output rows.
//
// Seeding from the host rather than from a device VF means the seed pattern
// exists in ONE place -- here -- so it cannot drift from the oracle that
// mirrors it.
//
// WHAT IT DOES NOT COVER
// Tiles outside each lane's sample set never run, so a gate/up indexing bug
// confined to them is invisible here; Dsv4MoeBlockSim is the backstop for the
// dense grid. It also cannot time anything: a CAModel cycle count is not
// latency (CLAUDE.md §5.7).

#include "dsv4_moe_block_sim_common.hpp"

#include "dsv4_moe_block_tiling.h"

#include <array>

namespace moe::kernels::arch35 {
// Defined in tests/sim/kernel/dsv4_moe_block_decode_inductive_launch.cpp,
// which only this target links.
Status LaunchDsv4MoeBlockDecodeInductiveSim(
    const float* x_gm, const Mxfp4Pair* w1_packed_gm, const E8M0* w1_scale_gm,
    const Mxfp4Pair* w3_packed_gm, const E8M0* w3_scale_gm,
    const Mxfp4Pair* w2_packed_gm, const E8M0* w2_scale_gm,
    const std::int32_t* topk_expert_ids_gm, const float* routing_weights_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t top_k, std::uint32_t aiv_count, float* y_gm,
    float* activated_ws_gm, Dsv4MoeBlockTilingData* tiling_out,
    aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

constexpr std::uint32_t kRowsPerTile = 32;
constexpr std::uint32_t kTileSlots = 2;
constexpr std::uint32_t kGateUpSamples = kTileSlots + 2;  // 4
constexpr std::uint32_t kDownSamples = kTileSlots + 3;    // 5, odd on purpose

// THE SIMULATED GRID. aiv_count = 4 with kTopK = 2 gives gang_size = 2 and
// block_dim = 4: two routed slots, two lanes each. That is the smallest grid
// that exercises everything the gang introduced -- slot mapping (two slots),
// the strided partition (two lanes), the barrier across more than one core,
// and atomic accumulation of two slots into one y row.
//
// It is PINNED rather than queried because a simulated core count that
// changed with the host's SKU would make this run's cost and its coverage
// irreproducible. The production wrapper queries; this one is told.
constexpr std::uint32_t kSimAivCount = 4;

// MIRRORS the host-side seed the driver writes into `activated_ws`. Multiples
// of 1/16 in [0.25, 1.0]: exactly representable, so seeding contributes no
// rounding error of its own; numerically benign, so it cannot swamp the real
// contributions through the down reduction (0x5A5A5A5A as a float is ~1.5e16,
// which would); and index-VARYING, so a phase-2 column-index bug cannot hide
// behind a constant.
float ActSeed(std::uint32_t index) {
  return 0.25F + static_cast<float>(index % 13U) * 0.0625F;
}

// MIRRORS decode_sim::TilesOwned, which is the production TilesOwned.
std::uint32_t TilesOwned(std::uint32_t total, std::uint32_t lo,
                         std::uint32_t stride) {
  if (lo >= total) return 0U;
  return (total - lo + stride - 1U) / stride;
}

// MIRRORS decode_sim::BuildLaneSamples. Positions within ONE LANE'S OWN
// subsequence, not global tile indices.
std::vector<std::uint32_t> LaneSamples(std::uint32_t owned,
                                       std::uint32_t want) {
  std::vector<std::uint32_t> out;
  if (owned == 0U) return out;
  std::array<std::uint32_t, kDownSamples> raw = {
      0U, 1U, owned / 2U, owned / 2U + 1U, owned - 1U};
  if (want < kDownSamples) raw[3] = owned - 1U;
  for (std::uint32_t i = 0; i < want && i < kDownSamples; ++i) {
    if (raw[i] >= owned) continue;
    bool seen = false;
    for (std::uint32_t kept : out) {
      if (kept == raw[i]) seen = true;
    }
    if (!seen) out.push_back(raw[i]);
  }
  return out;
}

// The GLOBAL tiles a whole gang touches in a phase: the union over its lanes
// of (lane + pos*gang) for that lane's sampled positions. Identical for every
// slot, because the rule depends only on (lane, gang, total).
std::vector<std::uint32_t> GangTiles(std::uint32_t total, std::uint32_t gang,
                                     std::uint32_t want) {
  std::vector<std::uint32_t> tiles;
  for (std::uint32_t lane = 0; lane < gang; ++lane) {
    const std::uint32_t owned = TilesOwned(total, lane, gang);
    for (std::uint32_t pos : LaneSamples(owned, want)) {
      tiles.push_back(lane + pos * gang);
    }
  }
  std::sort(tiles.begin(), tiles.end());
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  return tiles;
}

std::vector<bool> SampledRowMask(std::uint32_t extent,
                                 const std::vector<std::uint32_t>& tiles) {
  std::vector<bool> mask(extent, false);
  for (std::uint32_t tile : tiles) {
    for (std::uint32_t r = 0; r < kRowsPerTile; ++r) {
      const std::uint32_t row = tile * kRowsPerTile + r;
      if (row < extent) mask[row] = true;
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Oracle, composed from audited primitives
// ---------------------------------------------------------------------------
void GateUpActivated(const moe::emu::Mxfp4Matrix& w1,
                     const moe::emu::Mxfp4Matrix& w3,
                     const std::vector<float>& x, float limit,
                     std::vector<float>* activated) {
  const std::size_t inter = static_cast<std::size_t>(w1.rows);
  std::vector<float> gate(inter, 0.0F);
  std::vector<float> up(inter, 0.0F);
  moe::emu::mxfp4_gemv_cpu(w1, x.data(), gate.data());
  moe::emu::mxfp4_gemv_cpu(w3, x.data(), up.data());
  activated->assign(inter, 0.0F);
  for (std::size_t i = 0; i < inter; ++i) {
    // ASYMMETRIC, exactly as the kernel and the audited oracle do it: the
    // gate is bounded ABOVE only, `up` both ways.
    const float g = gate[i] < limit ? gate[i] : limit;
    const float u = up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
    (*activated)[i] = (g / (1.0F + std::exp(-g))) * u;
  }
}

// Phase 1 of the two-phase oracle protocol: the decomposition must reproduce
// the audited whole-expert oracle BEFORE it is trusted with a seeded body.
void SelfAuditOracle(const ExpertBank& w1_bank, const ExpertBank& w3_bank,
                     const ExpertBank& w2_bank, const std::vector<float>& x) {
  const moe::emu::Mxfp4Matrix w1 = w1_bank.View(0);
  const moe::emu::Mxfp4Matrix w3 = w3_bank.View(0);
  const moe::emu::Mxfp4Matrix w2 = w2_bank.View(0);

  std::vector<float> activated;
  GateUpActivated(w1, w3, x, kLimit, &activated);
  Require(NonDegenerate(activated),
          "composed `activated` is zero, constant or non-finite");

  std::vector<float> composed(kHidden, 0.0F);
  moe::emu::mxfp4_gemv_cpu(w2, activated.data(), composed.data());

  std::vector<float> scratch;
  std::vector<float> audited(kHidden, 0.0F);
  moe::emu::moe_expert_swiglu_mxfp4_cpu(w1, w3, w2, x.data(), audited.data(),
                                        kLimit, &scratch);

  // BIT-EXACT: both paths are the same primitives in the same order on the
  // same host. Any tolerance here would hide a drift in the clamp or the
  // SwiGLU, which is what this audit exists to catch.
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    if (composed[i] != audited[i]) {
      Fail("oracle decomposition disagrees with moe_expert_swiglu_mxfp4_cpu "
           "at row " + std::to_string(i) + ": composed " +
           std::to_string(composed[i]) + " vs audited " +
           std::to_string(audited[i]));
    }
  }
  std::printf("Oracle self-audit: composed gate/up/down reproduces "
              "moe_expert_swiglu_mxfp4_cpu bit-exactly over %u rows\n",
              kHidden);
}

void RunDecodeInductiveTest() {
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);

  const ExpertBank w1_bank = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w3_bank = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w2_bank = MakeBank(kHidden, kInter, kExperts, rng);

  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);

  SelfAuditOracle(w1_bank, w3_bank, w2_bank, x);

  // --- the routed slots ---------------------------------------------------
  const std::vector<std::int32_t> topk_ids = {3, 1};
  const std::vector<float> routing_weights = {0.625F, 0.375F};
  Require(topk_ids.size() == kTopK && routing_weights.size() == kTopK,
          "the routed slot arrays must have kTopK entries");

  // --- the grid, as the production builder derives it ---------------------
  moe::kernels::arch35::Dsv4MoeBlockTilingData plan{};
  const auto plan_status = moe::kernels::arch35::MakeDsv4MoeDecodeTopkTiling(
      kHidden, kInter, kExperts, kTopK, kSimAivCount, &plan);
  Require(plan_status == moe::kernels::arch35::Status::success,
          "the decode tiling builder rejected the sim geometry");
  const std::uint32_t gang = plan.gang_size;
  Require(gang >= 2U,
          "the sim grid must give each slot at least two lanes, or the "
          "barrier and the strided ring are not exercised");
  Require(plan.block_dim == gang * kTopK,
          "block_dim must be gang_size * top_k");

  // --- the sampled grids, mirrored from the kernel ------------------------
  const std::uint32_t inter_tiles = (kInter + kRowsPerTile - 1U) / kRowsPerTile;
  const std::uint32_t hidden_tiles =
      (kHidden + kRowsPerTile - 1U) / kRowsPerTile;
  const std::vector<std::uint32_t> p1 =
      GangTiles(inter_tiles, gang, kGateUpSamples);
  const std::vector<std::uint32_t> p2 =
      GangTiles(hidden_tiles, gang, kDownSamples);
  Require(!p1.empty() && !p2.empty(), "both sampled grids must be non-empty");

  // The odd sample count is what makes the ring's epilogue fire; assert the
  // property rather than trusting the arithmetic stayed odd.
  for (std::uint32_t lane = 0; lane < gang; ++lane) {
    const std::uint32_t owned = TilesOwned(hidden_tiles, lane, gang);
    Require(LaneSamples(owned, kDownSamples).size() % 2U == 1U,
            "each lane's down sample count must be ODD so the tile ring's "
            "odd-remainder epilogue executes -- that branch is live only "
            "since the gang split");
  }

  // --- the oracle over the SEEDED body, per slot --------------------------
  const std::vector<bool> act_sampled = SampledRowMask(kInter, p1);
  std::vector<float> expected(kHidden, 0.0F);
  std::vector<std::vector<float>> seeded_act(kTopK);
  std::uint32_t seeded_rows = 0;

  for (std::uint32_t s = 0; s < kTopK; ++s) {
    const std::uint32_t expert = static_cast<std::uint32_t>(topk_ids[s]);
    const moe::emu::Mxfp4Matrix w1 = w1_bank.View(expert);
    const moe::emu::Mxfp4Matrix w3 = w3_bank.View(expert);
    const moe::emu::Mxfp4Matrix w2 = w2_bank.View(expert);

    std::vector<float> activated;
    GateUpActivated(w1, w3, x, kLimit, &activated);
    // Everything phase 1 did NOT visit keeps the host's seed -- exactly what
    // the slot's `activated` plane holds when phase 2 reduces over it.
    for (std::uint32_t i = 0; i < kInter; ++i) {
      if (!act_sampled[i]) {
        activated[i] = ActSeed(i);
        if (s == 0) ++seeded_rows;
      }
    }
    Require(NonDegenerate(activated),
            "seeded `activated` is zero, constant or non-finite");
    seeded_act[s] = activated;

    std::vector<float> down(kHidden, 0.0F);
    moe::emu::mxfp4_gemv_cpu(w2, activated.data(), down.data());
    // Atomic accumulation across slots, with the router gate folded in.
    for (std::uint32_t r = 0; r < kHidden; ++r) {
      expected[r] += routing_weights[s] * down[r];
    }
  }

  // --- device -------------------------------------------------------------
  std::vector<float> actual(kHidden, 0.0F);
  std::uint32_t launched_grid = 0;
  {
    Runtime runtime;
    {
      DeviceBuffer d_w1p(w1_bank.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w1s(w1_bank.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w3p(w3_bank.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w3s(w3_bank.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w2p(w2_bank.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w2s(w2_bank.scales.size() * sizeof(E8M0));
      DeviceBuffer d_x(x.size() * sizeof(float));
      DeviceBuffer d_ids(topk_ids.size() * sizeof(std::int32_t));
      DeviceBuffer d_rw(routing_weights.size() * sizeof(float));

      CopyToDevice(d_w1p, w1_bank.packed);
      CopyToDevice(d_w1s, w1_bank.scales);
      CopyToDevice(d_w3p, w3_bank.packed);
      CopyToDevice(d_w3s, w3_bank.scales);
      CopyToDevice(d_w2p, w2_bank.packed);
      CopyToDevice(d_w2s, w2_bank.scales);
      CopyToDevice(d_x, x);
      CopyToDevice(d_ids, topk_ids);
      CopyToDevice(d_rw, routing_weights);

      // THE `activated` PLANE, PRE-SEEDED BY THE HOST. [top_k, inter] FP32,
      // every slot's row carrying the same deterministic pattern. Phase 1
      // overwrites only the sampled tiles; the rest reaches phase 2 as seed,
      // which is exactly what the oracle above mirrored.
      std::vector<float> act_seed(static_cast<std::size_t>(kTopK) * kInter);
      for (std::uint32_t s = 0; s < kTopK; ++s) {
        for (std::uint32_t i = 0; i < kInter; ++i) {
          act_seed[static_cast<std::size_t>(s) * kInter + i] = ActSeed(i);
        }
      }
      DeviceBuffer d_actws(act_seed.size() * sizeof(float));
      CopyToDevice(d_actws, act_seed);

      // DECODE ACCUMULATES, so y must start at a known value. It is POISONED
      // rather than zeroed over its logical extent plus a guard band: an
      // unsampled phase-2 row must come back STILL POISONED, which is what
      // makes a store to a wrong y row visible -- the exact risk the strided
      // tile-ring reindex introduced.
      //
      // Poison is not a valid accumulator start, so the sampled rows are
      // zeroed below; the two regions have different jobs.
      auto d_y = MakeGuardedOutput(actual.size());
      const std::vector<bool> y_sampled_pre = SampledRowMask(kHidden, p2);
      {
        std::vector<float> y_init(kHidden, CanaryFloat());
        for (std::uint32_t r = 0; r < kHidden; ++r) {
          if (y_sampled_pre[r]) y_init[r] = 0.0F;
        }
        RequireAcl(aclrtMemcpy(d_y->get(), kHidden * sizeof(float),
                               y_init.data(), kHidden * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(y poison + zeroed sampled rows)");
      }

      LaunchWatchdog watchdog(kLaunchBudgetSeconds,
                              "[dsv4_block_decode_inductive]");
      moe::kernels::arch35::Dsv4MoeBlockTilingData used{};
      const auto status =
          moe::kernels::arch35::LaunchDsv4MoeBlockDecodeInductiveSim(
              static_cast<const float*>(d_x.get()),
              static_cast<const Mxfp4Pair*>(d_w1p.get()),
              static_cast<const E8M0*>(d_w1s.get()),
              static_cast<const Mxfp4Pair*>(d_w3p.get()),
              static_cast<const E8M0*>(d_w3s.get()),
              static_cast<const Mxfp4Pair*>(d_w2p.get()),
              static_cast<const E8M0*>(d_w2s.get()),
              static_cast<const std::int32_t*>(d_ids.get()),
              static_cast<const float*>(d_rw.get()), kHidden, kInter, kExperts,
              kTopK, kSimAivCount, static_cast<float*>(d_y->get()),
              static_cast<float*>(d_actws.get()), &used, runtime.stream());
      if (status != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoeBlockDecodeInductiveSim", status));
      }
      watchdog.Arm("dsv4_moe_block_decode_inductive_sim");
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(decode inductive)");
      watchdog.Disarm();
      launched_grid = used.block_dim;

      RequireGuardIntact(*d_y, actual.size(), "decode inductive y");
      CopyFromDevice(actual, *d_y);

      const DumpCensus dumps = ExceptionDumps();
      if (dumps.non_empty > 0) {
        Fail("device left " + std::to_string(dumps.non_empty) +
             " non-empty excp_log*.dump file(s) of " +
             std::to_string(dumps.files) +
             "; the launch faulted even though ACL reported success");
      }
      std::printf("Exception dumps: %zu file(s) seen, %zu non-empty\n",
                  static_cast<std::size_t>(dumps.files),
                  static_cast<std::size_t>(dumps.non_empty));
    }
    runtime.Close();
  }

  // --- compare, on the sampled phase-2 rows ONLY --------------------------
  const std::vector<bool> y_sampled = SampledRowMask(kHidden, p2);
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::uint32_t compared = 0;
  std::uint32_t untouched = 0;
  for (std::uint32_t r = 0; r < kHidden; ++r) {
    if (y_sampled[r]) {
      const double diff =
          std::fabs(static_cast<double>(actual[r]) - expected[r]);
      const double scale = std::fmax(1.0, std::fabs(expected[r]));
      if (diff > max_abs) max_abs = diff;
      if (diff / scale > max_rel) max_rel = diff / scale;
      ++compared;
      if (diff > kAbsTolerance &&
          diff / scale > static_cast<double>(kRelTolerance)) {
        Fail("sampled y row " + std::to_string(r) + " differs: got " +
             std::to_string(actual[r]) + ", expected " +
             std::to_string(expected[r]));
      }
    } else if (IsCanary(actual[r])) {
      ++untouched;
    } else {
      Fail("unsampled y row " + std::to_string(r) +
           " was written (no longer the 0x5A5A5A5A poison): a lane stored to "
           "a row outside its strided sample set");
    }
  }
  Require(compared > 0, "the compare must cover at least one sampled y row");
  Require(untouched == kHidden - compared,
          "every unsampled y row must still hold the poison");

  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf(
      "Executed dsv4_moe_block_decode_inductive_sim: hidden=%u inter=%u "
      "top_k=%u gang_size=%u block_dim=%u (aiv_count=%u), slots={%d,%d}\n",
      kHidden, kInter, kTopK, gang, launched_grid, kSimAivCount, topk_ids[0],
      topk_ids[1]);
  std::printf("Sampled %zu of %u phase-1 tiles and %zu of %u phase-2 tiles "
              "(per-lane positions, %u and %u per lane)\n",
              p1.size(), inter_tiles, p2.size(), hidden_tiles, kGateUpSamples,
              kDownSamples);
  std::printf(
      "Seeded %u of %u `activated` entries per slot; compared %u sampled y "
      "rows (max abs %.8g, max rel %.8g, tolerances abs %.2g / rel %.2g); "
      "%u unsampled rows still poisoned; guard band intact\n",
      seeded_rows, kInter, compared, max_abs, max_rel, kAbsTolerance,
      kRelTolerance, untouched);
}

}  // namespace

int main() {
  try {
    RunDecodeInductiveTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] %s\n", error.what());
    return 1;
  }
  std::printf("[PASS] Dsv4MoeBlockDecodeInductiveSim\n");
  return 0;
}
