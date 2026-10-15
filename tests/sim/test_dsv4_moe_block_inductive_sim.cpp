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

// Collapsed boundary-grid CAModel gate for dsv4_moe_block.
//
// Runs tests/sim/kernel/dsv4_moe_block_inductive_kernel.cpp: the production
// compute bodies on a 4-tile grid per phase, one token, one core. The full
// grid stays covered by Dsv4MoeBlockSim, which this does not replace.
//
// WHY A COLLAPSED GRID IS STILL A FAITHFUL COMPARISON
// Phase 2 is a LINEAR reduction over `activated`:
//
//     y[r] = route_weight * sum_c w2[r][c] * activated[c]
//
// so the oracle does not need `activated` to hold the true gate/up output --
// it only needs to KNOW what it holds. The kernel seeds the whole extent with
// a deterministic pattern, phase 1 overwrites only the sampled tiles, and the
// oracle below reproduces exactly that vector before running the down leg.
// The arithmetic under test is therefore the real thing, at FULL reduction
// length, on the sampled output rows. Nothing is approximated away.
//
// WHAT THIS GATE COVERS THAT THE HOST ORACLE CANNOT
// tests/emu/test_dsv4_moe_block_addressing_emu.cpp proves the index and ring
// arithmetic in seconds. What it cannot do is EXECUTE: the MXFP4 decode, the
// E8M0 scaling, the warp butterfly, the UB residency of `activated` across a
// phase boundary, and the tile ring's real event handshakes. That is this
// file's job.
//
// WHAT IT DOES NOT COVER
// Phase 1 tiles outside the sample set never run, so a gate/up indexing bug
// confined to them is invisible here. Dsv4MoeBlockSim is the backstop.

#include "dsv4_moe_block_sim_common.hpp"

#include <array>

namespace moe::kernels::arch35 {
// Defined in tests/sim/kernel/dsv4_moe_block_inductive_launch.cpp, which only
// this target links.
Status LaunchDsv4MoeBlockInductiveSim(
    const float* x_tokens, const Mxfp4Pair* w1_packed, const E8M0* w1_scale,
    const Mxfp4Pair* w3_packed, const E8M0* w3_scale,
    const Mxfp4Pair* w2_packed, const E8M0* w2_scale,
    const std::int32_t* token_to_expert, std::uint32_t hidden,
    std::uint32_t inter, std::uint32_t num_experts, std::uint32_t total_tokens,
    float* y_tokens, aclrtStream stream);
}  // namespace moe::kernels::arch35

namespace {

// One token on one core: the whole point is to minimise simulated work.
constexpr std::uint32_t kSimTokens = 1;
constexpr std::uint32_t kRowsPerTile = 32;
constexpr std::uint32_t kTileSlots = 2;
constexpr std::uint32_t kSampleTiles = kTileSlots + 2;  // 4

// MIRRORS sim::ActSeed in the sim kernel, bit for bit. Multiples of 1/16 in
// [0.25, 1.0]: exactly representable, so seeding contributes no rounding
// error of its own, numerically benign so it cannot swamp the real
// contributions through the down reduction, and index-VARYING so a phase-2
// column-index bug cannot hide behind a constant.
float ActSeed(std::uint32_t index) {
  return 0.25F + static_cast<float>(index % 13U) * 0.0625F;
}

// MIRRORS sim::BuildBoundarySamples: first, its ping-pong partner, an
// interior tile, the last tile; clipped and deduplicated, ascending order of
// insertion preserved.
std::vector<std::uint32_t> BoundarySamples(std::uint32_t total) {
  std::vector<std::uint32_t> out;
  if (total == 0U) return out;
  const std::array<std::uint32_t, kSampleTiles> raw = {0U, 1U, total / 2U,
                                                       total - 1U};
  for (std::uint32_t candidate : raw) {
    if (candidate >= total) continue;
    bool seen = false;
    for (std::uint32_t kept : out) {
      if (kept == candidate) seen = true;
    }
    if (!seen) out.push_back(candidate);
  }
  return out;
}

// True for rows inside a sampled tile.
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
// The oracle, composed from audited primitives
// ---------------------------------------------------------------------------
// moe_expert_swiglu_mxfp4_cpu returns the FINAL output, not the intermediate
// `activated`, so the collapsed comparison has to compose the two legs from
// mxfp4_gemv_cpu. SelfAuditOracle below proves that decomposition reproduces
// the audited whole-expert oracle bit-for-bit on the full grid, BEFORE it is
// trusted with a seeded body. Oracle phase 1, then phase 2.
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
    const float u =
        up[i] < -limit ? -limit : (up[i] > limit ? limit : up[i]);
    (*activated)[i] = (g / (1.0F + std::exp(-g))) * u;
  }
}

// Phase 1 of the two-phase oracle protocol: the decomposition must reproduce
// the audited oracle before it is allowed to be the reference.
void SelfAuditOracle(const ExpertBank& w1_bank, const ExpertBank& w3_bank,
                     const ExpertBank& w2_bank,
                     const std::vector<float>& x) {
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

  // BIT-EXACT, and it has to be: both paths are the same primitives in the
  // same order on the same host. Any tolerance here would hide a mismatch in
  // the clamp or the SwiGLU, which is precisely what this audit is for.
  for (std::uint32_t i = 0; i < kHidden; ++i) {
    if (composed[i] != audited[i]) {
      Fail("oracle decomposition disagrees with moe_expert_swiglu_mxfp4_cpu "
           "at row " + std::to_string(i) + ": composed " +
           std::to_string(composed[i]) + " vs audited " +
           std::to_string(audited[i]) +
           ". The gate/up clamp or the SwiGLU has drifted; the collapsed "
           "comparison cannot be trusted until this agrees exactly.");
    }
  }
  std::printf("Oracle self-audit: composed gate/up/down reproduces "
              "moe_expert_swiglu_mxfp4_cpu bit-exactly over %u rows\n",
              kHidden);
}

void RunInductiveTest() {
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);

  const ExpertBank w1_bank = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w3_bank = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w2_bank = MakeBank(kHidden, kInter, kExperts, rng);

  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);

  // Oracle phase 1.
  SelfAuditOracle(w1_bank, w3_bank, w2_bank, x);

  // --- the sampled grids, mirrored from the kernel ------------------------
  const std::uint32_t inter_tiles =
      (kInter + kRowsPerTile - 1U) / kRowsPerTile;
  const std::uint32_t hidden_tiles =
      (kHidden + kRowsPerTile - 1U) / kRowsPerTile;
  const std::vector<std::uint32_t> p1 = BoundarySamples(inter_tiles);
  const std::vector<std::uint32_t> p2 = BoundarySamples(hidden_tiles);
  Require(p1.size() == kSampleTiles && p2.size() == kSampleTiles,
          "the boundary grids must both have kTileSlots + 2 tiles at this "
          "geometry");

  // --- the oracle over the SEEDED body ------------------------------------
  const std::uint32_t expert = 2;
  const std::vector<std::int32_t> token_to_expert = {
      static_cast<std::int32_t>(expert)};
  const moe::emu::Mxfp4Matrix w1 = w1_bank.View(expert);
  const moe::emu::Mxfp4Matrix w3 = w3_bank.View(expert);
  const moe::emu::Mxfp4Matrix w2 = w2_bank.View(expert);

  std::vector<float> activated;
  GateUpActivated(w1, w3, x, kLimit, &activated);

  // Everything phase 1 did NOT visit keeps the seed -- exactly what the
  // kernel's `activated` holds when phase 2 reduces over it.
  const std::vector<bool> act_sampled = SampledRowMask(kInter, p1);
  std::uint32_t seeded = 0;
  for (std::uint32_t i = 0; i < kInter; ++i) {
    if (!act_sampled[i]) {
      activated[i] = ActSeed(i);
      ++seeded;
    }
  }
  Require(seeded == kInter - kSampleTiles * kRowsPerTile,
          "seeded row count must be the complement of the sampled phase-1 "
          "tiles");
  Require(NonDegenerate(activated),
          "seeded `activated` is zero, constant or non-finite");

  std::vector<float> expected(kHidden, 0.0F);
  moe::emu::mxfp4_gemv_cpu(w2, activated.data(), expected.data());

  // --- device -------------------------------------------------------------
  std::vector<float> actual(static_cast<std::size_t>(kSimTokens) * kHidden,
                            0.0F);
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
      DeviceBuffer d_t2e(token_to_expert.size() * sizeof(std::int32_t));

      CopyToDevice(d_w1p, w1_bank.packed);
      CopyToDevice(d_w1s, w1_bank.scales);
      CopyToDevice(d_w3p, w3_bank.packed);
      CopyToDevice(d_w3s, w3_bank.scales);
      CopyToDevice(d_w2p, w2_bank.packed);
      CopyToDevice(d_w2s, w2_bank.scales);
      CopyToDevice(d_x, x);
      CopyToDevice(d_t2e, token_to_expert);

      // y is POISONED, not zeroed, over its whole logical extent plus a guard
      // band. Unsampled phase-2 rows must come back still poisoned: that is
      // what makes a store to a wrong y row visible, which is the exact risk
      // the tile-ring reindex introduced.
      auto d_y = MakeGuardedOutput(actual.size());

      LaunchWatchdog watchdog(kLaunchBudgetSeconds, "[dsv4_block_inductive]");
      const auto status =
          moe::kernels::arch35::LaunchDsv4MoeBlockInductiveSim(
              static_cast<const float*>(d_x.get()),
              static_cast<const Mxfp4Pair*>(d_w1p.get()),
              static_cast<const E8M0*>(d_w1s.get()),
              static_cast<const Mxfp4Pair*>(d_w3p.get()),
              static_cast<const E8M0*>(d_w3s.get()),
              static_cast<const Mxfp4Pair*>(d_w2p.get()),
              static_cast<const E8M0*>(d_w2s.get()),
              static_cast<const std::int32_t*>(d_t2e.get()), kHidden, kInter,
              kExperts, kSimTokens, static_cast<float*>(d_y->get()),
              runtime.stream());
      if (status != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoeBlockInductiveSim", status));
      }
      watchdog.Arm("dsv4_moe_block_inductive_sim");
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(inductive)");
      watchdog.Disarm();

      RequireGuardIntact(*d_y, actual.size(), "inductive y");
      CopyFromDevice(actual, *d_y);

      const DumpCensus dumps = ExceptionDumps();
      if (dumps.non_empty > 0) {
        Fail("device left " + std::to_string(dumps.non_empty) +
             " non-empty excp_log*.dump file(s) of " +
             std::to_string(dumps.files) +
             "; the launch faulted even though ACL reported success");
      }
    }
    runtime.Close();
  }

  // --- compare, on the sampled phase-2 rows ONLY --------------------------
  // Unsampled rows are supposed to still hold the poison. Comparing them
  // against the oracle would fail every collapsed run for the wrong reason.
  const std::vector<bool> y_sampled = SampledRowMask(kHidden, p2);
  double max_abs = 0.0;
  std::uint32_t compared = 0;
  std::uint32_t untouched = 0;
  for (std::uint32_t r = 0; r < kHidden; ++r) {
    if (y_sampled[r]) {
      const double diff =
          std::fabs(static_cast<double>(actual[r]) - expected[r]);
      const double scale = std::fmax(1.0, std::fabs(expected[r]));
      if (diff / scale > max_abs) max_abs = diff / scale;
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
           " was written (no longer the 0x5A5A5A5A poison): the collapsed "
           "phase-2 grid stored to a row it never sampled");
    }
  }

  Require(compared == kSampleTiles * kRowsPerTile,
          "the compare must cover exactly the sampled phase-2 rows");
  Require(untouched == kHidden - compared,
          "every unsampled y row must still hold the poison");

  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf(
      "Executed dsv4_moe_block_inductive_sim: hidden=%u inter=%u expert=%u, "
      "phase-1 tiles {%u,%u,%u,%u} of %u, phase-2 tiles {%u,%u,%u,%u} of %u\n",
      kHidden, kInter, expert, p1[0], p1[1], p1[2], p1[3], inter_tiles, p2[0],
      p2[1], p2[2], p2[3], hidden_tiles);
  std::printf(
      "Seeded %u of %u `activated` entries; compared %u sampled y rows "
      "(max rel diff %.8g, tolerance %.2g); %u unsampled rows still poisoned\n",
      seeded, kInter, compared, max_abs, kRelTolerance, untouched);
}

}  // namespace

int main() {
  try {
    RunInductiveTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] %s\n", error.what());
    return 1;
  }
  std::printf("[PASS] Dsv4MoeBlockInductiveSim\n");
  return 0;
}
