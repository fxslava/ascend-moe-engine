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

// Host validation of the collapsed trip schedule for dsv4_moe_prefill_cube.
//
// The sample-set builder decides WHICH trips the CAModel gate executes, so it
// decides what that gate proves. It is pure index arithmetic, and index
// arithmetic is validated here rather than on the simulator (CLAUDE.md §5.5):
// "it compiles for dav-3510" is not evidence that a schedule covers its
// boundaries. A wrong builder does not fail loudly on the simulator -- it runs
// a slightly different, slightly smaller set of trips and still passes, which
// is the quietest possible way to lose coverage.
//
// Checked against a BRUTE-FORCE construction (std::set, no shared code with
// the builder) for every N in [1, 4096] and every D in [1, 4], plus the
// structural obligations the induction argument actually rests on.
//
// No device, no CANN, no toolkit. This is a tests/sim host-plumbing binary
// (§4.3) and builds with a stock g++.

#include "kernel/dsv4_moe_prefill_cube_sample_set.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

using moe::sim::prefill_cube::BuildTileSampleSet;
using moe::sim::prefill_cube::kMaxSamples;
using moe::sim::prefill_cube::kRingDepth;

int g_failures = 0;
int g_checks = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  [ FAIL ] %s\n", what.c_str());
  }
}

// The specification, written independently of the implementation: the union of
// the three groups, clipped and deduplicated, in ascending order.
std::vector<std::uint32_t> BruteForce(std::uint32_t n, std::uint32_t d) {
  std::set<std::uint32_t> s;
  if (n == 0u) {
    return {};
  }
  const std::uint32_t depth = d > 4u ? 4u : d;
  for (std::uint32_t i = 0; i <= depth; ++i) {
    if (i < n) s.insert(i);
  }
  const std::uint32_t mid = n / 2u;
  if (mid < n) s.insert(mid);
  if (mid + 1u < n) s.insert(mid + 1u);
  if (n >= 2u) s.insert(n - 2u);
  s.insert(n - 1u);
  return std::vector<std::uint32_t>(s.begin(), s.end());
}

}  // namespace

int main() {
  std::printf("dsv4_moe_prefill_cube collapsed trip schedule -- host audit\n");
  std::printf("ring depth D = %u, max samples = %u\n", kRingDepth, kMaxSamples);

  // --- 1. agreement with the specification, exhaustively over N and D ------
  std::uint32_t worst_n = 0;
  std::uint32_t worst_d = 0;
  bool agreed = true;
  for (std::uint32_t d = 1; d <= 4u && agreed; ++d) {
    for (std::uint32_t n = 1; n <= 4096u; ++n) {
      std::uint32_t out[kMaxSamples] = {};
      const std::uint32_t count = BuildTileSampleSet(n, d, out);
      const std::vector<std::uint32_t> want = BruteForce(n, d);
      bool ok = count == want.size() && count <= kMaxSamples;
      for (std::uint32_t i = 0; ok && i < count; ++i) {
        if (out[i] != want[i]) ok = false;
      }
      if (!ok) {
        agreed = false;
        worst_n = n;
        worst_d = d;
        std::printf("           N=%u D=%u: got %u entries, want %zu\n", n, d,
                    count, want.size());
        break;
      }
    }
  }
  Check(agreed,
        "the builder matches a brute-force construction for every N in "
        "[1, 4096] and every D in [1, 4] (first divergence at N=" +
            std::to_string(worst_n) + " D=" + std::to_string(worst_d) + ")");

  // --- 2. the output is always sorted, unique and in range ----------------
  bool well_formed = true;
  for (std::uint32_t d = 1; d <= 4u; ++d) {
    for (std::uint32_t n = 1; n <= 4096u; ++n) {
      std::uint32_t out[kMaxSamples] = {};
      const std::uint32_t count = BuildTileSampleSet(n, d, out);
      if (count == 0u || count > kMaxSamples) well_formed = false;
      for (std::uint32_t i = 0; i < count; ++i) {
        if (out[i] >= n) well_formed = false;
        if (i > 0 && out[i] <= out[i - 1]) well_formed = false;
      }
    }
  }
  Check(well_formed,
        "every sample is in [0, N), strictly ascending, and the count never "
        "exceeds the array bound the device declares");

  // --- 3. N <= |S| falls back to the FULL loop ----------------------------
  // The template's small-shape branch, which must not be optimized away: the
  // shapes used in CI are exactly the small ones, and a collapse that silently
  // skipped a trip there would be invisible.
  bool full_when_small = true;
  for (std::uint32_t n = 1; n <= kMaxSamples; ++n) {
    std::uint32_t out[kMaxSamples] = {};
    const std::uint32_t count = BuildTileSampleSet(n, kRingDepth, out);
    const std::uint32_t head = kRingDepth + 1u;  // {0..D}
    // For N small enough that the three groups overlap into the whole range,
    // the set must BE the whole range.
    if (n <= head + 2u) {
      if (count != n) full_when_small = false;
      for (std::uint32_t i = 0; i < count; ++i) {
        if (out[i] != i) full_when_small = false;
      }
    }
  }
  Check(full_when_small,
        "for N small enough that the groups meet, the set is the entire trip "
        "range -- no trip is skipped at a CI shape");

  // --- 4. the structural obligations of the induction ---------------------
  // These are the reason the set has the shape it has. Asserting them
  // separately means a future edit to the builder that still "looks right"
  // cannot quietly drop one.
  bool roles_present = true;
  std::uint32_t missing_at = 0;
  for (std::uint32_t n = 2u * (kRingDepth + 3u); n <= 4096u; ++n) {
    std::uint32_t out[kMaxSamples] = {};
    const std::uint32_t count = BuildTileSampleSet(n, kRingDepth, out);
    std::set<std::uint32_t> s(out, out + count);
    const std::uint32_t mid = n / 2u;
    const bool cold_start = s.count(0u) != 0u;
    // One full rotation of the ring: 0..D must all be present.
    bool rotation = true;
    for (std::uint32_t i = 0; i <= kRingDepth; ++i) {
      if (s.count(i) == 0u) rotation = false;
    }
    // An interior trip WITH A SUCCESSOR: mid alone would prove that mid
    // writes its state and never that the next trip reads the right state.
    const bool interior_pair = s.count(mid) != 0u && s.count(mid + 1u) != 0u;
    // The epilogue, also as a pair, for the same reason.
    const bool tail_pair = s.count(n - 2u) != 0u && s.count(n - 1u) != 0u;
    if (!(cold_start && rotation && interior_pair && tail_pair)) {
      roles_present = false;
      missing_at = n;
      break;
    }
  }
  Check(roles_present,
        "for every N large enough to separate them, the set carries a cold "
        "start, a full 0..D rotation, an interior PAIR and a tail PAIR (first "
        "gap at N=" + std::to_string(missing_at) + ")");

  // --- 5. the collapse ratio is real at the gate's own shape --------------
  // The gate runs 1024 tokens at a 16-token tile, so N = 64. Pinned here
  // because the claim the gate is allowed to make quotes this number.
  {
    constexpr std::uint32_t kGateTiles = 64u;
    std::uint32_t out[kMaxSamples] = {};
    const std::uint32_t count = BuildTileSampleSet(kGateTiles, kRingDepth, out);
    std::printf("           gate shape: N=%u -> |S|=%u  S = {", kGateTiles,
                count);
    for (std::uint32_t i = 0; i < count; ++i) {
      std::printf("%s%u", i ? ", " : "", out[i]);
    }
    std::printf("}  collapse %.1fx\n",
                static_cast<double>(kGateTiles) / static_cast<double>(count));
    // {0, 1, 32, 33, 62, 63}: the head pair, the interior pair at mid = 32,
    // and the tail pair.
    const std::uint32_t want[6] = {0u, 1u, 32u, 33u, 62u, 63u};
    bool matches = count == 6u;
    for (std::uint32_t i = 0; matches && i < count; ++i) {
      if (out[i] != want[i]) matches = false;
    }
    Check(matches,
          "at the gate's N = 64 the set is exactly {0, 1, 32, 33, 62, 63}");
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) {
    std::puts("[PASS] the collapsed trip schedule covers its boundary roles "
              "and matches the specification for every N and D tested");
  }
  return g_failures == 0 ? 0 : 1;
}
