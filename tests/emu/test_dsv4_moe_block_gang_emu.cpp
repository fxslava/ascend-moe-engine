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

// Host validation of the dsv4_moe_block DECODE GANG's index arithmetic
// (CLAUDE.md section 5.5: device logic that is pure integer / index
// arithmetic is validated here, not left to the NPU).
//
// WHY THIS FILE EXISTS SEPARATELY FROM test_dsv4_moe_block_addressing_emu.cpp
// That file validates the WITHIN-ROW map -- how a row's 16 lanes cover its
// scale groups, and how the grid-stride token map covers a prefill batch. It
// predates the gang and assumes one core owns a whole expert.
//
// This file validates the ACROSS-CORE map the gang decode introduced: how the
// gang_size lanes of one routed slot divide that expert's row tiles between
// them, in both the gate/up and the down phase.
//
// THE OBLIGATION, AND WHY IT IS NOT COSMETIC
// The down phase accumulates into y with SetAtomicAdd. So a tile owned by two
// lanes is added TWICE and a tile owned by none is silently omitted -- in both
// cases y is wrong by a plausible-looking amount, with no fault, no error code
// and nothing in the output that says which tile was wrong. The partition
// being exact is the whole correctness argument for splitting an expert, and
// it costs no hardware to check.
//
// Mirrored, not linked, per this suite's convention and section 4.1: the emu
// suite links moe_emu only and must build under a stock host g++ with no CANN
// present. tests/sim/test_dsv4_moe_block_grid_sim.cpp links the REAL builder
// and checks it against the same formula, so a drift between this transcript
// and the operator is caught there rather than hidden here.

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// --- mirrored from src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.h --
constexpr std::uint32_t kMoeBlockAivFallback = 56u;
constexpr std::uint32_t kMoeBlockAivMax = 256u;
constexpr std::uint32_t kMoeBlockMinGangSize = 1u;
constexpr std::uint32_t kRowsPerTile = 32u;

// MakeDsv4MoeDecodeTopkTiling's gang derivation, mirrored.
std::uint32_t ResolveAiv(std::uint32_t aiv) {
  return (aiv == 0u || aiv > kMoeBlockAivMax) ? kMoeBlockAivFallback : aiv;
}
std::uint32_t GangSize(std::uint32_t aiv, std::uint32_t top_k) {
  const std::uint32_t g = ResolveAiv(aiv) / top_k;
  return g < kMoeBlockMinGangSize ? kMoeBlockMinGangSize : g;
}

// --- mirrored from op_kernel/dsv4_moe_block_kernel.cpp :: TilesOwned --------
// Keep textually identical to the kernel. If these drift the test is void,
// which is exactly what the sim-side builder check exists to notice.
std::uint32_t TilesOwned(std::uint32_t total, std::uint32_t lo,
                         std::uint32_t stride) {
  if (lo >= total) {
    return 0u;
  }
  return (total - lo + stride - 1u) / stride;
}

int g_failures = 0;

void Report(const char* what, std::uint32_t aiv, std::uint32_t top_k,
            std::uint32_t extra) {
  std::printf("[FAIL] %s (aiv=%u top_k=%u value=%u)\n", what, aiv, top_k, extra);
  ++g_failures;
}

// The core obligation: under the kernel's (tile_lo = lane, tile_stride =
// gang_size) map, the gang's lanes cover [0, total) exactly once.
void CheckPartition(std::uint32_t aiv, std::uint32_t top_k,
                    std::uint32_t gang, std::uint32_t total) {
  std::vector<int> cover(total, 0);
  for (std::uint32_t lane = 0; lane < gang; ++lane) {
    const std::uint32_t owned = TilesOwned(total, lane, gang);
    for (std::uint32_t i = 0; i < owned; ++i) {
      const std::uint32_t tile = lane + i * gang;
      if (tile >= total) {
        Report("a lane walked past the last tile", aiv, top_k, tile);
        return;
      }
      ++cover[tile];
    }
  }
  for (std::uint32_t tile = 0; tile < total; ++tile) {
    if (cover[tile] != 1) {
      std::printf("[FAIL] tile %u of %u covered %d times "
                  "(aiv=%u top_k=%u gang=%u) -- atomic-add double counts it\n",
                  tile, total, cover[tile], aiv, top_k, gang);
      ++g_failures;
    }
  }
}

}  // namespace

int main() {
  // Every 950PR Vector count the toolkit's platform configs list, plus the
  // dev part, a zero (capability query failed) and an out-of-range reading.
  const std::uint32_t aivs[] = {8u, 56u, 64u, 72u, 0u, 1000u};
  const std::uint32_t topks[] = {1u, 2u, 4u, 6u, 8u};

  // Tile counts: the production geometry's two phases (hidden 4096 and inter
  // 2048 over 32-row tiles give 128 and 64), the admissible floor, and small
  // / odd / prime counts where a gang is wider than the work.
  const std::uint32_t totals[] = {128u, 64u, 16u, 1u, 2u, 7u, 13u, 31u};

  for (std::uint32_t aiv : aivs) {
    for (std::uint32_t top_k : topks) {
      const std::uint32_t gang = GangSize(aiv, top_k);

      if (gang == 0u) {
        // The kernel divides GetBlockIdx() by this.
        Report("gang_size is zero", aiv, top_k, 0u);
        continue;
      }
      // The grid must fit the device it was derived for.
      const std::uint32_t grid = gang * top_k;
      if (aiv != 0u && aiv <= kMoeBlockAivMax && grid > aiv) {
        Report("grid oversubscribes the device", aiv, top_k, grid);
      }
      // ...and must not waste more than the unavoidable aiv % top_k.
      const std::uint32_t resolved = ResolveAiv(aiv);
      if (resolved >= top_k && resolved - grid >= top_k) {
        Report("grid leaves a whole extra gang idle", aiv, top_k, grid);
      }

      for (std::uint32_t total : totals) {
        CheckPartition(aiv, top_k, gang, total);
      }
    }
  }

  // The production geometry's tile counts are what the gang actually divides,
  // so pin them rather than leaving them implied by the loop above.
  if (4096u / kRowsPerTile != 128u || 2048u / kRowsPerTile != 64u) {
    std::puts("[FAIL] production tile counts are not 128 (hidden) / 64 (inter)");
    ++g_failures;
  }

  // A gang wider than the tile count must leave the surplus lanes with zero
  // tiles, not with a wrapped or negative count. This is the case that makes
  // the down phase's `owned == 0` early return load-bearing.
  for (std::uint32_t lane = 0; lane < 8u; ++lane) {
    const std::uint32_t owned = TilesOwned(4u, lane, 8u);
    const std::uint32_t want = lane < 4u ? 1u : 0u;
    if (owned != want) {
      std::printf("[FAIL] lane %u of a gang of 8 over 4 tiles owns %u, want %u\n",
                  lane, owned, want);
      ++g_failures;
    }
  }

  if (g_failures != 0) {
    std::printf("\n%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("[PASS] decode gang: grid fits every 950PR SKU, and the lane map "
            "partitions both phases' tiles exactly once");
  return 0;
}
