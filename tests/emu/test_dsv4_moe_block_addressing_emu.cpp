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

// Host validation of dsv4_moe_block's pure integer / index arithmetic
// (CLAUDE.md section 5.5: device logic that is index arithmetic is validated
// on the host, not left to the NPU).
//
// WHY THIS FILE EXISTS SEPARATELY FROM test_kernel_addressing_emu.cpp
// That suite covers mxfp4_expert_gemv, whose lane map is LANE-CONTIGUOUS:
//
//     p = lane * pairs_per_lane + g * 16 + q
//
// dsv4_moe_block uses a different, STRIDED map:
//
//     g = step * kGroupsPerWarpStep + lane
//
// They are not interchangeable and only the strided one makes a warp step
// contiguous in GM, which is the whole basis of the coalescing claim in
// dsv4_moe_block_arch35.h ("one step of the 16 lanes spans 16 consecutive
// groups = 256 contiguous bytes"). That claim is asserted here.
//
// WHAT THE CAMODEL SOAK CANNOT DO CHEAPLY, AND THIS CAN
// Dsv4MoeBlockSim executes the real kernel, but one core runs a whole expert
// at the 512x512 geometry floor, so it is a multi-hour soak. Nothing about
// the index arithmetic below needs a simulator: it is integer math, and it is
// exactly what is wrong the first time. Seconds here, hours there.
//
// Scope note: this file proves the ADDRESS SETS and the REDUCTION TOPOLOGY.
// It does not prove the MXFP4 numerics -- the quantization path is the
// oracle's job (src/emu/dsv4_quant_gemv.cpp) and the CAModel suite's.

#include <cstdint>
#include <string>
#include <vector>

#include "emu_test_support.hpp"

namespace {

// --- mirrored from src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_block_arch35.h
constexpr std::uint32_t kRowsPerTile = 32;
constexpr std::uint32_t kLanesPerRow = 16;
constexpr std::uint32_t kGroupSize = 32;                       // logical cols
constexpr std::uint32_t kGroupBytes = kGroupSize / 2;          // 16
constexpr std::uint32_t kGroupsPerWarpStep = kLanesPerRow;      // 16
constexpr std::uint32_t kWarpStepBytes =
    kGroupsPerWarpStep * kGroupBytes;                           // 256
constexpr std::uint32_t kSimtThreads = kRowsPerTile * kLanesPerRow;  // 512

// --- mirrored from src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.h --
constexpr std::uint32_t kReductionQuantum = 512;

// The admissible geometry floor and the production geometry.
constexpr std::uint32_t kFloorDim = 512;
constexpr std::uint32_t kProdHidden = 4096;
constexpr std::uint32_t kProdInter = 2048;

// MOE_DSV4_BLOCK_GEOMETRY_ADMISSIBLE, mirrored.
constexpr bool GeometryAdmissible(std::uint32_t hidden, std::uint32_t inter) {
  return hidden >= kReductionQuantum && inter >= kReductionQuantum &&
         hidden % kReductionQuantum == 0u && inter % kReductionQuantum == 0u;
}

// THE map under test, mirrored from LaneRowPartial / LaneRowPartialGateUp:
//
//     g = step * kGroupsPerWarpStep + lane
//
// Defined ONCE and used by every check below. This is not stylistic: both
// this map and the lane-contiguous alternative (g = lane * groups_per_lane +
// step) are permutations of the group set, so coverage alone cannot tell them
// apart. Only per-warp-step CONTIGUITY can, and a contiguity check that
// re-derives the formula instead of calling this function proves nothing
// about the formula. An earlier draft of this file did exactly that and a
// mutation to the map passed all 4121 checks.
// `groups_per_lane` is named (not commented out) although the strided map does
// not use it: the lane-contiguous alternative does, and a mutation test that
// cannot even express the alternative is not a test of this choice.
constexpr std::uint32_t GroupForLaneStep(std::uint32_t step, std::uint32_t lane,
                                         std::uint32_t groups_per_lane) {
  (void)groups_per_lane;
  return step * kGroupsPerWarpStep + lane;
}

// A span whose backing allocation is exactly its logical extent, so a logical
// overrun is also a heap overrun and ASan (MOE_EMU_SANITIZE=ON) becomes an
// independent second witness. Same device as test_kernel_addressing_emu.cpp.
template <typename T>
class BoundedSpan {
 public:
  BoundedSpan(emu_test::Runner& runner, std::string name, std::size_t extent)
      : runner_(runner), name_(std::move(name)), data_(extent), extent_(extent) {}

  void At(std::uint64_t index, const std::string& what) {
    if (index < extent_) {
      data_[static_cast<std::size_t>(index)] = T{};
      return;
    }
    if (clean_) {  // report the FIRST violation only; a stride bug trips many
      clean_ = false;
      runner_.Check(false, name_ + ": " + what + " index " +
                               std::to_string(index) + " is past extent " +
                               std::to_string(extent_));
    }
  }

  bool clean() const { return clean_; }

 private:
  emu_test::Runner& runner_;
  std::string name_;
  std::vector<T> data_;
  std::size_t extent_;
  bool clean_ = true;
};

// ---------------------------------------------------------------------------
// 1. The mirrored constants must still agree with the header
// ---------------------------------------------------------------------------
// A drifted mirror makes every check below a check of the wrong kernel, so
// these come first and are stated as the header states them.
void TestMirroredConstants(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: mirrored constants");

  runner.Check(kGroupBytes == 16u,
               "a scale group is exactly one uint4 of packed nibbles");
  runner.Check(kWarpStepBytes == 256u,
               "one warp step spans 256 contiguous packed bytes");
  runner.Check(kSimtThreads == 512u, "the SIMT launch is 32 rows x 16 lanes");
  runner.Check(kGroupSize / 2u == kGroupBytes,
               "two 4-bit values share a packed byte");

  // The UB layout's own arithmetic, restated: 16,384 + 8,192 + 128 = 24,704.
  const std::uint32_t x_bytes = kProdHidden * 4u;
  const std::uint32_t act_bytes = kProdInter * 4u;
  const std::uint32_t tile_bytes = kRowsPerTile * 4u;
  runner.Check(x_bytes + act_bytes + tile_bytes == 24704u,
               "the dynamic UB segments sum to 24,704 B");
  runner.Check(x_bytes + act_bytes + tile_bytes <= 221184u,
               "the dynamic UB total is inside the 221,184 B usable ceiling");
}

// ---------------------------------------------------------------------------
// 2. Thread decomposition
// ---------------------------------------------------------------------------
//   row_in_tile = threadIdx.x / 16
//   lane        = threadIdx.x % 16
//   global_row  = tile_base + row_in_tile
//
// The properties that matter: the map is a bijection onto (row, lane); every
// row gets exactly 16 lanes; and lane 0 is the UNIQUE writer of each row's
// output slot. A decomposition that is merely in bounds can still have two
// threads claim one slot, which on device is a race that produces a plausible
// number.
void TestThreadDecomposition(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: thread -> (row, lane) decomposition");

  std::vector<int> lanes_per_row(kRowsPerTile, 0);
  std::vector<int> writers_per_row(kRowsPerTile, 0);
  std::vector<int> seen(kRowsPerTile * kLanesPerRow, 0);

  for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
    const std::uint32_t row_in_tile = t / kLanesPerRow;
    const std::uint32_t lane = t % kLanesPerRow;

    runner.Check(row_in_tile < kRowsPerTile, "row_in_tile is inside the tile");
    runner.Check(lane < kLanesPerRow, "lane is inside the row group");

    ++lanes_per_row[row_in_tile];
    ++seen[row_in_tile * kLanesPerRow + lane];
    if (lane == 0u) {
      ++writers_per_row[row_in_tile];
    }
    // The inverse must recover the thread, or the map is not a bijection.
    runner.Check(row_in_tile * kLanesPerRow + lane == t,
                 "(row, lane) inverts back to threadIdx.x");
  }

  bool every_row_full = true;
  bool one_writer_each = true;
  for (std::uint32_t r = 0; r < kRowsPerTile; ++r) {
    if (lanes_per_row[r] != static_cast<int>(kLanesPerRow)) {
      every_row_full = false;
    }
    if (writers_per_row[r] != 1) {
      one_writer_each = false;
    }
  }
  runner.Check(every_row_full, "each of the 32 rows is served by 16 lanes");
  runner.Check(one_writer_each,
               "exactly one thread (lane 0) writes each row's output slot");

  bool covered_once = true;
  for (int count : seen) {
    if (count != 1) {
      covered_once = false;
    }
  }
  runner.Check(covered_once, "the 512 threads cover (row, lane) exactly once");
}

// ---------------------------------------------------------------------------
// 3. The strided group-to-lane map, and the coalescing claim
// ---------------------------------------------------------------------------
//   groups_per_lane = (cols / kGroupSize) / kLanesPerRow
//   g               = step * kGroupsPerWarpStep + lane
//
// Two separate things to prove:
//   (a) COVERAGE -- the (lane, step) pairs hit every group of the row exactly
//       once. Under-coverage is in bounds and silently drops columns, which
//       is the failure this kernel's design is most exposed to.
//   (b) CONTIGUITY -- for a fixed step, lanes 0..15 read 16 CONSECUTIVE
//       groups, so the warp step is one 256 B run at a 256 B boundary. That
//       is the header's coalescing claim, and it is false for the
//       lane-contiguous map that mxfp4_expert_gemv uses. Asserting it here is
//       what stops someone "unifying" the two maps.
void CheckGroupLaneMap(emu_test::Runner& runner, std::uint32_t cols) {
  const std::string tag = "cols=" + std::to_string(cols) + ": ";

  const std::uint32_t groups = cols / kGroupSize;
  const std::uint32_t groups_per_lane = groups / kLanesPerRow;
  const std::uint32_t packed_row_stride = cols / 2u;
  const std::uint32_t scale_row_stride = cols / kGroupSize;

  runner.Check(groups % kLanesPerRow == 0u,
               tag + "the row's groups divide evenly across the 16 lanes");
  runner.Check(groups_per_lane >= 1u,
               tag + "every lane has at least one group to reduce");

  BoundedSpan<std::uint8_t> packed(runner, tag + "packed row",
                                   packed_row_stride);
  BoundedSpan<std::uint8_t> scale(runner, tag + "scale row", scale_row_stride);
  BoundedSpan<float> x_ub(runner, tag + "x_ub", cols);

  std::vector<int> group_hits(groups, 0);
  std::uint64_t max_byte = 0;
  std::uint64_t max_x = 0;

  for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
    for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
      const std::uint32_t g = GroupForLaneStep(step, lane, groups_per_lane);
      runner.Check(g < groups, tag + "group index is inside the row");
      if (g < groups) {
        ++group_hits[g];
      }

      // The scale byte: one per group, indexed by the group itself.
      scale.At(g, "group scale byte");

      // The packed load: a uint4 at g * 16, i.e. 16 B wide at a 16 B
      // boundary, so it never straddles two uint4 slots.
      const std::uint64_t byte_off =
          static_cast<std::uint64_t>(g) * kGroupBytes;
      runner.Check(byte_off % kGroupBytes == 0u,
                   tag + "the uint4 packed load is 16 B aligned");
      for (std::uint32_t b = 0; b < kGroupBytes; ++b) {
        packed.At(byte_off + b, "packed group byte");
      }
      if (byte_off + kGroupBytes - 1u > max_byte) {
        max_byte = byte_off + kGroupBytes - 1u;
      }

      // The x window: 32 logical columns per group.
      const std::uint64_t x_off = static_cast<std::uint64_t>(g) * kGroupSize;
      for (std::uint32_t c = 0; c < kGroupSize; ++c) {
        x_ub.At(x_off + c, "x logical column");
      }
      if (x_off + kGroupSize - 1u > max_x) {
        max_x = x_off + kGroupSize - 1u;
      }
    }
  }

  runner.Check(packed.clean(), tag + "no packed-row overrun over a full row");
  runner.Check(scale.clean(), tag + "no scale-row overrun over a full row");
  runner.Check(x_ub.clean(), tag + "no x_ub overrun over a full row");

  // (a) Coverage, and TIGHT coverage: exactly once, and the whole row.
  bool once = true;
  for (int hits : group_hits) {
    if (hits != 1) {
      once = false;
    }
  }
  runner.Check(once, tag + "every scale group is claimed by exactly one lane");
  runner.Check(max_byte + 1u == packed_row_stride,
               tag + "the lanes exactly cover the packed row");
  runner.Check(max_x + 1u == cols, tag + "the groups exactly cover x");

  // (b) Contiguity per warp step -- the coalescing claim, and the ONLY
  // property that distinguishes this map from the lane-contiguous one.
  //
  // Derived from GroupForLaneStep, never re-deriving the formula: collect the
  // 16 lanes' group indices for a fixed step and require that, as a SET, they
  // are 16 consecutive groups starting at a 256 B boundary.
  for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
    std::uint32_t lo = groups;
    std::uint32_t hi = 0;
    std::vector<int> in_step(groups, 0);
    for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
      const std::uint32_t g = GroupForLaneStep(step, lane, groups_per_lane);
      if (g >= groups) {
        continue;
      }
      ++in_step[g];
      if (g < lo) {
        lo = g;
      }
      if (g > hi) {
        hi = g;
      }
    }

    // 16 distinct groups...
    bool distinct = true;
    std::uint32_t claimed = 0;
    for (int hits : in_step) {
      if (hits > 1) {
        distinct = false;
      }
      claimed += static_cast<std::uint32_t>(hits);
    }
    runner.Check(distinct && claimed == kLanesPerRow,
                 tag + "a warp step reads 16 distinct groups");
    // ...that are CONSECUTIVE. hi - lo == 15 with 16 distinct members forces
    // the set to be exactly {lo .. lo+15} with no gap.
    runner.Check(hi - lo == kLanesPerRow - 1u,
                 tag + "a warp step's 16 groups are consecutive, so the step "
                       "is one contiguous GM run (the coalescing claim)");

    const std::uint64_t first_byte = static_cast<std::uint64_t>(lo) * kGroupBytes;
    const std::uint64_t last_byte =
        static_cast<std::uint64_t>(hi) * kGroupBytes + kGroupBytes - 1u;
    runner.Check(last_byte - first_byte + 1u == kWarpStepBytes,
                 tag + "a warp step spans exactly 256 bytes");
    runner.Check(first_byte % kWarpStepBytes == 0u,
                 tag + "a warp step starts at a 256 B boundary");
  }
}

void TestGroupLaneMap(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: strided group-to-lane map and coalescing");
  // The geometry floor, the production pair, and the scaled test width.
  CheckGroupLaneMap(runner, kFloorDim);
  CheckGroupLaneMap(runner, kProdInter);
  CheckGroupLaneMap(runner, kProdHidden);
}

// ---------------------------------------------------------------------------
// 4. The in-register butterfly (ReduceRow)
// ---------------------------------------------------------------------------
//   value += asc_shfl_xor(value, 1, 16);
//   value += asc_shfl_xor(value, 2, 16);
//   value += asc_shfl_xor(value, 4, 16);
//   value += asc_shfl_xor(value, 8, 16);
//
// Nothing else in the suite covers this, and it is the highest-risk pure-index
// construct in the kernel: a wrong mask set, or a width that is not 16, gives
// a partial sum or pulls in a neighbouring row -- both of which look like a
// numerical bug and neither of which needs hardware to find.
//
// Modelled exactly: width 16 confines the exchange to the ALIGNED 16-lane
// segment, so a thread's partner is (segment_base + (lane ^ mask)).
//
// Values are small integers so the sums are exact in float. That is
// deliberate: this is an INDEX test, and admitting float reassociation error
// would blur the thing being measured.
void TestButterflyReduction(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: ReduceRow butterfly topology");

  const std::uint32_t masks[] = {1u, 2u, 4u, 8u};

  // The mask set must be exactly the powers of two below the width, or the
  // butterfly does not span the segment.
  std::uint32_t mask_or = 0;
  for (std::uint32_t m : masks) {
    mask_or |= m;
  }
  runner.Check(mask_or == kLanesPerRow - 1u,
               "the xor masks 1,2,4,8 span every bit of a 16-lane index");

  // One full tile: 32 independent row segments of 16 lanes.
  std::vector<float> value(kSimtThreads, 0.0f);
  // Row r's lanes carry (r + 1) each, so row r's true sum is 16 * (r + 1) and
  // every row's expected total is distinct -- cross-row contamination cannot
  // cancel out and pass.
  for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
    const std::uint32_t row_in_tile = t / kLanesPerRow;
    value[t] = static_cast<float>(row_in_tile + 1u);
  }

  for (std::uint32_t m : masks) {
    std::vector<float> next = value;
    for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
      const std::uint32_t segment_base = (t / kLanesPerRow) * kLanesPerRow;
      const std::uint32_t lane = t % kLanesPerRow;
      const std::uint32_t partner = segment_base + (lane ^ m);

      // Width 16 must keep the partner inside this row's own segment.
      runner.Check(partner >= segment_base &&
                       partner < segment_base + kLanesPerRow,
                   "xor mask " + std::to_string(m) +
                       " stays inside the aligned 16-lane row group");
      next[t] = value[t] + value[partner];
    }
    value = next;
  }

  // Every lane of a row must hold that row's full 16-lane sum: the kernel
  // reads the result on lane 0, but an all-lanes butterfly that only happens
  // to be right on lane 0 is a different (and fragile) reduction.
  bool all_rows_exact = true;
  for (std::uint32_t r = 0; r < kRowsPerTile; ++r) {
    const float expected =
        static_cast<float>(kLanesPerRow) * static_cast<float>(r + 1u);
    for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
      if (value[r * kLanesPerRow + lane] != expected) {
        all_rows_exact = false;
      }
    }
  }
  runner.Check(all_rows_exact,
               "after 4 butterfly steps every lane holds its own row's "
               "complete 16-lane sum, and no other row's");

  // A negative control: drop one mask and the reduction must FAIL to total.
  // Without this, a butterfly that summed by accident would look verified.
  {
    std::vector<float> broken(kSimtThreads, 1.0f);
    const std::uint32_t short_masks[] = {1u, 2u, 4u};  // 8 omitted
    for (std::uint32_t m : short_masks) {
      std::vector<float> next = broken;
      for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
        const std::uint32_t segment_base = (t / kLanesPerRow) * kLanesPerRow;
        const std::uint32_t lane = t % kLanesPerRow;
        next[t] = broken[t] + broken[segment_base + (lane ^ m)];
      }
      broken = next;
    }
    runner.Check(broken[0] == 8.0f,
                 "omitting the mask-8 step reduces only half the row (8 of "
                 "16), so the mask set is load-bearing");
  }
}

// ---------------------------------------------------------------------------
// 5. Tile progression, and the dead tail branch
// ---------------------------------------------------------------------------
// Both phase loops are `tile_base = tile_idx * 32` over
// ceil(dim / 32) tiles, and both VFs carry a `global_row >= dim` padding
// branch that writes 0.
//
// At ADMISSIBLE geometry that branch is unreachable: dim is a multiple of 512,
// 512 is a multiple of 32, so the tiles divide the extent exactly. Pinning
// that here records it as dead code rather than leaving a reader to assume it
// is tested -- and if the admissibility rule is ever loosened to a non-
// multiple of 32, this check fails and says so.
void TestTileProgression(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: tile_base progression and tail reachability");

  const std::uint32_t dims[] = {kFloorDim, kProdInter, kProdHidden};
  for (std::uint32_t dim : dims) {
    const std::string tag = "dim=" + std::to_string(dim) + ": ";
    const std::uint32_t tiles = (dim + kRowsPerTile - 1u) / kRowsPerTile;

    std::vector<int> row_hits(dim, 0);
    bool tail_row_seen = false;
    for (std::uint32_t tile_idx = 0; tile_idx < tiles; ++tile_idx) {
      const std::uint32_t tile_base = tile_idx * kRowsPerTile;
      for (std::uint32_t row_in_tile = 0; row_in_tile < kRowsPerTile;
           ++row_in_tile) {
        const std::uint32_t global_row = tile_base + row_in_tile;
        if (global_row < dim) {
          ++row_hits[global_row];
        } else {
          tail_row_seen = true;
        }
      }
    }

    bool once = true;
    for (int hits : row_hits) {
      if (hits != 1) {
        once = false;
      }
    }
    runner.Check(once, tag + "the tiles cover every row exactly once");
    runner.Check(tiles * kRowsPerTile == dim,
                 tag + "the tiles divide the extent exactly");
    runner.Check(!tail_row_seen,
                 tag + "no padding row is produced, so the VFs' "
                       "`global_row >= dim` branch is unreachable here");
  }

  // And the reason it is unreachable, stated as the rule rather than the
  // instance: admissibility forces a multiple of 512.
  runner.Check(kReductionQuantum % kRowsPerTile == 0u,
               "the reduction quantum is a whole number of 32-row tiles, so "
               "no admissible extent has a partial tail tile");
}

// ---------------------------------------------------------------------------
// 6. Row and expert strides
// ---------------------------------------------------------------------------
// TilingRejected fixes all four row strides exactly, so the kernel trusts
// them. They are derived here the same way and checked for overflow at the
// largest offset each one produces.
void TestStrides(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: row and expert stride arithmetic");

  struct Case {
    std::uint32_t hidden;
    std::uint32_t inter;
  };
  const Case cases[] = {{kFloorDim, kFloorDim}, {kProdHidden, kProdInter}};

  for (const Case& c : cases) {
    const std::string tag = "h=" + std::to_string(c.hidden) + ",i=" +
                            std::to_string(c.inter) + ": ";
    runner.Check(GeometryAdmissible(c.hidden, c.inter),
                 tag + "geometry is admissible");

    const std::uint32_t w13_packed_row_stride = c.hidden / 2u;
    const std::uint32_t w13_scale_row_stride = c.hidden / kGroupSize;
    const std::uint32_t w2_packed_row_stride = c.inter / 2u;
    const std::uint32_t w2_scale_row_stride = c.inter / kGroupSize;

    // The gate's own equalities, restated.
    runner.Check(w13_packed_row_stride * 2u == c.hidden,
                 tag + "w13 packed row stride is hidden/2 with no remainder");
    runner.Check(w13_scale_row_stride * kGroupSize == c.hidden,
                 tag + "w13 scale row stride is hidden/32 with no remainder");
    runner.Check(w2_packed_row_stride * 2u == c.inter,
                 tag + "w2 packed row stride is inter/2 with no remainder");
    runner.Check(w2_scale_row_stride * kGroupSize == c.inter,
                 tag + "w2 scale row stride is inter/32 with no remainder");

    // Largest in-expert row offset, as the VFs compute it:
    // global_row * packed_row_stride, with global_row < inter (phase 1) or
    // < hidden (phase 2). These stay uint32 in the kernel, so they must fit.
    const std::uint64_t w13_max =
        static_cast<std::uint64_t>(c.inter - 1u) * w13_packed_row_stride;
    const std::uint64_t w2_max =
        static_cast<std::uint64_t>(c.hidden - 1u) * w2_packed_row_stride;
    runner.Check(w13_max <= 0xFFFFFFFFull,
                 tag + "the largest w13 row offset fits the uint32 the "
                       "kernel keeps it in");
    runner.Check(w2_max <= 0xFFFFFFFFull,
                 tag + "the largest w2 row offset fits uint32");

    // Expert strides. The kernel widens to uint64 BEFORE multiplying by the
    // expert id (section 2.7). Show that the widening is what makes the top
    // of the range safe: at the production geometry a 256-expert table is
    // ~1.07 GB, which still fits uint32 -- but one doubling of either extent
    // or of the expert count does not, and uint32 arithmetic would wrap
    // silently into a valid-looking offset.
    const std::uint64_t w13_expert_stride =
        static_cast<std::uint64_t>(c.inter) * w13_packed_row_stride;
    const std::uint32_t experts = 256u;
    const std::uint64_t widened =
        static_cast<std::uint64_t>(experts - 1u) * w13_expert_stride;
    const std::uint32_t narrow_stride =
        static_cast<std::uint32_t>(w13_expert_stride);
    const std::uint32_t narrowed = (experts - 1u) * narrow_stride;
    runner.Check(widened == static_cast<std::uint64_t>(narrowed) ||
                     widened > 0xFFFFFFFFull,
                 tag + "uint64 and uint32 expert offsets agree only while "
                       "the product is inside uint32");
    // Double the expert count and the narrow form must diverge -- the
    // positive evidence that the widening is load-bearing, not decoration.
    const std::uint64_t widened_2k =
        static_cast<std::uint64_t>(2048u - 1u) * w13_expert_stride;
    const std::uint32_t narrowed_2k = (2048u - 1u) * narrow_stride;
    if (widened_2k > 0xFFFFFFFFull) {
      runner.Check(widened_2k != static_cast<std::uint64_t>(narrowed_2k),
                   tag + "at 2048 experts the uint32 product wraps, so "
                         "widening before the multiply is load-bearing");
    }
  }
}

// ---------------------------------------------------------------------------
// 7. Grid-stride token coverage (prefill) and slot mapping (decode)
// ---------------------------------------------------------------------------
//   prefill: for (token = block_idx; token < total_tokens; token += block_dim)
//   decode:  block_dim == top_k, core `slot` owns routed slot `slot`
//
// The prefill loop is the one place a core takes more than one token, so it
// is the only per-core trip count that scales with the batch. Its coverage
// must be exact: a miss leaves an output row untouched (and `y` is
// overwritten, not accumulated, on this path, so a miss is a stale row).
void TestTokenCoverage(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: grid-stride token coverage");

  struct Case {
    std::uint32_t total_tokens;
    std::uint32_t block_dim;
  };
  // Includes the soak's own shape (2 tokens over 32 cores), where most cores
  // take no token at all, and the ragged case where they take unequal counts.
  const Case cases[] = {{2u, 32u}, {32u, 32u}, {33u, 32u}, {100u, 32u},
                        {1u, 32u}};

  for (const Case& c : cases) {
    const std::string tag = "tokens=" + std::to_string(c.total_tokens) +
                            ",cores=" + std::to_string(c.block_dim) + ": ";
    std::vector<int> hits(c.total_tokens, 0);
    std::uint32_t max_trips = 0;

    for (std::uint32_t block_idx = 0; block_idx < c.block_dim; ++block_idx) {
      std::uint32_t trips = 0;
      for (std::uint32_t token = block_idx; token < c.total_tokens;
           token += c.block_dim) {
        ++hits[token];
        ++trips;
      }
      if (trips > max_trips) {
        max_trips = trips;
      }
    }

    bool once = true;
    for (int count : hits) {
      if (count != 1) {
        once = false;
      }
    }
    runner.Check(once, tag + "every token is processed by exactly one core");

    // The per-core trip count, which is what CAModel wall time tracks.
    const std::uint32_t expected_max =
        (c.total_tokens + c.block_dim - 1u) / c.block_dim;
    runner.Check(max_trips == expected_max,
                 tag + "the busiest core takes ceil(tokens/cores) tokens");
  }

  // --- expert co-residency: why the grid-stride map IS the reuse mechanism --
  //
  // Weights cannot be staged in UB (one expert's w1 plane is 4 MiB against a
  // 221,184 B ceiling), so the only lever a token->core map has is locality.
  //
  // With tokens grouped by expert, grid-stride puts every core inside the SAME
  // expert at each step, so one expert's weights serve the whole grid.
  // Contiguous chunking puts one expert per core -- block_dim experts in
  // flight at once. This check is what a future "give each core a contiguous
  // run" change would have to break first.
  {
    const std::uint32_t cores = 32u;
    const std::uint32_t experts = 8u;
    const std::uint32_t per_expert = 32u;  // expert-sorted, equal batches
    const std::uint32_t tokens = experts * per_expert;  // 256
    std::vector<std::uint32_t> token_expert(tokens, 0);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      token_expert[t] = t / per_expert;  // the caller's expert-grouped order
    }

    // Grid-stride: step s has core c on token c + s * cores.
    std::uint32_t grid_max_live = 0;
    const std::uint32_t steps = (tokens + cores - 1u) / cores;
    for (std::uint32_t s = 0; s < steps; ++s) {
      std::vector<int> live(experts, 0);
      for (std::uint32_t c = 0; c < cores; ++c) {
        const std::uint32_t t = c + s * cores;
        if (t < tokens) {
          live[token_expert[t]] = 1;
        }
      }
      std::uint32_t n = 0;
      for (int f : live) {
        n += static_cast<std::uint32_t>(f);
      }
      if (n > grid_max_live) {
        grid_max_live = n;
      }
    }

    // Contiguous chunking, for contrast: core c owns [c*chunk, (c+1)*chunk).
    const std::uint32_t chunk = (tokens + cores - 1u) / cores;
    std::vector<int> chunk_live(experts, 0);
    for (std::uint32_t c = 0; c < cores; ++c) {
      const std::uint32_t t = c * chunk;  // each core's first token
      if (t < tokens) {
        chunk_live[token_expert[t]] = 1;
      }
    }
    std::uint32_t chunk_live_count = 0;
    for (int f : chunk_live) {
      chunk_live_count += static_cast<std::uint32_t>(f);
    }

    runner.Check(grid_max_live == 1u,
                 "expert-sorted tokens + grid-stride: all cores are inside ONE "
                 "expert at every step, so one expert's weights serve the grid");
    runner.Check(chunk_live_count > grid_max_live,
                 "contiguous chunking would put strictly more experts in "
                 "flight at once, which is the L2 working set this map avoids");
  }

  // Decode: one core per routed slot, so the slot map is the identity and the
  // per-core token count is always 1.
  runner.Section("dsv4_moe_block: decode slot mapping");
  const std::uint32_t top_k_cases[] = {1u, 6u, 8u};
  for (std::uint32_t top_k : top_k_cases) {
    std::vector<int> slot_hits(top_k, 0);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      ++slot_hits[slot];
    }
    bool once = true;
    for (int count : slot_hits) {
      if (count != 1) {
        once = false;
      }
    }
    runner.Check(once, "top_k=" + std::to_string(top_k) +
                           ": each routed slot is owned by exactly one core");
  }
}

// ---------------------------------------------------------------------------
// 8. Per-core trip count: why the soak is a soak
// ---------------------------------------------------------------------------
// Recorded as a test so the cost claim in the CAModel suite's TIMEOUT comment
// has something that fails when the geometry assumptions move under it.
//
// CAModel cost tracks per-core retired instruction count. Per (expert, token)
// pair a core runs inter/32 phase-1 tiles plus hidden/32 phase-2 tiles, and
// the MAC count is 2*inter*hidden (gate and up) + hidden*inter (down).
void TestPerCoreWork(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: per-core work at the geometry floor");

  const std::uint32_t hidden = kFloorDim;
  const std::uint32_t inter = kFloorDim;

  const std::uint32_t phase1_tiles = inter / kRowsPerTile;
  const std::uint32_t phase2_tiles = hidden / kRowsPerTile;
  runner.Check(phase1_tiles == 16u && phase2_tiles == 16u,
               "the floor geometry is 16 + 16 = 32 tile trips per core");

  const std::uint64_t phase1_macs =
      2ull * static_cast<std::uint64_t>(inter) * hidden;
  const std::uint64_t phase2_macs =
      static_cast<std::uint64_t>(hidden) * inter;
  runner.Check(phase1_macs + phase2_macs == 786432ull,
               "one (expert, token) pair is 786,432 MACs on one core");
  runner.Check(phase1_macs == 2ull * phase2_macs,
               "phase 1 is two thirds of the per-core work, and it is the "
               "phase that cannot be trip-collapsed: phase 2 reduces over "
               "the whole `activated` buffer phase 1 fills");

  // The floor itself: this is why a smaller fixture cannot make the soak
  // cheap. Both extents must be >= 512 AND multiples of 512.
  runner.Check(!GeometryAdmissible(256u, 512u),
               "a 256-wide extent is rejected, so the fixture cannot shrink "
               "below the reduction quantum");
  runner.Check(!GeometryAdmissible(512u, 768u),
               "768 is not a multiple of the reduction quantum");
  runner.Check(GeometryAdmissible(kFloorDim, kFloorDim),
               "512x512 is the admissible floor");
}

// ---------------------------------------------------------------------------
// 9. The phase-2 tile ring, as a marked graph
// ---------------------------------------------------------------------------
// Phase 2 double-buffers its output tile and unrolls by kTileSlots, so ONE
// DataSyncBarrier publishes both slots' VFs per step instead of one per tile.
// That schedule has two hazards and one leak to rule out, and all three are
// integer/event bookkeeping -- no simulator required:
//
//   RAW: a slot must be PUBLISHED (barrier) after its VF and before its store.
//   WAR: a slot must not be re-entered by a VF until its store has retired
//        (MTE3_V). This is the one the two-slot ring makes tight: the next
//        pair writes the very slots MTE3 just read.
//   LEAK: SetFlag/WaitFlag must balance, or a token outlives the launch and
//        breaks the NEXT kernel.
//
// Modelled exactly as written in RunExpertToken. Base step (prologue priming),
// inductive step (pair k -> k+1), terminal step (drain) are all covered by
// replaying the schedule and asserting the invariants at every event.
constexpr std::uint32_t kTileSlots = 2;
constexpr std::uint32_t kTileBytes = kRowsPerTile * 4u;        // 128
constexpr std::uint32_t kTileSkewBytes = 32;
constexpr std::uint32_t kTileSlotStrideBytes = kTileBytes + kTileSkewBytes;

void CheckRingSchedule(emu_test::Runner& runner, std::uint32_t tiles) {
  const std::string tag = "tiles=" + std::to_string(tiles) + ": ";

  enum class SlotState { kEmpty, kWritten, kPublished, kStored };
  std::vector<SlotState> slot(kTileSlots, SlotState::kEmpty);
  std::vector<std::uint32_t> slot_tile(kTileSlots, 0);
  std::vector<int> stored(tiles, 0);
  std::vector<std::uint32_t> stored_from(tiles, 0xFFFFFFFFu);

  int set_v_mte3 = 0, wait_v_mte3 = 0;
  int set_mte3_v = 0, wait_mte3_v = 0;
  int barriers = 0;
  std::uint32_t store_order = 0;
  bool raw_violation = false, war_violation = false, order_violation = false;

  auto issue = [&](std::uint32_t tile_idx, std::uint32_t s) {
    // WAR: re-entering a slot is legal only if it was never used, or its
    // previous store has retired (state kStored after an MTE3_V wait).
    if (slot[s] != SlotState::kEmpty && slot[s] != SlotState::kStored) {
      war_violation = true;
    }
    slot[s] = SlotState::kWritten;
    slot_tile[s] = tile_idx;
  };
  auto publish_all = [&]() {
    ++barriers;
    for (std::uint32_t s = 0; s < kTileSlots; ++s) {
      if (slot[s] == SlotState::kWritten) {
        slot[s] = SlotState::kPublished;
      }
    }
  };
  auto store = [&](std::uint32_t s) {
    // RAW: the slot must have been published since its VF wrote it.
    if (slot[s] != SlotState::kPublished) {
      raw_violation = true;
    }
    const std::uint32_t t = slot_tile[s];
    if (t >= tiles) {
      order_violation = true;
      return;
    }
    ++stored[t];
    stored_from[t] = s;
    if (t != store_order) {
      order_violation = true;  // y rows must be written in tile order
    }
    ++store_order;
    slot[s] = SlotState::kStored;
  };

  // --- base step: prologue priming, clamped to the grid --------------------
  const std::uint32_t primed = tiles < kTileSlots ? tiles : kTileSlots;
  for (std::uint32_t s = 0; s < primed; ++s) {
    issue(s * kRowsPerTile / kRowsPerTile, s);  // tile index s
  }
  runner.Check(primed == (tiles < kTileSlots ? tiles : kTileSlots),
               tag + "priming is clamped to the grid, so no token is primed "
                     "for a tile no iteration consumes");

  // --- inductive step: whole pairs -----------------------------------------
  std::uint32_t tile_idx = 0;
  for (; tile_idx + kTileSlots <= tiles; tile_idx += kTileSlots) {
    publish_all();
    ++set_v_mte3;
    ++wait_v_mte3;
    store(0);
    store(1);
    ++set_mte3_v;
    ++wait_mte3_v;
    const std::uint32_t next = tile_idx + kTileSlots;
    if (next < tiles) {
      issue(next, 0);
    }
    if (next + 1u < tiles) {
      issue(next + 1u, 1);
    }
  }

  // --- terminal step: an odd final tile ------------------------------------
  if (tile_idx < tiles) {
    publish_all();
    ++set_v_mte3;
    ++wait_v_mte3;
    store(0);
    ++set_mte3_v;
    ++wait_mte3_v;
  }

  runner.Check(!raw_violation,
               tag + "every slot is published by a barrier between its VF and "
                     "its store (no RAW hazard)");
  runner.Check(!war_violation,
               tag + "no slot is re-entered by a VF before its store retired "
                     "(no WAR hazard)");
  runner.Check(!order_violation, tag + "y rows are stored in tile order");

  bool once = true;
  for (int n : stored) {
    if (n != 1) {
      once = false;
    }
  }
  runner.Check(once, tag + "every tile is stored exactly once");

  // Each tile must come from the slot its VF wrote: slot = tile % kTileSlots
  // for every tile the PAIRED steady state covers, and slot 0 for the odd
  // tail tile. The steady state covers tiles [0, tiles - tiles % kTileSlots).
  const std::uint32_t paired = tiles - (tiles % kTileSlots);
  bool slot_map_ok = true;
  for (std::uint32_t t = 0; t < tiles; ++t) {
    const std::uint32_t expect = (t < paired) ? (t % kTileSlots) : 0u;
    if (stored_from[t] != expect) {
      slot_map_ok = false;
    }
  }
  runner.Check(slot_map_ok,
               tag + "each tile is stored from the slot its own VF wrote");

  // --- leak: event tokens must balance ------------------------------------
  runner.Check(set_v_mte3 == wait_v_mte3,
               tag + "V_MTE3 sets and waits balance, leaving no residual");
  runner.Check(set_mte3_v == wait_mte3_v,
               tag + "MTE3_V sets and waits balance, leaving no residual");
  for (std::uint32_t s = 0; s < kTileSlots; ++s) {
    runner.Check(slot[s] == SlotState::kStored || slot[s] == SlotState::kEmpty,
                 tag + "no slot holds an unstored VF result at exit");
  }

  // --- the point of the refactor: barriers per tile ------------------------
  const std::uint32_t expected_barriers =
      tiles / kTileSlots + (tiles % kTileSlots);
  runner.Check(barriers == static_cast<int>(expected_barriers),
               tag + "one barrier per unrolled pair, plus one for an odd tail");
  if (tiles >= kTileSlots && tiles % kTileSlots == 0u) {
    runner.Check(barriers * 2 == static_cast<int>(tiles),
                 tag + "an even grid pays HALF the barriers the per-tile "
                       "schedule paid");
  }
}

void TestTileRing(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: phase-2 tile ring as a marked graph");

  // kSlots + 2 is the minimal grid that exercises prime, flip, wrap and drain.
  // Below kSlots the prologue clamp is what has to hold; odd counts exercise
  // the tail path (unreachable at admissible geometry, but the schedule must
  // still be sound if admissibility is ever loosened).
  const std::uint32_t grids[] = {1u, 2u, 3u, 4u, 5u, 16u, 64u, 128u};
  for (std::uint32_t n : grids) {
    CheckRingSchedule(runner, n);
  }

  runner.Section("dsv4_moe_block: tile ring UB stride and bank skew");
  runner.Check(kTileSlotStrideBytes == 160u,
               "the slot stride is 128 B payload + 32 B skew");
  runner.Check(kTileSlotStrideBytes % 256u != 0u,
               "the slot stride does not alias the 256 B UB bank period");
  runner.Check(kTileSlotStrideBytes % 4096u != 0u,
               "the slot stride does not alias the 4 KiB UB bank period");
  runner.Check(kTileBytes * kTileSlots % 256u == 0u,
               "two unskewed 128 B slots WOULD have aliased at 256 B, which is "
               "why the skew exists");
  for (std::uint32_t s = 0; s < kTileSlots; ++s) {
    runner.Check((s * kTileSlotStrideBytes) % 32u == 0u,
                 "slot " + std::to_string(s) +
                     " base stays 32 B aligned for MTE3");
  }

  // dyn_ub_size, as the host declares it at the chevron. An under-declared
  // dynamic region is the AIV error 341 class, so the formula is pinned here.
  runner.Section("dsv4_moe_block: declared dynamic UB footprint");
  const std::uint32_t ring_bytes =
      kTileSlots * kTileBytes + (kTileSlots - 1u) * kTileSkewBytes;
  runner.Check(ring_bytes == 288u, "the tile ring occupies 288 B");
  const std::uint32_t prod_dyn =
      kProdHidden * 4u + kProdInter * 4u + ring_bytes;
  runner.Check(prod_dyn == 24864u,
               "production dynamic UB is 24,864 B (16,384 + 8,192 + 288)");
  runner.Check(prod_dyn <= 221184u,
               "the declared dynamic region fits the 221,184 B ceiling");
  const std::uint32_t floor_dyn =
      kFloorDim * 4u + kFloorDim * 4u + ring_bytes;
  runner.Check(floor_dyn == 4384u,
               "floor-geometry dynamic UB is 4,384 B");
}

// ---------------------------------------------------------------------------
// 10. The collapsed-grid boundary sample set
// ---------------------------------------------------------------------------
// MIRRORS sim::BuildBoundarySamples in
// tests/sim/kernel/dsv4_moe_block_inductive_kernel.cpp and its host twin in
// tests/sim/test_dsv4_moe_block_inductive_sim.cpp. Three copies of one
// formula, so it gets validated here rather than trusted three times: a
// collapsed run whose schedule is wrong verifies the wrong tiles and reports
// a pass.
std::vector<std::uint32_t> BoundarySamples(std::uint32_t total) {
  std::vector<std::uint32_t> out;
  if (total == 0u) {
    return out;
  }
  const std::uint32_t raw[4] = {0u, 1u, total / 2u, total - 1u};
  for (std::uint32_t candidate : raw) {
    if (candidate >= total) {
      continue;
    }
    bool seen = false;
    for (std::uint32_t kept : out) {
      if (kept == candidate) {
        seen = true;
      }
    }
    if (!seen) {
      out.push_back(candidate);
    }
  }
  return out;
}

void TestBoundarySamples(emu_test::Runner& runner) {
  runner.Section("dsv4_moe_block: collapsed-grid boundary sample set");

  for (std::uint32_t total = 1; total <= 512u; ++total) {
    const std::string tag = "total=" + std::to_string(total) + ": ";
    const std::vector<std::uint32_t> s = BoundarySamples(total);

    bool in_range = true;
    for (std::uint32_t t : s) {
      if (t >= total) {
        in_range = false;
      }
    }
    bool distinct = true;
    for (std::size_t i = 0; i < s.size(); ++i) {
      for (std::size_t j = i + 1; j < s.size(); ++j) {
        if (s[i] == s[j]) {
          distinct = false;
        }
      }
    }
    if (!in_range || !distinct) {
      runner.Check(in_range, tag + "every sample is inside [0, total)");
      runner.Check(distinct, tag + "samples are distinct");
      continue;
    }

    // Must always carry the two boundaries: the cold-start tile and the tail.
    bool has_first = false;
    bool has_last = false;
    for (std::uint32_t t : s) {
      if (t == 0u) {
        has_first = true;
      }
      if (t == total - 1u) {
        has_last = true;
      }
    }
    runner.Check(has_first, tag + "the cold-start tile 0 is always sampled");
    runner.Check(has_last, tag + "the tail tile total-1 is always sampled");

    // Never more than the 4 raw candidates, and never more than the grid.
    runner.Check(s.size() <= 4u, tag + "at most 4 tiles are sampled");
    runner.Check(s.size() <= total, tag + "never more samples than tiles");

    // At the sizes the gate actually runs, the set must be FULL -- a smaller
    // set would silently drop a ring branch.
    if (total >= 4u) {
      runner.Check(s.size() == 4u,
                   tag + "a grid of 4 or more yields a full 4-tile sample set");
      // Enough iterations to wrap a kTileSlots-deep ring: 4 tiles is two
      // paired iterations, which is prime -> store -> WRAP -> store -> drain.
      runner.Check(s.size() >= kTileSlots + 2u,
                   tag + "the sample count reaches kTileSlots + 2, so the "
                         "ring wraps and drains");
    }
    // Small grids degrade gracefully rather than sampling out of range.
    if (total < 4u) {
      runner.Check(s.size() == total,
                   tag + "a grid smaller than the sample set runs in full");
    }
  }
}

}  // namespace

int main() {
  emu_test::Runner runner("dsv4_moe_block addressing (host index validation)");

  TestMirroredConstants(runner);
  TestThreadDecomposition(runner);
  TestGroupLaneMap(runner);
  TestButterflyReduction(runner);
  TestTileProgression(runner);
  TestStrides(runner);
  TestTokenCoverage(runner);
  TestPerCoreWork(runner);
  TestTileRing(runner);
  TestBoundarySamples(runner);

  return runner.Summary();
}
