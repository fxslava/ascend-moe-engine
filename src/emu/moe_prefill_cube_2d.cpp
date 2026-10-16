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

// The TWO-PHASE ON-CHIP FUSED CUBE golden for the W4A16 MoE prefill FFN,
// modelling the Ascend 950PR Cube core (DaVinci v3, dav-3510).
//
// A SEPARATE TU FROM moe_prefill_w4a16.cpp, AND A SEPARATE FUNCTION. The
// 1D-strip reference there tiles only the contraction and reduces each output
// row whole; it is the AIV/Vector shape and stays that way. This file models
// the Cube's schedule, which is a different function -- not by intent but by
// arithmetic, because its reduction hierarchy is three levels deep where the
// strip's is flat. Collapsing them behind a flag would hide which one a given
// test asserts, which is the distinction the numerics argument rests on.
//
// WHAT REPLACED WHAT, because the previous schedule's cost is the reason this
// file exists in this shape. The nested (m, h) output grid computed gate and
// up INSIDE the hidden-tile loop, so the whole intermediate was recomputed
// once per H tile -- 64x the gate/up GEMM work at hidden 4096 with H_tile 64.
// This schedule pays for a resident `act[M_tile, inter]` plane instead and
// evaluates gate and up EXACTLY ONCE. The two identities that say so are
// observed, not asserted:
//
//     trace.gate_up_macs          == 2 * M * inter * hidden
//     trace.act_elements_written  == M * inter
//
// The grid is now ONE-DIMENSIONAL, over token tiles. Phase 2's h0 loop lives
// inside a tile body because `act` must exist in full before any of it runs;
// its direction is still irrelevant to the answer, and `CubeTileVisit`
// carries that so the suite can check it.
//
// See include/moe/emu/dsv4_layers.h for the schedule in full, the three-level
// decomposition the sub-tile extents name, what is deliberately NOT modelled
// (mmad product order, MTE timing, which on-chip buffer holds the level-2
// subtotal, the device's Silu), and the residual DMA cost this schedule keeps:
// with r outer and k inner, the X macro-tile is re-read ceil(inter / R_sub)
// times per token tile, which `CubeTileTrace::x_fragment_reads` reports.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("moe_prefill_cube_2d: " + message);
  }
}

// silu(z) = z * sigmoid(z). Byte-for-byte the guard in moe_block.cpp and
// moe_prefill_w4a16.cpp: the oracles of this family must not disagree about
// the activation, only about the order of the reductions around it.
float Silu(float z) noexcept {
  if (z < -80.0F) {
    return 0.0F;
  }
  return z / (1.0F + std::exp(-z));
}

// ONE mmad's worth of reduction and the accumulation level above it.
//
//   total = sum over fractals of ( sum over the <= 16 products in the fractal )
//
// `a` and `b` are contiguous BF16-valued runs of `extent` elements, and
// `extent` is ONE SUB-TILE of the contraction axis -- K_sub in phase 1,
// R_sub2 in phase 2. The level above (summing sub-tiles over the whole axis)
// is the caller's accumulator, which is why this function sees only two of
// the three levels.
//
// Every product is exact in FP32 -- two BF16 factors need at most 16
// significand bits -- so this function's only arithmetic character is the
// SHAPE of its additions, which is the point of it.
float FractalDot(const float* a, const float* b, int extent) noexcept {
  float total = 0.0F;
  for (int base = 0; base < extent; base += kCubeFractal) {
    const int span = std::min(kCubeFractal, extent - base);
    float fragment = 0.0F;
    for (int i = 0; i < span; ++i) {
      fragment += a[base + i] * b[base + i];
    }
    total += fragment;
  }
  return total;
}

int CeilDiv(int value, int divisor) noexcept {
  return (value + divisor - 1) / divisor;
}

bool IsFractalMultiple(int extent) noexcept {
  return extent > 0 && extent % kCubeFractal == 0;
}

// The token-tile list, in the requested visit order. Built explicitly rather
// than walked as a loop so that the order is a VALUE the caller can vary --
// which is how the suite turns "the token tiles are independent" into an
// executable check instead of a claim about the source.
std::vector<int> BuildTokenTileOrder(int total_tokens,
                                     const CubeTileGeometry& geometry) {
  const int m_tiles = CeilDiv(total_tokens, geometry.m_tile);
  std::vector<int> tiles(static_cast<std::size_t>(m_tiles));
  for (int i = 0; i < m_tiles; ++i) {
    tiles[static_cast<std::size_t>(i)] = i;
  }
  if (geometry.visit == CubeTileVisit::kReversed) {
    std::reverse(tiles.begin(), tiles.end());
  } else if (geometry.visit == CubeTileVisit::kShuffled) {
    std::mt19937_64 rng(geometry.visit_seed);
    std::shuffle(tiles.begin(), tiles.end(), rng);
  }
  return tiles;
}

// Phase 2's h0 blocks run backwards for two of the four visit orders. The
// direction cannot change the answer -- the (m0, h0) output blocks share no
// state either -- and the suite checks that rather than taking it on trust.
bool Phase2Descending(CubeTileVisit visit) noexcept {
  return visit == CubeTileVisit::kColumnMajor ||
         visit == CubeTileVisit::kReversed;
}

void MergePeak(CubeBufferFootprint* into,
               const CubeBufferFootprint& observed) noexcept {
  into->l0a_bytes = std::max(into->l0a_bytes, observed.l0a_bytes);
  into->l0b_bytes = std::max(into->l0b_bytes, observed.l0b_bytes);
  into->l0c_bytes = std::max(into->l0c_bytes, observed.l0c_bytes);
  into->l1_bytes = std::max(into->l1_bytes, observed.l1_bytes);
  into->ub_bytes = std::max(into->ub_bytes, observed.ub_bytes);
}

void TrackMin(int* slot, int value) noexcept {
  *slot = (*slot == 0) ? value : std::min(*slot, value);
}

void RecordPhase1(CubeTileTrace* trace, int m, int r, int k, int inter,
                  int depth) noexcept {
  if (trace == nullptr) {
    return;
  }
  trace->max_m_extent = std::max(trace->max_m_extent, m);
  trace->max_r_extent = std::max(trace->max_r_extent, r);
  trace->max_k_extent = std::max(trace->max_k_extent, k);
  TrackMin(&trace->min_m_extent, m);
  TrackMin(&trace->min_r_extent, r);
  TrackMin(&trace->min_k_extent, k);
  const CubeBufferFootprint observed =
      moe_prefill_cube_phase1_footprint(m, r, k, inter, depth);
  MergePeak(&trace->phase1_peak, observed);
  MergePeak(&trace->peak, observed);
}

void RecordPhase2(CubeTileTrace* trace, int m, int h, int r_down, int inter,
                  int depth) noexcept {
  if (trace == nullptr) {
    return;
  }
  trace->max_m_extent = std::max(trace->max_m_extent, m);
  trace->max_h_extent = std::max(trace->max_h_extent, h);
  trace->max_r_down_extent = std::max(trace->max_r_down_extent, r_down);
  TrackMin(&trace->min_m_extent, m);
  TrackMin(&trace->min_h_extent, h);
  TrackMin(&trace->min_r_down_extent, r_down);
  const CubeBufferFootprint observed =
      moe_prefill_cube_phase2_footprint(m, h, r_down, inter, depth);
  MergePeak(&trace->phase2_peak, observed);
  MergePeak(&trace->peak, observed);
}

}  // namespace

// ---------------------------------------------------------------------------
// Footprints and admissibility
// ---------------------------------------------------------------------------

CubeBufferFootprint moe_prefill_cube_phase1_footprint(int m_extent, int r_extent,
                                                      int k_extent, int inter,
                                                      int buffer_depth) noexcept {
  CubeBufferFootprint out;
  if (m_extent <= 0 || r_extent <= 0 || k_extent <= 0 || inter <= 0 ||
      buffer_depth <= 0) {
    return out;
  }
  const std::size_t m = static_cast<std::size_t>(m_extent);
  const std::size_t r = static_cast<std::size_t>(r_extent);
  const std::size_t k = static_cast<std::size_t>(k_extent);
  const std::size_t n = static_cast<std::size_t>(inter);
  const std::size_t depth = static_cast<std::size_t>(buffer_depth);
  constexpr std::size_t kBf16 = 2;
  constexpr std::size_t kFp32 = 4;

  // L0A: the A matrix of the gate/up GEMMs is X[M, K_sub]. ONE fragment serves
  // both, which is the whole reason k is the inner loop.
  out.l0a_bytes = depth * kBf16 * m * k;
  // L0B: W1[R_sub, K_sub] and W3[R_sub, K_sub] feed the Cube one after the
  // other and share a slot, so the per-slot B fragment is R_sub * K_sub.
  out.l0b_bytes = depth * kBf16 * r * k;
  // The gate and up accumulators, charged TOGETHER and not as a max: they are
  // produced side by side from the same X fragment and both must be live when
  // the SwiGLU reads them.
  const std::size_t accumulators = kFp32 * 2U * m * r;
  out.l0c_bytes = accumulators;
  // L1 stages everything streamed for one (r0, k0) step: both weight
  // sub-tiles and the X sub-tile.
  out.l1_bytes = depth * kBf16 * (2U * r * k + m * k);
  // UB holds the resident `act` plane for the WHOLE inter axis -- that is the
  // residency the schedule is built around, and the term that puts `inter` in
  // the budget at all. It is charged at depth 1 whatever buffer_depth says:
  // it is one plane, not a streamed fragment.
  //
  // The accumulators are charged to UB **as well as** to L0C, deliberately.
  // The idealized schedule keeps them L0C-resident; a kernel driven through
  // the high-level Matmul API with a per-sub-tile B cannot, because L0C does
  // not survive between separately driven single-core GEMMs, so it holds them
  // in UB instead. Both are admissible drivers of the same arithmetic, and a
  // budget that charged only one of the two would declare a geometry feasible
  // for a driver nobody can write.
  //
  // THE DEQUANT STAGING IS CHARGED TOO, and it has to be: on-the-fly
  // dequantization is a VECTOR operation, so the MXFP4 sub-tile lands in UB as
  // packed nibbles (r*k/2 bytes) and is expanded there to BF16 (r*k*2 bytes)
  // before anything reaches L1. The host model holds ONE float array where the
  // device holds a UB staging copy and an L1 copy; the footprint charges both,
  // because the device pays for both. W1 and W3 reuse one staging buffer, as
  // they reuse one L0B slot.
  out.ub_bytes = kBf16 * m * n + accumulators + kBf16 * r * k + (r * k) / 2U;
  return out;
}

CubeBufferFootprint moe_prefill_cube_phase2_footprint(int m_extent, int h_extent,
                                                      int r_down_extent, int inter,
                                                      int buffer_depth) noexcept {
  CubeBufferFootprint out;
  if (m_extent <= 0 || h_extent <= 0 || r_down_extent <= 0 || inter <= 0 ||
      buffer_depth <= 0) {
    return out;
  }
  const std::size_t m = static_cast<std::size_t>(m_extent);
  const std::size_t h = static_cast<std::size_t>(h_extent);
  const std::size_t rd = static_cast<std::size_t>(r_down_extent);
  const std::size_t n = static_cast<std::size_t>(inter);
  const std::size_t depth = static_cast<std::size_t>(buffer_depth);
  constexpr std::size_t kBf16 = 2;
  constexpr std::size_t kFp32 = 4;

  // L0A: the down GEMM's A matrix is act[M, R_sub2], taken straight from the
  // resident UB plane -- there is no L1 hop for it, which is why l1_bytes
  // below charges only the weight sub-tile.
  out.l0a_bytes = depth * kBf16 * m * rd;
  out.l0b_bytes = depth * kBf16 * h * rd;
  const std::size_t accumulator = kFp32 * m * h;
  out.l0c_bytes = accumulator;
  out.l1_bytes = depth * kBf16 * h * rd;
  // `act` is still resident: phase 2 is what it was kept for. The dequant
  // staging is charged on the same argument as phase 1's, at phase 2's own
  // sub-tile shape W2[H_sub, R_sub2].
  out.ub_bytes =
      kBf16 * m * n + accumulator + kBf16 * h * rd + (h * rd) / 2U;
  return out;
}

CubeBufferFootprint moe_prefill_cube_tile_footprint(
    const CubeTileGeometry& geometry, int inter) noexcept {
  CubeBufferFootprint out;
  MergePeak(&out, moe_prefill_cube_phase1_footprint(
                      geometry.m_tile, geometry.r_tile, geometry.k_tile, inter,
                      geometry.buffer_depth));
  MergePeak(&out, moe_prefill_cube_phase2_footprint(
                      geometry.m_tile, geometry.h_tile, geometry.r_tile_down,
                      inter, geometry.buffer_depth));
  return out;
}

bool moe_prefill_cube_tile_geometry_admissible(const CubeTileGeometry& geometry,
                                               int inter) noexcept {
  if (!IsFractalMultiple(geometry.m_tile) ||
      !IsFractalMultiple(geometry.h_tile) ||
      !IsFractalMultiple(geometry.r_tile) ||
      !IsFractalMultiple(geometry.k_tile) ||
      !IsFractalMultiple(geometry.r_tile_down)) {
    return false;
  }
  if (geometry.buffer_depth < 1 || geometry.buffer_depth > 2) {
    return false;
  }
  if (inter <= 0) {
    return false;
  }
  return moe_prefill_cube_tile_footprint(geometry, inter).Fits();
}

std::size_t moe_prefill_cube_2d_scratch_floats(const CubeTileGeometry& geometry,
                                               int inter) noexcept {
  if (geometry.m_tile <= 0 || geometry.h_tile <= 0 || geometry.r_tile <= 0 ||
      geometry.k_tile <= 0 || geometry.r_tile_down <= 0 || inter <= 0) {
    return 0;
  }
  const std::size_t m = static_cast<std::size_t>(geometry.m_tile);
  const std::size_t h = static_cast<std::size_t>(geometry.h_tile);
  const std::size_t rt = static_cast<std::size_t>(geometry.r_tile);
  const std::size_t kt = static_cast<std::size_t>(geometry.k_tile);
  const std::size_t rd = static_cast<std::size_t>(geometry.r_tile_down);
  const std::size_t n = static_cast<std::size_t>(inter);
  // The host mirror of the modelled buffer set, and nothing else:
  //   phase 1   x_frag[m, kt]   w1_frag[rt, kt]  w3_frag[rt, kt]
  //             gate[m, rt]     up[m, rt]
  //   resident  act[m, inter]                        <- the UB plane
  //   phase 2   a_frag[m, rd]   w2_frag[h, rd]   y_acc[m, h]
  // Every term is a product of SUB-TILE extents except act[m, inter], and
  // that exception is the schedule. Nothing here scales with `hidden` or with
  // the token count, and act grows with M_TILE times inter -- not with M
  // times inter, which is the no-global-workspace claim as a formula a test
  // can check.
  return m * kt + 2U * rt * kt + 2U * m * rt + m * n + m * rd + h * rd + m * h;
}

// ---------------------------------------------------------------------------
// The golden
// ---------------------------------------------------------------------------

void moe_prefill_cube_flash_2d_w4a16_cpu(const Mxfp4Matrix* w1,
                                         const Mxfp4Matrix* w3,
                                         const Mxfp4Matrix* w2,
                                         const float* x_tokens,
                                         const std::int32_t* token_to_expert,
                                         int total_tokens, int num_experts,
                                         float* y_tokens, float limit,
                                         const CubeTileGeometry& geometry,
                                         CubeTileTrace* trace,
                                         std::vector<float>* scratch) {
  Require(w1 != nullptr && w3 != nullptr && w2 != nullptr,
          "prefill expert weight array is null");
  Require(x_tokens != nullptr && y_tokens != nullptr, "prefill operand is null");
  Require(token_to_expert != nullptr, "prefill routing operand is null");
  Require(scratch != nullptr, "prefill scratch is null");
  Require(total_tokens > 0 && num_experts > 0, "prefill has an empty extent");

  const int hidden = w2[0].rows;
  const int inter = w1[0].rows;
  // Every expert must share one geometry: the device addresses expert e at a
  // FIXED stride, so a ragged expert set reads the wrong bytes rather than
  // merely running slowly.
  for (int e = 0; e < num_experts; ++e) {
    Require(w1[e].rows == inter && w3[e].rows == inter,
            "every expert must share the intermediate size");
    Require(w2[e].rows == hidden, "every expert must share the hidden size");
    Require(w1[e].cols == hidden && w3[e].cols == hidden,
            "w1/w3 must reduce over the hidden size");
    Require(w2[e].cols == inter, "w2 must reduce over the intermediate size");
    Require(w1[e].packed != nullptr && w1[e].scale != nullptr &&
                w3[e].packed != nullptr && w3[e].scale != nullptr &&
                w2[e].packed != nullptr && w2[e].scale != nullptr,
            "an expert's weight matrix has a null payload");
  }
  Require(hidden > 0 && inter > 0, "the expert bank has an empty extent");
  Require(hidden % kMxfp4GroupSize == 0 && inter % kMxfp4GroupSize == 0,
          "both reduction axes must be whole 32-element MXFP4 groups");

  // The geometry is rejected HERE, before a single tile runs. On device an
  // over-budget tile is a compile- or load-time failure; on the host the
  // equivalent honesty is a throw, because the alternative -- clamping the
  // sub-tile to something that fits -- would silently verify a schedule
  // nobody asked for.
  //
  // IT IS CHECKED AFTER `inter` IS KNOWN, and it has to be: the resident
  // `act[M_tile, inter]` plane is now part of the budget, so admissibility is
  // a property of the geometry AND the problem, not of the geometry alone.
  // Nothing has been written to `y_tokens` at this point, so the rejection is
  // still total.
  Require(moe_prefill_cube_tile_geometry_admissible(geometry, inter),
          "the Cube sub-tile geometry is not admissible at this inter: every "
          "extent must be a positive multiple of the 16x16x16 fractal, "
          "buffer_depth must be 1 or 2, and the peak of the two phases must "
          "fit L0A/L0B/L0C/L1 and the 248 KiB UB that the resident act plane "
          "is charged against");

  const CubeTileGeometry& g = geometry;
  scratch->assign(moe_prefill_cube_2d_scratch_floats(g, inter), 0.0F);
  float* cursor = scratch->data();
  const auto take = [&cursor](std::size_t count) {
    float* out = cursor;
    cursor += count;
    return out;
  };
  const std::size_t m = static_cast<std::size_t>(g.m_tile);
  const std::size_t h = static_cast<std::size_t>(g.h_tile);
  const std::size_t rt = static_cast<std::size_t>(g.r_tile);
  const std::size_t kt = static_cast<std::size_t>(g.k_tile);
  const std::size_t rd = static_cast<std::size_t>(g.r_tile_down);
  const std::size_t nn = static_cast<std::size_t>(inter);
  float* x_frag = take(m * kt);   // phase 1 L0A: X[M_tile, K_sub]
  float* w1_frag = take(rt * kt); // phase 1 L0B: W1[R_sub, K_sub]
  float* w3_frag = take(rt * kt); // phase 1 L0B: W3[R_sub, K_sub]
  float* gate = take(m * rt);     // phase 1 accumulator, FP32
  float* up = take(m * rt);       // phase 1 accumulator, FP32
  float* act = take(m * nn);      // RESIDENT in UB, BF16-valued
  float* a_frag = take(m * rd);   // phase 2 L0A: act[M_tile, R_sub2]
  float* w2_frag = take(h * rd);  // phase 2 L0B: W2[H_sub, R_sub2]
  float* y_acc = take(m * h);     // phase 2 accumulator, FP32

  if (trace != nullptr) {
    *trace = CubeTileTrace{};
  }

  for (int t = 0; t < total_tokens; ++t) {
    const std::int32_t e = token_to_expert[t];
    Require(e >= 0 && e < num_experts, "token's expert id is out of range");
  }

  const std::vector<int> tiles = BuildTokenTileOrder(total_tokens, g);
  const bool h_descending = Phase2Descending(g.visit);
  const int h_blocks = CeilDiv(hidden, g.h_tile);

  for (const int tile : tiles) {
    const int m0 = tile * g.m_tile;
    const int m_extent = std::min(g.m_tile, total_tokens - m0);

    // =====================================================================
    // PHASE 1 -- gate / up / clamped SwiGLU, into the resident `act` plane
    // =====================================================================
    // The whole intermediate is produced here, ONCE, and nothing in phase 2
    // can re-enter this loop. That is the property the trace's
    // `gate_up_macs` and `act_elements_written` counters exist to witness.
    for (int r0 = 0; r0 < inter; r0 += g.r_tile) {
      const int r_extent = std::min(g.r_tile, inter - r0);

      for (int mm = 0; mm < m_extent; ++mm) {
        for (int rr = 0; rr < r_extent; ++rr) {
          const std::size_t i =
              static_cast<std::size_t>(mm) * rt + static_cast<std::size_t>(rr);
          gate[i] = 0.0F;
          up[i] = 0.0F;
        }
      }

      for (int k0 = 0; k0 < hidden; k0 += g.k_tile) {
        const int k_extent = std::min(g.k_tile, hidden - k0);
        RecordPhase1(trace, m_extent, r_extent, k_extent, inter,
                     g.buffer_depth);

        // X[M_tile, K_sub] -> L0A, read ONCE and used for BOTH the W1 and the
        // W3 contraction. That reuse is why k is the inner loop and why the
        // gate/up multiplier is 1.0x rather than 2.0x in DMA terms as well as
        // in MACs. Rounded to BF16 on the way in; the rounding is idempotent,
        // so doing it per fragment is the same function as rounding the whole
        // row once.
        for (int mm = 0; mm < m_extent; ++mm) {
          const std::size_t row = static_cast<std::size_t>(m0 + mm) *
                                  static_cast<std::size_t>(hidden);
          for (int kk = 0; kk < k_extent; ++kk) {
            x_frag[static_cast<std::size_t>(mm) * kt +
                   static_cast<std::size_t>(kk)] =
                bf16_round_cpu(x_tokens[row + static_cast<std::size_t>(k0 + kk)]);
          }
        }
        if (trace != nullptr) {
          ++trace->x_fragment_reads;
        }

        // The weight sub-tiles. Dequantized here and nowhere else: there is no
        // workspace holding W1/W3/W2 in BF16, at any scale, which is the whole
        // point of the pivot this model verifies. Both W1 and W3 are staged
        // for the same (r0, k0) sub-tile because they reduce against the SAME
        // X fragment; on device they occupy one L0B slot one after the other,
        // which is how the footprint charges them.
        //
        // STAGED PER EXPERT, NOT PER TOKEN. The Cube's reason for existing is
        // that one B fragment serves all M rows, and a kernel running on a
        // grouped dispatch stages each sub-tile exactly once per token tile.
        // This golden does NOT assume the grouping -- it restages when a tile
        // straddles an expert boundary -- because a golden that assumed it
        // could not tell a wrong-expert read from a legitimate one. The memo
        // makes the common case cost what it costs on hardware, and the trace
        // counts the stagings so the difference is visible.
        int staged_expert = -1;
        for (int mm = 0; mm < m_extent; ++mm) {
          const int ex = token_to_expert[static_cast<std::size_t>(m0 + mm)];
          if (ex != staged_expert) {
            const std::size_t e2 = static_cast<std::size_t>(ex);
            for (int rr = 0; rr < r_extent; ++rr) {
              for (int kk = 0; kk < k_extent; ++kk) {
                const std::size_t i = static_cast<std::size_t>(rr) * kt +
                                      static_cast<std::size_t>(kk);
                w1_frag[i] =
                    mxfp4_dequant_element_bf16_cpu(w1[e2], r0 + rr, k0 + kk);
                w3_frag[i] =
                    mxfp4_dequant_element_bf16_cpu(w3[e2], r0 + rr, k0 + kk);
              }
            }
            staged_expert = ex;
            if (trace != nullptr) {
              ++trace->gate_up_fragment_stagings;
            }
          }
          // mmad: C += A * B^T, the fractal reduction inside, FP32 out.
          for (int rr = 0; rr < r_extent; ++rr) {
            const std::size_t ci =
                static_cast<std::size_t>(mm) * rt + static_cast<std::size_t>(rr);
            gate[ci] += FractalDot(x_frag + static_cast<std::size_t>(mm) * kt,
                                   w1_frag + static_cast<std::size_t>(rr) * kt,
                                   k_extent);
            up[ci] += FractalDot(x_frag + static_cast<std::size_t>(mm) * kt,
                                 w3_frag + static_cast<std::size_t>(rr) * kt,
                                 k_extent);
          }
        }
        if (trace != nullptr) {
          trace->gate_up_macs += 2ULL * static_cast<std::uint64_t>(m_extent) *
                                 static_cast<std::uint64_t>(r_extent) *
                                 static_cast<std::uint64_t>(k_extent);
        }
      }

      // The SwiGLU, on the Vector unit, straight into the resident plane. The
      // clamp is ASYMMETRIC -- the gate bounded ABOVE only, the up projection
      // on both sides -- and `act` is stored BF16 because it is the down
      // GEMM's A matrix and the Cube reads BF16.
      for (int mm = 0; mm < m_extent; ++mm) {
        for (int rr = 0; rr < r_extent; ++rr) {
          const std::size_t i =
              static_cast<std::size_t>(mm) * rt + static_cast<std::size_t>(rr);
          const float clamped_gate = std::min(gate[i], limit);
          const float clamped_up = std::min(limit, std::max(-limit, up[i]));
          act[static_cast<std::size_t>(mm) * nn +
              static_cast<std::size_t>(r0 + rr)] =
              bf16_round_cpu(Silu(clamped_gate) * clamped_up);
        }
      }
      if (trace != nullptr) {
        trace->act_elements_written += static_cast<std::uint64_t>(m_extent) *
                                       static_cast<std::uint64_t>(r_extent);
      }
    }

    // =====================================================================
    // PHASE 2 -- the down projection, reading `act` and nothing else
    // =====================================================================
    for (int hb = 0; hb < h_blocks; ++hb) {
      const int h_index = h_descending ? (h_blocks - 1 - hb) : hb;
      const int h0 = h_index * g.h_tile;
      const int h_extent = std::min(g.h_tile, hidden - h0);

      // The accumulator starts at zero and stays live for the whole r loop.
      for (int mm = 0; mm < m_extent; ++mm) {
        for (int hh = 0; hh < h_extent; ++hh) {
          y_acc[static_cast<std::size_t>(mm) * h +
                static_cast<std::size_t>(hh)] = 0.0F;
        }
      }

      for (int r0 = 0; r0 < inter; r0 += g.r_tile_down) {
        const int rd_extent = std::min(g.r_tile_down, inter - r0);
        RecordPhase2(trace, m_extent, h_extent, rd_extent, inter,
                     g.buffer_depth);

        // act[M_tile, R_sub2] -> L0A. A pure move out of the resident plane:
        // the values are already BF16, so this cannot round and therefore
        // cannot change the answer. It is modelled because L0A is a buffer of
        // its own and the footprint charges it.
        for (int mm = 0; mm < m_extent; ++mm) {
          for (int rr = 0; rr < rd_extent; ++rr) {
            a_frag[static_cast<std::size_t>(mm) * rd +
                   static_cast<std::size_t>(rr)] =
                act[static_cast<std::size_t>(mm) * nn +
                    static_cast<std::size_t>(r0 + rr)];
          }
        }

        int staged_down_expert = -1;
        for (int mm = 0; mm < m_extent; ++mm) {
          const int ex = token_to_expert[static_cast<std::size_t>(m0 + mm)];
          if (ex != staged_down_expert) {
            const std::size_t e2 = static_cast<std::size_t>(ex);
            for (int hh = 0; hh < h_extent; ++hh) {
              for (int rr = 0; rr < rd_extent; ++rr) {
                w2_frag[static_cast<std::size_t>(hh) * rd +
                        static_cast<std::size_t>(rr)] =
                    mxfp4_dequant_element_bf16_cpu(w2[e2], h0 + hh, r0 + rr);
              }
            }
            staged_down_expert = ex;
            if (trace != nullptr) {
              ++trace->down_fragment_stagings;
            }
          }
          for (int hh = 0; hh < h_extent; ++hh) {
            y_acc[static_cast<std::size_t>(mm) * h +
                  static_cast<std::size_t>(hh)] +=
                FractalDot(a_frag + static_cast<std::size_t>(mm) * rd,
                           w2_frag + static_cast<std::size_t>(hh) * rd,
                           rd_extent);
          }
        }
        if (trace != nullptr) {
          trace->down_macs += static_cast<std::uint64_t>(m_extent) *
                              static_cast<std::uint64_t>(h_extent) *
                              static_cast<std::uint64_t>(rd_extent);
        }
      }

      // One write to GM per (m0, h0) block, at the end. Assignment, not
      // accumulation: the block owns its slice of the output and nothing else
      // touches it.
      for (int mm = 0; mm < m_extent; ++mm) {
        const std::size_t row = static_cast<std::size_t>(m0 + mm) *
                                static_cast<std::size_t>(hidden);
        for (int hh = 0; hh < h_extent; ++hh) {
          y_tokens[row + static_cast<std::size_t>(h0 + hh)] =
              y_acc[static_cast<std::size_t>(mm) * h +
                    static_cast<std::size_t>(hh)];
        }
      }
      if (trace != nullptr) {
        ++trace->output_tiles;
        trace->output_elements_written += static_cast<std::uint64_t>(m_extent) *
                                          static_cast<std::uint64_t>(h_extent);
      }
    }
    if (trace != nullptr) {
      ++trace->token_tiles;
    }
  }
}

}  // namespace moe::emu
