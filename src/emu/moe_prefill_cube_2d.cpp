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

// The 2D/3D BLOCK-TILED CUBE golden for the W4A16 MoE prefill FFN, modelling
// the Ascend 950PR Cube core (DaVinci v3, dav-3510).
//
// A SEPARATE TU FROM moe_prefill_w4a16.cpp, AND A SEPARATE FUNCTION. The
// 1D-strip reference there tiles only the contraction and reduces each output
// row whole; it is the AIV/Vector shape and stays that way. This file models
// the Cube's schedule, which is a different function -- not by intent but by
// arithmetic, because its reduction hierarchy is three levels deep where the
// strip's is flat. Collapsing them behind a flag would hide which one a given
// test asserts, which is the distinction the numerics argument rests on.
//
// WHAT IS MODELLED, AND WHAT IS DELIBERATELY NOT
//
// Modelled, because each one can change the answer or the feasibility:
//   * the 2-D output grid and the fact that tiles share NOTHING -- the visit
//     order is a parameter precisely so a test can prove it does not matter;
//   * the L0C-resident accumulator, written to GM exactly once per tile;
//   * the three-level reduction (fractal C0 = 16, then the K/R block, then
//     the axis), each level accumulating in FP32;
//   * on-the-fly dequantization into per-block fragments -- the host scratch
//     IS the modelled buffer set, so "no global workspace" is a property of
//     this file and not a promise about a future kernel;
//   * the BF16 dtypes the budget is charged at, even though the host holds
//     those fragments in float.
//
// NOT modelled, stated because a golden that pretends otherwise stops being
// one:
//   * the order of the 16 products inside one mmad. The ISA does not specify
//     it; natural order is one admissible choice, and the error bound the
//     suite derives covers any of them.
//   * MTE1/MTE2 timing, BufferID ownership, double-buffer ping-pong. Buffer
//     depth enters only as a multiplier on the footprint budget, which is the
//     one way it can make a geometry infeasible.
//   * the device's `Silu`. It is a vector intrinsic over the hardware's
//     exp/reciprocal, not std::exp, and its error is not derivable from the
//     format -- the one term a device comparison must measure rather than
//     derive.
//
// THE SCHEDULE'S COST, recorded here because it is a design consequence the
// kernel author must see before writing the kernel rather than after: gate and
// up are computed INSIDE the (m, h) tile loop, so the whole intermediate is
// recomputed once per H tile. At hidden 4096 with H_tile 64 that is 64x the
// gate/up GEMM work. `CubeTileTrace::intermediate_recompute_factor` reports
// it, and H_tile = hidden removes it entirely at the cost of a wider L0C
// accumulator. This file implements the schedule as specified; it does not
// quietly hoist the intermediate out of the loop, because that would be a
// different schedule with a different footprint and a different answer.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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

// ONE mmad's worth of reduction and the two accumulation levels above it.
//
//   total = sum over fractals of ( sum over the <= 16 products in the fractal )
//
// `a` and `b` are contiguous BF16-valued runs of `extent` elements. Every
// product is exact in FP32 -- two BF16 factors need at most 16 significand
// bits -- so this function's only arithmetic character is the SHAPE of its
// additions, which is the point of it.
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

// The tile list, in the requested visit order. Built explicitly rather than
// walked as a nested loop so that the order is a VALUE the caller can vary --
// which is how the suite turns "the tiles are independent" into an executable
// check instead of a claim about the source.
std::vector<std::pair<int, int>> BuildTileOrder(int total_tokens, int hidden,
                                                const CubeTileGeometry& geometry) {
  const int m_tiles = CeilDiv(total_tokens, geometry.m_tile);
  const int h_tiles = CeilDiv(hidden, geometry.h_tile);
  std::vector<std::pair<int, int>> tiles;
  tiles.reserve(static_cast<std::size_t>(m_tiles) *
                static_cast<std::size_t>(h_tiles));
  if (geometry.visit == CubeTileVisit::kColumnMajor) {
    for (int hi = 0; hi < h_tiles; ++hi) {
      for (int mi = 0; mi < m_tiles; ++mi) {
        tiles.emplace_back(mi * geometry.m_tile, hi * geometry.h_tile);
      }
    }
  } else {
    for (int mi = 0; mi < m_tiles; ++mi) {
      for (int hi = 0; hi < h_tiles; ++hi) {
        tiles.emplace_back(mi * geometry.m_tile, hi * geometry.h_tile);
      }
    }
  }
  if (geometry.visit == CubeTileVisit::kReversed) {
    std::reverse(tiles.begin(), tiles.end());
  } else if (geometry.visit == CubeTileVisit::kShuffled) {
    std::mt19937_64 rng(geometry.visit_seed);
    std::shuffle(tiles.begin(), tiles.end(), rng);
  }
  return tiles;
}

void RecordExtents(CubeTileTrace* trace, int m, int h, int r, int k) noexcept {
  if (trace == nullptr) {
    return;
  }
  trace->max_m_extent = std::max(trace->max_m_extent, m);
  trace->max_h_extent = std::max(trace->max_h_extent, h);
  trace->max_r_extent = std::max(trace->max_r_extent, r);
  trace->max_k_extent = std::max(trace->max_k_extent, k);
  trace->min_m_extent = trace->min_m_extent == 0
                            ? m
                            : std::min(trace->min_m_extent, m);
  trace->min_h_extent = trace->min_h_extent == 0
                            ? h
                            : std::min(trace->min_h_extent, h);
  trace->min_r_extent = trace->min_r_extent == 0
                            ? r
                            : std::min(trace->min_r_extent, r);
  trace->min_k_extent = trace->min_k_extent == 0
                            ? k
                            : std::min(trace->min_k_extent, k);
  const CubeBufferFootprint observed =
      moe_prefill_cube_fragment_footprint(m, h, r, k, 1);
  trace->peak.l0a_bytes = std::max(trace->peak.l0a_bytes, observed.l0a_bytes);
  trace->peak.l0b_bytes = std::max(trace->peak.l0b_bytes, observed.l0b_bytes);
  trace->peak.l0c_bytes = std::max(trace->peak.l0c_bytes, observed.l0c_bytes);
  trace->peak.l1_bytes = std::max(trace->peak.l1_bytes, observed.l1_bytes);
}

}  // namespace

// ---------------------------------------------------------------------------
// Footprints and admissibility
// ---------------------------------------------------------------------------

CubeBufferFootprint moe_prefill_cube_fragment_footprint(int m_extent, int h_extent,
                                                        int r_extent, int k_extent,
                                                        int buffer_depth) noexcept {
  CubeBufferFootprint out;
  if (m_extent <= 0 || h_extent <= 0 || r_extent <= 0 || k_extent <= 0 ||
      buffer_depth <= 0) {
    return out;
  }
  const std::size_t m = static_cast<std::size_t>(m_extent);
  const std::size_t h = static_cast<std::size_t>(h_extent);
  const std::size_t r = static_cast<std::size_t>(r_extent);
  const std::size_t k = static_cast<std::size_t>(k_extent);
  const std::size_t depth = static_cast<std::size_t>(buffer_depth);
  constexpr std::size_t kBf16 = 2;
  constexpr std::size_t kFp32 = 4;

  // L0A holds the A matrix of whichever phase is running: X[M, K] for the
  // gate/up GEMMs, act[M, R] for the down GEMM. The phases do not overlap, so
  // the charge is the larger of the two, times the buffer depth.
  out.l0a_bytes = depth * kBf16 * m * std::max(k, r);
  // L0B likewise: W1[R, K] and W3[R, K] feed the Cube one after the other and
  // share a slot, so the per-slot B fragment is max(R*K, H*R).
  out.l0b_bytes = depth * kBf16 * std::max(r * k, h * r);
  // L0C holds all three accumulators AT ONCE. Y_acc lives across the whole r
  // loop by construction -- that is what makes the tile independent -- while
  // gate and up are produced inside it. A max() here would be the one modelling
  // error that lets an infeasible geometry look feasible.
  out.l0c_bytes = kFp32 * (m * h + 2U * m * r);
  // L1 stages everything streamed for one r step: both weight blocks for the
  // gate/up GEMMs, the down GEMM's weight block, and the X block.
  out.l1_bytes = depth * kBf16 * (2U * r * k + h * r + m * k);
  return out;
}

CubeBufferFootprint moe_prefill_cube_tile_footprint(
    const CubeTileGeometry& geometry) noexcept {
  return moe_prefill_cube_fragment_footprint(geometry.m_tile, geometry.h_tile,
                                             geometry.r_tile, geometry.k_tile,
                                             geometry.buffer_depth);
}

bool moe_prefill_cube_tile_geometry_admissible(
    const CubeTileGeometry& geometry) noexcept {
  if (!IsFractalMultiple(geometry.m_tile) ||
      !IsFractalMultiple(geometry.h_tile) ||
      !IsFractalMultiple(geometry.r_tile) ||
      !IsFractalMultiple(geometry.k_tile)) {
    return false;
  }
  if (geometry.buffer_depth < 1 || geometry.buffer_depth > 2) {
    return false;
  }
  return moe_prefill_cube_tile_footprint(geometry).Fits();
}

std::size_t moe_prefill_cube_2d_scratch_floats(
    const CubeTileGeometry& geometry) noexcept {
  if (geometry.m_tile <= 0 || geometry.h_tile <= 0 || geometry.r_tile <= 0 ||
      geometry.k_tile <= 0) {
    return 0;
  }
  const std::size_t m = static_cast<std::size_t>(geometry.m_tile);
  const std::size_t h = static_cast<std::size_t>(geometry.h_tile);
  const std::size_t r = static_cast<std::size_t>(geometry.r_tile);
  const std::size_t k = static_cast<std::size_t>(geometry.k_tile);
  // The host mirror of the modelled buffer set, and nothing else:
  //   L0A-side   x_frag[m, k]     act[m, r]
  //   L0B-side   w1_frag[r, k]    w3_frag[r, k]   w2_frag[h, r]
  //   L0C-side   gate[m, r]       up[m, r]        y_acc[m, h]
  // Every term is a product of TILE extents. Nothing here scales with hidden,
  // inter or the token count, which is the no-global-workspace claim as a
  // formula a test can check.
  return m * k + m * r + 2U * r * k + h * r + 2U * m * r + m * h;
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
  // The geometry is rejected HERE, before a single tile runs. On device an
  // over-budget tile is a compile- or load-time failure; on the host the
  // equivalent honesty is a throw, because the alternative -- clamping the
  // tile to something that fits -- would silently verify a schedule nobody
  // asked for.
  Require(moe_prefill_cube_tile_geometry_admissible(geometry),
          "the Cube tile geometry is not admissible: every tile extent must be "
          "a positive multiple of the 16x16x16 fractal, buffer_depth must be 1 "
          "or 2, and the full-tile footprint must fit L0A/L0B/L0C/L1");

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

  const CubeTileGeometry& g = geometry;
  scratch->assign(moe_prefill_cube_2d_scratch_floats(g), 0.0F);
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
  float* x_frag = take(m * kt);   // L0A: X[M_tile, K_tile], BF16-valued
  float* act = take(m * rt);      // L0A: act[M_tile, R_tile], BF16-valued
  float* w1_frag = take(rt * kt); // L0B: W1[R_tile, K_tile], BF16-valued
  float* w3_frag = take(rt * kt); // L0B: W3[R_tile, K_tile], BF16-valued
  float* w2_frag = take(h * rt);  // L0B: W2[H_tile, R_tile], BF16-valued
  float* gate = take(m * rt);     // L0C: FP32
  float* up = take(m * rt);       // L0C: FP32
  float* y_acc = take(m * h);     // L0C: FP32, resident across the r loop

  if (trace != nullptr) {
    *trace = CubeTileTrace{};
    trace->intermediate_recompute_factor =
        static_cast<std::uint64_t>(CeilDiv(hidden, g.h_tile));
  }

  for (int t = 0; t < total_tokens; ++t) {
    const std::int32_t e = token_to_expert[t];
    Require(e >= 0 && e < num_experts, "token's expert id is out of range");
  }

  const std::vector<std::pair<int, int>> tiles =
      BuildTileOrder(total_tokens, hidden, g);

  for (const auto& tile : tiles) {
    const int m0 = tile.first;
    const int h0 = tile.second;
    const int m_extent = std::min(g.m_tile, total_tokens - m0);
    const int h_extent = std::min(g.h_tile, hidden - h0);

    // 1. The accumulator starts at zero and stays in L0C for the whole tile.
    for (int mm = 0; mm < m_extent; ++mm) {
      for (int hh = 0; hh < h_extent; ++hh) {
        y_acc[static_cast<std::size_t>(mm) * h + static_cast<std::size_t>(hh)] =
            0.0F;
      }
    }

    // 2. The inner contraction over `inter`.
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

      // 2a. gate / up, blocked along the hidden axis.
      //
      // NOTE the per-token expert lookup: it is inside the m loop, so a token
      // tile that straddles an expert boundary is handled correctly even
      // though the device kernel will assume the dispatch grouped the tokens.
      // A golden that assumed the grouping could not tell a wrong-expert read
      // from a legitimate one.
      for (int k0 = 0; k0 < hidden; k0 += g.k_tile) {
        const int k_extent = std::min(g.k_tile, hidden - k0);
        RecordExtents(trace, m_extent, h_extent, r_extent, k_extent);

        // X[M_tile, K_tile] -> L0A. Rounded to BF16 on the way in; the
        // rounding is idempotent, so doing it per fragment is the same
        // function as rounding the whole row once.
        for (int mm = 0; mm < m_extent; ++mm) {
          const std::size_t row = static_cast<std::size_t>(m0 + mm) *
                                  static_cast<std::size_t>(hidden);
          for (int kk = 0; kk < k_extent; ++kk) {
            x_frag[static_cast<std::size_t>(mm) * kt +
                   static_cast<std::size_t>(kk)] =
                bf16_round_cpu(x_tokens[row + static_cast<std::size_t>(k0 + kk)]);
          }
        }

        // The weight blocks. Dequantized here and nowhere else: there is no
        // workspace holding W1/W3/W2 in BF16, at any scale, which is the whole
        // point of the pivot this model verifies. Both W1 and W3 are staged
        // for the same (r0, k0) block because they reduce against the SAME X
        // fragment; on device they occupy one L0B slot one after the other,
        // which is how the footprint charges them.
        //
        // STAGED PER EXPERT, NOT PER TOKEN TILE. The Cube's reason for
        // existing is that one B fragment serves all M rows, and a kernel
        // running on a grouped dispatch stages each block exactly once per
        // tile. This golden does NOT assume the grouping -- it restages when a
        // tile straddles an expert boundary -- because a golden that assumed
        // it could not tell a wrong-expert read from a legitimate one. The
        // memo makes the common case cost what it costs on hardware, and the
        // trace counts the stagings so the difference is visible.
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
          // mmad: C += A * B^T, the fractal reduction inside, FP32 into L0C.
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

      // 2b. The SwiGLU, on the tile, in the Vector unit. The clamp is
      // ASYMMETRIC -- the gate bounded ABOVE only, the up projection on both
      // sides -- and `act` is stored BF16 because it is the down GEMM's A
      // matrix and the Cube reads BF16.
      for (int mm = 0; mm < m_extent; ++mm) {
        for (int rr = 0; rr < r_extent; ++rr) {
          const std::size_t i =
              static_cast<std::size_t>(mm) * rt + static_cast<std::size_t>(rr);
          const float clamped_gate = std::min(gate[i], limit);
          const float clamped_up = std::min(limit, std::max(-limit, up[i]));
          act[i] = bf16_round_cpu(Silu(clamped_gate) * clamped_up);
        }
      }

      // 2c. The down GEMM, accumulated in place into the L0C tile.
      int staged_down_expert = -1;
      for (int mm = 0; mm < m_extent; ++mm) {
        const int ex = token_to_expert[static_cast<std::size_t>(m0 + mm)];
        if (ex != staged_down_expert) {
          const std::size_t e2 = static_cast<std::size_t>(ex);
          for (int hh = 0; hh < h_extent; ++hh) {
            for (int rr = 0; rr < r_extent; ++rr) {
              w2_frag[static_cast<std::size_t>(hh) * rt +
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
              FractalDot(act + static_cast<std::size_t>(mm) * rt,
                         w2_frag + static_cast<std::size_t>(hh) * rt, r_extent);
        }
      }
      if (trace != nullptr) {
        trace->down_macs += static_cast<std::uint64_t>(m_extent) *
                            static_cast<std::uint64_t>(h_extent) *
                            static_cast<std::uint64_t>(r_extent);
      }
    }

    // 3. One write to GM per tile, at the end. Assignment, not accumulation:
    // the tile owns its slice of the output and nothing else touches it.
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
}

}  // namespace moe::emu
