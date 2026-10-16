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

// HOST VERIFICATION OF THE TWO-PHASE ON-CHIP FUSED CUBE PIPELINE, before any
// kernel.
//
// The schedule under audit is the one a 950PR MIX AIC+AIV core can actually
// run: a 1-D grid over TOKEN TILES, an `act[M_tile, inter]` plane resident on
// chip across a phase boundary, weights dequantized per sub-tile on the way to
// L0B, and a three-level tile decomposition (GM macro-tile -> L1/UB sub-tile
// -> L0 micro-tile). Nothing here touches a device, a simulator or a toolkit:
// this is the CI gate (CLAUDE.md section 4.1), it links moe_emu only, and it
// must not skip (section 4.2).
//
// IT REPLACED AN (m, h) OUTPUT GRID whose gate/up GEMMs sat inside the
// hidden-tile loop, recomputing the whole intermediate once per H tile -- 64x
// the gate/up work at the production geometry. Claim 7 below is what that
// pivot bought, as an identity rather than an improvement.
//
// WHAT IS CLAIMED, AND HOW EACH CLAIM IS EARNED
//
//   1. THE OUTPUT DEPENDS ON (K_sub, R_sub2) AND ON NOTHING ELSE ABOUT THE
//      TILING. M_tile, R_sub and H_sub partition the output space -- of the
//      token axis, the intermediate axis and the hidden axis respectively --
//      and none of them enters a reduction. So geometries differing only in
//      those three must agree BIT FOR BIT, which is spatial independence in
//      its sharpest numerical form and is asserted with no tolerance at all.
//      Only K_sub (which blocks the gate/up reduction over `hidden`) and
//      R_sub2 (which blocks the down reduction over `inter`) may move a value.
//
//   2. ZERO SYNC BETWEEN TILES, operationally. The visit order is a parameter
//      of the model, so the suite runs all four traversals and several
//      shuffles and requires bit-identical output. An implementation that
//      leaked an accumulator or a scratchpad between token tiles fails this; a
//      source inspection for barriers would not.
//
//   3. WRITE-ONCE COVERAGE, OF THE OUTPUT AND OF `act`. The destination is
//      poisoned before every run and the trace's element count must equal
//      M * hidden exactly, which excludes a gap, an overlap and an
//      accumulate-into-GM. The same count over `act` must equal M * inter,
//      which is claim 7 in its sharpest form.
//
//   4. THE 950PR BUFFER BUDGET IS A HARD CONSTRAINT, AND UB IS NOW IN IT.
//      L0A/L0B/L0C/L1 and the 248 KiB UB are checked three ways: the formula,
//      the OBSERVED peak of every fragment actually staged during the run, and
//      a negative control -- geometries one step past the admissible set must
//      be rejected, or the check has no teeth. The resident `act` plane is
//      what puts `inter` in the budget, so admissibility is a property of the
//      geometry AND the problem here, not of the geometry alone.
//
//   5. NUMERICAL EQUIVALENCE WITH THE MONOLITHIC ORACLE, inside a bound
//      derived from the formats and the two reduction shapes. This model is
//      NEVER bit-exact with that oracle, by construction: the Cube's fractal
//      reduction (C0 = 16) is a level the flat oracle does not have. So the
//      bound is built in two parts, because one number would hide the
//      interesting half:
//        (a) the gate/up GEMMs reassociate the hidden-axis sum. Bounded by the
//            chain depths of the two trees times 2^-24 times the absolute-value
//            sum of the SAME products.
//        (b) that FP32 reassociation is then AMPLIFIED by the BF16 store of
//            `act`: a gate within the reassociation of a BF16 tie rounds the
//            other way, so `act` can move by one BF16 ULP -- 2^-9 relative,
//            four orders above the cause. The suite asserts the one-ULP claim
//            elementwise, counts the flips, and propagates them through the
//            down GEMM exactly.
//      The bit-exact anchor is an INDEPENDENT transcription of the schedule,
//      not the monolithic oracle.
//
//   6. RAGGED EVERYTHING: token counts, hidden and inter that divide none of
//      the sub-tile extents. The trace's min/max fragment extents prove the
//      tails were actually staged rather than skipped.
//
//   7. ZERO REDUNDANT COMPUTATION, as two exact identities on the trace:
//        gate_up_macs         == 2 * M * inter * hidden   (multiplier 1.0x)
//        act_elements_written == M * inter                (each stored once)
//      and the residual cost that replaced the recompute is reported rather
//      than discovered on hardware: with r outer and k inner the X macro-tile
//      is re-read ceil(inter / R_sub) times per token tile, which
//      `x_fragment_reads` counts.
//
// Inputs are generated in memory from a PRINTED seed. No disk fixtures.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include "kernel_test_harness.hpp"
#include "mxfp4_bank_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using moe::emu::bf16_round_cpu;
using moe::emu::CubeBufferFootprint;
using moe::emu::CubeTileGeometry;
using moe::emu::CubeTileTrace;
using moe::emu::CubeTileVisit;
using moe::emu::kCubeFractal;
using moe::emu::kCubeL0ABytes;
using moe::emu::kCubeL0BBytes;
using moe::emu::kCubeL0CBytes;
using moe::emu::kCubeL1Bytes;
using moe::emu::kCubeUbBytes;
using moe::emu::kMxfp4GroupSize;
using moe::emu::Mxfp4Matrix;
using moe::kerntest::CosineSimilarity;
using moe::kerntest::MaxAbsDiff;
using moe::kerntest::MeasureSpread;
using moe::kerntest::RelativeL2;
using moe::kerntest::Report;
using moe::kerntest::section;
using mxfp4_fixture::Bank;
using mxfp4_fixture::MakeBank;
using mxfp4_fixture::SiluRef;

constexpr std::uint64_t kSeed = 0xC0BE'2D16'5EEDULL;

// The principal fixture. hidden = 128 and inter = 256 make every sub-tile
// extent in the menu interesting -- each axis is split more than one way:
//   hidden 128 vs K_sub 64 -> 2 blocks, vs K_sub 128 -> 1
//              vs H_sub 64 -> 2 blocks, vs H_sub 128 -> 1
//   inter  256 vs R_sub 64 -> 4 blocks, vs R_sub 128 -> 2
//              vs R_sub2 64 -> 4 blocks, vs R_sub2 128 -> 2
// R_sub2 must be able to split `inter` or the down GEMM has a single
// reduction block and claim 1's "R_sub2 moves the output" half is vacuous,
// which is why the menu's R_sub2 values are below 256 here even though the
// production value is 256 against inter 2048.
constexpr int kHidden = 128;
constexpr int kInter = 256;
constexpr int kExperts = 3;
// 51 tokens: ragged against both M_tile = 16 (3 full tiles + a tail of 3) and
// M_tile = 32 (1 full tile + a tail of 19). FOUR token tiles at M_tile = 16 is
// the point -- the grid is the only independent axis this schedule has, so a
// fixture with two tiles would let the visit-order check pass on a 24th of the
// permutations by coincidence.
constexpr int kTokens = 51;
constexpr float kLimit = 10.0F;

// The RAGGED fixture: every extent chosen so that no tile size divides it.
// 192 = 6 x 32 and 160 = 5 x 32, so both are still whole MXFP4 groups.
//   hidden 192  vs H_tile 128 -> 128 + 64     vs K_tile 128 -> 128 + 64
//   inter  160  vs R_tile 64  -> 64 + 64 + 32 vs R_tile 128 -> 128 + 32
//   M 37        vs M_tile 16  -> 16 + 16 + 5  vs M_tile 32  -> 32 + 5
constexpr int kRaggedHidden = 192;
constexpr int kRaggedInter = 160;
constexpr int kRaggedTokens = 37;

// FP32's unit roundoff.
constexpr double kU = 1.0 / 16777216.0;  // 2^-24

// The sub-tile extents, in the order the schedule introduces them: the token
// tile, then phase 1's (R_sub, K_sub), then phase 2's (H_sub, R_sub2). Spelled
// positionally because every call site names all five -- a defaulted extent
// here is how a geometry audit quietly stops covering a knob.
CubeTileGeometry Geom(int m, int r, int k, int h, int r_down) {
  CubeTileGeometry g;
  g.m_tile = m;
  g.r_tile = r;
  g.k_tile = k;
  g.h_tile = h;
  g.r_tile_down = r_down;
  return g;
}

std::string GeomName(const CubeTileGeometry& g) {
  return "(M " + std::to_string(g.m_tile) + ", Rsub " +
         std::to_string(g.r_tile) + ", Ksub " + std::to_string(g.k_tile) +
         ", Hsub " + std::to_string(g.h_tile) + ", Rsub2 " +
         std::to_string(g.r_tile_down) + ", depth " +
         std::to_string(g.buffer_depth) + ")";
}

int CeilDiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Chain depths, from which the reassociation bounds are built
// ---------------------------------------------------------------------------
// A flat FP32 sum of n terms puts each term through at most n - 1 additions.
// The Cube's hierarchy puts a term through at most
//     (C0 - 1) inside its fractal
//   + (fractals per block - 1) accumulating fractals in L0C
//   + (blocks - 1)             accumulating blocks in L0C
// and the largest block is the full tile extent. Both numbers are properties
// of the SHAPES only -- nothing is read off an observed error (section 4.7).
double FlatChainDepth(int n) { return static_cast<double>(n - 1); }

double HierarchicalChainDepth(int n, int block) {
  const int widest = std::min(block, n);
  const int fractals = CeilDiv(widest, kCubeFractal);
  const int blocks = CeilDiv(n, block);
  return static_cast<double>(kCubeFractal - 1) +
         static_cast<double>(fractals - 1) + static_cast<double>(blocks - 1);
}

// |hierarchical - flat| for the same products, first order in u.
double ReassociationBound(int n, int block, double abs_sum) {
  return (FlatChainDepth(n) + HierarchicalChainDepth(n, block)) * kU * abs_sum;
}

// The BF16 ULP at `value`, i.e. the distance to the next BF16 away from zero.
// `value` is assumed to be a BF16-representable float, which every `act` is.
double Bf16Ulp(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x00010000u;  // step the BF16 significand, away from zero
  float stepped = 0.0F;
  std::memcpy(&stepped, &bits, sizeof(stepped));
  return std::fabs(static_cast<double>(stepped) - static_cast<double>(value));
}

bool BitIdentical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> RandomRows(int rows, int cols, double sigma,
                              std::mt19937_64& rng) {
  std::vector<float> v(static_cast<std::size_t>(rows) *
                       static_cast<std::size_t>(cols));
  std::normal_distribution<float> draw(0.0F, static_cast<float>(sigma));
  for (float& e : v) e = draw(rng);
  return v;
}

// ---------------------------------------------------------------------------
// The fixture, as the reference sees it
// ---------------------------------------------------------------------------
struct ExpertSet {
  Bank w1, w3, w2;
  std::vector<Mxfp4Matrix> v1, v3, v2;
  int hidden = 0;
  int inter = 0;

  void Build(int hidden_in, int inter_in, int experts, std::mt19937_64& rng) {
    hidden = hidden_in;
    inter = inter_in;
    w1 = MakeBank(inter, hidden, experts, rng);
    w3 = MakeBank(inter, hidden, experts, rng);
    w2 = MakeBank(hidden, inter, experts, rng);
    for (int e = 0; e < experts; ++e) {
      v1.push_back(w1.View(e));
      v3.push_back(w3.View(e));
      v2.push_back(w2.View(e));
    }
  }
};

// ---------------------------------------------------------------------------
// The independent transcription of the Cube schedule
// ---------------------------------------------------------------------------
// The bit-exact anchor. It computes the same answer by DIFFERENT code paths:
// weights come from the public `mxfp4_dequant_row_bf16_cpu` row helper rather
// than the reference's per-element decode, and the hierarchy is written out
// here as three explicit nested sums rather than called.
//
// It also exposes everything the error bound needs -- the FP32 gate/up, the
// BF16 `act`, and the absolute-value sums of both reductions -- which is why
// the bound can be split into the reassociation and the BF16-flip halves
// instead of being one fitted number.
//
// NOTE what it does NOT model: M_tile, R_sub and H_sub. They partition the
// output -- along the token, intermediate and hidden axes -- and cannot enter
// the arithmetic, and claim 1 of this suite is exactly that. The two extents
// it DOES read are K_sub (which blocks the gate/up reduction over `hidden`)
// and R_sub2 (which blocks the down reduction over `inter`). If the reference
// disagreed with this transcription for some (M_tile, R_sub, H_sub) triple,
// that would be the finding.
struct Transcript {
  std::vector<float> y;     // [M, hidden]
  std::vector<float> gate;  // [M, inter]
  std::vector<float> up;    // [M, inter]
  std::vector<float> act;   // [M, inter]
  std::vector<double> gate_abs_sum;  // [M, inter] sum_k |xb| * |W1b|
  std::vector<double> up_abs_sum;    // [M, inter] sum_k |xb| * |W3b|
  std::vector<double> down_abs_sum;  // [M, hidden] sum_r |act| * |W2b|
};

// One hierarchical dot product: fractals of C0, accumulated per block, blocks
// accumulated over the axis. Written out here, independently of the
// reference's FractalDot.
float HierDot(const float* a, const float* b, int n, int block) {
  float total = 0.0F;
  for (int base = 0; base < n; base += block) {
    const int extent = std::min(block, n - base);
    float block_total = 0.0F;
    for (int f = 0; f < extent; f += kCubeFractal) {
      const int span = std::min(kCubeFractal, extent - f);
      float fragment = 0.0F;
      for (int i = 0; i < span; ++i) {
        fragment += a[base + f + i] * b[base + f + i];
      }
      block_total += fragment;
    }
    total += block_total;
  }
  return total;
}

double AbsSum(const float* a, const float* b, int n) {
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    total += static_cast<double>(std::fabs(a[i])) *
             static_cast<double>(std::fabs(b[i]));
  }
  return total;
}

Transcript Transcribe(const ExpertSet& set, const float* x, int tokens,
                      const std::int32_t* route, float limit,
                      const CubeTileGeometry& g) {
  const int hidden = set.hidden;
  const int inter = set.inter;
  Transcript t;
  t.y.assign(static_cast<std::size_t>(tokens) * hidden, 0.0F);
  t.gate.assign(static_cast<std::size_t>(tokens) * inter, 0.0F);
  t.up.assign(static_cast<std::size_t>(tokens) * inter, 0.0F);
  t.act.assign(static_cast<std::size_t>(tokens) * inter, 0.0F);
  t.gate_abs_sum.assign(static_cast<std::size_t>(tokens) * inter, 0.0);
  t.up_abs_sum.assign(static_cast<std::size_t>(tokens) * inter, 0.0);
  t.down_abs_sum.assign(static_cast<std::size_t>(tokens) * hidden, 0.0);

  std::vector<float> xb(static_cast<std::size_t>(hidden));
  std::vector<float> w1row(static_cast<std::size_t>(hidden));
  std::vector<float> w3row(static_cast<std::size_t>(hidden));
  std::vector<float> w2row(static_cast<std::size_t>(inter));

  for (int m = 0; m < tokens; ++m) {
    const int e = route[m];
    const std::size_t xrow = static_cast<std::size_t>(m) * hidden;
    for (int c = 0; c < hidden; ++c) {
      xb[static_cast<std::size_t>(c)] = bf16_round_cpu(x[xrow + static_cast<std::size_t>(c)]);
    }
    // gate / up: the hidden axis, blocked by K_tile, fractals of 16 inside.
    for (int r = 0; r < inter; ++r) {
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v1[static_cast<std::size_t>(e)], r,
                                           w1row.data());
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v3[static_cast<std::size_t>(e)], r,
                                           w3row.data());
      const std::size_t i = static_cast<std::size_t>(m) * inter +
                            static_cast<std::size_t>(r);
      t.gate[i] = HierDot(xb.data(), w1row.data(), hidden, g.k_tile);
      t.up[i] = HierDot(xb.data(), w3row.data(), hidden, g.k_tile);
      t.gate_abs_sum[i] = AbsSum(xb.data(), w1row.data(), hidden);
      t.up_abs_sum[i] = AbsSum(xb.data(), w3row.data(), hidden);
      t.act[i] = bf16_round_cpu(
          SiluRef(std::min(t.gate[i], limit)) *
          std::min(limit, std::max(-limit, t.up[i])));
    }
    // down: the inter axis, blocked by R_sub2 -- phase 2's contraction
    // sub-tile, NOT phase 1's R_sub -- with fractals of 16 inside. The two
    // phases block different axes with different extents, and using the wrong
    // one here would make the anchor agree with a schedule nobody runs.
    for (int h = 0; h < hidden; ++h) {
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v2[static_cast<std::size_t>(e)], h,
                                           w2row.data());
      const std::size_t yi = static_cast<std::size_t>(m) * hidden +
                             static_cast<std::size_t>(h);
      t.y[yi] = HierDot(t.act.data() + static_cast<std::size_t>(m) * inter,
                        w2row.data(), inter, g.r_tile_down);
      t.down_abs_sum[yi] = AbsSum(
          t.act.data() + static_cast<std::size_t>(m) * inter, w2row.data(), inter);
    }
  }
  return t;
}

// The monolithic oracle's intermediates, for the two-part bound. Flat sums,
// which is the whole difference.
Transcript TranscribeFlat(const ExpertSet& set, const float* x, int tokens,
                          const std::int32_t* route, float limit) {
  CubeTileGeometry flat;
  flat.k_tile = set.hidden;       // one gate/up block ...
  flat.r_tile_down = set.inter;   // ... one down block ...
                                  // ... but still fractals of 16 inside
  Transcript t = Transcribe(set, x, tokens, route, limit, flat);
  // Recompute the three reductions FLAT, which is what the monolithic oracle
  // does, overwriting the hierarchical values.
  std::vector<float> xb(static_cast<std::size_t>(set.hidden));
  std::vector<float> row(static_cast<std::size_t>(
      std::max(set.hidden, set.inter)));
  for (int m = 0; m < tokens; ++m) {
    const int e = route[m];
    const std::size_t xrow = static_cast<std::size_t>(m) * set.hidden;
    for (int c = 0; c < set.hidden; ++c) {
      xb[static_cast<std::size_t>(c)] =
          bf16_round_cpu(x[xrow + static_cast<std::size_t>(c)]);
    }
    for (int r = 0; r < set.inter; ++r) {
      const std::size_t i = static_cast<std::size_t>(m) * set.inter +
                            static_cast<std::size_t>(r);
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v1[static_cast<std::size_t>(e)], r,
                                           row.data());
      float total = 0.0F;
      for (int c = 0; c < set.hidden; ++c) total += row[static_cast<std::size_t>(c)] * xb[static_cast<std::size_t>(c)];
      t.gate[i] = total;
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v3[static_cast<std::size_t>(e)], r,
                                           row.data());
      total = 0.0F;
      for (int c = 0; c < set.hidden; ++c) total += row[static_cast<std::size_t>(c)] * xb[static_cast<std::size_t>(c)];
      t.up[i] = total;
      t.act[i] = bf16_round_cpu(
          SiluRef(std::min(t.gate[i], limit)) *
          std::min(limit, std::max(-limit, t.up[i])));
    }
    for (int h = 0; h < set.hidden; ++h) {
      moe::emu::mxfp4_dequant_row_bf16_cpu(set.v2[static_cast<std::size_t>(e)], h,
                                           row.data());
      float total = 0.0F;
      for (int r = 0; r < set.inter; ++r) {
        total += t.act[static_cast<std::size_t>(m) * set.inter +
                       static_cast<std::size_t>(r)] *
                 row[static_cast<std::size_t>(r)];
      }
      t.y[static_cast<std::size_t>(m) * set.hidden +
          static_cast<std::size_t>(h)] = total;
    }
  }
  return t;
}

std::vector<float> RunCube(const ExpertSet& set, const float* x, int tokens,
                           const std::int32_t* route, float limit,
                           const CubeTileGeometry& g, CubeTileTrace* trace) {
  std::vector<float> y(static_cast<std::size_t>(tokens) * set.hidden,
                       -98765.0F);  // poison: a gap in the grid cannot hide
  std::vector<float> scratch;
  moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
      set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
      static_cast<int>(set.v1.size()), y.data(), limit, g, trace, &scratch);
  return y;
}

// ---------------------------------------------------------------------------
// 1. The 950PR on-chip buffer budget
// ---------------------------------------------------------------------------
// The budget is now a function of the GEOMETRY AND OF `inter`, because the
// resident `act[M_tile, inter]` plane is charged against UB. That is the one
// structural change this section carries over the (m, h)-grid version, and it
// is the reason every call here passes an `inter`.
void AuditBufferBudget(Report& r) {
  section("1. the 950PR on-chip buffer budget");

  // The capacities themselves, pinned against the platform configs. These are
  // the numbers every footprint claim rests on, so they are written down once
  // and asserted rather than being implicit in a comparison.
  r.check(kCubeL0ABytes == 65536 && kCubeL0BBytes == 65536 &&
              kCubeL0CBytes == 262144 && kCubeL1Bytes == 524288 &&
              kCubeUbBytes == 253952,
          "L0A/L0B = 64 KiB, L0C = 256 KiB, L1 = 512 KiB, UB = 248 KiB, as the "
          "950PR platform configs give them");
  r.check(kCubeFractal == 16,
          "the fractal unit is 16 elements -- C0 for BF16, and the MAC array's "
          "K depth");

  // --- THE PRODUCTION GEOMETRY, at the production problem. -----------------
  // This is the one the task names and the one the kernel will run, so it is
  // checked at hidden 4096 / inter 2048 and not only on the small fixture.
  {
    constexpr int kProdInter = 2048;
    const CubeTileGeometry prod = Geom(16, 128, 128, 64, 256);
    const CubeBufferFootprint f =
        moe::emu::moe_prefill_cube_tile_footprint(prod, kProdInter);
    r.check(moe::emu::moe_prefill_cube_tile_geometry_admissible(prod, kProdInter),
            "the production geometry (M 16, R_sub 128, K_sub 128, H_sub 64, "
            "R_sub2 256) is admissible at inter 2048",
            "L0A " + std::to_string(f.l0a_bytes) + " L0B " +
                std::to_string(f.l0b_bytes) + " L0C " +
                std::to_string(f.l0c_bytes) + " L1 " +
                std::to_string(f.l1_bytes) + " UB " +
                std::to_string(f.ub_bytes));
    // act[16, 2048] in BF16 is EXACTLY 64 KiB. That number is the whole reason
    // the token tile is 16, so it is pinned rather than recomputed from a
    // comment.
    const CubeBufferFootprint p1 = moe::emu::moe_prefill_cube_phase1_footprint(
        16, 128, 128, kProdInter, 2);
    const CubeBufferFootprint p2 = moe::emu::moe_prefill_cube_phase2_footprint(
        16, 64, 256, kProdInter, 2);
    r.check(p1.ub_bytes >= 65536u && p2.ub_bytes >= 65536u,
            "both phases carry the 65,536 B act[16, 2048] plane -- it is "
            "resident ACROSS the phase boundary, which is what the schedule "
            "spends its UB on",
            "phase 1 UB " + std::to_string(p1.ub_bytes) + ", phase 2 UB " +
                std::to_string(p2.ub_bytes));
    // L0B is exactly full in BOTH phases at this geometry: R_sub * K_sub and
    // H_sub * R_sub2 are both 16,384 elements, and at double buffering that is
    // 65,536 B. The production sub-tiles are the largest the Cube's B buffer
    // admits, which is worth stating because it means the menu cannot be
    // widened without dropping to depth 1.
    r.check(p1.l0b_bytes == kCubeL0BBytes && p2.l0b_bytes == kCubeL0BBytes,
            "L0B is exactly full in both phases at the production geometry "
            "(R_sub * K_sub == H_sub * R_sub2 == 16,384 elements at depth 2), "
            "so these sub-tiles are maximal rather than merely feasible",
            "phase 1 L0B " + std::to_string(p1.l0b_bytes) + ", phase 2 L0B " +
                std::to_string(p2.l0b_bytes));
    std::printf("           production @ inter 2048: L0A %zu, L0B %zu, L0C %zu, "
                "L1 %zu, UB %zu (of %zu)\n",
                f.l0a_bytes, f.l0b_bytes, f.l0c_bytes, f.l1_bytes, f.ub_bytes,
                kCubeUbBytes);
  }

  // --- THE ENUMERATED MENU must be admissible, or the suite's geometry list
  // is not actually runnable. The (H_sub, R_sub2) pairs are enumerated
  // TOGETHER rather than crossed, because their product is what L0B charges
  // and an independent cross would list pairs that cannot run.
  const std::pair<int, int> gate_pairs[] = {{64, 64}, {128, 64}, {64, 128},
                                            {128, 128}};
  const std::pair<int, int> down_pairs[] = {{64, 64}, {64, 128}, {128, 64},
                                            {64, 256}, {128, 128}};
  int admissible = 0;
  int total = 0;
  std::size_t worst_l0a = 0, worst_l0b = 0, worst_l0c = 0, worst_l1 = 0,
              worst_ub = 0;
  for (const int m : {16, 32}) {
    for (const auto& rk : gate_pairs) {
      for (const auto& hd : down_pairs) {
        const CubeTileGeometry g =
            Geom(m, rk.first, rk.second, hd.first, hd.second);
        const CubeBufferFootprint f =
            moe::emu::moe_prefill_cube_tile_footprint(g, kInter);
        ++total;
        if (moe::emu::moe_prefill_cube_tile_geometry_admissible(g, kInter)) {
          ++admissible;
        }
        worst_l0a = std::max(worst_l0a, f.l0a_bytes);
        worst_l0b = std::max(worst_l0b, f.l0b_bytes);
        worst_l0c = std::max(worst_l0c, f.l0c_bytes);
        worst_l1 = std::max(worst_l1, f.l1_bytes);
        worst_ub = std::max(worst_ub, f.ub_bytes);
      }
    }
  }
  r.check(admissible == total,
          "all " + std::to_string(total) +
              " enumerated geometries are admissible at inter " +
              std::to_string(kInter) + " with double buffering",
          std::to_string(admissible) + " of " + std::to_string(total) +
              " passed");
  std::printf("           worst of the %d: L0A %zu B, L0B %zu B (of %zu), "
              "L0C %zu B (of %zu), L1 %zu B, UB %zu B\n",
              total, worst_l0a, worst_l0b, kCubeL0BBytes, worst_l0c,
              kCubeL0CBytes, worst_l1, worst_ub);
  r.check(worst_l0b == kCubeL0BBytes,
          "L0B is the binding buffer: the widest sub-tile in the menu fills it "
          "exactly at double buffering, which is what stops the menu there",
          "worst L0B " + std::to_string(worst_l0b));

  // --- THE FORMULAE, stated independently of the implementation, and PER
  // PHASE. One merged number would not say which phase is tight, and the two
  // phases stage different shapes.
  {
    constexpr int kN = 256;
    const CubeBufferFootprint p1 =
        moe::emu::moe_prefill_cube_phase1_footprint(16, 64, 64, kN, 2);
    r.check(p1.l0a_bytes == 2u * 2u * 16u * 64u &&
                p1.l0b_bytes == 2u * 2u * 64u * 64u &&
                p1.l0c_bytes == 4u * 2u * 16u * 64u &&
                p1.l1_bytes == 2u * 2u * (2u * 64u * 64u + 16u * 64u),
            "phase 1 is depth*2B*M*K_sub in L0A, depth*2B*R_sub*K_sub in L0B "
            "(W1 and W3 share the slot), 4B*2*M*R_sub in L0C, and "
            "depth*2B*(2*R_sub*K_sub + M*K_sub) in L1",
            "L0A " + std::to_string(p1.l0a_bytes) + " L0B " +
                std::to_string(p1.l0b_bytes) + " L0C " +
                std::to_string(p1.l0c_bytes) + " L1 " +
                std::to_string(p1.l1_bytes));
    // gate AND up, together and not as a max: both must be live when the
    // SwiGLU reads them, so a max() model would be the one modelling error
    // that lets an infeasible geometry look feasible.
    r.check(p1.l0c_bytes == 2u * (4u * 16u * 64u),
            "phase 1 charges gate AND up at once, not the larger of them");

    const CubeBufferFootprint p2 =
        moe::emu::moe_prefill_cube_phase2_footprint(16, 64, 128, kN, 2);
    r.check(p2.l0a_bytes == 2u * 2u * 16u * 128u &&
                p2.l0b_bytes == 2u * 2u * 64u * 128u &&
                p2.l0c_bytes == 4u * 16u * 64u &&
                p2.l1_bytes == 2u * 2u * 64u * 128u,
            "phase 2 is depth*2B*M*R_sub2 in L0A, depth*2B*H_sub*R_sub2 in "
            "L0B, 4B*M*H_sub in L0C, and depth*2B*H_sub*R_sub2 in L1 -- the A "
            "matrix comes straight out of the resident UB plane, so L1 charges "
            "the weight sub-tile ALONE",
            "L0A " + std::to_string(p2.l0a_bytes) + " L0B " +
                std::to_string(p2.l0b_bytes) + " L0C " +
                std::to_string(p2.l0c_bytes) + " L1 " +
                std::to_string(p2.l1_bytes));

    // THE TILE PEAK IS THE ELEMENTWISE MAX OF THE TWO PHASES, not their sum.
    // That is sound only because the phases are separated in time -- phase 2
    // cannot start until the last `act` column is stored -- so it is checked
    // as the max it claims to be rather than assumed.
    const CubeTileGeometry g = Geom(16, 64, 64, 64, 128);
    const CubeBufferFootprint f =
        moe::emu::moe_prefill_cube_tile_footprint(g, kN);
    r.check(f.l0a_bytes == std::max(p1.l0a_bytes, p2.l0a_bytes) &&
                f.l0b_bytes == std::max(p1.l0b_bytes, p2.l0b_bytes) &&
                f.l0c_bytes == std::max(p1.l0c_bytes, p2.l0c_bytes) &&
                f.l1_bytes == std::max(p1.l1_bytes, p2.l1_bytes) &&
                f.ub_bytes == std::max(p1.ub_bytes, p2.ub_bytes),
            "the tile peak is the elementwise MAX of the two phases, which is "
            "sound because they are separated in time and share only `act`");
    r.check(f.l0a_bytes < p1.l0a_bytes + p2.l0a_bytes,
            "and it is strictly less than their sum, so the max is really "
            "being taken rather than the phases coinciding");
  }

  // --- UB IS A REAL CONSTRAINT, and this is the check that says so. The same
  // geometry that fits at a small `inter` must be REJECTED at an `inter` whose
  // act plane does not fit. Nothing else in this suite can fail if UB is
  // silently ignored, so without this the ub_bytes field would be decoration.
  {
    const CubeTileGeometry g = Geom(16, 64, 64, 64, 64);
    r.check(moe::emu::moe_prefill_cube_tile_geometry_admissible(g, 2048),
            "a narrow geometry is admissible at inter 2048 (act 64 KiB)");
    // act[16, inter] * 2 B must stay under 248 KiB, so inter above ~7700 is
    // out on the act plane alone. 16384 gives act = 512 KiB, twice the whole
    // buffer.
    r.check(!moe::emu::moe_prefill_cube_tile_geometry_admissible(g, 16384),
            "the SAME geometry is rejected at inter 16384, where the resident "
            "act plane alone is 512 KiB against 248 KiB of UB -- so "
            "admissibility is a property of the geometry AND the problem");
    const CubeBufferFootprint huge =
        moe::emu::moe_prefill_cube_tile_footprint(g, 16384);
    r.check(huge.l0a_bytes <= kCubeL0ABytes &&
                huge.l0b_bytes <= kCubeL0BBytes &&
                huge.l0c_bytes <= kCubeL0CBytes &&
                huge.l1_bytes <= kCubeL1Bytes && huge.ub_bytes > kCubeUbBytes,
            "and it is rejected on UB ALONE -- every Cube buffer still fits, "
            "so the UB condition is not being conflated with the others",
            "UB " + std::to_string(huge.ub_bytes) + " of " +
                std::to_string(kCubeUbBytes));
  }

  // --- NEGATIVE CONTROLS. Without these the budget check proves nothing: a
  // predicate that returns true for everything would pass every test above.
  struct Bad {
    CubeTileGeometry g;
    int inter;
    const char* why;
  };
  std::vector<Bad> bad;
  bad.push_back({Geom(16, 256, 128, 64, 128), kInter,
                 "R_sub 256 with K_sub 128 overflows L0B (128 KiB)"});
  bad.push_back({Geom(16, 64, 64, 128, 256), kInter,
                 "H_sub 128 with R_sub2 256 overflows L0B (128 KiB)"});
  bad.push_back({Geom(32, 2048, 64, 64, 64), kInter,
                 "R_sub 2048 overflows L0B long before L0C"});
  bad.push_back({Geom(24, 64, 64, 64, 64), kInter,
                 "M_tile 24 is not a fractal multiple"});
  bad.push_back({Geom(16, 64, 64, 63, 64), kInter,
                 "H_sub 63 is not a fractal multiple"});
  bad.push_back({Geom(16, 64, 64, 64, 24), kInter,
                 "R_sub2 24 is not a fractal multiple"});
  bad.push_back({Geom(16, 0, 64, 64, 64), kInter,
                 "R_sub 0 is not a positive extent"});
  bad.push_back({Geom(16, 64, 64, 64, -64), kInter,
                 "R_sub2 -64 is not a positive extent"});
  bad.push_back({Geom(16, 64, 64, 64, 64), 0,
                 "inter 0 leaves the act plane undefined"});
  CubeTileGeometry deep = Geom(16, 64, 64, 64, 64);
  deep.buffer_depth = 3;
  bad.push_back({deep, kInter,
                 "buffer_depth 3 is not a modelled buffering scheme"});
  CubeTileGeometry zero_depth = Geom(16, 64, 64, 64, 64);
  zero_depth.buffer_depth = 0;
  bad.push_back({zero_depth, kInter, "buffer_depth 0 is not a buffering scheme"});
  for (const Bad& b : bad) {
    r.check(!moe::emu::moe_prefill_cube_tile_geometry_admissible(b.g, b.inter),
            std::string("rejected: ") + b.why);
  }

  // And a geometry that fails ONLY on the budget must still be fractal-legal,
  // or the negative control above would be passing for the wrong reason.
  const CubeTileGeometry over = Geom(16, 256, 128, 64, 128);
  const CubeBufferFootprint over_f =
      moe::emu::moe_prefill_cube_tile_footprint(over, kInter);
  r.check(over.m_tile % kCubeFractal == 0 && over.h_tile % kCubeFractal == 0 &&
              over.r_tile % kCubeFractal == 0 &&
              over.k_tile % kCubeFractal == 0 &&
              over.r_tile_down % kCubeFractal == 0 && !over_f.Fits() &&
              over_f.l0b_bytes > kCubeL0BBytes &&
              over_f.l0c_bytes <= kCubeL0CBytes &&
              over_f.ub_bytes <= kCubeUbBytes,
          "the over-budget control is fractal-legal and fails on L0B ALONE, so "
          "the fractal, UB and L0 conditions are not conflated",
          "L0B " + std::to_string(over_f.l0b_bytes) + " L0C " +
              std::to_string(over_f.l0c_bytes) + " UB " +
              std::to_string(over_f.ub_bytes));
}

// ---------------------------------------------------------------------------
// 2. Which extents partition the output, and which enter the arithmetic
// ---------------------------------------------------------------------------
// FIVE EXTENTS, SPLIT THREE TO TWO, and the split is not arbitrary -- it is
// read off the schedule:
//
//   M_tile   partitions the TOKEN axis of the output          no reduction
//   R_sub    partitions the INTERMEDIATE axis of `act`        no reduction
//   H_sub    partitions the HIDDEN axis of the output         no reduction
//   K_sub    blocks the gate/up reduction over `hidden`       REASSOCIATES
//   R_sub2   blocks the down reduction over `inter`           REASSOCIATES
//
// So three of the five must be bit-irrelevant and two must not be, and both
// halves are asserted. Asserting only the first half would pass for a model
// that had no reduction hierarchy at all.
void AuditTilingInvariance(Report& r, const ExpertSet& set, const float* x,
                           int tokens, const std::int32_t* route) {
  section("2. M_tile / R_sub / H_sub partition the output and nothing else");

  const CubeTileGeometry base = Geom(16, 64, 64, 64, 64);
  CubeTileTrace trace_base;
  const std::vector<float> want =
      RunCube(set, x, tokens, route, kLimit, base, &trace_base);

  // Each variant changes ONE partitioning extent, then one changes all three
  // at once -- a knob that leaked through only in combination would survive
  // the singles.
  struct Variant {
    CubeTileGeometry g;
    const char* knob;
  };
  const Variant variants[] = {
      {Geom(32, 64, 64, 64, 64), "M_tile 16 -> 32"},
      {Geom(16, 128, 64, 64, 64), "R_sub 64 -> 128"},
      {Geom(16, 64, 64, 128, 64), "H_sub 64 -> 128"},
      {Geom(32, 128, 64, 128, 64), "all three at once"},
  };
  for (const Variant& v : variants) {
    CubeTileTrace trace;
    const std::vector<float> got =
        RunCube(set, x, tokens, route, kLimit, v.g, &trace);
    r.check(BitIdentical(got, want),
            std::string(v.knob) + " -- " + GeomName(v.g) + " agrees with " +
                GeomName(base) +
                " BIT FOR BIT: changing only the output partitioning cannot "
                "change the arithmetic",
            "max abs " +
                std::to_string(MaxAbsDiff(got.data(), want.data(), got.size())));
    // ... and the schedule really was different, or the check is vacuous. Two
    // independent witnesses, because no single counter moves for all three
    // knobs: M_tile and H_sub change the GM write grid, R_sub changes how many
    // times the X macro-tile is re-read.
    r.check(trace.output_tiles != trace_base.output_tiles ||
                trace.x_fragment_reads != trace_base.x_fragment_reads,
            std::string(v.knob) + " really did run a different schedule (" +
                std::to_string(trace.output_tiles) + " output blocks and " +
                std::to_string(trace.x_fragment_reads) + " X reads, against " +
                std::to_string(trace_base.output_tiles) + " and " +
                std::to_string(trace_base.x_fragment_reads) + ")");
  }

  // THE TWO KNOBS THAT DO ENTER THE ARITHMETIC, and they do NOT enter it
  // symmetrically. This asymmetry is a real property of a W4A16 path and the
  // obvious version of this check gets it wrong, so it is worth spelling out:
  //
  //   R_sub2 re-associates the DOWN GEMM, whose FP32 result goes straight to
  //   the output. The perturbation lands in the answer.
  //
  //   K_sub re-associates the GATE/UP GEMMs, and their result is then STORED
  //   AS BF16 in `act`. The reassociation is ~hidden * 2^-24 relative, four
  //   orders below BF16's 2^-9 granularity, so it almost always rounds to the
  //   same BF16 value and the output is bit-identical. "Almost always": the
  //   probability of crossing a tie is about hidden * 2^-15 per element, so at
  //   a larger hidden it does happen -- which is what section 5b's
  //   amplification fixture exercises.
  //
  // So: assert that R_sub2 moves the OUTPUT, and that K_sub moves the
  // INTERMEDIATE. Asserting that K_sub moves the output would be a test
  // fitted to a fixture, and it would fail on a correct model.
  {
    const CubeTileGeometry wide_rd = Geom(16, 64, 64, 64, 128);
    const std::vector<float> got =
        RunCube(set, x, tokens, route, kLimit, wide_rd, nullptr);
    r.check(!BitIdentical(got, want),
            GeomName(wide_rd) + " DOES change the output -- R_sub2 "
                                "reassociates the down GEMM and nothing rounds "
                                "after it",
            "it agreed bit for bit, so the hierarchy is not being modelled");
    std::printf("           %s against the base: relative-L2 %.3e\n",
                GeomName(wide_rd).c_str(),
                RelativeL2(got.data(), want.data(), got.size()));
  }
  {
    const CubeTileGeometry wide_k = Geom(16, 64, 128, 64, 64);
    const Transcript narrow = Transcribe(set, x, tokens, route, kLimit, base);
    const Transcript wide = Transcribe(set, x, tokens, route, kLimit, wide_k);
    const double gate_gap = MaxAbsDiff(wide.gate.data(), narrow.gate.data(),
                                       narrow.gate.size());
    r.check(gate_gap > 0.0,
            GeomName(wide_k) + " DOES change the gate/up values -- K_sub "
                               "reassociates the hidden-axis sum",
            "the gate was bit-identical, so the K blocking is not being "
            "modelled");
    int flips = 0;
    for (std::size_t i = 0; i < wide.act.size(); ++i) {
      if (wide.act[i] != narrow.act[i]) ++flips;
    }
    const std::vector<float> got =
        RunCube(set, x, tokens, route, kLimit, wide_k, nullptr);
    std::printf("           %s: gate moves by up to %.3e, which the BF16 store "
                "of act absorbs on %zu of %zu elements; the output is %s\n",
                GeomName(wide_k).c_str(), gate_gap,
                wide.act.size() - static_cast<std::size_t>(flips),
                wide.act.size(),
                BitIdentical(got, want) ? "bit-identical" : "different");
    // Whatever the flips did, the output must be consistent with them: no
    // flips means a bit-identical output, and that implication is the real
    // content of this check.
    r.check(flips > 0 || BitIdentical(got, want),
            GeomName(wide_k) + ": with no activation flip the output is "
                               "bit-identical, which is what makes the BF16 "
                               "intermediate insensitive to the K blocking");
  }
}

// ---------------------------------------------------------------------------
// 3. Zero sync between token tiles, and write-once coverage
// ---------------------------------------------------------------------------
void AuditSpatialIndependence(Report& r, const ExpertSet& set, const float* x,
                              int tokens, const std::int32_t* route) {
  section("3. zero sync between token tiles");

  const CubeTileGeometry base = Geom(16, 64, 64, 64, 64);
  CubeTileTrace trace;
  const std::vector<float> want = RunCube(set, x, tokens, route, kLimit, base, &trace);

  // THE VISIT ORDER IS A PARAMETER so that this can be a check and not a
  // claim. The grid's only independent axis is the token tile, and the enum
  // permutes those; it also flips the direction of phase 2's h-block loop,
  // which is inside a tile body and so cannot be a grid axis but is just as
  // irrelevant to the answer. If a tile -- or an h block -- depended on any
  // state another left behind, these would differ.
  struct Order {
    CubeTileVisit visit;
    std::uint64_t seed;
    const char* name;
  };
  const Order orders[] = {
      {CubeTileVisit::kColumnMajor, 0, "tiles ascending, h blocks descending"},
      {CubeTileVisit::kReversed, 0, "tiles descending, h blocks descending"},
      {CubeTileVisit::kShuffled, 1, "tiles shuffled (seed 1)"},
      {CubeTileVisit::kShuffled, 0xABCDEF, "tiles shuffled (seed 0xABCDEF)"},
  };
  for (const Order& o : orders) {
    CubeTileGeometry g = base;
    g.visit = o.visit;
    g.visit_seed = o.seed;
    const std::vector<float> got = RunCube(set, x, tokens, route, kLimit, g, nullptr);
    r.check(BitIdentical(got, want),
            std::string("visiting ") + o.name + " gives a bit-identical result",
            "max abs " +
                std::to_string(MaxAbsDiff(got.data(), want.data(), got.size())));
  }

  // THE SHUFFLE MUST REALLY HAVE SHUFFLED, or all four orders agreeing says
  // nothing at all. Checking that enough tiles EXIST for a non-identity
  // permutation would only bound the coincidence; this reconstructs the
  // permutation the model actually walks -- same std::mt19937_64, same seed,
  // same std::shuffle over the same element type in the same translation unit
  // -- and requires it to be non-identity.
  {
    const int m_tiles = CeilDiv(tokens, base.m_tile);
    r.check(m_tiles >= 3,
            "the fixture has " + std::to_string(m_tiles) +
                " token tiles, so the grid has an interior to permute");
    int non_identity = 0;
    for (const Order& o : orders) {
      if (o.visit != CubeTileVisit::kShuffled) continue;
      std::vector<int> order(static_cast<std::size_t>(m_tiles));
      for (int i = 0; i < m_tiles; ++i) {
        order[static_cast<std::size_t>(i)] = i;
      }
      std::mt19937_64 rng(o.seed);
      std::shuffle(order.begin(), order.end(), rng);
      if (!std::is_sorted(order.begin(), order.end())) ++non_identity;
    }
    r.check(non_identity > 0,
            "at least one of the shuffle seeds really does reorder the token "
            "tiles, so the bit-identical results above are a statement about "
            "independence and not about an identity permutation",
            std::to_string(non_identity) + " of the shuffle seeds permuted");
  }

  // WRITE-ONCE COVERAGE OF THE OUTPUT. The destination went in poisoned, so a
  // gap in the grid would survive as -98765; the element count excludes an
  // overlap.
  const std::uint64_t expected_elements =
      static_cast<std::uint64_t>(tokens) * static_cast<std::uint64_t>(set.hidden);
  r.check(trace.output_elements_written == expected_elements,
          "the tiles wrote exactly M * hidden elements (" +
              std::to_string(expected_elements) + "), so they partition the "
              "output with no overlap and no gap",
          "wrote " + std::to_string(trace.output_elements_written));
  bool poison_survived = false;
  for (const float v : want) {
    if (v == -98765.0F) poison_survived = true;
  }
  r.check(!poison_survived,
          "no element of the poisoned destination survived, so every output "
          "element was assigned");

  // WRITE-ONCE COVERAGE OF `act`, which is the new half. The intermediate is
  // not observable from outside the golden, so the trace is the only place
  // this can be checked -- and it is the sharpest form of "gate and up run
  // exactly once": the previous (m, h)-grid schedule would report
  // M * inter * ceil(hidden / H_sub) here.
  const std::uint64_t expected_act =
      static_cast<std::uint64_t>(tokens) * static_cast<std::uint64_t>(set.inter);
  r.check(trace.act_elements_written == expected_act,
          "the activation plane was written exactly M * inter times (" +
              std::to_string(expected_act) +
              ") -- each activation computed ONCE, with no H-tile recompute",
          "wrote " + std::to_string(trace.act_elements_written));

  // The grid's shape, both axes of it: the token tiles are the grid, the
  // (m0, h0) blocks are the GM writes.
  r.check(trace.token_tiles ==
              static_cast<std::uint64_t>(CeilDiv(tokens, base.m_tile)),
          "the grid is exactly ceil(M / M_tile) token tiles",
          "ran " + std::to_string(trace.token_tiles));
  r.check(trace.output_tiles ==
              static_cast<std::uint64_t>(CeilDiv(tokens, base.m_tile)) *
                  static_cast<std::uint64_t>(CeilDiv(set.hidden, base.h_tile)),
          "and it writes ceil(M / M_tile) x ceil(hidden / H_sub) output blocks",
          "wrote " + std::to_string(trace.output_tiles) + " blocks");

  const auto spread = MeasureSpread(want.data(), want.size());
  r.check(spread.NonDegenerate(),
          "the Cube golden carries magnitude and variation",
          "rms " + std::to_string(spread.rms) + " stddev " +
              std::to_string(spread.stddev));
  std::printf("           golden: rms %.4f stddev %.4f min %.4f max %.4f\n",
              spread.rms, spread.stddev, spread.min, spread.max);
}

// ---------------------------------------------------------------------------
// 4. The observed footprint of every fragment actually staged
// ---------------------------------------------------------------------------
void AuditObservedFootprint(Report& r, const ExpertSet& set, const float* x,
                            int tokens, const std::int32_t* route,
                            const std::vector<CubeTileGeometry>& geometries) {
  section("4. the footprint OBSERVED during the run, fragment by fragment");

  for (const CubeTileGeometry& g : geometries) {
    CubeTileTrace trace;
    RunCube(set, x, tokens, route, kLimit, g, &trace);

    // The trace's peak is already charged at the geometry's own buffer depth
    // -- the footprint helpers take `buffer_depth` and the recorder passes it
    // -- so this compares like with like and does not re-multiply.
    const bool fits = trace.peak.Fits();
    r.check(fits,
            GeomName(g) +
                ": every fragment set staged during the run fits "
                "L0A/L0B/L0C/L1 and UB at the geometry's buffer depth",
            "peak L0A " + std::to_string(trace.peak.l0a_bytes) + " L0B " +
                std::to_string(trace.peak.l0b_bytes) + " L0C " +
                std::to_string(trace.peak.l0c_bytes) + " L1 " +
                std::to_string(trace.peak.l1_bytes) + " UB " +
                std::to_string(trace.peak.ub_bytes));

    // The observed peak must EQUAL the full-sub-tile formula wherever a full
    // sub-tile exists -- otherwise the model is charging for fragments it
    // never stages, or staging ones it does not charge for.
    const CubeBufferFootprint full =
        moe::emu::moe_prefill_cube_tile_footprint(g, set.inter);
    const bool has_full_tile =
        tokens >= g.m_tile && set.hidden >= g.h_tile && set.inter >= g.r_tile &&
        set.hidden >= g.k_tile && set.inter >= g.r_tile_down;
    if (has_full_tile) {
      r.check(trace.peak.l0a_bytes == full.l0a_bytes &&
                  trace.peak.l0b_bytes == full.l0b_bytes &&
                  trace.peak.l0c_bytes == full.l0c_bytes &&
                  trace.peak.l1_bytes == full.l1_bytes &&
                  trace.peak.ub_bytes == full.ub_bytes,
              GeomName(g) +
                  ": the observed peak is exactly the full-sub-tile footprint, "
                  "so the budget formula describes what actually ran",
              "observed L0A " + std::to_string(trace.peak.l0a_bytes) + " vs " +
                  std::to_string(full.l0a_bytes) + ", L0B " +
                  std::to_string(trace.peak.l0b_bytes) + " vs " +
                  std::to_string(full.l0b_bytes) + ", L0C " +
                  std::to_string(trace.peak.l0c_bytes) + " vs " +
                  std::to_string(full.l0c_bytes) + ", UB " +
                  std::to_string(trace.peak.ub_bytes) + " vs " +
                  std::to_string(full.ub_bytes));
    } else {
      r.check(trace.peak.l0a_bytes <= full.l0a_bytes &&
                  trace.peak.l0b_bytes <= full.l0b_bytes,
              GeomName(g) +
                  ": with no full sub-tile in the problem the observed peak "
                  "stays under the full-sub-tile footprint");
    }

    // BOTH PHASES RAN. A merged peak would be satisfied by a model that
    // skipped one of them entirely, so the per-phase peaks are required to be
    // nonzero and to differ -- they stage different shapes, so an
    // implementation that reported one for both would be the finding.
    r.check(trace.phase1_peak.l0b_bytes > 0 && trace.phase2_peak.l0b_bytes > 0,
            GeomName(g) + ": both phases staged at least one fragment set, so "
                          "neither was skipped",
            "phase 1 L0B " + std::to_string(trace.phase1_peak.l0b_bytes) +
                ", phase 2 L0B " +
                std::to_string(trace.phase2_peak.l0b_bytes));
    r.check(trace.peak.l0b_bytes == std::max(trace.phase1_peak.l0b_bytes,
                                             trace.phase2_peak.l0b_bytes) &&
                trace.peak.ub_bytes == std::max(trace.phase1_peak.ub_bytes,
                                                trace.phase2_peak.ub_bytes),
            GeomName(g) + ": the merged peak is the max of the two phases' "
                          "observed peaks");
    std::printf("           %s: peak L0A %zu, L0B %zu, L0C %zu, L1 %zu, UB %zu "
                "(phase 1 UB %zu / phase 2 UB %zu)\n",
                GeomName(g).c_str(), trace.peak.l0a_bytes,
                trace.peak.l0b_bytes, trace.peak.l0c_bytes,
                trace.peak.l1_bytes, trace.peak.ub_bytes,
                trace.phase1_peak.ub_bytes, trace.phase2_peak.ub_bytes);
  }

  // The host scratch is the mirror of that buffer set: sub-tile sized except
  // for the one term that is deliberately not -- act[M_tile, inter]. That is
  // the no-global-workspace claim, and the shape of the formula is what makes
  // it checkable.
  const CubeTileGeometry g = Geom(16, 64, 64, 64, 128);
  const std::size_t want_floats =
      /* x_frag  */ 16u * 64u +
      /* w1 + w3 */ 2u * 64u * 64u +
      /* gate+up */ 2u * 16u * 64u +
      /* act     */ 16u * static_cast<std::size_t>(kInter) +
      /* a_frag  */ 16u * 128u +
      /* w2_frag */ 64u * 128u +
      /* y_acc   */ 16u * 64u;
  r.check(moe::emu::moe_prefill_cube_2d_scratch_floats(g, kInter) == want_floats,
          "the scratch is exactly the modelled buffer set: x_frag + w1/w3 + "
          "gate + up + act[M_tile, inter] + a_frag + w2_frag + Y_acc",
          "got " + std::to_string(
                       moe::emu::moe_prefill_cube_2d_scratch_floats(g, kInter)));
  std::vector<float> scratch;
  std::vector<float> y(static_cast<std::size_t>(tokens) * set.hidden);
  moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
      set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
      static_cast<int>(set.v1.size()), y.data(), kLimit, g, nullptr, &scratch);
  r.check(scratch.size() == want_floats,
          "the run allocates exactly that and nothing more -- no weight "
          "workspace, no M x inter intermediate",
          "allocated " + std::to_string(scratch.size()));

  // THE SHAPE OF THE FORMULA, as three independent statements. Any one of them
  // alone would be satisfied by a formula that is wrong in the other two
  // directions, which is why "it does not grow with the problem" is spelled
  // out as which variable it does and does not grow with.
  {
    // (i) INDEPENDENT OF THE TOKEN COUNT. act is M_TILE x inter, not M x
    // inter; that difference is the whole claim, so it is asserted as the
    // absence of an M in the formula -- the only argument that could carry one
    // is not even passed.
    r.check(moe::emu::moe_prefill_cube_2d_scratch_floats(g, kInter) ==
                want_floats,
            "the scratch formula takes no token count at all, so it cannot "
            "grow with M -- the act plane is M_TILE x inter");
    // (ii) INDEPENDENT OF `hidden`. hidden enters only through K_sub and
    // H_sub, which are sub-tile extents and are held fixed here.
    r.check(moe::emu::moe_prefill_cube_2d_scratch_floats(g, kInter) ==
                want_floats,
            "and it takes no `hidden` either -- the hidden axis is streamed in "
            "K_sub and H_sub blocks and never materialized");
    // (iii) LINEAR IN `inter`, WITH SLOPE M_TILE. The act plane is the ONLY
    // term that grows with inter, so doubling inter must add exactly
    // M_tile * inter floats and nothing else.
    const std::size_t at_1x =
        moe::emu::moe_prefill_cube_2d_scratch_floats(g, kInter);
    const std::size_t at_2x =
        moe::emu::moe_prefill_cube_2d_scratch_floats(g, 2 * kInter);
    r.check(at_2x - at_1x ==
                static_cast<std::size_t>(g.m_tile) *
                    static_cast<std::size_t>(kInter),
            "doubling `inter` grows the scratch by exactly M_tile * inter "
            "floats -- the resident act plane and nothing else",
            "grew by " + std::to_string(at_2x - at_1x) + ", expected " +
                std::to_string(static_cast<std::size_t>(g.m_tile) * kInter));
  }
  std::printf("           scratch: %zu floats (%zu B), of which act is %zu B; "
              "the dequantized weights alone would be %zu B\n",
              scratch.size(), scratch.size() * sizeof(float),
              16u * static_cast<std::size_t>(kInter) * sizeof(float),
              static_cast<std::size_t>(set.hidden) * set.inter * 3u * 2u);
}

// ---------------------------------------------------------------------------
// 5. Numerical equivalence with the monolithic oracle
// ---------------------------------------------------------------------------
void AuditNumericalEquivalence(Report& r, const ExpertSet& set, const float* x,
                               int tokens, const std::int32_t* route,
                               const std::vector<CubeTileGeometry>& geometries) {
  section("5. against the monolithic W4A16 oracle, inside a derived bound");

  const int hidden = set.hidden;
  const int inter = set.inter;

  std::vector<float> mono(static_cast<std::size_t>(tokens) * hidden);
  std::vector<float> mono_scratch;
  moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
      set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
      static_cast<int>(set.v1.size()), mono.data(), kLimit, &mono_scratch);
  const auto mono_spread = MeasureSpread(mono.data(), mono.size());
  r.check(mono_spread.NonDegenerate(),
          "the monolithic golden being compared against is non-degenerate",
          "rms " + std::to_string(mono_spread.rms));

  // The monolithic oracle's own intermediates, reproduced flat. Pinned
  // against the oracle's output first, so the decomposition that follows is
  // about the real reference and not about a second model of it.
  const Transcript flat = TranscribeFlat(set, x, tokens, route, kLimit);
  r.check(BitIdentical(flat.y, mono),
          "the flat transcription reproduces moe_prefill_batched_mxfp4_w4a16_cpu "
          "bit for bit, so its gate/up/act ARE that oracle's intermediates",
          "max abs " +
              std::to_string(MaxAbsDiff(flat.y.data(), mono.data(), mono.size())));

  for (const CubeTileGeometry& g : geometries) {
    const std::vector<float> got = RunCube(set, x, tokens, route, kLimit, g, nullptr);
    const Transcript hier = Transcribe(set, x, tokens, route, kLimit, g);

    // (0) THE BIT-EXACT ANCHOR: the reference IS the schedule, checked against
    // an independent transcription that uses the public row dequantizer and
    // writes the hierarchy out by hand.
    r.check(BitIdentical(got, hier.y),
            GeomName(g) + ": matches the independent hierarchical "
                          "transcription bit for bit",
            "max abs " +
                std::to_string(MaxAbsDiff(got.data(), hier.y.data(), got.size())));

    // (a) THE GATE/UP REASSOCIATION, elementwise against a bound built from
    // the chain depths of the two trees and the absolute-value sum of the same
    // products.
    int gate_violations = 0;
    double worst_gate_ratio = 0.0;
    for (std::size_t i = 0; i < hier.gate.size(); ++i) {
      const double dg = std::fabs(static_cast<double>(hier.gate[i]) -
                                  static_cast<double>(flat.gate[i]));
      const double du = std::fabs(static_cast<double>(hier.up[i]) -
                                  static_cast<double>(flat.up[i]));
      const double bg = ReassociationBound(hidden, g.k_tile, hier.gate_abs_sum[i]);
      const double bu = ReassociationBound(hidden, g.k_tile, hier.up_abs_sum[i]);
      if (dg > bg || du > bu) ++gate_violations;
      if (bg > 0.0) worst_gate_ratio = std::max(worst_gate_ratio, dg / bg);
      if (bu > 0.0) worst_gate_ratio = std::max(worst_gate_ratio, du / bu);
    }
    r.check(gate_violations == 0,
            GeomName(g) + ": every gate and up value is inside the derived "
                          "hidden-axis reassociation bound",
            std::to_string(gate_violations) + " exceeded it, worst ratio " +
                std::to_string(worst_gate_ratio));

    // (b) THE BF16 AMPLIFICATION. An FP32 reassociation of ~hidden * 2^-24 is
    // four orders below BF16's 2^-9 granularity, so it can only ever push a
    // gate across a single tie: `act` must differ by AT MOST ONE BF16 ULP.
    // That is the claim; here it is checked elementwise, and the flips are
    // counted because their number is what the output bound needs.
    int act_flips = 0;
    int act_over_one_ulp = 0;
    for (std::size_t i = 0; i < hier.act.size(); ++i) {
      if (hier.act[i] == flat.act[i]) continue;
      ++act_flips;
      const double delta = std::fabs(static_cast<double>(hier.act[i]) -
                                     static_cast<double>(flat.act[i]));
      const double ulp = std::max(Bf16Ulp(hier.act[i]), Bf16Ulp(flat.act[i]));
      if (delta > ulp) ++act_over_one_ulp;
    }
    r.check(act_over_one_ulp == 0,
            GeomName(g) + ": where the activation differs it differs by at "
                          "most ONE BF16 ULP -- the gate reassociation can "
                          "only ever cross a single tie",
            std::to_string(act_over_one_ulp) + " of " +
                std::to_string(act_flips) + " flips moved further than one ULP");
    std::printf("           %s: %d of %zu activations flipped one BF16 ULP "
                "(%.3f%%)\n",
                GeomName(g).c_str(), act_flips, hier.act.size(),
                100.0 * static_cast<double>(act_flips) /
                    static_cast<double>(hier.act.size()));

    // (c) THE OUTPUT, as the sum of the two causes:
    //       |y_hier - y_flat|
    //         <= |hierarchical sum of act_hier*w  -  flat sum of act_hier*w|
    //          + |flat sum of (act_hier - act_flat)*w|
    //         <= reassociation(inter, R_tile, S_hier[h])
    //          + sum_r |act_hier[r] - act_flat[r]| * |W2b[h, r]|
    // Both terms are derived: the first from the chain depths, the second from
    // the measured one-ULP flips propagated exactly through a linear map.
    std::vector<float> w2row(static_cast<std::size_t>(inter));
    int out_violations = 0;
    double worst_ratio = 0.0;
    double worst_rel = 0.0;
    double worst_cos = 1.0;
    for (int m = 0; m < tokens; ++m) {
      const std::size_t e = static_cast<std::size_t>(route[m]);
      for (int h = 0; h < hidden; ++h) {
        moe::emu::mxfp4_dequant_row_bf16_cpu(set.v2[e], h, w2row.data());
        double flip_term = 0.0;
        for (int rr = 0; rr < inter; ++rr) {
          const std::size_t ai = static_cast<std::size_t>(m) * inter +
                                 static_cast<std::size_t>(rr);
          flip_term += std::fabs(static_cast<double>(hier.act[ai]) -
                                 static_cast<double>(flat.act[ai])) *
                       std::fabs(static_cast<double>(w2row[static_cast<std::size_t>(rr)]));
        }
        const std::size_t yi = static_cast<std::size_t>(m) * hidden +
                               static_cast<std::size_t>(h);
        const double bound =
            ReassociationBound(inter, g.r_tile_down, hier.down_abs_sum[yi]) +
            flip_term;
        const double diff = std::fabs(static_cast<double>(got[yi]) -
                                      static_cast<double>(mono[yi]));
        if (diff > bound) ++out_violations;
        if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
      }
      const std::size_t row = static_cast<std::size_t>(m) * hidden;
      worst_rel = std::max(worst_rel,
                           RelativeL2(got.data() + row, mono.data() + row,
                                      static_cast<std::size_t>(hidden)));
      worst_cos = std::min(worst_cos,
                           CosineSimilarity(got.data() + row, mono.data() + row,
                                            static_cast<std::size_t>(hidden)));
    }
    r.check(out_violations == 0,
            GeomName(g) + ": every output element is inside the derived "
                          "reassociation + one-ULP-flip bound",
            std::to_string(out_violations) + " exceeded it, worst diff/bound " +
                std::to_string(worst_ratio));

    // Cosine is kept strict even where the magnitude bound must be loose:
    // rounding moves magnitude, structural bugs move direction (4.7). The
    // flips are a 2^-9 relative perturbation on a small fraction of the
    // activations, so the direction can move by at most about (2^-9)^2 / 2 in
    // the worst case of total correlation -- 1e-6 is two orders conservative.
    constexpr double kCosineFloor = 1.0 - 1e-6;
    r.check(worst_cos >= kCosineFloor,
            GeomName(g) + ": per-token cosine against the monolithic oracle "
                          "stays above 1 - 1e-6",
            "worst cosine " + std::to_string(worst_cos));
    std::printf("           %s: worst diff/bound %.4f, worst per-token "
                "relative-L2 %.3e, worst cosine %.12f\n",
                GeomName(g).c_str(), worst_ratio, worst_rel, worst_cos);

    // And the difference must be NONZERO, or the hierarchy is not being
    // modelled and this whole section is comparing a function with itself.
    r.check(MaxAbsDiff(got.data(), mono.data(), got.size()) > 0.0,
            GeomName(g) + ": the Cube model and the flat oracle are genuinely "
                          "different functions (the C0 = 16 fractal level is "
                          "real)",
            "they agreed exactly");
  }
}

// ---------------------------------------------------------------------------
// 5b. The BF16 amplification lemma, and what it is worth at scale
// ---------------------------------------------------------------------------
// The output bound in section 5 has two halves, and the second one -- the
// activation flips -- measured ZERO on the principal fixture. A term that is
// always zero is a term nobody has checked, so this section does three things
// instead of quietly leaving it at that:
//
//   1. It proves the LEMMA the term rests on, directly and independently of
//      any GEMM: an FP32 perturbation of at most half a BF16 ULP moves the
//      rounded value by at most one ULP. That is a property of RNE rounding,
//      so it is tested as one, swept across binades, signs and exact ties.
//
//   2. It shows WHY the term is zero here, as an inequality on real data
//      rather than as a shrug: the gate/up reassociation, propagated through
//      the clamped SwiGLU, stays below half a BF16 ULP for every single
//      activation -- so the lemma PREDICTS zero flips, and zero is observed.
//      The prediction and the observation are both asserted, which is a
//      stronger statement than either alone.
//
//   3. It records what changes at the production geometry, where the term
//      stops being zero. The relative reassociation of a sum of n terms grows
//      like 2^-24 * sqrt(n), so the share of activations that can cross a
//      BF16 tie grows with the hidden axis. The number is printed, not
//      asserted: it is an estimate about hardware-scale shapes, and section
//      5.7 forbids claiming more than that.
//
// A fixture engineered to force flips was considered and rejected: at hidden
// 512 the measured gate shift is four orders below the BF16 ULP, and reaching
// a flip probability near 1 would need a reduction axis of order 1e9. Forcing
// it would mean fabricating adversarial inputs the path never sees, which
// tests the fabrication rather than the path.
void AuditBf16Amplification(Report& r, std::mt19937_64& rng) {
  section("5b. the BF16 amplification lemma, and what it is worth at scale");

  // --- 1. THE LEMMA, as a property of RNE rounding.
  //
  // Sweep magnitudes across binades, perturb by just under and just over half
  // an ULP in both directions, and require: under half an ULP the rounded
  // value moves by at most one ULP; and the bound is TIGHT, i.e. there exist
  // perturbations under half an ULP that do move it by exactly one (otherwise
  // the lemma would be a vacuous over-estimate).
  {
    int violations = 0;
    int exercised = 0;
    for (int binade = -12; binade <= 12; ++binade) {
      const float scale = std::ldexp(1.0F, binade);
      for (int step = 0; step < 64; ++step) {
        // A BF16-representable base, plus a sub-ULP offset that walks across
        // the whole interval between two BF16 neighbours -- so the ties are
        // hit, not stepped over.
        const float base =
            bf16_round_cpu(scale * (1.0F + static_cast<float>(step) / 64.0F));
        const double ulp = Bf16Ulp(base);
        for (const double frac : {-0.499, -0.25, 0.0, 0.25, 0.499}) {
          for (const float sign : {1.0F, -1.0F}) {
            const float perturbed =
                sign * static_cast<float>(static_cast<double>(base) + frac * ulp);
            const float rounded = bf16_round_cpu(perturbed);
            const float reference = bf16_round_cpu(sign * base);
            const double moved = std::fabs(static_cast<double>(rounded) -
                                           static_cast<double>(reference));
            const double allowed =
                std::max(Bf16Ulp(rounded), Bf16Ulp(reference));
            if (moved > allowed) ++violations;
            if (moved > 0.0) ++exercised;
          }
        }
      }
    }
    r.check(violations == 0,
            "LEMMA: an FP32 perturbation of at most half a BF16 ULP moves the "
            "rounded value by at most one ULP, over 25 binades and both signs",
            std::to_string(violations) + " perturbations moved further");
    r.check(exercised > 0,
            "the lemma sweep actually crosses ties -- some sub-half-ULP "
            "perturbations DO move the rounded value, so the bound is tight "
            "and not vacuously satisfied",
            "no perturbation changed the rounding, so the sweep missed every "
            "tie");
    std::printf("           lemma: %d of the swept perturbations crossed a "
                "tie and moved exactly one ULP\n",
                exercised);
  }

  // --- 2. WHY THE TERM IS ZERO HERE, as an inequality on real data.
  constexpr int kWideHidden = 512;
  constexpr int kNarrowInter = 64;
  constexpr int kWideTokens = 16;

  ExpertSet wide;
  wide.Build(kWideHidden, kNarrowInter, 1, rng);
  // Scaled down so the SwiGLU clamp does NOT saturate: a clamped gate is
  // exactly insensitive to its own perturbation, which would make the
  // inequality below true for an uninteresting reason.
  const std::vector<float> x = RandomRows(kWideTokens, kWideHidden, 0.05, rng);
  const std::vector<std::int32_t> route(static_cast<std::size_t>(kWideTokens), 0);

  const CubeTileGeometry g = Geom(16, 64, 64, 128, 64);
  const std::vector<float> got =
      RunCube(wide, x.data(), kWideTokens, route.data(), kLimit, g, nullptr);
  const Transcript hier =
      Transcribe(wide, x.data(), kWideTokens, route.data(), kLimit, g);
  const Transcript flat =
      TranscribeFlat(wide, x.data(), kWideTokens, route.data(), kLimit);
  r.check(BitIdentical(got, hier.y),
          "at hidden 512 the reference still matches the independent "
          "transcription bit for bit");

  int unclamped = 0;
  for (std::size_t i = 0; i < hier.gate.size(); ++i) {
    if (hier.gate[i] < kLimit && std::fabs(hier.up[i]) < kLimit) ++unclamped;
  }
  r.check(unclamped == static_cast<int>(hier.gate.size()),
          "no activation is clamped in this fixture, so every one of them is "
          "genuinely sensitive to its gate's perturbation",
          std::to_string(unclamped) + " of " +
              std::to_string(hier.gate.size()) + " were unclamped");

  // Propagate the measured gate/up shift through the clamped SwiGLU:
  //   act = silu(clamp(gate)) * clamp(up)
  //   |d act| <= |silu'|_max * |up| * |d gate| + |silu(gate)| * |d up|
  // with |silu'| <= 1.1 everywhere (the maximum is ~1.0998 near z = 2.4) and
  // both clamps 1-Lipschitz. Every factor is derived; only the shifts are
  // measured.
  constexpr double kSiluLipschitz = 1.1;
  int predicted_flips_possible = 0;
  int observed_flips = 0;
  int over_one_ulp = 0;
  double worst_margin = 0.0;  // predicted act shift / (half an ULP)
  for (std::size_t i = 0; i < hier.act.size(); ++i) {
    const double dgate = std::fabs(static_cast<double>(hier.gate[i]) -
                                   static_cast<double>(flat.gate[i]));
    const double dup = std::fabs(static_cast<double>(hier.up[i]) -
                                 static_cast<double>(flat.up[i]));
    const double predicted =
        kSiluLipschitz * std::fabs(static_cast<double>(flat.up[i])) * dgate +
        std::fabs(static_cast<double>(SiluRef(flat.gate[i]))) * dup;
    const double half_ulp = 0.5 * Bf16Ulp(flat.act[i]);
    if (predicted > half_ulp) ++predicted_flips_possible;
    if (half_ulp > 0.0) worst_margin = std::max(worst_margin, predicted / half_ulp);
    if (hier.act[i] != flat.act[i]) {
      ++observed_flips;
      const double delta = std::fabs(static_cast<double>(hier.act[i]) -
                                     static_cast<double>(flat.act[i]));
      const double ulp = std::max(Bf16Ulp(hier.act[i]), Bf16Ulp(flat.act[i]));
      if (delta > ulp) ++over_one_ulp;
    }
  }
  r.check(predicted_flips_possible == 0 && observed_flips == 0,
          "the propagated gate/up shift stays under half a BF16 ULP for every "
          "activation, so the lemma PREDICTS no flip -- and none is observed",
          std::to_string(predicted_flips_possible) +
              " element(s) could flip, " + std::to_string(observed_flips) +
              " did");
  // The guard stays in place regardless: if the model ever does produce a
  // flip, it may not exceed one ULP.
  r.check(over_one_ulp == 0,
          "no activation moved further than one BF16 ULP",
          std::to_string(over_one_ulp) + " of " +
              std::to_string(observed_flips) + " flips moved further");
  std::printf("           at hidden %d: the propagated activation shift is at "
              "most %.3e of half a BF16 ULP, so the flip term is exactly zero "
              "here\n",
              kWideHidden, worst_margin);

  // --- 3. WHAT CHANGES AT SCALE. Reported, never asserted (section 5.7).
  // The relative reassociation of a flat sum against a blocked one grows like
  // 2^-24 * sqrt(n) for random signs, and a flip needs it to reach BF16's
  // 2^-9 half-granularity. The ratio below is that estimate carried from the
  // margin measured above, which is the only honest way to extrapolate it.
  const double scale_ratio = std::sqrt(4096.0 / static_cast<double>(kWideHidden));
  std::printf("           estimate only, never measured: at hidden 4096 the "
              "same worst-case margin scales by ~sqrt(4096/%d) = %.1fx to "
              "%.3e -- still %.0fx short of a flip. The flip term stays in the "
              "bound because it is the correct FORM, not because this geometry "
              "activates it; an adversarially conditioned row, or a reduction "
              "axis orders longer, is what would.\n",
              kWideHidden, scale_ratio, worst_margin * scale_ratio,
              1.0 / (worst_margin * scale_ratio));

  // --- and the output bound still holds, with its flip term at zero.
  std::vector<float> mono(static_cast<std::size_t>(kWideTokens) * kWideHidden);
  std::vector<float> scratch;
  moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
      wide.v1.data(), wide.v3.data(), wide.v2.data(), x.data(), route.data(),
      kWideTokens, 1, mono.data(), kLimit, &scratch);
  int violations = 0;
  double worst_ratio = 0.0;
  double worst_cos = 1.0;
  for (int m = 0; m < kWideTokens; ++m) {
    for (int h = 0; h < kWideHidden; ++h) {
      const std::size_t yi = static_cast<std::size_t>(m) * kWideHidden +
                             static_cast<std::size_t>(h);
      const double bound =
          ReassociationBound(kNarrowInter, g.r_tile_down,
                             hier.down_abs_sum[yi]);
      const double diff = std::fabs(static_cast<double>(got[yi]) -
                                    static_cast<double>(mono[yi]));
      if (diff > bound) ++violations;
      if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
    }
    const std::size_t row = static_cast<std::size_t>(m) * kWideHidden;
    worst_cos = std::min(worst_cos,
                         CosineSimilarity(got.data() + row, mono.data() + row,
                                          static_cast<std::size_t>(kWideHidden)));
  }
  r.check(violations == 0,
          "with the flip term provably zero, the output sits inside the pure "
          "reassociation bound",
          std::to_string(violations) + " exceeded it, worst diff/bound " +
              std::to_string(worst_ratio));
  r.check(worst_cos >= 1.0 - 1e-6,
          "cosine at hidden 512 stays above 1 - 1e-6",
          "worst cosine " + std::to_string(worst_cos));
  std::printf("           worst diff/bound %.4f, worst cosine %.12f\n",
              worst_ratio, worst_cos);
}

// ---------------------------------------------------------------------------
// 6. Ragged boundaries
// ---------------------------------------------------------------------------
void AuditRagged(Report& r, const ExpertSet& ragged, std::mt19937_64& rng) {
  section("6. ragged tokens, ragged hidden, ragged inter");

  const std::vector<float> x =
      RandomRows(kRaggedTokens, kRaggedHidden, 1.0, rng);
  std::vector<std::int32_t> route(static_cast<std::size_t>(kRaggedTokens));
  for (int t = 0; t < kRaggedTokens; ++t) {
    route[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(t % kExperts);
  }

  std::vector<float> mono(static_cast<std::size_t>(kRaggedTokens) * kRaggedHidden);
  std::vector<float> scratch;
  moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
      ragged.v1.data(), ragged.v3.data(), ragged.v2.data(), x.data(),
      route.data(), kRaggedTokens, kExperts, mono.data(), kLimit, &scratch);

  for (const CubeTileGeometry& g : {Geom(16, 64, 64, 64, 64),
                                    Geom(32, 128, 64, 128, 128),
                                    Geom(32, 128, 128, 64, 256)}) {
    CubeTileTrace trace;
    const std::vector<float> got =
        RunCube(ragged, x.data(), kRaggedTokens, route.data(), kLimit, g, &trace);

    // THE TAILS WERE ACTUALLY STAGED. The trace's minima are the smallest
    // fragment extent seen, so these numbers distinguish "ran the tail" from
    // "skipped the tail and got away with it because the destination happened
    // to be right".
    const int want_min_m = kRaggedTokens % g.m_tile;
    const int want_min_h = kRaggedHidden % g.h_tile == 0
                               ? std::min(g.h_tile, kRaggedHidden)
                               : kRaggedHidden % g.h_tile;
    const int want_min_r = kRaggedInter % g.r_tile == 0
                               ? std::min(g.r_tile, kRaggedInter)
                               : kRaggedInter % g.r_tile;
    const int want_min_k = kRaggedHidden % g.k_tile == 0
                               ? std::min(g.k_tile, kRaggedHidden)
                               : kRaggedHidden % g.k_tile;
    const int want_min_rd = kRaggedInter % g.r_tile_down == 0
                                ? std::min(g.r_tile_down, kRaggedInter)
                                : kRaggedInter % g.r_tile_down;
    r.check(trace.min_m_extent == want_min_m && trace.min_h_extent == want_min_h &&
                trace.min_r_extent == want_min_r &&
                trace.min_k_extent == want_min_k &&
                trace.min_r_down_extent == want_min_rd,
            GeomName(g) + ": the ragged tails were staged at their true "
                          "extents (M " + std::to_string(want_min_m) +
                ", R_sub " + std::to_string(want_min_r) + ", K_sub " +
                std::to_string(want_min_k) + ", H_sub " +
                std::to_string(want_min_h) + ", R_sub2 " +
                std::to_string(want_min_rd) + ")",
            "observed M " + std::to_string(trace.min_m_extent) + " R_sub " +
                std::to_string(trace.min_r_extent) + " K_sub " +
                std::to_string(trace.min_k_extent) + " H_sub " +
                std::to_string(trace.min_h_extent) + " R_sub2 " +
                std::to_string(trace.min_r_down_extent));

    // The ragged run must STILL be non-redundant: a tail is exactly where a
    // schedule most plausibly recomputes something to simplify its bookkeeping.
    r.check(trace.act_elements_written ==
                static_cast<std::uint64_t>(kRaggedTokens) * kRaggedInter &&
                trace.gate_up_macs == 2ULL *
                                          static_cast<std::uint64_t>(kRaggedTokens) *
                                          kRaggedInter * kRaggedHidden,
            GeomName(g) + ": the ragged run is still 1.0x -- each activation "
                          "stored once, gate/up exactly 2 * M * inter * hidden",
            "act stores " + std::to_string(trace.act_elements_written) +
                ", gate/up MACs " + std::to_string(trace.gate_up_macs));

    r.check(trace.output_elements_written ==
                static_cast<std::uint64_t>(kRaggedTokens) * kRaggedHidden,
            GeomName(g) + ": the ragged grid still covers the output exactly "
                          "once",
            "wrote " + std::to_string(trace.output_elements_written));

    // The numbers still have to be right at the boundary, against the same
    // two-part bound as the aligned case. Re-derived here from the ragged
    // transcription rather than reused.
    const Transcript hier =
        Transcribe(ragged, x.data(), kRaggedTokens, route.data(), kLimit, g);
    const Transcript flat =
        TranscribeFlat(ragged, x.data(), kRaggedTokens, route.data(), kLimit);
    r.check(BitIdentical(got, hier.y),
            GeomName(g) + ": the ragged run matches the independent "
                          "transcription bit for bit");
    std::vector<float> w2row(static_cast<std::size_t>(kRaggedInter));
    int violations = 0;
    double worst_ratio = 0.0;
    double worst_cos = 1.0;
    for (int m = 0; m < kRaggedTokens; ++m) {
      const std::size_t e = static_cast<std::size_t>(route[static_cast<std::size_t>(m)]);
      for (int h = 0; h < kRaggedHidden; ++h) {
        moe::emu::mxfp4_dequant_row_bf16_cpu(ragged.v2[e], h, w2row.data());
        double flip_term = 0.0;
        for (int rr = 0; rr < kRaggedInter; ++rr) {
          const std::size_t ai = static_cast<std::size_t>(m) * kRaggedInter +
                                 static_cast<std::size_t>(rr);
          flip_term += std::fabs(static_cast<double>(hier.act[ai]) -
                                 static_cast<double>(flat.act[ai])) *
                       std::fabs(static_cast<double>(w2row[static_cast<std::size_t>(rr)]));
        }
        const std::size_t yi = static_cast<std::size_t>(m) * kRaggedHidden +
                               static_cast<std::size_t>(h);
        const double bound =
            ReassociationBound(kRaggedInter, g.r_tile_down,
                               hier.down_abs_sum[yi]) +
            flip_term;
        const double diff = std::fabs(static_cast<double>(got[yi]) -
                                      static_cast<double>(mono[yi]));
        if (diff > bound) ++violations;
        if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
      }
      const std::size_t row = static_cast<std::size_t>(m) * kRaggedHidden;
      worst_cos = std::min(
          worst_cos, CosineSimilarity(got.data() + row, mono.data() + row,
                                      static_cast<std::size_t>(kRaggedHidden)));
    }
    r.check(violations == 0,
            GeomName(g) + ": every ragged output element is inside the derived "
                          "bound against the monolithic oracle",
            std::to_string(violations) + " exceeded it, worst diff/bound " +
                std::to_string(worst_ratio));
    r.check(worst_cos >= 1.0 - 1e-6,
            GeomName(g) + ": ragged per-token cosine stays above 1 - 1e-6",
            "worst cosine " + std::to_string(worst_cos));

    // The tail rows and tail columns specifically must carry real values.
    const auto tail_rows = MeasureSpread(
        got.data() + static_cast<std::size_t>(kRaggedTokens - want_min_m) *
                         kRaggedHidden,
        static_cast<std::size_t>(want_min_m) * kRaggedHidden);
    r.check(tail_rows.NonDegenerate(),
            GeomName(g) + ": the " + std::to_string(want_min_m) +
                " tail tokens carry magnitude and variation",
            "rms " + std::to_string(tail_rows.rms));

    // And the visit order still cannot matter when the grid is ragged, which
    // is where an independence bug would most plausibly hide.
    CubeTileGeometry shuffled = g;
    shuffled.visit = CubeTileVisit::kShuffled;
    shuffled.visit_seed = 0x5A5A;
    const std::vector<float> shuffled_y = RunCube(
        ragged, x.data(), kRaggedTokens, route.data(), kLimit, shuffled, nullptr);
    r.check(BitIdentical(shuffled_y, got),
            GeomName(g) + ": shuffling the ragged grid's visit order changes "
                          "nothing, bit for bit");
    std::printf("           %s ragged: %llu tiles, worst diff/bound %.4f, "
                "worst cosine %.12f\n",
                GeomName(g).c_str(),
                static_cast<unsigned long long>(trace.output_tiles), worst_ratio,
                worst_cos);
  }
}

// ---------------------------------------------------------------------------
// 7. Zero redundant computation, and what it cost instead
// ---------------------------------------------------------------------------
// THE CLAIM THIS SECTION REPLACED is worth recording, because the assertion
// inverted. The (m, h)-grid schedule computed gate and up inside the
// hidden-tile loop, so the suite's job there was to REPORT a recompute factor
// of ceil(hidden / H_tile) -- 64x at hidden 4096 with H_tile 64. The two-phase
// schedule's job is to show that factor is gone, and "gone" is an exact
// identity rather than a smaller number:
//
//     gate_up_macs         == 2 * M * inter * hidden
//     down_macs            ==     M * hidden * inter
//     act_elements_written ==     M * inter
//
// All three are OBSERVED counters, and all three are independent of every
// sub-tile extent -- which is the strongest available statement that no choice
// of tiling reintroduces the recompute.
void AuditScheduleCost(Report& r, const ExpertSet& set, const float* x,
                       int tokens, const std::int32_t* route) {
  section("7. zero redundant computation, and the cost that replaced it");

  const std::uint64_t single_pass_gate_up =
      2ULL * static_cast<std::uint64_t>(tokens) *
      static_cast<std::uint64_t>(set.inter) *
      static_cast<std::uint64_t>(set.hidden);
  const std::uint64_t single_pass_down = static_cast<std::uint64_t>(tokens) *
                                         static_cast<std::uint64_t>(set.hidden) *
                                         static_cast<std::uint64_t>(set.inter);
  const std::uint64_t expected_act = static_cast<std::uint64_t>(tokens) *
                                     static_cast<std::uint64_t>(set.inter);

  // Swept across geometries that differ in EVERY extent, including the two
  // that reassociate. The MAC counts must not budge: a schedule whose work
  // depended on its tiling would be caught here and nowhere else.
  const CubeTileGeometry sweep[] = {
      Geom(16, 64, 64, 64, 64),
      Geom(16, 128, 128, 64, 128),
      Geom(32, 128, 64, 128, 64),
      Geom(32, 64, 128, 64, 256),
  };
  for (const CubeTileGeometry& g : sweep) {
    CubeTileTrace trace;
    RunCube(set, x, tokens, route, kLimit, g, &trace);

    r.check(trace.gate_up_macs == single_pass_gate_up,
            GeomName(g) + ": gate/up is EXACTLY 2 * M * inter * hidden -- "
                          "multiplier 1.0x, no H-tile recompute",
            std::to_string(trace.gate_up_macs) + " against " +
                std::to_string(single_pass_gate_up) + " (x" +
                std::to_string(static_cast<double>(trace.gate_up_macs) /
                               static_cast<double>(single_pass_gate_up)) + ")");
    r.check(trace.down_macs == single_pass_down,
            GeomName(g) + ": the down GEMM is exactly M * hidden * inter -- "
                          "each output block visited once",
            std::to_string(trace.down_macs) + " against " +
                std::to_string(single_pass_down));
    r.check(trace.act_elements_written == expected_act,
            GeomName(g) + ": each of the M * inter activations is stored "
                          "exactly once",
            std::to_string(trace.act_elements_written) + " against " +
                std::to_string(expected_act));

    // THE RESIDUAL COST, as a number rather than a surprise. With r outer and
    // k inner, the X macro-tile is re-read once per (r0, k0) step, i.e.
    // ceil(inter / R_sub) times over the hidden axis per token tile. This is
    // the DMA price of removing the 64x MAC recompute, and it is asserted so
    // that a later schedule change cannot quietly trade one for the other.
    const std::uint64_t expected_x_reads =
        static_cast<std::uint64_t>(CeilDiv(tokens, g.m_tile)) *
        static_cast<std::uint64_t>(CeilDiv(set.inter, g.r_tile)) *
        static_cast<std::uint64_t>(CeilDiv(set.hidden, g.k_tile));
    r.check(trace.x_fragment_reads == expected_x_reads,
            GeomName(g) + ": the X macro-tile is re-read ceil(inter / R_sub) = " +
                std::to_string(CeilDiv(set.inter, g.r_tile)) +
                " times per token tile, which is this schedule's residual cost",
            std::to_string(trace.x_fragment_reads) + " fragment reads against " +
                std::to_string(expected_x_reads));
    // And ONE X fragment serves BOTH the W1 and the W3 contraction, which is
    // why the gate/up multiplier is 1.0x in DMA terms too. If X were re-read
    // per weight matrix there would be twice as many reads as there are
    // (r0, k0) steps.
    r.check(trace.x_fragment_reads * 2ULL ==
                static_cast<std::uint64_t>(CeilDiv(tokens, g.m_tile)) *
                    static_cast<std::uint64_t>(CeilDiv(set.inter, g.r_tile)) *
                    static_cast<std::uint64_t>(CeilDiv(set.hidden, g.k_tile)) *
                    2ULL,
            GeomName(g) + ": one X fragment serves both the W1 and the W3 "
                          "contraction -- it is read once per (r0, k0) step, "
                          "not once per weight matrix");

    std::printf("           %s: gate/up %llu MACs (1.0x), down %llu, act "
                "stores %llu, X reads %llu\n",
                GeomName(g).c_str(),
                static_cast<unsigned long long>(trace.gate_up_macs),
                static_cast<unsigned long long>(trace.down_macs),
                static_cast<unsigned long long>(trace.act_elements_written),
                static_cast<unsigned long long>(trace.x_fragment_reads));
  }

  // WHAT THE OLD SCHEDULE WOULD HAVE COST, at the production geometry, so the
  // identity above is read against the thing it replaced rather than in
  // isolation. Arithmetic on the shapes, printed and not asserted -- it is a
  // statement about a schedule this file no longer implements.
  {
    constexpr std::uint64_t kProdHidden = 4096;
    constexpr std::uint64_t kProdHsub = 64;
    std::printf("           at hidden %llu with H_sub %llu the (m, h)-grid "
                "schedule evaluated gate/up %llux; this one evaluates it 1x\n",
                static_cast<unsigned long long>(kProdHidden),
                static_cast<unsigned long long>(kProdHsub),
                static_cast<unsigned long long>(kProdHidden / kProdHsub));
  }

  // B-FRAGMENT REUSE. The Cube exists because one weight fragment serves all
  // M rows. A grouped dispatch stages each sub-tile once per token tile; an
  // interleaved map must restage per token, and the golden stays correct
  // either way. The counter is what makes that visible instead of implied.
  const CubeTileGeometry g = Geom(16, 64, 64, 64, 64);
  std::vector<std::int32_t> grouped(static_cast<std::size_t>(tokens));
  for (int t = 0; t < tokens; ++t) {
    grouped[static_cast<std::size_t>(t)] = 0;  // one expert: perfect reuse
  }
  CubeTileTrace grouped_trace;
  RunCube(set, x, tokens, grouped.data(), kLimit, g, &grouped_trace);

  // Phase 1's sub-tiles no longer multiply by the H grid -- that is the same
  // saving as the MAC count, seen from the DMA side.
  const std::uint64_t gate_up_blocks =
      static_cast<std::uint64_t>(CeilDiv(tokens, g.m_tile)) *
      static_cast<std::uint64_t>(CeilDiv(set.inter, g.r_tile)) *
      static_cast<std::uint64_t>(CeilDiv(set.hidden, g.k_tile));
  r.check(grouped_trace.gate_up_fragment_stagings == gate_up_blocks,
          "with every token on one expert each [R_sub, K_sub] weight sub-tile "
          "is dequantized exactly once per token tile -- the B-fragment reuse "
          "the Cube exists for, and with no H-grid multiplier on it",
          std::to_string(grouped_trace.gate_up_fragment_stagings) +
              " stagings against " + std::to_string(gate_up_blocks) +
              " sub-tiles");
  const std::uint64_t down_blocks =
      static_cast<std::uint64_t>(CeilDiv(tokens, g.m_tile)) *
      static_cast<std::uint64_t>(CeilDiv(set.hidden, g.h_tile)) *
      static_cast<std::uint64_t>(CeilDiv(set.inter, g.r_tile_down));
  r.check(grouped_trace.down_fragment_stagings == down_blocks,
          "and each [H_sub, R_sub2] sub-tile of W2 exactly once per token tile",
          std::to_string(grouped_trace.down_fragment_stagings) +
              " against " + std::to_string(down_blocks));

  CubeTileTrace interleaved_trace;
  RunCube(set, x, tokens, route, kLimit, g, &interleaved_trace);
  r.check(interleaved_trace.gate_up_fragment_stagings >
              grouped_trace.gate_up_fragment_stagings,
          "an interleaved expert map forces restaging, so the golden pays for "
          "being correct on a map the kernel's grouped dispatch would not "
          "produce",
          std::to_string(interleaved_trace.gate_up_fragment_stagings) +
              " against " +
              std::to_string(grouped_trace.gate_up_fragment_stagings));
  // The MACs do NOT change with the expert map: restaging is a DMA cost, not
  // extra arithmetic. Without this the staging counter could be read as a
  // work multiplier, which it is not.
  r.check(interleaved_trace.gate_up_macs == grouped_trace.gate_up_macs &&
              interleaved_trace.act_elements_written ==
                  grouped_trace.act_elements_written,
          "restaging costs DMA and not MACs -- the work is identical on the "
          "grouped and the interleaved map");
  std::printf("           stagings: %llu grouped, %llu interleaved, over %llu "
              "gate/up sub-tiles\n",
              static_cast<unsigned long long>(
                  grouped_trace.gate_up_fragment_stagings),
              static_cast<unsigned long long>(
                  interleaved_trace.gate_up_fragment_stagings),
              static_cast<unsigned long long>(gate_up_blocks));
}

// ---------------------------------------------------------------------------
// 8. Structural inputs and the argument contract
// ---------------------------------------------------------------------------
void AuditContract(Report& r, const ExpertSet& set, const float* x, int tokens,
                   const std::int32_t* route) {
  section("8. structural inputs and the argument contract");

  const CubeTileGeometry g = Geom(16, 64, 64, 64, 64);
  const int experts = static_cast<int>(set.v1.size());
  std::vector<float> scratch;
  std::vector<float> y(static_cast<std::size_t>(tokens) * set.hidden);

  // --- zero activations: gate = up = 0, so act = 0 and every tile's
  // accumulator sums zeros. Over a poisoned destination, so this also proves
  // the accumulator is initialized per tile.
  {
    const std::vector<float> zero(
        static_cast<std::size_t>(tokens) * set.hidden, 0.0F);
    std::fill(y.begin(), y.end(), 4321.0F);
    moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
        set.v1.data(), set.v3.data(), set.v2.data(), zero.data(), route, tokens,
        experts, y.data(), kLimit, g, nullptr, &scratch);
    bool all_zero = true;
    for (const float v : y) {
      if (v != 0.0F) all_zero = false;
    }
    r.check(all_zero,
            "a zero activation gives an exactly zero output over a poisoned "
            "destination (so each tile's L0C accumulator starts at zero)");
  }

  // --- a zero clamp limit annihilates the expert exactly, at every geometry.
  {
    bool all_zero = true;
    for (const CubeTileGeometry& gg : {Geom(16, 64, 64, 64, 64),
                                       Geom(32, 128, 128, 64, 256)}) {
      moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
          set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens, experts,
          y.data(), 0.0F, gg, nullptr, &scratch);
      for (const float v : y) {
        if (v != 0.0F) all_zero = false;
      }
    }
    r.check(all_zero, "a zero clamp limit gives an exactly zero output");
  }

  auto throws = [](auto&& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };

  // The geometry is rejected BEFORE anything runs. Clamping an over-budget
  // tile to something that fits would silently verify a schedule nobody asked
  // for, and returning a zeroed output is the one failure mode a golden must
  // not have.
  for (const CubeTileGeometry& bad :
       {Geom(16, 256, 128, 64, 128), Geom(24, 64, 64, 64, 64),
        Geom(16, 0, 64, 64, 64), Geom(16, 64, 64, 128, 256),
        Geom(16, 64, 64, 64, 24)}) {
    r.check(throws([&] {
              moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                  set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
                  experts, y.data(), kLimit, bad, nullptr, &scratch);
            }),
            "an inadmissible geometry " + GeomName(bad) +
                " is rejected, not clamped and not silently zeroed");
  }

  r.check(throws([&] {
            moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
                experts, y.data(), kLimit, g, nullptr, nullptr);
          }),
          "a null scratch is rejected");
  r.check(throws([&] {
            std::vector<Mxfp4Matrix> bad = set.v1;
            bad[1].rows = set.inter - kMxfp4GroupSize;
            moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                bad.data(), set.v3.data(), set.v2.data(), x, route, tokens,
                experts, y.data(), kLimit, g, nullptr, &scratch);
          }),
          "an expert whose intermediate size differs from the others is "
          "rejected, not silently read at the wrong stride");
  r.check(throws([&] {
            std::vector<Mxfp4Matrix> bad = set.v2;
            bad[0].packed = nullptr;
            moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                set.v1.data(), set.v3.data(), bad.data(), x, route, tokens,
                experts, y.data(), kLimit, g, nullptr, &scratch);
          }),
          "a weight matrix with a null payload is rejected");
  r.check(throws([&] {
            std::vector<std::int32_t> bad(static_cast<std::size_t>(tokens), 0);
            bad[static_cast<std::size_t>(tokens) - 1] = experts;  // one past the end
            moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                set.v1.data(), set.v3.data(), set.v2.data(), x, bad.data(),
                tokens, experts, y.data(), kLimit, g, nullptr, &scratch);
          }),
          "a token routed to an out-of-range expert id is rejected -- and "
          "before any tile runs, so no partial output is produced");

  // A null trace is allowed: the trace is instrumentation, not a contract.
  r.check(!throws([&] {
            moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
                set.v1.data(), set.v3.data(), set.v2.data(), x, route, tokens,
                experts, y.data(), kLimit, g, nullptr, &scratch);
          }),
          "a null trace is accepted -- the instrumentation is optional");
}

}  // namespace

int main() {
  std::printf("dsv4 MoE prefill: TWO-PHASE on-chip fused CUBE pipeline, host "
              "verification\n");
  std::printf("950PR / DaVinci v3: L0A %zu B, L0B %zu B, L0C %zu B, L1 %zu B, "
              "UB %zu B, fractal %d\n",
              kCubeL0ABytes, kCubeL0BBytes, kCubeL0CBytes, kCubeL1Bytes,
              kCubeUbBytes, kCubeFractal);
  std::printf("seed 0x%llX  hidden %d  inter %d  experts %d  tokens %d  "
              "limit %.1f\n",
              static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
              kTokens, static_cast<double>(kLimit));
  std::printf("ragged fixture: hidden %d  inter %d  tokens %d\n", kRaggedHidden,
              kRaggedInter, kRaggedTokens);

  std::mt19937_64 rng(kSeed);
  Report r;

  ExpertSet set;
  set.Build(kHidden, kInter, kExperts, rng);
  const std::vector<float> x = RandomRows(kTokens, kHidden, 1.0, rng);
  // Interleaved rather than grouped: the golden is per-token, and an
  // interleaved map is the stronger statement about state leaking between
  // tokens or tiles.
  std::vector<std::int32_t> route(static_cast<std::size_t>(kTokens));
  for (int t = 0; t < kTokens; ++t) {
    route[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(t % kExperts);
  }

  // The production sub-tile shape (M 16, R_sub 128, K_sub 128, H_sub 64) with
  // R_sub2 scaled to this fixture's `inter`, plus two that move every extent.
  // R_sub2 is deliberately BELOW `inter` in all three: at R_sub2 == inter the
  // down GEMM has a single reduction block and section 5's bound for it loses
  // its level-2 term, so the suite would be testing a degenerate hierarchy.
  const std::vector<CubeTileGeometry> geometries = {
      Geom(16, 128, 128, 64, 128),
      Geom(16, 64, 64, 64, 64),
      Geom(32, 128, 64, 128, 128),
  };

  AuditBufferBudget(r);
  AuditTilingInvariance(r, set, x.data(), kTokens, route.data());
  AuditSpatialIndependence(r, set, x.data(), kTokens, route.data());
  AuditObservedFootprint(r, set, x.data(), kTokens, route.data(), geometries);
  AuditNumericalEquivalence(r, set, x.data(), kTokens, route.data(), geometries);
  AuditBf16Amplification(r, rng);

  ExpertSet ragged;
  ragged.Build(kRaggedHidden, kRaggedInter, kExperts, rng);
  AuditRagged(r, ragged, rng);

  AuditScheduleCost(r, set, x.data(), kTokens, route.data());
  AuditContract(r, set, x.data(), kTokens, route.data());

  std::printf("\n%d checks, %d failures, %d skipped\n", r.checks, r.failures,
              r.skipped);
  return r.exit_code();
}
