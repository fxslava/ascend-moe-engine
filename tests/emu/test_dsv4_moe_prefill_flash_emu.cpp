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

// HOST VERIFICATION OF THE FLASH-STYLE TILED W4A16 FFN, before any kernel.
//
// The pivot being verified: `dsv4_moe_prefill_cube` stops staging whole
// dequantized weights in a GM workspace and stops materializing the M x inter
// intermediate. Instead the CONTRACTION axis is sliced into chunks of R, each
// chunk's W1/W3/W2 slices are dequantized on the fly, the SwiGLU happens on
// the chunk, and Y accumulates the chunk's contribution.
//
// Nothing here touches a device, a simulator or a toolkit: this is the CI gate
// (CLAUDE.md section 4.1), it links moe_emu only, and it must not skip
// (section 4.2). Per section 5.5 the indexing of a tiled loop is exactly the
// thing that is wrong the first time and needs no hardware to check -- which
// is why the algorithm is verified on the host FIRST and the kernel is written
// against a reference that has already been audited.
//
// WHAT IS CLAIMED, AND HOW EACH CLAIM IS EARNED
//
//   1. The tiled form computes the SAME FUNCTION as the monolithic W4A16
//      oracle, up to ONE reassociation and no other difference. The claim is
//      sharp, so the test is sharp in both directions:
//        * gate, up and act are bit-identical at every R -- established by
//          matching an INDEPENDENT transcription that computes them with the
//          full monolithic GEMV, a different code path from the per-element
//          decode the tiled reference uses;
//        * the down GEMM's additions are reordered into a two-level tree, and
//          the gap against the monolithic flat sum is held to an ELEMENTWISE
//          bound derived from the format and the two summation shapes -- not
//          to a figure read off the output (section 4.7);
//        * the two extremes must be BIT-EXACT: R >= inter is a single chunk
//          whose accumulator is the flat sum, and R == 1 adds every product to
//          Y on its own, which is also the flat sum. A tolerance would hide
//          both.
//
//   2. The comparison has TEETH. A bound nothing can violate proves nothing,
//      so two realistic indexing bugs -- the W2 column offset dropped, the
//      act offset dropped -- are injected into the transcription and must
//      exceed the bound by orders of magnitude.
//
//   3. Nothing is materialized. Held to an arithmetic statement rather than a
//      comment: the scratch footprint is `hidden + 3 * min(R, inter)` floats,
//      it is the SAME NUMBER for two different `inter` at fixed R, and it is
//      smaller than a single dequantized W2 row.
//
//   4. Ragged everything. A token count that is not a multiple of the
//      kernel's 16-token tile, an `inter` that is not a multiple of R (so the
//      final chunk is short), and an R that splits a 32-element MXFP4 group.
//
//   5. The boundary regimes: both SwiGLU clamps saturating, a zero clamp
//      limit, an effectively infinite one, zero activations and zero weights.
//      DSV4 prefill carries no router WEIGHTS -- the dispatch has already
//      grouped the tokens, so a token is routed, not blended -- so the
//      "extreme routing" case is realized as the per-token expert map plus
//      extreme activation amplitudes, and a wrong-expert read is shown to be
//      visible rather than assumed to be.
//
//   6. Non-degeneracy of the golden itself: nonzero RMS and nonzero stddev,
//      never CosineSimilarity(want, want) == 1.0 (section 4.8).
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

// Printed, so a failure here is reproducible without a fixture on disk
// (CLAUDE.md section 4.5).
constexpr std::uint64_t kSeed = 0xF1A5'4A16'5EEDULL;

// Scaled-down geometry. The reduction axes must be whole 32-element MXFP4
// groups, which is all the reference requires; the kernel's own admissible
// geometry (hidden/inter multiples of 512) is a Cube tiling constraint and is
// checked where it binds, in tests/sim.
//
// inter = 256 is chosen so that the audited chunk sizes give genuinely
// different decompositions: 256 chunks at R = 1, 8 at R = 32, 4 at R = 64,
// 2 at R = 128, 1 at R = 256, and ragged tails at R = 48 and R = 100.
constexpr int kHidden = 128;
constexpr int kInter = 256;
constexpr int kExperts = 3;
constexpr float kLimit = 10.0F;

// The chunk sizes under audit. 1 and 256 are the bit-exact extremes; 48 splits
// a 32-element MXFP4 group AND leaves a ragged tail; 100 is both and is not
// even a multiple of 4; 512 exceeds `inter` and must collapse to one chunk.
constexpr int kChunkSizes[] = {1, 32, 48, 64, 100, 128, 256, 512};

// FP32's unit roundoff: the significand is 24 bits, so one RNE add carries at
// most 2^-24 relative.
constexpr double kU = 1.0 / 16777216.0;  // 2^-24

int EffectiveSpan(int chunk, int inter) { return chunk < inter ? chunk : inter; }

int ChunkCount(int chunk, int inter) {
  const int span = EffectiveSpan(chunk, inter);
  return (inter + span - 1) / span;
}

// ---------------------------------------------------------------------------
// The independent transcription of the tiled algorithm
// ---------------------------------------------------------------------------
// This is what makes the suite an audit rather than a restatement. It computes
// the same answer by DIFFERENT code paths at every step:
//
//   * gate and up come from `mxfp4_gemv_w4a16_cpu` over the FULL intermediate
//     axis -- the monolithic GEMV, not a row range -- which is what pins the
//     claim "slicing W1/W3's output rows reorders nothing";
//   * W2 is dequantized by the public `mxfp4_dequant_row_bf16_cpu` row
//     helper, not by the reference's per-element decode, which is what pins
//     the nibble order and the group-scale indexing inside a chunk;
//   * only the ADDITION ORDER is transcribed from the algorithm: a per-chunk
//     accumulator added into Y.
//
// It also returns the quantity the error bound needs: S[h] = sum_r |act[r]| *
// |W2b[h,r]|, the absolute-value sum of the down reduction's terms. That is a
// property of the INPUTS, which is what makes a bound built from it derived
// rather than fitted.
enum class Sabotage {
  kNone,
  kW2ColumnOffsetDropped,  // reads W2's columns from 0 in every chunk
  kActOffsetDropped,       // reads act from 0 in every chunk
};

struct Tiled {
  std::vector<float> y;
  std::vector<float> act;      // the full activation, chunk-independent
  std::vector<double> abs_sum;  // S[h], in double so the bound's own arithmetic is not the lossy part
  int chunks = 0;
  int span = 0;
};

Tiled TranscribeTiled(const Mxfp4Matrix& w1, const Mxfp4Matrix& w3,
                      const Mxfp4Matrix& w2, const float* x, float limit,
                      int chunk, Sabotage sabotage = Sabotage::kNone) {
  const int hidden = w1.cols;
  const int inter = w1.rows;

  std::vector<float> xb(static_cast<std::size_t>(hidden));
  for (int c = 0; c < hidden; ++c) {
    xb[static_cast<std::size_t>(c)] = bf16_round_cpu(x[c]);
  }

  std::vector<float> gate(static_cast<std::size_t>(inter));
  std::vector<float> up(static_cast<std::size_t>(inter));
  moe::emu::mxfp4_gemv_w4a16_cpu(w1, xb.data(), gate.data());
  moe::emu::mxfp4_gemv_w4a16_cpu(w3, xb.data(), up.data());

  Tiled t;
  t.span = EffectiveSpan(chunk, inter);
  t.act.resize(static_cast<std::size_t>(inter));
  for (int r = 0; r < inter; ++r) {
    const float g = std::min(gate[static_cast<std::size_t>(r)], limit);
    const float u =
        std::min(limit, std::max(-limit, up[static_cast<std::size_t>(r)]));
    t.act[static_cast<std::size_t>(r)] = bf16_round_cpu(SiluRef(g) * u);
  }

  // W2 dequantized through the public row helper, once per output row.
  std::vector<float> w2rows(static_cast<std::size_t>(hidden) *
                            static_cast<std::size_t>(inter));
  for (int h = 0; h < hidden; ++h) {
    moe::emu::mxfp4_dequant_row_bf16_cpu(
        w2, h, w2rows.data() + static_cast<std::size_t>(h) * inter);
  }

  t.y.assign(static_cast<std::size_t>(hidden), 0.0F);
  t.abs_sum.assign(static_cast<std::size_t>(hidden), 0.0);
  for (int r0 = 0; r0 < inter; r0 += t.span) {
    const int rows = std::min(t.span, inter - r0);
    ++t.chunks;
    for (int h = 0; h < hidden; ++h) {
      float partial = 0.0F;
      for (int i = 0; i < rows; ++i) {
        const int act_index =
            (sabotage == Sabotage::kActOffsetDropped) ? i : r0 + i;
        const int w2_index =
            (sabotage == Sabotage::kW2ColumnOffsetDropped) ? i : r0 + i;
        const float a = t.act[static_cast<std::size_t>(act_index)];
        const float w = w2rows[static_cast<std::size_t>(h) * inter +
                               static_cast<std::size_t>(w2_index)];
        partial += a * w;
        t.abs_sum[static_cast<std::size_t>(h)] +=
            static_cast<double>(std::fabs(a)) * static_cast<double>(std::fabs(w));
      }
      t.y[static_cast<std::size_t>(h)] += partial;
    }
  }
  return t;
}

// THE DERIVED BOUND on |tiled[h] - monolithic[h]|, in full.
//
// Both forms reduce the SAME products p_r = act[r] * W2b[h,r]. Each factor is
// a BF16 value, i.e. carries at most 8 significand bits, so p_r needs at most
// 16 and is EXACT in FP32. Nothing but the additions can differ, and that is
// what makes this a pure reassociation question rather than an error-budget
// one.
//
// For a flat FP32 sum of n terms the standard first-order bound is
//     |computed - exact| <= (n - 1) * u * sum_r |p_r|,        u = 2^-24.
// For the blocked sum with span R and K = ceil(n / R) chunks, a term passes
// through at most (R - 1) additions inside its chunk and (K - 1)
// accumulations into Y, so
//     |computed - exact| <= (R - 1 + K - 1) * u * sum_r |p_r|.
// The two computed values therefore differ by at most the sum of the two:
//     ((n - 1) + (R - 1) + (K - 1)) * u * S[h],   S[h] = sum_r |p_r|.
//
// Every factor comes from the formats and the two summation shapes. S[h] is a
// property of the inputs, computed by the transcription above. Nothing in it
// is read off the observed error.
double DownReductionBound(int inter, int span, int chunks, double abs_sum) {
  const double chain = static_cast<double>(inter - 1) +
                       static_cast<double>(span - 1) +
                       static_cast<double>(chunks - 1);
  return chain * kU * abs_sum;
}

// Cosine is kept strict even where the magnitude bound must be loose:
// rounding moves magnitude, structural bugs move direction (section 4.7). A
// relative perturbation of size d moves the cosine by about d^2 / 2, and the
// bound above is at most 2 * inter * 2^-24 * (S/|y|) ~ 1e-4 even at a
// condition number of 10, so 1 - 1e-6 is conservative by two orders.
constexpr double kCosineFloor = 1.0 - 1e-6;

bool BitIdentical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<float> RandomRow(int n, double sigma, std::mt19937_64& rng) {
  std::vector<float> v(static_cast<std::size_t>(n));
  std::normal_distribution<float> draw(0.0F, static_cast<float>(sigma));
  for (float& e : v) e = draw(rng);
  return v;
}

// ---------------------------------------------------------------------------
// 1. The tiled form is the monolithic form, up to one reassociation
// ---------------------------------------------------------------------------
void AuditChunkEquivalence(Report& r, const Bank& w1, const Bank& w3,
                           const Bank& w2, const std::vector<float>& x) {
  section("1. tiled == monolithic, up to the down GEMM's reassociation");

  std::vector<float> scratch;
  std::vector<float> want(static_cast<std::size_t>(kHidden));
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0), w2.View(0),
                                              x.data(), want.data(), kLimit,
                                              &scratch);
  const auto spread = MeasureSpread(want.data(), want.size());
  r.check(spread.NonDegenerate(),
          "the monolithic golden the tiled form is compared against carries "
          "magnitude and variation",
          "rms " + std::to_string(spread.rms) + " stddev " +
              std::to_string(spread.stddev));

  // The activation is chunk-independent, so one transcription of it serves as
  // the cross-chunk witness: every R's transcription must produce the same
  // `act`, and the implementation must match the transcription.
  const Tiled reference_act =
      TranscribeTiled(w1.View(0), w3.View(0), w2.View(0), x.data(), kLimit, 1);

  for (const int chunk : kChunkSizes) {
    const int span = EffectiveSpan(chunk, kInter);
    const int chunks = ChunkCount(chunk, kInter);
    const Tiled t = TranscribeTiled(w1.View(0), w3.View(0), w2.View(0),
                                    x.data(), kLimit, chunk);
    std::vector<float> got(static_cast<std::size_t>(kHidden));
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
        w1.View(0), w3.View(0), w2.View(0), x.data(), got.data(), kLimit, chunk,
        &scratch);

    const std::string tag =
        "R = " + std::to_string(chunk) + " (" + std::to_string(chunks) +
        " chunk(s) of " + std::to_string(span) +
        (kInter % span == 0 ? "" : ", ragged tail of " +
                                       std::to_string(kInter % span)) +
        ")";

    // (a) the implementation IS the algorithm: bit for bit against the
    // independent transcription, at every chunk size.
    r.check(BitIdentical(got, t.y),
            tag + ": matches the independent tiled transcription bit for bit",
            "max abs " + std::to_string(MaxAbsDiff(got.data(), t.y.data(),
                                                   got.size())));

    // (b) the intermediate does not depend on the chunking at all.
    r.check(BitIdentical(t.act, reference_act.act),
            tag + ": gate/up/act are bit-identical to the R = 1 transcription "
                  "(the chunking cannot reorder a row's own reduction)",
            "max abs " + std::to_string(MaxAbsDiff(t.act.data(),
                                                   reference_act.act.data(),
                                                   t.act.size())));

    // (c) against the monolithic oracle: elementwise, inside the derived
    // bound -- and BIT-EXACT at the two extremes, where the two summation
    // orders coincide and a tolerance would hide a real difference.
    const bool extreme = (span == 1) || (chunks == 1);
    if (extreme) {
      r.check(BitIdentical(got, want),
              tag + ": bit-exact against the monolithic oracle (this "
                    "decomposition reproduces the flat sum exactly)",
              "max abs " +
                  std::to_string(MaxAbsDiff(got.data(), want.data(), got.size())));
    } else {
      int violations = 0;
      double worst_ratio = 0.0;
      double worst_bound = 0.0;
      for (int h = 0; h < kHidden; ++h) {
        const std::size_t i = static_cast<std::size_t>(h);
        const double diff = std::fabs(static_cast<double>(got[i]) -
                                      static_cast<double>(want[i]));
        const double bound =
            DownReductionBound(kInter, span, chunks, t.abs_sum[i]);
        if (diff > bound) ++violations;
        if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
        worst_bound = std::max(worst_bound, bound);
      }
      r.check(violations == 0,
              tag + ": every output element is inside the derived "
                    "((inter-1)+(R-1)+(K-1)) * 2^-24 * S[h] reassociation bound",
              std::to_string(violations) +
                  " element(s) exceeded it; worst diff/bound ratio " +
                  std::to_string(worst_ratio));
      std::printf("           %s: worst diff/bound %.4f (largest bound "
                  "%.3e)\n",
                  tag.c_str(), worst_ratio, worst_bound);
    }

    const double rel = RelativeL2(got.data(), want.data(), got.size());
    const double cos = CosineSimilarity(got.data(), want.data(), got.size());
    r.check(cos >= kCosineFloor,
            tag + ": cosine against the monolithic oracle stays above "
                  "1 - 1e-6 (reassociation moves magnitude, not direction)",
            "cosine " + std::to_string(cos));
    std::printf("           %s: relative-L2 %.3e, cosine %.12f\n", tag.c_str(),
                rel, cos);
  }
}

// ---------------------------------------------------------------------------
// 2. The comparison has teeth
// ---------------------------------------------------------------------------
void AuditBoundHasTeeth(Report& r, const Bank& w1, const Bank& w3,
                        const Bank& w2, const std::vector<float>& x) {
  section("2. a tiled indexing bug cannot pass the bound");

  // A bound nothing can violate is not a check. These are the two indexing
  // bugs a chunked down GEMM actually has -- forgetting to advance W2's
  // column offset with the chunk, and forgetting to rebase the activation
  // index inside the chunk. Both leave the FIRST chunk correct, which is
  // precisely why a test that only ran one chunk would miss them.
  constexpr int kChunk = 64;
  std::vector<float> scratch;
  std::vector<float> want(static_cast<std::size_t>(kHidden));
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0), w2.View(0),
                                              x.data(), want.data(), kLimit,
                                              &scratch);
  const Tiled honest = TranscribeTiled(w1.View(0), w3.View(0), w2.View(0),
                                       x.data(), kLimit, kChunk);

  struct Case {
    Sabotage kind;
    const char* what;
  };
  const Case cases[] = {
      {Sabotage::kW2ColumnOffsetDropped,
       "W2's column offset not advanced with the chunk"},
      {Sabotage::kActOffsetDropped,
       "the activation index not rebased inside the chunk"},
  };

  for (const Case& c : cases) {
    const Tiled broken = TranscribeTiled(w1.View(0), w3.View(0), w2.View(0),
                                         x.data(), kLimit, kChunk, c.kind);
    int exceeded = 0;
    double worst_ratio = 0.0;
    for (int h = 0; h < kHidden; ++h) {
      const std::size_t i = static_cast<std::size_t>(h);
      const double diff = std::fabs(static_cast<double>(broken.y[i]) -
                                    static_cast<double>(want[i]));
      // The bound is built from the HONEST run's term magnitudes, which is
      // how the real comparison in section 1 computes it.
      const double bound = DownReductionBound(kInter, honest.span,
                                              honest.chunks, honest.abs_sum[i]);
      if (diff > bound) ++exceeded;
      if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
    }
    r.check(exceeded > kHidden / 2,
            std::string("a tiled run with ") + c.what +
                " violates the reassociation bound on most output elements",
            std::to_string(exceeded) + " of " + std::to_string(kHidden) +
                " elements exceeded it, worst ratio " +
                std::to_string(worst_ratio));
    std::printf("           %s: %d/%d elements out of bound, worst diff/bound "
                "%.3e\n",
                c.what, exceeded, kHidden, worst_ratio);
  }
}

// ---------------------------------------------------------------------------
// 3. Nothing is materialized
// ---------------------------------------------------------------------------
void AuditNoMaterialization(Report& r, const Bank& w1, const Bank& w3,
                            const Bank& w2, const std::vector<float>& x,
                            std::mt19937_64& rng) {
  section("3. the resident set is set by the chunk, not by the intermediate");

  // The formula, stated independently of the implementation.
  for (const int chunk : kChunkSizes) {
    const int span = EffectiveSpan(chunk, kInter);
    const std::size_t want = static_cast<std::size_t>(kHidden) +
                             3U * static_cast<std::size_t>(span);
    r.check(moe::emu::moe_prefill_flash_scratch_floats(kHidden, kInter, chunk) ==
                want,
            "the documented footprint at R = " + std::to_string(chunk) +
                " is hidden + 3 * min(R, inter) floats",
            "got " + std::to_string(moe::emu::moe_prefill_flash_scratch_floats(
                         kHidden, kInter, chunk)));
  }

  // And the reference actually allocates exactly that -- no row buffer, no
  // weight tile, nothing proportional to `inter`.
  std::vector<float> scratch;
  std::vector<float> y(static_cast<std::size_t>(kHidden));
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
      w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kLimit, 32,
      &scratch);
  const std::size_t small_inter_footprint = scratch.size();
  r.check(small_inter_footprint ==
              moe::emu::moe_prefill_flash_scratch_floats(kHidden, kInter, 32),
          "the tiled expert's scratch is exactly the documented footprint "
          "after a run",
          "allocated " + std::to_string(small_inter_footprint));

  // THE CLAIM THAT MATTERS: double `inter` and the footprint does not move.
  // The monolithic form's does, by construction -- it holds gate and up over
  // the whole intermediate axis.
  const int wide_inter = kInter * 2;
  const Bank wide1 = MakeBank(wide_inter, kHidden, 1, rng);
  const Bank wide3 = MakeBank(wide_inter, kHidden, 1, rng);
  const Bank wide2 = MakeBank(kHidden, wide_inter, 1, rng);
  std::vector<float> wide_scratch;
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
      wide1.View(0), wide3.View(0), wide2.View(0), x.data(), y.data(), kLimit,
      32, &wide_scratch);
  r.check(wide_scratch.size() == small_inter_footprint,
          "doubling `inter` from " + std::to_string(kInter) + " to " +
              std::to_string(wide_inter) +
              " leaves the tiled footprint unchanged",
          std::to_string(small_inter_footprint) + " -> " +
              std::to_string(wide_scratch.size()));

  std::vector<float> mono_scratch;
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(wide1.View(0), wide3.View(0),
                                              wide2.View(0), x.data(), y.data(),
                                              kLimit, &mono_scratch);
  r.check(mono_scratch.size() > wide_scratch.size(),
          "the monolithic form, on the same geometry, holds strictly more -- "
          "the whole intermediate axis",
          std::to_string(mono_scratch.size()) + " vs " +
              std::to_string(wide_scratch.size()));

  // Smaller than ONE dequantized W2 row, let alone the hidden x R tile a
  // staging implementation would hold. This is the GM workspace that the
  // pivot removes, stated as a number.
  const std::size_t one_w2_row = static_cast<std::size_t>(wide_inter);
  const std::size_t w2_tile = static_cast<std::size_t>(kHidden) * 32U;
  r.check(wide_scratch.size() - static_cast<std::size_t>(kHidden) < one_w2_row,
          "the chunk's own buffers (3R floats) are smaller than a single "
          "dequantized W2 row",
          std::to_string(wide_scratch.size() - kHidden) + " vs " +
              std::to_string(one_w2_row));
  std::printf("           footprint: %zu floats at hidden %d / R 32, for "
              "inter %d AND inter %d; a staged hidden x R W2 tile alone would "
              "be %zu\n",
              wide_scratch.size(), kHidden, kInter, wide_inter, w2_tile);

  // At the production geometry the same formula says what the pivot buys on
  // the intermediate: 3R against 2 * inter floats per token.
  constexpr int kProdHidden = 4096;
  constexpr int kProdInter = 2048;
  constexpr int kProdChunk = 128;
  const std::size_t prod_flash = moe::emu::moe_prefill_flash_scratch_floats(
      kProdHidden, kProdInter, kProdChunk);
  const std::size_t prod_mono = static_cast<std::size_t>(kProdHidden) +
                                2U * static_cast<std::size_t>(kProdInter);
  r.check(prod_flash - kProdHidden < (prod_mono - kProdHidden) / 10,
          "at hidden 4096 / inter 2048 / R 128 the chunk's buffers are under a "
          "tenth of the monolithic intermediate",
          std::to_string(prod_flash - kProdHidden) + " vs " +
              std::to_string(prod_mono - kProdHidden));
  std::printf("           production geometry: 3R = %zu floats against "
              "2 * inter = %zu\n",
              prod_flash - kProdHidden, prod_mono - kProdHidden);
}

// ---------------------------------------------------------------------------
// 4. The batch: ragged tokens, ragged chunks, and the expert map
// ---------------------------------------------------------------------------
void AuditRaggedBatch(Report& r, const Bank& w1, const Bank& w3,
                      const Bank& w2, std::mt19937_64& rng) {
  section("4. ragged token count, ragged chunk, and the per-token expert map");

  // 19 tokens against the kernel's 16-token tile: one full tile and a partial
  // one of 3. The tail is where every off-by-one lives.
  constexpr int kTokens = 19;
  constexpr int kMTile = 16;  // kPrefillCubeMTile
  static_assert(kTokens % kMTile != 0, "the point is a ragged final tile");

  const std::vector<float> x = RandomRow(kTokens * kHidden, 1.0, rng);

  // Grouped by expert, which is the dispatch order the kernel assumes.
  std::vector<std::int32_t> route(static_cast<std::size_t>(kTokens));
  for (int t = 0; t < kTokens; ++t) {
    route[static_cast<std::size_t>(t)] = t < 7 ? 0 : (t < 13 ? 1 : 2);
  }

  std::vector<Mxfp4Matrix> v1;
  std::vector<Mxfp4Matrix> v3;
  std::vector<Mxfp4Matrix> v2;
  for (int e = 0; e < kExperts; ++e) {
    v1.push_back(w1.View(e));
    v3.push_back(w3.View(e));
    v2.push_back(w2.View(e));
  }

  std::vector<float> scratch;
  std::vector<float> one(static_cast<std::size_t>(kHidden));

  // R = 100 on inter = 256: 2 chunks of 100 and a ragged 56, and 100 splits a
  // 32-element MXFP4 group. Run the whole batch through the worst chunking
  // available and through a clean one.
  for (const int chunk : {32, 100}) {
    std::vector<float> y(static_cast<std::size_t>(kTokens) * kHidden);
    moe::emu::moe_prefill_flash_ffn_w4a16_cpu(
        v1.data(), v3.data(), v2.data(), x.data(), route.data(), kTokens,
        kExperts, y.data(), kLimit, chunk, &scratch);

    // ROW INDEPENDENCE under tiling: a token's output must be the
    // single-token tiled expert on that token's row, bit for bit. This is
    // what licenses a collapsed token grid on device.
    int mismatched = 0;
    int first_bad = -1;
    for (int t = 0; t < kTokens; ++t) {
      const int e = route[static_cast<std::size_t>(t)];
      const std::size_t row = static_cast<std::size_t>(t) * kHidden;
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          v1[static_cast<std::size_t>(e)], v3[static_cast<std::size_t>(e)],
          v2[static_cast<std::size_t>(e)], x.data() + row, one.data(), kLimit,
          chunk, &scratch);
      if (std::memcmp(y.data() + row, one.data(), one.size() * sizeof(float)) !=
          0) {
        ++mismatched;
        if (first_bad < 0) first_bad = t;
      }
    }
    r.check(mismatched == 0,
            "R = " + std::to_string(chunk) +
                ": all 19 tokens -- including the 3 in the ragged final "
                "16-token tile -- equal the single-token tiled expert on their "
                "own row, bit for bit",
            std::to_string(mismatched) + " token(s) differed, first at index " +
                std::to_string(first_bad));

    // And against the MONOLITHIC batch, elementwise inside the derived bound.
    // Computed per token against that token's own expert, because S[h]
    // depends on the row.
    std::vector<float> y_mono(static_cast<std::size_t>(kTokens) * kHidden);
    moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
        v1.data(), v3.data(), v2.data(), x.data(), route.data(), kTokens,
        kExperts, y_mono.data(), kLimit, &scratch);
    int violations = 0;
    double worst_ratio = 0.0;
    double worst_rel = 0.0;
    double worst_cos = 1.0;
    for (int t = 0; t < kTokens; ++t) {
      const int e = route[static_cast<std::size_t>(t)];
      const std::size_t row = static_cast<std::size_t>(t) * kHidden;
      const Tiled tr = TranscribeTiled(v1[static_cast<std::size_t>(e)],
                                       v3[static_cast<std::size_t>(e)],
                                       v2[static_cast<std::size_t>(e)],
                                       x.data() + row, kLimit, chunk);
      for (int h = 0; h < kHidden; ++h) {
        const std::size_t i = row + static_cast<std::size_t>(h);
        const double diff = std::fabs(static_cast<double>(y[i]) -
                                      static_cast<double>(y_mono[i]));
        const double bound =
            DownReductionBound(kInter, tr.span, tr.chunks,
                               tr.abs_sum[static_cast<std::size_t>(h)]);
        if (diff > bound) ++violations;
        if (bound > 0.0) worst_ratio = std::max(worst_ratio, diff / bound);
      }
      worst_rel = std::max(worst_rel,
                           RelativeL2(y.data() + row, y_mono.data() + row,
                                      static_cast<std::size_t>(kHidden)));
      worst_cos = std::min(worst_cos,
                           CosineSimilarity(y.data() + row, y_mono.data() + row,
                                            static_cast<std::size_t>(kHidden)));
    }
    r.check(violations == 0,
            "R = " + std::to_string(chunk) +
                ": every element of the 19-token batch is inside the derived "
                "reassociation bound against the monolithic batch",
            std::to_string(violations) +
                " element(s) exceeded it, worst diff/bound " +
                std::to_string(worst_ratio));
    r.check(worst_cos >= kCosineFloor,
            "R = " + std::to_string(chunk) +
                ": per-token cosine against the monolithic batch stays above "
                "1 - 1e-6",
            "worst cosine " + std::to_string(worst_cos));
    std::printf("           R = %d: worst diff/bound %.4f, worst per-token "
                "relative-L2 %.3e, worst cosine %.12f\n",
                chunk, worst_ratio, worst_rel, worst_cos);

    // A WRONG-EXPERT READ MUST BE VISIBLE, or the comparison above could pass
    // on a lucky fixture.
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
        v1[1], v3[1], v2[1], x.data(), one.data(), kLimit, chunk, &scratch);
    const double wrong_expert_gap =
        RelativeL2(one.data(), y.data(), static_cast<std::size_t>(kHidden));
    r.check(wrong_expert_gap > 0.1,
            "R = " + std::to_string(chunk) +
                ": running token 0 through expert 1 instead of expert 0 "
                "changes the answer by more than 10%, so a wrong-expert read "
                "cannot hide",
            "relative-L2 " + std::to_string(wrong_expert_gap));

    // The ragged tail must have been computed, not skipped, and the golden
    // must be non-degenerate (section 4.8).
    const auto tail = MeasureSpread(
        y.data() + static_cast<std::size_t>(kMTile) * kHidden,
        static_cast<std::size_t>(kTokens - kMTile) * kHidden);
    r.check(tail.NonDegenerate(),
            "R = " + std::to_string(chunk) +
                ": the ragged tail rows carry magnitude and variation",
            "rms " + std::to_string(tail.rms) + " stddev " +
                std::to_string(tail.stddev));
    const auto whole = MeasureSpread(y.data(), y.size());
    r.check(whole.NonDegenerate(),
            "R = " + std::to_string(chunk) +
                ": the whole tiled batch is non-degenerate",
            "rms " + std::to_string(whole.rms) + " stddev " +
                std::to_string(whole.stddev));
    std::printf("           R = %d: golden rms %.4f stddev %.4f min %.4f "
                "max %.4f\n",
                chunk, whole.rms, whole.stddev, whole.min, whole.max);
  }

  // ARBITRARY M, not just one ragged case. The kernel tiles tokens in groups
  // of 16, so the interesting counts are the ones around that boundary: a
  // single token, one short of a tile, exactly a tile, one past it, and more
  // than two tiles. Each must equal the per-token tiled expert bit for bit --
  // the batch loop may not carry state between tokens, and the chunk loop
  // living inside it must not either.
  for (const int tokens : {1, 15, 16, 17, 33}) {
    const std::vector<float> xm =
        RandomRow(tokens * kHidden, 1.0, rng);
    std::vector<std::int32_t> rm(static_cast<std::size_t>(tokens));
    for (int t = 0; t < tokens; ++t) {
      // Cycled rather than grouped here: the batch oracle is per-token, so an
      // interleaved map is the stronger statement about state leaking between
      // tokens. (The kernel's own grouped-dispatch assumption is a device
      // tiling constraint and is checked in tests/sim.)
      rm[static_cast<std::size_t>(t)] = static_cast<std::int32_t>(t % kExperts);
    }
    std::vector<float> ym(static_cast<std::size_t>(tokens) * kHidden);
    moe::emu::moe_prefill_flash_ffn_w4a16_cpu(
        v1.data(), v3.data(), v2.data(), xm.data(), rm.data(), tokens, kExperts,
        ym.data(), kLimit, 48, &scratch);
    int mismatched = 0;
    for (int t = 0; t < tokens; ++t) {
      const std::size_t e = static_cast<std::size_t>(rm[static_cast<std::size_t>(t)]);
      const std::size_t row = static_cast<std::size_t>(t) * kHidden;
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          v1[e], v3[e], v2[e], xm.data() + row, one.data(), kLimit, 48,
          &scratch);
      if (std::memcmp(ym.data() + row, one.data(),
                      one.size() * sizeof(float)) != 0) {
        ++mismatched;
      }
    }
    r.check(mismatched == 0,
            "M = " + std::to_string(tokens) +
                " with an interleaved expert map at R = 48 (ragged chunk): "
                "every token equals the per-token tiled expert bit for bit",
            std::to_string(mismatched) + " of " + std::to_string(tokens) +
                " token(s) differed");
    const auto spread = MeasureSpread(ym.data(), ym.size());
    r.check(spread.NonDegenerate(),
            "M = " + std::to_string(tokens) +
                ": the golden carries magnitude and variation",
            "rms " + std::to_string(spread.rms) + " stddev " +
                std::to_string(spread.stddev));
  }
}

// ---------------------------------------------------------------------------
// 5. Boundary regimes
// ---------------------------------------------------------------------------
void AuditBoundaryRegimes(Report& r, const Bank& w1, const Bank& w3,
                          const Bank& w2, std::mt19937_64& rng) {
  section("5. clamp boundaries, extreme amplitudes, and structural inputs");

  std::vector<float> scratch;
  std::vector<float> y(static_cast<std::size_t>(kHidden));

  // --- BOTH CLAMPS SATURATING. Amplify x until the gate GEMM clears the
  // limit on both sides; then the asymmetric clamp (gate bounded ABOVE only)
  // is doing real work inside the chunk loop, and the transcription -- which
  // applies the clamp outside any chunk -- must still agree bit for bit.
  {
    const std::vector<float> x = RandomRow(kHidden, 6.0, rng);
    std::vector<float> xb(static_cast<std::size_t>(kHidden));
    for (int c = 0; c < kHidden; ++c) {
      xb[static_cast<std::size_t>(c)] =
          bf16_round_cpu(x[static_cast<std::size_t>(c)]);
    }
    std::vector<float> gate(static_cast<std::size_t>(kInter));
    std::vector<float> up(static_cast<std::size_t>(kInter));
    moe::emu::mxfp4_gemv_w4a16_cpu(w1.View(0), xb.data(), gate.data());
    moe::emu::mxfp4_gemv_w4a16_cpu(w3.View(0), xb.data(), up.data());
    int above = 0, below = 0, up_clamped = 0;
    for (int i = 0; i < kInter; ++i) {
      const float g = gate[static_cast<std::size_t>(i)];
      const float u = up[static_cast<std::size_t>(i)];
      if (g > kLimit) ++above;
      if (g < -kLimit) ++below;
      if (std::fabs(u) > kLimit) ++up_clamped;
    }
    r.check(above > 0 && below > 0 && up_clamped > 0,
            "the amplified probe drives the gate past the limit on BOTH sides "
            "and clamps the up projection, so the boundary is exercised",
            std::to_string(above) + " above, " + std::to_string(below) +
                " below, " + std::to_string(up_clamped) + " up-clamped");

    for (const int chunk : {32, 48, 128}) {
      const Tiled t = TranscribeTiled(w1.View(0), w3.View(0), w2.View(0),
                                      x.data(), kLimit, chunk);
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kLimit, chunk,
          &scratch);
      r.check(BitIdentical(y, t.y),
              "at saturating amplitude, R = " + std::to_string(chunk) +
                  " still matches the transcription bit for bit (the clamp is "
                  "applied per chunk exactly as it is applied whole)",
              "max abs " +
                  std::to_string(MaxAbsDiff(y.data(), t.y.data(), y.size())));
    }
  }

  // --- A ZERO CLAMP LIMIT annihilates the expert exactly: up clamps to 0, so
  // every activation is a product with zero and the down GEMM sums zeros.
  // Exact at every R, which is a statement a tolerance cannot make.
  {
    const std::vector<float> x = RandomRow(kHidden, 2.0, rng);
    bool all_zero = true;
    for (const int chunk : kChunkSizes) {
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), 0.0F, chunk,
          &scratch);
      for (const float v : y) {
        if (v != 0.0F) all_zero = false;
      }
    }
    r.check(all_zero,
            "a zero clamp limit gives an exactly zero output at every chunk "
            "size");
  }

  // --- AN EFFECTIVELY INFINITE LIMIT: nothing clamps, so the tiled form must
  // still track the monolithic one. Checked against the monolithic oracle at
  // the same limit, which is the comparison that would break if the chunk
  // loop applied the clamp to a partial sum.
  {
    const std::vector<float> x = RandomRow(kHidden, 1.0, rng);
    constexpr float kHuge = 1.0e30F;
    std::vector<float> want(static_cast<std::size_t>(kHidden));
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0),
                                                w2.View(0), x.data(),
                                                want.data(), kHuge, &scratch);
    const Tiled t = TranscribeTiled(w1.View(0), w3.View(0), w2.View(0),
                                    x.data(), kHuge, 64);
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
        w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kHuge, 64,
        &scratch);
    r.check(BitIdentical(y, t.y),
            "with the clamp effectively disabled, R = 64 matches the "
            "transcription bit for bit");
    int violations = 0;
    for (int h = 0; h < kHidden; ++h) {
      const std::size_t i = static_cast<std::size_t>(h);
      const double diff = std::fabs(static_cast<double>(y[i]) -
                                    static_cast<double>(want[i]));
      if (diff > DownReductionBound(kInter, t.span, t.chunks, t.abs_sum[i])) {
        ++violations;
      }
    }
    r.check(violations == 0,
            "with the clamp effectively disabled, R = 64 stays inside the "
            "derived bound against the monolithic oracle",
            std::to_string(violations) + " element(s) exceeded it");
  }

  // --- ZERO ACTIVATIONS: gate = up = 0, silu(0) * 0 = 0, so Y is exactly 0 --
  // and in the tiled form that also means every chunk's subtotal was zero,
  // i.e. the accumulator was initialized and not left holding stale data.
  {
    const std::vector<float> zero(static_cast<std::size_t>(kHidden), 0.0F);
    bool all_zero = true;
    for (const int chunk : kChunkSizes) {
      std::fill(y.begin(), y.end(), 1234.5F);  // poison, to catch a missing init
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          w1.View(0), w3.View(0), w2.View(0), zero.data(), y.data(), kLimit,
          chunk, &scratch);
      for (const float v : y) {
        if (v != 0.0F) all_zero = false;
      }
    }
    r.check(all_zero,
            "a zero activation gives an exactly zero output at every chunk "
            "size, over a poisoned destination (so Y is initialized, not "
            "accumulated into whatever was there)");
  }

  // --- ZERO W1: the gate is identically zero, so silu(0) = 0 and the whole
  // expert collapses whatever x and W3 are.
  {
    Bank zeros;
    zeros.rows = kInter;
    zeros.cols = kHidden;
    zeros.packed.assign(static_cast<std::size_t>(kInter) * (kHidden / 2), 0u);
    zeros.scales.assign(
        static_cast<std::size_t>(kInter) * (kHidden / kMxfp4GroupSize), 127u);
    const std::vector<float> x = RandomRow(kHidden, 1.0, rng);
    bool all_zero = true;
    for (const int chunk : {1, 48, 256}) {
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          zeros.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kLimit,
          chunk, &scratch);
      for (const float v : y) {
        if (v != 0.0F) all_zero = false;
      }
    }
    r.check(all_zero,
            "an all-zero W1 zeroes the activation and so the output, exactly, "
            "at every chunk size");
  }

  // --- A ONE-HOT ACTIVATION. x = e_j makes gate[r] = W1b[r][j] and up[r] =
  // W3b[r][j], so the expert reduces to a hand-computable expression over the
  // dequantized columns -- computed here through the public row helper and
  // explicit loops, which is an audit of the tiled path's W2 COLUMN offsets in
  // particular: with a one-hot input every chunk contributes a different,
  // identifiable set of columns.
  {
    constexpr int kHot = 37;
    static_assert(kHot < kHidden, "the one-hot column must be inside the row");
    std::vector<float> x(static_cast<std::size_t>(kHidden), 0.0F);
    x[kHot] = 1.0F;  // exactly representable, so no rounding enters

    std::vector<float> rowbuf;
    std::vector<float> act(static_cast<std::size_t>(kInter));
    for (int rr = 0; rr < kInter; ++rr) {
      const float gate =
          mxfp4_fixture::DequantElement(w1.View(0), rr, kHot, &rowbuf);
      const float up =
          mxfp4_fixture::DequantElement(w3.View(0), rr, kHot, &rowbuf);
      act[static_cast<std::size_t>(rr)] = bf16_round_cpu(
          SiluRef(std::min(gate, kLimit)) *
          std::min(kLimit, std::max(-kLimit, up)));
    }
    for (const int chunk : {32, 48, 100}) {
      const int span = EffectiveSpan(chunk, kInter);
      std::vector<float> want(static_cast<std::size_t>(kHidden), 0.0F);
      for (int r0 = 0; r0 < kInter; r0 += span) {
        const int rows = std::min(span, kInter - r0);
        for (int h = 0; h < kHidden; ++h) {
          float partial = 0.0F;
          for (int i = 0; i < rows; ++i) {
            partial += act[static_cast<std::size_t>(r0 + i)] *
                       mxfp4_fixture::DequantElement(w2.View(0), h, r0 + i,
                                                     &rowbuf);
          }
          want[static_cast<std::size_t>(h)] += partial;
        }
      }
      moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
          w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kLimit, chunk,
          &scratch);
      r.check(BitIdentical(y, want),
              "a one-hot activation at R = " + std::to_string(chunk) +
                  " reproduces an independent hand transcription of the tiled "
                  "expert, bit for bit",
              "max abs " +
                  std::to_string(MaxAbsDiff(y.data(), want.data(), y.size())));
      const auto spread = MeasureSpread(want.data(), want.size());
      r.check(spread.NonDegenerate(),
              "the one-hot golden at R = " + std::to_string(chunk) +
                  " itself carries magnitude and variation",
              "rms " + std::to_string(spread.rms) + " stddev " +
                  std::to_string(spread.stddev));
    }
  }
}

// ---------------------------------------------------------------------------
// 6. The argument contract
// ---------------------------------------------------------------------------
void AuditArgumentContract(Report& r, const Bank& w1, const Bank& w3,
                           const Bank& w2, const std::vector<float>& x) {
  section("6. the argument contract");

  std::vector<float> scratch;
  std::vector<float> y(static_cast<std::size_t>(kHidden));

  auto throws = [](auto&& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };

  // A NONPOSITIVE CHUNK is the one argument error that must not be tolerated:
  // the loop would either not advance or skip the reduction entirely and
  // return a zeroed output, which is an oracle that silently says "zero".
  for (const int bad : {0, -1, -64}) {
    r.check(throws([&] {
              moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
                  w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(),
                  kLimit, bad, &scratch);
            }),
            "R = " + std::to_string(bad) +
                " is rejected rather than silently producing a zeroed output");
  }

  // Null operands and a null scratch.
  r.check(throws([&] {
            moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
                w1.View(0), w3.View(0), w2.View(0), x.data(), y.data(), kLimit,
                32, nullptr);
          }),
          "a null scratch is rejected");
  r.check(throws([&] {
            Mxfp4Matrix broken = w1.View(0);
            broken.packed = nullptr;
            moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
                broken, w3.View(0), w2.View(0), x.data(), y.data(), kLimit, 32,
                &scratch);
          }),
          "a weight matrix with a null payload is rejected");

  // Mismatched geometry between the three weight matrices.
  r.check(throws([&] {
            Mxfp4Matrix broken = w2.View(0);
            broken.cols = kInter - kMxfp4GroupSize;
            moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
                w1.View(0), w3.View(0), broken, x.data(), y.data(), kLimit, 32,
                &scratch);
          }),
          "a W2 whose input extent is not the intermediate size is rejected");
  r.check(throws([&] {
            Mxfp4Matrix broken = w2.View(0);
            broken.rows = kHidden - 1;
            moe::emu::moe_expert_swiglu_mxfp4_w4a16_flash_cpu(
                w1.View(0), w3.View(0), broken, x.data(), y.data(), kLimit, 32,
                &scratch);
          }),
          "a W2 whose OUTPUT extent is not the hidden size is rejected -- Y is "
          "indexed over w2's rows inside the chunk loop, so a short w2 would "
          "be read past its end");

  // The batch entry: a ragged expert set and an out-of-range expert id. Both
  // would otherwise read the wrong bytes at a fixed device stride.
  std::vector<Mxfp4Matrix> v1, v3, v2;
  for (int e = 0; e < kExperts; ++e) {
    v1.push_back(w1.View(e));
    v3.push_back(w3.View(e));
    v2.push_back(w2.View(e));
  }
  std::vector<std::int32_t> route(4, 0);
  std::vector<float> xb(static_cast<std::size_t>(4) * kHidden, 0.25F);
  std::vector<float> yb(static_cast<std::size_t>(4) * kHidden);

  r.check(throws([&] {
            std::vector<Mxfp4Matrix> bad = v1;
            bad[1].rows = kInter - kMxfp4GroupSize;
            moe::emu::moe_prefill_flash_ffn_w4a16_cpu(
                bad.data(), v3.data(), v2.data(), xb.data(), route.data(), 4,
                kExperts, yb.data(), kLimit, 32, &scratch);
          }),
          "an expert whose intermediate size differs from the others is "
          "rejected, not silently read at the wrong stride");
  r.check(throws([&] {
            std::vector<std::int32_t> bad = route;
            bad[2] = kExperts;  // one past the end
            moe::emu::moe_prefill_flash_ffn_w4a16_cpu(
                v1.data(), v3.data(), v2.data(), xb.data(), bad.data(), 4,
                kExperts, yb.data(), kLimit, 32, &scratch);
          }),
          "a token routed to an out-of-range expert id is rejected");
  r.check(throws([&] {
            moe::emu::moe_prefill_flash_ffn_w4a16_cpu(
                v1.data(), v3.data(), v2.data(), xb.data(), route.data(), 4,
                kExperts, yb.data(), kLimit, 0, &scratch);
          }),
          "the batch entry rejects a zero chunk as well");

  // The documented footprint of a nonpositive extent is 0, not a wild number
  // a caller would then allocate.
  r.check(moe::emu::moe_prefill_flash_scratch_floats(kHidden, kInter, 0) == 0 &&
              moe::emu::moe_prefill_flash_scratch_floats(kHidden, 0, 32) == 0 &&
              moe::emu::moe_prefill_flash_scratch_floats(0, kInter, 32) == 0,
          "the footprint helper returns 0 for a nonpositive extent");
}

}  // namespace

int main() {
  std::printf("dsv4 MoE prefill: flash-style tiled W4A16 FFN, host verification\n");
  std::printf("seed 0x%llX  hidden %d  inter %d  experts %d  limit %.1f\n",
              static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
              static_cast<double>(kLimit));
  std::printf("chunk sizes R:");
  for (const int chunk : kChunkSizes) std::printf(" %d", chunk);
  std::printf("\n");

  std::mt19937_64 rng(kSeed);
  Report r;

  const Bank w1 = MakeBank(kInter, kHidden, kExperts, rng);
  const Bank w3 = MakeBank(kInter, kHidden, kExperts, rng);
  const Bank w2 = MakeBank(kHidden, kInter, kExperts, rng);
  const std::vector<float> x = RandomRow(kHidden, 1.0, rng);

  AuditChunkEquivalence(r, w1, w3, w2, x);
  AuditBoundHasTeeth(r, w1, w3, w2, x);
  AuditNoMaterialization(r, w1, w3, w2, x, rng);
  AuditRaggedBatch(r, w1, w3, w2, rng);
  AuditBoundaryRegimes(r, w1, w3, w2, rng);
  AuditArgumentContract(r, w1, w3, w2, x);

  std::printf("\n%d checks, %d failures, %d skipped\n", r.checks, r.failures,
              r.skipped);
  return r.exit_code();
}
