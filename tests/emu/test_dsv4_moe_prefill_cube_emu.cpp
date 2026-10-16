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

// ORACLE SELF-AUDIT for the W4A16 golden behind dsv4_moe_prefill_cube.
//
// Phase 1 of the two-phase oracle rule: a kernel diffed against an unaudited
// CPU model proves that two things agree, not that either is right. Nothing in
// this file touches a device, a simulator or a toolkit -- it is the CI gate
// (CLAUDE.md section 4.1), links moe_emu only, and must not skip (section
// 4.2).
//
// WHAT IS BEING AUDITED, AND WHY EACH CHECK EARNS ITS PLACE
//
//   1. MXFP4 -> BF16 is exact. Asserted over the WHOLE 16 x 255 code product,
//      not sampled: the claim is what licenses calling the weight path
//      lossless, and the code space is small enough to enumerate. The exact
//      boundaries of the domain are pinned too, so "lossless except at the
//      exponent extremes" names an interval instead of gesturing at one.
//
//   2. The group scale's PLACEMENT is free. The W4A32 oracle scales each
//      32-element group subtotal; the Cube can only distribute the scale into
//      every product. On a single-group matrix the two must agree BIT FOR BIT
//      -- which simultaneously pins the dequantization, the scale commutation
//      and the nibble order between the two oracles. Across groups they
//      differ only by FP32 reassociation, and that is bounded, not asserted
//      away.
//
//   3. The BF16 rounding happens exactly once, in the right place. Checked by
//      idempotence under pre-rounding, which fails if the oracle rounds twice,
//      rounds after the clamp, or forgets to round `activated`.
//
//   4. Structural inputs whose answer can be written down: zero activations,
//      zero weights, and a ONE-HOT activation, for which the whole expert
//      reduces to a hand-computable expression over the dequantized columns.
//      The one-hot case is an independent transcription of the expert -- it
//      goes through mxfp4_dequant_row_bf16_cpu and explicit loops, not through
//      the function under audit.
//
//   5. The clamp's ASYMMETRY is load-bearing. Shown by exhibiting an input on
//      which bounding the gate from below too changes the answer, rather than
//      by asserting that the code contains a min and not a max.
//
//   6. The W4A16 / W4A32 gap is inside a bound DERIVED FROM THE FORMAT. The
//      derivation is written out at the assertion; the observed figure is
//      printed but never used to pick the bound (section 4.7).
//
//   7. Ragged geometry: a token count that is not a multiple of the kernel's
//      16-token tile, and an expert grouping where a wrong-expert read would
//      change the answer.
//
//   8. Real non-degeneracy of the golden -- nonzero RMS and nonzero stddev,
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
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using moe::emu::bf16_round_cpu;
using moe::emu::DecodeFp4E2M1;
using moe::emu::DecodeFp8E8M0;
using moe::emu::kMxfp4GroupSize;
using moe::emu::kMxfp4LowNibbleFirst;
using moe::emu::Mxfp4DequantIsExactInBf16;
using moe::emu::Mxfp4Matrix;
using moe::kerntest::CosineSimilarity;
using moe::kerntest::MaxAbsDiff;
using moe::kerntest::MeasureSpread;
using moe::kerntest::RelativeL2;
using moe::kerntest::Report;
using moe::kerntest::section;
using mxfp4_fixture::Bank;
using mxfp4_fixture::DequantElement;
using mxfp4_fixture::MakeBank;
using mxfp4_fixture::SiluRef;

// The seed is printed, so a failure here is reproducible without a fixture on
// disk (CLAUDE.md section 4.5).
constexpr std::uint64_t kSeed = 0xC0BE'4A16'5EEDULL;

// Scaled-down geometry. The reduction axes must be whole 32-element MXFP4
// groups, which is all the oracle requires; the kernel's own admissible
// geometry (hidden/inter multiples of 512) is a Cube tiling constraint and is
// checked where it binds, in tests/sim.
constexpr int kHidden = 128;
constexpr int kInter = 64;
constexpr int kExperts = 3;
constexpr float kLimit = 10.0F;

// ---------------------------------------------------------------------------
// MXFP4 fixture
// ---------------------------------------------------------------------------
// Shared with test_dsv4_moe_prefill_flash_emu.cpp, which audits the
// flash-tiled shape of this same arithmetic: the two suites are only
// comparable if their weight banks come from the same generator and the same
// seed, and the nibble-order convention must be transcribed on the test side
// exactly once. See tests/emu/mxfp4_bank_fixture.hpp.

// ---------------------------------------------------------------------------
// 1. MXFP4 -> BF16 exactness, enumerated over the whole code product
// ---------------------------------------------------------------------------
void AuditDequantExactness(Report& r) {
  section("1. MXFP4 -> BF16 dequantization is exact");

  std::size_t exact = 0;
  std::size_t inexact = 0;
  int worst_nibble = -1;
  int worst_scale = -1;
  for (int scale_code = 0; scale_code <= 255; ++scale_code) {
    for (int nibble = 0; nibble < 16; ++nibble) {
      const bool in_domain = Mxfp4DequantIsExactInBf16(
          static_cast<std::uint8_t>(nibble), static_cast<std::uint8_t>(scale_code));
      if (!in_domain) {
        ++inexact;
        continue;
      }
      ++exact;
      // Inside the domain the round MUST be the identity, bit for bit. A
      // tolerance here would defeat the purpose: the claim is exactness.
      const float value = DecodeFp4E2M1(static_cast<std::uint8_t>(nibble)) *
                          DecodeFp8E8M0(static_cast<std::uint8_t>(scale_code));
      if (bf16_round_cpu(value) != value) {
        worst_nibble = nibble;
        worst_scale = scale_code;
      }
    }
  }
  r.check(worst_nibble < 0,
          "every in-domain (nibble, scale) product survives a BF16 round "
          "bit-exactly",
          worst_nibble < 0 ? std::string()
                           : "nibble " + std::to_string(worst_nibble) +
                                 " scale code " + std::to_string(worst_scale) +
                                 " is claimed exact but rounds");
  std::printf("           domain: %zu of %zu code pairs exact, %zu outside\n",
              exact, exact + inexact, inexact);

  // Pin the BOUNDARIES rather than trusting the predicate to be the whole
  // story. The derivation, from the formats:
  //   upper -- the largest magnitude is 6 = 1.5 * 2^2, so the product is
  //            1.5 * 2^(code - 125) and leaves FP32's range once
  //            code - 125 > 127, i.e. at code 253.
  //   lower -- the smallest nonzero magnitude is 0.5 = 2^-1, so the product is
  //            2^(code - 128) and drops below BF16's smallest NORMAL (2^-126)
  //            once code < 2.
  // Scale code 0xFF is the OCP NaN code and is outside by definition.
  const bool upper_ok =
      Mxfp4DequantIsExactInBf16(7u, 252u) &&       // 6 * 2^125, still finite
      !Mxfp4DequantIsExactInBf16(7u, 253u);        // 6 * 2^126 overflows
  r.check(upper_ok,
          "the exact domain's upper edge is scale code 252 for the magnitude-6 "
          "nibble",
          "codes 252/253 did not bracket the overflow");
  const bool lower_ok =
      Mxfp4DequantIsExactInBf16(1u, 2u) &&         // 0.5 * 2^-125 = 2^-126
      !Mxfp4DequantIsExactInBf16(1u, 1u);          // 0.5 * 2^-126 = 2^-127
  r.check(lower_ok,
          "the exact domain's lower edge is scale code 2 for the magnitude-0.5 "
          "nibble (below it the device flushes the subnormal)",
          "codes 1/2 did not bracket the subnormal edge");
  r.check(!Mxfp4DequantIsExactInBf16(1u, 255u),
          "the OCP NaN scale code 0xFF is outside the domain of agreement");
  // A zero nibble is exact at EVERY finite scale, including the extremes --
  // worth pinning because it is what makes a sparse weight row safe.
  bool zero_ok = true;
  for (int code = 0; code <= 254; ++code) {
    if (!Mxfp4DequantIsExactInBf16(0u, static_cast<std::uint8_t>(code))) {
      zero_ok = false;
    }
  }
  r.check(zero_ok, "a zero nibble is exact at every non-NaN scale code");

  // And the band a real checkpoint uses must be wholly inside.
  bool checkpoint_band = true;
  for (int code = 100; code <= 150; ++code) {
    for (int nibble = 0; nibble < 16; ++nibble) {
      if (!Mxfp4DequantIsExactInBf16(static_cast<std::uint8_t>(nibble),
                                     static_cast<std::uint8_t>(code))) {
        checkpoint_band = false;
      }
    }
  }
  r.check(checkpoint_band,
          "scale codes 100..150 -- the band real MXFP4 checkpoints occupy -- "
          "are exact for all 16 nibbles");
}

// ---------------------------------------------------------------------------
// 2. The group scale's placement, and what reassociation costs
// ---------------------------------------------------------------------------
void AuditScalePlacement(Report& r, std::mt19937_64& rng) {
  section("2. the E8M0 group scale commutes with the reduction");

  // ONE group: the W4A32 oracle's per-subtotal scaling and the W4A16 oracle's
  // per-product scaling must then agree BIT FOR BIT. Scaling by a power of
  // two does not touch the mantissa, so each running partial in the
  // distributed form is exactly 2^k times its counterpart -- including after
  // every intermediate rounding.
  const Bank single = MakeBank(8, kMxfp4GroupSize, 1, rng);
  std::vector<float> x(kMxfp4GroupSize);
  std::normal_distribution<float> draw(0.0F, 1.0F);
  for (float& v : x) v = bf16_round_cpu(draw(rng));  // BF16-representable

  std::vector<float> y_w4a32(8);
  std::vector<float> y_w4a16(8);
  moe::emu::mxfp4_gemv_cpu(single.View(0), x.data(), y_w4a32.data());
  moe::emu::mxfp4_gemv_w4a16_cpu(single.View(0), x.data(), y_w4a16.data());

  bool bit_exact = true;
  for (int i = 0; i < 8; ++i) {
    if (std::memcmp(&y_w4a32[static_cast<std::size_t>(i)],
                    &y_w4a16[static_cast<std::size_t>(i)], sizeof(float)) != 0) {
      bit_exact = false;
    }
  }
  r.check(bit_exact,
          "on a single-group matrix the W4A32 and W4A16 GEMVs agree bit for "
          "bit (so the dequantization, the scale commutation and the nibble "
          "order all match)",
          "a row differed; max abs " +
              std::to_string(MaxAbsDiff(y_w4a16.data(), y_w4a32.data(), 8)));

  // MANY groups: now the two differ, and only by the order of the additions.
  // The W4A32 form does one add per group; the W4A16 form does 32. Bound:
  // cols * 2^-24 relative, the worst case for cols sequential FP32 adds.
  const Bank wide = MakeBank(8, kHidden, 1, rng);
  std::vector<float> xw(kHidden);
  for (float& v : xw) v = bf16_round_cpu(draw(rng));
  std::vector<float> wide32(8);
  std::vector<float> wide16(8);
  moe::emu::mxfp4_gemv_cpu(wide.View(0), xw.data(), wide32.data());
  moe::emu::mxfp4_gemv_w4a16_cpu(wide.View(0), xw.data(), wide16.data());

  const double reassociation_bound =
      static_cast<double>(kHidden) * std::pow(2.0, -24);
  const double observed = RelativeL2(wide16.data(), wide32.data(), 8);
  r.check(observed <= reassociation_bound,
          "across groups the two forms differ only by FP32 reassociation "
          "(<= cols * 2^-24)",
          "relative-L2 " + std::to_string(observed) + " exceeds " +
              std::to_string(reassociation_bound));
  std::printf("           reassociation: relative-L2 %.3e against the "
              "cols * 2^-24 bound %.3e\n",
              observed, reassociation_bound);
}

// ---------------------------------------------------------------------------
// 3. The BF16 rounding happens once, and inside
// ---------------------------------------------------------------------------
void AuditRoundingIdempotence(Report& r, const Bank& w1, const Bank& w3,
                              const Bank& w2, std::mt19937_64& rng) {
  section("3. the activation rounding is applied exactly once, inside");

  std::vector<float> x(kHidden);
  std::normal_distribution<float> draw(0.0F, 1.0F);
  for (float& v : x) v = draw(rng);  // deliberately NOT pre-rounded

  std::vector<float> pre(kHidden);
  for (int i = 0; i < kHidden; ++i) pre[static_cast<std::size_t>(i)] =
      bf16_round_cpu(x[static_cast<std::size_t>(i)]);

  std::vector<float> y_raw(kHidden);
  std::vector<float> y_pre(kHidden);
  std::vector<float> scratch;
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0), w2.View(0),
                                              x.data(), y_raw.data(), kLimit,
                                              &scratch);
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0), w2.View(0),
                                              pre.data(), y_pre.data(), kLimit,
                                              &scratch);
  // Bit-exact: the oracle rounds x itself, so handing it an already-rounded x
  // must change nothing. This fails if the rounding is missing (then raw x
  // carries more precision than the kernel has) or doubled.
  bool identical = std::memcmp(y_raw.data(), y_pre.data(),
                               y_raw.size() * sizeof(float)) == 0;
  r.check(identical,
          "pre-rounding the activations to BF16 changes nothing, bit for bit",
          "max abs " + std::to_string(MaxAbsDiff(y_raw.data(), y_pre.data(),
                                                 y_raw.size())));

  // And the rounding is not a no-op in disguise: an unrounded x must differ
  // from the W4A32 answer, or the whole W4A16 distinction is vacuous.
  std::vector<float> y_w4a32(kHidden);
  moe::emu::moe_expert_swiglu_mxfp4_cpu(w1.View(0), w3.View(0), w2.View(0),
                                        x.data(), y_w4a32.data(), kLimit,
                                        &scratch);
  const double gap = MaxAbsDiff(y_raw.data(), y_w4a32.data(), y_raw.size());
  r.check(gap > 0.0,
          "the W4A16 and W4A32 oracles are genuinely different functions",
          "they agreed exactly, so one of them is not doing what it claims");
}

// ---------------------------------------------------------------------------
// 4. Structural inputs whose answer can be written down
// ---------------------------------------------------------------------------
void AuditStructuralInputs(Report& r, const Bank& w1, const Bank& w3,
                           const Bank& w2) {
  section("4. structural inputs");

  std::vector<float> scratch;
  std::vector<float> y(kHidden);

  // --- zero activations: gate = up = 0, silu(0) * 0 = 0, so y is exactly 0.
  {
    const std::vector<float> zero(kHidden, 0.0F);
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0),
                                                w2.View(0), zero.data(),
                                                y.data(), kLimit, &scratch);
    bool all_zero = true;
    for (const float v : y) {
      if (v != 0.0F) all_zero = false;
    }
    r.check(all_zero, "a zero activation gives an exactly zero output");
  }

  // --- zero weights: every nibble 0, so y is exactly 0 whatever x is.
  {
    Bank zeros;
    zeros.rows = kInter;
    zeros.cols = kHidden;
    zeros.packed.assign(
        static_cast<std::size_t>(kInter) * (kHidden / 2), 0u);
    zeros.scales.assign(
        static_cast<std::size_t>(kInter) * (kHidden / kMxfp4GroupSize), 127u);
    std::vector<float> x(kHidden);
    for (int i = 0; i < kHidden; ++i) {
      x[static_cast<std::size_t>(i)] = 0.5F * static_cast<float>(i % 7) - 1.0F;
    }
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(zeros.View(0), w3.View(0),
                                                w2.View(0), x.data(), y.data(),
                                                kLimit, &scratch);
    // gate is identically 0 -> silu(0) = 0 -> activated is identically 0 ->
    // the down GEMM sums zeros.
    bool all_zero = true;
    for (const float v : y) {
      if (v != 0.0F) all_zero = false;
    }
    r.check(all_zero,
            "an all-zero W1 zeroes the activation and so the output, exactly");
  }

  // --- ONE-HOT activation: x = e_j. Then gate[rr] = W1b[rr][j] and
  // up[rr] = W3b[rr][j], so the whole expert is
  //   y[h] = sum_rr bf16(silu(min(W1b[rr][j], L)) * clamp(W3b[rr][j])) * W2b[h][rr]
  // which this block computes INDEPENDENTLY, from the dequant row helper and
  // explicit loops. That is what makes it an audit of the oracle rather than a
  // restatement of it.
  {
    constexpr int kHot = 37;
    static_assert(kHot < kHidden, "the one-hot column must be inside the row");
    std::vector<float> x(kHidden, 0.0F);
    x[kHot] = 1.0F;  // exactly representable, so no rounding enters

    std::vector<float> rowbuf;
    std::vector<float> activated(kInter);
    for (int rr = 0; rr < kInter; ++rr) {
      const float gate = DequantElement(w1.View(0), rr, kHot, &rowbuf);
      const float up = DequantElement(w3.View(0), rr, kHot, &rowbuf);
      const float clamped_gate = std::min(gate, kLimit);
      const float clamped_up = std::min(kLimit, std::max(-kLimit, up));
      activated[static_cast<std::size_t>(rr)] =
          bf16_round_cpu(SiluRef(clamped_gate) * clamped_up);
    }
    std::vector<float> want(kHidden);
    for (int h = 0; h < kHidden; ++h) {
      float total = 0.0F;
      for (int rr = 0; rr < kInter; ++rr) {
        total += activated[static_cast<std::size_t>(rr)] *
                 DequantElement(w2.View(0), h, rr, &rowbuf);
      }
      want[static_cast<std::size_t>(h)] = total;
    }

    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0),
                                                w2.View(0), x.data(), y.data(),
                                                kLimit, &scratch);
    // Bit-exact is the right bar: the independent transcription performs the
    // same operations in the same order, so any difference is a real
    // disagreement and not a rounding artefact.
    const bool identical =
        std::memcmp(y.data(), want.data(), y.size() * sizeof(float)) == 0;
    r.check(identical,
            "a one-hot activation reproduces an independent hand transcription "
            "of the expert, bit for bit",
            "max abs " +
                std::to_string(MaxAbsDiff(y.data(), want.data(), y.size())));

    const auto spread = MeasureSpread(want.data(), want.size());
    r.check(spread.NonDegenerate(),
            "the one-hot golden itself carries magnitude and variation",
            "rms " + std::to_string(spread.rms) + " stddev " +
                std::to_string(spread.stddev));
  }
}

// ---------------------------------------------------------------------------
// 5. The clamp's asymmetry is load-bearing
// ---------------------------------------------------------------------------
void AuditClampAsymmetry(Report& r, const Bank& w1, const Bank& w3,
                         const Bank& w2, std::mt19937_64& rng) {
  section("5. the SwiGLU clamp is asymmetric, and that matters");

  // Scale the activations up until the gate GEMM clears the limit on both
  // sides. At that amplitude a symmetric clamp -- gate bounded BELOW at -L
  // too -- is a different function, and this exhibits an input that separates
  // them instead of asserting it from the source.
  std::vector<float> x(kHidden);
  std::normal_distribution<float> draw(0.0F, 1.0F);
  for (float& v : x) v = bf16_round_cpu(draw(rng) * 6.0F);

  std::vector<float> xb(kHidden);
  for (int i = 0; i < kHidden; ++i) xb[static_cast<std::size_t>(i)] =
      bf16_round_cpu(x[static_cast<std::size_t>(i)]);

  std::vector<float> gate(kInter);
  std::vector<float> up(kInter);
  moe::emu::mxfp4_gemv_w4a16_cpu(w1.View(0), xb.data(), gate.data());
  moe::emu::mxfp4_gemv_w4a16_cpu(w3.View(0), xb.data(), up.data());

  int above = 0;
  int below = 0;
  for (int i = 0; i < kInter; ++i) {
    if (gate[static_cast<std::size_t>(i)] > kLimit) ++above;
    if (gate[static_cast<std::size_t>(i)] < -kLimit) ++below;
  }
  r.check(above > 0 && below > 0,
          "the probe input drives the gate past the limit on BOTH sides, so "
          "the asymmetry is observable",
          std::to_string(above) + " above, " + std::to_string(below) +
              " below -- both must be nonzero for the next check to mean "
              "anything");

  // The asymmetric (correct) activation, and the symmetric one.
  std::vector<float> act_asym(kInter);
  std::vector<float> act_sym(kInter);
  for (int i = 0; i < kInter; ++i) {
    const float g = gate[static_cast<std::size_t>(i)];
    const float u = up[static_cast<std::size_t>(i)];
    const float cu = std::min(kLimit, std::max(-kLimit, u));
    act_asym[static_cast<std::size_t>(i)] =
        bf16_round_cpu(SiluRef(std::min(g, kLimit)) * cu);
    act_sym[static_cast<std::size_t>(i)] =
        bf16_round_cpu(SiluRef(std::min(kLimit, std::max(-kLimit, g))) * cu);
  }

  // THE CHECK IS STRUCTURAL, NOT A MAGNITUDE THRESHOLD, and the reason is
  // worth recording because the obvious version of this test is wrong.
  //
  // At L = 10 the two clamps are numerically very close: silu(-10) =
  // -10/(1+e^10) = -4.54e-4, while silu(z) -> 0 from below as z decreases, so
  // the largest activation difference a below-clamp can produce is
  // |silu(-L)| * L = 4.5e-3 on an element, and after the down GEMM that lands
  // around 4e-5 relative. A "differs by more than 1e-4" assertion therefore
  // FAILS on a correct oracle -- the asymmetry is structurally real and
  // numerically small, and conflating the two is how a test ends up fitted to
  // a threshold nobody can derive.
  //
  // So: assert WHERE the two differ, not by how much. They must differ on
  // exactly the indices whose gate fell below -L, and agree bit for bit
  // everywhere else. That statement is exact, follows from the definition of
  // the clamp, and is independent of L.
  int differ_in_set = 0;
  int differ_out_of_set = 0;
  for (int i = 0; i < kInter; ++i) {
    const bool below_limit = gate[static_cast<std::size_t>(i)] < -kLimit;
    const bool differs = std::memcmp(&act_asym[static_cast<std::size_t>(i)],
                                     &act_sym[static_cast<std::size_t>(i)],
                                     sizeof(float)) != 0;
    if (differs && below_limit) ++differ_in_set;
    if (differs && !below_limit) ++differ_out_of_set;
  }
  r.check(differ_out_of_set == 0,
          "the two clamps agree bit for bit on every element whose gate is "
          "inside [-L, inf)",
          std::to_string(differ_out_of_set) +
              " element(s) differed outside the below-limit set, so the "
              "difference is not the clamp asymmetry");
  // Not every below-limit element need differ: where `up` clamps to 0, or
  // where both silu values round to the same BF16, the difference vanishes.
  // What must hold is that the set is where the differences live, and that at
  // least one of them survives -- otherwise the probe proved nothing.
  r.check(differ_in_set > 0,
          "at least one below-limit element's activation actually changes, so "
          "the asymmetry is exercised and not merely available",
          std::to_string(below) + " elements were below -L but none of their "
          "activations differed");
  std::printf("           asymmetry: %d of %d below-limit elements change; "
              "the largest change a below-clamp can make is "
              "|silu(-L)| * L = %.3e\n",
              differ_in_set, below,
              static_cast<double>(std::fabs(SiluRef(-kLimit)) * kLimit));

  std::vector<float> y_asym(kHidden);
  std::vector<float> y_sym(kHidden);
  moe::emu::mxfp4_gemv_w4a16_cpu(w2.View(0), act_asym.data(), y_asym.data());
  moe::emu::mxfp4_gemv_w4a16_cpu(w2.View(0), act_sym.data(), y_sym.data());
  // Reported, never gated: the downstream figure depends on L, on the weight
  // draw and on how many elements clamped, none of which is derivable.
  std::printf("           downstream, for information only: symmetric vs "
              "asymmetric relative-L2 %.3e\n",
              RelativeL2(y_sym.data(), y_asym.data(), kHidden));

  // And the oracle under audit must agree with the ASYMMETRIC one.
  std::vector<float> scratch;
  std::vector<float> y(kHidden);
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(0), w3.View(0), w2.View(0),
                                              x.data(), y.data(), kLimit,
                                              &scratch);
  const bool matches_asym =
      std::memcmp(y.data(), y_asym.data(), y.size() * sizeof(float)) == 0;
  r.check(matches_asym,
          "the W4A16 oracle implements the asymmetric clamp (gate bounded "
          "above only)",
          "max abs against the asymmetric transcription " +
              std::to_string(MaxAbsDiff(y.data(), y_asym.data(), y.size())));
}

// ---------------------------------------------------------------------------
// 6. The W4A16 / W4A32 gap, against a bound derived from the format
// ---------------------------------------------------------------------------
void AuditFormatErrorBound(Report& r, const Bank& w1, const Bank& w3,
                           const Bank& w2, std::mt19937_64& rng) {
  section("6. the activation-rounding gap is inside the format's bound");

  // THE DERIVATION. BF16 keeps an 8-bit significand, so one RNE store carries
  // at most 2^-9 relative error. The path performs exactly two such stores,
  // and nothing else in it is lossy (check 1 settled the weights, check 2 the
  // scale placement):
  //
  //   X -> BF16                           <= 2^-9 relative
  //   both gate/up GEMMs are LINEAR in X, so they carry that relative error
  //   through unchanged -- a worst case, since the per-term errors have
  //   independent signs and partly cancel.
  //   the clamped SwiGLU composes silu and a product; both are Lipschitz with
  //   a relative gain of at most 1 here, and the product of two perturbed
  //   factors carries the SUM of their relative errors
  //                                       <= 2 * 2^-9
  //   activated -> BF16                   +  2^-9
  //   the down GEMM is linear, so it carries that through
  //                                       <= 4 * 2^-9 = 2^-7
  //
  // 2^-7 is therefore the bound, and it comes from the format and the shape of
  // the composition -- NOT from the number printed below (CLAUDE.md 4.7).
  constexpr double kDerivedRelL2Bound = 1.0 / 128.0;  // 2^-7
  // Cosine is kept strict on purpose: a relative perturbation of size d moves
  // the cosine by about d^2 / 2, so 2^-7 of magnitude error is ~3e-5 of
  // direction error. Rounding moves magnitude; a structural bug moves
  // direction.
  constexpr double kCosineFloor = 0.9999;

  std::vector<float> x(kHidden);
  std::normal_distribution<float> draw(0.0F, 1.0F);
  for (float& v : x) v = draw(rng);

  std::vector<float> scratch;
  std::vector<float> y16(kHidden);
  std::vector<float> y32(kHidden);
  double worst_rel = 0.0;
  double worst_cos = 1.0;
  for (int e = 0; e < kExperts; ++e) {
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(w1.View(e), w3.View(e),
                                                w2.View(e), x.data(),
                                                y16.data(), kLimit, &scratch);
    moe::emu::moe_expert_swiglu_mxfp4_cpu(w1.View(e), w3.View(e), w2.View(e),
                                          x.data(), y32.data(), kLimit,
                                          &scratch);
    worst_rel = std::max(worst_rel, RelativeL2(y16.data(), y32.data(), kHidden));
    worst_cos = std::min(worst_cos,
                         CosineSimilarity(y16.data(), y32.data(), kHidden));
  }

  r.check(worst_rel <= kDerivedRelL2Bound,
          "W4A16 against W4A32: relative-L2 within the derived 4 * 2^-9 bound",
          "relative-L2 " + std::to_string(worst_rel) + " exceeds 2^-7");
  r.check(worst_cos >= kCosineFloor,
          "W4A16 against W4A32: cosine stays above 0.9999 (direction is "
          "preserved)",
          "cosine " + std::to_string(worst_cos));
  std::printf("           observed: relative-L2 %.3e (bound %.3e), cosine "
              "%.8f\n",
              worst_rel, kDerivedRelL2Bound, worst_cos);
}

// ---------------------------------------------------------------------------
// 7. Ragged token geometry and the expert grouping
// ---------------------------------------------------------------------------
void AuditRaggedBatch(Report& r, const Bank& w1, const Bank& w3, const Bank& w2,
                      std::mt19937_64& rng) {
  section("7. ragged batch geometry and the per-token expert map");

  // 19 tokens against the kernel's 16-token tile: one full tile and a partial
  // one of 3. The tail is where every off-by-one lives.
  constexpr int kTokens = 19;
  constexpr int kMTile = 16;  // kPrefillCubeMTile
  static_assert(kTokens % kMTile != 0, "the point is a ragged final tile");

  std::vector<float> x(static_cast<std::size_t>(kTokens) * kHidden);
  std::normal_distribution<float> draw(0.0F, 1.0F);
  for (float& v : x) v = draw(rng);

  // Grouped by expert, which is the dispatch order the kernel assumes: it
  // reads token_to_expert at the FIRST row of each tile and applies that
  // expert to the whole tile.
  std::vector<std::int32_t> route(kTokens);
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

  std::vector<float> y(static_cast<std::size_t>(kTokens) * kHidden);
  std::vector<float> scratch;
  moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
      v1.data(), v3.data(), v2.data(), x.data(), route.data(), kTokens,
      kExperts, y.data(), kLimit, &scratch);

  // ROW INDEPENDENCE, which is what licenses the collapsed CAModel grid in
  // tests/sim: a token's output must be the single-token expert applied to
  // that token's row, with no dependence on its neighbours or on which tile
  // it landed in. Bit-exact -- the batch oracle is a loop over the same
  // function, so anything else is a real defect.
  std::vector<float> one(kHidden);
  int mismatched = 0;
  int first_bad = -1;
  for (int t = 0; t < kTokens; ++t) {
    const int e = route[static_cast<std::size_t>(t)];
    const std::size_t row = static_cast<std::size_t>(t) * kHidden;
    moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(v1[static_cast<std::size_t>(e)],
                                                v3[static_cast<std::size_t>(e)],
                                                v2[static_cast<std::size_t>(e)],
                                                x.data() + row, one.data(),
                                                kLimit, &scratch);
    if (std::memcmp(y.data() + row, one.data(), one.size() * sizeof(float)) !=
        0) {
      ++mismatched;
      if (first_bad < 0) first_bad = t;
    }
  }
  r.check(mismatched == 0,
          "all 19 tokens -- including the 3 in the ragged final 16-token tile "
          "-- equal the single-token expert on their own row, bit for bit",
          std::to_string(mismatched) + " token(s) differed, first at index " +
              std::to_string(first_bad));

  // A WRONG-EXPERT READ MUST BE VISIBLE. The fixture varies the scale pattern
  // with the expert id for exactly this reason; this check proves the
  // variation is enough by running one token's row through a neighbouring
  // expert and requiring a large difference. Without it, a tile that picked
  // the wrong expert could pass the comparison above on a lucky fixture.
  moe::emu::moe_expert_swiglu_mxfp4_w4a16_cpu(v1[1], v3[1], v2[1], x.data(),
                                              one.data(), kLimit, &scratch);
  const double wrong_expert_gap = RelativeL2(one.data(), y.data(), kHidden);
  r.check(wrong_expert_gap > 0.1,
          "running token 0 through expert 1 instead of expert 0 changes the "
          "answer by more than 10%, so a wrong-expert read cannot hide",
          "relative-L2 " + std::to_string(wrong_expert_gap));

  // The ragged tail must not be left unwritten, and the whole batch must be
  // non-degenerate -- nonzero RMS and nonzero stddev, measured on the GOLDEN
  // (section 4.8).
  const auto tail = MeasureSpread(
      y.data() + static_cast<std::size_t>(kMTile) * kHidden,
      static_cast<std::size_t>(kTokens - kMTile) * kHidden);
  r.check(tail.NonDegenerate(),
          "the ragged tail rows carry magnitude and variation (they were "
          "computed, not skipped)",
          "rms " + std::to_string(tail.rms) + " stddev " +
              std::to_string(tail.stddev));
  const auto whole = MeasureSpread(y.data(), y.size());
  r.check(whole.NonDegenerate(),
          "the whole batch golden is non-degenerate",
          "rms " + std::to_string(whole.rms) + " stddev " +
              std::to_string(whole.stddev));
  std::printf("           golden: rms %.4f stddev %.4f min %.4f max %.4f\n",
              whole.rms, whole.stddev, whole.min, whole.max);

  // Finally, the per-expert geometry guard must reject a ragged expert set
  // rather than read the wrong bytes: the device addresses expert e at a
  // FIXED stride, so this is not a cosmetic argument check.
  std::vector<Mxfp4Matrix> bad = v1;
  bad[1].rows = kInter - kMxfp4GroupSize;
  bool threw = false;
  try {
    moe::emu::moe_prefill_batched_mxfp4_w4a16_cpu(
        bad.data(), v3.data(), v2.data(), x.data(), route.data(), kTokens,
        kExperts, y.data(), kLimit, &scratch);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  r.check(threw,
          "an expert whose intermediate size differs from the others is "
          "rejected, not silently read at the wrong stride");
}

}  // namespace

int main() {
  std::printf("dsv4_moe_prefill_cube W4A16 oracle self-audit\n");
  std::printf("seed 0x%llX  hidden %d  inter %d  experts %d  limit %.1f\n",
              static_cast<unsigned long long>(kSeed), kHidden, kInter, kExperts,
              static_cast<double>(kLimit));

  std::mt19937_64 rng(kSeed);
  Report r;

  const Bank w1 = MakeBank(kInter, kHidden, kExperts, rng);
  const Bank w3 = MakeBank(kInter, kHidden, kExperts, rng);
  const Bank w2 = MakeBank(kHidden, kInter, kExperts, rng);

  AuditDequantExactness(r);
  AuditScalePlacement(r, rng);
  AuditRoundingIdempotence(r, w1, w3, w2, rng);
  AuditStructuralInputs(r, w1, w3, w2);
  AuditClampAsymmetry(r, w1, w3, w2, rng);
  AuditFormatErrorBound(r, w1, w3, w2, rng);
  AuditRaggedBatch(r, w1, w3, w2, rng);

  std::printf("\n%d checks, %d failures, %d skipped\n", r.checks, r.failures,
              r.skipped);
  return r.exit_code();
}
