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

// Paged KV compressor host emulation: HOLD/EMIT cadence, the gated 4:1
// reduction, and the RMSNorm + RoPE epilog.
//
// The numerical reference is recomputed here independently of the kernel --
// straight from compressor_golden.py's formulas, in the most literal form
// (explicit loops, no shared helpers) -- so that a bug in a helper cannot
// cancel itself out between the kernel and the expectation.

#include "moe/emu/cpu_kernels.h"

#include "emu_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace moe::emu;

constexpr int kH = 256;          // a small hidden size keeps the test fast
constexpr int kDHead = 64;       // head_dim D
constexpr int kRatio = 4;
constexpr int kStateWidth = kCompressorCoff * kDHead;  // 128

// --- independent reference implementation ----------------------------------

// softmax down `rows`, per lane, with the golden's denormal flush.
std::vector<float> ReferenceColumnSoftmax(const std::vector<float>& logits, int rows,
                                          int cols) {
  std::vector<float> out(logits.size());
  const float tiny = std::numeric_limits<float>::min();
  for (int c = 0; c < cols; ++c) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int r = 0; r < rows; ++r) {
      maximum = std::max(maximum, logits[static_cast<std::size_t>(r) * cols + c]);
    }
    if (!std::isfinite(maximum)) {
      for (int r = 0; r < rows; ++r) {
        out[static_cast<std::size_t>(r) * cols + c] = 0.0F;
      }
      continue;
    }
    float sum = 0.0F;
    for (int r = 0; r < rows; ++r) {
      float e = std::exp(logits[static_cast<std::size_t>(r) * cols + c] - maximum);
      if (e < tiny) {
        e = 0.0F;
      }
      out[static_cast<std::size_t>(r) * cols + c] = e;
      sum += e;
    }
    for (int r = 0; r < rows; ++r) {
      out[static_cast<std::size_t>(r) * cols + c] /= sum;
    }
  }
  return out;
}

struct Weights {
  std::vector<std::uint16_t> wkv;    // [coff*D, H]
  std::vector<std::uint16_t> wgate;  // [coff*D, H]
  std::vector<float> ape;            // [ratio, coff*D]
  std::vector<std::uint16_t> norm;   // [D]
  std::vector<float> cos;            // [kRopeHeadDim]
  std::vector<float> sin;            // [kRopeHeadDim]
};

Weights MakeWeights() {
  Weights w;
  w.wkv.resize(static_cast<std::size_t>(kStateWidth) * kH);
  w.wgate.resize(static_cast<std::size_t>(kStateWidth) * kH);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kH));
  for (auto& v : w.wkv) {
    v = FloatToBf16(emu_test::UniformFloat(-scale, scale));
  }
  for (auto& v : w.wgate) {
    v = FloatToBf16(emu_test::UniformFloat(-scale, scale));
  }
  w.ape.resize(static_cast<std::size_t>(kRatio) * kStateWidth);
  for (auto& v : w.ape) {
    v = emu_test::UniformFloat(-0.5F, 0.5F);
  }
  w.norm.resize(kDHead);
  for (auto& v : w.norm) {
    v = FloatToBf16(emu_test::UniformFloat(0.5F, 1.5F));
  }
  // A real RoPE table at position 0 would be cos=1, sin=0; use a non-trivial
  // angle so the rotation is actually observable.
  w.cos.resize(kRopeHeadDim);
  w.sin.resize(kRopeHeadDim);
  for (int i = 0; i < kRopeHeadDim / 2; ++i) {
    const float angle = 0.3F + 0.05F * static_cast<float>(i);
    w.cos[i] = std::cos(angle);
    w.sin[i] = std::sin(angle);
    w.cos[kRopeHeadDim / 2 + i] = std::cos(angle);  // duplicated halves
    w.sin[kRopeHeadDim / 2 + i] = std::sin(angle);
  }
  return w;
}

std::vector<std::uint16_t> MakeToken() {
  std::vector<std::uint16_t> x(kH);
  for (auto& v : x) {
    v = FloatToBf16(emu_test::UniformFloat(-1.0F, 1.0F));
  }
  return x;
}

// The whole EMIT path, recomputed from the golden formulas for one group.
// `tokens` are the ratio tokens of the current group, `previous` the ratio
// tokens of the group before it (empty for the first group).
std::vector<float> ReferenceEmit(const Weights& w,
                                 const std::vector<std::vector<std::uint16_t>>& tokens,
                                 const std::vector<std::vector<std::uint16_t>>& previous) {
  const int rows = kCompressorCoff * kRatio;  // 8

  // stage 1: project every token this reduction touches
  auto project = [&](const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& m) {
    std::vector<float> out(kStateWidth);
    for (int r = 0; r < kStateWidth; ++r) {
      float dot = 0.0F;
      for (int c = 0; c < kH; ++c) {
        dot += Bf16ToFloat(m[static_cast<std::size_t>(r) * kH + c]) *
               Bf16ToFloat(x[static_cast<std::size_t>(c)]);
      }
      out[static_cast<std::size_t>(r)] = dot;
    }
    return out;
  };

  std::vector<float> window_kv(static_cast<std::size_t>(rows) * kDHead, 0.0F);
  std::vector<float> window_score(static_cast<std::size_t>(rows) * kDHead,
                                  -std::numeric_limits<float>::infinity());

  for (int t = 0; t < kRatio; ++t) {
    // rows 0..3: the PREVIOUS group's a-halves (lanes [0, D) of its row)
    if (!previous.empty()) {
      std::vector<float> kv = project(previous[static_cast<std::size_t>(t)], w.wkv);
      std::vector<float> sc = project(previous[static_cast<std::size_t>(t)], w.wgate);
      for (int i = 0; i < kStateWidth; ++i) {
        sc[static_cast<std::size_t>(i)] +=
            w.ape[static_cast<std::size_t>(t) * kStateWidth + i];
      }
      for (int i = 0; i < kDHead; ++i) {
        window_kv[static_cast<std::size_t>(t) * kDHead + i] = kv[static_cast<std::size_t>(i)];
        window_score[static_cast<std::size_t>(t) * kDHead + i] = sc[static_cast<std::size_t>(i)];
      }
    }
    // rows 4..7: THIS group's b-halves (lanes [D, 2D) of its row)
    std::vector<float> kv = project(tokens[static_cast<std::size_t>(t)], w.wkv);
    std::vector<float> sc = project(tokens[static_cast<std::size_t>(t)], w.wgate);
    for (int i = 0; i < kStateWidth; ++i) {
      sc[static_cast<std::size_t>(i)] +=
          w.ape[static_cast<std::size_t>(t) * kStateWidth + i];
    }
    for (int i = 0; i < kDHead; ++i) {
      window_kv[static_cast<std::size_t>(kRatio + t) * kDHead + i] =
          kv[static_cast<std::size_t>(kDHead + i)];
      window_score[static_cast<std::size_t>(kRatio + t) * kDHead + i] =
          sc[static_cast<std::size_t>(kDHead + i)];
    }
  }

  // stages 3-5
  const std::vector<float> weights = ReferenceColumnSoftmax(window_score, rows, kDHead);
  std::vector<float> compressed(kDHead, 0.0F);
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < kDHead; ++i) {
      compressed[static_cast<std::size_t>(i)] +=
          weights[static_cast<std::size_t>(r) * kDHead + i] *
          window_kv[static_cast<std::size_t>(r) * kDHead + i];
    }
  }

  // stage 6: RMSNorm
  double sum_squares = 0.0;
  for (int i = 0; i < kDHead; ++i) {
    sum_squares += static_cast<double>(compressed[static_cast<std::size_t>(i)]) *
                   static_cast<double>(compressed[static_cast<std::size_t>(i)]);
  }
  const float inv_rms =
      1.0F / std::sqrt(static_cast<float>(sum_squares / kDHead) + kRmsNormEpsilon);
  std::vector<float> out(kDHead);
  for (int i = 0; i < kDHead; ++i) {
    out[static_cast<std::size_t>(i)] =
        Bf16ToFloat(w.norm[static_cast<std::size_t>(i)]) *
        compressed[static_cast<std::size_t>(i)] * inv_rms;
  }

  // stage 7: half-mode RoPE over the last kRopeHeadDim lanes
  const int base = kDHead - kRopeHeadDim;
  const int half = kRopeHeadDim / 2;
  for (int i = 0; i < half; ++i) {
    const float lo = out[static_cast<std::size_t>(base + i)];
    const float hi = out[static_cast<std::size_t>(base + half + i)];
    out[static_cast<std::size_t>(base + i)] =
        w.cos[static_cast<std::size_t>(i)] * lo - w.sin[static_cast<std::size_t>(i)] * hi;
    out[static_cast<std::size_t>(base + half + i)] =
        w.sin[static_cast<std::size_t>(half + i)] * lo +
        w.cos[static_cast<std::size_t>(half + i)] * hi;
  }
  return out;
}

// ---------------------------------------------------------------------------

void TestColumnSoftmax(emu_test::Runner& runner) {
  runner.Section("softmax_columns: down the window, not across the lanes");
  constexpr int rows = 8;
  constexpr int cols = 5;
  std::vector<float> logits(rows * cols);
  for (auto& v : logits) {
    v = emu_test::UniformFloat(-4.0F, 4.0F);
  }
  std::vector<float> got(logits.size());
  compressor_column_softmax(logits.data(), got.data(), rows, cols);
  const std::vector<float> want = ReferenceColumnSoftmax(logits, rows, cols);

  for (std::size_t i = 0; i < got.size(); ++i) {
    runner.CheckNear(got[i], want[i], 1e-7, "softmax entry " + std::to_string(i));
  }
  // Each LANE must sum to 1 (axis 0). Rows must NOT.
  for (int c = 0; c < cols; ++c) {
    float sum = 0.0F;
    for (int r = 0; r < rows; ++r) {
      sum += got[static_cast<std::size_t>(r) * cols + c];
    }
    runner.CheckNear(sum, 1.0, 1e-6, "lane " + std::to_string(c) + " sums to 1");
  }
  bool some_row_differs = false;
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (int c = 0; c < cols; ++c) {
      sum += got[static_cast<std::size_t>(r) * cols + c];
    }
    if (std::fabs(sum - 1.0F) > 1e-3F) {
      some_row_differs = true;
    }
  }
  runner.Check(some_row_differs,
               "rows do NOT sum to 1 -- the softmax is columnar (a row-wise "
               "implementation would pass the lane check too)");

  // -inf rows contribute exactly zero, which is what makes the first group's
  // missing predecessor harmless.
  {
    std::vector<float> masked(rows * cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        masked[static_cast<std::size_t>(r) * cols + c] =
            (r < 4) ? -std::numeric_limits<float>::infinity() : 1.0F;
      }
    }
    std::vector<float> out(masked.size());
    compressor_column_softmax(masked.data(), out.data(), rows, cols);
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < cols; ++c) {
        runner.CheckNear(out[static_cast<std::size_t>(r) * cols + c], 0.0, 0.0,
                         "a -inf row gets exactly zero weight");
      }
    }
    for (int c = 0; c < cols; ++c) {
      runner.CheckNear(out[static_cast<std::size_t>(4) * cols + c], 0.25, 1e-6,
                       "the four live rows share the weight equally");
    }
  }

  // An all--inf lane is written as zero, not NaN.
  {
    std::vector<float> all_masked(rows * 1, -std::numeric_limits<float>::infinity());
    std::vector<float> out(all_masked.size());
    compressor_column_softmax(all_masked.data(), out.data(), rows, 1);
    for (int r = 0; r < rows; ++r) {
      runner.Check(out[static_cast<std::size_t>(r)] == 0.0F,
                   "an all--inf lane is zero, not NaN");
    }
  }
}

void TestRopeAndNorm(emu_test::Runner& runner) {
  runner.Section("epilog: RMSNorm over D, half-mode RoPE on the LAST 64 lanes");
  const Weights w = MakeWeights();

  std::vector<float> x(kDHead);
  for (auto& v : x) {
    v = emu_test::UniformFloat(-2.0F, 2.0F);
  }
  std::vector<float> normalized(kDHead);
  const float inv_rms =
      compressor_rms_norm(x.data(), w.norm.data(), normalized.data(), kDHead,
                          kRmsNormEpsilon);

  double sum_squares = 0.0;
  for (int i = 0; i < kDHead; ++i) {
    sum_squares += static_cast<double>(x[static_cast<std::size_t>(i)]) *
                   static_cast<double>(x[static_cast<std::size_t>(i)]);
  }
  const double expected_inv_rms =
      1.0 / std::sqrt(sum_squares / kDHead + kRmsNormEpsilon);
  runner.CheckNear(inv_rms, expected_inv_rms, 1e-6, "inv_rms");
  for (int i = 0; i < kDHead; ++i) {
    runner.CheckNear(normalized[static_cast<std::size_t>(i)],
                     Bf16ToFloat(w.norm[static_cast<std::size_t>(i)]) *
                         x[static_cast<std::size_t>(i)] * inv_rms,
                     1e-6, "rms norm lane " + std::to_string(i));
  }

  // RoPE must leave lanes [0, D - 64) untouched and rotate the rest.
  std::vector<float> before = normalized;
  std::vector<float> after = normalized;
  compressor_apply_rope(after.data(), w.cos.data(), w.sin.data(), kDHead, kRopeHeadDim,
                        RotaryMode::kHalf);
  const int base = kDHead - kRopeHeadDim;
  for (int i = 0; i < base; ++i) {
    runner.CheckNear(after[static_cast<std::size_t>(i)],
                     before[static_cast<std::size_t>(i)], 0.0,
                     "lane " + std::to_string(i) + " is before the RoPE slice");
  }
  const int half = kRopeHeadDim / 2;
  for (int i = 0; i < half; ++i) {
    const float lo = before[static_cast<std::size_t>(base + i)];
    const float hi = before[static_cast<std::size_t>(base + half + i)];
    runner.CheckNear(after[static_cast<std::size_t>(base + i)],
                     w.cos[static_cast<std::size_t>(i)] * lo -
                         w.sin[static_cast<std::size_t>(i)] * hi,
                     1e-6, "rope lo lane " + std::to_string(i));
    runner.CheckNear(after[static_cast<std::size_t>(base + half + i)],
                     w.sin[static_cast<std::size_t>(half + i)] * lo +
                         w.cos[static_cast<std::size_t>(half + i)] * hi,
                     1e-6, "rope hi lane " + std::to_string(i));
  }

  // A rotation preserves the norm of each (lo, hi) pair: the cheapest check
  // that the two output lanes were not swapped.
  for (int i = 0; i < half; ++i) {
    const double before_energy =
        static_cast<double>(before[static_cast<std::size_t>(base + i)]) *
            before[static_cast<std::size_t>(base + i)] +
        static_cast<double>(before[static_cast<std::size_t>(base + half + i)]) *
            before[static_cast<std::size_t>(base + half + i)];
    const double after_energy =
        static_cast<double>(after[static_cast<std::size_t>(base + i)]) *
            after[static_cast<std::size_t>(base + i)] +
        static_cast<double>(after[static_cast<std::size_t>(base + half + i)]) *
            after[static_cast<std::size_t>(base + half + i)];
    runner.CheckRelative(after_energy, before_energy, 1e-5,
                         "rope pair " + std::to_string(i) + " is a rotation");
  }

  // Interleave mode must differ from half mode -- otherwise the attribute is
  // not actually being honoured.
  std::vector<float> interleaved = normalized;
  compressor_apply_rope(interleaved.data(), w.cos.data(), w.sin.data(), kDHead,
                        kRopeHeadDim, RotaryMode::kInterleave);
  bool differs = false;
  for (int i = base; i < kDHead; ++i) {
    if (std::fabs(interleaved[static_cast<std::size_t>(i)] -
                  after[static_cast<std::size_t>(i)]) > 1e-6F) {
      differs = true;
    }
  }
  runner.Check(differs, "rotary_mode 2 (interleave) differs from mode 1 (half)");

  // A null table is a pass-through, not a crash.
  std::vector<float> untouched = normalized;
  compressor_apply_rope(untouched.data(), nullptr, nullptr, kDHead, kRopeHeadDim,
                        RotaryMode::kHalf);
  for (int i = 0; i < kDHead; ++i) {
    runner.CheckNear(untouched[static_cast<std::size_t>(i)],
                     normalized[static_cast<std::size_t>(i)], 0.0,
                     "no rope table means pass-through");
  }
}

void TestC4aCadence(emu_test::Runner& runner) {
  runner.Section("C4A cadence: steps 0-2 HOLD, step 3 EMIT, rollover at 4-7");
  const Weights w = MakeWeights();
  CompressorRingState state;

  // A canary pattern in the destination: a HOLD step must not write ANY of it.
  constexpr std::uint16_t kCanary = 0xBEEF;
  std::vector<std::uint16_t> slot(kDHead, kCanary);

  std::vector<std::vector<std::uint16_t>> tokens;
  for (int step = 0; step < 8; ++step) {
    tokens.push_back(MakeToken());
  }

  // --- steps 0, 1, 2: HOLD -------------------------------------------------
  for (int step = 0; step < 3; ++step) {
    const CompressorStepResult result = paged_compressor_decode_c4a(
        tokens[static_cast<std::size_t>(step)].data(), w.wkv.data(), w.wgate.data(),
        w.ape.data(), w.norm.data(), step, &state, w.cos.data(), w.sin.data(),
        slot.data(), kH, kDHead, kRatio);
    runner.Check(!result.emitted, "step " + std::to_string(step) + " is a HOLD");
    runner.Check(result.compressed_index == -1,
                 "step " + std::to_string(step) + " produced no compressed index");
    bool clean = true;
    for (const std::uint16_t v : slot) {
      if (v != kCanary) {
        clean = false;
      }
    }
    runner.Check(clean, "step " + std::to_string(step) +
                            " left the paged slot completely untouched");
    runner.Check(!state.have_previous,
                 "no previous group exists yet at step " + std::to_string(step));
    runner.Check(state.tokens_seen == step + 1,
                 "state counted " + std::to_string(step + 1) + " tokens");
  }

  // --- step 3: EMIT --------------------------------------------------------
  {
    const CompressorStepResult result = paged_compressor_decode_c4a(
        tokens[3].data(), w.wkv.data(), w.wgate.data(), w.ape.data(), w.norm.data(), 3,
        &state, w.cos.data(), w.sin.data(), slot.data(), kH, kDHead, kRatio);
    runner.Check(result.emitted, "step 3 EMITs");
    runner.Check(result.compressed_index == 0, "step 3 emits compressed token 0");
    runner.Check(result.inv_rms > 0.0F && std::isfinite(result.inv_rms),
                 "step 3 reports a finite inv_rms");
    runner.Check(state.have_previous, "step 3 rolls its a-halves into `previous`");

    const std::vector<std::vector<std::uint16_t>> group0(tokens.begin(),
                                                         tokens.begin() + 4);
    const std::vector<float> want = ReferenceEmit(w, group0, {});
    for (int i = 0; i < kDHead; ++i) {
      runner.CheckRelative(Bf16ToFloat(slot[static_cast<std::size_t>(i)]),
                           want[static_cast<std::size_t>(i)], 0.01,
                           "group 0 lane " + std::to_string(i));
    }
    bool wrote_something = false;
    for (const std::uint16_t v : slot) {
      if (v != kCanary) {
        wrote_something = true;
      }
    }
    runner.Check(wrote_something, "step 3 actually wrote the slot");
  }

  // --- steps 4, 5, 6: HOLD again, with a predecessor now present -----------
  std::fill(slot.begin(), slot.end(), kCanary);
  for (int step = 4; step < 7; ++step) {
    const CompressorStepResult result = paged_compressor_decode_c4a(
        tokens[static_cast<std::size_t>(step)].data(), w.wkv.data(), w.wgate.data(),
        w.ape.data(), w.norm.data(), step, &state, w.cos.data(), w.sin.data(),
        slot.data(), kH, kDHead, kRatio);
    runner.Check(!result.emitted, "step " + std::to_string(step) + " is a HOLD");
    bool clean = true;
    for (const std::uint16_t v : slot) {
      if (v != kCanary) {
        clean = false;
      }
    }
    runner.Check(clean, "step " + std::to_string(step) + " left the slot untouched");
  }

  // --- step 7: EMIT with the coff=2 overlap ---------------------------------
  {
    const CompressorStepResult result = paged_compressor_decode_c4a(
        tokens[7].data(), w.wkv.data(), w.wgate.data(), w.ape.data(), w.norm.data(), 7,
        &state, w.cos.data(), w.sin.data(), slot.data(), kH, kDHead, kRatio);
    runner.Check(result.emitted, "step 7 EMITs");
    runner.Check(result.compressed_index == 1, "step 7 emits compressed token 1");

    const std::vector<std::vector<std::uint16_t>> group0(tokens.begin(),
                                                         tokens.begin() + 4);
    const std::vector<std::vector<std::uint16_t>> group1(tokens.begin() + 4,
                                                         tokens.begin() + 8);
    const std::vector<float> want = ReferenceEmit(w, group1, group0);
    for (int i = 0; i < kDHead; ++i) {
      runner.CheckRelative(Bf16ToFloat(slot[static_cast<std::size_t>(i)]),
                           want[static_cast<std::size_t>(i)], 0.01,
                           "group 1 lane " + std::to_string(i) +
                               " (overlapped with group 0)");
    }

    // The overlap has to matter: a reduction that ignored the predecessor
    // would produce a different token. Verify the coff=1-style answer is NOT
    // what we got.
    const std::vector<float> without_overlap = ReferenceEmit(w, group1, {});
    bool overlap_changed_result = false;
    for (int i = 0; i < kDHead; ++i) {
      if (std::fabs(without_overlap[static_cast<std::size_t>(i)] -
                    want[static_cast<std::size_t>(i)]) > 1e-3F) {
        overlap_changed_result = true;
      }
    }
    runner.Check(overlap_changed_result,
                 "the coff=2 overlap genuinely changes group 1's token, so the "
                 "predecessor rows are really being consumed");
  }

  // --- the cadence is periodic over many groups ---------------------------
  int emits = 0;
  for (int step = 8; step < 40; ++step) {
    const std::vector<std::uint16_t> token = MakeToken();
    const CompressorStepResult result = paged_compressor_decode_c4a(
        token.data(), w.wkv.data(), w.wgate.data(), w.ape.data(), w.norm.data(), step,
        &state, w.cos.data(), w.sin.data(), slot.data(), kH, kDHead, kRatio);
    if (result.emitted) {
      ++emits;
      runner.Check(result.compressed_index == step / kRatio,
                   "step " + std::to_string(step) + " emits token " +
                       std::to_string(step / kRatio));
      runner.Check(step % kRatio == kRatio - 1,
                   "an EMIT only happens on the last step of a group");
    }
  }
  runner.Check(emits == 8, "steps 8..39 produce exactly 8 compressed tokens");
}

void TestDeclaredCoff1EntryPoint(emu_test::Runner& runner) {
  runner.Section("declared entry point (coff=1, [4, 512] ring buffer)");
  constexpr int width = 512;
  constexpr int head_dim = width / 2;

  std::vector<float> ring(static_cast<std::size_t>(kCompressorRatio) * width, 0.0F);
  std::vector<std::uint16_t> slot(head_dim, 0xBEEF);
  std::vector<std::vector<std::uint16_t>> rows;
  for (int step = 0; step < 4; ++step) {
    std::vector<std::uint16_t> row(width);
    for (auto& v : row) {
      v = FloatToBf16(emu_test::UniformFloat(-2.0F, 2.0F));
    }
    rows.push_back(row);
  }

  std::vector<float> cos_table(kRopeHeadDim, 1.0F);
  std::vector<float> sin_table(kRopeHeadDim, 0.0F);

  for (int step = 0; step < 3; ++step) {
    const bool emitted = paged_compressor_decode_cpu(
        rows[static_cast<std::size_t>(step)].data(), step, ring.data(),
        cos_table.data(), sin_table.data(), slot.data(), kCompressorRatio);
    runner.Check(!emitted, "coff=1 step " + std::to_string(step) + " HOLDs");
    bool slot_clean = true;
    for (const std::uint16_t v : slot) {
      if (v != 0xBEEF) {
        slot_clean = false;
      }
    }
    runner.Check(slot_clean, "coff=1 HOLD leaves the whole slot untouched");
    // The buffered row must be exactly what was handed in.
    for (int i = 0; i < width; i += 97) {
      runner.CheckNear(ring[static_cast<std::size_t>(step) * width + i],
                       Bf16ToFloat(rows[static_cast<std::size_t>(step)]
                                       [static_cast<std::size_t>(i)]),
                       0.0, "coff=1 buffered lane " + std::to_string(i));
    }
  }

  const bool emitted = paged_compressor_decode_cpu(
      rows[3].data(), 3, ring.data(), cos_table.data(), sin_table.data(), slot.data(),
      kCompressorRatio);
  runner.Check(emitted, "coff=1 step 3 EMITs");

  // Reference: split each 512-wide row into [kv 256 | score 256], softmax the
  // score halves down the 4 rows, gate-reduce the kv halves, RMSNorm with
  // gain 1, identity RoPE.
  std::vector<float> score(static_cast<std::size_t>(kCompressorRatio) * head_dim);
  std::vector<float> kv(score.size());
  for (int r = 0; r < kCompressorRatio; ++r) {
    for (int i = 0; i < head_dim; ++i) {
      kv[static_cast<std::size_t>(r) * head_dim + i] =
          Bf16ToFloat(rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)]);
      score[static_cast<std::size_t>(r) * head_dim + i] = Bf16ToFloat(
          rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(head_dim + i)]);
    }
  }
  const std::vector<float> weights =
      ReferenceColumnSoftmax(score, kCompressorRatio, head_dim);
  std::vector<float> compressed(head_dim, 0.0F);
  for (int r = 0; r < kCompressorRatio; ++r) {
    for (int i = 0; i < head_dim; ++i) {
      compressed[static_cast<std::size_t>(i)] +=
          weights[static_cast<std::size_t>(r) * head_dim + i] *
          kv[static_cast<std::size_t>(r) * head_dim + i];
    }
  }
  double sum_squares = 0.0;
  for (int i = 0; i < head_dim; ++i) {
    sum_squares += static_cast<double>(compressed[static_cast<std::size_t>(i)]) *
                   static_cast<double>(compressed[static_cast<std::size_t>(i)]);
  }
  const float inv_rms =
      1.0F / std::sqrt(static_cast<float>(sum_squares / head_dim) + kRmsNormEpsilon);
  for (int i = 0; i < head_dim; i += 13) {
    runner.CheckRelative(Bf16ToFloat(slot[static_cast<std::size_t>(i)]),
                         compressed[static_cast<std::size_t>(i)] * inv_rms, 0.01,
                         "coff=1 compressed lane " + std::to_string(i));
  }
}

void TestRejectsBadArguments(emu_test::Runner& runner) {
  runner.Section("argument validation");
  std::vector<float> ring(static_cast<std::size_t>(kCompressorRatio) * 512, 0.0F);
  std::vector<std::uint16_t> row(512, 0);
  std::vector<std::uint16_t> slot(256, 0);

  const auto throws = [](auto&& callable) {
    try {
      callable();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  };

  runner.Check(throws([&] {
                 paged_compressor_decode_cpu(nullptr, 0, ring.data(), nullptr, nullptr,
                                             slot.data(), kCompressorRatio);
               }),
               "a null cur_kv is rejected");
  runner.Check(throws([&] {
                 paged_compressor_decode_cpu(row.data(), -1, ring.data(), nullptr,
                                             nullptr, slot.data(), kCompressorRatio);
               }),
               "a negative step is rejected");
  runner.Check(throws([&] {
                 paged_compressor_decode_cpu(row.data(), 0, ring.data(), nullptr,
                                             nullptr, slot.data(), 0);
               }),
               "ratio = 0 is rejected");
  // The ring buffer is contractually [4, 512] and the function indexes it at
  // step % ratio, so a larger ratio would write past a correctly sized caller
  // buffer. The signature carries no length, so the bound is the only defence.
  for (const int bad_ratio : {5, 8, 64}) {
    runner.Check(throws([&] {
                   paged_compressor_decode_cpu(row.data(), 0, ring.data(), nullptr,
                                               nullptr, slot.data(), bad_ratio);
                 }),
                 "ratio = " + std::to_string(bad_ratio) +
                     " is rejected (it would overrun the [4, 512] ring buffer)");
  }
  // And the whole legal range is accepted.
  for (const int good_ratio : {1, 2, 3, 4}) {
    std::vector<float> fresh(static_cast<std::size_t>(kCompressorRatio) * 512, 0.0F);
    bool accepted = true;
    try {
      paged_compressor_decode_cpu(row.data(), 0, fresh.data(), nullptr, nullptr,
                                  slot.data(), good_ratio);
    } catch (...) {
      accepted = false;
    }
    runner.Check(accepted, "ratio = " + std::to_string(good_ratio) + " is accepted");
  }
  // An EMIT step with no destination must be refused rather than writing to
  // address zero.
  runner.Check(throws([&] {
                 paged_compressor_decode_cpu(row.data(), 3, ring.data(), nullptr,
                                             nullptr, nullptr, kCompressorRatio);
               }),
               "a null slot on an EMIT step is rejected");
  // ... but a HOLD step legitimately does not need one.
  bool hold_without_slot_ok = true;
  try {
    paged_compressor_decode_cpu(row.data(), 0, ring.data(), nullptr, nullptr, nullptr,
                                kCompressorRatio);
  } catch (...) {
    hold_without_slot_ok = false;
  }
  runner.Check(hold_without_slot_ok, "a HOLD step does not require a slot");
}

}  // namespace

int main() {
  emu_test::Runner runner("test_paged_compressor_emu");
  try {
    TestColumnSoftmax(runner);
    TestRopeAndNorm(runner);
    TestC4aCadence(runner);
    TestDeclaredCoff1EntryPoint(runner);
    TestRejectsBadArguments(runner);
  } catch (const std::exception& error) {
    std::printf("   FATAL: unexpected exception: %s\n", error.what());
    return 1;
  }
  return runner.Summary();
}
