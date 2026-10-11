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

// The layer machinery behind the semi-end-to-end harness: quantized GEMVs,
// the sliding-window cache, shared-KV attention, and the MoE router/experts.
//
// The tests that matter most here are the INDEXING ones. The checkpoint
// carries two different quantization schemes that look alike -- bytes plus
// E8M0 scales -- but differ in block geometry (FP8: 128x128 blocks; FP4: per
// row, per 32-element group). Reading one with the other's indexing produces
// finite, plausibly-scaled garbage, so each GEMV is checked by perturbing ONE
// scale and asserting exactly which outputs move.

#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"
#include "moe/emu/semi_e2e_pipeline.hpp"

#include "emu_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using namespace moe::emu;

// --- FP4 E2M1 ---------------------------------------------------------------

void TestFp4Decode(emu_test::Runner& runner) {
  runner.Section("FP4 E2M1: all 16 codes, no inf and no NaN");
  // 1-2-1 with bias 1: {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3.
  const float expected[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  for (int code = 0; code < 8; ++code) {
    runner.CheckNear(DecodeFp4E2M1(static_cast<std::uint8_t>(code)), expected[code],
                     0.0, "code " + std::to_string(code));
    runner.CheckNear(DecodeFp4E2M1(static_cast<std::uint8_t>(code | 0x08)),
                     -expected[code], 0.0,
                     "code " + std::to_string(code | 0x08) + " is the negation");
  }
  // Every code is finite: E2M1 has no reserved encodings at all, which is why
  // the 4-bit range reaches 6 rather than stopping at 4.
  for (int code = 0; code < 16; ++code) {
    runner.Check(std::isfinite(DecodeFp4E2M1(static_cast<std::uint8_t>(code))),
                 "code " + std::to_string(code) + " is finite");
  }
  runner.CheckNear(DecodeFp4E2M1(0x07), 6.0, 0.0, "the maximum magnitude is 6");
}

// --- FP8 block-scaled GEMV --------------------------------------------------

void TestFp8BlockGemv(emu_test::Runner& runner) {
  runner.Section("FP8 block GEMV: 128x128 scale grid");
  // Two block rows by three block columns, so both grid axes are exercised.
  constexpr int rows = 256;
  constexpr int cols = 384;
  constexpr int scale_rows = rows / kFp8BlockSize;  // 2
  constexpr int scale_cols = cols / kFp8BlockSize;  // 3

  std::vector<std::uint8_t> weight(static_cast<std::size_t>(rows) * cols);
  for (auto& byte : weight) {
    byte = EncodeFp8E4M3(emu_test::UniformFloat(-4.0F, 4.0F));
  }
  std::vector<std::uint8_t> scale(scale_rows * scale_cols);
  for (std::size_t i = 0; i < scale.size(); ++i) {
    scale[i] = static_cast<std::uint8_t>(125 + i);  // distinct powers of two
  }
  std::vector<float> x(cols);
  for (float& value : x) {
    value = emu_test::UniformFloat(-1.0F, 1.0F);
  }

  Fp8BlockMatrix matrix;
  matrix.weight = weight.data();
  matrix.scale = scale.data();
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.scale_cols = scale_cols;

  std::vector<float> got(rows);
  fp8_block_gemv_cpu(matrix, x.data(), got.data());

  // Naive reference, written longhand.
  std::vector<float> want(rows, 0.0F);
  for (int row = 0; row < rows; ++row) {
    double total = 0.0;
    for (int col = 0; col < cols; ++col) {
      const float value =
          DecodeFp8E4M3(weight[static_cast<std::size_t>(row) * cols + col]);
      const float block_scale = DecodeFp8E8M0(
          scale[static_cast<std::size_t>(row / kFp8BlockSize) * scale_cols +
                col / kFp8BlockSize]);
      total += static_cast<double>(value) * block_scale * x[static_cast<std::size_t>(col)];
    }
    want[static_cast<std::size_t>(row)] = static_cast<float>(total);
  }
  for (int row = 0; row < rows; ++row) {
    runner.CheckRelative(got[static_cast<std::size_t>(row)],
                         want[static_cast<std::size_t>(row)], 1e-4,
                         "fp8 gemv row " + std::to_string(row));
  }

  // THE INDEXING TEST: bump one block scale and assert exactly the rows and
  // columns of that 128x128 block change. This is what separates a correct
  // block walk from a per-row or per-tensor one.
  const int target_row_block = 1;
  const int target_col_block = 2;
  std::vector<std::uint8_t> bumped = scale;
  bumped[static_cast<std::size_t>(target_row_block) * scale_cols + target_col_block] =
      static_cast<std::uint8_t>(bumped[static_cast<std::size_t>(target_row_block) *
                                           scale_cols + target_col_block] + 1);
  matrix.scale = bumped.data();
  std::vector<float> after(rows);
  fp8_block_gemv_cpu(matrix, x.data(), after.data());

  int changed_inside = 0;
  int changed_outside = 0;
  for (int row = 0; row < rows; ++row) {
    const bool in_block = (row / kFp8BlockSize) == target_row_block;
    const bool differs = std::fabs(after[static_cast<std::size_t>(row)] -
                                   got[static_cast<std::size_t>(row)]) > 1e-6F;
    if (in_block && differs) {
      ++changed_inside;
    }
    if (!in_block && differs) {
      ++changed_outside;
    }
  }
  runner.Check(changed_inside > kFp8BlockSize / 2,
               "bumping one block scale moves the rows inside that block row");
  runner.Check(changed_outside == 0,
               "and moves NO row outside it (the scale grid really is 2-D)");

  // A scale grid too narrow for the declared columns must be refused rather
  // than read out of bounds.
  bool threw = false;
  try {
    Fp8BlockMatrix narrow = matrix;
    narrow.scale_cols = 1;
    std::vector<float> ignored(rows);
    fp8_block_gemv_cpu(narrow, x.data(), ignored.data());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  runner.Check(threw, "an undersized fp8 scale grid is rejected");
}

// --- MXFP4 group-scaled GEMV ------------------------------------------------

void TestMxfp4Gemv(emu_test::Runner& runner) {
  runner.Section("MXFP4 GEMV: per-row, per-32-group scales");
  constexpr int rows = 8;
  constexpr int cols = 128;  // 4 groups of 32
  constexpr int groups = cols / kMxfp4GroupSize;

  std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows) * (cols / 2));
  for (auto& byte : packed) {
    byte = static_cast<std::uint8_t>(emu_test::Rng()() & 0xFFU);
  }
  std::vector<std::uint8_t> scale(static_cast<std::size_t>(rows) * groups);
  for (std::size_t i = 0; i < scale.size(); ++i) {
    scale[i] = static_cast<std::uint8_t>(124 + (i % 5));
  }
  std::vector<float> x(cols);
  for (float& value : x) {
    value = emu_test::UniformFloat(-1.0F, 1.0F);
  }

  Mxfp4Matrix matrix;
  matrix.packed = packed.data();
  matrix.scale = scale.data();
  matrix.rows = rows;
  matrix.cols = cols;

  std::vector<float> got(rows);
  mxfp4_gemv_cpu(matrix, x.data(), got.data());

  // Naive reference with the nibble order spelled out, so the GEMV's pair
  // table cannot agree with itself by construction.
  std::vector<float> want(rows, 0.0F);
  for (int row = 0; row < rows; ++row) {
    double total = 0.0;
    for (int col = 0; col < cols; ++col) {
      const std::uint8_t byte =
          packed[static_cast<std::size_t>(row) * (cols / 2) + col / 2];
      const bool even = (col % 2) == 0;
      const std::uint8_t low = byte & 0x0FU;
      const std::uint8_t high = static_cast<std::uint8_t>(byte >> 4U);
      std::uint8_t code;
      if (kMxfp4LowNibbleFirst) {
        code = even ? low : high;
      } else {
        code = even ? high : low;
      }
      const float value = DecodeFp4E2M1(code);
      const float group_scale = DecodeFp8E8M0(
          scale[static_cast<std::size_t>(row) * groups + col / kMxfp4GroupSize]);
      total += static_cast<double>(value) * group_scale * x[static_cast<std::size_t>(col)];
    }
    want[static_cast<std::size_t>(row)] = static_cast<float>(total);
  }
  for (int row = 0; row < rows; ++row) {
    runner.CheckRelative(got[static_cast<std::size_t>(row)],
                         want[static_cast<std::size_t>(row)], 1e-4,
                         "fp4 gemv row " + std::to_string(row));
  }

  // The dequantized row must agree with the GEMV, so the pair table and the
  // row walk cannot drift apart.
  std::vector<float> row_values(cols);
  mxfp4_dequant_row_cpu(matrix, 3, row_values.data());
  double dot = 0.0;
  for (int col = 0; col < cols; ++col) {
    dot += static_cast<double>(row_values[static_cast<std::size_t>(col)]) *
           x[static_cast<std::size_t>(col)];
  }
  runner.CheckRelative(dot, got[3], 1e-4, "dequant_row dotted with x matches the GEMV");

  // THE INDEXING TEST: the FP4 scale is per ROW per GROUP, not per block.
  // Bump row 5's group 2 and assert only row 5 moves.
  std::vector<std::uint8_t> bumped = scale;
  bumped[static_cast<std::size_t>(5) * groups + 2] =
      static_cast<std::uint8_t>(bumped[static_cast<std::size_t>(5) * groups + 2] + 1);
  matrix.scale = bumped.data();
  std::vector<float> after(rows);
  mxfp4_gemv_cpu(matrix, x.data(), after.data());
  for (int row = 0; row < rows; ++row) {
    const bool differs = std::fabs(after[static_cast<std::size_t>(row)] -
                                   got[static_cast<std::size_t>(row)]) > 1e-6F;
    runner.Check(differs == (row == 5),
                 "bumping row 5's group scale moves row " + std::to_string(row) +
                     (row == 5 ? " (expected)" : " not at all"));
  }
  // And within row 5, only columns [64, 96) contributed to the change.
  matrix.scale = scale.data();
  std::vector<float> baseline_row(cols);
  mxfp4_dequant_row_cpu(matrix, 5, baseline_row.data());
  matrix.scale = bumped.data();
  std::vector<float> bumped_row(cols);
  mxfp4_dequant_row_cpu(matrix, 5, bumped_row.data());
  for (int col = 0; col < cols; ++col) {
    const bool in_group = col >= 64 && col < 96;
    const bool differs = std::fabs(bumped_row[static_cast<std::size_t>(col)] -
                                   baseline_row[static_cast<std::size_t>(col)]) > 1e-9F;
    // A zero-valued element does not move however the scale changes.
    if (baseline_row[static_cast<std::size_t>(col)] != 0.0F) {
      runner.Check(differs == in_group,
                   "column " + std::to_string(col) + " is " +
                       (in_group ? "inside" : "outside") + " the bumped 32-group");
    }
  }

  // A column count that is not a multiple of 32 cannot be a valid MX layout.
  bool threw = false;
  try {
    Mxfp4Matrix ragged = matrix;
    ragged.cols = 100;
    std::vector<float> ignored(rows);
    mxfp4_gemv_cpu(ragged, x.data(), ignored.data());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  runner.Check(threw, "a column count that is not a multiple of 32 is rejected");
}

// --- RMSNorm ----------------------------------------------------------------

void TestRmsNorm(emu_test::Runner& runner) {
  runner.Section("RMSNorm, with and without a learned gain");
  constexpr int count = 512;
  std::vector<float> x(count);
  for (float& value : x) {
    value = emu_test::UniformFloat(-3.0F, 3.0F);
  }
  std::vector<std::uint16_t> weight(count);
  for (auto& value : weight) {
    value = FloatToBf16(emu_test::UniformFloat(0.5F, 1.5F));
  }

  std::vector<float> got(count);
  const float inv_rms = rms_norm_cpu(x.data(), weight.data(), got.data(), count, 1e-6F);
  double sum_squares = 0.0;
  for (float value : x) {
    sum_squares += static_cast<double>(value) * value;
  }
  const double expected_inv = 1.0 / std::sqrt(sum_squares / count + 1e-6);
  runner.CheckNear(inv_rms, expected_inv, 1e-6, "inv_rms");
  for (int i = 0; i < count; ++i) {
    runner.CheckNear(got[static_cast<std::size_t>(i)],
                     Bf16ToFloat(weight[static_cast<std::size_t>(i)]) *
                         x[static_cast<std::size_t>(i)] * inv_rms,
                     1e-5, "lane " + std::to_string(i));
  }

  // The has_weight=False variant (q_norm_without_weight) must normalize with
  // unit gain, and its result must have unit RMS.
  std::vector<float> plain(count);
  rms_norm_cpu(x.data(), nullptr, plain.data(), count, 1e-6F);
  double normed_squares = 0.0;
  for (float value : plain) {
    normed_squares += static_cast<double>(value) * value;
  }
  runner.CheckNear(std::sqrt(normed_squares / count), 1.0, 1e-4,
                   "a weightless RMSNorm yields unit RMS");
}

// --- the sliding-window cache ----------------------------------------------

void TestWindowCache(emu_test::Runner& runner) {
  runner.Section("sliding-window KV cache: ring indexing and eviction");
  constexpr int window = 4;
  constexpr int head_dim = 8;
  SharedKvWindowCache cache(window, head_dim);
  runner.Check(cache.live_tokens() == 0, "a fresh cache holds nothing");
  runner.Check(cache.At(0) == nullptr, "and returns null for any position");

  for (int position = 0; position < 10; ++position) {
    std::vector<float> row(head_dim,
                           static_cast<float>(position) + 0.5F);  // identifiable
    const int live = cache.Append(row.data(), position);
    runner.Check(live == std::min(window, position + 1),
                 "after position " + std::to_string(position) + " the window holds " +
                     std::to_string(live));

    // Everything inside the window reads back its own value; everything
    // older is evicted.
    for (int probe = 0; probe <= position; ++probe) {
      const float* stored = cache.At(probe);
      if (probe > position - window) {
        runner.Check(stored != nullptr,
                     "position " + std::to_string(probe) + " is still live");
        if (stored != nullptr) {
          runner.CheckNear(stored[0], static_cast<double>(probe) + 0.5, 0.0,
                           "position " + std::to_string(probe) +
                               " reads back its own row");
        }
      } else {
        runner.Check(stored == nullptr,
                     "position " + std::to_string(probe) + " has been evicted");
      }
    }
    runner.Check(cache.oldest_position() == std::max(0, position - window + 1),
                 "oldest_position tracks the window");
  }

  // Positions must advance: a decode step cannot rewrite history, and letting
  // it would corrupt the ring silently.
  std::vector<float> row(head_dim, 1.0F);
  bool threw = false;
  try {
    cache.Append(row.data(), 5);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  runner.Check(threw, "appending a stale position is rejected");
}

// --- the routers ------------------------------------------------------------

void TestSqrtSoftplus(emu_test::Runner& runner) {
  runner.Section("sqrtsoftplus scoring");
  // softplus(x) = ln(1 + e^x) below threshold 20, identity above it.
  for (const float logit : {-10.0F, -1.0F, 0.0F, 1.0F, 5.0F, 19.0F}) {
    const double expected = std::sqrt(std::log1p(std::exp(static_cast<double>(logit))));
    runner.CheckRelative(SqrtSoftplus(logit), expected, 1e-5,
                         "sqrtsoftplus(" + std::to_string(logit) + ")");
  }
  // Above the threshold softplus is linear, which is what stops exp() from
  // overflowing. A naive ln(1+exp(x)) would be inf at 1000 and the sqrt NaN.
  runner.CheckRelative(SqrtSoftplus(100.0F), std::sqrt(100.0), 1e-5,
                       "sqrtsoftplus is linear above the threshold");
  runner.Check(std::isfinite(SqrtSoftplus(1.0e4F)),
               "a huge logit stays finite (no exp overflow)");
  runner.Check(SqrtSoftplus(-100.0F) >= 0.0F && std::isfinite(SqrtSoftplus(-100.0F)),
               "a very negative logit is finite and non-negative");
  // Monotone, so it is a usable score.
  runner.Check(SqrtSoftplus(1.0F) > SqrtSoftplus(0.0F), "monotone increasing");
}

void TestHashRouter(emu_test::Runner& runner) {
  runner.Section("hash router: tid2eid selects, the gate only weights");
  constexpr int num_experts = 256;
  std::vector<float> logits(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    logits[static_cast<std::size_t>(i)] = emu_test::UniformFloat(-3.0F, 3.0F);
  }
  // Give one non-selected expert an overwhelming logit. A score-based router
  // would pick it; a hash router must not.
  logits[7] = 1000.0F;

  const std::int64_t row[kNumExpertsPerTok] = {30, 205, 192, 177, 242, 102};
  const MoeRouting routing = moe_route_hash_cpu(row, "I64", kNumExpertsPerTok,
                                                logits.data(), num_experts, true);
  runner.Check(routing.count == kNumExpertsPerTok, "six experts are selected");
  runner.Check(routing.from_hash_table, "and reported as coming from the table");
  for (int i = 0; i < kNumExpertsPerTok; ++i) {
    runner.Check(routing.experts[i] == static_cast<int>(row[i]),
                 "slot " + std::to_string(i) + " is the table's expert, in order");
  }
  bool selected_the_decoy = false;
  for (int i = 0; i < routing.count; ++i) {
    if (routing.experts[i] == 7) {
      selected_the_decoy = true;
    }
  }
  runner.Check(!selected_the_decoy,
               "an expert with a 1000 logit is NOT selected -- the table, not "
               "the gate, decides membership");

  float total = 0.0F;
  for (int i = 0; i < routing.count; ++i) {
    total += routing.weights[i];
    runner.Check(routing.weights[i] >= 0.0F,
                 "weight " + std::to_string(i) + " is non-negative");
  }
  runner.CheckNear(total, 1.0, 1e-5, "norm_topk_prob renormalizes the weights to 1");

  // The weights must be in the ratio of the raw sqrtsoftplus scores.
  float raw_total = 0.0F;
  for (int i = 0; i < kNumExpertsPerTok; ++i) {
    raw_total += SqrtSoftplus(logits[static_cast<std::size_t>(row[i])]);
  }
  for (int i = 0; i < kNumExpertsPerTok; ++i) {
    runner.CheckRelative(routing.weights[i],
                         SqrtSoftplus(logits[static_cast<std::size_t>(row[i])]) /
                             raw_total,
                         1e-5, "weight " + std::to_string(i) + " is the gate's share");
  }

  // Without renormalization the weights are the raw scores.
  const MoeRouting raw = moe_route_hash_cpu(row, "I64", kNumExpertsPerTok,
                                            logits.data(), num_experts, false);
  for (int i = 0; i < kNumExpertsPerTok; ++i) {
    runner.CheckRelative(raw.weights[i],
                         SqrtSoftplus(logits[static_cast<std::size_t>(row[i])]), 1e-5,
                         "unrenormalized weight " + std::to_string(i));
  }

  // I32 tables are accepted too (hash_router.py validates int32); anything
  // else is refused rather than reinterpreted at the wrong width.
  const std::int32_t narrow_row[kNumExpertsPerTok] = {1, 2, 3, 4, 5, 6};
  const MoeRouting narrow = moe_route_hash_cpu(narrow_row, "I32", kNumExpertsPerTok,
                                               logits.data(), num_experts, true);
  runner.Check(narrow.experts[2] == 3, "an I32 table reads at the right width");
  bool threw = false;
  try {
    moe_route_hash_cpu(row, "F32", kNumExpertsPerTok, logits.data(), num_experts, true);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  runner.Check(threw, "a non-integer tid2eid dtype is rejected");

  // An out-of-range expert id is a corrupt table, not something to clamp.
  const std::int64_t bad_row[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 999};
  threw = false;
  try {
    moe_route_hash_cpu(bad_row, "I64", kNumExpertsPerTok, logits.data(), num_experts,
                       true);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  runner.Check(threw, "an expert id outside [0, n_routed_experts) is rejected");
}

void TestScoreRouter(emu_test::Runner& runner) {
  runner.Section("noaux_tc router: the bias shifts selection, NOT the weight");
  constexpr int num_experts = 16;
  constexpr int top_k = 4;
  std::vector<float> logits(num_experts, 0.0F);
  std::vector<float> bias(num_experts, 0.0F);

  // Experts 0..3 have the best raw scores. The bias then promotes 10..13 past
  // them. A correct noaux_tc selects 10..13 but reports their UNSHIFTED
  // scores as the weights.
  for (int i = 0; i < 4; ++i) {
    logits[static_cast<std::size_t>(i)] = 3.0F;
  }
  for (int i = 10; i < 14; ++i) {
    logits[static_cast<std::size_t>(i)] = 1.0F;
    bias[static_cast<std::size_t>(i)] = 10.0F;
  }

  const MoeRouting routing =
      moe_route_score_cpu(logits.data(), bias.data(), num_experts, top_k, false);
  runner.Check(routing.count == top_k, "top_k experts are selected");
  runner.Check(!routing.from_hash_table, "and reported as score-gated");
  std::set<int> chosen(routing.experts, routing.experts + routing.count);
  runner.Check(chosen == std::set<int>({10, 11, 12, 13}),
               "the bias promotes experts 10-13 into the top-k");
  for (int i = 0; i < routing.count; ++i) {
    runner.CheckRelative(routing.weights[i], SqrtSoftplus(1.0F), 1e-5,
                         "weight " + std::to_string(i) +
                             " is the UNSHIFTED score (the bias is not in it)");
    runner.Check(routing.weights[i] < 5.0F,
                 "the weight is nowhere near the +10 bias");
  }

  // With no bias, selection falls back to the raw scores.
  const MoeRouting unbiased =
      moe_route_score_cpu(logits.data(), nullptr, num_experts, top_k, true);
  std::set<int> unbiased_set(unbiased.experts, unbiased.experts + unbiased.count);
  runner.Check(unbiased_set == std::set<int>({0, 1, 2, 3}),
               "without a bias the best raw scores win");
  float total = 0.0F;
  for (int i = 0; i < unbiased.count; ++i) {
    total += unbiased.weights[i];
  }
  runner.CheckNear(total, 1.0, 1e-5, "and renormalization still sums to 1");

  // Ties must break deterministically on the lower index.
  std::vector<float> flat(num_experts, 2.0F);
  const MoeRouting tied =
      moe_route_score_cpu(flat.data(), nullptr, num_experts, top_k, false);
  for (int i = 0; i < top_k; ++i) {
    runner.Check(tied.experts[i] == i,
                 "an all-ties gate selects expert " + std::to_string(i));
  }
}

// --- the clamped SwiGLU -----------------------------------------------------

void TestClampedSwiglu(emu_test::Runner& runner) {
  runner.Section("expert SwiGLU: the gate clamp is asymmetric");
  // A 1x4 -> 1 expert built so gate and up are directly controllable. w1/w3
  // are [intermediate, hidden], w2 is [hidden, intermediate].
  constexpr int hidden = 32;
  constexpr int intermediate = 32;
  const auto make = [&](float value, int rows, int cols) {
    // An FP8 matrix whose first column is `value` and the rest zero, so
    // W @ x = value * x[0] for every row.
    struct Holder {
      std::vector<std::uint8_t> weight;
      std::vector<std::uint8_t> scale;
    };
    auto holder = std::make_shared<Holder>();
    holder->weight.assign(static_cast<std::size_t>(rows) * cols, EncodeFp8E4M3(0.0F));
    for (int row = 0; row < rows; ++row) {
      holder->weight[static_cast<std::size_t>(row) * cols] = EncodeFp8E4M3(value);
    }
    const int scale_rows = (rows + kFp8BlockSize - 1) / kFp8BlockSize;
    const int scale_cols = (cols + kFp8BlockSize - 1) / kFp8BlockSize;
    holder->scale.assign(static_cast<std::size_t>(scale_rows) * scale_cols, 127);  // 2^0
    Fp8BlockMatrix matrix;
    matrix.weight = holder->weight.data();
    matrix.scale = holder->scale.data();
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.scale_cols = scale_cols;
    return std::make_pair(matrix, holder);
  };

  const float limit = kSwigluLimit;  // 10
  std::vector<float> scratch;
  std::vector<float> out(hidden);

  // gate = 4 * x0, up = 1 * x0, down = identity-ish (1 in the first column).
  auto w1 = make(4.0F, intermediate, hidden);
  auto w3 = make(1.0F, intermediate, hidden);
  auto w2 = make(1.0F, hidden, intermediate);

  const auto expected_first_lane = [&](float x0) {
    const float gate = std::min(4.0F * x0, limit);
    const float up = std::min(limit, std::max(-limit, 1.0F * x0));
    const float silu = gate / (1.0F + std::exp(-gate));
    // w2's first column is 1 and the activation vector is constant, so
    // out[row] = h[0].
    return silu * up;
  };

  for (const float x0 : {0.5F, 1.0F, 2.0F, 5.0F, -5.0F, -40.0F}) {
    std::vector<float> x(hidden, 0.0F);
    x[0] = x0;
    moe_expert_swiglu_fp8_cpu(w1.first, w3.first, w2.first, x.data(), out.data(), limit,
                              &scratch);
    runner.CheckRelative(out[0], expected_first_lane(x0), 1e-4,
                         "swiglu at x0 = " + std::to_string(x0));
  }

  // THE ASYMMETRY: at x0 = 5 the gate is 20, clamped to 10; at x0 = -5 the
  // gate is -20 and is NOT clamped, because the gate is bounded above only.
  {
    std::vector<float> x(hidden, 0.0F);
    x[0] = 5.0F;
    moe_expert_swiglu_fp8_cpu(w1.first, w3.first, w2.first, x.data(), out.data(), limit,
                              &scratch);
    const float clamped = out[0];
    // If the gate were clamped symmetrically, the -5 case would give
    // silu(-10) * -5 instead of silu(-20) * -5. Both are tiny but distinct.
    x[0] = -5.0F;
    moe_expert_swiglu_fp8_cpu(w1.first, w3.first, w2.first, x.data(), out.data(), limit,
                              &scratch);
    const float negative = out[0];
    const float silu_minus_20 = -20.0F / (1.0F + std::exp(20.0F));
    const float silu_minus_10 = -10.0F / (1.0F + std::exp(10.0F));
    runner.CheckRelative(negative, silu_minus_20 * -5.0F, 1e-3,
                         "the gate is NOT clamped from below (silu(-20) is used)");
    runner.Check(std::fabs(negative - silu_minus_10 * -5.0F) >
                     std::fabs(negative - silu_minus_20 * -5.0F),
                 "and a symmetric clamp would give a measurably different answer");
    runner.Check(clamped > 0.0F, "the positive branch is positive");
  }
}

// --- the assembled pipeline, when the weight file is present ---------------

void TestPipelineIfPresent(emu_test::Runner& runner) {
  runner.Section("the assembled pipeline on real weights, if fetched");
  const char* override_dir = std::getenv("DSV4_SEMI_E2E_DIR");
  const std::string directory =
      (override_dir != nullptr) ? std::string(override_dir) : std::string("data/dsv4_layer0");
  const std::string weights = directory + "/dsv4_semi_e2e.safetensors";
  const std::string config = directory + "/config.json";
  std::ifstream probe(weights, std::ios::binary);
  if (!probe.good()) {
    std::printf("   SKIP: %s not present "
                "(run scripts/download_dsv4_semi_e2e_weights.sh)\n",
                weights.c_str());
    return;
  }
  probe.close();

  SemiE2ePipeline pipeline(weights, config);
  SemiE2eOptions options;
  // The token id has to match what the file was fetched for; read it back
  // from the metadata rather than hard-coding it.
  const auto& metadata = pipeline.weights().metadata();
  const auto token = metadata.find("token_id");
  runner.Check(token != metadata.end(), "the weight file records its token id");
  if (token != metadata.end()) {
    options.token_id = std::stoll(token->second);
  }
  options.decode_steps = 4;
  options.indexer_seq_len = 512;
  options.warmup_window_tokens = 4;

  std::vector<std::uint16_t> x_next;
  const SemiE2eReport report = pipeline.Run(options, &x_next);

  runner.Check(x_next.size() == static_cast<std::size_t>(kHcStreams) * kHiddenSize,
               "x_next is [1, 4, 4096]");
  runner.Check(report.all_finite, "every activation is finite");
  runner.Check(report.x_after_layer0_l2 > 0.0F, "layer 0 produced a non-zero bundle");
  runner.Check(report.x_after_layer2_l2 > 0.0F, "layer 2 produced a non-zero bundle");
  runner.Check(report.x_after_moe_l2 > 0.0F, "the MoE layer produced a non-zero bundle");
  runner.Check(report.stages.size() >= 6, "every stage reported");

  // The compressor cadence: 4 steps at ratio 4 is exactly one EMIT.
  runner.Check(report.compressed_tokens_emitted == 1,
               "4 decode steps emit 1 compressed token");
  runner.Check(report.indexer_selected == std::min<int>(kIndexerTopK, 512),
               "the indexer filled its index list");

  // Routing against the real table.
  runner.Check(report.routing.count == kNumExpertsPerTok, "six experts routed");
  runner.Check(report.routing.from_hash_table,
               "layer 1 is a hash layer, so the table selected");
  runner.Check(report.routing_experts_all_present,
               "and every selected expert is present in the weight file");
  const auto routed = metadata.find("routed_experts");
  if (routed != metadata.end()) {
    // The router must agree with what the fetch script read out of tid2eid.
    std::string rendered;
    for (int i = 0; i < report.routing.count; ++i) {
      if (i != 0) {
        rendered += ",";
      }
      rendered += std::to_string(report.routing.experts[i]);
    }
    runner.Check(rendered == routed->second,
                 "the router reproduces the fetch script's tid2eid row (" + rendered +
                     " vs " + routed->second + ")");
  }
  float weight_total = 0.0F;
  for (int i = 0; i < report.routing.count; ++i) {
    weight_total += report.routing.weights[i];
  }
  runner.CheckNear(weight_total, 1.0, 1e-5, "the routed weights sum to 1");

  // Attention must actually have attended to the primed window.
  runner.Check(report.layer0_attention.attended_tokens == options.warmup_window_tokens + 1,
               "layer 0 attended the whole primed window");
  runner.Check(report.layer0_attention.mean_sink_weight >= 0.0F &&
                   report.layer0_attention.mean_sink_weight <= 1.0F,
               "the attention sink absorbed a valid fraction of the softmax mass");
  runner.Check(report.layer0_attention.q_lora_inv_rms > 0.0F,
               "q_lora was normalized");

  // Memory: the whole point of the per-row dequantization.
  if (report.memory.available) {
    const double peak_mb =
        static_cast<double>(report.memory.peak_rss_bytes) / 1048576.0;
    std::printf("   peak RSS %.1f MB, wall %.1f ms\n", peak_mb,
                report.total_microseconds / 1000.0);
    runner.Check(peak_mb < 4096.0, "peak RSS stays under the 4 GB budget");
  }

  // A token id the file was not fetched for must be refused, not silently
  // routed to absent experts.
  SemiE2eOptions wrong = options;
  wrong.token_id = options.token_id + 1;
  bool threw = false;
  try {
    std::vector<std::uint16_t> ignored;
    pipeline.Run(wrong, &ignored);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  runner.Check(threw, "a token id the file was not fetched for is refused");
}

}  // namespace

int main() {
  emu_test::Runner runner("test_semi_e2e_pipeline");
  try {
    TestFp4Decode(runner);
    TestFp8BlockGemv(runner);
    TestMxfp4Gemv(runner);
    TestRmsNorm(runner);
    TestWindowCache(runner);
    TestSqrtSoftplus(runner);
    TestHashRouter(runner);
    TestScoreRouter(runner);
    TestClampedSwiglu(runner);
    TestPipelineIfPresent(runner);
  } catch (const std::exception& error) {
    std::printf("   FATAL: unexpected exception: %s\n", error.what());
    return 1;
  }
  return runner.Summary();
}
