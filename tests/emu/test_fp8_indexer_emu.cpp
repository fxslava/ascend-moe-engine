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

// Fast FP8 lightning indexer: FP8 round trip, scoring, and top-512 selection
// against a naive full-sort baseline.
//
// The baseline below is deliberately the slow, obvious implementation --
// dequantize everything, score everything, sort everything -- which is what
// quant_lightning_indexer_golden.py does. The kernel under test replaces the
// sort with nth_element; the assertion is that the two agree as an ORDERED
// LIST, not merely as a set, because the sparse-attention consumer reads the
// indices in rank order.

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"

#include "emu_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using namespace moe::emu;

constexpr int kHeads = kIndexerHeads;      // 64
constexpr int kHeadDim = kIndexerHeadDim;  // 128
constexpr std::int32_t kBlockSize = 128;

// --- FP8 E4M3 ---------------------------------------------------------------

void TestFp8E4M3(emu_test::Runner& runner) {
  runner.Section("FP8 E4M3FN: the exact value set, including the top binade");

  // The encodings that make E4M3FN different from an IEEE-shaped format.
  runner.CheckNear(DecodeFp8E4M3(0x00), 0.0, 0.0, "0x00 is +0");
  runner.Check(std::signbit(DecodeFp8E4M3(0x80)), "0x80 is -0");
  runner.CheckNear(DecodeFp8E4M3(0x38), 1.0, 0.0, "0x38 is 1.0");
  runner.CheckNear(DecodeFp8E4M3(0xB8), -1.0, 0.0, "0xB8 is -1.0");
  // exp == 15 is NOT inf/NaN in E4M3FN except for mantissa == 7.
  runner.CheckNear(DecodeFp8E4M3(0x7E), 448.0, 0.0,
                   "0x7E is 448, the format maximum -- NOT an infinity");
  runner.CheckNear(DecodeFp8E4M3(0x78), 256.0, 0.0, "0x78 is 256");
  runner.Check(std::isnan(DecodeFp8E4M3(0x7F)), "0x7F is the only positive NaN");
  runner.Check(std::isnan(DecodeFp8E4M3(0xFF)), "0xFF is the only negative NaN");
  for (std::uint32_t code = 0; code <= 0x7EU; ++code) {
    runner.Check(std::isfinite(DecodeFp8E4M3(static_cast<std::uint8_t>(code))),
                 "code " + std::to_string(code) + " is finite");
  }
  // Smallest subnormal: 1/8 * 2^-6 = 2^-9.
  runner.CheckNear(DecodeFp8E4M3(0x01), std::ldexp(1.0, -9), 0.0,
                   "0x01 is the smallest subnormal, 2^-9");
  // Smallest normal: 2^-6.
  runner.CheckNear(DecodeFp8E4M3(0x08), std::ldexp(1.0, -6), 0.0,
                   "0x08 is the smallest normal, 2^-6");

  // Monotonicity across the positive codes: the selection logic relies on the
  // table being ordered.
  for (std::uint32_t code = 1; code <= 0x7EU; ++code) {
    runner.Check(DecodeFp8E4M3(static_cast<std::uint8_t>(code)) >
                     DecodeFp8E4M3(static_cast<std::uint8_t>(code - 1)),
                 "code " + std::to_string(code) + " is strictly larger than " +
                     std::to_string(code - 1));
  }

  // Round trip: every finite code must survive encode(decode(code)).
  for (std::uint32_t code = 0; code <= 0xFFU; ++code) {
    const float value = DecodeFp8E4M3(static_cast<std::uint8_t>(code));
    if (std::isnan(value)) {
      continue;
    }
    if (value == 0.0F) {
      continue;  // +0/-0 both re-encode to a zero; sign is checked above
    }
    runner.Check(EncodeFp8E4M3(value) == static_cast<std::uint8_t>(code),
                 "encode(decode(" + std::to_string(code) + ")) round-trips");
  }
  // Saturation rather than overflow to NaN.
  runner.Check(EncodeFp8E4M3(1.0e6F) == 0x7EU, "a huge value saturates to 448");
  runner.Check(EncodeFp8E4M3(-1.0e6F) == 0xFEU, "a huge negative saturates to -448");

  // The table and the scalar decoder must not drift apart.
  const Fp8Table& table = Fp8E4M3Table();
  for (std::uint32_t code = 0; code <= 0xFFU; ++code) {
    const float direct = DecodeFp8E4M3(static_cast<std::uint8_t>(code));
    if (std::isnan(direct)) {
      runner.Check(std::isnan(table[code]), "table NaN agrees at " + std::to_string(code));
    } else {
      runner.CheckNear(table[code], direct, 0.0,
                       "table agrees with the decoder at " + std::to_string(code));
    }
  }
}

void TestFp8E5M2AndE8M0(emu_test::Runner& runner) {
  runner.Section("FP8 E5M2 (IEEE-shaped) and E8M0 micro-scales");

  runner.CheckNear(DecodeFp8E5M2(0x3C), 1.0, 0.0, "E5M2 0x3C is 1.0");
  runner.CheckNear(DecodeFp8E5M2(0x7B), 57344.0, 0.0, "E5M2 0x7B is the max, 57344");
  // Unlike E4M3FN, exp == 31 really is inf/NaN here.
  runner.Check(std::isinf(DecodeFp8E5M2(0x7C)) && DecodeFp8E5M2(0x7C) > 0,
               "E5M2 0x7C is +inf");
  runner.Check(std::isinf(DecodeFp8E5M2(0xFC)) && DecodeFp8E5M2(0xFC) < 0,
               "E5M2 0xFC is -inf");
  runner.Check(std::isnan(DecodeFp8E5M2(0x7D)), "E5M2 0x7D is NaN");
  runner.CheckNear(DecodeFp8E5M2(0x01), std::ldexp(1.0, -16), 0.0,
                   "E5M2 0x01 is the smallest subnormal, 2^-16");
  for (std::uint32_t code = 0; code <= 0xFFU; ++code) {
    const float value = DecodeFp8E5M2(static_cast<std::uint8_t>(code));
    if (!std::isfinite(value) || value == 0.0F) {
      continue;
    }
    runner.Check(EncodeFp8E5M2(value) == static_cast<std::uint8_t>(code),
                 "E5M2 encode(decode(" + std::to_string(code) + ")) round-trips");
  }

  // E8M0 is the "scale_fmt": "ue8m0" type: value = 2^(code - 127).
  runner.CheckNear(DecodeFp8E8M0(127), 1.0, 0.0, "E8M0 127 is 2^0");
  runner.CheckNear(DecodeFp8E8M0(128), 2.0, 0.0, "E8M0 128 is 2^1");
  runner.CheckNear(DecodeFp8E8M0(126), 0.5, 0.0, "E8M0 126 is 2^-1");
  runner.CheckNear(DecodeFp8E8M0(0), std::ldexp(1.0, -127), 0.0, "E8M0 0 is 2^-127");
  runner.Check(std::isnan(DecodeFp8E8M0(255)),
               "E8M0 255 is the NaN code, not 2^128");
  for (std::uint32_t code = 0; code < 255U; ++code) {
    runner.Check(DecodeFp8E8M0(static_cast<std::uint8_t>(code)) > 0.0F,
                 "E8M0 " + std::to_string(code) + " is positive");
  }
}

// --- the naive baseline -----------------------------------------------------

struct Fixture {
  std::vector<std::uint8_t> query;       // [heads, head_dim]
  std::vector<std::uint8_t> key_cache;   // [blocks, block_size, 1, head_dim]
  std::vector<float> scales;             // [heads + seq_len]
  std::vector<std::int32_t> block_table;
  std::int32_t seq_len = 0;
};

Fixture MakeFixture(std::int32_t seq_len, bool shuffle_blocks, bool coarse_scores) {
  Fixture f;
  f.seq_len = seq_len;
  const std::int32_t blocks = (seq_len + kBlockSize - 1) / kBlockSize;

  f.query.resize(static_cast<std::size_t>(kHeads) * kHeadDim);
  for (auto& byte : f.query) {
    byte = EncodeFp8E4M3(emu_test::UniformFloat(-2.0F, 2.0F));
  }

  // The physical cache is deliberately larger than the sequence, and the
  // block table is permuted when asked, so a kernel that ignores the table
  // and reads blocks in logical order gets the wrong keys.
  const std::int32_t physical_blocks = blocks + 3;
  f.key_cache.assign(static_cast<std::size_t>(physical_blocks) * kBlockSize * kHeadDim,
                     0);
  for (auto& byte : f.key_cache) {
    byte = EncodeFp8E4M3(emu_test::UniformFloat(-2.0F, 2.0F));
  }

  f.block_table.resize(static_cast<std::size_t>(blocks));
  std::iota(f.block_table.begin(), f.block_table.end(), 0);
  if (shuffle_blocks) {
    // A fixed, non-identity permutation of the available physical blocks.
    for (std::int32_t i = 0; i < blocks; ++i) {
      f.block_table[static_cast<std::size_t>(i)] = (blocks + 2 - i) % physical_blocks;
    }
  }

  f.scales.resize(static_cast<std::size_t>(kHeads) + seq_len);
  for (int h = 0; h < kHeads; ++h) {
    f.scales[static_cast<std::size_t>(h)] = emu_test::UniformFloat(0.1F, 1.0F);
  }
  for (std::int32_t j = 0; j < seq_len; ++j) {
    // `coarse_scores` quantizes the key scales hard, which forces many exact
    // BF16 ties and makes the tie-break path the one under test.
    const float scale = coarse_scores
                            ? (1.0F + static_cast<float>(j % 4))
                            : emu_test::UniformFloat(0.5F, 2.0F);
    f.scales[static_cast<std::size_t>(kHeads + j)] = scale;
  }
  return f;
}

// relu-per-head, weighted sum, key scale -- written out longhand.
std::vector<float> BaselineScores(const Fixture& f) {
  const std::int32_t blocks_per_seq = static_cast<std::int32_t>(f.block_table.size());
  (void)blocks_per_seq;
  std::vector<float> scores(static_cast<std::size_t>(f.seq_len), 0.0F);
  for (std::int32_t j = 0; j < f.seq_len; ++j) {
    const std::int32_t logical = j / kBlockSize;
    const std::int32_t physical = f.block_table[static_cast<std::size_t>(logical)];
    const std::size_t base =
        (static_cast<std::size_t>(physical) * kBlockSize +
         static_cast<std::size_t>(j % kBlockSize)) *
        static_cast<std::size_t>(kHeadDim);
    float total = 0.0F;
    for (int h = 0; h < kHeads; ++h) {
      float dot = 0.0F;
      for (int d = 0; d < kHeadDim; ++d) {
        dot += DecodeFp8E4M3(f.query[static_cast<std::size_t>(h) * kHeadDim + d]) *
               DecodeFp8E4M3(f.key_cache[base + static_cast<std::size_t>(d)]);
      }
      if (dot > 0.0F) {
        total += f.scales[static_cast<std::size_t>(h)] * dot;
      }
    }
    scores[static_cast<std::size_t>(j)] =
        total * f.scales[static_cast<std::size_t>(kHeads + j)];
  }
  return scores;
}

// The golden's selection: round to BF16, then a full stable sort on
// (-value, index).
std::vector<std::int32_t> BaselineTopK(const std::vector<float>& scores,
                                       std::int32_t top_k) {
  std::vector<std::pair<float, std::int32_t>> ranked;
  ranked.reserve(scores.size());
  for (std::size_t j = 0; j < scores.size(); ++j) {
    ranked.emplace_back(Bf16ToFloat(FloatToBf16(scores[j])),
                        static_cast<std::int32_t>(j));
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::int32_t> out(static_cast<std::size_t>(top_k), -1);
  const std::size_t selected =
      std::min<std::size_t>(static_cast<std::size_t>(top_k), ranked.size());
  for (std::size_t i = 0; i < selected; ++i) {
    out[i] = ranked[i].second;
  }
  return out;
}

void TestScoringMatchesBaseline(emu_test::Runner& runner) {
  runner.Section("scoring: relu per head BEFORE the weighted reduction");
  const Fixture f = MakeFixture(600, true, false);
  std::vector<float> got(static_cast<std::size_t>(f.seq_len));
  fast_fp8_indexer_scores_cpu(f.query.data(), f.key_cache.data(), f.scales.data(),
                              f.block_table.data(), f.seq_len, got.data(), kBlockSize,
                              kHeads, kHeadDim);
  const std::vector<float> want = BaselineScores(f);
  for (std::int32_t j = 0; j < f.seq_len; ++j) {
    runner.CheckRelative(got[static_cast<std::size_t>(j)],
                         want[static_cast<std::size_t>(j)], 1e-5,
                         "score for key " + std::to_string(j));
  }
  // Every score must be non-negative: the per-head relu guarantees it, and a
  // kernel that reduced before clamping could produce negatives.
  bool all_non_negative = true;
  for (const float value : got) {
    if (value < 0.0F) {
      all_non_negative = false;
    }
  }
  runner.Check(all_non_negative,
               "all scores are >= 0, as the per-head relu requires");

  // ORDER-OF-OPERATIONS PROOF: compute what a reduce-then-relu kernel would
  // give and show it differs. Without this, both implementations pass.
  std::vector<float> wrong_order(static_cast<std::size_t>(f.seq_len), 0.0F);
  for (std::int32_t j = 0; j < f.seq_len; ++j) {
    const std::int32_t physical =
        f.block_table[static_cast<std::size_t>(j / kBlockSize)];
    const std::size_t base =
        (static_cast<std::size_t>(physical) * kBlockSize +
         static_cast<std::size_t>(j % kBlockSize)) *
        static_cast<std::size_t>(kHeadDim);
    float total = 0.0F;
    for (int h = 0; h < kHeads; ++h) {
      float dot = 0.0F;
      for (int d = 0; d < kHeadDim; ++d) {
        dot += DecodeFp8E4M3(f.query[static_cast<std::size_t>(h) * kHeadDim + d]) *
               DecodeFp8E4M3(f.key_cache[base + static_cast<std::size_t>(d)]);
      }
      total += f.scales[static_cast<std::size_t>(h)] * dot;  // no per-head relu
    }
    wrong_order[static_cast<std::size_t>(j)] =
        std::max(0.0F, total) * f.scales[static_cast<std::size_t>(kHeads + j)];
  }
  std::size_t differing = 0;
  for (std::int32_t j = 0; j < f.seq_len; ++j) {
    if (std::fabs(wrong_order[static_cast<std::size_t>(j)] -
                  got[static_cast<std::size_t>(j)]) > 1e-3F) {
      ++differing;
    }
  }
  runner.Check(differing > static_cast<std::size_t>(f.seq_len) / 2,
               "a reduce-then-relu kernel disagrees on most keys, so the "
               "per-head clamp is genuinely being exercised");
}

void TestTopKMatchesFullSort(emu_test::Runner& runner) {
  runner.Section("top-512 selection vs a naive full-sort baseline");

  struct Scenario {
    const char* name;
    std::int32_t seq_len;
    bool shuffle;
    bool coarse;
  };
  const Scenario scenarios[] = {
      {"seq 4096, permuted blocks", 4096, true, false},
      {"seq 1000, identity blocks", 1000, false, false},
      {"seq 2048, heavy BF16 ties", 2048, true, true},
      {"seq 513, one past top_k", 513, true, false},
      {"seq 512, exactly top_k", 512, false, false},
      {"seq 200, shorter than top_k", 200, true, false},
      {"seq 1, single key", 1, false, false},
  };

  for (const Scenario& scenario : scenarios) {
    const Fixture f = MakeFixture(scenario.seq_len, scenario.shuffle, scenario.coarse);
    std::vector<std::int32_t> got(static_cast<std::size_t>(kIndexerTopK), 0);
    fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(), f.scales.data(),
                              f.block_table.data(), f.seq_len, kIndexerTopK, got.data(),
                              kBlockSize, kHeads, kHeadDim);

    const std::vector<float> scores = BaselineScores(f);
    const std::vector<std::int32_t> want = BaselineTopK(scores, kIndexerTopK);

    // (a) identical as an ORDERED list -- the strong claim.
    std::size_t order_mismatches = 0;
    for (std::int32_t i = 0; i < kIndexerTopK; ++i) {
      if (got[static_cast<std::size_t>(i)] != want[static_cast<std::size_t>(i)]) {
        ++order_mismatches;
      }
    }
    runner.Check(order_mismatches == 0,
                 std::string(scenario.name) +
                     ": ranked index list matches the full sort exactly (" +
                     std::to_string(order_mismatches) + " mismatches)");

    // (b) 100% set intersection, as the brief asks for explicitly.
    std::set<std::int32_t> got_set(got.begin(), got.end());
    std::set<std::int32_t> want_set(want.begin(), want.end());
    runner.Check(got_set == want_set,
                 std::string(scenario.name) + ": 100% set intersection");

    // (c) structural: in range, no duplicates, -1 padding only at the tail.
    const std::int32_t expected_selected = std::min(kIndexerTopK, scenario.seq_len);
    std::set<std::int32_t> unique;
    for (std::int32_t i = 0; i < expected_selected; ++i) {
      const std::int32_t index = got[static_cast<std::size_t>(i)];
      runner.Check(index >= 0 && index < scenario.seq_len,
                   std::string(scenario.name) + ": index " + std::to_string(i) +
                       " is inside [0, seq_len)");
      unique.insert(index);
    }
    runner.Check(unique.size() == static_cast<std::size_t>(expected_selected),
                 std::string(scenario.name) + ": no duplicate indices");
    for (std::int32_t i = expected_selected; i < kIndexerTopK; ++i) {
      runner.Check(got[static_cast<std::size_t>(i)] == -1,
                   std::string(scenario.name) + ": slot " + std::to_string(i) +
                       " is -1 padding");
    }

    // (d) the selected scores really are the largest, and monotonically
    //     non-increasing in rank.
    float previous = std::numeric_limits<float>::infinity();
    for (std::int32_t i = 0; i < expected_selected; ++i) {
      const float value = Bf16ToFloat(
          FloatToBf16(scores[static_cast<std::size_t>(got[static_cast<std::size_t>(i)])]));
      runner.Check(value <= previous,
                   std::string(scenario.name) + ": rank " + std::to_string(i) +
                       " is not larger than rank " + std::to_string(i - 1));
      previous = value;
    }
    if (expected_selected < scenario.seq_len) {
      // Nothing left out may beat the weakest selected entry.
      float weakest = std::numeric_limits<float>::infinity();
      for (std::int32_t i = 0; i < expected_selected; ++i) {
        weakest = std::min(weakest, Bf16ToFloat(FloatToBf16(
                                        scores[static_cast<std::size_t>(
                                            got[static_cast<std::size_t>(i)])])));
      }
      bool rejected_any_better = false;
      for (std::int32_t j = 0; j < scenario.seq_len; ++j) {
        if (got_set.count(j) != 0) {
          continue;
        }
        if (Bf16ToFloat(FloatToBf16(scores[static_cast<std::size_t>(j)])) > weakest) {
          rejected_any_better = true;
        }
      }
      runner.Check(!rejected_any_better,
                   std::string(scenario.name) +
                       ": no rejected key outscores a selected one");
    }
  }
}

void TestTieBreakIsLowestIndex(emu_test::Runner& runner) {
  runner.Section("tie-break: equal scores select the LOWER index");
  // All keys identical -> every score is bit-identical -> the selection is
  // pure tie-break, so the answer must be 0, 1, 2, ... exactly.
  constexpr std::int32_t seq_len = 300;
  constexpr std::int32_t top_k = 64;
  Fixture f;
  f.seq_len = seq_len;
  f.query.assign(static_cast<std::size_t>(kHeads) * kHeadDim, EncodeFp8E4M3(0.5F));
  const std::int32_t blocks = (seq_len + kBlockSize - 1) / kBlockSize;
  f.key_cache.assign(static_cast<std::size_t>(blocks) * kBlockSize * kHeadDim,
                     EncodeFp8E4M3(0.25F));
  f.block_table.resize(static_cast<std::size_t>(blocks));
  std::iota(f.block_table.begin(), f.block_table.end(), 0);
  f.scales.assign(static_cast<std::size_t>(kHeads) + seq_len, 1.0F);

  std::vector<std::int32_t> got(static_cast<std::size_t>(top_k), 0);
  fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(), f.scales.data(),
                            f.block_table.data(), seq_len, top_k, got.data(), kBlockSize,
                            kHeads, kHeadDim);
  for (std::int32_t i = 0; i < top_k; ++i) {
    runner.Check(got[static_cast<std::size_t>(i)] == i,
                 "an all-ties field selects index " + std::to_string(i) + " at rank " +
                     std::to_string(i));
  }
}

void TestDegenerateInputs(emu_test::Runner& runner) {
  runner.Section("degenerate inputs");
  const Fixture f = MakeFixture(256, false, false);

  // seq_len == 0: all padding, no read of the key cache.
  {
    std::vector<std::int32_t> out(static_cast<std::size_t>(kIndexerTopK), 7);
    fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(), f.scales.data(),
                              f.block_table.data(), 0, kIndexerTopK, out.data(),
                              kBlockSize, kHeads, kHeadDim);
    bool all_padding = true;
    for (const std::int32_t value : out) {
      if (value != -1) {
        all_padding = false;
      }
    }
    runner.Check(all_padding, "seq_len = 0 yields all -1");
  }

  // top_k == 0: nothing written, nothing read.
  {
    std::vector<std::int32_t> out;
    bool threw = false;
    try {
      fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(), f.scales.data(),
                                f.block_table.data(), f.seq_len, 0, out.data(),
                                kBlockSize, kHeads, kHeadDim);
    } catch (...) {
      threw = true;
    }
    runner.Check(!threw, "top_k = 0 is a no-op, not an error");
  }

  // A null block table means identity mapping -- used by the unpaged path.
  {
    std::vector<std::int32_t> paged(static_cast<std::size_t>(kIndexerTopK), 0);
    std::vector<std::int32_t> unpaged(static_cast<std::size_t>(kIndexerTopK), 0);
    Fixture identity = f;
    std::iota(identity.block_table.begin(), identity.block_table.end(), 0);
    fast_fp8_indexer_topk_cpu(identity.query.data(), identity.key_cache.data(),
                              identity.scales.data(), identity.block_table.data(),
                              identity.seq_len, kIndexerTopK, paged.data(), kBlockSize,
                              kHeads, kHeadDim);
    fast_fp8_indexer_topk_cpu(identity.query.data(), identity.key_cache.data(),
                              identity.scales.data(), nullptr, identity.seq_len,
                              kIndexerTopK, unpaged.data(), kBlockSize, kHeads,
                              kHeadDim);
    runner.Check(paged == unpaged,
                 "a null block table behaves as the identity mapping");
  }

  // Validation.
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
  std::vector<std::int32_t> out(static_cast<std::size_t>(kIndexerTopK), 0);
  runner.Check(throws([&] {
                 fast_fp8_indexer_topk_cpu(nullptr, f.key_cache.data(), f.scales.data(),
                                           f.block_table.data(), f.seq_len,
                                           kIndexerTopK, out.data(), kBlockSize, kHeads,
                                           kHeadDim);
               }),
               "a null query is rejected");
  runner.Check(throws([&] {
                 fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(),
                                           f.scales.data(), f.block_table.data(),
                                           f.seq_len, -1, out.data(), kBlockSize,
                                           kHeads, kHeadDim);
               }),
               "a negative top_k is rejected");
  runner.Check(throws([&] {
                 fast_fp8_indexer_topk_cpu(f.query.data(), f.key_cache.data(),
                                           f.scales.data(), f.block_table.data(),
                                           f.seq_len, kIndexerTopK, out.data(), 0,
                                           kHeads, kHeadDim);
               }),
               "block_size = 0 is rejected");
}

}  // namespace

int main() {
  emu_test::Runner runner("test_fp8_indexer_emu");
  try {
    TestFp8E4M3(runner);
    TestFp8E5M2AndE8M0(runner);
    TestScoringMatchesBaseline(runner);
    TestTopKMatchesFullSort(runner);
    TestTieBreakIsLowestIndex(runner);
    TestDegenerateInputs(runner);
  } catch (const std::exception& error) {
    std::printf("   FATAL: unexpected exception: %s\n", error.what());
    return 1;
  }
  return runner.Summary();
}
