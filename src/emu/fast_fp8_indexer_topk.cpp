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

// Fast FP8 lightning indexer scoring + top-k, on the host CPU.
//
// Ported from ops-transformer attention/quant_lightning_indexer/tests/pytest/
// quant_lightning_indexer_golden.py :: GeneralizedQLI::cal_atten_per_batch_fp8.
//
// THE SCORE
//     logit[j] = k_scale[j] * sum_h  qw[h] * relu( q[h] . k[j] )
//
//   The golden computes it in this order and the order matters:
//
//     qk_bmm_res  = bmm(q, k^T)              per head, FP32
//     qk_relu_out = clamp_min(qk_bmm_res, 0) PER HEAD, before weighting
//     brc_vmul    = bmm(w^T, qk_relu_out)    the weighted head sum
//     reduce_sum  = brc_vmul * k_scale       per key
//
//   `relu` sits between the per-head dot product and the head reduction. A
//   kernel that sums the heads first and then applies the ReLU computes a
//   different, strictly larger function and will still look plausible --
//   negative head contributions get clipped collectively instead of
//   individually. The weights are also folded as `cur_w = cur_wt *
//   cur_q_scale` before the reduction, which is why this API takes one fused
//   per-head scale rather than separate weights and scales.
//
// SELECTION, AND WHY THE TIE-BREAK IS THE CONTRACT
//   The golden rounds the FP32 scores to BF16 (`to_be_sort_ele.to(
//   torch.bfloat16)`) and only then sorts, stably, on the pair
//   (-value, index). BF16 keeps 8 mantissa bits, so over a few thousand keys
//   exact ties are routine, and "stable sort on -value" resolves every tie
//   toward the lower index. Reproducing the rounding but not the tie-break
//   gives a set that differs from the device's by a handful of indices --
//   which is exactly the kind of mismatch that is impossible to attribute
//   later. Both are reproduced here.
//
//   The golden pays a full O(N log N) sort. This uses std::nth_element to
//   partition at top_k and then sorts only the winners: O(N) expected plus
//   O(k log k). Because the comparator below is a strict total order (no two
//   distinct keys compare equivalent -- the index breaks every tie), the
//   partition point is unambiguous and the result is identical to the
//   golden's, not merely equivalent up to ties.

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("fast_fp8_indexer_topk: " + message);
  }
}

// Resolve key `j` through the block table into a pointer at its 128 FP8
// lanes. PA_BSND is [num_blocks, block_size, num_kv_heads, head_dim] with
// num_kv_heads == 1 for DSV4 ("num_key_value_heads": 1), so the row stride is
// head_dim and the block stride is block_size * head_dim.
const std::uint8_t* KeyRow(const std::uint8_t* key_cache, const std::int32_t* block_tables,
                           std::int32_t j, std::int32_t block_size, int head_dim) {
  const std::int32_t logical_block = j / block_size;
  const std::int32_t within_block = j % block_size;
  const std::int32_t physical_block =
      (block_tables == nullptr) ? logical_block : block_tables[logical_block];
  const std::size_t offset =
      (static_cast<std::size_t>(physical_block) * static_cast<std::size_t>(block_size) +
       static_cast<std::size_t>(within_block)) *
      static_cast<std::size_t>(head_dim);
  return key_cache + offset;
}

// A score paired with its key index. The comparator is a strict total order:
// higher score first, lower index first on a tie.
struct ScoredKey {
  float score = 0.0F;  // already rounded through BF16
  std::int32_t index = 0;
};

bool Beats(const ScoredKey& a, const ScoredKey& b) noexcept {
  if (a.score != b.score) {
    return a.score > b.score;
  }
  return a.index < b.index;
}

}  // namespace

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

void fast_fp8_indexer_scores_cpu(
    const std::uint8_t* query_fp8,
    const std::uint8_t* key_cache_fp8,
    const float* dequant_scales,
    const std::int32_t* block_tables,
    std::int32_t seq_len,
    float* out_scores,
    std::int32_t block_size,
    int heads,
    int head_dim) {
  Require(query_fp8 != nullptr, "query_fp8 is null");
  Require(key_cache_fp8 != nullptr, "key_cache_fp8 is null");
  Require(dequant_scales != nullptr, "dequant_scales is null");
  Require(out_scores != nullptr, "out_scores is null");
  Require(seq_len >= 0, "seq_len must be non-negative");
  Require(block_size > 0, "block_size must be positive");
  Require(heads > 0 && head_dim > 0, "heads and head_dim must be positive");

  const Fp8Table& table = Fp8E4M3Table();

  // Dequantize the 64x128 query once. 8192 FP8 bytes -> 32 KB of FP32, which
  // is reused for every key, so the per-key inner loop is a pure FP32 dot
  // product over cached data. Dequantizing inside the key loop would redo
  // this seq_len times.
  std::vector<float> query(static_cast<std::size_t>(heads) * head_dim);
  for (std::size_t i = 0; i < query.size(); ++i) {
    query[i] = table[query_fp8[i]];
  }

  // dequant_scales = [qw(heads) | k_scale(seq_len)], see cpu_kernels.h.
  const float* head_weights = dequant_scales;
  const float* key_scales = dequant_scales + heads;

  std::vector<float> key(static_cast<std::size_t>(head_dim));
  for (std::int32_t j = 0; j < seq_len; ++j) {
    const std::uint8_t* key_row =
        KeyRow(key_cache_fp8, block_tables, j, block_size, head_dim);
    for (int d = 0; d < head_dim; ++d) {
      key[static_cast<std::size_t>(d)] = table[key_row[d]];
    }

    float accumulator = 0.0F;
    for (int h = 0; h < heads; ++h) {
      const float* query_head = query.data() + static_cast<std::size_t>(h) * head_dim;
      float dot = 0.0F;
      for (int d = 0; d < head_dim; ++d) {
        dot += query_head[d] * key[static_cast<std::size_t>(d)];
      }
      // relu per head, BEFORE the weighted reduction. See the file header.
      if (dot > 0.0F) {
        accumulator += head_weights[h] * dot;
      }
    }
    out_scores[j] = accumulator * key_scales[j];
  }
}

// ---------------------------------------------------------------------------
// scoring + selection
// ---------------------------------------------------------------------------

void fast_fp8_indexer_topk_cpu(
    const std::uint8_t* query_fp8,
    const std::uint8_t* key_cache_fp8,
    const float* dequant_scales,
    const std::int32_t* block_tables,
    std::int32_t seq_len,
    std::int32_t top_k,
    std::int32_t* out_indices,
    std::int32_t block_size,
    int heads,
    int head_dim) {
  Require(top_k >= 0, "top_k must be non-negative");
  // top_k == 0 is checked BEFORE the output pointer: a caller selecting
  // nothing has no buffer to hand over, so requiring one would make the
  // degenerate case an error instead of the no-op it should be.
  if (top_k == 0) {
    return;
  }
  Require(out_indices != nullptr, "out_indices is null");

  // -1 is the golden's fill value for an unselected slot
  // (torch.full(shape, -1, dtype=int32)), so a short sequence is reported as
  // padding rather than as index 0 repeated.
  std::fill_n(out_indices, top_k, -1);
  if (seq_len <= 0) {
    return;
  }

  std::vector<float> scores(static_cast<std::size_t>(seq_len));
  fast_fp8_indexer_scores_cpu(query_fp8, key_cache_fp8, dequant_scales, block_tables,
                              seq_len, scores.data(), block_size, heads, head_dim);

  // Round through BF16 before comparing, as the golden does.
  std::vector<ScoredKey> ranked(static_cast<std::size_t>(seq_len));
  for (std::int32_t j = 0; j < seq_len; ++j) {
    ranked[static_cast<std::size_t>(j)] = ScoredKey{
        Bf16ToFloat(FloatToBf16(scores[static_cast<std::size_t>(j)])), j};
  }

  const std::int32_t selected = std::min(top_k, seq_len);
  if (selected < seq_len) {
    std::nth_element(ranked.begin(), ranked.begin() + selected, ranked.end(), Beats);
  }
  // The winners still have to come out in rank order: the sparse-attention
  // consumer reads them as a ranked list, not as a set.
  std::sort(ranked.begin(), ranked.begin() + selected, Beats);

  for (std::int32_t i = 0; i < selected; ++i) {
    out_indices[i] = ranked[static_cast<std::size_t>(i)].index;
  }
}

}  // namespace moe::emu
