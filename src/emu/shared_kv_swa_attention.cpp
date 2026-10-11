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

// Shared-KV sliding-window attention, decode step, on the host CPU.
//
// Ported from DeepseekV4Attention in vllm-ascend
// vllm_ascend/models/deepseek_v4.py. Every module and extent below is taken
// from that __init__ and confirmed against the shard header:
//
//   wq_a   ReplicatedLinear(dim=4096 -> q_lora_rank=1024)      [1024, 4096]
//   q_norm RMSNorm(1024)
//   wq_b   Linear(1024 -> n_heads * head_dim = 32768)          [32768, 1024]
//   wkv    ReplicatedLinear(4096 -> head_dim = 512)            [512, 4096]
//   kv_norm RMSNorm(512)
//   wo_a   Linear(n_heads*head_dim/n_groups=4096 -> n_groups*o_lora_rank=8192)
//   wo_b   Linear(8192 -> 4096)
//   attn_sink  Parameter(n_heads) FP32
//   scale  = head_dim ** -0.5
//   window = config.sliding_window = 128
//
// ONE SHARED KV ROW, NOT ONE PER HEAD
//   wkv maps 4096 -> 512, which is head_dim -- NOT n_heads * head_dim. With
//   "num_key_value_heads": 1, all 64 query heads score against the SAME
//   512-wide key row per token, and that same row is the value. This is what
//   the vendored operator kv_quant_sparse_attn_sharedkv is named for, and it
//   is why the window cache is 128 x 512 floats (256 KB) instead of 64 times
//   that. Allocating per-head keys would be a 64x memory error that still
//   produces finite output.
//
// WHAT THE ATTENTION SINK DOES
//   attn_sink is one learned FP32 logit per head. It enters the softmax as an
//   extra column with no value vector, so the weights over real tokens sum to
//   less than one and a head can decline to attend. (This is the standard
//   "off-by-one"/sink softmax; the exact placement lives inside the fused NPU
//   operator rather than in the Python, so it is implemented here in that
//   standard form and the absorbed mass is REPORTED in AttentionStats rather
//   than silently discarded -- a sink weight near 1.0 across heads is the
//   signal that the window or the position is wrong.)
//
// RoPE
//   Applied to the LAST rope_head_dim = 64 lanes of each 512-wide q head and
//   of the 512-wide kv row, half mode -- the same convention and the same
//   helper the compressor uses (nope_head_dim = 512 - 64 = 448 lanes pass
//   through). The caller supplies the tables for `position`.

#include "moe/emu/dsv4_layers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("shared_kv_swa_attention: " + message);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// the window cache
// ---------------------------------------------------------------------------

SharedKvWindowCache::SharedKvWindowCache(int window, int head_dim)
    : window_(window), head_dim_(head_dim) {
  Require(window > 0, "window must be positive");
  Require(head_dim > 0, "head_dim must be positive");
  storage_.assign(static_cast<std::size_t>(window) * head_dim, 0.0F);
}

int SharedKvWindowCache::Append(const float* kv_row, int position) {
  Require(kv_row != nullptr, "kv row is null");
  Require(position >= 0, "position must be non-negative");
  Require(position > newest_position_,
          "positions must advance: a decode step cannot rewrite history");
  // A ring indexed by position % window: appending position p evicts p -
  // window, which is exactly the token leaving the sliding window.
  float* slot = storage_.data() + static_cast<std::size_t>(position % window_) * head_dim_;
  std::copy_n(kv_row, head_dim_, slot);
  newest_position_ = position;
  live_ = std::min(window_, position + 1);
  return live_;
}

const float* SharedKvWindowCache::At(int position) const {
  if (position < 0 || position > newest_position_) {
    return nullptr;
  }
  if (position <= newest_position_ - window_) {
    return nullptr;  // evicted
  }
  return storage_.data() + static_cast<std::size_t>(position % window_) * head_dim_;
}

int SharedKvWindowCache::oldest_position() const {
  if (newest_position_ < 0) {
    return -1;
  }
  return std::max(0, newest_position_ - window_ + 1);
}

// ---------------------------------------------------------------------------
// the decode step
// ---------------------------------------------------------------------------

void shared_kv_swa_prefill_kv_cpu(const float* h_in,
                                  const SharedKvAttentionWeights& weights,
                                  SharedKvWindowCache* cache, int position,
                                  const float* rope_cos, const float* rope_sin) {
  Require(h_in != nullptr, "h_in is null");
  Require(cache != nullptr, "cache is null");
  Require(weights.kv_norm != nullptr, "kv_norm is null");
  Require(cache->head_dim() == kAttnHeadDim, "cache head_dim must be 512");
  std::vector<float> kv(static_cast<std::size_t>(kAttnHeadDim));
  fp8_block_gemv_cpu(weights.wkv, h_in, kv.data());
  rms_norm_cpu(kv.data(), weights.kv_norm, kv.data(), kAttnHeadDim, kRmsNormEpsilon);
  compressor_apply_rope(kv.data(), rope_cos, rope_sin, kAttnHeadDim, kAttnRopeHeadDim,
                        RotaryMode::kHalf);
  cache->Append(kv.data(), position);
}

void shared_kv_swa_attention_decode_cpu(const float* h_in,
                                        const SharedKvAttentionWeights& weights,
                                        SharedKvWindowCache* cache, int position,
                                        const float* rope_cos, const float* rope_sin,
                                        float* out, AttentionStats* stats) {
  Require(h_in != nullptr && out != nullptr, "h_in / out is null");
  Require(cache != nullptr, "cache is null");
  Require(position >= 0, "position must be non-negative");
  Require(weights.q_norm != nullptr && weights.kv_norm != nullptr,
          "q_norm / kv_norm is null");
  Require(weights.attn_sink != nullptr, "attn_sink is null");
  Require(cache->head_dim() == kAttnHeadDim, "cache head_dim must be 512");

  const int head_dim = kAttnHeadDim;
  const int heads = kAttnHeads;

  AttentionStats local;
  AttentionStats* report = (stats != nullptr) ? stats : &local;
  *report = AttentionStats{};

  // --- query: wq_a -> q_norm -> wq_b --------------------------------------
  std::vector<float> q_lora(static_cast<std::size_t>(kQLoraRank));
  fp8_block_gemv_cpu(weights.wq_a, h_in, q_lora.data());
  std::vector<float> q_lora_normed(static_cast<std::size_t>(kQLoraRank));
  report->q_lora_inv_rms = rms_norm_cpu(q_lora.data(), weights.q_norm,
                                        q_lora_normed.data(), kQLoraRank,
                                        kRmsNormEpsilon);

  std::vector<float> q(static_cast<std::size_t>(heads) * head_dim);
  fp8_block_gemv_cpu(weights.wq_b, q_lora_normed.data(), q.data());

  // Per head: the weightless RMSNorm (q_norm_without_weight), then RoPE over
  // the last 64 lanes.
  for (int head = 0; head < heads; ++head) {
    float* row = q.data() + static_cast<std::size_t>(head) * head_dim;
    rms_norm_cpu(row, nullptr, row, head_dim, kRmsNormEpsilon);
    compressor_apply_rope(row, rope_cos, rope_sin, head_dim, kAttnRopeHeadDim,
                          RotaryMode::kHalf);
  }

  // --- shared key/value: wkv -> kv_norm -> RoPE ---------------------------
  std::vector<float> kv(static_cast<std::size_t>(head_dim));
  fp8_block_gemv_cpu(weights.wkv, h_in, kv.data());
  report->kv_inv_rms =
      rms_norm_cpu(kv.data(), weights.kv_norm, kv.data(), head_dim, kRmsNormEpsilon);
  compressor_apply_rope(kv.data(), rope_cos, rope_sin, head_dim, kAttnRopeHeadDim,
                        RotaryMode::kHalf);
  cache->Append(kv.data(), position);

  // --- the attended window -------------------------------------------------
  const int oldest = cache->oldest_position();
  const int attended = position - oldest + 1;
  report->attended_tokens = attended;

  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  // context[head] accumulates the weighted shared value rows.
  std::vector<float> context(static_cast<std::size_t>(heads) * head_dim, 0.0F);
  std::vector<float> logits(static_cast<std::size_t>(attended));
  double sink_total = 0.0;
  float max_logit = -std::numeric_limits<float>::infinity();

  for (int head = 0; head < heads; ++head) {
    const float* query = q.data() + static_cast<std::size_t>(head) * head_dim;

    float head_max = weights.attn_sink[head];  // the sink competes in the max
    for (int offset = 0; offset < attended; ++offset) {
      const float* key = cache->At(oldest + offset);
      Require(key != nullptr, "a position inside the window is missing from the cache");
      float dot = 0.0F;
      for (int lane = 0; lane < head_dim; ++lane) {
        dot += query[lane] * key[lane];
      }
      const float logit = dot * scale;
      logits[static_cast<std::size_t>(offset)] = logit;
      head_max = std::max(head_max, logit);
    }
    max_logit = std::max(max_logit, head_max);

    // Softmax over [logits..., attn_sink[head]]. The sink contributes to the
    // denominator only -- it has no value row -- so the attended weights sum
    // to 1 - sink_weight.
    float denominator = std::exp(weights.attn_sink[head] - head_max);
    const float sink_numerator = denominator;
    for (int offset = 0; offset < attended; ++offset) {
      logits[static_cast<std::size_t>(offset)] =
          std::exp(logits[static_cast<std::size_t>(offset)] - head_max);
      denominator += logits[static_cast<std::size_t>(offset)];
    }
    const float inverse = 1.0F / denominator;
    sink_total += static_cast<double>(sink_numerator * inverse);

    float* accumulator = context.data() + static_cast<std::size_t>(head) * head_dim;
    for (int offset = 0; offset < attended; ++offset) {
      const float weight = logits[static_cast<std::size_t>(offset)] * inverse;
      // The key IS the value: one shared KV row per token.
      const float* value = cache->At(oldest + offset);
      for (int lane = 0; lane < head_dim; ++lane) {
        accumulator[lane] += weight * value[lane];
      }
    }
  }
  report->mean_sink_weight = static_cast<float>(sink_total / heads);
  report->max_logit = max_logit;

  // --- output: grouped wo_a, then wo_b ------------------------------------
  // wo_a is [8192, 4096] but acts as n_groups = 8 independent
  // [o_lora_rank=1024, 4096] blocks: group g consumes context lanes
  // [g*4096, (g+1)*4096) of the flattened 64x512 = 32768 context and produces
  // o_lora[g*1024, (g+1)*1024). Treating it as one dense [8192, 4096] GEMV
  // over the first 4096 context lanes would silently drop seven eighths of
  // the attention output.
  Require(weights.wo_a.rows == kWoAOutput, "wo_a must have 8192 rows");
  Require(weights.wo_a.cols == kWoAGroupInput, "wo_a must have 4096 columns");
  Require(weights.wo_b.rows == kHiddenSize, "wo_b must have 4096 rows");
  Require(weights.wo_b.cols == kWoAOutput, "wo_b must have 8192 columns");

  std::vector<float> o_lora(static_cast<std::size_t>(kWoAOutput));
  for (int group = 0; group < kOGroups; ++group) {
    Fp8BlockMatrix block = weights.wo_a;
    block.rows = kOLoraRank;
    // Advance into wo_a's rows and into its scale grid by the same group.
    block.weight = weights.wo_a.weight +
                   static_cast<std::size_t>(group) * kOLoraRank * weights.wo_a.cols;
    block.scale = weights.wo_a.scale +
                  static_cast<std::size_t>(group) * (kOLoraRank / kFp8BlockSize) *
                      weights.wo_a.scale_cols;
    fp8_block_gemv_cpu(block,
                       context.data() + static_cast<std::size_t>(group) * kWoAGroupInput,
                       o_lora.data() + static_cast<std::size_t>(group) * kOLoraRank);
  }
  fp8_block_gemv_cpu(weights.wo_b, o_lora.data(), out);
}

}  // namespace moe::emu
