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

// The DeepSeek-V4 MoE block: routing and clamped-SwiGLU expert execution.
//
// Ported from DeepseekV4MoE and DeepseekV2MLP in vllm-ascend
// vllm_ascend/models/deepseek_v4.py, plus AscendSiluAndMulWithClamp in
// vllm_ascend/ops/activation.py.
//
// TWO ROUTERS, PICKED BY LAYER INDEX -- the thing most likely to be got wrong
//   DeepseekV4MoE.__init__ sets `self.hash = layer_idx < config.num_hash_layers`
//   and then, in the hash branch, installs `gate.tid2eid` and sets
//   `gate.e_score_correction_bias = None`. In the other branch it does the
//   reverse. So:
//
//     layers 0, 1, 2   ("num_hash_layers": 3)
//       WHICH experts  = tid2eid[token_id], a pure function of the token id
//       HOW MUCH       = the gate's scores for those six experts
//       no bias exists
//
//     layers 3 .. 42
//       WHICH + HOW MUCH = bias-shifted top-k over all 256 gate scores
//       ("topk_method": "noaux_tc")
//
//   The checkpoint corroborates it exactly: `ffn.gate.tid2eid` is present on
//   precisely 3 layers and `ffn.gate.bias` on precisely the other 40. Running
//   noaux_tc on a hash layer would select six plausible-looking experts that
//   are simply not the ones the model was trained to use for that token.
//
// THE GATE RUNS IN FP32
//   `router_logits = F.linear(hidden_states.float(), self.gate.weight)` --
//   the activation is widened before the projection, not after.
//
// SCALING ORDER
//   DeepseekV4MoE.forward normalizes the top-k weights first and applies
//   routed_scaling_factor to the ROUTED SUM afterwards, then adds the shared
//   expert:  out = routed_sum * 1.5 + shared.  The comment in the reference
//   is explicit that the scaling is kept outside the router path for this
//   reason. Folding 1.5 into the per-expert weights instead would also scale
//   the shared expert once the two are added, which is a different function.

#include "moe/emu/dsv4_layers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::emu {
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("moe_block: " + message);
  }
}

// silu(z) = z * sigmoid(z), guarded against exp overflow for very negative z.
float Silu(float z) noexcept {
  if (z < -80.0F) {
    return 0.0F;
  }
  return z / (1.0F + std::exp(-z));
}

// The clamped SwiGLU, then the down projection. Shared by the FP4 and FP8
// expert paths so the activation cannot drift between them.
// Three separate template parameters: each lambda below has its own closure
// type, so one shared GemvFn cannot bind all three.
template <typename GateFn, typename UpFn, typename DownFn>
void ExpertSwiglu(GateFn&& gate_gemv, UpFn&& up_gemv, DownFn&& down_gemv,
                  int intermediate, int hidden, const float* x, float* out, float limit,
                  std::vector<float>* scratch) {
  Require(x != nullptr && out != nullptr, "expert operand is null");
  Require(intermediate > 0 && hidden > 0, "expert has an empty extent");
  Require(scratch != nullptr, "expert scratch is null");
  // One owned buffer for gate, up and the activation, resized once and reused
  // across every expert and every token.
  scratch->resize(static_cast<std::size_t>(intermediate) * 2);
  float* gate = scratch->data();
  float* up = scratch->data() + intermediate;

  gate_gemv(x, gate);
  up_gemv(x, up);

  for (int i = 0; i < intermediate; ++i) {
    // AscendSiluAndMulWithClamp: the gate is bounded ABOVE ONLY, the up
    // projection on both sides. The asymmetry is in the reference and is
    // load-bearing wherever w1 @ x is strongly negative.
    const float clamped_gate = std::min(gate[i], limit);
    const float clamped_up = std::min(limit, std::max(-limit, up[i]));
    gate[i] = Silu(clamped_gate) * clamped_up;
  }
  (void)hidden;
  down_gemv(gate, out);
}

}  // namespace

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

float SqrtSoftplus(float logit) noexcept {
  // aclnnSoftplus with beta = 1, threshold = 20: softplus(x) = ln(1 + e^x)
  // below the threshold and the identity above it, which is what keeps e^x
  // from overflowing for a large logit. softplus is positive everywhere, so
  // the sqrt is always real.
  const float softplus = (logit > 20.0F) ? logit : std::log1p(std::exp(logit));
  return std::sqrt(softplus);
}

// ---------------------------------------------------------------------------
// routing
// ---------------------------------------------------------------------------

namespace {

// Normalize the selected weights to sum 1 ("norm_topk_prob": true).
void MaybeRenormalize(MoeRouting* routing, bool renormalize) {
  if (!renormalize || routing->count == 0) {
    return;
  }
  float total = 0.0F;
  for (int i = 0; i < routing->count; ++i) {
    total += routing->weights[i];
  }
  // A zero total would mean every selected score was zero; leaving the
  // weights alone is the only option that does not produce NaN.
  if (!(total > 0.0F)) {
    return;
  }
  for (int i = 0; i < routing->count; ++i) {
    routing->weights[i] /= total;
  }
}

}  // namespace

MoeRouting moe_route_hash_cpu(const void* tid2eid_row, const char* tid2eid_dtype,
                              int count, const float* gate_logits, int num_experts,
                              bool renormalize) {
  Require(tid2eid_row != nullptr, "tid2eid row is null");
  Require(gate_logits != nullptr, "gate logits are null");
  Require(count > 0 && count <= kNumExpertsPerTok,
          "tid2eid row width must be in (0, num_experts_per_tok]");
  Require(num_experts > 0, "num_experts must be positive");
  Require(tid2eid_dtype != nullptr, "tid2eid dtype is null");

  // The checkpoint ships I64; hash_router.py validates int32. Accept both and
  // refuse anything else rather than reinterpreting bytes at the wrong width.
  const bool is_i64 = std::strcmp(tid2eid_dtype, "I64") == 0;
  const bool is_i32 = std::strcmp(tid2eid_dtype, "I32") == 0;
  Require(is_i64 || is_i32,
          std::string("tid2eid must be I64 or I32, got ") + tid2eid_dtype);

  MoeRouting routing;
  routing.from_hash_table = true;
  for (int i = 0; i < count; ++i) {
    std::int64_t expert = 0;
    if (is_i64) {
      std::memcpy(&expert, static_cast<const std::uint8_t*>(tid2eid_row) +
                               static_cast<std::size_t>(i) * sizeof(std::int64_t),
                  sizeof(std::int64_t));
    } else {
      std::int32_t narrow = 0;
      std::memcpy(&narrow, static_cast<const std::uint8_t*>(tid2eid_row) +
                               static_cast<std::size_t>(i) * sizeof(std::int32_t),
                  sizeof(std::int32_t));
      expert = narrow;
    }
    Require(expert >= 0 && expert < num_experts,
            "tid2eid references expert " + std::to_string(expert) +
                ", outside [0, " + std::to_string(num_experts) + ")");
    routing.experts[routing.count] = static_cast<int>(expert);
    // The gate supplies only the weight here; selection was the table's job.
    routing.weights[routing.count] = SqrtSoftplus(gate_logits[expert]);
    ++routing.count;
  }
  MaybeRenormalize(&routing, renormalize);
  return routing;
}

MoeRouting moe_route_score_cpu(const float* gate_logits, const float* bias,
                               int num_experts, int top_k, bool renormalize) {
  Require(gate_logits != nullptr, "gate logits are null");
  Require(num_experts > 0, "num_experts must be positive");
  Require(top_k > 0 && top_k <= kNumExpertsPerTok,
          "top_k must be in (0, num_experts_per_tok]");
  Require(top_k <= num_experts, "top_k must not exceed num_experts");

  std::vector<float> scores(static_cast<std::size_t>(num_experts));
  for (int expert = 0; expert < num_experts; ++expert) {
    scores[static_cast<std::size_t>(expert)] = SqrtSoftplus(gate_logits[expert]);
  }

  // noaux_tc: the bias shifts the SELECTION only. The weight returned for a
  // selected expert is its UNSHIFTED score -- that distinction is the whole
  // point of the method, and adding the bias into the weight would change
  // every routed magnitude.
  std::vector<int> order(static_cast<std::size_t>(num_experts));
  std::iota(order.begin(), order.end(), 0);
  const auto shifted = [&](int expert) {
    return scores[static_cast<std::size_t>(expert)] +
           ((bias == nullptr) ? 0.0F : bias[expert]);
  };
  std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                    [&](int a, int b) {
                      const float sa = shifted(a);
                      const float sb = shifted(b);
                      if (sa != sb) {
                        return sa > sb;
                      }
                      return a < b;  // a total order, so selection is deterministic
                    });

  MoeRouting routing;
  routing.from_hash_table = false;
  for (int i = 0; i < top_k; ++i) {
    routing.experts[i] = order[static_cast<std::size_t>(i)];
    routing.weights[i] = scores[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    ++routing.count;
  }
  MaybeRenormalize(&routing, renormalize);
  return routing;
}

// ---------------------------------------------------------------------------
// expert execution
// ---------------------------------------------------------------------------

void moe_expert_swiglu_mxfp4_cpu(const Mxfp4Matrix& w1, const Mxfp4Matrix& w3,
                                 const Mxfp4Matrix& w2, const float* x, float* out,
                                 float limit, std::vector<float>* scratch) {
  Require(w1.rows == w3.rows, "w1 and w3 must agree on the intermediate size");
  Require(w1.cols == w3.cols, "w1 and w3 must agree on the hidden size");
  Require(w2.cols == w1.rows, "w2's input must be the intermediate size");
  const auto gate = [&](const float* in, float* dst) { mxfp4_gemv_cpu(w1, in, dst); };
  const auto up = [&](const float* in, float* dst) { mxfp4_gemv_cpu(w3, in, dst); };
  const auto down = [&](const float* in, float* dst) { mxfp4_gemv_cpu(w2, in, dst); };
  ExpertSwiglu(gate, up, down, w1.rows, w2.rows, x, out, limit, scratch);
}

void moe_expert_swiglu_fp8_cpu(const Fp8BlockMatrix& w1, const Fp8BlockMatrix& w3,
                               const Fp8BlockMatrix& w2, const float* x, float* out,
                               float limit, std::vector<float>* scratch) {
  Require(w1.rows == w3.rows, "w1 and w3 must agree on the intermediate size");
  Require(w1.cols == w3.cols, "w1 and w3 must agree on the hidden size");
  Require(w2.cols == w1.rows, "w2's input must be the intermediate size");
  const auto gate = [&](const float* in, float* dst) {
    fp8_block_gemv_cpu(w1, in, dst);
  };
  const auto up = [&](const float* in, float* dst) { fp8_block_gemv_cpu(w3, in, dst); };
  const auto down = [&](const float* in, float* dst) {
    fp8_block_gemv_cpu(w2, in, dst);
  };
  ExpertSwiglu(gate, up, down, w1.rows, w2.rows, x, out, limit, scratch);
}

// ---------------------------------------------------------------------------
// Whole-block oracles
// ---------------------------------------------------------------------------

void moe_decode_topk_mxfp4_cpu(const Mxfp4Matrix* w1, const Mxfp4Matrix* w3,
                               const Mxfp4Matrix* w2, const float* x,
                               const std::int32_t* topk_expert_ids,
                               const float* routing_weights, int top_k,
                               int num_experts, float* out, float limit,
                               std::vector<float>* scratch) {
  Require(w1 != nullptr && w3 != nullptr && w2 != nullptr,
          "decode expert weight array is null");
  Require(x != nullptr && out != nullptr, "decode operand is null");
  Require(topk_expert_ids != nullptr && routing_weights != nullptr,
          "decode routing operand is null");
  Require(scratch != nullptr, "decode scratch is null");
  Require(top_k > 0 && num_experts > 0, "decode has an empty extent");

  const int hidden = w2[0].rows;
  const int inter = w1[0].rows;
  // Every expert must share one geometry: the device kernel addresses expert
  // e at a FIXED stride, so a ragged expert set would not merely be slow, it
  // would read the wrong bytes.
  for (int e = 0; e < num_experts; ++e) {
    Require(w1[e].rows == inter && w3[e].rows == inter,
            "every expert must share the intermediate size");
    Require(w2[e].rows == hidden, "every expert must share the hidden size");
    Require(w1[e].cols == hidden && w3[e].cols == hidden,
            "w1/w3 must reduce over the hidden size");
    Require(w2[e].cols == inter, "w2 must reduce over the intermediate size");
  }

  std::fill(out, out + hidden, 0.0F);
  // One expert's unweighted output, reused across slots.
  std::vector<float> partial(static_cast<std::size_t>(hidden));
  for (int s = 0; s < top_k; ++s) {
    const std::int32_t e = topk_expert_ids[s];
    Require(e >= 0 && e < num_experts, "routed expert id is out of range");
    moe_expert_swiglu_mxfp4_cpu(w1[e], w3[e], w2[e], x, partial.data(), limit,
                                scratch);
    const float weight = routing_weights[s];
    // Ascending slot order -- see the contract note on the declaration.
    for (int c = 0; c < hidden; ++c) {
      out[c] += weight * partial[static_cast<std::size_t>(c)];
    }
  }
}

void moe_prefill_batched_mxfp4_cpu(const Mxfp4Matrix* w1, const Mxfp4Matrix* w3,
                                   const Mxfp4Matrix* w2, const float* x_tokens,
                                   const std::int32_t* token_to_expert,
                                   int total_tokens, int num_experts,
                                   float* y_tokens, float limit,
                                   std::vector<float>* scratch) {
  Require(w1 != nullptr && w3 != nullptr && w2 != nullptr,
          "prefill expert weight array is null");
  Require(x_tokens != nullptr && y_tokens != nullptr, "prefill operand is null");
  Require(token_to_expert != nullptr, "prefill routing operand is null");
  Require(scratch != nullptr, "prefill scratch is null");
  Require(total_tokens > 0 && num_experts > 0, "prefill has an empty extent");

  const int hidden = w2[0].rows;
  const int inter = w1[0].rows;
  for (int e = 0; e < num_experts; ++e) {
    Require(w1[e].rows == inter && w3[e].rows == inter,
            "every expert must share the intermediate size");
    Require(w2[e].rows == hidden, "every expert must share the hidden size");
    Require(w1[e].cols == hidden && w3[e].cols == hidden,
            "w1/w3 must reduce over the hidden size");
    Require(w2[e].cols == inter, "w2 must reduce over the intermediate size");
  }

  for (int t = 0; t < total_tokens; ++t) {
    const std::int32_t e = token_to_expert[t];
    Require(e >= 0 && e < num_experts, "token's expert id is out of range");
    const std::size_t row = static_cast<std::size_t>(t) *
                            static_cast<std::size_t>(hidden);
    moe_expert_swiglu_mxfp4_cpu(w1[e], w3[e], w2[e], x_tokens + row,
                                y_tokens + row, limit, scratch);
  }
}

}  // namespace moe::emu
