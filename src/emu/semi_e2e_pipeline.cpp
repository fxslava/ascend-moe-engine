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

#include "moe/emu/semi_e2e_pipeline.hpp"

#include "moe/core/json.hpp"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace moe::emu {
namespace {

namespace json = ascend_moe::moe_json;
using Clock = std::chrono::steady_clock;

double MicrosecondsSince(const Clock::time_point& start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("semi_e2e_pipeline: " + message);
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    Fail("cannot open " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

float L2Norm(const std::uint16_t* values, std::size_t count) {
  double total = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = Bf16ToFloat(values[i]);
    total += value * value;
  }
  return static_cast<float>(std::sqrt(total));
}

float L2Norm(const float* values, std::size_t count) {
  double total = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    total += static_cast<double>(values[i]) * values[i];
  }
  return static_cast<float>(std::sqrt(total));
}

bool AllFinite(const float* values, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

// RoPE tables for one position. `theta` differs between the dense attention
// path ("rope_theta": 10000) and the compressor path
// ("compress_rope_theta": 160000), which is why it is a parameter rather than
// a constant -- using one for the other is a silent, plausible error.
void BuildRopeTables(int position, double theta, int rope_head_dim,
                     std::vector<float>* cos_table, std::vector<float>* sin_table) {
  cos_table->assign(static_cast<std::size_t>(rope_head_dim), 0.0F);
  sin_table->assign(static_cast<std::size_t>(rope_head_dim), 0.0F);
  const int half = rope_head_dim / 2;
  for (int i = 0; i < half; ++i) {
    const double inverse_frequency =
        1.0 / std::pow(theta, static_cast<double>(2 * i) / rope_head_dim);
    const double angle = static_cast<double>(position) * inverse_frequency;
    const float c = static_cast<float>(std::cos(angle));
    const float s = static_cast<float>(std::sin(angle));
    (*cos_table)[static_cast<std::size_t>(i)] = c;
    (*sin_table)[static_cast<std::size_t>(i)] = s;
    (*cos_table)[static_cast<std::size_t>(half + i)] = c;
    (*sin_table)[static_cast<std::size_t>(half + i)] = s;
  }
}

std::vector<std::string> SplitCsv(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// weight resolution
// ---------------------------------------------------------------------------

namespace {

// A small helper bundle so every lookup below reports the same way.
struct Resolver {
  const SafetensorsView* view;

  const TensorView& Get(const std::string& name) const {
    const TensorView* tensor = view->Find(name);
    if (tensor == nullptr) {
      Fail("the weight file has no tensor '" + name +
           "'. Re-run scripts/download_dsv4_semi_e2e_weights.sh with the "
           "layers this pipeline needs.");
    }
    if (!tensor->available) {
      Fail("tensor '" + name + "' is in the header but its bytes are missing");
    }
    return *tensor;
  }

  const TensorView& Shaped(const std::string& name,
                           std::initializer_list<std::int64_t> expected) const {
    const TensorView& tensor = Get(name);
    if (!tensor.is_shape(expected)) {
      std::string wanted = "[";
      std::size_t index = 0;
      for (const std::int64_t extent : expected) {
        if (index++ != 0) {
          wanted += ", ";
        }
        wanted += std::to_string(extent);
      }
      Fail("tensor '" + name + "' is " + tensor.dtype + " " + tensor.shape_string() +
           " but this build expects " + wanted + "]");
    }
    return tensor;
  }

  const std::uint8_t* Bytes(const std::string& name) const {
    return static_cast<const std::uint8_t*>(Get(name).data_ptr);
  }

  // An FP8 E4M3 matrix plus its E8M0 128x128 block scale grid, with the grid
  // derived from the declared shapes rather than assumed.
  Fp8BlockMatrix Fp8(const std::string& stem, std::int64_t rows,
                     std::int64_t cols) const {
    const TensorView& weight = Shaped(stem + ".weight", {rows, cols});
    if (weight.dtype != "F8_E4M3") {
      Fail("'" + stem + ".weight' is " + weight.dtype + ", expected F8_E4M3");
    }
    const std::int64_t scale_rows = (rows + kFp8BlockSize - 1) / kFp8BlockSize;
    const std::int64_t scale_cols = (cols + kFp8BlockSize - 1) / kFp8BlockSize;
    const TensorView& scale = Shaped(stem + ".scale", {scale_rows, scale_cols});
    if (scale.dtype != "F8_E8M0") {
      Fail("'" + stem + ".scale' is " + scale.dtype + ", expected F8_E8M0");
    }
    Fp8BlockMatrix matrix;
    matrix.weight = static_cast<const std::uint8_t*>(weight.data_ptr);
    matrix.scale = static_cast<const std::uint8_t*>(scale.data_ptr);
    matrix.rows = static_cast<int>(rows);
    matrix.cols = static_cast<int>(cols);
    matrix.scale_cols = static_cast<int>(scale_cols);
    return matrix;
  }

  // A routed-expert MXFP4 matrix. `cols` is LOGICAL; the stored tensor is
  // half as wide because two E2M1 nibbles share a byte, and the scale grid is
  // one column per 32 logical elements.
  Mxfp4Matrix Mxfp4(const std::string& stem, std::int64_t rows,
                    std::int64_t logical_cols) const {
    const TensorView& weight = Shaped(stem + ".weight", {rows, logical_cols / 2});
    // The checkpoint spells the packed payload I8 -- a byte container -- not
    // F4_E2M1. config.json's "expert_dtype": "fp4" plus the scale grid width
    // (logical_cols / 32 groups over logical_cols / 2 stored bytes) is what
    // establishes that these bytes are nibble pairs.
    if (weight.dtype != "I8" && weight.dtype != "U8" && weight.dtype != "F4_E2M1") {
      Fail("'" + stem + ".weight' is " + weight.dtype +
           ", expected the packed-FP4 byte container (I8/U8/F4_E2M1)");
    }
    const TensorView& scale =
        Shaped(stem + ".scale", {rows, logical_cols / kMxfp4GroupSize});
    if (scale.dtype != "F8_E8M0") {
      Fail("'" + stem + ".scale' is " + scale.dtype + ", expected F8_E8M0");
    }
    Mxfp4Matrix matrix;
    matrix.packed = static_cast<const std::uint8_t*>(weight.data_ptr);
    matrix.scale = static_cast<const std::uint8_t*>(scale.data_ptr);
    matrix.rows = static_cast<int>(rows);
    matrix.cols = static_cast<int>(logical_cols);
    return matrix;
  }

  SharedKvAttentionWeights Attention(int layer) const {
    const std::string prefix = "layers." + std::to_string(layer) + ".attn";
    SharedKvAttentionWeights weights;
    weights.wq_a = Fp8(prefix + ".wq_a", kQLoraRank, kHiddenSize);
    weights.wq_b = Fp8(prefix + ".wq_b", kAttnHeads * kAttnHeadDim, kQLoraRank);
    weights.wkv = Fp8(prefix + ".wkv", kAttnHeadDim, kHiddenSize);
    weights.wo_a = Fp8(prefix + ".wo_a", kWoAOutput, kWoAGroupInput);
    weights.wo_b = Fp8(prefix + ".wo_b", kHiddenSize, kWoAOutput);
    weights.q_norm = static_cast<const std::uint16_t*>(
        Shaped(prefix + ".q_norm.weight", {kQLoraRank}).data_ptr);
    weights.kv_norm = static_cast<const std::uint16_t*>(
        Shaped(prefix + ".kv_norm.weight", {kAttnHeadDim}).data_ptr);
    weights.attn_sink = reinterpret_cast<const float*>(
        Shaped(prefix + ".attn_sink", {kAttnHeads}).data_ptr);
    return weights;
  }
};

// The three mHC tensors for one round.
struct MhcWeights {
  const float* phi = nullptr;
  const float* alpha = nullptr;
  const float* bias = nullptr;
  std::vector<float> phi_storage;
  std::vector<float> alpha_storage;
  std::vector<float> bias_storage;
};

// phi is F32 [24, 16384] and the mapped data segment is NOT 4-byte aligned in
// a repacked file, so it is copied out rather than aliased: reading it through
// a `const float*` cast would be undefined behaviour. 1.6 MB per round.
MhcWeights LoadMhc(const Resolver& resolver, int layer, const char* stem) {
  const std::string prefix = "layers." + std::to_string(layer) + ".hc_" + stem;
  MhcWeights weights;
  const TensorView& phi = resolver.Shaped(prefix + "_fn", {kHcPhiRows, kHcPhiCols});
  const TensorView& alpha = resolver.Shaped(prefix + "_scale", {kHcAlphaElements});
  const TensorView& bias = resolver.Shaped(prefix + "_base", {kHcProjectionRows});
  weights.phi_storage.resize(static_cast<std::size_t>(kHcPhiRows) * kHcPhiCols);
  weights.alpha_storage.resize(kHcAlphaElements);
  weights.bias_storage.resize(kHcProjectionRows);
  phi.CopyTo(weights.phi_storage.data(), weights.phi_storage.size());
  alpha.CopyTo(weights.alpha_storage.data(), weights.alpha_storage.size());
  bias.CopyTo(weights.bias_storage.data(), weights.bias_storage.size());
  weights.phi = weights.phi_storage.data();
  weights.alpha = weights.alpha_storage.data();
  weights.bias = weights.bias_storage.data();
  return weights;
}

}  // namespace

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------

SemiE2ePipeline::SemiE2ePipeline(const std::string& weights_path,
                                 const std::string& config_path)
    : weights_(weights_path) {
  const json::Value config = json::Parse(ReadWholeFile(config_path));
  const auto required_int = [&](const char* key) {
    const json::Value* value = config.find(key);
    if (value == nullptr) {
      Fail(std::string("config.json has no \"") + key + "\"");
    }
    return value->as_int();
  };

  // Refuse a checkpoint whose geometry diverges from the compiled constants,
  // naming the field. A silent re-plan would read every matrix at the wrong
  // stride.
  struct Check {
    const char* field;
    std::int64_t expected;
  };
  const Check checks[] = {
      {"hidden_size", kHiddenSize},
      {"hc_mult", kHcStreams},
      {"hc_sinkhorn_iters", kHcSinkhornIters},
      {"num_attention_heads", kAttnHeads},
      {"head_dim", kAttnHeadDim},
      {"q_lora_rank", kQLoraRank},
      {"o_lora_rank", kOLoraRank},
      {"o_groups", kOGroups},
      {"qk_rope_head_dim", kAttnRopeHeadDim},
      {"sliding_window", kSlidingWindow},
      {"n_routed_experts", kNumRoutedExperts},
      {"num_experts_per_tok", kNumExpertsPerTok},
      {"moe_intermediate_size", kMoeIntermediate},
      {"index_topk", kIndexerTopK},
      {"index_n_heads", kIndexerHeads},
      {"index_head_dim", kIndexerHeadDim},
      {"num_hash_layers", kNumHashLayers},
  };
  std::ostringstream summary;
  std::string divergences;
  for (const Check& check : checks) {
    const std::int64_t actual = required_int(check.field);
    if (actual != check.expected) {
      divergences += std::string("\n  ") + check.field + ": checkpoint " +
                     std::to_string(actual) + ", binary " +
                     std::to_string(check.expected);
    }
    summary << "  " << check.field << " = " << actual << "\n";
  }
  if (!divergences.empty()) {
    Fail("the checkpoint geometry diverges from this binary:" + divergences);
  }
  geometry_summary_ = summary.str();

  // The weight file records which token it was fetched for and which experts
  // that token routes to; Run() refuses a different token so the MoE block
  // cannot route to an expert the file does not carry.
  const auto& metadata = weights_.metadata();
  const auto token = metadata.find("token_id");
  if (token != metadata.end()) {
    metadata_token_id_ = std::stoll(token->second);
  }
  const auto hash_flag = metadata.find("moe_is_hash_layer");
  if (hash_flag != metadata.end()) {
    moe_is_hash_layer_ = hash_flag->second == "true";
  }
  const auto routed = metadata.find("routed_experts");
  if (routed != metadata.end()) {
    for (const std::string& item : SplitCsv(routed->second)) {
      metadata_routed_experts_.push_back(std::stoi(item));
    }
  }
}

// ---------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------

SemiE2eReport SemiE2ePipeline::Run(const SemiE2eOptions& options,
                                   std::vector<std::uint16_t>* x_next) {
  if (x_next == nullptr) {
    Fail("x_next is null");
  }
  if (metadata_token_id_ >= 0 && options.token_id != metadata_token_id_) {
    Fail("the weight file was fetched for token id " +
         std::to_string(metadata_token_id_) + " but this run uses " +
         std::to_string(options.token_id) +
         ". The hash router would select experts that are not in the file; "
         "re-fetch with --token-id " + std::to_string(options.token_id) + ".");
  }

  const Resolver resolver{&weights_};
  SemiE2eReport report;
  report.token_id = options.token_id;
  const auto wall_start = Clock::now();

  const std::size_t bundle = static_cast<std::size_t>(kHcStreams) * kHiddenSize;
  std::vector<std::uint16_t> x(bundle);
  std::vector<std::uint16_t> scratch_bundle(bundle);
  x_next->assign(bundle, 0);

  const auto record = [&report](const std::string& name, Fidelity fidelity,
                                const std::string& provenance, const std::string& note,
                                double microseconds) {
    report.stages.push_back(StageReport{name, fidelity, provenance, note, microseconds});
  };

  // ===== stage 0: the input token =========================================
  // The real layer 0 input is embed.weight[token_id] broadcast across the four
  // mHC streams. embed.weight is 1.06 GB and lives in shard 1, which the
  // semi-e2e fetch deliberately skips, so the embedding row is SYNTHESIZED --
  // deterministically from the token id, so the golden dump is reproducible.
  {
    const auto start = Clock::now();
    std::mt19937 engine(static_cast<std::uint32_t>(0x9E3779B9u ^ options.token_id));
    std::normal_distribution<float> distribution(0.0F, 0.02F);  // initializer_range
    std::vector<float> embedding(kHiddenSize);
    for (float& value : embedding) {
      value = distribution(engine);
    }
    for (int stream = 0; stream < kHcStreams; ++stream) {
      for (int feature = 0; feature < kHiddenSize; ++feature) {
        x[static_cast<std::size_t>(stream) * kHiddenSize + feature] =
            FloatToBf16(embedding[static_cast<std::size_t>(feature)]);
      }
    }
    record("input: embed row -> [1, 4, 4096] BF16", Fidelity::kSyntheticInput,
           "synthesized from the token id",
           "embed.weight is 1.06 GB in shard 1, which this fetch skips; the row "
           "is drawn deterministically at initializer_range 0.02",
           MicrosecondsSince(start));
  }

  std::vector<float> rope_cos;
  std::vector<float> rope_sin;
  std::vector<float> h_in_normed(kHiddenSize);
  std::vector<float> attention_out(kHiddenSize);
  std::vector<std::uint16_t> h_out_bf16(kHiddenSize);

  // One full mHC-wrapped attention layer, as DeepseekV2DecoderLayer.forward
  // writes it. Returns the attention statistics.
  const auto run_attention_layer =
      [&](int layer, MhcSinkhornTrace* trace, AttentionStats* attention_stats,
          const std::string& label) {
        const MhcWeights mhc = LoadMhc(resolver, layer, "attn");
        const SharedKvAttentionWeights attention = resolver.Attention(layer);
        const auto* attn_norm = static_cast<const std::uint16_t*>(
            resolver
                .Shaped("layers." + std::to_string(layer) + ".attn_norm.weight",
                        {kHiddenSize})
                .data_ptr);

        const auto start = Clock::now();
        // 1. hc_pre: the 24-row projection, the gates, and the Sinkhorn solve.
        mhc_pre_sinkhorn_cpu(x.data(), mhc.phi, mhc.alpha, mhc.bias, trace,
                             kHcEpsilon, kHcSinkhornIters, kRmsNormEpsilon);
        // 2. input_layernorm on h_in -- the [1, D] mHC output, NOT the bundle.
        rms_norm_cpu(trace->h_in, attn_norm, h_in_normed.data(), kHiddenSize,
                     kRmsNormEpsilon);

        // 3. Prime the sliding window, then the measured step. Without this the
        //    window holds one token and the softmax is a no-op, which would
        //    hide any window or RoPE error.
        SharedKvWindowCache cache(kSlidingWindow, kAttnHeadDim);
        for (int warm = 0; warm < options.warmup_window_tokens; ++warm) {
          BuildRopeTables(warm, 10000.0, kAttnRopeHeadDim, &rope_cos, &rope_sin);
          // KV only: a warmup token exists to populate the window, and the
          // query/output projections it does not need are 98% of the cost.
          shared_kv_swa_prefill_kv_cpu(h_in_normed.data(), attention, &cache, warm,
                                       rope_cos.data(), rope_sin.data());
        }
        const int position = options.warmup_window_tokens;
        BuildRopeTables(position, 10000.0, kAttnRopeHeadDim, &rope_cos, &rope_sin);
        shared_kv_swa_attention_decode_cpu(h_in_normed.data(), attention, &cache,
                                           position, rope_cos.data(), rope_sin.data(),
                                           attention_out.data(), attention_stats);

        // 4. hc_post: fold the attention output back into the four streams.
        for (int i = 0; i < kHiddenSize; ++i) {
          h_out_bf16[static_cast<std::size_t>(i)] =
              FloatToBf16(attention_out[static_cast<std::size_t>(i)]);
        }
        mhc_post_cpu(x.data(), trace->birkhoff, trace->h_post, h_out_bf16.data(),
                     scratch_bundle.data());
        x.swap(scratch_bundle);
        const double elapsed = MicrosecondsSince(start);

        if (!AllFinite(attention_out.data(), kHiddenSize)) {
          report.all_finite = false;
        }
        record(label, Fidelity::kReferenceWithAssumption,
               "layers." + std::to_string(layer) +
                   ".{hc_attn_*, attn_norm, attn.*} (real)",
               "attn_sink enters the softmax as an extra sinkless column; its "
               "exact placement lives in the fused NPU operator, not in the "
               "reference Python",
               elapsed);
      };

  // ===== layer 0: mHC + shared-KV SWA attention ===========================
  run_attention_layer(options.layer0, &report.layer0_attn_trace,
                      &report.layer0_attention,
                      "layer 0: mHC round + shared-KV SWA attention (window 128)");
  report.x_after_layer0_l2 = L2Norm(x.data(), x.size());

  // ===== layer 2: mHC + SWA attention + compressor + indexer ==============
  run_attention_layer(options.layer2, &report.layer2_attn_trace,
                      &report.layer2_attention,
                      "layer 2: mHC round + shared-KV SWA attention");
  report.x_after_layer2_l2 = L2Norm(x.data(), x.size());

  // --- the 4:1 compressor cadence over `decode_steps` tokens --------------
  // Driven by layer 2's h_in, which is what the real compressor consumes.
  std::vector<std::uint16_t> compressed_tokens;
  {
    const std::string prefix = "layers." + std::to_string(options.layer2) + ".attn.compressor";
    const std::int64_t state_width = kCompressorCoff * kCompressorHeadDim;  // 1024
    const auto* wkv = static_cast<const std::uint16_t*>(
        resolver.Shaped(prefix + ".wkv.weight", {state_width, kHiddenSize}).data_ptr);
    const auto* wgate = static_cast<const std::uint16_t*>(
        resolver.Shaped(prefix + ".wgate.weight", {state_width, kHiddenSize}).data_ptr);
    const TensorView& ape_view =
        resolver.Shaped(prefix + ".ape", {kCompressorRatio, state_width});
    const auto* norm_weight = static_cast<const std::uint16_t*>(
        resolver.Shaped(prefix + ".norm.weight", {kCompressorHeadDim}).data_ptr);
    // ape is F32 and the repacked data segment is not 4-byte aligned, so copy.
    std::vector<float> ape(static_cast<std::size_t>(kCompressorRatio) * state_width);
    ape_view.CopyTo(ape.data(), ape.size());

    const auto start = Clock::now();
    CompressorRingState ring;
    std::vector<std::uint16_t> slot(kCompressorHeadDim, 0);
    std::vector<std::uint16_t> token(kHiddenSize);
    for (int step = 0; step < options.decode_steps; ++step) {
      for (int i = 0; i < kHiddenSize; ++i) {
        token[static_cast<std::size_t>(i)] = FloatToBf16(
            h_in_normed[static_cast<std::size_t>(i)] *
            (1.0F + 0.01F * static_cast<float>(step)));
      }
      // The compressor path uses "compress_rope_theta": 160000, NOT
      // "rope_theta": 10000, and the table is for the COMPRESSED token index.
      BuildRopeTables(step / kCompressorRatio, 160000.0, kRopeHeadDim, &rope_cos,
                      &rope_sin);
      const CompressorStepResult result = paged_compressor_decode_c4a(
          token.data(), wkv, wgate, ape.data(), norm_weight, step, &ring,
          rope_cos.data(), rope_sin.data(), slot.data(), kHiddenSize,
          kCompressorHeadDim, kCompressorRatio, kRmsNormEpsilon, RotaryMode::kHalf);
      if (result.emitted) {
        ++report.compressed_tokens_emitted;
        compressed_tokens.insert(compressed_tokens.end(), slot.begin(), slot.end());
      }
    }
    record("layer 2: compressor C4A cadence, " + std::to_string(options.decode_steps) +
               " steps -> " + std::to_string(report.compressed_tokens_emitted) +
               " tokens",
           Fidelity::kReference,
           "layers." + std::to_string(options.layer2) + ".attn.compressor.* (real)",
           "coff 2, ratio 4, D 512; RoPE on compress_rope_theta = 160000",
           MicrosecondsSince(start));
  }

  // --- the FP8 lightning indexer, top-512 ---------------------------------
  std::vector<std::int32_t> top_indices(static_cast<std::size_t>(kIndexerTopK), -1);
  {
    const std::string prefix = "layers." + std::to_string(options.layer2) + ".attn.indexer";
    const auto* weights_proj = static_cast<const std::uint16_t*>(
        resolver.Shaped(prefix + ".weights_proj.weight", {kIndexerHeads, kHiddenSize})
            .data_ptr);

    const auto start = Clock::now();
    // The per-head gate: w = weights_proj @ h_in, folded into the query
    // dequant scale exactly as the golden's `cur_w = cur_wt * cur_q_scale`
    // does. Magnitudes only -- the scale multiplies a relu'd score.
    std::vector<float> head_weights(kIndexerHeads);
    bf16_gemv_cpu(weights_proj, kIndexerHeads, kHiddenSize, h_in_normed.data(),
                  head_weights.data());

    const std::int32_t block_size = 128;
    const std::int32_t blocks = (options.indexer_seq_len + block_size - 1) / block_size;
    std::vector<float> scales(static_cast<std::size_t>(kIndexerHeads) +
                              options.indexer_seq_len);
    for (int head = 0; head < kIndexerHeads; ++head) {
      scales[static_cast<std::size_t>(head)] =
          std::fabs(head_weights[static_cast<std::size_t>(head)]);
    }

    // The indexer's FP8 query and key cache are runtime ACTIVATIONS produced
    // by indexer.wq_b and indexer.compressor; both are in the fetched set,
    // but the key cache needs `indexer_seq_len` historical compressed tokens
    // that a single decode step has not produced. The Q/K bytes are therefore
    // synthesized at the exact PA_BSND geometry, while the per-head gate above
    // is real.
    std::mt19937 engine(0xC0FFEEu ^ static_cast<std::uint32_t>(options.token_id));
    std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
    std::vector<std::uint8_t> query(static_cast<std::size_t>(kIndexerHeads) *
                                    kIndexerHeadDim);
    for (std::uint8_t& byte : query) {
      byte = EncodeFp8E4M3(distribution(engine));
    }
    std::vector<std::uint8_t> key_cache(static_cast<std::size_t>(blocks) * block_size *
                                        kIndexerHeadDim);
    for (std::uint8_t& byte : key_cache) {
      byte = EncodeFp8E4M3(distribution(engine));
    }
    for (std::int32_t j = 0; j < options.indexer_seq_len; ++j) {
      scales[static_cast<std::size_t>(kIndexerHeads + j)] = 0.5F + distribution(engine) * 0.1F;
    }
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(blocks));
    for (std::int32_t b = 0; b < blocks; ++b) {
      block_table[static_cast<std::size_t>(b)] = b;
    }

    fast_fp8_indexer_topk_cpu(query.data(), key_cache.data(), scales.data(),
                              block_table.data(), options.indexer_seq_len,
                              kIndexerTopK, top_indices.data(), block_size,
                              kIndexerHeads, kIndexerHeadDim);
    for (const std::int32_t index : top_indices) {
      if (index >= 0) {
        ++report.indexer_selected;
      }
    }
    record("layer 2: lightning indexer, top-" + std::to_string(kIndexerTopK) + " of " +
               std::to_string(options.indexer_seq_len),
           Fidelity::kSyntheticInput,
           "layers." + std::to_string(options.layer2) +
               ".attn.indexer.weights_proj (real head gate)",
           "the FP8 Q/K bytes are synthesized: a single decode step cannot "
           "produce " + std::to_string(options.indexer_seq_len) +
               " historical compressed tokens",
           MicrosecondsSince(start));
  }

  // ===== layer 1 MoE: hash router + routed experts + shared expert ========
  {
    const int layer = options.moe_layer;
    const std::string prefix = "layers." + std::to_string(layer) + ".ffn";
    const MhcWeights mhc = LoadMhc(resolver, layer, "ffn");
    const auto* ffn_norm = static_cast<const std::uint16_t*>(
        resolver.Shaped("layers." + std::to_string(layer) + ".ffn_norm.weight",
                        {kHiddenSize})
            .data_ptr);
    const auto* gate_weight = static_cast<const std::uint16_t*>(
        resolver.Shaped(prefix + ".gate.weight", {kNumRoutedExperts, kHiddenSize})
            .data_ptr);

    const auto start = Clock::now();
    MhcSinkhornTrace ffn_trace;
    mhc_pre_sinkhorn_cpu(x.data(), mhc.phi, mhc.alpha, mhc.bias, &ffn_trace,
                         kHcEpsilon, kHcSinkhornIters, kRmsNormEpsilon);
    rms_norm_cpu(ffn_trace.h_in, ffn_norm, h_in_normed.data(), kHiddenSize,
                 kRmsNormEpsilon);

    // The gate runs in FP32 on the normed activation:
    //   router_logits = F.linear(hidden_states.float(), gate.weight)
    std::vector<float> gate_logits(kNumRoutedExperts);
    bf16_gemv_cpu(gate_weight, kNumRoutedExperts, kHiddenSize, h_in_normed.data(),
                  gate_logits.data());

    if (moe_is_hash_layer_) {
      const TensorView& table = resolver.Get(prefix + ".gate.tid2eid");
      if (table.shape.size() != 2 || table.shape[1] != kNumExpertsPerTok) {
        Fail("'" + prefix + ".gate.tid2eid' is " + table.shape_string() +
             ", expected [vocab, num_experts_per_tok]");
      }
      if (options.token_id < 0 || options.token_id >= table.shape[0]) {
        Fail("token id " + std::to_string(options.token_id) + " is outside [0, " +
             std::to_string(table.shape[0]) + ")");
      }
      const std::size_t element = table.element_size();
      const auto* row = static_cast<const std::uint8_t*>(table.data_ptr) +
                        static_cast<std::size_t>(options.token_id) *
                            kNumExpertsPerTok * element;
      report.routing = moe_route_hash_cpu(row, table.dtype.c_str(),
                                          kNumExpertsPerTok, gate_logits.data(),
                                          kNumRoutedExperts, true);
    } else {
      const TensorView& bias_view =
          resolver.Shaped(prefix + ".gate.bias", {kNumRoutedExperts});
      std::vector<float> bias(kNumRoutedExperts);
      bias_view.CopyTo(bias.data(), bias.size());
      report.routing = moe_route_score_cpu(gate_logits.data(), bias.data(),
                                           kNumRoutedExperts, kNumExpertsPerTok, true);
    }

    // Every selected expert must be in the file. With hash routing the
    // selection is fixed by the token id, so this is a real check and not a
    // formality: a token whose experts were not fetched fails here instead of
    // silently running the wrong six.
    report.routing_experts_all_present = true;
    for (int i = 0; i < report.routing.count; ++i) {
      const std::string stem = prefix + ".experts." +
                               std::to_string(report.routing.experts[i]) + ".w1.weight";
      if (weights_.Find(stem) == nullptr) {
        report.routing_experts_all_present = false;
      }
    }
    if (!report.routing_experts_all_present) {
      std::string missing;
      for (int i = 0; i < report.routing.count; ++i) {
        missing += " " + std::to_string(report.routing.experts[i]);
      }
      Fail("the router selected experts" + missing +
           " but the weight file does not carry all of them. Re-fetch with "
           "--token-id " + std::to_string(options.token_id) + ".");
    }

    // --- routed experts --------------------------------------------------
    std::vector<float> routed(kHiddenSize, 0.0F);
    std::vector<float> expert_out(kHiddenSize);
    std::vector<float> expert_scratch;
    for (int i = 0; i < report.routing.count; ++i) {
      const std::string expert_prefix =
          prefix + ".experts." + std::to_string(report.routing.experts[i]);
      const Mxfp4Matrix w1 =
          resolver.Mxfp4(expert_prefix + ".w1", kMoeIntermediate, kHiddenSize);
      const Mxfp4Matrix w3 =
          resolver.Mxfp4(expert_prefix + ".w3", kMoeIntermediate, kHiddenSize);
      const Mxfp4Matrix w2 =
          resolver.Mxfp4(expert_prefix + ".w2", kHiddenSize, kMoeIntermediate);
      moe_expert_swiglu_mxfp4_cpu(w1, w3, w2, h_in_normed.data(), expert_out.data(),
                                  kSwigluLimit, &expert_scratch);
      const float weight = report.routing.weights[i];
      for (int feature = 0; feature < kHiddenSize; ++feature) {
        routed[static_cast<std::size_t>(feature)] +=
            weight * expert_out[static_cast<std::size_t>(feature)];
      }
    }
    report.moe_routed_l2 = L2Norm(routed.data(), routed.size());

    // --- the shared expert (FP8, intermediate = moe_intermediate * n_shared)
    std::vector<float> shared(kHiddenSize, 0.0F);
    {
      const Fp8BlockMatrix w1 =
          resolver.Fp8(prefix + ".shared_experts.w1", kMoeIntermediate, kHiddenSize);
      const Fp8BlockMatrix w3 =
          resolver.Fp8(prefix + ".shared_experts.w3", kMoeIntermediate, kHiddenSize);
      const Fp8BlockMatrix w2 =
          resolver.Fp8(prefix + ".shared_experts.w2", kHiddenSize, kMoeIntermediate);
      moe_expert_swiglu_fp8_cpu(w1, w3, w2, h_in_normed.data(), shared.data(),
                                kSwigluLimit, &expert_scratch);
    }
    report.moe_shared_l2 = L2Norm(shared.data(), shared.size());

    // DeepseekV4MoE.forward: the top-k weights are already normalized, so the
    // scaling factor multiplies the ROUTED SUM and the shared expert is added
    // afterwards at unit weight. Folding 1.5 into the per-expert weights would
    // scale the shared expert too.
    std::vector<std::uint16_t> ffn_out(kHiddenSize);
    for (int feature = 0; feature < kHiddenSize; ++feature) {
      const float value = routed[static_cast<std::size_t>(feature)] *
                              kRoutedScalingFactor +
                          shared[static_cast<std::size_t>(feature)];
      ffn_out[static_cast<std::size_t>(feature)] = FloatToBf16(value);
      if (!std::isfinite(value)) {
        report.all_finite = false;
      }
    }

    mhc_post_cpu(x.data(), ffn_trace.birkhoff, ffn_trace.h_post, ffn_out.data(),
                 scratch_bundle.data());
    x.swap(scratch_bundle);

    record("layer " + std::to_string(layer) + ": mHC round + MoE (" +
               (moe_is_hash_layer_ ? "hash router" : "noaux_tc router") + ", " +
               std::to_string(report.routing.count) + " routed + 1 shared)",
           Fidelity::kReference,
           "layers." + std::to_string(layer) + ".{hc_ffn_*, ffn_norm, ffn.gate.*, "
           "ffn.experts.*, ffn.shared_experts.*} (real)",
           "routed experts are MXFP4 group-32; the nibble order is the "
           "documented assumption in dsv4_layers.h",
           MicrosecondsSince(start));
  }

  report.x_after_moe_l2 = L2Norm(x.data(), x.size());
  *x_next = x;
  report.memory = ReadMemoryUsage();
  report.total_microseconds = MicrosecondsSince(wall_start);
  return report;
}

}  // namespace moe::emu
