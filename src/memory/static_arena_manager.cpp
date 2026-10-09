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

// StaticArenaManager: reservations, backbone ingestion and every descriptor
// the graph consumes. Extracted from Dsv4Pipeline so that memory shape and
// stage scheduling evolve independently (SRP); the pipeline orchestrates, it
// does not allocate.

#include "moe/memory/static_arena_manager.hpp"
#include "moe/core/arch35_contract.hpp"

#include <aclnn/acl_meta.h>

#include "moe/memory/weight_transpose.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "moe/core/error.hpp"
#include "moe/core/resource_scope.hpp"

namespace ascend_moe {
namespace {

// This runner decodes one token at a time (greedy, batch 1). Every buffer the
// manager reserves is sized for it.
constexpr int64_t kTokensPerStep = 1;

// Scale dtype for the dense block-128 weight scales and for the MX block-32
// activation scales: both are OCP E8M0, one byte per block.
constexpr int32_t kScaleDtype = kAclFloat8E8m0;

// The folded MLA projections this runner consumes. A real checkpoint ships
// q_b_proj and o_proj in their unfolded form plus kv_b_proj; folding W_UK into
// q_b and W_UV into o_proj is a dequantize -> matmul -> requantize pass at full
// precision, which belongs to a checkpoint converter and not to a runtime that
// must not lose accuracy silently. The runner expects the folded tensors by
// these names and says so when they are missing.
const char* kFoldedQName = "model.layers.{L}.self_attn.q_b_proj_latent.weight";
const char* kFoldedQScaleName = "model.layers.{L}.self_attn.q_b_proj_latent.weight_scale_inv";
const char* kFoldedOName = "model.layers.{L}.self_attn.o_proj_folded.weight";
const char* kFoldedOScaleName = "model.layers.{L}.self_attn.o_proj_folded.weight_scale_inv";

std::string LayerTensorName(const char* pattern, int64_t layer) {
  std::string name(pattern);
  const std::string token = "{L}";
  const size_t position = name.find(token);
  if (position != std::string::npos) {
    name.replace(position, token.size(), std::to_string(layer));
  }
  return name;
}

}  // namespace

bool IsAuxiliaryCheckpointTensor(const std::string& name) {
  return name.compare(0, 4, "mtp.") == 0 || name.compare(0, 10, "model.mtp.") == 0;
}

std::string ResolveCheckpointTensorName(const std::string& requested,
                                        const CheckpointHasTensor& has_tensor) {
  if (IsAuxiliaryCheckpointTensor(requested)) return {};
  std::vector<std::string> candidates{requested};
  if (requested == "model.embed_tokens.weight") {
    candidates.push_back("embed.weight");
    candidates.push_back("embed_tokens.weight");
  }
  if (requested == "model.norm.weight") candidates.push_back("norm.weight");
  if (requested == "lm_head.weight") candidates.push_back("head.weight");

  const std::string prefix = "model.layers.";
  if (requested.compare(0, prefix.size(), prefix) == 0) {
    const size_t dot = requested.find('.', prefix.size());
    if (dot != std::string::npos) {
      const std::string layer = requested.substr(prefix.size(), dot - prefix.size());
      if (!layer.empty() && layer.find_first_not_of("0123456789") == std::string::npos) {
        const std::string flat = "layers." + layer + ".";
        const std::string leaf = requested.substr(dot + 1);
        candidates.push_back(flat + leaf);
        const std::pair<const char*, const char*> aliases[] = {
            {"input_layernorm.weight", "attn_norm.weight"},
            {"post_attention_layernorm.weight", "ffn_norm.weight"},
            {"mlp.gate.weight", "ffn.gate.weight"},
            {"mlp.gate.e_score_correction_bias", "ffn.gate.bias"},
            {"self_attn.q_a_layernorm.weight", "attn.q_norm.weight"},
            {"self_attn.kv_a_layernorm.weight", "attn.kv_norm.weight"},
        };
        for (const auto& alias : aliases) {
          if (leaf == alias.first) candidates.push_back(flat + alias.second);
        }
        // Both flattened expert layouts occur in converted checkpoints.
        const std::string experts = "mlp.experts.";
        if (leaf.compare(0, experts.size(), experts) == 0) {
          const size_t expert_dot = leaf.find('.', experts.size());
          if (expert_dot != std::string::npos) {
            const std::string expert = leaf.substr(experts.size(), expert_dot - experts.size());
            const std::string projection = leaf.substr(expert_dot + 1);
            const std::pair<const char*, const char*> projections[] = {
                {"gate_proj.weight", "w1.weight"}, {"up_proj.weight", "w3.weight"},
                {"down_proj.weight", "w2.weight"}, {"gate_proj.weight_scale_inv", "w1.scale"},
                {"up_proj.weight_scale_inv", "w3.scale"}, {"down_proj.weight_scale_inv", "w2.scale"},
            };
            for (const auto& alias : projections) {
              if (projection == alias.first) {
                candidates.push_back(flat + "ffn.experts." + expert + "." + alias.second);
                candidates.push_back(flat + "experts." + expert + "." + alias.second);
              }
            }
          }
        }
      }
    }
  }
  for (const auto& candidate : candidates) {
    if (has_tensor(candidate)) return candidate;
  }
  return {};
}

CheckpointTopology ValidateCheckpointTopology(const CheckpointIndex& index, int64_t num_layers) {
  DSV4_REQUIRE(num_layers > 0, "checkpoint topology needs a positive layer count");
  CheckpointTopology topology;
  const auto has = [&](const std::string& name) { return index.count(name) != 0; };
  for (const auto& entry : index) {
    if (IsAuxiliaryCheckpointTensor(entry.first)) {
      ++topology.ignored_auxiliary_tensors;
      continue;
    }
    const std::string& shard = entry.second;
    const std::string suffix = ".safetensors";
    DSV4_REQUIRE(shard.size() > suffix.size() &&
                     shard.compare(shard.size() - suffix.size(), suffix.size(), suffix) == 0,
                 "invalid shard target for " << entry.first << ": " << shard);
  }
  auto require = [&](std::map<std::string, std::string>& bindings, const std::string& role,
                     const std::vector<std::string>& names) {
    for (const auto& name : names) {
      const std::string resolved = ResolveCheckpointTensorName(name, has);
      if (!resolved.empty()) {
        bindings.emplace(role, resolved);
        return;
      }
    }
    throw Dsv4Error("checkpoint topology missing " + role + ": " + names.front());
  };
  require(topology.globals, "embedding", {"model.embed_tokens.weight"});
  require(topology.globals, "final_norm", {"model.norm.weight"});
  require(topology.globals, "lm_head", {"lm_head.weight"});
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    std::map<std::string, std::string> bindings;
    const std::string hf = "model.layers." + std::to_string(layer) + ".";
    const std::string flat = "layers." + std::to_string(layer) + ".";
    require(bindings, "input_norm", {hf + "input_layernorm.weight"});
    require(bindings, "post_norm", {hf + "post_attention_layernorm.weight"});
    require(bindings, "router", {hf + "mlp.gate.weight"});
    require(bindings, "expert_gate", {hf + "mlp.experts.0.gate_proj.weight"});
    if (has(flat + "hc_attn_base") || has(flat + "hc_ffn_base")) {
      require(bindings, "attention", {flat + "hc_attn_base"});
      require(bindings, "feed_forward", {flat + "hc_ffn_base"});
    } else {
      require(bindings, "attention", {hf + "self_attn.q_proj.weight", hf + "self_attn.q_a_proj.weight",
                                      flat + "attn.wq_a.weight"});
      require(bindings, "attention_output", {hf + "self_attn.o_proj.weight",
                                             hf + "self_attn.o_proj_folded.weight", flat + "attn.wo_b.weight"});
    }
    topology.layers.push_back(std::move(bindings));
  }
  return topology;
}

int64_t DivideUp(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }
size_t Fp8Bytes(int64_t elements) { return static_cast<size_t>(elements); }
size_t Bf16Bytes(int64_t elements) { return static_cast<size_t>(elements) * 2; }
size_t Fp32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int32Bytes(int64_t elements) { return static_cast<size_t>(elements) * 4; }
size_t Int64Bytes(int64_t elements) { return static_cast<size_t>(elements) * 8; }
int64_t DenseScaleCols(int64_t k) { return DivideUp(k, kDenseScaleBlock); }
int64_t MxScaleCols(int64_t k) { return DivideUp(k, kRoutedScaleBlock); }

StaticArenaManager::StaticArenaManager(IDeviceAllocator& allocator, IStreamEngine& streams, const RuntimeConfig& config)
    : allocator_(allocator), streams_(streams), config_(config), arena_(allocator) {
  DSV4_REQUIRE(config_.block_size > 0, "paged block size must be positive");
  DSV4_REQUIRE(config_.max_context_len >= config_.block_size,
               "max context " << config_.max_context_len << " is below one block of " << config_.block_size);
  num_blocks_ = DivideUp(config_.max_context_len, config_.block_size);
  tensors_ = std::make_unique<ArenaTensors>();
  backbone_ = std::make_unique<BackboneWeights>();
  backbone_->layers.resize(static_cast<size_t>(kNumLayers));

  // The compression schedule decides what gets reserved at all, so it is
  // resolved before any reservation rather than per layer.
  uses_csa_ = config_.any_layer_uses(AttentionPath::kCompressedSparse);
  uses_hca_ = config_.any_layer_uses(AttentionPath::kHyperCompressed);
  max_compress_ratio_ = kCompressRatioSwa;
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    max_compress_ratio_ = std::max(max_compress_ratio_, config_.compress_ratio(layer));
  }
  // The compressed hybrid entry is only coherent if its two halves really are
  // this checkpoint's two halves. Checked here, before a single byte is
  // reserved against it, because kCompressedKvEntryBytes sizes the cache and
  // the attention core addresses it with that stride.
  if (uses_compression()) {
    DSV4_REQUIRE(kCompressedKvNopeChannels + kCompressedKvRopeChannels == config_.mla.kv_lora_rank,
                 "the " << kCompressedKvEntryBytes << "-byte compressed KV entry packs "
                        << kCompressedKvNopeChannels << " nope + " << kCompressedKvRopeChannels
                        << " rope channels, which is " << (kCompressedKvNopeChannels + kCompressedKvRopeChannels)
                        << ", but this checkpoint's kv_lora_rank is " << config_.mla.kv_lora_rank
                        << ". Dsv4CompressedKvEntry in kv_cache_layout.hpp has to be re-derived for that "
                           "geometry before any compressed layer can run.");
    DSV4_REQUIRE(kCompressedKvRopeChannels == config_.mla.qk_rope_head_dim,
                 "the compressed KV entry keeps " << kCompressedKvRopeChannels
                                                  << " rope channels at BF16 but this checkpoint's "
                                                     "qk_rope_head_dim is "
                                                  << config_.mla.qk_rope_head_dim);
  }
}

size_t StaticArenaManager::CompressedWindowLayerStrideBytes() const {
  return Bf16Bytes(max_compress_ratio_ * kHiddenSize);
}

size_t StaticArenaManager::CompressedStateLayerStrideBytes() const {
  return Fp32Bytes(kCompressorStateBlocks * kCompressorStateBlockSize * 2 * config_.mla.kv_lora_rank);
}

size_t StaticArenaManager::CompressedKvLayerStrideBytes() const {
  return static_cast<size_t>(compressed_slots()) * static_cast<size_t>(kCompressedKvEntryBytes);
}

size_t StaticArenaManager::IndexerKeyLayerStrideBytes() const {
  return static_cast<size_t>(compressed_slots()) * static_cast<size_t>(kIndexHeadDim);
}

size_t StaticArenaManager::IndexerKeyScaleLayerStrideBytes() const {
  return Fp32Bytes(compressed_slots());
}

StaticArenaManager::~StaticArenaManager() = default;

size_t StaticArenaManager::BackboneDeviceBytes(const MlaGeometry& mla, int64_t block_size,
                                               int64_t max_context_len,
                                               const std::vector<int64_t>& compress_ratios) {
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = kNumAttentionHeads * kv_row;          // folded q_b output
  const int64_t o_input = kNumAttentionHeads * mla.kv_lora_rank;  // folded o_proj input
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;

  size_t per_layer = 0;
  per_layer += Bf16Bytes(kHiddenSize);                                   // input norm
  per_layer += Fp8Bytes(kQLoraRank * kHiddenSize);                       // q_a
  per_layer += Fp8Bytes(kQLoraRank * DenseScaleCols(kHiddenSize));       // q_a scale
  per_layer += Bf16Bytes(kQLoraRank);                                    // q_a norm
  per_layer += Fp8Bytes(q_b_width * kQLoraRank);                         // q_b (folded)
  per_layer += Fp8Bytes(q_b_width * DenseScaleCols(kQLoraRank));         // q_b scale
  per_layer += Fp8Bytes(kv_row * kHiddenSize);                           // kv_a
  per_layer += Fp8Bytes(kv_row * DenseScaleCols(kHiddenSize));           // kv_a scale
  per_layer += Bf16Bytes(mla.kv_lora_rank);                              // kv_a norm
  per_layer += Fp8Bytes(kHiddenSize * o_input);                          // o (folded)
  per_layer += Fp8Bytes(kHiddenSize * DenseScaleCols(o_input));          // o scale
  per_layer += Bf16Bytes(kHiddenSize);                                   // post norm
  per_layer += Bf16Bytes(kNumRoutedExperts * kHiddenSize);               // router
  per_layer += Fp32Bytes(kNumRoutedExperts);                             // router bias
  per_layer += Fp8Bytes(2 * shared_inter * kHiddenSize);                 // shared gate/up
  per_layer += Fp8Bytes(2 * shared_inter * DenseScaleCols(kHiddenSize)); // shared gate/up scale
  per_layer += Fp8Bytes(kHiddenSize * shared_inter);                     // shared down
  per_layer += Fp8Bytes(kHiddenSize * DenseScaleCols(shared_inter));     // shared down scale
  // mHC, unconditional: two hyper-connections per layer (attention, MoE).
  per_layer += 2 * (Fp32Bytes(kMhcMixRows * kMhcMixCols) + Fp32Bytes(kMhcAlphaElements) +
                    Fp32Bytes(kMhcMixRows) + Fp32Bytes(kNhcStreams * kHiddenSize));

  // The compression schedule, resolved the same way the manager resolves it.
  bool uses_csa = false;
  bool uses_hca = false;
  int64_t max_ratio = kCompressRatioSwa;
  for (const int64_t ratio : compress_ratios) {
    switch (AttentionPathForRatio(ratio)) {
      case AttentionPath::kCompressedSparse: uses_csa = true; break;
      case AttentionPath::kHyperCompressed: uses_hca = true; break;
      default: break;
    }
    max_ratio = std::max(max_ratio, ratio);
  }
  if (uses_csa || uses_hca) {
    per_layer += Bf16Bytes(kHiddenSize * mla.kv_lora_rank);  // compressor wkv
    per_layer += Bf16Bytes(kHiddenSize * mla.kv_lora_rank); // compressor wgate
    per_layer += Fp32Bytes(max_ratio * mla.kv_lora_rank);    // positional bias
    per_layer += Fp32Bytes(mla.kv_lora_rank);                // compressor norm gain
  }
  if (uses_csa) {
    per_layer += Bf16Bytes(kQLoraRank * kIndexNumHeads * kIndexHeadDim);  // indexer q
    per_layer += Bf16Bytes(mla.kv_lora_rank * kIndexHeadDim);             // indexer k
    per_layer += Bf16Bytes(kIndexNumHeads);                               // indexer head gains
  }

  const int64_t blocks = DivideUp(max_context_len, block_size);
  size_t total = per_layer * static_cast<size_t>(kNumLayers);
  total += Bf16Bytes(kVocabSize * kHiddenSize) * 2;                // embed_tokens + lm_head
  total += Bf16Bytes(kHiddenSize);                                 // final norm
  total += Bf16Bytes(max_context_len * mla.qk_rope_head_dim) * 2;  // cos / sin tables
  // Paged MLA cache: one latent row and one rope row per token slot, per layer.
  total += Bf16Bytes(blocks * block_size * mla.kv_lora_rank) * kNumLayers;
  total += Bf16Bytes(blocks * block_size * mla.qk_rope_head_dim) * kNumLayers;
  if (uses_csa || uses_hca) {
    // One Dsv4CompressedKvEntry per slot per layer, plus the window ring and
    // the compressor's recurrent pooling state.
    total += static_cast<size_t>(blocks * block_size) *
             static_cast<size_t>(kCompressedKvEntryBytes) * static_cast<size_t>(kNumLayers);
    total += Bf16Bytes(max_ratio * kHiddenSize) * static_cast<size_t>(kNumLayers);
    total += Fp32Bytes(kCompressorStateBlocks * kCompressorStateBlockSize * 2 * mla.kv_lora_rank) *
             static_cast<size_t>(kNumLayers);
  }
  if (uses_csa) {
    total += static_cast<size_t>(blocks * block_size) * static_cast<size_t>(kIndexHeadDim) *
             static_cast<size_t>(kNumLayers);                                  // indexer key cache
    total += Fp32Bytes(blocks * block_size) * static_cast<size_t>(kNumLayers);  // its dequant scales
  }
  return total;
}

void StaticArenaManager::ReserveActivations() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * mla.kv_lora_rank;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  ArenaTensors& t = *tensors_;

  t.h_hidden = arena_.Reserve("act.hidden", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed_fp8", Fp8Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("act.normed_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(kHiddenSize)));
  arena_.Reserve("act.rstd", Fp32Bytes(kTokensPerStep));
  arena_.Reserve("act.q_a", Bf16Bytes(kTokensPerStep * kQLoraRank));
  arena_.Reserve("act.q_a_fp8", Fp8Bytes(kTokensPerStep * kQLoraRank));
  arena_.Reserve("act.q_a_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(kQLoraRank)));
  arena_.Reserve("act.q_b", Bf16Bytes(kTokensPerStep * q_b_width));
  arena_.Reserve("act.kv_a", Bf16Bytes(kTokensPerStep * kv_row));
  arena_.Reserve("act.kv_latent_normed", Bf16Bytes(kTokensPerStep * mla.kv_lora_rank));
  arena_.Reserve("act.attn_out", Bf16Bytes(kTokensPerStep * o_input));
  arena_.Reserve("act.attn_fp8", Fp8Bytes(kTokensPerStep * o_input));
  arena_.Reserve("act.attn_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(o_input)));
  arena_.Reserve("act.softmax_lse", Fp32Bytes(kTokensPerStep * heads_));
  arena_.Reserve("act.proj_out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("moe.router_matmul", Bf16Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.combine_weights", Bf16Bytes(kTokensPerStep * kNumExpertsPerTok));
  arena_.Reserve("moe.router_logits", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.router_softplus", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.router_scores", Fp32Bytes(kTokensPerStep * kNumRoutedExperts));
  arena_.Reserve("moe.gating_weights", Fp32Bytes(kTokensPerStep * kNumExpertsPerTok));
  t.h_gating_indices = arena_.Reserve("moe.gating_indices", Int32Bytes(kTokensPerStep * kNumExpertsPerTok));
  t.h_local_indices = arena_.Reserve("moe.local_indices", Int32Bytes(kTokensPerStep * kNumExpertsPerTok));
  arena_.Reserve("moe.expanded_x", Fp8Bytes(expanded_rows * kHiddenSize));
  arena_.Reserve("moe.expanded_row_idx", Int32Bytes(expanded_rows));
  arena_.Reserve("moe.expanded_scale", Fp8Bytes(expanded_rows * MxScaleCols(kHiddenSize)));
  arena_.Reserve("moe.expanded_weights", Fp32Bytes(expanded_rows));
  t.h_group_list = arena_.Reserve("moe.group_list", Int64Bytes(kNumExpertsPerTok));
  arena_.Reserve("moe.gemm1_raw", Bf16Bytes(expanded_rows * 2 * kMoeIntermediateSize));
  arena_.Reserve("moe.gemm1_out", Fp8Bytes(expanded_rows * kMoeIntermediateSize));
  arena_.Reserve("moe.gemm1_scale", Fp8Bytes(expanded_rows * MxScaleCols(kMoeIntermediateSize)));
  arena_.Reserve("moe.gemm2_out", Bf16Bytes(expanded_rows * kHiddenSize));
  arena_.Reserve("moe.routed_out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("shared.gate_up", Bf16Bytes(kTokensPerStep * 2 * shared_inter));
  arena_.Reserve("shared.act", Bf16Bytes(kTokensPerStep * shared_inter));
  arena_.Reserve("shared.act_fp8", Fp8Bytes(kTokensPerStep * shared_inter));
  arena_.Reserve("shared.act_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(shared_inter)));
  arena_.Reserve("shared.out", Bf16Bytes(kTokensPerStep * kHiddenSize));

  arena_.Reserve("head.final_normed", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("head.logits", Bf16Bytes(kTokensPerStep * kVocabSize));
  t.h_argmax = arena_.Reserve("head.argmax", Int64Bytes(kTokensPerStep));

  backbone_->block_table = arena_.Reserve("kv.block_table", Int32Bytes(kTokensPerStep * num_blocks_));
  t.h_slot_mapping = arena_.Reserve("kv.slot_mapping", Int32Bytes(kTokensPerStep));
  backbone_->slot_mapping = t.h_slot_mapping;

  ReserveMhc();
  ReserveCompression();
}

// The mHC residual stream and the two per-sub-block rounds. Unconditional:
// DSV4-Flash carries four streams in every layer, so this is the activation
// path, not an option.
void StaticArenaManager::ReserveMhc() {
  ArenaTensors& t = *tensors_;
  const int64_t stream_elements = kTokensPerStep * kNhcStreams * kHiddenSize;
  t.h_residual_stream[0] = arena_.Reserve("mhc.stream_a", Bf16Bytes(stream_elements));
  t.h_residual_stream[1] = arena_.Reserve("mhc.stream_b", Bf16Bytes(stream_elements));

  t.mhc_attn.h_h_in = arena_.Reserve("mhc.attn_h_in", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("mhc.attn_h_post", Fp32Bytes(kTokensPerStep * kNhcStreams));
  arena_.Reserve("mhc.attn_h_res", Fp32Bytes(kTokensPerStep * kNhcStreams * kNhcStreams));
  arena_.Reserve("mhc.attn_h_res_sink", Fp32Bytes(kTokensPerStep * kNhcStreams * kNhcStreams));

  t.mhc_moe.h_h_in = arena_.Reserve("mhc.moe_h_in", Bf16Bytes(kTokensPerStep * kHiddenSize));
  arena_.Reserve("mhc.moe_h_post", Fp32Bytes(kTokensPerStep * kNhcStreams));
  arena_.Reserve("mhc.moe_h_res", Fp32Bytes(kTokensPerStep * kNhcStreams * kNhcStreams));
  arena_.Reserve("mhc.moe_h_res_sink", Fp32Bytes(kTokensPerStep * kNhcStreams * kNhcStreams));
}

// Nothing here is reserved on a SWA-only checkpoint: `uses_compression()` is
// false, the pipeline plans no compressed stages, and the ~GB of compressor
// and indexer projections never exists.
void StaticArenaManager::ReserveCompression() {
  if (!uses_compression()) {
    return;
  }
  const int64_t compressed_width = config_.mla.kv_lora_rank;
  BackboneWeights& b = *backbone_;

  b.cmp_window = arena_.Reserve("cmp.window", CompressedWindowLayerStrideBytes() *
                                                  static_cast<size_t>(kNumLayers));
  b.cmp_state_cache = arena_.Reserve("cmp.state_cache", CompressedStateLayerStrideBytes() *
                                                            static_cast<size_t>(kNumLayers));
  // The hybrid cache is sized from Dsv4CompressedKvEntry, never from a
  // (width, dtype) product computed here.
  b.cmp_kv_cache =
      arena_.Reserve("cmp.kv_cache", CompressedKvLayerStrideBytes() * static_cast<size_t>(kNumLayers));
  b.cmp_state_block_table = arena_.Reserve("cmp.state_block_table", Int32Bytes(kCompressorStateBlocks));
  b.cmp_block_table = arena_.Reserve("cmp.block_table", Int32Bytes(kTokensPerStep * num_blocks_));
  b.cmp_slot_mapping = arena_.Reserve("cmp.slot_mapping", Int32Bytes(kTokensPerStep));
  b.cmp_seq_k = arena_.Reserve("cmp.seq_k", Int32Bytes(kTokensPerStep));
  // [0,T] cumulative lengths, seqused, startPos, and original indexer length.
  // One pinned 20-byte update serves all four views at emission.
  b.cmp_window_meta = arena_.Reserve("cmp.window_meta", Int32Bytes(5));
  arena_.Reserve("cmp.kv_out", Bf16Bytes(kCompressorWindowOutputRows * compressed_width));
  arena_.Reserve("cmp.rope_cos", Fp32Bytes(2 * config_.mla.qk_rope_head_dim));
  arena_.Reserve("cmp.rope_sin", Fp32Bytes(2 * config_.mla.qk_rope_head_dim));
  b.cmp_sparse_indices_dense = arena_.Reserve("cmp.sparse_indices_dense", Int32Bytes(kIndexTopK));

  if (!uses_csa_) {
    return;
  }
  arena_.Reserve("idx.metadata", Int32Bytes(kIndexerMetadataElements));
  arena_.Reserve("idx.head_weights", Fp32Bytes(kIndexNumHeads));
  const int64_t indexer_width = kIndexNumHeads * kIndexHeadDim;
  arena_.Reserve("idx.q_bf16", Bf16Bytes(kTokensPerStep * indexer_width));
  arena_.Reserve("idx.q_fp8", Fp8Bytes(kTokensPerStep * indexer_width));
  arena_.Reserve("idx.q_mx_scale", Fp8Bytes(kTokensPerStep * MxScaleCols(indexer_width)));
  arena_.Reserve("idx.k_bf16", Bf16Bytes(kTokensPerStep * kIndexHeadDim));
  b.index_k_cache =
      arena_.Reserve("idx.k_cache", IndexerKeyLayerStrideBytes() * static_cast<size_t>(kNumLayers));
  b.index_k_dequant =
      arena_.Reserve("idx.k_dequant", IndexerKeyScaleLayerStrideBytes() * static_cast<size_t>(kNumLayers));
  b.index_q_dequant = arena_.Reserve("idx.q_dequant", Fp32Bytes(kTokensPerStep * kIndexNumHeads));
  b.index_sparse_indices = arena_.Reserve("idx.sparse_indices", Int32Bytes(kIndexTopK));
}

void StaticArenaManager::ReserveBackbone() {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * mla.kv_lora_rank;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;

  backbone_->embed_tokens = arena_.Reserve("w.embed_tokens", Bf16Bytes(kVocabSize * kHiddenSize));
  backbone_->lm_head = arena_.Reserve("w.lm_head", Bf16Bytes(kVocabSize * kHiddenSize));
  backbone_->final_norm = arena_.Reserve("w.final_norm", Bf16Bytes(kHiddenSize));
  backbone_->rope_cos = arena_.Reserve("w.rope_cos", Bf16Bytes(config_.max_context_len * mla.qk_rope_head_dim));
  backbone_->rope_sin = arena_.Reserve("w.rope_sin", Bf16Bytes(config_.max_context_len * mla.qk_rope_head_dim));
  backbone_->kv_latent_cache = arena_.Reserve(
      "kv.latent_cache", Bf16Bytes(kNumLayers * num_blocks_ * config_.block_size * mla.kv_lora_rank));
  backbone_->kv_rope_cache = arena_.Reserve(
      "kv.rope_cache", Bf16Bytes(kNumLayers * num_blocks_ * config_.block_size * mla.qk_rope_head_dim));

  for (int64_t index = 0; index < kNumLayers; ++index) {
    BackboneWeights::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
    layer.input_norm = arena_.Reserve("w.layer.input_norm", Bf16Bytes(kHiddenSize));
    layer.q_a_weight = arena_.Reserve("w.layer.q_a", Fp8Bytes(kQLoraRank * kHiddenSize));
    layer.q_a_scale = arena_.Reserve("w.layer.q_a_scale", Fp8Bytes(kQLoraRank * DenseScaleCols(kHiddenSize)));
    layer.q_a_norm = arena_.Reserve("w.layer.q_a_norm", Bf16Bytes(kQLoraRank));
    layer.q_b_weight = arena_.Reserve("w.layer.q_b_folded", Fp8Bytes(q_b_width * kQLoraRank));
    layer.q_b_scale = arena_.Reserve("w.layer.q_b_scale", Fp8Bytes(q_b_width * DenseScaleCols(kQLoraRank)));
    layer.kv_a_weight = arena_.Reserve("w.layer.kv_a", Fp8Bytes(kv_row * kHiddenSize));
    layer.kv_a_scale = arena_.Reserve("w.layer.kv_a_scale", Fp8Bytes(kv_row * DenseScaleCols(kHiddenSize)));
    layer.kv_a_norm = arena_.Reserve("w.layer.kv_a_norm", Bf16Bytes(mla.kv_lora_rank));
    layer.o_weight = arena_.Reserve("w.layer.o_folded", Fp8Bytes(kHiddenSize * o_input));
    layer.o_scale = arena_.Reserve("w.layer.o_scale", Fp8Bytes(kHiddenSize * DenseScaleCols(o_input)));
    layer.post_norm = arena_.Reserve("w.layer.post_norm", Bf16Bytes(kHiddenSize));
    layer.router_weight = arena_.Reserve("w.layer.router", Bf16Bytes(kNumRoutedExperts * kHiddenSize));
    layer.router_bias = arena_.Reserve("w.layer.router_bias", Fp32Bytes(kNumRoutedExperts));
    layer.shared_gate_up_weight =
        arena_.Reserve("w.layer.shared_gate_up", Fp8Bytes(2 * shared_inter * kHiddenSize));
    layer.shared_gate_up_scale =
        arena_.Reserve("w.layer.shared_gate_up_scale", Fp8Bytes(2 * shared_inter * DenseScaleCols(kHiddenSize)));
    layer.shared_down_weight = arena_.Reserve("w.layer.shared_down", Fp8Bytes(kHiddenSize * shared_inter));
    layer.shared_down_scale =
        arena_.Reserve("w.layer.shared_down_scale", Fp8Bytes(kHiddenSize * DenseScaleCols(shared_inter)));

    // Two independent hyper-connections per layer, one per sub-block.
    const auto reserve_mhc = [this](BackboneWeights::Layer::MhcWeights* weights, const char* phi_name,
                                    const char* alpha_name, const char* bias_name, const char* gamma_name) {
      weights->phi = arena_.Reserve(phi_name, Fp32Bytes(kMhcMixRows * kMhcMixCols));
      weights->alpha = arena_.Reserve(alpha_name, Fp32Bytes(kMhcAlphaElements));
      weights->bias = arena_.Reserve(bias_name, Fp32Bytes(kMhcMixRows));
      weights->gamma = arena_.Reserve(gamma_name, Fp32Bytes(kNhcStreams * kHiddenSize));
    };
    reserve_mhc(&layer.mhc_attn, "w.layer.mhc_attn_phi", "w.layer.mhc_attn_alpha", "w.layer.mhc_attn_bias",
                "w.layer.mhc_attn_gamma");
    reserve_mhc(&layer.mhc_moe, "w.layer.mhc_moe_phi", "w.layer.mhc_moe_alpha", "w.layer.mhc_moe_bias",
                "w.layer.mhc_moe_gamma");

    if (uses_compression()) {
      layer.cmp_wkv = arena_.Reserve("w.layer.cmp_wkv", Bf16Bytes(kHiddenSize * mla.kv_lora_rank));
      layer.cmp_wgate = arena_.Reserve("w.layer.cmp_wgate", Bf16Bytes(kHiddenSize * mla.kv_lora_rank));
      layer.cmp_ape = arena_.Reserve("w.layer.cmp_ape", Fp32Bytes(max_compress_ratio_ * mla.kv_lora_rank));
      layer.cmp_norm_weight = arena_.Reserve("w.layer.cmp_norm_weight", Fp32Bytes(mla.kv_lora_rank));
    }
    if (uses_csa_) {
      layer.index_q_weight =
          arena_.Reserve("w.layer.index_q", Bf16Bytes(kQLoraRank * kIndexNumHeads * kIndexHeadDim));
      layer.index_k_weight = arena_.Reserve("w.layer.index_k", Bf16Bytes(mla.kv_lora_rank * kIndexHeadDim));
      layer.index_head_weight = arena_.Reserve("w.layer.index_head_weight", Bf16Bytes(kIndexNumHeads));
    }
  }
}

void StaticArenaManager::Commit() { arena_.Commit(); }

void StaticArenaManager::IngestBackbone(WeightByteSource& source) {
  struct Binding {
    ArenaHandle handle;
    const char* pattern;
  };

  // Streamed through a bounded pinned staging buffer, not a plain heap vector:
  // it is the DMA source for every backbone transfer, and page-locked staging
  // is what the transfer path can validate end to end (the exclusive
  // hierarchy's transit scratch takes the same stance). Init-only work, freed
  // before this function returns.
  ResourceScope staging_scope(allocator_, streams_);
  uint8_t* staging = static_cast<uint8_t*>(staging_scope.HostPinnedMalloc(kTransferChunkBytes));
  DSV4_REQUIRE(staging != nullptr, "the backbone staging buffer could not be pinned");

  auto ingest = [&](ArenaHandle handle, const std::string& name, bool required) -> bool {
    const size_t bytes = arena_.Bytes(handle);
    const std::string resolved = ResolveCheckpointTensorName(
        name, [&](const std::string& candidate) { return source.HasNamed(candidate); });
    if (resolved.empty()) {
      DSV4_REQUIRE(!required,
                   "the checkpoint has no tensor '"
                       << name
                       << "'. If that is q_b_proj_latent or o_proj_folded, the checkpoint has not been through "
                          "the MLA absorption pass: this runner consumes q_b with W_UK folded in and o_proj with "
                          "W_UV folded in, because folding block-scaled FP8 at startup would requantize and lose "
                          "accuracy without saying so.");
      return false;
    }
    const size_t available = source.NamedByteSize(resolved);
    DSV4_REQUIRE(available == 0 || available == bytes,
                 "tensor '" << resolved << "' (requested '" << name << "') holds " << available
                             << " bytes, the arena reserved " << bytes);
    uint8_t* destination = arena_.AddressAs<uint8_t>(handle);
    // Dense BF16 weights are stored [N,K] in checkpoints; aclnnMatmul has no
    // transpose flag. Materialize [K,N] once before creating the descriptors.
    const bool head = name == "lm_head.weight";
    const bool router = name.find(".mlp.gate.weight") != std::string::npos;
    if (head || router) {
      IngestTransposedBf16(source, resolved, destination, head ? kVocabSize : kNumRoutedExperts,
                           kHiddenSize, allocator_, streams_);
      return true;
    }
    for (size_t offset = 0; offset < bytes; offset += kTransferChunkBytes) {
      const size_t count = std::min(kTransferChunkBytes, bytes - offset);
      source.ReadNamed(resolved, staging, kTransferChunkBytes, offset, count);
      streams_.MemcpySync(destination + offset, bytes - offset, staging, count, MemcpyKind::kHostToDevice);
    }
    return true;
  };

  ingest(backbone_->embed_tokens, "model.embed_tokens.weight", true);
  ingest(backbone_->lm_head, "lm_head.weight", true);
  ingest(backbone_->final_norm, "model.norm.weight", true);

  // "Populated" has to mean ALL of them: a checkpoint that ships phi but not
  // alpha would mix the streams with a zero gain, which is not a partial
  // result, it is a wrong one.
  bool mhc_present = false;
  bool mhc_missing = false;

  for (int64_t index = 0; index < kNumLayers; ++index) {
    const BackboneWeights::Layer& layer = backbone_->layers[static_cast<size_t>(index)];
    const Binding bindings[] = {
        {layer.input_norm, "model.layers.{L}.input_layernorm.weight"},
        {layer.q_a_weight, "model.layers.{L}.self_attn.q_a_proj.weight"},
        {layer.q_a_scale, "model.layers.{L}.self_attn.q_a_proj.weight_scale_inv"},
        {layer.q_a_norm, "model.layers.{L}.self_attn.q_a_layernorm.weight"},
        {layer.q_b_weight, kFoldedQName},
        {layer.q_b_scale, kFoldedQScaleName},
        {layer.kv_a_weight, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"},
        {layer.kv_a_scale, "model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight_scale_inv"},
        {layer.kv_a_norm, "model.layers.{L}.self_attn.kv_a_layernorm.weight"},
        {layer.o_weight, kFoldedOName},
        {layer.o_scale, kFoldedOScaleName},
        {layer.post_norm, "model.layers.{L}.post_attention_layernorm.weight"},
        {layer.router_weight, "model.layers.{L}.mlp.gate.weight"},
        {layer.router_bias, "model.layers.{L}.mlp.gate.e_score_correction_bias"},
        {layer.shared_gate_up_weight, "model.layers.{L}.mlp.shared_experts.gate_up_proj.weight"},
        {layer.shared_gate_up_scale, "model.layers.{L}.mlp.shared_experts.gate_up_proj.weight_scale_inv"},
        {layer.shared_down_weight, "model.layers.{L}.mlp.shared_experts.down_proj.weight"},
        {layer.shared_down_scale, "model.layers.{L}.mlp.shared_experts.down_proj.weight_scale_inv"},
    };
    for (const Binding& binding : bindings) {
      ingest(binding.handle, LayerTensorName(binding.pattern, index), true);
    }

    // mHC is NOT required of the checkpoint: a conversion that predates the
    // hyper-connection export would otherwise be unloadable, and the report
    // below says plainly when the weights are missing.
    const Binding mhc_bindings[] = {
        {layer.mhc_attn.phi, "model.layers.{L}.self_attn.hc.phi"},
        {layer.mhc_attn.alpha, "model.layers.{L}.self_attn.hc.alpha"},
        {layer.mhc_attn.bias, "model.layers.{L}.self_attn.hc.bias"},
        {layer.mhc_attn.gamma, "model.layers.{L}.self_attn.hc.gamma"},
        {layer.mhc_moe.phi, "model.layers.{L}.mlp.hc.phi"},
        {layer.mhc_moe.alpha, "model.layers.{L}.mlp.hc.alpha"},
        {layer.mhc_moe.bias, "model.layers.{L}.mlp.hc.bias"},
        {layer.mhc_moe.gamma, "model.layers.{L}.mlp.hc.gamma"},
    };
    for (const Binding& binding : mhc_bindings) {
      if (ingest(binding.handle, LayerTensorName(binding.pattern, index), false)) {
        mhc_present = true;
      } else {
        mhc_missing = true;
      }
    }

    // The compressor and indexer projections, on the other hand, ARE required
    // of a checkpoint that marks this layer compressed: it asked for the path,
    // so the weights that path reads have to be there.
    if (uses_compression()) {
      const Binding cmp_bindings[] = {
          {layer.cmp_wkv, "model.layers.{L}.self_attn.compressor.wkv.weight"},
          {layer.cmp_wgate, "model.layers.{L}.self_attn.compressor.wgate.weight"},
          {layer.cmp_ape, "model.layers.{L}.self_attn.compressor.ape"},
          {layer.cmp_norm_weight, "model.layers.{L}.self_attn.compressor.norm.weight"},
      };
      for (const Binding& binding : cmp_bindings) {
        ingest(binding.handle, LayerTensorName(binding.pattern, index), false);
      }
    }
    if (uses_csa_) {
      const Binding index_bindings[] = {
          {layer.index_q_weight, "model.layers.{L}.self_attn.indexer.wq.weight"},
          {layer.index_k_weight, "model.layers.{L}.self_attn.indexer.wk.weight"},
          {layer.index_head_weight, "model.layers.{L}.self_attn.indexer.weights_proj.weight"},
      };
      for (const Binding& binding : index_bindings) {
        ingest(binding.handle, LayerTensorName(binding.pattern, index), false);
      }
    }
  }

  // The rope tables are derived, not stored. A checkpoint that ships them is
  // honoured; otherwise they stay zeroed and the report says the rope tables
  // were not populated, because attending with cos=0 is a wrong answer with no
  // symptom.
  const bool cos = ingest(backbone_->rope_cos, "model.rotary_emb.cos_cached", false);
  const bool sin = ingest(backbone_->rope_sin, "model.rotary_emb.sin_cached", false);
  backbone_->rope_tables_populated = cos && sin;
  backbone_->mhc_weights_populated = mhc_present && !mhc_missing;
}

void* StaticArenaManager::ReservationAddress(const char* name) const {
  for (size_t index = 0; index < arena_.reservations().size(); ++index) {
    if (std::strcmp(arena_.reservations()[index].name, name) == 0) {
      return arena_.Address(index);
    }
  }
  throw Dsv4Error(std::string("no arena reservation named ") + name);
}

void StaticArenaManager::CreateDescriptors(const ExpertSlotLayout& slots, const ExpertSlotAddresses& experts) {
  const MlaGeometry& mla = config_.mla;
  const int64_t kv_lora = mla.kv_lora_rank;
  const int64_t rope = mla.qk_rope_head_dim;
  const int64_t kv_row = mla.kv_row_elements();
  const int64_t q_b_width = heads_ * kv_row;
  const int64_t o_input = heads_ * kv_lora;
  const int64_t shared_inter = kMoeIntermediateSize * kNumSharedExperts;
  const int64_t expanded_rows = kTokensPerStep * kNumExpertsPerTok;
  ArenaTensors& t = *tensors_;
  BackboneWeights& b = *backbone_;

  // Reservations are looked up by name here rather than threaded through as
  // handles: every buffer is created exactly once and the name is the same
  // string literal the reservation used.
  auto address = [&](const char* name) -> void* { return ReservationAddress(name); };

  uint8_t* q_b_base = static_cast<uint8_t*>(address("act.q_b"));
  uint8_t* kv_a_base = static_cast<uint8_t*>(address("act.kv_a"));

  // --- activations -------------------------------------------------------
  t.hidden = arena_.CreateTensor("hidden", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.hidden"));
  t.normed = arena_.CreateTensor("normed", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.normed"));
  t.normed_fp8 =
      arena_.CreateTensor("normed_fp8", {kTokensPerStep, kHiddenSize}, kAclFloat8E4m3Fn, address("act.normed_fp8"));
  t.normed_mx_scale = arena_.CreateTensor("normed_mx_scale", {kTokensPerStep, MxScaleCols(kHiddenSize)}, kScaleDtype,
                                          address("act.normed_mx_scale"));
  t.rstd = arena_.CreateTensor("rstd", {kTokensPerStep, 1}, kAclFloat32, address("act.rstd"));
  t.q_a = arena_.CreateTensor("q_a", {kTokensPerStep, kQLoraRank}, kAclBf16, address("act.q_a"));
  t.q_a_fp8 = arena_.CreateTensor("q_a_fp8", {kTokensPerStep, kQLoraRank}, kAclFloat8E4m3Fn, address("act.q_a_fp8"));
  t.q_a_mx_scale = arena_.CreateTensor("q_a_mx_scale", {kTokensPerStep, MxScaleCols(kQLoraRank)}, kScaleDtype,
                                       address("act.q_a_mx_scale"));
  t.q_b = arena_.CreateTensor("q_b", {kTokensPerStep, q_b_width}, kAclBf16, q_b_base);
  // q_b is laid out per head as [kv_lora | rope], so the latent and the rope
  // slice are views at a stride of kv_row, not contiguous halves.
  t.q_latent = arena_.CreateTensor("q_latent", {kTokensPerStep, heads_, kv_lora}, kAclBf16, q_b_base);
  t.q_rope =
      arena_.CreateTensor("q_rope", {kTokensPerStep, heads_, rope}, kAclBf16, q_b_base + Bf16Bytes(kv_lora));
  t.kv_a = arena_.CreateTensor("kv_a", {kTokensPerStep, kv_row}, kAclBf16, kv_a_base);
  t.kv_latent = arena_.CreateTensor("kv_latent", {kTokensPerStep, kv_lora}, kAclBf16, kv_a_base);
  t.k_rope = arena_.CreateTensor("k_rope", {kTokensPerStep, 1, rope}, kAclBf16, kv_a_base + Bf16Bytes(kv_lora));
  t.kv_latent_normed =
      arena_.CreateTensor("kv_latent_normed", {kTokensPerStep, kv_lora}, kAclBf16, address("act.kv_latent_normed"));
  t.attn_out = arena_.CreateTensor("attn_out", {kTokensPerStep, heads_, kv_lora}, kAclBf16, address("act.attn_out"));
  t.attn_flat = arena_.CreateTensor("attn_flat", {kTokensPerStep, o_input}, kAclBf16, address("act.attn_out"));
  t.attn_fp8 = arena_.CreateTensor("attn_fp8", {kTokensPerStep, o_input}, kAclFloat8E4m3Fn, address("act.attn_fp8"));
  t.attn_mx_scale = arena_.CreateTensor("attn_mx_scale", {kTokensPerStep, MxScaleCols(o_input)}, kScaleDtype,
                                        address("act.attn_mx_scale"));
  t.softmax_lse =
      arena_.CreateTensor("softmax_lse", {kTokensPerStep, heads_, 1}, kAclFloat32, address("act.softmax_lse"));
  t.proj_out = arena_.CreateTensor("proj_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("act.proj_out"));

  // --- routing -----------------------------------------------------------
  t.router_matmul = arena_.CreateTensor("router_matmul", {kTokensPerStep, kNumRoutedExperts}, kAclBf16, address("moe.router_matmul"));
  t.combine_weights = arena_.CreateTensor("combine_weights", {kTokensPerStep, expanded_rows}, kAclBf16, address("moe.combine_weights"));
  t.router_logits = arena_.CreateTensor("router_logits", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                        address("moe.router_logits"));
  t.router_softplus = arena_.CreateTensor("router_softplus", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                          address("moe.router_softplus"));
  t.router_scores = arena_.CreateTensor("router_scores", {kTokensPerStep, kNumRoutedExperts}, kAclFloat32,
                                        address("moe.router_scores"));
  {
    // The host scalars must carry fp32 bytes: the constants are double, and
    // passing a double's bit pattern as an ACL_FLOAT scalar would silently
    // change softplus's beta/threshold.
    const float beta = static_cast<float>(kSoftplusBeta);
    const float threshold = static_cast<float>(kSoftplusThreshold);
    t.softplus_beta = arena_.CreateScalar("softplus_beta", kAclFloat32, &beta);
    t.softplus_threshold = arena_.CreateScalar("softplus_threshold", kAclFloat32, &threshold);
  }
  t.gating_weights = arena_.CreateTensor("gating_weights", {kTokensPerStep, kNumExpertsPerTok}, kAclFloat32,
                                         address("moe.gating_weights"));
  t.gating_indices = arena_.CreateTensor("gating_indices", {kTokensPerStep, kNumExpertsPerTok}, kAclInt32,
                                         address("moe.gating_indices"));
  t.local_indices = arena_.CreateTensor("local_indices", {kTokensPerStep, kNumExpertsPerTok}, kAclInt32,
                                        address("moe.local_indices"));
  t.expanded_x =
      arena_.CreateTensor("expanded_x", {expanded_rows, kHiddenSize}, kAclFloat8E4m3Fn, address("moe.expanded_x"));
  t.expanded_row_idx =
      arena_.CreateTensor("expanded_row_idx", {expanded_rows}, kAclInt32, address("moe.expanded_row_idx"));
  t.expanded_scale = arena_.CreateTensor("expanded_scale", {expanded_rows, MxScaleCols(kHiddenSize)}, kScaleDtype,
                                         address("moe.expanded_scale"));
  t.expanded_weights =
      arena_.CreateTensor("expanded_weights", {expanded_rows}, kAclFloat32, address("moe.expanded_weights"));
  t.expanded_weights_row = arena_.CreateTensor("expanded_weights_row", {kTokensPerStep, expanded_rows},
                                               kAclFloat32, address("moe.expanded_weights"));
  t.group_list = arena_.CreateTensor("group_list", {kNumExpertsPerTok}, kAclInt64, address("moe.group_list"));
  t.gemm1_raw = arena_.CreateTensor("gemm1_raw", {expanded_rows, 2 * kMoeIntermediateSize}, kAclBf16,
                                    address("moe.gemm1_raw"));
  t.gemm1_out = arena_.CreateTensor("gemm1_out", {expanded_rows, kMoeIntermediateSize}, kAclFloat8E4m3Fn,
                                    address("moe.gemm1_out"));
  t.gemm1_scale = arena_.CreateTensor("gemm1_scale", {expanded_rows, MxScaleCols(kMoeIntermediateSize)},
                                      kScaleDtype, address("moe.gemm1_scale"));
  t.gemm2_out = arena_.CreateTensor("gemm2_out", {expanded_rows, kHiddenSize}, kAclBf16, address("moe.gemm2_out"));
  t.routed_out =
      arena_.CreateTensor("routed_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("moe.routed_out"));
  t.gemm1_x_list = arena_.CreateTensorList("gemm1_x_list", {t.expanded_x});
  t.gemm1_x_scale_list = arena_.CreateTensorList("gemm1_x_scale_list", {t.expanded_scale});
  t.gemm1_raw_out_list = arena_.CreateTensorList("gemm1_raw_out_list", {t.gemm1_raw});
  t.gemm2_x_list = arena_.CreateTensorList("gemm2_x_list", {t.gemm1_out});
  t.gemm2_x_scale_list = arena_.CreateTensorList("gemm2_x_scale_list", {t.gemm1_scale});
  t.gemm2_out_list = arena_.CreateTensorList("gemm2_out_list", {t.gemm2_out});

  // --- shared expert -----------------------------------------------------
  t.shared_gate_up =
      arena_.CreateTensor("shared_gate_up", {kTokensPerStep, 2 * shared_inter}, kAclBf16, address("shared.gate_up"));
  t.shared_act = arena_.CreateTensor("shared_act", {kTokensPerStep, shared_inter}, kAclBf16, address("shared.act"));
  t.shared_act_fp8 = arena_.CreateTensor("shared_act_fp8", {kTokensPerStep, shared_inter}, kAclFloat8E4m3Fn,
                                         address("shared.act_fp8"));
  t.shared_act_scale = arena_.CreateTensor("shared_act_scale", {kTokensPerStep, MxScaleCols(shared_inter)},
                                           kScaleDtype, address("shared.act_mx_scale"));
  t.shared_out = arena_.CreateTensor("shared_out", {kTokensPerStep, kHiddenSize}, kAclBf16, address("shared.out"));

  // --- head --------------------------------------------------------------
  t.final_normed =
      arena_.CreateTensor("final_normed", {kTokensPerStep, kHiddenSize}, kAclBf16, address("head.final_normed"));
  t.logits = arena_.CreateTensor("logits", {kTokensPerStep, kVocabSize}, kAclBf16, address("head.logits"));
  t.argmax = arena_.CreateTensor("argmax", {kTokensPerStep}, kAclInt64, address("head.argmax"));

  // --- paged KV ----------------------------------------------------------
  // One cache per layer, carved out of one reservation. FIA V5 reads the
  // compressed latent through `key`/`value` and the rope slice through
  // `keyRope`, which is the native MLA input set.
  uint8_t* latent_cache_base = arena_.AddressAs<uint8_t>(b.kv_latent_cache);
  uint8_t* rope_cache_base = arena_.AddressAs<uint8_t>(b.kv_rope_cache);
  t.kv_latent_cache = arena_.CreateTensor("kv_latent_cache", {num_blocks_, config_.block_size, 1, kv_lora},
                                          kAclBf16, latent_cache_base);
  t.kv_rope_cache =
      arena_.CreateTensor("kv_rope_cache", {num_blocks_, config_.block_size, 1, rope}, kAclBf16, rope_cache_base);
  t.key_rope_cache_view = arena_.CreateTensor("key_rope_cache_view", {num_blocks_, config_.block_size, 1, rope},
                                              kAclBf16, rope_cache_base);
  t.key_list = arena_.CreateTensorList("fia_key_list", {t.kv_latent_cache});
  t.value_list = arena_.CreateTensorList("fia_value_list", {t.kv_latent_cache});
  t.block_table =
      arena_.CreateTensor("block_table", {kTokensPerStep, num_blocks_}, kAclInt32, arena_.Address(b.block_table));
  t.slot_mapping = arena_.CreateTensor("slot_mapping", {kTokensPerStep}, kAclInt32, arena_.Address(b.slot_mapping));
  // TND with one query token. The kv length is the reserved context: FIA
  // derives the live length from the block table, and `kv_padding_size` is left
  // null, so neither array changes per step and both are built once.
  t.actual_seq_q = arena_.CreateIntArray("actual_seq_q", {kTokensPerStep});
  t.actual_seq_kv = arena_.CreateIntArray("actual_seq_kv", {config_.max_context_len});

  t.rope_cos = arena_.CreateTensor("rope_cos", {kTokensPerStep, rope}, kAclBf16, arena_.Address(b.rope_cos));
  t.rope_sin = arena_.CreateTensor("rope_sin", {kTokensPerStep, rope}, kAclBf16, arena_.Address(b.rope_sin));

  // --- layer weights, pointed at layer 0 ---------------------------------
  const BackboneWeights::Layer& first = b.layers[0];
  t.w_input_norm = arena_.CreateTensor("w_input_norm", {kHiddenSize}, kAclBf16, arena_.Address(first.input_norm));
  t.w_q_a =
      arena_.CreateTensor("w_q_a", {kQLoraRank, kHiddenSize}, kAclFloat8E4m3Fn, arena_.Address(first.q_a_weight));
  t.w_q_a_scale = arena_.CreateTensor("w_q_a_scale", {kQLoraRank, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                                      arena_.Address(first.q_a_scale));
  t.w_q_a_norm = arena_.CreateTensor("w_q_a_norm", {kQLoraRank}, kAclBf16, arena_.Address(first.q_a_norm));
  t.w_q_b =
      arena_.CreateTensor("w_q_b", {q_b_width, kQLoraRank}, kAclFloat8E4m3Fn, arena_.Address(first.q_b_weight));
  t.w_q_b_scale = arena_.CreateTensor("w_q_b_scale", {q_b_width, DenseScaleCols(kQLoraRank)}, kScaleDtype,
                                      arena_.Address(first.q_b_scale));
  t.w_kv_a =
      arena_.CreateTensor("w_kv_a", {kv_row, kHiddenSize}, kAclFloat8E4m3Fn, arena_.Address(first.kv_a_weight));
  t.w_kv_a_scale = arena_.CreateTensor("w_kv_a_scale", {kv_row, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                                       arena_.Address(first.kv_a_scale));
  t.w_kv_a_norm = arena_.CreateTensor("w_kv_a_norm", {kv_lora}, kAclBf16, arena_.Address(first.kv_a_norm));
  t.w_o = arena_.CreateTensor("w_o", {kHiddenSize, o_input}, kAclFloat8E4m3Fn, arena_.Address(first.o_weight));
  t.w_o_scale = arena_.CreateTensor("w_o_scale", {kHiddenSize, DenseScaleCols(o_input)}, kScaleDtype,
                                    arena_.Address(first.o_scale));
  t.w_post_norm = arena_.CreateTensor("w_post_norm", {kHiddenSize}, kAclBf16, arena_.Address(first.post_norm));
  t.w_router = arena_.CreateTensor("w_router", {kHiddenSize, kNumRoutedExperts}, kAclBf16,
                                   arena_.Address(first.router_weight));
  t.w_router_bias =
      arena_.CreateTensor("w_router_bias", {kNumRoutedExperts}, kAclFloat32, arena_.Address(first.router_bias));
  t.w_shared_gate_up = arena_.CreateTensor("w_shared_gate_up", {2 * shared_inter, kHiddenSize}, kAclFloat8E4m3Fn,
                                           arena_.Address(first.shared_gate_up_weight));
  t.w_shared_gate_up_scale =
      arena_.CreateTensor("w_shared_gate_up_scale", {2 * shared_inter, DenseScaleCols(kHiddenSize)}, kScaleDtype,
                          arena_.Address(first.shared_gate_up_scale));
  t.w_shared_down = arena_.CreateTensor("w_shared_down", {kHiddenSize, shared_inter}, kAclFloat8E4m3Fn,
                                        arena_.Address(first.shared_down_weight));
  t.w_shared_down_scale =
      arena_.CreateTensor("w_shared_down_scale", {kHiddenSize, DenseScaleCols(shared_inter)}, kScaleDtype,
                          arena_.Address(first.shared_down_scale));
  t.w_final_norm = arena_.CreateTensor("w_final_norm", {kHiddenSize}, kAclBf16, arena_.Address(b.final_norm));
  t.w_lm_head = arena_.CreateTensor("w_lm_head", {kHiddenSize, kVocabSize}, kAclBf16, arena_.Address(b.lm_head));

  // --- the six active experts -------------------------------------------
  // Views into the exclusive manager's HBM slot pool, repointed every layer by
  // aclSetDynamicTensorAddr. They start pointed at slots 0..5 so the plan phase
  // sees real, resident addresses.
  const ExpertRegionSpec& gate_up = slots.region(ExpertRegionId::kGateUpWeight);
  const ExpertRegionSpec& gate_up_scale = slots.region(ExpertRegionId::kGateUpScale);
  const ExpertRegionSpec& down = slots.region(ExpertRegionId::kDownWeight);
  const ExpertRegionSpec& down_scale = slots.region(ExpertRegionId::kDownScale);
  for (int64_t index = 0; index < kNumExpertsPerTok; ++index) {
    const size_t slot = static_cast<size_t>(index);
    t.expert_gate_up.push_back(
        arena_.CreateFp4Tensor("expert_gate_up", {gate_up.rows, gate_up.cols}, experts.gate_up_weight[slot]));
    t.expert_gate_up_scale.push_back(arena_.CreateTensor(
        "expert_gate_up_scale", {gate_up_scale.rows, static_cast<int64_t>(gate_up_scale.stored_cols())},
        kScaleDtype, experts.gate_up_scale[slot]));
    t.expert_down.push_back(
        arena_.CreateFp4Tensor("expert_down", {down.rows, down.cols}, experts.down_weight[slot]));
    t.expert_down_scale.push_back(arena_.CreateTensor(
        "expert_down_scale", {down_scale.rows, static_cast<int64_t>(down_scale.stored_cols())}, kScaleDtype,
        experts.down_scale[slot]));
  }
  t.expert_gate_up_list = arena_.CreateTensorList("expert_gate_up_list", t.expert_gate_up);
  t.expert_gate_up_scale_list = arena_.CreateTensorList("expert_gate_up_scale_list", t.expert_gate_up_scale);
  t.expert_down_list = arena_.CreateTensorList("expert_down_list", t.expert_down);
  t.expert_down_scale_list = arena_.CreateTensorList("expert_down_scale_list", t.expert_down_scale);

  CreateMhcDescriptors();
  CreateCompressionDescriptors();
  SeedStaticTables();
}

// ---------------------------------------------------------------------------
// mHC descriptors: the uniform BSND spelling
// ---------------------------------------------------------------------------
//
// Pure TND, at the ranks the 950PR run proved Repeatable=true for both
// `aclnnMhcPre` and `aclnnMhcPost`: rank 3 for x / hRes / B_l / out, rank 2
// for hIn / hOut / hPost. Nothing is mixed, and nothing is reshaped -- hIn
// comes out at [1, 4096], which is exactly what RMSNorm, the attention
// projections and the MoE router take, so it is handed to them as it is.
void StaticArenaManager::CreateMhcDescriptors() {
  ArenaTensors& t = *tensors_;
  const int64_t tokens = kTokensPerStep;

  static const char* const kStreamLabels[2] = {"mhc_stream_a", "mhc_stream_b"};
  static const char* const kSliceLabels[2][kNhcStreams] = {
      {"mhc_stream_a.s0", "mhc_stream_a.s1", "mhc_stream_a.s2", "mhc_stream_a.s3"},
      {"mhc_stream_b.s0", "mhc_stream_b.s1", "mhc_stream_b.s2", "mhc_stream_b.s3"},
  };
  for (size_t buffer = 0; buffer < 2; ++buffer) {
    uint8_t* base = arena_.AddressAs<uint8_t>(t.h_residual_stream[buffer]);
    // [T, n, D], contiguous -- which is also what makes each stream slice
    // below a plain byte offset rather than a strided view.
    t.residual_stream[buffer] =
        arena_.CreateTensor(kStreamLabels[buffer], {tokens, kNhcStreams, kHiddenSize}, kAclBf16, base);
    static const char* const kStreamBshdLabels[2] = {"mhc_stream_a.bshd", "mhc_stream_b.bshd"};
    t.residual_stream_bshd[buffer] = arena_.CreateTensor(kStreamBshdLabels[buffer],
                                                         {tokens, 1, kNhcStreams, kHiddenSize}, kAclBf16, base);
    for (int64_t stream = 0; stream < kNhcStreams; ++stream) {
      t.stream_slice[buffer][stream] =
          arena_.CreateTensor(kSliceLabels[buffer][static_cast<size_t>(stream)], {tokens, kHiddenSize}, kAclBf16,
                              base + Bf16Bytes(stream * kHiddenSize));
    }
  }

  struct RoundLabels {
    const char* h_in;
    const char* h_post;
    const char* h_res;
    const char* h_res_sink;
    const char* h_out;
    const char* h_post_bshd;
    const char* b_l_bshd;
    const char* h_out_bshd;
    const char* h_in_reservation;
    const char* h_post_reservation;
    const char* h_res_reservation;
    const char* h_res_sink_reservation;
    // The sub-block output buffer h_out describes: the attention round folds
    // back o_proj's output, the MoE round folds back the routed + shared sum.
    // Neither is copied -- h_out is a descriptor over bytes that already exist.
    const char* h_out_reservation;
  };
  const RoundLabels rounds[2] = {
      {"mhc_attn_h_in", "mhc_attn_h_post", "mhc_attn_h_res", "mhc_attn_h_res_sink", "mhc_attn_h_out",
       "mhc_attn_h_post.bshd", "mhc_attn_b_l.bshd", "mhc_attn_h_out.bshd",
       "mhc.attn_h_in", "mhc.attn_h_post", "mhc.attn_h_res", "mhc.attn_h_res_sink", "act.proj_out"},
      {"mhc_moe_h_in", "mhc_moe_h_post", "mhc_moe_h_res", "mhc_moe_h_res_sink", "mhc_moe_h_out",
       "mhc_moe_h_post.bshd", "mhc_moe_b_l.bshd", "mhc_moe_h_out.bshd",
       "mhc.moe_h_in", "mhc.moe_h_post", "mhc.moe_h_res", "mhc.moe_h_res_sink", "moe.routed_out"},
  };
  MhcRoundTensors* targets[2] = {&t.mhc_attn, &t.mhc_moe};
  for (size_t round = 0; round < 2; ++round) {
    const RoundLabels& labels = rounds[round];
    MhcRoundTensors& set = *targets[round];
    set.h_in = arena_.CreateTensor(labels.h_in, {tokens, kHiddenSize}, kAclBf16,
                                   ReservationAddress(labels.h_in_reservation));
    set.h_post = arena_.CreateTensor(labels.h_post, {tokens, kNhcStreams}, kAclFloat32,
                                     ReservationAddress(labels.h_post_reservation));
    set.h_res = arena_.CreateTensor(labels.h_res, {tokens, kNhcStreams, kNhcStreams}, kAclFloat32,
                                    ReservationAddress(labels.h_res_reservation));
    // CONTIGUOUS, and that is now the ONLY admissible form: the 950PR run
    // refused a non-contiguous Sinkhorn output with 561103, so the gapped-view
    // workaround the CANN 4.31 note once recommended is simply unavailable.
    // CreateTensor emits contiguous strides, which for [1, 4, 4] FP32 is
    // [16, 4, 1].
    set.h_res_sink = arena_.CreateTensor(labels.h_res_sink, {tokens, kNhcStreams, kNhcStreams}, kAclFloat32,
                                         ReservationAddress(labels.h_res_sink_reservation));
    set.h_out = arena_.CreateTensor(labels.h_out, {tokens, kHiddenSize}, kAclBf16,
                                    ReservationAddress(labels.h_out_reservation));
    // The BSHD relabels HcPost requires, over the very same addresses.
    set.h_post_bshd = arena_.CreateTensor(labels.h_post_bshd, {tokens, 1, kNhcStreams}, kAclFloat32,
                                          ReservationAddress(labels.h_post_reservation));
    set.b_l_bshd = arena_.CreateTensor(labels.b_l_bshd, {tokens, 1, kNhcStreams, kNhcStreams}, kAclFloat32,
                                       ReservationAddress(labels.h_res_sink_reservation));
    set.h_out_bshd = arena_.CreateTensor(labels.h_out_bshd, {tokens, 1, kHiddenSize}, kAclBf16,
                                         ReservationAddress(labels.h_out_reservation));
  }

  // Pointed at layer 0's attention round; repointed per layer and per round.
  const BackboneWeights::Layer::MhcWeights& first = backbone_->layers[0].mhc_attn;
  t.w_mhc_phi = arena_.CreateTensor("w_mhc_phi", {kMhcMixRows, kMhcMixCols}, kAclFloat32,
                                    arena_.Address(first.phi));
  t.w_mhc_alpha = arena_.CreateTensor("w_mhc_alpha", {kMhcAlphaElements}, kAclFloat32,
                                      arena_.Address(first.alpha));
  t.w_mhc_bias = arena_.CreateTensor("w_mhc_bias", {kMhcMixRows}, kAclFloat32, arena_.Address(first.bias));
  t.w_mhc_gamma = arena_.CreateTensor("w_mhc_gamma", {kNhcStreams, kHiddenSize}, kAclFloat32,
                                      arena_.Address(first.gamma));
}

// ---------------------------------------------------------------------------
// Compression and sparse-attention descriptors
// ---------------------------------------------------------------------------
//
// Every per-layer tensor is created over LAYER 0's slice and repointed by the
// decode loop, exactly like the dense weights. The paged caches get two
// descriptors over one allocation where two operators disagree on the
// spelling -- the hybrid KV cache is FP8 [blocks, block_size, 604] for both
// the epilog and the attention core, the indexer key cache is UINT8
// [blocks, block_size, 128] for its epilog and FP8 [blocks, block_size, 1,
// 128] for the indexer -- so no byte is ever copied to change rank or dtype.
void StaticArenaManager::CreateCompressionDescriptors() {
  if (!uses_compression()) {
    return;
  }
  ArenaTensors& t = *tensors_;
  BackboneWeights& b = *backbone_;
  const int64_t tokens = kTokensPerStep;
  const int64_t width = config_.mla.kv_lora_rank;
  const int64_t rope = config_.mla.qk_rope_head_dim;
  const int64_t slots = compressed_slots();

  uint8_t* window_base = arena_.AddressAs<uint8_t>(b.cmp_window);
  void* rope_cos_base = ReservationAddress("cmp.rope_cos");
  void* rope_sin_base = ReservationAddress("cmp.rope_sin");
  // Window-length views. The compressor takes x as [T, H] with ropeSin /
  // ropeCos at the rank of x, so these are the rank-2 TND spelling -- the
  // operator admits no rank-4 x (mock and wrapper both cap it at [B, S, H]),
  // which is why the mHC graph's BSND rule stops at the mHC boundary.
  if (uses_csa_) {
    t.cmp_window_csa =
        arena_.CreateTensor("cmp_window_csa", {kCompressRatioCsa, kHiddenSize}, kAclBf16, window_base);
    t.cmp_rope_cos_csa = arena_.CreateTensor("cmp_rope_cos_csa", {kCompressorWindowOutputRows, rope}, kAclFloat32,
                                             rope_cos_base);
    t.cmp_rope_sin_csa = arena_.CreateTensor("cmp_rope_sin_csa", {kCompressorWindowOutputRows, rope}, kAclFloat32,
                                             rope_sin_base);
  }
  if (uses_hca_) {
    t.cmp_window_hca =
        arena_.CreateTensor("cmp_window_hca", {kCompressRatioHca, kHiddenSize}, kAclBf16, window_base);
    t.cmp_rope_cos_hca = arena_.CreateTensor("cmp_rope_cos_hca", {kCompressorWindowOutputRows, rope}, kAclFloat32,
                                             rope_cos_base);
    t.cmp_rope_sin_hca = arena_.CreateTensor("cmp_rope_sin_hca", {kCompressorWindowOutputRows, rope}, kAclFloat32,
                                             rope_sin_base);
  }

  t.cmp_state_cache = arena_.CreateTensor("cmp_state_cache",
                                          {kCompressorStateBlocks, kCompressorStateBlockSize, 2 * width},
                                          kAclFloat32, arena_.Address(b.cmp_state_cache));
  t.cmp_state_block_table = arena_.CreateTensor("cmp_state_block_table", {tokens, kCompressorStateBlocks},
                                                kAclInt32, arena_.Address(b.cmp_state_block_table));
  uint8_t* meta_base = static_cast<uint8_t*>(ReservationAddress("cmp.window_meta"));
  t.cmp_cu_seqlens = arena_.CreateTensor("cmp_cu_seqlens", {tokens + 1}, kAclInt32, meta_base);
  t.cmp_seqused = arena_.CreateTensor("cmp_seqused", {tokens}, kAclInt32, meta_base + Int32Bytes(2));
  t.cmp_start_pos = arena_.CreateTensor("cmp_start_pos", {tokens}, kAclInt32, meta_base + Int32Bytes(3));

  t.cmp_rope_cos_row = arena_.CreateTensor("cmp_rope_cos_row", {tokens, rope}, kAclFloat32, rope_cos_base);
  t.cmp_rope_sin_row = arena_.CreateTensor("cmp_rope_sin_row", {tokens, rope}, kAclFloat32, rope_sin_base);
  void* kv_out_base = ReservationAddress("cmp.kv_out");
  t.cmp_kv_padded = arena_.CreateTensor("cmp_kv_padded", {kCompressorWindowOutputRows, width}, kAclBf16, kv_out_base);
  t.cmp_kv_out = arena_.CreateTensor("cmp_kv_out", {tokens, width}, kAclBf16, kv_out_base);
  uint8_t* kv_cache_base = arena_.AddressAs<uint8_t>(b.cmp_kv_cache);
  // The third axis counts BYTES of one Dsv4CompressedKvEntry, viewed as FP8
  // because that is the dtype both the epilog and the attention core demand of
  // this cache. Its axis-0 stride -- which is what every stride attribute is
  // derived from -- is therefore block_size * 604 elements by construction.
  t.cmp_kv_cache = arena_.CreateTensor("cmp_kv_cache",
                                       {num_blocks_, config_.block_size, kCompressedKvEntryBytes},
                                       kAclFloat8E4m3Fn, kv_cache_base);
  t.cmp_slot_mapping =
      arena_.CreateTensor("cmp_slot_mapping", {tokens}, kAclInt32, arena_.Address(b.cmp_slot_mapping));
  t.cmp_block_table =
      arena_.CreateTensor("cmp_block_table", {tokens, num_blocks_}, kAclInt32, arena_.Address(b.cmp_block_table));
  t.cmp_seq_k = arena_.CreateTensor("cmp_seq_k", {tokens}, kAclInt32, arena_.Address(b.cmp_seq_k));
  t.cmp_sparse_indices_dense = arena_.CreateTensor("cmp_sparse_indices_dense", {tokens, 1, 1, kIndexTopK},
                                                   kAclInt32, arena_.Address(b.cmp_sparse_indices_dense));

  // The sparse attention core's q and out, as rank-4 BSND views over the
  // buffers the dense path already uses: q_b's leading heads x kv_lora block
  // and act.attn_out. Byte-identical to the q_latent / attn_out descriptors
  // the SWA path binds, so both cores read and write the same activations and
  // `attn_quant` -> `o_proj` needs no variant.
  t.sparse_q = arena_.CreateTensor("sparse_q", {tokens, 1, heads_, width}, kAclBf16,
                                   ReservationAddress("act.q_b"));
  t.sparse_attn_out = arena_.CreateTensor("sparse_attn_out", {tokens, 1, heads_, width}, kAclBf16,
                                          ReservationAddress("act.attn_out"));
  // returnSoftmaxLse = false is signalled by a [0] placeholder, which the
  // wrapper reads as "issue no copy for that output".
  t.sparse_lse_empty = arena_.CreateTensor("sparse_lse_empty", {0}, kAclFloat32,
                                           ReservationAddress("act.softmax_lse"));

  // Compressor weights, pointed at layer 0 and repointed per layer.
  const BackboneWeights::Layer& first = b.layers[0];
  t.w_cmp_wkv = arena_.CreateTensor("w_cmp_wkv", {width, kHiddenSize}, kAclBf16, arena_.Address(first.cmp_wkv));
  t.w_cmp_wgate = arena_.CreateTensor("w_cmp_wgate", {width, kHiddenSize}, kAclBf16, arena_.Address(first.cmp_wgate));
  t.w_cmp_norm_weight = arena_.CreateTensor("w_cmp_norm_weight", {width}, kAclFloat32,
                                            arena_.Address(first.cmp_norm_weight));
  if (uses_csa_) {
    t.w_cmp_ape_csa = arena_.CreateTensor("w_cmp_ape_csa", {kCompressRatioCsa, width}, kAclFloat32,
                                          arena_.Address(first.cmp_ape));
  }
  if (uses_hca_) {
    t.w_cmp_ape_hca = arena_.CreateTensor("w_cmp_ape_hca", {kCompressRatioHca, width}, kAclFloat32,
                                          arena_.Address(first.cmp_ape));
  }

  if (!uses_csa_) {
    return;
  }
  const int64_t indexer_width = kIndexNumHeads * kIndexHeadDim;
  void* q_fp8_base = ReservationAddress("idx.q_fp8");
  t.index_q_bf16 =
      arena_.CreateTensor("index_q_bf16", {tokens, indexer_width}, kAclBf16, ReservationAddress("idx.q_bf16"));
  t.index_q_fp8 = arena_.CreateTensor("index_q_fp8", {tokens, indexer_width}, kAclFloat8E4m3Fn, q_fp8_base);
  t.index_q_mx_scale = arena_.CreateTensor("index_q_mx_scale", {tokens, MxScaleCols(indexer_width)}, kScaleDtype,
                                           ReservationAddress("idx.q_mx_scale"));
  // The indexer's own BSND spelling of the same FP8 bytes: [B, S, N1, D].
  t.index_q = arena_.CreateTensor("index_q", {tokens, 1, kIndexNumHeads, kIndexHeadDim}, kAclFloat8E4m3Fn,
                                  q_fp8_base);
  t.index_k_bf16 =
      arena_.CreateTensor("index_k_bf16", {tokens, kIndexHeadDim}, kAclBf16, ReservationAddress("idx.k_bf16"));

  uint8_t* index_cache_base = arena_.AddressAs<uint8_t>(b.index_k_cache);
  t.index_k_cache_u8 = arena_.CreateTensor("index_k_cache_u8", {num_blocks_, config_.block_size, kIndexHeadDim},
                                           kAclUint8, index_cache_base);
  t.index_k_cache = arena_.CreateTensor("index_k_cache",
                                        {num_blocks_, config_.block_size, 1, kIndexHeadDim}, kAclFloat8E4m3Fn,
                                        index_cache_base);
  t.index_k_dequant = arena_.CreateTensor("index_k_dequant", {num_blocks_, config_.block_size, 1}, kAclFloat32,
                                          arena_.Address(b.index_k_dequant));
  t.index_q_dequant = arena_.CreateTensor("index_q_dequant", {tokens, 1, kIndexNumHeads}, kAclFloat32,
                                          arena_.Address(b.index_q_dequant));
  t.index_sparse_indices = arena_.CreateTensor("index_sparse_indices", {tokens, 1, 1, kIndexTopK}, kAclInt32,
                                               arena_.Address(b.index_sparse_indices));
  t.index_sparse_values_empty = arena_.CreateTensor("index_sparse_values_empty", {0}, kAclFloat32,
                                                    arena_.Address(b.index_q_dequant));
  t.index_metadata = arena_.CreateTensor("index_metadata", {kIndexerMetadataElements}, kAclInt32, ReservationAddress("idx.metadata"));
  t.index_seq_k = arena_.CreateTensor("index_seq_k", {tokens}, kAclInt32, meta_base + Int32Bytes(4));
  t.index_head_weights_bf16 = arena_.CreateTensor("index_head_weights_bf16", {tokens, 1, kIndexNumHeads}, kAclBf16, arena_.Address(first.index_head_weight));
  // [1, 1, 64] FP32, the learned per-head scoring gain, viewed over the
  // layer's flat [64] reservation at the rank the indexer takes.
  t.index_head_weights = arena_.CreateTensor("index_head_weights", {tokens, 1, kIndexNumHeads}, kAclFloat32,
                                             ReservationAddress("idx.head_weights"));
  // [K, N] for aclnnMatmul, which has no transpose flag.
  t.w_index_q = arena_.CreateTensor("w_index_q", {kQLoraRank, indexer_width}, kAclBf16,
                                    arena_.Address(first.index_q_weight));
  t.w_index_k = arena_.CreateTensor("w_index_k", {width, kIndexHeadDim}, kAclBf16,
                                    arena_.Address(first.index_k_weight));
  (void)slots;
}

// ---------------------------------------------------------------------------
// The constants the graph reads but never computes
// ---------------------------------------------------------------------------
//
// Written here, once, at descriptor time -- not inside a decode step, and not
// left as the zeros Commit() memset. A zero block table makes every paged read
// address block 0, which is a wrong answer with no symptom; the engine's
// paging is one contiguous run of blocks per sequence, so the identity IS the
// table.
void StaticArenaManager::SeedStaticTables() {
  const BackboneWeights& b = *backbone_;
  ResourceScope staging_scope(allocator_, streams_);

  const auto upload = [&](ArenaHandle handle, const std::vector<int32_t>& values) {
    const size_t bytes = Int32Bytes(static_cast<int64_t>(values.size()));
    DSV4_REQUIRE(arena_.Bytes(handle) >= bytes,
                 "a static table of " << values.size() << " entries does not fit its "
                                      << arena_.Bytes(handle) << "-byte reservation");
    int32_t* staging = static_cast<int32_t*>(staging_scope.HostPinnedMalloc(bytes));
    DSV4_REQUIRE(staging != nullptr, "the static-table staging buffer could not be pinned");
    std::copy(values.begin(), values.end(), staging);
    streams_.MemcpySync(arena_.Address(handle), bytes, staging, bytes, MemcpyKind::kHostToDevice);
  };
  const auto identity = [](int64_t count) {
    std::vector<int32_t> values(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
      values[static_cast<size_t>(index)] = static_cast<int32_t>(index);
    }
    return values;
  };

  // The dense paged cache's block table. Previously left zeroed, which pointed
  // every FIA V5 block lookup at block 0.
  upload(b.block_table, identity(num_blocks_));
  if (!uses_compression()) {
    return;
  }
  upload(b.cmp_block_table, identity(num_blocks_));
  upload(b.cmp_state_block_table, identity(kCompressorStateBlocks));
  // The HCA path has no indexer, but a bound cmpKv still requires an INT32
  // selection, so "attend the whole compressed stream" is spelled as the
  // identity top-k.
  //
  // CLAMPED, not a plain 0..k-1 ramp. kIndexTopK is 512 while one layer's
  // compressed cache holds `compressed_slots()` entries, which is smaller
  // whenever the reserved context is short (256 at a 2-block reservation). A
  // raw identity would then hand the attention core indices past the end of
  // the cache, and it gathers at these indices with no bounds check of its own
  // -- the same out-of-bounds read H4's range checker exists to rule out. Past
  // the last real slot the vector repeats it, which for an "attend everything"
  // selection over a stream shorter than k re-reads one entry instead of
  // reading memory that is not the cache. The live length still bounds the
  // kernel through cmp_seq_k; this is the floor under that.
  {
    const int64_t last = compressed_slots() - 1;
    std::vector<int32_t> dense(static_cast<size_t>(kIndexTopK));
    for (int64_t index = 0; index < kIndexTopK; ++index) {
      dense[static_cast<size_t>(index)] = static_cast<int32_t>(std::min(index, last));
    }
    upload(b.cmp_sparse_indices_dense, dense);
  }

  if (!uses_csa_) {
    return;
  }
  const auto metadata = DecodeIndexerMetadata();
  void* metadata_staging = staging_scope.HostPinnedMalloc(sizeof(metadata));
  std::memcpy(metadata_staging, metadata.data(), sizeof(metadata));
  streams_.MemcpySync(ReservationAddress("idx.metadata"), sizeof(metadata), metadata_staging, sizeof(metadata), MemcpyKind::kHostToDevice);
  // The indexer's two FP32 dequant-scale streams, set to unit scale.
  //
  // NAMED DEVIATION, reported by Dsv4Pipeline::DescribeStages: nothing in this
  // graph can produce them. The operator documents them as per-token-head FP32
  // (queryDequantScale matching `weights`, keyDequantScale matching the key
  // layout without D), and the only quantizers this engine has --
  // aclnnDynamicMxQuant and aclnnRmsNormDynamicMxQuant -- emit OCP E8M0
  // block-32 microscales, a different quantity in a different dtype at a
  // different granularity. aclnnIndexerCompressEpilogV2, which writes the
  // indexer key cache, has no scale output at all. So the indexer RUNS and its
  // top-k is real, but with a unit scale the score ORDERING is only correct
  // once a per-token-head scale producer exists. Unit scale is the honest
  // placeholder: it is the identity, not a guess at a magnitude.
  const size_t query_scale_bytes = Fp32Bytes(kTokensPerStep * kIndexNumHeads);
  const size_t key_scale_bytes = IndexerKeyScaleLayerStrideBytes() * static_cast<size_t>(kNumLayers);
  const size_t scale_bytes = std::max(query_scale_bytes, key_scale_bytes);
  float* scale_staging = static_cast<float*>(staging_scope.HostPinnedMalloc(scale_bytes));
  DSV4_REQUIRE(scale_staging != nullptr, "the indexer scale staging buffer could not be pinned");
  std::fill(scale_staging, scale_staging + scale_bytes / sizeof(float), 1.0f);
  streams_.MemcpySync(arena_.Address(b.index_q_dequant), query_scale_bytes, scale_staging, query_scale_bytes,
                      MemcpyKind::kHostToDevice);
  streams_.MemcpySync(arena_.Address(b.index_k_dequant), key_scale_bytes, scale_staging, key_scale_bytes,
                      MemcpyKind::kHostToDevice);
}

void StaticArenaManager::CommitWorkspace() { arena_.CommitWorkspace(); }

void StaticArenaManager::Seal() { arena_.Seal(); }

}  // namespace ascend_moe
