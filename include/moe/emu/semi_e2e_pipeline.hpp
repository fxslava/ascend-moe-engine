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

// semi_e2e_pipeline.hpp -- threads one token through a representative slice of
// the DeepSeek-V4-Flash stack on the host CPU:
//
//   layer 0   mHC round -> shared-KV SWA attention -> mHC fold-back
//   layer 2   mHC round -> SWA attention
//             + the 4:1 compressor cadence (HOLD, HOLD, HOLD, EMIT)
//             + the FP8 lightning indexer selecting top-512 context slots
//             -> mHC fold-back
//   layer 1   mHC round -> MoE block (hash router + routed experts
//             + shared expert) -> mHC fold-back
//
// "SEMI" IS LOAD-BEARING. This is not a 43-layer forward pass and does not
// claim to be one. It is the smallest arrangement that exercises every fused
// operator against real weights with real inter-stage data dependencies, so
// that a layout or ordering error shows up as a number instead of as a
// plausible-looking activation. What is faithful and what is structural is
// enumerated per stage in SemiE2eReport::stages, and the bench prints it.
//
// LAYER ORDER
// The run is layer 0 -> layer 2 -> layer 1's MoE. Layer 1's ATTENTION is not
// executed, so this is deliberately not the model's own layer order; the MoE
// block is taken from layer 1 because layer 1 is a hash layer whose routed
// experts were fetched, and the weight file carries only those six. The
// report says so.

#ifndef MOE_EMU_SEMI_E2E_PIPELINE_HPP
#define MOE_EMU_SEMI_E2E_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/emu/safetensors_view.hpp"

namespace moe::emu {

// How faithful a stage is to the reference forward pass.
enum class Fidelity {
  // Real weights, and the math is ported from a named upstream reference.
  kReference,
  // Real weights and real math, but one convention could not be settled from
  // the checkpoint or the available sources. `note` says which.
  kReferenceWithAssumption,
  // Correct geometry and real math, but the inputs are synthesized because
  // the producing weights are outside the fetched set.
  kSyntheticInput,
};

struct StageReport {
  std::string name;
  Fidelity fidelity = Fidelity::kReference;
  std::string provenance;  // where the weights came from
  std::string note;        // the assumption or the substitution, if any
  double microseconds = 0.0;
};

struct SemiE2eReport {
  std::vector<StageReport> stages;

  // Routing actually taken by the MoE block.
  MoeRouting routing;
  bool routing_experts_all_present = false;
  std::int64_t token_id = 0;

  // Per-stage activation summaries, for comparing against a device run.
  float x_after_layer0_l2 = 0.0F;
  float x_after_layer2_l2 = 0.0F;
  float x_after_moe_l2 = 0.0F;
  int compressed_tokens_emitted = 0;
  int indexer_selected = 0;
  float moe_routed_l2 = 0.0F;
  float moe_shared_l2 = 0.0F;

  AttentionStats layer0_attention;
  AttentionStats layer2_attention;
  MhcSinkhornTrace layer0_attn_trace;
  MhcSinkhornTrace layer2_attn_trace;

  MemoryUsage memory;
  double total_microseconds = 0.0;
  bool all_finite = true;
};

struct SemiE2eOptions {
  // Token id drives BOTH the synthesized embedding and the hash router, so it
  // has to be the id the weight file was fetched for -- the file's
  // __metadata__ records which, and Run() refuses a mismatch rather than
  // routing to experts that are not present.
  std::int64_t token_id = 3;
  int layer0 = 0;
  int layer2 = 2;
  int moe_layer = 1;
  // Decode steps driven through layer 2, to exercise the compressor cadence.
  int decode_steps = 4;
  // Context length the indexer scores over.
  std::int32_t indexer_seq_len = 2048;
  // Prime the SWA window with this many tokens before the measured step, so
  // attention runs against a populated window rather than a single token.
  int warmup_window_tokens = 8;
};

// Owns the mapping and the resolved weight views for one run.
class SemiE2ePipeline {
 public:
  // `weights_path` is the compact file scripts/download_dsv4_semi_e2e_weights.sh
  // produces. Throws (std::runtime_error) naming any tensor that is missing or
  // the wrong shape.
  SemiE2ePipeline(const std::string& weights_path, const std::string& config_path);

  // Threads one token through. `x_next` receives [1, n, D] BF16.
  SemiE2eReport Run(const SemiE2eOptions& options, std::vector<std::uint16_t>* x_next);

  const SafetensorsView& weights() const { return weights_; }
  const std::string& describe_geometry() const { return geometry_summary_; }

 private:
  SafetensorsView weights_;
  std::string geometry_summary_;
  std::int64_t metadata_token_id_ = -1;
  std::vector<int> metadata_routed_experts_;
  bool moe_is_hash_layer_ = true;
};

}  // namespace moe::emu

#endif  // MOE_EMU_SEMI_E2E_PIPELINE_HPP
