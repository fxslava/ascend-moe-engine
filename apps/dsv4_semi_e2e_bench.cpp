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

// dsv4_semi_e2e_bench -- one token through layer 0, layer 2 and a MoE block on
// real DeepSeek-V4-Flash weights, on the host CPU.
//
//   layer 0   mHC round -> shared-KV SWA attention -> mHC fold-back
//   layer 2   mHC round -> SWA attention
//             + 4:1 compressor cadence (HOLD x3, EMIT)
//             + FP8 lightning indexer, top-512
//             -> mHC fold-back
//   layer 1   mHC round -> MoE (hash router, 6 routed experts + 1 shared)
//             -> mHC fold-back
//
// Budgets checked and reported: peak RSS under --memory-budget-mb (default
// 4096) and wall clock under --time-budget-ms (default 1500).
//
// USAGE
//   scripts/download_dsv4_semi_e2e_weights.sh
//   ./build_emu/apps/dsv4_semi_e2e_bench [--data-dir DIR] [--token-id N]
//       [--decode-steps N] [--indexer-seq-len N] [--memory-budget-mb N]
//       [--time-budget-ms N] [--no-dump]

#include "moe/emu/semi_e2e_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace moe::emu;

struct Options {
  std::string data_dir = "data/dsv4_layer0";
  std::int64_t token_id = 3;
  int decode_steps = 4;
  std::int32_t indexer_seq_len = 2048;
  double memory_budget_mb = 4096.0;
  double time_budget_ms = 1500.0;
  bool dump = true;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("option ") + flag + " needs a value");
      }
      return argv[++i];
    };
    if (argument == "--data-dir") {
      options.data_dir = next("--data-dir");
    } else if (argument == "--token-id") {
      options.token_id = std::stoll(next("--token-id"));
    } else if (argument == "--decode-steps") {
      options.decode_steps = std::stoi(next("--decode-steps"));
    } else if (argument == "--indexer-seq-len") {
      options.indexer_seq_len =
          static_cast<std::int32_t>(std::stol(next("--indexer-seq-len")));
    } else if (argument == "--memory-budget-mb") {
      options.memory_budget_mb = std::stod(next("--memory-budget-mb"));
    } else if (argument == "--time-budget-ms") {
      options.time_budget_ms = std::stod(next("--time-budget-ms"));
    } else if (argument == "--no-dump") {
      options.dump = false;
    } else if (argument == "-h" || argument == "--help") {
      std::printf(
          "usage: dsv4_semi_e2e_bench [--data-dir DIR] [--token-id N]\n"
          "       [--decode-steps N] [--indexer-seq-len N]\n"
          "       [--memory-budget-mb N] [--time-budget-ms N] [--no-dump]\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.decode_steps < 1 || options.indexer_seq_len < 1) {
    throw std::runtime_error("--decode-steps and --indexer-seq-len must be >= 1");
  }
  return options;
}

bool FileExists(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  return probe.good();
}

const char* FidelityLabel(Fidelity fidelity) {
  switch (fidelity) {
    case Fidelity::kReference:
      return "reference";
    case Fidelity::kReferenceWithAssumption:
      return "reference*";
    case Fidelity::kSyntheticInput:
      return "synth-input";
  }
  return "?";
}

}  // namespace

int main(int argc, char** argv) try {
  const Options options = ParseOptions(argc, argv);

  const std::string weights_path = options.data_dir + "/dsv4_semi_e2e.safetensors";
  const std::string config_path = options.data_dir + "/config.json";
  const std::string dump_path = options.data_dir + "/golden_semi_e2e_ref.bin";

  std::printf(
      "=================================================================\n"
      " DeepSeek-V4-Flash semi-end-to-end decode -- host CPU emulation\n"
      " layer 0 (SWA) -> layer 2 (compressed + indexer) -> layer 1 (MoE)\n"
      "=================================================================\n");

  for (const std::string& path : {config_path, weights_path}) {
    if (!FileExists(path)) {
      std::fprintf(stderr,
                   "error: %s not found.\n"
                   "       Run scripts/download_dsv4_semi_e2e_weights.sh first.\n",
                   path.c_str());
      return 2;
    }
  }

  const MemoryUsage before = ReadMemoryUsage();
  SemiE2ePipeline pipeline(weights_path, config_path);

  std::printf("\n[0] weights: %s\n", weights_path.c_str());
  std::printf("    %zu tensors, %llu-byte header, %zu data bytes mapped\n",
              pipeline.weights().size(),
              static_cast<unsigned long long>(pipeline.weights().header_length()),
              pipeline.weights().data_segment_bytes());
  for (const auto& entry : pipeline.weights().metadata()) {
    std::printf("    %-22s %s\n", entry.first.c_str(), entry.second.c_str());
  }
  std::printf("\n[1] geometry checked against the compiled contract:\n%s",
              pipeline.describe_geometry().c_str());

  SemiE2eOptions run_options;
  run_options.token_id = options.token_id;
  run_options.decode_steps = options.decode_steps;
  run_options.indexer_seq_len = options.indexer_seq_len;

  std::vector<std::uint16_t> x_next;
  std::printf("\n[2] running the pipeline\n");
  const SemiE2eReport report = pipeline.Run(run_options, &x_next);

  // ---- per-stage report ---------------------------------------------------
  std::printf("\n[3] stages\n");
  for (const StageReport& stage : report.stages) {
    std::printf("  %-11s %10.1f us  %s\n", FidelityLabel(stage.fidelity),
                stage.microseconds, stage.name.c_str());
    std::printf("                             weights: %s\n", stage.provenance.c_str());
    if (!stage.note.empty()) {
      std::printf("                             note:    %s\n", stage.note.c_str());
    }
  }
  std::printf(
      "\n  reference   = real weights, math ported from a named upstream golden\n"
      "  reference*  = real weights and math, one convention assumed (see note)\n"
      "  synth-input = real geometry and math, inputs synthesized (see note)\n");

  // ---- activations --------------------------------------------------------
  std::printf("\n[4] activations (L2 of the [1, 4, 4096] stream bundle)\n");
  std::printf("    after layer 0           %12.4f\n", report.x_after_layer0_l2);
  std::printf("    after layer 2           %12.4f\n", report.x_after_layer2_l2);
  std::printf("    after the MoE layer     %12.4f\n", report.x_after_moe_l2);
  std::printf("    all values finite       %s\n", report.all_finite ? "yes" : "NO");

  std::printf("\n[5] attention\n");
  const auto print_attention = [](const char* label, const AttentionStats& stats) {
    std::printf("    %-8s attended %3d tokens, q_lora inv_rms %.4f, kv inv_rms %.4f\n",
                label, stats.attended_tokens, stats.q_lora_inv_rms, stats.kv_inv_rms);
    std::printf("             max logit %.4f, mean attn_sink weight %.6f\n",
                stats.max_logit, stats.mean_sink_weight);
  };
  print_attention("layer 0", report.layer0_attention);
  print_attention("layer 2", report.layer2_attention);

  std::printf("\n[6] mHC Birkhoff maps (worst |sum - 1|)\n");
  const auto print_birkhoff = [](const char* label, const MhcSinkhornTrace& trace) {
    double worst_row = 0.0;
    double worst_col = 0.0;
    for (int row = 0; row < kHcStreams; ++row) {
      double sum = 0.0;
      for (int col = 0; col < kHcStreams; ++col) {
        sum += trace.birkhoff[row * kHcStreams + col];
      }
      worst_row = std::max(worst_row, std::fabs(sum - 1.0));
    }
    for (int col = 0; col < kHcStreams; ++col) {
      double sum = 0.0;
      for (int row = 0; row < kHcStreams; ++row) {
        sum += trace.birkhoff[row * kHcStreams + col];
      }
      worst_col = std::max(worst_col, std::fabs(sum - 1.0));
    }
    std::printf("    %-8s rows %.3e, columns %.3e\n", label, worst_row, worst_col);
  };
  print_birkhoff("layer 0", report.layer0_attn_trace);
  print_birkhoff("layer 2", report.layer2_attn_trace);
  std::printf(
      "    (the row sums are NOT within 1e-5 at 20 iterations on these weights;\n"
      "     that is the checkpoint's behaviour, not a solver fault -- see the\n"
      "     Sinkhorn note in README.md)\n");

  std::printf("\n[7] compressor and indexer\n");
  std::printf("    compressed tokens emitted  %d (from %d decode steps, %d:1)\n",
              report.compressed_tokens_emitted, options.decode_steps,
              kCompressorRatio);
  std::printf("    indexer indices selected   %d of top-%d over %d keys\n",
              report.indexer_selected, kIndexerTopK, options.indexer_seq_len);

  std::printf("\n[8] MoE routing (token id %lld)\n",
              static_cast<long long>(report.token_id));
  std::printf("    router      %s\n",
              report.routing.from_hash_table
                  ? "tid2eid hash table (layer < num_hash_layers)"
                  : "noaux_tc bias-shifted top-k");
  std::printf("    experts    ");
  for (int i = 0; i < report.routing.count; ++i) {
    std::printf(" %3d", report.routing.experts[i]);
  }
  std::printf("\n    weights    ");
  float weight_total = 0.0F;
  for (int i = 0; i < report.routing.count; ++i) {
    std::printf(" %.4f", report.routing.weights[i]);
    weight_total += report.routing.weights[i];
  }
  std::printf("\n    weight sum  %.6f (norm_topk_prob renormalizes to 1)\n",
              weight_total);
  std::printf("    all selected experts present in the weight file: %s\n",
              report.routing_experts_all_present ? "yes" : "NO");
  std::printf("    routed L2 %.4f, shared L2 %.4f, routed scaled by %.2f\n",
              report.moe_routed_l2, report.moe_shared_l2, kRoutedScalingFactor);

  // ---- the golden dump ----------------------------------------------------
  if (options.dump) {
    std::printf("\n[9] golden reference dump\n");
    std::ofstream out(dump_path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      std::fprintf(stderr, "error: cannot write %s\n", dump_path.c_str());
      return 5;
    }
    struct Header {
      char magic[8];
      std::uint32_t version;
      std::uint32_t dtype_code;  // 1 = BF16
      std::int32_t streams;
      std::int32_t hidden;
      std::int64_t token_id;
      std::int32_t routed_count;
      std::int32_t sinkhorn_n;
      std::int32_t compressed_tokens;
      std::int32_t indexer_top_k;
    } header{};
    std::memcpy(header.magic, "DSV4SE2E", 8);
    header.version = 3;
    header.dtype_code = 1;
    header.streams = kHcStreams;
    header.hidden = kHiddenSize;
    header.token_id = report.token_id;
    header.routed_count = report.routing.count;
    header.sinkhorn_n = kHcStreams;
    header.compressed_tokens = report.compressed_tokens_emitted;
    header.indexer_top_k = kIndexerTopK;

    const auto write = [&out](const void* data, std::size_t bytes) {
      out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    write(&header, sizeof(header));
    write(x_next.data(), x_next.size() * sizeof(std::uint16_t));
    write(report.layer0_attn_trace.birkhoff, sizeof(report.layer0_attn_trace.birkhoff));
    write(report.layer2_attn_trace.birkhoff, sizeof(report.layer2_attn_trace.birkhoff));
    write(report.routing.experts, sizeof(report.routing.experts));
    write(report.routing.weights, sizeof(report.routing.weights));
    // Version 2 appends the precise stage inputs needed by an independent
    // PyTorch reference. The version-1 prefix above is kept byte-for-byte.
    const auto write_vector = [&write](const auto& values) {
      write(values.data(), values.size() * sizeof(values[0]));
    };
    write_vector(report.x_input);
    write_vector(report.x_after_layer0);
    write_vector(report.x_after_layer2);
    write_vector(report.layer0_h_in);
    write_vector(report.layer2_h_in);
    write_vector(report.moe_h_in);
    write(report.moe_ffn_trace.birkhoff, sizeof(report.moe_ffn_trace.birkhoff));
    write_vector(report.expert_outputs);
    write_vector(report.moe_routed);
    write_vector(report.moe_shared);
    write_vector(report.layer2_compressed_tokens);
    out.close();
    if (!out.good()) {
      std::fprintf(stderr, "error: writing %s failed\n", dump_path.c_str());
      return 5;
    }
    std::ifstream probe(dump_path, std::ios::binary | std::ios::ate);
    std::printf("    %s (%lld bytes)\n", dump_path.c_str(),
                static_cast<long long>(probe.tellg()));
    std::printf("      header %zu B, x_next [1, 4, 4096] BF16, 2 x hRes [4, 4] F32,\n"
                "      routed expert ids + weights, v3 stage diagnostics\n",
                sizeof(header));
  } else {
    std::printf("\n[9] golden reference dump skipped (--no-dump)\n");
  }

  // ---- budgets ------------------------------------------------------------
  std::printf("\n=================================================================\n"
              " budgets\n"
              "=================================================================\n");
  const double milliseconds = report.total_microseconds / 1000.0;
  std::printf("  wall clock            %10.1f ms   budget %7.1f ms   %s\n",
              milliseconds, options.time_budget_ms,
              milliseconds < options.time_budget_ms ? "OK" : "EXCEEDED");

  bool memory_ok = true;
  if (report.memory.available) {
    const double peak_mb = static_cast<double>(report.memory.peak_rss_bytes) / 1048576.0;
    const double current_mb =
        static_cast<double>(report.memory.current_rss_bytes) / 1048576.0;
    const double start_mb =
        before.available ? static_cast<double>(before.peak_rss_bytes) / 1048576.0 : 0.0;
    memory_ok = peak_mb < options.memory_budget_mb;
    std::printf("  peak RSS (VmHWM)      %10.1f MB   budget %7.1f MB   %s\n", peak_mb,
                options.memory_budget_mb, memory_ok ? "OK" : "EXCEEDED");
    std::printf("  current RSS (VmRSS)   %10.1f MB\n", current_mb);
    std::printf("  RSS before the run    %10.1f MB\n", start_mb);
    std::printf("\n  The 365 MB weight file is mmap'd and every quantized matrix is\n"
                "  dequantized one output row at a time, so the resident set tracks\n"
                "  the pages actually touched rather than an FP32 copy of the\n"
                "  weights (which would be 4x for FP8 and 8x for FP4).\n");
  } else {
    std::printf("  peak RSS              unavailable (/proc/self/status not readable)\n");
  }

  const bool ok = report.all_finite && memory_ok &&
                  milliseconds < options.time_budget_ms &&
                  report.routing_experts_all_present;
  std::printf("\n  semi-end-to-end multi-layer decode: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 6;
} catch (const std::exception& error) {
  std::fprintf(stderr, "fatal: %s\n", error.what());
  return 1;
}
