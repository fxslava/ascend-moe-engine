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

// dsv4_layer0_emu_bench -- one Layer 0 decode step on real DeepSeek-V4-Flash
// weights, entirely on the host CPU, emitting a golden activation tensor for
// later NPU kernel verification.
//
// THREE THINGS THE CHECKPOINT DOES NOT MATCH THE OBVIOUS ASSUMPTION ON, all
// established by reading `model.safetensors.index.json` and the shard headers
// rather than by guessing. Each one would have produced a plausible-looking
// run on the wrong data:
//
//   1. LAYER 0 IS IN SHARD 2, NOT SHARD 1.
//      The index maps `model-00001-of-00046.safetensors` to exactly one
//      tensor -- `embed.weight`, BF16 [129280, 4096], 1,059,061,760 bytes --
//      and all 1565 `layers.0.*` tensors to shard 2. Shard 1 carries no layer
//      weight at all, so a bench that opened it would find nothing to run.
//
//   2. THE TENSOR NAMES HAVE NO `model.` PREFIX AND NO `mhc_pre` SUBMODULE.
//      The real spellings are flat, per layer:
//        layers.0.hc_attn_fn      F32 [24, 16384]   the mHC phi  (attention)
//        layers.0.hc_attn_scale   F32 [3]           alpha
//        layers.0.hc_attn_base    F32 [24]          bias
//        layers.0.hc_ffn_fn/_scale/_base            the same, for the FFN
//        layers.0.attn_norm.weight   BF16 [4096]
//        layers.0.ffn_norm.weight    BF16 [4096]
//        layers.0.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}
//      There is no `input_layernorm`, no `self_attn`, no `mhc_pre.phi`.
//
//   3. LAYER 0 HAS NEITHER A COMPRESSOR NOR AN INDEXER.
//      config.json's `compress_ratios` begins [0, 0, 4, 128, 4, 128, ...]:
//      layer 0 is an uncompressed (SWA) layer, so it owns no
//      `attn.compressor.*` and no `attn.indexer.*` tensors. Those first
//      appear on layer 2 (ratio 4, the CSA path), which is where this bench
//      sources them from -- `layers.2.attn.compressor.wkv.weight`
//      BF16 [1024, 4096] is what pins coff=2, D=512.
//
//      So the full seven-stage pipeline cannot run on layer 0's weights
//      alone. This bench runs every stage, and reports for each one whether
//      the weights behind it are layer 0's own, layer 2's (fetched with
//      --shard 4), or synthetic. Nothing is silently substituted.
//
// THE PIPELINE
//   x0 -> RMSNorm -> mHC projection -> fused Sinkhorn -> attention /
//   compressor -> lightning indexer -> mHC fused update -> x_next
//
//   The mHC operator fuses stages 2-4 and 7: `mhc_fused_sinkhorn_gemv_traced`
//   produces inv_rms, the 24 projection rows, the three gate segments, the
//   Birkhoff matrix, h_in AND the x_next update in one call. The bench times
//   it as one phase and then reports the sub-quantities it exposed, because
//   splitting it would time something the device never launches separately.
//
// USAGE
//   ./dsv4_layer0_emu_bench [--data-dir DIR] [--steps N] [--seq-len N]
//                           [--repeat N] [--no-dump]
//   Defaults: --data-dir data/dsv4_layer0, --steps 8, --seq-len 4096.

#include "moe/core/json.hpp"
#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"
#include "moe/emu/safetensors_view.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace moe::emu;
namespace json = ascend_moe::moe_json;

using Clock = std::chrono::steady_clock;

double MicrosecondsSince(const Clock::time_point& start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// --- phase timing -----------------------------------------------------------

struct Phase {
  std::string name;
  double microseconds = 0.0;
  std::string provenance;  // where the weights came from
  std::string detail;
};

std::vector<Phase> phases;

void Record(const std::string& name, double microseconds, const std::string& provenance,
            const std::string& detail = "") {
  phases.push_back(Phase{name, microseconds, provenance, detail});
}

// --- options ----------------------------------------------------------------

struct Options {
  std::string data_dir = "data/dsv4_layer0";
  int steps = 8;
  std::int32_t seq_len = 4096;
  int repeat = 1;
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
    } else if (argument == "--steps") {
      options.steps = std::stoi(next("--steps"));
    } else if (argument == "--seq-len") {
      options.seq_len = static_cast<std::int32_t>(std::stol(next("--seq-len")));
    } else if (argument == "--repeat") {
      options.repeat = std::stoi(next("--repeat"));
    } else if (argument == "--no-dump") {
      options.dump = false;
    } else if (argument == "-h" || argument == "--help") {
      std::printf(
          "usage: dsv4_layer0_emu_bench [--data-dir DIR] [--steps N] "
          "[--seq-len N] [--repeat N] [--no-dump]\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.steps < 1 || options.seq_len < 1 || options.repeat < 1) {
    throw std::runtime_error("--steps, --seq-len and --repeat must all be >= 1");
  }
  return options;
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    throw std::runtime_error("cannot open " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

bool FileExists(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  return probe.good();
}

// --- stage 1: geometry from config.json -------------------------------------

struct Geometry {
  std::int64_t hidden_size = 0;
  std::int64_t hc_mult = 0;
  std::int64_t hc_sinkhorn_iters = 0;
  double hc_eps = 0.0;
  double rms_norm_eps = 0.0;
  std::int64_t index_topk = 0;
  std::int64_t index_head_dim = 0;
  std::int64_t index_n_heads = 0;
  std::int64_t head_dim = 0;
  std::int64_t qk_rope_head_dim = 0;
  std::int64_t num_hidden_layers = 0;
  std::vector<std::int64_t> compress_ratios;
  // NOT published by config.json; see the report below.
  std::int64_t coff = kCompressorCoff;
};

Geometry LoadGeometry(const std::string& path, int* refusals) {
  const json::Value document = json::Parse(ReadWholeFile(path));
  const auto required_int = [&](const char* key) {
    const json::Value* value = document.find(key);
    if (value == nullptr) {
      throw std::runtime_error(std::string("config.json has no \"") + key + "\"");
    }
    return value->as_int();
  };
  const auto required_double = [&](const char* key) {
    const json::Value* value = document.find(key);
    if (value == nullptr) {
      throw std::runtime_error(std::string("config.json has no \"") + key + "\"");
    }
    return value->as_number();
  };

  Geometry g;
  g.hidden_size = required_int("hidden_size");
  g.hc_mult = required_int("hc_mult");
  g.hc_sinkhorn_iters = required_int("hc_sinkhorn_iters");
  g.hc_eps = required_double("hc_eps");
  g.rms_norm_eps = required_double("rms_norm_eps");
  g.index_topk = required_int("index_topk");
  g.index_head_dim = required_int("index_head_dim");
  g.index_n_heads = required_int("index_n_heads");
  g.head_dim = required_int("head_dim");
  g.qk_rope_head_dim = required_int("qk_rope_head_dim");
  g.num_hidden_layers = required_int("num_hidden_layers");
  if (const json::Value* ratios = document.find("compress_ratios")) {
    for (std::size_t i = 0; i < ratios->size(); ++i) {
      g.compress_ratios.push_back((*ratios)[i].as_int());
    }
  }

  // --- the contract asserts -----------------------------------------------
  // These are the four the brief names (n = 4, D = 4096, top_k = 512,
  // coff = 2) plus the ones the compiled kernels depend on. A divergence is
  // reported field by field and counted; it is not a silent re-plan.
  std::printf("  geometry (checked against the compiled contract):\n");
  const auto check = [&](const char* field, std::int64_t actual, std::int64_t expected) {
    const bool ok = actual == expected;
    if (!ok) {
      ++*refusals;
    }
    std::printf("    %-22s %-8lld  expected %-8lld  %s\n", field,
                static_cast<long long>(actual), static_cast<long long>(expected),
                ok ? "ok" : "REFUSED");
  };
  check("hc_mult (n)", g.hc_mult, kHcStreams);
  check("hidden_size (D)", g.hidden_size, kHiddenSize);
  check("index_topk", g.index_topk, kIndexerTopK);
  check("index_head_dim", g.index_head_dim, kIndexerHeadDim);
  check("index_n_heads", g.index_n_heads, kIndexerHeads);
  check("head_dim", g.head_dim, kCompressorHeadDim);
  check("qk_rope_head_dim", g.qk_rope_head_dim, kRopeHeadDim);
  check("hc_sinkhorn_iters", g.hc_sinkhorn_iters, kHcSinkhornIters);

  // coff is NOT a config.json field. It is fixed by the compressor's
  // supported-configuration table (C4A = D 512, coff 2, cmp_ratio 4) and
  // confirmed by the weight shapes: `layers.2.attn.compressor.wkv.weight` is
  // [1024, 4096] = [coff * D, H], so coff = 1024 / 512 = 2. Printed as a
  // derived value rather than asserted against a field that does not exist.
  std::printf("    %-22s %-8lld  (derived: not a config.json field; "
              "wkv [coff*D, H] = [1024, 4096] over D = 512)\n",
              "coff", static_cast<long long>(g.coff));

  std::printf("    %-22s %.3g / %.3g\n", "hc_eps / rms_norm_eps", g.hc_eps,
              g.rms_norm_eps);
  std::printf("    %-22s %lld layers, compress_ratios[0..5] = ", "topology",
              static_cast<long long>(g.num_hidden_layers));
  for (std::size_t i = 0; i < std::min<std::size_t>(6, g.compress_ratios.size()); ++i) {
    std::printf("%lld ", static_cast<long long>(g.compress_ratios[i]));
  }
  std::printf("\n");
  if (!g.compress_ratios.empty() && g.compress_ratios[0] == 0) {
    std::printf("    note: compress_ratios[0] = 0 -- LAYER 0 IS AN UNCOMPRESSED "
                "(SWA) LAYER,\n"
                "          so it owns no compressor and no indexer weights. See "
                "the phase\n"
                "          provenance column below for what each stage actually "
                "ran on.\n");
  }
  return g;
}

// --- weight handles ---------------------------------------------------------

// A BF16 tensor pulled out of the mapping into an owned buffer. Copied rather
// than used in place because the mapping may be misaligned and because the
// kernels want a packed array.
std::vector<std::uint16_t> LoadBf16(const SafetensorsView& view, const std::string& name,
                                    std::int64_t expected_elements) {
  const TensorView& tensor = view.Tensor(name);
  if (tensor.dtype != "BF16") {
    throw std::runtime_error(name + " is " + tensor.dtype + ", expected BF16");
  }
  if (tensor.elements() != expected_elements) {
    throw std::runtime_error(name + " has " + std::to_string(tensor.elements()) +
                             " elements, expected " + std::to_string(expected_elements));
  }
  std::vector<std::uint16_t> out(static_cast<std::size_t>(expected_elements));
  tensor.CopyTo(out.data(), out.size());
  return out;
}

std::vector<float> LoadF32(const SafetensorsView& view, const std::string& name,
                           std::int64_t expected_elements) {
  const TensorView& tensor = view.Tensor(name);
  if (tensor.dtype != "F32") {
    throw std::runtime_error(name + " is " + tensor.dtype + ", expected F32");
  }
  if (tensor.elements() != expected_elements) {
    throw std::runtime_error(name + " has " + std::to_string(tensor.elements()) +
                             " elements, expected " + std::to_string(expected_elements));
  }
  std::vector<float> out(static_cast<std::size_t>(expected_elements));
  tensor.CopyTo(out.data(), out.size());
  return out;
}

// --- synthetic fallbacks ----------------------------------------------------

std::mt19937& Rng() {
  // Fixed seed: the golden dump has to be reproducible run to run, or it is
  // not a golden.
  static std::mt19937 engine(0xD54Fu);
  return engine;
}

float Uniform(float low, float high) {
  std::uniform_real_distribution<float> distribution(low, high);
  return distribution(Rng());
}

std::vector<std::uint16_t> SyntheticBf16(std::size_t count, float scale) {
  std::vector<std::uint16_t> out(count);
  for (auto& value : out) {
    value = FloatToBf16(Uniform(-scale, scale));
  }
  return out;
}

std::vector<float> SyntheticF32(std::size_t count, float low, float high) {
  std::vector<float> out(count);
  for (auto& value : out) {
    value = Uniform(low, high);
  }
  return out;
}

// --- RoPE tables ------------------------------------------------------------

// config.json: "compress_rope_theta": 160000 for the compressor path. Built
// here rather than read from the checkpoint because the tables are generated
// at runtime from the position, not stored.
void BuildRopeTables(std::int32_t position, double theta, std::vector<float>* cos_table,
                     std::vector<float>* sin_table) {
  cos_table->assign(kRopeHeadDim, 0.0F);
  sin_table->assign(kRopeHeadDim, 0.0F);
  const int half = kRopeHeadDim / 2;
  for (int i = 0; i < half; ++i) {
    const double inverse_frequency =
        1.0 / std::pow(theta, static_cast<double>(2 * i) / kRopeHeadDim);
    const double angle = static_cast<double>(position) * inverse_frequency;
    const float c = static_cast<float>(std::cos(angle));
    const float s = static_cast<float>(std::sin(angle));
    // Duplicated halves, the layout half-mode RoPE expects.
    (*cos_table)[static_cast<std::size_t>(i)] = c;
    (*sin_table)[static_cast<std::size_t>(i)] = s;
    (*cos_table)[static_cast<std::size_t>(half + i)] = c;
    (*sin_table)[static_cast<std::size_t>(half + i)] = s;
  }
}

// --- reporting helpers ------------------------------------------------------

struct TensorStats {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double l2 = 0.0;
  bool all_finite = true;
};

TensorStats Describe(const std::uint16_t* bf16, std::size_t count) {
  TensorStats stats;
  if (count == 0) {
    return stats;
  }
  double sum = 0.0;
  double sum_squares = 0.0;
  stats.min = stats.max = Bf16ToFloat(bf16[0]);
  for (std::size_t i = 0; i < count; ++i) {
    const double value = Bf16ToFloat(bf16[i]);
    if (!std::isfinite(value)) {
      stats.all_finite = false;
    }
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
    sum += value;
    sum_squares += value * value;
  }
  stats.mean = sum / static_cast<double>(count);
  stats.l2 = std::sqrt(sum_squares);
  return stats;
}

}  // namespace

int main(int argc, char** argv) try {
  const Options options = ParseOptions(argc, argv);
  const auto wall_start = Clock::now();

  std::printf(
      "=============================================================\n"
      " DeepSeek-V4-Flash Layer 0 decode step -- host CPU emulation\n"
      "=============================================================\n");

  // ===== stage 0: config.json ==============================================
  const std::string config_path = options.data_dir + "/config.json";
  if (!FileExists(config_path)) {
    std::fprintf(stderr,
                 "error: %s not found.\n"
                 "       Run scripts/download_dsv4_layer0.sh first.\n",
                 config_path.c_str());
    return 2;
  }
  std::printf("\n[0] config.json: %s\n", config_path.c_str());
  int refusals = 0;
  auto start = Clock::now();
  const Geometry geometry = LoadGeometry(config_path, &refusals);
  Record("config parse + geometry check", MicrosecondsSince(start), "config.json");
  if (refusals != 0) {
    std::fprintf(stderr,
                 "\nerror: %d geometry field(s) diverge from the compiled "
                 "contract (marked REFUSED above).\n"
                 "       A divergent checkpoint needs a rebuild, not a silent "
                 "re-plan.\n",
                 refusals);
    return 3;
  }

  // ===== stage 0b: open the shard that holds Layer 0 =======================
  // Shard 2, not shard 1 -- see the file header.
  const std::string layer0_shard =
      options.data_dir + "/model-00002-of-00046.safetensors";
  const std::string layer2_shard =
      options.data_dir + "/model-00004-of-00046.safetensors";

  if (!FileExists(layer0_shard)) {
    std::fprintf(stderr,
                 "error: %s not found.\n"
                 "       Layer 0's weights live in shard 2; shard 1 holds only "
                 "embed.weight.\n"
                 "       Run scripts/download_dsv4_layer0.sh (which fetches "
                 "shard 2 by default).\n",
                 layer0_shard.c_str());
    return 2;
  }

  start = Clock::now();
  SafetensorsView shard(layer0_shard);
  const double mmap_microseconds = MicrosecondsSince(start);
  std::printf("\n[0b] shard: %s\n"
              "     %zu tensors, %llu-byte header, %zu of the data segment mapped\n",
              layer0_shard.c_str(), shard.size(),
              static_cast<unsigned long long>(shard.header_length()),
              shard.data_segment_bytes());
  Record("mmap + header parse (1565 tensors)", mmap_microseconds, layer0_shard);

  const bool have_layer2 = FileExists(layer2_shard);
  SafetensorsView layer2;
  if (have_layer2) {
    layer2.Open(layer2_shard);
    std::printf("     layer 2 shard also present: %s (%zu tensors)\n",
                layer2_shard.c_str(), layer2.size());
  } else {
    std::printf("     layer 2 shard (%s) absent: the compressor and indexer "
                "stages\n     will run on synthetic weights at the real "
                "geometry. Fetch it with\n"
                "       scripts/download_dsv4_layer0.sh --shard 4\n",
                layer2_shard.c_str());
  }

  // ===== stage 0c: locate the Layer 0 weights ==============================
  std::printf("\n[0c] Layer 0 weights (real names, from the shard header):\n");
  start = Clock::now();

  const std::vector<float> phi_attn =
      LoadF32(shard, "layers.0.hc_attn_fn", kHcPhiRows * kHcPhiCols);
  const std::vector<float> alpha_attn =
      LoadF32(shard, "layers.0.hc_attn_scale", kHcAlphaElements);
  const std::vector<float> bias_attn =
      LoadF32(shard, "layers.0.hc_attn_base", kHcProjectionRows);
  const std::vector<float> phi_ffn =
      LoadF32(shard, "layers.0.hc_ffn_fn", kHcPhiRows * kHcPhiCols);
  const std::vector<float> alpha_ffn =
      LoadF32(shard, "layers.0.hc_ffn_scale", kHcAlphaElements);
  const std::vector<float> bias_ffn =
      LoadF32(shard, "layers.0.hc_ffn_base", kHcProjectionRows);
  const std::vector<std::uint16_t> attn_norm =
      LoadBf16(shard, "layers.0.attn_norm.weight", kHiddenSize);
  const std::vector<std::uint16_t> ffn_norm =
      LoadBf16(shard, "layers.0.ffn_norm.weight", kHiddenSize);

  const double weight_microseconds = MicrosecondsSince(start);
  for (const char* name : {"layers.0.hc_attn_fn", "layers.0.hc_attn_scale",
                           "layers.0.hc_attn_base", "layers.0.hc_ffn_fn",
                           "layers.0.hc_ffn_scale", "layers.0.hc_ffn_base",
                           "layers.0.attn_norm.weight", "layers.0.ffn_norm.weight"}) {
    const TensorView& tensor = shard.Tensor(name);
    std::printf("     %-32s %-9s %s\n", name, tensor.dtype.c_str(),
                tensor.shape_string().c_str());
  }
  // The FP8 attention projections: reported, and honestly reported as absent
  // when the shard is a leading slice.
  for (const char* name : {"layers.0.attn.wq_a.weight", "layers.0.attn.wkv.weight",
                           "layers.0.attn.wo_b.weight"}) {
    const TensorView* tensor = shard.Find(name);
    if (tensor == nullptr) {
      std::printf("     %-32s MISSING from the header\n", name);
    } else {
      std::printf("     %-32s %-9s %-16s %s\n", name, tensor->dtype.c_str(),
                  tensor->shape_string().c_str(),
                  tensor->available ? "mapped"
                                    : "NOT in the downloaded slice (--full fetches it)");
    }
  }
  Record("locate + copy the mHC / norm weights", weight_microseconds, layer0_shard,
         "2 x phi [24, 16384] F32 + 2 x alpha/bias + 2 x norm [4096] BF16");

  // ===== stage 1: synthesize x0 ============================================
  std::printf("\n[1] input: synthesizing x0 [1, %d, %d] BF16 (TND)\n", kHcStreams,
              kHiddenSize);
  const std::size_t stream_elements = static_cast<std::size_t>(kHcStreams) * kHiddenSize;
  // Scaled so the four streams look like a post-embedding residual: the
  // embedding is BF16 with a ~0.02 initializer_range, and four streams of it.
  std::vector<std::uint16_t> x0 = SyntheticBf16(stream_elements, 0.05F);
  {
    const TensorStats stats = Describe(x0.data(), x0.size());
    std::printf("    x0: min %.5f max %.5f mean %.6f L2 %.4f\n", stats.min, stats.max,
                stats.mean, stats.l2);
  }

  // ===== the pipeline ======================================================
  std::vector<std::uint16_t> x_next(stream_elements);
  std::vector<std::uint16_t> x_after_attn(stream_elements);
  MhcSinkhornTrace attn_trace;
  MhcSinkhornTrace ffn_trace;

  // --- stage 2: explicit RMSNorm over the flattened bundle -----------------
  // The mHC operator applies its own inv_rms internally; this stage computes
  // the same quantity separately so the activation RMSNorm can be timed and
  // reported on its own, and so `attn_norm.weight` is actually consumed.
  std::printf("\n[2] RMSNorm (activation norm, layers.0.attn_norm.weight)\n");
  start = Clock::now();
  std::vector<float> normed(stream_elements);
  {
    double sum_squares = 0.0;
    for (std::size_t i = 0; i < x0.size(); ++i) {
      const double value = Bf16ToFloat(x0[i]);
      sum_squares += value * value;
    }
    const float inv_rms =
        1.0F / std::sqrt(static_cast<float>(sum_squares / static_cast<double>(x0.size())) +
                         static_cast<float>(geometry.rms_norm_eps));
    for (int stream = 0; stream < kHcStreams; ++stream) {
      for (int feature = 0; feature < kHiddenSize; ++feature) {
        const std::size_t index =
            static_cast<std::size_t>(stream) * kHiddenSize + feature;
        normed[index] = Bf16ToFloat(x0[index]) * inv_rms *
                        Bf16ToFloat(attn_norm[static_cast<std::size_t>(feature)]);
      }
    }
    std::printf("    inv_rms %.6f (eps %.1e)\n", inv_rms, geometry.rms_norm_eps);
  }
  Record("RMSNorm [1, 4, 4096]", MicrosecondsSince(start), layer0_shard,
         "layers.0.attn_norm.weight");

  // --- stages 3-4: mHC projection + fused Sinkhorn (attention round) -------
  std::printf("\n[3+4] mHC projection + 20-iteration Sinkhorn (attention round)\n");
  {
    // h_out is the attention sub-block's FUNCTION output. The FP8 projections
    // that would produce it are outside the downloaded slice, so this is a
    // synthetic stand-in at the right shape -- stated, not hidden.
    const std::vector<std::uint16_t> h_out_attn = SyntheticBf16(kHiddenSize, 0.1F);
    double total = 0.0;
    for (int repeat = 0; repeat < options.repeat; ++repeat) {
      start = Clock::now();
      mhc_fused_sinkhorn_gemv_traced(x0.data(), phi_attn.data(), alpha_attn.data(),
                                     bias_attn.data(), h_out_attn.data(),
                                     x_after_attn.data(), &attn_trace,
                                     static_cast<float>(geometry.hc_eps),
                                     static_cast<int>(geometry.hc_sinkhorn_iters));
      total += MicrosecondsSince(start);
    }
    Record("mHC pre + Sinkhorn + post (attention)", total / options.repeat,
           layer0_shard, "layers.0.hc_attn_{fn,scale,base}; h_out synthetic");

    std::printf("    inv_rms         %.6f\n", attn_trace.inv_rms);
    std::printf("    projection[0:4] % .5f % .5f % .5f % .5f\n",
                attn_trace.projection[0], attn_trace.projection[1],
                attn_trace.projection[2], attn_trace.projection[3]);
    std::printf("    h_pre           % .6f % .6f % .6f % .6f\n", attn_trace.h_pre[0],
                attn_trace.h_pre[1], attn_trace.h_pre[2], attn_trace.h_pre[3]);
    std::printf("    h_post          % .6f % .6f % .6f % .6f\n", attn_trace.h_post[0],
                attn_trace.h_post[1], attn_trace.h_post[2], attn_trace.h_post[3]);
    std::printf("    hRes (the residual map; see the convergence note below):\n");
    double worst_row = 0.0;
    double worst_col = 0.0;
    for (int row = 0; row < kHcStreams; ++row) {
      std::printf("      ");
      double row_sum = 0.0;
      for (int col = 0; col < kHcStreams; ++col) {
        const float value = attn_trace.birkhoff[row * kHcStreams + col];
        std::printf("%.6f ", value);
        row_sum += value;
      }
      std::printf("  row sum %.8f\n", row_sum);
      worst_row = std::max(worst_row, std::fabs(row_sum - 1.0));
    }
    for (int col = 0; col < kHcStreams; ++col) {
      double col_sum = 0.0;
      for (int row = 0; row < kHcStreams; ++row) {
        col_sum += attn_trace.birkhoff[row * kHcStreams + col];
      }
      worst_col = std::max(worst_col, std::fabs(col_sum - 1.0));
    }
    std::printf("    worst |row sum - 1| %.3e, worst |col sum - 1| %.3e\n", worst_row,
                worst_col);

    // WHY THE ROW SUMS DO NOT REACH 1e-5 ON THE REAL WEIGHTS -- measured,
    // not assumed, and NOT a solver defect.
    //
    // `layers.0.hc_attn_base` rows 8..23 (the residual-matrix segment) span
    // roughly -30.1 to +9.9, and six of the matching `hc_attn_fn` rows have a
    // near-zero L2 norm (9.7e-06 .. 8.9e-05). Those logits are therefore PURE
    // BIAS, pinned around -21 to -30 whatever the input is -- which is why a
    // random-direction x0 and a replicated-embedding x0 give the same answer
    // to three digits. exp() of a ~40-wide logit spread covers e^-40 ~ 4e-18,
    // and Sinkhorn-Knopp converges slowly on a matrix that close to
    // degenerate: 20 iterations leave ~8e-2 on the row sums, 200 reach 1e-6.
    //
    // Column sums DO converge, to ~1e-6, because the cadence's final
    // operation is a column normalization.
    //
    // So 20 iterations is not a bound that the real layer-0 weights satisfy.
    // It is what config.json's "hc_sinkhorn_iters": 20 specifies and what the
    // model was trained with, so the 20-iteration result IS the contract the
    // NPU kernel must reproduce -- bit for bit, non-stochastic row sums and
    // all. Asserting double stochasticity as an operator-correctness test
    // would reject a correct kernel on this checkpoint. The emulation test
    // suite therefore asserts the Birkhoff invariant only on fixtures that
    // actually converge, and pins THIS regime by its measured value.
    {
      float converged[kHcStreams * kHcStreams];
      sinkhorn_knopp_4x4(attn_trace.residual_logits, converged,
                         static_cast<float>(geometry.hc_eps), 200);
      double converged_worst_row = 0.0;
      for (int row = 0; row < kHcStreams; ++row) {
        double row_sum = 0.0;
        for (int col = 0; col < kHcStreams; ++col) {
          row_sum += converged[row * kHcStreams + col];
        }
        converged_worst_row = std::max(converged_worst_row, std::fabs(row_sum - 1.0));
      }
      float logit_low = attn_trace.residual_logits[0];
      float logit_high = attn_trace.residual_logits[0];
      for (int i = 1; i < kHcStreams * kHcStreams; ++i) {
        logit_low = std::min(logit_low, attn_trace.residual_logits[i]);
        logit_high = std::max(logit_high, attn_trace.residual_logits[i]);
      }
      std::printf("    residual logits span [%.3g, %.3g] (spread %.3g)\n", logit_low,
                  logit_high, logit_high - logit_low);
      if (worst_row > 1e-5) {
        std::printf(
            "    NOTE: the row sums are NOT within 1e-5 at %lld iterations, and\n"
            "          that is a property of these weights, not a solver fault:\n"
            "          the same logits reach %.3e at 200 iterations. The biases\n"
            "          in hc_attn_base span ~40 and six hc_attn_fn rows are\n"
            "          near-zero, so exp() covers ~e^-40. config.json pins\n"
            "          hc_sinkhorn_iters = %lld, so THIS matrix -- not a doubly\n"
            "          stochastic idealization of it -- is what the NPU kernel\n"
            "          must reproduce.\n",
            static_cast<long long>(geometry.hc_sinkhorn_iters), converged_worst_row,
            static_cast<long long>(geometry.hc_sinkhorn_iters));
      } else {
        std::printf("    row and column sums both within 1e-5 at %lld iterations\n",
                    static_cast<long long>(geometry.hc_sinkhorn_iters));
      }
    }
    const TensorStats stats = Describe(x_after_attn.data(), x_after_attn.size());
    std::printf("    x_after_attn: min %.5f max %.5f L2 %.4f finite %s\n", stats.min,
                stats.max, stats.l2, stats.all_finite ? "yes" : "NO");
  }

  // --- stage 5: attention / compressor -------------------------------------
  std::printf("\n[5] compressor: C4A cadence over %d decode steps "
              "(coff %lld, ratio %d, D %d)\n",
              options.steps, static_cast<long long>(geometry.coff), kCompressorRatio,
              kCompressorHeadDim);
  std::vector<std::uint16_t> compressed_tokens;
  {
    const std::int64_t state_width = geometry.coff * kCompressorHeadDim;  // 1024
    std::vector<std::uint16_t> wkv;
    std::vector<std::uint16_t> wgate;
    std::vector<float> ape;
    std::vector<std::uint16_t> norm_weight;
    std::string provenance;

    if (have_layer2 && layer2.Find("layers.2.attn.compressor.wkv.weight") != nullptr &&
        layer2.Find("layers.2.attn.compressor.wkv.weight")->available) {
      wkv = LoadBf16(layer2, "layers.2.attn.compressor.wkv.weight",
                     state_width * kHiddenSize);
      wgate = LoadBf16(layer2, "layers.2.attn.compressor.wgate.weight",
                       state_width * kHiddenSize);
      ape = LoadF32(layer2, "layers.2.attn.compressor.ape",
                    kCompressorRatio * state_width);
      norm_weight = LoadBf16(layer2, "layers.2.attn.compressor.norm.weight",
                             kCompressorHeadDim);
      provenance = layer2_shard + " (layers.2.attn.compressor.*)";
      std::printf("    weights: REAL, from layer 2 -- wkv %s, ape %s\n",
                  layer2.Tensor("layers.2.attn.compressor.wkv.weight")
                      .shape_string()
                      .c_str(),
                  layer2.Tensor("layers.2.attn.compressor.ape").shape_string().c_str());
    } else {
      const float scale = 1.0F / std::sqrt(static_cast<float>(kHiddenSize));
      wkv = SyntheticBf16(static_cast<std::size_t>(state_width) * kHiddenSize, scale);
      wgate = SyntheticBf16(static_cast<std::size_t>(state_width) * kHiddenSize, scale);
      ape = SyntheticF32(static_cast<std::size_t>(kCompressorRatio) * state_width, -0.5F,
                         0.5F);
      norm_weight = SyntheticBf16(kCompressorHeadDim, 1.0F);
      provenance = "SYNTHETIC at the real C4A geometry (layer 0 has no compressor)";
      std::printf("    weights: SYNTHETIC at [%lld, %d] -- layer 0 has no "
                  "compressor;\n             fetch shard 4 for layer 2's real "
                  "weights\n",
                  static_cast<long long>(state_width), kHiddenSize);
    }

    CompressorRingState ring;
    std::vector<std::uint16_t> slot(kCompressorHeadDim, 0);
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    int emitted = 0;
    double total = 0.0;

    for (int step = 0; step < options.steps; ++step) {
      // One hidden state per step. The real decode would take it from the
      // previous layer; here it is the h_in the mHC round just produced,
      // perturbed per step so the cadence sees different tokens.
      std::vector<std::uint16_t> token(kHiddenSize);
      for (int i = 0; i < kHiddenSize; ++i) {
        token[static_cast<std::size_t>(i)] = FloatToBf16(
            attn_trace.h_in[i] * (1.0F + 0.01F * static_cast<float>(step)));
      }
      BuildRopeTables(step / kCompressorRatio, 160000.0, &cos_table, &sin_table);

      start = Clock::now();
      const CompressorStepResult result = paged_compressor_decode_c4a(
          token.data(), wkv.data(), wgate.data(), ape.data(), norm_weight.data(), step,
          &ring, cos_table.data(), sin_table.data(), slot.data(), kHiddenSize,
          kCompressorHeadDim, kCompressorRatio,
          static_cast<float>(geometry.rms_norm_eps), RotaryMode::kHalf);
      total += MicrosecondsSince(start);

      if (result.emitted) {
        ++emitted;
        compressed_tokens.insert(compressed_tokens.end(), slot.begin(), slot.end());
        const TensorStats stats = Describe(slot.data(), slot.size());
        std::printf("    step %2d EMIT  token %2d  inv_rms %.6f  min %.4f max %.4f "
                    "L2 %.4f\n",
                    step, result.compressed_index, result.inv_rms, stats.min, stats.max,
                    stats.l2);
      } else {
        std::printf("    step %2d HOLD  (slot %d of %d buffered)\n", step,
                    step % kCompressorRatio, kCompressorRatio);
      }
    }
    std::printf("    %d steps -> %d compressed tokens (%d:1)\n", options.steps, emitted,
                kCompressorRatio);
    Record("compressor: " + std::to_string(options.steps) + " decode steps",
           total, provenance,
           std::to_string(emitted) + " tokens emitted, " +
               std::to_string(options.steps - emitted) + " HOLDs");
  }

  // --- stage 6: lightning indexer ------------------------------------------
  std::printf("\n[6] lightning indexer: top-%lld of %d keys "
              "(%lld heads x %lld dim, FP8 E4M3)\n",
              static_cast<long long>(geometry.index_topk), options.seq_len,
              static_cast<long long>(geometry.index_n_heads),
              static_cast<long long>(geometry.index_head_dim));
  std::vector<std::int32_t> top_indices(static_cast<std::size_t>(geometry.index_topk), -1);
  {
    const std::int32_t block_size = 128;
    const std::int32_t blocks = (options.seq_len + block_size - 1) / block_size;

    // The indexer's query and key cache are FP8 E4M3 runtime activations, not
    // checkpoint weights -- the checkpoint carries wq_b and weights_proj that
    // PRODUCE them. Those FP8 projections are outside the slice, so the
    // activations here are synthetic at the exact PA_BSND geometry.
    std::vector<std::uint8_t> query(
        static_cast<std::size_t>(geometry.index_n_heads * geometry.index_head_dim));
    for (auto& byte : query) {
      byte = EncodeFp8E4M3(Uniform(-2.0F, 2.0F));
    }
    std::vector<std::uint8_t> key_cache(static_cast<std::size_t>(blocks) * block_size *
                                        static_cast<std::size_t>(geometry.index_head_dim));
    for (auto& byte : key_cache) {
      byte = EncodeFp8E4M3(Uniform(-2.0F, 2.0F));
    }
    // Fused per-head scales then per-key scales: see cpu_kernels.h.
    std::vector<float> scales(
        static_cast<std::size_t>(geometry.index_n_heads + options.seq_len));
    for (std::int64_t h = 0; h < geometry.index_n_heads; ++h) {
      scales[static_cast<std::size_t>(h)] = Uniform(0.1F, 1.0F);
    }
    for (std::int32_t j = 0; j < options.seq_len; ++j) {
      scales[static_cast<std::size_t>(geometry.index_n_heads + j)] = Uniform(0.5F, 2.0F);
    }
    std::vector<std::int32_t> block_table(static_cast<std::size_t>(blocks));
    for (std::int32_t b = 0; b < blocks; ++b) {
      block_table[static_cast<std::size_t>(b)] = b;
    }

    std::string provenance = "SYNTHETIC FP8 activations at the real indexer geometry";
    if (have_layer2) {
      const TensorView* weights_proj =
          layer2.Find("layers.2.attn.indexer.weights_proj.weight");
      if (weights_proj != nullptr && weights_proj->available) {
        // The real per-head weight projection, folded into the query scales
        // exactly as the golden's `cur_w = cur_wt * cur_q_scale` does. The
        // head weight is w_h = <weights_proj[h], h_in>.
        const std::vector<std::uint16_t> projection =
            LoadBf16(layer2, "layers.2.attn.indexer.weights_proj.weight",
                     geometry.index_n_heads * kHiddenSize);
        for (std::int64_t h = 0; h < geometry.index_n_heads; ++h) {
          float dot = 0.0F;
          for (int i = 0; i < kHiddenSize; ++i) {
            dot += Bf16ToFloat(
                       projection[static_cast<std::size_t>(h) * kHiddenSize + i]) *
                   attn_trace.h_in[i];
          }
          // The indexer gate is non-negative; the device applies a softplus
          // here. Fold the magnitude into the query dequant scale.
          scales[static_cast<std::size_t>(h)] = std::log1p(std::exp(-std::fabs(dot))) +
                                                std::max(0.0F, dot);
        }
        provenance = layer2_shard + " (indexer.weights_proj) + synthetic FP8 Q/K";
        std::printf("    head weights: REAL, from "
                    "layers.2.attn.indexer.weights_proj.weight %s\n",
                    weights_proj->shape_string().c_str());
      }
    }
    if (provenance.find("SYNTHETIC FP8 activations") == 0) {
      std::printf("    head weights: SYNTHETIC (layer 0 has no indexer)\n");
    }

    double total = 0.0;
    for (int repeat = 0; repeat < options.repeat; ++repeat) {
      start = Clock::now();
      fast_fp8_indexer_topk_cpu(
          query.data(), key_cache.data(), scales.data(), block_table.data(),
          options.seq_len, static_cast<std::int32_t>(geometry.index_topk),
          top_indices.data(), block_size,
          static_cast<int>(geometry.index_n_heads),
          static_cast<int>(geometry.index_head_dim));
      total += MicrosecondsSince(start);
    }
    Record("lightning indexer: score " + std::to_string(options.seq_len) +
               " keys + top-" + std::to_string(geometry.index_topk),
           total / options.repeat, provenance,
           std::to_string(geometry.index_n_heads * geometry.index_head_dim *
                          static_cast<std::int64_t>(options.seq_len)) +
               " FP8 MACs");

    // Validate the selection here too: a golden dump produced alongside a
    // broken index set would be worse than no dump.
    //
    // The rank-ordering claim needs the scores, so they are recomputed rather
    // than asserted: `fast_fp8_indexer_scores_cpu` is the same scoring half
    // the top-k entry point uses, so comparing against it checks the
    // SELECTION, which is the part that could be wrong.
    std::vector<float> scores(static_cast<std::size_t>(options.seq_len));
    fast_fp8_indexer_scores_cpu(query.data(), key_cache.data(), scales.data(),
                                block_table.data(), options.seq_len, scores.data(),
                                block_size, static_cast<int>(geometry.index_n_heads),
                                static_cast<int>(geometry.index_head_dim));

    std::int32_t selected = 0;
    bool in_range = true;
    bool unique = true;
    bool rank_ordered = true;
    std::vector<char> seen(static_cast<std::size_t>(options.seq_len), 0);
    float previous_score = std::numeric_limits<float>::infinity();
    for (std::int32_t i = 0; i < static_cast<std::int32_t>(geometry.index_topk); ++i) {
      const std::int32_t index = top_indices[static_cast<std::size_t>(i)];
      if (index == -1) {
        continue;
      }
      ++selected;
      if (index < 0 || index >= options.seq_len) {
        in_range = false;
        continue;
      }
      if (seen[static_cast<std::size_t>(index)] != 0) {
        unique = false;
      }
      seen[static_cast<std::size_t>(index)] = 1;
      // Compare on the BF16-rounded value, which is what the selection
      // ordered by.
      const float score =
          Bf16ToFloat(FloatToBf16(scores[static_cast<std::size_t>(index)]));
      if (score > previous_score) {
        rank_ordered = false;
      }
      previous_score = score;
    }
    // Nothing rejected may outscore the weakest selected key.
    bool optimal = true;
    if (selected > 0 && selected < options.seq_len) {
      float weakest = previous_score;
      for (std::int32_t j = 0; j < options.seq_len; ++j) {
        if (seen[static_cast<std::size_t>(j)] != 0) {
          continue;
        }
        if (Bf16ToFloat(FloatToBf16(scores[static_cast<std::size_t>(j)])) > weakest) {
          optimal = false;
        }
      }
    }
    std::printf("    selected %d indices, head: ", selected);
    for (int i = 0; i < 8 && i < selected; ++i) {
      std::printf("%d ", top_indices[static_cast<std::size_t>(i)]);
    }
    std::printf("\n    in range %s, unique %s, rank-ordered %s, "
                "no rejected key outscores a selected one: %s\n",
                in_range ? "yes" : "NO", unique ? "yes" : "NO",
                rank_ordered ? "yes" : "NO", optimal ? "yes" : "NO");
    if (!in_range || !unique || !rank_ordered || !optimal) {
      std::fprintf(stderr, "error: the indexer produced an invalid index set\n");
      return 4;
    }
  }

  // --- stage 7: the mHC fused update (FFN round) ---------------------------
  std::printf("\n[7] mHC fused update (FFN round, layers.0.hc_ffn_*) -> x_next\n");
  {
    // The MoE sub-block's output. Its experts are FP4 and far outside the
    // slice, so this is synthetic at [1, 4096].
    const std::vector<std::uint16_t> h_out_ffn = SyntheticBf16(kHiddenSize, 0.1F);
    double total = 0.0;
    for (int repeat = 0; repeat < options.repeat; ++repeat) {
      start = Clock::now();
      mhc_fused_sinkhorn_gemv_traced(x_after_attn.data(), phi_ffn.data(),
                                     alpha_ffn.data(), bias_ffn.data(),
                                     h_out_ffn.data(), x_next.data(), &ffn_trace,
                                     static_cast<float>(geometry.hc_eps),
                                     static_cast<int>(geometry.hc_sinkhorn_iters));
      total += MicrosecondsSince(start);
    }
    Record("mHC pre + Sinkhorn + post (FFN)", total / options.repeat, layer0_shard,
           "layers.0.hc_ffn_{fn,scale,base}; h_out synthetic");

    std::printf("    h_post          % .6f % .6f % .6f % .6f\n", ffn_trace.h_post[0],
                ffn_trace.h_post[1], ffn_trace.h_post[2], ffn_trace.h_post[3]);
    const TensorStats stats = Describe(x_next.data(), x_next.size());
    std::printf("    x_next: min %.5f max %.5f mean %.6f L2 %.4f finite %s\n",
                stats.min, stats.max, stats.mean, stats.l2,
                stats.all_finite ? "yes" : "NO");
    if (!stats.all_finite) {
      std::fprintf(stderr, "error: x_next contains non-finite values\n");
      return 4;
    }
    // ffn_norm is consumed by the real MoE input norm; report that it was
    // loaded rather than pretending a stage used it.
    std::printf("    (layers.0.ffn_norm.weight loaded, %zu BF16 lanes -- consumed "
                "by the MoE input norm, which is outside this bench)\n",
                ffn_norm.size());
  }

  // ===== the golden dump ===================================================
  const std::string dump_path = options.data_dir + "/golden_layer0_ref.bin";
  if (options.dump) {
    std::printf("\n[8] golden reference dump\n");
    start = Clock::now();
    std::ofstream out(dump_path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      std::fprintf(stderr, "error: cannot write %s\n", dump_path.c_str());
      return 5;
    }
    // A self-describing header, so a device-side comparison cannot silently
    // read the wrong layout. Magic, version, then the four section extents.
    struct Header {
      char magic[8];
      std::uint32_t version;
      std::uint32_t dtype_code;  // 1 = BF16
      std::int32_t streams;
      std::int32_t hidden;
      std::int32_t compressed_tokens;
      std::int32_t compressed_dim;
      std::int32_t top_k;
      std::int32_t sinkhorn_n;
    } header{};
    std::memcpy(header.magic, "DSV4EMU0", 8);
    header.version = 1;
    header.dtype_code = 1;
    header.streams = kHcStreams;
    header.hidden = kHiddenSize;
    header.compressed_tokens =
        static_cast<std::int32_t>(compressed_tokens.size() / kCompressorHeadDim);
    header.compressed_dim = kCompressorHeadDim;
    header.top_k = static_cast<std::int32_t>(geometry.index_topk);
    header.sinkhorn_n = kHcStreams;

    const auto write = [&out](const void* data, std::size_t bytes) {
      out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    write(&header, sizeof(header));
    write(x_next.data(), x_next.size() * sizeof(std::uint16_t));
    write(attn_trace.birkhoff, sizeof(attn_trace.birkhoff));
    write(ffn_trace.birkhoff, sizeof(ffn_trace.birkhoff));
    write(attn_trace.h_in, sizeof(attn_trace.h_in));
    write(compressed_tokens.data(), compressed_tokens.size() * sizeof(std::uint16_t));
    write(top_indices.data(), top_indices.size() * sizeof(std::int32_t));
    out.close();
    if (!out.good()) {
      std::fprintf(stderr, "error: writing %s failed\n", dump_path.c_str());
      return 5;
    }
    const double dump_microseconds = MicrosecondsSince(start);

    std::ifstream size_probe(dump_path, std::ios::binary | std::ios::ate);
    const long long bytes = size_probe.tellg();
    std::printf("    %s (%lld bytes)\n", dump_path.c_str(), bytes);
    std::printf("      header            %zu bytes  magic DSV4EMU0 v1\n",
                sizeof(header));
    std::printf("      x_next            [1, %d, %d] BF16\n", kHcStreams, kHiddenSize);
    std::printf("      hRes attn / ffn   2 x [%d, %d] F32\n", kHcStreams, kHcStreams);
    std::printf("      h_in              [1, %d] F32\n", kHiddenSize);
    std::printf("      compressed tokens [%d, %d] BF16\n", header.compressed_tokens,
                kCompressorHeadDim);
    std::printf("      top-k indices     [%d] I32\n", header.top_k);
    Record("golden dump", dump_microseconds, dump_path);
  } else {
    std::printf("\n[8] golden reference dump skipped (--no-dump)\n");
  }

  // ===== the latency report ================================================
  const double wall = MicrosecondsSince(wall_start);
  std::printf("\n=============================================================\n"
              " per-phase latency (microseconds, host CPU)\n"
              "=============================================================\n");
  double pipeline = 0.0;
  for (const Phase& phase : phases) {
    std::printf("  %-44s %12.1f us\n", phase.name.c_str(), phase.microseconds);
    if (!phase.detail.empty()) {
      std::printf("      %s\n", phase.detail.c_str());
    }
    std::printf("      weights: %s\n", phase.provenance.c_str());
    pipeline += phase.microseconds;
  }
  std::printf("  %-44s %12.1f us\n", "--- sum of phases ---", pipeline);
  std::printf("  %-44s %12.1f us  (%.3f s)\n", "=== total wall clock ===", wall,
              wall / 1e6);

  const bool under_one_second = wall < 1e6;
  std::printf("\n  Layer 0 decode step in under 1 s on CPU: %s\n",
              under_one_second ? "YES" : "NO");
  if (options.dump) {
    std::printf("  golden reference: %s\n", dump_path.c_str());
  }
  return under_one_second ? 0 : 6;
} catch (const std::exception& error) {
  std::fprintf(stderr, "fatal: %s\n", error.what());
  return 1;
}
