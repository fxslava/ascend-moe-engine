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

// Extracts isolated unit-test fixtures for the arch35 Stage-1 kernels
// (KER-01..KER-05) from golden_semi_e2e_ref.bin v3 plus the checkpoint.
//
// WHY THIS EXISTS AS A SEPARATE TOOL
//   The arch35 kernels are verified by running the real Ascend C sources under
//   CANN's CPU interpreter, which must be built with _GLIBCXX_USE_CXX11_ABI=0.
//   That makes it unable to link against libmoe_emu (built with the default
//   ABI), so the oracle cannot be called from the harness. This tool runs on
//   the normal-ABI side, writes every input and every expected output as raw
//   little-endian bytes, and the harness then only ever reads bytes.
//
// WHAT IS GOLDEN AND WHAT IS DERIVED
//   Each fixture's manifest.txt records, per tensor, where it came from:
//     golden   a byte slice of golden_semi_e2e_ref.bin v3, at the offset the
//              kernel contract spec tabulates
//     weights  a tensor copied out of the checkpoint
//     oracle   computed by src/emu (the numerical oracle) for a value v3 does
//              not serialize
//     derived  reconstructed from captured stage boundaries
//   A spec-mandated distinction: a missing v3 field is NOT a passing golden
//   check, so anything marked `oracle` is an oracle agreement test, not
//   evidence from the serialized reference.

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"
#include "moe/emu/safetensors_view.hpp"
#include "moe/emu/semi_e2e_pipeline.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace moe::emu;

namespace {

// --- golden v3 byte directory (docs/kernel_contracts_spec.md) --------------
constexpr std::size_t kGoldenBytes = 312608U;
constexpr std::size_t kHeaderBytes = 48U;
constexpr std::uint32_t kGoldenVersion = 3U;

constexpr std::size_t kOffXFinal = 48U;          // BF16 [4, 4096]
constexpr std::size_t kOffB0 = 32816U;           // FP32 [4, 4]
constexpr std::size_t kOffB2 = 32880U;
constexpr std::size_t kOffXInput = 32992U;       // BF16 [4, 4096]
constexpr std::size_t kOffXAfter0 = 65760U;
constexpr std::size_t kOffXAfter2 = 98528U;
constexpr std::size_t kOffHIn0 = 131296U;        // FP32 [4096]
constexpr std::size_t kOffHIn2 = 147680U;
constexpr std::size_t kOffHInMoe = 164064U;
constexpr std::size_t kOffBMoe = 180448U;        // FP32 [4, 4]
constexpr std::size_t kOffExperts = 180512U;     // FP32 [6, 4096]
constexpr std::size_t kOffCompressed = 311584U;  // BF16 [1, 512]

constexpr int kStreams = kHcStreams;
constexpr int kHidden = kHiddenSize;
constexpr int kFlat = kHcFlatWidth;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("extract_kernel_fixtures: " + message);
}

std::vector<std::uint8_t> ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    Fail("cannot open '" + path.string() + "'");
  }
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                   std::istreambuf_iterator<char>());
}

// One fixture directory: raw .bin files plus a provenance manifest.
class Fixture {
 public:
  Fixture(const fs::path& root, std::string name, std::string purpose)
      : dir_(root / name), name_(std::move(name)) {
    fs::create_directories(dir_);
    manifest_.open(dir_ / "manifest.txt");
    if (!manifest_) {
      Fail("cannot write manifest in '" + dir_.string() + "'");
    }
    manifest_ << "# fixture: " << name_ << "\n";
    manifest_ << "# " << purpose << "\n";
    manifest_ << "# columns: file dtype count bytes provenance\n";
  }

  template <typename T>
  void Add(const std::string& file, const char* dtype, const T* data,
           std::size_t count, const char* provenance) {
    const fs::path path = dir_ / (file + ".bin");
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      Fail("cannot write '" + path.string() + "'");
    }
    const std::size_t bytes = count * sizeof(T);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
      Fail("short write to '" + path.string() + "'");
    }
    manifest_ << file << ".bin " << dtype << " " << count << " " << bytes << " "
              << provenance << "\n";
    total_bytes_ += bytes;
  }

  // A slice of the golden file, bounds-checked against its declared length.
  void AddGolden(const std::string& file, const char* dtype,
                 const std::vector<std::uint8_t>& golden, std::size_t offset,
                 std::size_t bytes, std::size_t element_size) {
    if (offset + bytes > golden.size()) {
      Fail("golden slice for '" + file + "' runs past the file");
    }
    const fs::path path = dir_ / (file + ".bin");
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(golden.data() + offset),
              static_cast<std::streamsize>(bytes));
    if (!out) {
      Fail("short write to '" + path.string() + "'");
    }
    manifest_ << file << ".bin " << dtype << " " << (bytes / element_size) << " "
              << bytes << " golden@" << offset << "\n";
    total_bytes_ += bytes;
  }

  const std::string& name() const { return name_; }
  std::size_t total_bytes() const { return total_bytes_; }

 private:
  fs::path dir_;
  std::string name_;
  std::ofstream manifest_;
  std::size_t total_bytes_ = 0;
};

const TensorView& Need(const SafetensorsView& weights, const std::string& name) {
  const TensorView* view = weights.Find(name);
  if (view == nullptr || !view->available) {
    Fail("checkpoint has no usable tensor '" + name + "'");
  }
  return *view;
}

// BF16 round-trip, which is what every BF16 kernel output is compared against.
std::vector<std::uint16_t> ToBf16(const std::vector<float>& values) {
  std::vector<std::uint16_t> out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = FloatToBf16(values[i]);
  }
  return out;
}

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

}  // namespace

int main(int argc, char** argv) {
  std::string data_dir = "data/dsv4_layer0";
  std::string out_dir = "data/dsv4_layer0/kernel_fixtures";
  std::int64_t token_id = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_next = (i + 1) < argc;
    if (arg == "--data-dir" && has_next) {
      data_dir = argv[++i];
    } else if (arg == "--out-dir" && has_next) {
      out_dir = argv[++i];
    } else if (arg == "--token-id" && has_next) {
      token_id = std::stoll(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: dsv4_extract_kernel_fixtures [--data-dir DIR]"
                   " [--out-dir DIR] [--token-id N]\n");
      return 2;
    }
  }

  try {
    const fs::path data(data_dir);
    const fs::path root(out_dir);
    fs::create_directories(root);

    // --- 1. validate the golden reference -------------------------------
    const std::vector<std::uint8_t> golden =
        ReadFile(data / "golden_semi_e2e_ref.bin");
    if (golden.size() != kGoldenBytes) {
      Fail("golden_semi_e2e_ref.bin is " + std::to_string(golden.size()) +
           " bytes, expected exactly " + std::to_string(kGoldenBytes) + " (v3)");
    }
    if (std::memcmp(golden.data(), "DSV4SE2E", 8) != 0) {
      Fail("golden_semi_e2e_ref.bin has the wrong magic");
    }
    std::uint32_t version = 0;
    std::memcpy(&version, golden.data() + 8, sizeof(version));
    if (version != kGoldenVersion) {
      Fail("golden_semi_e2e_ref.bin is version " + std::to_string(version) +
           ", expected 3");
    }
    std::int32_t streams = 0;
    std::int32_t hidden = 0;
    std::memcpy(&streams, golden.data() + 16, sizeof(streams));
    std::memcpy(&hidden, golden.data() + 20, sizeof(hidden));
    if (streams != kStreams || hidden != kHidden) {
      Fail("golden geometry is [" + std::to_string(streams) + ", " +
           std::to_string(hidden) + "], expected [4, 4096]");
    }
    std::printf("golden_semi_e2e_ref.bin: v%u, %zu bytes, [%d, %d]  OK\n", version,
                golden.size(), streams, hidden);

    // --- 2. run the oracle pipeline -------------------------------------
    SemiE2ePipeline pipeline((data / "dsv4_semi_e2e.safetensors").string(),
                             (data / "config.json").string());
    SemiE2eOptions options;
    options.token_id = token_id;
    std::vector<std::uint16_t> x_next;
    const SemiE2eReport report = pipeline.Run(options, &x_next);
    if (!report.all_finite) {
      Fail("the oracle run produced a non-finite activation");
    }
    const SafetensorsView& weights = pipeline.weights();
    std::printf("oracle run: token %lld, %d routed experts, %d compressed token(s)\n",
                static_cast<long long>(report.token_id), report.routing.count,
                report.compressed_tokens_emitted);

    // Cross-check the serialized golden against the live oracle, so a stale
    // golden file is caught here rather than silently weakening every fixture.
    {
      const auto* golden_h_in0 =
          reinterpret_cast<const float*>(golden.data() + kOffHIn0);
      double worst = 0.0;
      for (int i = 0; i < kHidden; ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(
                                    golden_h_in0[i] - report.layer0_h_in[static_cast<std::size_t>(i)])));
      }
      std::printf("golden vs live oracle, layer0 h_in: max abs delta %.3e\n", worst);
      if (worst > 1e-5) {
        Fail("the golden file disagrees with the current oracle; regenerate it");
      }
    }

    std::vector<Fixture> written;

    // =====================================================================
    // KER-01 -- three rounds, one per mHC site.
    // =====================================================================
    struct PreCase {
      const char* name;
      const char* stem;   // hc_attn / hc_ffn
      const char* norm;   // attn_norm / ffn_norm
      int layer;
      std::size_t x_offset;
      std::size_t h_in_offset;
      std::size_t b_offset;
      const std::vector<float>* h_norm;
      const MhcSinkhornTrace* trace;
    };
    const PreCase pre_cases[] = {
        {"ker01_layer0", "hc_attn", "attn_norm", 0, kOffXInput, kOffHIn0, kOffB0,
         &report.layer0_h_norm, &report.layer0_attn_trace},
        {"ker01_layer2", "hc_attn", "attn_norm", 2, kOffXAfter0, kOffHIn2, kOffB2,
         &report.layer2_h_norm, &report.layer2_attn_trace},
        {"ker01_moe", "hc_ffn", "ffn_norm", 1, kOffXAfter2, kOffHInMoe, kOffBMoe,
         &report.moe_h_norm, &report.moe_ffn_trace},
    };
    for (const PreCase& c : pre_cases) {
      Fixture f(root, c.name,
                "KER-01 fused_mhc_pre_norm_sinkhorn, one mHC round. h_raw and B "
                "are golden; h_norm and post are not serialized in v3.");
      const std::string prefix = "layers." + std::to_string(c.layer) + ".";
      std::vector<float> phi(static_cast<std::size_t>(kHcPhiRows) * kHcPhiCols);
      std::vector<float> alpha(kHcAlphaElements);
      std::vector<float> bias(kHcProjectionRows);
      Need(weights, prefix + c.stem + "_fn").CopyTo(phi.data(), phi.size());
      Need(weights, prefix + c.stem + "_scale").CopyTo(alpha.data(), alpha.size());
      Need(weights, prefix + c.stem + "_base").CopyTo(bias.data(), bias.size());
      const TensorView& gamma = Need(weights, prefix + c.norm + ".weight");

      f.AddGolden("x", "bf16", golden, c.x_offset, 2U * kFlat, 2U);
      f.Add("phi", "fp32", phi.data(), phi.size(), "weights");
      f.Add("alpha", "fp32", alpha.data(), alpha.size(), "weights");
      f.Add("bias", "fp32", bias.data(), bias.size(), "weights");
      f.Add("gamma", "bf16", static_cast<const std::uint16_t*>(gamma.data_ptr),
            static_cast<std::size_t>(kHidden), "weights");
      f.AddGolden("expect_h_raw", "fp32", golden, c.h_in_offset, 4U * kHidden, 4U);
      f.AddGolden("expect_birkhoff", "fp32", golden, c.b_offset, 64U, 4U);
      const std::vector<std::uint16_t> h_norm_bf = ToBf16(*c.h_norm);
      f.Add("expect_h_norm", "bf16", h_norm_bf.data(), h_norm_bf.size(),
            "derived(stage-boundary-capture)");
      f.Add("expect_post", "fp32", c.trace->h_post, static_cast<std::size_t>(kStreams),
            "oracle(mhc_pre_sinkhorn_cpu)");
      written.push_back(std::move(f));
    }

    // =====================================================================
    // KER-02 -- three fold-backs. Every input and the expected y are golden
    // or captured stage boundaries, so this ties directly to v3.
    // =====================================================================
    struct PostCase {
      const char* name;
      std::size_t x_offset;
      std::size_t y_offset;
      std::size_t b_offset;
      const std::vector<std::uint16_t>* h_out;
      const MhcSinkhornTrace* trace;
    };
    const PostCase post_cases[] = {
        {"ker02_layer0", kOffXInput, kOffXAfter0, kOffB0, &report.layer0_h_out,
         &report.layer0_attn_trace},
        {"ker02_layer2", kOffXAfter0, kOffXAfter2, kOffB2, &report.layer2_h_out,
         &report.layer2_attn_trace},
        {"ker02_moe", kOffXAfter2, kOffXFinal, kOffBMoe, &report.moe_h_out,
         &report.moe_ffn_trace},
    };
    for (const PostCase& c : post_cases) {
      Fixture f(root, c.name,
                "KER-02 fused_mhc_post_residual, y = B^T x + post (x) h_out. x, B "
                "and y are golden; h_out and post are captured stage boundaries.");
      f.AddGolden("x", "bf16", golden, c.x_offset, 2U * kFlat, 2U);
      f.AddGolden("birkhoff", "fp32", golden, c.b_offset, 64U, 4U);
      f.Add("post", "fp32", c.trace->h_post, static_cast<std::size_t>(kStreams),
            "oracle(mhc_pre_sinkhorn_cpu)");
      f.Add("h_out", "bf16", c.h_out->data(), c.h_out->size(),
            "derived(stage-boundary-capture)");
      f.AddGolden("expect_y", "bf16", golden, c.y_offset, 2U * kFlat, 2U);
      written.push_back(std::move(f));
    }

    // =====================================================================
    // KER-03 -- the compressor, two groups.
    //
    // Group 1 reproduces the pipeline's four synthetic decode tokens and is
    // checked against the golden emitted token. Group 2 runs eight tokens so
    // the SECOND emission exercises the previous-a path, which a four-row
    // window would silently get wrong; v3 does not contain it, so its
    // expectation is the oracle's.
    // =====================================================================
    {
      const std::string prefix = "layers.2.attn.compressor.";
      const std::int64_t state_width = kCompressorCoff * kCompressorHeadDim;
      const TensorView& wkv = Need(weights, prefix + "wkv.weight");
      const TensorView& wgate = Need(weights, prefix + "wgate.weight");
      const TensorView& norm_w = Need(weights, prefix + "norm.weight");
      std::vector<float> ape(static_cast<std::size_t>(kCompressorRatio) * state_width);
      Need(weights, prefix + "ape").CopyTo(ape.data(), ape.size());

      for (int groups = 1; groups <= 2; ++groups) {
        const int steps = groups * kCompressorRatio;
        Fixture f(root, groups == 1 ? "ker03_group1" : "ker03_group2",
                  groups == 1
                      ? "KER-03 C4A compressor, one group (4 steps). The emitted "
                        "token is golden; HOLD steps must leave the slot untouched."
                      : "KER-03 C4A compressor, two groups (8 steps). Certifies the "
                        "cross-boundary previous-a path; expectation is the oracle's.");
        // The pipeline's synthetic tokens: layer 2's normalized h_in, scaled.
        std::vector<std::uint16_t> tokens(static_cast<std::size_t>(steps) * kHidden);
        for (int step = 0; step < steps; ++step) {
          for (int i = 0; i < kHidden; ++i) {
            tokens[static_cast<std::size_t>(step) * kHidden + i] =
                FloatToBf16(report.layer2_h_norm[static_cast<std::size_t>(i)] *
                            (1.0F + 0.01F * static_cast<float>(step)));
          }
        }
        f.Add("x_tokens", "bf16", tokens.data(), tokens.size(),
              "derived(stage-boundary-capture)");
        f.Add("wkv", "bf16", static_cast<const std::uint16_t*>(wkv.data_ptr),
              static_cast<std::size_t>(state_width) * kHidden, "weights");
        f.Add("wgate", "bf16", static_cast<const std::uint16_t*>(wgate.data_ptr),
              static_cast<std::size_t>(state_width) * kHidden, "weights");
        f.Add("ape", "fp32", ape.data(), ape.size(), "weights");
        f.Add("norm_weight", "bf16",
              static_cast<const std::uint16_t*>(norm_w.data_ptr),
              static_cast<std::size_t>(kCompressorHeadDim), "weights");

        // One RoPE table pair per compressed index, for the compressor theta.
        std::vector<float> rope_cos_all;
        std::vector<float> rope_sin_all;
        std::vector<float> cos_table;
        std::vector<float> sin_table;
        for (int g = 0; g < groups; ++g) {
          BuildRopeTables(g, 160000.0, kRopeHeadDim, &cos_table, &sin_table);
          rope_cos_all.insert(rope_cos_all.end(), cos_table.begin(), cos_table.end());
          rope_sin_all.insert(rope_sin_all.end(), sin_table.begin(), sin_table.end());
        }
        f.Add("rope_cos", "fp32", rope_cos_all.data(), rope_cos_all.size(),
              "derived(compress_rope_theta=160000)");
        f.Add("rope_sin", "fp32", rope_sin_all.data(), rope_sin_all.size(),
              "derived(compress_rope_theta=160000)");

        // Run the oracle over the same cadence and record every emission.
        CompressorRingState ring;
        std::vector<std::uint16_t> slot(kCompressorHeadDim, 0);
        std::vector<std::uint16_t> emissions;
        std::vector<std::uint32_t> emitted_flags(static_cast<std::size_t>(steps));
        for (int step = 0; step < steps; ++step) {
          BuildRopeTables(step / kCompressorRatio, 160000.0, kRopeHeadDim, &cos_table,
                          &sin_table);
          const CompressorStepResult result = paged_compressor_decode_c4a(
              tokens.data() + static_cast<std::size_t>(step) * kHidden,
              wkv.data_ptr, wgate.data_ptr, ape.data(), norm_w.data_ptr, step, &ring,
              cos_table.data(), sin_table.data(), slot.data(), kHidden,
              kCompressorHeadDim, kCompressorRatio, kRmsNormEpsilon, RotaryMode::kHalf);
          emitted_flags[static_cast<std::size_t>(step)] = result.emitted ? 1U : 0U;
          if (result.emitted) {
            emissions.insert(emissions.end(), slot.begin(), slot.end());
          }
        }
        f.Add("expect_emitted", "u32", emitted_flags.data(), emitted_flags.size(),
              "oracle(paged_compressor_decode_c4a)");
        f.Add("expect_tokens", "bf16", emissions.data(), emissions.size(),
              "oracle(paged_compressor_decode_c4a)");
        if (groups == 1) {
          // The single emission of group 1 is the serialized golden token.
          f.AddGolden("expect_token_golden", "bf16", golden, kOffCompressed, 1024U, 2U);
          const auto* want =
              reinterpret_cast<const std::uint16_t*>(golden.data() + kOffCompressed);
          int mismatches = 0;
          for (int i = 0; i < kCompressorHeadDim; ++i) {
            if (emissions[static_cast<std::size_t>(i)] != want[i]) {
              ++mismatches;
            }
          }
          std::printf("ker03_group1: oracle emission vs golden token: %d/%d BF16 "
                      "words differ\n", mismatches, kCompressorHeadDim);
          if (mismatches != 0) {
            Fail("the oracle's first emission does not reproduce the golden token");
          }
        }
        written.push_back(std::move(f));
      }
    }

    // =====================================================================
    // KER-04 -- the FP8 indexer. v3 serializes NOTHING for this path, so the
    // whole fixture (inputs and expected indices) is built here, reproducing
    // the pipeline's synthetic Q/K bytes bit for bit.
    // =====================================================================
    {
      Fixture f(root, "ker04_indexer",
                "KER-04 fused_fp8_indexer_topk. v3 serializes no indexer output, "
                "so expected indices are the oracle's; Q/K bytes are synthetic at "
                "the real PA_BSND geometry and the per-head gate is real.");
      const std::int32_t seq_len = 2048;
      const std::int32_t block_size = 128;
      const std::int32_t blocks = (seq_len + block_size - 1) / block_size;

      // The real per-head gate: weights_proj against layer 2's normalized h_in.
      const TensorView& weights_proj =
          Need(weights, "layers.2.attn.indexer.weights_proj.weight");
      std::vector<float> head_weights(kIndexerHeads);
      bf16_gemv_cpu(static_cast<const std::uint16_t*>(weights_proj.data_ptr),
                    kIndexerHeads, kHidden, report.layer2_h_norm.data(),
                    head_weights.data());

      std::vector<float> scales(static_cast<std::size_t>(kIndexerHeads) + seq_len);
      for (int head = 0; head < kIndexerHeads; ++head) {
        scales[static_cast<std::size_t>(head)] = std::fabs(head_weights[static_cast<std::size_t>(head)]);
      }
      // Same engine, same seed, same draw order as the pipeline.
      std::mt19937 engine(0xC0FFEEu ^ static_cast<std::uint32_t>(token_id));
      std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
      std::vector<std::uint8_t> query(static_cast<std::size_t>(kIndexerHeads) *
                                      kIndexerHeadDim);
      for (std::uint8_t& byte : query) {
        byte = EncodeFp8E4M3(distribution(engine));
      }
      std::vector<std::uint8_t> key_cache(
          static_cast<std::size_t>(blocks) * block_size * kIndexerHeadDim);
      for (std::uint8_t& byte : key_cache) {
        byte = EncodeFp8E4M3(distribution(engine));
      }
      for (std::int32_t j = 0; j < seq_len; ++j) {
        scales[static_cast<std::size_t>(kIndexerHeads + j)] =
            0.5F + distribution(engine) * 0.1F;
      }
      std::vector<std::int32_t> block_table(static_cast<std::size_t>(blocks));
      for (std::int32_t b = 0; b < blocks; ++b) {
        block_table[static_cast<std::size_t>(b)] = b;
      }

      std::vector<float> scores(static_cast<std::size_t>(seq_len));
      fast_fp8_indexer_scores_cpu(query.data(), key_cache.data(), scales.data(),
                                  block_table.data(), seq_len, scores.data(),
                                  block_size, kIndexerHeads, kIndexerHeadDim);
      std::vector<std::int32_t> indices(kIndexerTopK, -1);
      fast_fp8_indexer_topk_cpu(query.data(), key_cache.data(), scales.data(),
                                block_table.data(), seq_len, kIndexerTopK,
                                indices.data(), block_size, kIndexerHeads,
                                kIndexerHeadDim);

      // The composite-key tie-break in the kernel assumes non-negative scores.
      float min_score = scores.empty() ? 0.0F : scores[0];
      for (const float s : scores) {
        min_score = std::min(min_score, s);
      }
      std::printf("ker04_indexer: %d scores, min %.6g (the kernel's ordering "
                  "requires >= 0)\n", seq_len, static_cast<double>(min_score));
      if (min_score < 0.0F) {
        Fail("a negative indexer score would break the composite-key ordering");
      }

      f.Add("query", "fp8e4m3", query.data(), query.size(), "synthetic(mt19937)");
      f.Add("key_cache", "fp8e4m3", key_cache.data(), key_cache.size(),
            "synthetic(mt19937)");
      f.Add("dequant_scales", "fp32", scales.data(), scales.size(),
            "weights(head gate) + synthetic(key scales)");
      f.Add("block_table", "i32", block_table.data(), block_table.size(),
            "synthetic(identity)");
      f.Add("expect_scores", "fp32", scores.data(), scores.size(),
            "oracle(fast_fp8_indexer_scores_cpu)");
      f.Add("expect_indices", "i32", indices.data(), indices.size(),
            "oracle(fast_fp8_indexer_topk_cpu)");
      written.push_back(std::move(f));
    }

    // =====================================================================
    // KER-05 -- one fixture per GEMV of the first routed expert, plus the
    // golden six expert outputs for the integrated check.
    // =====================================================================
    {
      const int expert = report.routing.experts[0];
      const std::string prefix =
          "layers.1.ffn.experts." + std::to_string(expert) + ".";
      struct GemvCase {
        const char* name;
        const char* stem;
        int rows;
        int cols;
      };
      const GemvCase gemv_cases[] = {
          {"ker05_w1", "w1", kMoeIntermediate, kHidden},
          {"ker05_w3", "w3", kMoeIntermediate, kHidden},
          {"ker05_w2", "w2", kHidden, kMoeIntermediate},
      };
      for (const GemvCase& c : gemv_cases) {
        Fixture f(root, c.name,
                  "KER-05 LaunchMxfp4ExpertGemv, one standalone GEMV of a real "
                  "routed expert. v3 stores no isolated GEMV output, so the "
                  "expectation is the oracle's; the input is nonuniform so a "
                  "within-byte nibble swap cannot pass.");
        const TensorView& packed = Need(weights, prefix + c.stem + ".weight");
        const TensorView& scale = Need(weights, prefix + c.stem + ".scale");
        Mxfp4Matrix matrix;
        matrix.packed = static_cast<const std::uint8_t*>(packed.data_ptr);
        matrix.scale = static_cast<const std::uint8_t*>(scale.data_ptr);
        matrix.rows = c.rows;
        matrix.cols = c.cols;

        // A nonuniform input: an all-ones vector cannot distinguish a nibble
        // swap, because it sums both halves of every byte symmetrically.
        std::vector<float> x(static_cast<std::size_t>(c.cols));
        for (int i = 0; i < c.cols; ++i) {
          x[static_cast<std::size_t>(i)] =
              std::sin(0.013F * static_cast<float>(i) + 0.7F) *
              (1.0F + 0.25F * static_cast<float>(i % 7));
        }
        std::vector<float> y(static_cast<std::size_t>(c.rows));
        mxfp4_gemv_cpu(matrix, x.data(), y.data());

        f.Add("packed", "u8", matrix.packed,
              static_cast<std::size_t>(c.rows) * (c.cols / 2), "weights");
        f.Add("scale", "fp8e8m0", matrix.scale,
              static_cast<std::size_t>(c.rows) * (c.cols / kMxfp4GroupSize), "weights");
        f.Add("x", "fp32", x.data(), x.size(), "synthetic(nonuniform)");
        f.Add("expect_y", "fp32", y.data(), y.size(), "oracle(mxfp4_gemv_cpu)");
        written.push_back(std::move(f));
      }

      Fixture f(root, "ker05_integrated",
                "KER-05 integrated reference: the six golden expert outputs and "
                "the routed/shared sums, for the full SwiGLU path.");
      f.AddGolden("expect_expert_outputs", "fp32", golden, kOffExperts,
                  98304U, 4U);
      f.AddGolden("expect_routed_sum", "fp32", golden, 278816U, 16384U, 4U);
      f.AddGolden("expect_shared_sum", "fp32", golden, 295200U, 16384U, 4U);
      f.Add("routed_expert_ids", "i32", report.routing.experts,
            static_cast<std::size_t>(report.routing.count), "oracle(routing)");
      written.push_back(std::move(f));
    }

    // --- summary ---------------------------------------------------------
    std::size_t total = 0;
    std::printf("\nfixtures written to %s\n", root.string().c_str());
    for (const Fixture& f : written) {
      std::printf("  %-20s %9.2f MiB\n", f.name().c_str(),
                  static_cast<double>(f.total_bytes()) / (1024.0 * 1024.0));
      total += f.total_bytes();
    }
    std::printf("  %-20s %9.2f MiB total, %zu fixtures\n", "", 
                static_cast<double>(total) / (1024.0 * 1024.0), written.size());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
