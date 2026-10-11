#include "moe/core/acl_guard.hpp"
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

// dsv4_runner -- standalone DeepSeek-V4 Flash decode on Ascend 950PR.
//
//   ./dsv4_runner --weights <path> --prompt "..." --max-new-tokens 512 --vram-slots <K>
//   ./dsv4_runner --config /mnt/c/models/DeepSeek-V4-Flash/config.json --dry-run
//
// No Python, no PyTorch: `ldd` on this binary carries libascendcl, libnnopbase
// and the libopapi family, and nothing else from a framework.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/model_config.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/pipeline/lattice_moe_block.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/pipeline/standard_aclnn_moe_block.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/diagnostics/inference_stats.hpp"
#if ASCEND_MOCK_RUNTIME
#include "mock_allocator.hpp"
#include "mock_weight_source.hpp"
#endif

namespace ascend_moe {
namespace {

// The checkpoint configuration read when neither --config nor a config.json
// beside the weights names a file. WSL keeps the Windows model directory
// under /mnt/c.
#if defined(_WIN32) || defined(__CYGWIN__)
constexpr const char* kDefaultModelConfigPath = "C:/models/DeepSeek-V4-Flash/config.json";
#else
constexpr const char* kDefaultModelConfigPath = "/mnt/c/models/DeepSeek-V4-Flash/config.json";
#endif

void PrintUsage() {
  std::printf(
      "dsv4_runner -- DeepSeek-V4 Flash decode, Ascend 950PR, ACLNN V5\n"
      "\n"
      "  --weights <path>          checkpoint directory or .safetensors file\n"
      "  --config <path>           model config.json to parse and validate (default: try the weights\n"
      "                            directory, then %s; a missing file falls back to the\n"
      "                            compiled DeepSeek-V4 Flash contract, an unreadable EXPLICIT path is\n"
      "                            an error)\n"
      "  --prompt \"...\"            prompt text (token ids are read from --prompt-ids)\n"
      "  --prompt-ids a,b,c        explicit prompt token ids; this runner embeds no tokenizer\n"
      "  --prompt-tokens <n>      synthesize n zero token IDs (requires --synthetic-weights)\n"
      "  --max-new-tokens <n>      tokens to generate (default 512)\n"
      "  --vram-slots <K>          resident routed-expert slots in HBM (default: as many as fit)\n"
      "  --device <id>             NPU device id (default 0)\n"
      "  --block-size <n>          paged KV block size (default 128)\n"
      "  --max-context <n>         reserved context length (default 8192)\n"
      "  --kv-lora-rank <n>        MLA latent width (default %lld, family default)\n"
      "  --qk-rope-head-dim <n>    MLA rope slice width (default %lld, family default)\n"
      "  --qk-nope-head-dim <n>    MLA nope head width (default %lld, family default)\n"
      "  --v-head-dim <n>          MLA value head width (default %lld, family default)\n"
      "  --moe-path fused|decomposed   expert GEMM chain inside the standard aclnn block (default fused)\n"
      "  --moe-backend standard|lattice  IRoutedMoeBlock selection (default standard; the 2-bit\n"
      "                            Leech-lattice backend is a skeleton and refuses at plan time)\n"
      "  --mhc-mixing fused|staged how the mHC residual map B_l is produced (default fused):\n"
      "                            fused  = aclnnHcPre -> aclnnHcPost, 2 launches/round, 0 host plans;\n"
      "                                     the 20-iteration Sinkhorn runs in HcPre's vector epilogue\n"
      "                            staged = aclnnMhcPre -> aclnnMhcSinkhorn -> aclnnMhcPost; the middle\n"
      "                                     operator has no repeatable form, so it costs one host plan\n"
      "                                     per round. Fallback only\n"
      "  --compress-ratios <list>  comma-separated per-layer token-compression ratio, one per layer:\n"
      "                            0 or 1 = SWA, 4 = CSA (compressor 4:1 + indexer top-512 + shared-KV\n"
      "                            sparse attention), 128 = HCA (compressor 128:1 + direct shared-KV).\n"
      "                            Overrides the checkpoint; omitted leaves every layer on SWA\n"
      "  --gating-norm-type <n>    aclnnMoeGatingTopKV2 normType (default -1: scores arrive pre-normalized\n"
      "                            from the decomposed aclnnSoftplus -> aclnnSqrt sqrtsoftplus chain)\n"
      "  --dense-group-size <n>    aclnnQuantMatmulV5 groupSize (default 0, UNVERIFIED)\n"
      "  --routed-coverage <n>     |Set_Device| + |Set_Host|, default %lld (bring-up subset below that)\n"
      "  --synthetic-weights       deterministic in-memory weights; no files are opened\n"
      "  --dry-run                 build, plan, report and exit without decoding; with no weights\n"
      "                            given, parse and validate the model configuration only\n"
      "  --report <path>           write the full report there as well as to stdout\n"
      "  --diag-json <path>        structured diagnostics, including dry runs; null if unmeasured\n"
      "  --stats-json <path>       write machine-readable run stats (TTFT, TPOT, swap bytes, launch\n"
      "                            counters) there after a decode; consumed by python/benchmarks\n"
      "  --verbose                 print the arena ledger and operator inventory\n"
      "  --help\n",
      kDefaultModelConfigPath, static_cast<long long>(kDefaultKvLoraRank),
      static_cast<long long>(kDefaultQkRopeHeadDim), static_cast<long long>(kDefaultQkNopeHeadDim),
      static_cast<long long>(kDefaultVHeadDim), static_cast<long long>(kTotalRoutedExperts));
}

int64_t ParseInt(const char* text, const char* flag) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  DSV4_REQUIRE(end != nullptr && *end == '\0', flag << " expects an integer, got '" << text << "'");
  return static_cast<int64_t>(value);
}

std::vector<int32_t> ParseIdList(const std::string& text) {
  std::vector<int32_t> ids;
  std::stringstream stream(text);
  std::string piece;
  while (std::getline(stream, piece, ',')) {
    if (!piece.empty()) {
      ids.push_back(static_cast<int32_t>(ParseInt(piece.c_str(), "--prompt-ids")));
    }
  }
  return ids;
}

struct Arguments {
  RuntimeConfig config;
  std::vector<int32_t> prompt_ids;
  std::string model_config_path;  // empty = no --config flag given
  std::string stats_json_path;    // empty = no --stats-json flag given
  std::string diag_json_path;
  int64_t prompt_tokens = 0;
  bool help = false;
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments arguments;
  RuntimeConfig& config = arguments.config;
  bool mla_from_cli = false;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto next = [&](const char* name) -> const char* {
      DSV4_REQUIRE(index + 1 < argc, name << " needs a value");
      return argv[++index];
    };
    if (flag == "--help" || flag == "-h") {
      arguments.help = true;
    } else if (flag == "--weights") {
      config.weights_path = next("--weights");
    } else if (flag == "--config") {
      arguments.model_config_path = next("--config");
    } else if (flag == "--prompt") {
      config.prompt = next("--prompt");
    } else if (flag == "--prompt-ids") {
      arguments.prompt_ids = ParseIdList(next("--prompt-ids"));
    } else if (flag == "--prompt-tokens") {
      arguments.prompt_tokens = ParseInt(next("--prompt-tokens"), "--prompt-tokens");
      DSV4_REQUIRE(arguments.prompt_tokens > 0, "--prompt-tokens must be positive");
    } else if (flag == "--max-new-tokens") {
      config.max_new_tokens = ParseInt(next("--max-new-tokens"), "--max-new-tokens");
    } else if (flag == "--vram-slots") {
      config.vram_slots = ParseInt(next("--vram-slots"), "--vram-slots");
    } else if (flag == "--device") {
      config.device_id = static_cast<int32_t>(ParseInt(next("--device"), "--device"));
    } else if (flag == "--block-size") {
      config.block_size = ParseInt(next("--block-size"), "--block-size");
    } else if (flag == "--max-context") {
      config.max_context_len = ParseInt(next("--max-context"), "--max-context");
    } else if (flag == "--kv-lora-rank") {
      config.mla.kv_lora_rank = ParseInt(next("--kv-lora-rank"), "--kv-lora-rank");
      mla_from_cli = true;
    } else if (flag == "--qk-rope-head-dim") {
      config.mla.qk_rope_head_dim = ParseInt(next("--qk-rope-head-dim"), "--qk-rope-head-dim");
      mla_from_cli = true;
    } else if (flag == "--qk-nope-head-dim") {
      config.mla.qk_nope_head_dim = ParseInt(next("--qk-nope-head-dim"), "--qk-nope-head-dim");
      mla_from_cli = true;
    } else if (flag == "--v-head-dim") {
      config.mla.v_head_dim = ParseInt(next("--v-head-dim"), "--v-head-dim");
      mla_from_cli = true;
    } else if (flag == "--moe-path") {
      const std::string value = next("--moe-path");
      DSV4_REQUIRE(value == "fused" || value == "decomposed",
                   "--moe-path expects 'fused' or 'decomposed', got '" << value << "'");
      config.moe_path = value == "fused" ? MoePath::kFused : MoePath::kDecomposed;
    } else if (flag == "--moe-backend") {
      const std::string value = next("--moe-backend");
      DSV4_REQUIRE(value == "standard" || value == "lattice",
                   "--moe-backend expects 'standard' or 'lattice', got '" << value << "'");
      config.moe_backend = value == "lattice" ? MoeBackend::kLattice24 : MoeBackend::kStandardAclnn;
    } else if (flag == "--compress-ratios") {
      // A comma-separated ratio per layer, selecting that layer's attention
      // path (<=1 SWA, 4 CSA, 128 HCA). Overrides the checkpoint, the same
      // all-or-nothing way --kv-lora-rank overrides its geometry.
      config.compress_ratios.clear();
      const std::string value = next("--compress-ratios");
      size_t start = 0;
      while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string field = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        DSV4_REQUIRE(!field.empty(), "--compress-ratios has an empty entry");
        config.compress_ratios.push_back(ParseInt(field.c_str(), "--compress-ratios"));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      config.compress_ratios_provenance = GeometryProvenance::kCommandLine;
    } else if (flag == "--mhc-mixing") {
      const std::string value = next("--mhc-mixing");
      DSV4_REQUIRE(value == "fused" || value == "staged",
                   "--mhc-mixing expects 'fused' or 'staged', got '" << value << "'");
      config.mhc_mixing_mode =
          value == "staged" ? MhcMixingMode::kStagedMhcSinkhorn : MhcMixingMode::kFusedHcPre;
    } else if (flag == "--gating-norm-type") {
      config.gating_norm_type = ParseInt(next("--gating-norm-type"), "--gating-norm-type");
    } else if (flag == "--dense-group-size") {
      config.dense_group_size = ParseInt(next("--dense-group-size"), "--dense-group-size");
    } else if (flag == "--routed-coverage") {
      config.routed_coverage = ParseInt(next("--routed-coverage"), "--routed-coverage");
    } else if (flag == "--synthetic-weights") {
      config.synthetic_weights = true;
    } else if (flag == "--dry-run") {
      config.dry_run = true;
    } else if (flag == "--report") {
      config.report_path = next("--report");
    } else if (flag == "--stats-json") {
      arguments.stats_json_path = next("--stats-json");
    } else if (flag == "--diag-json") {
      arguments.diag_json_path = next("--diag-json");
    } else if (flag == "--verbose") {
      config.verbose = true;
    } else {
      throw Dsv4Error("unknown flag: " + flag);
    }
  }
  if (mla_from_cli) {
    config.mla.provenance = GeometryProvenance::kCommandLine;
  }
  return arguments;
}

// The model configuration, in priority order: an explicit --config, a
// config.json beside the weights, the platform default path, compiled
// defaults. Explicit-but-missing is an error; implicit-but-missing falls
// back silently, and the report says which happened.
ModelConfig LoadModelConfig(const Arguments& arguments, const RuntimeConfig& config, std::string* source_path) {
  if (!arguments.model_config_path.empty()) {
    *source_path = arguments.model_config_path;
    return ModelConfig::FromJsonFile(arguments.model_config_path);
  }
  if (!config.weights_path.empty()) {
    const std::string beside_weights = config.weights_path + "/config.json";
    std::ifstream probe(beside_weights);
    if (probe.is_open()) {
      probe.close();
      *source_path = beside_weights;
      return ModelConfig::FromJsonFile(beside_weights);
    }
  }
  {
    std::ifstream probe(kDefaultModelConfigPath);
    if (probe.is_open()) {
      probe.close();
      *source_path = kDefaultModelConfigPath;
      return ModelConfig::FromJsonFile(kDefaultModelConfigPath);
    }
  }
  source_path->clear();
  return ModelConfig{};  // the compiled DSV4-Flash contract
}

std::string DescribeConfiguration(const RuntimeConfig& config, const ModelConfig& model,
                                  const std::string& model_source, const std::string& moe_backend_description,
                                  const ExpertSlotLayout& layout) {
  std::ostringstream out;
  out << "DeepSeek-V4 Flash standalone runner\n";
  out << model.DescribeSummary(model_source);
  out << "  MLA geometry  provenance: " << config.mla.provenance_name() << "\n";
  out << "  precision     dense FP8 E4M3 (" << kAclFloat8E4m3Fn << ") block-" << kDenseScaleBlock
      << " scales; routed FP4 E2M1 (" << kAclFloat4E2m1 << ") + E8M0 (" << kAclFloat8E8m0 << ") block-"
      << kRoutedScaleBlock << "\n";
  out << "  moe backend   " << moe_backend_description << "\n";
  out << "  paged KV      block " << config.block_size << ", context " << config.max_context_len << "\n";
  out << layout.DescribeTable();
  return out.str();
}

int Run(int argc, char** argv) {
  Arguments arguments = ParseArguments(argc, argv);
  if (arguments.help || (argc == 1)) {
    PrintUsage();
    return 0;
  }
  RuntimeConfig& config = arguments.config;

  DSV4_REQUIRE(config.max_context_len > 0 && config.block_size > 0,
               "context length and block size must be positive");
  DSV4_REQUIRE(config.max_new_tokens > 0, "--max-new-tokens must be positive");
  if (arguments.prompt_tokens > 0) {
    DSV4_REQUIRE(config.synthetic_weights && arguments.prompt_ids.empty(),
                 "--prompt-tokens requires --synthetic-weights and cannot be combined with --prompt-ids");
    DSV4_REQUIRE(arguments.prompt_tokens <= config.max_context_len, "synthetic prompt exceeds context capacity");
    arguments.prompt_ids.assign(static_cast<size_t>(arguments.prompt_tokens), 0);
  }
  DSV4_REQUIRE(arguments.prompt_ids.size() <= static_cast<size_t>(config.max_context_len),
               "prompt exceeds context capacity");
  InferenceDiagnostics diagnostics;
  diagnostics.synthetic_weights = config.synthetic_weights;
  diagnostics.dry_run = config.dry_run;
  diagnostics.prompt_tokens = arguments.prompt_ids.size();
  diagnostics.paged_attention.block_size = static_cast<uint64_t>(config.block_size);
#if ASCEND_MOCK_RUNTIME
  diagnostics.mock_runtime = true;
#endif

  // ---- the model configuration (Task: parse, assert, then plan) ---------
  std::string model_source;
  const ModelConfig model = LoadModelConfig(arguments, config, &model_source);
  model.Validate();
  model.AssertMatchesBinaryContract();
  // The checkpoint's MLA geometry is authoritative unless the command line
  // overrode it (all-or-nothing, like the reader it replaces).
  if (config.mla.provenance != GeometryProvenance::kCommandLine) {
    config.mla = model.mla;
  }
  // The per-layer compression schedule, same precedence. An empty schedule
  // from both sources leaves every layer on the SWA path; nothing guesses an
  // interleave (see RuntimeConfig::compress_ratios).
  if (config.compress_ratios_provenance != GeometryProvenance::kCommandLine &&
      !model.compress_ratios.empty()) {
    config.compress_ratios = model.compress_ratios;
    config.compress_ratios_provenance = GeometryProvenance::kCheckpointConfig;
  }

  // The slot planner runs from the PARSED geometry, not the compiled
  // constants (they agree: AssertMatchesBinaryContract just proved it).
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);

  std::ostringstream report;
  report << DescribeConfiguration(
      config, model, model_source,
      config.moe_backend == MoeBackend::kLattice24 ? Lattice24MoeBlock().DescribeBackend()
                                                   : "standard aclnn (pipeline default): GroupedMatmulV5 family "
                                                     "over FP4/E8M0 slots",
      layout);

  // A --dry-run with no weights is a configuration check: parse, validate,
  // report, exit -- no aclnn runtime, no device, no weight source.
  if (config.dry_run && config.weights_path.empty() && !config.synthetic_weights) {
    if (!arguments.diag_json_path.empty()) diagnostics.DumpJson(arguments.diag_json_path);
    report << "\n--dry-run --config: model configuration parsed and validated; no device, no weights, "
              "no decode.\n";
    const std::string text = report.str();
    std::fputs(text.c_str(), stdout);
    if (!config.report_path.empty()) {
      std::ofstream file(config.report_path);
      DSV4_REQUIRE(file.is_open(), "cannot write the report to " << config.report_path);
      file << text;
    }
    return 0;
  }

  DSV4_REQUIRE(!config.weights_path.empty() || config.synthetic_weights,
               "--weights is required unless --synthetic-weights is given");

  OpTable ops;
  if (config.verbose) {
    report << ops.DescribeInventory();
  }
  DSV4_REQUIRE(ops.runtime_reachable(),
               "the aclnn runtime is not on this process's loader path: no operator resolved. Source "
               "set_env.sh, or add $ASCEND_TOOLKIT_HOME/<arch>-linux/lib64 to LD_LIBRARY_PATH.");

  // The device. A missing NPU is reported as itself, not as a fallback.
  AclDeviceOps device(config.device_id);
  report << "  device        " << config.device_id << ", SoC " << device.soc_name() << "\n";

  size_t free_hbm = 0;
  size_t total_hbm = 0;
  DSV4_REQUIRE(device.QueryDeviceMemory(&free_hbm, &total_hbm), "aclrtGetMemInfo(ACL_HBM_MEM) failed");
  const size_t backbone_bytes =
      Dsv4Pipeline::BackboneDeviceBytes(config.mla, config.block_size, config.max_context_len,
                                        config.compress_ratios);
  const int64_t slots = ExclusiveExpertManager::PlanDeviceSlots(
      free_hbm, backbone_bytes, layout.slot_num_bytes(), config.routed_coverage, kDeviceReserveBytes,
      kTransferChunkBytes, model.num_experts_per_tok, config.vram_slots);
  report << "  HBM           " << (free_hbm >> 20) << " MiB free of " << (total_hbm >> 20) << " MiB; backbone "
         << (backbone_bytes >> 20) << " MiB; K = " << slots << " routed slots\n";

  ExclusiveExpertManager::Options options;
  options.num_layers = model.num_hidden_layers;
  options.num_experts = model.n_routed_experts;
  options.top_k = model.num_experts_per_tok;
  options.routed_coverage = config.routed_coverage;
  options.device_slots = slots;
#if ASCEND_MOCK_RUNTIME
  options.host_available_bytes = 256ll << 30;  // interval bookkeeping, not physical pinned RAM
#endif
  ExclusiveExpertManager experts(device, device, layout, options);
  report << experts.DescribeHierarchy();

  // The router is constructed by the runner and shared by the pipeline and
  // the MoE block (the block consumes its dispatch output).
  MoeRouterEngine router(device, device);

  std::unique_ptr<IRoutedMoeBlock> moe_block;
  if (config.moe_backend == MoeBackend::kLattice24) {
    moe_block = std::make_unique<Lattice24MoeBlock>();
  } else {
    moe_block = std::make_unique<StandardAclnnMoeBlock>(ops, router, experts, config);
    // The block owns the slot byte contract for the hierarchy it runs on.
    DSV4_REQUIRE(moe_block->GetExpertSlotBytes() == layout.slot_num_bytes(),
                 "the MoE block's slot contract (" << moe_block->GetExpertSlotBytes()
                                                   << " B) disagrees with the planned slot layout ("
                                                   << layout.slot_num_bytes() << " B)");
  }

  std::unique_ptr<WeightByteSource> source;
  if (config.synthetic_weights) {
#if ASCEND_MOCK_RUNTIME
    source = std::make_unique<mock::SymbolicWeightSource>(layout, model.num_hidden_layers, model.n_routed_experts);
    mock::MockSetD2HSeed(&mock::SeedSyntheticReadback);
    report << "  mock outputs  synthetic routing and token readbacks; no numerical inference\n";
#else
    source = std::make_unique<SyntheticWeightSource>(layout, model.num_hidden_layers, model.n_routed_experts);
#endif
  } else {
    source = std::make_unique<SafetensorsWeightSource>(config.weights_path, layout, CheckpointNaming::kDsv4Flat,
                                                       model.num_hidden_layers, model.n_routed_experts);
  }

  Dsv4Pipeline pipeline(device, device, ops, experts, router, config, std::move(moe_block));
  // Order matters: the pipeline takes the backbone tensors first, then the
  // expert manager takes the routed slots and seals the source. Exactly one
  // owner closes it, and after that the hierarchy is the only copy.
  pipeline.Build(*source);
  diagnostics = pipeline.diagnostics();
  diagnostics.prompt_tokens = arguments.prompt_ids.size();
  experts.Ingest(*source, {});
  WeightByteSource::AssertNoOpenWeightDescriptors(config.synthetic_weights ? std::string() : config.weights_path);
  report << "  ingestion     " << (experts.stats().startup_bytes >> 20) << " MiB of routed experts, source '"
         << source->source_name() << "' closed, no descriptor left open\n";

  report << pipeline.DescribeStages();
  report << pipeline.DescribeSlotIndexMap();
  if (config.verbose) {
    report << pipeline.arena_manager().arena().DescribeLedger();
  }

  if (config.dry_run) {
    report << "\n--dry-run: built, planned and verified; no token was decoded.\n";
  } else {
    DSV4_REQUIRE(!arguments.prompt_ids.empty(),
                 "--prompt-ids is required to decode: this runner embeds no tokenizer, so the prompt text in "
                 "--prompt is recorded but not tokenized. Pass the ids your tokenizer produced.");
    // Timing anchors for the benchmark harness: TTFT spans the whole prefill
    // (prompt steps + the first readback), TPOT is the mean span of every
    // subsequent step. This runner runs prefill AS decode steps (one token
    // each), so TTFT over a P-token prompt is P decode steps -- the honest
    // number for this graph shape, and the harness labels it as such.
    const std::chrono::steady_clock::time_point decode_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point first_token_time = decode_start;
    std::vector<int32_t> generated;
    int64_t position = 0;
    // Prefill is run as a sequence of single-token steps: the decode graph is
    // shaped for one token (see kTokensPerStep), so a batched prefill would need
    // a second set of descriptors. It is correct, and it is O(prompt) steps.
    for (size_t index = 0; index < arguments.prompt_ids.size(); ++index) {
      pipeline.DecodeStep(arguments.prompt_ids[index], position++);
    }
    int32_t token = pipeline.ReadArgmaxToken();
    first_token_time = std::chrono::steady_clock::now();
    generated.push_back(token);
    for (int64_t index = 1; index < config.max_new_tokens && position < config.max_context_len; ++index) {
      pipeline.DecodeStep(token, position++);
      token = pipeline.ReadArgmaxToken();
      generated.push_back(token);
    }
    const std::chrono::steady_clock::time_point decode_end = std::chrono::steady_clock::now();
    const double ttft_seconds =
        std::chrono::duration<double>(first_token_time - decode_start).count();
    const double decode_span_seconds = std::chrono::duration<double>(decode_end - first_token_time).count();
    const double total_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
    const double tpot_seconds =
        generated.size() > 1 ? decode_span_seconds / static_cast<double>(generated.size() - 1) : 0.0;
    diagnostics = pipeline.diagnostics();
    diagnostics.prompt_tokens = arguments.prompt_ids.size();
    diagnostics.generated_tokens = generated.size();
    diagnostics.ttft_ms = ttft_seconds * 1000.0;
    if (generated.size() > 1) diagnostics.tpot_ms = tpot_seconds * 1000.0;
    report << "\ngenerated " << generated.size() << " token ids:";
    for (int32_t id : generated) {
      report << " " << id;
    }
    report << "\n";
    const StepCounters& counters = pipeline.counters();
    report << "steps " << counters.steps << ", layers " << counters.layers << ", launches " << counters.launches
           << "\n";
    report << "timing: ttft " << ttft_seconds << " s (prefill of " << arguments.prompt_ids.size()
           << " single-token steps + first readback), tpot " << tpot_seconds << " s, total " << total_seconds
           << " s\n";
    report << "expert residency: " << counters.expert_slot_hits << " hits, " << counters.expert_slot_misses
           << " misses\n";
    report << "host synchronizations " << counters.host_synchronizations << " (expected "
           << counters.steps * model.num_hidden_layers + generated.size()
           << ": one per MoE layer plus one per generated-token readback)\n";
    report << "allocations inside a step " << counters.device_allocations_in_step << " (must be 0); descriptors "
           << counters.descriptors_built_in_step << " (must be 0)\n";
    DSV4_REQUIRE(counters.device_allocations_in_step == 0,
                 counters.device_allocations_in_step << " device allocations happened inside a decode step");
    DSV4_REQUIRE(counters.descriptors_built_in_step == 0,
                 counters.descriptors_built_in_step << " descriptors were built inside a decode step");

    if (!arguments.stats_json_path.empty()) {
      // Machine-readable mirror of the report's counters, plus the swap
      // engine's byte tallies: what python/benchmarks/bench_runner.py parses.
      std::ofstream stats(arguments.stats_json_path);
      DSV4_REQUIRE(stats.is_open(), "cannot write the stats to " << arguments.stats_json_path);
      const ExclusiveStagingStats& swap = experts.stats();
      stats << "{\n"
            << "  \"generated_tokens\": " << generated.size() << ",\n"
            << "  \"prompt_tokens\": " << arguments.prompt_ids.size() << ",\n"
            << "  \"steps\": " << counters.steps << ",\n"
            << "  \"layers\": " << counters.layers << ",\n"
            << "  \"launches\": " << counters.launches << ",\n"
            << "  \"host_synchronizations\": " << counters.host_synchronizations << ",\n"
            << "  \"expert_slot_hits\": " << counters.expert_slot_hits << ",\n"
            << "  \"expert_slot_misses\": " << counters.expert_slot_misses << ",\n"
            << "  \"swap_h2d_bytes\": " << swap.host_to_device_bytes << ",\n"
            << "  \"swap_d2h_bytes\": " << swap.device_to_host_bytes << ",\n"
            << "  \"swap_d2d_bytes\": " << swap.device_to_device_bytes << ",\n"
            << "  \"swap_chunks\": " << swap.swap_chunks << ",\n"
            << "  \"startup_ingest_bytes\": " << swap.startup_bytes << ",\n"
            << "  \"device_allocations_in_step\": " << counters.device_allocations_in_step << ",\n"
            << "  \"descriptors_built_in_step\": " << counters.descriptors_built_in_step << ",\n"
            << "  \"ttft_seconds\": " << ttft_seconds << ",\n"
            << "  \"tpot_seconds\": " << tpot_seconds << ",\n"
            << "  \"decode_span_seconds\": " << decode_span_seconds << ",\n"
            << "  \"total_seconds\": " << total_seconds << ",\n"
            << "  \"device_slots\": " << slots << ",\n"
            << "  \"slot_bytes\": " << layout.slot_num_bytes() << ",\n"
            << "  \"backbone_bytes\": " << backbone_bytes << "\n"
            << "}\n";
      report << "stats written to " << arguments.stats_json_path << "\n";
    }
  }

  experts.Synchronize();
  experts.ValidateResidency();
  diagnostics.moe_cache = experts.cache_stats();
  if (!arguments.diag_json_path.empty()) {
    diagnostics.DumpJson(arguments.diag_json_path);
    report << "diagnostics written to " << arguments.diag_json_path << "\n";
  }
  report << "exclusive residency invariant holds after the run.\n";

  const std::string text = report.str();
  std::fputs(text.c_str(), stdout);
  if (!config.report_path.empty()) {
    std::ofstream file(config.report_path);
    DSV4_REQUIRE(file.is_open(), "cannot write the report to " << config.report_path);
    file << text;
  }
  return 0;
}

}  // namespace
}  // namespace ascend_moe

int RunMain(int argc, char** argv) {
  try {
    return ascend_moe::Run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "dsv4_runner: %s\n", error.what());
    return 1;
  }
}

int main(int argc, char** argv) {
  return ascend_moe::GuardedMain([&] { return RunMain(argc, argv); });
}
