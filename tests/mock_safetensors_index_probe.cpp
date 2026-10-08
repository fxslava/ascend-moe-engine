// Host-only metadata contract: ordinary JSON allocations, zero tensor storage.
// Fixtures are generated inline; no checkpoint download or shard is required.
#include "moe/core/checkpoint_keys.hpp"
#include "moe/core/error.hpp"
#include "moe/core/weight_source.hpp"
#include "mock_allocator.hpp"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ascend_moe;
constexpr int64_t kProbeLayers = 43;
size_t checks = 0;

void Check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
  ++checks;
}

template <class F>
void Refuses(F action, const std::string& message) {
  bool refused = false;
  try {
    action();
  } catch (const Dsv4Error&) {
    refused = true;
  }
  Check(refused, message);
}

// The same tiny tensors can describe huge models without any tensor storage.
std::string Header(const std::vector<std::string>& names) {
  std::string json = R"({"__metadata__":{"format":"pt"})";
  for (const auto& name : names) {
    json += ",\"" + name + R"(":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]})";
  }
  return json + "}";
}

std::string IndexJson(const CheckpointIndex& index) {
  std::string json = R"({"metadata":{"total_size":159609485896},"weight_map":{)";
  bool first = true;
  for (const auto& entry : index) {
    if (!first) json += ',';
    first = false;
    json += '"' + entry.first + "\":\"" + entry.second + '"';
  }
  return json + "}}";
}

CheckpointIndex ParseIndex(const std::string& json) {
  return ParseSafetensorsIndexJson(json.data(), json.size());
}

struct TemporaryDirectory {
  std::string path;
  TemporaryDirectory() {
    char pattern[] = "/tmp/dsv4-index-probe-XXXXXX";
    const char* result = ::mkdtemp(pattern);
    if (!result) throw std::runtime_error("cannot create temporary fixture directory");
    path = result;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void WriteContainer(const std::string& path, const std::string& json, uint64_t declared_size) {
  std::ofstream file(path, std::ios::binary);
  for (unsigned byte = 0; byte < 8; ++byte) {
    file.put(static_cast<char>((declared_size >> (byte * 8)) & 0xff));
  }
  file.write(json.data(), static_cast<std::streamsize>(json.size()));
  file.close();
  Check(static_cast<bool>(file), "write temporary header fixture");
}

void TestHeaders() {
  const std::vector<std::string> hf = {
      "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight",
      "model.norm.weight", "lm_head.weight"};
  const std::vector<std::string> hc = {
      "embed.weight", "layers.0.hc_attn_base", "layers.0.hc_ffn_base",
      "layers.0.experts.0.w1.weight", "mtp.0.head.weight"};
  TemporaryDirectory temporary;
  for (const auto& names : {hf, hc}) {
    std::string json = Header(names);
    const auto parsed = ParseSafetensorsHeaderJson(json.data(), json.size());
    Check(parsed.size() == names.size(), "metadata is excluded from tensor count");
    for (const auto& name : names) {
      Check(parsed.at(name).dtype == "BF16" && parsed.at(name).shape == std::vector<int64_t>{0} &&
                parsed.at(name).num_bytes() == 0,
            "zero-length header metadata for " + name);
      Check(ResolveCheckpointTensorName(name, [&](const std::string& key) { return parsed.count(key); }) ==
                (IsAuxiliaryCheckpointTensor(name) ? std::string{} : name),
            "parsed header key resolution for " + name);
    }
    Check(json.size() <= 1016, "fixture header fits 1 KB container");
    json.resize(1016, ' ');  // 8-byte prefix + aligned JSON = exactly 1024 bytes.
    const std::string path = temporary.path + "/tiny.safetensors";
    WriteContainer(path, json, json.size());
    Check(std::filesystem::file_size(path) == 1024, "fixture has exactly 1 KB and no payload");
    SafetensorsWeightSource source(path, ExpertSlotLayout::ForDeepSeekV4Flash(),
                                   CheckpointNaming::kHfDeepseek, 1, 1);
    Check(source.tensor_count() == names.size(), "disk header tensor count");
    for (const auto& name : names) {
      Check(source.HasNamed(name) && source.NamedByteSize(name) == 0, "disk metadata for " + name);
    }
    const auto embedding = ResolveCheckpointTensorName("model.embed_tokens.weight",
        [&](const std::string& key) { return source.HasNamed(key); });
    Check(embedding == names.front(), "real source uses the shared embedding resolver");
    Check(source.bytes_read() == 1024 && source.read_requests() == 2,
          "disk source read only the prefix and JSON header");
    source.Close();
    WeightByteSource::AssertNoOpenWeightDescriptors(temporary.path);
  }

  const std::string huge = R"({"huge":{"dtype":"U8","shape":[8589934592],"data_offsets":[0,8589934592]}})";
  const auto metadata = ParseSafetensorsHeaderJson(huge.data(), huge.size());
  Check(metadata.at("huge").num_bytes() == (8ull << 30), "8 GiB span is parsed as metadata only");

  for (const std::string& bad : std::vector<std::string>{
           "not json", "{\"a\":{", Header(hf) + "garbage",
           R"({"a":{"dtype":"BF16","shape":[0]}})",
           R"({"a":{"dtype":"BF16","shape":[-1],"data_offsets":[0,0]}})",
           R"({"a":{"dtype":"BF16","shape":[0],"data_offsets":[-1,0]}})",
           R"({"a":{"dtype":"BF16","shape":[0],"data_offsets":[2,1]}})",
           Header({"duplicate", "duplicate"})}) {
    Refuses([&] { ParseSafetensorsHeaderJson(bad.data(), bad.size()); }, "reject malformed header");
  }
  const std::string path = temporary.path + "/invalid.safetensors";
  for (uint64_t length : {0ull, 4096ull, 1ull << 40}) {
    WriteContainer(path, "{}", length);
    Refuses([&] {
      SafetensorsWeightSource source(path, ExpertSlotLayout::ForDeepSeekV4Flash(),
                                     CheckpointNaming::kHfDeepseek, 1, 1);
    }, "reject invalid header length without allocating payload");
    WeightByteSource::AssertNoOpenWeightDescriptors(temporary.path);
  }
  WriteContainer(path, huge, huge.size());
  Refuses([&] {
    SafetensorsWeightSource source(path, ExpertSlotLayout::ForDeepSeekV4Flash(),
                                   CheckpointNaming::kHfDeepseek, 1, 1);
  }, "reject an 8 GiB payload span absent from disk without reading it");
  WeightByteSource::AssertNoOpenWeightDescriptors(temporary.path);
  std::cout << "[ PASS ] Header parser: memory + 1 KB disk fixtures; two header reads, no payload reads\n";
}

// Model A is HF; Model B uses HC and deliberately varies expert prefixes.
CheckpointIndex SyntheticIndex(bool hc, bool mixed = false) {
  CheckpointIndex index;
  const auto add = [&](const std::string& name) { index.emplace(name, "absent-shard.safetensors"); };
  add(hc ? "embed.weight" : "model.embed_tokens.weight");
  add(hc ? "norm.weight" : "model.norm.weight");
  add(hc ? "head.weight" : "lm_head.weight");
  for (int64_t layer = 0; layer < kProbeLayers; ++layer) {
    const bool compiled = hc && (!mixed || layer % 2 == 0);
    const std::string prefix = (compiled ? "layers." : "model.layers.") + std::to_string(layer) + ".";
    if (compiled) {
      add(prefix + "hc_attn_base");
      add(prefix + "hc_ffn_base");
      add(prefix + "attn_norm.weight");
      add(prefix + "ffn_norm.weight");
      add(prefix + "ffn.gate.weight");
      add(prefix + "ffn.gate.bias");
      add(prefix + (layer % 2 == 0 ? "experts.0.w1.weight" : "ffn.experts.0.w1.weight"));
    } else {
      add(prefix + "self_attn.q_proj.weight");
      add(prefix + "self_attn.o_proj.weight");
      add(prefix + "input_layernorm.weight");
      add(prefix + "post_attention_layernorm.weight");
      add(prefix + "mlp.gate.weight");
      add(prefix + "mlp.experts.0.gate_proj.weight");
    }
  }
  add("mtp.0.head.weight");
  add("model.mtp.0.embed.weight");
  return index;
}

void TestMappingAndTopology() {
  for (const auto& fixture : {SyntheticIndex(false), SyntheticIndex(true), SyntheticIndex(true, true)}) {
    const auto index = ParseIndex(IndexJson(fixture));
    Check(index == fixture, "JSON index round trip preserves every shard mapping");
    const auto topology = ValidateCheckpointTopology(index, kProbeLayers);
    Check(topology.layers.size() == 43 && topology.globals.size() == 3, "all 43 backbone layers mapped");
    Check(topology.ignored_auxiliary_tensors == 2, "MTP tensors safely excluded");
    for (const auto& layer : topology.layers) {
      for (const auto& role : layer) {
        Check(index.count(role.second) == 1 && !IsAuxiliaryCheckpointTensor(role.second),
              "backbone role has an existing non-MTP target");
      }
    }
  }
  auto index = SyntheticIndex(true);
  const auto resolve = [&](const std::string& name) {
    return ResolveCheckpointTensorName(name, [&](const std::string& key) { return index.count(key); });
  };
  index["embed_tokens.weight"] = "fallback.safetensors";
  Check(resolve("model.embed_tokens.weight") == "embed.weight", "embed.weight wins over embed_tokens.weight");
  index.erase("embed.weight");
  Check(resolve("model.embed_tokens.weight") == "embed_tokens.weight", "unprefixed embedding fallback");
  index["embed.weight"] = "absent-shard.safetensors";
  for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
           {"model.embed_tokens.weight", "embed.weight"}, {"lm_head.weight", "head.weight"},
           {"model.norm.weight", "norm.weight"},
           {"model.layers.0.input_layernorm.weight", "layers.0.attn_norm.weight"},
           {"model.layers.0.post_attention_layernorm.weight", "layers.0.ffn_norm.weight"},
           {"model.layers.0.mlp.gate.weight", "layers.0.ffn.gate.weight"},
           {"model.layers.0.mlp.gate.e_score_correction_bias", "layers.0.ffn.gate.bias"},
           {"model.layers.0.mlp.experts.0.gate_proj.weight", "layers.0.experts.0.w1.weight"},
           {"model.layers.1.mlp.experts.0.gate_proj.weight", "layers.1.ffn.experts.0.w1.weight"}}) {
    Check(resolve(pair.first) == pair.second, "fallback for " + pair.first);
    index[pair.first] = "canonical.safetensors";
    Check(resolve(pair.first) == pair.first, "canonical precedence for " + pair.first);
    index.erase(pair.first);
  }
  index["layers.0.attn.q_norm.weight"] = "norms.safetensors";
  index["layers.0.attn.kv_norm.weight"] = "norms.safetensors";
  Check(resolve("model.layers.0.self_attn.q_a_layernorm.weight") == "layers.0.attn.q_norm.weight" &&
            resolve("model.layers.0.self_attn.kv_a_layernorm.weight") == "layers.0.attn.kv_norm.weight",
        "Q and KV norms have distinct aliases");
  Check(resolve("mtp.0.head.weight").empty() && resolve("model.mtp.0.embed.weight").empty(),
        "auxiliary keys never resolve into the main graph");
  index["layers.0.attn.wq_b.weight"] = "raw.safetensors";
  index["layers.0.attn.wo_b.weight"] = "raw.safetensors";
  Check(resolve("model.layers.0.self_attn.q_b_proj_latent.weight").empty() &&
            resolve("model.layers.0.self_attn.o_proj_folded.weight").empty(),
        "raw HC projections must not impersonate absorbed MLA weights");

  for (const std::string missing : {"embed.weight", "head.weight", "norm.weight", "layers.42.hc_attn_base",
                                    "layers.42.hc_ffn_base", "layers.42.attn_norm.weight",
                                    "layers.42.ffn_norm.weight", "layers.42.ffn.gate.weight",
                                    "layers.42.experts.0.w1.weight"}) {
    auto broken = SyntheticIndex(true);
    broken.erase(missing);
    broken["mtp.0." + missing] = "auxiliary.safetensors";
    Refuses([&] { ValidateCheckpointTopology(broken, 43); }, "missing role is not supplied by MTP: " + missing);
  }
  auto broken = SyntheticIndex(false);
  broken.erase("model.layers.42.self_attn.o_proj.weight");
  Refuses([&] { ValidateCheckpointTopology(broken, 43); }, "HF output projection is required");
  broken = SyntheticIndex(true);
  broken["layers.42.ffn.gate.weight"] = "";
  Refuses([&] { ValidateCheckpointTopology(broken, 43); }, "mapped targets must be valid shard names");
  for (const std::string& bad : std::vector<std::string>{
           "{}", R"({"weight_map":{}})", R"({"weight_map":{"a":12}})",
           R"({"weight_map":{"a":""}})", R"({"weight_map":{"a":"a","a":"b"}})",
           IndexJson(index) + "garbage", R"({"weight_map":{"a":"b"})"}) {
    Refuses([&] { ParseIndex(bad); }, "malformed index must fail");
  }
  std::cout << "[ PASS ] HF, HC and mixed topology: layers 0..42, aliases, precedence, MTP and negative cases\n";
}

void ProbeLocalIndex() {
  const char* configured = std::getenv("DSV4_MODEL_DIR");
#if defined(_WIN32) || defined(__CYGWIN__)
  const char* default_root = "C:/models/DeepSeek-V4-Flash";
#else
  const char* default_root = "/mnt/c/models/DeepSeek-V4-Flash";
#endif
  const std::filesystem::path root = configured && *configured ? configured : default_root;
  const auto path = root / "model.safetensors.index.json";
  std::error_code error;
  const bool exists = std::filesystem::exists(path, error);
  Check(!error, "cannot check local index: " + error.message());
  if (!exists) {
    std::cout << "[ SKIP ] Local index file not found\n";
    return;
  }
  std::ifstream file(path, std::ios::binary);
  Check(static_cast<bool>(file), "local index exists but cannot be opened: " + path.string());
  std::ostringstream text;
  text << file.rdbuf();
  Check(!file.bad(), "local index read failed");
  const auto index = ParseIndex(text.str());
  const auto topology = ValidateCheckpointTopology(index, kProbeLayers);
  Check(topology.layers.size() == 43, "local index covers layers 0..42");
  const auto has = [&](const std::string& key) { return index.count(key) != 0; };
  for (int64_t layer = 0; layer < kProbeLayers; ++layer) {
    for (int64_t expert = 0; expert < 256; ++expert) {
      const std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.experts." +
                                 std::to_string(expert) + ".";
      for (const char* projection : {"gate_proj", "up_proj", "down_proj"}) {
        for (const char* suffix : {".weight", ".weight_scale_inv"}) {
          const std::string requested = prefix + projection + suffix;
          Check(!ResolveCheckpointTensorName(requested, has).empty(), "local expert descriptor missing: " + requested);
        }
      }
    }
  }
  std::cout << "[ PASS ] Local index: " << path << "; " << index.size() << " keys; layers 0..42 mapped; "
            << topology.ignored_auxiliary_tensors << " MTP keys ignored; 0 shards opened\n";
  std::cout << "[ INFO ] Index topology only; dtype/shape and folded-MLA execution compatibility are not tested\n";
}
}  // namespace

int main() {
  try {
    TestHeaders();
    TestMappingAndTopology();
    ProbeLocalIndex();
    const auto& stats = ascend_moe::mock::MockMemoryStatistics();
    Check(stats.device_allocations == 0 && stats.host_allocations == 0 && stats.active_spans == 0 &&
              stats.real_host_bytes == 0 && stats.symbolic_device_bytes == 0 && stats.would_copy_bytes == 0,
          "metadata probe must never allocate or transfer tensor storage");
    std::cout << "[ PASS ] " << checks << " checks; 0 MB weight allocation; 0 device/pinned allocations; "
                 "0 payload bytes read\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[ FAIL ] " << error.what() << '\n';
    return 1;
  }
}
