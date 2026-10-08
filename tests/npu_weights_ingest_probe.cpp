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

// npu_weights_ingest_probe -- the ingestion path, end to end, one slot.
//
//   model dir (DSV4_MODEL_DIR or --model-dir)
//     -> find safetensors shard 00001
//     -> parse its JSON header (the engine's own parser)
//     -> aclrtMalloc ONE expert slot in HBM (12.75 MiB, the FP4/E8M0 slot)
//     -> aclrtMallocHost a pinned staging chunk
//     -> fread real shard bytes into the pinned chunk
//     -> aclrtMemcpyAsync pinned -> HBM on a stream + synchronize
//
// This is exactly the host-pinned -> HBM path the exclusive hierarchy's
// startup ingestion uses, exercised once with real bytes. No weights is a
// SKIP with exit 0 -- the probe is a bring-up tool, not a gate on hosts
// that simply do not carry the checkpoint. A provided directory whose shard
// is unreadable, or a DMA that returns a status, IS a failure.

#include <acl/acl.h>
#include "npu_test_support.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/model_config.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/core/weight_source.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  std::printf("  [ %s ] %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition) {
    ++g_failures;
  }
}

void Skip(const std::string& what) {
  std::printf("  [ SKIP ] %s\n", what.c_str());
}

// The model directory: --model-dir wins, then DSV4_MODEL_DIR.
std::string ModelDirFrom(int argc, char** argv) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::strcmp(argv[index], "--model-dir") == 0) {
      return argv[index + 1];
    }
  }
  const char* from_env = std::getenv("DSV4_MODEL_DIR");
  return from_env != nullptr ? std::string(from_env) : std::string();
}

bool DirectoryExists(const std::string& path) {
  DIR* directory = opendir(path.c_str());
  if (directory == nullptr) {
    return false;
  }
  closedir(directory);
  return true;
}

// Shard 00001 by convention (<name>-00001-of-000NN.safetensors); any shard
// proves the path, so fall back to the first .safetensors in sorted order.
std::string FindShard00001(const std::string& directory) {
  std::vector<std::string> shards;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    return std::string();
  }
  while (const dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.size() > 12 && name.compare(name.size() - 12, 12, ".safetensors") == 0) {
      shards.push_back(name);
    }
  }
  closedir(dir);
  std::sort(shards.begin(), shards.end());
  for (const std::string& name : shards) {
    if (name.find("00001") != std::string::npos) {
      return directory + "/" + name;
    }
  }
  return shards.empty() ? std::string() : directory + "/" + shards.front();
}

struct FileCloser {
  void operator()(std::FILE* file) const {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FileHandle = std::unique_ptr<std::FILE, FileCloser>;

// Reads the safetensors header: 8 bytes little-endian length, then that many
// bytes of JSON. Returns false when the file is not even a plausible shard.
bool ReadShardHeader(const std::string& path, std::vector<char>* header_json, uint64_t* data_begin,
                     uint64_t* file_size) {
  FileHandle file(std::fopen(path.c_str(), "rb"));
  if (file == nullptr) {
    return false;
  }
  if (std::fseek(file.get(), 0, SEEK_END) != 0) {
    return false;
  }
  const long size = std::ftell(file.get());
  if (size < 8) {
    return false;
  }
  *file_size = static_cast<uint64_t>(size);
  if (std::fseek(file.get(), 0, SEEK_SET) != 0) {
    return false;
  }
  uint8_t length_bytes[8];
  if (std::fread(length_bytes, 1, 8, file.get()) != 8) {
    return false;
  }
  uint64_t header_length = 0;
  for (int index = 7; index >= 0; --index) {
    header_length = (header_length << 8) | length_bytes[index];
  }
  if (header_length == 0 || header_length > (1ull << 30) ||
      *file_size < 8ull + header_length) {
    return false;
  }
  header_json->resize(header_length);
  if (std::fread(header_json->data(), 1, header_length, file.get()) != header_length) {
    return false;
  }
  *data_begin = 8ull + header_length;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (!PhysicalNpuPresent()) return 0;
  std::printf("npu_weights_ingest_probe -- shard header + one pinned->HBM DMA, real bytes\n");

  const std::string model_dir = ModelDirFrom(argc, argv);
  if (model_dir.empty()) {
    Skip("No model weights directory provided (set DSV4_MODEL_DIR or pass --model-dir)");
    return 0;
  }
  if (!DirectoryExists(model_dir)) {
    Skip("model directory does not exist: " + model_dir);
    return 0;
  }
  const std::string shard = FindShard00001(model_dir);
  if (shard.empty()) {
    Skip("no .safetensors shards in " + model_dir);
    return 0;
  }

  try {
    // The device backend (aclInit + aclrtSetDevice + context). On a host with
    // no NPU this throws, which for this probe is a SKIP, not a defect.
    ascend_moe::AclDeviceOps device(0);
    std::printf("  device SoC: %s\n", device.soc_name().c_str());

    // Slot geometry: from the checkpoint's config.json when the directory
    // carries one, else the compiled DeepSeek-V4 Flash contract. Either way
    // the slot is the 12.75 MiB FP4/E8M0 unit the hierarchy sizes everything
    // by; the probe allocates exactly one.
    ascend_moe::ExpertSlotLayout layout = ascend_moe::ExpertSlotLayout::ForDeepSeekV4Flash();
    const std::string config_path = model_dir + "/config.json";
    {
      std::FILE* probe = std::fopen(config_path.c_str(), "rb");
      if (probe != nullptr) {
        std::fclose(probe);
        const ascend_moe::ModelConfig model = ascend_moe::ModelConfig::FromJsonFile(config_path);
        model.AssertMatchesBinaryContract();
        layout = ascend_moe::ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);
        std::printf("  slot geometry from %s: hidden=%" PRId64 " intermediate=%" PRId64 "\n", config_path.c_str(),
                    model.hidden_size, model.moe_intermediate_size);
      } else {
        std::printf("  slot geometry from the compiled DSV4-Flash contract (no config.json in %s)\n",
                    model_dir.c_str());
      }
    }
    const size_t slot_bytes = layout.slot_num_bytes();

    // 1. the shard's header, parsed by the engine's own parser.
    std::vector<char> header_json;
    uint64_t data_begin = 0;
    uint64_t file_size = 0;
    Check(ReadShardHeader(shard, &header_json, &data_begin, &file_size),
          "shard header read: " + shard);
    if (g_failures > 0) {
      std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
      return 1;
    }
    const std::map<std::string, ascend_moe::SafetensorsTensor> tensors =
        ascend_moe::ParseSafetensorsHeaderJson(header_json.data(), header_json.size());
    Check(!tensors.empty(), "header parses: " + std::to_string(tensors.size()) + " tensors, header "
                                + std::to_string(header_json.size()) + " bytes, data starts at "
                                + std::to_string(data_begin) + ", file " + std::to_string(file_size) + " bytes");
    size_t named = 0;
    for (const auto& entry : tensors) {
      if (named++ < 3) {
        std::printf("           e.g. %s [%s] %" PRIu64 " bytes\n", entry.first.c_str(),
                    entry.second.dtype.c_str(), entry.second.num_bytes());
      }
    }

    // 2. one HBM slot, one pinned staging chunk, chunked DMAs.
    void* device_slot = device.DeviceMalloc(slot_bytes);
    Check(device_slot != nullptr,
          "aclrtMalloc: one " + std::to_string(slot_bytes) + "-byte expert slot in HBM");
    const size_t chunk_bytes =
        ascend_moe::kTransferChunkBytes < slot_bytes ? ascend_moe::kTransferChunkBytes : slot_bytes;
    void* pinned = device.HostPinnedMalloc(chunk_bytes);
    Check(pinned != nullptr,
          "aclrtMallocHost: a pinned staging chunk of " + std::to_string(chunk_bytes) + " bytes");

    if (device_slot != nullptr && pinned != nullptr) {
      const uint64_t data_bytes = file_size - data_begin;
      const uint64_t proof_bytes = data_bytes < slot_bytes ? data_bytes : slot_bytes;
      FileHandle file(std::fopen(shard.c_str(), "rb"));
      Check(file != nullptr, "shard reopened for the payload read");
      if (file != nullptr) {
        // Chunked exactly like the exclusive hierarchy's ingestion: one
        // bounded pinned staging chunk, filled by the host, drained H2D,
        // per chunk, bounding physical pinned staging memory.
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream) == 0 && stream != nullptr, "aclrtCreateStream");
        uint64_t transferred = 0;
        bool reads_ok = true;
        bool copies_ok = true;
        for (uint64_t offset = 0; offset < proof_bytes; offset += chunk_bytes) {
          const size_t count =
              static_cast<size_t>(proof_bytes - offset < chunk_bytes ? proof_bytes - offset : chunk_bytes);
          if (std::fseek(file.get(), static_cast<long>(data_begin + offset), SEEK_SET) != 0 ||
              std::fread(pinned, 1, count, file.get()) != count) {
            reads_ok = false;
            break;
          }
          const aclError copy_status = aclrtMemcpyAsync(
              static_cast<uint8_t*>(device_slot) + offset, slot_bytes - offset, pinned, count,
              ACL_MEMCPY_HOST_TO_DEVICE, stream);
          if (copy_status != 0) {
            copies_ok = false;
            Check(false, "aclrtMemcpyAsync chunk at offset " + std::to_string(offset) +
                             " returned " + std::to_string(copy_status));
            break;
          }
          transferred += count;
          // Ingestion is the one place a host synchronization is correct:
          // the next read overwrites the staging chunk the DMA is sourcing.
          if (aclrtSynchronizeStream(stream) != 0) {
            copies_ok = false;
            Check(false, "aclrtSynchronizeStream failed after chunk at offset " + std::to_string(offset));
            break;
          }
        }
        Check(reads_ok, "fread " + std::to_string(proof_bytes) + " real payload bytes through the pinned chunk (" +
                            std::to_string(chunk_bytes) + "-byte chunks)");
        Check(copies_ok, "aclrtMemcpyAsync pinned->HBM moved " + std::to_string(transferred) +
                             " bytes with zero status errors");
        if (stream != nullptr) {
          aclrtDestroyStream(stream);
        }
      }
    }

    if (pinned != nullptr) {
      device.HostPinnedFree(pinned);
    }
    if (device_slot != nullptr) {
      device.DeviceFree(device_slot);
    }
  } catch (const std::exception& error) {
    // A host without an NPU cannot prove the DMA path; that is a SKIP. Any
    // other exception (a shard that lies about its own header, a refused
    // allocation with a device present) is a real failure.
    const std::string message = error.what();
    if (message.find("aclrtSetDevice") != std::string::npos ||
        message.find("aclrtCreateContext") != std::string::npos) {
      Skip("no usable NPU device on this host: " + message);
      return 0;
    }
    std::printf("  [FAIL] %s\n", message.c_str());
    ++g_failures;
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
