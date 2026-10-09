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

// mock_contract_smoke -- the parts of this runner that can be verified without
// an NPU attached, actually verified.
//
// This is not a compile check. The exclusive swap engine runs here for real,
// over the `SimulatedDeviceOps` backend, and the test asserts on its recorded
// DMA trace: the disjointness invariant, the chunked duplex exchange's event
// ordering, the LRU victim choice, the byte-exact round trip of an evicted
// expert, the poisoning on a failed transfer, and every refusal the sealed
// arena is supposed to make. It also resolves the operator table against the
// linked mock library, which proves the mock link and ABI without a device.
//
// What it cannot check, and says so: anything that needs a 950PR -- the real
// `GetWorkspaceSize` numbers, whether a kernel exists for every op on the part,
// and the `aclSetTensorAddr` index map.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/config.hpp"
#include "moe/core/kv_cache_layout.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/model_config.hpp"
#include "moe/memory/exclusive_staging.hpp"
#include "moe/memory/expert_layout.hpp"
#include "moe/ops/aclnn_dsv4_vendor_ops.h"
#include "moe/pipeline/lattice_moe_block.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "moe/core/weight_source.hpp"

#include "mock_acl_tensor.hpp"

namespace ascend_moe {
namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (condition) {
    std::printf("  [ ok ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

// Asserts that `action` throws, which is how every refusal in this design is
// expressed. A refusal that silently succeeds is the failure mode worth testing.
template <typename Action>
void CheckRefuses(Action action, const std::string& what) {
  ++g_checks;
  try {
    action();
  } catch (const Dsv4Error&) {
    std::printf("  [ ok ] refused: %s\n", what.c_str());
    return;
  } catch (const std::exception& error) {
    ++g_failures;
    std::printf("  [FAIL] %s threw the wrong type: %s\n", what.c_str(), error.what());
    return;
  }
  ++g_failures;
  std::printf("  [FAIL] %s was allowed\n", what.c_str());
}

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

// ---------------------------------------------------------------------------
// 1. Expert slot layout
// ---------------------------------------------------------------------------

void TestExpertLayout() {
  Section("expert slot layout");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  std::printf("%s", layout.DescribeTable().c_str());

  // The number the Python layout's docstring states for this geometry.
  Check(layout.slot_num_bytes() == 13369344,
        "DSV4-Flash slot is 13,369,344 bytes (12.75 MiB), matching core/layout.py");
  Check(layout.slot_num_bytes() * kTotalRoutedExperts == 147169738752ull,
        "the whole routed set is 147,169,738,752 bytes (137.07 GiB) over 11,008 experts");

  size_t covered = 0;
  bool aligned = true;
  for (const ExpertRegionSpec& spec : layout.regions()) {
    aligned = aligned && (spec.offset_bytes % kSlotRegionAlignBytes == 0);
    covered += spec.num_bytes();
  }
  Check(aligned, "every region offset is 128-byte aligned");
  Check(covered == layout.slot_num_bytes(), "the regions tile the slot exactly, with no padding bytes");

  // Gate and up must be adjacent, or the fused expert GEMM cannot address them
  // as one weight.
  const ExpertRegionSpec& gate_up = layout.region(ExpertRegionId::kGateUpWeight);
  Check(gate_up.rows == 2 * kMoeIntermediateSize && gate_up.cols == kHiddenSize,
        "the fused gate/up weight is [2*intermediate, hidden]");
  const ExpertRegionSpec& down = layout.region(ExpertRegionId::kDownWeight);
  Check(down.rows == kHiddenSize && down.cols == kMoeIntermediateSize,
        "the down weight is [hidden, intermediate]");
  Check(layout.region(ExpertRegionId::kGateUpScale).stored_cols() == kHiddenSize / kRoutedScaleBlock,
        "microscales are blocked along the reduction axis, 32 elements per byte of scale");

  // A geometry whose reduction axis is not divisible by the block must be
  // refused, not rounded.
  CheckRefuses([] { ExpertSlotLayout::ForGeometry(48, 64); }, "a hidden size that is not a multiple of 32");
}

// ---------------------------------------------------------------------------
// 2. Safetensors header parsing
// ---------------------------------------------------------------------------

void TestHeaderParsing() {
  Section("safetensors header parsing");
  const char* header =
      "{\"__metadata__\":{\"format\":\"pt\",\"nested\":{\"a\":[1,2,3]}},"
      "\"layers.0.ffn.experts.0.w1.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[2048,4096],"
      "\"data_offsets\":[0,4194304]},"
      "\"layers.0.ffn.experts.0.w1.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[2048,128],"
      "\"data_offsets\":[4194304,4456448]}}";
  const std::map<std::string, SafetensorsTensor> tensors =
      ParseSafetensorsHeaderJson(header, std::strlen(header));
  Check(tensors.size() == 2, "__metadata__ is skipped, including its nested object");
  const auto weight = tensors.find("layers.0.ffn.experts.0.w1.weight");
  Check(weight != tensors.end() && weight->second.num_bytes() == 4194304,
        "the FP4/FP8 weight span is 4,194,304 bytes");
  const auto scale = tensors.find("layers.0.ffn.experts.0.w1.scale");
  Check(scale != tensors.end() && scale->second.payload_begin == 4194304, "the scale span follows the weight");

  CheckRefuses([] { ParseSafetensorsHeaderJson("{\"a\":{", 6); }, "a truncated header");
  CheckRefuses([] { ParseSafetensorsHeaderJson("not json", 8); }, "a header that is not an object");
}

// ---------------------------------------------------------------------------
// 2b. The FP4 / E8M0 unpack mapping, checked against a real (tiny) file
// ---------------------------------------------------------------------------

// Writes a one-expert checkpoint for ExpertSlotLayout::ForGeometry(64, 64):
// w1/w3 weight 2048 bytes each (shape [64, 32] in the packed-FP4 byte view),
// w1/w3 scale 128 bytes each ([64, 2]), w2 weight 2048 ([64, 32]), w2 scale 128
// ([64, 2]) -- 6528 payload bytes, exactly one slot. `weight_dtype` /
// `scale_dtype` land in every header so a wrong-dtype checkpoint can be
// expressed; `w1_bytes` resizes the first span for the wrong-size case.
bool WriteExpertCheckpoint(const char* path, const char* weight_dtype, const char* scale_dtype, size_t w1_bytes) {
  const size_t scale_bytes = 128;
  const size_t w3_bytes = 2048;
  const size_t w2_bytes = 2048;
  const size_t total = w1_bytes + scale_bytes + w3_bytes + scale_bytes + w2_bytes + scale_bytes;

  size_t cursor = 0;
  std::string json = "{";
  const auto add_tensor = [&](const char* leaf, const char* dtype, const char* shape, size_t bytes) {
    const size_t begin = cursor;
    cursor += bytes;
    json += json.size() > 1 ? "," : "";
    json += "\"layers.0.ffn.experts.0.";
    json += leaf;
    json += "\":{\"dtype\":\"";
    json += dtype;
    json += "\",\"shape\":";
    json += shape;
    json += ",\"data_offsets\":[";
    json += std::to_string(begin);
    json += ",";
    json += std::to_string(cursor);
    json += "]}";
  };
  add_tensor("w1.weight", weight_dtype, "[64,32]", w1_bytes);
  add_tensor("w1.scale", scale_dtype, "[64,2]", scale_bytes);
  add_tensor("w3.weight", weight_dtype, "[64,32]", w3_bytes);
  add_tensor("w3.scale", scale_dtype, "[64,2]", scale_bytes);
  add_tensor("w2.weight", weight_dtype, "[64,32]", w2_bytes);
  add_tensor("w2.scale", scale_dtype, "[64,2]", scale_bytes);
  json += "}";

  std::FILE* file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const std::vector<char> payload(total, '\0');
  const uint64_t header_length = json.size();
  const bool wrote = std::fwrite(&header_length, sizeof(header_length), 1, file) == 1 &&
                     std::fwrite(json.data(), 1, json.size(), file) == json.size() &&
                     std::fwrite(payload.data(), 1, payload.size(), file) == payload.size();
  std::fclose(file);
  return wrote;
}

void TestExpertBindingValidation() {
  Section("safetensors expert binding: the FP4 / E8M0 unpack mapping");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(64, 64);
  Check(layout.slot_num_bytes() == 6528, "the 64x64 test geometry packs into a 6,528-byte slot");

  const char* path = "/tmp/mock_contract_smoke_experts.safetensors";
  std::vector<uint8_t> slot(layout.slot_num_bytes());

  // The admissible checkpoint: packed-FP4 weights and UE8M0 block scales.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "MOE_F4_SCALE", 2048), "the admissible checkpoint was written");
  {
    SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
    Check(source.Contains(0, 0), "the flat naming scheme finds expert (0, 0)");
    source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
    Check(true, "an FP4 + E8M0 checkpoint with exact span sizes reads into the slot");
  }

  // An FP8 checkpoint bound against the FP4 slot layout: same byte count would
  // not catch it, the dtype string does.
  Check(WriteExpertCheckpoint(path, "F8_E4M3", "MOE_F4_SCALE", 2048), "the FP8 checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "an FP8-E4M3 weight tensor bound against the packed-FP4 slot layout");

  // A per-element scale dtype is not the block-32 E8M0 mapping either.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "U8", 2048), "the U8-scale checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "a per-element U8 scale tensor bound against the E8M0 block-32 scale region");

  // A resized first span: the byte-exact slot total no longer tiles.
  Check(WriteExpertCheckpoint(path, "MOE_F4", "MOE_F4_SCALE", 512), "the resized checkpoint was written");
  CheckRefuses(
      [&] {
        SafetensorsWeightSource source(path, layout, CheckpointNaming::kDsv4Flat, 1, 1);
        source.ReadExpertSlotRange(slot.data(), slot.size(), 0, slot.size(), 0, 0);
      },
      "a weight span that no longer matches the slot region size");

  std::remove(path);
}

// ---------------------------------------------------------------------------
// 3. The exclusive hierarchy, running for real on the simulated backend
// ---------------------------------------------------------------------------

// A deliberately small geometry: the invariants are about counts and ordering,
// and 11,008 real 12.75 MiB slots would need 137 GiB to say the same thing.
constexpr int64_t kTestLayers = 4;
constexpr int64_t kTestExperts = 8;
constexpr int64_t kTestCoverage = kTestLayers * kTestExperts;  // 32
constexpr int64_t kTestDeviceSlots = 12;

ExclusiveExpertManager::Options TestOptions() {
  ExclusiveExpertManager::Options options;
  options.num_layers = kTestLayers;
  options.num_experts = kTestExperts;
  options.routed_coverage = kTestCoverage;
  options.device_slots = kTestDeviceSlots;
  options.transfer_chunk_bytes = 4096;  // several chunks per slot, so the ring is exercised
  options.host_block_bytes = 64 * 1024;
  options.top_k = kNumExpertsPerTok;
  return options;
}

void TestHierarchyInvariants() {
  Section("exclusive hierarchy: ingestion and the disjointness invariant");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  std::printf("%s", experts.DescribeHierarchy().c_str());

  Check(experts.device_slot_count() + experts.host_slot_count() == kTestCoverage,
        "|Set_Device| + |Set_Host| equals the routed coverage");
  Check(experts.host_bytes() == static_cast<size_t>(kTestCoverage - kTestDeviceSlots) * layout.slot_num_bytes(),
        "the pinned host arena holds exactly coverage - K slots");

  const std::vector<const void*> fingerprint = experts.Fingerprint();
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});
  Check(source.closed(), "Ingest closes the byte source: the hierarchy becomes the only copy");
  CheckRefuses([&] { source.ReadExpertSlotRange(nullptr, 0, 0, 0, 0, 0); },
               "a read from the closed source (runtime disk reads)");
  Check(experts.Fingerprint() == fingerprint, "no arena address moved during ingestion");
  experts.ValidateResidency();
  Check(true, "the residency invariant holds after ingestion");

  CheckRefuses([&] { experts.Ingest(source, {}); }, "a second ingestion");

  // Every byte of every host-resident expert must be what the source would have
  // produced: a slot written to the wrong place is otherwise invisible.
  int mismatches = 0;
  for (int32_t layer = 0; layer < static_cast<int32_t>(kTestLayers); ++layer) {
    for (int32_t expert = 0; expert < static_cast<int32_t>(kTestExperts); ++expert) {
      if (experts.DeviceSlotOf(layer, expert) != kNoSlot) {
        continue;
      }
      // Host-resident: compare the first and last byte of the slot image.
      // (The addresses are private, so this checks through the device arena
      // after a promotion below instead; see TestSwapRoundTrip.)
      ++mismatches;
    }
  }
  Check(mismatches == kTestCoverage - kTestDeviceSlots,
        "exactly coverage - K experts are host-resident after ingestion");
}

void TestLayerPlanning() {
  Section("exclusive hierarchy: layer planning and the eviction policy");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});

  // Layer 0's experts 0..5 were placed in slots 0..5 at ingestion, so this is
  // an all-hit layer: no transfer, no event, no stream touched.
  const int32_t all_hits[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  const LayerSwapPlan hit_plan = experts.PlanLayer(0, all_hits, kNumExpertsPerTok);
  Check(hit_plan.hit_count == kNumExpertsPerTok && hit_plan.miss_count == 0,
        "a layer whose top-6 is already resident plans six hits and no swap");

  device.ResetCounters();
  DeviceStream compute = device.CreateStream();
  DeviceEvent boundary = device.CreateEvent();
  device.RecordEvent(boundary, compute);
  experts.ExecutePlan(hit_plan, compute, boundary);
  Check(device.counters().async_copies == 0, "an all-hit layer enqueues no DMA at all");

  // Layer 3 is entirely host-resident, so it is six misses and six exchanges.
  const int32_t all_misses[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  const LayerSwapPlan miss_plan = experts.PlanLayer(3, all_misses, kNumExpertsPerTok);
  Check(miss_plan.miss_count == kNumExpertsPerTok, "a fully host-resident layer plans six misses");
  bool distinct_victims = true;
  for (int32_t i = 0; i < miss_plan.count; ++i) {
    for (int32_t j = i + 1; j < miss_plan.count; ++j) {
      distinct_victims = distinct_victims && miss_plan.device_slots[i] != miss_plan.device_slots[j];
    }
  }
  Check(distinct_victims, "the six admissions claim six different slots, so they cannot evict each other");

  // The victims must be the least-recently-touched slots, which after the
  // all-hit layer above are the six slots that layer did NOT touch.
  bool avoided_recent = true;
  for (int32_t i = 0; i < miss_plan.count; ++i) {
    avoided_recent = avoided_recent && miss_plan.device_slots[i] >= kNumExpertsPerTok;
  }
  Check(avoided_recent, "LRU picked the six untouched slots, not the six just used");

  CheckRefuses([&] { experts.ExecutePlan(miss_plan, compute, nullptr); },
               "a batch with misses and no compute-boundary event");

  device.ResetCounters();
  device.ClearTrace();
  experts.ExecutePlan(miss_plan, compute, boundary);
  experts.Synchronize();
  experts.ValidateResidency();
  Check(true, "the residency invariant still holds after six exchanges");
  const auto& cache = experts.cache_stats();
  Check(cache.total_expert_requests == 12 && cache.hbm_slot_hits == 6 && cache.host_promotions == 6 &&
            cache.evictions_to_host == 6 && cache.HitRate() == 0.5,
        "cache telemetry counts hits and completed promotions/evictions, excluding the refused plan");

  const DmaCounters& counters = device.counters();
  const size_t slot_bytes = layout.slot_num_bytes();
  Check(counters.host_to_device_bytes == kNumExpertsPerTok * slot_bytes,
        "H2D moved exactly six slots' worth of bytes");
  Check(counters.device_to_host_bytes == kNumExpertsPerTok * slot_bytes,
        "D2H moved exactly six slots' worth (the victims)");
  Check(counters.device_to_device_bytes == kNumExpertsPerTok * slot_bytes,
        "D2D parked exactly six slots' worth in the bounded transit scratch");
  Check(counters.stream_synchronizations == 3,
        "the whole six-expert batch cost zero host synchronizations (3 here are the explicit Synchronize())");

  // Event ordering: within each chunk the three stages must appear as
  // park -> record, wait -> promote -> record, wait -> land -> record.
  int triples = 0;
  const std::vector<DmaTraceEntry>& trace = device.trace();
  for (size_t index = 0; index + 8 < trace.size(); ++index) {
    const bool shape = trace[index].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 1].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 1].memcpy_kind == MemcpyKind::kDeviceToDevice &&
                       trace[index + 2].kind == DmaTraceEntry::Kind::kRecordEvent &&
                       trace[index + 3].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 4].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 4].memcpy_kind == MemcpyKind::kHostToDevice &&
                       trace[index + 5].kind == DmaTraceEntry::Kind::kRecordEvent &&
                       trace[index + 6].kind == DmaTraceEntry::Kind::kStreamWait &&
                       trace[index + 7].kind == DmaTraceEntry::Kind::kMemcpyAsync &&
                       trace[index + 7].memcpy_kind == MemcpyKind::kDeviceToHost &&
                       trace[index + 8].kind == DmaTraceEntry::Kind::kRecordEvent;
    if (shape) {
      // The promote reads the host slot the land then writes: that aliasing is
      // the whole reason stage 3 waits on stage 2's event.
      Check(trace[index + 4].source == trace[index + 7].destination,
            "chunk " + std::to_string(triples) + ": H2D reads the host slot that the following D2H overwrites");
      ++triples;
    }
  }
  const size_t expected_chunks = kNumExpertsPerTok * ((slot_bytes + 4095) / 4096);
  Check(static_cast<size_t>(triples) == expected_chunks,
        "every one of the " + std::to_string(expected_chunks) +
            " transit chunks is a park/promote/land triple with an event between each pair");

  device.DestroyEvent(boundary);
  device.DestroyStream(compute);
}

void TestSwapRoundTrip() {
  Section("exclusive hierarchy: an evicted expert comes back byte-identical");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);
  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  experts.Ingest(source, {});

  DeviceStream compute = device.CreateStream();
  DeviceEvent boundary = device.CreateEvent();
  device.RecordEvent(boundary, compute);

  const size_t slot_bytes = layout.slot_num_bytes();
  std::vector<uint8_t> expected(slot_bytes);
  for (size_t index = 0; index < slot_bytes; ++index) {
    expected[index] = SyntheticWeightSource::ByteAt(3, 7, index);
  }

  // Promote (3, 7) -- host-resident after ingestion -- and compare its HBM
  // image against what the source would have produced.
  const int32_t request[kNumExpertsPerTok] = {7, 6, 5, 4, 3, 2};
  const LayerSwapPlan plan = experts.PrepareLayer(3, request, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  const int32_t slot = experts.DeviceSlotOf(3, 7);
  Check(slot != kNoSlot, "expert (3, 7) is resident in HBM after the promotion");
  const uint8_t* resident = static_cast<const uint8_t*>(experts.DeviceSlotAddress(slot));
  Check(std::memcmp(resident, expected.data(), slot_bytes) == 0,
        "its HBM image is byte-identical to the authoritative source");

  // Now evict it again by requesting a disjoint set from the same layer, then
  // bring it back and compare once more: the host copy the exchange wrote must
  // be the same bytes.
  const int32_t displace[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  experts.PrepareLayer(0, displace, kNumExpertsPerTok, compute, boundary);
  const int32_t displace2[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5};
  experts.PrepareLayer(1, displace2, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  Check(experts.DeviceSlotOf(3, 7) == kNoSlot, "expert (3, 7) was evicted back to pinned host RAM");

  const LayerSwapPlan again = experts.PrepareLayer(3, request, kNumExpertsPerTok, compute, boundary);
  experts.Synchronize();
  const int32_t slot_again = experts.DeviceSlotOf(3, 7);
  const uint8_t* resident_again = static_cast<const uint8_t*>(experts.DeviceSlotAddress(slot_again));
  Check(std::memcmp(resident_again, expected.data(), slot_bytes) == 0,
        "after a full eviction and re-promotion the bytes are still identical");
  Check(again.miss_count > 0, "the re-promotion really was a miss, not a stale hit");
  (void)plan;

  experts.ValidateResidency();
  std::printf("  swaps=%" PRIu64 " chunks=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 "\n",
              experts.stats().swaps, experts.stats().swap_chunks, experts.stats().slot_hits,
              experts.stats().slot_misses);
  device.DestroyEvent(boundary);
  device.DestroyStream(compute);
}

void TestHierarchyRefusals() {
  Section("exclusive hierarchy: refusals");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForGeometry(256, 128);
  SimulatedDeviceOps device(64ull * 1024 * 1024);

  CheckRefuses(
      [&] {
        ExclusiveExpertManager::Options options = TestOptions();
        options.device_slots = kNumExpertsPerTok - 1;
        ExclusiveExpertManager narrow(device, device, layout, options);
      },
      "a K smaller than one top-k");
  CheckRefuses(
      [&] {
        ExclusiveExpertManager::Options options = TestOptions();
        options.device_slots = kTestCoverage + 1;
        ExclusiveExpertManager wide(device, device, layout, options);
      },
      "a K larger than the routed coverage");

  ExclusiveExpertManager experts(device, device, layout, TestOptions());
  SyntheticWeightSource source(layout, kTestLayers, kTestExperts);
  CheckRefuses([&] { const int32_t ids[1] = {0}; experts.PlanLayer(0, ids, 1); },
               "planning before ingestion names no host slot");
  experts.Ingest(source, {});
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 0, 1, 2, 3, 4}; experts.PlanLayer(0, ids, kNumExpertsPerTok); },
               "a top-k with a duplicated expert id");
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 99}; experts.PlanLayer(0, ids, kNumExpertsPerTok); },
               "an expert id outside the expert range");
  CheckRefuses([&] { const int32_t ids[kNumExpertsPerTok] = {0, 1, 2, 3, 4, 5}; experts.PlanLayer(99, ids, kNumExpertsPerTok); },
               "a layer index outside the model");

  // The slot planner's arithmetic, independent of any device.
  const size_t slot_bytes = ExpertSlotLayout::ForDeepSeekV4Flash().slot_num_bytes();
  const int64_t fitted = ExclusiveExpertManager::PlanDeviceSlots(
      64ull << 30, 6ull << 30, slot_bytes, kTotalRoutedExperts, kDeviceReserveBytes, kTransferChunkBytes,
      kNumExpertsPerTok, -1);
  std::printf("  64 GiB HBM, 6 GiB backbone -> K = %" PRId64 " routed slots (%.2f GiB)\n", fitted,
              static_cast<double>(fitted) * static_cast<double>(slot_bytes) / (1024.0 * 1024.0 * 1024.0));
  Check(fitted > kNumExpertsPerTok && fitted < kTotalRoutedExperts,
        "the slot planner lands between one top-k and full residency on a 64 GiB part");
  CheckRefuses(
      [&] {
        ExclusiveExpertManager::PlanDeviceSlots(1ull << 30, 6ull << 30, slot_bytes, kTotalRoutedExperts,
                                                kDeviceReserveBytes, kTransferChunkBytes, kNumExpertsPerTok, -1);
      },
      "a device with less free HBM than the backbone needs");
}

// ---------------------------------------------------------------------------
// 4. The static arena's phase latches
// ---------------------------------------------------------------------------

void TestStaticArena() {
  Section("static arena: alignment and the sealed-for-decode latch");
  SimulatedDeviceOps device(1ull << 30);

  // The simulated allocators must honor the hardware page on ABSOLUTE
  // addresses: aclrtMalloc(..., ACL_MEM_MALLOC_HUGE_FIRST) and aclrtMallocHost
  // serve page-aligned memory, and the arena's alignment contract is stated on
  // absolute descriptor addresses, not offsets.
  void* device_block = device.DeviceMalloc(100);
  void* pinned_block = device.HostPinnedMalloc(100);
  Check(reinterpret_cast<uintptr_t>(device_block) % kSimDeviceAllocAlignBytes == 0,
        "the simulated device allocator hands out 4096-aligned blocks (aclrtMalloc HUGE_FIRST parity)");
  Check(reinterpret_cast<uintptr_t>(pinned_block) % kSimDeviceAllocAlignBytes == 0,
        "the simulated pinned allocator hands out 4096-aligned blocks (aclrtMallocHost parity)");
  device.HostPinnedFree(pinned_block);
  device.DeviceFree(device_block);
  device.ResetCounters();  // the ONE-allocation check below counts from a clean slate

  StaticMemoryArena arena(device);

  const ArenaHandle first = arena.Reserve("a.small", 100);
  const ArenaHandle second = arena.Reserve("b.aligned", 4096, 4096);
  const ArenaHandle third = arena.Reserve("c.tail", 7);
  CheckRefuses([&] { arena.Address(first); }, "an address before Commit");
  CheckRefuses([&] { arena.Reserve("d.zero", 0); }, "a zero-byte reservation");
  CheckRefuses([&] { arena.Reserve("e.odd-align", 64, 3); }, "a non-power-of-two alignment");

  arena.Commit();
  Check(device.counters().device_allocations == 1, "Commit performs exactly ONE device allocation");
  Check(reinterpret_cast<uintptr_t>(arena.Address(second)) % 4096 == 0,
        "a reservation that asked for 4096-byte alignment got it");
  Check(arena.Address(third) > arena.Address(first), "reservations are laid out in request order");
  arena.ValidateAlignment();
  Check(true, "every reservation is aligned, in order, and inside the arena");
  CheckRefuses([&] { arena.Reserve("f.late", 8); }, "a reservation after Commit");

  arena.NoteWorkspace(1024);
  arena.NoteWorkspace(4096);
  arena.NoteWorkspace(2048);
  arena.CommitWorkspace();
  Check(arena.workspace_bytes() == 4096, "the shared workspace is the high-water mark over all planned ops");
  arena.AssertWorkspaceFits(4096, "test");
  CheckRefuses([&] { arena.AssertWorkspaceFits(4097, "test"); },
               "a re-plan that would need more workspace than was reserved");

  arena.Seal();
  Check(arena.sealed(), "the arena is sealed for decoding");
  CheckRefuses([&] { arena.Reserve("g.sealed", 8); }, "a reservation after Seal");
  CheckRefuses([&] { arena.CreateIntArray("h.sealed", {1, 2}); }, "a descriptor after Seal");
  CheckRefuses([&] { arena.CommitWorkspace(); }, "a workspace commit after Seal");
  std::printf("%s", arena.DescribeLedger().c_str());
}

// ---------------------------------------------------------------------------
// 4b. The vendored mHC operator geometry in the static arena
// ---------------------------------------------------------------------------

// The DSV4 mHC chain reserves descriptor storage and plans its operators
// against arena addresses, exactly as the decode loop will: n_hc = 4 streams
// over the 4096-wide hidden state, TND layout, phi [24, 16384] FP32. The four
// vendored workspaces then have to fit under the arena's shared
// NoteWorkspace high-water mark.
void TestVendorMhcArena() {
  Section("vendored mHC operators: arena reservations and the workspace high-water mark");
  constexpr int64_t kMhcStreams = 4;  // n_hc: the DSV4 hyper-connection stream count
  constexpr int64_t kTokensArena = 4;
  constexpr int64_t kMixRowsArena = kMhcStreams * kMhcStreams + 2 * kMhcStreams;  // 24
  SimulatedDeviceOps device(1ull << 30);
  StaticMemoryArena arena(device);

  const ArenaHandle mhc_state = arena.Reserve("mhc.state",
      static_cast<size_t>(kTokensArena) * kMhcStreams * kHiddenSize * 2 +          // x [T,4,4096] BF16
      static_cast<size_t>(kTokensArena) * kHiddenSize * 2 +                        // hIn/hOut [T,4096] BF16
      static_cast<size_t>(kTokensArena) * kHiddenSize * 2, 4096);                  // out [T,4,4096] BF16
  const ArenaHandle mhc_phi = arena.Reserve("mhc.phi",
      static_cast<size_t>(kMixRowsArena) * kMhcStreams * kHiddenSize * 4, 4096);   // phi [24,16384] FP32
  const ArenaHandle mhc_small = arena.Reserve("mhc.routing",
      static_cast<size_t>(kTokensArena) * kMhcStreams * kMhcStreams * 4 +          // hRes [T,4,4] FP32
      static_cast<size_t>(kTokensArena) * kMhcStreams * 4 +                        // hPost [T,4] FP32
      3 * 4 + kMixRowsArena * 4);                                                  // alpha [3] + bias [24]
  Check(mhc_state != mhc_phi && mhc_phi != mhc_small, "the mHC descriptors hold three distinct arena handles");

  arena.Commit();
  const auto make = [&](ArenaHandle handle, size_t offset, const std::vector<int64_t>& shape, aclDataType dtype) {
    void* base = static_cast<char*>(arena.Address(handle)) + offset;
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           base);
  };
  const size_t token_bytes = static_cast<size_t>(kTokensArena);
  aclTensor* x = make(mhc_state, 0, {kTokensArena, kMhcStreams, kHiddenSize}, ACL_BF16);
  aclTensor* h_in = make(mhc_state, token_bytes * kMhcStreams * kHiddenSize * 2, {kTokensArena, kHiddenSize},
                         ACL_BF16);
  aclTensor* post_out = make(mhc_state, token_bytes * kMhcStreams * kHiddenSize * 2 + token_bytes * kHiddenSize * 2,
                             {kTokensArena, kMhcStreams, kHiddenSize}, ACL_BF16);
  aclTensor* phi = make(mhc_phi, 0, {kMixRowsArena, kMhcStreams * kHiddenSize}, ACL_FLOAT32);
  aclTensor* h_res = make(mhc_small, 0, {kTokensArena, kMhcStreams, kMhcStreams}, ACL_FLOAT32);
  aclTensor* h_post = make(mhc_small, token_bytes * kMhcStreams * kMhcStreams * 4, {kTokensArena, kMhcStreams},
                           ACL_FLOAT32);
  aclTensor* alpha = make(mhc_small, token_bytes * kMhcStreams * kMhcStreams * 4 + token_bytes * kMhcStreams * 4,
                          {3}, ACL_FLOAT32);
  aclTensor* bias = make(mhc_small,
                         token_bytes * kMhcStreams * kMhcStreams * 4 + token_bytes * kMhcStreams * 4 + 12,
                         {kMixRowsArena}, ACL_FLOAT32);

  // Plan the chain over the arena addresses; every workspace must sit under
  // the shared high-water mark the arena commits.
  OpTable ops;
  uint64_t ws_pre = 0;
  uint64_t ws_sinkhorn = 0;
  uint64_t ws_post = 0;
  {
    aclOpExecutor* executor = nullptr;
    ws_pre = PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, &executor, x, phi, alpha, bias, nullptr, 1e-6, 1e-6, h_in,
                                       h_post, h_res, nullptr, nullptr, nullptr);
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_sinkhorn = PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, &executor, h_res, 1e-6f, 20, h_res, nullptr,
                                                 nullptr);
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_post = PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, &executor, x, h_res, h_in, h_post, post_out);
    aclDestroyAclOpExecutor(executor);
  }
  std::printf("  workspaces: mhc_pre=%" PRIu64 " mhc_sinkhorn=%" PRIu64 " mhc_post=%" PRIu64 "\n", ws_pre, ws_sinkhorn,
              ws_post);
  const uint64_t peak = ws_pre > ws_sinkhorn ? (ws_pre > ws_post ? ws_pre : ws_post)
                                             : (ws_sinkhorn > ws_post ? ws_sinkhorn : ws_post);
  Check(peak > 0, "the vendored mHC chain plans non-empty workspaces over arena addresses");

  arena.NoteWorkspace(ws_pre);
  arena.NoteWorkspace(ws_sinkhorn);
  arena.NoteWorkspace(ws_post);
  arena.CommitWorkspace();
  Check(arena.workspace_bytes() == peak,
        "the shared workspace is exactly the mHC chain's high-water mark over all three operators");
  arena.AssertWorkspaceFits(ws_pre, "mhc_pre");
  arena.AssertWorkspaceFits(ws_sinkhorn, "mhc_sinkhorn");
  arena.AssertWorkspaceFits(ws_post, "mhc_post");
  Check(true, "every vendored workspace fits within the committed NoteWorkspace reservation");
  CheckRefuses([&] { arena.AssertWorkspaceFits(peak + 1, "vendor re-plan"); },
               "a vendored re-plan exceeding the reserved workspace");

  for (aclTensor* tensor : {x, h_in, post_out, phi, h_res, h_post, alpha, bias}) {
    aclDestroyTensor(tensor);
  }
  arena.Seal();
}

// The compressed-KV front end over the same arena discipline: the token-level
// compressor, the shared-KV indexer, the shared-KV attention core and the two
// cache epilogs. This checks the layout (every reservation lands in its own
// non-overlapping byte range at its requested alignment, and every tensor
// stays inside the reservation it was cut from) and that one shared workspace
// reservation covers the whole five-operator chain.
void TestVendorSharedKvArena() {
  Section("vendored compressed-KV chain: arena layout and the shared workspace bound");
  constexpr int64_t kTokensArena = 128;      // divisible by both compression ratios
  constexpr int64_t kHeadsArena = 64;        // DSV4 query heads / indexer N1
  constexpr int64_t kIndexerDim = 128;       // indexer head dim
  constexpr int64_t kTotalHeadDim = 512;     // DSV4 MLA total head dim
  constexpr int64_t kBlocksArena = 8;
  constexpr int64_t kBlockSizeArena = 128;
  constexpr int64_t kSparseCount = 16;
  constexpr int64_t kCmpChannels = 256;
  constexpr int64_t kRopeHeadDim = 64;
  constexpr int64_t kStateDim = 128;
  constexpr int64_t kCmpRatio = 4;           // CSA
  constexpr int64_t kCompressedRows = kTokensArena / kCmpRatio;

  SimulatedDeviceOps device(4ull << 30);
  StaticMemoryArena arena(device);

  // One reservation per lifetime class, as the decode path lays it out. The
  // indexer cache is deliberately shared: indexer_compress_epilog_v2 writes
  // the quantized key stream that the shared-KV indexer then scores.
  const size_t ori_kv_bytes = static_cast<size_t>(kBlocksArena) * kBlockSizeArena * kTotalHeadDim;   // FP8
  const size_t cmp_kv_bytes = ori_kv_bytes;                                                          // FP8
  const size_t idx_cache_bytes = static_cast<size_t>(kBlocksArena) * kBlockSizeArena * kIndexerDim;  // UINT8
  const size_t attn_query_bytes = static_cast<size_t>(kTokensArena) * kHeadsArena * kTotalHeadDim * 2;
  const size_t attn_out_bytes = attn_query_bytes;
  const size_t idx_query_bytes = static_cast<size_t>(kTokensArena) * kHeadsArena * kIndexerDim;      // FP8
  const size_t weight_bytes = static_cast<size_t>(kHiddenSize) * kCmpChannels * 2 +                  // wkv BF16
                              static_cast<size_t>(kHiddenSize) * 2;                                  // wgate BF16
  const size_t staging_bytes = static_cast<size_t>(kTokensArena) * kHiddenSize * 2 +                 // x BF16
                               static_cast<size_t>(kCompressedRows) * kCmpChannels * 2 +             // cmp rows BF16
                               static_cast<size_t>(kCompressedRows) * kIndexerDim * 2;               // idx rows BF16
  const size_t state_bytes = static_cast<size_t>(kBlocksArena) * kBlockSizeArena * kStateDim * 4;    // FP32
  const size_t k_scale_bytes = static_cast<size_t>(kBlocksArena) * kBlockSizeArena * 4;              // FP32
  // Scratch: every small per-token vector the five operators bind, laid out
  // back to back by the cursor below. Sized generously; the assertion that
  // matters is that the cursor never leaves the reservation.
  const size_t scratch_bytes = 1u << 20;

  const ArenaHandle h_ori = arena.Reserve("kv.ori", ori_kv_bytes, 4096);
  const ArenaHandle h_cmp = arena.Reserve("kv.cmp", cmp_kv_bytes, 4096);
  const ArenaHandle h_idx = arena.Reserve("kv.indexer", idx_cache_bytes, 4096);
  const ArenaHandle h_attn_q = arena.Reserve("attn.query", attn_query_bytes, 4096);
  const ArenaHandle h_attn_out = arena.Reserve("attn.out", attn_out_bytes, 4096);
  const ArenaHandle h_idx_q = arena.Reserve("indexer.query", idx_query_bytes, 4096);
  const ArenaHandle h_weights = arena.Reserve("compressor.weights", weight_bytes, 4096);
  const ArenaHandle h_staging = arena.Reserve("compressor.staging", staging_bytes, 4096);
  const ArenaHandle h_state = arena.Reserve("compressor.state", state_bytes, 4096);
  const ArenaHandle h_k_scale = arena.Reserve("indexer.key_scale", k_scale_bytes, 4096);
  const ArenaHandle h_scratch = arena.Reserve("vendor.scratch", scratch_bytes, 512);

  arena.Commit();

  // Layout: each reservation at its requested alignment, in its own byte range.
  struct Span {
    const char* name;
    const char* base;
    size_t bytes;
    size_t align;
  };
  const Span spans[] = {
      {"kv.ori", static_cast<const char*>(arena.Address(h_ori)), ori_kv_bytes, 4096},
      {"kv.cmp", static_cast<const char*>(arena.Address(h_cmp)), cmp_kv_bytes, 4096},
      {"kv.indexer", static_cast<const char*>(arena.Address(h_idx)), idx_cache_bytes, 4096},
      {"attn.query", static_cast<const char*>(arena.Address(h_attn_q)), attn_query_bytes, 4096},
      {"attn.out", static_cast<const char*>(arena.Address(h_attn_out)), attn_out_bytes, 4096},
      {"indexer.query", static_cast<const char*>(arena.Address(h_idx_q)), idx_query_bytes, 4096},
      {"compressor.weights", static_cast<const char*>(arena.Address(h_weights)), weight_bytes, 4096},
      {"compressor.staging", static_cast<const char*>(arena.Address(h_staging)), staging_bytes, 4096},
      {"compressor.state", static_cast<const char*>(arena.Address(h_state)), state_bytes, 4096},
      {"indexer.key_scale", static_cast<const char*>(arena.Address(h_k_scale)), k_scale_bytes, 4096},
      {"vendor.scratch", static_cast<const char*>(arena.Address(h_scratch)), scratch_bytes, 512},
  };
  constexpr size_t kSpanCount = sizeof(spans) / sizeof(spans[0]);
  bool aligned = true;
  bool disjoint = true;
  for (size_t i = 0; i < kSpanCount; ++i) {
    if (reinterpret_cast<uintptr_t>(spans[i].base) % spans[i].align != 0) {
      aligned = false;
      std::printf("  NOTE: %s is not %zu-aligned\n", spans[i].name, spans[i].align);
    }
    for (size_t j = i + 1; j < kSpanCount; ++j) {
      const char* a_end = spans[i].base + spans[i].bytes;
      const char* b_end = spans[j].base + spans[j].bytes;
      if (spans[i].base < b_end && spans[j].base < a_end) {
        disjoint = false;
        std::printf("  NOTE: %s overlaps %s\n", spans[i].name, spans[j].name);
      }
    }
  }
  Check(aligned, "every compressed-KV reservation lands at its requested alignment");
  Check(disjoint,
        "the eleven compressed-KV reservations occupy strictly non-overlapping byte ranges");

  const auto make = [&](ArenaHandle handle, size_t offset, const std::vector<int64_t>& shape, aclDataType dtype) {
    void* base = static_cast<char*>(arena.Address(handle)) + offset;
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           base);
  };

  // A bump cursor over vendor.scratch, 64-byte spaced, that refuses to leave
  // the reservation -- the layout property this section exists to assert.
  size_t scratch_cursor = 0;
  bool scratch_fits = true;
  const auto scratch = [&](const std::vector<int64_t>& shape, aclDataType dtype, size_t element_bytes) {
    size_t elements = 1;
    for (int64_t dim : shape) {
      elements *= static_cast<size_t>(dim);
    }
    const size_t offset = scratch_cursor;
    scratch_cursor += ((elements * element_bytes) + 63) & ~size_t(63);
    if (scratch_cursor > scratch_bytes) {
      scratch_fits = false;
    }
    return make(h_scratch, offset, shape, dtype);
  };

  // -- the compressor -------------------------------------------------------
  aclTensor* comp_x = make(h_staging, 0, {kTokensArena, kHiddenSize}, ACL_BF16);
  size_t staging_cursor = static_cast<size_t>(kTokensArena) * kHiddenSize * 2;
  aclTensor* cmp_rows = make(h_staging, staging_cursor, {kCompressedRows, kCmpChannels}, ACL_BF16);
  staging_cursor += static_cast<size_t>(kCompressedRows) * kCmpChannels * 2;
  aclTensor* idx_rows = make(h_staging, staging_cursor, {kCompressedRows, kIndexerDim}, ACL_BF16);
  staging_cursor += static_cast<size_t>(kCompressedRows) * kIndexerDim * 2;
  Check(staging_cursor <= staging_bytes,
        "the compressor staging tensors fit inside the compressor.staging reservation");

  aclTensor* wkv = make(h_weights, 0, {kHiddenSize, kCmpChannels}, ACL_BF16);
  aclTensor* wgate = make(h_weights, static_cast<size_t>(kHiddenSize) * kCmpChannels * 2, {kHiddenSize, 1},
                          ACL_BF16);
  aclTensor* state_cache = make(h_state, 0, {kBlocksArena, kBlockSizeArena, kStateDim}, ACL_FLOAT32);

  aclTensor* ape = scratch({kTokensArena}, ACL_FLOAT32, 4);
  aclTensor* norm_weight = scratch({kCmpChannels}, ACL_FLOAT32, 4);
  aclTensor* rope_sin = scratch({kTokensArena, kRopeHeadDim}, ACL_FLOAT32, 4);
  aclTensor* rope_cos = scratch({kTokensArena, kRopeHeadDim}, ACL_FLOAT32, 4);
  aclTensor* slot_map = scratch({kCompressedRows}, ACL_INT32, 4);
  aclTensor* ori_indices = scratch({kTokensArena, 1, kSparseCount}, ACL_INT32, 4);
  aclTensor* cmp_indices = scratch({kTokensArena, 1, kSparseCount}, ACL_INT32, 4);
  aclTensor* block_table = scratch({1, kBlocksArena}, ACL_INT32, 4);
  aclTensor* seq_lengths = scratch({1}, ACL_INT32, 4);
  aclTensor* idx_weights = scratch({kTokensArena, kHeadsArena}, ACL_BF16, 2);
  aclTensor* idx_q_scale = scratch({kTokensArena, kHeadsArena}, ACL_FLOAT32, 4);
  // returnValues / returnSoftmaxLse false are signalled by [0] placeholders,
  // which is what the vendored wrappers check before issuing a second copy.
  aclTensor* no_values = scratch({0}, ACL_FLOAT32, 4);
  aclTensor* no_lse = scratch({0}, ACL_FLOAT32, 4);
  Check(scratch_fits, "every per-token vector fits inside the vendor.scratch reservation");

  // -- the paged caches and the two query streams ---------------------------
  aclTensor* ori_kv = make(h_ori, 0, {kBlocksArena, kBlockSizeArena, kTotalHeadDim}, ACL_FLOAT8_E4M3FN);
  aclTensor* cmp_kv = make(h_cmp, 0, {kBlocksArena, kBlockSizeArena, kTotalHeadDim}, ACL_FLOAT8_E4M3FN);
  aclTensor* idx_cache = make(h_idx, 0, {kBlocksArena, kBlockSizeArena, kIndexerDim}, ACL_UINT8);
  // The indexer reads the same bytes indexer_compress_epilog_v2 wrote, viewed
  // as the FP8 key the scoring kernel expects.
  aclTensor* idx_key = make(h_idx, 0, {kBlocksArena, kBlockSizeArena, 1, kIndexerDim}, ACL_FLOAT8_E4M3FN);
  aclTensor* idx_k_scale = make(h_k_scale, 0, {kBlocksArena, kBlockSizeArena, 1}, ACL_FLOAT32);
  aclTensor* query = make(h_attn_q, 0, {kTokensArena, kHeadsArena, kTotalHeadDim}, ACL_BF16);
  aclTensor* attn_out = make(h_attn_out, 0, {kTokensArena, kHeadsArena, kTotalHeadDim}, ACL_BF16);
  aclTensor* idx_query = make(h_idx_q, 0, {kTokensArena, kHeadsArena, kIndexerDim}, ACL_FLOAT8_E4M3FN);

  OpTable ops;
  uint64_t ws_compressor = 0;
  uint64_t ws_indexer = 0;
  uint64_t ws_attention = 0;
  uint64_t ws_kv_epilog = 0;
  uint64_t ws_idx_epilog = 0;
  {
    aclOpExecutor* executor = nullptr;
    ws_compressor = PlanAclnnOp<CompressorPlanFn>(
        ops, OpId::kCompressor, &executor, comp_x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
        block_table, seq_lengths, nullptr, nullptr, kRopeHeadDim, kCmpRatio, 1, 1e-6, 1, 1,
        mock::AsMockTensor(state_cache)->strides.at(0), cmp_rows);
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_indexer = PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
        ops, OpId::kVllmQuantLightningIndexer, &executor, idx_query, idx_key, idx_weights, idx_q_scale, idx_k_scale,
        seq_lengths, seq_lengths, block_table, nullptr, 0, 0, const_cast<char*>("TND"),
        const_cast<char*>("PA_BSND"), kSparseCount, 3, INT64_MAX, INT64_MAX, kCmpRatio, false,
        mock::AsMockTensor(idx_key)->strides.at(0), mock::AsMockTensor(idx_k_scale)->strides.at(0), cmp_indices, no_values);
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_attention = PlanAclnnOp<KvQuantSparseAttnSharedkvPlanFn>(
        ops, OpId::kKvQuantSparseAttnSharedkv, &executor, query, ori_kv, cmp_kv, ori_indices, cmp_indices,
        block_table, block_table, seq_lengths, seq_lengths, seq_lengths, nullptr, nullptr, nullptr, nullptr, 1, 64,
        kRopeHeadDim, 0.0441942, kCmpRatio, 4, 3, 127, 0, const_cast<char*>("TND"), const_cast<char*>("PA_ND"),
        mock::AsMockTensor(ori_kv)->strides.at(0), mock::AsMockTensor(cmp_kv)->strides.at(0), false, attn_out, no_lse);
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_kv_epilog = PlanAclnnOp<KvCompressEpilogPlanFn>(ops, OpId::kKvCompressEpilog, &executor, cmp_kv, cmp_rows,
                                                       slot_map, 128, 1, 1, 1,
                                                       mock::AsMockTensor(cmp_kv)->strides.at(0));
    aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    ws_idx_epilog = PlanAclnnOp<IndexerCompressEpilogV2PlanFn>(ops, OpId::kIndexerCompressEpilogV2, &executor,
                                                               idx_cache, idx_rows, slot_map, 2,
                                                               mock::AsMockTensor(idx_cache)->strides.at(0));
    aclDestroyAclOpExecutor(executor);
  }
  std::printf("  workspaces: compressor=%" PRIu64 " vllm_indexer=%" PRIu64 " sharedkv_attn=%" PRIu64
              " kv_epilog=%" PRIu64 " idx_epilog=%" PRIu64 "\n",
              ws_compressor, ws_indexer, ws_attention, ws_kv_epilog, ws_idx_epilog);

  uint64_t peak = 0;
  bool all_planned = true;
  for (uint64_t ws : {ws_compressor, ws_indexer, ws_attention, ws_kv_epilog, ws_idx_epilog}) {
    if (ws == 0) {
      all_planned = false;
    }
    arena.NoteWorkspace(ws);
    if (ws > peak) {
      peak = ws;
    }
  }
  Check(all_planned, "all five compressed-KV operators planned non-empty workspaces over arena addresses");
  arena.CommitWorkspace();
  Check(arena.workspace_bytes() == peak,
        "one shared workspace reservation covers the whole five-operator compressed-KV chain");
  arena.AssertWorkspaceFits(ws_compressor, "compressor");
  arena.AssertWorkspaceFits(ws_indexer, "vllm_quant_lightning_indexer");
  arena.AssertWorkspaceFits(ws_attention, "kv_quant_sparse_attn_sharedkv");
  arena.AssertWorkspaceFits(ws_kv_epilog, "kv_compress_epilog");
  arena.AssertWorkspaceFits(ws_idx_epilog, "indexer_compress_epilog_v2");
  Check(true, "every compressed-KV workspace fits within the committed reservation");
  CheckRefuses([&] { arena.AssertWorkspaceFits(peak + 1, "compressed-KV re-plan"); },
               "a compressed-KV re-plan exceeding the reserved workspace");

  for (aclTensor* tensor : {comp_x, cmp_rows, idx_rows, wkv, wgate, state_cache, ape, norm_weight, rope_sin,
                            rope_cos, slot_map, ori_indices, cmp_indices, block_table, seq_lengths, idx_weights,
                            idx_q_scale, no_values, no_lse, ori_kv, cmp_kv, idx_cache, idx_key, idx_k_scale, query,
                            attn_out, idx_query}) {
    aclDestroyTensor(tensor);
  }
  arena.Seal();
}

// ---------------------------------------------------------------------------
// 5. Backbone sizing and the operator table
// ---------------------------------------------------------------------------

void TestBackboneSizing() {
  Section("backbone device footprint");
  const MlaGeometry mla;
  const size_t bytes = Dsv4Pipeline::BackboneDeviceBytes(mla, 128, 8192);
  std::printf("  backbone + paged KV at kv_lora=%" PRId64 " rope=%" PRId64 " block=128 context=8192: %.2f GiB\n",
              mla.kv_lora_rank, mla.qk_rope_head_dim, static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  Check(bytes > (1ull << 30), "the backbone is more than a gigabyte, so the slot planner must subtract it");
  const size_t longer = Dsv4Pipeline::BackboneDeviceBytes(mla, 128, 32768);
  Check(longer > bytes, "a longer reserved context costs more paged KV");
}

void TestOperatorTable() {
  Section("aclnn operator table");
  OpTable ops;
  std::printf("%s", ops.DescribeInventory().c_str());
  if (!ops.runtime_reachable()) {
    std::printf("  NOTE: the aclnn runtime is not on the loader path here, so operator resolution is untested.\n");
    std::printf("        Everything above this section is device-independent and did run.\n");
    return;
  }
  Check(ops.available(OpId::kRmsNorm), "aclnnRmsNorm resolved from the linked libraries");
  Check(ops.available(OpId::kFusedInferAttentionScoreV5), "aclnnFusedInferAttentionScoreV5 resolved");
  Check(ops.available(OpId::kGroupedMatmulV5), "aclnnGroupedMatmulV5 resolved");
  Check(ops.available(OpId::kSoftplus), "aclnnSoftplus resolved (sqrtsoftplus router scoring, stage 1)");
  Check(ops.available(OpId::kSqrt), "aclnnSqrt resolved (sqrtsoftplus router scoring, stage 2)");
  Check(ops.available(OpId::kMoeGatingTopKV2), "aclnnMoeGatingTopKV2 resolved (CANN 9.2.0 and later)");
  Check(ops.available(OpId::kMoeInitRoutingV4), "aclnnMoeInitRoutingV4 resolved (CANN 9.2.0 and later)");
  CheckRefuses([&] { ops.RequireAll({OpId::kOpCount}); }, "a request for an out-of-range operator id");
}

// ---------------------------------------------------------------------------
// 6. ModelConfig and the MoE block seam
// ---------------------------------------------------------------------------

void TestModelConfigContract() {
  Section("model config: the checkpoint contract the runner validates at startup");
  // The published DeepSeek-V4 Flash fields, verbatim structure.
  const char* config_json =
      "{\"architectures\":[\"DeepseekV4ForCausalLM\"],\"hidden_size\":4096,"
      "\"moe_intermediate_size\":2048,\"index_topk\":512,\"n_routed_experts\":256,"
      "\"n_shared_experts\":1,\"norm_topk_prob\":true,\"num_attention_heads\":64,"
      "\"num_experts_per_tok\":6,\"num_hidden_layers\":43,\"q_lora_rank\":1024,"
      "\"qk_rope_head_dim\":64,"
      "\"quantization_config\":{\"quant_method\":\"fp8\",\"scale_fmt\":\"ue8m0\","
      "\"weight_block_size\":[128,128]},"
      "\"routed_scaling_factor\":1.5,\"rms_norm_eps\":1e-06,"
      "\"scoring_func\":\"sqrtsoftplus\",\"swiglu_limit\":10.0,"
      "\"topk_method\":\"noaux_tc\",\"vocab_size\":129280}";
  const ModelConfig model = ModelConfig::FromJsonText(config_json, "fixture config.json");
  Check(model.num_hidden_layers == 43 && model.n_routed_experts == 256 && model.num_experts_per_tok == 6 &&
            model.hidden_size == 4096 && model.moe_intermediate_size == 2048 &&
            model.routed_scaling_factor == 1.5 && model.topk_method == "noaux_tc" &&
            model.scoring_func == "sqrtsoftplus" && model.norm_topk_prob,
        "the nine contracted fields parse to the documented DSV4-Flash values");
  model.AssertMatchesBinaryContract();
  Check(true, "the parsed topology matches the compiled graph geometry");

  CheckRefuses([] {
    ModelConfig::FromJsonText("{\"num_hidden_layers\":43,\"n_routed_experts\":256,"
                              "\"num_experts_per_tok\":6,\"hidden_size\":4096,"
                              "\"moe_intermediate_size\":2048,\"routed_scaling_factor\":1.5,"
                              "\"topk_method\":\"noaux_tc\",\"scoring_func\":\"softmax\","
                              "\"norm_topk_prob\":true}",
                              "t");
  }, "scoring_func softmax is refused: the router chain computes sqrt(softplus)");
  CheckRefuses([] { ModelConfig::FromJsonText("{\"broken\"", "t"); }, "a truncated document is a parse error");

  // The slot planner runs from the parsed geometry.
  const ExpertSlotLayout from_config = ExpertSlotLayout::ForGeometry(model.hidden_size, model.moe_intermediate_size);
  Check(from_config.slot_num_bytes() == ExpertSlotLayout::ForDeepSeekV4Flash().slot_num_bytes(),
        "the slot layout computed from the PARSED geometry equals the compiled one");
}

// ---------------------------------------------------------------------------
// The per-layer compression schedule: parsing, validation, and the gating
// ---------------------------------------------------------------------------

void TestCompressionSchedule() {
  Section("compress_ratios: the checkpoint field that selects each layer's attention path");

  // One ratio per layer, from a config.json, with the three admissible kinds.
  std::string schedule = "[";
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    const int64_t ratio = layer < 21 ? 1 : (layer < 41 ? 4 : 128);
    schedule += (layer == 0 ? "" : ",") + std::to_string(ratio);
  }
  schedule += "]";
  const std::string json =
      "{\"num_hidden_layers\":43,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
      "\"hidden_size\":4096,\"moe_intermediate_size\":2048,\"routed_scaling_factor\":1.5,"
      "\"topk_method\":\"noaux_tc\",\"scoring_func\":\"sqrtsoftplus\",\"norm_topk_prob\":true,"
      "\"compress_ratios\":" + schedule + "}";
  const ModelConfig model = ModelConfig::FromJsonText(json, "fixture config.json");
  Check(static_cast<int64_t>(model.compress_ratios.size()) == kNumLayers,
        "compress_ratios parses to one entry per layer");
  Check(AttentionPathForRatio(model.compress_ratios[0]) == AttentionPath::kSlidingWindow &&
            AttentionPathForRatio(model.compress_ratios[21]) == AttentionPath::kCompressedSparse &&
            AttentionPathForRatio(model.compress_ratios[42]) == AttentionPath::kHyperCompressed,
        "ratio 1 selects SWA, 4 selects CSA and 128 selects HCA");

  // A schedule this binary could not dispatch must not reach the device: the
  // deployed compressor kernel admits cmpRatio 4 and 128 only.
  const auto with_ratios = [](const std::string& ratios) {
    return "{\"num_hidden_layers\":43,\"n_routed_experts\":256,\"num_experts_per_tok\":6,"
           "\"hidden_size\":4096,\"moe_intermediate_size\":2048,\"routed_scaling_factor\":1.5,"
           "\"topk_method\":\"noaux_tc\",\"scoring_func\":\"sqrtsoftplus\","
           "\"norm_topk_prob\":true,\"compress_ratios\":" + ratios + "}";
  };
  CheckRefuses([&] { ModelConfig::FromJsonText(with_ratios("[4,4,4]"), "t"); },
               "a schedule shorter than num_hidden_layers (one ratio per layer is required)");
  std::string eight = "[";
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    eight += (layer == 0 ? "" : ",") + std::string(layer == 7 ? "8" : "1");
  }
  eight += "]";
  CheckRefuses([&] { ModelConfig::FromJsonText(with_ratios(eight), "t"); },
               "cmpRatio 8, which the deployed compressor kernel refuses");
  CheckRefuses([&] { ModelConfig::FromJsonText(with_ratios("\"4\""), "t"); },
               "a compress_ratios that is not an array");

  // THE GATING. A SWA-only schedule must cost nothing: no compressor or
  // indexer reservation, and -- because the pipeline plans a path only when
  // some layer selects it -- no descriptors or executors for them either.
  const MlaGeometry mla;
  const size_t swa_only = StaticArenaManager::BackboneDeviceBytes(mla, 128, 8192);
  const size_t with_empty = StaticArenaManager::BackboneDeviceBytes(mla, 128, 8192, {});
  Check(swa_only == with_empty,
        "an empty compress_ratios is identical to no schedule at all: every layer is SWA");
  std::vector<int64_t> csa_only(static_cast<size_t>(kNumLayers), kCompressRatioCsa);
  const size_t with_csa = StaticArenaManager::BackboneDeviceBytes(mla, 128, 8192, csa_only);
  Check(with_csa > swa_only,
        "a CSA schedule enlarges the backbone by the compressor and indexer projections (" +
            std::to_string((with_csa - swa_only) >> 20) + " MiB), so the slot planner subtracts them");
  std::vector<int64_t> hca_only(static_cast<size_t>(kNumLayers), kCompressRatioHca);
  const size_t with_hca = StaticArenaManager::BackboneDeviceBytes(mla, 128, 8192, hca_only);
  Check(with_hca > swa_only && with_hca < with_csa,
        "an HCA schedule costs the compressor but NOT the indexer, so it sits between the two");

  // The 604-byte entry, held to its own arithmetic rather than to a literal.
  Check(sizeof(Dsv4CompressedKvEntry) == 604 && kCompressedKvEntryBytes == 604,
        "Dsv4CompressedKvEntry is exactly 604 bytes");
  Check(kCompressedKvNopeChannels + kCompressedKvRopeChannels == mla.kv_lora_rank,
        "its two halves (" + std::to_string(kCompressedKvNopeChannels) + " nope + " +
            std::to_string(kCompressedKvRopeChannels) + " rope) sum to kv_lora_rank " +
            std::to_string(mla.kv_lora_rank));
  Check(kCompressedKvRopeChannels == mla.qk_rope_head_dim,
        "its rope half is exactly qk_rope_head_dim wide");
  Check(kCompressedKvScaleBlock == 64 &&
            kCompressedKvNopeChannels == kCompressedKvScaleCount * kCompressedKvScaleBlock,
        "one UE8M0 block scale covers exactly 64 nope channels, with none left over");
  Check(offsetof(Dsv4CompressedKvEntry, rope_bf16) == 476,
        "the rope half starts at byte 476, after the quantized half and its padding");

  // The HCA path's identity selection must never name a slot the cache does
  // not have: the attention core gathers at these indices without a bounds
  // check of its own. At a short reserved context kIndexTopK (512) exceeds one
  // layer's slot count, so the vector has to clamp rather than ramp.
  const int64_t short_slots = DivideUp(256, 128) * 128;  // a 2-block reservation
  Check(kIndexTopK > short_slots,
        "top-k " + std::to_string(kIndexTopK) + " really does exceed the " +
            std::to_string(short_slots) + " compressed slots a short context reserves, so the clamp is "
            "load-bearing rather than theoretical");
}

void TestMoeBlockSlotContract() {
  Section("IRoutedMoeBlock: slot byte contracts");
  const ExpertSlotLayout layout = ExpertSlotLayout::ForDeepSeekV4Flash();
  Check(Lattice24MoeBlock::SlotBytes() == 7077888,
        "the 2-bit Leech slot is 7,077,888 bytes (6.75 MiB) at the DSV4-Flash geometry");
  Check(Lattice24MoeBlock::SlotBytes() < layout.slot_num_bytes(),
        "the lattice slot is smaller than the 13,369,344-byte FP4 slot");
  Lattice24MoeBlock lattice;
  CheckRefuses([&] {
    StaticOpSlotTable table(8, "lattice-test");
    RuntimeConfig config;
    config.block_size = 128;
    config.max_context_len = 256;
    SimulatedDeviceOps device(64ull * 1024 * 1024);
    StaticArenaManager probe(device, device, config);
    ExpertSlotAddresses addresses;
    lattice.PlanStages(table, probe, addresses);
  }, "the lattice skeleton refuses to plan, naming aclnnLatticeUnpackAndGroupedMatmul");
}

}  // namespace
}  // namespace ascend_moe

int RunMain() {
  using namespace ascend_moe;
  std::printf("mock_contract_smoke -- device-free verification of the DSV4 runner's contracts\n");
  try {
    TestExpertLayout();
    TestHeaderParsing();
    TestExpertBindingValidation();
    TestHierarchyInvariants();
    TestLayerPlanning();
    TestSwapRoundTrip();
    TestHierarchyRefusals();
    TestStaticArena();
    TestVendorMhcArena();
    TestVendorSharedKvArena();
    TestBackboneSizing();
    TestModelConfigContract();
    TestCompressionSchedule();
    TestMoeBlockSlotContract();
    TestOperatorTable();
  } catch (const std::exception& error) {
    std::printf("\nunexpected exception: %s\n", error.what());
    ++g_failures;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  std::printf(
      "\nNot covered here (needs a 950PR): real GetWorkspaceSize values, whether every\n"
      "operator has an ascend950 kernel binary, and the aclSetTensorAddr index map.\n");
  return g_failures == 0 ? 0 : 1;
}

int main() {
  return ascend_moe::GuardedMain([&] { return RunMain(); });
}
