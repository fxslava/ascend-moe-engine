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

// Shared fixture for the dav-3510 CAModel block suites.
//
// Two executables include this: test_dsv4_moe_block_sim.cpp (the full-grid
// gate) and test_dsv4_moe_block_inductive_sim.cpp (the collapsed boundary
// grid). They are separate binaries, so the anonymous namespace below gives
// each its own copy and there is no ODR question.
//
// Holds: the geometry constants, the ACL error/require helpers, the launch
// hang watchdog, the exception-dump census, the output guard-band canary, the
// Runtime/DeviceBuffer RAII pair, the MXFP4 expert-bank fixture builder, and
// the diff/non-degeneracy metrics. The test BODIES stay in their own files.

#ifndef TESTS_SIM_DSV4_MOE_BLOCK_SIM_COMMON_HPP
#define TESTS_SIM_DSV4_MOE_BLOCK_SIM_COMMON_HPP

#include "dsv4_moe_block_tiling.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/kernels/arch35/launch.hpp"

#include <acl/acl.h>

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kHidden = 512;
constexpr std::uint32_t kInter = 512;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kExperts = 4;
constexpr std::uint32_t kTopK = 2;
constexpr std::uint32_t kTokens = 2;
constexpr float kLimit = 10.0F;
constexpr std::uint64_t kSeed = 0xD5A4B10CULL;
constexpr float kAbsTolerance = 1.0e-2F;
constexpr float kRelTolerance = 1.0e-2F;
// Per-launch hang budget. Generous on purpose: this converts an
// infinite wait into a named failure, it does not police speed.
// Each launch is top_k (or token count) whole experts of simulated
// work, so minutes are normal and hours are not.
constexpr int kLaunchBudgetSeconds = 5400;

using moe::kernels::arch35::E8M0;
using moe::kernels::arch35::Mxfp4Pair;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

std::string AclError(const char* call, aclError status) {
  const char* recent = aclGetRecentErrMsg();
  return std::string(call) + " failed with ACL status " +
         std::to_string(status) + ": " +
         (recent != nullptr ? recent : "<no recent ACL error>");
}

void RequireAcl(aclError status, const char* call) {
  if (status != ACL_SUCCESS) Fail(AclError(call, status));
}

// ---------------------------------------------------------------------------
// Hang watchdog
// ---------------------------------------------------------------------------
// A pipeline whose event handshakes do not balance does not fault -- it HANGS,
// and without a watchdog the only symptom is ctest expiring hours later with
// no indication of which launch stalled. Pattern taken from
// C:\vllm-ascend\csrc\tests\common\testing\camodel_guard.hpp.
//
// The budget is per launch and generous: it is here to convert an infinite
// wait into a named failure, not to police performance.
constexpr int kHangExitCode = 3;

class LaunchWatchdog {
 public:
  LaunchWatchdog(int budget_seconds, std::string tag)
      : budget_(budget_seconds),
        tag_(std::move(tag)),
        thread_([this] { Watch(); }) {}

  LaunchWatchdog(const LaunchWatchdog&) = delete;
  LaunchWatchdog& operator=(const LaunchWatchdog&) = delete;

  ~LaunchWatchdog() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
  }

  void Arm(const std::string& what) {
    std::lock_guard<std::mutex> lock(mutex_);
    what_ = what;
    deadline_ = std::chrono::steady_clock::now() + budget_;
    armed_ = true;
    ++generation_;
    wake_.notify_all();
  }

  void Disarm() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = false;
    ++generation_;
    wake_.notify_all();
  }

 private:
  void Watch() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
      if (!armed_) {
        wake_.wait(lock, [this] { return stop_ || armed_; });
        continue;
      }
      const std::uint64_t generation = generation_;
      const bool changed = wake_.wait_until(
          lock, deadline_,
          [this, generation] { return stop_ || generation_ != generation; });
      if (!changed) {
        std::printf(
            "\n%s HANG: %s did not return within %lld s; ending the process "
            "with %d. An unbalanced SetFlag/WaitFlag pair in the tile ring is "
            "the first thing to check.\n",
            tag_.c_str(), what_.c_str(),
            static_cast<long long>(budget_.count()), kHangExitCode);
        std::fflush(stdout);
        std::_Exit(kHangExitCode);
      }
    }
  }

  const std::chrono::seconds budget_;
  const std::string tag_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::string what_;
  std::chrono::steady_clock::time_point deadline_;
  bool armed_ = false;
  bool stop_ = false;
  std::uint64_t generation_ = 0;
  std::thread thread_;
};

// A launch can return ACL_SUCCESS and still have faulted on device, leaving an
// exception dump behind. Count non-empty ones so that case is not silently
// read as a pass.
struct DumpCensus {
  std::size_t files = 0;
  std::size_t non_empty = 0;
};

DumpCensus ExceptionDumps() {
  DumpCensus census;
  DIR* dir = opendir(".");
  if (dir == nullptr) return census;
  while (const dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    const bool is_dump =
        name.size() > 5 && name.compare(name.size() - 5, 5, ".dump") == 0;
    if (!is_dump || name.find("excp_log") == std::string::npos) continue;
    struct stat info {};
    if (stat(entry->d_name, &info) != 0) continue;
    ++census.files;
    if (info.st_size > 0) ++census.non_empty;
  }
  closedir(dir);
  return census;
}

// ---------------------------------------------------------------------------
// Output guard-bands
// ---------------------------------------------------------------------------
// The phase-2 refactor changed how the output tile ring maps slots to y rows,
// so an off-by-one in the store index is exactly the defect to fear. A
// tolerance compare over the LOGICAL extent cannot see a write past the end;
// a guard band can.
//
// 0x5A5A5A5A and not zero: zero is what a zeroed allocation, an early-return
// kernel and a half-issued burst all leave behind, so it cannot distinguish
// "nobody wrote here" from "something went wrong".
constexpr std::uint32_t kCanaryBits = 0x5A5A5A5AU;
constexpr std::size_t kGuardFloats = 64;  // 256 B past every output

float CanaryFloat() {
  float value = 0.0F;
  const std::uint32_t bits = kCanaryBits;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool IsCanary(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == kCanaryBits;
}

class Runtime {
 public:
  Runtime() {
    RequireAcl(aclInit(nullptr), "aclInit");
    initialized_ = true;
    std::uint32_t device_count = 0;
    RequireAcl(aclrtGetDeviceCount(&device_count), "aclrtGetDeviceCount");
    Require(device_count > 0,
            "CAModel reports no simulated Ascend device (device count is 0)");
    RequireAcl(aclrtSetDevice(0), "aclrtSetDevice(0)");
    device_set_ = true;
    RequireAcl(aclrtCreateStream(&stream_), "aclrtCreateStream");
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  ~Runtime() {
    if (stream_ != nullptr) aclrtDestroyStream(stream_);
    if (device_set_) aclrtResetDevice(0);
    if (initialized_) aclFinalize();
  }

  aclrtStream stream() const noexcept { return stream_; }

  void Close() {
    if (stream_ != nullptr) {
      RequireAcl(aclrtDestroyStream(stream_), "aclrtDestroyStream");
      stream_ = nullptr;
    }
    if (device_set_) {
      RequireAcl(aclrtResetDevice(0), "aclrtResetDevice(0)");
      device_set_ = false;
    }
    if (initialized_) {
      RequireAcl(aclFinalize(), "aclFinalize");
      initialized_ = false;
    }
  }

 private:
  aclrtStream stream_ = nullptr;
  bool initialized_ = false;
  bool device_set_ = false;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    Require(bytes_ > 0, "device buffer size must be nonzero");
    RequireAcl(aclrtMalloc(&pointer_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      const aclError status = aclrtFree(pointer_);
      if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "[CLEANUP] %s\n",
                     AclError("aclrtFree", status).c_str());
      }
    }
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

  void Zero() {
    RequireAcl(aclrtMemset(pointer_, bytes_, 0, bytes_), "aclrtMemset");
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_;
};

template <typename T>
void CopyToDevice(DeviceBuffer& destination, const std::vector<T>& source) {
  const std::size_t bytes = source.size() * sizeof(T);
  Require(destination.bytes() >= bytes, "host-to-device copy exceeds buffer");
  RequireAcl(aclrtMemcpy(destination.get(), destination.bytes(), source.data(),
                         bytes, ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(HOST_TO_DEVICE)");
}

template <typename T>
void CopyFromDevice(std::vector<T>& destination, const DeviceBuffer& source) {
  const std::size_t bytes = destination.size() * sizeof(T);
  Require(source.bytes() >= bytes, "device-to-host copy exceeds buffer");
  RequireAcl(aclrtMemcpy(destination.data(), bytes, source.get(), bytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(DEVICE_TO_HOST)");
}

// Allocates `logical + kGuardFloats` and fills the whole thing with the
// canary, so the guard tail is poisoned before the launch.
std::unique_ptr<DeviceBuffer> MakeGuardedOutput(std::size_t logical_floats) {
  auto buffer = std::make_unique<DeviceBuffer>(
      (logical_floats + kGuardFloats) * sizeof(float));
  const std::vector<float> poison(logical_floats + kGuardFloats, CanaryFloat());
  CopyToDevice(*buffer, poison);
  return buffer;
}

// Reads back only the guard tail and reports the first corrupted word.
void RequireGuardIntact(const DeviceBuffer& buffer, std::size_t logical_floats,
                        const char* what) {
  std::vector<float> guard(kGuardFloats, 0.0F);
  const std::size_t offset_bytes = logical_floats * sizeof(float);
  RequireAcl(aclrtMemcpy(guard.data(), guard.size() * sizeof(float),
                         static_cast<const std::uint8_t*>(buffer.get()) +
                             offset_bytes,
                         guard.size() * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(guard band)");
  for (std::size_t i = 0; i < guard.size(); ++i) {
    if (!IsCanary(guard[i])) {
      Fail(std::string(what) + ": guard band corrupted at float +" +
           std::to_string(i) + " past the logical extent (" +
           std::to_string(logical_floats) +
           " floats). The kernel stored past its output -- check the phase-2 "
           "tile ring's y row index.");
    }
  }
}

std::uint8_t EncodeFp4(float value) {
  std::uint8_t best = 0;
  float error = std::numeric_limits<float>::infinity();
  for (std::uint8_t code = 0; code < 16; ++code) {
    const float candidate = std::fabs(value - moe::emu::DecodeFp4E2M1(code));
    if (candidate < error) {
      best = code;
      error = candidate;
    }
  }
  return best;
}

// One contiguous buffer per role holding every expert, which is the device
// layout: expert e starts at e * <role>_expert_stride.
struct ExpertBank {
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::vector<Mxfp4Pair> packed;  // [experts, rows, cols/2]
  std::vector<E8M0> scales;       // [experts, rows, cols/32]

  std::size_t PackedStride() const {
    return static_cast<std::size_t>(rows) * (cols / 2u);
  }
  std::size_t ScaleStride() const {
    return static_cast<std::size_t>(rows) * (cols / kGroup);
  }
  moe::emu::Mxfp4Matrix View(std::uint32_t expert) const {
    return {packed.data() + static_cast<std::size_t>(expert) * PackedStride(),
            scales.data() + static_cast<std::size_t>(expert) * ScaleStride(),
            static_cast<int>(rows), static_cast<int>(cols)};
  }
};

ExpertBank MakeBank(std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t experts, std::mt19937_64& rng) {
  Require(cols % kGroup == 0, "MXFP4 width must be group aligned");
  ExpertBank bank;
  bank.rows = rows;
  bank.cols = cols;
  bank.packed.resize(static_cast<std::size_t>(experts) * rows * (cols / 2u));
  bank.scales.resize(static_cast<std::size_t>(experts) * rows * (cols / kGroup));
  std::normal_distribution<float> source_weight(0.0F, 1.7F);
  for (std::uint32_t e = 0; e < experts; ++e) {
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t group = 0; group < cols / kGroup; ++group) {
        // The expert id participates so no two experts share a scale
        // pattern -- otherwise a wrong-expert read could go unnoticed.
        const E8M0 scale_code =
            static_cast<E8M0>(126u + ((row + group + e) % 3u));
        bank.scales[static_cast<std::size_t>(e) * bank.ScaleStride() +
                    static_cast<std::size_t>(row) * (cols / kGroup) + group] =
            scale_code;
        const float scale =
            std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
        for (std::uint32_t col = group * kGroup; col < (group + 1u) * kGroup;
             col += 2u) {
          const std::uint8_t low = EncodeFp4(source_weight(rng) / scale);
          const std::uint8_t high = EncodeFp4(source_weight(rng) / scale);
          bank.packed[static_cast<std::size_t>(e) * bank.PackedStride() +
                      static_cast<std::size_t>(row) * (cols / 2u) + col / 2u] =
              static_cast<Mxfp4Pair>(low | (high << 4u));
        }
      }
    }
  }
  return bank;
}

struct Diff {
  float max_abs = 0.0F;
  bool within_tolerance = true;
};

Diff Compare(const std::vector<float>& actual,
             const std::vector<float>& expected) {
  Require(actual.size() == expected.size(), "comparison size mismatch");
  Diff diff;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      diff.max_abs = std::numeric_limits<float>::infinity();
      diff.within_tolerance = false;
      continue;
    }
    const float delta = std::fabs(actual[i] - expected[i]);
    diff.max_abs = std::max(diff.max_abs, delta);
    if (delta > kAbsTolerance + kRelTolerance * std::fabs(expected[i])) {
      diff.within_tolerance = false;
    }
  }
  return diff;
}

bool NonDegenerate(const std::vector<float>& values) {
  if (values.empty()) return false;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (const float value : values) {
    if (!std::isfinite(value)) return false;
    sum += value;
    sum_sq += static_cast<double>(value) * value;
  }
  const double mean = sum / static_cast<double>(values.size());
  const double mean_sq = sum_sq / static_cast<double>(values.size());
  const double variance = std::max(0.0, mean_sq - mean * mean);
  return std::sqrt(mean_sq) > 1.0e-6 && std::sqrt(variance) > 1.0e-6;
}

std::string LaunchFailure(const char* name, moe::kernels::arch35::Status st) {
  const auto& failure = moe::kernels::arch35::LastLaunchFailure();
  const char* call =
      failure.call != nullptr ? failure.call : "<no recorded ACL call>";
  return std::string(name) + " returned " +
         moe::kernels::arch35::ToString(st) + "; last ACL " + call +
         " status " + std::to_string(failure.acl_error) + "; " +
         (aclGetRecentErrMsg() != nullptr ? aclGetRecentErrMsg()
                                          : "<no recent ACL error>");
}

}  // namespace

#endif  // TESTS_SIM_DSV4_MOE_BLOCK_SIM_COMMON_HPP
