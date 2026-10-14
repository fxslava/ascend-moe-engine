#include "dsv4_moe_expert_tiling.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/kernels/arch35/launch.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// hidden is 1024, NOT 512, and that is load-bearing rather than arbitrary.
//
// The kernel's group-to-lane map is  g = step * 16 + lane  with
// groups_per_lane = (cols/32)/16. At cols = 512 groups_per_lane is 1, so the
// map degenerates to g = lane -- and the previous contiguous-block map,
// g = lane * groups_per_lane + step, degenerates to exactly the same thing.
// A 512-wide reduction therefore CANNOT distinguish the two geometries: run
// both against this suite at 512 and the max-abs diffs agree bit for bit,
// which looks like a pass and is really a blind spot.
//
// 1024 is the smallest admissible width (the quantum is 512) with
// groups_per_lane = 2, where the maps genuinely differ, so the gate/up legs
// -- which reduce over `hidden` and carry the fused W1/W3 sweep, the most
// heavily rewritten path -- actually exercise the interleaving on device.
//
// `inter` stays 512 to hold the CAModel wall time near 2x rather than 4x;
// the down leg's indexing is the same GroupDot walk, and the host simulation
// in tests/emu covers both legs at groups_per_lane = 2 for free.
constexpr std::uint32_t kHidden = 1024;
constexpr std::uint32_t kInter = 512;
constexpr std::uint32_t kGroup = 32;
constexpr float kLimit = 10.0F;
constexpr std::uint64_t kSeed = 0xD5A40001ULL;
constexpr float kAbsTolerance = 1.0e-2F;
constexpr float kRelTolerance = 1.0e-2F;

using moe::kernels::arch35::E8M0;
using moe::kernels::arch35::Mxfp4Pair;

struct Matrix {
  std::uint32_t rows;
  std::uint32_t cols;
  std::vector<Mxfp4Pair> packed;
  std::vector<E8M0> scales;

  moe::emu::Mxfp4Matrix View() const {
    return {packed.data(), scales.data(), static_cast<int>(rows),
            static_cast<int>(cols)};
  }
};

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
        std::fprintf(stderr, "[CLEANUP] %s\n", AclError("aclrtFree", status).c_str());
      }
    }
  }

  void* get() const noexcept { return pointer_; }
  std::size_t bytes() const noexcept { return bytes_; }

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

std::uint8_t EncodeFp4(float value) {
  std::uint8_t best = 0;
  float error = std::numeric_limits<float>::infinity();
  for (std::uint8_t code = 0; code < 16; ++code) {
    const float candidate =
        std::fabs(value - moe::emu::DecodeFp4E2M1(code));
    if (candidate < error) {
      best = code;
      error = candidate;
    }
  }
  return best;
}

Matrix MakeMatrix(std::uint32_t rows, std::uint32_t cols,
                  std::mt19937_64& rng) {
  Require(cols % kGroup == 0, "MXFP4 width must be group aligned");
  Matrix matrix{rows, cols,
                std::vector<Mxfp4Pair>(static_cast<std::size_t>(rows) *
                                       (cols / 2u)),
                std::vector<E8M0>(static_cast<std::size_t>(rows) *
                                  (cols / kGroup))};
  std::normal_distribution<float> source_weight(0.0F, 1.7F);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t group = 0; group < cols / kGroup; ++group) {
      const E8M0 scale_code =
          static_cast<E8M0>(126u + ((row + group) % 3u));
      matrix.scales[static_cast<std::size_t>(row) * (cols / kGroup) + group] =
          scale_code;
      const float scale =
          std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
      for (std::uint32_t col = group * kGroup;
           col < (group + 1u) * kGroup; col += 2u) {
        const std::uint8_t low = EncodeFp4(source_weight(rng) / scale);
        const std::uint8_t high = EncodeFp4(source_weight(rng) / scale);
        matrix.packed[static_cast<std::size_t>(row) * (cols / 2u) + col / 2u] =
            static_cast<Mxfp4Pair>(low | (high << 4u));
      }
    }
  }
  return matrix;
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

void RunTest() {
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);
  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);
  const Matrix w1 = MakeMatrix(kInter, kHidden, rng);
  const Matrix w3 = MakeMatrix(kInter, kHidden, rng);
  const Matrix w2 = MakeMatrix(kHidden, kInter, rng);

  std::vector<float> expected_y(kHidden);
  std::vector<float> scratch;
  moe::emu::moe_expert_swiglu_mxfp4_cpu(
      w1.View(), w3.View(), w2.View(), x.data(), expected_y.data(), kLimit,
      &scratch);
  Require(scratch.size() >= kInter, "CPU golden omitted activation data");
  std::vector<float> expected_activated(scratch.begin(),
                                        scratch.begin() + kInter);
  Require(NonDegenerate(expected_activated),
          "activation golden is zero, constant, or non-finite");
  Require(NonDegenerate(expected_y),
          "output golden is zero, constant, or non-finite");

  moe::kernels::arch35::Dsv4MoeExpertTilingData tiling{};
  const auto tiling_status = moe::kernels::arch35::MakeDsv4MoeExpertTiling(
      kHidden, kInter, kHidden / 2u, kHidden / kGroup, kInter / 2u,
      kInter / kGroup, &tiling);
  Require(tiling_status == moe::kernels::arch35::Status::success,
          std::string("scaled tiling rejected: ") +
              moe::kernels::arch35::ToString(tiling_status));

  std::vector<float> actual_activated(kInter);
  std::vector<float> actual_y(kHidden);
  {
    Runtime runtime;
    {
      DeviceBuffer d_w1_packed(w1.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w1_scale(w1.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w3_packed(w3.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w3_scale(w3.scales.size() * sizeof(E8M0));
      DeviceBuffer d_x(x.size() * sizeof(float));
      DeviceBuffer d_activated(actual_activated.size() * sizeof(float));
      DeviceBuffer d_w2_packed(w2.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w2_scale(w2.scales.size() * sizeof(E8M0));
      DeviceBuffer d_y(actual_y.size() * sizeof(float));

      CopyToDevice(d_w1_packed, w1.packed);
      CopyToDevice(d_w1_scale, w1.scales);
      CopyToDevice(d_w3_packed, w3.packed);
      CopyToDevice(d_w3_scale, w3.scales);
      CopyToDevice(d_x, x);
      CopyToDevice(d_w2_packed, w2.packed);
      CopyToDevice(d_w2_scale, w2.scales);

      const auto launch_status = moe::kernels::arch35::LaunchDsv4MoeExpert(
          static_cast<const Mxfp4Pair*>(d_w1_packed.get()),
          static_cast<const E8M0*>(d_w1_scale.get()),
          static_cast<const Mxfp4Pair*>(d_w3_packed.get()),
          static_cast<const E8M0*>(d_w3_scale.get()),
          static_cast<const Mxfp4Pair*>(d_w2_packed.get()),
          static_cast<const E8M0*>(d_w2_scale.get()), kHidden / 2u,
          kHidden / kGroup, kInter / 2u, kInter / kGroup,
          static_cast<const float*>(d_x.get()), kHidden, kInter,
          static_cast<float*>(d_y.get()), d_activated.get(),
          moe::kernels::arch35::Dsv4MoeExpertWorkspaceBytes(kInter),
          runtime.stream());
      if (launch_status != moe::kernels::arch35::Status::success) {
        const auto& failure = moe::kernels::arch35::LastLaunchFailure();
        const char* failed_call =
            failure.call != nullptr ? failure.call : "<no recorded ACL call>";
        Fail(std::string("LaunchDsv4MoeExpert returned ") +
             moe::kernels::arch35::ToString(launch_status) + "; last ACL " +
             failed_call + " status " + std::to_string(failure.acl_error) +
             "; " + (aclGetRecentErrMsg() != nullptr ? aclGetRecentErrMsg()
                                                      : "<no recent ACL error>"));
      }
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream");
      CopyFromDevice(actual_activated, d_activated);
      CopyFromDevice(actual_y, d_y);
    }
    runtime.Close();
  }

  const Diff activation_diff = Compare(actual_activated, expected_activated);
  const Diff output_diff = Compare(actual_y, expected_y);
  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf("Executed dsv4_moe_expert_gate_up and "
              "dsv4_moe_expert_down via AscendCL: hidden=%u inter=%u "
              "block_dim=%u, gate/up tiles=%u, down tiles=%u\n",
              kHidden, kInter, tiling.block_dim,
              kInter / tiling.rows_per_block, kHidden / tiling.rows_per_block);
  std::printf("Activated max abs diff: %.8g; output max abs diff: %.8g "
              "(abs/rel tolerance %.2g/%.2g)\n",
              activation_diff.max_abs, output_diff.max_abs, kAbsTolerance,
              kRelTolerance);
  Require(activation_diff.within_tolerance,
          "device activation differs from CPU golden beyond tolerance");
  Require(output_diff.within_tolerance,
          "device output differs from CPU golden beyond tolerance");
}
}  // namespace

int main() {
  try {
    RunTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] Dsv4MoeExpertCamodel: %s\n", error.what());
    return 1;
  }
  std::puts("[PASS] Dsv4MoeExpertCamodel");
  return 0;
}
