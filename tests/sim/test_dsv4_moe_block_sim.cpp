// dav-3510 CAModel execution test for the two whole-block entry points.
//
//   dsv4_moe_decode_topk      one token, top_k routed experts, atomic combine
//   dsv4_moe_prefill_batched  several tokens, one expert each
//
// Both run the REAL kernels through the REAL Launch* wrappers against the
// CPU golden in src/emu. References are generated in memory from a printed
// seed, never from a disk fixture (CLAUDE.md §4.5).
//
// GEOMETRY IS CHOSEN FOR SIMULATOR COST, and the reason is worth recording.
// These kernels are monolithic: one core runs a whole expert for a whole
// token. CAModel wall time scales with TOTAL simulated work (measured on the
// single-expert suite: 786 K MACs -> ~260 s, 1.57 M -> ~540 s), and total
// work here is top_k (or token count) times a full expert. So the widths are
// the smallest the host gate admits -- 512, the reduction quantum -- and the
// slot and token counts are small. The LANE geometry at 512 is degenerate
// (groups_per_lane = 1, where the coalesced and the old contiguous-block maps
// coincide); that is fine HERE because this suite is testing expert-stride
// addressing, routing and the UB-resident `activated`, not the lane map,
// which Dsv4MoeExpertSim covers at 1024 and tests/emu covers on the host.

#include "dsv4_moe_block_tiling.h"
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

void RunTest() {
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);

  const ExpertBank w1 = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w3 = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w2 = MakeBank(kHidden, kInter, kExperts, rng);

  std::vector<moe::emu::Mxfp4Matrix> v1(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v3(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v2(kExperts);
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    v1[e] = w1.View(e);
    v3[e] = w3.View(e);
    v2[e] = w2.View(e);
  }

  // --- goldens -----------------------------------------------------------
  // Routed slots are deliberately neither sorted nor the identity, so a
  // kernel that ignored topk_expert_ids and used the slot index would fail.
  const std::vector<std::int32_t> topk_ids = {3, 1};
  const std::vector<float> routing_weights = {0.625F, 0.375F};
  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);

  std::vector<float> scratch;
  std::vector<float> expected_decode(kHidden);
  moe::emu::moe_decode_topk_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x.data(), topk_ids.data(),
      routing_weights.data(), static_cast<int>(kTopK),
      static_cast<int>(kExperts), expected_decode.data(), kLimit, &scratch);
  Require(NonDegenerate(expected_decode),
          "decode golden is zero, constant, or non-finite");

  // Both tokens routed to DIFFERENT experts, so a kernel that applied one
  // expert to every token fails.
  const std::vector<std::int32_t> token_to_expert = {2, 0};
  std::vector<float> x_tokens(static_cast<std::size_t>(kTokens) * kHidden);
  for (float& value : x_tokens) value = x_distribution(rng);
  std::vector<float> expected_prefill(static_cast<std::size_t>(kTokens) * kHidden);
  moe::emu::moe_prefill_batched_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x_tokens.data(), token_to_expert.data(),
      static_cast<int>(kTokens), static_cast<int>(kExperts),
      expected_prefill.data(), kLimit, &scratch);
  Require(NonDegenerate(expected_prefill),
          "prefill golden is zero, constant, or non-finite");

  // --- device ------------------------------------------------------------
  std::vector<float> actual_decode(kHidden);
  std::vector<float> actual_prefill(static_cast<std::size_t>(kTokens) * kHidden);
  {
    Runtime runtime;
    {
      DeviceBuffer d_w1p(w1.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w1s(w1.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w3p(w3.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w3s(w3.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w2p(w2.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w2s(w2.scales.size() * sizeof(E8M0));
      DeviceBuffer d_x(x.size() * sizeof(float));
      DeviceBuffer d_ids(topk_ids.size() * sizeof(std::int32_t));
      DeviceBuffer d_rw(routing_weights.size() * sizeof(float));
      DeviceBuffer d_y(static_cast<std::size_t>(kHidden) * sizeof(float));
      DeviceBuffer d_xt(x_tokens.size() * sizeof(float));
      DeviceBuffer d_t2e(token_to_expert.size() * sizeof(std::int32_t));
      DeviceBuffer d_yt(actual_prefill.size() * sizeof(float));

      CopyToDevice(d_w1p, w1.packed);
      CopyToDevice(d_w1s, w1.scales);
      CopyToDevice(d_w3p, w3.packed);
      CopyToDevice(d_w3s, w3.scales);
      CopyToDevice(d_w2p, w2.packed);
      CopyToDevice(d_w2s, w2.scales);
      CopyToDevice(d_x, x);
      CopyToDevice(d_ids, topk_ids);
      CopyToDevice(d_rw, routing_weights);
      CopyToDevice(d_xt, x_tokens);
      CopyToDevice(d_t2e, token_to_expert);

      // DECODE ACCUMULATES: the caller zeroes y, per the wrapper contract.
      d_y.Zero();

      const auto decode_st = moe::kernels::arch35::LaunchDsv4MoeDecodeTopk(
          static_cast<const float*>(d_x.get()),
          static_cast<const Mxfp4Pair*>(d_w1p.get()),
          static_cast<const E8M0*>(d_w1s.get()),
          static_cast<const Mxfp4Pair*>(d_w3p.get()),
          static_cast<const E8M0*>(d_w3s.get()),
          static_cast<const Mxfp4Pair*>(d_w2p.get()),
          static_cast<const E8M0*>(d_w2s.get()),
          static_cast<const std::int32_t*>(d_ids.get()),
          static_cast<const float*>(d_rw.get()), kHidden, kInter, kExperts,
          kTopK, static_cast<float*>(d_y.get()), runtime.stream());
      if (decode_st != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoeDecodeTopk", decode_st));
      }
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(decode)");
      CopyFromDevice(actual_decode, d_y);

      const auto prefill_st = moe::kernels::arch35::LaunchDsv4MoePrefillBatched(
          static_cast<const float*>(d_xt.get()),
          static_cast<const Mxfp4Pair*>(d_w1p.get()),
          static_cast<const E8M0*>(d_w1s.get()),
          static_cast<const Mxfp4Pair*>(d_w3p.get()),
          static_cast<const E8M0*>(d_w3s.get()),
          static_cast<const Mxfp4Pair*>(d_w2p.get()),
          static_cast<const E8M0*>(d_w2s.get()),
          static_cast<const std::int32_t*>(d_t2e.get()), kHidden, kInter,
          kExperts, kTokens, static_cast<float*>(d_yt.get()),
          runtime.stream());
      if (prefill_st != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoePrefillBatched", prefill_st));
      }
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(prefill)");
      CopyFromDevice(actual_prefill, d_yt);
    }
    runtime.Close();
  }

  const Diff decode_diff = Compare(actual_decode, expected_decode);
  const Diff prefill_diff = Compare(actual_prefill, expected_prefill);
  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf(
      "Executed dsv4_moe_decode_topk (block_dim=%u, slots={3,1}) and "
      "dsv4_moe_prefill_batched (tokens=%u, experts={2,0}) via AscendCL: "
      "hidden=%u inter=%u num_experts=%u\n",
      kTopK, kTokens, kHidden, kInter, kExperts);
  std::printf(
      "Decode max abs diff: %.8g; prefill max abs diff: %.8g "
      "(abs/rel tolerance %.2g/%.2g)\n",
      decode_diff.max_abs, prefill_diff.max_abs, kAbsTolerance, kRelTolerance);
  Require(decode_diff.within_tolerance,
          "device decode differs from CPU golden beyond tolerance");
  Require(prefill_diff.within_tolerance,
          "device prefill differs from CPU golden beyond tolerance");
}
}  // namespace

int main() {
  try {
    RunTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] Dsv4MoeBlockCamodel: %s\n", error.what());
    return 1;
  }
  std::puts("[PASS] Dsv4MoeBlockCamodel");
  return 0;
}
