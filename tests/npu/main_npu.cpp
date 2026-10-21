#include "dsv4_moe_prefill_cube_human_writed.h"
#include "../sim/test_patterns.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kXCount = 16ULL * 4096;
constexpr std::size_t kGateCount = 16ULL * 2048;
constexpr std::size_t kDownCount = 16ULL * 4096;
constexpr std::size_t kWeightCount = 4096ULL * 2048;
constexpr std::size_t kPackedBytes = kWeightCount / 2;
constexpr std::size_t kScaleCount = kWeightCount / 32;

bool Check(aclError error, const char* operation) {
  if (error == ACL_SUCCESS) return true;
  std::fprintf(stderr, "%s failed: ACL status %d\n", operation,
               static_cast<int>(error));
  return false;
}

template <class T>
bool DeviceAlloc(T** ptr, std::size_t count) {
  return Check(aclrtMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T),
                            ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc");
}

template <class T>
bool CopyToDevice(T* dst, const T* src, std::size_t count) {
  return Check(aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                            ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy H2D");
}

void CpuGolden(const std::vector<uint16_t>& x,
               const std::vector<uint8_t>& gate_packed,
               const std::vector<uint8_t>& gate_scales,
               const std::vector<uint8_t>& up_packed,
               const std::vector<uint8_t>& up_scales,
               const std::vector<uint8_t>& down_packed,
               const std::vector<uint8_t>& down_scales,
               std::vector<float>& y) {
  auto round_bf16 = [](float value) {
    union Bits { float f; uint32_t u; } bits{};
    bits.f = value;
    const uint32_t increment = 0x7fffU + ((bits.u >> 16) & 1U);
    return dsv4::sim::Bf16ToFloat(static_cast<uint16_t>((bits.u + increment) >> 16));
  };
  std::vector<float> gate_weights(kWeightCount);
  std::vector<float> up_weights(kWeightCount);
  std::vector<float> down_weights(kWeightCount);
  for (std::size_t i = 0; i < kWeightCount; ++i) {
    gate_weights[i] = dsv4::sim::DecodePackedWeight(
        gate_packed.data(), gate_scales.data(), i);
    up_weights[i] = dsv4::sim::DecodePackedWeight(
        up_packed.data(), up_scales.data(), i);
    down_weights[i] = dsv4::sim::DecodePackedWeight(
        down_packed.data(), down_scales.data(), i);
  }

  std::vector<float> gate(kGateCount), up(kGateCount), inter(kGateCount);
  for (std::size_t row = 0; row < 16; ++row) {
    for (std::size_t n = 0; n < 2048; ++n) {
      float gate_sum = 0.0F, up_sum = 0.0F;
      for (std::size_t k = 0; k < 4096; ++k) {
        const float xv = dsv4::sim::Bf16ToFloat(x[row * 4096 + k]);
        const std::size_t wi = k * 2048 + n;
        gate_sum = round_bf16(gate_sum + round_bf16(xv * round_bf16(gate_weights[wi])));
        up_sum = round_bf16(up_sum + round_bf16(xv * round_bf16(up_weights[wi])));
      }
      const std::size_t oi = row * 2048 + n;
      gate[oi] = round_bf16(gate_sum);
      up[oi] = round_bf16(up_sum);
      inter[oi] = round_bf16((gate[oi] / (1.0F + std::exp(-gate[oi]))) * up[oi]);
    }
  }
  for (std::size_t row = 0; row < 16; ++row) {
    for (std::size_t n = 0; n < 4096; ++n) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < 2048; ++k)
        sum = round_bf16(sum + round_bf16(inter[row * 2048 + k] * round_bf16(down_weights[k * 4096 + n])));
      y[row * 4096 + n] = sum;
    }
  }
}

void PrintAndValidate(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
  double square_sum = 0.0;
  float max_diff = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    square_sum += static_cast<double>(diff) * diff;
    max_diff = std::max(max_diff, diff);
  }
  const double rmse = std::sqrt(square_sum / actual.size());
  std::puts("index       NPU output        CPU golden");
  for (std::size_t i = 0; i < 8; ++i)
    std::printf("%5zu  %15.7f  %15.7f\n", i, actual[i], expected[i]);
  std::printf("RMSE %.7g, max abs %.7g\n", rmse, max_diff);
  std::puts(max_diff < 0.1F ? "TEST PASSED" : "TEST FAILED");
}

}  // namespace

int main(int argc, char** argv) {
  bool benchmark = false;
  if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--validate" &&
                                  std::string(argv[1]) != "--benchmark")) {
    std::fprintf(stderr, "usage: %s [--validate|--benchmark]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2) benchmark = std::string(argv[1]) == "--benchmark";

  std::vector<uint16_t> x(kXCount);
  std::vector<float> gate_source(kWeightCount, 1.0F);
  std::vector<float> up_source(kWeightCount, 1.0F);
  std::vector<float> down_source(kWeightCount, 1.0F);
  std::vector<uint8_t> gate_packed(kPackedBytes), gate_scales(kScaleCount);
  std::vector<uint8_t> up_packed(kPackedBytes), up_scales(kScaleCount);
  std::vector<uint8_t> down_packed(kPackedBytes), down_scales(kScaleCount);
  std::vector<float> y(kDownCount);
  std::vector<uint16_t> y_bf16(kDownCount);
  dsv4::sim::GenerateIdentityMatrix(x.data(), 16, 4096);
  dsv4::sim::PackWeightsFp4(gate_source.data(), gate_packed.data(),
                            gate_scales.data(), 4096, 2048);
  dsv4::sim::PackWeightsFp4(up_source.data(), up_packed.data(),
                            up_scales.data(), 4096, 2048);
  dsv4::sim::PackWeightsFp4(down_source.data(), down_packed.data(),
                            down_scales.data(), 2048, 4096);

  if (!Check(aclInit(nullptr), "aclInit") ||
      !Check(aclrtSetDevice(0), "aclrtSetDevice"))
    return EXIT_FAILURE;
  aclrtStream stream = nullptr;
  if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream"))
    return EXIT_FAILURE;

  uint16_t *dx = nullptr;
  uint8_t *dgp = nullptr, *dgs = nullptr, *dup = nullptr, *dus = nullptr;
  uint8_t *ddp = nullptr, *dds = nullptr;
  uint16_t *dgate = nullptr, *dup_out = nullptr, *dinter = nullptr, *dy = nullptr;
  if (!DeviceAlloc(&dx, kXCount) ||
      !DeviceAlloc(&dgp, kPackedBytes) || !DeviceAlloc(&dgs, kScaleCount) ||
      !DeviceAlloc(&dup, kPackedBytes) || !DeviceAlloc(&dus, kScaleCount) ||
      !DeviceAlloc(&ddp, kPackedBytes) || !DeviceAlloc(&dds, kScaleCount) ||
      !DeviceAlloc(&dgate, kGateCount) || !DeviceAlloc(&dup_out, kGateCount) ||
      !DeviceAlloc(&dinter, kGateCount) || !DeviceAlloc(&dy, kDownCount))
    return EXIT_FAILURE;

  if (!CopyToDevice(dx, x.data(), kXCount) ||
      !CopyToDevice(dgp, gate_packed.data(), kPackedBytes) ||
      !CopyToDevice(dgs, gate_scales.data(), kScaleCount) ||
      !CopyToDevice(dup, up_packed.data(), kPackedBytes) ||
      !CopyToDevice(dus, up_scales.data(), kScaleCount) ||
      !CopyToDevice(ddp, down_packed.data(), kPackedBytes) ||
      !CopyToDevice(dds, down_scales.data(), kScaleCount))
    return EXIT_FAILURE;

  auto launch = [&]() {
    moe::kernels::arch35::LaunchDsv4MoePrefillCubeHumanWrited(
        dx, dgp, dgs, dup, dus, ddp, dds, dgate, dup_out, dinter, dy, stream);
    return aclrtSynchronizeStream(stream) == ACL_SUCCESS;
  };
  if (benchmark) {
    constexpr int kWarmup = 5, kIterations = 100;
    for (int i = 0; i < kWarmup; ++i)
      if (!launch()) return EXIT_FAILURE;
    double total_us = 0.0;
    for (int i = 0; i < kIterations; ++i) {
      const auto start = std::chrono::steady_clock::now();
      if (!launch()) return EXIT_FAILURE;
      const auto stop = std::chrono::steady_clock::now();
      total_us += std::chrono::duration<double, std::micro>(stop - start).count();
    }
    std::printf("100 launches, average latency %.3f us\n", total_us / kIterations);
  } else {
    if (!launch() ||
        !Check(aclrtMemcpy(y_bf16.data(), kDownCount * sizeof(uint16_t), dy,
                           kDownCount * sizeof(uint16_t),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy D2H"))
      return EXIT_FAILURE;
    for (std::size_t i = 0; i < y.size(); ++i)
      y[i] = dsv4::sim::Bf16ToFloat(y_bf16[i]);
    std::vector<float> expected(kDownCount);
    CpuGolden(x, gate_packed, gate_scales, up_packed, up_scales, down_packed,
              down_scales, expected);
    PrintAndValidate(y, expected);
    float max_diff = 0.0F;
    for (std::size_t i = 0; i < y.size(); ++i)
      max_diff = std::max(max_diff, std::fabs(y[i] - expected[i]));
    if (max_diff >= 0.1F) return EXIT_FAILURE;
  }

  aclrtDestroyStream(stream);
  aclrtFree(dx); aclrtFree(dgp); aclrtFree(dgs); aclrtFree(dup); aclrtFree(dus);
  aclrtFree(ddp); aclrtFree(dds); aclrtFree(dgate); aclrtFree(dup_out);
  aclrtFree(dinter); aclrtFree(dy);
  aclrtResetDevice(0);
  aclFinalize();
  return EXIT_SUCCESS;
}
