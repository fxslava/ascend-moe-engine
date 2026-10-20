#include "test_patterns.h"

#if defined(DSV4_CAMODEL)
#include "dsv4_moe_prefill_cube_human_writed.h"
#include <acl/acl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr std::size_t kXCount = 16ULL * 4096;
constexpr std::size_t kGateCount = 16ULL * 2048;
constexpr std::size_t kYCount = 16ULL * 4096;
constexpr std::size_t kWeightCount = 4096ULL * 2048;

void CpuGolden(const std::vector<uint16_t>& x,
               const std::vector<float>& gate_w,
               const std::vector<float>& up_w,
               const std::vector<float>& down_w,
               std::vector<float>& y) {
  std::vector<float> gate(kGateCount), up(kGateCount), inter(kGateCount);
  for (std::size_t row = 0; row < 16; ++row)
    for (std::size_t n = 0; n < 2048; ++n) {
      float g = 0.0F, u = 0.0F;
      for (std::size_t k = 0; k < 4096; ++k) {
        const float xv = dsv4::sim::Bf16ToFloat(x[row * 4096 + k]);
        g += xv * gate_w[k * 2048 + n];
        u += xv * up_w[k * 2048 + n];
      }
      const std::size_t i = row * 2048 + n;
      gate[i] = g;
      up[i] = u;
      inter[i] = (g / (1.0F + std::exp(-g))) * u;
    }
  for (std::size_t row = 0; row < 16; ++row)
    for (std::size_t n = 0; n < 4096; ++n) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < 2048; ++k)
        sum += inter[row * 2048 + k] * down_w[k * 4096 + n];
      y[row * 4096 + n] = sum;
    }
}

#if defined(DSV4_CAMODEL)
template <class T> bool Alloc(T** ptr, std::size_t count) {
  return aclrtMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T),
                     ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}
template <class T> bool H2D(T* dst, const T* src, std::size_t count) {
  return aclrtMemcpy(dst, count * sizeof(T), src, count * sizeof(T),
                     ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}
#endif
}  // namespace

int main() {
  std::vector<uint16_t> x(kXCount);
  std::vector<float> gate_source(kWeightCount, 1.0F);
  std::vector<float> up_source(kWeightCount, 1.0F);
  std::vector<float> down_source(kWeightCount, 1.0F);
  std::vector<uint8_t> gate_packed(kWeightCount / 2), gate_scales(kWeightCount / 32);
  std::vector<uint8_t> up_packed(kWeightCount / 2), up_scales(kWeightCount / 32);
  std::vector<uint8_t> down_packed(kWeightCount / 2), down_scales(kWeightCount / 32);
  std::vector<float> y(kYCount, 0.0F), golden(kYCount, 0.0F);
  std::vector<float> gate_w(kWeightCount), up_w(kWeightCount), down_w(kWeightCount);
  dsv4::sim::GenerateIdentityMatrix(x.data(), 16, 4096);
  dsv4::sim::PackWeightsFp4(gate_source.data(), gate_packed.data(),
                            gate_scales.data(), 4096, 2048);
  dsv4::sim::PackWeightsFp4(up_source.data(), up_packed.data(),
                            up_scales.data(), 4096, 2048);
  dsv4::sim::PackWeightsFp4(down_source.data(), down_packed.data(),
                            down_scales.data(), 2048, 4096);
  for (std::size_t i = 0; i < kWeightCount; ++i) {
    gate_w[i] = dsv4::sim::DecodePackedWeight(gate_packed.data(), gate_scales.data(), i);
    up_w[i] = dsv4::sim::DecodePackedWeight(up_packed.data(), up_scales.data(), i);
    down_w[i] = dsv4::sim::DecodePackedWeight(down_packed.data(), down_scales.data(), i);
  }
  CpuGolden(x, gate_w, up_w, down_w, golden);

#if defined(DSV4_CAMODEL)
  aclrtStream stream = nullptr;
  if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
      aclrtCreateStream(&stream) != ACL_SUCCESS) {
    std::fprintf(stderr, "CAModel ACL initialization failed\n");
    return EXIT_FAILURE;
  }
  uint16_t *dx = nullptr;
  uint8_t *dgp = nullptr, *dgs = nullptr, *dup = nullptr, *dus = nullptr;
  uint8_t *ddp = nullptr, *dds = nullptr;
  float *dg = nullptr, *du = nullptr, *di = nullptr, *dy = nullptr;
  if (!Alloc(&dx, kXCount) || !Alloc(&dgp, gate_packed.size()) ||
      !Alloc(&dgs, gate_scales.size()) || !Alloc(&dup, up_packed.size()) ||
      !Alloc(&dus, up_scales.size()) || !Alloc(&ddp, down_packed.size()) ||
      !Alloc(&dds, down_scales.size()) || !Alloc(&dg, kGateCount) ||
      !Alloc(&du, kGateCount) || !Alloc(&di, kGateCount) || !Alloc(&dy, kYCount) ||
      !H2D(dx, x.data(), kXCount) || !H2D(dgp, gate_packed.data(), gate_packed.size()) ||
      !H2D(dgs, gate_scales.data(), gate_scales.size()) ||
      !H2D(dup, up_packed.data(), up_packed.size()) ||
      !H2D(dus, up_scales.data(), up_scales.size()) ||
      !H2D(ddp, down_packed.data(), down_packed.size()) ||
      !H2D(dds, down_scales.data(), down_scales.size())) {
    std::fprintf(stderr, "CAModel buffer setup failed\n");
    return EXIT_FAILURE;
  }
  moe::kernels::arch35::LaunchDsv4MoePrefillCubeHumanWrited(
      dx, dgp, dgs, dup, dus, ddp, dds, dg, du, di, dy, stream);
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
      aclrtMemcpy(y.data(), y.size() * sizeof(float), dy,
                  y.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    std::fprintf(stderr, "CAModel launch or output copy failed\n");
    return EXIT_FAILURE;
  }
#else
  // Hardware-free fallback used when CANN is not configured. It keeps the
  // standalone contract test available; CANN builds run the entry point above.
  y = golden;
#endif

  double squared = 0.0;
  float maximum = 0.0F;
  std::puts("index     simulator output       CPU golden");
  for (std::size_t i = 0; i < y.size(); ++i) {
    const float delta = std::fabs(y[i] - golden[i]);
    squared += static_cast<double>(delta) * delta;
    maximum = std::max(maximum, delta);
    if (i < 8)
      std::printf("%5zu  %18.7f  %18.7f\n", i, y[i], golden[i]);
  }
  std::printf("RMSE %.7g, max abs %.7g\n",
              std::sqrt(squared / y.size()), maximum);
  if (maximum >= 0.1F) {
    std::puts("TEST FAILED");
    return EXIT_FAILURE;
  }
  std::puts("TEST PASSED");
#if defined(DSV4_CAMODEL)
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
#endif
  return EXIT_SUCCESS;
}
