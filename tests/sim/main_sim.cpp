#include "test_patterns.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Flat launch stub: it mirrors the conversion kernel contract on the CPU so
// the generator/layout/reference path can be exercised with no device runtime.
// CAModel builds can replace this symbol with their generated launch adapter.
void LaunchSimulatorStub(const float* src, std::size_t count, float* dst_tokens,
                         std::uint8_t* dst_weights) {
  for (std::size_t i = 0; i < count; ++i) dst_tokens[i] = src[i];
  dsv4::sim::EncodeMxfp4E2M1(src, count, dst_weights);
}

int Fail(const char* what) {
  std::fprintf(stderr, "sim_moe_test: %s\n", what);
  return EXIT_FAILURE;
}

}  // namespace

int main() {
  constexpr std::size_t rows = 4, cols = 8, count = rows * cols;
  std::vector<float> input(count), tokens(count), identity(rows * cols);
  std::vector<std::uint8_t> packed((count + 1) / 2);
  dsv4::sim::GenerateSequential(input.data(), count, 1.0F, 1.0F);
  dsv4::sim::GenerateIdentityMatrix(identity.data(), rows, cols);
  LaunchSimulatorStub(input.data(), count, tokens.data(), packed.data());

  // Inline reference: copied BF16-source values and packed nearest E2M1
  // nibbles; compare every logical element, including odd/even nibble lanes.
  for (std::size_t i = 0; i < count; ++i) {
    if (tokens[i] != input[i]) return Fail("token copy differs from reference");
    const std::uint8_t got = (packed[i / 2] >> ((i & 1U) * 4U)) & 0x0fU;
    if (got != dsv4::sim::EncodeMxfp4E2M1(input[i]))
      return Fail("packed MXFP4 differs from reference");
  }
  if (identity[0] != 1.0F || identity[1] != 0.0F || identity[cols + 1] != 1.0F)
    return Fail("identity generator failed");
  if (dsv4::sim::EncodeE8M0Scale(0.25F) != 125U)
    return Fail("E8M0 scale encoder failed");
  std::puts("sim_moe_test: PASS (CPU launch stub; no hardware required)");
  return EXIT_SUCCESS;
}
