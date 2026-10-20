#include "test_patterns.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dsv4::sim {
namespace {
constexpr float kE2M1[] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
}

void GenerateSequential(float* buffer, std::size_t count, float start_val,
                        float step) {
  for (std::size_t i = 0; i < count; ++i)
    buffer[i] = start_val + static_cast<float>(i) * step;
}

void GenerateAllOnes(float* buffer, std::size_t count) {
  std::fill_n(buffer, count, 1.0F);
}

void GenerateIdentityMatrix(float* buffer, std::size_t rows,
                            std::size_t cols) {
  std::fill_n(buffer, rows * cols, 0.0F);
  for (std::size_t i = 0; i < std::min(rows, cols); ++i)
    buffer[i * cols + i] = 1.0F;
}

std::uint8_t EncodeMxfp4E2M1(float value) {
  if (!std::isfinite(value)) throw std::invalid_argument("MXFP4 value must be finite");
  const bool negative = std::signbit(value);
  const float magnitude = std::fabs(value);
  unsigned code = 0;
  float distance = std::numeric_limits<float>::infinity();
  for (unsigned i = 0; i < 8; ++i) {
    const float candidate = std::fabs(magnitude - kE2M1[i]);
    if (candidate < distance) { distance = candidate; code = i; }
  }
  return static_cast<std::uint8_t>(code | (negative ? 0x8U : 0U));
}

void EncodeMxfp4E2M1(const float* input, std::size_t count,
                      std::uint8_t* packed) {
  std::fill_n(packed, (count + 1U) / 2U, 0U);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t nibble = EncodeMxfp4E2M1(input[i]);
    packed[i / 2U] |= static_cast<std::uint8_t>(nibble << ((i & 1U) * 4U));
  }
}

std::uint8_t EncodeE8M0Scale(float scale) {
  if (!(scale > 0.0F) || !std::isfinite(scale))
    throw std::invalid_argument("E8M0 scale must be finite and positive");
  const int exponent = static_cast<int>(std::nearbyint(std::log2(scale))) + 127;
  return static_cast<std::uint8_t>(std::clamp(exponent, 0, 254));
}

}  // namespace dsv4::sim
