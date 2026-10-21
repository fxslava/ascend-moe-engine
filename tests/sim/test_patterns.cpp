#include "test_patterns.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace dsv4::sim {
namespace {
uint16_t FloatToBf16(float value) {
  union Bits { float f; uint32_t u; } bits{};
  bits.f = value;
  const uint32_t round = 0x7fffU + ((bits.u >> 16) & 1U);
  return static_cast<uint16_t>((bits.u + round) >> 16);
}
uint8_t EncodeE2M1(float value) {
  static constexpr float levels[] = {0.0F, 0.5F, 1.0F, 1.5F,
                                     2.0F, 3.0F, 4.0F, 6.0F};
  const bool negative = std::signbit(value);
  const float magnitude = std::min(std::fabs(value), 6.0F);
  unsigned best = 0;
  float error = std::numeric_limits<float>::max();
  for (unsigned i = 0; i < 8; ++i) {
    const float candidate = std::fabs(magnitude - levels[i]);
    if (candidate < error) { error = candidate; best = i; }
  }
  return static_cast<uint8_t>(best | (negative ? 8U : 0U));
}
}  // namespace

void GenerateSequential(uint16_t* ptr, std::size_t count,
                        float start_val, float step) {
  for (std::size_t i = 0; i < count; ++i)
    ptr[i] = FloatToBf16(start_val + static_cast<float>(i) * step);
}

void GenerateAllOnes(uint16_t* ptr, std::size_t count) {
  std::fill_n(ptr, count, FloatToBf16(1.0F));
}

void GenerateIdentityMatrix(uint16_t* ptr, std::size_t rows,
                            std::size_t cols) {
  std::fill_n(ptr, rows * cols, FloatToBf16(0.0F));
  for (std::size_t i = 0; i < std::min(rows, cols); ++i)
    ptr[i * cols + i] = FloatToBf16(1.0F);
}

void PackWeightsFp4(const float* src, uint8_t* dst_packed,
                    uint8_t* dst_scales, std::size_t k, std::size_t n,
                    std::size_t group_size) {
  if (group_size == 0) throw std::invalid_argument("group_size must be positive");
  const std::size_t count = k * n;
  const std::size_t groups = (count + group_size - 1) / group_size;
  std::fill_n(dst_packed, (count + 1) / 2, 0U);
  for (std::size_t group = 0; group < groups; ++group) {
    const std::size_t begin = group * group_size;
    const std::size_t end = std::min(count, begin + group_size);
    float max_abs = 0.0F;
    for (std::size_t i = begin; i < end; ++i)
      max_abs = std::max(max_abs, std::fabs(src[i]));

    // E8M0 is an unsigned biased exponent. Choosing ceil(log2(max/6))
    // ensures the largest normalized E2M1 value stays representable.
    int exponent = max_abs == 0.0F
                       ? 0
                       : static_cast<int>(std::ceil(std::log2(max_abs / 6.0F)));
    exponent = std::clamp(exponent, -126, 127);
    const float scale = std::ldexp(1.0F, exponent);
    dst_scales[group] = static_cast<uint8_t>(exponent + 127);
    for (std::size_t i = begin; i < end; ++i) {
      const uint8_t code = EncodeE2M1(src[i] / scale);
      dst_packed[i / 2] |= static_cast<uint8_t>(code << ((i & 1U) * 4U));
    }
  }
}

float Bf16ToFloat(uint16_t value) {
  union Bits { uint32_t u; float f; } bits{};
  bits.u = static_cast<uint32_t>(value) << 16;
  return bits.f;
}

float DecodePackedWeight(const uint8_t* packed, const uint8_t* scales,
                         std::size_t element_index, std::size_t group_size) {
  const uint8_t byte = packed[element_index / 2];
  const uint8_t nibble = (element_index & 1U) == 0
                             ? static_cast<uint8_t>(byte & 0x0fU)
                             : static_cast<uint8_t>(byte >> 4);
  const float levels[] = {0.0F, 0.5F, 1.0F, 1.5F,
                          2.0F, 3.0F, 4.0F, 6.0F};
  const int exponent = static_cast<int>(scales[element_index / group_size]) - 127;
  return ((nibble & 8U) ? -1.0F : 1.0F) *
         levels[nibble & 7U] * std::ldexp(1.0F, exponent);
}
}  // namespace dsv4::sim
