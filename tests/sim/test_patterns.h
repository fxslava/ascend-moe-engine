#ifndef DSV4_TEST_PATTERNS_H
#define DSV4_TEST_PATTERNS_H

#include <cstddef>
#include <cstdint>

namespace dsv4::sim {

void GenerateSequential(float* buffer, std::size_t count, float start_val,
                        float step);
void GenerateAllOnes(float* buffer, std::size_t count);
void GenerateIdentityMatrix(float* buffer, std::size_t rows,
                            std::size_t cols);

// Encode a value using the nearest finite E2M1 code, packed low nibble first.
std::uint8_t EncodeMxfp4E2M1(float value);
void EncodeMxfp4E2M1(const float* input, std::size_t count,
                      std::uint8_t* packed);

// Encode a positive scale as the nearest E8M0 power of two (biased exponent).
std::uint8_t EncodeE8M0Scale(float scale);

}  // namespace dsv4::sim

#endif
