#ifndef DSV4_TEST_PATTERNS_H
#define DSV4_TEST_PATTERNS_H

#include <cstddef>
#include <cstdint>

namespace dsv4::sim {
void GenerateSequential(uint16_t* ptr, std::size_t count,
                        float start_val = 1.0F, float step = 1.0F);
void GenerateAllOnes(uint16_t* ptr, std::size_t count);
void GenerateIdentityMatrix(uint16_t* ptr, std::size_t rows,
                            std::size_t cols);
void PackWeightsFp4(const float* src, uint8_t* dst_packed,
                    uint8_t* dst_scales, std::size_t k, std::size_t n,
                    std::size_t group_size = 32);
float Bf16ToFloat(uint16_t value);
float DecodePackedWeight(const uint8_t* packed, const uint8_t* scales,
                         std::size_t element_index,
                         std::size_t group_size = 32);
}  // namespace dsv4::sim

#endif
