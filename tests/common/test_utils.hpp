#ifndef TESTS_COMMON_TEST_UTILS_HPP
#define TESTS_COMMON_TEST_UTILS_HPP

#include "mhc_fused_sinkhorn_gemv_ref.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace test_common {

using moe::ops::mhc_fused_sinkhorn_gemv::Inputs;

void Require(bool condition, const std::string& message);
void RequireNear(float actual, float expected, float tolerance,
                 const std::string& label);
void RequireBirkhoff(
    const std::array<float, moe::ops::mhc_fused_sinkhorn_gemv::kResidualElements>& matrix,
    float tolerance);
Inputs MakeUniformSinkhornCase();

}  // namespace test_common

#endif  // TESTS_COMMON_TEST_UTILS_HPP
