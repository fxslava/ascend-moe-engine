#ifndef TESTS_COMMON_TENSOR_FACTORY_HPP
#define TESTS_COMMON_TENSOR_FACTORY_HPP

#include "test_utils.hpp"

namespace test_common {

inline Inputs MakeDeterministicMhcFusedSinkhornGemvInputs() {
  return MakeUniformSinkhornCase();
}

}  // namespace test_common

#endif  // TESTS_COMMON_TENSOR_FACTORY_HPP
