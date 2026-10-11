/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// A self-contained check harness for the host emulation suite.
//
// Deliberately NOT tests/common/test_utils.hpp: that header pulls in the
// mhc_fused_sinkhorn_gemv cpu_ref target, and the whole point of
// ENABLE_HOST_EMU is a tree that configures and builds with nothing but a
// host compiler. This file has no dependency outside the standard library.
//
// Every generator seeds a fixed mt19937, never std::random_device: a flaky
// numerical test that cannot be reproduced from its failure message is worse
// than no test.

#ifndef TESTS_EMU_EMU_TEST_SUPPORT_HPP
#define TESTS_EMU_EMU_TEST_SUPPORT_HPP

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace emu_test {

class Runner {
 public:
  explicit Runner(std::string suite) : suite_(std::move(suite)) {
    std::printf("== %s ==\n", suite_.c_str());
  }

  void Section(const std::string& name) {
    section_ = name;
    std::printf("-- %s\n", name.c_str());
  }

  void Check(bool condition, const std::string& what) {
    ++checks_;
    if (condition) {
      return;
    }
    ++failures_;
    std::printf("   FAIL [%s] %s\n", section_.c_str(), what.c_str());
  }

  void CheckNear(double actual, double expected, double tolerance,
                 const std::string& what) {
    ++checks_;
    const double error = std::fabs(actual - expected);
    if (std::isfinite(error) && error <= tolerance) {
      return;
    }
    ++failures_;
    std::printf("   FAIL [%s] %s: got %.10g, expected %.10g (|err| %.3g > %.3g)\n",
                section_.c_str(), what.c_str(), actual, expected, error, tolerance);
  }

  // Relative closeness, for quantities whose scale is not known up front.
  void CheckRelative(double actual, double expected, double tolerance,
                     const std::string& what) {
    const double scale = std::fmax(1.0, std::fabs(expected));
    CheckNear(actual / scale, expected / scale, tolerance, what);
  }

  // Reports how many checks ran, so a test that silently stopped generating
  // cases cannot pass by vacuity.
  int Summary() const {
    if (failures_ == 0) {
      std::printf("== %s: PASS (%d checks) ==\n", suite_.c_str(), checks_);
      return 0;
    }
    std::printf("== %s: FAIL (%d of %d checks failed) ==\n", suite_.c_str(), failures_,
                checks_);
    return 1;
  }

  int checks() const { return checks_; }
  int failures() const { return failures_; }

 private:
  std::string suite_;
  std::string section_ = "init";
  int checks_ = 0;
  int failures_ = 0;
};

// A fixed-seed generator. The seed is part of the test, so a failure is
// always reproducible by re-running the binary.
inline std::mt19937& Rng(std::uint32_t seed = 0x5EEDu) {
  static std::mt19937 engine(seed);
  return engine;
}

inline float UniformFloat(float low, float high) {
  std::uniform_real_distribution<float> distribution(low, high);
  return distribution(Rng());
}

inline std::vector<float> UniformVector(std::size_t count, float low, float high) {
  std::vector<float> out(count);
  for (float& value : out) {
    value = UniformFloat(low, high);
  }
  return out;
}

}  // namespace emu_test

#endif  // TESTS_EMU_EMU_TEST_SUPPORT_HPP
