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

// Fixture loading and the acceptance comparator for the arch35 CPU-debug
// harnesses.
//
// This header is compiled with _GLIBCXX_USE_CXX11_ABI=0, which is what CANN's
// cpudebug interpreter requires, so it deliberately has NO dependency on
// libmoe_emu. Everything it needs is raw bytes written by
// apps/dsv4_extract_kernel_fixtures.
//
// ACCEPTANCE (docs/kernel_contracts_spec.md): cosine >= 0.9995 AND relative
// L2 <= 1e-4, plus a finiteness check, plus an explicit guard against the
// degenerate all-zero output -- cosine alone must never pass a kernel that
// wrote nothing.

#ifndef MOE_ARCH35_CPUDEBUG_FIXTURE_IO_H
#define MOE_ARCH35_CPUDEBUG_FIXTURE_IO_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace arch35_test {

constexpr double kCosineFloor = 0.9995;
constexpr double kRelL2Ceiling = 1e-4;

inline std::vector<unsigned char> ReadBytes(const std::string& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "FATAL: cannot open fixture '%s'\n", path.c_str());
    std::exit(2);
  }
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::vector<unsigned char> out(static_cast<size_t>(size));
  if (size > 0 && std::fread(out.data(), 1, out.size(), file) != out.size()) {
    std::fprintf(stderr, "FATAL: short read on '%s'\n", path.c_str());
    std::exit(2);
  }
  std::fclose(file);
  return out;
}

// Reads a fixture and checks it is exactly the declared element count, so a
// regenerated fixture of the wrong shape fails loudly instead of being padded.
template <typename T>
std::vector<T> ReadTyped(const std::string& dir, const std::string& name,
                         size_t expected_count) {
  const std::vector<unsigned char> raw = ReadBytes(dir + "/" + name + ".bin");
  if (raw.size() != expected_count * sizeof(T)) {
    std::fprintf(stderr,
                 "FATAL: fixture '%s/%s.bin' is %zu bytes, expected %zu\n",
                 dir.c_str(), name.c_str(), raw.size(), expected_count * sizeof(T));
    std::exit(2);
  }
  std::vector<T> out(expected_count);
  if (!out.empty()) {
    std::memcpy(out.data(), raw.data(), raw.size());
  }
  return out;
}

inline float Bf16ToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

struct Metrics {
  double cosine = 0.0;
  double relative_l2 = 0.0;
  double max_abs = 0.0;
  size_t count = 0;
  size_t nonfinite = 0;
  double got_norm = 0.0;
  double want_norm = 0.0;
  size_t exact_mismatches = 0;
};

inline Metrics Compare(const float* got, const float* want, size_t count) {
  Metrics m;
  m.count = count;
  double dot = 0.0;
  double got_sq = 0.0;
  double want_sq = 0.0;
  double diff_sq = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double g = static_cast<double>(got[i]);
    const double w = static_cast<double>(want[i]);
    if (!std::isfinite(g)) {
      ++m.nonfinite;
    }
    if (g != w) {
      ++m.exact_mismatches;
    }
    dot += g * w;
    got_sq += g * g;
    want_sq += w * w;
    diff_sq += (g - w) * (g - w);
    const double abs_diff = std::fabs(g - w);
    if (abs_diff > m.max_abs) {
      m.max_abs = abs_diff;
    }
  }
  m.got_norm = std::sqrt(got_sq);
  m.want_norm = std::sqrt(want_sq);
  m.cosine = (m.got_norm > 0.0 && m.want_norm > 0.0)
                 ? dot / (m.got_norm * m.want_norm)
                 : 0.0;
  m.relative_l2 = (m.want_norm > 0.0) ? std::sqrt(diff_sq) / m.want_norm
                                      : std::sqrt(diff_sq);
  return m;
}

// Tracks every check in one harness so the process exit code is the verdict.
class Verdict {
 public:
  explicit Verdict(const char* kernel) : kernel_(kernel) {
    std::printf("=== %s (CANN cpudebug, Ascend950PR_9599) ===\n", kernel);
  }

  void CheckMetrics(const char* label, const Metrics& m) {
    const bool finite_ok = (m.nonfinite == 0);
    // A kernel that wrote nothing has zero norm; cosine is then meaningless.
    const bool nonzero_ok = (m.got_norm > 0.0) || (m.want_norm == 0.0);
    const bool cosine_ok = m.cosine >= kCosineFloor;
    const bool l2_ok = m.relative_l2 <= kRelL2Ceiling;
    const bool ok = finite_ok && nonzero_ok && cosine_ok && l2_ok;
    std::printf("  %-34s n=%-7zu cos=%.9f  relL2=%.3e  maxabs=%.3e  %s\n", label,
                m.count, m.cosine, m.relative_l2, m.max_abs, ok ? "PASS" : "FAIL");
    if (!ok) {
      if (!finite_ok) {
        std::printf("      %zu non-finite outputs\n", m.nonfinite);
      }
      if (!nonzero_ok) {
        std::printf("      output norm is zero against a non-zero reference\n");
      }
      if (!cosine_ok) {
        std::printf("      cosine %.9f below the %.4f floor\n", m.cosine, kCosineFloor);
      }
      if (!l2_ok) {
        std::printf("      relative L2 %.3e above the %.1e ceiling\n",
                    m.relative_l2, kRelL2Ceiling);
      }
      failed_ = true;
    }
    ++checks_;
  }

  // For structural invariants: exact equality, untouched bytes, index order.
  void CheckTrue(const char* label, bool condition, const char* detail = "") {
    std::printf("  %-34s %s%s%s\n", label, condition ? "PASS" : "FAIL",
                condition ? "" : "  <- ", condition ? "" : detail);
    if (!condition) {
      failed_ = true;
    }
    ++checks_;
  }

  int Finish() const {
    std::printf("  ---- %zu checks, %s ----\n", checks_,
                failed_ ? "FAILED" : "all passed");
    return failed_ ? 1 : 0;
  }

 private:
  const char* kernel_;
  size_t checks_ = 0;
  bool failed_ = false;
};

}  // namespace arch35_test

#endif  // MOE_ARCH35_CPUDEBUG_FIXTURE_IO_H
