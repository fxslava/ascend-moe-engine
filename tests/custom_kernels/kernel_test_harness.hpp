#pragma once
// -----------------------------------------------------------------------------
// Shared scaffolding for the two decoupled custom-kernel binaries:
//
//   test_custom_kernels_verification   -- numerical correctness vs the CPU golden
//   bench_custom_kernels_performance   -- latency / throughput / efficiency
//
// Everything here is host-portable C++17 over moe_emu. It deliberately does NOT
// touch any host CPU debug/interpretation runtime (no cpudebug, no tikicpulib,
// no libascendc_cpudebug.so): the references in src/emu are ordinary portable
// C++ compiled for the host, which is a different thing entirely from
// interpreting device SIMT code on x86.
//
// References are generated IN MEMORY, per run. Nothing is compared against a
// pre-generated disk dump, so a stale fixture cannot make a broken kernel pass.
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace moe::kerntest {

// ---------------------------------------------------------------------------
// Minimal assertion harness (this tree uses custom harnesses, not gtest)
// ---------------------------------------------------------------------------
struct Report {
  int checks = 0;
  int failures = 0;
  int skipped = 0;

  void pass(const std::string& what) {
    ++checks;
    std::printf("  [  OK  ] %s\n", what.c_str());
  }
  void fail(const std::string& what, const std::string& detail) {
    ++checks;
    ++failures;
    std::printf("  [ FAIL ] %s\n           %s\n", what.c_str(), detail.c_str());
  }
  void skip(const std::string& what, const std::string& why) {
    ++skipped;
    std::printf("  [ SKIP ] %s\n           %s\n", what.c_str(), why.c_str());
  }
  void check(bool ok, const std::string& what, const std::string& detail = {}) {
    ok ? pass(what) : fail(what, detail);
  }
  int exit_code() const { return failures == 0 ? 0 : 1; }
};

inline void section(const char* name) {
  std::printf("\n=== %s ===\n", name);
}

// ---------------------------------------------------------------------------
// Numerical metrics
//
// Thresholds are asserted by the caller; these only measure. All of them run
// the accumulation in double so that the metric itself is never the thing that
// loses precision when judging a 16k-element tensor.
// ---------------------------------------------------------------------------

// 1.0 for identical direction. Returns 1.0 when both inputs are all-zero
// (vacuously parallel) and 0.0 when exactly one is, which keeps a silently
// zeroed output from scoring as a pass.
inline double CosineSimilarity(const float* a, const float* b, std::size_t n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = a[i], y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na == 0.0 && nb == 0.0) return 1.0;
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ||a-b||_2 / ||b||_2, with b the reference. Zero reference and zero error is
// 0.0; zero reference and nonzero error is infinity rather than a divide-by-0.
inline double RelativeL2(const float* a, const float* b, std::size_t n) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (den == 0.0) return num == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  return std::sqrt(num / den);
}

inline double MaxAbsDiff(const float* a, const float* b, std::size_t n) {
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) -
                                      static_cast<double>(b[i])));
  }
  return worst;
}

inline bool AllFinite(const float* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Birkhoff / doubly-stochastic bounds
//
// A Sinkhorn solve must land on the Birkhoff polytope: every row sums to 1,
// every column sums to 1, every entry in [0,1]. `worst_row` / `worst_col` are
// the largest deviation of any row / column sum from 1.
// ---------------------------------------------------------------------------
struct BirkhoffResidual {
  double worst_row = 0.0;
  double worst_col = 0.0;
  double min_entry = 0.0;
  double max_entry = 0.0;
  bool finite = true;

  double worst() const { return std::max(worst_row, worst_col); }
};

inline BirkhoffResidual BirkhoffCheck(const float* m, int n) {
  BirkhoffResidual r;
  r.min_entry = std::numeric_limits<double>::infinity();
  r.max_entry = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n * n; ++i) {
    if (!std::isfinite(m[i])) r.finite = false;
    r.min_entry = std::min(r.min_entry, static_cast<double>(m[i]));
    r.max_entry = std::max(r.max_entry, static_cast<double>(m[i]));
  }
  for (int i = 0; i < n; ++i) {
    double row = 0.0, col = 0.0;
    for (int j = 0; j < n; ++j) {
      row += m[i * n + j];
      col += m[j * n + i];
    }
    r.worst_row = std::max(r.worst_row, std::fabs(row - 1.0));
    r.worst_col = std::max(r.worst_col, std::fabs(col - 1.0));
  }
  return r;
}

// ---------------------------------------------------------------------------
// BF16 helpers (the kernels store BF16; the references take void* BF16)
// ---------------------------------------------------------------------------
inline std::uint16_t Fp32ToBf16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, 4);
  // round-to-nearest-even
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

inline float Bf16ToFp32(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline std::vector<std::uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<std::uint16_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = Fp32ToBf16(v[i]);
  return out;
}

inline std::vector<float> FromBf16(const std::vector<std::uint16_t>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = Bf16ToFp32(v[i]);
  return out;
}

// ---------------------------------------------------------------------------
// Deterministic input generation
//
// Seeded per call site so a failure is reproducible from the printed seed, and
// so the verification and benchmark binaries can ask for the same geometry
// without sharing state.
// ---------------------------------------------------------------------------
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : gen_(seed), seed_(seed) {}

  std::uint64_t seed() const { return seed_; }

  std::vector<float> normal(std::size_t n, float mean = 0.0F, float sd = 1.0F) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(gen_);
    return v;
  }

  std::vector<float> uniform(std::size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(gen_);
    return v;
  }

  std::vector<std::uint8_t> bytes(std::size_t n, std::uint8_t lo = 0,
                                  std::uint8_t hi = 255) {
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<std::uint8_t> v(n);
    for (auto& x : v) x = static_cast<std::uint8_t>(d(gen_));
    return v;
  }

 private:
  std::mt19937_64 gen_;
  std::uint64_t seed_;
};

// ---------------------------------------------------------------------------
// Device availability
//
// Two independent gates, reported separately because they fail for different
// reasons and want different fixes:
//
//   * kDeviceLaunchLayerAvailable -- whether the host-side Launch* wrappers in
//     docs/kernel_contracts_spec.md are IMPLEMENTED. As of this writing they
//     are specified but absent from the tree, so no amount of hardware makes
//     an on-device comparison possible. Flip this on with
//     -DMOE_KERNEL_LAUNCH_LAYER_AVAILABLE once they land.
//
//   * physical NPU presence -- a runtime question, answered only in a CANN
//     build. tests/npu/npu_test_support.hpp owns that probe; this header does
//     not duplicate it, so the host-emu build stays free of any acl include.
// ---------------------------------------------------------------------------
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
inline constexpr bool kDeviceLaunchLayerAvailable = true;
#else
inline constexpr bool kDeviceLaunchLayerAvailable = false;
#endif

inline const char* DeviceUnavailableReason() {
  return "host-side Launch* wrappers for the five arch35 kernels are specified "
         "in docs/kernel_contracts_spec.md but not implemented in the tree, so "
         "kernel output cannot be produced for comparison. Build with "
         "-DMOE_KERNEL_LAUNCH_LAYER_AVAILABLE once they exist.";
}

}  // namespace moe::kerntest
