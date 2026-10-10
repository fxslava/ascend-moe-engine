#ifndef MHC_FUSED_SINKHORN_GEMV_REF_HPP
#define MHC_FUSED_SINKHORN_GEMV_REF_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace moe::ops::mhc_fused_sinkhorn_gemv {

inline constexpr std::size_t kHeads = 4;
inline constexpr std::size_t kHiddenSize = 4096;
inline constexpr std::size_t kInputElements = kHeads * kHiddenSize;
inline constexpr std::size_t kProjectionRows = 24;
inline constexpr std::size_t kPhiElements = kProjectionRows * kInputElements;
inline constexpr std::size_t kResidualElements = kHeads * kHeads;
inline constexpr std::size_t kSinkhornIterations = 20;

struct Inputs {
  std::vector<std::uint16_t> x_bf16;
  std::vector<float> phi;
  std::array<float, 3> alpha{};
  std::array<float, kProjectionRows> bias{};
  std::vector<std::uint16_t> h_out_bf16;
};

struct Result {
  std::array<float, kHeads> h_in{};
  std::array<float, kHeads> h_post{};
  std::array<float, kResidualElements> residual_matrix{};
  std::vector<std::uint16_t> y_bf16;
};

class MhcFusedSinkhornGemvReference final {
 public:
  static void Validate(const Inputs& inputs);
  static Result Compute(const Inputs& inputs,
                        std::size_t sinkhorn_iterations = kSinkhornIterations);
};

std::uint16_t FloatToBf16(float value) noexcept;
float Bf16ToFloat(std::uint16_t value) noexcept;

}  // namespace moe::ops::mhc_fused_sinkhorn_gemv

#endif  // MHC_FUSED_SINKHORN_GEMV_REF_HPP
