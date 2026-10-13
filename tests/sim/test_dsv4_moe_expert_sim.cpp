// Host SIMT functional model for the arch35 expert kernel.
// It follows the device lane reduction and tile-stride mapping; it does not
// interpret device code through a host CPU debug runtime.
#include "dsv4_moe_expert_tiling.h"
#include "moe/emu/dsv4_layers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint32_t kHidden = 512;
constexpr std::uint32_t kInter = 512;
constexpr std::uint32_t kRowsPerTile = 32;
constexpr std::uint32_t kLanesPerRow = 16;
constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kBlockDim = 2;
constexpr float kLimit = 10.0F;
constexpr std::uint64_t kSeed = 0xD5A40001ULL;
constexpr float kAbsTolerance = 1.0e-2F;
constexpr float kRelTolerance = 1.0e-2F;

struct Matrix {
  std::uint32_t rows;
  std::uint32_t cols;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> scales;

  moe::emu::Mxfp4Matrix View() const {
    return {packed.data(), scales.data(), static_cast<int>(rows),
            static_cast<int>(cols)};
  }
};

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint8_t EncodeFp4(float value) {
  std::uint8_t best = 0;
  float error = std::numeric_limits<float>::infinity();
  for (std::uint8_t code = 0; code < 16; ++code) {
    const float candidate =
        std::fabs(value - moe::emu::DecodeFp4E2M1(code));
    if (candidate < error) {
      best = code;
      error = candidate;
    }
  }
  return best;
}

Matrix MakeMatrix(std::uint32_t rows, std::uint32_t cols,
                  std::mt19937_64& rng) {
  Require(cols % kGroup == 0, "MXFP4 width must be group aligned");
  Matrix m{rows, cols,
           std::vector<std::uint8_t>(static_cast<std::size_t>(rows) *
                                     (cols / 2u)),
           std::vector<std::uint8_t>(static_cast<std::size_t>(rows) *
                                     (cols / kGroup))};
  std::normal_distribution<float> source_weight(0.0F, 1.7F);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t group = 0; group < cols / kGroup; ++group) {
      const std::uint8_t scale_code =
          static_cast<std::uint8_t>(126u + ((row + group) % 3u));
      m.scales[static_cast<std::size_t>(row) * (cols / kGroup) + group] =
          scale_code;
      const float scale =
          std::ldexp(1.0F, static_cast<int>(scale_code) - 127);
      for (std::uint32_t col = group * kGroup;
           col < (group + 1u) * kGroup; col += 2u) {
        const std::uint8_t low = EncodeFp4(source_weight(rng) / scale);
        const std::uint8_t high = EncodeFp4(source_weight(rng) / scale);
        m.packed[static_cast<std::size_t>(row) * (cols / 2u) + col / 2u] =
            static_cast<std::uint8_t>(low | (high << 4u));
      }
    }
  }
  return m;
}

float LanePartial(const std::uint8_t* packed, const std::uint8_t* scales,
                  const float* x, std::uint32_t lane, std::uint32_t cols) {
  const std::uint32_t pairs_per_lane = (cols / 2u) / kLanesPerRow;
  const std::uint32_t scale_base = lane * pairs_per_lane / 16u;
  float total = 0.0F;
  for (std::uint32_t group = 0; group < pairs_per_lane / 16u; ++group) {
    float subtotal = 0.0F;
    const std::uint32_t pair_base = lane * pairs_per_lane + group * 16u;
    for (std::uint32_t pair_offset = 0; pair_offset < 16u; ++pair_offset) {
      const std::uint32_t pair = pair_base + pair_offset;
      const std::uint8_t byte = packed[pair];
      subtotal +=
          moe::emu::DecodeFp4E2M1(byte & 0x0Fu) * x[2u * pair] +
          moe::emu::DecodeFp4E2M1(byte >> 4u) * x[2u * pair + 1u];
    }
    const float scale =
        std::ldexp(1.0F, static_cast<int>(scales[scale_base + group]) - 127);
    total += subtotal * scale;
  }
  return total;
}

float SimtGemvRow(const Matrix& weight, std::uint32_t row, const float* x) {
  const std::size_t packed_stride = weight.cols / 2u;
  const std::size_t scale_stride = weight.cols / kGroup;
  const std::uint8_t* packed =
      weight.packed.data() + static_cast<std::size_t>(row) * packed_stride;
  const std::uint8_t* scales =
      weight.scales.data() + static_cast<std::size_t>(row) * scale_stride;
  std::array<float, kLanesPerRow> lane_values{};
  for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
    lane_values[lane] =
        LanePartial(packed, scales, x, lane, weight.cols);
  }
  for (std::uint32_t mask = 1; mask < kLanesPerRow; mask <<= 1u) {
    const auto previous = lane_values;
    for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
      lane_values[lane] = previous[lane] + previous[lane ^ mask];
    }
  }
  return lane_values[0];
}

template <typename RowFunction>
std::vector<std::uint32_t> RunTiles(std::uint32_t rows,
                                    RowFunction&& process_row) {
  const std::uint32_t tile_count = rows / kRowsPerTile;
  std::vector<std::uint32_t> visits(tile_count, 0u);
  for (std::uint32_t core = 0; core < kBlockDim; ++core) {
    for (std::uint32_t tile = core; tile < tile_count; tile += kBlockDim) {
      ++visits[tile];
      for (std::uint32_t offset = 0; offset < kRowsPerTile; ++offset) {
        process_row(tile * kRowsPerTile + offset);
      }
    }
  }
  return visits;
}

std::vector<float> SimulateGateUp(const Matrix& w1, const Matrix& w3,
                                  const std::vector<float>& x,
                                  std::vector<std::uint32_t>* visits) {
  std::vector<float> activated(kInter);
  *visits = RunTiles(kInter, [&](std::uint32_t row) {
    const float gate = SimtGemvRow(w1, row, x.data());
    const float up = SimtGemvRow(w3, row, x.data());
    const float g = gate < kLimit ? gate : kLimit;
    const float u = up < -kLimit ? -kLimit : (up > kLimit ? kLimit : up);
    activated[row] = (g / (1.0F + std::exp(-g))) * u;
  });
  return activated;
}

std::vector<float> SimulateDown(const Matrix& w2,
                                const std::vector<float>& activated,
                                std::vector<std::uint32_t>* visits) {
  std::vector<float> output(kHidden);
  *visits = RunTiles(kHidden, [&](std::uint32_t row) {
    output[row] = SimtGemvRow(w2, row, activated.data());
  });
  return output;
}

struct Diff {
  float max_abs = 0.0F;
  bool within_tolerance = true;
};

Diff Compare(const std::vector<float>& actual,
             const std::vector<float>& expected) {
  Require(actual.size() == expected.size(), "comparison size mismatch");
  Diff diff;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      diff.max_abs = std::numeric_limits<float>::infinity();
      diff.within_tolerance = false;
      continue;
    }
    const float delta = std::fabs(actual[i] - expected[i]);
    diff.max_abs = std::max(diff.max_abs, delta);
    if (delta > kAbsTolerance + kRelTolerance * std::fabs(expected[i])) {
      diff.within_tolerance = false;
    }
  }
  return diff;
}

bool NonDegenerate(const std::vector<float>& values) {
  if (values.empty()) return false;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (const float value : values) {
    if (!std::isfinite(value)) return false;
    sum += value;
    sum_sq += static_cast<double>(value) * value;
  }
  const double mean = sum / static_cast<double>(values.size());
  const double mean_sq = sum_sq / static_cast<double>(values.size());
  const double rms = std::sqrt(mean_sq);
  const double variance = std::max(0.0, mean_sq - mean * mean);
  return rms > 1.0e-6 && std::sqrt(variance) > 1.0e-6;
}

void RunTest() {
  moe::kernels::arch35::Dsv4MoeExpertTilingData tiling{};
  const auto status = moe::kernels::arch35::MakeDsv4MoeExpertTiling(
      kHidden, kInter, kHidden / 2u, kHidden / kGroup, kInter / 2u,
      kInter / kGroup, &tiling);
  Require(status == moe::kernels::arch35::Status::success,
          "test override rejected the scaled tiling");
  Require(tiling.block_dim == kBlockDim && tiling.hidden == kHidden &&
              tiling.inter == kInter &&
              tiling.rows_per_block == kRowsPerTile &&
              tiling.group_size == kGroup &&
              tiling.w13_packed_row_stride == kHidden / 2u &&
              tiling.w13_scale_row_stride == kHidden / kGroup &&
              tiling.w2_packed_row_stride == kInter / 2u &&
              tiling.w2_scale_row_stride == kInter / kGroup &&
              tiling.low_nibble_first == 1u && tiling.reserved0 == 0u &&
              tiling.reserved1 == 0u && tiling.swiglu_limit == kLimit &&
              moe::kernels::arch35::Dsv4ExpertTilingGeometryValid(
                  tiling.hidden, tiling.inter, tiling.block_dim),
          "test tiling fields do not satisfy the kernel geometry guard");

  // Each lane must own at least one whole 16-pair scale group. Thus 256/128
  // would make the kernel's group loop empty; 512 is the smallest valid width.
  const auto group_aligned = [](std::uint32_t cols) {
    const std::uint32_t pairs_per_lane =
        (cols / 2u) / kLanesPerRow;
    return pairs_per_lane >= 16u && pairs_per_lane % 16u == 0u;
  };
  Require(group_aligned(kHidden) && group_aligned(kInter),
          "SIMT lanes do not own whole MXFP4 scale groups");

  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);
  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);
  const Matrix w1 = MakeMatrix(kInter, kHidden, rng);
  const Matrix w3 = MakeMatrix(kInter, kHidden, rng);
  const Matrix w2 = MakeMatrix(kHidden, kInter, rng);

  std::vector<float> reference_gate(kInter);
  std::vector<float> reference_up(kInter);
  moe::emu::mxfp4_gemv_cpu(w1.View(), x.data(), reference_gate.data());
  moe::emu::mxfp4_gemv_cpu(w3.View(), x.data(), reference_up.data());
  std::uint32_t gate_upper_clamps = 0;
  std::uint32_t up_upper_clamps = 0;
  std::uint32_t up_lower_clamps = 0;
  for (std::uint32_t row = 0; row < kInter; ++row) {
    gate_upper_clamps += reference_gate[row] > kLimit ? 1u : 0u;
    up_upper_clamps += reference_up[row] > kLimit ? 1u : 0u;
    up_lower_clamps += reference_up[row] < -kLimit ? 1u : 0u;
  }
  Require(gate_upper_clamps > 0u && gate_upper_clamps < kInter &&
              up_upper_clamps > 0u && up_lower_clamps > 0u,
          "fixture did not exercise each asymmetric clamp branch");

  std::vector<float> reference_y(kHidden);
  std::vector<float> scratch;
  moe::emu::moe_expert_swiglu_mxfp4_cpu(
      w1.View(), w3.View(), w2.View(), x.data(), reference_y.data(), kLimit,
      &scratch);
  Require(scratch.size() >= kInter, "CPU reference omitted activation data");
  const std::vector<float> reference_activated(scratch.begin(),
                                                scratch.begin() + kInter);

  std::vector<std::uint32_t> gate_visits;
  const std::vector<float> actual_activated =
      SimulateGateUp(w1, w3, x, &gate_visits);
  std::vector<std::uint32_t> down_visits;
  const std::vector<float> actual_y =
      SimulateDown(w2, actual_activated, &down_visits);
  const auto every_once = [](const std::vector<std::uint32_t>& visits) {
    return std::all_of(visits.begin(), visits.end(),
                       [](std::uint32_t n) { return n == 1u; });
  };
  Require(gate_visits.size() == 16u && down_visits.size() == 16u &&
              every_once(gate_visits) && every_once(down_visits),
          "grid-stride simulation skipped or duplicated a tile");

  const Diff activation_diff =
      Compare(actual_activated, reference_activated);
  const Diff output_diff = Compare(actual_y, reference_y);
  Require(NonDegenerate(reference_activated),
          "activation golden is zero, constant, or non-finite");
  Require(NonDegenerate(reference_y),
          "output golden is zero, constant, or non-finite");
  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf("Tiling accepted: hidden=%u inter=%u block_dim=%u rows/tile=%u; "
              "gate/up tiles=16, down tiles=16 (8 tiles/core)\n",
              tiling.hidden, tiling.inter, tiling.block_dim,
              tiling.rows_per_block);
  std::printf("Activated max abs diff: %.8g; output max abs diff: %.8g "
              "(abs/rel tolerance %.2g/%.2g)\n",
              activation_diff.max_abs, output_diff.max_abs, kAbsTolerance,
              kRelTolerance);
  Require(activation_diff.within_tolerance,
          "gate/up activation differs from CPU reference beyond tolerance");
  Require(output_diff.within_tolerance,
          "down output differs from CPU reference beyond tolerance");
}
}  // namespace

int main() {
  try {
    RunTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] Dsv4MoeExpertSim: %s\n", error.what());
    return 1;
  }
  std::puts("[PASS] Dsv4MoeExpertSim");
  return 0;
}
