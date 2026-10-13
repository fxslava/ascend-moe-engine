// -----------------------------------------------------------------------------
// test_bf16_block_quant_emu -- oracle for moe::emu::bf16_block_quant_fp8_cpu,
// the host golden the bf16_block_quant_fp8 device kernel is verified against.
//
// Per CLAUDE.md §4.1/§4.2 this links moe_emu only, includes no device header,
// builds with a stock host compiler, and has no skip path: every case below
// runs on every machine.
//
// TOLERANCES ARE SET BY THE FORMAT, NOT BY OBSERVATION (§4.7)
// OCP FP8 E4M3 keeps a 3-bit stored mantissa, so within a binade the
// representable points are spaced 2^-3 apart relative to the binade's base
// and round-to-nearest lands within half of that: a relative error of at
// most 2^-4 = 0.0625 on any single value whose magnitude is at least the
// smallest normal. The quantizer scales each block so its largest magnitude
// maps exactly onto 448 (the top of the format), so every element of a block
// is measured against THAT block's maximum rather than against itself --
// which is the honest bound, because a value far below the block max is
// quantized coarsely in relative terms by construction.
//
// Cosine stays strict while relative-L2 is allowed to be loose, per §4.7:
// rounding moves magnitude, structural bugs (a transposed pair, a
// misattributed scale, an off-by-one block) move direction.
// -----------------------------------------------------------------------------

#include "emu_test_support.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using moe::emu::Bf16ToFloat;
using moe::emu::bf16_block_quant_fp8_cpu;
using moe::emu::bf16_block_quant_fp8_dequant;
using moe::emu::FloatToBf16;
using moe::emu::kBlockQuantRefBlockSize;
using moe::emu::kFp8E4M3Max;

constexpr int kBlock = kBlockQuantRefBlockSize;

// Half an E4M3 mantissa step, relative. See the header comment.
constexpr double kE4M3RelativeHalfUlp = 1.0 / 16.0;

// The scales are stored as FP32, and the expectations below are computed in
// double, so an exact comparison between them is wrong by construction: the
// FP32 quotient carries one rounding of its own. FP32 keeps a 23-bit stored
// mantissa, so consecutive representable values are 2^-23 apart relative,
// and one such step is the whole budget a single stored quotient may differ
// by. Bounds derived from this are format-justified, not fitted -- an
// earlier revision of this test used a flat 1e-12 and failed at 3.3e-11 and
// 4.0e-10, both of which are inside one FP32 ulp at those magnitudes.
constexpr double kFp32RelativeUlp = 1.1920929e-7;  // 2^-23

// Absolute tolerance for an FP32-stored quantity whose true value is `want`.
double Fp32Tolerance(double want) {
  return std::fabs(want) * kFp32RelativeUlp;
}

// The widths the model actually asks for, all with M = 1.
const int kModelWidths[] = {1024, 2048, 4096, 8192, 32768};

std::vector<std::uint16_t> MakeBf16Vector(int width, float low, float high) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(width));
  for (std::uint16_t& value : out) {
    // Round-tripped through BF16 so the stored bits ARE the value the
    // reference will widen -- no hidden FP32 precision the device never saw.
    value = FloatToBf16(emu_test::UniformFloat(low, high));
  }
  return out;
}

double CosineSimilarity(const std::vector<double>& a,
                        const std::vector<double>& b) {
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  if (norm_a <= 0.0 || norm_b <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// ---------------------------------------------------------------------------
// 1. The scale identity, exactly
// ---------------------------------------------------------------------------
// scale[b] = max|x| / 448 is the defining equation, computed independently
// here from the BF16 bits. The reference forms it as a divide, so this is an
// exact FP32 comparison, not an approximate one.
void ScaleIdentity(emu_test::Runner& runner) {
  runner.Section("scale identity: scale[b] == max|x| / 448, exactly");

  for (const int width : kModelWidths) {
    const std::vector<std::uint16_t> x = MakeBf16Vector(width, -6.0F, 6.0F);
    const int blocks = width / kBlock;
    std::vector<std::uint8_t> q(static_cast<std::size_t>(width));
    std::vector<float> scale(static_cast<std::size_t>(blocks));
    bf16_block_quant_fp8_cpu(x.data(), width, q.data(), scale.data());

    int mismatches = 0;
    for (int b = 0; b < blocks; ++b) {
      float want_max = 0.0F;
      for (int i = 0; i < kBlock; ++i) {
        want_max = std::max(want_max, std::fabs(Bf16ToFloat(x[static_cast<std::size_t>(b * kBlock + i)])));
      }
      const float want = want_max > 0.0F ? want_max / kFp8E4M3Max : 1.0F;
      if (scale[static_cast<std::size_t>(b)] != want) {
        ++mismatches;
      }
    }
    runner.Check(mismatches == 0,
                 "width " + std::to_string(width) + ": " +
                     std::to_string(mismatches) + " of " +
                     std::to_string(blocks) + " block scales differ from max|x|/448");
  }
}

// ---------------------------------------------------------------------------
// 2. Non-degeneracy of the golden (§4.8)
// ---------------------------------------------------------------------------
// Asserts properties of the REFERENCE DATA, not a value against itself:
// that the input really varies, that the scales are not all equal (which a
// stuck block-max would produce), that the codes span more than one value
// (which a dead quantizer would not), and that the top of the format is
// actually reached -- the last is what proves the scaling is doing its job
// rather than leaving the format mostly unused.
void NonDegeneracy(emu_test::Runner& runner) {
  runner.Section("non-degeneracy of the golden");

  constexpr int kWidth = 4096;
  const std::vector<std::uint16_t> x = MakeBf16Vector(kWidth, -6.0F, 6.0F);
  const int blocks = kWidth / kBlock;
  std::vector<std::uint8_t> q(kWidth);
  std::vector<float> scale(static_cast<std::size_t>(blocks));
  bf16_block_quant_fp8_cpu(x.data(), kWidth, q.data(), scale.data());

  // Input really varies.
  double mean = 0.0;
  for (const std::uint16_t bits : x) mean += Bf16ToFloat(bits);
  mean /= static_cast<double>(kWidth);
  double variance = 0.0;
  double norm = 0.0;
  for (const std::uint16_t bits : x) {
    const double value = Bf16ToFloat(bits);
    variance += (value - mean) * (value - mean);
    norm += value * value;
  }
  variance /= static_cast<double>(kWidth);
  runner.Check(norm > 0.0, "input has a nonzero L2 norm");
  runner.Check(variance > 0.1, "input has real variance (got " +
                                   std::to_string(variance) + ")");

  // Scales differ between blocks.
  const float min_scale = *std::min_element(scale.begin(), scale.end());
  const float max_scale = *std::max_element(scale.begin(), scale.end());
  runner.Check(min_scale > 0.0F, "every block scale is strictly positive");
  runner.Check(max_scale > min_scale,
               "block scales vary across blocks (a constant scale would mean "
               "the block max is stuck)");

  // Codes span the format and reach its top.
  std::vector<int> histogram(256, 0);
  for (const std::uint8_t code : q) ++histogram[code];
  int distinct = 0;
  for (const int count : histogram) {
    if (count > 0) ++distinct;
  }
  runner.Check(distinct > 16,
               "quantized codes take many distinct values (got " +
                   std::to_string(distinct) + ")");

  // Every block's largest magnitude is scaled onto 448, so the saturating
  // code 0x7E (+448) or 0xFE (-448) must appear at least once per block.
  int blocks_reaching_top = 0;
  for (int b = 0; b < blocks; ++b) {
    bool reached = false;
    for (int i = 0; i < kBlock; ++i) {
      const std::uint8_t code = q[static_cast<std::size_t>(b * kBlock + i)];
      if ((code & 0x7FU) == 0x7EU) {
        reached = true;
        break;
      }
    }
    if (reached) ++blocks_reaching_top;
  }
  runner.Check(blocks_reaching_top == blocks,
               "every block reaches the top of the E4M3 range (" +
                   std::to_string(blocks_reaching_top) + " of " +
                   std::to_string(blocks) + ")");
}

// ---------------------------------------------------------------------------
// 3. Dequantization round trip, bounded by the format
// ---------------------------------------------------------------------------
void RoundTrip(emu_test::Runner& runner) {
  runner.Section("dequantization round trip within the E4M3 bound");

  for (const int width : kModelWidths) {
    const std::vector<std::uint16_t> x = MakeBf16Vector(width, -6.0F, 6.0F);
    const int blocks = width / kBlock;
    std::vector<std::uint8_t> q(static_cast<std::size_t>(width));
    std::vector<float> scale(static_cast<std::size_t>(blocks));
    bf16_block_quant_fp8_cpu(x.data(), width, q.data(), scale.data());

    std::vector<double> want(static_cast<std::size_t>(width));
    std::vector<double> got(static_cast<std::size_t>(width));
    double worst_relative = 0.0;
    double sq_error = 0.0;
    double sq_signal = 0.0;

    for (int b = 0; b < blocks; ++b) {
      // Each element is judged against ITS BLOCK's maximum, which is what
      // the format bound is relative to -- see the file header.
      float block_max = 0.0F;
      for (int i = 0; i < kBlock; ++i) {
        block_max = std::max(block_max, std::fabs(Bf16ToFloat(x[static_cast<std::size_t>(b * kBlock + i)])));
      }
      for (int i = 0; i < kBlock; ++i) {
        const std::size_t index = static_cast<std::size_t>(b * kBlock + i);
        const double original = Bf16ToFloat(x[index]);
        const double decoded =
            bf16_block_quant_fp8_dequant(q[index], scale[static_cast<std::size_t>(b)]);
        want[index] = original;
        got[index] = decoded;
        sq_error += (decoded - original) * (decoded - original);
        sq_signal += original * original;
        if (block_max > 0.0F) {
          worst_relative =
              std::max(worst_relative,
                       std::fabs(decoded - original) / static_cast<double>(block_max));
        }
      }
    }

    runner.Check(worst_relative <= kE4M3RelativeHalfUlp,
                 "width " + std::to_string(width) +
                     ": worst per-element error vs its block max is " +
                     std::to_string(worst_relative) + ", bound " +
                     std::to_string(kE4M3RelativeHalfUlp));

    // Relative L2 is allowed to be loose; cosine is not.
    const double relative_l2 =
        sq_signal > 0.0 ? std::sqrt(sq_error / sq_signal) : 0.0;
    runner.Check(relative_l2 < 0.05,
                 "width " + std::to_string(width) + ": relative L2 " +
                     std::to_string(relative_l2) + " < 0.05");
    const double cosine = CosineSimilarity(want, got);
    runner.Check(cosine > 0.999,
                 "width " + std::to_string(width) + ": cosine " +
                     std::to_string(cosine) + " > 0.999 (direction preserved)");
  }
}

// ---------------------------------------------------------------------------
// 4. Edge cases the kernel's guards exist for
// ---------------------------------------------------------------------------
void EdgeCases(emu_test::Runner& runner) {
  runner.Section("edge cases: zero blocks, saturation, per-block independence");

  // An all-zero block must take scale 1 and codes 0, so that dequantization
  // is exactly zero and nothing downstream multiplies by a zero scale.
  {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kBlock) * 2, 0u);
    // Second block is nonzero, so this also proves blocks are independent.
    for (int i = 0; i < kBlock; ++i) {
      x[static_cast<std::size_t>(kBlock + i)] = FloatToBf16(1.5F);
    }
    std::vector<std::uint8_t> q(x.size());
    std::vector<float> scale(2);
    bf16_block_quant_fp8_cpu(x.data(), static_cast<int>(x.size()), q.data(),
                             scale.data());

    runner.Check(scale[0] == 1.0F, "all-zero block takes scale exactly 1");
    bool all_zero_codes = true;
    for (int i = 0; i < kBlock; ++i) {
      if (q[static_cast<std::size_t>(i)] != 0u) all_zero_codes = false;
    }
    runner.Check(all_zero_codes, "all-zero block quantizes to all-zero codes");
    runner.CheckNear(bf16_block_quant_fp8_dequant(q[0], scale[0]), 0.0, 0.0,
                     "all-zero block dequantizes to exactly 0");
    const double want_neighbour = 1.5 / static_cast<double>(kFp8E4M3Max);
    runner.CheckNear(scale[1], want_neighbour, Fp32Tolerance(want_neighbour),
                     "nonzero neighbour block keeps its own scale (no bleed "
                     "from the zero block)");
  }

  // A constant block: every element equals the max, so every code must be
  // the top of the format and the round trip must be exact.
  {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kBlock),
                                 FloatToBf16(3.25F));
    std::vector<std::uint8_t> q(x.size());
    std::vector<float> scale(1);
    bf16_block_quant_fp8_cpu(x.data(), kBlock, q.data(), scale.data());
    runner.Check((q[0] & 0x7FU) == 0x7EU,
                 "constant block maps onto the top E4M3 code");
    runner.CheckNear(bf16_block_quant_fp8_dequant(q[0], scale[0]), 3.25, 1e-6,
                     "constant block round-trips exactly");
  }

  // Sign must survive: a block with a negative maximum still scales on
  // magnitude, and the sign bit must be carried per element.
  {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kBlock));
    for (int i = 0; i < kBlock; ++i) {
      x[static_cast<std::size_t>(i)] =
          FloatToBf16((i % 2 == 0) ? -4.0F : 2.0F);
    }
    std::vector<std::uint8_t> q(x.size());
    std::vector<float> scale(1);
    bf16_block_quant_fp8_cpu(x.data(), kBlock, q.data(), scale.data());
    const double want_scale = 4.0 / static_cast<double>(kFp8E4M3Max);
    runner.CheckNear(scale[0], want_scale, Fp32Tolerance(want_scale),
                     "scale comes from the magnitude, not the signed max");
    runner.Check((q[0] & 0x80U) != 0u, "negative element keeps its sign bit");
    runner.Check((q[1] & 0x80U) == 0u, "positive element keeps its sign bit");
    runner.CheckNear(bf16_block_quant_fp8_dequant(q[0], scale[0]), -4.0, 1e-5,
                     "negative extreme round-trips");
  }

  // Values beyond the E4M3 range after scaling cannot occur by construction
  // (the block max maps to exactly 448), so saturation is only ever a
  // rounding boundary. Assert no code is the NaN pattern.
  {
    const std::vector<std::uint16_t> x = MakeBf16Vector(1024, -200.0F, 200.0F);
    std::vector<std::uint8_t> q(x.size());
    std::vector<float> scale(1024 / kBlock);
    bf16_block_quant_fp8_cpu(x.data(), 1024, q.data(), scale.data());
    int nan_codes = 0;
    for (const std::uint8_t code : q) {
      if ((code & 0x7FU) == 0x7FU) ++nan_codes;
    }
    runner.Check(nan_codes == 0,
                 "no element quantizes to the E4M3 NaN pattern, even with a "
                 "wide input range (got " + std::to_string(nan_codes) + ")");
  }
}

}  // namespace

int main() {
  emu_test::Runner runner("bf16_block_quant_fp8 oracle");
  ScaleIdentity(runner);
  NonDegeneracy(runner);
  RoundTrip(runner);
  EdgeCases(runner);
  return runner.Summary();
}
