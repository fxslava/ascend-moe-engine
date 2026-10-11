// KER-05 numerical acceptance under CANN's CPU interpreter.
//
// Verifies the MXFP4 GEMV against the oracle on a real routed expert's
// weights, with a NONUNIFORM x: an all-ones probe sums both halves of every
// packed byte symmetrically and so cannot distinguish a within-byte nibble
// swap. The swap is also checked directly, by re-running against a
// deliberately swapped copy of the weights and requiring the result to differ.

#include "tikicpulib.h"
#include "fixture_io.h"
#include "moe/kernels/arch35/kernel_contracts.h"

#include <cstdlib>
#include <string>
#include <vector>

extern "C" __global__ __aicore__ void ker05_mxfp4_expert_gemv(
    GM_ADDR packed_gm, GM_ADDR scale_gm, GM_ADDR x_gm, GM_ADDR y_gm,
    GM_ADDR workspace_gm, GM_ADDR tiling_gm);

namespace mk = moe::kernels::arch35;
using namespace arch35_test;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ker05_cpudebug <fixture-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Verdict verdict("KER-05 LaunchMxfp4ExpertGemv");

  // Shape comes from the fixture: x gives cols, expect_y gives rows.
  const std::vector<unsigned char> x_raw = ReadBytes(dir + "/x.bin");
  const std::vector<unsigned char> y_raw = ReadBytes(dir + "/expect_y.bin");
  const uint32_t cols = static_cast<uint32_t>(x_raw.size() / sizeof(float));
  const uint32_t rows = static_cast<uint32_t>(y_raw.size() / sizeof(float));
  const uint32_t packed_stride = cols / 2U;
  const uint32_t scale_stride = cols / mk::kMxfp4GroupSize;
  std::printf("  shape: rows=%u cols=%u packed_stride=%u scale_stride=%u\n", rows, cols,
              packed_stride, scale_stride);

  const std::vector<float> x = ReadTyped<float>(dir, "x", cols);
  const std::vector<float> want_y = ReadTyped<float>(dir, "expect_y", rows);
  const std::vector<uint8_t> packed =
      ReadTyped<uint8_t>(dir, "packed", static_cast<size_t>(rows) * packed_stride);
  const std::vector<uint8_t> scale =
      ReadTyped<uint8_t>(dir, "scale", static_cast<size_t>(rows) * scale_stride);

  mk::Mxfp4GemvTilingData tiling{};
  const mk::Status status = mk::BuildMxfp4GemvTiling(rows, cols, packed_stride,
                                                     scale_stride, 1, &tiling);
  verdict.CheckTrue("host tiler accepts the geometry", status == mk::Status::success);
  // Diagnostic knob: KER05_BLOCK_DIM=1 collapses the grid onto one core, which
  // separates a per-row arithmetic fault from a cross-core one.
  if (const char* override_dim = std::getenv("KER05_BLOCK_DIM")) {
    tiling.block_dim = static_cast<uint32_t>(std::atoi(override_dim));
    std::printf("  block_dim overridden to %u\n", tiling.block_dim);
  }
  // The contract says low_nibble_first=0 and a wrong scale stride must be
  // refused rather than adapted.
  mk::Mxfp4GemvTilingData reject{};
  verdict.CheckTrue("host tiler rejects low_nibble_first=0",
                    mk::BuildMxfp4GemvTiling(rows, cols, packed_stride, scale_stride,
                                             0, &reject) != mk::Status::success);
  verdict.CheckTrue("host tiler rejects an FP8-style scale stride",
                    mk::BuildMxfp4GemvTiling(rows, cols, packed_stride,
                                             scale_stride / 4U, 1, &reject) !=
                        mk::Status::success);

  auto* packed_gm = (uint8_t*)AscendC::GmAlloc(packed.size());
  auto* scale_gm = (uint8_t*)AscendC::GmAlloc(scale.size());
  auto* x_gm = (uint8_t*)AscendC::GmAlloc(x.size() * sizeof(float));
  auto* y_gm = (uint8_t*)AscendC::GmAlloc(static_cast<size_t>(rows) * sizeof(float));
  auto* tiling_gm = (uint8_t*)AscendC::GmAlloc(sizeof(tiling));

  std::memcpy(packed_gm, packed.data(), packed.size());
  std::memcpy(scale_gm, scale.data(), scale.size());
  std::memcpy(x_gm, x.data(), x.size() * sizeof(float));
  std::memcpy(tiling_gm, &tiling, sizeof(tiling));
  std::memset(y_gm, 0xA5, static_cast<size_t>(rows) * sizeof(float));

  ICPU_RUN_KF(ker05_mxfp4_expert_gemv, tiling.block_dim, (GM_ADDR)packed_gm,
              (GM_ADDR)scale_gm, (GM_ADDR)x_gm, (GM_ADDR)y_gm, (GM_ADDR) nullptr,
              (GM_ADDR)tiling_gm);

  const auto* got_y = reinterpret_cast<const float*>(y_gm);
  verdict.CheckMetrics("y vs oracle mxfp4_gemv (FP32)",
                       Compare(got_y, want_y.data(), rows));

  // Localize any non-finite row: print its index, which block tile owned it,
  // and its position inside that tile.
  {
    size_t shown = 0;
    for (uint32_t r = 0; r < rows && shown < 8; ++r) {
      if (!std::isfinite(got_y[r])) {
        const uint32_t tile = r / tiling.rows_per_block;
        std::printf("      nonfinite row %u: tile %u, block %u, lane %u, got %g want %g\n",
                    r, tile, tile % tiling.block_dim, r % tiling.rows_per_block,
                    static_cast<double>(got_y[r]), static_cast<double>(want_y[r]));
        ++shown;
      }
    }
  }

  // Every row must have been written by some block: 0xA5A5A5A5 as a float is
  // a large negative value, so a missed row is unmistakable.
  size_t untouched = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    uint32_t bits;
    std::memcpy(&bits, &got_y[r], sizeof(bits));
    if (bits == 0xA5A5A5A5u) {
      ++untouched;
    }
  }
  verdict.CheckTrue("every row written by its block tile", untouched == 0,
                    "a 32-row tile was never claimed by any block");

  // Nibble-order sensitivity: swapping the halves of every packed byte must
  // change the answer. If it does not, the kernel is not reading the order it
  // claims and an all-ones probe would never have told us.
  std::vector<uint8_t> swapped(packed.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    swapped[i] = static_cast<uint8_t>(((packed[i] & 0x0FU) << 4U) |
                                      ((packed[i] >> 4U) & 0x0FU));
  }
  std::memcpy(packed_gm, swapped.data(), swapped.size());
  std::memset(y_gm, 0xA5, static_cast<size_t>(rows) * sizeof(float));
  ICPU_RUN_KF(ker05_mxfp4_expert_gemv, tiling.block_dim, (GM_ADDR)packed_gm,
              (GM_ADDR)scale_gm, (GM_ADDR)x_gm, (GM_ADDR)y_gm, (GM_ADDR) nullptr,
              (GM_ADDR)tiling_gm);
  const Metrics swapped_metrics = Compare(got_y, want_y.data(), rows);
  std::printf("  %-34s cos=%.6f relL2=%.3e (must NOT match)\n",
              "nibble-swapped control", swapped_metrics.cosine,
              swapped_metrics.relative_l2);
  verdict.CheckTrue("a nibble swap changes the result",
                    swapped_metrics.relative_l2 > kRelL2Ceiling,
                    "the kernel is insensitive to within-byte nibble order");

  AscendC::GmFree(packed_gm);
  AscendC::GmFree(scale_gm);
  AscendC::GmFree(x_gm);
  AscendC::GmFree(y_gm);
  AscendC::GmFree(tiling_gm);
  return verdict.Finish();
}
