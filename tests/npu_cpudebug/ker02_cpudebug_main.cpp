// KER-02 numerical acceptance under CANN's CPU interpreter.
//
// Every input and the expected y are golden byte slices or captured stage
// boundaries, so this ties the kernel directly to golden v3: x and y are the
// serialized bundle streams and B is the serialized Birkhoff matrix.
//
// Also checks the two things the contract calls out: the B^T orientation (an
// asymmetric B makes a transpose visible) and that the four blocks write
// disjoint y rows.

#include "tikicpulib.h"
#include "fixture_io.h"
#include "moe/kernels/arch35/kernel_contracts.h"

#include <string>
#include <vector>

extern "C" __global__ __aicore__ void ker02_mhc_post_residual(
    GM_ADDR x_gm, GM_ADDR birkhoff_gm, GM_ADDR post_gm, GM_ADDR h_out_gm,
    GM_ADDR y_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm);

namespace mk = moe::kernels::arch35;
using namespace arch35_test;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ker02_cpudebug <fixture-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Verdict verdict("KER-02 fused_mhc_post_residual");

  constexpr uint32_t kHidden = mk::kHiddenSize;
  constexpr uint32_t kFlat = mk::kHcFlatWidth;
  constexpr uint32_t kStreams = mk::kHcStreams;

  const std::vector<uint16_t> x = ReadTyped<uint16_t>(dir, "x", kFlat);
  const std::vector<float> b = ReadTyped<float>(dir, "birkhoff", kStreams * kStreams);
  const std::vector<float> post = ReadTyped<float>(dir, "post", kStreams);
  const std::vector<uint16_t> h_out = ReadTyped<uint16_t>(dir, "h_out", kHidden);
  const std::vector<uint16_t> want_y = ReadTyped<uint16_t>(dir, "expect_y", kFlat);

  mk::MhcPostTilingData tiling{};
  const mk::Status status =
      mk::BuildMhcPostTiling(kHidden, kHidden, kStreams, &tiling);
  verdict.CheckTrue("host tiler accepts the geometry", status == mk::Status::success);

  auto* x_gm = (uint8_t*)AscendC::GmAlloc(x.size() * sizeof(uint16_t));
  auto* b_gm = (uint8_t*)AscendC::GmAlloc(b.size() * sizeof(float));
  auto* post_gm = (uint8_t*)AscendC::GmAlloc(post.size() * sizeof(float));
  auto* h_out_gm = (uint8_t*)AscendC::GmAlloc(h_out.size() * sizeof(uint16_t));
  auto* y_gm = (uint8_t*)AscendC::GmAlloc(kFlat * sizeof(uint16_t));
  auto* tiling_gm = (uint8_t*)AscendC::GmAlloc(sizeof(tiling));

  std::memcpy(x_gm, x.data(), x.size() * sizeof(uint16_t));
  std::memcpy(b_gm, b.data(), b.size() * sizeof(float));
  std::memcpy(post_gm, post.data(), post.size() * sizeof(float));
  std::memcpy(h_out_gm, h_out.data(), h_out.size() * sizeof(uint16_t));
  std::memcpy(tiling_gm, &tiling, sizeof(tiling));
  std::memset(y_gm, 0xA5, kFlat * sizeof(uint16_t));

  ICPU_RUN_KF(ker02_mhc_post_residual, tiling.block_dim, (GM_ADDR)x_gm, (GM_ADDR)b_gm,
              (GM_ADDR)post_gm, (GM_ADDR)h_out_gm, (GM_ADDR)y_gm, (GM_ADDR) nullptr,
              (GM_ADDR)tiling_gm);

  const auto* got_y = reinterpret_cast<const uint16_t*>(y_gm);
  std::vector<float> got_f(kFlat);
  std::vector<float> want_f(kFlat);
  for (uint32_t i = 0; i < kFlat; ++i) {
    got_f[i] = Bf16ToFloat(got_y[i]);
    want_f[i] = Bf16ToFloat(want_y[i]);
  }
  verdict.CheckMetrics("y vs golden next bundle (BF16)",
                       Compare(got_f.data(), want_f.data(), kFlat));

  // Per-stream, so one wrong output row cannot hide inside the flat norm --
  // which is exactly what a transposed B would look like.
  for (uint32_t s = 0; s < kStreams; ++s) {
    char label[64];
    std::snprintf(label, sizeof(label), "y stream %u vs golden", s);
    verdict.CheckMetrics(label, Compare(got_f.data() + s * kHidden,
                                        want_f.data() + s * kHidden, kHidden));
  }

  // Every block must have written: the poison byte pattern 0xA5A5 must be gone
  // from all four rows, which is the disjoint-write invariant.
  size_t untouched = 0;
  for (uint32_t i = 0; i < kFlat; ++i) {
    if (got_y[i] == 0xA5A5) {
      ++untouched;
    }
  }
  verdict.CheckTrue("all 4 output streams written", untouched == 0,
                    "a block left its y row at the poison value");

  AscendC::GmFree(x_gm);
  AscendC::GmFree(b_gm);
  AscendC::GmFree(post_gm);
  AscendC::GmFree(h_out_gm);
  AscendC::GmFree(y_gm);
  AscendC::GmFree(tiling_gm);
  return verdict.Finish();
}
