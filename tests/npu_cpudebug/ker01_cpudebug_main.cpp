// KER-01 numerical acceptance, run through CANN's CPU interpreter for
// Ascend950PR_9599 against the fixtures extracted from golden v3.
//
// h_raw and B are compared against GOLDEN byte slices; h_norm and post are
// compared against the oracle/stage-boundary values, because v3 does not
// serialize them. Structural invariants on B (row and column sums) are
// checked too: a doubly stochastic matrix is the Sinkhorn solve's contract.

#include "tikicpulib.h"
#include "fixture_io.h"
#include "moe/kernels/arch35/kernel_contracts.h"

#include <string>
#include <vector>

extern "C" __global__ __aicore__ void ker01_mhc_pre_norm_sinkhorn(
    GM_ADDR x_gm, GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm,
    GM_ADDR gamma_gm, GM_ADDR h_raw_gm, GM_ADDR h_norm_gm, GM_ADDR post_gm,
    GM_ADDR birkhoff_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm);

namespace mk = moe::kernels::arch35;
using namespace arch35_test;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ker01_cpudebug <fixture-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Verdict verdict("KER-01 fused_mhc_pre_norm_sinkhorn");

  constexpr uint32_t kHidden = mk::kHiddenSize;
  constexpr uint32_t kFlat = mk::kHcFlatWidth;
  constexpr uint32_t kRows = mk::kHcProjectionRows;
  constexpr uint32_t kStreams = mk::kHcStreams;
  constexpr uint32_t kPhiCols = kFlat;

  const std::vector<uint16_t> x = ReadTyped<uint16_t>(dir, "x", kFlat);
  const std::vector<float> phi = ReadTyped<float>(dir, "phi", kRows * kPhiCols);
  const std::vector<float> alpha = ReadTyped<float>(dir, "alpha", 3);
  const std::vector<float> bias = ReadTyped<float>(dir, "bias", kRows);
  const std::vector<uint16_t> gamma = ReadTyped<uint16_t>(dir, "gamma", kHidden);
  const std::vector<float> want_h_raw = ReadTyped<float>(dir, "expect_h_raw", kHidden);
  const std::vector<uint16_t> want_h_norm =
      ReadTyped<uint16_t>(dir, "expect_h_norm", kHidden);
  const std::vector<float> want_post = ReadTyped<float>(dir, "expect_post", kStreams);
  const std::vector<float> want_b =
      ReadTyped<float>(dir, "expect_birkhoff", kStreams * kStreams);

  // The real host tiler builds and validates the tiling buffer.
  mk::MhcPreNormTilingData tiling{};
  const mk::Status status = mk::BuildMhcPreNormTiling(kHidden, kPhiCols, &tiling);
  verdict.CheckTrue("host tiler accepts the geometry", status == mk::Status::success);

  auto* x_gm = (uint8_t*)AscendC::GmAlloc(x.size() * sizeof(uint16_t));
  auto* phi_gm = (uint8_t*)AscendC::GmAlloc(phi.size() * sizeof(float));
  auto* alpha_gm = (uint8_t*)AscendC::GmAlloc(alpha.size() * sizeof(float));
  auto* bias_gm = (uint8_t*)AscendC::GmAlloc(bias.size() * sizeof(float));
  auto* gamma_gm = (uint8_t*)AscendC::GmAlloc(gamma.size() * sizeof(uint16_t));
  auto* h_raw_gm = (uint8_t*)AscendC::GmAlloc(kHidden * sizeof(float));
  auto* h_norm_gm = (uint8_t*)AscendC::GmAlloc(kHidden * sizeof(uint16_t));
  auto* post_gm = (uint8_t*)AscendC::GmAlloc(kStreams * sizeof(float));
  auto* b_gm = (uint8_t*)AscendC::GmAlloc(kStreams * kStreams * sizeof(float));
  auto* tiling_gm = (uint8_t*)AscendC::GmAlloc(sizeof(tiling));

  std::memcpy(x_gm, x.data(), x.size() * sizeof(uint16_t));
  std::memcpy(phi_gm, phi.data(), phi.size() * sizeof(float));
  std::memcpy(alpha_gm, alpha.data(), alpha.size() * sizeof(float));
  std::memcpy(bias_gm, bias.data(), bias.size() * sizeof(float));
  std::memcpy(gamma_gm, gamma.data(), gamma.size() * sizeof(uint16_t));
  std::memcpy(tiling_gm, &tiling, sizeof(tiling));
  // Poison the outputs so a kernel that never writes cannot pass by accident.
  std::memset(h_raw_gm, 0xA5, kHidden * sizeof(float));
  std::memset(h_norm_gm, 0xA5, kHidden * sizeof(uint16_t));
  std::memset(post_gm, 0xA5, kStreams * sizeof(float));
  std::memset(b_gm, 0xA5, kStreams * kStreams * sizeof(float));

  // GM scratch contract is 0 bytes: pass nothing and prove it is not needed.
  ICPU_RUN_KF(ker01_mhc_pre_norm_sinkhorn, tiling.block_dim, (GM_ADDR)x_gm,
              (GM_ADDR)phi_gm, (GM_ADDR)alpha_gm, (GM_ADDR)bias_gm, (GM_ADDR)gamma_gm,
              (GM_ADDR)h_raw_gm, (GM_ADDR)h_norm_gm, (GM_ADDR)post_gm, (GM_ADDR)b_gm,
              (GM_ADDR) nullptr, (GM_ADDR)tiling_gm);

  const auto* got_h_raw = reinterpret_cast<const float*>(h_raw_gm);
  const auto* got_h_norm = reinterpret_cast<const uint16_t*>(h_norm_gm);
  const auto* got_post = reinterpret_cast<const float*>(post_gm);
  const auto* got_b = reinterpret_cast<const float*>(b_gm);

  verdict.CheckMetrics("h_raw vs golden h_in (FP32)",
                       Compare(got_h_raw, want_h_raw.data(), kHidden));

  std::vector<float> got_norm_f(kHidden);
  std::vector<float> want_norm_f(kHidden);
  for (uint32_t i = 0; i < kHidden; ++i) {
    got_norm_f[i] = Bf16ToFloat(got_h_norm[i]);
    want_norm_f[i] = Bf16ToFloat(want_h_norm[i]);
  }
  verdict.CheckMetrics("h_norm vs derived (BF16)",
                       Compare(got_norm_f.data(), want_norm_f.data(), kHidden));
  verdict.CheckMetrics("post vs oracle (FP32)",
                       Compare(got_post, want_post.data(), kStreams));
  verdict.CheckMetrics("B vs golden Birkhoff (FP32)",
                       Compare(got_b, want_b.data(), kStreams * kStreams));

  // Structural check on B, with the right axis.
  //
  // The Sinkhorn cadence's LAST operation is always a column normalization, so
  // columns are the tight axis and sum to 1 within ~1e-6. Rows are only
  // approximately 1 after 20 iterations: the golden file's own layer-0 B has
  // row sums from 0.923 to 1.082. Asserting rows sum to 1 would therefore be
  // asserting something the reference algorithm does not provide, so the row
  // axis is checked against the REFERENCE's row sums instead.
  double worst_col = 0.0;
  double worst_row_delta = 0.0;
  for (uint32_t i = 0; i < kStreams; ++i) {
    double got_row = 0.0;
    double want_row = 0.0;
    double got_col = 0.0;
    for (uint32_t j = 0; j < kStreams; ++j) {
      got_row += got_b[i * kStreams + j];
      want_row += want_b[i * kStreams + j];
      got_col += got_b[j * kStreams + i];
    }
    worst_col = std::max(worst_col, std::fabs(got_col - 1.0));
    worst_row_delta = std::max(worst_row_delta, std::fabs(got_row - want_row));
  }
  std::printf("  %-34s col dev %.3e, row delta vs ref %.3e\n",
              "B structure", worst_col, worst_row_delta);
  verdict.CheckTrue("B column sums are 1 within 1e-5", worst_col < 1e-5,
                    "the final column normalization did not take");
  verdict.CheckTrue("B row sums match the reference", worst_row_delta < 1e-4,
                    "row mass diverged from the oracle");

  AscendC::GmFree(x_gm);
  AscendC::GmFree(phi_gm);
  AscendC::GmFree(alpha_gm);
  AscendC::GmFree(bias_gm);
  AscendC::GmFree(gamma_gm);
  AscendC::GmFree(h_raw_gm);
  AscendC::GmFree(h_norm_gm);
  AscendC::GmFree(post_gm);
  AscendC::GmFree(b_gm);
  AscendC::GmFree(tiling_gm);
  return verdict.Finish();
}
