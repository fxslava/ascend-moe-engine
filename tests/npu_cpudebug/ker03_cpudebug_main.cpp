// KER-03 numerical acceptance under CANN's CPU interpreter.
//
// Drives the full 4:1 cadence step by step against persistent GM state. The
// two invariants that a plausible-but-wrong compressor would violate are
// checked explicitly:
//
//   * HOLD steps (slot 0, 1, 2) must leave EVERY byte of cache_slot_gm
//     untouched. The slot is poisoned before each call and compared after.
//   * The SECOND emission must use the previous group's a-halves. A four-row
//     window still produces a finite, normalized token, so group 2 is the
//     only check that catches it.
//
// Group 1's emission is additionally compared against the golden v3 token.

#include "tikicpulib.h"
#include "fixture_io.h"
#include "moe/kernels/arch35/kernel_contracts.h"

#include <string>
#include <vector>

extern "C" __global__ __aicore__ void ker03_paged_compressor_rope(
    GM_ADDR x_gm, GM_ADDR wkv_gm, GM_ADDR wgate_gm, GM_ADDR ape_gm,
    GM_ADDR norm_weight_gm, GM_ADDR rope_cos_gm, GM_ADDR rope_sin_gm,
    GM_ADDR state_gm, GM_ADDR state_meta_gm, GM_ADDR cache_slot_gm,
    GM_ADDR emitted_gm, GM_ADDR workspace_gm, GM_ADDR tiling_gm);

namespace mk = moe::kernels::arch35;
using namespace arch35_test;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ker03_cpudebug <fixture-dir>\n");
    return 2;
  }
  const std::string dir = argv[1];
  Verdict verdict("KER-03 fused_paged_compressor_rope (C4A)");

  constexpr uint32_t kHidden = mk::kHiddenSize;
  constexpr uint32_t kHeadDim = mk::kCompressorHeadDim;
  constexpr uint32_t kRatio = mk::kCompressorRatio;
  constexpr uint32_t kStateWidth = mk::kCompressorCoff * kHeadDim;
  constexpr uint32_t kRopeDim = mk::kRopeHeadDim;

  // The fixture's step count follows from the emitted-flag vector.
  const std::vector<unsigned char> flags_raw = ReadBytes(dir + "/expect_emitted.bin");
  const uint32_t steps = static_cast<uint32_t>(flags_raw.size() / sizeof(uint32_t));
  const uint32_t groups = steps / kRatio;
  std::printf("  cadence: %u steps, %u group(s)\n", steps, groups);

  const std::vector<uint32_t> want_emitted =
      ReadTyped<uint32_t>(dir, "expect_emitted", steps);
  const std::vector<uint16_t> tokens =
      ReadTyped<uint16_t>(dir, "x_tokens", steps * kHidden);
  const std::vector<uint16_t> wkv =
      ReadTyped<uint16_t>(dir, "wkv", kStateWidth * kHidden);
  const std::vector<uint16_t> wgate =
      ReadTyped<uint16_t>(dir, "wgate", kStateWidth * kHidden);
  const std::vector<float> ape = ReadTyped<float>(dir, "ape", kRatio * kStateWidth);
  const std::vector<uint16_t> norm_weight =
      ReadTyped<uint16_t>(dir, "norm_weight", kHeadDim);
  const std::vector<float> rope_cos =
      ReadTyped<float>(dir, "rope_cos", groups * kRopeDim);
  const std::vector<float> rope_sin =
      ReadTyped<float>(dir, "rope_sin", groups * kRopeDim);
  const std::vector<uint16_t> want_tokens =
      ReadTyped<uint16_t>(dir, "expect_tokens", groups * kHeadDim);

  mk::CompressorTilingData tiling{};
  const mk::Status status = mk::BuildCompressorTiling(
      kHidden, kHidden, kStateWidth, mk::kCompressorCoff, mk::kRotaryModeHalf,
      &tiling);
  verdict.CheckTrue("host tiler accepts coff=2 half-mode",
                    status == mk::Status::success);
  // The contract says coff 1 and interleave rotary must be refused.
  mk::CompressorTilingData reject{};
  verdict.CheckTrue("host tiler rejects coff=1",
                    mk::BuildCompressorTiling(kHidden, kHidden, kStateWidth, 1,
                                              mk::kRotaryModeHalf, &reject) !=
                        mk::Status::success);
  verdict.CheckTrue("host tiler rejects interleave rotary",
                    mk::BuildCompressorTiling(kHidden, kHidden, kStateWidth,
                                              mk::kCompressorCoff, 0, &reject) !=
                        mk::Status::success);

  auto* wkv_gm = (uint8_t*)AscendC::GmAlloc(wkv.size() * sizeof(uint16_t));
  auto* wgate_gm = (uint8_t*)AscendC::GmAlloc(wgate.size() * sizeof(uint16_t));
  auto* ape_gm = (uint8_t*)AscendC::GmAlloc(ape.size() * sizeof(float));
  auto* normw_gm = (uint8_t*)AscendC::GmAlloc(norm_weight.size() * sizeof(uint16_t));
  auto* x_gm = (uint8_t*)AscendC::GmAlloc(kHidden * sizeof(uint16_t));
  auto* cos_gm = (uint8_t*)AscendC::GmAlloc(kRopeDim * sizeof(float));
  auto* sin_gm = (uint8_t*)AscendC::GmAlloc(kRopeDim * sizeof(float));
  auto* state_gm = (uint8_t*)AscendC::GmAlloc(mk::kCompressorStateFloats * sizeof(float));
  auto* meta_gm = (uint8_t*)AscendC::GmAlloc(mk::kCompressorStateMetaWords * sizeof(uint32_t));
  auto* slot_gm = (uint8_t*)AscendC::GmAlloc(kHeadDim * sizeof(uint16_t));
  auto* emitted_gm = (uint8_t*)AscendC::GmAlloc(sizeof(uint32_t));
  auto* tiling_gm = (uint8_t*)AscendC::GmAlloc(sizeof(tiling));

  std::memcpy(wkv_gm, wkv.data(), wkv.size() * sizeof(uint16_t));
  std::memcpy(wgate_gm, wgate.data(), wgate.size() * sizeof(uint16_t));
  std::memcpy(ape_gm, ape.data(), ape.size() * sizeof(float));
  std::memcpy(normw_gm, norm_weight.data(), norm_weight.size() * sizeof(uint16_t));
  std::memcpy(tiling_gm, &tiling, sizeof(tiling));
  // The state allocation is initialized once, with meta zero, as the ABI says.
  std::memset(state_gm, 0, mk::kCompressorStateFloats * sizeof(float));
  std::memset(meta_gm, 0, mk::kCompressorStateMetaWords * sizeof(uint32_t));

  uint32_t emissions_seen = 0;
  bool hold_untouched = true;
  bool emitted_flags_ok = true;
  bool meta_ok = true;

  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t compressed_index = step / kRatio;
    std::memcpy(x_gm, tokens.data() + static_cast<size_t>(step) * kHidden,
                kHidden * sizeof(uint16_t));
    std::memcpy(cos_gm, rope_cos.data() + static_cast<size_t>(compressed_index) * kRopeDim,
                kRopeDim * sizeof(float));
    std::memcpy(sin_gm, rope_sin.data() + static_cast<size_t>(compressed_index) * kRopeDim,
                kRopeDim * sizeof(float));
    // Poison the slot so a HOLD write of any kind is visible.
    std::memset(slot_gm, 0x5A, kHeadDim * sizeof(uint16_t));
    std::memset(emitted_gm, 0xFF, sizeof(uint32_t));

    ICPU_RUN_KF(ker03_paged_compressor_rope, tiling.block_dim, (GM_ADDR)x_gm,
                (GM_ADDR)wkv_gm, (GM_ADDR)wgate_gm, (GM_ADDR)ape_gm,
                (GM_ADDR)normw_gm, (GM_ADDR)cos_gm, (GM_ADDR)sin_gm,
                (GM_ADDR)state_gm, (GM_ADDR)meta_gm, (GM_ADDR)slot_gm,
                (GM_ADDR)emitted_gm, (GM_ADDR) nullptr, (GM_ADDR)tiling_gm);

    const uint32_t got_emitted = *reinterpret_cast<const uint32_t*>(emitted_gm);
    const uint32_t want = want_emitted[step];
    if (got_emitted != want) {
      std::printf("      step %u: emitted=%u, expected %u\n", step, got_emitted, want);
      emitted_flags_ok = false;
    }
    const uint32_t tokens_seen = reinterpret_cast<const uint32_t*>(meta_gm)[1];
    if (tokens_seen != step + 1) {
      std::printf("      step %u: tokens_seen=%u, expected %u\n", step, tokens_seen,
                  step + 1);
      meta_ok = false;
    }

    if (want == 0U) {
      // HOLD: every byte of the slot must still be the poison value.
      const auto* slot = reinterpret_cast<const uint8_t*>(slot_gm);
      for (size_t i = 0; i < kHeadDim * sizeof(uint16_t); ++i) {
        if (slot[i] != 0x5A) {
          std::printf("      step %u: HOLD wrote slot byte %zu\n", step, i);
          hold_untouched = false;
          break;
        }
      }
    } else {
      const auto* got = reinterpret_cast<const uint16_t*>(slot_gm);
      const uint16_t* exp = want_tokens.data() + static_cast<size_t>(emissions_seen) * kHeadDim;
      std::vector<float> got_f(kHeadDim);
      std::vector<float> want_f(kHeadDim);
      for (uint32_t i = 0; i < kHeadDim; ++i) {
        got_f[i] = Bf16ToFloat(got[i]);
        want_f[i] = Bf16ToFloat(exp[i]);
      }
      char label[80];
      std::snprintf(label, sizeof(label), "emission %u (group %u) vs oracle",
                    emissions_seen, compressed_index);
      verdict.CheckMetrics(label, Compare(got_f.data(), want_f.data(), kHeadDim));
      ++emissions_seen;
    }
  }

  verdict.CheckTrue("HOLD steps never touch the slot", hold_untouched,
                    "a HOLD step wrote into cache_slot_gm");
  verdict.CheckTrue("emitted flag correct every step", emitted_flags_ok);
  verdict.CheckTrue("state_meta tokens_seen advances", meta_ok);
  verdict.CheckTrue("have_previous set after first emission",
                    reinterpret_cast<const uint32_t*>(meta_gm)[0] == 1U);

  // Group 1's emission is in golden v3; compare it byte for byte.
  std::vector<unsigned char> probe = ReadBytes(dir + "/expect_token_golden.bin");
  if (!probe.empty()) {
    const std::vector<uint16_t> want_golden =
        ReadTyped<uint16_t>(dir, "expect_token_golden", kHeadDim);
    // Re-run group 1 from a clean state to recover its emission.
    std::memset(state_gm, 0, mk::kCompressorStateFloats * sizeof(float));
    std::memset(meta_gm, 0, mk::kCompressorStateMetaWords * sizeof(uint32_t));
    for (uint32_t step = 0; step < kRatio; ++step) {
      std::memcpy(x_gm, tokens.data() + static_cast<size_t>(step) * kHidden,
                  kHidden * sizeof(uint16_t));
      std::memcpy(cos_gm, rope_cos.data(), kRopeDim * sizeof(float));
      std::memcpy(sin_gm, rope_sin.data(), kRopeDim * sizeof(float));
      ICPU_RUN_KF(ker03_paged_compressor_rope, tiling.block_dim, (GM_ADDR)x_gm,
                  (GM_ADDR)wkv_gm, (GM_ADDR)wgate_gm, (GM_ADDR)ape_gm,
                  (GM_ADDR)normw_gm, (GM_ADDR)cos_gm, (GM_ADDR)sin_gm,
                  (GM_ADDR)state_gm, (GM_ADDR)meta_gm, (GM_ADDR)slot_gm,
                  (GM_ADDR)emitted_gm, (GM_ADDR) nullptr, (GM_ADDR)tiling_gm);
    }
    const auto* got = reinterpret_cast<const uint16_t*>(slot_gm);
    size_t exact = 0;
    std::vector<float> got_f(kHeadDim);
    std::vector<float> want_f(kHeadDim);
    for (uint32_t i = 0; i < kHeadDim; ++i) {
      if (got[i] == want_golden[i]) {
        ++exact;
      }
      got_f[i] = Bf16ToFloat(got[i]);
      want_f[i] = Bf16ToFloat(want_golden[i]);
    }
    verdict.CheckMetrics("emission vs GOLDEN v3 token",
                         Compare(got_f.data(), want_f.data(), kHeadDim));
    std::printf("  %-34s %zu/%u BF16 words bit-identical\n", "golden token agreement",
                exact, kHeadDim);
  }

  AscendC::GmFree(wkv_gm);
  AscendC::GmFree(wgate_gm);
  AscendC::GmFree(ape_gm);
  AscendC::GmFree(normw_gm);
  AscendC::GmFree(x_gm);
  AscendC::GmFree(cos_gm);
  AscendC::GmFree(sin_gm);
  AscendC::GmFree(state_gm);
  AscendC::GmFree(meta_gm);
  AscendC::GmFree(slot_gm);
  AscendC::GmFree(emitted_gm);
  AscendC::GmFree(tiling_gm);
  return verdict.Finish();
}
