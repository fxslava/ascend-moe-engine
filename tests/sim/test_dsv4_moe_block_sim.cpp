// dav-3510 CAModel execution test for the two whole-block entry points.
//
//   dsv4_moe_decode_topk      one token, top_k routed experts, atomic combine
//   dsv4_moe_prefill_batched  several tokens, one expert each
//
// Both run the REAL kernels through the REAL Launch* wrappers against the
// CPU golden in src/emu. References are generated in memory from a printed
// seed, never from a disk fixture (CLAUDE.md §4.5).
//
// GEOMETRY IS CHOSEN FOR SIMULATOR COST, and the reason is worth recording.
// These kernels are monolithic: one core runs a whole expert for a whole
// token. So the widths are the smallest the host gate admits -- 512, the
// reduction quantum -- and the slot and token counts are small.
//
// MEASURED COST, 2026-10-15: this suite takes ~4,930 s (~82 min) at the
// constants below, over ~1.5 M simulated cycles.
//
// An earlier version of this comment claimed CAModel wall time scales with
// TOTAL simulated work, citing the single-expert suite (786 K MACs -> ~260 s,
// 1.57 M -> ~540 s) and scaling linearly. That predicts ~18 min here and is
// wrong; the opposite per-core reading in tests/sim/CMakeLists.txt predicts
// ~4.6 h and is also wrong. The observed ~82 min sits between them.
//
// The common error was extrapolating across KERNELS: the single-expert suite
// is two-phase over 32 cores, this one is monolithic over 2, and CAModel
// joins simulated threads in parallel, so neither rate transfers. Time a
// suite at its own shape rather than deriving it from another's. The LANE geometry at 512 is degenerate
// (groups_per_lane = 1, where the coalesced and the old contiguous-block maps
// coincide); that is fine HERE because this suite is testing expert-stride
// addressing, routing and the UB-resident `activated`, not the lane map,
// which Dsv4MoeExpertSim covers at 1024 and tests/emu covers on the host.

#include "dsv4_moe_block_sim_common.hpp"

namespace {


void RunTest() {
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<float> x_distribution(0.0F, 0.25F);

  const ExpertBank w1 = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w3 = MakeBank(kInter, kHidden, kExperts, rng);
  const ExpertBank w2 = MakeBank(kHidden, kInter, kExperts, rng);

  std::vector<moe::emu::Mxfp4Matrix> v1(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v3(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v2(kExperts);
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    v1[e] = w1.View(e);
    v3[e] = w3.View(e);
    v2[e] = w2.View(e);
  }

  // --- goldens -----------------------------------------------------------
  // Routed slots are deliberately neither sorted nor the identity, so a
  // kernel that ignored topk_expert_ids and used the slot index would fail.
  const std::vector<std::int32_t> topk_ids = {3, 1};
  const std::vector<float> routing_weights = {0.625F, 0.375F};
  std::vector<float> x(kHidden);
  for (float& value : x) value = x_distribution(rng);

  std::vector<float> scratch;
  std::vector<float> expected_decode(kHidden);
  moe::emu::moe_decode_topk_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x.data(), topk_ids.data(),
      routing_weights.data(), static_cast<int>(kTopK),
      static_cast<int>(kExperts), expected_decode.data(), kLimit, &scratch);
  Require(NonDegenerate(expected_decode),
          "decode golden is zero, constant, or non-finite");

  // Both tokens routed to DIFFERENT experts, so a kernel that applied one
  // expert to every token fails.
  const std::vector<std::int32_t> token_to_expert = {2, 0};
  std::vector<float> x_tokens(static_cast<std::size_t>(kTokens) * kHidden);
  for (float& value : x_tokens) value = x_distribution(rng);
  std::vector<float> expected_prefill(static_cast<std::size_t>(kTokens) * kHidden);
  moe::emu::moe_prefill_batched_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x_tokens.data(), token_to_expert.data(),
      static_cast<int>(kTokens), static_cast<int>(kExperts),
      expected_prefill.data(), kLimit, &scratch);
  Require(NonDegenerate(expected_prefill),
          "prefill golden is zero, constant, or non-finite");

  // --- device ------------------------------------------------------------
  std::vector<float> actual_decode(kHidden);
  std::vector<float> actual_prefill(static_cast<std::size_t>(kTokens) * kHidden);
  {
    Runtime runtime;
    {
      DeviceBuffer d_w1p(w1.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w1s(w1.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w3p(w3.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w3s(w3.scales.size() * sizeof(E8M0));
      DeviceBuffer d_w2p(w2.packed.size() * sizeof(Mxfp4Pair));
      DeviceBuffer d_w2s(w2.scales.size() * sizeof(E8M0));
      DeviceBuffer d_x(x.size() * sizeof(float));
      DeviceBuffer d_ids(topk_ids.size() * sizeof(std::int32_t));
      DeviceBuffer d_rw(routing_weights.size() * sizeof(float));
      // The decode gang's `activated` plane: [top_k, inter] FP32, caller-owned.
      DeviceBuffer d_actws(static_cast<std::size_t>(
          moe::kernels::arch35::Dsv4MoeDecodeWorkspaceBytes(kTopK, kInter)));
      auto d_y = MakeGuardedOutput(kHidden);
      DeviceBuffer d_xt(x_tokens.size() * sizeof(float));
      DeviceBuffer d_t2e(token_to_expert.size() * sizeof(std::int32_t));
      auto d_yt = MakeGuardedOutput(actual_prefill.size());

      CopyToDevice(d_w1p, w1.packed);
      CopyToDevice(d_w1s, w1.scales);
      CopyToDevice(d_w3p, w3.packed);
      CopyToDevice(d_w3s, w3.scales);
      CopyToDevice(d_w2p, w2.packed);
      CopyToDevice(d_w2s, w2.scales);
      CopyToDevice(d_x, x);
      CopyToDevice(d_ids, topk_ids);
      CopyToDevice(d_rw, routing_weights);
      CopyToDevice(d_xt, x_tokens);
      CopyToDevice(d_t2e, token_to_expert);

      // DECODE ACCUMULATES: the caller zeroes y, per the wrapper contract.
      // Only the LOGICAL extent -- the guard tail past it stays poisoned, or
      // it could not witness a store past the end.
      RequireAcl(aclrtMemset(d_y->get(), d_y->bytes(), 0,
                             static_cast<std::size_t>(kHidden) * sizeof(float)),
                 "aclrtMemset(decode y, logical extent only)");

      LaunchWatchdog watchdog(kLaunchBudgetSeconds, "[dsv4_moe_block_sim]");

      const auto decode_st = moe::kernels::arch35::LaunchDsv4MoeDecodeTopk(
          static_cast<const float*>(d_x.get()),
          static_cast<const Mxfp4Pair*>(d_w1p.get()),
          static_cast<const E8M0*>(d_w1s.get()),
          static_cast<const Mxfp4Pair*>(d_w3p.get()),
          static_cast<const E8M0*>(d_w3s.get()),
          static_cast<const Mxfp4Pair*>(d_w2p.get()),
          static_cast<const E8M0*>(d_w2s.get()),
          static_cast<const std::int32_t*>(d_ids.get()),
          static_cast<const float*>(d_rw.get()), kHidden, kInter, kExperts,
          kTopK, static_cast<float*>(d_y->get()),
          static_cast<float*>(d_actws.get()), runtime.stream());
      if (decode_st != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoeDecodeTopk", decode_st));
      }
      watchdog.Arm("dsv4_moe_decode_topk");
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(decode)");
      watchdog.Disarm();
      RequireGuardIntact(*d_y, kHidden, "decode y");
      CopyFromDevice(actual_decode, *d_y);

      const auto prefill_st = moe::kernels::arch35::LaunchDsv4MoePrefillBatched(
          static_cast<const float*>(d_xt.get()),
          static_cast<const Mxfp4Pair*>(d_w1p.get()),
          static_cast<const E8M0*>(d_w1s.get()),
          static_cast<const Mxfp4Pair*>(d_w3p.get()),
          static_cast<const E8M0*>(d_w3s.get()),
          static_cast<const Mxfp4Pair*>(d_w2p.get()),
          static_cast<const E8M0*>(d_w2s.get()),
          static_cast<const std::int32_t*>(d_t2e.get()), kHidden, kInter,
          kExperts, kTokens, static_cast<float*>(d_yt->get()),
          runtime.stream());
      if (prefill_st != moe::kernels::arch35::Status::success) {
        Fail(LaunchFailure("LaunchDsv4MoePrefillBatched", prefill_st));
      }
      watchdog.Arm("dsv4_moe_prefill_batched");
      RequireAcl(aclrtSynchronizeStream(runtime.stream()),
                 "aclrtSynchronizeStream(prefill)");
      watchdog.Disarm();
      RequireGuardIntact(*d_yt, actual_prefill.size(), "prefill y");
      CopyFromDevice(actual_prefill, *d_yt);

      // A launch can return ACL_SUCCESS and still have faulted on device.
      const DumpCensus dumps = ExceptionDumps();
      if (dumps.non_empty > 0) {
        Fail("device left " + std::to_string(dumps.non_empty) +
             " non-empty excp_log*.dump file(s) of " +
             std::to_string(dumps.files) +
             "; a launch faulted even though ACL reported success");
      }
    }
    runtime.Close();
  }

  const Diff decode_diff = Compare(actual_decode, expected_decode);
  const Diff prefill_diff = Compare(actual_prefill, expected_prefill);
  std::printf("Seed: 0x%llX\n", static_cast<unsigned long long>(kSeed));
  std::printf(
      "Executed dsv4_moe_decode_topk (block_dim=%u, slots={3,1}) and "
      "dsv4_moe_prefill_batched (tokens=%u, experts={2,0}) via AscendCL: "
      "hidden=%u inter=%u num_experts=%u\n",
      kTopK, kTokens, kHidden, kInter, kExperts);
  std::printf(
      "Decode max abs diff: %.8g; prefill max abs diff: %.8g "
      "(abs/rel tolerance %.2g/%.2g)\n",
      decode_diff.max_abs, prefill_diff.max_abs, kAbsTolerance, kRelTolerance);
  Require(decode_diff.within_tolerance,
          "device decode differs from CPU golden beyond tolerance");
  Require(prefill_diff.within_tolerance,
          "device prefill differs from CPU golden beyond tolerance");
}
}  // namespace

int main() {
  try {
    RunTest();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[FAIL] Dsv4MoeBlockCamodel: %s\n", error.what());
    return 1;
  }
  std::puts("[PASS] Dsv4MoeBlockCamodel");
  return 0;
}
