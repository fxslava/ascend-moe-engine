// SIM ONLY -- host launch wrapper for the collapsed-grid sim kernel.
//
// Compiled by BiSheng as an ASC TU, for the reason in CLAUDE.md §3.2-3.3: the
// triple-chevron form is the only dispatch and BiSheng is what lowers it, so
// a host compiler cannot build this file.
//
// A deliberate near-copy of LaunchDsv4MoePrefillBatched in
// src/ops/dsv4_moe_expert/op_host/dsv4_moe_block.cpp: SAME tiling builder,
// SAME UB ceiling check, SAME staging helper, SAME dyn_ub_size formula, SAME
// FinishLaunch. Only the kernel symbol differs. That matters -- a shape this
// wrapper admits is exactly a shape the production wrapper would admit, so
// the collapsed grid is a property of the KERNEL and never of the tiling.
//
// The wrapper owns validation, the tiling staging buffer, the dispatch and
// its teardown. It does not allocate or copy operand data (§3.4).

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_block_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_block_inductive_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR tiling);

constexpr const char* kSimOp = "dsv4_moe_block_inductive_sim";

// Same ceiling check the production wrapper applies, and for the same reason:
// the geometry predicate leaves UB out because the byte-count helper is
// host-space and the kernel's device-side backstop cannot link it.
bool UbWithinCeiling(const Dsv4MoeBlockTilingData& t) noexcept {
  return Dsv4MoeBlockDynUbBytes(t.hidden, t.inter) <=
         kMoeBlockDynUbCeilingBytes;
}

}  // namespace

// Pass total_tokens == 1 to keep the simulated work minimal: the sim kernel
// keeps the production grid-stride token loop, so with one token only core 0
// does anything and the other cores fall straight out of the loop. That way
// no tiling field has to be overridden to shrink the grid.
Status LaunchDsv4MoeBlockInductiveSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, float* y_tokens_gm, aclrtStream stream) {
  if (x_tokens_gm == nullptr || w1_packed_gm == nullptr ||
      w1_scale_gm == nullptr || w3_packed_gm == nullptr ||
      w3_scale_gm == nullptr || w2_packed_gm == nullptr ||
      w2_scale_gm == nullptr || token_to_expert_gm == nullptr ||
      y_tokens_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  if (static_cast<const void*>(y_tokens_gm) ==
      static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoeBlockTilingData tiling{};
  // PASS THE OLD CONSTANT, NOT THE QUERIED COUNT. This is a CAModel gate, so
  // its grid must not depend on which machine's SKU it runs on -- a simulated
  // core count that changes under the test would make its cost and its
  // coverage irreproducible. 32 is what this gate has always launched.
  //
  // The builder now returns min(32, total_tokens) rather than a flat 32, so a
  // run with fewer tokens than cores stops launching cores that only idle.
  // That cannot change the result: the grid-stride map covers every token
  // exactly once for any block_dim, which tests/sim/test_gang_partition.cpp
  // proves on the host.
  const Status shape = MakeDsv4MoePrefillTiling(hidden, inter, num_experts,
                                                total_tokens, 32u, &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (!UbWithinCeiling(tiling)) {
    return Status::unsupported_shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kSimOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  const std::uint32_t dyn_ub =
      static_cast<std::uint32_t>(Dsv4MoeBlockDynUbBytes(hidden, inter));

  dsv4_moe_block_inductive_sim<<<tiling.block_dim, dyn_ub, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(w1_packed_gm),
      detail::Gm(w1_scale_gm), detail::Gm(w3_packed_gm),
      detail::Gm(w3_scale_gm), detail::Gm(w2_packed_gm),
      detail::Gm(w2_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kSimOp, stream);
}

}  // namespace moe::kernels::arch35
