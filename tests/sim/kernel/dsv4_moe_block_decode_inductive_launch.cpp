// SIM ONLY -- host launch wrapper for the collapsed GANG DECODE sim kernel.
//
// Compiled by BiSheng as an ASC TU, for the reason in CLAUDE.md §3.2-3.3: the
// triple-chevron form is the only dispatch and BiSheng is what lowers it, so
// a host compiler cannot build this file.
//
// A deliberate near-copy of LaunchDsv4MoeDecodeTopk in
// src/ops/dsv4_moe_expert/op_host/dsv4_moe_block.cpp: SAME tiling builder,
// SAME UB ceiling check, SAME staging helper, SAME dyn_ub_size formula, SAME
// FinishLaunch, SAME aliasing rejections. Only the kernel symbol differs. That
// matters -- a shape this wrapper admits is exactly a shape the production
// wrapper would admit, so the collapsed grid is a property of the KERNEL and
// never of the tiling.
//
// ONE DEVIATION, AND IT IS THE POINT OF THE GATE: `aiv_count` is an explicit
// parameter here instead of detail::QueryAivCoreCount(). Under CAModel the
// capability query reports the simulated part, and the grid has to be pinned
// to a small, reproducible gang -- a simulated core count that changes with
// the host's SKU would make the run's cost and its coverage irreproducible.
// The production wrapper queries; this one is told. Everything downstream of
// the builder is identical, so the gang arithmetic under test is production's.
//
// The wrapper owns validation, the tiling staging buffer, the dispatch and
// its teardown. It does not allocate or copy operand data (§3.4) -- including
// `activated_ws`, which the caller owns and pre-fills with the seed.

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_block_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_block_decode_inductive_sim(
    GM_ADDR x, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR topk_expert_ids, GM_ADDR routing_weights, GM_ADDR y,
    GM_ADDR activated_ws, GM_ADDR tiling);

constexpr const char* kSimOp = "dsv4_moe_block_decode_inductive_sim";

bool UbWithinCeiling(const Dsv4MoeBlockTilingData& t) noexcept {
  return Dsv4MoeBlockDynUbBytes(t.hidden, t.inter) <=
         kMoeBlockDynUbCeilingBytes;
}

}  // namespace

// `aiv_count` pins the gang: block_dim comes back as
// floor(aiv_count / top_k) * top_k. The caller reports both so the run's grid
// is on stdout rather than inferred.
Status LaunchDsv4MoeBlockDecodeInductiveSim(
    const float* x_gm, const Mxfp4Pair* w1_packed_gm, const E8M0* w1_scale_gm,
    const Mxfp4Pair* w3_packed_gm, const E8M0* w3_scale_gm,
    const Mxfp4Pair* w2_packed_gm, const E8M0* w2_scale_gm,
    const std::int32_t* topk_expert_ids_gm, const float* routing_weights_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t top_k, std::uint32_t aiv_count, float* y_gm,
    float* activated_ws_gm, Dsv4MoeBlockTilingData* tiling_out,
    aclrtStream stream) {
  if (x_gm == nullptr || w1_packed_gm == nullptr || w1_scale_gm == nullptr ||
      w3_packed_gm == nullptr || w3_scale_gm == nullptr ||
      w2_packed_gm == nullptr || w2_scale_gm == nullptr ||
      topk_expert_ids_gm == nullptr || routing_weights_gm == nullptr ||
      y_gm == nullptr || activated_ws_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  if (static_cast<const void*>(y_gm) == static_cast<const void*>(x_gm)) {
    return Status::invalid_argument;
  }
  if (static_cast<const void*>(activated_ws_gm) ==
          static_cast<const void*>(x_gm) ||
      static_cast<const void*>(activated_ws_gm) ==
          static_cast<const void*>(y_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoeBlockTilingData tiling{};
  const Status shape = MakeDsv4MoeDecodeTopkTiling(hidden, inter, num_experts,
                                                   top_k, aiv_count, &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (!UbWithinCeiling(tiling)) {
    return Status::unsupported_shape;
  }
  if (tiling_out != nullptr) {
    *tiling_out = tiling;  // so the driver can print and mirror the grid
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kSimOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  const std::uint32_t dyn_ub =
      static_cast<std::uint32_t>(Dsv4MoeBlockDynUbBytes(hidden, inter));

  dsv4_moe_block_decode_inductive_sim<<<tiling.block_dim, dyn_ub, stream>>>(
      detail::Gm(x_gm), detail::Gm(w1_packed_gm), detail::Gm(w1_scale_gm),
      detail::Gm(w3_packed_gm), detail::Gm(w3_scale_gm),
      detail::Gm(w2_packed_gm), detail::Gm(w2_scale_gm),
      detail::Gm(topk_expert_ids_gm), detail::Gm(routing_weights_gm),
      detail::Gm(y_gm), detail::Gm(activated_ws_gm),
      detail::Gm(staging.get()));

  return detail::FinishLaunch(kSimOp, stream);
}

}  // namespace moe::kernels::arch35
