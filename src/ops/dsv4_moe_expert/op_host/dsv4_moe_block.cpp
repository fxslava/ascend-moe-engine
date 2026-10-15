// -----------------------------------------------------------------------------
// dsv4_moe_block.cpp -- host-side launch wrappers for the two whole-block
// entry points. Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
//
// ONE DISPATCH EACH. Both kernels give one core a whole expert-token pair, so
// `activated` is a UB buffer with no sharer and there is no inter-phase
// barrier to express -- which is what collapses the retired
// LaunchDsv4MoeExpert's two dispatches (and, for the decode path, the six
// per-expert wrapper calls plus the aclnn weight-broadcast/mul/add trio above
// it) into a single launch.
//
// NEITHER WRAPPER HAS A WORKSPACE. `activated` never reaches GM, so there is
// no inter-stage operand left to place. This is the one respect in which the
// new ABI is strictly simpler than the old one.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_block_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_decode_topk(
    GM_ADDR x, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR topk_expert_ids, GM_ADDR routing_weights, GM_ADDR y,
    GM_ADDR activated_ws, GM_ADDR tiling);

extern "C" __global__ __aicore__ void dsv4_moe_prefill_batched(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR tiling);

constexpr const char* kDecodeOp = "dsv4_moe_decode_topk";
constexpr const char* kPrefillOp = "dsv4_moe_prefill_batched";

// Shared by both wrappers: the UB ceiling. This is the ONLY place it is
// enforced -- the geometry predicate deliberately leaves it out, because the
// byte-count helper is host-space and the kernel's device-side backstop
// cannot link it.
bool UbWithinCeiling(const Dsv4MoeBlockTilingData& t) noexcept {
  return Dsv4MoeBlockDynUbBytes(t.hidden, t.inter) <=
         kMoeBlockDynUbCeilingBytes;
}

}  // namespace

Status LaunchDsv4MoeDecodeTopk(
    const float* x_gm, const Mxfp4Pair* w1_packed_gm, const E8M0* w1_scale_gm,
    const Mxfp4Pair* w3_packed_gm, const E8M0* w3_scale_gm,
    const Mxfp4Pair* w2_packed_gm, const E8M0* w2_scale_gm,
    const std::int32_t* topk_expert_ids_gm, const float* routing_weights_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t top_k, float* y_gm, float* activated_ws_gm,
    aclrtStream stream) {
  if (x_gm == nullptr || w1_packed_gm == nullptr || w1_scale_gm == nullptr ||
      w3_packed_gm == nullptr || w3_scale_gm == nullptr ||
      w2_packed_gm == nullptr || w2_scale_gm == nullptr ||
      topk_expert_ids_gm == nullptr || routing_weights_gm == nullptr ||
      y_gm == nullptr || activated_ws_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // The gang writes `activated` here and reads it back after the barrier, so
  // an overlap with x or y is a self-corrupting launch, not a slow one.
  if (static_cast<const void*>(activated_ws_gm) ==
          static_cast<const void*>(x_gm) ||
      static_cast<const void*>(activated_ws_gm) ==
          static_cast<const void*>(y_gm)) {
    return Status::invalid_argument;
  }
  // y is ACCUMULATED INTO, so it must not alias the token it reads.
  if (static_cast<const void*>(y_gm) == static_cast<const void*>(x_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoeBlockTilingData tiling{};
  // The grid comes from the device, not from a literal: see
  // detail::QueryAivCoreCount(). A 0 means the query failed and the builder
  // takes its conservative fallback.
  const Status shape = MakeDsv4MoeDecodeTopkTiling(
      hidden, inter, num_experts, top_k, detail::QueryAivCoreCount(), &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (!UbWithinCeiling(tiling)) {
    return Status::unsupported_shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kDecodeOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // The second chevron slot is dyn_ub_size (uint32 BYTES), not an l2ctrl
  // pointer: the 950PR SIMD/SIMT hybrid launch is
  // kernel<<<block_num, dyn_ub_size, stream>>>. Both entry points hand
  // __ubuf__ pointers to a SIMT VF, so a declared size of 0 leaves the VF
  // addressing a region the launch never reserved -- the error-341 / 507035
  // class. CLAUDE.md §3.1 prescribes nullptr for this slot; the user granted
  // an explicit exception for the dynamic-UB argument.
  const std::uint32_t dyn_ub =
      static_cast<std::uint32_t>(Dsv4MoeBlockDynUbBytes(hidden, inter));

  // block_dim == gang_size * top_k: a gang of AIVs per routed slot, splitting
  // that expert's row tiles. See dsv4_moe_block_tiling.h for why this
  // reverses the earlier one-core-per-slot grid.
  dsv4_moe_decode_topk<<<tiling.block_dim, dyn_ub, stream>>>(
      detail::Gm(x_gm), detail::Gm(w1_packed_gm), detail::Gm(w1_scale_gm),
      detail::Gm(w3_packed_gm), detail::Gm(w3_scale_gm),
      detail::Gm(w2_packed_gm), detail::Gm(w2_scale_gm),
      detail::Gm(topk_expert_ids_gm), detail::Gm(routing_weights_gm),
      detail::Gm(y_gm), detail::Gm(activated_ws_gm),
      detail::Gm(staging.get()));

  return detail::FinishLaunch(kDecodeOp, stream);
}

Status LaunchDsv4MoePrefillBatched(
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
  // Each token's output row is written while its input row is still being
  // read by the same core, so in-place is not supported.
  if (static_cast<const void*>(y_tokens_gm) ==
      static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoeBlockTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillTiling(
      hidden, inter, num_experts, total_tokens, detail::QueryAivCoreCount(),
      &tiling);
  if (shape != Status::success) {
    return shape;
  }
  if (!UbWithinCeiling(tiling)) {
    return Status::unsupported_shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kPrefillOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  const std::uint32_t dyn_ub =
      static_cast<std::uint32_t>(Dsv4MoeBlockDynUbBytes(hidden, inter));

  dsv4_moe_prefill_batched<<<tiling.block_dim, dyn_ub, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(w1_packed_gm),
      detail::Gm(w1_scale_gm), detail::Gm(w3_packed_gm),
      detail::Gm(w3_scale_gm), detail::Gm(w2_packed_gm),
      detail::Gm(w2_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(staging.get()));

  return detail::FinishLaunch(kPrefillOp, stream);
}

}  // namespace moe::kernels::arch35
