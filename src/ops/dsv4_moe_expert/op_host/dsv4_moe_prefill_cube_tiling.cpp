#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35 {
namespace {

std::uint32_t ResolveAivCount(std::uint32_t aiv_count) noexcept {
  if (aiv_count == 0u || aiv_count > kPrefillCubeAivMax) {
    return kPrefillCubeAivFallback;
  }
  return aiv_count;
}

}  // namespace

Status MakeDsv4MoePrefillCubeTiling(std::uint32_t hidden, std::uint32_t inter,
                                    std::uint32_t num_experts,
                                    std::uint32_t total_tokens,
                                    std::uint32_t aiv_count,
                                    Dsv4MoePrefillCubeTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (num_experts == 0u || total_tokens == 0u) {
    return Status::invalid_argument;
  }
  if (!Dsv4MoePrefillCubeAdmissible(hidden, inter)) {
    return Status::unsupported_shape;
  }
  // EVERY sub-tile extent must divide its axis. Four conditions, not two,
  // because the two-phase schedule splits both axes of both phases:
  //
  //   phase 1   inter  % R_SUB   output columns of `act` per pass
  //             hidden % K_SUB   contraction blocks of the gate/up GEMMs
  //   phase 2   hidden % H_SUB   output columns of Y per pass
  //             inter  % R_SUB2  contraction blocks of the down GEMM
  //
  // A ragged sub-tile on any of them would need the burst extents and the
  // Matmul tail recomputed per step, which this version does not drive -- the
  // only raggedness it handles is the TOKEN tail, where SetTail(rows, ...)
  // and the per-row burst block count already carry it.
  if (inter % kPrefillCubeRSub != 0u || hidden % kPrefillCubeKSub != 0u ||
      hidden % kPrefillCubeHSub != 0u || inter % kPrefillCubeRSub2 != 0u) {
    return Status::unsupported_shape;
  }
  // The reduction axes must be whole MXFP4 groups, or a dequantized sub-tile
  // would end mid-group and the scale index would run past it.
  if (hidden % kPrefillCubeGroupSize != 0u ||
      inter % kPrefillCubeGroupSize != 0u) {
    return Status::unsupported_shape;
  }
  // The strided bursts count their block length in 32-BYTE units, so each
  // sub-tile row must be a whole number of them on both sides of every copy:
  //
  //   X      K_SUB floats                       (k_sub * 4) % 32
  //   Y      H_SUB floats                       (h_sub * 4) % 32
  //   W      half a row of K_SUB / R_SUB2 nibbles
  //
  // and the SOURCE strides likewise, which is what brings `hidden` and
  // `inter` into these conditions rather than the sub-tile extents alone. The
  // sub-tile halves are static_asserted in the header; these are the ones
  // that depend on the caller's geometry.
  if (((hidden - kPrefillCubeKSub) * 4u) % 32u != 0u ||
      ((hidden - kPrefillCubeHSub) * 4u) % 32u != 0u ||
      ((hidden - kPrefillCubeKSub) / 2u) % 32u != 0u ||
      ((inter - kPrefillCubeRSub2) / 2u) % 32u != 0u) {
    return Status::unsupported_shape;
  }

  // Expert strides, widened before every multiply (§2.7).
  const std::uint64_t w13_packed_expert =
      static_cast<std::uint64_t>(inter) * (hidden / 2u);
  const std::uint64_t w13_scale_expert =
      static_cast<std::uint64_t>(inter) * (hidden / kPrefillCubeGroupSize);
  const std::uint64_t w2_packed_expert =
      static_cast<std::uint64_t>(hidden) * (inter / 2u);
  const std::uint64_t w2_scale_expert =
      static_cast<std::uint64_t>(hidden) * (inter / kPrefillCubeGroupSize);

  constexpr std::uint64_t kU32Max = 0xFFFFFFFFull;
  if (w13_packed_expert > kU32Max || w13_scale_expert > kU32Max ||
      w2_packed_expert > kU32Max || w2_scale_expert > kU32Max) {
    return Status::unsupported_shape;
  }
  const std::uint64_t last = static_cast<std::uint64_t>(num_experts - 1u);
  if (last * w13_packed_expert > kU32Max ||
      last * w2_packed_expert > kU32Max) {
    return Status::unsupported_shape;
  }

  // THERE IS NO WORKSPACE SIZE TO BOUND ANY MORE. The predecessor checked
  // that its three dequantized planes stayed addressable, because at the
  // production geometry they were 48 MiB and grew with the problem. The
  // two-phase schedule has no planes: the workspace is the Matmul KFC message
  // queue reserve, a constant. Recorded rather than silently deleted, because
  // the absence of the check is now a property of the schedule.

  Dsv4MoePrefillCubeTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.hidden = hidden;
  t.inter = inter;
  t.num_experts = num_experts;
  t.total_tokens = total_tokens;
  t.m_tile = kPrefillCubeMTile;
  t.r_sub = kPrefillCubeRSub;
  t.k_sub = kPrefillCubeKSub;
  t.h_sub = kPrefillCubeHSub;
  t.r_sub2 = kPrefillCubeRSub2;
  t.group_size = kPrefillCubeGroupSize;
  t.low_nibble_first = 1u;
  t.w13_packed_expert_stride = static_cast<std::uint32_t>(w13_packed_expert);
  t.w13_scale_expert_stride = static_cast<std::uint32_t>(w13_scale_expert);
  t.w2_packed_expert_stride = static_cast<std::uint32_t>(w2_packed_expert);
  t.w2_scale_expert_stride = static_cast<std::uint32_t>(w2_scale_expert);
  t.swiglu_limit = kPrefillCubeSwigluLimit;

  // One token TILE per core, grid-striding, and never more cores than tiles.
  const std::uint32_t tiles =
      (total_tokens + kPrefillCubeMTile - 1u) / kPrefillCubeMTile;
  const std::uint32_t aiv = ResolveAivCount(aiv_count);
  t.block_dim = tiles < aiv ? tiles : aiv;

  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
