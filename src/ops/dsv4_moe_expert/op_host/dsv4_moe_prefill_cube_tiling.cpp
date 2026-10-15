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
  // The Cube N chunks must divide their axes: a ragged final chunk would need
  // SetTail() before Iterate, which this version does not drive.
  if (inter % kPrefillCubeInterTile != 0u ||
      hidden % kPrefillCubeHiddenTile != 0u) {
    return Status::unsupported_shape;
  }
  // The reduction axes must be whole MXFP4 groups, or a dequantized row would
  // end mid-group and the scale index would run past the row.
  if (hidden % kPrefillCubeGroupSize != 0u ||
      inter % kPrefillCubeGroupSize != 0u) {
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

  // The workspace is indexed with uint64 on device, but a single allocation
  // this large is not addressable either, so decline it here rather than at
  // the first wrong store.
  //
  // Spelled locally rather than calling Dsv4MoePrefillCubeWorkspaceBytes():
  // that helper is public ABI in launch.hpp, which pulls in <acl/acl.h>, and
  // this TU must stay ACL-free because the tiling header it implements is
  // included by the device side (§2.4). Only the ORDER OF MAGNITUDE matters
  // here -- the authoritative size, including the KFC reserve, is the public
  // helper, and the caller allocates from that.
  const std::uint64_t x_bf16 =
      static_cast<std::uint64_t>(total_tokens) * hidden * 2u;
  const std::uint64_t w_bf16 =
      3ull * static_cast<std::uint64_t>(inter) * hidden * 2u;
  if (x_bf16 + w_bf16 > (1ull << 40)) {
    return Status::unsupported_shape;
  }

  Dsv4MoePrefillCubeTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.hidden = hidden;
  t.inter = inter;
  t.num_experts = num_experts;
  t.total_tokens = total_tokens;
  t.m_tile = kPrefillCubeMTile;
  t.inter_tile = kPrefillCubeInterTile;
  t.hidden_tile = kPrefillCubeHiddenTile;
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
