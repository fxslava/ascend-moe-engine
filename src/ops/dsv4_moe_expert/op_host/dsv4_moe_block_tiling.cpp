#include "dsv4_moe_block_tiling.h"

namespace moe::kernels::arch35 {
namespace {

// The geometry both entry points share. Returns success, or the HONEST
// rejection (§2.6): invalid_argument for a caller mistake, unsupported_shape
// for a geometry this version does not implement.
Status FillCommon(std::uint32_t hidden, std::uint32_t inter,
                  std::uint32_t num_experts,
                  Dsv4MoeBlockTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // A zero or absurd expert count is a caller mistake, not a geometry: the
  // weight buffers are indexed by expert id, so there is no shape here to
  // decline implementing.
  if (num_experts == 0u) {
    return Status::invalid_argument;
  }
  if (!Dsv4MoeBlockGeometryAdmissible(hidden, inter)) {
    return Status::unsupported_shape;
  }
  if (hidden % kMoeBlockRowsPerTile != 0u ||
      inter % kMoeBlockRowsPerTile != 0u) {
    return Status::unsupported_shape;
  }

  // Expert strides, widened before every multiply (§2.7). These are the
  // whole reason the strides are explicit POD fields: the device must not
  // recompute them from caller extents.
  const std::uint64_t w13_packed_expert =
      static_cast<std::uint64_t>(inter) * (hidden / 2u);
  const std::uint64_t w13_scale_expert =
      static_cast<std::uint64_t>(inter) * (hidden / kMoeBlockGroupSize);
  const std::uint64_t w2_packed_expert =
      static_cast<std::uint64_t>(hidden) * (inter / 2u);
  const std::uint64_t w2_scale_expert =
      static_cast<std::uint64_t>(hidden) * (inter / kMoeBlockGroupSize);

  // The POD is 32-bit only (§2.1), so a stride that does not fit is a
  // geometry this version cannot express -- not a silent truncation.
  constexpr std::uint64_t kU32Max = 0xFFFFFFFFull;
  if (w13_packed_expert > kU32Max || w13_scale_expert > kU32Max ||
      w2_packed_expert > kU32Max || w2_scale_expert > kU32Max) {
    return Status::unsupported_shape;
  }
  // The largest byte offset the kernel forms is (num_experts - 1) * stride.
  // It is computed in uint64 on device, but a buffer that large is not
  // addressable as a single allocation here either, so decline it up front
  // rather than at the first wrong read.
  const std::uint64_t last = static_cast<std::uint64_t>(num_experts - 1u);
  if (last * w13_packed_expert > kU32Max ||
      last * w2_packed_expert > kU32Max) {
    return Status::unsupported_shape;
  }

  Dsv4MoeBlockTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.hidden = hidden;
  t.inter = inter;
  t.num_experts = num_experts;
  t.rows_per_block = kMoeBlockRowsPerTile;
  t.group_size = kMoeBlockGroupSize;
  t.low_nibble_first = 1u;  // the kernel rejects 0
  t.w13_packed_row_stride = hidden / 2u;
  t.w13_scale_row_stride = hidden / kMoeBlockGroupSize;
  t.w2_packed_row_stride = inter / 2u;
  t.w2_scale_row_stride = inter / kMoeBlockGroupSize;
  t.w13_packed_expert_stride = static_cast<std::uint32_t>(w13_packed_expert);
  t.w13_scale_expert_stride = static_cast<std::uint32_t>(w13_scale_expert);
  t.w2_packed_expert_stride = static_cast<std::uint32_t>(w2_packed_expert);
  t.w2_scale_expert_stride = static_cast<std::uint32_t>(w2_scale_expert);
  t.swiglu_limit = kMoeBlockSwigluLimit;
  *out = t;
  return Status::success;
}

// The AIV count to plan the grid against: the queried value when it is
// usable, else the smallest production part's Vector count. Clamped so a
// nonsense capability reading cannot become an absurd grid.
std::uint32_t ResolveAivCount(std::uint32_t aiv_count) noexcept {
  if (aiv_count == 0u || aiv_count > kMoeBlockAivMax) {
    return kMoeBlockAivFallback;
  }
  return aiv_count;
}

}  // namespace

Status MakeDsv4MoeDecodeTopkTiling(std::uint32_t hidden, std::uint32_t inter,
                                   std::uint32_t num_experts,
                                   std::uint32_t top_k,
                                   std::uint32_t aiv_count,
                                   Dsv4MoeBlockTilingData* out) noexcept {
  const Status common = FillCommon(hidden, inter, num_experts, out);
  if (common != Status::success) {
    return common;
  }
  // top_k == 0 is a caller mistake. top_k > num_experts cannot be satisfied
  // by any routing, so it is also a caller mistake rather than a shape.
  if (top_k == 0u || top_k > num_experts) {
    return Status::invalid_argument;
  }
  // The ceiling is on the ROUTED SLOT COUNT, which is what the workspace and
  // the slot-to-gang map are sized by. It is no longer the grid.
  if (top_k > kMoeBlockMaxTopK) {
    return Status::unsupported_shape;
  }

  // GANG DERIVATION. gang_size = floor(aiv / top_k), at least 1, so the grid
  // gang_size * top_k never exceeds the device. The aiv % top_k remainder is
  // deliberately left unlaunched: every launched core must reach the
  // cross-core barrier, so a core holding a partial slot would have to be
  // carried through both phases to keep the barrier counts equal, and an
  // unlaunched core costs nothing.
  const std::uint32_t aiv = ResolveAivCount(aiv_count);
  std::uint32_t gang_size = aiv / top_k;
  if (gang_size < kMoeBlockMinGangSize) {
    gang_size = kMoeBlockMinGangSize;  // top_k > aiv: fall back to 1 core/slot
  }

  out->mode = kMoeBlockModeDecode;
  out->top_k = top_k;
  out->total_tokens = 1u;
  out->gang_size = gang_size;
  out->block_dim = gang_size * top_k;
  return Status::success;
}

Status MakeDsv4MoePrefillTiling(std::uint32_t hidden, std::uint32_t inter,
                                std::uint32_t num_experts,
                                std::uint32_t total_tokens,
                                std::uint32_t aiv_count,
                                Dsv4MoeBlockTilingData* out) noexcept {
  const Status common = FillCommon(hidden, inter, num_experts, out);
  if (common != Status::success) {
    return common;
  }
  if (total_tokens == 0u) {
    return Status::invalid_argument;
  }
  out->mode = kMoeBlockModePrefill;
  // Each token carries exactly one expert on the prefill path; the router
  // weighting happens in the combine stage above this operator.
  out->top_k = 1u;
  out->total_tokens = total_tokens;
  // A core owns whole tokens here, so there is no gang and no slot sharing.
  out->gang_size = 1u;
  // Every AIV, but never more cores than there are tokens: the body is a
  // grid-stride loop, so a core past the last token would do nothing except
  // occupy a slot in the launch.
  const std::uint32_t aiv = ResolveAivCount(aiv_count);
  out->block_dim = total_tokens < aiv ? total_tokens : aiv;
  return Status::success;
}

}  // namespace moe::kernels::arch35
