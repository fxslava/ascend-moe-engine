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

}  // namespace

Status MakeDsv4MoeDecodeTopkTiling(std::uint32_t hidden, std::uint32_t inter,
                                   std::uint32_t num_experts,
                                   std::uint32_t top_k,
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
  // block_dim = top_k is the decode grid (see the header). The ceiling exists
  // because the grid is the slot count: a top_k above it would ask for more
  // AIVs per launch than this version plans for.
  if (top_k > kMoeBlockMaxTopK) {
    return Status::unsupported_shape;
  }
  out->mode = kMoeBlockModeDecode;
  out->top_k = top_k;
  out->total_tokens = 1u;
  out->block_dim = top_k;
  return Status::success;
}

Status MakeDsv4MoePrefillTiling(std::uint32_t hidden, std::uint32_t inter,
                                std::uint32_t num_experts,
                                std::uint32_t total_tokens,
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
  out->block_dim = kMoeBlockPrefillBlockDim;
  return Status::success;
}

}  // namespace moe::kernels::arch35
