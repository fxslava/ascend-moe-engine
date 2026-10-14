#include "dsv4_moe_expert_tiling.h"

namespace moe::kernels::arch35 {

Status MakeDsv4MoeExpertTiling(std::uint32_t hidden, std::uint32_t inter,
                               std::uint32_t w13_packed_row_stride,
                               std::uint32_t w13_scale_row_stride,
                               std::uint32_t w2_packed_row_stride,
                               std::uint32_t w2_scale_row_stride,
                               Dsv4MoeExpertTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // w1/w3 are [inter, hidden], w2 is [hidden, inter]. The admissible set and
  // the reason the reduction quantum is load-bearing both live on
  // Dsv4ExpertGeometryAdmissible() in the header, which the kernel's
  // TilingRejected() backstop consumes verbatim. This is the GATE (§2.5):
  // the kernel can only reject by returning early, which produces a zeroed
  // output rather than an error.
  if (!Dsv4ExpertGeometryAdmissible(hidden, inter)) {
    return Status::unsupported_shape;
  }
  // Exact strides. Never infer a shared FP8-style 128x128 scale grid: these
  // are MXFP4 group-32 scales, one byte per 32 LOGICAL columns.
  if (w13_packed_row_stride != hidden / 2u ||
      w13_scale_row_stride != hidden / kDsv4ExpertGroupSize ||
      w2_packed_row_stride != inter / 2u ||
      w2_scale_row_stride != inter / kDsv4ExpertGroupSize) {
    return Status::unsupported_shape;
  }
  // Both legs are tiled 32 rows at a time, and the tail-tile path writes a
  // padding row rather than reading past the end -- but an exact multiple is
  // what v1 admits, so require it instead of relying on that path.
  if (hidden % kDsv4ExpertRowsPerTile != 0u ||
      inter % kDsv4ExpertRowsPerTile != 0u) {
    return Status::unsupported_shape;
  }

  Dsv4MoeExpertTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = kDsv4ExpertBlockDim;
  t.hidden = hidden;
  t.inter = inter;
  t.rows_per_block = kDsv4ExpertRowsPerTile;
  t.group_size = kDsv4ExpertGroupSize;
  t.w13_packed_row_stride = w13_packed_row_stride;
  t.w13_scale_row_stride = w13_scale_row_stride;
  t.w2_packed_row_stride = w2_packed_row_stride;
  t.w2_scale_row_stride = w2_scale_row_stride;
  t.low_nibble_first = 1;  // the kernel rejects 0
  t.swiglu_limit = kDsv4ExpertSwigluLimit;
  t.reserved0 = 0;
  t.reserved1 = 0;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
