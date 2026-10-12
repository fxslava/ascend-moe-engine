#include "fused_mhc_post_residual_tiling.h"

namespace moe::kernels::arch35 {

Status MakeMhcPostTiling(std::uint32_t x_stream_stride,
                         std::uint32_t y_stream_stride,
                         std::uint32_t b_row_stride,
                         MhcPostTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (x_stream_stride != kMhcPostHidden || y_stream_stride != kMhcPostHidden ||
      b_row_stride != kMhcPostStreams) {
    return Status::unsupported_shape;
  }

  MhcPostTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = kMhcPostStreams;  // one output stream per block
  t.hidden = kMhcPostHidden;
  t.streams = kMhcPostStreams;
  t.feature_tile = kMhcPostFeatureTile;
  t.x_stream_stride = x_stream_stride;
  t.y_stream_stride = y_stream_stride;
  t.b_row_stride = b_row_stride;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
