#include "fused_mhc_pre_norm_sinkhorn_tiling.h"

namespace moe::kernels::arch35 {

Status MakeMhcPreNormTiling(std::uint32_t x_stream_stride,
                            std::uint32_t phi_row_stride,
                            MhcPreNormTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // v1 fixes both strides. A different stride is a shape this kernel does not
  // implement rather than a caller mistake, which is why it is
  // unsupported_shape and not invalid_argument.
  if (x_stream_stride != kMhcHidden || phi_row_stride != kMhcFlatWidth) {
    return Status::unsupported_shape;
  }

  MhcPreNormTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = 1;
  t.hidden = kMhcHidden;
  t.streams = kMhcStreams;
  t.projection_rows = kMhcProjectionRows;
  t.projection_tile = kMhcProjectionTile;
  t.projection_chunks = kMhcProjectionChunks;
  t.queue_depth = kMhcQueueDepth;
  t.sinkhorn_iterations = kMhcSinkhornIterations;
  t.x_stream_stride = x_stream_stride;
  t.phi_row_stride = phi_row_stride;
  t.reserved = 0;
  t.hc_eps = kMhcHcEps;
  t.bundle_norm_eps = kMhcNormEps;
  t.activation_norm_eps = kMhcNormEps;
  t.reserved_fp32 = 0.0F;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
