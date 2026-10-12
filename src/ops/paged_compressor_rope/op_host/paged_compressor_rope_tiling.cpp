#include "paged_compressor_rope_tiling.h"

namespace moe::kernels::arch35 {

Status MakeCompressorTiling(std::uint32_t wkv_row_stride,
                            std::uint32_t wgate_row_stride,
                            std::uint32_t ape_row_stride,
                            CompressorTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (wkv_row_stride != kCompressorHidden ||
      wgate_row_stride != kCompressorHidden ||
      ape_row_stride != kCompressorStateWidthElems) {
    return Status::unsupported_shape;
  }

  CompressorTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = 1;
  t.hidden = kCompressorHidden;
  t.head_dim = kCompressorHeadDim;
  t.coff = kCompressorCoff;
  t.ratio = kCompressorRatio;
  t.reduction_rows = kCompressorReduceRows;
  t.rope_head_dim = kRopeHeadDim;
  t.rotary_mode = kRotaryModeHalf;
  t.projection_row_tile = 1;
  t.wkv_row_stride = wkv_row_stride;
  t.wgate_row_stride = wgate_row_stride;
  t.ape_row_stride = ape_row_stride;
  t.reserved = 0;
  t.norm_eps = kCompressorNormEps;
  t.reserved_fp32 = 0.0F;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
