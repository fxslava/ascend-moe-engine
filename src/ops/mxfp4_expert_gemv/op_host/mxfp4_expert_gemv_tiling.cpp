#include "mxfp4_expert_gemv_tiling.h"

namespace moe::kernels::arch35 {

Status MakeMxfp4GemvTiling(std::uint32_t packed_row_stride,
                           std::uint32_t scale_row_stride, std::uint32_t rows,
                           std::uint32_t cols,
                           Mxfp4GemvTilingData* out) noexcept {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // v1 geometries, per the contract: w1/w3 are [2048,4096], w2 is [4096,2048].
  const bool geometry_ok = (rows == 2048u && cols == 4096u) ||
                           (rows == 4096u && cols == 2048u);
  if (!geometry_ok) {
    return Status::unsupported_shape;
  }
  // "Require packed_row_stride = cols/2, scale_row_stride = cols/32 for v1;
  // never infer a shared FP8-style 128x128 scale grid."
  if (packed_row_stride != cols / 2u ||
      scale_row_stride != cols / kMxfp4GroupSize) {
    return Status::unsupported_shape;
  }
  if (rows % kMxfp4RowsPerTile != 0) {
    return Status::unsupported_shape;
  }

  Mxfp4GemvTilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = kMxfp4BlockDim;
  t.rows = rows;
  t.cols = cols;
  t.rows_per_block = kMxfp4RowsPerTile;
  t.group_size = kMxfp4GroupSize;
  t.packed_row_stride = packed_row_stride;
  t.scale_row_stride = scale_row_stride;
  t.low_nibble_first = 1;  // the kernel rejects 0
  t.reserved = 0;
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
