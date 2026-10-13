// -----------------------------------------------------------------------------
// bf16_block_quant_fp8_tiling.cpp -- the validating tiling builder.
//
// ALL shape validation lives here, host-side, before the enqueue (CLAUDE.md
// §2.5). The kernel rejects bad tiling by returning early, which on device is
// a silent no-op that leaves the output buffer at whatever it held -- so a
// wrong shape reaching the device is an undetectable wrong answer, not an
// error. This function is the only place that cannot happen.
// -----------------------------------------------------------------------------

#include "bf16_block_quant_fp8_tiling.h"

namespace moe::kernels::arch35 {

Status MakeBf16BlockQuantFp8Tiling(
    std::uint32_t width, Bf16BlockQuantFp8TilingData* out) noexcept {
  // A null destination is a caller mistake, not a geometry we decline (§2.6).
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (width == 0u) {
    return Status::invalid_argument;
  }

  // Geometry v1 does not implement: a width that is not a whole number of
  // scale blocks, or one past the envelope. Both are unsupported_shape --
  // the caller asked for something coherent that this version declines.
  //
  // The block-alignment rule is load-bearing rather than cosmetic: the SIMT
  // VF guards whole blocks, so a partial trailing block would be either
  // skipped (leaving stale bytes in the output) or given a scale derived from
  // elements outside the tensor.
  if (width % kBlockQuantBlockSize != 0u) {
    return Status::unsupported_shape;
  }
  if (width > kBlockQuantMaxWidth) {
    return Status::unsupported_shape;
  }

  // Widen before multiplying a caller-supplied extent (§2.7). width is
  // already bounded by kBlockQuantMaxWidth above, so this cannot overflow --
  // the widening is kept because the bound and the arithmetic are separate
  // facts, and a later bound change must not silently become a wrap.
  const std::uint64_t tiles =
      (static_cast<std::uint64_t>(width) + kBlockQuantElemsPerTile - 1u) /
      kBlockQuantElemsPerTile;
  if (tiles == 0u || tiles > 0xFFFFFFFFull) {
    return Status::unsupported_shape;
  }

  // The UB envelope is part of the launch contract, so it is checked rather
  // than assumed. Fixed-size here, but a tile-size change must fail loudly.
  if (BlockQuantDynUbBytes() > kBlockQuantDynUbCeilingBytes) {
    return Status::unsupported_shape;
  }

  Bf16BlockQuantFp8TilingData t{};
  t.abi_version = kTilingAbiVersion;
  t.block_dim = kBlockQuantBlockDim;
  t.width = width;
  t.block_size = kBlockQuantBlockSize;
  t.elems_per_tile = kBlockQuantElemsPerTile;
  t.blocks_per_tile = kBlockQuantBlocksPerTile;
  t.num_tiles = static_cast<std::uint32_t>(tiles);
  t.input_is_fp32 = 0u;  // Legacy callers retain BF16 input.
  *out = t;
  return Status::success;
}

}  // namespace moe::kernels::arch35
