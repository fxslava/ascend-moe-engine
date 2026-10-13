// bf16_block_quant_fp8_kernel.cpp -- device kernel (Ascend 950PR, arch35 /
// dav-3510, 32 AIVs, hybrid SIMD/SIMT).
//
// Math (oracle: moe::emu::bf16_block_quant_fp8_cpu):
//   for each block b of 32 consecutive elements:
//     m        = max over the block of |x|
//     scale[b] = (m > 0) ? m / 448 : 1          FP32, NOT a power of two
//     q[i]     = e4m3( x[i] * (448 / m) )       saturating, round-to-nearest
//   with q[i] = 0 and scale[b] = 1 for an all-zero block, so the dequantized
//   value is exactly 0 and the consuming matmul never multiplies by a zero
//   or denormal scale.
//
// Grid: 32 AIV cores, grid-stride over 1024-element tiles. Each core owns
// disjoint tiles, so there is no cross-core synchronization and no GM
// scratch.
//
// Per tile: MTE2 the BF16 tile and widen with a SIMD Cast, or MTE2 FP32
// directly into the same FP32 buffer, then ONE
// SIMT launch of 512 threads = 32 blocks x 16 lanes. Each lane owns a
// float2, reduces the block max through an in-register butterfly, converts
// its pair with the hardware E4M3 encoder, and lane 0 writes the block's
// FP32 scale. The SIMD side then stores both outputs.
//
// See bf16_block_quant_fp8_arch35.h for why the reduction and the conversion
// are both in the VF: this toolkit has no WholeReduceMax/BlockReduceMax and
// no SIMD float->E4M3 Cast, so neither half has a vector formulation here.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "bf16_block_quant_fp8_arch35.h"
#include "bf16_block_quant_fp8_tiling.h"

#include "simt_api/asc_fp8.h"

namespace moe::kernels::arch35::bf16_block_quant_fp8 {

using namespace AscendC;

__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
BlockQuantTileSimt(__ubuf__ const float* x_ub,     // [kElemsPerTile]
                   __ubuf__ std::uint8_t* q_ub,    // [kElemsPerTile]
                   __ubuf__ float* scale_ub,       // [kBlocksPerTile]
                   std::uint32_t tile_elems) {     // multiple of 32
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t block = t / kLanesPerBlock;  // 0..31
  const std::uint32_t lane = t % kLanesPerBlock;   // 0..15
  const std::uint32_t i0 = block * kBlockSize + 2u * lane;

  // Whole blocks only: MakeBf16BlockQuantFp8Tiling rejects a width that is
  // not a multiple of kBlockSize, so a block is entirely inside the tile or
  // entirely outside it. There is no partial-block case to guard per element.
  const bool active = (block * kBlockSize) < tile_elems;

  // Inactive lanes still take part in the shuffles below -- asc_shfl_xor is
  // a warp-wide operation and must be reached by every lane of the group --
  // so they contribute a magnitude that cannot win the max.
  float a0 = 0.0F;
  float a1 = 0.0F;
  if (active) {
    a0 = x_ub[i0];
    a1 = x_ub[i0 + 1u];
  }

  float m = fmaxf(fabsf(a0), fabsf(a1));
  // In-register block reduction across the 16 lanes of this block. The xor
  // masks 1,2,4,8 at width 16 stay inside the aligned 16-lane group, so the
  // 32 blocks of the launch reduce independently and every lane ends holding
  // its own block's maximum -- which is what removes the broadcast step.
  m = fmaxf(m, asc_shfl_xor(m, 1, kLanesPerBlock));
  m = fmaxf(m, asc_shfl_xor(m, 2, kLanesPerBlock));
  m = fmaxf(m, asc_shfl_xor(m, 4, kLanesPerBlock));
  m = fmaxf(m, asc_shfl_xor(m, 8, kLanesPerBlock));

  // An all-zero block gets scale 1 and codes 0. Deriving the scale from a
  // zero max would emit 0 or a denormal, and the consuming matmul would then
  // multiply by it -- the output is the same either way, but a zero
  // pertoken_scale is the kind of value that turns into a NaN one stage
  // later.
  const bool nonzero = m > 0.0F;
  const float scale = nonzero ? (m * (1.0F / kE4M3Max)) : 1.0F;
  const float inv = nonzero ? (kE4M3Max / m) : 0.0F;

  float2 v;
  v.x = a0 * inv;
  v.y = a1 * inv;
  // The only FP8 E4M3 encoder in this toolkit (simt_api/asc_fp8.h, gated on
  // __NPU_ARCH__ == 3510). __ASC_SATFINITE clamps to +/-448 rather than
  // emitting an infinity; after the scaling above the in-range case is
  // already exact at the top of the format, so saturation only ever catches
  // a rounding step at the boundary.
  const __asc_fp8x2_storage_t packed =
      __asc_cvt_float2_to_fp8x2(v, __ASC_SATFINITE, __ASC_E4M3);

  if (active) {
    // UNVERIFIED (CLAUDE.md §5.7): the byte order within
    // __asc_fp8x2_storage_t is taken to be little-endian with respect to the
    // float2, i.e. low byte <- v.x. The toolkit headers declare the storage
    // as `unsigned short` and document no lane order, and this host has no
    // NPU, so the pairing has never been observed. If tests/npu shows the
    // two halves of every pair transposed, swap these two lines -- that is
    // the whole fix, and the emu golden will localize it immediately because
    // a transposed pair changes the output while leaving the scales right.
    q_ub[i0] = static_cast<std::uint8_t>(packed & 0xFFu);
    q_ub[i0 + 1u] = static_cast<std::uint8_t>((packed >> 8u) & 0xFFu);
    if (lane == 0u) {
      scale_ub[block] = scale;
    }
  }
}

class Bf16BlockQuantFp8 {
 public:
  __aicore__ inline Bf16BlockQuantFp8(
      GM_ADDR x, GM_ADDR y, GM_ADDR scale,
      const __gm__ Bf16BlockQuantFp8TilingData* tiling) {
    block_idx_ = GetBlockIdx();
    // CCE address spaces forbid struct copy from __gm__; field-by-field.
    tiling_.block_dim = tiling->block_dim;
    tiling_.width = tiling->width;
    tiling_.elems_per_tile = tiling->elems_per_tile;
    tiling_.blocks_per_tile = tiling->blocks_per_tile;
    tiling_.num_tiles = tiling->num_tiles;
    tiling_.input_is_fp32 = tiling->input_is_fp32;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x),
                         tiling_.width);
    xF32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling_.width);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(y),
                         tiling_.width);
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale),
                             tiling_.width / kBlockSize);
  }

  __aicore__ inline void Init() {
    // Dynamic UB only; allocation order fixes the offsets and must match
    // BlockQuantDynUbBytes(). The allocator is a function-local bump pointer
    // rather than a member for the reason mxfp4_expert_gemv documents: as a
    // member, its inline constructor has to be emitted from this class's
    // constructor and the toolkit leaves LocalMemAllocator::LocalMemAllocator()
    // without a reachable definition on this include path, which produces
    // -Wundefined-inline. head_ starts at 0 per instance, so a local is
    // equivalent -- all four tensors are allocated here, once, in this order.
    //
    // Alloc()'s tileSize is in ELEMENTS, not bytes.
    LocalMemAllocator<> alloc;
    xBf16Tensor_ = alloc.Alloc<bfloat16_t>(kElemsPerTile);
    xF32Tensor_ = alloc.Alloc<float>(kElemsPerTile);
    qTensor_ = alloc.Alloc<std::uint8_t>(kElemsPerTile);
    scaleTensor_ = alloc.Alloc<float>(kBlocksPerTile);
  }

  __aicore__ inline void Process() {
    for (std::uint32_t tile = static_cast<std::uint32_t>(block_idx_);
         tile < tiling_.num_tiles; tile += tiling_.block_dim) {
      const std::uint32_t base = tile * kElemsPerTile;
      // The last tile may be short; width is a multiple of kBlockSize, so
      // tile_elems is too.
      const std::uint32_t remaining = tiling_.width - base;
      const std::uint32_t tile_elems =
          remaining < kElemsPerTile ? remaining : kElemsPerTile;
      const std::uint32_t tile_blocks = tile_elems / kBlockSize;

      if (tiling_.input_is_fp32 == 1u) {
        DataCopy(xF32Tensor_, xF32Gm_[base], tile_elems);
      } else {
        DataCopy(xBf16Tensor_, xGm_[base], tile_elems);
      }
      SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);

      // Widen on the SIMD side: the VF reads FP32 so the per-lane max and the
      // multiply happen at full precision, and BF16 -> FP32 is exact, so this
      // introduces no error of its own.
      if (tiling_.input_is_fp32 == 0u) {
        Cast(xF32Tensor_, xBf16Tensor_, RoundMode::CAST_NONE, tile_elems);
      }
      DataSyncBarrier<MemDsbT::ALL>();

      asc_vf_call<BlockQuantTileSimt>(
          dim3(kSimtThreads),
          reinterpret_cast<__ubuf__ float*>(xF32Tensor_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ std::uint8_t*>(qTensor_.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(scaleTensor_.GetPhyAddr()),
          tile_elems);
      DataSyncBarrier<MemDsbT::ALL>();

      SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
      WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
      DataCopy(yGm_[base], qTensor_, tile_elems);
      DataCopy(scaleGm_[base / kBlockSize], scaleTensor_, tile_blocks);
    }
  }

 private:
  int32_t block_idx_;
  Bf16BlockQuantFp8TilingData tiling_;

  LocalTensor<bfloat16_t> xBf16Tensor_;
  LocalTensor<float> xF32Tensor_;
  LocalTensor<std::uint8_t> qTensor_;
  LocalTensor<float> scaleTensor_;

  GlobalTensor<bfloat16_t> xGm_;
  GlobalTensor<float> xF32Gm_;
  GlobalTensor<std::uint8_t> yGm_;
  GlobalTensor<float> scaleGm_;
};

// GM_ADDR parameter order mirrors LaunchBf16BlockQuantFp8 (workspace unused:
// GM scratch is 0 bytes for this kernel).
extern "C" __global__ __aicore__ void bf16_block_quant_fp8(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  (void)workspace;
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Bf16BlockQuantFp8TilingData* td =
      reinterpret_cast<const __gm__ Bf16BlockQuantFp8TilingData*>(tiling);
  // §2.3: reject an unknown ABI version. The rest re-checks the invariants
  // the host already enforced -- cheap, and the alternative is a silent
  // no-op producing a plausible-looking buffer.
  if (unlikely(td->abi_version != kTilingAbiVersion ||
               td->block_dim != kBlockQuantBlockDim ||
               td->block_size != kBlockSize ||
               td->elems_per_tile != kElemsPerTile ||
               td->blocks_per_tile != kBlocksPerTile || td->input_is_fp32 > 1u ||
               td->width == 0u || td->width % kBlockSize != 0u ||
               td->width > kBlockQuantMaxWidth || td->num_tiles == 0u)) {
    return;
  }
  Bf16BlockQuantFp8 op(x, y, scale, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::bf16_block_quant_fp8
