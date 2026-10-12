// mxfp4_expert_gemv.cpp -- mxfp4_expert_gemv device kernel (Ascend 950PR,
// arch35 / dav-3510, 32 AIVs, hybrid SIMD/SIMT).
//
// Math (oracle src/emu/dsv4_quant_gemv.cpp :: mxfp4_gemv_cpu):
//   y[row] = sum over groups g of ( sum_{col in g} e2m1(packed) * x[col] )
//            * e8m0(scale[row][g])
// with one E8M0 scale per 32 LOGICAL columns per row, kMxfp4LowNibbleFirst
// == 1 (even logical column in bits 0..3 of the packed byte), FP32
// accumulation throughout.
//
// Grid: 32 AIV cores, grid-stride over 32-row tiles (rows 2048 or 4096 =>
// 64 or 128 tiles, 2 or 4 per core); each core owns disjoint tiles, so
// there is no cross-core synchronization and no GM scratch.
//
// Per tile, one SIMT launch of 512 threads = 32 rows x 16 lanes:
//   * the lane's packed bytes stream from GM through the SIMT DataCache
//     (asc_ldca) and never touch UB;
//   * OCP E2M1 is decoded branch-free in registers (select, no branch);
//   * the E8M0 group scale is formed with ldexpf and applied ONCE per
//     16-pair (32-column) group partial -- the oracle's grouping, so the
//     per-group scale-after-subtotal order is preserved bit-for-bit;
//   * x[cols] is resident in UB and read directly by the lanes;
//   * the 16 lane partials of each row reduce through an in-register
//     butterfly (xor masks 1,2,4,8 within the aligned 16-lane row group),
//     and lane 0 writes the row result into the UB y tile.
// The SIMD side owns the x MTE2 load and the y tile store per 32-row tile
// (one V->MTE3 guarded 128 B DataCopy each). A vector ReduceSum cannot
// express 32 independent 16-lane row reductions on this API generation
// (no sub-tensor views), which is why the dense reduction runs in
// registers instead; the UB budget is 16,512 B, all of it dynamic.
//
// PROBE (hardware error 341): the two SIMT-visible buffers are allocated
// from the DYNAMIC UB region via LocalMemAllocator, and the host declares
// that region's size as dyn_ub_size in the chevron. Previously they were
// TPipe/TBuf allocations in the STATIC region while the launch passed
// nullptr for dyn_ub_size, i.e. declared 0 dynamic bytes. Every operator
// that hands a __ubuf__ pointer to asc_vf_call faults with 341 on its first
// SIMT UB access, while fused_mhc_post_residual -- the only operator with no
// SIMT at all -- passes; the three failing footprints span 16 KB to 174 KB,
// which rules out a static tile-budget violation. This change tests that
// correlation. NOT YET VALIDATED ON HARDWARE.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "mxfp4_expert_gemv_arch35.h"
#include "mxfp4_expert_gemv_tiling.h"

namespace moe::kernels::arch35::mxfp4_expert_gemv {

using namespace AscendC;

__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
Mxfp4GemvTileSimt(__gm__ std::uint8_t* packed_gm, __gm__ std::uint8_t* scale_gm,
                  __ubuf__ const float* x_ub,  // [cols]
                  std::uint32_t rows, std::uint32_t cols,
                  std::uint32_t packed_row_stride, std::uint32_t scale_row_stride,
                  std::uint32_t tile_base,     // first row of this 32-row tile
                  __ubuf__ float* y_tile) {    // [32]
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t row_in_tile = t / kLanesPerRow;
  const std::uint32_t lane = t % kLanesPerRow;
  const std::uint32_t global_row = tile_base + row_in_tile;
  // The lane's contiguous block of pairs aligns exactly to 16-pair groups
  // (32 logical columns, one E8M0 scale), so per group g the lane touches
  // exactly one scale byte.
  const std::uint32_t pairs_per_lane = (cols / 2u) / kLanesPerRow;
  const std::uint32_t lane_group_base = (lane * pairs_per_lane) / 16u;

  if (global_row < rows) {
    __gm__ std::uint8_t* prow = packed_gm + global_row * packed_row_stride;
    __gm__ std::uint8_t* srow = scale_gm + global_row * scale_row_stride;
    float total = 0.0f;
    for (std::uint32_t g = 0; g < pairs_per_lane / 16u; ++g) {
      const float scale = DecodeE8M0(asc_ldca(srow + lane_group_base + g));
      float partial = 0.0f;
      const std::uint32_t p0 = lane * pairs_per_lane + g * 16u;
      for (std::uint32_t q = 0; q < 16u; ++q) {
        const std::uint32_t p = p0 + q;
        const std::uint32_t byte = asc_ldca(prow + p);
        // kMxfp4LowNibbleFirst: even logical column in bits 0..3.
        partial += DecodeFp4E2M1(byte & 0xFu) * x_ub[2u * p] +
                   DecodeFp4E2M1(byte >> 4u) * x_ub[2u * p + 1u];
      }
      total += partial * scale;  // scale once per 32-column group, as the oracle
    }
    // In-register dense reduction across the row's 16 lanes (the aligned
    // 16-lane groups of a warp; xor masks 1..8 never leave the group).
    total += asc_shfl_xor(total, 1, 16);
    total += asc_shfl_xor(total, 2, 16);
    total += asc_shfl_xor(total, 4, 16);
    total += asc_shfl_xor(total, 8, 16);
    if (lane == 0u) {
      y_tile[row_in_tile] = total;
    }
  } else if (lane == 0u) {
    y_tile[row_in_tile] = 0.0f;  // tail tile padding row
  }
}

class Mxfp4ExpertGemv {
 public:
  __aicore__ inline Mxfp4ExpertGemv(GM_ADDR packed, GM_ADDR scale, GM_ADDR x,
                                    GM_ADDR y,
                                    const __gm__ Mxfp4GemvTilingData* tiling) {
    block_idx_ = GetBlockIdx();
    // CCE address spaces forbid struct copy from __gm__; field-by-field.
    tiling_.block_dim = tiling->block_dim;
    tiling_.rows = tiling->rows;
    tiling_.cols = tiling->cols;
    tiling_.packed_row_stride = tiling->packed_row_stride;
    tiling_.scale_row_stride = tiling->scale_row_stride;

    packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(packed));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint8_t*>(scale));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling_.cols);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling_.rows);
  }

  __aicore__ inline void Init() {
    // DYNAMIC UB, not TPipe/TBuf. Both of these buffers are addressed by the
    // SIMT VF, and the dynamic region is the one whose size the host declares
    // through dyn_ub_size in the chevron; a TBuf lives in the static region,
    // which the launch never declares. See the probe rationale in the file
    // header. Allocation order fixes the offsets and must match
    // Mxfp4DynUbBytes(): xF32 first, then yTile.
    //
    // Alloc()'s tileSize is in ELEMENTS, not bytes (the implementation does
    // head_ += sizeof(DataType) * tileSize), so cols and kRowsPerTile are
    // passed as element counts.
    // The allocator is a function-local bump pointer, not a member: as a
    // member its inline constructor has to be emitted from this class's own
    // constructor, and the toolkit declares
    // LocalMemAllocator::LocalMemAllocator() without a reachable definition
    // (the body in impl/basic_api/kernel_tensor_impl.h sits behind an
    // __NPU_ARCH__ guard that this include path does not satisfy). That
    // produced two -Wundefined-inline warnings here while the other two SIMT
    // operators, which allocate from locals, compiled clean. It links either
    // way, but the gate's standard is zero warnings, and head_ starts at 0 on
    // each instance so a local is equivalent: both tensors are allocated here,
    // once, in this order.
    LocalMemAllocator<> alloc;
    xTensor_ = alloc.Alloc<float>(tiling_.cols);
    yTensor_ = alloc.Alloc<float>(kRowsPerTile);
  }

  __aicore__ inline void Process() {
    // x is resident once for all of this core's tiles.
    DataCopy(xTensor_, xGm_, tiling_.cols);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    DataSyncBarrier<MemDsbT::ALL>();

    const std::uint32_t num_tiles =
        (tiling_.rows + kRowsPerTile - 1u) / kRowsPerTile;
    for (std::uint32_t tile = static_cast<std::uint32_t>(block_idx_);
         tile < num_tiles; tile += tiling_.block_dim) {
      const std::uint32_t tile_base = tile * kRowsPerTile;
      asc_vf_call<Mxfp4GemvTileSimt>(
          dim3(kSimtThreads),
          const_cast<__gm__ std::uint8_t*>(
              reinterpret_cast<const __gm__ std::uint8_t*>(
                  packedGm_.GetPhyAddr())),
          const_cast<__gm__ std::uint8_t*>(
              reinterpret_cast<const __gm__ std::uint8_t*>(
                  scaleGm_.GetPhyAddr())),
          reinterpret_cast<__ubuf__ float*>(xTensor_.GetPhyAddr()),
          tiling_.rows, tiling_.cols, tiling_.packed_row_stride,
          tiling_.scale_row_stride, tile_base,
          reinterpret_cast<__ubuf__ float*>(yTensor_.GetPhyAddr()));
      DataSyncBarrier<MemDsbT::ALL>();
      SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
      WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
      DataCopy(yGm_[tile_base], yTensor_, kRowsPerTile);
    }
  }

 private:
  int32_t block_idx_;
  Mxfp4GemvTilingData tiling_;

  // No TPipe: this kernel allocates entirely from the dynamic UB region, so
  // static UB is 0 B. Keeping a TBuf alongside LocalMemAllocator would risk
  // the two mechanisms overlapping, which the 950PR docs call out as
  // undefined -- and the old scalarState TBuf was never read anyway.
  LocalTensor<float> xTensor_;
  LocalTensor<float> yTensor_;

  GlobalTensor<std::uint8_t> packedGm_;
  GlobalTensor<std::uint8_t> scaleGm_;
  GlobalTensor<float> xGm_;
  GlobalTensor<float> yGm_;
};

// GM_ADDR parameter order mirrors LaunchMxfp4ExpertGemv (workspace unused:
// GM scratch is 0 bytes for this kernel).
extern "C" __global__ __aicore__ void mxfp4_expert_gemv(
    GM_ADDR packed, GM_ADDR scale, GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  (void)workspace;
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Mxfp4GemvTilingData* td =
      reinterpret_cast<const __gm__ Mxfp4GemvTilingData*>(tiling);
  // v1 contract: rows 2048 or 4096, cols 4096 or 2048, exact strides, and
  // kMxfp4LowNibbleFirst == 1 (a 0 is a within-pair nibble swap, which the
  // all-ones GEMV cannot distinguish -- reject it outright).
  if (unlikely(td->abi_version != kTilingAbiVersion || td->block_dim != 32 ||
               td->rows_per_block != kRowsPerTile ||
               td->group_size != kGroupSize ||
               td->low_nibble_first != 1 || td->reserved != 0 ||
               !((td->rows == 2048u && td->cols == 4096u) ||
                 (td->rows == 4096u && td->cols == 2048u)) ||
               td->packed_row_stride != td->cols / 2u ||
               td->scale_row_stride != td->cols / 32u)) {
    return;
  }
  Mxfp4ExpertGemv op(packed, scale, x, y, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::mxfp4_expert_gemv
