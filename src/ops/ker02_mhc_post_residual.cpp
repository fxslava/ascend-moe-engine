// ker02_mhc_post_residual.cpp -- KER-02 device kernel (Ascend 950PR,
// arch35 / dav-3510, 4 AIVs, pure SIMD).
//
// Math (oracle src/emu/mhc_fused_sinkhorn_gemv.cpp :: mhc_post_cpu):
//   acc[j,d] = post[j] * BF16_to_FP32(h_out[d])
//   acc[j,d] += sum_i B[i,j] * BF16_to_FP32(x[i,d])      B row-major
//   y[j,d]    = BF16(acc[j,d])
//
// B is transposed BY INDEXING (mix[i] = birkhoff[i][j]), never by moving
// data: physically transposing a 4x4 is how the orientation silently flips.
// Block j owns output stream j and reads all four input streams; the four
// blocks' y rows are disjoint, so there is no barrier, no SyncAll, and no
// GM scratch (workspace = 0 bytes). In-place x == y is a host-side contract
// violation (parallel streams would overwrite another block's source) and
// is rejected before launch.
//
// Feature lanes run in 512-element tiles (8 per stream). The per-stream mix
// scalar is applied in place on each widened source tile (Muls) before the
// accumulate (Add), so no product scratch tile is needed.

#include "moe/kernels/arch35/arch35_common.hpp"
#include "moe/kernels/arch35/ker02_ub_layout.hpp"
#include "moe/kernels/arch35/tiling_data.hpp"

namespace moe::kernels::arch35::ker02 {

using namespace AscendC;

class MhcPostResidual {
 public:
  __aicore__ inline MhcPostResidual(GM_ADDR x, GM_ADDR birkhoff,
                                    GM_ADDR post, GM_ADDR h_out, GM_ADDR y,
                                    const __gm__ MhcPostTilingData* tiling) {
    block_idx_ = GetBlockIdx();
    // CCE address spaces forbid struct copy from __gm__; field-by-field.
    tiling_.block_dim = tiling->block_dim;
    tiling_.hidden = tiling->hidden;
    tiling_.streams = tiling->streams;
    tiling_.x_stream_stride = tiling->x_stream_stride;
    tiling_.y_stream_stride = tiling->y_stream_stride;
    tiling_.b_row_stride = tiling->b_row_stride;
    tiling_.abi_version = tiling->abi_version;
    tiling_.feature_tile = tiling->feature_tile;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    birkhoffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(birkhoff), 16);
    postGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(post), kStreams);
    hOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(h_out),
                            kHidden);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
  }

  __aicore__ inline void Init() {
    pipe_.InitBuffer(xTileBf16Buf_, kXTileBytes);
    for (std::uint32_t i = 0; i < kStreams; ++i) {
      pipe_.InitBuffer(xF32Bufs_[i], kXF32TileBytes);
    }
    pipe_.InitBuffer(hOutTileBf16Buf_, kXTileBytes);
    pipe_.InitBuffer(hOutF32Buf_, kXF32TileBytes);
    pipe_.InitBuffer(accBuf_, kAccBytes);
    pipe_.InitBuffer(yTileBf16Buf_, kYTileBytes);
    pipe_.InitBuffer(scalarBuf_, kScalarStateBytes);
  }

  __aicore__ inline void Process() {
    if (static_cast<std::uint32_t>(block_idx_) >= kStreams) {
      return;  // launched with block_dim == 4; extra blocks do nothing
    }
    const std::uint32_t j = static_cast<std::uint32_t>(block_idx_);

    // B^T x by indexing: mix[i] = B[i][j], i.e. birkhoff[i * 4 + j].
    float mix[kStreams];
    for (std::uint32_t i = 0; i < kStreams; ++i) {
      mix[i] = birkhoffGm_.GetValue(i * tiling_.b_row_stride + j);
    }
    const float gain = postGm_.GetValue(j);

    auto xTile = xTileBf16Buf_.Get<bfloat16_t>();
    auto hOutTile = hOutTileBf16Buf_.Get<bfloat16_t>();
    auto hOutF32 = hOutF32Buf_.Get<float>();
    auto acc = accBuf_.Get<float>();
    auto yTile = yTileBf16Buf_.Get<bfloat16_t>();

    for (std::uint32_t tile = 0; tile < kTilesPerStream; ++tile) {
      const std::uint32_t base = tile * kFeatureTile;

      // Widen the four source tiles and the h_out tile.
      for (std::uint32_t i = 0; i < kStreams; ++i) {
        if (i != 0 || tile != 0) {
          SetFlag<HardEvent::V_MTE2>(kEvVToMte2);
          WaitFlag<HardEvent::V_MTE2>(kEvVToMte2);
        }
        DataCopy(xTile, xGm_[i * tiling_.x_stream_stride + base],
                 kFeatureTile);
        SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
        Cast(xF32Bufs_[i].Get<float>(), xTile, RoundMode::CAST_NONE,
             kFeatureTile);
      }
      SetFlag<HardEvent::V_MTE2>(kEvVToMte2);
      WaitFlag<HardEvent::V_MTE2>(kEvVToMte2);
      DataCopy(hOutTile, hOutGm_[base], kFeatureTile);
      SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
      WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
      Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, kFeatureTile);

      // acc = post[j] * h_out + sum_i B[i][j] * x_i, oracle accumulation
      // order: the gain term first, then the four mixed streams.
      Muls(acc, hOutF32, gain, kFeatureTile);
      for (std::uint32_t i = 0; i < kStreams; ++i) {
        auto src = xF32Bufs_[i].Get<float>();
        Muls(src, src, mix[i], kFeatureTile);  // in place; tile is read once
        Add(acc, acc, src, kFeatureTile);
      }
      Cast(yTile, acc, RoundMode::CAST_RINT, kFeatureTile);
      SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
      WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
      DataCopy(yGm_[j * tiling_.y_stream_stride + base], yTile, kFeatureTile);
    }
  }

 private:
  int32_t block_idx_;
  MhcPostTilingData tiling_;

  TPipe pipe_;
  TBuf<> xTileBf16Buf_;
  TBuf<> xF32Bufs_[kStreams];
  TBuf<> hOutTileBf16Buf_;
  TBuf<> hOutF32Buf_;
  TBuf<> accBuf_;
  TBuf<> yTileBf16Buf_;
  TBuf<> scalarBuf_;

  GlobalTensor<bfloat16_t> xGm_;
  GlobalTensor<float> birkhoffGm_;
  GlobalTensor<float> postGm_;
  GlobalTensor<bfloat16_t> hOutGm_;
  GlobalTensor<bfloat16_t> yGm_;
};

// GM_ADDR parameter order mirrors LaunchMhcPostResidual (workspace unused:
// GM scratch is 0 bytes for this kernel).
extern "C" __global__ __aicore__ void ker02_mhc_post_residual(
    GM_ADDR x, GM_ADDR birkhoff, GM_ADDR post, GM_ADDR h_out, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ MhcPostTilingData* td =
      reinterpret_cast<const __gm__ MhcPostTilingData*>(tiling);
  if (unlikely(td->abi_version != kTilingAbiVersion ||
               td->block_dim != kStreams || td->hidden != kHidden ||
               td->streams != kStreams ||
               td->feature_tile != kFeatureTile ||
               td->x_stream_stride != kHidden ||
               td->y_stream_stride != kHidden || td->b_row_stride != 4)) {
    return;
  }
  MhcPostResidual op(x, birkhoff, post, h_out, y, td);
  op.Init();
  op.Process();
}

}  // namespace moe::kernels::arch35::ker02
