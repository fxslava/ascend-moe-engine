// dsv4_moe_expert_kernel.cpp -- dsv4_moe_expert device kernel (Ascend 950PR,
// arch35 / dav-3510, 32 AIVs, hybrid SIMD/SIMT).
//
// The fused DeepSeek-V4 routed expert for one token, in two stream-ordered
// entry points. Oracle: src/emu/moe_block.cpp :: moe_expert_swiglu_mxfp4_cpu,
// whose ExpertSwiglu body is the authority for the activation below.
//
//   dsv4_moe_expert_gate_up
//     gate[r] = mxfp4_gemv(w1, x)[r]        r over [0, inter)
//     up[r]   = mxfp4_gemv(w3, x)[r]
//     activated[r] = silu(min(gate[r], L)) * min(L, max(-L, up[r]))
//   dsv4_moe_expert_down
//     out[c]  = mxfp4_gemv(w2, activated)[c]   c over [0, hidden)
//
// THE CLAMP IS ASYMMETRIC AND THE TWO HALVES ARE NOT INTERCHANGEABLE.
// AscendSiluAndMulWithClamp bounds the GATE ABOVE ONLY and bounds UP ON BOTH
// SIDES. Clamping the gate below as well is nearly harmless (it only bites
// for gate < -L, where silu is already within 4.6e-4 of zero), but dropping
// the `up` clamp is not: `up` is a raw 4096-wide GEMV output, so it exceeds L
// readily and the error is unbounded. The vllm-ascend prototype this operator
// descends from had both halves inverted; see that repo's commit
// "fix(dsv4_moe_expert): correct the SwiGLU swish sign and the clamp shape".
//
// Evaluate the oracle's silu(clamped_gate) directly. If this is expressed
// through AscendC::SwiGLU, pass beta = +1.0f: both the common and 3510 outer
// wrappers negate beta before their inner exponential, yielding exp(-gate).
// Keeping the direct expression here makes that wrapper convention explicit
// and avoids a second interpretation of the public beta argument.
//
// Grid: 32 AIV cores, grid-stride over 32-row tiles. gate_up tiles over
// `inter` (2048 rows => 64 tiles, 2 per core); down tiles over `hidden`
// (4096 rows => 128 tiles, 4 per core). Each core owns disjoint tiles, so
// there is no cross-core synchronization WITHIN an entry point. The
// dependency BETWEEN them -- down reduces over all of `activated` -- is
// carried by stream ordering and the GM workspace, not by a device barrier.
//
// NEVER EXECUTED: compiled for dav-3510 only. No claim is made here about
// hardware behaviour (CLAUDE.md §5.7).

#include "moe/kernels/arch35/arch35_common.hpp"
#include "dsv4_moe_expert_arch35.h"
#include "dsv4_moe_expert_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_expert {

using namespace AscendC;

// One lane's share of a single MXFP4 GEMV row, scaled per 32-column group
// exactly as the oracle groups it. Returns this lane's partial; the caller
// reduces across the row's 16 lanes.
// ATTRIBUTE: __simt_callee__, not __simt_vf__ and not a bare __aicore__.
// __simt_vf__ marks a VF ENTRY and requires a void return. A bare
// __aicore__ inline does not compile either: asc_ldca and asc_shfl_xor are
// declared __SIMT_DEVICE_FUNCTIONS_DECL__, which simt_api/device_types.h:27
// defines as __simt_callee__, so they are not visible to overload resolution
// from an ordinary device function ("no matching function for call to
// asc_shfl_xor"). __simt_callee__ is the attribute for a helper CALLED FROM a
// VF, and unlike __simt_vf__ it permits a non-void return -- the toolkit uses
// it exactly this way in adv_api/hcomm/hcomm_simt.h.
//
// The other four operators inline their intrinsics directly into the VF entry
// and so never needed this; this operator runs the same GEMV row three times
// (w1, w3, w2) across two entry points, which is worth one shared helper.
__simt_callee__ __aicore__ inline float LaneRowPartial(
    __gm__ std::uint8_t* packed_row, __gm__ std::uint8_t* scale_row,
    __ubuf__ const float* x_ub, std::uint32_t lane, std::uint32_t cols) {
  // The lane's contiguous block of pairs aligns exactly to 16-pair groups
  // (32 logical columns, one E8M0 scale), so per group the lane touches
  // exactly one scale byte.
  const std::uint32_t pairs_per_lane = (cols / 2u) / kLanesPerRow;
  const std::uint32_t lane_group_base = (lane * pairs_per_lane) / 16u;
  float total = 0.0f;
  for (std::uint32_t g = 0; g < pairs_per_lane / 16u; ++g) {
    const float scale = DecodeE8M0(asc_ldca(scale_row + lane_group_base + g));
    float partial = 0.0f;
    const std::uint32_t p0 = lane * pairs_per_lane + g * 16u;
    for (std::uint32_t q = 0; q < 16u; ++q) {
      const std::uint32_t p = p0 + q;
      const std::uint32_t byte = asc_ldca(packed_row + p);
      // kMxfp4LowNibbleFirst: even logical column in bits 0..3.
      partial += DecodeFp4E2M1(byte & 0xFu) * x_ub[2u * p] +
                 DecodeFp4E2M1(byte >> 4u) * x_ub[2u * p + 1u];
    }
    total += partial * scale;  // scale once per 32-column group, as the oracle
  }
  return total;
}

// Reduce a row's 16 lane partials in registers. The aligned 16-lane groups of
// a warp: xor masks 1..8 never leave the group.
__simt_callee__ __aicore__ inline float ReduceRow(float value) {
  value += asc_shfl_xor(value, 1, 16);
  value += asc_shfl_xor(value, 2, 16);
  value += asc_shfl_xor(value, 4, 16);
  value += asc_shfl_xor(value, 8, 16);
  return value;
}

// ---------------------------------------------------------------------------
// Pass 1: gate GEMV + up GEMV + the asymmetric clamped SwiGLU.
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
Dsv4ExpertGateUpTileSimt(__gm__ std::uint8_t* w1_packed, __gm__ std::uint8_t* w1_scale,
                         __gm__ std::uint8_t* w3_packed, __gm__ std::uint8_t* w3_scale,
                         __ubuf__ const float* x_ub,  // [hidden]
                         std::uint32_t inter, std::uint32_t hidden,
                         std::uint32_t packed_row_stride,
                         std::uint32_t scale_row_stride, float limit,
                         std::uint32_t tile_base,
                         __ubuf__ float* act_tile) {  // [32]
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t row_in_tile = t / kLanesPerRow;
  const std::uint32_t lane = t % kLanesPerRow;
  const std::uint32_t global_row = tile_base + row_in_tile;

  if (global_row < inter) {
    const std::uint32_t packed_off = global_row * packed_row_stride;
    const std::uint32_t scale_off = global_row * scale_row_stride;
    const float gate = ReduceRow(LaneRowPartial(
        w1_packed + packed_off, w1_scale + scale_off, x_ub, lane, hidden));
    const float up = ReduceRow(LaneRowPartial(
        w3_packed + packed_off, w3_scale + scale_off, x_ub, lane, hidden));
    if (lane == 0u) {
      // ASYMMETRIC, per the file header: gate above only, up both ways.
      const float g = gate < limit ? gate : limit;
      const float u = up < -limit ? -limit : (up > limit ? limit : up);
      act_tile[row_in_tile] = (g / (1.0f + expf(-g))) * u;
    }
  } else if (lane == 0u) {
    act_tile[row_in_tile] = 0.0f;  // tail tile padding row
  }
}

// ---------------------------------------------------------------------------
// Pass 2: the down GEMV over the completed `activated`.
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
Dsv4ExpertDownTileSimt(__gm__ std::uint8_t* w2_packed, __gm__ std::uint8_t* w2_scale,
                       __ubuf__ const float* act_ub,  // [inter]
                       std::uint32_t hidden, std::uint32_t inter,
                       std::uint32_t packed_row_stride,
                       std::uint32_t scale_row_stride, std::uint32_t tile_base,
                       __ubuf__ float* y_tile) {  // [32]
  const std::uint32_t t = threadIdx.x;
  const std::uint32_t row_in_tile = t / kLanesPerRow;
  const std::uint32_t lane = t % kLanesPerRow;
  const std::uint32_t global_row = tile_base + row_in_tile;

  if (global_row < hidden) {
    const float total = ReduceRow(LaneRowPartial(
        w2_packed + global_row * packed_row_stride,
        w2_scale + global_row * scale_row_stride, act_ub, lane, inter));
    if (lane == 0u) {
      y_tile[row_in_tile] = total;
    }
  } else if (lane == 0u) {
    y_tile[row_in_tile] = 0.0f;  // tail tile padding row
  }
}

namespace {

// Field-by-field: CCE address spaces forbid struct copy from __gm__.
struct Geometry {
  std::uint32_t block_dim;
  std::uint32_t hidden;
  std::uint32_t inter;
  std::uint32_t w13_packed_row_stride;
  std::uint32_t w13_scale_row_stride;
  std::uint32_t w2_packed_row_stride;
  std::uint32_t w2_scale_row_stride;
  float swiglu_limit;
};

__aicore__ inline Geometry LoadGeometry(
    const __gm__ Dsv4MoeExpertTilingData* tiling) {
  Geometry g;
  g.block_dim = tiling->block_dim;
  g.hidden = tiling->hidden;
  g.inter = tiling->inter;
  g.w13_packed_row_stride = tiling->w13_packed_row_stride;
  g.w13_scale_row_stride = tiling->w13_scale_row_stride;
  g.w2_packed_row_stride = tiling->w2_packed_row_stride;
  g.w2_scale_row_stride = tiling->w2_scale_row_stride;
  g.swiglu_limit = tiling->swiglu_limit;
  return g;
}

// The v1 contract, rejected identically by both entry points. A kernel that
// returns early on bad tiling produces a ZEROED output, not an error
// (CLAUDE.md §2.5), which is why MakeDsv4MoeExpertTiling validates host-side
// too -- this is the backstop, not the gate.
__aicore__ inline bool TilingRejected(const __gm__ Dsv4MoeExpertTilingData* td) {
  return td->abi_version != kTilingAbiVersion ||
         td->block_dim != kDsv4ExpertBlockDim ||
         td->rows_per_block != kRowsPerTile || td->group_size != kGroupSize ||
         td->low_nibble_first != 1u || td->reserved0 != 0u ||
         td->reserved1 != 0u || td->hidden != kHiddenMax ||
         td->inter != kInterMax ||
         td->w13_packed_row_stride != td->hidden / 2u ||
         td->w13_scale_row_stride != td->hidden / kGroupSize ||
         td->w2_packed_row_stride != td->inter / 2u ||
         td->w2_scale_row_stride != td->inter / kGroupSize;
}

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR p) {
  return reinterpret_cast<__gm__ std::uint8_t*>(p);
}

}  // namespace

// GM_ADDR parameter order mirrors the two dispatches in LaunchDsv4MoeExpert.
extern "C" __global__ __aicore__ void dsv4_moe_expert_gate_up(
    GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed, GM_ADDR w3_scale,
    GM_ADDR x, GM_ADDR activated, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeExpertTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeExpertTilingData*>(tiling);
  if (unlikely(TilingRejected(td))) {
    return;
  }
  const Geometry g = LoadGeometry(td);
  const int32_t block_idx = GetBlockIdx();

  GlobalTensor<float> xGm;
  GlobalTensor<float> actGm;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), g.hidden);
  actGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activated), g.inter);

  // DYNAMIC UB, not TPipe/TBuf -- see the header. Allocation order fixes the
  // offsets and must match Dsv4ExpertGateUpDynUbBytes(): xF32, then actTile.
  // Alloc()'s tileSize is in ELEMENTS, not bytes. The allocator is a
  // function-local bump pointer rather than a member for the same reason it
  // is in mxfp4_expert_gemv: as a member, its inline constructor has to be
  // emitted here and the toolkit leaves LocalMemAllocator::LocalMemAllocator()
  // without a reachable definition on this include path (-Wundefined-inline).
  LocalMemAllocator<> alloc;
  LocalTensor<float> xTensor = alloc.Alloc<float>(g.hidden);
  LocalTensor<float> actTile = alloc.Alloc<float>(kRowsPerTile);

  // x is resident once for all of this core's tiles.
  DataCopy(xTensor, xGm, g.hidden);
  SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
  WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
  DataSyncBarrier<MemDsbT::ALL>();

  const std::uint32_t num_tiles = (g.inter + kRowsPerTile - 1u) / kRowsPerTile;
  for (std::uint32_t tile = static_cast<std::uint32_t>(block_idx);
       tile < num_tiles; tile += g.block_dim) {
    const std::uint32_t tile_base = tile * kRowsPerTile;
    asc_vf_call<Dsv4ExpertGateUpTileSimt>(
        dim3(kSimtThreads), Bytes(w1_packed), Bytes(w1_scale),
        Bytes(w3_packed), Bytes(w3_scale),
        reinterpret_cast<__ubuf__ float*>(xTensor.GetPhyAddr()), g.inter,
        g.hidden, g.w13_packed_row_stride, g.w13_scale_row_stride,
        g.swiglu_limit, tile_base,
        reinterpret_cast<__ubuf__ float*>(actTile.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(actGm[tile_base], actTile, kRowsPerTile);
  }
}

extern "C" __global__ __aicore__ void dsv4_moe_expert_down(
    GM_ADDR w2_packed, GM_ADDR w2_scale, GM_ADDR activated, GM_ADDR y,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeExpertTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeExpertTilingData*>(tiling);
  if (unlikely(TilingRejected(td))) {
    return;
  }
  const Geometry g = LoadGeometry(td);
  const int32_t block_idx = GetBlockIdx();

  GlobalTensor<float> actGm;
  GlobalTensor<float> yGm;
  actGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activated), g.inter);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), g.hidden);

  // Allocation order must match Dsv4ExpertDownDynUbBytes(): actF32, yTile.
  LocalMemAllocator<> alloc;
  LocalTensor<float> actTensor = alloc.Alloc<float>(g.inter);
  LocalTensor<float> yTile = alloc.Alloc<float>(kRowsPerTile);

  DataCopy(actTensor, actGm, g.inter);
  SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
  WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
  DataSyncBarrier<MemDsbT::ALL>();

  const std::uint32_t num_tiles = (g.hidden + kRowsPerTile - 1u) / kRowsPerTile;
  for (std::uint32_t tile = static_cast<std::uint32_t>(block_idx);
       tile < num_tiles; tile += g.block_dim) {
    const std::uint32_t tile_base = tile * kRowsPerTile;
    asc_vf_call<Dsv4ExpertDownTileSimt>(
        dim3(kSimtThreads), Bytes(w2_packed), Bytes(w2_scale),
        reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr()), g.hidden,
        g.inter, g.w2_packed_row_stride, g.w2_scale_row_stride, tile_base,
        reinterpret_cast<__ubuf__ float*>(yTile.GetPhyAddr()));
    DataSyncBarrier<MemDsbT::ALL>();
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(yGm[tile_base], yTile, kRowsPerTile);
  }
}

}  // namespace moe::kernels::arch35::dsv4_moe_expert
