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

// ---------------------------------------------------------------------------
// Lane geometry: one scale group per lane per step, groups interleaved
// ---------------------------------------------------------------------------
// Lane l takes group  g = s * 16 + l  on step s, so the 16 lanes of a row
// cover 16 CONSECUTIVE groups per step. Because a group is exactly 16 packed
// bytes (kGroupBytes) and is fetched by a single uint4 asc_ldca, the stride
// between lanes equals the load quantum:
//
//   lane l address = row_base + s*256 + l*16        (16 B apart, 16 B wide)
//   warp step shift = 16 * 16 B = 256 B             (kWarpStepBytes)
//
// so one warp-wide load instruction touches 256 contiguous bytes = two
// 128 B DataCache lines, with every fetched byte consumed. The scale bytes
// are read at `scale_row + g`, i.e. 16 CONSECUTIVE bytes across the warp,
// one line.
//
// The previous geometry gave each lane a contiguous BLOCK of groups
// (p0 = lane * pairs_per_lane + g * 16): at hidden = 4096 that put adjacent
// lanes 128 B apart, so a single warp-wide byte load straddled 16 distinct
// cache lines and used 1 of every 128 fetched bytes. See the transaction
// accounting in docs/.
//
// WHAT DID NOT CHANGE, and must not: a group is still owned END TO END by
// ONE lane, its 16 pairs are still summed in ascending column order into a
// register partial, and the E8M0 scale is still applied ONCE to that
// subtotal before it joins the lane total. So each group's partial is bit
// identical to the previous geometry's; only WHICH lane holds which group
// changes, which permutes the leaves of the butterfly below. Had the group
// been split across lanes instead, the scale could no longer be applied
// after the subtotal without extra in-loop shuffles, and the oracle's
// grouping would be gone.
//
// The host gate guarantees cols % 512 == 0 (kDsv4ExpertReductionQuantum), so
// num_groups is always a whole multiple of the 16 lanes and there is no
// partial warp step to handle. It also makes every row base 16 B aligned --
// cols/2 is a multiple of 256 -- which the uint4 loads require.

// Eight logical columns (one 32-bit packed word = 4 pairs) of BOTH legs,
// sharing the two x reads. This is where the W1/W3 fusion actually pays:
// x_even and x_odd are read from UB once and feed both accumulators, instead
// of each leg sweeping x on its own.
//
// Accumulation shape is deliberately `acc += D(lo)*x_even + D(hi)*x_odd` per
// pair, pair by pair -- the same expression and the same order as the
// one-byte-at-a-time loop it replaces, so the group subtotal is bit
// identical rather than merely close.
__simt_callee__ __aicore__ inline void WordDotGateUp(
    std::uint32_t w1_word, std::uint32_t w3_word, __ubuf__ const float* xp,
    float& gate_partial, float& up_partial) {
  for (std::uint32_t j = 0; j < 4u; ++j) {
    const std::uint32_t shift = 8u * j;
    const std::uint32_t b1 = (w1_word >> shift) & 0xFFu;
    const std::uint32_t b3 = (w3_word >> shift) & 0xFFu;
    const float x_even = xp[2u * j];
    const float x_odd = xp[2u * j + 1u];
    // kMxfp4LowNibbleFirst: even logical column in bits 0..3.
    gate_partial +=
        DecodeFp4E2M1(b1 & 0xFu) * x_even + DecodeFp4E2M1(b1 >> 4u) * x_odd;
    up_partial +=
        DecodeFp4E2M1(b3 & 0xFu) * x_even + DecodeFp4E2M1(b3 >> 4u) * x_odd;
  }
}

// Single-leg form, for the down pass.
__simt_callee__ __aicore__ inline void WordDot(std::uint32_t word,
                                               __ubuf__ const float* xp,
                                               float& partial) {
  for (std::uint32_t j = 0; j < 4u; ++j) {
    const std::uint32_t b = (word >> (8u * j)) & 0xFFu;
    partial += DecodeFp4E2M1(b & 0xFu) * xp[2u * j] +
               DecodeFp4E2M1(b >> 4u) * xp[2u * j + 1u];
  }
}

// One 32-column scale group of both legs: two uint4 loads (one per leg), 16
// pairs each, oracle order. Returns the UNSCALED subtotals; the caller
// applies the group scales.
__simt_callee__ __aicore__ inline void GroupDotGateUp(
    __gm__ std::uint8_t* w1_group, __gm__ std::uint8_t* w3_group,
    __ubuf__ const float* x_group, float& gate_partial, float& up_partial) {
  const ::uint4 v1 = asc_ldca(reinterpret_cast<__gm__ ::uint4*>(w1_group));
  const ::uint4 v3 = asc_ldca(reinterpret_cast<__gm__ ::uint4*>(w3_group));
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  WordDotGateUp(v1.x, v3.x, x_group + 0u, gate_acc, up_acc);
  WordDotGateUp(v1.y, v3.y, x_group + 8u, gate_acc, up_acc);
  WordDotGateUp(v1.z, v3.z, x_group + 16u, gate_acc, up_acc);
  WordDotGateUp(v1.w, v3.w, x_group + 24u, gate_acc, up_acc);
  gate_partial = gate_acc;
  up_partial = up_acc;
}

// One 32-column scale group, single leg. Unscaled subtotal.
__simt_callee__ __aicore__ inline float GroupDot(
    __gm__ std::uint8_t* packed_group, __ubuf__ const float* x_group) {
  const ::uint4 v = asc_ldca(reinterpret_cast<__gm__ ::uint4*>(packed_group));
  float acc = 0.0f;
  WordDot(v.x, x_group + 0u, acc);
  WordDot(v.y, x_group + 8u, acc);
  WordDot(v.z, x_group + 16u, acc);
  WordDot(v.w, x_group + 24u, acc);
  return acc;
}

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
// (w1, w3, w2) across two entry points, which is worth a shared helper.
//
// Since the gate/up legs were fused, THIS form serves only the down leg (w2);
// w1 and w3 go through LaneRowPartialGateUp below, which is the same walk with
// two accumulators. Both are kept rather than expressing the down leg as a
// fused call against itself, which would double its weight traffic.
__simt_callee__ __aicore__ inline float LaneRowPartial(
    __gm__ std::uint8_t* packed_row, __gm__ std::uint8_t* scale_row,
    __ubuf__ const float* x_ub, std::uint32_t lane, std::uint32_t cols) {
  const std::uint32_t groups_per_lane = (cols / kGroupSize) / kLanesPerRow;
  float total = 0.0f;
  for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
    const std::uint32_t g = step * kGroupsPerWarpStep + lane;
    const float scale = DecodeE8M0(asc_ldca(scale_row + g));
    const float partial =
        GroupDot(packed_row + g * kGroupBytes, x_ub + g * kGroupSize);
    total += partial * scale;  // scale once per 32-column group, as the oracle
  }
  return total;
}

// Both legs of the gate/up row in ONE sweep. The two decoupled
// LaneRowPartial() calls this replaces issued two independent weight streams
// and re-generated the same x addresses twice; here each x chunk is visited
// once and feeds both accumulators, and gate_total / up_total stay in
// registers across the whole reduction.
__simt_callee__ __aicore__ inline void LaneRowPartialGateUp(
    __gm__ std::uint8_t* w1_packed_row, __gm__ std::uint8_t* w1_scale_row,
    __gm__ std::uint8_t* w3_packed_row, __gm__ std::uint8_t* w3_scale_row,
    __ubuf__ const float* x_ub, std::uint32_t lane, std::uint32_t cols,
    float& gate_total, float& up_total) {
  const std::uint32_t groups_per_lane = (cols / kGroupSize) / kLanesPerRow;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
    const std::uint32_t g = step * kGroupsPerWarpStep + lane;
    const std::uint32_t byte_off = g * kGroupBytes;
    // w1 and w3 share the group index, so they share the scale index too.
    const float s1 = DecodeE8M0(asc_ldca(w1_scale_row + g));
    const float s3 = DecodeE8M0(asc_ldca(w3_scale_row + g));
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    GroupDotGateUp(w1_packed_row + byte_off, w3_packed_row + byte_off,
                   x_ub + g * kGroupSize, gate_partial, up_partial);
    gate_acc += gate_partial * s1;  // scale once per group, per leg
    up_acc += up_partial * s3;
  }
  gate_total = gate_acc;
  up_total = up_acc;
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
    float gate_lane = 0.0f;
    float up_lane = 0.0f;
    LaneRowPartialGateUp(w1_packed + packed_off, w1_scale + scale_off,
                         w3_packed + packed_off, w3_scale + scale_off, x_ub,
                         lane, hidden, gate_lane, up_lane);
    // The butterfly is unchanged, and still runs once per leg: the fusion
    // shares the SWEEP, not the reduction -- the two legs are independent
    // sums and each needs its own 16-lane reduction.
    const float gate = ReduceRow(gate_lane);
    const float up = ReduceRow(up_lane);
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
  // Loaded into locals before the predicate call: a __gm__-qualified scalar
  // lvalue does not take part in overload resolution against a plain
  // std::uint32_t parameter ("no matching function"), the same address-space
  // rule that forces LoadGeometry() to copy field by field.
  const std::uint32_t hidden = td->hidden;
  const std::uint32_t inter = td->inter;
  return td->abi_version != kTilingAbiVersion ||
         td->block_dim != kDsv4ExpertBlockDim ||
         td->rows_per_block != kRowsPerTile || td->group_size != kGroupSize ||
         td->low_nibble_first != 1u || td->reserved0 != 0u ||
         td->reserved1 != 0u ||
         !Dsv4ExpertGeometryAdmissible(hidden, inter) ||
         td->w13_packed_row_stride != hidden / 2u ||
         td->w13_scale_row_stride != hidden / kGroupSize ||
         td->w2_packed_row_stride != inter / 2u ||
         td->w2_scale_row_stride != inter / kGroupSize;
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
