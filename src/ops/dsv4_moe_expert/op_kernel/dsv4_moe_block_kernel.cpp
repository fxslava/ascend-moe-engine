// dsv4_moe_block_kernel.cpp -- the two whole-MoE-block entry points for
// DeepSeek-V4 routed experts (Ascend 950PR, arch35 / dav-3510, hybrid
// SIMD/SIMT).
//
//   dsv4_moe_decode_topk      M = 1,  a GANG of AIVs per routed expert slot,
//                             router-weighted atomic accumulation into y
//   dsv4_moe_prefill_batched  M >> 1, every AIV grid-striding over tokens
//
// Oracle: src/emu/moe_block.cpp :: moe_decode_topk_mxfp4_cpu and
// moe_prefill_batched_mxfp4_cpu, both of which compose
// moe_expert_swiglu_mxfp4_cpu -- the authority for the activation below.
//
// Each core runs ONE expert for ONE token end to end:
//     gate[r] = mxfp4_gemv(w1, x)[r]        r over [0, inter)
//     up[r]   = mxfp4_gemv(w3, x)[r]
//     act[r]  = silu(min(gate[r], L)) * min(L, max(-L, up[r]))
//     y[c]    = mxfp4_gemv(w2, act)[c]      c over [0, hidden)
// so `act` is a UB buffer with no other sharer and never reaches GM. That is
// the whole reason these two entry points replace the retired two-dispatch
// dsv4_moe_expert; see dsv4_moe_block_arch35.h.
//
// THE CLAMP IS ASYMMETRIC AND THE TWO HALVES ARE NOT INTERCHANGEABLE.
// AscendSiluAndMulWithClamp bounds the GATE ABOVE ONLY and bounds UP ON BOTH
// SIDES. Clamping the gate below as well is nearly harmless (it only bites
// for gate < -L, where silu is already within 4.6e-4 of zero), but dropping
// the `up` clamp is not: `up` is a raw hidden-wide GEMV output, so it exceeds
// L readily and the error is unbounded.
//
// Evaluate the oracle's silu(clamped_gate) directly. If this is expressed
// through AscendC::SwiGLU, pass beta = +1.0f: both the common and 3510 outer
// wrappers negate beta before their inner exponential, yielding exp(-gate).
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under the CAModel
// only (CLAUDE.md §5.7).

#include "moe/kernels/arch35/arch35_common.hpp"
#include "dsv4_moe_block_arch35.h"
#include "dsv4_moe_block_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_block {

using namespace AscendC;

// The tile ring is described twice on purpose: op_kernel/<op>_arch35.h owns
// the UB layout (§1.2, private), while dsv4_moe_block_tiling.h owns what the
// host declares as dyn_ub_size at the chevron. Both sides see this TU, so tie
// them together here. A drift would under-declare the dynamic region, and a
// SIMT VF handed a buffer outside it faults with AIV error 341 on its first
// UB access -- legal budget, clean link, immediate fault on every core.
static_assert(kTileSlots == kMoeBlockTileSlots,
              "tile ring slot count disagrees between the UB layout and the "
              "tiling header the host sizes dyn_ub_size from");
static_assert(kTileSkewBytes == kMoeBlockTileSkewBytes,
              "tile ring skew disagrees between the UB layout and the tiling "
              "header the host sizes dyn_ub_size from");
static_assert(kRowsPerTile == kMoeBlockRowsPerTile,
              "rows per tile disagrees between the UB layout and the tiling "
              "header");

// ---------------------------------------------------------------------------
// Lane geometry: one scale group per lane per step, groups interleaved
// ---------------------------------------------------------------------------
// Lane l takes group  g = step * 16 + l, so the 16 lanes of a row cover 16
// CONSECUTIVE groups per step. A group is exactly 16 packed bytes and is
// fetched by a single uint4 asc_ldca, so the stride between lanes equals the
// load quantum:
//
//   lane l address = row_base + step*256 + l*16     (16 B apart, 16 B wide)
//   warp step shift = 16 * 16 B = 256 B             (kWarpStepBytes)
//
// so one warp-wide load instruction touches 256 contiguous bytes = two 128 B
// DataCache lines, with every fetched byte consumed. Scale bytes are read at
// `scale_row + g`, i.e. 16 CONSECUTIVE bytes across the warp, one line.
//
// A group is owned END TO END by ONE lane, its 16 pairs summed in ascending
// column order into a register partial, and the E8M0 scale applied ONCE to
// that subtotal before it joins the lane total -- which is what preserves the
// oracle's grouping. That is why the load quantum is 16 B and not 4 B: a
// stride-4 uint32 pattern puts 4 lanes on each group, and then the scale
// cannot be applied after the subtotal without two extra in-loop
// asc_shfl_xor per group.
//
// The host gate guarantees cols % 512 == 0 (kMoeBlockReductionQuantum), so a
// row's group count is always a whole multiple of the 16 lanes and there is
// no partial warp step. It also makes every row base 16 B aligned -- cols/2
// is a multiple of 256 -- which the uint4 loads require.

// Eight logical columns (one 32-bit packed word = 4 pairs) of BOTH legs,
// sharing the two x reads. x_even and x_odd are read from UB once and feed
// both accumulators, instead of each leg sweeping x on its own.
//
// Accumulation shape is deliberately `acc += D(lo)*x_even + D(hi)*x_odd` per
// pair, pair by pair -- the same expression and order as a one-byte-at-a-time
// loop, so a group subtotal is bit-identical to the scalar form.
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

// One 32-column scale group of both legs: two uint4 loads, 16 pairs each,
// oracle order. Returns the UNSCALED subtotals; the caller applies the
// group scales.
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

// Reduce a row's 16 lane partials in registers. The aligned 16-lane groups of
// a warp: xor masks 1..8 never leave the group.
__simt_callee__ __aicore__ inline float ReduceRow(float value) {
  value += asc_shfl_xor(value, 1, 16);
  value += asc_shfl_xor(value, 2, 16);
  value += asc_shfl_xor(value, 4, 16);
  value += asc_shfl_xor(value, 8, 16);
  return value;
}

// One lane's share of a single MXFP4 GEMV row (down leg).
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

// Both gate/up legs of a row in ONE sweep: each x chunk is visited once and
// feeds both accumulators, and the two totals stay in registers throughout.
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

// ---------------------------------------------------------------------------
// Phase 1 VF: 32 rows of gate + up, the asymmetric clamped SwiGLU, into UB.
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
MoeBlockGateUpTileSimt(__gm__ std::uint8_t* w1_packed, __gm__ std::uint8_t* w1_scale,
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
    // One butterfly per leg: the fusion shares the SWEEP, not the reduction.
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
// Phase 2 VF: 32 rows of the down GEMV over `activated`, already in UB.
// ---------------------------------------------------------------------------
// `route_weight` folds the router gate into the store, so the decode path
// needs no separate scaling pass. Prefill passes 1.0f.
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
MoeBlockDownTileSimt(__gm__ std::uint8_t* w2_packed, __gm__ std::uint8_t* w2_scale,
                     __ubuf__ const float* act_ub,  // [inter]
                     std::uint32_t hidden, std::uint32_t inter,
                     std::uint32_t packed_row_stride,
                     std::uint32_t scale_row_stride, float route_weight,
                     std::uint32_t tile_base,
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
      y_tile[row_in_tile] = total * route_weight;
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
  std::uint32_t num_experts;
  std::uint32_t top_k;
  std::uint32_t total_tokens;
  std::uint32_t gang_size;
  std::uint32_t w13_packed_row_stride;
  std::uint32_t w13_scale_row_stride;
  std::uint32_t w2_packed_row_stride;
  std::uint32_t w2_scale_row_stride;
  std::uint32_t w13_packed_expert_stride;
  std::uint32_t w13_scale_expert_stride;
  std::uint32_t w2_packed_expert_stride;
  std::uint32_t w2_scale_expert_stride;
  float swiglu_limit;
};

__aicore__ inline Geometry LoadGeometry(
    const __gm__ Dsv4MoeBlockTilingData* tiling) {
  Geometry g;
  g.block_dim = tiling->block_dim;
  g.hidden = tiling->hidden;
  g.inter = tiling->inter;
  g.num_experts = tiling->num_experts;
  g.top_k = tiling->top_k;
  g.total_tokens = tiling->total_tokens;
  g.gang_size = tiling->gang_size;
  g.w13_packed_row_stride = tiling->w13_packed_row_stride;
  g.w13_scale_row_stride = tiling->w13_scale_row_stride;
  g.w2_packed_row_stride = tiling->w2_packed_row_stride;
  g.w2_scale_row_stride = tiling->w2_scale_row_stride;
  g.w13_packed_expert_stride = tiling->w13_packed_expert_stride;
  g.w13_scale_expert_stride = tiling->w13_scale_expert_stride;
  g.w2_packed_expert_stride = tiling->w2_packed_expert_stride;
  g.w2_scale_expert_stride = tiling->w2_scale_expert_stride;
  g.swiglu_limit = tiling->swiglu_limit;
  return g;
}

// The v1 contract. A kernel that returns early on bad tiling produces a
// ZEROED (or, under atomic accumulation, an UNCHANGED) output rather than an
// error (§2.5), which is why the Make*Tiling builders validate host-side too
// -- this is the backstop, not the gate.
//
// `expected_mode` is what makes handing the prefill kernel a decode plan an
// early return instead of a plausible-looking wrong answer.
__aicore__ inline bool TilingRejected(const __gm__ Dsv4MoeBlockTilingData* td,
                                      std::uint32_t expected_mode) {
  // Loaded into locals before the predicate call: a __gm__-qualified scalar
  // lvalue does not take part in overload resolution against a plain
  // std::uint32_t parameter, the same address-space rule that forces
  // LoadGeometry() to copy field by field.
  const std::uint32_t hidden = td->hidden;
  const std::uint32_t inter = td->inter;
  return td->abi_version != kTilingAbiVersion || td->mode != expected_mode ||
         td->rows_per_block != kRowsPerTile ||
         td->group_size != kGroupSize || td->low_nibble_first != 1u ||
         td->num_experts == 0u || td->top_k == 0u ||
         td->total_tokens == 0u ||
         // The decode kernel DIVIDES GetBlockIdx() by this, so a zero is a
         // fault rather than a wrong answer. Reject it here instead of
         // trusting that the host filled it.
         td->gang_size == 0u ||
         !Dsv4MoeBlockGeometryAdmissible(hidden, inter) ||
         td->w13_packed_row_stride != hidden / 2u ||
         td->w13_scale_row_stride != hidden / kGroupSize ||
         td->w2_packed_row_stride != inter / 2u ||
         td->w2_scale_row_stride != inter / kGroupSize;
}

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR p) {
  return reinterpret_cast<__gm__ std::uint8_t*>(p);
}

// One down-phase tile into one ring slot. Factored out so the unrolled loop
// body below names a slot and nothing else, and so the argument list cannot
// drift between the prologue and the steady state.
__aicore__ inline void IssueDownTile(const Geometry& g,
                                     __gm__ std::uint8_t* w2_packed,
                                     __gm__ std::uint8_t* w2_scale,
                                     __ubuf__ float* act_ub, float route_weight,
                                     std::uint32_t tile_base,
                                     __ubuf__ float* slot) {
  asc_vf_call<MoeBlockDownTileSimt>(
      dim3(kSimtThreads), w2_packed, w2_scale, act_ub, g.hidden, g.inter,
      g.w2_packed_row_stride, g.w2_scale_row_stride, route_weight, tile_base,
      slot);
}

// ---------------------------------------------------------------------------
// The two phases are separate functions so the DECODE path can put a
// cross-core barrier between them. Prefill calls them back to back on one
// core, which is the original fused body with `activated` never leaving UB.
//
// Both take a (tile_lo, tile_stride) pair naming which of the phase's row
// tiles THIS core owns:
//
//   prefill / one-core-per-slot   tile_lo = 0,    tile_stride = 1
//   decode gang lane L of G       tile_lo = L,    tile_stride = G
//
// A core's i-th tile is therefore tile (tile_lo + i*tile_stride), and the
// count below is the only place that arithmetic lives.
// ---------------------------------------------------------------------------
__aicore__ inline std::uint32_t TilesOwned(std::uint32_t total_tiles,
                                           std::uint32_t tile_lo,
                                           std::uint32_t tile_stride) {
  if (tile_lo >= total_tiles) {
    return 0u;  // a gang wider than the tile count: this lane has no work
  }
  return (total_tiles - tile_lo + tile_stride - 1u) / tile_stride;
}

// --- phase 1: gate + up + clamped SwiGLU, into actTensor -------------------
// Writes each owned tile's 32 activations at their FINAL offset inside
// actTensor, so there is no tile-to-vector copy. When `act_out` is non-null
// the tile is also published to GM at the same offset, which is what lets the
// gang's other lanes read it after the barrier.
__aicore__ inline void RunGateUpPhase(
    const Geometry& g, __gm__ std::uint8_t* w1_packed,
    __gm__ std::uint8_t* w1_scale, __gm__ std::uint8_t* w3_packed,
    __gm__ std::uint8_t* w3_scale, __ubuf__ float* x_ub,
    const LocalTensor<float>& actTensor, std::uint32_t tile_lo,
    std::uint32_t tile_stride, const GlobalTensor<float>* act_out,
    std::uint64_t act_out_base) {
  __ubuf__ float* act_ub =
      reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr());
  const std::uint32_t inter_tiles =
      (g.inter + kRowsPerTile - 1u) / kRowsPerTile;
  const std::uint32_t owned = TilesOwned(inter_tiles, tile_lo, tile_stride);

  for (std::uint32_t i = 0; i < owned; ++i) {
    const std::uint32_t tile_base = (tile_lo + i * tile_stride) * kRowsPerTile;
    asc_vf_call<MoeBlockGateUpTileSimt>(
        dim3(kSimtThreads), w1_packed, w1_scale, w3_packed, w3_scale, x_ub,
        g.inter, g.hidden, g.w13_packed_row_stride, g.w13_scale_row_stride,
        g.swiglu_limit, tile_base, act_ub + tile_base);
    DataSyncBarrier<MemDsbT::ALL>();
    if (act_out != nullptr) {
      // 128 B per store, well above the 32 B MTE quantum. V->MTE3 first: the
      // VF's UB writes must retire before DMA reads the same 32 floats.
      SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
      WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
      DataCopy((*act_out)[act_out_base + tile_base],
               actTensor[tile_base], kRowsPerTile);
      SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
      WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    }
  }
}

// --- phase 2: the down GEMV over the completed `activated` ----------------
__aicore__ inline void RunDownPhase(
    const Geometry& g, __gm__ std::uint8_t* w2_packed,
    __gm__ std::uint8_t* w2_scale, const LocalTensor<float>& actTensor,
    const LocalTensor<float> (&tiles)[kTileSlots],
    const GlobalTensor<float>& yGm, std::uint64_t y_base, float route_weight,
    std::uint32_t tile_lo, std::uint32_t tile_stride) {
  __ubuf__ float* act_ub =
      reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr());
  __ubuf__ float* tile_ub[kTileSlots];
  for (std::uint32_t slot = 0; slot < kTileSlots; ++slot) {
    tile_ub[slot] =
        reinterpret_cast<__ubuf__ float*>(tiles[slot].GetPhyAddr());
  }
  //
  // Double-buffered and unrolled by kTileSlots, so ONE DataSyncBarrier
  // publishes both slots' VFs per pipeline step instead of one per tile. The
  // two slots are bank-disjoint (kTileSkewBytes), so a single coalesced
  // barrier before the pair of stores is what the consumer needs: MTE3 reads
  // slot 0 and slot 1 only after both VFs have been published.
  //
  // WHY THE NEXT PAIR'S VFs ARE ISSUED AFTER MTE3_V, NOT BEFORE
  // They write the same two slots MTE3 is reading. Issuing them earlier is a
  // WAR hazard that no fence covers until MTE3_V, so the gain here is a
  // halved barrier and event count, NOT overlap across the pair. Overlapping
  // a VF with another tile's store requires a third slot. Do not "optimize"
  // these two calls above the MTE3_V wait.
  //
  // Set/Wait are balanced 1:1 per unrolled step, so no event token outlives
  // the loop and nothing leaks into the next launch.
  //
  // THE RING NOW WALKS THIS CORE'S OWN TILE SUBSEQUENCE, not a contiguous
  // run: `i` indexes the core's tiles and (tile_lo + i*tile_stride) is the
  // global tile. With tile_stride == 1 this reduces to the previous
  // contiguous schedule exactly.
  const std::uint32_t hidden_tiles =
      (g.hidden + kRowsPerTile - 1u) / kRowsPerTile;
  const std::uint32_t owned = TilesOwned(hidden_tiles, tile_lo, tile_stride);
  if (owned == 0u) {
    return;  // a gang wider than the tile count: this lane has no down work
  }

  // Prologue: prime both slots. Clamped, so a core owning fewer tiles than
  // the ring has slots does not prime a slot no iteration consumes.
  const std::uint32_t primed = owned < kTileSlots ? owned : kTileSlots;
  for (std::uint32_t slot = 0; slot < primed; ++slot) {
    IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                  (tile_lo + slot * tile_stride) * kRowsPerTile,
                  tile_ub[slot]);
  }

  // Steady state: whole pairs only.
  std::uint32_t i = 0;
  for (; i + kTileSlots <= owned; i += kTileSlots) {
    DataSyncBarrier<MemDsbT::ALL>();  // publishes BOTH slots' VFs
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);

    // Slot index is a compile-time constant in each unrolled body, so the
    // scalar dispatch path carries no modulo and no branch.
    DataCopy(yGm[y_base + (tile_lo + (i + 0u) * tile_stride) * kRowsPerTile],
             tiles[0], kRowsPerTile);
    DataCopy(yGm[y_base + (tile_lo + (i + 1u) * tile_stride) * kRowsPerTile],
             tiles[1], kRowsPerTile);

    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);

    // Both slots are free again: issue the next pair.
    const std::uint32_t next = i + kTileSlots;
    if (next < owned) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    (tile_lo + next * tile_stride) * kRowsPerTile,
                    tile_ub[0]);
    }
    if (next + 1u < owned) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    (tile_lo + (next + 1u) * tile_stride) * kRowsPerTile,
                    tile_ub[1]);
    }
  }

  // Epilogue: an odd final tile. This USED to be unreachable -- `hidden` is a
  // multiple of 512 and 512/32 = 16, so hidden_tiles is always even -- and
  // the old comment said so. The gang decode is what made it live: `owned` is
  // ceil((hidden_tiles - lane)/gang_size), which is odd for many gang sizes
  // (128 tiles over a gang of 10 gives lanes 0..7 thirteen tiles each). It is
  // now an exercised path, not a defensive one.
  if (i < owned) {
    DataSyncBarrier<MemDsbT::ALL>();
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(yGm[y_base + (tile_lo + i * tile_stride) * kRowsPerTile],
             tiles[0], kRowsPerTile);
    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);
  }
}

// One expert, one token, end to end, on ONE core: the fused body, which is
// what prefill uses. `activated` stays in UB between the phases -- the point
// of the original design, and still true wherever one core owns a whole
// expert.
__aicore__ inline void RunExpertToken(
    const Geometry& g, __gm__ std::uint8_t* w1_packed,
    __gm__ std::uint8_t* w1_scale, __gm__ std::uint8_t* w3_packed,
    __gm__ std::uint8_t* w3_scale, __gm__ std::uint8_t* w2_packed,
    __gm__ std::uint8_t* w2_scale, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& actTensor,
    const LocalTensor<float> (&tiles)[kTileSlots],
    const GlobalTensor<float>& yGm, std::uint64_t y_base, float route_weight) {
  __ubuf__ float* x_ub = reinterpret_cast<__ubuf__ float*>(xTensor.GetPhyAddr());
  RunGateUpPhase(g, w1_packed, w1_scale, w3_packed, w3_scale, x_ub, actTensor,
                 0u, 1u, nullptr, 0u);
  RunDownPhase(g, w2_packed, w2_scale, actTensor, tiles, yGm, y_base,
               route_weight, 0u, 1u);
}

}  // namespace

// ---------------------------------------------------------------------------
// Entry point 1: decode. One token, top_k routed experts, one AIV per slot.
// ---------------------------------------------------------------------------
// GRID: block_dim == gang_size * top_k. Core c owns slot c/gang_size as lane
// c%gang_size, and the gang's lanes SPLIT that expert's row tiles in both
// phases. `activated` therefore crosses cores, which is why this entry point
// takes an `activated_ws` GM plane and a cross-core barrier -- see the
// reversal recorded in dsv4_moe_block_tiling.h.
//
// `activated_ws` is [top_k, inter] FP32, CALLER-OWNED, sized by
// Dsv4MoeDecodeWorkspaceBytes(). It is written before it is read within the
// launch, so it needs no initialization and nothing in it outlives the call.
//
// `y` is ACCUMULATED INTO with an atomic-add DataCopy, so:
//   * THE CALLER MUST ZERO `y` BEFORE THE LAUNCH. The wrapper does not, by
//     §3.4 -- it owns validation, staging, dispatch and teardown, not operand
//     data. LaunchDsv4MoeDecodeTopk says so in its contract.
//   * the slot summation order is whatever the cores finish in, and varies
//     run to run. Six partials of comparable magnitude reassociate within
//     ~1e-7 relative, far inside the 1e-2 the suites allow.
extern "C" __global__ __aicore__ void dsv4_moe_decode_topk(
    GM_ADDR x, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR topk_expert_ids, GM_ADDR routing_weights, GM_ADDR y,
    GM_ADDR activated_ws, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeBlockTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeBlockTilingData*>(tiling);
  if (unlikely(TilingRejected(td, kMoeBlockModeDecode))) {
    return;
  }
  const Geometry g = LoadGeometry(td);

  // GANG MAPPING. Core c owns slot c/gang_size as lane c%gang_size, and the
  // lane grid-strides over its slot's row tiles in both phases.
  const std::uint32_t core = static_cast<std::uint32_t>(GetBlockIdx());
  const std::uint32_t slot = core / g.gang_size;
  const std::uint32_t lane = core % g.gang_size;

  // ------------------------------------------------------------------------
  // NO EARLY RETURN BEYOND THIS LINE. There is a cross-core barrier below,
  // and SyncAll synchronizes every LAUNCHED core: a core that returns before
  // reaching it hangs the ones that do. So a core with nothing to do -- past
  // the last slot, or holding an out-of-range expert id -- is carried to the
  // barrier with `active` false instead of returning.
  //
  // The tiling rejection above is exempt because it is UNIFORM: the tiling is
  // identical for every core, so either all of them return or none do, and
  // there is no barrier left for anyone to wait at.
  // ------------------------------------------------------------------------
  bool active = slot < g.top_k;

  // The routed expert for this slot, and its router gate. Guarded, because an
  // inactive core must not index these buffers at all.
  std::uint32_t expert = 0u;
  float route_weight = 0.0f;
  if (active) {
    expert = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(topk_expert_ids)[slot]);
    if (unlikely(expert >= g.num_experts)) {
      active = false;  // backstop; the host validates the ids it can see
    } else {
      route_weight =
          reinterpret_cast<const __gm__ float*>(routing_weights)[slot];
    }
  }

  // Expert bases: widen BEFORE the multiply (§2.7). At 256 experts the w1
  // offset reaches ~1.07 GB, which fits uint32 only by one doubling.
  const std::uint64_t e = static_cast<std::uint64_t>(expert);
  __gm__ std::uint8_t* w1p =
      Bytes(w1_packed) + e * g.w13_packed_expert_stride;
  __gm__ std::uint8_t* w1s = Bytes(w1_scale) + e * g.w13_scale_expert_stride;
  __gm__ std::uint8_t* w3p =
      Bytes(w3_packed) + e * g.w13_packed_expert_stride;
  __gm__ std::uint8_t* w3s = Bytes(w3_scale) + e * g.w13_scale_expert_stride;
  __gm__ std::uint8_t* w2p = Bytes(w2_packed) + e * g.w2_packed_expert_stride;
  __gm__ std::uint8_t* w2s = Bytes(w2_scale) + e * g.w2_scale_expert_stride;

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), g.hidden);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), g.hidden);

  // DYNAMIC UB, not TPipe/TBuf -- see the header. Allocation order fixes the
  // offsets and must match Dsv4MoeBlockDynUbBytes(): xF32, actF32, tile.
  // Alloc()'s tileSize is in ELEMENTS, not bytes. The allocator is a
  // function-local bump pointer rather than a member because as a member its
  // inline constructor has to be emitted here and the toolkit leaves
  // LocalMemAllocator::LocalMemAllocator() without a reachable definition on
  // this include path (-Wundefined-inline).
  LocalMemAllocator<> alloc;
  LocalTensor<float> xTensor = alloc.Alloc<float>(g.hidden);
  LocalTensor<float> actTensor = alloc.Alloc<float>(g.inter);
  // The output tile ring. Allocation ORDER IS THE LAYOUT, so the skew is
  // allocated as its own segment between the two slots: back-to-back 128 B
  // slots would give a 256 B stride, the UB bank period.
  LocalTensor<float> tile0 = alloc.Alloc<float>(kRowsPerTile);
  LocalTensor<float> tileSkew = alloc.Alloc<float>(kTileSkewBytes / 4u);
  LocalTensor<float> tile1 = alloc.Alloc<float>(kRowsPerTile);
  (void)tileSkew;  // reserved spacer; never read or written
  const LocalTensor<float> tiles[kTileSlots] = {tile0, tile1};

  // `activated` for every slot, as one [top_k, inter] FP32 plane. This slot's
  // row starts at slot*inter; the gang's lanes write disjoint tiles of it.
  GlobalTensor<float> actGm;
  actGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activated_ws),
                        static_cast<std::uint64_t>(g.top_k) * g.inter);
  const std::uint64_t act_base = static_cast<std::uint64_t>(slot) * g.inter;

  if (active) {
    DataCopy(xTensor, xGm, g.hidden);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    DataSyncBarrier<MemDsbT::ALL>();

    // Phase 1, this lane's share of the gate/up rows, published to GM.
    //
    // ATOMIC MODE MUST BE NONE HERE. These stores are a plain handoff to the
    // gang, not an accumulation: with atomic-add still set from a previous
    // phase they would add into whatever the workspace held, which is exactly
    // the kind of silent wrong answer the workspace has no way to reveal.
    // The atomic window is opened around the DOWN phase only, below.
    __ubuf__ float* x_ub =
        reinterpret_cast<__ubuf__ float*>(xTensor.GetPhyAddr());
    RunGateUpPhase(g, w1p, w1s, w3p, w3s, x_ub, actTensor, lane, g.gang_size,
                   &actGm, act_base);
  }

  // Publish phase 1 to the rest of the gang. The DataSyncBarrier drains this
  // core's own pipelines; SyncAll is the cross-core edge. Both are needed and
  // in this order -- a barrier that lets a core past before its MTE3 stores
  // have landed publishes nothing.
  DataSyncBarrier<MemDsbT::ALL>();
  SyncAll<true>();

  if (active) {
    // Re-read the WHOLE slot's `activated`: the down GEMV reduces over all of
    // it, and only this lane's tiles were computed locally. 8 KiB at the
    // production geometry, against a weight stream of ~12 MiB per expert.
    DataCopy(actTensor, actGm[act_base], g.inter);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    DataSyncBarrier<MemDsbT::ALL>();

    // Phase 2, this lane's share of the down rows. The router gate folds into
    // the store and the slots accumulate into y, so the caller must have
    // zeroed y (see the contract above).
    SetAtomicAdd<float>();
    RunDownPhase(g, w2p, w2s, actTensor, tiles, yGm, 0u, route_weight, lane,
                 g.gang_size);
    SetAtomicNone();
  }
}

// ---------------------------------------------------------------------------
// Entry point 2: prefill. M tokens, one expert each, 32 AIVs over tokens.
// ---------------------------------------------------------------------------
// GRID: every AIV the device reports, grid-striding over `total_tokens` (and
// never more cores than tokens). A core owns whole tokens,
// so the monolithic body applies per token and `activated` is private to the
// core for the duration of each one. `x_tokens` is expected ALREADY GROUPED
// BY EXPERT (the standard MoE dispatch output), which is what makes a core's
// consecutive tokens share an expert and therefore its weight stream.
//
// No router weighting and no accumulation: each token has exactly one expert
// and one output row, so the store is a plain overwrite and needs no zeroed
// output.
//
// DEVIATION FROM THE REQUESTED ABI: no `expert_offsets` and no `workspace`
// operand. `token_to_expert` is the per-token truth and one load per token is
// cheaper than locating a CSR segment, so the CSR offsets would be an operand
// the kernel never reads; and with `activated` resident in UB this entry
// point has no scratch to place, so a workspace would be a required non-null
// pointer with no meaning. Grouping `x_tokens` by expert still matters -- it
// is what makes a core's consecutive tokens share a weight stream -- it is
// just the caller's arrangement rather than something passed in.
extern "C" __global__ __aicore__ void dsv4_moe_prefill_batched(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeBlockTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeBlockTilingData*>(tiling);
  if (unlikely(TilingRejected(td, kMoeBlockModePrefill))) {
    return;
  }
  const Geometry g = LoadGeometry(td);
  const std::uint32_t block_idx = static_cast<std::uint32_t>(GetBlockIdx());

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  const std::uint64_t elems =
      static_cast<std::uint64_t>(g.total_tokens) * g.hidden;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_tokens), elems);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_tokens), elems);

  LocalMemAllocator<> alloc;
  LocalTensor<float> xTensor = alloc.Alloc<float>(g.hidden);
  LocalTensor<float> actTensor = alloc.Alloc<float>(g.inter);
  // The output tile ring. Allocation ORDER IS THE LAYOUT, so the skew is
  // allocated as its own segment between the two slots: back-to-back 128 B
  // slots would give a 256 B stride, the UB bank period.
  LocalTensor<float> tile0 = alloc.Alloc<float>(kRowsPerTile);
  LocalTensor<float> tileSkew = alloc.Alloc<float>(kTileSkewBytes / 4u);
  LocalTensor<float> tile1 = alloc.Alloc<float>(kRowsPerTile);
  (void)tileSkew;  // reserved spacer; never read or written
  const LocalTensor<float> tiles[kTileSlots] = {tile0, tile1};

  // WEIGHT REUSE: THE GRID-STRIDE MAP IS THE REUSE MECHANISM. DO NOT BLOCK IT.
  //
  // It is tempting to give each core a CONTIGUOUS chunk of tokens so that a
  // core sees a run of same-expert tokens and re-reads w1/w3/w2 out of its
  // SIMT DataCache. That trade is backwards here, and the arithmetic says so.
  //
  // Weights are never staged in UB -- they cannot be: one expert's w1 packed
  // plane alone is inter * hidden/2 = 4 MiB at the production geometry,
  // against a 221,184 B usable UB ceiling. So there is no staged buffer to
  // reuse and no DMA to elide; the only thing a mapping can change is the
  // GM/L2 access pattern's locality.
  //
  // With this grid-stride map, at step s every core is working on token
  // block_idx + s * block_dim. If the CALLER presents tokens grouped by
  // expert -- which is the cheap host-side sort, and the actual lever -- then
  // at any instant all 32 cores are inside the same expert, so ONE expert's
  // ~12 MiB of w1+w3+w2 serves the whole grid and sits comfortably in L2.
  //
  // Contiguous chunking would instead put 32 DIFFERENT experts in flight at
  // once, ~384 MiB of working set against a 128 MiB L2: it buys per-core
  // temporal reuse and pays for it with grid-wide L2 thrash.
  //
  // So intra-expert batching for this kernel is a HOST ordering concern, not
  // a kernel mapping change. tests/emu/test_dsv4_moe_block_addressing_emu.cpp
  // pins the co-residency property this map depends on.
  //
  // NEVER MEASURED: no L2 or DataCache behaviour here has been observed on
  // hardware (§5.7). This is a working-set argument, not a benchmark result.
  for (std::uint32_t token = block_idx; token < g.total_tokens;
       token += g.block_dim) {
    const std::uint32_t expert = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert)[token]);
    if (unlikely(expert >= g.num_experts)) {
      continue;  // backstop; the host validates the ids it can see
    }
    const std::uint64_t e = static_cast<std::uint64_t>(expert);
    __gm__ std::uint8_t* w1p =
        Bytes(w1_packed) + e * g.w13_packed_expert_stride;
    __gm__ std::uint8_t* w1s = Bytes(w1_scale) + e * g.w13_scale_expert_stride;
    __gm__ std::uint8_t* w3p =
        Bytes(w3_packed) + e * g.w13_packed_expert_stride;
    __gm__ std::uint8_t* w3s = Bytes(w3_scale) + e * g.w13_scale_expert_stride;
    __gm__ std::uint8_t* w2p = Bytes(w2_packed) + e * g.w2_packed_expert_stride;
    __gm__ std::uint8_t* w2s = Bytes(w2_scale) + e * g.w2_scale_expert_stride;

    // The token's element offset. Widened before the multiply AND kept wide
    // (§2.7): at hidden = 4096 a uint32 row offset overflows past ~1.05 M
    // tokens, and narrowing here would defeat the widening.
    const std::uint64_t row = static_cast<std::uint64_t>(token) * g.hidden;

    DataCopy(xTensor, xGm[row], g.hidden);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    DataSyncBarrier<MemDsbT::ALL>();

    // Plain overwrite: one token, one expert, one output row.
    RunExpertToken(g, w1p, w1s, w3p, w3s, w2p, w2s, xTensor, actTensor, tiles,
                   yGm, row, 1.0f);
  }
}

}  // namespace moe::kernels::arch35::dsv4_moe_block
