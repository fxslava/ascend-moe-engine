// SIM ONLY -- a boundary-grid variant of dsv4_moe_block for CAModel.
//
// Lives under tests/sim/kernel/ and is compiled into test targets only; no
// production library or wheel links it (tests/sim/CMakeLists.txt asserts
// that). Production op_kernel/ carries no verification branch because of it.
//
// WHAT IS SHARED AND WHAT IS NOT
// This TU #includes the production kernel, so every COMPUTE body here is
// verbatim production code: MoeBlockGateUpTileSimt, MoeBlockDownTileSimt,
// GroupDot, ReduceRow, LoadGeometry, TilingRejected, IssueDownTile. Only the
// MACRO SCHEDULE differs -- which tiles run. That is the whole point: a
// collapsed run against a forked body would prove nothing.
//
// Including a .cpp is deliberate. The VFs are `inline` inside the production
// TU, so they cannot be referenced across a link; sharing them verbatim means
// textual inclusion. The production __global__ entries come along unused, so
// this object must NOT be linked beside dsv4_moe_block_kernel.o.
//
// WHY A COLLAPSED GRID IS FAITHFUL HERE
// Phase 2 is a LINEAR reduction over `activated`:
//
//     y[r] = route_weight * sum_c w2[r][c] * activated[c]
//
// so the oracle does not need `activated` to be the true gate/up output -- it
// only needs to KNOW it. We therefore seed the whole `activated` extent with
// a deterministic, host-reproducible pattern, let phase 1 overwrite only the
// sampled tiles, and have the host oracle mirror exactly that vector. The
// comparison is then of the real kernel arithmetic, at full reduction length,
// on the sampled output rows.
//
// THE SEED MUST BE NUMERICALLY BENIGN AND INDEX-VARYING, and both halves of
// that matter:
//   * benign -- 0x5A5A5A5A reinterpreted as float is ~1.5e16. Pushing 384 of
//     those through the down reduction would swamp the real contributions and
//     the compare would lose all sensitivity. The pattern below lands in
//     [0.25, 1.0].
//   * varying -- a CONSTANT seed makes a phase-2 column-index bug invisible,
//     because every wrong slot holds the same value. Varying by index means a
//     misread column changes the sum.
//   * exactly representable -- multiples of 1/16, so seeding contributes no
//     rounding error of its own and host and device agree bit-for-bit on the
//     seeded entries.
//
// Bitwise 0x5A5A5A5A is still used, by the HOST, for GM regions nothing may
// write (output guard bands, unsampled y rows). Two canary roles, two values.
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under CAModel
// only (CLAUDE.md §5.7).

#include "dsv4_moe_block_kernel.cpp"  // verbatim production bodies

namespace moe::kernels::arch35::dsv4_moe_block {
namespace sim {

// kSlots + 2: the minimal grid that fires every structural branch of the
// phase-2 tile ring. With kTileSlots == 2 the collapsed loop runs two paired
// iterations -- prime(2), store pair, WRAP back into slots 0/1, store pair,
// then no re-issue (drain) -- so prime, flip, wrap and epilogue are all
// covered. The ring's state machine advances per ITERATION, not per tile
// index, which is why the sampled indices are free to be boundary values.
inline constexpr std::uint32_t kSampleTiles = kTileSlots + 2u;  // 4

// Deterministic `activated` seed. Mirrored host-side bit-for-bit.
__simt_callee__ __aicore__ inline float ActSeed(std::uint32_t index) {
  return 0.25f + static_cast<float>(index % 13u) * 0.0625f;
}

// Boundary sample set: first, its ping-pong partner, an interior tile, and the
// last tile. Clipped and deduplicated; returns how many are valid.
__aicore__ inline std::uint32_t BuildBoundarySamples(std::uint32_t total,
                                                     std::uint32_t* out) {
  if (total == 0u) {
    return 0u;
  }
  std::uint32_t raw[kSampleTiles] = {0u, 1u, total / 2u, total - 1u};
  std::uint32_t n = 0u;
  for (std::uint32_t i = 0; i < kSampleTiles; ++i) {
    if (raw[i] >= total) {
      continue;
    }
    bool seen = false;
    for (std::uint32_t j = 0; j < n; ++j) {
      if (out[j] == raw[i]) {
        seen = true;
      }
    }
    if (!seen) {
      out[n++] = raw[i];
    }
  }
  return n;
}

// Seed the whole `activated` extent before phase 1 runs.
__simt_vf__ __aicore__ __launch_bounds__(kSimtThreads) inline void
MoeBlockSeedActSimt(__ubuf__ float* act, std::uint32_t inter) {
  for (std::uint32_t i = threadIdx.x; i < inter; i += kSimtThreads) {
    act[i] = ActSeed(i);
  }
}

// One expert-token pair over the collapsed grid.
//
// Phase 1 runs the sampled `inter` tiles only; the rest of `activated` keeps
// its seed. Phase 2 runs the sampled `hidden` tiles only, each at FULL
// reduction length over all of `activated`, through the production tile ring.
__aicore__ inline void RunExpertTokenInductive(
    const Geometry& g, __gm__ std::uint8_t* w1_packed,
    __gm__ std::uint8_t* w1_scale, __gm__ std::uint8_t* w3_packed,
    __gm__ std::uint8_t* w3_scale, __gm__ std::uint8_t* w2_packed,
    __gm__ std::uint8_t* w2_scale, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& actTensor,
    const LocalTensor<float> (&tiles)[kTileSlots],
    const GlobalTensor<float>& yGm, std::uint64_t y_base, float route_weight) {
  __ubuf__ float* x_ub = reinterpret_cast<__ubuf__ float*>(xTensor.GetPhyAddr());
  __ubuf__ float* act_ub =
      reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr());
  __ubuf__ float* tile_ub[kTileSlots];
  for (std::uint32_t slot = 0; slot < kTileSlots; ++slot) {
    tile_ub[slot] =
        reinterpret_cast<__ubuf__ float*>(tiles[slot].GetPhyAddr());
  }

  // --- seed `activated`, then publish it ----------------------------------
  asc_vf_call<MoeBlockSeedActSimt>(dim3(kSimtThreads), act_ub, g.inter);
  DataSyncBarrier<MemDsbT::ALL>();

  // --- phase 1 on the sampled inter tiles ---------------------------------
  const std::uint32_t inter_tiles =
      (g.inter + kRowsPerTile - 1u) / kRowsPerTile;
  std::uint32_t p1[kSampleTiles];
  const std::uint32_t n1 = BuildBoundarySamples(inter_tiles, p1);
  for (std::uint32_t i = 0; i < n1; ++i) {
    const std::uint32_t tile_base = p1[i] * kRowsPerTile;
    asc_vf_call<MoeBlockGateUpTileSimt>(
        dim3(kSimtThreads), w1_packed, w1_scale, w3_packed, w3_scale, x_ub,
        g.inter, g.hidden, g.w13_packed_row_stride, g.w13_scale_row_stride,
        g.swiglu_limit, tile_base, act_ub + tile_base);
    DataSyncBarrier<MemDsbT::ALL>();
  }

  // --- phase 2 on the sampled hidden tiles, through the production ring ---
  //
  // The schedule below is the production loop with `tile_idx` replaced by
  // `p2[k]`: one fused barrier per unrolled pair, stores from both slots, then
  // the next pair issued only AFTER MTE3_V (the WAR constraint the two-slot
  // ring imposes). Set/Wait stay balanced 1:1 per step, so no event token
  // outlives the launch.
  const std::uint32_t hidden_tiles =
      (g.hidden + kRowsPerTile - 1u) / kRowsPerTile;
  std::uint32_t p2[kSampleTiles];
  const std::uint32_t n2 = BuildBoundarySamples(hidden_tiles, p2);

  const std::uint32_t primed = n2 < kTileSlots ? n2 : kTileSlots;
  for (std::uint32_t slot = 0; slot < primed; ++slot) {
    IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                  p2[slot] * kRowsPerTile, tile_ub[slot]);
  }

  std::uint32_t k = 0;
  for (; k + kTileSlots <= n2; k += kTileSlots) {
    DataSyncBarrier<MemDsbT::ALL>();  // publishes BOTH slots' VFs
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);

    DataCopy(yGm[y_base + p2[k + 0u] * kRowsPerTile], tiles[0], kRowsPerTile);
    DataCopy(yGm[y_base + p2[k + 1u] * kRowsPerTile], tiles[1], kRowsPerTile);

    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);

    const std::uint32_t next = k + kTileSlots;
    if (next < n2) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    p2[next] * kRowsPerTile, tile_ub[0]);
    }
    if (next + 1u < n2) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    p2[next + 1u] * kRowsPerTile, tile_ub[1]);
    }
  }

  // Odd remainder of the sample set.
  if (k < n2) {
    DataSyncBarrier<MemDsbT::ALL>();
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(yGm[y_base + p2[k] * kRowsPerTile], tiles[0], kRowsPerTile);
    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);
  }
}

}  // namespace sim
}  // namespace moe::kernels::arch35::dsv4_moe_block

// ---------------------------------------------------------------------------
// Sim entry: one token, one expert, one core, collapsed grid.
// ---------------------------------------------------------------------------
// Prefill-shaped operand list so the host side stays simple. block_dim is 1:
// the point is to minimise simulated work, and one core exercises both phases
// and the whole tile ring.
//
// `y` is OVERWRITTEN on the sampled rows and left untouched elsewhere, so the
// host can poison the whole output and assert the unsampled rows still hold
// the poison. That is what makes a store to a wrong y row visible.
extern "C" __global__ __aicore__ void dsv4_moe_block_inductive_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR tiling) {
  // Both levels: the tiling POD and the mode constants live in `arch35`, the
  // kernel internals in `arch35::dsv4_moe_block`. The production entry is
  // defined inside the inner namespace so it reaches both by ordinary lookup;
  // this entry is at global scope and has to say so.
  using namespace moe::kernels::arch35;
  using namespace moe::kernels::arch35::dsv4_moe_block;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeBlockTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeBlockTilingData*>(tiling);
  // The production predicate, verbatim: a wrong shape must not reach the
  // collapsed path either.
  if (unlikely(TilingRejected(td, kMoeBlockModePrefill))) {
    return;
  }
  const Geometry g = LoadGeometry(td);

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  const std::uint64_t elems =
      static_cast<std::uint64_t>(g.total_tokens) * g.hidden;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_tokens), elems);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_tokens), elems);

  LocalMemAllocator<> alloc;
  LocalTensor<float> xTensor = alloc.Alloc<float>(g.hidden);
  LocalTensor<float> actTensor = alloc.Alloc<float>(g.inter);
  LocalTensor<float> tile0 = alloc.Alloc<float>(kRowsPerTile);
  LocalTensor<float> tileSkew = alloc.Alloc<float>(kTileSkewBytes / 4u);
  LocalTensor<float> tile1 = alloc.Alloc<float>(kRowsPerTile);
  (void)tileSkew;
  const LocalTensor<float> tiles[kTileSlots] = {tile0, tile1};

  // The PRODUCTION grid-stride token loop, verbatim. The test passes
  // total_tokens == 1, so core 0 runs token 0 and the other 31 cores fall
  // straight out -- the grid shrinks without overriding any tiling field, and
  // the token->core mapping under test is still the real one.
  const std::uint32_t block_idx = static_cast<std::uint32_t>(GetBlockIdx());
  for (std::uint32_t token = block_idx; token < g.total_tokens;
       token += g.block_dim) {
    const std::uint32_t expert = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert)[token]);
    if (unlikely(expert >= g.num_experts)) {
      continue;
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

    const std::uint64_t row = static_cast<std::uint64_t>(token) * g.hidden;
    DataCopy(xTensor, xGm[row], g.hidden);
    SetFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvMte2ToV);
    DataSyncBarrier<MemDsbT::ALL>();

    sim::RunExpertTokenInductive(g, w1p, w1s, w3p, w3s, w2p, w2s, xTensor,
                                 actTensor, tiles, yGm, row, 1.0f);
  }
}
