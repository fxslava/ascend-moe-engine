// SIM ONLY -- a collapsed, MULTI-CORE boundary-grid variant of the GANG
// DECODE path of dsv4_moe_block, for CAModel.
//
// Lives under tests/sim/kernel/ and is compiled into test targets only; no
// production library or wheel links it (tests/sim/CMakeLists.txt asserts
// that). Production op_kernel/ carries no verification branch because of it.
//
// WHY THIS EXISTS BESIDE dsv4_moe_block_inductive_kernel.cpp
// That file collapses the PREFILL path: one token, ONE core, the monolithic
// fused body. It contains no cross-core barrier and no gang mapping, so it
// cannot observe any of what the gang decode introduced:
//
//   * AscendC::SyncAll<true>() between the two phases, which every launched
//     core must reach or the ones that do hang;
//   * the strided down-ring, where lane L of a gang of G owns tiles
//     L, L+G, L+2G, ... instead of a contiguous run;
//   * `activated` crossing cores through a GM plane rather than staying
//     resident in one core's UB;
//   * the atomic-add accumulation of top_k slots into one `y` row.
//
// A single-core run satisfies SyncAll trivially, so it proves nothing about
// the barrier. This gate therefore runs a REAL grid: gang_size * top_k cores.
//
// WHAT IS SHARED AND WHAT IS NOT
// This TU #includes the production kernel, so every COMPUTE body here is
// verbatim production code: MoeBlockGateUpTileSimt, MoeBlockDownTileSimt,
// GroupDot, ReduceRow, LoadGeometry, TilingRejected, IssueDownTile,
// TilesOwned. Only the MACRO SCHEDULE differs -- which of a lane's tiles run.
// A collapsed run against a forked body would prove nothing.
//
// Including a .cpp is deliberate: the VFs are `inline` inside the production
// TU and cannot be referenced across a link, so sharing them verbatim means
// textual inclusion. The production __global__ entries come along unused, so
// this object must NOT be linked beside dsv4_moe_block_kernel.o.
//
// THE BARRIER IS NOT COLLAPSED, AND THAT IS THE WHOLE POINT
// Collapsing one side of a cross-core handshake and not the other deadlocks
// the simulator -- an artifact of the technique, not a kernel bug. Here the
// barrier is reached exactly once by every launched core on a straight-line
// path, sampled trips or not, which is the production invariant being tested.
// Cores with no work are carried to it with `active` false rather than
// returning early, exactly as production does.
//
// `activated` IS SEEDED BY THE HOST, NOT BY A DEVICE VF
// The gang reads the whole slot's `activated` plane back out of GM after the
// barrier, and the plane is caller-owned, so the host simply pre-fills it
// with the deterministic seed before the launch. Phase 1 overwrites only the
// sampled tiles; the rest reaches phase 2 as seed. That removes the device
// seeding VF the prefill gate needs, and with it one copy of the seed
// function -- the pattern now lives host-side only, so it cannot drift from
// the oracle that mirrors it.
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under CAModel
// only (CLAUDE.md §5.7).

#include "dsv4_moe_block_kernel.cpp"  // verbatim production bodies

namespace moe::kernels::arch35::dsv4_moe_block {
namespace decode_sim {

// Phase 1 samples kTileSlots + 2 = 4 positions. That fires the gate/up loop's
// cold start, its induction step, an interior step and its tail; the phase has
// no ring, so four is sufficient for it.
inline constexpr std::uint32_t kGateUpSamples = kTileSlots + 2u;  // 4

// Phase 2 samples FIVE, not four, and the extra trip is load-bearing.
//
// The ring's state machine advances per ITERATION, so with an EVEN sample
// count the paired steady-state loop consumes every sampled trip and the
// `i < owned` epilogue never executes. That epilogue was dead code before the
// gang (hidden_tiles is always even at admissible geometry) and the gang is
// exactly what made it live: a lane owns ceil((tiles - lane)/gang) tiles,
// which is odd for many gang sizes. Sampling an odd count is what puts the
// newly-live branch under the simulator.
//
// Cost of the fifth trip is one 32-row tile at full reduction length, against
// a ~358 s fixed simulator cost -- so this is the cheapest structural
// coverage available here.
inline constexpr std::uint32_t kDownSamples = kTileSlots + 3u;  // 5

// Boundary positions WITHIN ONE LANE'S OWN TILE SUBSEQUENCE.
//
// The sample is taken over the lane's subsequence index, not over the global
// tile index, so the lane's real strided addressing (tile = lane + pos*gang)
// is what executes. Sampling global indices instead would hand the lane tiles
// it does not own and test an addressing map production never performs.
//
// Positions: first, its ping-pong partner, the wrap (slot 0 reused), an
// interior pair, and the last. Clipped and deduplicated; returns the count.
__aicore__ inline std::uint32_t BuildLaneSamples(std::uint32_t owned,
                                                 std::uint32_t want,
                                                 std::uint32_t* out) {
  if (owned == 0u) {
    return 0u;
  }
  std::uint32_t raw[kDownSamples] = {0u, 1u, owned / 2u, owned / 2u + 1u,
                                     owned - 1u};
  // With `want` == 4 drop the {mid+1} entry: the four kept are head, partner,
  // interior, tail, which is the prefill gate's set shape.
  if (want < kDownSamples) {
    raw[3] = owned - 1u;
  }
  std::uint32_t n = 0u;
  for (std::uint32_t i = 0; i < want && i < kDownSamples; ++i) {
    if (raw[i] >= owned) {
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

// --- phase 1, collapsed: this lane's sampled gate/up tiles, published to GM
__aicore__ inline void GateUpPhaseInductive(
    const Geometry& g, __gm__ std::uint8_t* w1_packed,
    __gm__ std::uint8_t* w1_scale, __gm__ std::uint8_t* w3_packed,
    __gm__ std::uint8_t* w3_scale, __ubuf__ float* x_ub,
    const LocalTensor<float>& actTensor, std::uint32_t lane,
    std::uint32_t gang, const GlobalTensor<float>& actGm,
    std::uint64_t act_base) {
  __ubuf__ float* act_ub =
      reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr());
  const std::uint32_t inter_tiles =
      (g.inter + kRowsPerTile - 1u) / kRowsPerTile;
  const std::uint32_t owned = TilesOwned(inter_tiles, lane, gang);

  std::uint32_t pos[kDownSamples];
  const std::uint32_t n = BuildLaneSamples(owned, kGateUpSamples, pos);

  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t tile_base = (lane + pos[i] * gang) * kRowsPerTile;
    asc_vf_call<MoeBlockGateUpTileSimt>(
        dim3(kSimtThreads), w1_packed, w1_scale, w3_packed, w3_scale, x_ub,
        g.inter, g.hidden, g.w13_packed_row_stride, g.w13_scale_row_stride,
        g.swiglu_limit, tile_base, act_ub + tile_base);
    DataSyncBarrier<MemDsbT::ALL>();
    // Publish to the gang. V->MTE3 first: the VF's UB writes must retire
    // before DMA reads the same 32 floats. Production does exactly this.
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(actGm[act_base + tile_base], actTensor[tile_base], kRowsPerTile);
    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);
  }
}

// --- phase 2, collapsed: this lane's sampled down tiles through the ring ---
//
// The schedule is the production loop with the lane's subsequence index `i`
// replaced by `pos[k]`: one fused barrier per unrolled pair, stores from both
// slots, the next pair issued only AFTER MTE3_V (the WAR constraint the
// two-slot ring imposes), and the odd-remainder epilogue. Set/Wait stay
// balanced 1:1 per step, so no event token outlives the launch.
__aicore__ inline void DownPhaseInductive(
    const Geometry& g, __gm__ std::uint8_t* w2_packed,
    __gm__ std::uint8_t* w2_scale, const LocalTensor<float>& actTensor,
    const LocalTensor<float> (&tiles)[kTileSlots],
    const GlobalTensor<float>& yGm, std::uint64_t y_base, float route_weight,
    std::uint32_t lane, std::uint32_t gang) {
  __ubuf__ float* act_ub =
      reinterpret_cast<__ubuf__ float*>(actTensor.GetPhyAddr());
  __ubuf__ float* tile_ub[kTileSlots];
  for (std::uint32_t slot = 0; slot < kTileSlots; ++slot) {
    tile_ub[slot] =
        reinterpret_cast<__ubuf__ float*>(tiles[slot].GetPhyAddr());
  }

  const std::uint32_t hidden_tiles =
      (g.hidden + kRowsPerTile - 1u) / kRowsPerTile;
  const std::uint32_t owned = TilesOwned(hidden_tiles, lane, gang);

  std::uint32_t pos[kDownSamples];
  const std::uint32_t n = BuildLaneSamples(owned, kDownSamples, pos);
  if (n == 0u) {
    return;  // this lane owns no down tile; the caller has already barriered
  }

  const std::uint32_t primed = n < kTileSlots ? n : kTileSlots;
  for (std::uint32_t slot = 0; slot < primed; ++slot) {
    IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                  (lane + pos[slot] * gang) * kRowsPerTile, tile_ub[slot]);
  }

  std::uint32_t k = 0;
  for (; k + kTileSlots <= n; k += kTileSlots) {
    DataSyncBarrier<MemDsbT::ALL>();  // publishes BOTH slots' VFs
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);

    DataCopy(yGm[y_base + (lane + pos[k + 0u] * gang) * kRowsPerTile],
             tiles[0], kRowsPerTile);
    DataCopy(yGm[y_base + (lane + pos[k + 1u] * gang) * kRowsPerTile],
             tiles[1], kRowsPerTile);

    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);

    const std::uint32_t next = k + kTileSlots;
    if (next < n) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    (lane + pos[next] * gang) * kRowsPerTile, tile_ub[0]);
    }
    if (next + 1u < n) {
      IssueDownTile(g, w2_packed, w2_scale, act_ub, route_weight,
                    (lane + pos[next + 1u] * gang) * kRowsPerTile,
                    tile_ub[1]);
    }
  }

  // The odd remainder: with kDownSamples == 5 this ALWAYS fires, which is the
  // reason the count is odd. See kDownSamples.
  if (k < n) {
    DataSyncBarrier<MemDsbT::ALL>();
    SetFlag<HardEvent::V_MTE3>(kEvVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvVToMte3);
    DataCopy(yGm[y_base + (lane + pos[k] * gang) * kRowsPerTile], tiles[0],
             kRowsPerTile);
    SetFlag<HardEvent::MTE3_V>(kEvMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvMte3ToV);
  }
}

}  // namespace decode_sim
}  // namespace moe::kernels::arch35::dsv4_moe_block

// ---------------------------------------------------------------------------
// Sim entry: the gang decode on a collapsed per-lane grid.
// ---------------------------------------------------------------------------
// Decode-shaped operand list, including the caller-owned `activated_ws` plane,
// which the HOST pre-fills with the seed. block_dim is gang_size * top_k, so
// this runs a real multi-core gang.
//
// The control flow around the barrier is production's, unchanged: the only
// early return before SyncAll is the UNIFORM tiling rejection, and a core
// with no work is carried to the barrier with `active` false.
extern "C" __global__ __aicore__ void dsv4_moe_block_decode_inductive_sim(
    GM_ADDR x, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR topk_expert_ids, GM_ADDR routing_weights, GM_ADDR y,
    GM_ADDR activated_ws, GM_ADDR tiling) {
  using namespace moe::kernels::arch35;
  using namespace moe::kernels::arch35::dsv4_moe_block;
  using namespace moe::kernels::arch35::dsv4_moe_block::decode_sim;
  using AscendC::GlobalTensor;
  using AscendC::LocalTensor;

  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  if (unlikely(tiling == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoeBlockTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoeBlockTilingData*>(tiling);
  if (unlikely(TilingRejected(td, kMoeBlockModeDecode))) {
    return;  // uniform across cores, so no core is left waiting at a barrier
  }
  const Geometry g = LoadGeometry(td);

  const std::uint32_t core = static_cast<std::uint32_t>(AscendC::GetBlockIdx());
  const std::uint32_t slot = core / g.gang_size;
  const std::uint32_t lane = core % g.gang_size;

  // NO EARLY RETURN BEYOND THIS LINE -- see the production entry point.
  bool active = slot < g.top_k;

  std::uint32_t expert = 0u;
  float route_weight = 0.0f;
  if (active) {
    expert = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(topk_expert_ids)[slot]);
    if (unlikely(expert >= g.num_experts)) {
      active = false;
    } else {
      route_weight =
          reinterpret_cast<const __gm__ float*>(routing_weights)[slot];
    }
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

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  GlobalTensor<float> actGm;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), g.hidden);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), g.hidden);
  actGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(activated_ws),
                        static_cast<std::uint64_t>(g.top_k) * g.inter);
  const std::uint64_t act_base = static_cast<std::uint64_t>(slot) * g.inter;

  AscendC::LocalMemAllocator<> alloc;
  LocalTensor<float> xTensor = alloc.Alloc<float>(g.hidden);
  LocalTensor<float> actTensor = alloc.Alloc<float>(g.inter);
  LocalTensor<float> tile0 = alloc.Alloc<float>(kRowsPerTile);
  LocalTensor<float> tileSkew = alloc.Alloc<float>(kTileSkewBytes / 4u);
  LocalTensor<float> tile1 = alloc.Alloc<float>(kRowsPerTile);
  (void)tileSkew;
  const LocalTensor<float> tiles[kTileSlots] = {tile0, tile1};

  if (active) {
    AscendC::DataCopy(xTensor, xGm, g.hidden);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvMte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvMte2ToV);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    __ubuf__ float* x_ub =
        reinterpret_cast<__ubuf__ float*>(xTensor.GetPhyAddr());
    // Atomic mode is NONE here: these stores hand `activated` to the gang,
    // they do not accumulate. Production is explicit about this too.
    GateUpPhaseInductive(g, w1p, w1s, w3p, w3s, x_ub, actTensor, lane,
                         g.gang_size, actGm, act_base);
  }

  // THE BARRIER, reached once by every launched core.
  AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
  AscendC::SyncAll<true>();

  if (active) {
    // The whole slot's `activated`: the sampled tiles as phase 1 computed
    // them, every other row still carrying the host's seed.
    AscendC::DataCopy(actTensor, actGm[act_base], g.inter);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kEvMte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kEvMte2ToV);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    AscendC::SetAtomicAdd<float>();
    DownPhaseInductive(g, w2p, w2s, actTensor, tiles, yGm, 0u, route_weight,
                       lane, g.gang_size);
    AscendC::SetAtomicNone();
  }
}
