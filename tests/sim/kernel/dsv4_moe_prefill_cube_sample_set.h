// SIM ONLY -- the inductive trip schedule for dsv4_moe_prefill_cube.
//
// Lives under tests/sim/kernel/ and is compiled into test targets only. It is
// ACL-free and toolkit-free so the HOST can unit-test it against a brute-force
// construction (tests/sim/test_dsv4_moe_prefill_cube_sample_set_sim.cpp): a
// sample-set builder is index arithmetic, and index arithmetic is validated on
// the host, never left to the simulator (CLAUDE.md §5.5).
//
// ---------------------------------------------------------------------------
// WHAT THE SAMPLING PROVES, AND WHAT IT CANNOT
// ---------------------------------------------------------------------------
// Running trips S instead of all N discharges exactly one obligation:
//
//   if the trip body is a function of (t, state) uniform in t away from the
//   ends, and state is carried only through a ring of depth D, then a
//   hazard-free rotation over 0..D plus an interior step with a successor plus
//   the tail proves the schedule for every N.
//
// For this kernel the body is `ProcessTokenTile(g, tile, ...)` and `tile` is
// its ONLY loop state -- the UB tensors are fully overwritten each trip, the
// Matmul objects are re-pointed with SetTensorA/SetTensorB and closed with
// End() inside the trip, and `y` rows are disjoint across tiles. So:
//
//   RING DEPTH D = 1.
//
// Not 2. The entry point calls pipe.InitBuffer(qAcc, 2, ...) and that 2 is a
// BUFFER COUNT, not a ping-pong rotation across trips: the two slots are
// `gate` and `up`, allocated once before the loop and held for the whole
// launch. There is no pointer swap between trips and nothing survives one, so
// a `t == 2` sample would add cost and prove nothing new. Recorded because
// reading `InitBuffer(q, 2, ...)` as "double buffered" and sampling {0,1,2}
// is the obvious mistake here.
//
// Four things the sampling is BLIND to, which must be covered elsewhere:
//
//   1. error that accrues over trips -- there is none to accrue: rows are
//      independent and nothing accumulates across tiles. Established on the
//      host, in tests/emu's row-independence check, where N is free.
//   2. indices outside the sample set -- with D = 1 and a tile map of
//      `row0 = tile * m_tile` there is no wrap, no modulo and no page
//      crossing to miss. The map itself is checked on the host.
//   3. full-count resource exhaustion -- the GM workspace high-water mark is
//      independent of the trip count here (one shared weight plane, see the
//      driver's note), and the event ids are released within each trip.
//   4. cross-core flag counts -- this kernel issues NO CrossCoreSetFlag /
//      CrossCoreWaitFlag and no SyncAll. Its cross-core traffic is the Matmul
//      KfcServer, where the AIC side is a SERVICE LOOP rather than a mirrored
//      trip loop: it answers however many messages the AIV sends and quits on
//      the client's teardown. Collapsing the AIV trip count therefore cannot
//      strand the AIC, which is the deadlock this technique usually has to be
//      careful about.
//
// ---------------------------------------------------------------------------
// THE SAMPLE SET
// ---------------------------------------------------------------------------
//   S = {0 .. D} U {mid, mid + 1} U {N - 2, N - 1},  mid = N / 2
//       clipped to [0, N - 1], deduplicated, ascending
//
// Role of each group:
//   0        -- cold start: the first MTE2 burst, the GM base pointer
//               arithmetic, the first expert's dequantization.
//   1 .. D   -- the induction step, one full rotation of the state ring. D = 1
//               here, so this is the single trip that must follow a cold one.
//   mid,     -- the invariant at an interior step, WITH A SUCCESSOR, so the
//   mid + 1     state a trip leaves is consumed by another trip and not merely
//               produced.
//   N-2, N-1 -- the epilogue, where `rows` goes partial when
//               total_tokens % m_tile != 0 and the last store lands at the
//               end of y.
//
// If N <= |S| the whole loop runs and the mechanism is bypassed. That branch
// is kept deliberately: small shapes are also the ones used in CI.

#ifndef TESTS_SIM_KERNEL_DSV4_MOE_PREFILL_CUBE_SAMPLE_SET_H
#define TESTS_SIM_KERNEL_DSV4_MOE_PREFILL_CUBE_SAMPLE_SET_H

#include <cstdint>

namespace moe::sim::prefill_cube {

// D = 1. See the derivation above; this is not the InitBuffer slot count.
inline constexpr std::uint32_t kRingDepth = 1u;

// (D + 1) head + 2 interior + 2 tail, at the largest ring depth the builder
// is unit-tested over (D = 4). Sized for the test, not for D = 1, so the host
// sweep and the device agree on one array bound.
inline constexpr std::uint32_t kMaxSamples = 9u;

// Macro plus two same-name wrappers, for the reason recorded on
// Dsv4MoePrefillCubeAdmissible in the operator's tiling header: BiSheng
// partitions functions into __host__ and __aicore__ and will not let one
// definition serve both. The body is written ONCE, here, so the host sweep and
// the device schedule cannot drift.
#define MOE_SIM_PREFILL_CUBE_BUILD_SAMPLES(total, depth, out, count)          \
  do {                                                                        \
    (count) = 0u;                                                             \
    if ((total) != 0u) {                                                      \
      const std::uint32_t _d = (depth) > 4u ? 4u : (depth);                   \
      const std::uint32_t _mid = (total) / 2u;                                \
      std::uint32_t _raw[moe::sim::prefill_cube::kMaxSamples];                \
      std::uint32_t _n_raw = 0u;                                              \
      for (std::uint32_t _i = 0u; _i <= _d; ++_i) {                           \
        _raw[_n_raw++] = _i;                                                   \
      }                                                                       \
      _raw[_n_raw++] = _mid;                                                   \
      _raw[_n_raw++] = _mid + 1u;                                              \
      _raw[_n_raw++] = (total) >= 2u ? (total) - 2u : 0u;                      \
      _raw[_n_raw++] = (total) - 1u;                                           \
      /* clip, deduplicate, and insert in ascending order. An insertion sort  \
       * over at most 9 entries: the point is that the device and the host    \
       * run the SAME construction, not that it is the fastest one. */        \
      for (std::uint32_t _i = 0u; _i < _n_raw; ++_i) {                        \
        const std::uint32_t _v = _raw[_i];                                     \
        if (_v >= (total)) {                                                   \
          continue;                                                            \
        }                                                                      \
        std::uint32_t _pos = 0u;                                               \
        bool _dup = false;                                                     \
        for (std::uint32_t _j = 0u; _j < (count); ++_j) {                      \
          if ((out)[_j] == _v) {                                               \
            _dup = true;                                                       \
          }                                                                    \
          if ((out)[_j] < _v) {                                                \
            _pos = _j + 1u;                                                    \
          }                                                                    \
        }                                                                      \
        if (_dup) {                                                            \
          continue;                                                            \
        }                                                                      \
        for (std::uint32_t _j = (count); _j > _pos; --_j) {                     \
          (out)[_j] = (out)[_j - 1u];                                           \
        }                                                                      \
        (out)[_pos] = _v;                                                       \
        ++(count);                                                              \
      }                                                                        \
    }                                                                          \
  } while (0)

// Writes the ascending, deduplicated sample set into `out` (which must hold
// kMaxSamples entries) and returns how many it wrote.
inline std::uint32_t BuildTileSampleSet(std::uint32_t total_tiles,
                                        std::uint32_t ring_depth,
                                        std::uint32_t* out) noexcept {
  std::uint32_t count = 0u;
  MOE_SIM_PREFILL_CUBE_BUILD_SAMPLES(total_tiles, ring_depth, out, count);
  return count;
}

#if defined(__CCE__)
__aicore__ inline std::uint32_t BuildTileSampleSet(std::uint32_t total_tiles,
                                                   std::uint32_t ring_depth,
                                                   std::uint32_t* out) {
  std::uint32_t count = 0u;
  MOE_SIM_PREFILL_CUBE_BUILD_SAMPLES(total_tiles, ring_depth, out, count);
  return count;
}
#endif

}  // namespace moe::sim::prefill_cube

#endif  // TESTS_SIM_KERNEL_DSV4_MOE_PREFILL_CUBE_SAMPLE_SET_H
