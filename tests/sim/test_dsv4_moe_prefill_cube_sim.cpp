/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// SPARSE BOUNDARY EXECUTION of the PRODUCTION entry point dsv4_moe_prefill_cube
// on the dav-3510 CAModel, KERNEL_TYPE_MIX_AIC_1_2.
//
// ===========================================================================
// WHAT IS NEW HERE, AND WHY IT IS A SEPARATE DRIVER
// ===========================================================================
// Every other prefill-cube gate in this directory drives something that is not
// the shipping kernel:
//
//   Dsv4MoePrefillCubeInductiveSim / ...SingleTileSim
//       drive dsv4_moe_prefill_cube_inductive_sim, a sim-only entry whose body
//       is the QUARANTINED Matmul-framework path (ProcessTokenTile, KfcServer,
//       REGIST_MATMUL_OBJ) in dsv4_moe_prefill_cube_legacy_matmul.h.
//   Dsv4FixpipeDualDstSim (STEP 1) / Dsv4DualDstPipelineSim (STEP 2)
//       drive hand-written probes that reproduce ONE primitive each.
//
// The bare-metal three-stage schedule that ships -- RunVectorCore + RunCubeCore
// reached through the dsv4_moe_prefill_cube __global__ entry -- has only ever
// been COMPILED AND LINKED for dav-3510 (see the kernel's own "WHAT IS VERIFIED
// AND WHAT IS NOT"). This driver is the first thing that executes it.
//
// It launches through the PRODUCTION host wrapper LaunchDsv4MoePrefillCube,
// unmodified: same tiling builder, same aliasing rejections, same staging blob,
// same QueryAivCoreCount(), same chevron. There is no sim-only launch wrapper
// and no sim-only kernel TU in this gate's link, which is what makes a pass a
// statement about the shipping operator rather than about a copy of it.
//
// ===========================================================================
// THE SPARSE BOUNDARY SCHEDULE, AND WHAT MAKES IT A BOUNDARY
// ===========================================================================
// Dense simulation of the production geometry (hidden 4096, inter 2048) is out
// of reach: phase 1 alone is (inter/R_SUB) * (hidden/K_SUB) = 16 * 32 = 512
// contraction steps per token tile, each with two dequantized weight streams.
//
// The collapse is applied to the AXIS, not to the loop. hidden is reduced until
// the phase 1 contraction is exactly K_TILES = 8 steps:
//
//     fwd_steps = hidden / K_SUB = 1024 / 128 = 8
//
// so one gate/up contraction runs k = 0..7, and the head boundary k in [0, 3]
// and the tail boundary k in [4, 7] TOGETHER ARE the whole contraction. The L1
// ring is depth 2, so the ring-depth-aware sample set
// S = {0..D} u {mid, mid+1} u {N-2, N-1} = {0,1,2,3,4,6,7} is a SUBSET of the 8
// steps actually executed -- which is why this run is dense over the collapsed
// axis rather than gated by a sampled trip predicate. Nothing is skipped, so
// nothing can be blind to an index that the sample set happened to miss.
//
// CAPPING THE LOOP INSTEAD WAS REJECTED, for the reason the single-tile driver
// already records: a capped k loop leaves L0C holding a PARTIAL contraction,
// which is not what the operator computes and could not be compared against the
// audited emu golden at all. Shrinking the axis keeps every contraction whole,
// so moe_prefill_cube_flash_2d_w4a16_cpu applies verbatim and the comparison is
// against the operator's real answer on a smaller problem.
//
// WHAT THE 8 STEPS BUY OVER THE EXISTING SINGLE-TILE SHAPE (hidden 128, one
// step), stated as what is unreachable below 8:
//
//   * L0C init-on-step-0 followed by SEVEN in-place FP32 accumulations. At one
//     step ccubeInitVal is always true and the accumulating path never runs.
//   * The depth-2 L1 ring WRAPS FOUR TIMES, so the AIV's WAR wait
//     (k >= kCubeRingSlots) and the AIC's slot release both execute.
//   * The AIC's Mode 4 point-to-point kCubeFlagStepFree release is guarded by
//     k + kCubeRingSlots < fwd_steps, so it NEVER FIRES below 3 steps. At 8 it
//     fires for k = 0..5, six times per contraction, on both sub-block channels
//     (base + 0 and base + kCubeIntraFlagStride).
//   * PipeBarrier<PIPE_M> between consecutive Mmads into one L0C. The rule is
//     (m/16)*(n/16) < 10, which phase 1 (1*8 = 8) and phase 2 (1*4 = 4) are
//     both under. Removing it was MEASURED to turn a four-tile contraction into
//     one tile's product with no fault; at one step there is no consecutive
//     pair for it to order.
//
// inter is held at its minimum admissible 256, because every unit of it
// multiplies BOTH phases: phase 1 runs inter/R_SUB = 2 contractions of 8 steps,
// phase 2 runs hidden/H_SUB = 16 contractions of inter/R_SUB2 = 1 step. The
// phase 2 contraction is therefore SINGLE-STEP at this shape and does not
// accumulate -- said plainly rather than left to be inferred. Raising inter to
// 512 (the override's cap) would make it two steps and roughly double the run;
// the phase 2 slot release is unreachable either way, since it needs
// down_steps > kCubeRingSlots, i.e. inter > 512.
//
// ===========================================================================
// WHAT A PASS LICENSES
// ===========================================================================
// "dsv4_moe_prefill_cube executes on the cycle-accurate dav-3510 CAModel under
// MIX_AIC_1_2 over an 8-step phase 1 contraction and a 16-block phase 2, and
// its 16 x hidden output matches the audited two-phase Cube golden."
//
// It does NOT license a hardware claim (CLAUDE.md section 5.7), the production
// geometry's 32-step contraction, the ragged token tail, the multi-cluster grid
// stride, or phase 2 accumulation across more than one step.
//
// The operand transpose is covered NUMERICALLY, and that is deliberate:
// CubeLoadWeightsToL0B reinterprets the AIVs' NZ(N, K) write as ZN(K, N) to
// obtain B = W^T with no instruction spent, a step the kernel marks "DERIVED
// FROM THE TOOLKIT'S LAYOUT ARITHMETIC, NOT MEASURED". An index corruption
// there is not a small error: it delivers W[n][k] where W[k][n] was wanted,
// which the golden sees as a structurally different matrix. Cosine is held
// strict at 0.999 for exactly that reason -- rounding moves magnitude, a
// transposed operand moves direction.
//
// TOLERANCE IS SET BY THE FORMAT (CLAUDE.md section 4.7), not fitted. The only
// lossy step in this W4A16 path is the FP32 -> BF16 cast of the activation,
// which keeps an 8-bit mantissa and so carries up to 2^-9 ~ 2.0e-3 relative
// error per store; MXFP4 -> BF16 and the E8M0 power-of-two group scale are both
// exact, and the contraction itself stays FP32 in L0C for the whole K axis, so
// there is NO per-step rounding to accumulate. Two chained phases each cast
// once, so 1e-2 absolute plus 1e-2 relative is about a 5x envelope over the
// format's own bound.

#include "dsv4_moe_block_sim_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// WHICH ENTRY POINT THIS BINARY DRIVES
// ---------------------------------------------------------------------------
// ONE host driver, two executables, so the comparison between them has exactly
// one variable. Without the macro this is the production gate; with
// DSV4_BOUNDARY_NOPRIME_SIM it is the bisection probe whose kernel differs from
// production only in that the AIV's V_MTE2 priming pair is replaced by a
// PipeBarrier<PIPE_ALL> (see kernel/dsv4_moe_prefill_cube_noprime_kernel.cpp
// for the measurement that motivates it).
//
// They cannot share one binary: the probe's TU textually includes the
// production kernel, so it also defines `dsv4_moe_prefill_cube`, and two such
// objects do not link together.
#if defined(DSV4_BOUNDARY_TRACE_SIM)
namespace moe::kernels::arch35 {
Status LaunchDsv4MoePrefillCubeTraceSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, float* y_tokens_gm, void* workspace_gm,
    aclrtStream stream);
}  // namespace moe::kernels::arch35
#define DSV4_BOUNDARY_LAUNCH \
  ::moe::kernels::arch35::LaunchDsv4MoePrefillCubeTraceSim
#define DSV4_BOUNDARY_LAUNCH_NAME "LaunchDsv4MoePrefillCubeTraceSim"
#define DSV4_BOUNDARY_ENTRY_NAME \
  "dsv4_moe_prefill_cube_trace_sim (PROGRESS TRACE: device prints at stage " \
  "boundaries)"
#elif defined(DSV4_BOUNDARY_NOPRIME_SIM)
namespace moe::kernels::arch35 {
Status LaunchDsv4MoePrefillCubeNoPrimeSim(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, float* y_tokens_gm, void* workspace_gm,
    aclrtStream stream);
}  // namespace moe::kernels::arch35
#define DSV4_BOUNDARY_LAUNCH \
  ::moe::kernels::arch35::LaunchDsv4MoePrefillCubeNoPrimeSim
#define DSV4_BOUNDARY_LAUNCH_NAME "LaunchDsv4MoePrefillCubeNoPrimeSim"
#define DSV4_BOUNDARY_ENTRY_NAME \
  "dsv4_moe_prefill_cube_noprime_sim (BISECTION PROBE: no V_MTE2 priming)"
#else
#define DSV4_BOUNDARY_LAUNCH ::moe::kernels::arch35::LaunchDsv4MoePrefillCube
#define DSV4_BOUNDARY_LAUNCH_NAME "LaunchDsv4MoePrefillCube"
#define DSV4_BOUNDARY_ENTRY_NAME "dsv4_moe_prefill_cube (PRODUCTION entry)"
#endif

namespace {

using moe::kernels::arch35::kPrefillCubeGroupSize;
using moe::kernels::arch35::kPrefillCubeHSub;
using moe::kernels::arch35::kPrefillCubeKSub;
using moe::kernels::arch35::kPrefillCubeMTile;
using moe::kernels::arch35::kPrefillCubeRSub;
using moe::kernels::arch35::kPrefillCubeRSub2;

// EXACTLY ONE FULL TOKEN TILE. total_tiles == 1 makes block_dim come out
// min(1, aiv) == 1 whatever the simulated core count is, so the grid is one AIC
// and its two paired AIVs -- the minimum topology this MIX kernel runs in, and
// the only one in which the boundary schedule is the whole of the work.
constexpr std::uint32_t kBoundaryTokens = kPrefillCubeMTile;
static_assert(kBoundaryTokens % kPrefillCubeMTile == 0u,
              "the tile must be FULL: the ragged token tail is a different "
              "gate's job and would change what the Fixpipe M split means");

// THE BOUNDARY SHAPE. hidden 1024 is chosen for one reason only: it is the axis
// length at which fwd_steps lands on 8.
constexpr std::uint32_t kBoundaryHidden = 1024;
constexpr std::uint32_t kBoundaryInter = 256;
constexpr std::uint32_t kBoundaryKTiles = kBoundaryHidden / kPrefillCubeKSub;
static_assert(kBoundaryKTiles == 8u,
              "this driver is the 8-step boundary run: 4 head steps k in [0,3] "
              "and 4 tail steps k in [4,7]. If K_SUB moved, re-derive hidden "
              "rather than letting the step count drift");
static_assert(kBoundaryHidden % kPrefillCubeKSub == 0u &&
                  kBoundaryHidden % kPrefillCubeHSub == 0u &&
                  kBoundaryInter % kPrefillCubeRSub == 0u &&
                  kBoundaryInter % kPrefillCubeRSub2 == 0u &&
                  kBoundaryHidden % kPrefillCubeGroupSize == 0u &&
                  kBoundaryInter % kPrefillCubeGroupSize == 0u,
              "MakeDsv4MoePrefillCubeTiling rejects a shape that is not a whole "
              "multiple of every sub-tile extent, and the run would never reach "
              "the kernel");

// The routed expert, deliberately not 0: a nonzero w*_packed_expert_stride is
// what makes a mis-computed stride visible instead of masked by a base read.
constexpr std::int32_t kBoundaryExpert = 2;
static_assert(kBoundaryExpert < static_cast<std::int32_t>(kExperts),
              "the routed expert must exist in the bank");

constexpr float kBoundaryAbsTol = 1.0e-2F;
constexpr float kBoundaryRelTol = 1.0e-2F;
constexpr double kBoundaryCosFloor = 0.999;

// kCubeRingSlots MIRRORED, NOT INCLUDED. The L1 ring depth lives in
// op_kernel/dsv4_moe_prefill_cube_arch35.h, which is operator-private
// (CLAUDE.md section 1.2) and must not be reached from a test. It is restated
// here only to predict the release census this run should drive; if the two ever
// disagree the printed census is wrong, not the kernel.
constexpr std::uint32_t kRingSlotsMirror = 2;

// A TIME BOX, NOT A DEADLOCK DETECTOR. A CAModel keeps advancing cycles while
// every core sits blocked on a flag, so host wall clock cannot tell a hang from
// a slow simulation; exceeding this means "did not finish in the budget" and
// nothing more. To call a hang, show the cycle count is not advancing between
// two trip points, or a WaitFlag with no matching SetFlag.
//
// NOT DERIVED FROM A COMPLETED RUN OF THIS SHAPE -- no run of this entry point
// existed when it was written. The nearest measurements on this workstation are
// the STEP 2 probe (one 4-step contraction of the same schedule at M 16 /
// N 128 / K_SUB 128: 337.6 s) and the STEP 1 probe (one Fixpipe step:
// 14.5-16.0 s). Pass an explicit budget as the third argument once this shape
// has a measurement of its own.
constexpr int kBoundaryBudgetSeconds = 7200;

struct Spread {
  double rms = 0.0, stddev = 0.0, min = 0.0, max = 0.0;
};

Spread Measure(const std::vector<float>& v) {
  Spread s;
  if (v.empty()) return s;
  double sum = 0.0, sum_sq = 0.0;
  s.min = v[0];
  s.max = v[0];
  for (const float value : v) {
    const double x = value;
    sum += x;
    sum_sq += x * x;
    s.min = std::min(s.min, x);
    s.max = std::max(s.max, x);
  }
  const double n = static_cast<double>(v.size());
  const double mean = sum / n;
  const double mean_sq = sum_sq / n;
  s.rms = std::sqrt(mean_sq);
  s.stddev = std::sqrt(std::max(0.0, mean_sq - mean * mean));
  return s;
}

struct BoundaryDiff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double cosine = 1.0;
  bool finite = true;
  std::size_t worst = 0;
  std::size_t wrong = 0;  // elements outside the envelope -- the headline count
};

BoundaryDiff CompareTile(const std::vector<float>& got,
                         const std::vector<float>& want) {
  BoundaryDiff d;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double a = got[i];
    const double b = want[i];
    if (!std::isfinite(a) || !std::isfinite(b)) {
      d.finite = false;
      ++d.wrong;
      continue;
    }
    const double delta = std::fabs(a - b);
    if (delta > d.max_abs) {
      d.max_abs = delta;
      d.worst = i;
    }
    if (std::fabs(b) > 1.0e-6) {
      d.max_rel = std::max(d.max_rel, delta / std::fabs(b));
    }
    if (delta > kBoundaryAbsTol + kBoundaryRelTol * std::fabs(b)) {
      ++d.wrong;
    }
    dot += a * b;
    na += a * a;
    nb += b * b;
  }
  d.cosine = (na == 0.0 && nb == 0.0)
                 ? 1.0
                 : ((na == 0.0 || nb == 0.0)
                        ? 0.0
                        : dot / (std::sqrt(na) * std::sqrt(nb)));
  return d;
}

// Tells "never written" apart from "written wrong". A kernel that returned early
// -- the silent failure mode CLAUDE.md section 2.5 is about -- leaves the tile
// untouched or zeroed, and both read as a wrong answer unless counted.
struct Occupancy {
  std::size_t canary = 0;
  std::size_t zero = 0;
  std::size_t live = 0;
};

Occupancy MeasureOccupancy(const std::vector<float>& y) {
  Occupancy o;
  for (const float v : y) {
    if (IsCanary(v)) {
      ++o.canary;
    } else if (v == 0.0F) {
      ++o.zero;
    } else {
      ++o.live;
    }
  }
  return o;
}

std::string CheckWorkspaceUnwritten(const std::vector<std::uint8_t>& probe) {
  for (std::size_t i = 0; i < probe.size(); ++i) {
    if (probe[i] != 0x5Au) {
      return "workspace byte " + std::to_string(i) +
             " past the declared reserve was written (0x" +
             std::to_string(static_cast<unsigned>(probe[i])) +
             "): the kernel spilled into GM it was not given";
    }
  }
  return {};
}

std::uint32_t ParseExtent(const char* text, std::uint32_t fallback) {
  if (text == nullptr) return fallback;
  const long value = std::strtol(text, nullptr, 10);
  if (value <= 0) return fallback;
  return static_cast<std::uint32_t>(value);
}

// The trip census this run is expected to drive, derived from the kernel's own
// loop bounds. Printed before the launch so the cost and the coverage are
// visible rather than discovered as a timeout, and so a shape passed on the
// command line cannot quietly stop being a boundary run.
struct Schedule {
  std::uint32_t fwd_steps = 0;       // phase 1 contraction length, the K_TILES
  std::uint32_t fwd_blocks = 0;      // phase 1 inter output blocks
  std::uint32_t down_steps = 0;      // phase 2 contraction length
  std::uint32_t down_blocks = 0;     // phase 2 hidden output blocks
  std::uint32_t head_last = 0;       // last k of the head boundary
  std::uint32_t tail_first = 0;      // first k of the tail boundary
  std::uint32_t steps = 0;           // total Mmad-accumulate steps driven
  std::uint32_t accumulations = 0;   // in-place L0C accumulations (init false)
  std::uint32_t l0c_inits = 0;       // ccubeInitVal true, once per contraction
  std::uint32_t slot_frees_fwd = 0;  // Mode 4 kCubeFlagStepFree, phase 1
  std::uint32_t slot_frees_down = 0; // Mode 4 kCubeFlagStepFree, phase 2
  std::uint32_t fixpipe_moveouts = 0;
};

Schedule DeriveSchedule(std::uint32_t hidden, std::uint32_t inter) {
  Schedule s;
  s.fwd_steps = hidden / kPrefillCubeKSub;
  s.fwd_blocks = inter / kPrefillCubeRSub;
  s.down_steps = inter / kPrefillCubeRSub2;
  s.down_blocks = hidden / kPrefillCubeHSub;
  s.head_last = s.fwd_steps < 4u ? (s.fwd_steps - 1u) : 3u;
  s.tail_first = s.fwd_steps > 4u ? (s.fwd_steps - 4u) : 0u;
  // Phase 1 runs TWO Mmads per step (gate and up) against one token tile, so
  // the accumulate count is doubled there and the L0C init count is two per
  // inter block.
  s.steps = s.fwd_blocks * s.fwd_steps + s.down_blocks * s.down_steps;
  s.accumulations = s.fwd_blocks * 2u * (s.fwd_steps - 1u) +
                    s.down_blocks * (s.down_steps - 1u);
  s.l0c_inits = s.fwd_blocks * 2u + s.down_blocks;
  s.slot_frees_fwd =
      s.fwd_blocks *
      (s.fwd_steps > kRingSlotsMirror ? s.fwd_steps - kRingSlotsMirror : 0u);
  s.slot_frees_down =
      s.down_blocks *
      (s.down_steps > kRingSlotsMirror ? s.down_steps - kRingSlotsMirror : 0u);
  // Phase 1 hands gate and up over through ONE landing zone, so two per block.
  s.fixpipe_moveouts = s.fwd_blocks * 2u + s.down_blocks;
  return s;
}

int RunBoundary(std::uint32_t hidden, std::uint32_t inter, int budget_seconds) {
  using namespace moe::kernels::arch35;

  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const Schedule sched = DeriveSchedule(hidden, inter);

  std::printf(
      "SPARSE BOUNDARY dav-3510 CAModel run (MIX_AIC_1_2)\n"
      "entry: %s\n"
      "seed 0x%llX  hidden %u  inter %u  experts %u  tokens %u  m_tile %u  "
      "tiles 1  expert %d\n",
      DSV4_BOUNDARY_ENTRY_NAME,
      static_cast<unsigned long long>(kSeed), hidden, inter, kExperts,
      kBoundaryTokens, kPrefillCubeMTile, static_cast<int>(kBoundaryExpert));
  std::printf(
      "  phase 1: %u inter block(s) x K_TILES %u   head k in [0,%u]  tail k in "
      "[%u,%u]\n"
      "  phase 2: %u hidden block(s) x %u step(s)\n"
      "  total contraction steps %u   L0C inits %u   in-place accumulations "
      "%u\n"
      "  Mode 4 kCubeFlagStepFree releases: %u phase 1 + %u phase 2, each on 2 "
      "sub-block channels\n"
      "  Fixpipe DUAL_DST_SPLIT_M move-outs %u   PipeBarrier<PIPE_M> sites %u\n",
      sched.fwd_blocks, sched.fwd_steps, sched.head_last, sched.tail_first,
      sched.fwd_steps - 1u, sched.down_blocks, sched.down_steps, sched.steps,
      sched.l0c_inits, sched.accumulations, sched.slot_frees_fwd,
      sched.slot_frees_down, sched.fixpipe_moveouts, sched.steps);

  const DumpCensus before = ExceptionDumps();

  try {
    if (!Dsv4MoePrefillCubeAdmissible(hidden, inter)) {
      Fail("hidden " + std::to_string(hidden) + " / inter " +
           std::to_string(inter) +
           " is not admissible; this binary must be built with "
           "DSV4_TILING_TEST_OVERRIDE=1 and the shape must satisfy it");
    }

    std::mt19937_64 rng(kSeed);
    const ExpertBank w1 = MakeBank(inter, hidden, kExperts, rng);
    const ExpertBank w3 = MakeBank(inter, hidden, kExperts, rng);
    const ExpertBank w2 = MakeBank(hidden, inter, kExperts, rng);

    const std::size_t y_floats =
        static_cast<std::size_t>(kBoundaryTokens) * hidden;
    std::vector<float> x(y_floats);
    std::normal_distribution<float> draw(0.0F, 1.0F);
    for (float& v : x) v = draw(rng);

    const std::vector<std::int32_t> route(kBoundaryTokens, kBoundaryExpert);

    Runtime runtime;
    LaunchWatchdog watchdog(budget_seconds, "[PREFILL-CUBE-BOUNDARY]");

    DeviceBuffer xDev(x.size() * sizeof(float));
    CopyToDevice(xDev, x);
    DeviceBuffer w1p(w1.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w1p, w1.packed);
    DeviceBuffer w1s(w1.scales.size() * sizeof(E8M0));
    CopyToDevice(w1s, w1.scales);
    DeviceBuffer w3p(w3.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w3p, w3.packed);
    DeviceBuffer w3s(w3.scales.size() * sizeof(E8M0));
    CopyToDevice(w3s, w3.scales);
    DeviceBuffer w2p(w2.packed.size() * sizeof(Mxfp4Pair));
    CopyToDevice(w2p, w2.packed);
    DeviceBuffer w2s(w2.scales.size() * sizeof(E8M0));
    CopyToDevice(w2s, w2.scales);
    DeviceBuffer routeDev(route.size() * sizeof(std::int32_t));
    CopyToDevice(routeDev, route);

    // THREE-ZONE OUTPUT: [pre 4 KiB canary][output][post 4 KiB canary]. A
    // post-only guard cannot catch a store that lands BELOW the base -- a row
    // index computed one sub-tile low -- because nothing is poisoned there to
    // notice. 4 KiB per band is one whole phase 2 output sub-tile (16 rows x 64
    // floats), so a mis-strided burst cannot step over the band into live
    // memory with the guard intact.
    constexpr std::size_t kCanaryBandBytes = 4096;
    constexpr std::size_t kCanaryBandFloats = kCanaryBandBytes / sizeof(float);
    const std::size_t y_total_floats = y_floats + 2u * kCanaryBandFloats;
    DeviceBuffer yDev(y_total_floats * sizeof(float));
    {
      const std::vector<float> poison(y_total_floats, CanaryFloat());
      CopyToDevice(yDev, poison);
    }
    float* const y_device = static_cast<float*>(yDev.get()) + kCanaryBandFloats;
    std::printf("  output: [pre %zu B 0x5A][%zu floats][post %zu B 0x5A]\n",
                kCanaryBandBytes, y_floats, kCanaryBandBytes);

    // THE WORKSPACE IS EXPECTED TO STAY UNTOUCHED, and that is a real assertion
    // about this kernel rather than boilerplate: the bare-metal path removed
    // REGIST_MATMUL_OBJ, so there is no KFC message queue any more and the entry
    // point reads shared_workspace only to null-check it. The declared reserve
    // is zeroed and everything past it is poisoned; a byte of either moving
    // means the schedule is writing GM it was not given.
    const std::uint64_t declared =
        Dsv4MoePrefillCubeWorkspaceBytes(kBoundaryTokens, hidden, inter);
    constexpr std::uint64_t kProbeBytes = 8ull << 20;
    DeviceBuffer wsDev(static_cast<std::size_t>(declared + kProbeBytes));
    {
      std::vector<std::uint8_t> init(
          static_cast<std::size_t>(declared + kProbeBytes), 0x5Au);
      std::fill(init.begin(), init.begin() + static_cast<std::ptrdiff_t>(declared),
                0u);
      CopyToDevice(wsDev, init);
    }
    std::printf("  workspace: %llu KiB declared reserve (zeroed), then %llu MiB "
                "poisoned 0x5A\n",
                static_cast<unsigned long long>(declared >> 10),
                static_cast<unsigned long long>(kProbeBytes >> 20));

    // ----------------------------------------------------------------- golden
    // Computed and audited BEFORE the launch, so a device result is never the
    // thing that decides what the expected answer was.
    std::vector<float> want(y_floats, CanaryFloat());
    {
      using moe::emu::CubeTileGeometry;
      CubeTileGeometry geom;
      geom.m_tile = static_cast<int>(kPrefillCubeMTile);
      geom.r_tile = static_cast<int>(kPrefillCubeRSub);
      geom.k_tile = static_cast<int>(kPrefillCubeKSub);
      geom.h_tile = static_cast<int>(kPrefillCubeHSub);
      geom.r_tile_down = static_cast<int>(kPrefillCubeRSub2);
      geom.buffer_depth = 2;

      std::vector<moe::emu::Mxfp4Matrix> v1, v3, v2;
      for (std::uint32_t e = 0; e < kExperts; ++e) {
        v1.push_back(w1.View(e));
        v3.push_back(w3.View(e));
        v2.push_back(w2.View(e));
      }
      std::vector<float> scratch;
      moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
          v1.data(), v3.data(), v2.data(), x.data(), route.data(),
          static_cast<int>(kBoundaryTokens), static_cast<int>(kExperts),
          want.data(), kLimit, geom, nullptr, &scratch);

      // The GOLDEN is tested here, not the comparison: a nonzero norm AND real
      // variance (CLAUDE.md section 4.8). Comparing a value with itself would
      // assert nothing.
      if (!NonDegenerate(want)) {
        Fail("the golden for the boundary tile is degenerate: a zero norm or a "
             "constant output would make the comparison below vacuous");
      }
      // "still canary" and "correct" must be DISJOINT or the occupancy census
      // cannot tell an unwritten word from a right one.
      for (const float v : want) {
        if (IsCanary(v)) {
          Fail("a golden element collides with the 0x5A5A5A5A canary; the "
               "occupancy census would be unsound");
        }
      }
      const Spread s = Measure(want);
      std::printf(
          "  golden: rms %.4f stddev %.4f min %.4f max %.4f (%zu floats)\n",
          s.rms, s.stddev, s.min, s.max, want.size());
    }

    // ----------------------------------------------------------------- launch
    // THE PRODUCTION WRAPPER, CALLED AS A CALLER WOULD. It queries the core
    // count itself; block_dim comes out min(total_tiles, aiv) = 1 whatever the
    // simulated part reports, which is why this gate needs no sim-only launch
    // wrapper to pin the grid.
    watchdog.Arm("boundary " DSV4_BOUNDARY_LAUNCH_NAME);
    const auto started = std::chrono::steady_clock::now();
    const Status status = DSV4_BOUNDARY_LAUNCH(
        static_cast<const float*>(xDev.get()),
        static_cast<const Mxfp4Pair*>(w1p.get()),
        static_cast<const E8M0*>(w1s.get()),
        static_cast<const Mxfp4Pair*>(w3p.get()),
        static_cast<const E8M0*>(w3s.get()),
        static_cast<const Mxfp4Pair*>(w2p.get()),
        static_cast<const E8M0*>(w2s.get()),
        static_cast<const std::int32_t*>(routeDev.get()), hidden, inter,
        kExperts, kBoundaryTokens, y_device, wsDev.get(), runtime.stream());
    const auto finished = std::chrono::steady_clock::now();
    watchdog.Disarm();
    const double seconds =
        std::chrono::duration<double>(finished - started).count();

    std::printf("  elapsed %.1f s for %u contraction steps (%u phase 1 + %u "
                "phase 2)\n",
                seconds, sched.steps, sched.fwd_blocks * sched.fwd_steps,
                sched.down_blocks * sched.down_steps);
    if (status != Status::success) {
      Fail(LaunchFailure(DSV4_BOUNDARY_LAUNCH_NAME, status));
    }

    // ------------------------------------------------------------- inspection
    // The WHOLE allocation in one copy, so both bands and the payload come from
    // the same snapshot and a late store cannot slip between two reads.
    std::vector<float> y_all(y_total_floats, 0.0F);
    CopyFromDevice(y_all, yDev);

    for (std::size_t i = 0; i < kCanaryBandFloats; ++i) {
      if (!IsCanary(y_all[i])) {
        Fail("PRE-canary band corrupted at float -" +
             std::to_string(kCanaryBandFloats - i) +
             " below the y base: the kernel stored BELOW its output. Check the "
             "phase 2 y row index (token_row_start + row_base) and the GM base "
             "passed to tokens_out.");
      }
    }
    for (std::size_t i = 0; i < kCanaryBandFloats; ++i) {
      const std::size_t at = kCanaryBandFloats + y_floats + i;
      if (!IsCanary(y_all[at])) {
        Fail("POST-canary band corrupted at float +" + std::to_string(i) +
             " past the " + std::to_string(y_floats) +
             "-float output: the kernel stored past its output. Check the phase "
             "2 output burst's dstStride.");
      }
    }
    std::printf("  canary bands: both intact (%zu floats each side)\n",
                kCanaryBandFloats);

    const std::vector<float> y(
        y_all.begin() + static_cast<std::ptrdiff_t>(kCanaryBandFloats),
        y_all.begin() +
            static_cast<std::ptrdiff_t>(kCanaryBandFloats + y_floats));

    const Occupancy occ = MeasureOccupancy(y);
    std::printf(
        "  occupancy: %zu live, %zu exactly zero, %zu STILL CANARY of %zu\n",
        occ.live, occ.zero, occ.canary, y.size());
    if (occ.canary != 0) {
      Fail(std::to_string(occ.canary) + " of " + std::to_string(y.size()) +
           " output words were never written (still 0x5A5A5A5A): the kernel did "
           "not cover its own tile. A TilingRejected() early return is the first "
           "thing to rule out -- on device it is a silent no-op.");
    }

    {
      std::vector<std::uint8_t> probe(static_cast<std::size_t>(kProbeBytes), 0u);
      RequireAcl(
          aclrtMemcpy(probe.data(), probe.size(),
                      static_cast<const std::uint8_t*>(wsDev.get()) + declared,
                      probe.size(), ACL_MEMCPY_DEVICE_TO_HOST),
          "aclrtMemcpy(workspace probe past the declared reserve)");
      const std::string ws = CheckWorkspaceUnwritten(probe);
      if (!ws.empty()) Fail(ws);
      std::printf(
          "  workspace: all %llu MiB past the declared reserve still 0x5A\n",
          static_cast<unsigned long long>(kProbeBytes >> 20));
    }

    for (std::size_t i = 0; i < 4; ++i) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &y[i], sizeof(bits));
      std::printf("  output[%zu]: actual %.9g (0x%08x), want %.9g\n", i, y[i],
                  bits, want[i]);
    }

    const BoundaryDiff diff = CompareTile(y, want);
    std::printf("  WRONG ELEMENTS: %zu of %zu\n", diff.wrong, y.size());
    std::printf("  diff: max abs %.6e at [%zu] (row %zu col %zu), max rel "
                "%.6e, cosine %.8f\n",
                diff.max_abs, diff.worst, diff.worst / hidden,
                diff.worst % hidden, diff.max_rel, diff.cosine);
    if (!diff.finite) Fail("the device produced a non-finite value");
    if (diff.wrong != 0) {
      Fail(std::to_string(diff.wrong) + " of " + std::to_string(y.size()) +
           " elements are outside the " + std::to_string(kBoundaryAbsTol) +
           " + " + std::to_string(kBoundaryRelTol) + "*|want| envelope (max abs " +
           std::to_string(diff.max_abs) + ", max rel " +
           std::to_string(diff.max_rel) + ")");
    }
    if (diff.cosine < kBoundaryCosFloor) {
      Fail("cosine " + std::to_string(diff.cosine) + " is below the " +
           std::to_string(kBoundaryCosFloor) +
           " floor: a direction error of this size is the signature of a "
           "structural defect -- the NZ(N,K) == ZN(K,N) operand "
           "reinterpretation in CubeLoadWeightsToL0B is the first suspect");
    }

    runtime.Close();

    const DumpCensus after = ExceptionDumps();
    std::printf("  excp_log*.dump: %zu before, %zu after (%zu non-empty)\n",
                before.files, after.files, after.non_empty);
    if (after.non_empty > before.non_empty) {
      Fail("the run left new non-empty excp_log*.dump file(s)");
    }

    std::printf(
        "\n[PASS] %s executes on the "
        "dav-3510 CAModel in MIX_AIC_1_2 at block_dim 1 over a %u-step phase 1 "
        "contraction (head k in [0,%u], tail k in [%u,%u]) and a %u-block phase "
        "2, and its %u x %u output matches the audited two-phase Cube golden "
        "with 0 wrong elements inside %g + %g*|want| and cosine %.8f >= %g. "
        "Wall %.1f s. Compiles, links and now EXECUTES ON THE SIMULATOR; never "
        "run on hardware.\n",
        DSV4_BOUNDARY_ENTRY_NAME,
        sched.fwd_steps, sched.head_last, sched.tail_first,
        sched.fwd_steps - 1u, sched.down_blocks, kBoundaryTokens, hidden,
        static_cast<double>(kBoundaryAbsTol),
        static_cast<double>(kBoundaryRelTol), diff.cosine, kBoundaryCosFloor,
        seconds);
    return 0;
  } catch (const std::exception& error) {
    std::printf("\n[FAIL] %s\n", error.what());
    return 1;
  }
}

}  // namespace

// argv: [hidden] [inter] [budget_seconds]. The defaults ARE the registered
// boundary shape; the overrides exist so the same binary can measure its own
// fixed cost at a cheaper shape, which is the only honest way to size a budget
// for the next one -- the CAModel cost model is measured, never derived.
int main(int argc, char** argv) {
  const std::uint32_t hidden =
      ParseExtent(argc > 1 ? argv[1] : nullptr, kBoundaryHidden);
  const std::uint32_t inter =
      ParseExtent(argc > 2 ? argv[2] : nullptr, kBoundaryInter);
  const int budget = static_cast<int>(
      ParseExtent(argc > 3 ? argv[3] : nullptr,
                  static_cast<std::uint32_t>(kBoundaryBudgetSeconds)));
  return RunBoundary(hidden, inter, budget);
}
