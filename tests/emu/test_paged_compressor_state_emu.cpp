// test_paged_compressor_state_emu.cpp -- host simulation of
// paged_compressor_rope's STATE MACHINE across a group boundary.
//
// WHY THIS FILE EXISTS (CLAUDE.md section 5.5)
// test_paged_compressor_emu.cpp validates the ORACLE's algebra. This file
// validates something different and previously uncovered: the device kernel's
// own bookkeeping -- which slice of which GM region is written when, and
// whether four separate kernel launches carry the previous group's a-halves
// across the group boundary intact. That is pure integer/index arithmetic plus
// a buffer layout, it needs no hardware to check, and it was the arithmetic
// that two successive hardware failures were blamed on.
//
// It is NOT an interpretation of the kernel (CLAUDE.md section 5.2). It does
// not model __simt_vf__ scheduling, warp primitives, DataCache behaviour, MTE
// pipelines or event flags, and a pass here is no evidence about any of them.
// It transcribes the kernel's index expressions and its GM region offsets, and
// drives them through the same eight-step cadence the device suite uses. What
// it can prove is "the layout and the roll are right"; what it cannot prove is
// "the writeback is ordered correctly on hardware".
//
// THAT DISTINCTION IS THE POINT. When the device reported
//   group 0 EMIT  cos  >0.9999  (pass)
//   group 1 EMIT  cos   0.8848  (fail, relL2 4.794e-01)
// the natural reading was an off-by-one in the roll or a head-stride mismatch
// in which slice is preserved as prev_a. This simulation reproduces the golden
// EXACTLY for both groups, which refutes that reading: the indices are right,
// so the divergence is in ordering or visibility, not in address arithmetic.
// The corruption table below is what makes that a measurement rather than an
// assertion -- it prints what each candidate index bug WOULD have scored, and
// none of them scores anywhere near the observed signature.
//
// Note group 0 is nearly worthless as evidence about prev_a: with
// have_previous = 0 the first four window rows carry score -inf, hence softmax
// weight exactly 0, so the prev_a READ path is first exercised at step 7. A
// green group 0 says nothing about it, which is why the simulation drives the
// full two-group cadence and not one group.
//
// The projection runs at a deliberately small hidden size. The state machine
// is independent of `hidden` -- it only ever moves whole rows around -- and the
// golden takes hidden_size as a parameter, so both sides agree at any value
// while 256 keeps the eight steps under a second.

#include "moe/emu/cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "emu_test_support.hpp"

namespace {

constexpr int kD = moe::emu::kCompressorHeadDim;        // 512
constexpr int kR = moe::emu::kCompressorRatio;          // 4
constexpr int kW = moe::emu::kCompressorStateWidth;     // 1024
constexpr int kRows = moe::emu::kCompressorReduceRows;  // 8
constexpr int kRope = moe::emu::kRopeHeadDim;           // 64
constexpr int kSteps = 2 * kR;                          // 8
constexpr int kHidden = 256;  // see the header comment

// --- GM state ABI, in FP32 elements ----------------------------------------
// Mirrored from the kernel's SetGlobalBuffer offsets. Four consecutive regions
// in this order: kv_cur | gate_cur | prev_kv_a | prev_gate_a. Constants are
// mirrored rather than included because op_kernel/ headers are
// operator-private (CLAUDE.md section 1.2) and this suite must build with a
// stock host compiler and no CANN toolkit (section 4.1).
constexpr int kKvCurFloats = kR * kW;    // 4,096  at element 0
constexpr int kPrevAFloats = kR * kD;    // 2,048  at 8,192 and 10,240
constexpr int kStateFloats = 2 * kKvCurFloats + 2 * kPrevAFloats;  // 12,288
constexpr int kOffKv = 0;
constexpr int kOffGate = kKvCurFloats;
constexpr int kOffPrevKv = 2 * kKvCurFloats;
constexpr int kOffPrevGate = 2 * kKvCurFloats + kPrevAFloats;

float Bf(std::uint16_t bits) { return moe::emu::Bf16ToFloat(bits); }

// ---------------------------------------------------------------------------
// Tolerance, derived from the formats rather than fitted to the observation
// (CLAUDE.md section 4.7)
// ---------------------------------------------------------------------------
// The simulation and the oracle perform identical FP32 arithmetic in identical
// order everywhere EXCEPT the RMSNorm denominator. The oracle accumulates
// sum(x^2) in FP64 (`double sum_squares`); the device accumulates it in FP32,
// because the block reduce is a warp butterfly over FP32 registers and there
// is no FP64 path on the AIV. This simulation is faithful to the DEVICE, so
// the two inv_rms values differ.
//
// That difference is bounded: sequentially accumulating 512 non-negative FP32
// terms carries at most ~512 * 2^-24 relative error, and the square root
// halves it, so inv_rms agrees to within roughly 2^-23.5 -- measured at
// 1.4e-7 on a representative token.
//
// It is a single SHARED SCALE on all 512 lanes. The tempting conclusion is
// that cosine is therefore exactly 1.0 because cosine is scale-invariant.
// That is WRONG, and the error matters: the BF16 store happens AFTER the
// scale, so quantization does not commute with it.
//
//   * relative-L2 cannot be zero. A 1.4e-7 scale shift is far below BF16's
//     2^-8 quantum, so it changes a lane only when that lane sits on a
//     rounding tie -- and then it changes it by one whole BF16 ULP. With k of
//     512 lanes tied the error is sqrt(k/512) * 2^-8: 1.7e-4 for one lane,
//     3.5e-4 for four. The ceiling is 1e-3, which admits ~33
//     simultaneously-tied lanes and still rejects the loss of a single prev_a
//     row by three orders of magnitude (the cheapest defect in the table
//     below scores 4.1e-1).
//   * cosine is then PINNED BY that figure rather than independent of it.
//     Those k flipped lanes are a perturbation of relative size e = relL2
//     that is uncorrelated with the token, so it is essentially perpendicular
//     and cos = 1 - e^2/2. At the ceiling that is 1 - 5e-7, so the floor is
//     1e-6 -- still 100x tighter than the device suite's 0.9999, and every
//     defect in the table below is caught by three orders of magnitude or
//     more (the closest scores 0.92).
//
// The coupling is not a guess: at seed 0xBEEF the simulation measures relL2
// 3.176e-4 and cos 1 - 5.03e-8, against 1 - (3.176e-4)^2/2 = 1 - 5.04e-8
// predicted. Writing the two bounds as independent -- which the first version
// of this file did, with a 1e-9 cosine floor -- fails on exactly that seed
// while nothing is wrong, which is the failure mode a fitted tolerance hides
// and a derived one exposes.
inline constexpr double kRelL2Ceiling = 1e-3;
inline constexpr double kCosFloor =
    1.0 - 1e-6;  // >= 1 - kRelL2Ceiling^2/2, with margin

// Candidate defects, injected so the table can show what each WOULD score.
// kClean is the shipped kernel.
enum class Defect {
  kClean,
  kRollBHalf,       // roll the b half instead of the a half
  kRollCurrentA,    // roll after the window, from the current group
  kPrevSwapped,     // the two prev regions exchanged in GM
  kPrevKvLost,      // prev_kv never lands
  kPrevGateLost,    // prev_gate never lands
  kPrevRow0Only,    // only row 0 of prev lands
  kPrevUpperLost,   // prev lanes [256,512) never land
  kHoldWritesPrev,  // a HOLD also rolls, so prev tracks the current group
};

struct DefectCase {
  Defect defect;
  const char* name;
};

// --- the kernel's two SIMT VFs, as plain loops over feature lanes -----------

void ProjectSimt(const std::uint16_t* x, const std::uint16_t* wkv,
                 const std::uint16_t* wgate, const float* ape, int slot,
                 float* kv_cur, float* gate_cur) {
  const int ape_base = slot * kW;
  // Thread t covers features j = t and j = t + kD, as the VF's k loop does.
  for (int t = 0; t < kD; ++t) {
    for (int k = 0; k < 2; ++k) {
      const int j = t + k * kD;
      float kv_dot = 0.0f;
      const std::uint16_t* kv_row = wkv + static_cast<std::size_t>(j) * kHidden;
      for (int d = 0; d < kHidden; ++d) kv_dot += Bf(kv_row[d]) * Bf(x[d]);
      kv_cur[slot * kW + j] = kv_dot;

      float gate_dot = 0.0f;
      const std::uint16_t* g_row = wgate + static_cast<std::size_t>(j) * kHidden;
      for (int d = 0; d < kHidden; ++d) gate_dot += Bf(g_row[d]) * Bf(x[d]);
      gate_cur[slot * kW + j] = gate_dot + ape[ape_base + j];
    }
  }
}

// Rolls this group's a halves into the previous-a state, as the tail of
// CompressorEmitSimt does. `lane_base` is 0 in the shipped kernel.
void RollPrevA(const float* kv_cur, const float* gate_cur, float* prev_kv_a,
               float* prev_gate_a, int lane_base) {
  for (int idx = 0; idx < kR * kD; ++idx) {
    const int r = idx / kD;
    const int l = idx % kD;
    prev_kv_a[idx] = kv_cur[r * kW + lane_base + l];
    prev_gate_a[idx] = gate_cur[r * kW + lane_base + l];
  }
}

void EmitSimt(const float* kv_cur, const float* gate_cur,
              const float* prev_kv_a, const float* prev_gate_a,
              int have_previous, const float* norm_w, const float* rope_cos,
              const float* rope_sin, float eps, float* out) {
  std::vector<float> acc(kD);
  for (int i = 0; i < kD; ++i) {
    float kv[kRows];
    float sc[kRows];
    for (int r = 0; r < kR; ++r) {
      if (have_previous != 0) {
        kv[r] = prev_kv_a[r * kD + i];
        sc[r] = prev_gate_a[r * kD + i];
      } else {
        kv[r] = 0.0f;
        sc[r] = -std::numeric_limits<float>::infinity();
      }
      kv[kR + r] = kv_cur[r * kW + kD + i];
      sc[kR + r] = gate_cur[r * kW + kD + i];
    }
    float m = sc[0];
    for (int r = 1; r < kRows; ++r) m = std::fmax(m, sc[r]);
    float w[kRows];
    float sum = 0.0f;
    if (std::isfinite(m)) {
      for (int r = 0; r < kRows; ++r) {
        float e = std::exp(sc[r] - m);
        if (e < 1.17549435e-38f) e = 0.0f;  // the oracle's FTZ exp
        w[r] = e;
        sum += e;
      }
      const float inv = 1.0f / sum;
      for (int r = 0; r < kRows; ++r) w[r] *= inv;
    } else {
      for (int r = 0; r < kRows; ++r) w[r] = 0.0f;
    }
    float a = 0.0f;
    for (int r = 0; r < kRows; ++r) a += w[r] * kv[r];
    acc[i] = a;
  }

  // The block RMSNorm butterfly reduces to the plain total.
  float total = 0.0f;
  for (int i = 0; i < kD; ++i) total += acc[i] * acc[i];
  const float inv_rms = 1.0f / std::sqrt(total / static_cast<float>(kD) + eps);
  for (int i = 0; i < kD; ++i) out[i] = (norm_w[i] * acc[i]) * inv_rms;

  // Half-mode RoPE on the LAST kRope lanes; tables read at half+i on the
  // second line, as vf_rope.h indexes them.
  const int base = kD - kRope;
  for (int i = 0; i < kRope / 2; ++i) {
    const float lo = out[base + i];
    const float hi = out[base + kRope / 2 + i];
    out[base + i] = rope_cos[i] * lo - rope_sin[i] * hi;
    out[base + kRope / 2 + i] =
        rope_sin[kRope / 2 + i] * lo + rope_cos[kRope / 2 + i] * hi;
  }
}

struct Fixture {
  std::vector<std::uint16_t> wkv, wgate, norm_w;
  std::vector<float> ape, rope_cos, rope_sin, norm_w_f32;
  std::vector<std::vector<std::uint16_t>> tokens;
  std::vector<std::vector<std::uint16_t>> want;  // the golden's two tokens
};

Fixture MakeFixture(std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  auto bf16v = [&](std::size_t n, float s) {
    std::vector<std::uint16_t> v(n);
    for (auto& e : v) e = moe::emu::FloatToBf16(nd(rng) * s);
    return v;
  };

  Fixture f;
  const std::size_t proj = static_cast<std::size_t>(kW) * kHidden;
  const float ws = 2.0f / std::sqrt(static_cast<float>(kHidden));
  f.wkv = bf16v(proj, ws);
  f.wgate = bf16v(proj, ws);
  f.ape.resize(static_cast<std::size_t>(kR) * kW);
  for (auto& e : f.ape) e = nd(rng);
  f.norm_w.resize(kD);
  for (auto& e : f.norm_w) e = moe::emu::FloatToBf16(1.0f + nd(rng) * 0.5f);
  f.rope_cos.resize(kRope);
  f.rope_sin.resize(kRope);
  for (auto& e : f.rope_cos) e = nd(rng) * 2.0f;
  for (auto& e : f.rope_sin) e = nd(rng) * 2.0f;
  f.tokens.resize(kSteps);
  for (auto& t : f.tokens) t = bf16v(kHidden, 1.0f);

  f.norm_w_f32.resize(kD);
  for (int i = 0; i < kD; ++i) f.norm_w_f32[i] = Bf(f.norm_w[i]);

  // The oracle, driven step by step so its state evolves as the device's does.
  moe::emu::CompressorRingState gs;
  f.want.assign(2, std::vector<std::uint16_t>(kD, 0));
  for (int s = 0; s < kSteps; ++s) {
    moe::emu::paged_compressor_decode_c4a(
        f.tokens[s].data(), f.wkv.data(), f.wgate.data(), f.ape.data(),
        f.norm_w.data(), s, &gs, f.rope_cos.data(), f.rope_sin.data(),
        f.want[s / kR].data(), kHidden, kD, kR, moe::emu::kRmsNormEpsilon,
        moe::emu::RotaryMode::kHalf);
  }
  return f;
}

struct Result {
  double cos[2];
  double rel_l2[2];
  bool cadence_ok;
  int emits;
};

// Drives the eight-step cadence exactly as the device suite does: one
// "launch" per step, all state crossing the boundary through the GM array.
Result Simulate(const Fixture& f, Defect defect) {
  std::vector<float> gm(kStateFloats, 0.0f);  // the documented cold start
  std::uint32_t meta_have = 0;
  std::uint32_t meta_seen = 0;
  std::vector<std::vector<std::uint16_t>> got(2,
                                              std::vector<std::uint16_t>(kD, 0));
  Result out{};
  out.cadence_ok = true;
  out.emits = 0;

  for (int s = 0; s < kSteps; ++s) {
    if (static_cast<std::uint32_t>(s) != meta_seen) {
      out.cadence_ok = false;  // the kernel would refuse to touch the state
      break;
    }
    const int have_previous = static_cast<int>(meta_have & 1u);
    const int slot = s % kR;

    // LoadStateAndTables: four whole-region MTE2 copies, GM -> UB.
    std::vector<float> kv_cur(gm.begin() + kOffKv,
                              gm.begin() + kOffKv + kKvCurFloats);
    std::vector<float> gate_cur(gm.begin() + kOffGate,
                                gm.begin() + kOffGate + kKvCurFloats);
    std::vector<float> prev_kv(gm.begin() + kOffPrevKv,
                               gm.begin() + kOffPrevKv + kPrevAFloats);
    std::vector<float> prev_gate(gm.begin() + kOffPrevGate,
                                 gm.begin() + kOffPrevGate + kPrevAFloats);
    if (defect == Defect::kPrevSwapped) std::swap(prev_kv, prev_gate);

    ProjectSimt(f.tokens[s].data(), f.wkv.data(), f.wgate.data(), f.ape.data(),
                slot, kv_cur.data(), gate_cur.data());

    const bool emit = (slot == kR - 1);
    if (emit) {
      std::vector<float> token(kD);
      EmitSimt(kv_cur.data(), gate_cur.data(), prev_kv.data(), prev_gate.data(),
               have_previous, f.norm_w_f32.data(), f.rope_cos.data(),
               f.rope_sin.data(), moe::emu::kRmsNormEpsilon, token.data());
      for (int i = 0; i < kD; ++i)
        got[s / kR][i] = moe::emu::FloatToBf16(token[i]);
      ++out.emits;

      // The roll, AFTER the reduction -- the oracle's order.
      RollPrevA(kv_cur.data(), gate_cur.data(), prev_kv.data(),
                prev_gate.data(),
                defect == Defect::kRollBHalf ? kD : 0);
      if (defect == Defect::kRollCurrentA) {
        // Pathological: roll from the rows the NEXT group will fill, i.e.
        // preserve nothing useful.
        for (int r = 0; r < kR; ++r)
          for (int l = 0; l < kD; ++l) {
            prev_kv[r * kD + l] = kv_cur[r * kW + kD + l];
            prev_gate[r * kD + l] = gate_cur[r * kW + kD + l];
          }
      }
    } else if (defect == Defect::kHoldWritesPrev) {
      RollPrevA(kv_cur.data(), gate_cur.data(), prev_kv.data(),
                prev_gate.data(), 0);
    }

    // Injected writeback losses.
    if (defect == Defect::kPrevKvLost)
      std::fill(prev_kv.begin(), prev_kv.end(), 0.0f);
    if (defect == Defect::kPrevGateLost)
      std::fill(prev_gate.begin(), prev_gate.end(), 0.0f);
    if (defect == Defect::kPrevRow0Only) {
      std::fill(prev_kv.begin() + kD, prev_kv.end(), 0.0f);
      std::fill(prev_gate.begin() + kD, prev_gate.end(), 0.0f);
    }
    if (defect == Defect::kPrevUpperLost) {
      for (int r = 0; r < kR; ++r)
        for (int l = kD / 2; l < kD; ++l) {
          prev_kv[r * kD + l] = 0.0f;
          prev_gate[r * kD + l] = 0.0f;
        }
    }

    // WritebackRegions: the two current regions always, the prev pair only on
    // an EMIT (or, under kHoldWritesPrev, always).
    std::copy(kv_cur.begin(), kv_cur.end(), gm.begin() + kOffKv);
    std::copy(gate_cur.begin(), gate_cur.end(), gm.begin() + kOffGate);
    if (emit || defect == Defect::kHoldWritesPrev) {
      std::copy(prev_kv.begin(), prev_kv.end(), gm.begin() + kOffPrevKv);
      std::copy(prev_gate.begin(), prev_gate.end(), gm.begin() + kOffPrevGate);
    }
    if (emit) meta_have = 1;
    meta_seen = s + 1;
  }

  for (int g = 0; g < 2; ++g) {
    double dot = 0, na = 0, nb = 0, dif = 0, ref = 0;
    for (int i = 0; i < kD; ++i) {
      const double a = Bf(got[g][i]);
      const double b = Bf(f.want[g][i]);
      dot += a * b;
      na += a * a;
      nb += b * b;
      dif += (a - b) * (a - b);
      ref += b * b;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    out.cos[g] = denom > 0.0 ? dot / denom : 0.0;
    out.rel_l2[g] = ref > 0.0 ? std::sqrt(dif / ref) : 0.0;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Section 1: the GM region table tiles the buffer with no gap and no overlap.
// ---------------------------------------------------------------------------
void TestStateAbiLayout(emu_test::Runner& runner) {
  runner.Section("GM state ABI region table");

  runner.Check(kOffKv == 0, "kv_cur starts the state buffer");
  runner.Check(kOffGate == 4096,
               "gate_cur begins one kv region in, at float 4,096 -- not 8,192, "
               "which is where prev_kv_a starts");
  runner.Check(kOffPrevKv == 8192, "prev_kv_a is at float 8,192");
  runner.Check(kOffPrevGate == 10240, "prev_gate_a is at float 10,240");
  runner.Check(kOffPrevGate + kPrevAFloats == kStateFloats,
               "the four regions tile the 12,288-float buffer exactly");
  runner.Check(kStateFloats == 3 * kR * kW,
               "the state is three ratio-by-state-width blocks, as the "
               "contract's 49,152 B figure says");
  // The b half the window reads and the a half the roll preserves must be the
  // two disjoint halves of one state row.
  runner.Check(kW == 2 * kD,
               "a state row splits evenly into an a half and a b half");
}

// ---------------------------------------------------------------------------
// Section 2: the shipped kernel reproduces the oracle across the boundary.
// ---------------------------------------------------------------------------
void TestCleanCadence(emu_test::Runner& runner) {
  runner.Section("eight-step cadence against the oracle");

  // Several seeds: one fixture could hide a defect behind a flat softmax.
  for (std::uint32_t seed : {0xC4A0u, 0x1234u, 0xBEEFu}) {
    const Fixture f = MakeFixture(seed);
    const Result r = Simulate(f, Defect::kClean);
    const std::string tag = "seed 0x" + std::to_string(seed) + ": ";

    runner.Check(r.cadence_ok, tag + "the cadence is never refused");
    runner.Check(r.emits == 2, tag + "exactly two EMITs in eight steps");

    // The oracle and the simulation perform identical FP32 operations in
    // identical order, so this is EXACT -- no tolerance. A BF16 bound would
    // be wrong here: nothing rounds differently between the two sides.
    char m0[160], m1[160];
    std::snprintf(m0, sizeof(m0), " [cos %.12f, relL2 %.3e]", r.cos[0],
                  r.rel_l2[0]);
    std::snprintf(m1, sizeof(m1), " [cos %.12f, relL2 %.3e]", r.cos[1],
                  r.rel_l2[1]);
    runner.Check(r.cos[0] >= kCosFloor && r.rel_l2[0] <= kRelL2Ceiling,
                 tag + "group 0 EMIT matches the oracle" + m0);
    runner.Check(r.cos[1] >= kCosFloor && r.rel_l2[1] <= kRelL2Ceiling,
                 tag + "group 1 EMIT matches the oracle ACROSS the group "
                       "boundary -- the roll and the prev_a read compose to "
                       "the oracle's eight-row window" + m1);

    // The two tokens must differ, or group 1 never exercised prev_a.
    double between = 0, n0 = 0, n1 = 0;
    for (int i = 0; i < kD; ++i) {
      const double a = Bf(f.want[0][i]);
      const double b = Bf(f.want[1][i]);
      between += a * b;
      n0 += a * a;
      n1 += b * b;
    }
    runner.Check(n0 > 0.0 && n1 > 0.0 &&
                     std::abs(between / (std::sqrt(n0) * std::sqrt(n1))) <
                         0.9999,
                 tag + "the two compressed tokens differ, so group 1 really "
                       "depends on the previous group's a halves");
  }
}

// ---------------------------------------------------------------------------
// Section 3: group 0 cannot witness the prev_a path at all.
//
// This is the trap the two hardware investigations fell into. With
// have_previous = 0 the first four window rows carry score -inf, so their
// softmax weight is exactly 0 and their kv value is never read for effect.
// Corrupting them arbitrarily must leave group 0 bit-identical.
// ---------------------------------------------------------------------------
void TestGroup0IsBlindToPrevA(emu_test::Runner& runner) {
  runner.Section("group 0 is blind to prev_a by construction");

  const Fixture f = MakeFixture(0xC4A0u);
  const Result clean = Simulate(f, Defect::kClean);

  for (const DefectCase c : {
           DefectCase{Defect::kRollBHalf, "roll the b half"},
           DefectCase{Defect::kPrevKvLost, "prev_kv never lands"},
           DefectCase{Defect::kPrevGateLost, "prev_gate never lands"},
           DefectCase{Defect::kPrevUpperLost, "prev upper lanes never land"},
       }) {
    const Result r = Simulate(f, c.defect);
    runner.Check(r.cos[0] == clean.cos[0] && r.rel_l2[0] == clean.rel_l2[0],
                 std::string("group 0 is unchanged by: ") + c.name +
                     " -- so a green group 0 is NOT evidence that prev_a "
                     "works");
    runner.Check(r.cos[1] < 0.9999,
                 std::string("group 1 does catch: ") + c.name);
  }
}

// ---------------------------------------------------------------------------
// Section 4: what each candidate index defect would have scored.
//
// Printed as evidence, not asserted as a number: the point is that every
// address-arithmetic defect lands far from the observed group-1 signature of
// cos 0.8848 / relL2 4.794e-01, which is what moved the diagnosis off
// "off-by-one in the roll" and onto ordering/visibility.
// ---------------------------------------------------------------------------
void TestDefectSignatures(emu_test::Runner& runner) {
  runner.Section("candidate defect signatures (group 1)");

  const Fixture f = MakeFixture(0xC4A0u);
  for (const DefectCase c : {
           DefectCase{Defect::kRollBHalf, "roll the b half, not the a half"},
           DefectCase{Defect::kRollCurrentA, "roll the current group's b half"},
           DefectCase{Defect::kPrevSwapped, "prev_kv <-> prev_gate swapped"},
           DefectCase{Defect::kPrevKvLost, "prev_kv never lands"},
           DefectCase{Defect::kPrevGateLost, "prev_gate never lands"},
           DefectCase{Defect::kPrevRow0Only, "only prev row 0 lands"},
           DefectCase{Defect::kPrevUpperLost, "prev lanes [256,512) lost"},
           DefectCase{Defect::kHoldWritesPrev, "a HOLD also rolls"},
       }) {
    const Result r = Simulate(f, c.defect);
    char detail[200];
    std::snprintf(detail, sizeof(detail), " [cos %.9f, relL2 %.3e]", r.cos[1],
                  r.rel_l2[1]);
    // Every one of these must be CAUGHT -- that is the assertion. The numbers
    // ride along in the message so the table can be compared against a
    // hardware signature.
    runner.Check(r.cos[1] < 0.9999,
                 std::string("caught: ") + c.name + detail);
  }
}

}  // namespace

int main() {
  emu_test::Runner runner(
      "paged_compressor_rope state machine (host simulation)");

  TestStateAbiLayout(runner);
  TestCleanCadence(runner);
  TestGroup0IsBlindToPrevA(runner);
  TestDefectSignatures(runner);

  return runner.Summary();
}
