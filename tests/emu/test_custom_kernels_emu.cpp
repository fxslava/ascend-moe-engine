// -----------------------------------------------------------------------------
// test_custom_kernels_cpu_ref -- Target A, the pure CPU oracle.
//
// Properties of the arch35 kernel contracts that are decidable WITHOUT any
// hardware: the algebraic invariants of the CPU golden in src/emu, the Birkhoff
// polytope bounds, the top-k selection and tie-breaking rules, the E2M1 / E8M0
// decode, and the metric machinery the device suite's thresholds rest on.
//
// Every reference value is generated in memory on this run from a printed
// seed. No disk dumps, no pre-baked fixtures, so a stale asset cannot make a
// broken reference pass.
//
// ZERO DEVICE DEPENDENCY, deliberately and checkably:
//   * no <acl/acl.h>, no device headers,
//   * does not link libascendcl.so,
//   * does not link any host CPU debug/interpretation runtime (no cpudebug,
//     no tikicpulib, no libascendc_cpudebug.so),
//   * builds with a stock host g++/clang++ and no CANN toolkit present.
//
// It therefore must pass on any machine, NPU or not, and it is this binary --
// not the device one -- that gates CI. On-device comparison against these same
// references lives in test_npu_device.cpp (Target B).
// -----------------------------------------------------------------------------

#include "kernel_test_harness.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"

#include <algorithm>
#include <array>
#include <functional>
#include <set>

namespace {

using namespace moe::kerntest;
using moe::kerntest::Report;

// Precision contract. Tightened deliberately: these compare two
// implementations of the SAME arithmetic, not an approximation of it.
constexpr double kCosineFloor = 0.99999;
constexpr double kRelL2Ceiling = 1e-4;
// Birkhoff: the solve is 20 iterations of Sinkhorn-Knopp, whose last operation
// is a COLUMN normalization. Columns therefore land on 1.0 to within float
// rounding regardless of the input; rows are one half-step behind and carry
// whatever the truncated iteration has not yet removed.
constexpr double kBirkhoffColTol = 1e-5;
// Row tolerance for well-conditioned inputs.
constexpr double kBirkhoffRowTol = 2e-2;
// Rank-deficient inputs (repeated rows) make Sinkhorn converge slowly: the
// duplicated rows compete for the same column mass, and 20 iterations leaves a
// visible row residual. That is the REFERENCE's behaviour, and the kernel is
// required to reproduce the truncated 20-iteration result, not the converged
// limit -- so this is a correctness bound on a degenerate case, not a license
// for sloppiness. Measured worst case for the fixtures below is ~3.7e-2.
constexpr double kBirkhoffRowTolDegenerate = 5e-2;

// ===========================================================================
// 1. fused_mhc_pre_norm_sinkhorn -- Birkhoff doubly-stochastic bounds
//
// Driven through sinkhorn_knopp_4x4 directly so the adversarial matrices can
// be chosen: near-zero entries, huge dynamic range, exact ties, and a row of
// equal values. No phi projection would produce these, which is exactly why
// they are worth asserting.
// ===========================================================================
void VerifySinkhornBirkhoff(Report& rep) {
  section("fused_mhc_pre_norm_sinkhorn: Birkhoff doubly-stochastic bounds");

  struct Case {
    const char* name;
    std::vector<float> logits;  // 4x4 row-major
    // Rows 1..3 identical => rank-deficient => slow convergence.
    bool rank_deficient = false;
  };

  const std::vector<Case> cases = {
      // All rows identical, but symmetric: converges immediately to uniform.
      {"uniform zeros (maximally degenerate)", std::vector<float>(16, 0.0F), false},
      {"all equal nonzero", std::vector<float>(16, 3.5F), false},
      {"huge dynamic range",
       {80.0F, -80.0F, 0.0F, 40.0F,
        -80.0F, 80.0F, 40.0F, 0.0F,
        0.0F, 40.0F, 80.0F, -80.0F,
        40.0F, 0.0F, -80.0F, 80.0F}, false},
      {"near-identity (strong diagonal)",
       {20.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 20.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 20.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 20.0F}, false},
      {"exact ties across a row",
       {1.0F, 1.0F, 1.0F, 1.0F,
        2.0F, -2.0F, 2.0F, -2.0F,
        0.5F, 0.5F, -0.5F, -0.5F,
        -3.0F, 3.0F, -3.0F, 3.0F}, false},
      // Row 0 claims column 0 almost entirely while rows 1..3 are mutually
      // identical and must share the remaining three columns. The slowest
      // case here, and the reason kBirkhoffRowTolDegenerate exists.
      {"single dominant entry (rank-deficient)",
       {80.0F, -80.0F, -80.0F, -80.0F,
        -80.0F, -80.0F, -80.0F, -80.0F,
        -80.0F, -80.0F, -80.0F, -80.0F,
        -80.0F, -80.0F, -80.0F, -80.0F}, true},
  };

  for (const auto& c : cases) {
    const double row_tol =
        c.rank_deficient ? kBirkhoffRowTolDegenerate : kBirkhoffRowTol;
    std::vector<float> out(16, 0.0F);
    moe::emu::sinkhorn_knopp_4x4(c.logits.data(), out.data(),
                                 moe::emu::kHcEpsilon,
                                 moe::emu::kHcSinkhornIters);

    const BirkhoffResidual r = BirkhoffCheck(out.data(), 4);

    char detail[320];
    std::snprintf(detail, sizeof(detail),
                  "row dev %.3e (tol %.1e), col dev %.3e (tol %.1e), "
                  "entries [%.6f, %.6f], finite=%d",
                  r.worst_row, row_tol, r.worst_col, kBirkhoffColTol,
                  r.min_entry, r.max_entry, static_cast<int>(r.finite));

    const bool ok = r.finite && r.worst_col <= kBirkhoffColTol &&
                    r.worst_row <= row_tol &&
                    r.min_entry >= 0.0 && r.max_entry <= 1.0 + 1e-6;
    rep.check(ok, std::string("Birkhoff polytope: ") + c.name, detail);
  }

  // Permutation invariance: the solve is symmetric under a simultaneous
  // row+column relabelling, so a permuted input must give the permuted output.
  // A transpose bug or a stray index would break this and nothing else.
  {
    Rng rng(0xB1CF0FFULL);
    const std::vector<float> logits = rng.normal(16, 0.0F, 4.0F);
    std::vector<float> base(16);
    moe::emu::sinkhorn_knopp_4x4(logits.data(), base.data(),
                                 moe::emu::kHcEpsilon,
                                 moe::emu::kHcSinkhornIters);

    const int perm[4] = {2, 0, 3, 1};
    std::vector<float> permuted(16);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        permuted[i * 4 + j] = logits[perm[i] * 4 + perm[j]];
      }
    }
    std::vector<float> got(16);
    moe::emu::sinkhorn_knopp_4x4(permuted.data(), got.data(),
                                 moe::emu::kHcEpsilon,
                                 moe::emu::kHcSinkhornIters);

    std::vector<float> want(16);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        want[i * 4 + j] = base[perm[i] * 4 + perm[j]];
      }
    }
    const double worst = MaxAbsDiff(got.data(), want.data(), 16);
    char detail[160];
    std::snprintf(detail, sizeof(detail), "max |diff| %.3e (seed 0x%llx)", worst,
                  static_cast<unsigned long long>(rng.seed()));
    rep.check(worst <= 1e-6, "Sinkhorn is permutation-equivariant", detail);
  }
}

// ===========================================================================
// 2. fused_mhc_post_residual -- reference structure
//
// y[j,:] = sum_i B[i,j] x[i,:] + post[j] * h_out[:]
//
// Two properties the kernel's transposed-indexing scheme must preserve, both
// checkable against the reference alone:
//   * with B = I and post = 0, the output is the input (BF16 round-trip),
//   * the map is additive in h_out for fixed B and post.
// ===========================================================================
void VerifyMhcPostStructure(Report& rep) {
  section("fused_mhc_post_residual: reference structure");

  // Both are fixed by the reference, which takes no dimension arguments and
  // reads exactly kHcStreams * kHiddenSize elements. Tie them to the named
  // constants so this suite cannot drift into an out-of-bounds read.
  constexpr int kStreams = moe::emu::kHcStreams;
  constexpr int kHidden = moe::emu::kHiddenSize;
  const std::size_t n = static_cast<std::size_t>(kStreams) * kHidden;

  Rng rng(0x9057ULL);
  const std::vector<float> x_f32 = rng.normal(n, 0.0F, 0.5F);
  const std::vector<std::uint16_t> x_bf16 = ToBf16(x_f32);
  const std::vector<float> h_out_f32 = rng.normal(kHidden, 0.0F, 0.5F);
  const std::vector<std::uint16_t> h_out_bf16 = ToBf16(h_out_f32);

  // --- identity B, zero post => BF16 passthrough -------------------------
  {
    std::vector<float> identity(kStreams * kStreams, 0.0F);
    for (int i = 0; i < kStreams; ++i) identity[i * kStreams + i] = 1.0F;
    const std::vector<float> post(kStreams, 0.0F);

    std::vector<std::uint16_t> y(n, 0);
    moe::emu::mhc_post_cpu(x_bf16.data(), identity.data(), post.data(),
                           h_out_bf16.data(), y.data());

    const std::vector<float> got = FromBf16(y);
    const std::vector<float> want = FromBf16(x_bf16);
    const double cos = CosineSimilarity(got.data(), want.data(), n);
    const double l2 = RelativeL2(got.data(), want.data(), n);

    char detail[200];
    std::snprintf(detail, sizeof(detail), "cos %.9f (floor %.5f), relL2 %.3e",
                  cos, kCosineFloor, l2);
    rep.check(cos >= kCosineFloor && l2 <= kRelL2Ceiling,
              "B=I, post=0 is a BF16 passthrough", detail);
  }

  // --- additivity in h_out ------------------------------------------------
  // f(h1) + f(h2) - f(0) == f(h1 + h2), since the B-mix term is common.
  // Computed in FP32 from BF16 inputs; the tolerance is set by the BF16 store
  // on each of the four evaluations, not by the algebra.
  {
    std::vector<float> B(kStreams * kStreams);
    {
      const std::vector<float> logits = rng.normal(kStreams * kStreams, 0.0F, 2.0F);
      moe::emu::sinkhorn_knopp_4x4(logits.data(), B.data(),
                                   moe::emu::kHcEpsilon,
                                   moe::emu::kHcSinkhornIters);
    }
    const std::vector<float> post = rng.uniform(kStreams, 0.2F, 1.8F);

    const std::vector<float> h1 = rng.normal(kHidden, 0.0F, 0.25F);
    const std::vector<float> h2 = rng.normal(kHidden, 0.0F, 0.25F);
    std::vector<float> h_sum(kHidden);
    std::vector<float> h_zero(kHidden, 0.0F);
    for (int d = 0; d < kHidden; ++d) h_sum[d] = h1[d] + h2[d];

    const auto eval = [&](const std::vector<float>& h) {
      const std::vector<std::uint16_t> hb = ToBf16(h);
      std::vector<std::uint16_t> y(n, 0);
      moe::emu::mhc_post_cpu(x_bf16.data(), B.data(), post.data(), hb.data(),
                             y.data());
      return FromBf16(y);
    };

    const std::vector<float> f1 = eval(h1);
    const std::vector<float> f2 = eval(h2);
    const std::vector<float> f0 = eval(h_zero);
    const std::vector<float> fs = eval(h_sum);

    std::vector<float> lhs(n);
    for (std::size_t i = 0; i < n; ++i) lhs[i] = f1[i] + f2[i] - f0[i];

    const double cos = CosineSimilarity(lhs.data(), fs.data(), n);
    const double l2 = RelativeL2(lhs.data(), fs.data(), n);
    char detail[220];
    std::snprintf(detail, sizeof(detail),
                  "cos %.9f (floor 0.9999), relL2 %.3e (ceiling %.1e, "
                  "BF16-store limited), seed 0x%llx",
                  cos, l2, 1e-2, static_cast<unsigned long long>(rng.seed()));
    // The bound is set by the storage format, not by the algebra. BF16 keeps an
    // 8-bit mantissa, so a single store carries up to 2^-9 ~ 2.0e-3 relative
    // error, and this identity combines FOUR independently rounded outputs
    // (f(h1), f(h2), f(0), f(h1+h2)). A 1e-2 relative-L2 ceiling is a little
    // over twice the measured 3.3e-3 -- tight enough that a real transposed-
    // index or stream-mixing bug (which moves whole feature rows, not last
    // bits) still fails it. Cosine stays strict: direction is unaffected by
    // rounding, so a structural error shows up there first.
    rep.check(cos >= 0.9999 && l2 <= 1e-2,
              "post-mix is additive in h_out", detail);
  }
}

// ===========================================================================
// 3. fast_fp8_indexer_topk -- edge cases and the selection contract
//
// The documented contract: top-k by BF16-rounded score descending, logical
// index ascending on a tie, and -1 in every slot beyond min(top_k, seq_len).
// ===========================================================================
void VerifyIndexerEdgeCases(Report& rep) {
  section("fast_fp8_indexer_topk: edge cases and selection contract");

  constexpr int kHeads = moe::emu::kIndexerHeads;      // 64
  constexpr int kHeadDim = moe::emu::kIndexerHeadDim;  // 128
  constexpr int kBlockSize = 128;

  // Geometry small enough to stay fast, large enough to span several blocks.
  const auto run = [&](std::int32_t seq_len, std::int32_t top_k,
                       std::vector<std::int32_t>& out) {
    Rng rng(0xF8ULL ^ static_cast<std::uint64_t>(seq_len * 131 + top_k));
    const int num_blocks = (seq_len + kBlockSize - 1) / kBlockSize + 1;

    const std::vector<std::uint8_t> query =
        rng.bytes(static_cast<std::size_t>(kHeads) * kHeadDim, 1, 120);
    const std::vector<std::uint8_t> keys = rng.bytes(
        static_cast<std::size_t>(num_blocks) * kBlockSize * kHeadDim, 1, 120);
    const std::vector<float> scales =
        rng.uniform(static_cast<std::size_t>(num_blocks) * kBlockSize, 0.5F, 1.5F);

    std::vector<std::int32_t> block_tables(num_blocks);
    for (int b = 0; b < num_blocks; ++b) block_tables[b] = b;

    out.assign(static_cast<std::size_t>(top_k), 0);
    moe::emu::fast_fp8_indexer_topk_cpu(
        query.data(), keys.data(), scales.data(), block_tables.data(), seq_len,
        top_k, out.data(), kBlockSize, kHeads, kHeadDim);
  };

  // --- seq_len < top_k: tail must be exactly -1 ---------------------------
  {
    const std::int32_t seq_len = 37;
    const std::int32_t top_k = 512;
    std::vector<std::int32_t> out;
    run(seq_len, top_k, out);

    bool tail_ok = true;
    for (std::int32_t i = seq_len; i < top_k; ++i) {
      if (out[static_cast<std::size_t>(i)] != -1) tail_ok = false;
    }
    bool head_in_range = true;
    std::set<std::int32_t> seen;
    for (std::int32_t i = 0; i < seq_len; ++i) {
      const std::int32_t v = out[static_cast<std::size_t>(i)];
      if (v < 0 || v >= seq_len) head_in_range = false;
      seen.insert(v);
    }
    char detail[220];
    std::snprintf(detail, sizeof(detail),
                  "seq_len=%d top_k=%d: tail_all_minus_one=%d head_in_range=%d "
                  "distinct=%zu (want %d)",
                  seq_len, top_k, static_cast<int>(tail_ok),
                  static_cast<int>(head_in_range), seen.size(),
                  static_cast<int>(seq_len));
    rep.check(tail_ok && head_in_range &&
                  seen.size() == static_cast<std::size_t>(seq_len),
              "seq_len < top_k pads with -1 and selects each key once", detail);
  }

  // --- zero-length sequence: every slot is -1 -----------------------------
  {
    const std::int32_t top_k = 512;
    std::vector<std::int32_t> out;
    run(0, top_k, out);
    const bool all_minus_one =
        std::all_of(out.begin(), out.end(), [](std::int32_t v) { return v == -1; });
    char detail[120];
    std::snprintf(detail, sizeof(detail), "first slot = %d, all -1 = %d", out[0],
                  static_cast<int>(all_minus_one));
    rep.check(all_minus_one, "zero-length sequence yields all -1", detail);
  }

  // --- seq_len == top_k exactly: a permutation, no padding ----------------
  {
    const std::int32_t n = 256;
    std::vector<std::int32_t> out;
    run(n, n, out);
    std::set<std::int32_t> seen(out.begin(), out.end());
    const bool is_perm = seen.size() == static_cast<std::size_t>(n) &&
                         *seen.begin() == 0 && *seen.rbegin() == n - 1;
    char detail[140];
    std::snprintf(detail, sizeof(detail), "distinct=%zu min=%d max=%d",
                  seen.size(), *seen.begin(), *seen.rbegin());
    rep.check(is_perm, "seq_len == top_k selects a full permutation", detail);
  }

  // --- tail padding: seq_len not a multiple of block_size -----------------
  // A block-boundary bug shows up as an out-of-range index or a duplicate.
  {
    const std::int32_t seq_len = 130;  // 128 + 2, straddles two blocks
    const std::int32_t top_k = 64;
    std::vector<std::int32_t> out;
    run(seq_len, top_k, out);
    std::set<std::int32_t> seen(out.begin(), out.end());
    bool in_range = true;
    for (const std::int32_t v : out) {
      if (v < 0 || v >= seq_len) in_range = false;
    }
    char detail[180];
    std::snprintf(detail, sizeof(detail),
                  "seq_len=%d (not a multiple of %d), top_k=%d: distinct=%zu "
                  "in_range=%d",
                  seq_len, kBlockSize, top_k, seen.size(),
                  static_cast<int>(in_range));
    rep.check(in_range && seen.size() == static_cast<std::size_t>(top_k),
              "unaligned tail selects distinct in-range keys", detail);
  }

  // --- selection really is score-ordered ----------------------------------
  // Recompute scores, BF16-round them as the contract says, and confirm the
  // emitted order is non-increasing under the documented total order.
  {
    const std::int32_t seq_len = 300;
    const std::int32_t top_k = 64;
    const int num_blocks = (seq_len + kBlockSize - 1) / kBlockSize + 1;
    Rng rng(0x5C0E5ULL);

    const std::vector<std::uint8_t> query =
        rng.bytes(static_cast<std::size_t>(kHeads) * kHeadDim, 1, 120);
    const std::vector<std::uint8_t> keys = rng.bytes(
        static_cast<std::size_t>(num_blocks) * kBlockSize * kHeadDim, 1, 120);
    const std::vector<float> scales =
        rng.uniform(static_cast<std::size_t>(num_blocks) * kBlockSize, 0.5F, 1.5F);
    std::vector<std::int32_t> block_tables(num_blocks);
    for (int b = 0; b < num_blocks; ++b) block_tables[b] = b;

    std::vector<float> scores(static_cast<std::size_t>(seq_len));
    moe::emu::fast_fp8_indexer_scores_cpu(query.data(), keys.data(),
                                          scales.data(), block_tables.data(),
                                          seq_len, scores.data(), kBlockSize,
                                          kHeads, kHeadDim);

    std::vector<std::int32_t> out(static_cast<std::size_t>(top_k));
    moe::emu::fast_fp8_indexer_topk_cpu(query.data(), keys.data(), scales.data(),
                                        block_tables.data(), seq_len, top_k,
                                        out.data(), kBlockSize, kHeads, kHeadDim);

    bool ordered = true;
    std::string why;
    for (std::int32_t i = 0; i + 1 < top_k; ++i) {
      const std::int32_t a = out[static_cast<std::size_t>(i)];
      const std::int32_t b = out[static_cast<std::size_t>(i + 1)];
      const float sa = Bf16ToFp32(Fp32ToBf16(scores[static_cast<std::size_t>(a)]));
      const float sb = Bf16ToFp32(Fp32ToBf16(scores[static_cast<std::size_t>(b)]));
      // score descending, then index ascending on an exact BF16 tie
      if (sa < sb || (sa == sb && a > b)) {
        ordered = false;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "rank %d: idx %d (%.6f) before idx %d (%.6f)", i, a, sa, b,
                      sb);
        why = buf;
        break;
      }
    }
    rep.check(ordered,
              "selection is score-descending then index-ascending on ties",
              ordered ? "" : why);
  }
}

// ===========================================================================
// 4. mxfp4_expert_gemv -- decode and group-scale accumulation
//
// The kernel decodes OCP E2M1 in registers and applies one E8M0 group scale
// per 32 logical columns, after the subtotal. Both properties are checkable
// against the reference with hand-built weights.
// ===========================================================================
void VerifyMxfp4Gemv(Report& rep) {
  section("mxfp4_expert_gemv: E2M1 decode and group-scale accumulation");

  constexpr int kRows = 32;
  constexpr int kCols = 256;  // 8 groups of 32
  constexpr int kGroup = moe::emu::kMxfp4GroupSize;

  // E2M1 code 0x2 is the value 1.0 (see DecodeFp4E2M1: {0,.5,1,1.5,2,3,4,6}).
  // Packing 0x22 in every byte makes every logical weight exactly 1.0, so with
  // an E8M0 scale of 1.0 the GEMV is a plain column sum -- a closed form the
  // reference must reproduce exactly.
  {
    const std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(kRows) * (kCols / 2), 0x22);
    // E8M0 bias is 127, so 127 encodes 2^0 == 1.0.
    const std::vector<std::uint8_t> scale(
        static_cast<std::size_t>(kRows) * (kCols / kGroup), 127);

    Rng rng(0x4F4ULL);
    const std::vector<float> x = rng.normal(kCols, 0.0F, 1.0F);

    moe::emu::Mxfp4Matrix w;
    w.packed = packed.data();
    w.scale = scale.data();
    w.rows = kRows;
    w.cols = kCols;

    std::vector<float> y(kRows, 0.0F);
    moe::emu::mxfp4_gemv_cpu(w, x.data(), y.data());

    // Closed form, accumulated in the reference's grouping order so that the
    // comparison is not sensitive to summation order.
    double want = 0.0;
    for (int g = 0; g < kCols / kGroup; ++g) {
      double sub = 0.0;
      for (int c = g * kGroup; c < (g + 1) * kGroup; ++c) sub += x[c];
      want += sub;
    }

    double worst = 0.0;
    for (int r = 0; r < kRows; ++r) {
      worst = std::max(worst, std::fabs(static_cast<double>(y[r]) - want));
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "all-ones weights, unit scale: want %.6f, max row dev %.3e",
                  want, worst);
    rep.check(worst <= 1e-3 && AllFinite(y.data(), kRows),
              "all-ones E2M1 with unit scale reduces to a column sum", detail);
  }

  // --- group scale applies per group, not globally ------------------------
  // Give group 0 a scale of 2^1 and the rest 2^0. The result must differ from
  // the unit-scale case by exactly the group-0 subtotal.
  {
    const std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(kRows) * (kCols / 2), 0x22);
    const int groups = kCols / kGroup;

    std::vector<std::uint8_t> scale_unit(
        static_cast<std::size_t>(kRows) * groups, 127);
    std::vector<std::uint8_t> scale_g0(scale_unit);
    for (int r = 0; r < kRows; ++r) scale_g0[r * groups + 0] = 128;  // 2^1

    Rng rng(0x4F5ULL);
    const std::vector<float> x = rng.normal(kCols, 0.0F, 1.0F);

    moe::emu::Mxfp4Matrix w;
    w.packed = packed.data();
    w.rows = kRows;
    w.cols = kCols;

    w.scale = scale_unit.data();
    std::vector<float> y_unit(kRows, 0.0F);
    moe::emu::mxfp4_gemv_cpu(w, x.data(), y_unit.data());

    w.scale = scale_g0.data();
    std::vector<float> y_g0(kRows, 0.0F);
    moe::emu::mxfp4_gemv_cpu(w, x.data(), y_g0.data());

    double g0_subtotal = 0.0;
    for (int c = 0; c < kGroup; ++c) g0_subtotal += x[c];

    double worst = 0.0;
    for (int r = 0; r < kRows; ++r) {
      const double delta = static_cast<double>(y_g0[r]) - y_unit[r];
      worst = std::max(worst, std::fabs(delta - g0_subtotal));
    }
    char detail[220];
    std::snprintf(detail, sizeof(detail),
                  "doubling group 0's scale shifts y by its subtotal %.6f; "
                  "max dev %.3e",
                  g0_subtotal, worst);
    rep.check(worst <= 1e-3,
              "E8M0 scale is applied per 32-column group", detail);
  }

  // --- zero code decodes to zero -----------------------------------------
  {
    const std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(kRows) * (kCols / 2), 0x00);
    const std::vector<std::uint8_t> scale(
        static_cast<std::size_t>(kRows) * (kCols / kGroup), 127);
    Rng rng(0x4F6ULL);
    const std::vector<float> x = rng.normal(kCols, 0.0F, 10.0F);

    moe::emu::Mxfp4Matrix w;
    w.packed = packed.data();
    w.scale = scale.data();
    w.rows = kRows;
    w.cols = kCols;

    std::vector<float> y(kRows, 1.0F);
    moe::emu::mxfp4_gemv_cpu(w, x.data(), y.data());
    const bool all_zero =
        std::all_of(y.begin(), y.end(), [](float v) { return v == 0.0F; });
    rep.check(all_zero && AllFinite(y.data(), kRows),
              "all-zero E2M1 codes produce an exactly zero result",
              all_zero ? "" : "a nonzero element survived");
  }
}

// ===========================================================================
// 5. fast_fp8_indexer_topk -- the SIMT selection network, simulated
//
// SelectTopKSimt (src/ops/fast_fp8_indexer_topk/op_kernel/...) sorts 2048
// composite keys with a bitonic network whose 66 levels are split into three
// tiers by how far the compare-exchange partner lives: this thread's own
// registers (jj < 4), another lane of the same warp via asc_shfl_xor
// (4 <= jj < 128), or another warp via UB (jj >= 128).
//
// That tiering is an indexing argument, and an indexing argument is exactly
// the kind of thing that is wrong the first time. It is also pure integer
// logic with no device dependency, so it can be checked HERE rather than
// hoped about: the loop below is a faithful transcription of the kernel's
// network -- same tier boundaries, same blocked ownership (thread t owns
// [4t, 4t+4)), same exchange rule -- run over 512 simulated threads, and its
// output is compared against std::sort of the same keys.
//
// This does not prove the kernel runs correctly on hardware. It proves the
// network it implements is a correct sort, which is the part that would
// otherwise only be discovered on an NPU.
// ===========================================================================
void VerifyIndexerSelectionNetwork(Report& rep) {
  section("fast_fp8_indexer_topk: SIMT selection network (host simulation)");

  // Must match the kernel's geometry constants exactly.
  constexpr std::uint32_t kSortLength = 2048;
  constexpr std::uint32_t kSimtThreads = 512;
  constexpr std::uint32_t kWarpSize = 32;
  constexpr std::uint32_t kKeysPerThread = kSortLength / kSimtThreads;  // 4

  // A faithful transcription of SelectTopKSimt's network. `keys` is the
  // cross-warp staging buffer; `reg[t][r]` stands in for thread t's
  // registers. Barriers are implicit: each tier-3 level reads a snapshot
  // taken after every thread has stored, which is what the two
  // asc_syncthreads() calls around it guarantee.
  const auto run_network = [&](std::vector<std::uint32_t> initial) {
    std::vector<std::array<std::uint32_t, kKeysPerThread>> reg(kSimtThreads);
    for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
      for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
        reg[t][r] = initial[t * kKeysPerThread + r];
      }
    }
    std::vector<std::uint32_t> staging(kSortLength, 0);

    for (std::uint32_t k = 2; k <= kSortLength; k <<= 1) {
      for (std::uint32_t jj = k >> 1; jj > 0; jj >>= 1) {
        if (jj < kKeysPerThread) {
          for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
            for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
              const std::uint32_t pr = r ^ jj;
              if (pr > r) {
                const std::uint32_t i = t * kKeysPerThread + r;
                const std::uint32_t a = reg[t][r], b = reg[t][pr];
                const std::uint32_t hi = a > b ? a : b;
                const std::uint32_t lo = a > b ? b : a;
                const bool want_desc = ((i & k) == 0u);
                reg[t][r] = want_desc ? hi : lo;
                reg[t][pr] = want_desc ? lo : hi;
              }
            }
          }
        } else if (jj < kKeysPerThread * kWarpSize) {
          // Warp shuffle: partner lane is lane ^ (jj / kKeysPerThread), same
          // r slot. Snapshot first, because a shuffle reads every lane's
          // pre-exchange value.
          const std::uint32_t lane_mask = jj / kKeysPerThread;
          const auto before = reg;
          for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
            const std::uint32_t warp_base = (t / kWarpSize) * kWarpSize;
            const std::uint32_t lane = t % kWarpSize;
            const std::uint32_t partner_t = warp_base + (lane ^ lane_mask);
            for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
              const std::uint32_t i = t * kKeysPerThread + r;
              const std::uint32_t a = before[t][r];
              const std::uint32_t b = before[partner_t][r];
              const std::uint32_t hi = a > b ? a : b;
              const std::uint32_t lo = a > b ? b : a;
              const bool want_desc = ((i & k) == 0u);
              const bool i_is_low = ((i & jj) == 0u);
              reg[t][r] = (i_is_low == want_desc) ? hi : lo;
            }
          }
        } else {
          for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
            for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
              staging[t * kKeysPerThread + r] = reg[t][r];
            }
          }
          for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
            for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
              const std::uint32_t i = t * kKeysPerThread + r;
              const std::uint32_t a = reg[t][r];
              const std::uint32_t b = staging[i ^ jj];
              const std::uint32_t hi = a > b ? a : b;
              const std::uint32_t lo = a > b ? b : a;
              const bool want_desc = ((i & k) == 0u);
              const bool i_is_low = ((i & jj) == 0u);
              reg[t][r] = (i_is_low == want_desc) ? hi : lo;
            }
          }
        }
      }
    }

    std::vector<std::uint32_t> out(kSortLength);
    for (std::uint32_t t = 0; t < kSimtThreads; ++t) {
      for (std::uint32_t r = 0; r < kKeysPerThread; ++r) {
        out[t * kKeysPerThread + r] = reg[t][r];
      }
    }
    return out;
  };

  struct Case {
    const char* name;
    std::vector<std::uint32_t> keys;
  };
  std::vector<Case> cases;

  // Random composites built the way the kernel builds them, so the low 16
  // bits are the inverted index and every key is distinct.
  {
    Rng rng(0x70F0ULL);
    const std::vector<float> scores = rng.normal(kSortLength, 0.0F, 2.0F);
    std::vector<std::uint32_t> keys(kSortLength);
    for (std::uint32_t j = 0; j < kSortLength; ++j) {
      const std::uint16_t bf = Fp32ToBf16(scores[j]);
      const std::uint16_t mono =
          static_cast<std::uint16_t>(bf ^ ((bf & 0x8000u) != 0u ? 0xFFFFu : 0x8000u));
      keys[j] = (static_cast<std::uint32_t>(mono) << 16) | (0xFFFFu ^ j);
    }
    cases.push_back({"random BF16 composites", keys});
  }
  // Already sorted, and reverse sorted: the two orders a buggy direction
  // flag passes or fails asymmetrically.
  {
    std::vector<std::uint32_t> asc(kSortLength), desc(kSortLength);
    for (std::uint32_t j = 0; j < kSortLength; ++j) {
      asc[j] = j + 1u;
      desc[j] = kSortLength - j;
    }
    cases.push_back({"ascending input (worst case for a direction bug)", asc});
    cases.push_back({"already descending (must be a no-op)", desc});
  }
  // Heavy padding: a short sequence leaves mostly kSelectionPadKey == 0,
  // which must all sink to the tail.
  {
    std::vector<std::uint32_t> keys(kSortLength, 0u);
    for (std::uint32_t j = 0; j < 37u; ++j) {
      keys[j] = (static_cast<std::uint32_t>(0x4000u + j * 7u) << 16) | (0xFFFFu ^ j);
    }
    cases.push_back({"seq_len 37 of 2048, rest padding", keys});
  }

  for (const auto& c : cases) {
    const std::vector<std::uint32_t> got = run_network(c.keys);
    std::vector<std::uint32_t> want = c.keys;
    std::sort(want.begin(), want.end(), std::greater<std::uint32_t>());

    std::size_t first_diff = kSortLength;
    for (std::size_t i = 0; i < kSortLength; ++i) {
      if (got[i] != want[i]) {
        first_diff = i;
        break;
      }
    }
    char detail[260];
    if (first_diff == kSortLength) {
      std::snprintf(detail, sizeof(detail),
                    "all %u ranks match std::sort descending", kSortLength);
    } else {
      std::snprintf(detail, sizeof(detail),
                    "first mismatch at rank %zu: network 0x%08X, std::sort "
                    "0x%08X",
                    first_diff, got[first_diff], want[first_diff]);
    }
    rep.check(first_diff == kSortLength,
              std::string("tiered bitonic network sorts: ") + c.name, detail);
  }
}

// ===========================================================================
// 6. Metric self-test
//
// The thresholds above are only meaningful if the metrics themselves are
// right. A metric that returns 1.0 for everything would make this whole
// binary vacuous, so it gets its own cases.
// ===========================================================================
// ===========================================================================
// 6. dsv4_moe_expert -- the fused expert's lane indexing and activation,
//    simulated
//
// Two things in this operator can only be wrong in ways a device run would
// discover late, and both are pure host-checkable logic (CLAUDE.md section
// 5.5):
//
//   * the lane decomposition. LaneRowPartial gives lane L of a 16-lane row
//     the pair range [L*pairs_per_lane, (L+1)*pairs_per_lane) and reads ONE
//     E8M0 scale byte per 16-pair group, starting at
//     (L*pairs_per_lane)/16. That alignment is an assumption: if a lane's
//     range ever straddled a group boundary, the scale applied to half its
//     pairs would silently be the wrong one.
//   * the asymmetric clamp. The gate is bounded ABOVE only and `up` is
//     bounded on BOTH sides. The two are not interchangeable, and the
//     prototype this operator descends from had both halves inverted.
//
// The simulation below transcribes the kernel's indexing -- same
// group-to-lane map, same uint4 word unpacking, same per-group scale fetch,
// same in-register butterfly order -- and compares against
// moe_expert_swiglu_mxfp4_cpu, the oracle the kernel's own header names.
// ===========================================================================
void VerifyDsv4MoeExpertSimt(Report& rep) {
  section("dsv4_moe_expert: lane indexing and asymmetric clamp (host simulation)");

  // Must match the kernel's geometry constants exactly.
  constexpr std::uint32_t kLanesPerRow = 16;
  constexpr int kGroup = moe::emu::kMxfp4GroupSize;
  // A reduced geometry, but NOT an arbitrary one. The lane decomposition
  // needs the row's group count to be a whole multiple of the 16 lanes, i.e.
  // cols % 512 == 0 (kDsv4ExpertReductionQuantum); below that
  // groups_per_lane floors to zero, every lane's step loop body never runs
  // and the GEMV silently returns zero. 256/128 fails that and produced
  // exactly that all-zero result when this test was first written.
  // Dsv4ExpertGeometryAdmissible() is what keeps a bad width away from the
  // kernel -- this is the smallest pair that exercises the decomposition.
  //
  // Both widths are 1024, not 1024/512, and that is load-bearing: the
  // group-to-lane map is  g = step * 16 + lane  with
  // groups_per_lane = (cols/32)/16, so at cols = 512 groups_per_lane is 1 and
  // the map degenerates to g = lane -- which is EXACTLY what the old
  // contiguous-block map g = lane * groups_per_lane + step also reduces to.
  // A 512-wide reduction therefore cannot tell the two geometries apart. 1024
  // gives groups_per_lane = 2, the smallest width at which they differ, on
  // BOTH the gate/up legs (cols = hidden) and the down leg (cols = inter).
  constexpr int kHidden = 1024;
  constexpr int kInter = 1024;
  constexpr float kLimit = moe::emu::kSwigluLimit;

  Rng rng(0xD5040001ULL);
  const std::vector<std::uint8_t> w1p =
      rng.bytes(static_cast<std::size_t>(kInter) * (kHidden / 2));
  const std::vector<std::uint8_t> w3p =
      rng.bytes(static_cast<std::size_t>(kInter) * (kHidden / 2));
  const std::vector<std::uint8_t> w2p =
      rng.bytes(static_cast<std::size_t>(kHidden) * (kInter / 2));
  // E8M0 codes near the 127 bias, so the products stay in a sane range.
  const std::vector<std::uint8_t> w1s =
      rng.bytes(static_cast<std::size_t>(kInter) * (kHidden / kGroup), 125, 129);
  const std::vector<std::uint8_t> w3s =
      rng.bytes(static_cast<std::size_t>(kInter) * (kHidden / kGroup), 125, 129);
  const std::vector<std::uint8_t> w2s =
      rng.bytes(static_cast<std::size_t>(kHidden) * (kInter / kGroup), 125, 129);
  const std::vector<float> x = rng.normal(kHidden, 0.0F, 1.0F);

  const auto matrix = [](const std::vector<std::uint8_t>& packed,
                         const std::vector<std::uint8_t>& scale, int rows, int cols) {
    moe::emu::Mxfp4Matrix m;
    m.packed = packed.data();
    m.scale = scale.data();
    m.rows = rows;
    m.cols = cols;
    return m;
  };
  const moe::emu::Mxfp4Matrix w1 = matrix(w1p, w1s, kInter, kHidden);
  const moe::emu::Mxfp4Matrix w3 = matrix(w3p, w3s, kInter, kHidden);
  const moe::emu::Mxfp4Matrix w2 = matrix(w2p, w2s, kHidden, kInter);

  // --- the oracle -------------------------------------------------------
  std::vector<float> want(kHidden);
  std::vector<float> scratch;
  moe::emu::moe_expert_swiglu_mxfp4_cpu(w1, w3, w2, x.data(), want.data(), kLimit,
                                        &scratch);

  // --- the kernel's own decomposition -----------------------------------
  // The coalesced geometry, transcribed. Two things are being simulated here
  // and neither needs hardware (CLAUDE.md §5.5):
  //
  //   1. the group-to-lane map  g = step * 16 + lane, replacing the old
  //      contiguous-block map  g = lane * groups_per_lane + step;
  //   2. the uint4 word unpacking -- the device issues ONE 16 B asc_ldca per
  //      group and then slices four 32-bit words, so the nibble order now
  //      runs through a word shift rather than straight off a byte pointer.
  //
  // (2) is the part most likely to be wrong the first time: a swapped word
  // order or a wrong shift still compiles for dav-3510 and still produces
  // plausible-looking numbers.
  constexpr std::uint32_t kGroupCols = 32u;   // logical columns per scale group
  constexpr std::uint32_t kGroupBytes = 16u;  // = one uint4

  // GroupDot, transcribed. The four 32-bit words of the uint4 are
  // little-endian, so word w is packed_group[4w..4w+3], low byte first.
  const auto group_dot = [](const std::uint8_t* packed_group,
                            const float* x_group) {
    float acc = 0.0F;
    for (std::uint32_t w = 0; w < 4u; ++w) {
      std::uint32_t word = 0;
      for (std::uint32_t b = 0; b < 4u; ++b) {
        word |= static_cast<std::uint32_t>(packed_group[4u * w + b]) << (8u * b);
      }
      const float* xp = x_group + 8u * w;  // word w holds 8 logical columns
      for (std::uint32_t j = 0; j < 4u; ++j) {
        const std::uint32_t by = (word >> (8u * j)) & 0xFFu;
        acc += moe::emu::DecodeFp4E2M1(static_cast<std::uint8_t>(by & 0xFu)) *
                   xp[2u * j] +
               moe::emu::DecodeFp4E2M1(static_cast<std::uint8_t>((by >> 4u) & 0xFu)) *
                   xp[2u * j + 1u];
      }
    }
    return acc;
  };

  // LaneRowPartial, transcribed.
  const auto lane_partial = [&](const std::uint8_t* packed_row,
                                const std::uint8_t* scale_row, std::uint32_t lane,
                                std::uint32_t cols, const float* in) {
    const std::uint32_t groups_per_lane = (cols / kGroupCols) / kLanesPerRow;
    float total = 0.0F;
    for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
      const std::uint32_t g = step * kLanesPerRow + lane;
      const float scale =
          std::ldexp(1.0F, static_cast<int>(scale_row[g]) - 127);
      const float partial =
          group_dot(packed_row + g * kGroupBytes, in + g * kGroupCols);
      total += partial * scale;  // scale once per group, after the subtotal
    }
    return total;
  };

  // --- check 0: the group-to-lane map is a bijection ---------------------
  // The optimization's whole premise is an address pattern, so the address
  // pattern is asserted directly rather than inferred from the fact that the
  // numbers came out close. Two properties:
  //
  //   coverage: every group of the row is visited exactly once across the 16
  //   lanes -- a dropped group is a silently wrong answer, and a group
  //   visited twice is double-counted;
  //
  //   coalescing: within one warp step the 16 lanes' packed addresses are
  //   16 B apart and ascending, i.e. the load quantum equals the lane
  //   stride, which is what makes a warp step span one contiguous 256 B run.
  {
    bool coverage_ok = true;
    bool coalesced_ok = true;
    for (const std::uint32_t cols : {static_cast<std::uint32_t>(kHidden),
                                     static_cast<std::uint32_t>(kInter)}) {
      const std::uint32_t groups = cols / kGroupCols;
      const std::uint32_t groups_per_lane = groups / kLanesPerRow;
      std::vector<int> visits(groups, 0);
      for (std::uint32_t step = 0; step < groups_per_lane; ++step) {
        for (std::uint32_t lane = 0; lane < kLanesPerRow; ++lane) {
          const std::uint32_t g = step * kLanesPerRow + lane;
          if (g >= groups) {
            coverage_ok = false;
            continue;
          }
          ++visits[g];
          // Lane l's byte offset against lane 0's, same step.
          const std::uint32_t base = step * kLanesPerRow * kGroupBytes;
          if (g * kGroupBytes != base + lane * kGroupBytes) coalesced_ok = false;
        }
      }
      for (const int v : visits) {
        if (v != 1) coverage_ok = false;
      }
    }
    rep.check(coverage_ok,
              "every scale group is owned by exactly one lane, exactly once",
              "the group-to-lane map drops or double-counts a group");
    rep.check(coalesced_ok,
              "a warp step's 16 lanes read 16 B apart (contiguous 256 B run)",
              "lane stride does not equal the uint4 load quantum");
  }
  // The in-register butterfly: xor masks 1, 2, 4, 8 over the row's 16 lanes.
  const auto reduce_row = [](std::vector<float> lanes) {
    for (std::uint32_t mask : {1u, 2u, 4u, 8u}) {
      std::vector<float> next(lanes.size());
      for (std::size_t l = 0; l < lanes.size(); ++l) {
        next[l] = lanes[l] + lanes[l ^ static_cast<std::size_t>(mask)];
      }
      lanes.swap(next);
    }
    return lanes[0];
  };
  const auto gemv_row = [&](const moe::emu::Mxfp4Matrix& w, int row, const float* in) {
    std::vector<float> lanes(kLanesPerRow);
    const std::size_t packed_stride = static_cast<std::size_t>(w.cols) / 2u;
    const std::size_t scale_stride = static_cast<std::size_t>(w.cols) / kGroup;
    for (std::uint32_t l = 0; l < kLanesPerRow; ++l) {
      lanes[l] = lane_partial(w.packed + static_cast<std::size_t>(row) * packed_stride,
                              w.scale + static_cast<std::size_t>(row) * scale_stride, l,
                              static_cast<std::uint32_t>(w.cols), in);
    }
    return reduce_row(lanes);
  };

  // --- check A: the lane decomposition, in isolation --------------------
  // Compared against mxfp4_gemv_cpu, the very function the oracle's legs
  // call, so the ONLY difference is how the per-group subtotals associate: a
  // 16-way lane tree here against the reference's sequential sum there.
  //
  // The bound is the standard summation-reordering bound applied to exactly
  // the quantity that reorders. Two FP32 summations of the same G addends in
  // different orders differ by at most 2 * G * u * sum|addend|, with
  // u = 2^-24. It is stated per row against that row's own sum of absolute
  // group subtotals rather than against the result, because a dot product of
  // signed terms can cancel to near zero and a result-relative bound would
  // then be reporting the row's condition number, not the kernel's error.
  // (A plain relative bound measured 2.6e-5 here for exactly that reason.)
  {
    constexpr double kUnitRoundoff = 1.0 / 16777216.0;  // 2^-24
    std::vector<float> ref_gate(kInter);
    moe::emu::mxfp4_gemv_cpu(w1, x.data(), ref_gate.data());
    bool within = true;
    double worst_ratio = 0.0;
    const std::uint32_t groups = static_cast<std::uint32_t>(kHidden) / kGroup;
    for (int r = 0; r < kInter; ++r) {
      // Re-walk the row to accumulate sum|group subtotal|, the addends both
      // summations share. Walked in plain group order, not lane order: this
      // quantity sums over the row's groups, and the group-to-lane map is a
      // bijection (check 0), so which lane owns which group cannot change it.
      // Carried in double because it is bounding the FP32 result's error.
      double abs_groups = 0.0;
      {
        const std::uint8_t* prow =
            w1.packed + static_cast<std::size_t>(r) * (kHidden / 2);
        const std::uint8_t* srow =
            w1.scale + static_cast<std::size_t>(r) * (kHidden / kGroup);
        for (std::uint32_t g = 0; g < groups; ++g) {
          const double scale =
              std::ldexp(1.0, static_cast<int>(srow[g]) - 127);
          // Same word/nibble order as group_dot, in double.
          double partial = 0.0;
          const std::uint8_t* pg = prow + g * 16u;
          for (std::uint32_t w = 0; w < 4u; ++w) {
            std::uint32_t word = 0;
            for (std::uint32_t b = 0; b < 4u; ++b) {
              word |= static_cast<std::uint32_t>(pg[4u * w + b]) << (8u * b);
            }
            const std::size_t xb =
                static_cast<std::size_t>(g) * 32u + static_cast<std::size_t>(w) * 8u;
            for (std::uint32_t j = 0; j < 4u; ++j) {
              const std::uint32_t by = (word >> (8u * j)) & 0xFFu;
              partial += static_cast<double>(moe::emu::DecodeFp4E2M1(
                             static_cast<std::uint8_t>(by & 0xFu))) *
                             x[xb + 2u * j] +
                         static_cast<double>(moe::emu::DecodeFp4E2M1(
                             static_cast<std::uint8_t>((by >> 4u) & 0xFu))) *
                             x[xb + 2u * j + 1u];
            }
          }
          abs_groups += std::fabs(partial * scale);
        }
      }
      const double allowed = 2.0 * groups * kUnitRoundoff * abs_groups;
      const double got_row = gemv_row(w1, r, x.data());
      const double ref_row = ref_gate[static_cast<std::size_t>(r)];
      const double deviation = std::fabs(got_row - ref_row);
      if (deviation > allowed) within = false;
      if (allowed > 0.0) worst_ratio = std::max(worst_ratio, deviation / allowed);
    }
    rep.check(within, "lane decomposition matches mxfp4_gemv_cpu per row",
              "a row exceeded the reordering bound; worst deviation/bound = " +
                  std::to_string(worst_ratio));
  }

  // --- check B: the clamped activation, over an exact gate/up -----------
  // Both sides start from the SAME reference GEMV outputs, so this isolates
  // the clamp and silu expression from any reduction-order difference. It is
  // the check that would have caught the inverted clamp and the swish sign
  // in the prototype this operator descends from.
  std::vector<float> activated(kInter);
  {
    std::vector<float> ref_gate(kInter);
    std::vector<float> ref_up(kInter);
    moe::emu::mxfp4_gemv_cpu(w1, x.data(), ref_gate.data());
    moe::emu::mxfp4_gemv_cpu(w3, x.data(), ref_up.data());
    bool clamp_ok = true;
    for (int r = 0; r < kInter; ++r) {
      const float gate = ref_gate[static_cast<std::size_t>(r)];
      const float up = ref_up[static_cast<std::size_t>(r)];
      const float g = std::min(gate, kLimit);
      const float u = std::min(kLimit, std::max(-kLimit, up));
      const float mine = (g / (1.0F + std::exp(-g))) * u;
      activated[static_cast<std::size_t>(r)] = mine;
      // An independent evaluation of the same contract in double precision.
      // This checks the FP32 expression against a higher-precision one, not
      // the contract itself -- that the contract IS DeepSeek-V4's activation
      // is what check C establishes, by comparing the assembled expert
      // against moe_expert_swiglu_mxfp4_cpu, which calls the real Silu.
      const double gd = static_cast<double>(g);
      const double want_el = (gd / (1.0 + std::exp(-gd))) * static_cast<double>(u);
      if (std::fabs(static_cast<double>(mine) - want_el) >
          1e-6 * std::max(1.0, std::fabs(want_el))) {
        clamp_ok = false;
      }
    }
    rep.check(clamp_ok, "the kernel's clamp and silu match the oracle elementwise",
              "an element diverged from Silu(min(gate, L)) * clamp(up, -L, L)");
  }

  // --- check C: the assembled expert ------------------------------------
  // Direction, not magnitude. The down leg sums 512 signed terms, so the
  // association difference check A bounds at 1e-5 per element is amplified by
  // whatever cancellation each output row happens to have -- that moves
  // magnitude and says nothing about correctness. A transposed weight, a
  // mis-strided row or a swapped nibble moves DIRECTION, which cosine sees
  // and a loosened relative-L2 bound would not (CLAUDE.md section 4.7).
  {
    std::vector<float> got(kHidden);
    for (int c = 0; c < kHidden; ++c) {
      got[static_cast<std::size_t>(c)] = gemv_row(w2, c, activated.data());
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int c = 0; c < kHidden; ++c) {
      const double a = got[static_cast<std::size_t>(c)];
      const double b = want[static_cast<std::size_t>(c)];
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    // Non-degeneracy on the REFERENCE, not a value against itself (section
    // 4.8): a zeroed oracle would make any output look perfectly correlated.
    rep.check(nb > 0.0 && na > 0.0 && AllFinite(want.data(), kHidden) &&
                  AllFinite(got.data(), kHidden),
              "both the simulated expert and its oracle are nonzero and finite",
              "the cosine below would be vacuous");
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    rep.check(cosine > 0.999999, "the assembled expert matches the oracle in direction",
              "cosine " + std::to_string(cosine));
  }

  // The two width properties the coalesced geometry assumes, stated as their
  // own checks over both the simulated widths and the production pair.
  //
  //   (a) the row's group count divides evenly across the 16 lanes, so every
  //       warp step is full and no group is dropped;
  //   (b) every row base is 16 B aligned, which the uint4 asc_ldca requires.
  //       The row base is row * (cols/2), so this is a property of the WIDTH:
  //       cols % 512 == 0 makes cols/2 a multiple of 256. Nothing else in the
  //       kernel enforces it, and a misaligned 16 B load is not a wrong
  //       answer but a fault.
  bool aligned = true;
  bool row_base_aligned = true;
  for (int cols : {kHidden, kInter, 4096, 2048}) {
    const std::uint32_t groups = static_cast<std::uint32_t>(cols) / 32u;
    aligned = aligned && (groups % kLanesPerRow == 0u);
    const std::uint32_t packed_row_stride = static_cast<std::uint32_t>(cols) / 2u;
    row_base_aligned = row_base_aligned && (packed_row_stride % 16u == 0u);
  }
  rep.check(row_base_aligned,
            "every packed row base is 16 B aligned for the uint4 group load",
            "a row stride is not a multiple of 16, so asc_ldca(uint4*) would "
            "be misaligned");
  rep.check(aligned, "the row's group count divides evenly across the 16 lanes",
            "a warp step would be partial, so some group is never summed");

  // The clamp is asymmetric, and the two halves are not interchangeable.
  {
    const float big = 1e6F;
    rep.check(std::min(big, kLimit) == kLimit && std::min(-big, kLimit) == -big,
              "gate is clamped above only", "the gate picked up a lower bound");
    rep.check(std::min(kLimit, std::max(-kLimit, big)) == kLimit &&
                  std::min(kLimit, std::max(-kLimit, -big)) == -kLimit,
              "up is clamped on both sides", "the up clamp is one-sided");
  }
}

void VerifyMetrics(Report& rep) {
  section("harness: metric self-test");

  const std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> same = a;
  const std::vector<float> scaled = {2.0F, 4.0F, 6.0F, 8.0F};
  const std::vector<float> flipped = {-1.0F, -2.0F, -3.0F, -4.0F};
  const std::vector<float> zeros(4, 0.0F);

  rep.check(std::fabs(CosineSimilarity(a.data(), same.data(), 4) - 1.0) < 1e-12,
            "cosine(x, x) == 1");
  rep.check(std::fabs(CosineSimilarity(a.data(), scaled.data(), 4) - 1.0) < 1e-12,
            "cosine is scale-invariant");
  rep.check(CosineSimilarity(a.data(), flipped.data(), 4) < -0.999,
            "cosine detects an inverted sign");
  rep.check(CosineSimilarity(a.data(), zeros.data(), 4) == 0.0,
            "cosine against an all-zero vector scores 0, not 1");
  rep.check(RelativeL2(a.data(), same.data(), 4) == 0.0, "relL2(x, x) == 0");
  rep.check(std::fabs(RelativeL2(scaled.data(), a.data(), 4) - 1.0) < 1e-12,
            "relL2 of a doubled vector is 1");
  rep.check(!AllFinite(std::vector<float>{1.0F, std::nanf("")}.data(), 2),
            "AllFinite rejects NaN");

  // A matrix that is NOT doubly stochastic must be caught. Note {0.5,0.5,0.5,
  // 0.5} would NOT do: as a 2x2 its rows and columns all sum to 1, so it is
  // the uniform doubly stochastic matrix. Use row sums 1.8 / 0.2 instead.
  const std::vector<float> not_ds = {0.9F, 0.9F, 0.1F, 0.1F};
  const BirkhoffResidual bad = BirkhoffCheck(not_ds.data(), 2);
  rep.check(bad.worst() > 1e-3, "BirkhoffCheck rejects a non-stochastic matrix",
            "row/col worst deviation did not exceed 1e-3");

  // ... and the uniform case really is on the polytope, which is what makes
  // the fixture above a meaningful negative rather than a lucky one.
  const std::vector<float> uniform_ds = {0.5F, 0.5F, 0.5F, 0.5F};
  rep.check(BirkhoffCheck(uniform_ds.data(), 2).worst() < 1e-6,
            "BirkhoffCheck accepts the uniform doubly stochastic matrix");

  const std::vector<float> perm = {1.0F, 0.0F, 0.0F, 1.0F};
  const BirkhoffResidual good = BirkhoffCheck(perm.data(), 2);
  rep.check(good.worst() < 1e-6, "BirkhoffCheck accepts a permutation matrix");

  // BF16 round-trip must be idempotent on already-rounded values.
  const float v = Bf16ToFp32(Fp32ToBf16(3.14159F));
  rep.check(Bf16ToFp32(Fp32ToBf16(v)) == v, "BF16 rounding is idempotent");
}

}  // namespace

int main() {
  std::printf("test_custom_kernels_cpu_ref\n");
  std::printf("  scope: host-only invariants of the CPU golden (no device)\n");
  std::printf("  references: in-memory per run (moe_emu), no disk fixtures\n");
  std::printf("  AscendCL: not linked   host CPU debug runtimes: not linked\n");

  Report rep;
  VerifyMetrics(rep);
  VerifySinkhornBirkhoff(rep);
  VerifyMhcPostStructure(rep);
  VerifyIndexerEdgeCases(rep);
  VerifyMxfp4Gemv(rep);
  VerifyIndexerSelectionNetwork(rep);
  VerifyDsv4MoeExpertSimt(rep);

  std::printf("\n---------------------------------------------------------\n");
  std::printf("checks: %d   failures: %d\n", rep.checks, rep.failures);
  // This target has no device gate and no conditional sections, so a skip here
  // would mean a case silently opted out. Treat it as a failure of the suite.
  if (rep.skipped > 0) {
    std::printf("RESULT: FAIL (%d skipped; the CPU oracle must not skip)\n",
                rep.skipped);
    return 1;
  }
  std::printf("%s\n", rep.failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
  return rep.exit_code();
}
