// -----------------------------------------------------------------------------
// test_custom_kernels_verification -- Target A.
//
// Functional and numerical correctness of the five arch35 kernels against the
// bit-exact CPU algorithmic reference in src/emu, with every reference value
// generated in memory on this run. No disk dumps, no pre-baked fixtures.
//
// Structure of each case:
//   1. synthesize inputs from a printed seed (reproducible),
//   2. compute the golden in memory via moe::emu,
//   3. compute the kernel result -- on device, via the Launch* wrappers,
//   4. assert cosine similarity / relative-L2 / Birkhoff bounds on the pair.
//
// Step 3 is currently unavailable: the Launch* host wrappers are specified in
// docs/kernel_contracts_spec.md but not implemented, so those comparisons
// report SKIP with that reason rather than silently passing. Everything that
// does NOT need the device -- reference invariants, the Birkhoff bounds, the
// edge-case contract, and the metric machinery itself -- runs for real here and
// will fail the binary if it breaks.
//
// This binary never loads a host CPU debug runtime (cpudebug / tikicpulib).
// -----------------------------------------------------------------------------

#include "kernel_test_harness.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"

#include <algorithm>
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
// 5. Metric self-test
//
// The thresholds above are only meaningful if the metrics themselves are
// right. A metric that returns 1.0 for everything would make this whole
// binary vacuous, so it gets its own cases.
// ===========================================================================
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

// ===========================================================================
// 6. On-device comparisons
//
// The actual kernel-vs-golden diffs. Reported as SKIP, with the reason, until
// the Launch* layer exists -- never as a pass.
// ===========================================================================
void VerifyAgainstDevice(Report& rep) {
  section("device: kernel output vs in-memory golden");

  if (!kDeviceLaunchLayerAvailable) {
    for (const char* k : {"fused_mhc_pre_norm_sinkhorn",
                          "fused_mhc_post_residual",
                          "paged_compressor_rope",
                          "fast_fp8_indexer_topk",
                          "mxfp4_expert_gemv"}) {
      rep.skip(std::string(k) + ": device output vs golden",
               DeviceUnavailableReason());
    }
    return;
  }

  // When the Launch* wrappers land, each case belongs here: allocate device
  // buffers, launch with the tiling struct from
  // include/moe/kernels/arch35/tiling_data.hpp, read back, and assert
  // CosineSimilarity >= kCosineFloor && RelativeL2 <= kRelL2Ceiling against the
  // golden computed above -- plus BirkhoffCheck on the Birkhoff output of
  // fused_mhc_pre_norm_sinkhorn.
  rep.skip("device comparisons",
           "launch layer reported available but no cases are wired yet");
}

}  // namespace

int main() {
  std::printf("test_custom_kernels_verification\n");
  std::printf("  references: in-memory per run (moe_emu), no disk fixtures\n");
  std::printf("  host CPU debug runtimes (cpudebug/tikicpulib): not linked\n");

  Report rep;
  VerifyMetrics(rep);
  VerifySinkhornBirkhoff(rep);
  VerifyMhcPostStructure(rep);
  VerifyIndexerEdgeCases(rep);
  VerifyMxfp4Gemv(rep);
  VerifyAgainstDevice(rep);

  std::printf("\n---------------------------------------------------------\n");
  std::printf("checks: %d   failures: %d   skipped: %d\n", rep.checks,
              rep.failures, rep.skipped);
  if (rep.skipped > 0) {
    std::printf("NOTE: %d device comparison(s) skipped -- see the reason above.\n",
                rep.skipped);
  }
  std::printf("%s\n", rep.failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
  return rep.exit_code();
}
