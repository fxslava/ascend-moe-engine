// -----------------------------------------------------------------------------
// bench_custom_kernels_performance -- Target B.
//
// Execution profiling: latency, standard deviation, throughput, effective
// bandwidth and compute utilization across token/batch geometries.
//
// Timing contract:
//   * warmup iterations run first and are discarded,
//   * only the measured region is timed -- allocation, host/device transfer and
//     stream synchronization setup all happen outside it,
//   * every geometry reports mean, stddev, min and p50 so a single outlier is
//     visible rather than averaged away.
//
// Device benchmarking goes through the Launch* host wrappers. Those now
// exist, so this binary measures real hardware when an NPU is visible: all
// five arch35 operators are wired, and the stream is synchronized inside the
// timed region (see TimedLaunch -- a chevron dispatch is asynchronous, so
// timing the launch call alone measures enqueue cost).
//
// paged_compressor_rope does not go through Measure() and reports HOLD and
// EMIT separately; see BenchPagedCompressorDevice for why a stateful operator
// cannot be timed by repeating an identical body.
//
// Not every device row is comparable to the host rows above it, and two are
// not even on the same axis: the Sinkhorn host sweep varies the iteration
// count, which is baked into the kernel and is not a device parameter, so
// fused_mhc_pre_norm_sinkhorn reports a single device row against four host
// rows. fused_mhc_post_residual does share its axis (tokens per step).
//
// --device-only skips the host reference rooflines entirely. They are scalar
// single-threaded C++ and cost ~0.7 s of wall clock at the default 3/20
// counts (measured on the dev workstation; the 2048x4096 MXFP4 GEMV at
// ~6.4 ms/call and the seq_len=2048 indexer scan at ~10 ms/call are most of
// it), scaling linearly with --iters. The device kernels they sit next to are
// microseconds, so in a profile-and-tune loop the host rooflines are
// effectively the entire run time while being a fixed baseline that does not
// move between runs.
//
// The host-reference figures are kept alongside, and are NOT comparable to the
// device rows: they are the roofline the kernel is beating.
//
// This file previously stated that the Launch* wrappers were "specified in
// docs/kernel_contracts_spec.md but not implemented in the tree", which stopped
// being true once the launch layer landed -- and that stale comment is why the
// benchmark went on quietly reporting every device row as unavailable.
//
// This binary never loads a host CPU debug runtime (cpudebug / tikicpulib).
// -----------------------------------------------------------------------------

#include "kernel_test_harness.hpp"

#include "device_session.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"

#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
#include "moe/kernels/arch35/launch.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <string>

namespace {

using namespace moe::kerntest;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
struct Timing {
  double mean_us = 0.0;
  double stddev_us = 0.0;
  double min_us = 0.0;
  double p50_us = 0.0;
  int iterations = 0;
};

// Runs `warmup` discarded iterations, then `iters` timed ones. `body` must do
// the work and nothing else: set up buffers before calling.
Timing Measure(int warmup, int iters, const std::function<void()>& body) {
  for (int i = 0; i < warmup; ++i) body();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  Timing t;
  t.iterations = iters;
  if (samples.empty()) return t;

  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  t.mean_us = sum / static_cast<double>(samples.size());
  double var = 0.0;
  for (const double s : samples) var += (s - t.mean_us) * (s - t.mean_us);
  t.stddev_us = std::sqrt(var / static_cast<double>(samples.size()));
  std::sort(samples.begin(), samples.end());
  t.min_us = samples.front();
  t.p50_us = samples[samples.size() / 2];
  return t;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
void Header(const char* kernel) {
  std::printf("\n=== %s ===\n", kernel);
  std::printf("%-26s %10s %9s %10s %12s %12s\n", "geometry", "mean us",
              "sd us", "min us", "GB/s", "GFLOP/s");
  std::printf("%-26s %10s %9s %10s %12s %12s\n", "--------------------------",
              "----------", "---------", "----------", "------------",
              "------------");
}

// bytes: the operand traffic a perfect implementation must move, once.
// flops: useful arithmetic, counting a multiply-add as 2.
void Row(const std::string& geometry, const Timing& t, double bytes,
         double flops) {
  const double secs = t.mean_us * 1e-6;
  const double gbs = secs > 0.0 ? bytes / secs / 1e9 : 0.0;
  const double gflops = secs > 0.0 ? flops / secs / 1e9 : 0.0;
  std::printf("%-26s %10.2f %9.2f %10.2f %12.2f %12.2f\n", geometry.c_str(),
              t.mean_us, t.stddev_us, t.min_us, gbs, gflops);
}

// Printed in --device-only mode in place of the host reference rows, so a
// section is never silently short: the rows are absent because they were asked
// for to be, not because a measurement failed.
void HostSkipped() {
  std::printf("  [host]   skipped (--device-only)\n");
}

// ---------------------------------------------------------------------------
// Device context
// ---------------------------------------------------------------------------
// Opened once in main(). Three states, kept apart on purpose:
//   * not linked against the launch layer  -> DeviceUnavailableReason()
//   * linked, but no NPU on this host      -> session's own reason
//   * linked and a device is open          -> measure
// A benchmark is never a correctness gate (CLAUDE.md section 4.6), so none of
// these is a failure: the unmeasured rows say so and the binary still exits 0.
struct DeviceCtx {
  bool usable = false;
  std::string reason = DeviceUnavailableReason();
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  moe::kerntest::device::AclSession session;
  aclrtStream stream = nullptr;
#endif
};

void DeviceRowUnavailable(const char* kernel, const DeviceCtx& ctx) {
  std::printf("  [device] %s: NOT MEASURED -- %s\n", kernel,
              ctx.reason.c_str());
}

#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)

using moe::kerntest::device::AclFail;
using moe::kerntest::device::DeviceBuffer;
using moe::kerntest::device::OpenResult;

// A launch plus the synchronize that makes it mean something.
//
// This is the one thing a device benchmark must not get wrong: a chevron
// dispatch is asynchronous and returns void (CLAUDE.md section 3.5), so timing
// the Launch* call alone measures enqueue cost and would report a kernel as
// orders of magnitude faster than it is. The stream is synchronized INSIDE the
// timed region; buffer allocation and H2D stay outside it.
//
// Returns false on the first failure, with `why` set, so a broken device path
// is reported rather than silently producing a plausible number.
bool TimedLaunch(const std::function<moe::kernels::arch35::Status()>& launch,
                 aclrtStream stream, int warmup, int iters, Timing* out,
                 std::string* why) {
  using moe::kernels::arch35::Status;
  bool ok = true;
  const auto one = [&] {
    if (!ok) return;
    const Status st = launch();
    if (st != Status::success) {
      // Every ACL failure inside a wrapper is recorded (CLAUDE.md section
      // 3.5), so report the record rather than just the Status -- otherwise a
      // tiling malloc, a bad grid and a device fault all read the same.
      const auto& f = moe::kernels::arch35::LastLaunchFailure();
      *why = "launch returned Status=" +
             std::to_string(static_cast<unsigned>(st)) + " [op=" +
             std::string(f.op != nullptr ? f.op : "-") + ", call=" +
             std::string(f.call != nullptr ? f.call : "-") +
             ", aclError=" + std::to_string(f.acl_error) + "]";
      ok = false;
      return;
    }
    const aclError rc = aclrtSynchronizeStream(stream);
    if (rc != ACL_SUCCESS) {
      *why = AclFail("aclrtSynchronizeStream", rc);
      ok = false;
    }
  };
  *out = Measure(warmup, iters, one);
  return ok;
}

// ---------------------------------------------------------------------------
// device: mxfp4_expert_gemv
//
// 32 AIVs, grid-stride over 32-row tiles. Operand traffic is the packed
// weights plus x plus y; the weights dominate and are read exactly once, so
// GB/s here is the figure that matters -- this kernel is bandwidth-bound and
// the host roofline above is not comparable to it.
// ---------------------------------------------------------------------------
void BenchMxfp4GemvDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  // The tiling contract admits exactly these two shapes.
  struct Shape {
    std::uint32_t rows;
    std::uint32_t cols;
  };
  for (const Shape s : {Shape{2048u, 4096u}, Shape{4096u, 2048u}}) {
    constexpr std::uint32_t kGroup = 32;
    const std::size_t packed_bytes =
        static_cast<std::size_t>(s.rows) * (s.cols / 2);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(s.rows) * (s.cols / kGroup);

    Rng rng(0xB4F4ULL + s.rows);
    const std::vector<std::uint8_t> packed = rng.bytes(packed_bytes);
    const std::vector<std::uint8_t> scale = rng.bytes(scale_bytes, 120, 134);
    const std::vector<float> x = rng.normal(s.cols);

    DeviceBuffer d_packed, d_scale, d_x, d_y;
    aclError rc = d_packed.Alloc(packed_bytes);
    if (rc == ACL_SUCCESS) rc = d_scale.Alloc(scale_bytes);
    if (rc == ACL_SUCCESS) rc = d_x.Alloc(s.cols * sizeof(float));
    if (rc == ACL_SUCCESS) rc = d_y.Alloc(s.rows * sizeof(float));
    if (rc == ACL_SUCCESS) rc = d_packed.Upload(packed.data(), packed_bytes);
    if (rc == ACL_SUCCESS) rc = d_scale.Upload(scale.data(), scale_bytes);
    if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), s.cols * sizeof(float));
    if (rc == ACL_SUCCESS) rc = d_y.Zero();
    if (rc != ACL_SUCCESS) {
      std::printf("  [device] setup failed: %s\n",
                  AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
      return;
    }

    Timing t;
    std::string why;
    const bool ok = TimedLaunch(
        [&] {
          return LaunchMxfp4ExpertGemv(
              d_packed.as<Mxfp4Pair>(), s.cols / 2, d_scale.as<E8M0>(),
              s.cols / kGroup, d_x.as<float>(), s.rows, s.cols,
              d_y.as<float>(), nullptr, kMxfp4GemvWorkspaceBytes, ctx.stream);
        },
        ctx.stream, warmup, iters, &t, &why);
    if (!ok) {
      std::printf("  [device] %ux%u: FAILED -- %s\n", s.rows, s.cols,
                  why.c_str());
      continue;
    }

    // One pass over the weights and scales, plus x in and y out.
    const double bytes = static_cast<double>(packed_bytes + scale_bytes) +
                         static_cast<double>(s.cols) * 4.0 +
                         static_cast<double>(s.rows) * 4.0;
    // rows*cols multiply-adds.
    const double flops =
        2.0 * static_cast<double>(s.rows) * static_cast<double>(s.cols);
    char geo[64];
    std::snprintf(geo, sizeof(geo), "[device] %ux%u", s.rows, s.cols);
    Row(geo, t, bytes, flops);
  }
}

// ---------------------------------------------------------------------------
// device: fast_fp8_indexer_topk
//
// Single AIV. Traffic is the key cache (seq_len x 128 FP8 bytes) read once
// per call, which is what the warp-per-key scoring streams through the SIMT
// DataCache; the 66-stage selection network is compute, not traffic.
// ---------------------------------------------------------------------------
void BenchIndexerDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  constexpr std::uint32_t kHeads = 64;
  constexpr std::uint32_t kHeadDim = 128;
  constexpr std::uint32_t kTopK = 512;
  constexpr std::uint32_t kBlockSize = 128;

  for (const std::uint32_t seq_len : {512u, 1024u, 2048u}) {
    const std::uint32_t blocks = (seq_len + kBlockSize - 1) / kBlockSize;
    const std::size_t query_bytes =
        static_cast<std::size_t>(kHeads) * kHeadDim;
    const std::size_t cache_bytes =
        static_cast<std::size_t>(blocks) * kBlockSize * kHeadDim;
    const std::size_t scales_count = kHeads + seq_len;

    Rng rng(0xF8DEULL + seq_len);
    const std::vector<std::uint8_t> query = rng.bytes(query_bytes, 56, 72);
    const std::vector<std::uint8_t> cache = rng.bytes(cache_bytes, 56, 72);
    const std::vector<float> scales = rng.normal(scales_count, 1.0F, 0.1F);
    std::vector<std::int32_t> table(blocks);
    for (std::uint32_t i = 0; i < blocks; ++i) {
      table[i] = static_cast<std::int32_t>(i);
    }

    const std::size_t ws_bytes = Fp8IndexerWorkspaceBytes(seq_len);

    DeviceBuffer d_q, d_cache, d_scales, d_table, d_idx, d_ws;
    aclError rc = d_q.Alloc(query_bytes);
    if (rc == ACL_SUCCESS) rc = d_cache.Alloc(cache_bytes);
    if (rc == ACL_SUCCESS) rc = d_scales.Alloc(scales_count * sizeof(float));
    if (rc == ACL_SUCCESS) {
      rc = d_table.Alloc(blocks * sizeof(std::int32_t));
    }
    if (rc == ACL_SUCCESS) rc = d_idx.Alloc(kTopK * sizeof(std::int32_t));
    if (rc == ACL_SUCCESS) rc = d_ws.Alloc(ws_bytes);
    if (rc == ACL_SUCCESS) rc = d_q.Upload(query.data(), query_bytes);
    if (rc == ACL_SUCCESS) rc = d_cache.Upload(cache.data(), cache_bytes);
    if (rc == ACL_SUCCESS) {
      rc = d_scales.Upload(scales.data(), scales_count * sizeof(float));
    }
    if (rc == ACL_SUCCESS) {
      rc = d_table.Upload(table.data(), blocks * sizeof(std::int32_t));
    }
    if (rc == ACL_SUCCESS) rc = d_idx.Zero();
    if (rc != ACL_SUCCESS) {
      std::printf("  [device] setup failed: %s\n",
                  AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
      return;
    }

    Timing t;
    std::string why;
    const bool ok = TimedLaunch(
        [&] {
          return LaunchFp8IndexerTopK(
              d_q.as<Fp8E4M3>(), kHeadDim, d_cache.as<Fp8E4M3>(),
              kBlockSize * kHeadDim, d_scales.as<float>(),
              d_table.as<std::int32_t>(), blocks, blocks, kBlockSize, seq_len,
              d_idx.as<std::int32_t>(), d_ws.get(), ws_bytes, ctx.stream);
        },
        ctx.stream, warmup, iters, &t, &why);
    if (!ok) {
      std::printf("  [device] seq_len=%u: FAILED -- %s\n", seq_len,
                  why.c_str());
      continue;
    }

    // Key cache read once, plus the query, the scales and the index output.
    const double bytes = static_cast<double>(cache_bytes + query_bytes) +
                         static_cast<double>(scales_count) * 4.0 +
                         static_cast<double>(kTopK) * 4.0;
    // heads * head_dim multiply-adds per key, plus the weighted reduction.
    const double flops = 2.0 * static_cast<double>(seq_len) *
                         static_cast<double>(kHeads) *
                         static_cast<double>(kHeadDim);
    char geo[64];
    std::snprintf(geo, sizeof(geo), "[device] seq_len %u, top-%u", seq_len,
                  kTopK);
    Row(geo, t, bytes, flops);
  }
}

// ---------------------------------------------------------------------------
// device: fused_mhc_pre_norm_sinkhorn
//
// Geometry is FIXED by the contract -- 4 streams x 4096, a [24, 16384] phi --
// so there is no device sweep to run. The host section above sweeps Sinkhorn
// iteration counts, which is NOT a device axis: the iteration count is baked
// into the kernel, so there is exactly one device row here and it is not
// row-comparable to the four host rows.
//
// Traffic is phi: [24, 16384] FP32 is 1.5 MB read once per call and everything
// else together is under 50 KB, so the GB/s column is effectively measuring
// the projection-matrix stream. The 16-thread register-resident Sinkhorn solve
// contributes essentially no traffic, which is the point of fusing it.
// ---------------------------------------------------------------------------
void BenchMhcPreNormDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  constexpr std::uint32_t kStreams = moe::emu::kHcStreams;      // 4
  constexpr std::uint32_t kHidden = moe::emu::kHiddenSize;      // 4096
  constexpr std::uint32_t kRows = moe::emu::kHcProjectionRows;  // 24
  constexpr std::uint32_t kFlat = moe::emu::kHcFlatWidth;       // 16384
  const std::size_t n = static_cast<std::size_t>(kStreams) * kHidden;
  const std::size_t phi_elems = static_cast<std::size_t>(kRows) * kFlat;

  Rng rng(0xD9E0ULL);
  const std::vector<std::uint16_t> x = ToBf16(rng.normal(n, 0.0F, 0.5F));
  // 1/sqrt(16384) == 1/128, so a row's 16384-term projection stays O(1) and
  // the measured path is the same arithmetic regime the verification suite
  // checks, not a denormal/overflow corner.
  const std::vector<float> phi = rng.normal(phi_elems, 0.0F, 1.0F / 128.0F);
  const std::vector<float> alpha = rng.uniform(3, 0.5F, 1.5F);
  const std::vector<float> bias = rng.normal(kRows, 0.0F, 0.25F);
  const std::vector<std::uint16_t> gamma =
      ToBf16(rng.uniform(kHidden, 0.5F, 1.5F));

  DeviceBuffer d_x, d_phi, d_alpha, d_bias, d_gamma;
  DeviceBuffer d_h_raw, d_h_norm, d_post, d_birkhoff;
  aclError rc = d_x.Alloc(n * 2);
  if (rc == ACL_SUCCESS) rc = d_phi.Alloc(phi_elems * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_alpha.Alloc(alpha.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_bias.Alloc(bias.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_gamma.Alloc(kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_h_raw.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_h_norm.Alloc(kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_post.Alloc(kStreams * sizeof(float));
  if (rc == ACL_SUCCESS) {
    rc = d_birkhoff.Alloc(kStreams * kStreams * sizeof(float));
  }
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), n * 2);
  if (rc == ACL_SUCCESS) {
    rc = d_phi.Upload(phi.data(), phi_elems * sizeof(float));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_alpha.Upload(alpha.data(), alpha.size() * sizeof(float));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_bias.Upload(bias.data(), bias.size() * sizeof(float));
  }
  if (rc == ACL_SUCCESS) rc = d_gamma.Upload(gamma.data(), kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_h_raw.Zero();
  if (rc == ACL_SUCCESS) rc = d_h_norm.Zero();
  if (rc == ACL_SUCCESS) rc = d_post.Zero();
  if (rc == ACL_SUCCESS) rc = d_birkhoff.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] setup failed: %s\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  Timing t;
  std::string why;
  const bool ok = TimedLaunch(
      [&] {
        return LaunchMhcPreNormSinkhorn(
            d_x.as<Bf16>(), kHidden, d_phi.as<float>(), kFlat,
            d_alpha.as<float>(), d_bias.as<float>(), d_gamma.as<Bf16>(),
            d_h_raw.as<float>(), d_h_norm.as<Bf16>(), d_post.as<float>(),
            d_birkhoff.as<float>(), nullptr, kMhcPreNormWorkspaceBytes,
            ctx.stream);
      },
      ctx.stream, warmup, iters, &t, &why);
  if (!ok) {
    std::printf("  [device] %ux%u: FAILED -- %s\n", kStreams, kHidden,
                why.c_str());
    return;
  }

  // phi read once, x read once, gamma read once, then the four outputs.
  const double bytes = static_cast<double>(phi_elems) * 4.0 +
                       static_cast<double>(n) * 2.0 +
                       static_cast<double>(kHidden) * 2.0 +
                       static_cast<double>(kHidden) * 4.0 +
                       static_cast<double>(kHidden) * 2.0 +
                       static_cast<double>(kStreams) * 4.0 +
                       static_cast<double>(kStreams) * kStreams * 4.0;
  // The [24, 16384] projection, plus the pre-gated mix h_in = sum_i pre[i]*x[i]
  // over 4x4096. The Sinkhorn solve is 4x4 and is not worth counting against
  // 393,216 projection MACs.
  const double flops = 2.0 * (static_cast<double>(phi_elems) +
                              static_cast<double>(n));
  char geo[64];
  std::snprintf(geo, sizeof(geo), "[device] %ux%u, phi %ux%u", kStreams,
                kHidden, kRows, kFlat);
  Row(geo, t, bytes, flops);
}

// ---------------------------------------------------------------------------
// device: fused_mhc_post_residual
//
// Bandwidth-bound and stateless, so unlike the compressor it can be repeated
// with an identical body. The sweep is over TOKENS per timed body, the same
// axis the host section sweeps and the one decode actually varies.
//
// WHAT THIS DOES NOT MEASURE: the v1 wrapper synchronizes the stream itself
// before releasing its tiling staging buffer (CLAUDE.md section 3.7), so
// `tokens` launches are `tokens` full host/device round trips, each with its
// own tiling malloc, H2D, dispatch, sync and free. Nothing is amortized or
// pipelined across them, and the expectation is therefore that mean time
// scales linearly and the GB/s column stays FLAT across the three rows. A
// flat row set is the correct result here, not a missing optimization; a
// superlinear row is the signal worth chasing. Measuring real per-launch
// amortization needs the pipelined wrapper variant with caller-owned tiling,
// which section 3.7 says not to invent.
//
// Re-launching over the same operand buffers is legitimate here precisely
// because this kernel carries no state across calls (contrast
// BenchPagedCompressorDevice, which cannot do this).
// ---------------------------------------------------------------------------
void BenchMhcPostDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  constexpr std::uint32_t kStreams = moe::emu::kHcStreams;  // 4, fixed
  constexpr std::uint32_t kHidden = moe::emu::kHiddenSize;  // 4096, fixed
  const std::size_t n = static_cast<std::size_t>(kStreams) * kHidden;

  Rng rng(0xD9057ULL);
  const std::vector<std::uint16_t> x = ToBf16(rng.normal(n, 0.0F, 0.5F));
  const std::vector<std::uint16_t> h_out =
      ToBf16(rng.normal(kHidden, 0.0F, 0.5F));
  // A real Birkhoff matrix, not arbitrary floats: the post-mix reads B as a
  // doubly-stochastic operator and a non-stochastic B would change the
  // magnitudes the accumulator sees.
  std::vector<float> B(static_cast<std::size_t>(kStreams) * kStreams);
  {
    const std::vector<float> logits =
        rng.normal(static_cast<std::size_t>(kStreams) * kStreams, 0.0F, 2.0F);
    moe::emu::sinkhorn_knopp_4x4(logits.data(), B.data(), moe::emu::kHcEpsilon,
                                 moe::emu::kHcSinkhornIters);
  }
  const std::vector<float> post =
      rng.uniform(static_cast<std::size_t>(kStreams), 0.2F, 1.8F);

  DeviceBuffer d_x, d_b, d_post, d_h, d_y;
  aclError rc = d_x.Alloc(n * 2);
  if (rc == ACL_SUCCESS) rc = d_b.Alloc(B.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_post.Alloc(post.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_h.Alloc(kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(n * 2);
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), n * 2);
  if (rc == ACL_SUCCESS) rc = d_b.Upload(B.data(), B.size() * sizeof(float));
  if (rc == ACL_SUCCESS) {
    rc = d_post.Upload(post.data(), post.size() * sizeof(float));
  }
  if (rc == ACL_SUCCESS) rc = d_h.Upload(h_out.data(), kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] setup failed: %s\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  for (const int tokens : {1, 8, 64}) {
    // One timed body = `tokens` launches. The trailing synchronize is a
    // belt-and-braces guard, not part of the cost being attributed: the
    // wrapper has already synchronized on its way out of the last launch, so
    // this call finds the stream idle and returns immediately. It is kept so
    // the body does not depend on that internal detail staying true.
    bool ok = true;
    std::string why;
    const auto body = [&] {
      if (!ok) return;
      for (int i = 0; i < tokens; ++i) {
        const Status st = LaunchMhcPostResidual(
            d_x.as<Bf16>(), kHidden, d_b.as<float>(), kStreams,
            d_post.as<float>(), d_h.as<Bf16>(), d_y.as<Bf16>(), kHidden,
            nullptr, kMhcPostWorkspaceBytes, ctx.stream);
        if (st != Status::success) {
          const auto& f = LastLaunchFailure();
          why = "launch returned Status=" +
                std::to_string(static_cast<unsigned>(st)) + " [op=" +
                std::string(f.op != nullptr ? f.op : "-") + ", call=" +
                std::string(f.call != nullptr ? f.call : "-") +
                ", aclError=" + std::to_string(f.acl_error) + "]";
          ok = false;
          return;
        }
      }
      const aclError sync_rc = aclrtSynchronizeStream(ctx.stream);
      if (sync_rc != ACL_SUCCESS) {
        why = AclFail("aclrtSynchronizeStream", sync_rc);
        ok = false;
      }
    };

    const Timing t = Measure(warmup, iters, body);
    if (!ok) {
      std::printf("  [device] %d tok: FAILED -- %s\n", tokens, why.c_str());
      continue;
    }

    // Per launch: x and y are the full 4x4096 BF16 operands, h_out is read
    // once. Matches the host row's accounting so the two columns mean the
    // same thing.
    const double elems = static_cast<double>(n) * tokens;
    const double bytes = elems * 2.0 * 2.0 +
                         static_cast<double>(kHidden) * 2.0 * tokens;
    const double flops = elems * (kStreams + 1) * 2.0;
    char geo[64];
    std::snprintf(geo, sizeof(geo), "[device] %d tok x %ux%u", tokens,
                  kStreams, kHidden);
    Row(geo, t, bytes, flops);
  }
}

// Summarize pre-collected samples. Needed because the compressor cannot use
// Measure(): see BenchPagedCompressorDevice.
Timing Summarize(std::vector<double> samples) {
  Timing t;
  t.iterations = static_cast<int>(samples.size());
  if (samples.empty()) return t;
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  t.mean_us = sum / static_cast<double>(samples.size());
  double var = 0.0;
  for (const double s : samples) var += (s - t.mean_us) * (s - t.mean_us);
  t.stddev_us = std::sqrt(var / static_cast<double>(samples.size()));
  std::sort(samples.begin(), samples.end());
  t.min_us = samples.front();
  t.p50_us = samples[samples.size() / 2];
  return t;
}

// ---------------------------------------------------------------------------
// device: paged_compressor_rope
//
// WHY THIS ONE DOES NOT USE Measure()
// Measure() repeats an identical body, which is exactly wrong for a stateful
// operator: LaunchPagedCompressorRope requires step == state_meta[1], so
// calling it twice with the same step is a contract violation, and resetting
// the state between iterations would put a device memset inside the timed
// region. Instead the cadence is driven forward continuously -- steps
// 0..8*groups-1, the way real decode does -- and each step is timed
// individually.
//
// HOLD and EMIT are reported SEPARATELY because they are not the same kernel
// path: a HOLD does the two projections and stores a state row, while an EMIT
// additionally does the column softmax over 8 rows, the Hadamard reduction,
// the RMSNorm and the RoPE rotation. Averaging them into one number would
// report neither, and with a 3:1 HOLD:EMIT ratio the blend is dominated by
// the cheaper path.
//
// Traffic is dominated by the two [coff*D, H] BF16 projection matrices, which
// are read once per step -- 16.8 MB at the real geometry -- so GB/s here is
// effectively measuring the weight stream, for both phases alike.
// ---------------------------------------------------------------------------
void BenchPagedCompressorDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  const std::uint32_t kH = static_cast<std::uint32_t>(moe::emu::kHiddenSize);
  const std::uint32_t kD =
      static_cast<std::uint32_t>(moe::emu::kCompressorHeadDim);
  const std::uint32_t kRatio =
      static_cast<std::uint32_t>(moe::emu::kCompressorRatio);
  const std::uint32_t kStateW =
      static_cast<std::uint32_t>(moe::emu::kCompressorStateWidth);
  const std::uint32_t kRopeDim =
      static_cast<std::uint32_t>(moe::emu::kRopeHeadDim);
  const std::uint32_t kStateFloats = 3u * kRatio * kStateW;

  // warmup groups are discarded, then `iters` groups are timed. At least two
  // timed groups, so the cross-boundary EMIT (the one that reads a populated
  // previous-a region) is always represented.
  const std::uint32_t warm_groups = static_cast<std::uint32_t>(warmup < 0 ? 0 : warmup);
  const std::uint32_t timed_groups =
      static_cast<std::uint32_t>(iters < 2 ? 2 : iters);

  const std::size_t proj = static_cast<std::size_t>(kStateW) * kH;
  Rng rng(0xC4A8ULL);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kH));
  const std::vector<std::uint16_t> wkv = ToBf16(rng.normal(proj, 0.0F, scale));
  const std::vector<std::uint16_t> wgate =
      ToBf16(rng.normal(proj, 0.0F, scale));
  const std::vector<float> ape =
      rng.normal(static_cast<std::size_t>(kRatio) * kStateW, 0.0F, 0.5F);
  const std::vector<std::uint16_t> norm_w = ToBf16(rng.normal(kD, 1.0F, 0.25F));
  const std::vector<float> rope_cos = rng.normal(kRopeDim, 0.0F, 1.0F);
  const std::vector<float> rope_sin = rng.normal(kRopeDim, 0.0F, 1.0F);
  const std::vector<std::uint16_t> x = ToBf16(rng.normal(kH, 0.0F, 0.5F));

  DeviceBuffer d_x, d_wkv, d_wgate, d_ape, d_norm, d_cos, d_sin;
  DeviceBuffer d_state, d_meta, d_slot, d_emitted;
  aclError rc = d_x.Alloc(kH * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_wkv.Alloc(proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_wgate.Alloc(proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_ape.Alloc(ape.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_norm.Alloc(kD * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_cos.Alloc(kRopeDim * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_sin.Alloc(kRopeDim * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_state.Alloc(kStateFloats * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_meta.Alloc(4 * sizeof(std::uint32_t));
  if (rc == ACL_SUCCESS) rc = d_slot.Alloc(kD * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_emitted.Alloc(sizeof(std::uint32_t));
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), kH * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) {
    rc = d_wkv.Upload(wkv.data(), proj * sizeof(std::uint16_t));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_wgate.Upload(wgate.data(), proj * sizeof(std::uint16_t));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_ape.Upload(ape.data(), ape.size() * sizeof(float));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_norm.Upload(norm_w.data(), kD * sizeof(std::uint16_t));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_cos.Upload(rope_cos.data(), kRopeDim * sizeof(float));
  }
  if (rc == ACL_SUCCESS) {
    rc = d_sin.Upload(rope_sin.data(), kRopeDim * sizeof(float));
  }
  // Cold start: have_previous = 0, tokens_seen = 0.
  if (rc == ACL_SUCCESS) rc = d_state.Zero();
  if (rc == ACL_SUCCESS) rc = d_meta.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] setup failed: %s\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  std::vector<double> hold_samples;
  std::vector<double> emit_samples;
  const std::uint32_t total_steps = (warm_groups + timed_groups) * kRatio;
  for (std::uint32_t s = 0; s < total_steps; ++s) {
    const bool is_emit = ((s % kRatio) == kRatio - 1);
    const bool timed = (s >= warm_groups * kRatio);

    const auto t0 = std::chrono::steady_clock::now();
    const Status st = LaunchPagedCompressorRope(
        d_x.as<Bf16>(), d_wkv.as<Bf16>(), kH, d_wgate.as<Bf16>(), kH,
        d_ape.as<float>(), kStateW, d_norm.as<Bf16>(), d_cos.as<float>(),
        d_sin.as<float>(), s, d_state.as<float>(), d_meta.as<std::uint32_t>(),
        d_slot.as<Bf16>(), d_emitted.as<std::uint32_t>(), nullptr,
        kPagedCompressorWorkspaceBytes, ctx.stream);
    if (st == Status::success) {
      const aclError sync_rc = aclrtSynchronizeStream(ctx.stream);
      if (sync_rc != ACL_SUCCESS) {
        std::printf("  [device] step %u: %s\n", s,
                    AclFail("aclrtSynchronizeStream", sync_rc).c_str());
        return;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (st != Status::success) {
      const auto& f = LastLaunchFailure();
      std::printf("  [device] step %u: FAILED Status=%u [op=%s, call=%s, "
                  "aclError=%d]\n",
                  s, static_cast<unsigned>(st),
                  f.op != nullptr ? f.op : "-", f.call != nullptr ? f.call : "-",
                  f.acl_error);
      return;
    }
    if (timed) {
      const double us =
          std::chrono::duration<double, std::micro>(t1 - t0).count();
      (is_emit ? emit_samples : hold_samples).push_back(us);
    }
  }

  // Both projection matrices stream once per step; everything else is small.
  const double weight_bytes =
      2.0 * static_cast<double>(proj) * 2.0;  // two BF16 [coff*D, H]
  const double bytes = weight_bytes +
                       static_cast<double>(kH) * 2.0 +
                       static_cast<double>(ape.size()) * 4.0 +
                       static_cast<double>(kD) * 2.0;
  // Two [coff*D, H] matrix-vector products, 2 flops per MAC.
  const double flops = 2.0 * 2.0 * static_cast<double>(proj);

  char geo[80];
  std::snprintf(geo, sizeof(geo), "[device] HOLD step (%u)",
                static_cast<unsigned>(hold_samples.size()));
  Row(geo, Summarize(hold_samples), bytes, flops);
  std::snprintf(geo, sizeof(geo), "[device] EMIT step (%u)",
                static_cast<unsigned>(emit_samples.size()));
  Row(geo, Summarize(emit_samples), bytes, flops);
  std::printf("           HOLD and EMIT are separate kernel paths; EMIT adds "
              "the 8-row column softmax,\n           the Hadamard reduction, "
              "the RMSNorm and the RoPE rotation on top of the projections.\n");
}

#endif  // MOE_KERNEL_LAUNCH_LAYER_AVAILABLE

// ---------------------------------------------------------------------------
// Device dispatchers
// ---------------------------------------------------------------------------
// Always defined so the host sections can call them unconditionally; each one
// either measures or prints why it did not. All five kernels are now wired.
void DeviceSectionMxfp4(const DeviceCtx& ctx, int warmup, int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchMxfp4GemvDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("mxfp4_expert_gemv", ctx);
}

void DeviceSectionPagedCompressor(const DeviceCtx& ctx, int warmup,
                                  int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchPagedCompressorDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("paged_compressor_rope", ctx);
}

void DeviceSectionIndexer(const DeviceCtx& ctx, int warmup, int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchIndexerDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("fast_fp8_indexer_topk", ctx);
}

void DeviceSectionMhcPreNorm(const DeviceCtx& ctx, int warmup, int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchMhcPreNormDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("fused_mhc_pre_norm_sinkhorn", ctx);
}

void DeviceSectionMhcPost(const DeviceCtx& ctx, int warmup, int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchMhcPostDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("fused_mhc_post_residual", ctx);
}

// ---------------------------------------------------------------------------
// 1. fused_mhc_pre_norm_sinkhorn (host reference roofline)
//
// The Sinkhorn solve alone, across iteration counts, since that is the part
// the kernel keeps entirely in registers across 16 SIMT threads.
// ---------------------------------------------------------------------------
void BenchSinkhorn(const DeviceCtx& ctx, int warmup, int iters,
                   bool device_only) {
  Header("fused_mhc_pre_norm_sinkhorn (host reference: Sinkhorn solve)");

  if (device_only) {
    HostSkipped();
    DeviceSectionMhcPreNorm(ctx, warmup, iters);
    return;
  }

  for (const int n_iters : {5, 10, 20, 40}) {
    Rng rng(0x51C0ULL + static_cast<std::uint64_t>(n_iters));
    const std::vector<float> logits = rng.normal(16, 0.0F, 2.0F);
    std::vector<float> out(16);

    const Timing t = Measure(warmup, iters, [&] {
      moe::emu::sinkhorn_knopp_4x4(logits.data(), out.data(),
                                   moe::emu::kHcEpsilon, n_iters);
    });

    // 4x4, two normalizations per iteration, ~2 ops per entry per pass.
    const double flops = 16.0 * 2.0 * 2.0 * n_iters;
    const double bytes = 16.0 * 4.0 * 2.0;  // read logits, write out
    char geo[64];
    std::snprintf(geo, sizeof(geo), "4x4, %d iters", n_iters);
    Row(geo, t, bytes, flops);
  }
  DeviceSectionMhcPreNorm(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 2. fused_mhc_post_residual (host reference roofline)
//
// Bandwidth-bound. NOTE the per-call geometry is NOT a free parameter:
// moe::emu::mhc_post_cpu takes no dimension arguments and hardcodes
// n = kHcStreams (4), d = kHiddenSize (4096). Passing a narrower buffer makes
// it read 16,384 elements out of a shorter allocation, which is an
// out-of-bounds read, not a smaller benchmark. The sweep is therefore over
// TOKENS per step -- one 4x4096 post-mix per token, which is the axis that
// actually varies at decode time.
// ---------------------------------------------------------------------------
void BenchMhcPost(const DeviceCtx& ctx, int warmup, int iters,
                  bool device_only) {
  Header("fused_mhc_post_residual (host reference: 4-stream post-mix)");

  if (device_only) {
    HostSkipped();
    DeviceSectionMhcPost(ctx, warmup, iters);
    return;
  }

  constexpr int kStreams = moe::emu::kHcStreams;   // 4, fixed by the reference
  constexpr int kHidden = moe::emu::kHiddenSize;   // 4096, fixed likewise
  const std::size_t per_token = static_cast<std::size_t>(kStreams) * kHidden;

  for (const int tokens : {1, 8, 64}) {
    Rng rng(0x9057ULL + static_cast<std::uint64_t>(tokens));

    // One full-width buffer per token, so each call sees its own contract-sized
    // operand and nothing is read past its end.
    std::vector<std::vector<std::uint16_t>> x(static_cast<std::size_t>(tokens));
    std::vector<std::vector<std::uint16_t>> y(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
      x[static_cast<std::size_t>(i)] = ToBf16(rng.normal(per_token, 0.0F, 0.5F));
      y[static_cast<std::size_t>(i)].assign(per_token, 0);
    }
    const std::vector<std::uint16_t> h_out =
        ToBf16(rng.normal(static_cast<std::size_t>(kHidden), 0.0F, 0.5F));

    std::vector<float> B(kStreams * kStreams);
    {
      const std::vector<float> logits =
          rng.normal(static_cast<std::size_t>(kStreams) * kStreams, 0.0F, 2.0F);
      moe::emu::sinkhorn_knopp_4x4(logits.data(), B.data(),
                                   moe::emu::kHcEpsilon,
                                   moe::emu::kHcSinkhornIters);
    }
    const std::vector<float> post =
        rng.uniform(static_cast<std::size_t>(kStreams), 0.2F, 1.8F);

    const Timing t = Measure(warmup, iters, [&] {
      for (int i = 0; i < tokens; ++i) {
        moe::emu::mhc_post_cpu(x[static_cast<std::size_t>(i)].data(), B.data(),
                               post.data(), h_out.data(),
                               y[static_cast<std::size_t>(i)].data());
      }
    });

    // Per output element: kStreams B-mix multiply-adds plus one post*h_out.
    const double elems = static_cast<double>(per_token) * tokens;
    const double flops = elems * (kStreams + 1) * 2.0;
    // BF16 throughout: x read, y written, h_out read once per call.
    const double bytes = elems * 2.0 * 2.0 +
                         static_cast<double>(kHidden) * 2.0 * tokens;
    char geo[64];
    std::snprintf(geo, sizeof(geo), "%d tok x %dx%d", tokens, kStreams, kHidden);
    Row(geo, t, bytes, flops);
  }
  DeviceSectionMhcPost(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 3. fast_fp8_indexer_topk (host reference roofline)
//
// Scales with sequence length; the scoring half dominates.
// ---------------------------------------------------------------------------
void BenchIndexer(const DeviceCtx& ctx, int warmup, int iters,
                  bool device_only) {
  Header("fast_fp8_indexer_topk (host reference: score + select)");

  if (device_only) {
    HostSkipped();
    DeviceSectionIndexer(ctx, warmup, iters);
    return;
  }

  constexpr int kHeads = moe::emu::kIndexerHeads;
  constexpr int kHeadDim = moe::emu::kIndexerHeadDim;
  constexpr int kBlockSize = 128;

  for (const std::int32_t seq_len : {128, 512, 2048}) {
    const std::int32_t top_k = std::min<std::int32_t>(512, seq_len);
    const int num_blocks = (seq_len + kBlockSize - 1) / kBlockSize + 1;
    Rng rng(0xF8ULL + static_cast<std::uint64_t>(seq_len));

    const std::vector<std::uint8_t> query =
        rng.bytes(static_cast<std::size_t>(kHeads) * kHeadDim, 1, 120);
    const std::vector<std::uint8_t> keys = rng.bytes(
        static_cast<std::size_t>(num_blocks) * kBlockSize * kHeadDim, 1, 120);
    const std::vector<float> scales =
        rng.uniform(static_cast<std::size_t>(num_blocks) * kBlockSize, 0.5F, 1.5F);
    std::vector<std::int32_t> block_tables(num_blocks);
    for (int b = 0; b < num_blocks; ++b) block_tables[b] = b;
    std::vector<std::int32_t> out(static_cast<std::size_t>(top_k));

    const Timing t = Measure(warmup, iters, [&] {
      moe::emu::fast_fp8_indexer_topk_cpu(
          query.data(), keys.data(), scales.data(), block_tables.data(),
          seq_len, top_k, out.data(), kBlockSize, kHeads, kHeadDim);
    });

    // One multiply-add per (head, head_dim, key).
    const double flops = static_cast<double>(kHeads) * kHeadDim * seq_len * 2.0;
    // FP8 keys are the traffic: one byte per (key, head_dim).
    const double bytes = static_cast<double>(seq_len) * kHeadDim +
                         static_cast<double>(kHeads) * kHeadDim;
    char geo[64];
    std::snprintf(geo, sizeof(geo), "seq %d, top_k %d", seq_len, top_k);
    Row(geo, t, bytes, flops);
  }
  DeviceSectionIndexer(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 4. mxfp4_expert_gemv (host reference roofline)
//
// The contract geometries: 2048x4096 and 4096x2048.
// ---------------------------------------------------------------------------
void BenchMxfp4Gemv(const DeviceCtx& ctx, int warmup, int iters,
                    bool device_only) {
  Header("mxfp4_expert_gemv (host reference: MXFP4 GEMV)");

  if (device_only) {
    HostSkipped();
    DeviceSectionMxfp4(ctx, warmup, iters);
    return;
  }

  constexpr int kGroup = moe::emu::kMxfp4GroupSize;
  const std::pair<int, int> geometries[] = {{2048, 4096}, {4096, 2048}};

  for (const auto& [rows, cols] : geometries) {
    Rng rng(0x4F4ULL + static_cast<std::uint64_t>(rows));
    const std::vector<std::uint8_t> packed =
        rng.bytes(static_cast<std::size_t>(rows) * (cols / 2));
    // E8M0 near the bias so the decode cannot overflow to inf.
    const std::vector<std::uint8_t> scale =
        rng.bytes(static_cast<std::size_t>(rows) * (cols / kGroup), 120, 134);
    const std::vector<float> x = rng.normal(static_cast<std::size_t>(cols));

    moe::emu::Mxfp4Matrix w;
    w.packed = packed.data();
    w.scale = scale.data();
    w.rows = rows;
    w.cols = cols;

    std::vector<float> y(static_cast<std::size_t>(rows), 0.0F);

    const Timing t = Measure(warmup, iters, [&] {
      moe::emu::mxfp4_gemv_cpu(w, x.data(), y.data());
    });

    const double flops = static_cast<double>(rows) * cols * 2.0;
    // Weights dominate: half a byte per logical element, plus the scales.
    const double bytes = static_cast<double>(rows) * (cols / 2) +
                         static_cast<double>(rows) * (cols / kGroup) +
                         static_cast<double>(cols) * 4.0;
    char geo[64];
    std::snprintf(geo, sizeof(geo), "%d x %d", rows, cols);
    Row(geo, t, bytes, flops);
  }
  DeviceSectionMxfp4(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 5. paged_compressor_rope
//
// The reference's decode cadence is stateful (a persistent ring buffer across
// steps), so a meaningful figure needs the full group of 4 steps driven in
// order. Left to the device path rather than approximated here.
// ---------------------------------------------------------------------------
void BenchPagedCompressor(const DeviceCtx& ctx, int warmup, int iters) {
  Header("paged_compressor_rope");
  std::printf("  [host]   not benchmarked: the decode cadence is stateful "
              "across a 4-step group;\n           a per-call figure would "
              "misrepresent the hold/emit asymmetry.\n");
  DeviceSectionPagedCompressor(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// Scaling across cores
//
// On device this is the AIV sweep (1, 4, 32 cores) reported as scaling
// efficiency. There is no device here, so state the intent rather than print
// a host number dressed up as a hardware figure.
// ---------------------------------------------------------------------------
void ReportScaling(const DeviceCtx& ctx) {
  std::printf("\n=== AIV scaling efficiency ===\n");
  if (!ctx.usable) {
    std::printf("  NOT MEASURED -- %s\n", ctx.reason.c_str());
    return;
  }
  // Not measurable as the comment above once described, and the reason is a
  // contract fact rather than a missing feature: each kernel's block_dim is
  // PINNED by its tiling, and the device-side entry gate rejects any other
  // value outright -- mxfp4_expert_gemv requires block_dim == 32, the
  // single-AIV kernels require 1. A launch with a swept block_dim therefore
  // does not run a smaller grid, it returns early and computes nothing, which
  // would time as a near-instant "perfectly scaling" kernel.
  //
  // Measuring t(N) honestly needs MakeMxfp4GemvTiling to accept a caller
  // block_dim and the kernel gate to accept it with it, i.e. an ABI change.
  // Printing nothing is better than printing a number that measures an early
  // return.
  std::printf("  NOT MEASURED -- block_dim is fixed by each kernel's tiling\n"
              "  contract (32 for mxfp4_expert_gemv, 1 for the single-AIV\n"
              "  kernels) and the device entry gate rejects any other value,\n"
              "  so a block_dim sweep would time early returns rather than\n"
              "  smaller grids. A real sweep needs the tiling builder and the\n"
              "  kernel gate to take block_dim as a parameter.\n");
}

}  // namespace

int main(int argc, char** argv) {
  int warmup = 3;
  int iters = 20;
  bool device_only = false;

  // Positional [warmup] [iters] as before, plus --device-only anywhere. The
  // host references are single-threaded scalar C++ costing ~0.7 s at the
  // default counts and scaling linearly with `iters`, and they are a fixed
  // roofline that does not change between runs. When the question is "did
  // this kernel get faster on hardware", re-measuring that roofline is pure
  // wall-clock cost, so --device-only drops it.
  {
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--device-only") {
        device_only = true;
      } else if (arg == "--help" || arg == "-h") {
        std::printf("usage: %s [warmup] [iters] [--device-only]\n", argv[0]);
        std::printf("  --device-only  skip the host reference rooflines and "
                    "measure only the device rows\n");
        return 0;
      } else if (++positional == 1) {
        warmup = std::atoi(arg.c_str());
      } else if (positional == 2) {
        iters = std::atoi(arg.c_str());
      } else {
        std::printf("unrecognized argument '%s' (try --help)\n", arg.c_str());
        return 2;
      }
    }
  }
  if (warmup < 0) warmup = 0;
  if (iters < 1) iters = 1;

  std::printf("bench_custom_kernels_performance\n");
  std::printf("  warmup=%d (discarded), timed iterations=%d\n", warmup, iters);
  std::printf("  allocation and setup are outside the timed region\n");
  std::printf("  host CPU debug runtimes (cpudebug/tikicpulib): not linked\n");
  if (device_only) {
    std::printf("  --device-only: host reference rooflines are SKIPPED\n");
  } else {
    std::printf("  NOTE: host-reference figures are a roofline baseline, NOT "
                "device numbers.\n");
  }

  // Open the device once. A benchmark is never a gate (CLAUDE.md section
  // 4.6), so neither "no NPU" nor a bring-up failure is fatal here -- both
  // just mean the device rows say why they are empty. The two are still kept
  // apart in the printed reason.
  DeviceCtx ctx;
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  {
    std::string why;
    const OpenResult res = ctx.session.Open(&why);
    if (res == OpenResult::kOk) {
      ctx.usable = true;
      ctx.stream = ctx.session.stream();
      std::printf("  device: %u NPU(s) visible, measuring on device 0\n",
                  ctx.session.device_count());
    } else {
      ctx.reason = (res == OpenResult::kNoDevice ? "no NPU on this host: "
                                                 : "device bring-up FAILED: ") +
                   why;
      std::printf("  device: %s\n", ctx.reason.c_str());
    }
  }
#else
  std::printf("  device: %s\n", ctx.reason.c_str());
#endif

  // --device-only with no usable device measures nothing at all. Still not a
  // failure -- a benchmark is never a gate (CLAUDE.md section 4.6) -- but say
  // so plainly, because "every row empty" otherwise looks like a broken run
  // rather than two flags that cancel out.
  if (device_only && !ctx.usable) {
    std::printf("  WARNING: --device-only was given but no device is usable, "
                "so nothing will be measured.\n");
  }

  BenchSinkhorn(ctx, warmup, iters, device_only);
  BenchMhcPost(ctx, warmup, iters, device_only);
  BenchIndexer(ctx, warmup, iters, device_only);
  BenchMxfp4Gemv(ctx, warmup, iters, device_only);
  BenchPagedCompressor(ctx, warmup, iters);
  ReportScaling(ctx);

  std::printf("\nDone.\n");
  return 0;
}
