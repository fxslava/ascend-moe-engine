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
// six arch35 operators plus the two whole-MoE-block entry points are wired,
// and the stream is synchronized inside the timed region (see TimedLaunch --
// a chevron dispatch is asynchronous, so timing the launch call alone
// measures enqueue cost).
//
// dsv4_moe_expert is the one operator here that issues TWO chevron dispatches
// per call, so its row is doubly sync-inclusive and carries two tiling
// allocations; BenchDsv4ExpertDevice says why that is structural.
//
// TWO TIMING PATHS, kept apart on purpose. Most rows use Measure(), a
// steady_clock bracket around launch-plus-synchronize. The two whole-MoE-block
// rows (dsv4_moe_decode_topk, dsv4_moe_prefill_batched) instead use
// TimedLaunchEvents(), which brackets the launch with ACL_EVENT_TIME_LINE
// events on the stream and subtracts them on the device clock. That is a
// tighter window -- the tiling aclrtMalloc and the host call overhead fall
// outside it -- but it is still NOT an isolated kernel time, because the v1
// wrappers synchronize and free inside it (CLAUDE.md section 3.7).
// TimedLaunchEvents says exactly what is in and out of the window. The two
// paths are not comparable to each other and are never mixed inside one row.
//
// The block rows also report p50 and p99 rather than mean and min, because
// decode is a latency-critical single-token step where the tail is the
// figure of interest. P99 is nearest-rank: below 100 timed iterations it IS
// the slowest sample, so a real tail wants `iters` of 100 or more -- the
// default is 20.
//
// NEITHER BLOCK KERNEL HAS EVER BEEN TIMED ON HARDWARE. Both passed
// functional verification on a 950PR; these rows exist so the timing claim
// can be made at all (CLAUDE.md section 5.7).
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
  // Nearest-rank P99: sorted[ceil(0.99*n) - 1]. With fewer than 100 samples
  // that index IS the maximum, which is the honest answer rather than an
  // interpolation dressed up as a tail estimate -- at the default
  // iters = 20 this column is the slowest of 20, nothing more. A real P99
  // wants `iters` at 100 or above.
  double p99_us = 0.0;
  int iterations = 0;
};

// Mean, stddev and order statistics from a sample vector. Shared so the
// steady_clock path, the event-timed path and the pre-collected-sample path
// (Summarize, below) cannot drift in how they compute the same columns.
Timing StatsOf(std::vector<double> samples) {
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
  const std::size_t rank = static_cast<std::size_t>(
      std::ceil(0.99 * static_cast<double>(samples.size())));
  t.p99_us = samples[(rank == 0 ? 1 : rank) - 1];
  return t;
}

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

  Timing t = StatsOf(std::move(samples));
  // Preserved from before StatsOf existed: a zero-sample run still reports
  // the iteration count it was asked for rather than 0.
  t.iterations = iters;
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
// Reporting: the whole-MoE-block rows
// ---------------------------------------------------------------------------
// Two extra tables rather than extra columns on Header/Row, for two reasons.
// The mean/sd/min table above is what every other section prints and widening
// it would churn six sections to serve two. And these two rows are read for a
// different question: decode is a latency-critical single-token step, so the
// columns that matter are p50 and the tail, not the mean -- a mean over a
// run that contains one descheduled iteration reports a latency the operator
// never had.
void HeaderBlockLatency(const char* kernel) {
  std::printf("\n=== %s ===\n", kernel);
  std::printf("%-30s %10s %10s %10s %9s %11s %11s\n", "geometry", "p50 us",
              "p99 us", "mean us", "sd us", "GB/s", "GFLOP/s");
  std::printf("%-30s %10s %10s %10s %9s %11s %11s\n",
              "------------------------------", "----------", "----------",
              "----------", "---------", "-----------", "-----------");
}

// p50 drives the derived rates, not the mean: the bandwidth and FLOP columns
// are meant to describe a typical iteration, and the mean is the one
// statistic here a single stall can move arbitrarily far.
//
// [[maybe_unused]]: every call site is a DEVICE bench, so all of them sit
// inside the MOE_KERNEL_LAUNCH_LAYER_AVAILABLE region below. In a host-emu
// build this definition is orphaned and -Wunused-function would make the TU an
// error. Kept out here rather than moved inside the guard so the row and
// header printers stay adjacent -- they are one output format.
[[maybe_unused]] void RowBlockLatency(const std::string& geometry, const Timing& t,
                     double bytes, double flops) {
  const double secs = t.p50_us * 1e-6;
  const double gbs = secs > 0.0 ? bytes / secs / 1e9 : 0.0;
  const double gflops = secs > 0.0 ? flops / secs / 1e9 : 0.0;
  std::printf("%-30s %10.2f %10.2f %10.2f %9.2f %11.2f %11.2f\n",
              geometry.c_str(), t.p50_us, t.p99_us, t.mean_us, t.stddev_us,
              gbs, gflops);
}

void HeaderBlockThroughput(const char* kernel) {
  std::printf("\n=== %s ===\n", kernel);
  std::printf("%-18s %10s %10s %9s %13s %10s %9s %10s\n", "geometry", "p50 us",
              "p99 us", "sd us", "tokens/s", "GB/s", "FLOP/B", "TFLOP/s");
  std::printf("%-18s %10s %10s %9s %13s %10s %9s %10s\n", "------------------",
              "----------", "----------", "---------", "-------------",
              "----------", "---------", "----------");
}

// `bytes` is COMPULSORY traffic -- each distinct operand moved once -- so
// FLOP/B here is the intensity of the problem, not of the execution. The
// prefill section says below why the executed traffic can be a multiple of
// it and why that multiple is not knowable from the host.
//
// [[maybe_unused]] for the same reason as RowBlockLatency above.
[[maybe_unused]] void RowBlockThroughput(const std::string& geometry, const Timing& t,
                        double tokens, double bytes, double flops) {
  const double secs = t.p50_us * 1e-6;
  const double toks = secs > 0.0 ? tokens / secs : 0.0;
  const double gbs = secs > 0.0 ? bytes / secs / 1e9 : 0.0;
  const double tflops = secs > 0.0 ? flops / secs / 1e12 : 0.0;
  const double intensity = bytes > 0.0 ? flops / bytes : 0.0;
  std::printf("%-18s %10.2f %10.2f %9.2f %13.1f %10.2f %9.3f %10.3f\n",
              geometry.c_str(), t.p50_us, t.p99_us, t.stddev_us, toks, gbs,
              intensity, tflops);
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
// device: dsv4_moe_expert
//
// One routed expert for one token, fused: gate GEMV, up GEMV, the asymmetric
// clamped SwiGLU, down GEMV. 32 AIVs, grid-stride over 32-row tiles.
//
// Three things make this row different from the four around it, and all three
// are properties of the operator rather than of the harness:
//
//  * TWO chevron dispatches per call. The down leg reduces over the whole
//    `activated` vector while the gate/up legs are split across 32 cores by
//    row tile, so the phases need a barrier and stream ordering is it
//    (launch.hpp, dsv4_moe_expert). Both dispatches are inside the timed
//    region because together they are one expert evaluation; the figure is
//    not divisible into per-leg times from here.
//  * TWO tiling staging allocations per call, not one, for the same reason.
//    Per CLAUDE.md section 3.7 the v1 wrappers also synchronize before the
//    free, so this row is sync-inclusive twice over -- a host-observed round
//    trip, not a kernel time, and the cost is structural until a
//    caller-owned-tiling variant exists.
//  * `workspace` is a real inter-stage operand here, not scratch: the gate_up
//    pass writes `activated` into it and the down pass reads it back. It is
//    allocated separately from x and y precisely because it must not alias
//    them.
//
// v1 geometry is pinned by MakeDsv4MoeExpertTiling to hidden = 4096,
// inter = 2048 -- any other shape returns unsupported_shape -- so this is one
// row, not a sweep. The three weight matrices are the same size
// (2048x4096 twice, 4096x2048 once), which is why the traffic below is
// 3 x (packed + scales): ~13.4 MB read once per token against ~32 KB of
// activation, so this is bandwidth-bound by a wide margin.
// ---------------------------------------------------------------------------
void BenchDsv4ExpertDevice(const DeviceCtx& ctx, int warmup, int iters) {
  using namespace moe::kernels::arch35;

  constexpr std::uint32_t kHidden = 4096;
  constexpr std::uint32_t kInter = 2048;
  // Spelled locally rather than pulled from op_host/dsv4_moe_expert_tiling.h:
  // that header is operator-private (CLAUDE.md section 1.2) and nothing
  // outside the operator may depend on it. 32 is the MXFP4 block size, the
  // same constant the host section above reads as moe::emu::kMxfp4GroupSize;
  // if the two ever disagree the tiling builder rejects the row strides, so
  // the mismatch surfaces as unsupported_shape rather than a wrong answer.
  constexpr std::uint32_t kGroup = 32;

  // w1/w3 are [inter, hidden]; w2 is [hidden, inter]. Equal element counts,
  // so equal payloads -- but the row strides differ between the two, and the
  // tiling builder rejects a mismatch, so they are kept named apart.
  constexpr std::size_t kW13Packed =
      static_cast<std::size_t>(kInter) * (kHidden / 2);
  constexpr std::size_t kW13Scale =
      static_cast<std::size_t>(kInter) * (kHidden / kGroup);
  constexpr std::size_t kW2Packed =
      static_cast<std::size_t>(kHidden) * (kInter / 2);
  constexpr std::size_t kW2Scale =
      static_cast<std::size_t>(kHidden) * (kInter / kGroup);

  Rng rng(0xD54EULL);
  const std::vector<std::uint8_t> w1_packed = rng.bytes(kW13Packed);
  // E8M0 near the bias (127) so the decode cannot overflow to inf and turn
  // the measurement into a run over NaNs.
  const std::vector<std::uint8_t> w1_scale = rng.bytes(kW13Scale, 120, 134);
  const std::vector<std::uint8_t> w3_packed = rng.bytes(kW13Packed);
  const std::vector<std::uint8_t> w3_scale = rng.bytes(kW13Scale, 120, 134);
  const std::vector<std::uint8_t> w2_packed = rng.bytes(kW2Packed);
  const std::vector<std::uint8_t> w2_scale = rng.bytes(kW2Scale, 120, 134);
  const std::vector<float> x = rng.normal(kHidden);

  const std::size_t workspace_bytes = Dsv4MoeExpertWorkspaceBytes(kInter);

  DeviceBuffer d_w1p, d_w1s, d_w3p, d_w3s, d_w2p, d_w2s, d_x, d_y, d_ws;
  aclError rc = d_w1p.Alloc(kW13Packed);
  if (rc == ACL_SUCCESS) rc = d_w1s.Alloc(kW13Scale);
  if (rc == ACL_SUCCESS) rc = d_w3p.Alloc(kW13Packed);
  if (rc == ACL_SUCCESS) rc = d_w3s.Alloc(kW13Scale);
  if (rc == ACL_SUCCESS) rc = d_w2p.Alloc(kW2Packed);
  if (rc == ACL_SUCCESS) rc = d_w2s.Alloc(kW2Scale);
  if (rc == ACL_SUCCESS) rc = d_x.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_ws.Alloc(workspace_bytes);
  if (rc == ACL_SUCCESS) rc = d_w1p.Upload(w1_packed.data(), kW13Packed);
  if (rc == ACL_SUCCESS) rc = d_w1s.Upload(w1_scale.data(), kW13Scale);
  if (rc == ACL_SUCCESS) rc = d_w3p.Upload(w3_packed.data(), kW13Packed);
  if (rc == ACL_SUCCESS) rc = d_w3s.Upload(w3_scale.data(), kW13Scale);
  if (rc == ACL_SUCCESS) rc = d_w2p.Upload(w2_packed.data(), kW2Packed);
  if (rc == ACL_SUCCESS) rc = d_w2s.Upload(w2_scale.data(), kW2Scale);
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  if (rc == ACL_SUCCESS) rc = d_ws.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] setup failed: %s\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  Timing t;
  std::string why;
  const bool ok = TimedLaunch(
      [&] {
        return LaunchDsv4MoeExpert(
            d_w1p.as<Mxfp4Pair>(), d_w1s.as<E8M0>(), d_w3p.as<Mxfp4Pair>(),
            d_w3s.as<E8M0>(), d_w2p.as<Mxfp4Pair>(), d_w2s.as<E8M0>(),
            kHidden / 2, kHidden / kGroup, kInter / 2, kInter / kGroup,
            d_x.as<float>(), kHidden, kInter, d_y.as<float>(), d_ws.get(),
            workspace_bytes, ctx.stream);
      },
      ctx.stream, warmup, iters, &t, &why);
  if (!ok) {
    std::printf("  [device] h=%u i=%u: FAILED -- %s\n", kHidden, kInter,
                why.c_str());
    return;
  }

  // All three weight matrices and their scale grids, read once, plus x in and
  // y out. `activated` is counted both ways: the gate_up pass writes it to GM
  // and the down pass reads it back, and that round trip is real traffic the
  // two-dispatch structure forces.
  const double bytes =
      static_cast<double>(kW13Packed + kW13Scale) * 2.0 +
      static_cast<double>(kW2Packed + kW2Scale) +
      static_cast<double>(kHidden) * 4.0 * 2.0 +
      static_cast<double>(workspace_bytes) * 2.0;
  // Three GEMVs of inter*hidden multiply-adds each. The SwiGLU itself is
  // O(inter) transcendental work and is deliberately NOT counted here: it
  // would be noise against 50 MFLOP and counting a sigmoid as one flop would
  // be arbitrary.
  const double flops = 3.0 * 2.0 * static_cast<double>(kInter) *
                       static_cast<double>(kHidden);
  char geo[64];
  std::snprintf(geo, sizeof(geo), "[device] h=%u i=%u", kHidden, kInter);
  Row(geo, t, bytes, flops);
}

// ---------------------------------------------------------------------------
// Device-timeline timing, for the whole-MoE-block rows
// ---------------------------------------------------------------------------
// WHAT THE EVENT WINDOW DOES AND DOES NOT CONTAIN. "Hardware event timer"
// invites reading the number as an isolated kernel time, and it is not one.
// The window is bracketed by two aclrtRecordEvent calls on `stream` around
// the Launch* call, so it spans the wrapper's tiling H2D and the kernel --
// but the v1 wrappers synchronize the stream and free the staging buffer
// before returning (CLAUDE.md section 3.7), so the trailing aclrtFree and the
// return into this loop land inside the window too. It is tighter than the
// steady_clock rows above, which additionally carry the tiling aclrtMalloc
// and the host-side call overhead, and it is still a host-observed round trip
// with a kernel inside it. Isolating the kernel needs a caller-owned-tiling
// wrapper variant, which does not exist (section 3.7).
//
// `prologue` runs BEFORE the start record. That is where the decode path's
// y zeroing goes: zeroing the accumulator is a caller obligation of the
// atomic-add summation contract (section 3.4), not work the kernel does, so
// timing it would charge the operator for its own ABI.
//
// STATUS ON HARDWARE (section 5.7). This loop HAS now been run on a 950PR
// and its first version failed there -- see the aclrtResetEvent note at the
// bottom of the loop. No timing has been collected from it yet: the rows it
// drives have been executed but never successfully measured.
class EventPair {
 public:
  EventPair() = default;
  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;
  ~EventPair() {
    if (stop_ != nullptr) (void)aclrtDestroyEvent(stop_);
    if (start_ != nullptr) (void)aclrtDestroyEvent(start_);
  }

  // ACL_EVENT_TIME_LINE is what makes the event carry a timestamp; an event
  // created without it can be waited on but not subtracted.
  aclError Create() {
    aclError rc = aclrtCreateEventWithFlag(&start_, ACL_EVENT_TIME_LINE);
    if (rc == ACL_SUCCESS) {
      rc = aclrtCreateEventWithFlag(&stop_, ACL_EVENT_TIME_LINE);
    }
    return rc;
  }

  aclrtEvent start() const noexcept { return start_; }
  aclrtEvent stop() const noexcept { return stop_; }

 private:
  aclrtEvent start_ = nullptr;
  aclrtEvent stop_ = nullptr;
};

bool TimedLaunchEvents(const std::function<moe::kernels::arch35::Status()>& launch,
                       const std::function<aclError()>& prologue,
                       aclrtStream stream, int warmup, int iters, Timing* out,
                       std::string* why) {
  using moe::kernels::arch35::Status;

  EventPair ev;
  aclError rc = ev.Create();
  if (rc != ACL_SUCCESS) {
    *why = AclFail("aclrtCreateEventWithFlag(ACL_EVENT_TIME_LINE)", rc);
    return false;
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters < 0 ? 0 : iters));

  const int total = (warmup < 0 ? 0 : warmup) + (iters < 1 ? 1 : iters);
  for (int i = 0; i < total; ++i) {
    if (prologue) {
      rc = prologue();
      if (rc != ACL_SUCCESS) {
        *why = AclFail("prologue (accumulator zeroing)", rc);
        return false;
      }
      // The prologue is enqueued on `stream`, so it must retire before the
      // start record or its cost lands inside the window.
      rc = aclrtSynchronizeStream(stream);
      if (rc != ACL_SUCCESS) {
        *why = AclFail("aclrtSynchronizeStream(prologue)", rc);
        return false;
      }
    }

    rc = aclrtRecordEvent(ev.start(), stream);
    if (rc != ACL_SUCCESS) {
      *why = AclFail("aclrtRecordEvent(start)", rc);
      return false;
    }

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
      return false;
    }

    rc = aclrtRecordEvent(ev.stop(), stream);
    if (rc == ACL_SUCCESS) rc = aclrtSynchronizeStream(stream);
    if (rc != ACL_SUCCESS) {
      *why = AclFail("aclrtRecordEvent(stop)/aclrtSynchronizeStream", rc);
      return false;
    }

    float ms = 0.0F;
    rc = aclrtEventElapsedTime(&ms, ev.start(), ev.stop());
    if (rc != ACL_SUCCESS) {
      *why = AclFail("aclrtEventElapsedTime", rc);
      return false;
    }
    if (i >= (warmup < 0 ? 0 : warmup)) {
      samples.push_back(static_cast<double>(ms) * 1000.0);
    }

    // NO aclrtResetEvent HERE, and this is the one place in the loop where
    // the obvious-looking call is wrong. An earlier version reset both
    // events at the end of each iteration, reading aclrtResetEvent's "wait
    // for the stream's tasks to complete before resetting" as an
    // instruction to reset once they had. On hardware that returned
    // aclError=207000, ACL_ERROR_RT_EVENT_STATE_INVALID, on the first
    // iteration of every block row -- so both rows failed in the harness
    // while the kernels themselves were passing 18/18 in the device suite.
    //
    // aclrtRecordEvent already re-arms the event: recording an event that
    // carries a previous timestamp overwrites it, which is what makes the
    // create-once/record-per-iteration loop the standard shape. Reset is for
    // a recorded event that will NOT be recorded again and must be returned
    // to the unrecorded state for a waiter; applying it to an event this
    // loop is about to record again is the state error the runtime reports.
  }

  *out = StatsOf(std::move(samples));
  return true;
}

// ---------------------------------------------------------------------------
// The whole-MoE-block geometry, shared by the decode and prefill sections
// ---------------------------------------------------------------------------
// Spelled locally rather than pulled from op_host/dsv4_moe_block_tiling.h:
// that header is operator-private (CLAUDE.md section 1.2). The strides below
// are the ones MakeDsv4Moe*Tiling derives and validates; if they ever
// disagree the builder returns unsupported_shape, so a drift surfaces as a
// failed row rather than as a wrong number.
struct BlockGeometry {
  static constexpr std::uint32_t kHidden = 4096;
  static constexpr std::uint32_t kInter = 2048;
  static constexpr std::uint32_t kGroup = 32;  // MXFP4 block size

  static constexpr std::size_t kW13PackedExpert =
      static_cast<std::size_t>(kInter) * (kHidden / 2);
  static constexpr std::size_t kW13ScaleExpert =
      static_cast<std::size_t>(kInter) * (kHidden / kGroup);
  static constexpr std::size_t kW2PackedExpert =
      static_cast<std::size_t>(kHidden) * (kInter / 2);
  static constexpr std::size_t kW2ScaleExpert =
      static_cast<std::size_t>(kHidden) * (kInter / kGroup);

  // One expert's whole weight stream: 3 x 4 MiB packed + 3 x 256 KiB scales
  // = 13,369,344 B = 12.75 MiB. This is the number the decode row's
  // bandwidth is built from.
  static constexpr std::size_t kBytesPerExpert =
      2 * (kW13PackedExpert + kW13ScaleExpert) +
      (kW2PackedExpert + kW2ScaleExpert);

  // Three GEMVs of inter*hidden multiply-adds per token-expert evaluation.
  // The SwiGLU is deliberately NOT counted, for the same reason as the
  // single-expert row above: it is O(inter) transcendental work against
  // 50 MFLOP, and pricing a sigmoid at one flop would be arbitrary.
  static constexpr double kFlopsPerTokenExpert =
      3.0 * 2.0 * static_cast<double>(kInter) * static_cast<double>(kHidden);
};

// The weight buffers for a pool of `num_experts`, with `fill_experts` of them
// carrying real data at their strided offsets.
//
// WHY NOT UPLOAD THE WHOLE POOL. At num_experts = 64 the pool is 816 MiB and
// the host-side random fill alone would dominate the run. Only the routed
// experts are ever read, so the rest is left zeroed: an all-zero MXFP4 row
// decodes to zero, which is finite and harmless, and the kernel never
// touches it anyway. This is a benchmark -- a wrong number in an unread
// expert cannot become a wrong answer (CLAUDE.md section 4.6).
struct ExpertPool {
  DeviceBuffer w1p, w1s, w3p, w3s, w2p, w2s;

  aclError Stage(std::uint32_t num_experts,
                 const std::vector<std::int32_t>& fill_experts,
                 std::uint64_t seed) {
    using G = BlockGeometry;
    const std::size_t n = num_experts;
    aclError rc = w1p.Alloc(n * G::kW13PackedExpert);
    if (rc == ACL_SUCCESS) rc = w3p.Alloc(n * G::kW13PackedExpert);
    if (rc == ACL_SUCCESS) rc = w2p.Alloc(n * G::kW2PackedExpert);
    if (rc == ACL_SUCCESS) rc = w1s.Alloc(n * G::kW13ScaleExpert);
    if (rc == ACL_SUCCESS) rc = w3s.Alloc(n * G::kW13ScaleExpert);
    if (rc == ACL_SUCCESS) rc = w2s.Alloc(n * G::kW2ScaleExpert);
    if (rc != ACL_SUCCESS) return rc;

    if (rc == ACL_SUCCESS) rc = w1p.Zero();
    if (rc == ACL_SUCCESS) rc = w3p.Zero();
    if (rc == ACL_SUCCESS) rc = w2p.Zero();
    if (rc == ACL_SUCCESS) rc = w1s.Zero();
    if (rc == ACL_SUCCESS) rc = w3s.Zero();
    if (rc == ACL_SUCCESS) rc = w2s.Zero();
    if (rc != ACL_SUCCESS) return rc;

    Rng rng(seed);
    // E8M0 near the bias (127) so the decode cannot overflow to inf and turn
    // the measurement into a run over NaNs -- the same band the single-expert
    // row uses.
    const std::vector<std::uint8_t> p13 = rng.bytes(G::kW13PackedExpert);
    const std::vector<std::uint8_t> p13b = rng.bytes(G::kW13PackedExpert);
    const std::vector<std::uint8_t> p2 = rng.bytes(G::kW2PackedExpert);
    const std::vector<std::uint8_t> s13 =
        rng.bytes(G::kW13ScaleExpert, 120, 134);
    const std::vector<std::uint8_t> s13b =
        rng.bytes(G::kW13ScaleExpert, 120, 134);
    const std::vector<std::uint8_t> s2 = rng.bytes(G::kW2ScaleExpert, 120, 134);

    for (const std::int32_t e : fill_experts) {
      const std::size_t i = static_cast<std::size_t>(e);
      rc = UploadAt(w1p, i * G::kW13PackedExpert, p13);
      if (rc == ACL_SUCCESS) rc = UploadAt(w3p, i * G::kW13PackedExpert, p13b);
      if (rc == ACL_SUCCESS) rc = UploadAt(w2p, i * G::kW2PackedExpert, p2);
      if (rc == ACL_SUCCESS) rc = UploadAt(w1s, i * G::kW13ScaleExpert, s13);
      if (rc == ACL_SUCCESS) rc = UploadAt(w3s, i * G::kW13ScaleExpert, s13b);
      if (rc == ACL_SUCCESS) rc = UploadAt(w2s, i * G::kW2ScaleExpert, s2);
      if (rc != ACL_SUCCESS) return rc;
    }
    return ACL_SUCCESS;
  }

 private:
  // DeviceBuffer::Upload only writes at the base, and the pool layout needs
  // an offset, so this is the one place a raw aclrtMemcpy is warranted.
  static aclError UploadAt(const DeviceBuffer& dst, std::size_t offset,
                           const std::vector<std::uint8_t>& src) {
    return aclrtMemcpy(static_cast<std::uint8_t*>(dst.get()) + offset,
                       dst.bytes() - offset, src.data(), src.size(),
                       ACL_MEMCPY_HOST_TO_DEVICE);
  }
};

// ---------------------------------------------------------------------------
// device: dsv4_moe_decode_topk
//
// One token, top_k routed experts, block_dim == top_k: one AIV owns a whole
// expert end to end, so `activated` never reaches GM. This is the row the
// design note in launch.hpp explicitly says has never been measured -- the
// decision to take parallelism from the experts rather than the row tiles
// leaves 26 of 32 AIVs idle at top_k = 6, bought in exchange for keeping the
// w1/w3 weight stream at 1x on a kernel measured at 3.76 FLOP/B.
//
// TRAFFIC. Each of the top_k cores reads its own expert's 12.75 MiB exactly
// once, so the compulsory figure IS the executed figure here -- no two cores
// share an expert in a top_k set of distinct ids, so there is no reuse to
// over- or under-count. Plus x read by every core (top_k x 16 KiB; it is 16
// KiB of the 76.5 MiB stream) and y written once per slot by the atomic add.
// At top_k = 6 that is 6 x 12.75 MiB = 76.5 MiB against 302 MFLOP, i.e.
// ~3.76 FLOP/B: bandwidth-bound by a wide margin, so GB/s is the column that
// says whether this kernel is doing well.
//
// THE ACCUMULATOR MUST BE ZEROED BEFORE EVERY TIMED LAUNCH. Decode sums the
// top_k slots into y with an atomic-add store and the wrapper does not touch
// operand data (CLAUDE.md section 3.4), so a benchmark loop that zeroed once
// and then ran 100 iterations would be measuring a kernel accumulating into
// a growing y -- arithmetically garbage, and on this geometry it would reach
// inf within a few dozen iterations. The zeroing is the `prologue` above and
// sits outside the event window.
//
// Returns the p50 in `out_p50_us` so the caller can form the speedup against
// the legacy sequential path; false means the row failed and said why.
bool BenchDecodePool(const DeviceCtx& ctx, int warmup, int iters,
                     std::uint32_t num_experts,
                     const std::vector<std::int32_t>& topk_ids,
                     const std::vector<float>& routing_weights,
                     double* out_p50_us) {
  using namespace moe::kernels::arch35;
  using G = BlockGeometry;

  const std::uint32_t top_k = static_cast<std::uint32_t>(topk_ids.size());

  ExpertPool pool;
  aclError rc = pool.Stage(num_experts, topk_ids, 0xD54E0DEULL);
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] experts=%u: pool staging failed (%s)\n",
                num_experts, AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return false;
  }

  Rng rng(0xD54E01ULL);
  const std::vector<float> x = rng.normal(G::kHidden);

  DeviceBuffer d_x, d_ids, d_rw, d_y, d_actws;
  rc = d_x.Alloc(G::kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_ids.Alloc(top_k * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_rw.Alloc(top_k * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(G::kHidden * sizeof(float));
  // The decode gang's `activated` plane, [top_k, inter] FP32. Allocated ONCE
  // here and reused across every timed iteration -- which is exactly why it
  // is an operand rather than a per-launch allocation: a malloc inside the
  // timed window would measure the device allocator, not the kernel.
  if (rc == ACL_SUCCESS)
    rc = d_actws.Alloc(static_cast<std::size_t>(
        Dsv4MoeDecodeWorkspaceBytes(top_k, G::kInter)));
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), G::kHidden * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_ids.Upload(topk_ids.data(), top_k * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS)
    rc = d_rw.Upload(routing_weights.data(), top_k * sizeof(float));
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] experts=%u: setup failed (%s)\n", num_experts,
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return false;
  }

  Timing t;
  std::string why;
  const bool ok = TimedLaunchEvents(
      [&] {
        return LaunchDsv4MoeDecodeTopk(
            d_x.as<float>(), pool.w1p.as<Mxfp4Pair>(), pool.w1s.as<E8M0>(),
            pool.w3p.as<Mxfp4Pair>(), pool.w3s.as<E8M0>(),
            pool.w2p.as<Mxfp4Pair>(), pool.w2s.as<E8M0>(),
            d_ids.as<std::int32_t>(), d_rw.as<float>(), G::kHidden, G::kInter,
            num_experts, top_k, d_y.as<float>(), d_actws.as<float>(),
            ctx.stream);
      },
      // The accumulator zeroing, outside the window. Async on `stream` so it
      // is ordered against the launch rather than against the host.
      [&] {
        return aclrtMemsetAsync(d_y.get(), d_y.bytes(), 0, d_y.bytes(),
                                ctx.stream);
      },
      ctx.stream, warmup, iters, &t, &why);
  if (!ok) {
    std::printf("  [device] experts=%u top_k=%u: FAILED -- %s\n", num_experts,
                top_k, why.c_str());
    return false;
  }

  const double bytes =
      static_cast<double>(top_k) * static_cast<double>(G::kBytesPerExpert) +
      static_cast<double>(top_k) * static_cast<double>(G::kHidden) * 4.0 +
      static_cast<double>(top_k) * static_cast<double>(G::kHidden) * 4.0;
  const double flops =
      static_cast<double>(top_k) * G::kFlopsPerTokenExpert;

  char geo[64];
  std::snprintf(geo, sizeof(geo), "[device] pool=%u top_k=%u", num_experts,
                top_k);
  RowBlockLatency(geo, t, bytes, flops);
  if (out_p50_us != nullptr) *out_p50_us = t.p50_us;
  return true;
}

void BenchDsv4MoeDecodeTopkDevice(const DeviceCtx& ctx, int warmup,
                                  int iters) {
  using namespace moe::kernels::arch35;
  using G = BlockGeometry;

  // Neither sorted nor the identity, so a row measuring a kernel that used
  // the SLOT index instead of topk_expert_ids[slot] would at least be
  // measuring the same addressing the correctness suite exercises.
  const std::vector<std::int32_t> ids_tight = {4, 0, 5, 2, 1, 3};
  const std::vector<float> routing_weights = {0.35F, 0.25F, 0.15F,
                                              0.10F, 0.08F, 0.07F};

  double p50_tight = 0.0;
  const bool have_tight = BenchDecodePool(ctx, warmup, iters, 6, ids_tight,
                                          routing_weights, &p50_tight);

  // A SECOND POOL SIZE, because the first one cannot answer the question it
  // looks like it answers. At num_experts = 6 the routed set IS the whole
  // resident pool and the six weight streams are contiguous -- which
  // launch.hpp blesses as a valid staging ("passing ids 0..top_k-1 is as
  // valid as keeping all 256 resident"), but which is also the most
  // favourable address pattern this kernel will ever see. A production
  // decode step picks 6 of a resident pool, so the six streams are scattered
  // across hundreds of MiB. The traffic is identical; the page and TLB
  // behaviour is not, and that difference is the whole reason to run this
  // row.
  //
  // 64 rather than 256: the pool is 64 x 12.75 MiB = 816 MiB, and 256 would
  // be 3.26 GiB of HBM for a benchmark. If even 816 MiB does not allocate the
  // row says so and the section continues -- a benchmark is never a gate
  // (CLAUDE.md section 4.6).
  const std::vector<std::int32_t> ids_sparse = {40, 0, 55, 23, 11, 37};
  (void)BenchDecodePool(ctx, warmup, iters, 64, ids_sparse, routing_weights,
                        nullptr);

  // ---- the legacy sequential baseline --------------------------------
  //
  // What the decode path replaces: top_k separate LaunchDsv4MoeExpert calls,
  // each of which is TWO chevron dispatches with a GM round trip for
  // `activated` between them, two tiling staging allocations and two
  // wrapper-internal stream synchronizations (CLAUDE.md sections 3.7 and the
  // dsv4_moe_expert note above).
  //
  // HOW TO READ THE SPEEDUP, because it flatters the new kernel in one way
  // and penalises it in another, and neither is separable from here:
  //   * it counts top_k x the legacy per-call HOST overhead (2 mallocs, 2
  //     syncs, 2 frees), which a batched legacy caller would pay once per
  //     expert but could not avoid -- the wrappers own their staging;
  //   * against that, the legacy row below runs ONE expert's weight stream
  //     and is multiplied by top_k, which assumes the six calls neither
  //     contend nor share cache. On a weight-stream-bound kernel they would
  //     contend, so the true sequential wall clock is likely WORSE than
  //     top_k x this figure.
  // It is a host-observed round-trip ratio, not a kernel-time ratio.
  if (!have_tight) return;

  ExpertPool one;
  aclError rc = one.Stage(1, {0}, 0xD54E0DEULL);
  const std::size_t workspace_bytes = Dsv4MoeExpertWorkspaceBytes(G::kInter);

  Rng rng(0xD54E01ULL);
  const std::vector<float> x = rng.normal(G::kHidden);

  DeviceBuffer d_x, d_y, d_ws;
  if (rc == ACL_SUCCESS) rc = d_x.Alloc(G::kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(G::kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_ws.Alloc(workspace_bytes);
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), G::kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  if (rc == ACL_SUCCESS) rc = d_ws.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] legacy baseline: setup failed (%s)\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  Timing t;
  std::string why;
  const bool ok = TimedLaunchEvents(
      [&] {
        return LaunchDsv4MoeExpert(
            one.w1p.as<Mxfp4Pair>(), one.w1s.as<E8M0>(),
            one.w3p.as<Mxfp4Pair>(), one.w3s.as<E8M0>(),
            one.w2p.as<Mxfp4Pair>(), one.w2s.as<E8M0>(), G::kHidden / 2,
            G::kHidden / G::kGroup, G::kInter / 2, G::kInter / G::kGroup,
            d_x.as<float>(), G::kHidden, G::kInter, d_y.as<float>(),
            d_ws.get(), workspace_bytes, ctx.stream);
      },
      // dsv4_moe_expert OVERWRITES y, so there is no accumulator to zero.
      nullptr, ctx.stream, warmup, iters, &t, &why);
  if (!ok) {
    std::printf("  [device] legacy baseline: FAILED -- %s\n", why.c_str());
    return;
  }

  const std::uint32_t top_k = static_cast<std::uint32_t>(ids_tight.size());
  const double legacy_p50 = t.p50_us * static_cast<double>(top_k);
  char geo[64];
  std::snprintf(geo, sizeof(geo), "[device] legacy 1 expert");
  RowBlockLatency(geo, t,
                  static_cast<double>(G::kBytesPerExpert) +
                      static_cast<double>(G::kHidden) * 4.0 * 2.0 +
                      static_cast<double>(workspace_bytes) * 2.0,
                  G::kFlopsPerTokenExpert);
  std::printf("  [device] legacy sequential baseline: %u x %.2f us = %.2f us "
              "p50; fused decode p50 %.2f us -> %.2fx\n",
              top_k, t.p50_us, legacy_p50, p50_tight,
              p50_tight > 0.0 ? legacy_p50 / p50_tight : 0.0);
  std::printf("           The ratio is host-observed round trips, not kernel "
              "times: it charges the legacy path\n"
              "           top_k x (2 dispatches + 2 tiling allocations + 2 "
              "syncs), and assumes six sequential\n"
              "           weight streams neither contend nor share cache. "
              "See the note above the baseline.\n");
}

// ---------------------------------------------------------------------------
// device: dsv4_moe_prefill_batched
//
// M >> 1, one expert per token, 32 AIVs grid-striding over tokens. Unlike
// decode this one saturates the grid at any M >= 32, and the question the
// sweep answers is where the per-token weight stream stops dominating.
//
// THE BANDWIDTH COLUMN IS COMPULSORY TRAFFIC AND THE EXECUTED TRAFFIC IS A
// MULTIPLE OF IT. This is the one number in this file most likely to be
// misread, so it is stated plainly rather than buried. `bytes` counts each
// distinct expert's 12.75 MiB once, plus x and y. The kernel grid-strides:
// core c takes tokens c, c+32, c+64, ... so two tokens a core processes
// consecutively are 32 apart in the batch. The caller is expected to hand in
// tokens already GROUPED by expert (launch.hpp), so with E distinct experts
// and M tokens a bucket spans ~M/E consecutive positions -- and a core sees
// the same expert twice in a row only when a bucket is wider than 32
// positions, i.e. only when M/E > 32. At M = 16 and 64 with six buckets
// every token a core touches is a different expert from the last, so the
// executed weight stream approaches (cores x experts touched) x 12.75 MiB,
// up to 32x the compulsory figure; at M = 256 the buckets are ~43 wide and
// some reuse appears.
//
// So GB/s here is "compulsory bytes per second" and a figure ABOVE the
// hardware's HBM bandwidth is not an error in the row -- it means the real
// stream is being served from cache. What the sweep shows honestly is the
// SHAPE: tokens/s should climb steeply from M = 16 to M = 256 if reuse is
// materialising and flatten if it is not. Neither has been observed on
// hardware (CLAUDE.md section 5.7).
void BenchDsv4MoePrefillBatchedDevice(const DeviceCtx& ctx, int warmup,
                                      int iters) {
  using namespace moe::kernels::arch35;
  using G = BlockGeometry;

  constexpr std::uint32_t kExperts = 6;  // a step's active expert set
  const std::uint32_t kBatches[] = {16, 64, 128, 256};
  const std::uint32_t kMaxTokens = 256;

  ExpertPool pool;
  aclError rc = pool.Stage(kExperts, {0, 1, 2, 3, 4, 5}, 0xD54E0DEULL);
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] pool staging failed (%s)\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  // One allocation at the largest M, reused by every row of the sweep: the
  // smaller rows pass a smaller total_tokens and the kernel reads a prefix.
  // Allocation is setup, not work, and must not sit between the rows.
  Rng rng(0xD54E02ULL);
  const std::vector<float> x_tokens =
      rng.normal(static_cast<std::size_t>(kMaxTokens) * G::kHidden);

  DeviceBuffer d_xt, d_yt, d_t2e;
  const std::size_t tok_bytes =
      static_cast<std::size_t>(kMaxTokens) * G::kHidden * sizeof(float);
  rc = d_xt.Alloc(tok_bytes);
  if (rc == ACL_SUCCESS) rc = d_yt.Alloc(tok_bytes);
  if (rc == ACL_SUCCESS) rc = d_t2e.Alloc(kMaxTokens * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_xt.Upload(x_tokens.data(), tok_bytes);
  // Prefill OVERWRITES y_tokens, so no zeroing is required between
  // iterations -- there is no accumulator and hence no prologue below. It is
  // zeroed once so a token the kernel never writes reads as 0 rather than as
  // whatever the allocator returned.
  if (rc == ACL_SUCCESS) rc = d_yt.Zero();
  if (rc != ACL_SUCCESS) {
    std::printf("  [device] setup failed (%s)\n",
                AclFail("aclrtMalloc/aclrtMemcpy", rc).c_str());
    return;
  }

  for (const std::uint32_t m : kBatches) {
    // Contiguous, near-equal buckets: token t goes to expert (t*E)/M, which
    // is the grouped-by-expert layout the contract expects from a dispatch.
    // Correctness does not depend on the grouping, only reuse does -- which
    // is exactly what this sweep is measuring.
    std::vector<std::int32_t> token_to_expert(m);
    for (std::uint32_t tok = 0; tok < m; ++tok) {
      token_to_expert[tok] = static_cast<std::int32_t>(
          (static_cast<std::uint64_t>(tok) * kExperts) / m);
    }
    rc = d_t2e.Upload(token_to_expert.data(), m * sizeof(std::int32_t));
    if (rc != ACL_SUCCESS) {
      std::printf("  [device] M=%u: token map upload failed (%s)\n", m,
                  AclFail("aclrtMemcpy", rc).c_str());
      continue;
    }

    Timing t;
    std::string why;
    const bool ok = TimedLaunchEvents(
        [&] {
          return LaunchDsv4MoePrefillBatched(
              d_xt.as<float>(), pool.w1p.as<Mxfp4Pair>(), pool.w1s.as<E8M0>(),
              pool.w3p.as<Mxfp4Pair>(), pool.w3s.as<E8M0>(),
              pool.w2p.as<Mxfp4Pair>(), pool.w2s.as<E8M0>(),
              d_t2e.as<std::int32_t>(), G::kHidden, G::kInter, kExperts, m,
              d_yt.as<float>(), ctx.stream);
        },
        nullptr, ctx.stream, warmup, iters, &t, &why);
    if (!ok) {
      std::printf("  [device] M=%u: FAILED -- %s\n", m, why.c_str());
      continue;
    }

    // Every one of the six buckets is non-empty at M = 16 (the smallest
    // bucket is 2 tokens), so all six experts are touched at every point of
    // the sweep and the compulsory weight term does not vary with M.
    const double bytes =
        static_cast<double>(kExperts) *
            static_cast<double>(G::kBytesPerExpert) +
        static_cast<double>(m) * static_cast<double>(G::kHidden) * 4.0 * 2.0;
    const double flops = static_cast<double>(m) * G::kFlopsPerTokenExpert;

    char geo[32];
    std::snprintf(geo, sizeof(geo), "[device] M=%u E=%u", m, kExperts);
    RowBlockThroughput(geo, t, static_cast<double>(m), bytes, flops);
  }

  std::printf("  [device] GB/s is COMPULSORY traffic (each distinct expert "
              "once); the grid-stride schedule\n"
              "           re-streams weights up to 32x that below M/E = 32, "
              "so read the tokens/s trend\n"
              "           across the sweep, not the absolute GB/s. See the "
              "note above this function.\n");
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
  return StatsOf(std::move(samples));
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
// either measures or prints why it did not. Every operator is now wired.
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

void DeviceSectionDsv4Expert(const DeviceCtx& ctx, int warmup, int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchDsv4ExpertDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("dsv4_moe_expert", ctx);
}

void DeviceSectionDsv4MoeDecodeTopk(const DeviceCtx& ctx, int warmup,
                                    int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchDsv4MoeDecodeTopkDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("dsv4_moe_decode_topk", ctx);
}

void DeviceSectionDsv4MoePrefillBatched(const DeviceCtx& ctx, int warmup,
                                        int iters) {
#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)
  if (ctx.usable) {
    BenchDsv4MoePrefillBatchedDevice(ctx, warmup, iters);
    return;
  }
#else
  (void)warmup;
  (void)iters;
#endif
  DeviceRowUnavailable("dsv4_moe_prefill_batched", ctx);
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
// dsv4_moe_expert
//
// The host roofline is the same fused expert the kernel implements -- three
// MXFP4 GEMVs with the asymmetric clamped SwiGLU between them -- so the two
// rows below describe the same arithmetic. They are still NOT comparable as
// speeds: the host figure is scalar single-threaded C++ over a dequantizing
// inner loop, and the device figure is 32 AIVs plus two sync-inclusive round
// trips. The host row is the baseline the kernel is beating, nothing more.
//
// One geometry only: MakeDsv4MoeExpertTiling pins v1 to hidden = 4096,
// inter = 2048 and returns unsupported_shape for anything else, so a sweep
// here would be measuring rejected launches.
// ---------------------------------------------------------------------------
void BenchDsv4Expert(const DeviceCtx& ctx, int warmup, int iters,
                     bool device_only) {
  Header("dsv4_moe_expert (host reference: fused MXFP4 expert + SwiGLU)");

  if (device_only) {
    HostSkipped();
    DeviceSectionDsv4Expert(ctx, warmup, iters);
    return;
  }

  constexpr int kGroup = moe::emu::kMxfp4GroupSize;
  constexpr int kHidden = 4096;
  constexpr int kInter = 2048;

  // w1/w3 project hidden -> inter; w2 projects inter -> hidden.
  constexpr std::size_t kW13Packed =
      static_cast<std::size_t>(kInter) * (kHidden / 2);
  constexpr std::size_t kW13Scale =
      static_cast<std::size_t>(kInter) * (kHidden / kGroup);
  constexpr std::size_t kW2Packed =
      static_cast<std::size_t>(kHidden) * (kInter / 2);
  constexpr std::size_t kW2Scale =
      static_cast<std::size_t>(kHidden) * (kInter / kGroup);

  Rng rng(0xD54EULL);
  const std::vector<std::uint8_t> w1_packed = rng.bytes(kW13Packed);
  // E8M0 near the bias so the decode cannot overflow to inf.
  const std::vector<std::uint8_t> w1_scale = rng.bytes(kW13Scale, 120, 134);
  const std::vector<std::uint8_t> w3_packed = rng.bytes(kW13Packed);
  const std::vector<std::uint8_t> w3_scale = rng.bytes(kW13Scale, 120, 134);
  const std::vector<std::uint8_t> w2_packed = rng.bytes(kW2Packed);
  const std::vector<std::uint8_t> w2_scale = rng.bytes(kW2Scale, 120, 134);
  const std::vector<float> x = rng.normal(static_cast<std::size_t>(kHidden));

  moe::emu::Mxfp4Matrix w1;
  w1.packed = w1_packed.data();
  w1.scale = w1_scale.data();
  w1.rows = kInter;
  w1.cols = kHidden;

  moe::emu::Mxfp4Matrix w3 = w1;
  w3.packed = w3_packed.data();
  w3.scale = w3_scale.data();

  moe::emu::Mxfp4Matrix w2;
  w2.packed = w2_packed.data();
  w2.scale = w2_scale.data();
  w2.rows = kHidden;
  w2.cols = kInter;

  std::vector<float> out(static_cast<std::size_t>(kHidden), 0.0F);
  // Hoisted so the reference reuses it across iterations: the allocation is
  // setup, not work, and must not land inside the timed region.
  std::vector<float> scratch;

  const Timing t = Measure(warmup, iters, [&] {
    moe::emu::moe_expert_swiglu_mxfp4_cpu(w1, w3, w2, x.data(), out.data(),
                                          moe::emu::kSwigluLimit, &scratch);
  });

  // Three GEMVs of inter*hidden multiply-adds; the SwiGLU is excluded for the
  // same reason as in the device row.
  const double flops =
      3.0 * 2.0 * static_cast<double>(kInter) * static_cast<double>(kHidden);
  const double bytes = static_cast<double>(kW13Packed + kW13Scale) * 2.0 +
                       static_cast<double>(kW2Packed + kW2Scale) +
                       static_cast<double>(kHidden) * 4.0 * 2.0 +
                       static_cast<double>(kInter) * 4.0 * 2.0;
  char geo[64];
  std::snprintf(geo, sizeof(geo), "h=%d i=%d", kHidden, kInter);
  Row(geo, t, bytes, flops);

  DeviceSectionDsv4Expert(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 6. dsv4_moe_decode_topk -- the whole MoE block, M = 1
//
// NO HOST ROOFLINE HERE, and the reason is arithmetic rather than a gap.
// The reference for this kernel is top_k independent evaluations of exactly
// the expert body the dsv4_moe_expert host row above already measures --
// moe_decode_topk_mxfp4_cpu loops moe_expert_swiglu_mxfp4_cpu over the
// routed slots and sums the results. So the host figure is 6 x a number
// already on screen, and measuring it costs ~6 x 19 ms x iters of scalar
// single-threaded wall clock: at the default counts that is ~2.6 s, roughly
// four times the entire current host-reference budget, to print a product
// the reader can form from the row above.
//
// The device rows are what this section exists for. Both the decode and the
// prefill entry points passed functional verification on hardware; neither
// has ever been TIMED on it (CLAUDE.md section 5.7).
// ---------------------------------------------------------------------------
void BenchDsv4MoeDecodeTopk(const DeviceCtx& ctx, int warmup, int iters) {
  HeaderBlockLatency("dsv4_moe_decode_topk (whole MoE block, M=1, top_k=6)");
  std::printf("  [host]   not benchmarked: the reference is top_k x the "
              "dsv4_moe_expert host row above\n           (same expert body, "
              "summed over the routed slots), so it is that figure x 6.\n");
  DeviceSectionDsv4MoeDecodeTopk(ctx, warmup, iters);
}

// ---------------------------------------------------------------------------
// 7. dsv4_moe_prefill_batched -- the whole MoE block, M >> 1
//
// Same reasoning as above and more so: the host reference is M evaluations
// of the expert body, which at M = 256 is ~5 s per iteration of scalar C++.
// The sweep is a device-side question anyway -- it is measuring whether the
// grid-stride schedule turns a grouped token batch into weight reuse, and a
// single-threaded host loop has no schedule to measure.
// ---------------------------------------------------------------------------
void BenchDsv4MoePrefillBatched(const DeviceCtx& ctx, int warmup, int iters) {
  HeaderBlockThroughput(
      "dsv4_moe_prefill_batched (whole MoE block, M sweep, 6 experts)");
  std::printf("  [host]   not benchmarked: the reference is M x the "
              "dsv4_moe_expert host row above, ~5 s per\n           iteration "
              "at M=256, and a scalar host loop has no grid schedule to "
              "measure.\n");
  DeviceSectionDsv4MoePrefillBatched(ctx, warmup, iters);
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
  BenchDsv4Expert(ctx, warmup, iters, device_only);
  BenchDsv4MoeDecodeTopk(ctx, warmup, iters);
  BenchDsv4MoePrefillBatched(ctx, warmup, iters);
  BenchPagedCompressor(ctx, warmup, iters);
  ReportScaling(ctx);

  std::printf("\nDone.\n");
  return 0;
}
