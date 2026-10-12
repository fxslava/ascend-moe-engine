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
// Device benchmarking needs the Launch* host wrappers, which are specified in
// docs/kernel_contracts_spec.md but not implemented in the tree. Until they
// land this binary measures the CPU reference instead, which is useful on its
// own: it is the roofline the kernel is beating, and it exercises the timing
// harness so the numbers are trustworthy the day the device path appears.
// Each device row is reported as unavailable rather than silently omitted.
//
// This binary never loads a host CPU debug runtime (cpudebug / tikicpulib).
// -----------------------------------------------------------------------------

#include "kernel_test_harness.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>

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

void DeviceRowUnavailable(const char* kernel) {
  std::printf("  [device] %s: NOT MEASURED -- %s\n", kernel,
              DeviceUnavailableReason());
}

// ---------------------------------------------------------------------------
// 1. fused_mhc_pre_norm_sinkhorn (host reference roofline)
//
// The Sinkhorn solve alone, across iteration counts, since that is the part
// the kernel keeps entirely in registers across 16 SIMT threads.
// ---------------------------------------------------------------------------
void BenchSinkhorn(int warmup, int iters) {
  Header("fused_mhc_pre_norm_sinkhorn (host reference: Sinkhorn solve)");

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
  DeviceRowUnavailable("fused_mhc_pre_norm_sinkhorn");
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
void BenchMhcPost(int warmup, int iters) {
  Header("fused_mhc_post_residual (host reference: 4-stream post-mix)");

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
  DeviceRowUnavailable("fused_mhc_post_residual");
}

// ---------------------------------------------------------------------------
// 3. fast_fp8_indexer_topk (host reference roofline)
//
// Scales with sequence length; the scoring half dominates.
// ---------------------------------------------------------------------------
void BenchIndexer(int warmup, int iters) {
  Header("fast_fp8_indexer_topk (host reference: score + select)");

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
  DeviceRowUnavailable("fast_fp8_indexer_topk");
}

// ---------------------------------------------------------------------------
// 4. mxfp4_expert_gemv (host reference roofline)
//
// The contract geometries: 2048x4096 and 4096x2048.
// ---------------------------------------------------------------------------
void BenchMxfp4Gemv(int warmup, int iters) {
  Header("mxfp4_expert_gemv (host reference: MXFP4 GEMV)");

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
  DeviceRowUnavailable("mxfp4_expert_gemv");
}

// ---------------------------------------------------------------------------
// 5. paged_compressor_rope
//
// The reference's decode cadence is stateful (a persistent ring buffer across
// steps), so a meaningful figure needs the full group of 4 steps driven in
// order. Left to the device path rather than approximated here.
// ---------------------------------------------------------------------------
void BenchPagedCompressor() {
  Header("paged_compressor_rope");
  std::printf("  [host]   not benchmarked: the decode cadence is stateful "
              "across a 4-step group;\n           a per-call figure would "
              "misrepresent the hold/emit asymmetry.\n");
  DeviceRowUnavailable("paged_compressor_rope");
}

// ---------------------------------------------------------------------------
// Scaling across cores
//
// On device this is the AIV sweep (1, 4, 32 cores) reported as scaling
// efficiency. There is no device here, so state the intent rather than print
// a host number dressed up as a hardware figure.
// ---------------------------------------------------------------------------
void ReportScaling() {
  std::printf("\n=== AIV scaling efficiency ===\n");
  if (!kDeviceLaunchLayerAvailable) {
    std::printf("  NOT MEASURED -- %s\n", DeviceUnavailableReason());
    std::printf("  When wired: sweep block_dim over the kernel's supported core\n"
                "  counts (1 for the single-AIV kernels, 4 for\n"
                "  fused_mhc_post_residual, 32 for mxfp4_expert_gemv) and report\n"
                "  t(1)/(N*t(N)) per geometry.\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  int warmup = 3;
  int iters = 20;
  if (argc >= 2) warmup = std::atoi(argv[1]);
  if (argc >= 3) iters = std::atoi(argv[2]);
  if (warmup < 0) warmup = 0;
  if (iters < 1) iters = 1;

  std::printf("bench_custom_kernels_performance\n");
  std::printf("  warmup=%d (discarded), timed iterations=%d\n", warmup, iters);
  std::printf("  allocation and setup are outside the timed region\n");
  std::printf("  host CPU debug runtimes (cpudebug/tikicpulib): not linked\n");
  std::printf("  NOTE: host-reference figures are a roofline baseline, NOT "
              "device numbers.\n");

  BenchSinkhorn(warmup, iters);
  BenchMhcPost(warmup, iters);
  BenchIndexer(warmup, iters);
  BenchMxfp4Gemv(warmup, iters);
  BenchPagedCompressor();
  ReportScaling();

  std::printf("\nDone.\n");
  return 0;
}
