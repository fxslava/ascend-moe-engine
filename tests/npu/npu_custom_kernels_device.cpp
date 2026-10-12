// -----------------------------------------------------------------------------
// test_custom_kernels_npu_device -- Target B, on-device verification.
//
// Runs the five arch35 kernels on real hardware through the Launch* wrappers
// and compares each result, on the fly, against the same in-memory CPU golden
// that Target A asserts the invariants of. Nothing is read from disk.
//
// Built only under -DENABLE_ARCH35_DEVICE_LAUNCH=ON, which also supplies the
// BiSheng-built launch layer and the AscendCL link interface.
//
// ENTRY GUARD
// An absent NPU is not a test failure, it is an environment without hardware.
// main() probes aclrtGetDeviceCount first and, on zero devices or a failed
// init, prints "[ SKIP ALL: No physical NPU device detected ]" and exits 0.
// Past that point there IS a device, so every ACL error and every non-success
// Status is a hard FAIL carrying its code and the CANN error message -- a
// launch that fails on a machine that has an NPU is a real defect.
// -----------------------------------------------------------------------------

#include "kernel_test_harness.hpp"
#include "moe/kernels/arch35/launch.hpp"

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"

#include <acl/acl.h>

#include <string>
#include <vector>

namespace moe::kerntest {
namespace {

using namespace moe::kernels::arch35;

// The stream main() opened, so the per-operator cases do not each have to
// thread it through RunDeviceCases. Set once, before any case runs, and only
// after the entry guard has confirmed a device exists.
aclrtStream g_stream = nullptr;

// ---------------------------------------------------------------------------
// RAII device buffer
// ---------------------------------------------------------------------------
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { Free(); }

  aclError Alloc(std::size_t bytes) {
    Free();
    bytes_ = bytes;
    if (bytes == 0) {
      return ACL_SUCCESS;  // a legitimately empty operand stays null
    }
    const aclError rc = aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
      ptr_ = nullptr;
      bytes_ = 0;
    }
    return rc;
  }

  aclError Upload(const void* host, std::size_t bytes) {
    if (bytes == 0) return ACL_SUCCESS;
    return aclrtMemcpy(ptr_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  }

  aclError Download(void* host, std::size_t bytes) const {
    if (bytes == 0) return ACL_SUCCESS;
    return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  }

  aclError Zero() {
    if (bytes_ == 0) return ACL_SUCCESS;
    return aclrtMemset(ptr_, bytes_, 0, bytes_);
  }

  void Free() noexcept {
    if (ptr_ != nullptr) {
      (void)aclrtFree(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  void* get() const noexcept { return ptr_; }
  template <typename T>
  T* as() const noexcept { return static_cast<T*>(ptr_); }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Device session.
//
// Two outcomes are deliberately kept apart:
//
//   NoDevice -- there is no NPU here (or no CANN runtime to ask). That is an
//               environment fact, not a defect, and main() turns it into the
//               skip banner and exit 0.
//   Failed   -- there IS hardware but bringing up a device, context or stream
//               failed. That is a real fault and must be reported as one,
//               with its aclError.
//
// Collapsing the two is how a suite ends up either crying wolf on every
// laptop or silently passing on a broken machine.
// ---------------------------------------------------------------------------
enum class OpenResult { kOk, kNoDevice, kFailed };

// Defined below, next to LaunchFail; declared here because AclSession::Open
// reports its post-detection failures through it.
std::string AclFail(const char* what, aclError rc);

class AclSession {
 public:
  OpenResult Open(std::string* why) {
    const aclError init_rc = aclInit(nullptr);
    if (init_rc != ACL_SUCCESS) {
      // No runtime to ask means no device to find.
      *why = "aclInit failed (no CANN runtime available), aclError=" +
             std::to_string(init_rc);
      return OpenResult::kNoDevice;
    }
    initialized_ = true;

    std::uint32_t count = 0;
    const aclError count_rc = aclrtGetDeviceCount(&count);
    if (count_rc != ACL_SUCCESS) {
      *why = "aclrtGetDeviceCount failed, aclError=" + std::to_string(count_rc);
      return OpenResult::kNoDevice;
    }
    if (count == 0) {
      *why = "aclrtGetDeviceCount reported 0 devices";
      return OpenResult::kNoDevice;
    }
    device_count_ = count;

    // Past this point hardware exists, so every failure is a real one.
    const aclError set_rc = aclrtSetDevice(0);
    if (set_rc != ACL_SUCCESS) {
      *why = AclFail("aclrtSetDevice(0)", set_rc);
      return OpenResult::kFailed;
    }
    device_set_ = true;
    const aclError stream_rc = aclrtCreateStream(&stream_);
    if (stream_rc != ACL_SUCCESS) {
      *why = AclFail("aclrtCreateStream", stream_rc);
      return OpenResult::kFailed;
    }
    return OpenResult::kOk;
  }

  std::uint32_t device_count() const noexcept { return device_count_; }

  ~AclSession() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    if (device_set_) (void)aclrtResetDevice(0);
    if (initialized_) (void)aclFinalize();
  }

  aclrtStream stream() const noexcept { return stream_; }

 private:
  bool initialized_ = false;
  bool device_set_ = false;
  std::uint32_t device_count_ = 0;
  aclrtStream stream_ = nullptr;
};

std::string AclFail(const char* what, aclError rc) {
  std::string out = std::string(what) + " failed, aclError=" + std::to_string(rc);
  const char* msg = aclGetRecentErrMsg();
  if (msg != nullptr && msg[0] != '\0') {
    out += "\n           CANN: ";
    out += msg;
  }
  return out;
}

// Everything known about a failed Launch*.
//
// `Status::launch_error` on its own says almost nothing: a chevron launch
// returns void, so the first observable symptom is the following stream
// synchronization, by which point a failed tiling allocation, a failed tiling
// copy, a rejected grid and a device-side fault are indistinguishable. The
// launch layer records which ACL call actually failed (see
// LastLaunchFailure() in moe/kernels/arch35/launch.hpp); this prints it
// together with the CANN error text so one run on real hardware identifies
// the cause instead of leaving it to guesswork.
std::string LaunchFail(const char* wrapper, Status st) {
  std::string out = std::string(wrapper) + " -> " + ToString(st);
  if (st == Status::launch_error) {
    const LaunchFailure& f = LastLaunchFailure();
    out += "\n           failing call: ";
    out += (f.call != nullptr) ? f.call : "(not recorded)";
    if (f.op != nullptr) {
      out += "  [op ";
      out += f.op;
      out += "]";
    }
    out += "  aclError=" + std::to_string(f.acl_error);
    const char* msg = aclGetRecentErrMsg();
    if (msg != nullptr && msg[0] != '\0') {
      out += "\n           CANN: ";
      out += msg;
    }
  } else {
    // invalid_argument / unsupported_shape never reach the device: the host
    // guards rejected the call, so the arguments are the thing to look at.
    out += " (rejected by the host guards before enqueue; check the operand "
           "pointers and the tiling geometry against the contract)";
  }
  return out;
}

// ---------------------------------------------------------------------------
// mxfp4_expert_gemv: packed/scale/x in, y out. The simplest full contract.
// ---------------------------------------------------------------------------
void DeviceMxfp4Gemv(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kRows = 2048;
  constexpr std::uint32_t kCols = 4096;
  constexpr std::uint32_t kGroup = 32;
  const std::size_t packed_bytes = static_cast<std::size_t>(kRows) * (kCols / 2);
  const std::size_t scale_bytes = static_cast<std::size_t>(kRows) * (kCols / kGroup);

  Rng rng(0xD4F4ULL);
  const std::vector<std::uint8_t> packed = rng.bytes(packed_bytes);
  // Keep E8M0 near the bias so the decode cannot overflow to inf.
  const std::vector<std::uint8_t> scale = rng.bytes(scale_bytes, 120, 134);
  const std::vector<float> x = rng.normal(kCols);

  // Golden, in memory, this run.
  moe::emu::Mxfp4Matrix w;
  w.packed = packed.data();
  w.scale = scale.data();
  w.rows = static_cast<int>(kRows);
  w.cols = static_cast<int>(kCols);
  std::vector<float> want(kRows, 0.0F);
  moe::emu::mxfp4_gemv_cpu(w, x.data(), want.data());

  DeviceBuffer d_packed, d_scale, d_x, d_y;
  aclError rc = d_packed.Alloc(packed_bytes);
  if (rc == ACL_SUCCESS) rc = d_scale.Alloc(scale_bytes);
  if (rc == ACL_SUCCESS) rc = d_x.Alloc(kCols * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(kRows * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device mxfp4_expert_gemv", AclFail("aclrtMalloc", rc));
    return;
  }
  rc = d_packed.Upload(packed.data(), packed_bytes);
  if (rc == ACL_SUCCESS) rc = d_scale.Upload(scale.data(), scale_bytes);
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), kCols * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device mxfp4_expert_gemv", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status st = LaunchMxfp4ExpertGemv(
      d_packed.as<Mxfp4Pair>(), kCols / 2, d_scale.as<E8M0>(), kCols / kGroup,
      d_x.as<float>(), kRows, kCols, d_y.as<float>(), nullptr,
      kMxfp4GemvWorkspaceBytes, stream);
  if (st != Status::success) {
    rep.fail("device mxfp4_expert_gemv",
             LaunchFail("LaunchMxfp4ExpertGemv", st));
    return;
  }

  std::vector<float> got(kRows, 0.0F);
  rc = d_y.Download(got.data(), kRows * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device mxfp4_expert_gemv", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  const double cos = CosineSimilarity(got.data(), want.data(), kRows);
  const double l2 = RelativeL2(got.data(), want.data(), kRows);
  char detail[220];
  std::snprintf(detail, sizeof(detail),
                "%ux%u: cos %.9f (floor 0.99999), relL2 %.3e (ceil 1e-4), "
                "finite=%d, seed 0x%llx",
                kRows, kCols, cos, l2, static_cast<int>(AllFinite(got.data(), kRows)),
                static_cast<unsigned long long>(rng.seed()));
  rep.check(AllFinite(got.data(), kRows) && cos >= 0.99999 && l2 <= 1e-4,
            "device mxfp4_expert_gemv matches the in-memory golden", detail);
}

// ---------------------------------------------------------------------------
// fused_mhc_post_residual: 4 streams x 4096, B + post + h_out in, y out.
// ---------------------------------------------------------------------------
void DeviceMhcPostResidual(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kStreams = moe::emu::kHcStreams;
  constexpr std::uint32_t kHidden = moe::emu::kHiddenSize;
  const std::size_t n = static_cast<std::size_t>(kStreams) * kHidden;

  Rng rng(0xD9057ULL);
  const std::vector<std::uint16_t> x = ToBf16(rng.normal(n, 0.0F, 0.5F));
  const std::vector<std::uint16_t> h_out =
      ToBf16(rng.normal(kHidden, 0.0F, 0.5F));
  std::vector<float> B(static_cast<std::size_t>(kStreams) * kStreams);
  {
    const std::vector<float> logits =
        rng.normal(static_cast<std::size_t>(kStreams) * kStreams, 0.0F, 2.0F);
    moe::emu::sinkhorn_knopp_4x4(logits.data(), B.data(), moe::emu::kHcEpsilon,
                                 moe::emu::kHcSinkhornIters);
  }
  const std::vector<float> post =
      rng.uniform(static_cast<std::size_t>(kStreams), 0.2F, 1.8F);

  std::vector<std::uint16_t> want(n, 0);
  moe::emu::mhc_post_cpu(x.data(), B.data(), post.data(), h_out.data(),
                         want.data());

  DeviceBuffer d_x, d_b, d_post, d_h, d_y;
  aclError rc = d_x.Alloc(n * 2);
  if (rc == ACL_SUCCESS) rc = d_b.Alloc(B.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_post.Alloc(post.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_h.Alloc(kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(n * 2);
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_post_residual", AclFail("aclrtMalloc", rc));
    return;
  }
  rc = d_x.Upload(x.data(), n * 2);
  if (rc == ACL_SUCCESS) rc = d_b.Upload(B.data(), B.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_post.Upload(post.data(), post.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_h.Upload(h_out.data(), kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_post_residual", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status st = LaunchMhcPostResidual(
      d_x.as<Bf16>(), kHidden, d_b.as<float>(), kStreams, d_post.as<float>(),
      d_h.as<Bf16>(), d_y.as<Bf16>(), kHidden, nullptr, kMhcPostWorkspaceBytes,
      stream);
  if (st != Status::success) {
    rep.fail("device fused_mhc_post_residual",
             LaunchFail("LaunchMhcPostResidual", st));
    return;
  }

  std::vector<std::uint16_t> got(n, 0);
  rc = d_y.Download(got.data(), n * 2);
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_post_residual", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  const std::vector<float> got_f = FromBf16(got);
  const std::vector<float> want_f = FromBf16(want);
  const double cos = CosineSimilarity(got_f.data(), want_f.data(), n);
  const double l2 = RelativeL2(got_f.data(), want_f.data(), n);
  // The spec's comparator for this kernel: "cosine >= 0.9995 plus reported
  // relative L2; no acceptance from cosine alone for an all-zero output."
  //
  // That last clause is a statement about the GOLDEN, and it is measured on
  // the golden. The previous form asked whether
  // CosineSimilarity(want, want) == 1.0, which is not a degeneracy test at
  // all: it reduces to dot / (sqrt(s) * sqrt(s)) compared exactly against
  // 1.0, and sqrt(s) * sqrt(s) != s in binary floating point, so it reported
  // "degenerate" for well-varied random inputs. That is why this case
  // previously failed on `non-degenerate=0` while simultaneously reporting
  // exact bit parity (cos 1.000000000, relL2 0.000e+00) --- the kernel was
  // right and the guard was wrong.
  const Spread ref = MeasureSpread(want_f.data(), n);
  const bool non_degenerate = ref.NonDegenerate();
  char detail[360];
  std::snprintf(detail, sizeof(detail),
                "4x%u: cos %.9f (floor 0.9995), relL2 %.3e, finite=%d, "
                "golden rms %.3e stddev %.3e range [%.3f, %.3f] "
                "non-degenerate=%d, seed 0x%llx",
                kHidden, cos, l2, static_cast<int>(AllFinite(got_f.data(), n)),
                ref.rms, ref.stddev, ref.min, ref.max,
                static_cast<int>(non_degenerate),
                static_cast<unsigned long long>(rng.seed()));
  rep.check(AllFinite(got_f.data(), n) && non_degenerate && cos >= 0.9995 &&
                l2 <= 1e-3,
            "device fused_mhc_post_residual matches the in-memory golden",
            detail);
}

// ---------------------------------------------------------------------------
// fast_fp8_indexer_topk: exact index match against the CPU selection.
// ---------------------------------------------------------------------------
void DeviceFp8IndexerTopK(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kHeads = 64;
  constexpr std::uint32_t kHeadDim = 128;
  constexpr std::uint32_t kBlockSize = 128;
  constexpr std::uint32_t kSeqLen = 2048;
  constexpr std::uint32_t kTopK = 512;
  const std::uint32_t num_blocks = kSeqLen / kBlockSize;
  const std::size_t key_bytes =
      static_cast<std::size_t>(num_blocks) * kBlockSize * kHeadDim;

  Rng rng(0xDF8ULL);
  const std::vector<std::uint8_t> query =
      rng.bytes(static_cast<std::size_t>(kHeads) * kHeadDim, 1, 120);
  const std::vector<std::uint8_t> keys = rng.bytes(key_bytes, 1, 120);
  const std::vector<float> scales = rng.uniform(kHeads + kSeqLen, 0.5F, 1.5F);
  std::vector<std::int32_t> table(num_blocks);
  for (std::uint32_t b = 0; b < num_blocks; ++b) {
    table[b] = static_cast<std::int32_t>(b);
  }

  std::vector<std::int32_t> want(kTopK, -1);
  moe::emu::fast_fp8_indexer_topk_cpu(
      query.data(), keys.data(), scales.data(), table.data(),
      static_cast<std::int32_t>(kSeqLen), static_cast<std::int32_t>(kTopK),
      want.data(), static_cast<std::int32_t>(kBlockSize),
      static_cast<int>(kHeads), static_cast<int>(kHeadDim));

  const std::size_t ws_bytes = Fp8IndexerWorkspaceBytes(kSeqLen);

  DeviceBuffer d_q, d_k, d_s, d_t, d_out, d_ws;
  aclError rc = d_q.Alloc(query.size());
  if (rc == ACL_SUCCESS) rc = d_k.Alloc(key_bytes);
  if (rc == ACL_SUCCESS) rc = d_s.Alloc(scales.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_t.Alloc(table.size() * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_out.Alloc(kTopK * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_ws.Alloc(ws_bytes);
  if (rc != ACL_SUCCESS) {
    rep.fail("device fast_fp8_indexer_topk", AclFail("aclrtMalloc", rc));
    return;
  }
  rc = d_q.Upload(query.data(), query.size());
  if (rc == ACL_SUCCESS) rc = d_k.Upload(keys.data(), key_bytes);
  if (rc == ACL_SUCCESS)
    rc = d_s.Upload(scales.data(), scales.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_t.Upload(table.data(), table.size() * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_out.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device fast_fp8_indexer_topk", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status st = LaunchFp8IndexerTopK(
      d_q.as<Fp8E4M3>(), kHeadDim, d_k.as<Fp8E4M3>(), kBlockSize * kHeadDim,
      d_s.as<float>(), d_t.as<std::int32_t>(), num_blocks, num_blocks,
      kBlockSize, kSeqLen, d_out.as<std::int32_t>(), d_ws.get(), ws_bytes,
      stream);
  if (st != Status::success) {
    rep.fail("device fast_fp8_indexer_topk",
             LaunchFail("LaunchFp8IndexerTopK", st));
    return;
  }

  std::vector<std::int32_t> got(kTopK, 0);
  rc = d_out.Download(got.data(), kTopK * sizeof(std::int32_t));
  if (rc != ACL_SUCCESS) {
    rep.fail("device fast_fp8_indexer_topk", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  // Selection is a deterministic total order, so this is an EXACT match --
  // no tolerance. A single differing rank is a real ordering bug.
  std::size_t first_diff = kTopK;
  for (std::size_t i = 0; i < kTopK; ++i) {
    if (got[i] != want[i]) {
      first_diff = i;
      break;
    }
  }
  char detail[220];
  if (first_diff == kTopK) {
    std::snprintf(detail, sizeof(detail),
                  "seq %u, top_k %u: all ranks identical", kSeqLen, kTopK);
  } else {
    std::snprintf(detail, sizeof(detail),
                  "seq %u, top_k %u: first mismatch at rank %zu (device %d, "
                  "golden %d)",
                  kSeqLen, kTopK, first_diff, got[first_diff],
                  want[first_diff]);
  }
  rep.check(first_diff == kTopK,
            "device fast_fp8_indexer_topk reproduces the golden ranking exactly",
            detail);
}

// ---------------------------------------------------------------------------
// fused_mhc_pre_norm_sinkhorn: the full pre-block fixture.
//
// Operands, per the contract: x [4,4096] BF16, phi [24,16384] FP32 (1.5 MB),
// alpha [3] FP32, bias [24] FP32, gamma [4096] BF16. Outputs: h_raw [4096]
// FP32, h_norm [4096] BF16, post [4] FP32, birkhoff [4,4] FP32.
//
// The golden comes from mhc_pre_sinkhorn_cpu, which writes a trace rather
// than the kernel's output tensors, so the mapping is stated explicitly:
//
//   h_raw    <- trace.h_in          (sum_i pre[i] * x[i,:])
//   post     <- trace.h_post
//   birkhoff <- trace.birkhoff
//   h_norm   <- BF16(gamma * h_raw * rsqrt(mean(h_raw^2) + eps)), applied
//               here because the reference stops before the activation norm
//               (the kernel's own formula, kernel header lines 11-12).
//
// phi/alpha/bias are synthesized rather than taken from a checkpoint: this
// asserts the kernel reproduces the reference arithmetic, which does not
// depend on the weights being the trained ones. Scales are kept small so the
// 16384-term projection cannot overflow BF16 on the way in.
// ---------------------------------------------------------------------------
void DeviceMhcPreNormSinkhorn(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kStreams = moe::emu::kHcStreams;      // 4
  constexpr std::uint32_t kHidden = moe::emu::kHiddenSize;      // 4096
  constexpr std::uint32_t kRows = moe::emu::kHcProjectionRows;  // 24
  constexpr std::uint32_t kFlat = moe::emu::kHcFlatWidth;       // 16384
  const std::size_t n = static_cast<std::size_t>(kStreams) * kHidden;
  const std::size_t phi_elems = static_cast<std::size_t>(kRows) * kFlat;

  Rng rng(0xD9E0ULL);
  const std::vector<float> x_f32 = rng.normal(n, 0.0F, 0.5F);
  const std::vector<std::uint16_t> x = ToBf16(x_f32);
  // 1/sqrt(16384) == 1/128, so a row's projection stays O(1).
  const std::vector<float> phi = rng.normal(phi_elems, 0.0F, 1.0F / 128.0F);
  const std::vector<float> alpha = rng.uniform(3, 0.5F, 1.5F);
  const std::vector<float> bias = rng.normal(kRows, 0.0F, 0.25F);
  const std::vector<std::uint16_t> gamma =
      ToBf16(rng.uniform(kHidden, 0.5F, 1.5F));

  // --- golden ------------------------------------------------------------
  moe::emu::MhcSinkhornTrace trace;
  moe::emu::mhc_pre_sinkhorn_cpu(x.data(), phi.data(), alpha.data(),
                                 bias.data(), &trace);

  std::vector<float> want_h_raw(kHidden);
  for (std::uint32_t d = 0; d < kHidden; ++d) want_h_raw[d] = trace.h_in[d];

  // The activation RMSNorm the kernel fuses on, computed in FP64 so the
  // reference is not the thing losing precision.
  std::vector<std::uint16_t> want_h_norm(kHidden);
  {
    double mean_sq = 0.0;
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      mean_sq += static_cast<double>(want_h_raw[d]) * want_h_raw[d];
    }
    mean_sq /= static_cast<double>(kHidden);
    const double inv_rms = 1.0 / std::sqrt(mean_sq + 1e-6);
    for (std::uint32_t d = 0; d < kHidden; ++d) {
      const double g = Bf16ToFp32(gamma[d]);
      want_h_norm[d] =
          Fp32ToBf16(static_cast<float>(g * want_h_raw[d] * inv_rms));
    }
  }

  // --- device ------------------------------------------------------------
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
  if (rc == ACL_SUCCESS) rc = d_birkhoff.Alloc(kStreams * kStreams * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_pre_norm_sinkhorn", AclFail("aclrtMalloc", rc));
    return;
  }
  rc = d_x.Upload(x.data(), n * 2);
  if (rc == ACL_SUCCESS) rc = d_phi.Upload(phi.data(), phi_elems * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_alpha.Upload(alpha.data(), alpha.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_bias.Upload(bias.data(), bias.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_gamma.Upload(gamma.data(), kHidden * 2);
  if (rc == ACL_SUCCESS) rc = d_h_raw.Zero();
  if (rc == ACL_SUCCESS) rc = d_h_norm.Zero();
  if (rc == ACL_SUCCESS) rc = d_post.Zero();
  if (rc == ACL_SUCCESS) rc = d_birkhoff.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_pre_norm_sinkhorn",
             AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status st = LaunchMhcPreNormSinkhorn(
      d_x.as<Bf16>(), kHidden, d_phi.as<float>(), kFlat, d_alpha.as<float>(),
      d_bias.as<float>(), d_gamma.as<Bf16>(), d_h_raw.as<float>(),
      d_h_norm.as<Bf16>(), d_post.as<float>(), d_birkhoff.as<float>(), nullptr,
      kMhcPreNormWorkspaceBytes, stream);
  if (st != Status::success) {
    rep.fail("device fused_mhc_pre_norm_sinkhorn",
             LaunchFail("LaunchMhcPreNormSinkhorn", st));
    return;
  }

  std::vector<float> got_h_raw(kHidden, 0.0F);
  std::vector<std::uint16_t> got_h_norm(kHidden, 0);
  std::vector<float> got_post(kStreams, 0.0F);
  std::vector<float> got_birkhoff(kStreams * kStreams, 0.0F);
  rc = d_h_raw.Download(got_h_raw.data(), kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_h_norm.Download(got_h_norm.data(), kHidden * 2);
  if (rc == ACL_SUCCESS)
    rc = d_post.Download(got_post.data(), kStreams * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_birkhoff.Download(got_birkhoff.data(),
                             kStreams * kStreams * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device fused_mhc_pre_norm_sinkhorn",
             AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  // --- four independent assertions, because they fail for different bugs --
  {
    const Spread ref = MeasureSpread(want_h_raw.data(), kHidden);
    const double cos = CosineSimilarity(got_h_raw.data(), want_h_raw.data(),
                                        kHidden);
    const double l2 = RelativeL2(got_h_raw.data(), want_h_raw.data(), kHidden);
    char detail[300];
    std::snprintf(detail, sizeof(detail),
                  "cos %.9f (floor 0.99999), relL2 %.3e (ceil 1e-4), "
                  "golden rms %.3e stddev %.3e, non-degenerate=%d",
                  cos, l2, ref.rms, ref.stddev,
                  static_cast<int>(ref.NonDegenerate()));
    rep.check(ref.NonDegenerate() && AllFinite(got_h_raw.data(), kHidden) &&
                  cos >= 0.99999 && l2 <= 1e-4,
              "device fused_mhc_pre_norm_sinkhorn h_raw matches the golden",
              detail);
  }
  {
    const std::vector<float> got_f = FromBf16(got_h_norm);
    const std::vector<float> want_f = FromBf16(want_h_norm);
    const Spread ref = MeasureSpread(want_f.data(), kHidden);
    const double cos = CosineSimilarity(got_f.data(), want_f.data(), kHidden);
    const double l2 = RelativeL2(got_f.data(), want_f.data(), kHidden);
    char detail[300];
    // BF16 store on both sides: an 8-bit mantissa carries up to ~2e-3
    // relative, so relL2 is bounded by the format and cosine stays strict.
    std::snprintf(detail, sizeof(detail),
                  "cos %.9f (floor 0.9999), relL2 %.3e (ceil 5e-3, "
                  "BF16-store limited), golden rms %.3e, non-degenerate=%d",
                  cos, l2, ref.rms, static_cast<int>(ref.NonDegenerate()));
    rep.check(ref.NonDegenerate() && AllFinite(got_f.data(), kHidden) &&
                  cos >= 0.9999 && l2 <= 5e-3,
              "device fused_mhc_pre_norm_sinkhorn h_norm matches the golden",
              detail);
  }
  {
    const double worst =
        MaxAbsDiff(got_post.data(), trace.h_post, kStreams);
    char detail[220];
    std::snprintf(detail, sizeof(detail),
                  "max |diff| %.3e (tol 1e-5) over %u streams; golden "
                  "[%.6f, %.6f, %.6f, %.6f]",
                  worst, kStreams, trace.h_post[0], trace.h_post[1],
                  trace.h_post[2], trace.h_post[3]);
    rep.check(worst <= 1e-5 && AllFinite(got_post.data(), kStreams),
              "device fused_mhc_pre_norm_sinkhorn post matches the golden",
              detail);
  }
  {
    // Two separate claims: it equals the reference solve, AND it is on the
    // Birkhoff polytope. The second is what a wrong normalization order
    // breaks without necessarily moving the first very much.
    const double worst =
        MaxAbsDiff(got_birkhoff.data(), trace.birkhoff, kStreams * kStreams);
    const BirkhoffResidual r =
        BirkhoffCheck(got_birkhoff.data(), static_cast<int>(kStreams));
    char detail[300];
    std::snprintf(detail, sizeof(detail),
                  "max |diff| vs golden %.3e (tol 1e-5); device row dev %.3e, "
                  "col dev %.3e (col tol 1e-5), entries [%.6f, %.6f]",
                  worst, r.worst_row, r.worst_col, r.min_entry, r.max_entry);
    rep.check(worst <= 1e-5 && r.finite && r.worst_col <= 1e-5 &&
                  r.min_entry >= 0.0 && r.max_entry <= 1.0 + 1e-6,
              "device fused_mhc_pre_norm_sinkhorn birkhoff matches the golden "
              "and is doubly stochastic",
              detail);
  }
}

}  // namespace

void RunDeviceCases(Report& rep) {
  DeviceMxfp4Gemv(rep, g_stream);
  DeviceMhcPostResidual(rep, g_stream);
  DeviceFp8IndexerTopK(rep, g_stream);
  DeviceMhcPreNormSinkhorn(rep, g_stream);

  // Still a skip, with the specific reason rather than a blanket one. Writing
  // a single-call comparison here would be worse than skipping: the operator
  // is stateful across a 4-step HOLD/EMIT group, and with coff=2 the eight
  // rows reduced into one compressed token are the PREVIOUS group's four
  // a-half state rows followed by the CURRENT group's four b-half rows. A
  // fixture that does not span two groups never exercises the cross-boundary
  // previous-a path at all, so it would certify the easy half of the cadence
  // and report a pass for the operator as a whole.
  rep.skip("device paged_compressor_rope",
           "needs a two-group (8-step) fixture driving state_gm/state_meta_gm "
           "across the group boundary so the coff=2 previous-a path is "
           "covered; moe::emu::paged_compressor_decode_c4a is the golden. A "
           "single-call comparison would not certify the cadence.");
}

}  // namespace moe::kerntest

int main() {
  using moe::kerntest::Report;

  std::printf("test_custom_kernels_npu_device\n");
  std::printf("  scope: the five arch35 operators on real hardware, compared\n"
              "         against the in-memory CPU golden (moe_emu)\n");
  std::printf("  references: generated per run, no disk fixtures\n\n");

  // ------------------------------------------------------------------
  // Entry guard. An absent NPU is an environment fact, not a defect:
  // print the banner and exit 0 so a host without hardware reports a
  // skip instead of a wall of false failures. The phrase "Physical
  // Ascend NPU not detected" is what this suite's ctest registration
  // matches with SKIP_REGULAR_EXPRESSION, so it must appear verbatim.
  // ------------------------------------------------------------------
  moe::kerntest::AclSession session;
  std::string why;
  const moe::kerntest::OpenResult opened = session.Open(&why);

  if (opened == moe::kerntest::OpenResult::kNoDevice) {
    std::printf("[ SKIP ALL: No physical NPU device detected ]\n");
    std::printf("Physical Ascend NPU not detected -- %s\n", why.c_str());
    std::printf("\nNothing was verified. This is not a pass: the operators\n"
                "were neither executed nor compared. Run this binary on a\n"
                "machine with an Ascend 950PR device to get a real result.\n");
    return 0;
  }

  if (opened == moe::kerntest::OpenResult::kFailed) {
    // Hardware exists but could not be brought up. That is a real fault.
    std::printf("=== device bring-up ===\n");
    std::printf("  [ FAIL ] device present (%u) but initialization failed\n"
                "           %s\n",
                session.device_count(), why.c_str());
    std::printf("\nRESULT: FAIL\n");
    return 1;
  }

  std::printf("device count: %u, using device 0\n", session.device_count());
  moe::kerntest::g_stream = session.stream();

  Report rep;
  moe::kerntest::section("device: kernel output vs in-memory golden");
  moe::kerntest::RunDeviceCases(rep);

  std::printf("\n---------------------------------------------------------\n");
  std::printf("checks: %d   failures: %d   skipped: %d\n", rep.checks,
              rep.failures, rep.skipped);
  if (rep.skipped > 0) {
    std::printf("NOTE: %d case(s) skipped -- see the reason above. A skip "
                "here is an unbuilt fixture,\n      not a passing kernel.\n",
                rep.skipped);
  }
  std::printf("%s\n", rep.failures == 0 ? "RESULT: PASS" : "RESULT: FAIL");
  return rep.exit_code();
}
