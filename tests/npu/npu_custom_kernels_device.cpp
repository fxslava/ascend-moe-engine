// -----------------------------------------------------------------------------
// test_custom_kernels_npu_device -- Target B, on-device verification.
//
// Runs the six arch35 kernels on real hardware through the Launch* wrappers
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

// ---------------------------------------------------------------------------
// paged_compressor_rope: the full two-group (8-step) decode cadence.
//
// WHY EIGHT STEPS AND NOT ONE
// The operator is stateful across a 4-step HOLD/EMIT group, and with coff=2
// the eight rows reduced into one compressed token are the PREVIOUS group's
// four a-half state rows followed by the CURRENT group's four b-half rows. A
// fixture confined to one group leaves previous_kv_a / previous_score_a at
// their initial kv 0 / score -inf, which contribute exactly nothing to the
// column softmax -- so it exercises only the easy half of the cadence while
// reporting a pass for the operator as a whole. Steps 0..7 drive
// state_gm/state_meta_gm across the boundary, so the EMIT at step 7 is the
// one that actually reads a populated previous-a region.
//
// What is asserted, per step:
//   HOLD (0,1,2,4,5,6)  emitted == 0 AND the destination row is bit-for-bit
//                       untouched -- the golden documents that a HOLD does not
//                       touch the slot at all, so a canary is the right test.
//   EMIT (3, 7)         emitted == 1 and the row matches the golden token.
//
// Each group writes its OWN destination row (compressed_index 0 then 1), which
// is how a paged cache is actually addressed, and it also keeps the two EMITs
// independently checkable instead of one overwriting the other's evidence.
// ---------------------------------------------------------------------------
void DevicePagedCompressorRope(Report& rep, aclrtStream stream) {
  // The real C4A geometry, from the emu contract rather than the operator's
  // private tiling header. Unlike tests/emu/test_paged_compressor_emu.cpp,
  // which drives the golden at a deliberately small hidden size to stay fast,
  // the device kernel's entry gate fixes these -- so the golden must be called
  // at its defaults, which are exactly these values.
  constexpr std::uint32_t kH = static_cast<std::uint32_t>(moe::emu::kHiddenSize);          // 4096
  constexpr std::uint32_t kD = static_cast<std::uint32_t>(moe::emu::kCompressorHeadDim);   // 512
  constexpr std::uint32_t kRatio = static_cast<std::uint32_t>(moe::emu::kCompressorRatio); // 4
  constexpr std::uint32_t kStateW =
      static_cast<std::uint32_t>(moe::emu::kCompressorStateWidth);                         // 1024
  constexpr std::uint32_t kRopeDim = static_cast<std::uint32_t>(moe::emu::kRopeHeadDim);   // 64
  constexpr std::uint32_t kSteps = 2 * kRatio;                                             // 8
  constexpr std::uint32_t kGroups = 2;
  // GM state ABI: kv_cur | gate_cur | prev_kv_a | prev_score_a, FP32.
  constexpr std::uint32_t kStateFloats = 3u * kRatio * kStateW;              // 12,288
  constexpr std::uint32_t kMetaElems = 4;

  Rng rng(0xC4A0ULL);
  const std::size_t proj = static_cast<std::size_t>(kStateW) * kH;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kH));
  const std::vector<std::uint16_t> wkv = ToBf16(rng.normal(proj, 0.0F, scale));
  const std::vector<std::uint16_t> wgate =
      ToBf16(rng.normal(proj, 0.0F, scale));
  const std::vector<float> ape =
      rng.normal(static_cast<std::size_t>(kRatio) * kStateW, 0.0F, 0.5F);
  const std::vector<std::uint16_t> norm_w =
      ToBf16(rng.normal(kD, 1.0F, 0.25F));
  const std::vector<float> rope_cos = rng.normal(kRopeDim, 0.0F, 1.0F);
  const std::vector<float> rope_sin = rng.normal(kRopeDim, 0.0F, 1.0F);

  std::vector<std::vector<std::uint16_t>> tokens(kSteps);
  for (std::uint32_t s = 0; s < kSteps; ++s) {
    tokens[s] = ToBf16(rng.normal(kH, 0.0F, 0.5F));
  }

  // --- golden, in memory, this run ----------------------------------------
  // Driven step by step so its state evolves exactly as the device's does.
  moe::emu::CompressorRingState golden_state;
  std::vector<std::vector<std::uint16_t>> want(kGroups,
                                               std::vector<std::uint16_t>(kD, 0));
  std::vector<bool> golden_emitted(kSteps, false);
  // A snapshot of the oracle's ring state after EVERY step, so the device's GM
  // state can be compared region by region rather than inferred from the
  // emitted token. See the state-region diff below for why that matters.
  std::vector<moe::emu::CompressorRingState> golden_snapshot(kSteps);
  for (std::uint32_t s = 0; s < kSteps; ++s) {
    const std::uint32_t g = s / kRatio;
    const moe::emu::CompressorStepResult r = moe::emu::paged_compressor_decode_c4a(
        tokens[s].data(), wkv.data(), wgate.data(), ape.data(), norm_w.data(),
        static_cast<std::int32_t>(s), &golden_state, rope_cos.data(),
        rope_sin.data(), want[g].data());
    golden_emitted[s] = r.emitted;
    golden_snapshot[s] = golden_state;
  }
  if (!golden_emitted[kRatio - 1] || !golden_emitted[kSteps - 1] ||
      !golden_state.have_previous) {
    rep.fail("device paged_compressor_rope",
             "the golden did not produce the expected two-group cadence; the "
             "fixture is wrong, so no device claim can be made from it");
    return;
  }

  // --- device -------------------------------------------------------------
  // The canary must be a BF16 bit pattern the kernel would never write, so a
  // HOLD that touched the row is detectable rather than plausibly equal.
  constexpr std::uint16_t kCanary = 0x7FC1u;  // a BF16 NaN payload
  const std::size_t slot_elems = static_cast<std::size_t>(kGroups) * kD;
  const std::vector<std::uint16_t> canary(slot_elems, kCanary);

  DeviceBuffer d_x, d_wkv, d_wgate, d_ape, d_norm, d_cos, d_sin;
  DeviceBuffer d_state, d_meta, d_slots, d_emitted;
  aclError rc = d_x.Alloc(kH * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_wkv.Alloc(proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_wgate.Alloc(proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_ape.Alloc(ape.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_norm.Alloc(kD * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_cos.Alloc(kRopeDim * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_sin.Alloc(kRopeDim * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_state.Alloc(kStateFloats * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_meta.Alloc(kMetaElems * sizeof(std::uint32_t));
  if (rc == ACL_SUCCESS) rc = d_slots.Alloc(slot_elems * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_emitted.Alloc(sizeof(std::uint32_t));
  if (rc != ACL_SUCCESS) {
    rep.fail("device paged_compressor_rope", AclFail("aclrtMalloc", rc));
    return;
  }

  rc = d_wkv.Upload(wkv.data(), proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_wgate.Upload(wgate.data(), proj * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_ape.Upload(ape.data(), ape.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_norm.Upload(norm_w.data(), kD * sizeof(std::uint16_t));
  if (rc == ACL_SUCCESS) rc = d_cos.Upload(rope_cos.data(), kRopeDim * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_sin.Upload(rope_sin.data(), kRopeDim * sizeof(float));
  // Zeroed state and meta are the documented cold start: have_previous = 0,
  // tokens_seen = 0, and Launch requires step == tokens_seen.
  if (rc == ACL_SUCCESS) rc = d_state.Zero();
  if (rc == ACL_SUCCESS) rc = d_meta.Zero();
  if (rc == ACL_SUCCESS) {
    rc = d_slots.Upload(canary.data(), slot_elems * sizeof(std::uint16_t));
  }
  if (rc != ACL_SUCCESS) {
    rep.fail("device paged_compressor_rope", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  bool cadence_ok = true;
  std::string cadence_detail;
  // Kept apart from the cadence verdict: "the HOLD/EMIT rhythm is wrong" and
  // "a state region did not survive the launch boundary" are different
  // defects and must not be reported under one name.
  bool state_ok = true;
  std::string state_detail;
  for (std::uint32_t s = 0; s < kSteps && cadence_ok && state_ok; ++s) {
    const std::uint32_t g = s / kRatio;
    const bool expect_emit = ((s % kRatio) == kRatio - 1);

    rc = d_x.Upload(tokens[s].data(), kH * sizeof(std::uint16_t));
    if (rc == ACL_SUCCESS) rc = d_emitted.Zero();
    if (rc != ACL_SUCCESS) {
      rep.fail("device paged_compressor_rope",
               AclFail("aclrtMemcpy(H2D, step)", rc));
      return;
    }

    const Status st = LaunchPagedCompressorRope(
        d_x.as<Bf16>(), d_wkv.as<Bf16>(), kH, d_wgate.as<Bf16>(), kH,
        d_ape.as<float>(), kStateW, d_norm.as<Bf16>(), d_cos.as<float>(),
        d_sin.as<float>(), s, d_state.as<float>(),
        d_meta.as<std::uint32_t>(), d_slots.as<Bf16>() + g * kD,
        d_emitted.as<std::uint32_t>(), nullptr, kPagedCompressorWorkspaceBytes,
        stream);
    if (st != Status::success) {
      rep.fail("device paged_compressor_rope",
               "step " + std::to_string(s) + ": " +
                   LaunchFail("LaunchPagedCompressorRope", st));
      return;
    }

    std::uint32_t emitted = 0xFFFFFFFFu;
    rc = d_emitted.Download(&emitted, sizeof(emitted));
    std::vector<std::uint16_t> slots(slot_elems, 0);
    if (rc == ACL_SUCCESS) {
      rc = d_slots.Download(slots.data(), slot_elems * sizeof(std::uint16_t));
    }
    if (rc != ACL_SUCCESS) {
      rep.fail("device paged_compressor_rope", AclFail("aclrtMemcpy(D2H)", rc));
      return;
    }

    const bool emit_flag_ok = (emitted == (expect_emit ? 1u : 0u));
    if (!emit_flag_ok) {
      cadence_ok = false;
      cadence_detail = "step " + std::to_string(s) + ": emitted=" +
                       std::to_string(emitted) + ", expected " +
                       std::to_string(expect_emit ? 1 : 0);
      break;
    }
    // The meta counter is a precondition for the NEXT call, so a drift here
    // would make every later step silently invalid rather than fail.
    std::uint32_t meta[kMetaElems] = {0, 0, 0, 0};
    rc = d_meta.Download(meta, sizeof(meta));
    if (rc == ACL_SUCCESS && meta[1] != s + 1u) {
      cadence_ok = false;
      cadence_detail = "step " + std::to_string(s) + ": state_meta[1] (tokens_seen)=" +
                       std::to_string(meta[1]) + ", expected " +
                       std::to_string(s + 1);
      break;
    }

    if (!expect_emit) {
      // A HOLD must not touch ANY lane of the destination row.
      bool untouched = true;
      for (std::uint32_t i = 0; i < kD; ++i) {
        if (slots[g * kD + i] != kCanary) {
          untouched = false;
          break;
        }
      }
      if (!untouched) {
        cadence_ok = false;
        cadence_detail = "step " + std::to_string(s) +
                         " is a HOLD but wrote the destination row";
      }
    }

    // --- the persistent state, region by region --------------------------
    // WHY THIS IS CHECKED SEPARATELY FROM THE TOKEN
    // The emitted token is a lossy witness of the state: it is the output of a
    // column softmax, an 8-row reduction, an RMSNorm and a RoPE, so a damaged
    // state region shows up as one cosine with no indication of WHICH region,
    // row or lane is wrong. Worse, group 0's token cannot witness the prev_a
    // regions at all -- with have_previous = 0 those rows carry score -inf and
    // therefore softmax weight exactly 0, so the whole prev_a path is first
    // exercised at step 7. A group-0 pass is not evidence that the roll works.
    //
    // tests/emu/test_paged_compressor_state_emu.cpp simulates this state
    // machine on the host and reproduces the oracle through the group
    // boundary, so the region offsets and the roll indexing are known-good
    // arithmetic. What that simulation cannot model is ordering and
    // visibility on real hardware -- which is exactly what this diff measures,
    // and the reason it reports a per-region number instead of a verdict: a
    // zeroed or stale region scores ~1.0 and FP32 projection noise scores
    // ~1e-6, so the two are never ambiguous.
    std::vector<float> state_host(kStateFloats, 0.0F);
    rc = d_state.Download(state_host.data(), kStateFloats * sizeof(float));
    if (rc != ACL_SUCCESS) {
      rep.fail("device paged_compressor_rope",
               AclFail("aclrtMemcpy(D2H, state)", rc));
      return;
    }
    const moe::emu::CompressorRingState& gref = golden_snapshot[s];
    struct Region {
      const char* name;
      std::uint32_t offset;  // in FP32 elements
      std::uint32_t rows;
      std::uint32_t row_len;   // elements per row in the GM region
      const float* src;        // golden, row-major with stride src_stride
      std::uint32_t src_stride;
      bool live;               // false before the first EMIT rolls it
    };
    const Region regions[4] = {
        {"kv_cur", 0u, kRatio, kStateW, &gref.kv_rows[0][0], kStateW, true},
        {"gate_cur", kRatio * kStateW, kRatio, kStateW, &gref.score_rows[0][0],
         kStateW, true},
        {"prev_kv_a", 2u * kRatio * kStateW, kRatio, kD,
         &gref.previous_kv_a[0][0], kD, gref.have_previous},
        {"prev_gate_a", 2u * kRatio * kStateW + kRatio * kD, kRatio, kD,
         &gref.previous_score_a[0][0], kD, gref.have_previous},
    };
    for (const Region& reg : regions) {
      if (!reg.live) {
        continue;  // the oracle has not populated it yet; nothing to compare
      }
      double dif = 0.0;
      double ref = 0.0;
      std::uint32_t worst_row = 0;
      std::uint32_t worst_lane = 0;
      double worst_abs = -1.0;
      for (std::uint32_t r = 0; r < reg.rows; ++r) {
        for (std::uint32_t l = 0; l < reg.row_len; ++l) {
          const double got_v =
              state_host[reg.offset + r * reg.row_len + l];
          const double want_v = reg.src[r * reg.src_stride + l];
          const double d = got_v - want_v;
          dif += d * d;
          ref += want_v * want_v;
          if (std::fabs(d) > worst_abs) {
            worst_abs = std::fabs(d);
            worst_row = r;
            worst_lane = l;
          }
        }
      }
      const double rel = ref > 0.0 ? std::sqrt(dif / ref) : std::sqrt(dif);
      // The projection is a 4,096-term FP32 dot on both sides, in the same
      // order, so the bound is FP32 accumulation error: 4096 * 2^-24 = 2.4e-4
      // worst case. 1e-3 leaves margin for an FMA contraction on the device
      // and is still three orders of magnitude below a lost region.
      if (rel > 1e-3) {
        state_ok = false;
        char d[260];
        std::snprintf(d, sizeof(d),
                      "step %u: state region %s diverged, relL2 %.3e (ceil "
                      "1e-3); largest absolute delta %.6e at row %u lane %u "
                      "(device %.6e, golden %.6e)",
                      s, reg.name, rel, worst_abs, worst_row, worst_lane,
                      static_cast<double>(
                          state_host[reg.offset + worst_row * reg.row_len +
                                     worst_lane]),
                      static_cast<double>(
                          reg.src[worst_row * reg.src_stride + worst_lane]));
        state_detail = d;
        break;
      }
    }
  }
  rep.check(cadence_ok,
            "device paged_compressor_rope drives the 8-step cadence: HOLD "
            "steps emit nothing and leave the row untouched, EMIT steps flag "
            "once, tokens_seen advances every step",
            cadence_ok ? ("steps 0..7, emit at 3 and 7, seed 0x" +
                          std::to_string(rng.seed()))
                       : cadence_detail);
  rep.check(state_ok,
            "device paged_compressor_rope keeps all four GM state regions in "
            "step with the oracle across every launch boundary",
            state_ok ? std::string("kv_cur, gate_cur, prev_kv_a, prev_gate_a "
                                   "all within relL2 1e-3 of the oracle at "
                                   "every step (prev_a compared from the "
                                   "first EMIT onward)")
                     : state_detail);
  if (!cadence_ok || !state_ok) {
    return;
  }

  // --- the two emitted tokens ---------------------------------------------
  std::vector<std::uint16_t> got_slots(slot_elems, 0);
  rc = d_slots.Download(got_slots.data(), slot_elems * sizeof(std::uint16_t));
  if (rc != ACL_SUCCESS) {
    rep.fail("device paged_compressor_rope", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  for (std::uint32_t g = 0; g < kGroups; ++g) {
    std::vector<std::uint16_t> got_row(got_slots.begin() + g * kD,
                                       got_slots.begin() + (g + 1) * kD);
    const std::vector<float> got_f = FromBf16(got_row);
    const std::vector<float> want_f = FromBf16(want[g]);

    const double cos = CosineSimilarity(got_f.data(), want_f.data(), kD);
    const double l2 = RelativeL2(got_f.data(), want_f.data(), kD);
    // Non-degeneracy on the REFERENCE, not a self-comparison (CLAUDE.md
    // section 4.8): an all-zero or constant golden would make cosine
    // meaningless, and the second group is the one that must differ from the
    // first or the previous-a path is not actually contributing.
    double rms = 0.0;
    for (const float v : want_f) rms += static_cast<double>(v) * v;
    rms = std::sqrt(rms / static_cast<double>(kD));
    const bool non_degenerate = rms > 1e-6 && AllFinite(want_f.data(), kD);

    // BF16 store on both sides: an 8-bit mantissa carries up to ~2e-3
    // relative error per store, and this row is the product of a column
    // softmax, an 8-row reduction, an RMSNorm and a RoPE rotation, each
    // independently rounded -- so relative-L2 must stay loose while cosine
    // stays strict. Rounding moves magnitude; a wrong reduction order or a
    // dropped previous-a row moves direction.
    char detail[300];
    std::snprintf(detail, sizeof(detail),
                  "group %u (compressed token %u): cos %.9f (floor 0.9999), "
                  "relL2 %.3e (ceil 5e-3, BF16-store limited), golden rms "
                  "%.3e, non-degenerate=%d",
                  g, g, cos, l2, rms, static_cast<int>(non_degenerate));
    rep.check(non_degenerate && AllFinite(got_f.data(), kD) &&
                  cos >= 0.9999 && l2 <= 5e-3,
              g == 0 ? "device paged_compressor_rope group 0 EMIT matches the "
                       "golden (cold previous-a)"
                     : "device paged_compressor_rope group 1 EMIT matches the "
                       "golden ACROSS the group boundary (populated "
                       "previous-a, the coff=2 path)",
              detail);
  }

  // The two tokens must not be identical: if they were, the previous-a region
  // would be contributing nothing and both EMITs would be the same easy case.
  {
    const std::vector<float> a = FromBf16(want[0]);
    const std::vector<float> b = FromBf16(want[1]);
    const double between = CosineSimilarity(a.data(), b.data(), kD);
    char detail[200];
    std::snprintf(detail, sizeof(detail),
                  "cos(group0, group1) = %.9f; equality would mean the "
                  "cross-boundary path changed nothing",
                  between);
    rep.check(between < 0.9999,
              "the two compressed tokens differ, so the second EMIT really "
              "exercised the previous-a path",
              detail);
  }
}

// ---------------------------------------------------------------------------
// dsv4_moe_expert: the fused routed expert, gate/up GEMV + asymmetric clamped
// SwiGLU + down GEMV, for one token.
//
// Operands: w1/w3 [2048,4096] and w2 [4096,2048] MXFP4 with their E8M0 scale
// planes, x [4096] FP32. Output y [4096] FP32. The only geometry
// MakeDsv4MoeExpertTiling admits is hidden=4096 / inter=2048.
//
// WHAT IS AND IS NOT OBSERVABLE.
// This operator materializes exactly two tensors in GM: `activated` (in the
// caller's workspace, written by the gate_up dispatch and read by the down
// dispatch) and `y`. The raw gate and up projections are never stored --
// gate_up fuses silu(clamp(gate)) * clamp(up) inside a 32-row UB tile and
// writes only the product. So there are two checkable stages here, not four,
// and this case asserts both:
//
//   activated <- scratch[0, inter)  after moe_expert_swiglu_mxfp4_cpu, which
//                leaves the activation in the first half of its scratch
//   y         <- the reference's `out`
//
// Checking `activated` is what separates a gate_up bug from a down bug: with
// only `y` asserted, an error in either dispatch arrives as the same failure.
// It costs nothing, because the test already owns the workspace.
//
// THE FIXTURE, AND WHY THE SCALE BAND IS NARROW.
// The DSV4 clamp SATURATES. Once |w1 @ x| is well past the limit, min(gate,
// limit) returns the limit regardless of the GEMV result, and the comparison
// stops being sensitive to the arithmetic it exists to verify. E8M0 codes in
// [121,123] put the gate's standard deviation at 7.54 against a limit of 10,
// measured on this reference at this geometry with this seed: 9.0% of gate
// rows and 20.4% of up rows saturate, so both halves of the clamp are hit
// while the large majority of rows stay in the linear region where a wrong
// product still shows. Codes near the 127 bias do NOT work here -- at
// hidden=4096 they pin ~98% of `up` at +/-limit and would let a broken up
// GEMV pass. (tests/emu uses a band near the bias at hidden=1024, a quarter
// of the reduction width; the band does not transfer.)
//
// TOLERANCE.
// Both sides decode the same MXFP4 lattice, and an E2M1 nibble times a
// power-of-two E8M0 scale is exact in FP32, so there is no quantization term:
// the only difference is reduction ORDER. The device splits each row over 16
// lanes and factors the scale per 32-element group, then reduces the lanes in
// a butterfly; the reference accumulates sequentially. FP32 carries a 24-bit
// mantissa, so a 4096-term reorder is bounded in the random-walk model by
// ~sqrt(4096) * 2^-24 ~= 4e-6 relative, and the down GEMV's 2048 terms add
// the same order again. 1e-4 on relative L2 is that bound with room to spare,
// and the floor on cosine stays tight: reordering moves magnitude, while a
// wrong lane range or a mismatched scale byte moves direction.
// ---------------------------------------------------------------------------
void DeviceDsv4MoeExpert(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kHidden = 4096;
  constexpr std::uint32_t kInter = 2048;
  constexpr std::uint32_t kGroup = 32;
  // The clamp bound is NOT a launch argument: MakeDsv4MoeExpertTiling writes
  // kDsv4ExpertSwigluLimit into the POD host-side. That constant lives in
  // op_host/dsv4_moe_expert_tiling.h, which is op-private (§1.2) and must not
  // be reached across from here (§1.3), so the golden is driven by the
  // oracle's own public copy. The two are the same model constant; the guard
  // below fails loudly if either drifts, because the fixture's scale band was
  // chosen against this value.
  constexpr float kLimit = moe::emu::kSwigluLimit;

  const std::size_t w13_packed = static_cast<std::size_t>(kInter) * (kHidden / 2);
  const std::size_t w13_scale = static_cast<std::size_t>(kInter) * (kHidden / kGroup);
  const std::size_t w2_packed = static_cast<std::size_t>(kHidden) * (kInter / 2);
  const std::size_t w2_scale = static_cast<std::size_t>(kHidden) * (kInter / kGroup);

  Rng rng(0xD50E4ULL);
  const std::vector<std::uint8_t> w1p = rng.bytes(w13_packed);
  const std::vector<std::uint8_t> w3p = rng.bytes(w13_packed);
  const std::vector<std::uint8_t> w2p = rng.bytes(w2_packed);
  const std::vector<std::uint8_t> w1s = rng.bytes(w13_scale, 121, 123);
  const std::vector<std::uint8_t> w3s = rng.bytes(w13_scale, 121, 123);
  const std::vector<std::uint8_t> w2s = rng.bytes(w2_scale, 121, 123);
  const std::vector<float> x = rng.normal(kHidden);

  if (kLimit != 10.0F) {
    rep.fail("device dsv4_moe_expert",
             "moe::emu::kSwigluLimit is not the model's 10.0, so it no longer "
             "agrees with the kernel's kDsv4ExpertSwigluLimit and the "
             "fixture's scale band is no longer calibrated to the clamp");
    return;
  }

  // --- golden, in memory, this run -------------------------------------
  const auto matrix = [](const std::vector<std::uint8_t>& packed,
                         const std::vector<std::uint8_t>& scale, int rows,
                         int cols) {
    moe::emu::Mxfp4Matrix m;
    m.packed = packed.data();
    m.scale = scale.data();
    m.rows = rows;
    m.cols = cols;
    return m;
  };
  const moe::emu::Mxfp4Matrix w1 =
      matrix(w1p, w1s, static_cast<int>(kInter), static_cast<int>(kHidden));
  const moe::emu::Mxfp4Matrix w3 =
      matrix(w3p, w3s, static_cast<int>(kInter), static_cast<int>(kHidden));
  const moe::emu::Mxfp4Matrix w2 =
      matrix(w2p, w2s, static_cast<int>(kHidden), static_cast<int>(kInter));

  std::vector<float> want_y(kHidden, 0.0F);
  std::vector<float> scratch;
  moe::emu::moe_expert_swiglu_mxfp4_cpu(w1, w3, w2, x.data(), want_y.data(),
                                        kLimit, &scratch);
  // ExpertSwiglu leaves the activation in scratch[0, inter) and the raw up
  // projection in scratch[inter, 2*inter); the kernel's workspace holds the
  // former.
  if (scratch.size() < static_cast<std::size_t>(kInter)) {
    rep.fail("device dsv4_moe_expert",
             "reference scratch is shorter than inter; the activated-stage "
             "comparison would read past it");
    return;
  }
  const float* want_activated = scratch.data();

  // --- device ------------------------------------------------------------
  const std::size_t workspace_bytes = Dsv4MoeExpertWorkspaceBytes(kInter);
  if (workspace_bytes < static_cast<std::size_t>(kInter) * sizeof(float)) {
    rep.fail("device dsv4_moe_expert",
             "Dsv4MoeExpertWorkspaceBytes is smaller than the activated "
             "vector it must hold");
    return;
  }

  DeviceBuffer d_w1p, d_w1s, d_w3p, d_w3s, d_w2p, d_w2s, d_x, d_y, d_ws;
  aclError rc = d_w1p.Alloc(w13_packed);
  if (rc == ACL_SUCCESS) rc = d_w1s.Alloc(w13_scale);
  if (rc == ACL_SUCCESS) rc = d_w3p.Alloc(w13_packed);
  if (rc == ACL_SUCCESS) rc = d_w3s.Alloc(w13_scale);
  if (rc == ACL_SUCCESS) rc = d_w2p.Alloc(w2_packed);
  if (rc == ACL_SUCCESS) rc = d_w2s.Alloc(w2_scale);
  if (rc == ACL_SUCCESS) rc = d_x.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_ws.Alloc(workspace_bytes);
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_expert", AclFail("aclrtMalloc", rc));
    return;
  }

  rc = d_w1p.Upload(w1p.data(), w13_packed);
  if (rc == ACL_SUCCESS) rc = d_w1s.Upload(w1s.data(), w13_scale);
  if (rc == ACL_SUCCESS) rc = d_w3p.Upload(w3p.data(), w13_packed);
  if (rc == ACL_SUCCESS) rc = d_w3s.Upload(w3s.data(), w13_scale);
  if (rc == ACL_SUCCESS) rc = d_w2p.Upload(w2p.data(), w2_packed);
  if (rc == ACL_SUCCESS) rc = d_w2s.Upload(w2s.data(), w2_scale);
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  // Zeroed so a dispatch that never writes `activated` is visible as zeros
  // rather than as whatever the allocator handed back.
  if (rc == ACL_SUCCESS) rc = d_ws.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_expert", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status st = LaunchDsv4MoeExpert(
      d_w1p.as<Mxfp4Pair>(), d_w1s.as<E8M0>(), d_w3p.as<Mxfp4Pair>(),
      d_w3s.as<E8M0>(), d_w2p.as<Mxfp4Pair>(), d_w2s.as<E8M0>(),
      kHidden / 2, kHidden / kGroup, kInter / 2, kInter / kGroup,
      d_x.as<float>(), kHidden, kInter, d_y.as<float>(), d_ws.get(),
      workspace_bytes, stream);
  if (st != Status::success) {
    rep.fail("device dsv4_moe_expert", LaunchFail("LaunchDsv4MoeExpert", st));
    return;
  }

  // The wrapper synchronizes the stream before it returns (§3.7), so both
  // dispatches have retired and `activated` is readable.
  std::vector<float> got_activated(kInter, 0.0F);
  std::vector<float> got_y(kHidden, 0.0F);
  rc = d_ws.Download(got_activated.data(),
                     static_cast<std::size_t>(kInter) * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_y.Download(got_y.data(), kHidden * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_expert", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  // --- stage 1: activated, the gate_up dispatch -------------------------
  {
    const double cos = CosineSimilarity(got_activated.data(), want_activated, kInter);
    const double l2 = RelativeL2(got_activated.data(), want_activated, kInter);
    const Spread ref = MeasureSpread(want_activated, kInter);
    const bool finite = AllFinite(got_activated.data(), kInter);
    char detail[400];
    std::snprintf(detail, sizeof(detail),
                  "activated[%u] (gate_up, from workspace): cos %.9f (floor "
                  "0.99999), relL2 %.3e (ceil 1e-4), finite=%d, golden rms "
                  "%.3e stddev %.3e range [%.3f, %.3f] non-degenerate=%d, "
                  "seed 0x%llx",
                  kInter, cos, l2, static_cast<int>(finite), ref.rms,
                  ref.stddev, ref.min, ref.max,
                  static_cast<int>(ref.NonDegenerate()),
                  static_cast<unsigned long long>(rng.seed()));
    rep.check(finite && ref.NonDegenerate() && cos >= 0.99999 && l2 <= 1e-4,
              "device dsv4_moe_expert activated (silu(clamp(gate)) * "
              "clamp(up)) matches the in-memory golden",
              detail);
  }

  // --- stage 2: y, the down dispatch ------------------------------------
  {
    const double cos = CosineSimilarity(got_y.data(), want_y.data(), kHidden);
    const double l2 = RelativeL2(got_y.data(), want_y.data(), kHidden);
    const Spread ref = MeasureSpread(want_y.data(), kHidden);
    const bool finite = AllFinite(got_y.data(), kHidden);
    char detail[400];
    std::snprintf(detail, sizeof(detail),
                  "y[%u] (down): cos %.9f (floor 0.9999), relL2 %.3e (ceil "
                  "1e-4), finite=%d, golden rms %.3e stddev %.3e range "
                  "[%.3f, %.3f] non-degenerate=%d, seed 0x%llx",
                  kHidden, cos, l2, static_cast<int>(finite), ref.rms,
                  ref.stddev, ref.min, ref.max,
                  static_cast<int>(ref.NonDegenerate()),
                  static_cast<unsigned long long>(rng.seed()));
    rep.check(finite && ref.NonDegenerate() && cos >= 0.9999 && l2 <= 1e-4,
              "device dsv4_moe_expert y matches moe_expert_swiglu_mxfp4_cpu",
              detail);
  }

  // --- the clamp was actually exercised ---------------------------------
  // The fixture's whole point. If neither clamp half ever engaged, the case
  // verified three GEMVs and a silu, not the DSV4 activation -- and if a half
  // engaged everywhere, the GEMV feeding it was unobservable. Measured on the
  // GOLDEN (§4.8), not by comparing a device value with itself.
  {
    std::vector<float> gate(kInter, 0.0F);
    std::vector<float> up(kInter, 0.0F);
    moe::emu::mxfp4_gemv_cpu(w1, x.data(), gate.data());
    moe::emu::mxfp4_gemv_cpu(w3, x.data(), up.data());
    std::size_t gate_sat = 0;
    std::size_t up_sat = 0;
    for (std::uint32_t i = 0; i < kInter; ++i) {
      if (gate[i] >= kLimit) ++gate_sat;
      if (up[i] >= kLimit || up[i] <= -kLimit) ++up_sat;
    }
    const double gate_pct = 100.0 * static_cast<double>(gate_sat) / kInter;
    const double up_pct = 100.0 * static_cast<double>(up_sat) / kInter;
    char detail[320];
    std::snprintf(detail, sizeof(detail),
                  "golden saturation at limit %.1f: gate(upper) %.1f%%, "
                  "up(two-sided) %.1f%%; both must land in (1%%, 90%%) or the "
                  "fixture is testing the clamp instead of the GEMVs, or the "
                  "GEMVs instead of the clamp",
                  static_cast<double>(kLimit), gate_pct, up_pct);
    rep.check(gate_pct > 1.0 && gate_pct < 90.0 && up_pct > 1.0 &&
                  up_pct < 90.0,
              "the fixture exercises both halves of the asymmetric clamp "
              "without saturating either",
              detail);
  }
}

// ---------------------------------------------------------------------------
// dsv4_moe_decode_topk + dsv4_moe_prefill_batched -- the whole MoE block
// ---------------------------------------------------------------------------
// These are the entry points that supersede dsv4_moe_expert above: each core
// runs ONE expert for ONE token end to end, so `activated` stays in UB and
// never reaches GM. Decode parallelises over the top_k routed experts (one
// AIV per slot, block_dim == top_k); prefill over the token batch.
//
// NOTHING HERE HAS EVER RUN ON HARDWARE. This case exists so it can
// (CLAUDE.md §5.7). Past the suite's entry guard there IS a device, so every
// ACL error and every non-success Status is a hard FAIL carrying its code,
// LastLaunchFailure() and aclGetRecentErrMsg() (§4.4) -- never a skip.
//
// WHAT THIS CASE ADDS over DeviceDsv4MoeExpert, which already covers the
// expert body (the lane map, the uint4 group load, the butterfly, the
// asymmetric clamp):
//
//   1. EXPERT-STRIDE ADDRESSING. All experts live in one buffer per role,
//      expert e at e * <role>_expert_stride. A wrong stride reads a
//      DIFFERENT expert and produces finite, plausible numbers with no shape
//      to catch it. The fixture makes experts distinguishable on purpose (the
//      expert id participates in the scale band) and the golden is built
//      through the same stride arithmetic.
//   2. THE ATOMIC COMBINE. Decode accumulates top_k slots into y with an
//      atomic-add store, so y is ZEROED HERE before the launch -- the
//      wrapper does not touch operand data (§3.4). If the zeroing or the
//      atomic mode were wrong, y would carry allocator garbage or only the
//      last slot.
//   3. THE ROUTER WEIGHTING, folded into the down store.
//   4. PER-TOKEN EXPERT SELECTION on the prefill path.
//
// GEOMETRY is the production 4096/2048, which is what the unoverridden host
// gate admits -- unlike the CAModel suite, hardware has no reason to run a
// scaled-down width.
//
// TOLERANCE, same reasoning as the single-expert case above: both sides
// decode the same MXFP4 lattice and an E2M1 nibble times a power-of-two E8M0
// scale is exact in FP32, so there is no quantization term and the only
// difference is reduction ORDER. FP32 carries a 24-bit mantissa, so a
// 4096-term reorder is bounded in the random-walk model by
// ~sqrt(4096) * 2^-24 ~= 4e-6 relative, and the down GEMV's 2048 terms add
// the same order again.
//
// Decode carries ONE MORE reordering term than the single-expert case: the
// top_k slots are summed by atomic adds in whatever order the cores retire,
// so the slot order is not reproducible run to run. That is top_k = 6 terms
// of comparable magnitude, ~sqrt(6) * 2^-24 ~= 1.5e-7 relative -- negligible
// against the per-row reorder, so the same 1e-4 relative-L2 ceiling holds.
// Cosine stays tight: reordering moves magnitude, a wrong expert or a
// dropped router weight moves direction.
void DeviceDsv4MoeBlock(Report& rep, aclrtStream stream) {
  constexpr std::uint32_t kHidden = 4096;
  constexpr std::uint32_t kInter = 2048;
  constexpr std::uint32_t kGroup = 32;
  constexpr std::uint32_t kExperts = 6;  // a decode step's routed set
  constexpr std::uint32_t kTopK = 6;
  constexpr std::uint32_t kTokens = 4;
  constexpr float kLimit = moe::emu::kSwigluLimit;

  if (kLimit != 10.0F) {
    rep.fail("device dsv4_moe_block",
             "moe::emu::kSwigluLimit is not the model's 10.0, so the fixture's "
             "scale band is no longer calibrated to the clamp");
    return;
  }

  // Per-expert strides, exactly as MakeDsv4Moe*Tiling computes them.
  const std::size_t w13_packed_stride =
      static_cast<std::size_t>(kInter) * (kHidden / 2);
  const std::size_t w13_scale_stride =
      static_cast<std::size_t>(kInter) * (kHidden / kGroup);
  const std::size_t w2_packed_stride =
      static_cast<std::size_t>(kHidden) * (kInter / 2);
  const std::size_t w2_scale_stride =
      static_cast<std::size_t>(kHidden) * (kInter / kGroup);

  Rng rng(0xD5B10C4ULL);
  // ONE contiguous buffer per role, all experts -- the device layout.
  const std::vector<std::uint8_t> w1p = rng.bytes(w13_packed_stride * kExperts);
  const std::vector<std::uint8_t> w3p = rng.bytes(w13_packed_stride * kExperts);
  const std::vector<std::uint8_t> w2p = rng.bytes(w2_packed_stride * kExperts);
  // The scale band is the same one the single-expert case calibrated against
  // the clamp at hidden=4096. Per-expert offsetting would move the band and
  // decalibrate it, so experts are made distinguishable by their PACKED bytes
  // (independently drawn above) rather than by shifting the exponent.
  const std::vector<std::uint8_t> w1s =
      rng.bytes(w13_scale_stride * kExperts, 121, 123);
  const std::vector<std::uint8_t> w3s =
      rng.bytes(w13_scale_stride * kExperts, 121, 123);
  const std::vector<std::uint8_t> w2s =
      rng.bytes(w2_scale_stride * kExperts, 121, 123);

  // Views built through the SAME stride arithmetic the kernel performs.
  std::vector<moe::emu::Mxfp4Matrix> v1(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v3(kExperts);
  std::vector<moe::emu::Mxfp4Matrix> v2(kExperts);
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    v1[e] = {w1p.data() + static_cast<std::size_t>(e) * w13_packed_stride,
             w1s.data() + static_cast<std::size_t>(e) * w13_scale_stride,
             static_cast<int>(kInter), static_cast<int>(kHidden)};
    v3[e] = {w3p.data() + static_cast<std::size_t>(e) * w13_packed_stride,
             w3s.data() + static_cast<std::size_t>(e) * w13_scale_stride,
             static_cast<int>(kInter), static_cast<int>(kHidden)};
    v2[e] = {w2p.data() + static_cast<std::size_t>(e) * w2_packed_stride,
             w2s.data() + static_cast<std::size_t>(e) * w2_scale_stride,
             static_cast<int>(kHidden), static_cast<int>(kInter)};
  }

  // Routed slots: neither sorted nor the identity, so a kernel that used the
  // SLOT index instead of topk_expert_ids[slot] fails. Weights are distinct
  // and non-unit so a dropped weighting is visible.
  const std::vector<std::int32_t> topk_ids = {4, 0, 5, 2, 1, 3};
  const std::vector<float> routing_weights = {0.30F, 0.25F, 0.18F,
                                              0.12F, 0.09F, 0.06F};
  const std::vector<float> x = rng.normal(kHidden);

  // Tokens routed to different experts, including a repeat, so a kernel that
  // applied one expert to the whole batch fails.
  const std::vector<std::int32_t> token_to_expert = {1, 1, 4, 0};
  const std::vector<float> x_tokens =
      rng.normal(static_cast<std::size_t>(kTokens) * kHidden);

  // --- goldens, in memory, this run (§4.5) -------------------------------
  std::vector<float> scratch;
  std::vector<float> want_decode(kHidden, 0.0F);
  moe::emu::moe_decode_topk_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x.data(), topk_ids.data(),
      routing_weights.data(), static_cast<int>(kTopK),
      static_cast<int>(kExperts), want_decode.data(), kLimit, &scratch);

  std::vector<float> want_prefill(static_cast<std::size_t>(kTokens) * kHidden,
                                  0.0F);
  moe::emu::moe_prefill_batched_mxfp4_cpu(
      v1.data(), v3.data(), v2.data(), x_tokens.data(), token_to_expert.data(),
      static_cast<int>(kTokens), static_cast<int>(kExperts),
      want_prefill.data(), kLimit, &scratch);

  // --- device ------------------------------------------------------------
  DeviceBuffer d_w1p, d_w1s, d_w3p, d_w3s, d_w2p, d_w2s;
  DeviceBuffer d_x, d_ids, d_rw, d_y, d_xt, d_t2e, d_yt, d_actws;
  aclError rc = d_w1p.Alloc(w1p.size());
  if (rc == ACL_SUCCESS) rc = d_w1s.Alloc(w1s.size());
  if (rc == ACL_SUCCESS) rc = d_w3p.Alloc(w3p.size());
  if (rc == ACL_SUCCESS) rc = d_w3s.Alloc(w3s.size());
  if (rc == ACL_SUCCESS) rc = d_w2p.Alloc(w2p.size());
  if (rc == ACL_SUCCESS) rc = d_w2s.Alloc(w2s.size());
  if (rc == ACL_SUCCESS) rc = d_x.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_ids.Alloc(topk_ids.size() * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_rw.Alloc(routing_weights.size() * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_y.Alloc(kHidden * sizeof(float));
  if (rc == ACL_SUCCESS) rc = d_xt.Alloc(x_tokens.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_t2e.Alloc(token_to_expert.size() * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS) rc = d_yt.Alloc(want_prefill.size() * sizeof(float));
  // The decode gang's `activated` plane, [top_k, inter] FP32. Caller-owned:
  // the gang splits one expert across cores, so `activated` crosses cores.
  if (rc == ACL_SUCCESS)
    rc = d_actws.Alloc(static_cast<std::size_t>(
        Dsv4MoeDecodeWorkspaceBytes(kTopK, kInter)));
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_block", AclFail("aclrtMalloc", rc));
    return;
  }

  rc = d_w1p.Upload(w1p.data(), w1p.size());
  if (rc == ACL_SUCCESS) rc = d_w1s.Upload(w1s.data(), w1s.size());
  if (rc == ACL_SUCCESS) rc = d_w3p.Upload(w3p.data(), w3p.size());
  if (rc == ACL_SUCCESS) rc = d_w3s.Upload(w3s.data(), w3s.size());
  if (rc == ACL_SUCCESS) rc = d_w2p.Upload(w2p.data(), w2p.size());
  if (rc == ACL_SUCCESS) rc = d_w2s.Upload(w2s.data(), w2s.size());
  if (rc == ACL_SUCCESS) rc = d_x.Upload(x.data(), kHidden * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_ids.Upload(topk_ids.data(), topk_ids.size() * sizeof(std::int32_t));
  if (rc == ACL_SUCCESS)
    rc = d_rw.Upload(routing_weights.data(),
                     routing_weights.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_xt.Upload(x_tokens.data(), x_tokens.size() * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_t2e.Upload(token_to_expert.data(),
                      token_to_expert.size() * sizeof(std::int32_t));
  // DECODE ACCUMULATES INTO y, so the caller zeroes it (§3.4). This is not
  // defensive tidying: without it y carries allocator garbage and the atomic
  // adds land on top of it.
  if (rc == ACL_SUCCESS) rc = d_y.Zero();
  // Prefill OVERWRITES, so zeroing y_tokens is not required -- it is zeroed
  // anyway so that a token the kernel never writes reads as 0 rather than as
  // whatever the allocator returned.
  if (rc == ACL_SUCCESS) rc = d_yt.Zero();
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_block", AclFail("aclrtMemcpy(H2D)", rc));
    return;
  }

  const Status decode_st = LaunchDsv4MoeDecodeTopk(
      d_x.as<float>(), d_w1p.as<Mxfp4Pair>(), d_w1s.as<E8M0>(),
      d_w3p.as<Mxfp4Pair>(), d_w3s.as<E8M0>(), d_w2p.as<Mxfp4Pair>(),
      d_w2s.as<E8M0>(), d_ids.as<std::int32_t>(), d_rw.as<float>(), kHidden,
      kInter, kExperts, kTopK, d_y.as<float>(), d_actws.as<float>(),
      stream);
  if (decode_st != Status::success) {
    rep.fail("device dsv4_moe_decode_topk",
             LaunchFail("LaunchDsv4MoeDecodeTopk", decode_st));
    return;
  }

  const Status prefill_st = LaunchDsv4MoePrefillBatched(
      d_xt.as<float>(), d_w1p.as<Mxfp4Pair>(), d_w1s.as<E8M0>(),
      d_w3p.as<Mxfp4Pair>(), d_w3s.as<E8M0>(), d_w2p.as<Mxfp4Pair>(),
      d_w2s.as<E8M0>(), d_t2e.as<std::int32_t>(), kHidden, kInter, kExperts,
      kTokens, d_yt.as<float>(), stream);
  if (prefill_st != Status::success) {
    rep.fail("device dsv4_moe_prefill_batched",
             LaunchFail("LaunchDsv4MoePrefillBatched", prefill_st));
    return;
  }

  // Both wrappers synchronize the stream before returning (§3.7).
  std::vector<float> got_decode(kHidden, 0.0F);
  std::vector<float> got_prefill(want_prefill.size(), 0.0F);
  rc = d_y.Download(got_decode.data(), kHidden * sizeof(float));
  if (rc == ACL_SUCCESS)
    rc = d_yt.Download(got_prefill.data(), got_prefill.size() * sizeof(float));
  if (rc != ACL_SUCCESS) {
    rep.fail("device dsv4_moe_block", AclFail("aclrtMemcpy(D2H)", rc));
    return;
  }

  // --- stage 1: decode, the router-weighted sum over top_k slots ---------
  {
    const double cos = CosineSimilarity(got_decode.data(), want_decode.data(),
                                        kHidden);
    const double l2 = RelativeL2(got_decode.data(), want_decode.data(), kHidden);
    const Spread ref = MeasureSpread(want_decode.data(), kHidden);
    const bool finite = AllFinite(got_decode.data(), kHidden);
    char detail[480];
    std::snprintf(detail, sizeof(detail),
                  "y[%u] = sum over %u slots of w*Expert(id): cos %.9f (floor "
                  "0.9999), relL2 %.3e (ceil 1e-4), finite=%d, golden rms "
                  "%.3e stddev %.3e range [%.3f, %.3f] non-degenerate=%d, "
                  "slots={4,0,5,2,1,3}, seed 0x%llx",
                  kHidden, kTopK, cos, l2, static_cast<int>(finite), ref.rms,
                  ref.stddev, ref.min, ref.max,
                  static_cast<int>(ref.NonDegenerate()),
                  static_cast<unsigned long long>(rng.seed()));
    rep.check(finite && ref.NonDegenerate() && cos >= 0.9999 && l2 <= 1e-4,
              "device dsv4_moe_decode_topk matches moe_decode_topk_mxfp4_cpu",
              detail);
  }

  // --- stage 2: prefill, one expert per token ---------------------------
  {
    const std::size_t n = got_prefill.size();
    const double cos =
        CosineSimilarity(got_prefill.data(), want_prefill.data(), n);
    const double l2 = RelativeL2(got_prefill.data(), want_prefill.data(), n);
    const Spread ref = MeasureSpread(want_prefill.data(), n);
    const bool finite = AllFinite(got_prefill.data(), n);
    char detail[480];
    std::snprintf(detail, sizeof(detail),
                  "y[%u tokens x %u]: cos %.9f (floor 0.9999), relL2 %.3e "
                  "(ceil 1e-4), finite=%d, golden rms %.3e stddev %.3e range "
                  "[%.3f, %.3f] non-degenerate=%d, routing={1,1,4,0}, "
                  "seed 0x%llx",
                  kTokens, kHidden, cos, l2, static_cast<int>(finite),
                  ref.rms, ref.stddev, ref.min, ref.max,
                  static_cast<int>(ref.NonDegenerate()),
                  static_cast<unsigned long long>(rng.seed()));
    rep.check(finite && ref.NonDegenerate() && cos >= 0.9999 && l2 <= 1e-4,
              "device dsv4_moe_prefill_batched matches "
              "moe_prefill_batched_mxfp4_cpu",
              detail);
  }

  // --- per-token independence -------------------------------------------
  // Tokens 0 and 1 share expert 1 but have DIFFERENT x, so their outputs must
  // differ. If the kernel reused one token's `activated` across its bucket --
  // the obvious bug in a design where `activated` is a core-resident UB
  // buffer reused per token -- these two rows would be identical, and both
  // the cosine and the relative-L2 above would still look fine because the
  // golden's own rows are similar in magnitude.
  {
    double row_diff = 0.0;
    double row_mag = 0.0;
    for (std::uint32_t c = 0; c < kHidden; ++c) {
      const double a = got_prefill[c];
      const double b = got_prefill[static_cast<std::size_t>(kHidden) + c];
      row_diff += std::fabs(a - b);
      row_mag += std::fabs(a);
    }
    char detail[320];
    std::snprintf(detail, sizeof(detail),
                  "tokens 0 and 1 both route to expert 1 with different x: "
                  "sum|row0-row1| %.6e against sum|row0| %.6e; equal rows "
                  "would mean `activated` leaked between tokens in UB",
                  row_diff, row_mag);
    rep.check(row_mag > 0.0 && row_diff > 1e-3 * row_mag,
              "prefill does not reuse one token's activated across its bucket",
              detail);
  }
}

}  // namespace

void RunDeviceCases(Report& rep) {
  DeviceMxfp4Gemv(rep, g_stream);
  DeviceMhcPostResidual(rep, g_stream);
  DeviceFp8IndexerTopK(rep, g_stream);
  DeviceMhcPreNormSinkhorn(rep, g_stream);
  DevicePagedCompressorRope(rep, g_stream);
  DeviceDsv4MoeExpert(rep, g_stream);
  DeviceDsv4MoeBlock(rep, g_stream);
}

}  // namespace moe::kerntest

int main() {
  using moe::kerntest::Report;

  std::printf("test_custom_kernels_npu_device\n");
  std::printf("  scope: the six arch35 operators on real hardware, compared\n"
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
