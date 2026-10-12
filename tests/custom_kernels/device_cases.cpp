// -----------------------------------------------------------------------------
// device_cases.cpp -- on-device kernel output vs the in-memory CPU golden.
//
// Compiled into test_custom_kernels_verification ONLY when the launch layer is
// available (MOE_KERNEL_LAUNCH_LAYER_AVAILABLE, set by the
// arch35_device_launch target). Without it the harness uses the stub in
// device_cases_stub.cpp and reports the cases as skipped.
//
// NEVER EXECUTED AS WRITTEN: the development workstation has no Ascend NPU
// (no /dev/davinci*, no driver), so every line below is verified to compile
// and link only. It is written to fail loudly rather than quietly: a missing
// device is a SKIP, but a failed ACL call is a FAIL with its error code.
// -----------------------------------------------------------------------------

#include "device_cases.hpp"

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
// Device session. Absence of hardware is a SKIP, not a failure.
// ---------------------------------------------------------------------------
class AclSession {
 public:
  bool Open(std::string* why) {
    if (aclInit(nullptr) != ACL_SUCCESS) {
      *why = "aclInit failed (no CANN runtime or no device)";
      return false;
    }
    initialized_ = true;
    std::uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
      *why = "no physical Ascend NPU detected (aclrtGetDeviceCount == 0)";
      return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
      *why = "aclrtSetDevice(0) failed";
      return false;
    }
    device_set_ = true;
    if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
      *why = "aclrtCreateStream failed";
      return false;
    }
    return true;
  }

  ~AclSession() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    if (device_set_) (void)aclrtResetDevice(0);
    if (initialized_) (void)aclFinalize();
  }

  aclrtStream stream() const noexcept { return stream_; }

 private:
  bool initialized_ = false;
  bool device_set_ = false;
  aclrtStream stream_ = nullptr;
};

std::string AclFail(const char* what, aclError rc) {
  return std::string(what) + " failed, aclError=" + std::to_string(rc);
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
             std::string("LaunchMxfp4ExpertGemv -> ") + ToString(st));
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
             std::string("LaunchMhcPostResidual -> ") + ToString(st));
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
  const bool nonzero = CosineSimilarity(want_f.data(), want_f.data(), n) == 1.0 &&
                       RelativeL2(got_f.data(), want_f.data(), n) <
                           std::numeric_limits<double>::infinity();
  char detail[240];
  std::snprintf(detail, sizeof(detail),
                "4x%u: cos %.9f (floor 0.9995), relL2 %.3e, finite=%d, "
                "non-degenerate=%d",
                kHidden, cos, l2, static_cast<int>(AllFinite(got_f.data(), n)),
                static_cast<int>(nonzero));
  rep.check(AllFinite(got_f.data(), n) && nonzero && cos >= 0.9995,
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
             std::string("LaunchFp8IndexerTopK -> ") + ToString(st));
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

}  // namespace

void RunDeviceCases(Report& rep) {
  section("device: kernel output vs in-memory golden");

  AclSession session;
  std::string why;
  if (!session.Open(&why)) {
    // Expected on any machine without an NPU. Not a failure.
    for (const char* k : {"mxfp4_expert_gemv", "fused_mhc_post_residual",
                          "fast_fp8_indexer_topk"}) {
      rep.skip(std::string("device ") + k, why);
    }
    rep.skip("device fused_mhc_pre_norm_sinkhorn", why);
    rep.skip("device paged_compressor_rope", why);
    return;
  }

  DeviceMxfp4Gemv(rep, session.stream());
  DeviceMhcPostResidual(rep, session.stream());
  DeviceFp8IndexerTopK(rep, session.stream());

  // Not yet wired, with specific reasons rather than a blanket skip:
  rep.skip("device fused_mhc_pre_norm_sinkhorn",
           "needs the 1.5 MB phi projection and a checkpoint norm weight as "
           "device operands; the contract also states h_norm and post are not "
           "serialized, so the golden must come from mhc_pre_sinkhorn_cpu plus "
           "a checkpoint gamma. Fixture not built yet.");
  rep.skip("device paged_compressor_rope",
           "stateful across a 4-step HOLD/EMIT group and needs a two-group "
           "fixture to exercise the cross-boundary previous-a path (per the "
           "contract); a single-call comparison would not certify it.");
}

}  // namespace moe::kerntest
