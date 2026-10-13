// -----------------------------------------------------------------------------
// test_bf16_block_quant_fp8_npu -- on-device verification of
// bf16_block_quant_fp8 against the tests/emu golden.
//
// WHAT THIS IS FOR, SPECIFICALLY
// Besides ordinary numerical verification, this test exists to settle one
// question the host cannot answer: the byte order within
// __asc_fp8x2_storage_t. The toolkit declares that storage as
// `unsigned short` and documents no lane order, so the kernel ASSUMES
// low byte <- float2.x. If that is backwards, every quantized pair comes out
// transposed while every scale stays correct -- a failure mode that is
// invisible to a cosine check over a whole tensor but obvious here, because
// the test explicitly re-checks the golden with the pairs swapped and says
// which orientation matched. See "pair orientation" below.
//
// SUITE CONTRACT (CLAUDE.md §4.3, §4.4)
//   * main() probes aclrtGetDeviceCount first. On zero devices or a failed
//     init it prints the skip banner containing "Physical Ascend NPU not
//     detected" -- the phrase the ctest SKIP_REGULAR_EXPRESSION matches --
//     and exits 0.
//   * PAST that guard there is a device, so every ACL error and every
//     non-success Status is a hard FAIL carrying its code, the recorded
//     LastLaunchFailure() and aclGetRecentErrMsg(). A launch that fails on a
//     machine that HAS an NPU is a real defect, never a skip.
//   * "No device" and "device present but bring-up failed" are kept apart.
//
// The reference is generated in memory, per run, from a PRINTED seed
// (§4.5) -- no disk fixture can go stale and make a broken kernel pass.
//
// This binary never loads a host CPU debug runtime (cpudebug / tikicpulib).
// -----------------------------------------------------------------------------

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/emu/fp8_types.hpp"
#include "moe/kernels/arch35/launch.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using moe::emu::Bf16ToFloat;
using moe::emu::bf16_block_quant_fp8_cpu;
using moe::emu::bf16_block_quant_fp8_dequant;
using moe::emu::FloatToBf16;
using moe::emu::kBlockQuantRefBlockSize;
using namespace moe::kernels::arch35;

constexpr int kBlock = kBlockQuantRefBlockSize;

// The five widths the model asks for, all M = 1.
const std::uint32_t kModelWidths[] = {1024u, 2048u, 4096u, 8192u, 32768u};

int g_checks = 0;
int g_failures = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (ok) {
    std::printf("  [ OK ] %s\n", what.c_str());
    return;
  }
  ++g_failures;
  std::printf("  [FAIL] %s\n", what.c_str());
}

std::string AclFail(const char* call, aclError rc) {
  const char* recent = aclGetRecentErrMsg();
  return std::string(call) + " returned aclError=" + std::to_string(rc) +
         " | aclGetRecentErrMsg: " + (recent != nullptr ? recent : "<none>");
}

// Every non-success Status past the entry guard is a defect, and the recorded
// failure is what distinguishes a tiling malloc from a bad grid from a
// device-side fault (CLAUDE.md §3.5).
std::string LaunchFail(const char* call, Status st) {
  const LaunchFailure& f = LastLaunchFailure();
  const char* recent = aclGetRecentErrMsg();
  return std::string(call) + " returned Status=" +
         std::to_string(static_cast<unsigned>(st)) + " (" + ToString(st) +
         ") [op=" + (f.op != nullptr ? f.op : "-") +
         ", call=" + (f.call != nullptr ? f.call : "-") +
         ", aclError=" + std::to_string(f.acl_error) +
         "] | aclGetRecentErrMsg: " + (recent != nullptr ? recent : "<none>");
}

// A device allocation that frees itself. Deliberately local to this file: the
// suite's other binary has its own, and this one must stay loadable with
// nothing but the launch layer.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)aclrtFree(ptr_);
    }
  }
  aclError Alloc(std::size_t bytes) {
    bytes_ = bytes;
    return aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  }
  aclError Upload(const void* host, std::size_t bytes) {
    return aclrtMemcpy(ptr_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  aclError Download(void* host, std::size_t bytes) const {
    return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  }
  aclError Fill(std::uint8_t byte) {
    return aclrtMemset(ptr_, bytes_, byte, bytes_);
  }
  void* get() const { return ptr_; }
  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// One width, end to end
// ---------------------------------------------------------------------------
void VerifyWidth(std::uint32_t width, std::uint32_t seed, aclrtStream stream) {
  std::printf("\n-- width %u (%u blocks of %d), seed 0x%08X\n", width,
              width / kBlock, kBlock, seed);

  const int blocks = static_cast<int>(width) / kBlock;
  std::mt19937 engine(seed);
  std::uniform_real_distribution<float> dist(-6.0F, 6.0F);

  std::vector<std::uint16_t> x(width);
  for (std::uint16_t& bits : x) {
    bits = FloatToBf16(dist(engine));
  }

  // Host golden, from the same bits the device will see.
  std::vector<std::uint8_t> want_q(width);
  std::vector<float> want_scale(static_cast<std::size_t>(blocks));
  bf16_block_quant_fp8_cpu(x.data(), static_cast<int>(width), want_q.data(),
                           want_scale.data());

  DeviceBuffer d_x;
  DeviceBuffer d_q;
  DeviceBuffer d_scale;
  aclError rc = d_x.Alloc(static_cast<std::size_t>(width) * 2);
  if (rc == ACL_SUCCESS) rc = d_q.Alloc(width);
  if (rc == ACL_SUCCESS) rc = d_scale.Alloc(static_cast<std::size_t>(blocks) * 4);
  if (rc == ACL_SUCCESS) {
    rc = d_x.Upload(x.data(), static_cast<std::size_t>(width) * 2);
  }
  // Poison the outputs: a kernel that returns early without writing must not
  // be able to pass by inheriting a zeroed buffer that happens to look right.
  if (rc == ACL_SUCCESS) rc = d_q.Fill(0xA5u);
  if (rc == ACL_SUCCESS) rc = d_scale.Fill(0xA5u);
  if (rc != ACL_SUCCESS) {
    Check(false, AclFail("aclrtMalloc/aclrtMemcpy/aclrtMemset", rc));
    return;
  }

  const Status st = LaunchBf16BlockQuantFp8(
      reinterpret_cast<const Bf16*>(d_x.get()), width,
      reinterpret_cast<Fp8E4M3*>(d_q.get()), d_scale.as<float>(), nullptr,
      kBlockQuantFp8WorkspaceBytes, stream);
  if (st != Status::success) {
    Check(false, LaunchFail("LaunchBf16BlockQuantFp8", st));
    return;
  }
  rc = aclrtSynchronizeStream(stream);
  if (rc != ACL_SUCCESS) {
    Check(false, AclFail("aclrtSynchronizeStream", rc));
    return;
  }
  Check(true, "launch and synchronize returned success");

  std::vector<std::uint8_t> got_q(width);
  std::vector<float> got_scale(static_cast<std::size_t>(blocks));
  rc = d_q.Download(got_q.data(), width);
  if (rc == ACL_SUCCESS) {
    rc = d_scale.Download(got_scale.data(),
                          static_cast<std::size_t>(blocks) * 4);
  }
  if (rc != ACL_SUCCESS) {
    Check(false, AclFail("aclrtMemcpy D2H", rc));
    return;
  }

  // --- scales: bit-exact ---------------------------------------------------
  // Both sides compute max|x|/448 in FP32 from identical inputs. The device
  // forms it as m * (1/448) and the host as m / 448, and 1/448 is not exactly
  // representable, so one FP32 ulp of disagreement is allowed and no more.
  // FP32 keeps a 23-bit stored mantissa, hence 2^-23 relative.
  constexpr double kFp32RelativeUlp = 1.1920929e-7;
  int scale_bad = 0;
  int scale_exact = 0;
  double worst_scale_rel = 0.0;
  for (int b = 0; b < blocks; ++b) {
    const double want = want_scale[static_cast<std::size_t>(b)];
    const double got = got_scale[static_cast<std::size_t>(b)];
    if (got == want) ++scale_exact;
    const double rel = want != 0.0 ? std::fabs(got - want) / std::fabs(want)
                                   : std::fabs(got - want);
    worst_scale_rel = std::max(worst_scale_rel, rel);
    if (!(rel <= kFp32RelativeUlp)) ++scale_bad;
  }
  Check(scale_bad == 0,
        "all " + std::to_string(blocks) +
            " block scales agree within one FP32 ulp (" +
            std::to_string(scale_exact) + " bit-exact, worst relative " +
            std::to_string(worst_scale_rel) + ")");

  // --- pair orientation: the question this test exists for ------------------
  // Count code mismatches as-is, then count them again with each even/odd
  // pair swapped. If the swapped count is dramatically lower, the kernel's
  // assumption about __asc_fp8x2_storage_t byte order is backwards and the
  // fix is the two-line swap marked in the kernel.
  int mismatch_direct = 0;
  int mismatch_swapped = 0;
  for (std::uint32_t i = 0; i < width; ++i) {
    if (got_q[i] != want_q[i]) ++mismatch_direct;
    const std::uint32_t partner = (i % 2u == 0u) ? i + 1u : i - 1u;
    if (got_q[partner] != want_q[i]) ++mismatch_swapped;
  }

  if (mismatch_direct == 0) {
    Check(true,
          "all " + std::to_string(width) +
              " E4M3 codes are bit-identical to the golden; "
              "__asc_fp8x2_storage_t byte order is low-byte <- float2.x, as "
              "the kernel assumes");
  } else if (mismatch_swapped == 0) {
    Check(false,
          "PAIR ORIENTATION IS REVERSED: " + std::to_string(mismatch_direct) +
              " of " + std::to_string(width) +
              " codes differ as written, but ZERO differ when each even/odd "
              "pair is swapped. __asc_fp8x2_storage_t packs high-byte <- "
              "float2.x. Fix: swap the two q_ub stores in "
              "bf16_block_quant_fp8_kernel.cpp (the site is commented "
              "UNVERIFIED).");
  } else {
    // Neither orientation explains it, so report how wrong it is numerically
    // rather than just counting bytes -- a scale misattribution and a
    // genuine arithmetic bug look different here.
    double worst_abs = 0.0;
    for (std::uint32_t i = 0; i < width; ++i) {
      const int b = static_cast<int>(i) / kBlock;
      const double want = bf16_block_quant_fp8_dequant(
          want_q[i], want_scale[static_cast<std::size_t>(b)]);
      const double got = bf16_block_quant_fp8_dequant(
          got_q[i], got_scale[static_cast<std::size_t>(b)]);
      worst_abs = std::max(worst_abs, std::fabs(got - want));
    }
    Check(false,
          std::to_string(mismatch_direct) + " of " + std::to_string(width) +
              " codes differ from the golden (" +
              std::to_string(mismatch_swapped) +
              " still differ with pairs swapped, so this is NOT a byte-order "
              "transposition); worst dequantized delta " +
              std::to_string(worst_abs));
  }

  // --- the output was actually written -------------------------------------
  // 0xA5 is the poison byte. Finding it everywhere means the kernel returned
  // early, which on device is a silent no-op (§2.5) and would otherwise look
  // like a numerical disagreement rather than a rejected launch.
  const bool all_poison =
      std::all_of(got_q.begin(), got_q.end(),
                  [](std::uint8_t c) { return c == 0xA5u; });
  Check(!all_poison,
        "the kernel wrote the output buffer (not left at the 0xA5 poison "
        "pattern, which would mean the tiling gate rejected the launch)");
}

// ---------------------------------------------------------------------------
// Rejection paths: the host must refuse what the contract excludes
// ---------------------------------------------------------------------------
void VerifyRejections(aclrtStream stream) {
  std::printf("\n-- host-side contract rejections\n");

  DeviceBuffer d_x;
  DeviceBuffer d_q;
  DeviceBuffer d_scale;
  if (d_x.Alloc(4096 * 2) != ACL_SUCCESS || d_q.Alloc(4096) != ACL_SUCCESS ||
      d_scale.Alloc(128 * 4) != ACL_SUCCESS) {
    Check(false, "setup allocation for the rejection cases failed");
    return;
  }

  const Bf16* x = reinterpret_cast<const Bf16*>(d_x.get());
  Fp8E4M3* q = reinterpret_cast<Fp8E4M3*>(d_q.get());
  float* s = d_scale.as<float>();

  Check(LaunchBf16BlockQuantFp8(nullptr, 4096, q, s, nullptr,
                                kBlockQuantFp8WorkspaceBytes,
                                stream) == Status::invalid_argument,
        "a null input is invalid_argument (a caller mistake, §2.6)");

  // 1000 is not a whole number of 32-element blocks: geometry v1 declines,
  // which is unsupported_shape and must NOT be collapsed into
  // invalid_argument.
  Check(LaunchBf16BlockQuantFp8(x, 1000, q, s, nullptr,
                                kBlockQuantFp8WorkspaceBytes,
                                stream) == Status::unsupported_shape,
        "a width that is not a multiple of 32 is unsupported_shape");

  Check(LaunchBf16BlockQuantFp8(x, 1u << 20, q, s, nullptr,
                                kBlockQuantFp8WorkspaceBytes,
                                stream) == Status::unsupported_shape,
        "a width past the v1 envelope is unsupported_shape");

  // x aliasing an output would corrupt the result mid-tile, and the device
  // cannot see it.
  Check(LaunchBf16BlockQuantFp8(reinterpret_cast<const Bf16*>(d_q.get()), 4096,
                                q, s, nullptr, kBlockQuantFp8WorkspaceBytes,
                                stream) == Status::invalid_argument,
        "an input aliasing an output is invalid_argument");
}

}  // namespace

int main() {
  std::printf("test_bf16_block_quant_fp8_npu\n");
  std::printf("  host CPU debug runtimes (cpudebug/tikicpulib): not linked\n");

  // ---- entry guard (§4.3): no device is a SKIP with exit 0 ----------------
  const aclError init_rc = aclInit(nullptr);
  if (init_rc != ACL_SUCCESS && init_rc != ACL_ERROR_REPEAT_INITIALIZE) {
    std::printf("Physical Ascend NPU not detected -- aclInit returned %d "
                "(no usable CANN backend on this host)\n",
                init_rc);
    return 0;
  }
  std::uint32_t device_count = 0;
  const aclError count_rc = aclrtGetDeviceCount(&device_count);
  if (count_rc != ACL_SUCCESS) {
    std::printf("Physical Ascend NPU not detected -- aclrtGetDeviceCount "
                "returned %d\n",
                count_rc);
    return 0;
  }
  if (device_count == 0) {
    std::printf("Physical Ascend NPU not detected -- aclrtGetDeviceCount "
                "reports 0 devices\n");
    return 0;
  }

  // ---- past the guard a device EXISTS, so bring-up failure is a FAIL -----
  // (§4.4: "no device" and "device present but bring-up failed" must not be
  // collapsed -- the first is an environment fact, the second a defect.)
  std::printf("  %u NPU(s) visible; verifying on device 0\n", device_count);
  const char* soc = aclrtGetSocName();
  std::printf("  aclrtGetSocName: %s\n", soc != nullptr ? soc : "<null>");

  aclError rc = aclrtSetDevice(0);
  if (rc != ACL_SUCCESS) {
    std::printf("  [FAIL] %s\n", AclFail("aclrtSetDevice(0)", rc).c_str());
    return 1;
  }
  aclrtStream stream = nullptr;
  rc = aclrtCreateStream(&stream);
  if (rc != ACL_SUCCESS) {
    std::printf("  [FAIL] %s\n", AclFail("aclrtCreateStream", rc).c_str());
    (void)aclrtResetDevice(0);
    return 1;
  }

  // The seed is printed so any failure is reproducible (§4.5).
  constexpr std::uint32_t kSeedBase = 0xB10CB10Cu;
  for (std::uint32_t i = 0; i < sizeof(kModelWidths) / sizeof(kModelWidths[0]);
       ++i) {
    VerifyWidth(kModelWidths[i], kSeedBase + i * 0x9E3779B9u, stream);
  }
  VerifyRejections(stream);

  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(0);

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("FAILED\n");
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
