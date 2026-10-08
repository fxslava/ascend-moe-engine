#include "moe/core/acl_guard.hpp"

#include <aclnn/aclnn_base.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>

#include "moe/core/error.hpp"

namespace ascend_moe {
namespace {
volatile std::sig_atomic_t interrupted = 0;
void Interrupt(int signal) { interrupted = signal; }
void Fatal(int signal) {
  // ACL, C++ exceptions, allocation and stdio are NOT async-signal-safe.
  constexpr char message[] = "ascend-moe: fatal signal; driver resources require process reclamation\n";
  const auto written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)written;
  std::_Exit(128 + signal);
}
struct RuntimeLifetime {
  bool owned = false;
  RuntimeLifetime() {
    const auto status = aclInit(nullptr);
    if (status == ACL_ERROR_REPEAT_INITIALIZE) return;
    if (status != ACL_SUCCESS) throw AclError("aclInit", __FILE__, __LINE__, status);
    const auto nn_status = aclnnInit(nullptr);
    if (nn_status != 0) {
      // Capture recent error before cleanup can overwrite it.
      AclError error("aclnnInit", __FILE__, __LINE__, nn_status);
      ReportAclCleanup("aclFinalize", aclFinalize());
      throw error;
    }
    owned = true;
  }
  ~RuntimeLifetime() {
    if (!owned) return;
    // Manual sections 4.39-4.40: once per process, after all device owners.
    ReportAclCleanup("aclnnFinalize", aclnnFinalize());
    ReportAclCleanup("aclFinalize", aclFinalize());
  }
};
}  // namespace

void EnsureAclRuntime() { static RuntimeLifetime runtime; }
void InstallProcessGuards() {
  std::signal(SIGINT, Interrupt);
  std::signal(SIGTERM, Interrupt);
  std::signal(SIGSEGV, Fatal);
  std::signal(SIGABRT, Fatal);
  std::set_terminate([] { Fatal(SIGABRT); });
}
void CheckForInterrupt() {
  if (interrupted) throw Dsv4Error("interrupted; unwinding device resources");
}
void ReportAclCleanup(const char* call, int status) noexcept {
  if (status == 0) return;
  const char* recent = aclGetRecentErrMsg();
  std::fprintf(stderr, "ascend-moe cleanup: %s: %d%s%s\n", call, status, recent ? ": " : "",
               recent ? recent : "");
}
void AclStreamGuard::Drain() noexcept {
  if (!stream_) return;
  const auto status = aclrtSynchronizeStreamWithTimeout(stream_, 5000);
  if (status != ACL_SUCCESS) {
    ReportAclCleanup("aclrtSynchronizeStreamWithTimeout", status);
    ReportAclCleanup("aclrtDestroyStreamForce", aclrtDestroyStreamForce(stream_));
    stream_ = nullptr;
  }
}
void AclStreamGuard::reset() noexcept {
  Drain();
  if (stream_) ReportAclCleanup("aclrtDestroyStream", aclrtDestroyStream(stream_));
  stream_ = nullptr;
}
}  // namespace ascend_moe
