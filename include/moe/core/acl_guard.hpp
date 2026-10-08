#pragma once

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdio>
#include <exception>
#include <memory>
#include <utility>

namespace ascend_moe {

/// Initialize the process-owned ACL runtime once; finalize at normal process exit.
/// Externally initialized runtimes remain owned by their caller.
void EnsureAclRuntime();
/// Install signal handlers that request cooperative cancellation at ACL boundaries.
void InstallProcessGuards();
/// Throw on SIGINT/SIGTERM in ordinary thread context, never in a signal handler.
void CheckForInterrupt();
/// Log cleanup failures without throwing or replacing the primary exception.
void ReportAclCleanup(const char* call, int status) noexcept;

template <typename T, auto Destroy>
class AclHandleGuard {
 public:
  explicit AclHandleGuard(T value = nullptr) noexcept : value_(value) {}
  ~AclHandleGuard() { reset(); }
  AclHandleGuard(const AclHandleGuard&) = delete;
  AclHandleGuard& operator=(const AclHandleGuard&) = delete;
  T get() const noexcept { return value_; }
  T release() noexcept { return std::exchange(value_, nullptr); }
  void reset(T value = nullptr) noexcept {
    if (value_) ReportAclCleanup("ACL handle destruction", Destroy(value_));
    value_ = value;
  }

 private:
  T value_;
};

using AclContextGuard = AclHandleGuard<aclrtContext, aclrtDestroyContext>;
using AclEventGuard = AclHandleGuard<aclrtEvent, aclrtDestroyEvent>;
using OpExecutorGuard = AclHandleGuard<aclOpExecutor*, aclDestroyAclOpExecutor>;
using DeviceMemoryGuard = AclHandleGuard<void*, aclrtFree>;
using HostMemoryGuard = AclHandleGuard<void*, aclrtFreeHost>;

/** @brief Own a stream and drain it before destruction.
 * @warning A finite synchronization timeout bounds the wait for queued work;
 * it cannot bound a wedged driver API itself. Failed drains use force destruction.
 */
class AclStreamGuard {
 public:
  explicit AclStreamGuard(aclrtStream stream = nullptr) noexcept : stream_(stream) {}
  ~AclStreamGuard() { reset(); }
  AclStreamGuard(const AclStreamGuard&) = delete;
  AclStreamGuard& operator=(const AclStreamGuard&) = delete;
  aclrtStream get() const noexcept { return stream_; }
  void Drain() noexcept;
  void reset() noexcept;

 private:
  aclrtStream stream_;
};

/// Catch at the outermost application boundary so all inner scopes unwind first.
template <typename F>
int GuardedMain(F&& run) {
  try {
    InstallProcessGuards();
    return run();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ascend-moe: %s\n", error.what());
  } catch (...) {
    std::fputs("ascend-moe: unknown exception\n", stderr);
  }
  return 1;
}
}  // namespace ascend_moe
