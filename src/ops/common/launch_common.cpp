// -----------------------------------------------------------------------------
// launch_common.cpp -- the one definition of the launch-layer failure record
// and of Status::ToString, shared by the five op_host launchers.
// -----------------------------------------------------------------------------

#include "launch_common.hpp"

namespace moe::kernels::arch35 {
namespace {

// Not thread_local, deliberately: BiSheng's ASC frontend is not required to
// support the TLS model here, and the v1 launch layer is already serialized by
// its per-call aclrtSynchronizeStream, so a single record is the honest
// representation. Concurrent launches from several host threads would race on
// it; that is documented in launch.hpp rather than papered over, because the
// v1 wrappers are not safe to call concurrently on one stream anyway.
LaunchFailure g_last_failure{};

}  // namespace

namespace detail {

void RecordAclFailure(const char* op, const char* call, int acl_error) noexcept {
  g_last_failure.op = op;
  g_last_failure.call = call;
  g_last_failure.acl_error = acl_error;
}

void ClearLaunchFailure() noexcept { g_last_failure = LaunchFailure{}; }

std::uint32_t QueryAivCoreCount() noexcept {
  // The capability is per device, so the current device must be known. In a
  // process that has not called aclrtSetDevice there is nothing to ask about,
  // which is a 0 rather than an error: the caller's fallback is correct.
  std::int32_t device = 0;
  aclError rc = aclrtGetDevice(&device);
  if (rc != ACL_SUCCESS) {
    RecordAclFailure("QueryAivCoreCount", "aclrtGetDevice", static_cast<int>(rc));
    return 0u;
  }
  // ACL_DEVICE_INFO_VECTOR_CORE_NUM is the AIV count, as distinct from
  // ACL_DEVICE_INFO_AI_CORE_NUM, which is the Cube count. Asking for the
  // latter and using it as a pure-AIV grid is exactly the halving this
  // function exists to stop.
  std::int64_t value = 0;
  rc = aclGetDeviceCapability(static_cast<std::uint32_t>(device),
                              ACL_DEVICE_INFO_VECTOR_CORE_NUM, &value);
  if (rc != ACL_SUCCESS) {
    RecordAclFailure("QueryAivCoreCount", "aclGetDeviceCapability(VECTOR_CORE_NUM)",
                     static_cast<int>(rc));
    return 0u;
  }
  if (value <= 0 || value > 1024) {
    return 0u;  // nonsense reading; let the caller take its fallback
  }
  return static_cast<std::uint32_t>(value);
}

}  // namespace detail

const LaunchFailure& LastLaunchFailure() noexcept { return g_last_failure; }

const char* ToString(Status status) noexcept {
  switch (status) {
    case Status::success:
      return "success";
    case Status::invalid_argument:
      return "invalid_argument";
    case Status::unsupported_shape:
      return "unsupported_shape";
    case Status::launch_error:
      return "launch_error";
  }
  return "unknown";
}

}  // namespace moe::kernels::arch35
