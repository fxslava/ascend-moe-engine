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
