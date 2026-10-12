// -----------------------------------------------------------------------------
// status.hpp -- the arch35 launch-layer result code and storage-bit typedefs,
// with NO dependency on <acl/acl.h>.
//
// Split out of launch.hpp so that the per-op op_host/<op>_tiling.h headers can
// declare a validating tiling builder returning `Status` while remaining safe
// to include from a device (ASC) translation unit, which must not see the ACL
// runtime headers. launch.hpp includes this and adds the ACL-dependent part.
// -----------------------------------------------------------------------------

#ifndef MOE_KERNELS_ARCH35_STATUS_HPP
#define MOE_KERNELS_ARCH35_STATUS_HPP

#include <cstdint>

namespace moe::kernels::arch35 {

// Storage-bit typedefs, exactly as the contract spells them.
using Bf16 = std::uint16_t;      // round-to-nearest-even on store
using Fp8E4M3 = std::uint8_t;    // storage bits
using Mxfp4Pair = std::uint8_t;  // low nibble = even logical column
using E8M0 = std::uint8_t;       // one exponent scale code

enum class Status : std::uint32_t {
  success = 0,
  invalid_argument = 1,
  unsupported_shape = 2,
  launch_error = 3,
};

// Human-readable form, for test output and logs.
const char* ToString(Status status) noexcept;

}  // namespace moe::kernels::arch35

#endif  // MOE_KERNELS_ARCH35_STATUS_HPP
