// -----------------------------------------------------------------------------
// tiling_abi.hpp -- what every arch35 tiling POD shares.
//
// The PODs themselves live next to the operator that owns them, in
// src/ops/<op>/op_host/<op>_tiling.h, following the directory pattern set by
// src/ops/mhc_fused_sinkhorn_gemv. This header carries only the parts that
// must agree ACROSS operators: the ABI version both sides check, and the
// layout assertion every POD is required to pass.
//
// Every tiling struct is a standard-layout, trivially-copyable POD of 32-bit
// fields only, so host and device translation units see an identical layout
// with no pointer-width padding. The device kernel reads it through the
// `tiling` GM_ADDR argument; the host copies it into device memory verbatim.
// `reserved` fields must be zero; kernels reject an unknown `abi_version`.
//
// These PODs carry no pointers. Pointer addresses are kernel GM_ADDR
// arguments; the numerical order, not any private UB layout, is normative.
// -----------------------------------------------------------------------------

#ifndef MOE_KERNELS_ARCH35_TILING_ABI_HPP
#define MOE_KERNELS_ARCH35_TILING_ABI_HPP

#include <cstdint>
#include <type_traits>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kTilingAbiVersion = 1;

// Asserted per POD at its definition site. Both properties matter: the fixed
// size is the wire format the device side reads, and standard-layout +
// trivially-copyable is what makes a verbatim memcpy across the host/device
// boundary well-defined rather than merely working today.
#define MOE_ARCH35_ASSERT_TILING_POD(T, bytes)                               \
  static_assert(sizeof(T) == (bytes), #T " tiling must be " #bytes " B");    \
  static_assert(std::is_standard_layout<T>::value &&                         \
                    std::is_trivially_copyable<T>::value,                    \
                #T " tiling must be a standard-layout trivially-copyable POD")

}  // namespace moe::kernels::arch35

#endif  // MOE_KERNELS_ARCH35_TILING_ABI_HPP
