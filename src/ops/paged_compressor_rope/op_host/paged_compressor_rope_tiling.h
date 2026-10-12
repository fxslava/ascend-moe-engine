#ifndef PAGED_COMPRESSOR_ROPE_TILING_H
#define PAGED_COMPRESSOR_ROPE_TILING_H

// Tiling POD and C4A (coff=2) contract geometry for paged_compressor_rope.
// Included by both op_kernel/ and op_host/; no <acl/acl.h>.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kCompressorHidden = 4096;
inline constexpr std::uint32_t kCompressorHeadDim = 512;
inline constexpr std::uint32_t kCompressorCoff = 2;   // the kernel rejects 1
inline constexpr std::uint32_t kCompressorRatio = 4;  // HOLD/EMIT period
inline constexpr std::uint32_t kCompressorReduceRows =
    kCompressorCoff * kCompressorRatio;  // 8
inline constexpr std::uint32_t kCompressorStateWidthElems =
    kCompressorCoff * kCompressorHeadDim;  // 1024
inline constexpr std::uint32_t kRopeHeadDim = 64;
inline constexpr std::uint32_t kRotaryModeHalf = 1;  // not interleave
inline constexpr float kCompressorNormEps = 1e-6F;

// ---------------------------------------------------------------------------
// Dynamic UB size (dyn_ub_size, the second chevron argument)
// ---------------------------------------------------------------------------
// EVERY buffer this kernel has is allocated from the DYNAMIC UB region via
// LocalMemAllocator. There is no TPipe at all, so the static region is 0 B,
// and that is a correctness requirement rather than a preference.
//
// Two reasons, and they are different failures:
//
//   1. A buffer a SIMT VF dereferences MUST be in the dynamic region, because
//      that is the region whose size the host declares at launch. Hand a VF a
//      static-region pointer while the launch declares 0 dynamic bytes and
//      every SIMT UB access is outside the reserved window -- AIV error 341
//      on the first touch.
//   2. TPipe/TBuf and LocalMemAllocator bump a head pointer from the SAME UB
//      base, so holding a TBuf alongside the allocator makes the TBuf alias
//      the allocator's first segments. This kernel carried two 1 KiB BF16
//      staging TBufs that landed on top of kvCur_ row 0, silently corrupting
//      the persistent state; see the kernel's Init() for the full account.
//      Mixing the two is undefined, and the 950PR docs say so.
//
// The host must therefore compute the same byte count the device allocates,
// so the formula lives here: this header is included by both sides (CLAUDE.md
// section 2.4) and is ACL-free. op_kernel/paged_compressor_rope_arch35.h
// static_asserts its own segment table against this number, so the two
// cannot drift.
//
// LocalMemAllocator bumps a head pointer, so the ALLOCATION ORDER IS THE
// LAYOUT. Init() must allocate in exactly this order:
//   kvCur      ratio*stateWidth floats   16,384 B
//   gateCur    ratio*stateWidth floats   16,384 B
//   prevKvA    ratio*headDim    floats    8,192 B
//   prevGateA  ratio*headDim    floats    8,192 B
//   ape        ratio*stateWidth floats   16,384 B
//   normWF32   headDim          floats    2,048 B
//   ropeCos    ropeHeadDim      floats      256 B
//   ropeSin    ropeHeadDim      floats      256 B
//   outF32     headDim          floats    2,048 B
//   reduce     64               floats      256 B
//   normWBf16  headDim          bf16      1,024 B   (was a TBuf)
//   outBf16    headDim          bf16      1,024 B   (was a TBuf)
inline constexpr std::uint32_t kCompressorReduceSlackFloats = 64;

inline constexpr std::uint32_t kCompressorDynUbBytes =
    2u * kCompressorRatio * kCompressorStateWidthElems * 4u +  // kv + gate cur
    2u * kCompressorRatio * kCompressorHeadDim * 4u +          // prev a halves
    kCompressorRatio * kCompressorStateWidthElems * 4u +       // ape, resident
    kCompressorHeadDim * 4u +                                  // normWF32
    2u * kRopeHeadDim * 4u +                                   // cos + sin
    kCompressorHeadDim * 4u +                                  // outF32
    kCompressorReduceSlackFloats * 4u +                        // reduce slack
    kCompressorHeadDim * 2u +                                  // normWBf16
    kCompressorHeadDim * 2u;                                   // outBf16

static_assert(kCompressorDynUbBytes == 72448,
              "paged_compressor_rope dynamic UB must sum to 72,448 B");

// No TPipe, so the static region is empty. Keep the constant: the kernel's
// layout header asserts against it, which is what would catch a TBuf being
// reintroduced without the dyn_ub figure being updated to match.
inline constexpr std::uint32_t kCompressorStaticUbBytes = 0u;

// static + dynamic must fit 256 KiB - 8 KiB reserved - 32 KiB min DataCache.
static_assert(kCompressorStaticUbBytes + kCompressorDynUbBytes <= 221184u,
              "paged_compressor_rope exceeds the 216 KiB static+dynamic UB ceiling");

// ---------------------------------------------------------------------------
// sizeof == 64
// ---------------------------------------------------------------------------
struct CompressorTilingData {
  std::uint32_t abi_version;          // 1
  std::uint32_t block_dim;            // 1 AIV
  std::uint32_t hidden;               // 4096
  std::uint32_t head_dim;             // 512
  std::uint32_t coff;                 // 2, reject 1
  std::uint32_t ratio;                // 4
  std::uint32_t reduction_rows;       // 8
  std::uint32_t rope_head_dim;        // 64
  std::uint32_t rotary_mode;          // 1 = half, not interleave
  std::uint32_t projection_row_tile;  // 1 row at a time
  std::uint32_t wkv_row_stride;       // 4096 BF16 elements
  std::uint32_t wgate_row_stride;     // 4096 BF16 elements
  std::uint32_t ape_row_stride;       // 1024 FP32 elements
  std::uint32_t reserved;             // 0
  float norm_eps;                     // 1e-6
  float reserved_fp32;                // 0
};

MOE_ARCH35_ASSERT_TILING_POD(CompressorTilingData, 64);

Status MakeCompressorTiling(std::uint32_t wkv_row_stride,
                            std::uint32_t wgate_row_stride,
                            std::uint32_t ape_row_stride,
                            CompressorTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // PAGED_COMPRESSOR_ROPE_TILING_H
