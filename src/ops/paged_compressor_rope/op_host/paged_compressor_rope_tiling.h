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
// Every buffer this kernel's two SIMT VFs dereference is allocated from the
// DYNAMIC UB region via LocalMemAllocator, because the dynamic region is the
// one whose size the host declares at launch. A TPipe/TBuf buffer lands in
// the STATIC region instead, and handing a SIMT VF a static-region pointer
// while the launch declares 0 dynamic bytes puts every SIMT UB access outside
// the reserved window -- the kernel then faults on its first SIMT UB touch.
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
inline constexpr std::uint32_t kCompressorReduceSlackFloats = 64;

inline constexpr std::uint32_t kCompressorDynUbBytes =
    2u * kCompressorRatio * kCompressorStateWidthElems * 4u +  // kv + gate cur
    2u * kCompressorRatio * kCompressorHeadDim * 4u +          // prev a halves
    kCompressorRatio * kCompressorStateWidthElems * 4u +       // ape, resident
    kCompressorHeadDim * 4u +                                  // normWF32
    2u * kRopeHeadDim * 4u +                                   // cos + sin
    kCompressorHeadDim * 4u +                                  // outF32
    kCompressorReduceSlackFloats * 4u;                         // reduce slack

static_assert(kCompressorDynUbBytes == 70400,
              "paged_compressor_rope dynamic UB must sum to 70,400 B");

// The static region holds only the two SIMD-private BF16 staging buffers
// (norm gain in, cache-slot token out); nothing else is allocated from TPipe.
inline constexpr std::uint32_t kCompressorStaticUbBytes =
    kCompressorHeadDim * 2u + kCompressorHeadDim * 2u;  // 2,048

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
