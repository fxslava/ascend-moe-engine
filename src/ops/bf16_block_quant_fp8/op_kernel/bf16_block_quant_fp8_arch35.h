// bf16_block_quant_fp8_arch35.h -- UB layout and event plan for
// bf16_block_quant_fp8 (Ascend 950PR, arch35 / dav-3510, 32 AIVs,
// hybrid SIMD/SIMT).
//
// UB REGION: all DYNAMIC, static is 0 B. No TPipe, no TBuf.
//
// Three of the four buffers are addressed by the SIMT VF, so they must come
// from the dynamic region -- the one whose size the host declares as
// dyn_ub_size in the chevron. The BF16 staging tile is SIMD-only and could
// have been a TBuf, but keeping one allocation mechanism is deliberate: the
// 950PR notes call mixing TPipe/TBuf with LocalMemAllocator undefined, and
// this tree has already paid for a static/dynamic mismatch twice (error 341
// on first SIMT UB access, aclError 507035 in paged_compressor_rope).
//
// Segment table, in LocalMemAllocator allocation order -- the order IS the
// layout, and it must match BlockQuantDynUbBytes() exactly:
//
//   offset   size     segment
//   0        2,048    xBf16     1024 BF16, MTE2 staging
//   2,048    4,096    xF32      1024 FP32, widened; read by the VF
//   6,144    1,024    qOut      1024 E4M3 bytes; written by the VF
//   7,168      128    scaleOut  32 FP32 block scales; written by the VF
//
// Dynamic total: 7,296 B. Static: 0 B. Width-independent -- a narrow tensor
// runs fewer tiles, not smaller ones.
// DataCache = min(256 KiB - 0 - 7,296 - 8 KiB, 128 KiB) = 131,072 B, so the
// SIMT DataCache sits at its 128 KiB cap with room to spare.
//
// ALGORITHM, and why the reduction is in registers
// Per 32-element scale block: 16 lanes each hold a float2. The block max of
// |x| forms through an in-register butterfly (asc_shfl_xor, xor masks 1,2,4,8
// at width 16, which never leaves the aligned 16-lane group), so after four
// shuffles EVERY lane in the block holds the same max. That is the whole
// reason this shape was chosen: the per-element scale is then already in the
// lane that needs it, with no broadcast step at all.
//
// A SIMD formulation was rejected on API grounds, not preference. This
// toolkit's ASC include tree has no WholeReduceMax and no BlockReduceMax
// (verified by grep over x86_64-linux/asc/include), so 32 independent
// 32-element maxima cannot be expressed in one vector call; and the only
// FP8 E4M3 encoder anywhere in the tree is the SIMT-side
// __asc_cvt_float2_to_fp8x2 in simt_api/asc_fp8.h -- there is no float->E4M3
// Cast in the SIMD vconv interface. So the conversion has to happen in a VF
// regardless, and doing the reduction there too costs nothing extra.
//
// Event plan (EVENT_ID0..3 only; arch35 has no EVENT_ID4..7):
//   EVENT_ID0  MTE2 -> V    xBf16 tile arrival
//   EVENT_ID1  V    -> MTE3 qOut / scaleOut store readiness
// Phase boundaries around the SIMT launch use DataSyncBarrier<MemDsbT::ALL>.

#ifndef MOE_KERNELS_ARCH35_BF16_BLOCK_QUANT_FP8_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_BF16_BLOCK_QUANT_FP8_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::bf16_block_quant_fp8 {

inline constexpr std::uint32_t kBlockSize = 32;
inline constexpr std::uint32_t kLanesPerBlock = 16;   // one float2 per lane
inline constexpr std::uint32_t kBlocksPerTile = 32;
inline constexpr std::uint32_t kElemsPerTile = kBlockSize * kBlocksPerTile;

// SIMT launch geometry: 32 blocks x 16 lanes.
inline constexpr std::uint32_t kSimtThreads = kBlocksPerTile * kLanesPerBlock;
static_assert(kSimtThreads == 512,
              "bf16_block_quant_fp8 expects a 512-thread SIMT launch");

// E4M3 finite range. 448 is the largest finite OCP E4M3 magnitude
// (exponent 8, mantissa 0b111 -> 1.75 * 2^8), so scaling the block by
// blockmax/448 maps the block's largest magnitude onto the top of the format
// and leaves no representable range unused.
inline constexpr float kE4M3Max = 448.0F;

inline constexpr std::uint32_t kXBf16Bytes = kElemsPerTile * 2;    // 2,048
inline constexpr std::uint32_t kXF32Bytes = kElemsPerTile * 4;     // 4,096
inline constexpr std::uint32_t kQOutBytes = kElemsPerTile * 1;     // 1,024
inline constexpr std::uint32_t kScaleOutBytes = kBlocksPerTile * 4;  // 128

inline constexpr std::uint32_t kStaticUbBytes = 0;
inline constexpr std::uint32_t kDynUbBytes =
    kXBf16Bytes + kXF32Bytes + kQOutBytes + kScaleOutBytes;

static_assert(kDynUbBytes == 7296,
              "bf16_block_quant_fp8 dynamic UB must sum to 7,296 B");
static_assert(kStaticUbBytes + kDynUbBytes <= kUsableUbBytes,
              "bf16_block_quant_fp8 UB layout exceeds the arch35 usable ceiling");
static_assert(kUbTotalBytes - kUbReservedBytes - kStaticUbBytes -
                      kDynUbBytes >= kUbDataCacheMinBytes,
              "bf16_block_quant_fp8 UB layout starves the SIMT DataCache "
              "below 32 KiB");

inline constexpr std::int32_t kEvMte2ToV = 0;
inline constexpr std::int32_t kEvVToMte3 = 1;

}  // namespace moe::kernels::arch35::bf16_block_quant_fp8

#endif  // MOE_KERNELS_ARCH35_BF16_BLOCK_QUANT_FP8_UB_LAYOUT_HPP
