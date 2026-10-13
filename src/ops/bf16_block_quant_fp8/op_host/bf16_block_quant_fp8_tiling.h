#ifndef BF16_BLOCK_QUANT_FP8_TILING_H
#define BF16_BLOCK_QUANT_FP8_TILING_H

// Tiling POD and v1 contract geometry for bf16_block_quant_fp8: BF16 ->
// FP8 E4M3 with one FP32 scale per 32 elements.
//
// WHY THIS OPERATOR EXISTS
// The five activation quantizer sites in apps/dsv4_semi_e2e_bench_npu.cpp
// (quant_h, quant_q, quant_attn, quant_o, shared_quant) each ran
// aclnnDynamicMxQuant followed by an aclnnCast, because DynamicMxQuant emits
// E8M0 scales while the QuantBatchMatmulV3 binary this graph binds to takes
// pertoken_scale in FP32. On the 950PR that pair failed on hardware with
// EZ1009 ("SoC version ascend950 verification failed ... not configured
// through the AddConfig API of the OpDef class"). This operator replaces BOTH
// stages with one dispatch, so it is a launch REDUCTION, not an addition.
//
// SCALES ARE FP32 AND NOT POWERS OF TWO -- A DELIBERATE DIVERGENCE
// DynamicMxQuant produced OCP microscaling E8M0 scales, i.e. exact powers of
// two. This operator emits scale = blockmax/448 in FP32, which is what the
// ND_ND_fp8e4m3_fp8e4m3_fp32_bf16 binary wants directly. Arbitrary FP32
// scales use the E4M3 range strictly better than a power-of-two scale does
// (no up-to-2x headroom thrown away), so quantization error goes DOWN -- but
// the numbers differ from the old path, so golden parity shifts and is not a
// regression. State this when comparing against any figure taken before.
//
// Included by both op_kernel/ and op_host/; no <acl/acl.h> (CLAUDE.md §2.4).

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

// One scale per 32 elements. Matches kRoutedScaleBlock in
// include/moe/core/config.hpp, which is what the consuming matmul's
// `groupSize` describes; the two must not drift apart.
inline constexpr std::uint32_t kBlockQuantBlockSize = 32;

// SIMT geometry: a 32-element scale block is owned by 16 lanes, each handling
// a float2. 16 is forced by the hardware conversion primitive -- the only
// FP8 E4M3 encoder in this toolkit is
// __asc_cvt_float2_to_fp8x2(float2, saturation, interpretation) in
// simt_api/asc_fp8.h, which converts exactly TWO floats per call. One lane
// per element would leave half the conversion slots idle and need a second
// reduction width; one lane per pair makes the block reduction the same
// aligned 16-lane butterfly (xor masks 1,2,4,8) that mxfp4_expert_gemv and
// dsv4_moe_expert already use.
inline constexpr std::uint32_t kBlockQuantLanesPerBlock = 16;
inline constexpr std::uint32_t kBlockQuantBlocksPerTile = 32;
inline constexpr std::uint32_t kBlockQuantElemsPerTile =
    kBlockQuantBlockSize * kBlockQuantBlocksPerTile;  // 1024
inline constexpr std::uint32_t kBlockQuantBlockDim = 32;  // 32 AIVs

// Largest width v1 admits. The widest live site is kAttnQkvWidth = 32,768
// (64 heads x 512); 65,536 leaves a doubling of headroom without letting a
// caller-supplied extent run away. Anything larger is unsupported_shape, not
// invalid_argument: it is geometry v1 does not implement (§2.6).
inline constexpr std::uint32_t kBlockQuantMaxWidth = 65536;

// This operator needs no GM scratch: every tile is loaded, quantized and
// stored within one grid-stride iteration, and the cores own disjoint tiles.
inline constexpr std::size_t kBlockQuantWorkspaceBytes = 0;

// ---------------------------------------------------------------------------
// Dynamic UB size (dyn_ub_size, the second chevron argument)
// ---------------------------------------------------------------------------
// Everything is allocated from the DYNAMIC region through LocalMemAllocator,
// static UB is 0 B, and there is no TPipe/TBuf. Two reasons, both learned
// the hard way in this tree: a buffer a SIMT VF addresses must live in the
// region the host declares at launch or it faults (error 341 / aclError
// 507035), and the 950PR notes call mixing TPipe with LocalMemAllocator
// undefined, so this kernel uses exactly one mechanism.
//
// Allocation order in Init() IS the layout and MUST match this sum:
//   xBf16   1024 bf16    2,048 B   MTE2 staging
//   xF32    1024 floats  4,096 B   widened, read by the VF
//   qOut    1024 bytes   1,024 B   E4M3 result, written by the VF
//   scaleOut  32 floats    128 B   per-block FP32 scale, written by the VF
//
// Fixed at 7,296 B regardless of width, because the tile is fixed and a
// narrow width simply runs fewer tiles. That is why this returns no
// width parameter.
inline constexpr std::uint64_t BlockQuantDynUbBytes() noexcept {
  return static_cast<std::uint64_t>(kBlockQuantElemsPerTile) * 2u +
         static_cast<std::uint64_t>(kBlockQuantElemsPerTile) * 4u +
         static_cast<std::uint64_t>(kBlockQuantElemsPerTile) * 1u +
         static_cast<std::uint64_t>(kBlockQuantBlocksPerTile) * 4u;
}

inline constexpr std::uint32_t kBlockQuantDynUbCeilingBytes = 221184u;  // 216 KiB

// ---------------------------------------------------------------------------
// sizeof == 32
// ---------------------------------------------------------------------------
struct Bf16BlockQuantFp8TilingData {
  std::uint32_t abi_version;      // 1
  std::uint32_t block_dim;        // 32 AIVs, grid-stride over tiles
  std::uint32_t width;            // elements; multiple of 32
  std::uint32_t block_size;       // 32
  std::uint32_t elems_per_tile;   // 1024
  std::uint32_t blocks_per_tile;  // 32
  std::uint32_t num_tiles;        // ceil(width / 1024)
  std::uint32_t reserved;         // 0
};

MOE_ARCH35_ASSERT_TILING_POD(Bf16BlockQuantFp8TilingData, 32);

Status MakeBf16BlockQuantFp8Tiling(
    std::uint32_t width, Bf16BlockQuantFp8TilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // BF16_BLOCK_QUANT_FP8_TILING_H
