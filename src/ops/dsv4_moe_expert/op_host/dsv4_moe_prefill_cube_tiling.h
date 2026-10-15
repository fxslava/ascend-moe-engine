#ifndef DSV4_MOE_PREFILL_CUBE_TILING_H
#define DSV4_MOE_PREFILL_CUBE_TILING_H

// Tiling POD and contract for the Cube-accelerated batched prefill entry
// point:
//
//   dsv4_moe_prefill_cube   M >> 1, one expert per token, grouped by expert,
//                           W1/W3/W2 on the Cube in BF16
//
// Included by both op_kernel/ and op_host/; no <acl/acl.h> (CLAUDE.md §2.4).
//
// WHY THIS IS A SEPARATE ENTRY POINT AND NOT A MODE OF dsv4_moe_block
// The AIV prefill path in dsv4_moe_block_kernel.cpp decodes MXFP4 nibbles in
// SIMT registers and reduces them with a warp butterfly. This one hands whole
// GEMMs to the Cube. They share no inner loop, no UB layout and no grid, so
// `mode` on one POD would select between two unrelated kernels -- the shape
// §2.5 exists to prevent. The two POD contracts stay separate, as does the
// `activated` residency argument each rests on.
//
// ---------------------------------------------------------------------------
// WHY BF16 AND NOT NATIVE CUBE MXFP4
// ---------------------------------------------------------------------------
// The third-generation Cube does have an MXFP4 tensor path (1730 TFLOPS on
// 950PR against a 27 TFLOPS Vector FP32 ceiling), but `MmadMx` computes
// C = (scaleA ⊗ A) * (scaleB ⊗ B) and its operand table admits ONLY fp4/fp8
// for BOTH A and B -- there is no float or bfloat16_t A-matrix overload.
// VERIFIED against this toolkit, not inferred: a fp4x2_e2m1_t x fp4x2_e2m1_t
// `asc_mmad_mx` compiles for dav-3510, while the same call with a
// `__ca__ float*` A is rejected with "no matching function".
//
// So reaching 1730 TFLOPS would require quantizing the ACTIVATIONS to FP4 --
// a ~2-bit mantissa -- which changes the computed answer and retires the
// weight-only-quantization golden that tests/emu is built on. This entry
// point therefore takes the lossless route:
//
//   * weights MXFP4 -> BF16 is EXACT. E2M1 magnitudes are {0, .5, 1, 1.5, 2,
//     3, 4, 6}, at most two significant mantissa bits, and the E8M0 scale is
//     a power of two; BF16 keeps 8 mantissa bits, so nibble * 2^k is
//     representable exactly wherever it does not leave BF16's exponent range.
//   * the E8M0 group scale COMMUTES with the reduction. The CPU oracle applies
//     one scale to each 32-element group's subtotal; a Cube GEMM can only
//     distribute it into every product. Those differ in floating point in
//     general, but scaling by a power of two is exact and does not touch the
//     mantissa, so here they agree.
//   * the FP32 -> BF16 ACTIVATION cast is the only new error. BF16 keeps an
//     8-bit mantissa, so one store carries up to 2^-9 relative error. That is
//     a W4A16 kernel, and the oracle it is compared against must cast X the
//     same way (§4.7: the bound comes from the format, not from the observed
//     error).
//
// Cube BF16 is 432 TFLOPS. Against 1.6 TB/s the demand is 4*M FLOP/B, so this
// path is bandwidth-bound below M ~ 68 and compute-bound above -- which means
// below M ~ 68 it is not slower than the MXFP4 path would be, because neither
// can beat the memory system there. NEVER MEASURED on hardware (§5.7): this
// is a roofline argument, not a benchmark result.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kPrefillCubeGroupSize = 32;

inline constexpr float kPrefillCubeSwigluLimit = 10.0F;

// ---------------------------------------------------------------------------
// Cube tile shape, and the budget that fixes it
// ---------------------------------------------------------------------------
// DaVinci v3 per-core capacities, from the toolkit's own platform configs
// (x86_64-linux/data/platform_config/Ascend950PR_*.ini), not from the
// whitepaper's per-subsystem figures:
//
//     L1          524,288 B   (512 KiB)
//     L0A / L0B    65,536 B   (64 KiB each)
//     L0C         262,144 B   (256 KiB)
//     UB          253,952 B   (248 KiB)  <-- PER VECTOR CORE
//
// The whitepaper's "512 KB Unified Buffer" is per AI subsystem, and a
// subsystem holds TWO vector cores. Budgeting 512 KiB of UB per AIV would
// over-commit it by 2x. The repo's kUsableUbBytes (221,184 B) already encodes
// the smaller figure and this header budgets against that.
//
// M_TILE = 16 TOKENS, and the UB budget is what picks it. Per token tile the
// Vector side holds, at the production geometry:
//
//     gate chunk   FP32  [16, 512]  =  32,768 B   Cube output at VECIN
//     up   chunk   FP32  [16, 512]  =  32,768 B   Cube output at VECIN
//     activated    BF16  [16, 2048] =  65,536 B   SwiGLU result, feeds W2
//                                      ---------
//                                      131,072 B   of 221,184 B
//
// At M_TILE = 32 that triples to 262,144 B and does not fit. X is NOT in this
// budget: it is cast to BF16 once into a workspace plane and read by the Cube
// from GM, which is what buys the room for `activated` to stay resident
// between the gate/up and down GEMMs.
inline constexpr std::uint32_t kPrefillCubeMTile = 16;

// The gate/up GEMM's N chunk. Chosen so one gate and one up chunk together
// are a quarter of the UB budget, leaving `activated` resident.
inline constexpr std::uint32_t kPrefillCubeInterTile = 512;

// The down GEMM's N chunk, same reasoning on the hidden axis.
inline constexpr std::uint32_t kPrefillCubeHiddenTile = 512;

// C0 for BF16 is 16 ELEMENTS (32 B), not 64.
//
// 64 is the C0 of int4b_t / fp4x2_e2m1_t / fp4x2_e1m2_t. Since this path
// dequantizes to BF16 before the Cube sees anything, 16 is the alignment that
// binds, and a tile shape aligned only to 64 elements would be over-
// constrained rather than wrong. Source: the 950PR C0 ladder, where
// float/int32 is 8, half/bfloat16_t is 16, the b8 family is 32 and the b4
// family is 64.
inline constexpr std::uint32_t kPrefillCubeC0Bf16 = 16;

static_assert(kPrefillCubeInterTile % kPrefillCubeC0Bf16 == 0,
              "the gate/up N chunk must be a whole number of BF16 C0 blocks");
static_assert(kPrefillCubeHiddenTile % kPrefillCubeC0Bf16 == 0,
              "the down N chunk must be a whole number of BF16 C0 blocks");
// C at VECIN in ND requires a 32-byte-aligned last axis; FP32 x 512 is 2 KiB.
static_assert((kPrefillCubeInterTile * 4u) % 32u == 0,
              "a VECIN ND C tile needs its last axis 32-byte aligned");

// Grid: one token tile per core, grid-striding. The AIV count comes from the
// runtime (detail::QueryAivCoreCount), never a literal -- 950PR ships 56-, 64-
// and 72-vector parts. Same resolution rule as the dsv4_moe_block builders.
inline constexpr std::uint32_t kPrefillCubeAivFallback = 56;
inline constexpr std::uint32_t kPrefillCubeAivMax = 256;

// ---------------------------------------------------------------------------
// Admissible geometry
// ---------------------------------------------------------------------------
// Macro plus two same-name wrappers, for the reason recorded on
// Dsv4MoeBlockGeometryAdmissible: BiSheng partitions functions into __host__
// and __aicore__ and will not let one definition serve both.
#if defined(DSV4_TILING_TEST_OVERRIDE) && DSV4_TILING_TEST_OVERRIDE
#define MOE_DSV4_PREFILL_CUBE_ADMISSIBLE(hidden, inter)                      \
  ((hidden) >= 512u && (inter) >= 512u && (hidden) % 512u == 0u &&            \
   (inter) % 512u == 0u)
#else
#define MOE_DSV4_PREFILL_CUBE_ADMISSIBLE(hidden, inter)                      \
  ((hidden) == 4096u && (inter) == 2048u)
#endif

inline constexpr bool Dsv4MoePrefillCubeAdmissible(std::uint32_t hidden,
                                                   std::uint32_t inter) noexcept {
  return MOE_DSV4_PREFILL_CUBE_ADMISSIBLE(hidden, inter);
}

#if defined(__CCE__)
__aicore__ inline bool Dsv4MoePrefillCubeAdmissible(std::uint32_t hidden,
                                                    std::uint32_t inter) {
  return MOE_DSV4_PREFILL_CUBE_ADMISSIBLE(hidden, inter);
}
#endif

// ---------------------------------------------------------------------------
// sizeof == 64
// ---------------------------------------------------------------------------
struct Dsv4MoePrefillCubeTilingData {
  std::uint32_t abi_version;               // 1
  std::uint32_t block_dim;                 // token tiles in flight; <= aiv
  std::uint32_t hidden;                    // 4096
  std::uint32_t inter;                     // 2048
  std::uint32_t num_experts;               // total experts in the buffers
  std::uint32_t total_tokens;              // M
  std::uint32_t m_tile;                    // kPrefillCubeMTile
  std::uint32_t inter_tile;                // kPrefillCubeInterTile
  std::uint32_t hidden_tile;               // kPrefillCubeHiddenTile
  std::uint32_t group_size;                // 32
  std::uint32_t low_nibble_first;          // 1; reject 0
  std::uint32_t w13_packed_expert_stride;  // inter * hidden/2 bytes
  std::uint32_t w13_scale_expert_stride;   // inter * hidden/32 bytes
  std::uint32_t w2_packed_expert_stride;   // hidden * inter/2 bytes
  std::uint32_t w2_scale_expert_stride;    // hidden * inter/32 bytes
  float swiglu_limit;                      // 10.0
};

MOE_ARCH35_ASSERT_TILING_POD(Dsv4MoePrefillCubeTilingData, 64);

// ---------------------------------------------------------------------------
// Workspace -- CALLER-OWNED, and larger than the operand data
// ---------------------------------------------------------------------------
// Three planes, and the reason each exists:
//
//   x_bf16     total_tokens * hidden * 2      X cast once, read by the Cube
//   w13_bf16   2 * inter * hidden * 2         W1 and W3 of ONE expert
//   w2_bf16    hidden * inter * 2             W2 of ONE expert
//
// THE DEQUANTIZED WEIGHTS GO TO GM, NOT L1, AND THAT IS A REAL CONCESSION.
// The intent was to dequantize into L1 and stream L1 -> UB on the direct
// Cube-Vector path with no GM hop. At the production geometry one expert's W1
// alone is inter * hidden * 2 = 16 MiB in BF16, against 512 KiB of L1 and
// 248 KiB of UB, so the whole single-core B matrix cannot be resident -- and
// `TPosition::VECOUT` and TSCM both require exactly that. Reaching the direct
// path needs the GEMM re-driven per staged B tile (a [baseK, baseN] chunk
// dequantized into UB, singleCoreK/N set to match), which is a different
// Matmul driver, not a different buffer position. Recorded here rather than
// implied, because the cost is visible: BF16 weights are 4x the MXFP4 bytes,
// so the weight stream from GM grows 4x.
//
// It is amortized by the batch, which is the point of the prefill path: the
// dequantization is O(weights) and runs ONCE per expert, while the GEMM is
// O(weights * tokens_in_that_expert). With tokens grouped by expert -- the
// caller's arrangement, as for the AIV prefill kernel -- one dequantization
// serves the whole group.
// THE SIZE HELPER IS PUBLIC, AND DELIBERATELY NOT HERE.
// Dsv4MoePrefillCubeWorkspaceBytes() lives in
// include/moe/kernels/arch35/launch.hpp because the CALLER allocates this
// buffer. This header is operator-private (§1.2) and is not on the npu or
// bench include path, so a definition here is unreachable by exactly the
// callers that must allocate -- the build break 85f7e06 fixed for the decode
// workspace, not repeated here.
//
// The public helper also adds the KFC message-queue reserve that this
// operator's MIX Matmul needs at the head of the workspace; the kernel's own
// copy of that constant is kCubeKfcReserveBytes in
// op_kernel/dsv4_moe_prefill_cube_arch35.h.
Status MakeDsv4MoePrefillCubeTiling(std::uint32_t hidden, std::uint32_t inter,
                                    std::uint32_t num_experts,
                                    std::uint32_t total_tokens,
                                    std::uint32_t aiv_count,
                                    Dsv4MoePrefillCubeTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_PREFILL_CUBE_TILING_H
