#ifndef DSV4_MOE_PREFILL_CUBE_TILING_H
#define DSV4_MOE_PREFILL_CUBE_TILING_H

// Tiling POD and contract for the Cube-accelerated batched prefill entry
// point:
//
//   dsv4_moe_prefill_cube   M >> 1, one expert per token, grouped by expert,
//                           W1/W3/W2 on the Cube in BF16, TWO-PHASE and with
//                           NO GM INTERMEDIATE WORKSPACE
//
// Included by both op_kernel/ and op_host/; no <acl/acl.h> (CLAUDE.md §2.4).
//
// WHY THIS IS A SEPARATE ENTRY POINT AND NOT A MODE OF dsv4_moe_block
// The AIV prefill path in dsv4_moe_block_kernel.cpp decodes MXFP4 nibbles in
// SIMT registers and reduces them with a warp butterfly. This one hands whole
// GEMMs to the Cube. They share no inner loop, no UB layout and no grid, so
// `mode` on one POD would select between two unrelated kernels -- the shape
// §2.5 exists to prevent. The two POD contracts stay separate, as does the
// `act` residency argument each rests on.
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
//
// ---------------------------------------------------------------------------
// THE TWO-PHASE SCHEDULE, AND THE ABI BREAK THAT PAID FOR IT
// ---------------------------------------------------------------------------
// THIS POD IS 72 BYTES AND WAS 64. That is a deliberate ABI break (§2.2) and
// it is the visible half of a schedule change, so it is recorded rather than
// migrated silently. The previous version carried ONE N chunk per GEMM
// (`inter_tile`, `hidden_tile`) because each GEMM reduced over its whole axis
// in a single call, reading dequantized BF16 weight planes out of a GM
// workspace. This version splits BOTH axes of BOTH phases, so it needs four
// sub-tile extents instead of two, and it has no weight planes at all.
//
// WHAT CHANGED, AND WHY EACH PIECE HAD TO
//
//   1. ZERO GM INTERMEDIATE. The old schedule dequantized a whole expert's
//      W1/W3/W2 into GM -- 48 MiB in BF16 at 4096x2048, four times the MXFP4
//      bytes -- and the Cube streamed them back. That workspace is gone: each
//      weight SUB-TILE is decoded on the Vector side straight into UB and
//      pushed to L1 (TPosition::TSCM), and the Cube never reads a weight from
//      GM. The weight traffic is now the 12 MiB of MXFP4 and nothing else.
//      `act` never reaches GM either; it is UB-resident across the phase
//      boundary, which is what the whole budget is built around.
//
//   2. GATE AND UP ARE EVALUATED ONCE. The host golden's trace asserts
//      gate_up_macs == 2 * M * inter * hidden exactly. The predecessor
//      schedule computed them inside a hidden-tile loop and paid
//      ceil(hidden / H_sub) = 64x for it at the production geometry.
//
//   3. THE SUB-GEMM IS THE UNIT. A `TPosition::TSCM` B operand must be the
//      WHOLE single-core B matrix, so splitting K means each (r0, k0) step is
//      its own single-core GEMM. Hence four extents: phase 1 contracts
//      `hidden` in K_SUB blocks to produce R_SUB columns of `act`; phase 2
//      contracts `inter` in R_SUB2 blocks to produce H_SUB columns of Y.
//
//   4. CONSEQUENCE, STATED BECAUSE IT IS A DEVIATION FROM THE IDEALIZED
//      SCHEDULE: L0C cannot hold the level-2 subtotal. It does not survive
//      between separately driven single-core GEMMs, and `IterateAll`'s
//      accumulating form is GM-atomic, which rule 1 forbids. The subtotals
//      are therefore summed in an FP32 UB accumulator. The ARITHMETIC is
//      unchanged -- fractal, then sub-tile block, then axis, FP32 at every
//      level -- which is exactly what the emu golden models, so the oracle
//      comparison is unaffected. Only the buffer differs.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kPrefillCubeGroupSize = 32;

inline constexpr float kPrefillCubeSwigluLimit = 10.0F;

// ---------------------------------------------------------------------------
// Sub-tile shape, and the budget that fixes it
// ---------------------------------------------------------------------------
// DaVinci v3 per-core capacities, from the toolkit's own platform configs
// (x86_64-linux/data/platform_config/Ascend950PR_*.ini), not from the
// whitepaper's per-subsystem figures:
//
//     L1          524,288 B   (512 KiB)   shared with the paired Cube
//     L0A / L0B    65,536 B   (64 KiB each)
//     L0C         262,144 B   (256 KiB)
//     UB          253,952 B   (248 KiB)  <-- PER VECTOR CORE
//
// The whitepaper's "512 KB Unified Buffer" is per AI subsystem, and a
// subsystem holds TWO vector cores. Budgeting 512 KiB of UB per AIV would
// over-commit it by 2x. The repo's kUsableUbBytes (221,184 B) already encodes
// the smaller figure, further less the compiler reserve and the minimum SIMT
// DataCache, and this operator budgets against that.
//
// M_TILE = 16 TOKENS, because act[16, 2048] in BF16 is EXACTLY 64 KiB and it
// has to stay resident from the last gate/up GEMM to the first down GEMM. At
// M_TILE = 32 it is 128 KiB, and with the FP32 accumulators and the dequant
// staging also doubling or staying put the layout no longer leaves the Matmul
// framework room. The full table is in
// op_kernel/dsv4_moe_prefill_cube_arch35.h, which is where it belongs (§1.2);
// what the host needs is the extents it validates and declares.
inline constexpr std::uint32_t kPrefillCubeMTile = 16;

// PHASE 1. R_SUB is the `inter`-axis OUTPUT sub-tile -- how many columns of
// `act` one pass produces; K_SUB is the `hidden`-axis CONTRACTION sub-tile.
// Their product is what L0B charges, and at 128 x 128 BF16 with double
// buffering that is exactly 64 KiB -- the maximum, not merely a fit.
inline constexpr std::uint32_t kPrefillCubeRSub = 128;
inline constexpr std::uint32_t kPrefillCubeKSub = 128;

// PHASE 2. H_SUB is the `hidden`-axis OUTPUT sub-tile; R_SUB2 is the
// `inter`-axis CONTRACTION sub-tile. 64 x 256 is the same 16,384 elements as
// phase 1's sub-tile, which is why one UB staging buffer serves both phases.
inline constexpr std::uint32_t kPrefillCubeHSub = 64;
inline constexpr std::uint32_t kPrefillCubeRSub2 = 256;

static_assert(kPrefillCubeRSub * kPrefillCubeKSub ==
                  kPrefillCubeHSub * kPrefillCubeRSub2,
              "the two phases' weight sub-tiles must hold the same number of "
              "elements, or they cannot share one dequantization staging "
              "buffer and the UB table in the kernel's layout header is wrong");

// C0 for BF16 is 16 ELEMENTS (32 B), not 64.
//
// 64 is the C0 of int4b_t / fp4x2_e2m1_t / fp4x2_e1m2_t. Since this path
// dequantizes to BF16 before the Cube sees anything, 16 is the alignment that
// binds, and a tile shape aligned only to 64 elements would be over-
// constrained rather than wrong. Source: the 950PR C0 ladder, where
// float/int32 is 8, half/bfloat16_t is 16, the b8 family is 32 and the b4
// family is 64.
inline constexpr std::uint32_t kPrefillCubeC0Bf16 = 16;

static_assert(kPrefillCubeRSub % kPrefillCubeC0Bf16 == 0 &&
                  kPrefillCubeKSub % kPrefillCubeC0Bf16 == 0 &&
                  kPrefillCubeHSub % kPrefillCubeC0Bf16 == 0 &&
                  kPrefillCubeRSub2 % kPrefillCubeC0Bf16 == 0,
              "every sub-tile extent must be a whole number of BF16 C0 blocks");
// C at VECIN in ND requires a 32-byte-aligned last axis; FP32 x 64 is 256 B.
static_assert((kPrefillCubeRSub * 4u) % 32u == 0 &&
                  (kPrefillCubeHSub * 4u) % 32u == 0,
              "a VECIN ND C tile needs its last axis 32-byte aligned");
// The packed-nibble sub-tile is read as a STRIDED BURST whose block length is
// counted in 32-byte units, so half a row of K_SUB nibbles must be a whole
// number of them. K_SUB = 128 gives 64 B = 2 blocks.
static_assert((kPrefillCubeKSub / 2u) % 32u == 0 &&
                  (kPrefillCubeRSub2 / 2u) % 32u == 0,
              "half a sub-tile row of packed nibbles must be a whole number of "
              "32-byte burst blocks");
// Both contraction sub-tiles must be whole MXFP4 groups, or a staged sub-tile
// would end mid-group and the scale index would run past it.
static_assert(kPrefillCubeKSub % kPrefillCubeGroupSize == 0 &&
                  kPrefillCubeRSub2 % kPrefillCubeGroupSize == 0,
              "a contraction sub-tile must be a whole number of MXFP4 groups");

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
//
// The test override admits any multiple of 512 on both axes. 512 is the
// smallest value that is a whole multiple of every sub-tile extent above
// (R_SUB 128, K_SUB 128, H_SUB 64, R_SUB2 256), which is what keeps the
// kernel's sub-GEMM loops square on the reduced geometries the CAModel gate
// can afford to simulate.
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
// sizeof == 72
// ---------------------------------------------------------------------------
// WAS 64. See the ABI note at the head of this header: the four sub-tile
// extents replaced two N chunks, which is +8 bytes, and the change is
// deliberate rather than a field snuck into padding. There is no `reserved`
// word to grow into and none is added -- a reserved field that exists only to
// absorb the next break invites breaking the layout quietly.
struct Dsv4MoePrefillCubeTilingData {
  std::uint32_t abi_version;               // 1
  std::uint32_t block_dim;                 // token tiles in flight; <= aiv
  std::uint32_t hidden;                    // 4096
  std::uint32_t inter;                     // 2048
  std::uint32_t num_experts;               // total experts in the buffers
  std::uint32_t total_tokens;              // M
  std::uint32_t m_tile;                    // kPrefillCubeMTile
  std::uint32_t r_sub;                     // phase 1 inter OUTPUT sub-tile
  std::uint32_t k_sub;                     // phase 1 hidden CONTRACTION
  std::uint32_t h_sub;                     // phase 2 hidden OUTPUT sub-tile
  std::uint32_t r_sub2;                    // phase 2 inter CONTRACTION
  std::uint32_t group_size;                // 32
  std::uint32_t low_nibble_first;          // 1; reject 0
  std::uint32_t w13_packed_expert_stride;  // inter * hidden/2 bytes
  std::uint32_t w13_scale_expert_stride;   // inter * hidden/32 bytes
  std::uint32_t w2_packed_expert_stride;   // hidden * inter/2 bytes
  std::uint32_t w2_scale_expert_stride;    // hidden * inter/32 bytes
  float swiglu_limit;                      // 10.0
};

MOE_ARCH35_ASSERT_TILING_POD(Dsv4MoePrefillCubeTilingData, 72);

// ---------------------------------------------------------------------------
// Workspace -- CALLER-OWNED, and now only the KFC message queue
// ---------------------------------------------------------------------------
// It used to carry three data planes: X cast to BF16, then W1+W3 of the
// expert being processed, then W2 -- 48 MiB of dequantized weights at the
// production geometry, because the whole single-core B matrix a
// `TPosition::TSCM` operand requires could not fit 512 KiB of L1.
//
// THE SUB-GEMM REMOVED THAT REQUIREMENT RATHER THAN SATISFYING IT. Splitting
// K makes each (r0, k0) step its own single-core GEMM, whose B is one
// [R_SUB, K_SUB] sub-tile -- 32 KiB in BF16, which L1 holds comfortably. So
// the weights are dequantized per sub-tile into UB, pushed UB -> L1, and the
// Cube reads them from there. Nothing dequantized is ever written to GM, and
// X is cast per sub-tile in UB rather than into a plane.
//
// What is LEFT is the Matmul framework's own requirement:
// REGIST_MATMUL_OBJ stands up a KfcServer whose message queue lives at the
// head of the workspace, so a MIX Matmul kernel cannot run without one at
// all. That is the entire remaining size.
//
// THE SIZE HELPER IS PUBLIC, AND DELIBERATELY NOT HERE.
// Dsv4MoePrefillCubeWorkspaceBytes() lives in
// include/moe/kernels/arch35/launch.hpp because the CALLER allocates this
// buffer. This header is operator-private (§1.2) and is not on the npu or
// bench include path, so a definition here is unreachable by exactly the
// callers that must allocate -- the build break 85f7e06 fixed for the decode
// workspace, not repeated here. The kernel's own copy of the reserve constant
// is kCubeKfcReserveBytes in op_kernel/dsv4_moe_prefill_cube_arch35.h.
Status MakeDsv4MoePrefillCubeTiling(std::uint32_t hidden, std::uint32_t inter,
                                    std::uint32_t num_experts,
                                    std::uint32_t total_tokens,
                                    std::uint32_t aiv_count,
                                    Dsv4MoePrefillCubeTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_PREFILL_CUBE_TILING_H
