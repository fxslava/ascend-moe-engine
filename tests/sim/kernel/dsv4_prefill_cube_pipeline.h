#ifndef DSV4_PREFILL_CUBE_PIPELINE_H
#define DSV4_PREFILL_CUBE_PIPELINE_H

// SIM ONLY -- STEP 2: the shapes, flags and wire format of the three-stage
// asynchronous pipeline probe. Shared by the device kernel, the launch
// wrapper and the host checker so no two can disagree about a size.
//
// ACL-free and device-header-free: the host TU includes it with a stock g++.
//
// ---------------------------------------------------------------------------
// WHAT STEP 2 ADDS OVER STEP 1
// ---------------------------------------------------------------------------
// STEP 1 (test_fixpipe_dualdst_kernel.cpp) proved ONE Cube step: a single
// GM -> L1 -> L0 -> Mmad -> Fixpipe(DUAL_DST_SPLIT_M, bf16) chain delivers
// rows 0..M/2-1 to AIV0 and M/2..M-1 to AIV1. It computed one K block and
// moved the result out immediately.
//
// STEP 2 keeps that epilogue and puts a pipelined K loop in front of it:
//
//   AIV   MTE2 prefetches weight tile k+1's MXFP4 nibbles while the Vector
//         unit dequantizes tile k to BF16, then pushes tile k UB -> L1 with
//         an inline ND2NZ conversion.
//   AIC   consumes L1 slot k into L0B, loads the matching A tile into L0A,
//         and Mmads into L0C ACCUMULATING -- ccubeInitVal is true only for
//         k == 0.
//
// THE ACCUMULATION INVARIANCE IS THE POINT. Every partial sum stays in L0C
// in FP32 for the whole K axis, and the float -> bfloat16_t Fixpipe runs
// EXACTLY ONCE, on a completed contraction. That is what makes the in-line
// cast lossless: casting per-K-block partials, as a schedule that evicted
// each sub-GEMM to a UB accumulator would have to, injects a BF16 rounding
// into every one of them before they are summed. It also deletes the FP32 UB
// accumulator entirely.
//
// The high-level Matmul framework cannot express this -- it owns L0C and
// retires it per sub-GEMM -- which is the second reason this path is built
// from Tensor API primitives.
//
// ---------------------------------------------------------------------------
// SHAPES
// ---------------------------------------------------------------------------
// Deliberately small: this is a cycle-accurate simulator probe, and the
// question it answers (does the schedule retire with the right numbers) does
// not get a better answer from a bigger tile. kPipeTiles = 4 is the smallest
// trip count that exercises a prologue, a steady-state main loop and a
// two-tile epilogue drain against a depth-2 ring.

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kPipeM     = 16;   // split 8 / 8 by the Fixpipe
inline constexpr std::uint32_t kPipeN     = 128;  // Cube N; also W's row count
inline constexpr std::uint32_t kPipeKSub  = 128;  // one K tile
inline constexpr std::uint32_t kPipeTiles = 4;    // K = 512 in total

inline constexpr std::uint32_t kPipeK     = kPipeKSub * kPipeTiles;
inline constexpr std::uint32_t kPipeMHalf = kPipeM / 2;

// Each AIV dequantizes HALF the weight tile's COLUMNS (the N axis). This is
// the "never run identical SPMD allocations across both AIVs" rule:
// duplicating the dequant would double the vector work and buy nothing,
// since both halves land in the same shared L1 slot.
inline constexpr std::uint32_t kPipeNHalf = kPipeN / 2;
static_assert(kPipeN % 2 == 0, "the weight rows must split evenly across the AIVs");

inline constexpr std::uint32_t kPipeGroup = 32;  // MXFP4 group
static_assert(kPipeKSub % kPipeGroup == 0, "a K tile must be whole MXFP4 groups");

inline constexpr std::uint32_t kPipeRingSlots = 2;

// --- GM layouts -----------------------------------------------------------
// A IS TILE-MAJOR: [tiles][M][K_SUB], not [M][K].
//
// So the AIC's per-tile GM -> L1 copy reads a CONTIGUOUS block and the ND
// source layout needs no row stride. A [M][K] layout would make tile k a
// strided window, which CopyGM2L1 can express but which adds a second thing
// to get wrong while answering a question about the pipeline.
inline constexpr std::uint32_t kPipeAElems = kPipeTiles * kPipeM * kPipeKSub;

// W IS [K][N], CONTRACTION-MAJOR -- NOT production's [N][K]. That is a
// deliberate probe simplification and it is the fix for the defect the first
// STEP 2 run exposed.
//
// WHAT WENT WRONG WITH [N][K]. The Cube wants B = [K, N] in ZN. With W held
// as [N, K] the natural move is to declare the L1 tensor NZ(N, K) and the
// L0B tensor ZN(K, N), which routes CopyL12L0B to its "transpose copy"
// (LoadDataL12L0BNZ2ZN). It does not do what that name suggests. MEASURED,
// with a selector A making the output a direct readout of B and an
// asymmetric W: every delivered value matched W[m][n], never W[n][m]. The
// NZ -> ZN copy transposes each FRACTAL -- it converts STORAGE between two
// encodings of the SAME logical matrix -- it does not transpose the fractal
// grid, so it cannot turn a logical [N, K] into a logical [K, N].
//
// So the logical transpose has to happen before L1. Holding W as [K][N]
// makes the dequantizer's output already contraction-major, the UB -> L1
// ND2NZ gives NZ(K_SUB, N), and NZ -> ZN then converts storage only, which
// is exactly what it does do. Each AIV owns a COLUMN half of the tile
// instead of a row half.
//
// CONSEQUENCE TO CARRY INTO INTEGRATION: the MXFP4 groups now run along N
// rather than along K. That is self-consistent here -- the dequantizer only
// sees rows and columns -- but production quantizes along the contraction
// axis, so wiring this schedule to the real weights needs the logical
// transpose solved another way (LoadDataWithTranspose, or a dequantizer that
// emits contraction-major). This probe deliberately does not pretend to have
// solved that.
inline constexpr std::uint32_t kPipeWRows = kPipeK;  // contraction axis
inline constexpr std::uint32_t kPipeWCols = kPipeN;  // output axis
inline constexpr std::uint32_t kPipeWPackedBytes = kPipeWRows * kPipeWCols / 2u;
inline constexpr std::uint32_t kPipeWScaleBytes  = kPipeWRows * kPipeWCols / kPipeGroup;

static_assert(kPipeNHalf % kPipeGroup == 0,
              "each AIV's column half must be a whole number of MXFP4 groups");

inline constexpr std::uint32_t kPipeHalfElems = kPipeMHalf * kPipeN;
inline constexpr std::uint32_t kPipeOutElems  = 2u * kPipeHalfElems;

// --- cross-core flags -----------------------------------------------------
// Mode 2 on every set, default template arguments on every wait. The AIV ->
// AIC direction is exercised first by construction: the AIC's first act is to
// wait for slot 0, which is the rendezvous STEP 1 measured to be mandatory
// (see dsv4_prefill_cube_dualdst.h -- a one-way AIC -> AIV flag never fires).
inline constexpr std::uint8_t  kPipeSyncMode = 0x2;

// THE AIC WAITS ONCE PER TILE, NOT ONCE PER AIV, AND THAT IS MEASURED.
//
// Mode 2's AIV -> AIC direction is documented as incrementing the AIC's
// counter only after ALL M AIVs have set, which makes a single wait correct.
// The shipping turboquant kernel carries a probe variant that waits twice
// (kCubeLoadDoubleOperandsWait), so the question was worth settling rather
// than assuming: a second wait HANGS here, past 170,000 simulated cycles and
// a 240 s watchdog. The documented all-M semantics hold, one wait is right,
// and a partially-written L1 slot is NOT explained by the Cube overtaking
// the second AIV.
inline constexpr std::uint16_t kPipeFlagSlotReady = 0;  // + slot : AIV -> AIC
inline constexpr std::uint16_t kPipeFlagSlotFree  = 2;  // + slot : AIC -> AIV
inline constexpr std::uint16_t kPipeFlagProduct   = 4;  //         AIC -> AIV

// --- HOW THE LAYOUT DEFECT WAS FOUND, 2026-10-17 --------------------------
// Kept because the method is reusable and the conclusion is not guessable
// from the API docs.
//
// The first build of this probe delivered the right answer only in column 0.
// The diagnosis came from making the output a READOUT of B rather than a
// product: with a selector A[m][k] = (k == m), C[m][n] IS B[m][n], so the
// operand that reaches L0B can be read straight off the result. Every
// delivered value matched W[m][n] and never W[n][m] -- B was arriving
// untransposed.
//
// That is not a bug in CopyL12L0B. It derives its CopyMode from the layout
// patterns automatically (`noTrans = is_same_v<DstPattern, SrcPattern>` in
// impl/tensor_api/arch/cube/l1_to_l0b/copy.h), so NZ source + ZN destination
// really does route to LoadDataL12L0BNZ2ZN. The copy transposes each
// FRACTAL -- it converts between two storage encodings of the SAME logical
// matrix -- and does NOT transpose the fractal grid, so it cannot turn a
// logical [N, K] into a logical [K, N]. The logical transpose has to happen
// before L1, which is why the weights here are contraction-major.
//
// Two traps the diagnostic itself hit, worth avoiding next time:
//   * the first probe matrix was ((n * 5 + k) % 4) + 1, which is SYMMETRIC
//     in n and k because 5 == 1 (mod 4). It matched both the transposed and
//     the untransposed model and proved nothing.
//   * the turboquant kernel's kCubeLoadDoubleOperandsWait variant suggested
//     the AIC might need one wait per AIV. It does not -- a second
//     CrossCoreWaitFlag hangs. Mode 2's all-AIVs semantics hold as written.

// --- stages ---------------------------------------------------------------
//   1  ONE tile only. Proves the plumbing and the epilogue without the
//      accumulation or the ring.
//   2  all tiles, ring depth 2, but the MXFP4 load and the dequant are issued
//      back to back -- no MTE2 prefetch. Proves L0C accumulation.
//   3  all tiles, and the MTE2 for tile k+1 is issued BEFORE the Vector
//      dequant of tile k. The full three-stage overlap.
inline constexpr std::uint32_t kPipeStageOneTile  = 1;
inline constexpr std::uint32_t kPipeStageAccum    = 2;
inline constexpr std::uint32_t kPipeStagePrefetch = 3;

// --- status block ---------------------------------------------------------
// One 128 B region per writer. A shared cache line loses writes, because
// DataCacheCleanAndInvalid writes back the whole line and the three cores
// would clobber each other -- measured in STEP 1.
inline constexpr std::uint32_t kPipeStatusStride = 32;  // words, = 128 B
inline constexpr std::uint32_t kPipeStatusWords  = 4u * kPipeStatusStride;

inline constexpr std::uint32_t kPipeAicRegion  = 0;
inline constexpr std::uint32_t kPipeAivRegion  = 1;  // + sub-block index
inline constexpr std::uint32_t kPipeHostRegion = 3;

inline constexpr std::uint32_t kPipeSlotLive  = 0;
inline constexpr std::uint32_t kPipeSlotAddr  = 1;  // AIV landing-zone UB addr
inline constexpr std::uint32_t kPipeSlotTiles = 2;  // tiles the core processed

inline constexpr std::uint32_t kPipeStageWord =
    kPipeHostRegion * kPipeStatusStride;

inline constexpr std::uint32_t kPipeAicDone = 0xA1C2DEE5u;
inline constexpr std::uint32_t kPipeAivDone = 0xA1D2DEE5u;

}  // namespace moe::kernels::arch35

#endif  // DSV4_PREFILL_CUBE_PIPELINE_H
