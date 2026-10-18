// dsv4_moe_prefill_cube_arch35.h -- UB / L1 budget, NZ layout arithmetic, flag
// plan and event plan for the Cube-accelerated batched prefill entry point
// (Ascend 950PR, arch35 / dav-3510, KERNEL_TYPE_MIX_AIC_1_2), TWO-PHASE and
// with no GM intermediate workspace.
//
// op_kernel/ is PRIVATE (CLAUDE.md §1.2): nothing outside this operator may
// depend on these numbers. The sub-tile extents the HOST must also know --
// because it validates the geometry and the tiling POD carries them -- live in
// op_host/dsv4_moe_prefill_cube_tiling.h instead, and the kernel TU
// static_asserts the two copies against each other so a drift is a compile
// error rather than a silently wrong buffer.
//
// ===========================================================================
// THE MATMUL FRAMEWORK IS GONE FROM THIS OPERATOR
// ===========================================================================
// The kernel is now built from Tensor API primitives, on the three-stage
// asynchronous pipeline the STEP 1 / STEP 2 probes established
// (tests/sim/kernel/dsv4_prefill_cube_dualdst.h and
// tests/sim/kernel/dsv4_prefill_cube_pipeline.h). Three facts forced it, and
// all three are recorded rather than argued:
//
//   1. The framework cannot reach Fixpipe DUAL_DST_SPLIT_M with a UB
//      destination. The mode requires `A_TYPE::ibShare && B_TYPE::ibShare`
//      (copy_cube_out_fixpipe.h SplitMNParamsCheck), and with ibShare set
//      `Iterate()` returns false immediately under USE_WORKSPACE, which is
//      the mode this toolchain selects (matmul_client.h:716-721;
//      KFC_C310_SSBUF is pre-defined to 0). So there is no surviving
//      move-out call.
//   2. `TQue<TPosition::TSCM, 2>` does NOT allocate from L1 on this
//      toolchain. MEASURED by the GetPhyAddr census in
//      tests/sim/kernel/test_subgemm_stage_kernel.cpp, 2026-10-17: all four
//      ring slots came back contiguous with the sixteen UB buffers, running
//      127,520 -> 258,592. The weight ring was 131,072 B of UB -- half this
//      operator's budget -- and it is what made the production geometry
//      (hidden 4096, inter 2048) over-commit UB by 45,600 B. `TBuf<A1>` /
//      `TBuf<B1>` are real L1, which is where the staging belongs.
//   3. The framework owns L0C and retires it per sub-GEMM, so every K block's
//      partial sum had to be evicted to a UB FP32 accumulator and added
//      there. That injects a rounding per K block AND spends the
//      accumulators. The bare-metal path accumulates in L0C in FP32 across
//      the whole contraction and moves out ONCE.
//
// ===========================================================================
// WHO DOES WHAT, AND ON WHICH AXIS
// ===========================================================================
// Both vector cores are busy in both phases, and the two bisections are on
// DIFFERENT axes. That is not a quirk; each is forced by a different
// hardware contract.
//
//   PROLOGUE, bisected on the CONTRACTION axis (K).
//     AIV `s` loads and dequantizes the weight tile's K-half
//     [k_start = s * K/2, +K/2) and loads the same K-half of the token tile.
//     Each writes a DISJOINT RANGE OF WHOLE NZ C0 BLOCKS of the shared L1
//     slot. Why the contraction axis and not the output axis: see
//     "THE C0-BLOCK RULE" below -- a row-range write into an NZ slot is not
//     expressible as one strided burst, and the UB -> L1 ND2NZ path cannot do
//     it either.
//
//   EPILOGUE, bisected on the TOKEN axis (M).
//     One Fixpipe with DUAL_DST_SPLIT_M delivers rows [0, M/2) to AIV0's UB
//     and [M/2, M) to AIV1's, with the destination layout carrying the FULL M
//     -- MEASURED in STEP 1; passing M/2 there delivers M/4 rows per core.
//     So each AIV owns token rows [s * M/2, +M/2) from the activation plane
//     through SwiGLU to the output store, and the resident act plane is
//     M/2 rows wide per core rather than M.
//
// ===========================================================================
// THE WEIGHTS ARE [N][K] AND THE CUBE WANTS [K][N]. THE FIX IS FREE.
// ===========================================================================
// STEP 2 left this open: its probe held W contraction-major ([K][N]) and said
// so explicitly, because production quantizes along the contraction axis and
// stores W as [N][K] (gate/up: [inter][hidden]; down: [hidden][inter]).
// `CopyL12L0B`'s NZ -> ZN route does NOT solve it -- MEASURED in STEP 2, it
// transposes each FRACTAL, converting between two storage encodings of the
// SAME logical matrix, and cannot turn a logical [N, K] into a logical
// [K, N].
//
// The transpose is free, and it is a REINTERPRETATION rather than a copy.
// Derived from impl/tensor_api/tensor/layout_pattern.h, with FRACTAL_FIXED
// = 16 and C0_ELEMENT<bfloat16_t> = 16:
//
//   NZ(R, C) places logical (r, c), r = r0 + 16*r1, c = c0 + 16*c1, at
//       c1*(16*R) + r1*256 + r0*16 + c0
//   ZN(R, C) places logical (r, c), r = i  + 16*j,  c = m  + 16*n,  at
//       j*(16*C)  + n*256  + m*16  + i
//
// Substitute R' = N, C' = K into NZ and R' = K, C' = N into ZN and the two
// address functions are the SAME function of the same four digits, with
// (i, j) <-> (c0, c1) and (m, n) <-> (r0, r1):
//
//       NZ(N, K) at (n, k)  ==  ZN(K, N) at (k, n)
//
// So the AIV WRITES the L1 slot as NZ(N, K_SUB) -- the natural destination
// for a dequantizer whose output is already [N][K] with the MXFP4 groups
// along K -- and the AIC READS the same bytes as ZN(K_SUB, N), which is
// logically W^T. `CopyL12L0B` then sees ZN -> ZN, takes its `noTrans` /
// CopyMode::NORMAL route (l1_to_l0b/copy.h: `noTrans =
// is_same_v<DstPattern, SrcPattern>`), and no fractal transpose runs at all.
//
// THIS IS DERIVED FROM THE TOOLKIT'S LAYOUT ARITHMETIC AND HAS NOT BEEN
// MEASURED ON HARDWARE OR IN THE SIMULATOR. The STEP 2 history is the
// warning: the first attempt at this operand was wrong, it was wrong
// silently, and the diagnostic that settled it was a SELECTOR A (A[m][k] =
// (k == m)) which makes C a direct readout of B. Any doubt about this
// reinterpretation should be settled that way and not by reading the
// addresses again.
//
// ===========================================================================
// THE C0-BLOCK RULE: WHY THE PROLOGUE SPLITS ON K
// ===========================================================================
// An NZ(R, C) tile is C/16 blocks, each a dense [R rows][16 cols] row-major
// run of 16*R elements, blocks strided 16*R apart. (Fold the address function
// above: r1*256 + r0*16 = (16*r1 + r0)*16 = r*16.)
//
// So a COLUMN-BLOCK RANGE of an NZ slot is a contiguous, densely packed
// sub-tile, and a ROW RANGE is not -- it is R_sub rows out of every block.
// Two consequences:
//
//   * `DataCopy(.., .., Nd2NzParams)` UB -> L1 cannot write a row range.
//     Its dav-3510 implementation (DataCopyUB2L1ND2NZImpl, taken because
//     __MIX_CORE_AIC_RATION__ != 1) runs TransND2NZ into a PopStackBuffer and
//     then ONE contiguous `CopyUbufToCbuf` of
//         actualSize = ((dValue*sizeof(T)/32 - 1)*dstNzC0Stride
//                       + (nValue - 1)*dstNzNStride + 1) * 32
//     bytes. That equals the dense payload only when dstNzC0Stride == nValue.
//     With dstNzC0Stride = R (the full slot) and nValue = R/2 it is larger
//     than what TransND2NZ wrote, so the burst would carry uninitialized
//     stack bytes over the OTHER AIV's rows -- a silent wrong answer.
//   * It also means the C0-block range IS expressible as one plain strided
//     burst per block, which is exactly what the toolkit's own
//     DataCopyUB2L1ND2NZImplV2 emits.
//
// SO EVERY UB -> L1 PUSH IN THIS KERNEL IS A PLAIN `DataCopy` WITH
// `DataCopyParams`, one burst per C0 block, through `PushUbTileToL1Nz`.
// That is a deliberate step DOWN from the Nd2NzParams overload the STEP 2
// probe uses, for three reasons beyond the row-range one:
//
//   - It is pure MTE3. The Nd2NzParams path is TransND2NZ (PIPE_V) plus a
//     burst, so it spends vector cycles on the stage that already binds.
//   - It carries no hidden shared scratch. The Nd2NzParams path funnels
//     every push through ONE PopStackBuffer and fences only V -> MTE3
//     internally, so two pushes issued back to back in the same iteration
//     race on that buffer. This kernel issues THREE per step.
//   - `DataCopyUB2L1Impl` lowers straight to `CopyUbufToCbuf` under
//     __MIX_CORE_AIC_RATION__ != 1, with no KfcClient and no message queue.
//
// ===========================================================================
// PER-CORE CAPACITIES
// ===========================================================================
//     L1          524,288 B    shared with the paired Cube
//     L0A / L0B    65,536 B    each
//     L0C         262,144 B
//     UB          262,144 B    PER VECTOR CORE (the paper's 512 KiB is per
//                              subsystem, and a subsystem holds two AIVs)

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

inline constexpr std::uint32_t kCubeMTile = 16;    // tokens per token tile
inline constexpr std::uint32_t kCubeRSub = 128;    // phase 1 inter OUTPUT   (N)
inline constexpr std::uint32_t kCubeKSub = 128;    // phase 1 hidden CONTRACT (K)
inline constexpr std::uint32_t kCubeHSub = 64;     // phase 2 hidden OUTPUT  (N)
inline constexpr std::uint32_t kCubeRSub2 = 256;   // phase 2 inter CONTRACT (K)
inline constexpr std::uint32_t kCubeGroup = 32;    // MXFP4 group

inline constexpr std::uint32_t kCubeHiddenMax = 4096;
inline constexpr std::uint32_t kCubeInterMax = 2048;

// --- the two bisections ---------------------------------------------------
// M/2 is the Fixpipe's split and therefore the token rows one AIV owns.
inline constexpr std::uint32_t kCubeMHalf = kCubeMTile / 2u;
static_assert(kCubeMTile % 2u == 0u,
              "DUAL_DST_SPLIT_M requires an even M");

// K/2 is the contraction half one AIV loads, dequantizes and pushes.
inline constexpr std::uint32_t kCubeKHalf  = kCubeKSub / 2u;
inline constexpr std::uint32_t kCubeRHalf2 = kCubeRSub2 / 2u;

// An NZ C0 block is 32 B, so 16 elements at BF16. Everything about the
// prologue's disjointness argument is stated in these units.
inline constexpr std::uint32_t kCubeC0 = 16;
static_assert(kCubeKHalf % kCubeC0 == 0u && kCubeRHalf2 % kCubeC0 == 0u,
              "each AIV's contraction half must be a whole number of NZ C0 "
              "blocks, or the two cores' bursts overlap inside one block");
// And a whole number of MXFP4 groups, or the dequantizer's scale broadcast
// straddles a group boundary.
static_assert(kCubeKHalf % kCubeGroup == 0u && kCubeRHalf2 % kCubeGroup == 0u,
              "each AIV's contraction half must be a whole number of MXFP4 "
              "groups");

// THE ACT PLANE'S CAPACITY IS NOT kCubeInterMax.
//
// kCubeInterMax is the MODEL's widest inter axis. kCubeActInterCap is how wide
// an `inter` the resident act plane is actually built to hold. The HOST side
// is authoritative: op_host/dsv4_moe_prefill_cube_tiling.h VALIDATES the
// geometry (§2.5) and op_kernel/ is private (§1.2), so each side carries its
// own copy and the kernel TU static_asserts them equal. Without the host
// rejection an over-wide `inter` is a silent write past the end of the plane,
// because the kernel's early return produces a zeroed output rather than a
// fault.
//
// THE PRODUCTION CAP NOW FITS, which it did not before this refactor. The
// sim override is kept because a cycle-accurate run has no use for the extra
// capacity, not because the production cap over-commits.
#if defined(DSV4_TILING_TEST_OVERRIDE) && DSV4_TILING_TEST_OVERRIDE
inline constexpr std::uint32_t kCubeActInterCap = 512;
#else
inline constexpr std::uint32_t kCubeActInterCap = kCubeInterMax;
#endif

// Both phases' weight HALVES hold the same number of elements, which is why
// ONE staging triple (nibbles + scales + BF16) serves both. Asserted rather
// than assumed: the two shapes are chosen independently and this coincidence
// is load-bearing.
inline constexpr std::uint32_t kCubeWeightHalfElems = kCubeRSub * kCubeKHalf;
static_assert(kCubeWeightHalfElems == kCubeHSub * kCubeRHalf2,
              "the phase 1 and phase 2 weight halves must hold the same "
              "element count to share one dequant staging buffer");

// ---------------------------------------------------------------------------
// THE UB TABLE
// ---------------------------------------------------------------------------
// Resident across BOTH phases:
//
//   landing     FP32 [M/2, max(R_SUB, H_SUB)]   4,096 B  the Fixpipe's
//                                                        destination. MUST BE
//                                                        THE FIRST BUFFER --
//                                                        see
//                                                        kCubeLandingUbOffset.
//   act         BF16 [M/2, inter_cap]          32,768 B  carries the
//                                                        intermediate from the
//                                                        last gate/up GEMM to
//                                                        the first down GEMM
//                                                        with no GM hop.
//                                                        8,192 B at the sim
//                                                        cap of 512.
//
// Phase 1 only:
//
//   gate_acc    FP32 [M/2, R_SUB]               4,096 B  one inter block's
//   up_acc      FP32 [M/2, R_SUB]               4,096 B  SwiGLU operands
//   act_sub     BF16 [M/2, R_SUB]               2,048 B  the activated block,
//                                                        before the plane
//
// Per contraction ring slot (2 slots), shared by both phases:
//
//   x_f32       FP32 [M, K/2]                   4,096 B  token half as read
//   x_bf16      BF16 [M, K/2]                   2,048 B  token half as pushed
//   nib(x2)     4bit [R_SUB, K/2]           2 x 4,096 B  packed nibbles
//   scale(x2)   u8   [R_SUB, 32]            2 x 4,096 B  E8M0, one 32 B burst
//                                                        block per row
//   wide(x2)    BF16 [R_SUB, K/2]          2 x 16,384 B  dequantized half
//
// `nib`, `scale` and `wide` come in TWO per slot because phase 1 drives two
// weight streams -- W1 (gate) and W3 (up) -- through the SAME contraction
// loop, so that one token tile serves both. Phase 2 has a single stream and
// uses stream 0 only.
//
// Scratch, shared (serialized on the one Vector unit):
//
//   scale_plane BF16 [max(K/2, R_SUB2/2)]         256 B
//   scale_temp  u16  [16]                          32 B
//
// The phase-only segments are NOT overlaid even though the phases are
// disjoint in time: TPipe allocates statically, and overlaying them would
// mean aliasing two LocalTensors onto one TBuf, which is how a buffer-reuse
// bug becomes invisible.
inline constexpr std::uint32_t kCubeRingSlots = 2;
inline constexpr std::uint32_t kCubeWeightStreams = 2;  // W1 and W3

inline constexpr std::uint32_t kCubeOutColsMax =
    kCubeRSub > kCubeHSub ? kCubeRSub : kCubeHSub;
inline constexpr std::uint32_t kCubeLandingBytes = kCubeMHalf * kCubeOutColsMax * 4u;

inline constexpr std::uint32_t kCubeActPlaneBytes      = kCubeMHalf * kCubeActInterCap * 2u;
inline constexpr std::uint32_t kCubeAccBytes      = kCubeMHalf * kCubeRSub * 4u;
inline constexpr std::uint32_t kCubeActSubBytes   = kCubeMHalf * kCubeRSub * 2u;
inline constexpr std::uint32_t kCubeTokenF32Bytes     = kCubeMTile * kCubeKHalf * 4u;
inline constexpr std::uint32_t kCubeTokenSubBytes     = kCubeMTile * kCubeKHalf * 2u;
inline constexpr std::uint32_t kCubeNibHalfBytes      = kCubeWeightHalfElems / 2u;
inline constexpr std::uint32_t kCubeWideHalfBytes     = kCubeWeightHalfElems * 2u;

// The scale burst lands one 32-byte block per weight-tile row, because a row
// contributes only K/2 / 32 scale bytes and a UB burst block is 32 B. The row
// stride in the staging buffer is therefore 32 ENTRIES, and the kernel indexes
// it that way.
inline constexpr std::uint32_t kCubeScaleRowStride = 32;
inline constexpr std::uint32_t kCubeScaleRowsMax =
    kCubeRSub > kCubeHSub ? kCubeRSub : kCubeHSub;
inline constexpr std::uint32_t kCubeScaleBytes =
    kCubeScaleRowsMax * kCubeScaleRowStride;
static_assert(kCubeKHalf / kCubeGroup <= kCubeScaleRowStride &&
                  kCubeRHalf2 / kCubeGroup <= kCubeScaleRowStride,
              "a weight half's row of E8M0 scales must fit one 32-byte burst "
              "block");

// A ONE-ROW SCALE BUFFER, not a plane. Every row's scales are used only by
// that row's multiply, so one row of the widest half serves and the Mul lives
// inside the row loop. MEASURED CONSEQUENCE of the old plane, on the dav-3510
// CAModel: the Broadcast into it was the access reported by
// `rvec_dec.cc judge_valid_addr ub_addr_overflow`, 1,280 diagnostics per
// DequantSubTileVector call, in EVERY run that reached the dequantizer and
// none that did not.
// SIZED FROM THE FULL SUB-TILE, NOT THE HALF. The new path's widest Broadcast is
// max(K_SUB/2, R_SUB2/2) = 128 elements, but the legacy routines retained for the
// sim bisection probes broadcast up to R_SUB2 = 256, and one constant serving both
// is worth 256 B.
inline constexpr std::uint32_t kCubeScaleRowElems =
    kCubeKSub > kCubeRSub2 ? kCubeKSub : kCubeRSub2;
inline constexpr std::uint32_t kCubeScaleRowBytes = kCubeScaleRowElems * 2u;
inline constexpr std::uint32_t kCubeScaleTempBytes = 16u * 2u;

// THE SUM MUST LIST EVERY InitBuffer, ONCE PER CALL, and in the kernel's
// order so the two can be diffed by eye. The previous version of this
// constant did not, and its static_assert was therefore vacuous: it
// under-counted by 45,088 B. If a buffer is added to the kernel and not here,
// the assert stops meaning anything again.
inline constexpr std::uint32_t kCubeUbBytesLessAct =
    kCubeLandingBytes +                                        // q_landing
    kCubeAccBytes +                                            // q_gate_acc
    kCubeAccBytes +                                            // q_up_acc
    kCubeActSubBytes +                                         // q_act_sub
    kCubeRingSlots * kCubeTokenF32Bytes +                          // q_x_f32[s]
    kCubeRingSlots * kCubeTokenSubBytes +                          // q_x_bf16[s]
    kCubeRingSlots * kCubeWeightStreams * kCubeNibHalfBytes +      // q_nib[s][w]
    kCubeRingSlots * kCubeWeightStreams * kCubeScaleBytes +    // q_scale[s][w]
    kCubeRingSlots * kCubeWeightStreams * kCubeWideHalfBytes +     // q_wide[s][w]
    kCubeScaleRowBytes +                                       // q_scale_plane
    kCubeScaleTempBytes;                                       // q_scale_temp

inline constexpr std::uint32_t kCubeUbBytes = kCubeUbBytesLessAct + kCubeActPlaneBytes;

static_assert(kCubeUbBytesLessAct == 125472,
              "every InitBuffer except the act plane sums to 125,472 B. "
              "Re-derive it from the kernel rather than adjusting the literal");
static_assert(kCubeUbBytes == (kCubeActInterCap == 2048u ? 158240u : 133664u),
              "the Cube prefill UB budget is 158,240 B at the production "
              "inter cap of 2048 and 133,664 B at the sim cap of 512");

// THIS ASSERT IS NO LONGER VACUOUS, AND THAT IS THE POINT OF THE REFACTOR.
//
// Before it, the operator's real UB charge was kCubeUbBytes PLUS the 131,072 B
// TPosition::TSCM weight ring that the GetPhyAddr census found sharing the UB
// pool -- 307,744 B against a 262,144 B physical UB at the production cap,
// over by 45,600 B, which is the "~46 KB" the ub_addr_overflow diagnostics
// were reporting. The ring is now `TBuf<A1>` / `TBuf<B1>`, so it is charged to
// L1 (see kCubeL1Bytes) and kCubeUbBytes IS the whole UB charge.
//
// Both bounds are asserted: kUsableUbBytes (221,184) is the conservative
// budget that holds back the SIMT DataCache floor, kUbTotalBytes (262,144) is
// the physical ceiling the census measured the pool to span from 0.
static_assert(kCubeUbBytes <= kUsableUbBytes,
              "the Cube prefill UB budget exceeds the arch35 usable ceiling");
static_assert(kCubeUbBytes <= kUbTotalBytes,
              "the Cube prefill UB budget exceeds physical UB");

// The act plane's ceiling, derived rather than chosen. It is now well clear of
// the model maximum, so kCubeActInterCap = kCubeInterMax is a real capacity
// and not an aspiration.
inline constexpr std::uint32_t kCubeActInterCapMax =
    ((kUsableUbBytes - kCubeUbBytesLessAct) / (kCubeMHalf * 2u)) /
    kCubeRSub2 * kCubeRSub2;
static_assert(kCubeActInterCapMax >= kCubeInterMax,
              "the act plane must hold the model's widest inter axis");

// --- UB offsets the AIC has to know ---------------------------------------
//
// THE LANDING ZONE IS AT UB OFFSET 0, AND THAT IS A MEASURED FACT RATHER THAN
// A CONVENTION.
//
// The AIC issues the Fixpipe with a UB address that is interpreted in EACH
// AIV's own UB -- the two are separately addressed -- so both sides have to
// agree on one offset, and the AIC cannot ask TPipe for it because the buffer
// is not its own. Offset 0 works because the AIV's TPipe dynamic pool starts
// at 0 on this toolchain: `TPipe::ResetPool()` seeds it with
// `GetDynamicMemStartPos<Hardware::UB>()` = `&dynamicStartUB[0]`, and that
// linker symbol resolves to zero here. MEASURED by the GetPhyAddr census in
// tests/sim/kernel/test_subgemm_stage_kernel.cpp and re-checked by the STEP 1
// probe, which reports the AIV's first InitBuffer address in its status block.
//
// CONSEQUENCE FOR THE KERNEL: `q_landing` must be the FIRST InitBuffer call on
// the AIV, and there must be exactly ONE landing zone. A second one would need
// a second literal offset, and that offset would depend on the allocation
// order of everything before it. This kernel instead reuses the one zone and
// serializes the two phase 1 move-outs with the landing-free flag below --
// which costs two cross-core round trips per inter block against a whole
// contraction's worth of Cube work.
inline constexpr std::uint32_t kCubeLandingUbOffset = 0;

// ---------------------------------------------------------------------------
// L1: the ring the Cube reads its operands from
// ---------------------------------------------------------------------------
// `TBuf<TPosition::A1>` and `TBuf<TPosition::B1>`, allocated IDENTICALLY ON
// BOTH CORES and in the same order, OUTSIDE the ASCEND_IS_* branches. That is
// not tidiness: the AIV writes a slot and the AIC reads it, so the two cores
// have to land on the same L1 offset, and nothing in the toolchain checks
// that they do. Allocating identically on both is the only thing that makes
// the addresses agree.
// The Cube-side capacities, from the toolkit's platform configs rather than
// the whitepaper's per-subsystem figures. Declared here rather than in
// include/moe/kernels/arch35/arch35_common.hpp because only this operator
// budgets L1 and L0 by hand -- everything else leaves them to the Matmul
// template, which sizes them from its compile-time tiling.
inline constexpr std::uint32_t kL1TotalBytes   = 512u * 1024u;
inline constexpr std::uint32_t kL0ABTotalBytes = 64u * 1024u;
inline constexpr std::uint32_t kL0CTotalBytes  = 256u * 1024u;

inline constexpr std::uint32_t kCubeL1ASlotBytes  = kCubeMTile * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeL1WSlotBytes  = kCubeRSub * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeL1A2SlotBytes = kCubeMTile * kCubeRSub2 * 2u;
inline constexpr std::uint32_t kCubeL1W2SlotBytes = kCubeHSub * kCubeRSub2 * 2u;

inline constexpr std::uint32_t kCubeL1Bytes =
    kCubeRingSlots * kCubeL1ASlotBytes +
    kCubeRingSlots * kCubeWeightStreams * kCubeL1WSlotBytes +
    kCubeRingSlots * kCubeL1A2SlotBytes +
    kCubeRingSlots * kCubeL1W2SlotBytes;
static_assert(kCubeL1Bytes == 221184,
              "the L1 operand ring costs 221,184 B");
static_assert(kCubeL1Bytes <= kL1TotalBytes,
              "the L1 operand ring exceeds L1");

// --- L0 ------------------------------------------------------------------
// Raw __ca__ / __cb__ / __cc__ arrays inside the AIC branch: they are
// Cube-private, the AIV has no such memory, and TPipe would have to be told
// so. Sized for the larger phase of each.
//
//   L0A  BF16 [M, max(K_SUB, R_SUB2)]                 8,192 B
//   L0B  BF16 [max(K_SUB, R_SUB2), N] x 2 streams    65,536 B
//   L0C  FP32 [M, R_SUB] x 2 accumulators            16,384 B
//
// L0B is EXACTLY FULL at the phase 1 shape, and that is deliberate: the two
// weight streams need their operands resident simultaneously, because the
// alternative -- one L0B reloaded between the gate Mmad and the up Mmad --
// puts an L0B write-after-read fence between every pair of Mmads and makes
// MTE1 and the Cube alternate instead of overlap.
inline constexpr std::uint32_t kCubeKMax =
    kCubeKSub > kCubeRSub2 ? kCubeKSub : kCubeRSub2;
inline constexpr std::uint32_t kCubeL0AElems = kCubeMTile * kCubeKMax;
inline constexpr std::uint32_t kCubeL0BElems = kCubeKSub * kCubeRSub;
inline constexpr std::uint32_t kCubeL0CElems = kCubeMTile * kCubeRSub;
static_assert(kCubeL0AElems * 2u <= kL0ABTotalBytes, "L0A overflows");
static_assert(kCubeWeightStreams * kCubeL0BElems * 2u <= kL0ABTotalBytes,
              "the two L0B streams overflow L0B");
static_assert(kCubeWeightStreams * kCubeL0CElems * 4u <= kL0CTotalBytes,
              "the two L0C accumulators overflow L0C");
static_assert(kCubeL0BElems >= kCubeRSub2 * kCubeHSub,
              "one L0B stream must also hold the phase 2 operand");

// ---------------------------------------------------------------------------
// CROSS-CORE FLAGS
// ---------------------------------------------------------------------------
// Mode 2 at EVERY set AND EVERY wait, with the pipe named at every site.
//
// `CrossCoreWaitFlag` declares `template <uint8_t modeId = 0,
// pipe_t pipe = PIPE_S>`, so the bare `CrossCoreWaitFlag(f)` spelling is
// mode 0 + PIPE_S -- a different mode from the mode 2 the set side uses, and
// the one pipe scope the spec's constraint list excludes for modes 0/1/2. On
// 950PR both template arguments are EFFECTIVE (on A2/A3 the hardware ignores
// them and blocks every pipeline), and an effective wait blocks only the
// pipeline it names. So the pipe has to be the one that CONSUMES what the flag
// protects, per site. Choosing the wrong pipe does not fail loudly: it stops
// gating the consumer, which is a silent wrong answer rather than a hang.
//
// AND THE AIV -> AIC DIRECTION MUST BE EXERCISED FIRST. MEASURED in STEP 1: a
// one-way AIC -> AIV release does not fire, whichever way the wait is spelled
// (stage 2 and stage 3 both hang, stage 4 retires in 14.6 s once the AIVs also
// signal the AIC). This kernel satisfies it structurally -- the AIC's first act
// in every contraction is to wait on kCubeFlagStepReady, which the AIVs set.
inline constexpr std::uint8_t kCubeSyncMode = 0x2;

// ONE READY FLAG PER RING SLOT COVERS THE WHOLE STEP, not one per buffer.
// A step writes three L1 slots (the token half, W1's half, W3's half); the AIV
// raises the flag once, after the last push, and MTE3 is in order so that
// covers all three. Mode 2's AIV -> AIC direction increments the AIC's counter
// only once BOTH AIVs have set, so ONE wait on the AIC matches two setters --
// MEASURED in STEP 2: a second wait HANGS, past 170,000 simulated cycles and a
// 240 s watchdog.
//
// --- flagId allocation: USE ONLY IDS SOMETHING HAS BEEN MEASURED WITH ----
//
// Modes 0/1/2 give each core 16 flagIds and `GetffstMsg` masks the id with
// 0xf, so 0..15 are all representable. The documented reservations are
// [0, 2N-1] for the Matmul high-level API with N DEFINED objects and [11, 14]
// for SyncAll (SYNC_AIC_FLAG 11, SYNC_AIV_FLAG 12, SYNC_AIC_AIV_FLAG 13,
// SYNC_AIV_ONLY_ALL 14, in dav_3510/kernel_operator_sync_impl.h).
//
// WHICH IDS ACTUALLY WORK IS NOT ESTABLISHED, AND THE ONE ATTEMPT TO
// ESTABLISH IT WAS CONFOUNDED BY A FLAKY PROBE. The record, because the wrong
// conclusion is easy to re-derive:
//
//   The STEP 1 probe moved its two flags from 0/1 to 8/9 on exactly the
//   reasoning above -- outside both reserved ranges, therefore "isolated" --
//   and hung past its 200 s watchdog. Reverting to 0/1 was expected to fix
//   it. It did not: a configuration that is behaviourally IDENTICAL to the
//   committed one (which had retired in 13.0 s minutes earlier) also hung.
//   So STEP 1 is NONDETERMINISTIC on the CAModel, and every "8/9 hangs"
//   observation is equally consistent with a flake.
//
// What is positively known is only this: STEP 2's handshake, which uses 4, 5,
// 6, 7 and 8, retires on all three cores every time it has been run; and
// STEP 1 has passed with 0/1. 9..15 have never been exercised green by
// anything. That is weak evidence, not a rule, and it is why this operator
// stays inside {0, 1} u [4, 8] -- not because 9 is known bad, but because
// nothing recommends it over an id that has demonstrably carried traffic.
//
// It needs six flags,
// and the fifth and sixth cannot both come from [4, 8] without colliding with
// the per-slot ranges, so the landing-zone acknowledgement takes 0. That is
// safe here for the same reason [0, 3] is free at all: this TU defines no
// Matmul OBJECT and never calls REGIST_MATMUL_OBJ, so the Matmul API's flagId
// usage never happens. If a Matmul object is ever instantiated in this TU,
// this allocation breaks and the static_assert below will not catch it --
// nothing in the build can.
inline constexpr std::uint16_t kCubeFlagStepReady   = 4;  // + slot : AIV -> AIC
inline constexpr std::uint16_t kCubeFlagStepFree    = 6;  // + slot : AIC -> AIV
inline constexpr std::uint16_t kCubeFlagProduct     = 8;  //         AIC -> AIV
inline constexpr std::uint16_t kCubeFlagLandingFree = 0;  //         AIV -> AIC

static_assert(kCubeFlagStepReady + kCubeRingSlots - 1u < kCubeFlagStepFree &&
                  kCubeFlagStepFree + kCubeRingSlots - 1u < kCubeFlagProduct,
              "the per-slot flag ranges must not overlap each other");
static_assert(kCubeFlagLandingFree < kCubeFlagStepReady,
              "the landing acknowledgement must not fall inside the per-slot "
              "ranges");
static_assert(kCubeFlagProduct <= 8 && kCubeFlagStepReady >= 4,
              "flagIds 9-15 have never been exercised green by any probe and "
              "11-14 are reserved by SyncAll: read the note above before "
              "widening this");

// ---------------------------------------------------------------------------
// INTRA-CORE EVENT IDS
// ---------------------------------------------------------------------------
// An intra-core flag is BINARY, not a counter: a second SetFlag on the same
// (srcPipe, dstPipe, eventID) triple before the first is consumed is undefined
// behaviour -- the simulator reports it as
// `pem_ccu.cc execute_set_flag already has same set_flag!` -- and a later
// PipeBarrier<PIPE_ALL> then hangs. Every fence that is DEFERRED across a loop
// iteration therefore needs one id per ring slot, because two of its tokens
// are outstanding at once.
//
// Ids 0..5 are usable. Id 6 is reserved for internal system planning and id 7
// is used by TPipe's automatic synchronization feature. (arch35_common.hpp's
// kMaxEventId = 3 is the conservative figure; the STEP 2 probe uses 0..5 on
// dav-3510 and retires, which is the measurement.)
//
// A flag is keyed by the FULL triple, so MTE2_V id 0, V_MTE2 id 0 and
// MTE3_V id 0 are three different hardware flags and the tables below may
// repeat values across HardEvents without aliasing.
//
//   kEvCubeSlotFull[s]  MTE2 -> V    slot s's token/nibble/scale bursts have
//                                    landed; the Vector may read them. Raised
//                                    ONCE per slot, after the LAST burst,
//                                    because MTE2 is in order.
//   kEvCubeSlotStage[s] V -> MTE2    the Vector has finished READING slot s's
//                                    staging; the next prefetch may overwrite
//                                    it. Deferred one full ring lap.
//   kEvCubeSlotPushed[s] MTE3 -> V   slot s's UB -> L1 pushes have drained;
//                                    the dequant may overwrite `wide`.
//                                    Deferred one full ring lap.
//   kEvCubeWideReady    V -> MTE3    the dequant wrote `wide`; MTE3 may read
//                                    it. Consumed immediately, so one id.
inline constexpr std::int32_t kEvCubeSlotFull[kCubeRingSlots]   = {0, 1};
inline constexpr std::int32_t kEvCubeSlotStage[kCubeRingSlots]  = {0, 1};
inline constexpr std::int32_t kEvCubeSlotPushed[kCubeRingSlots] = {0, 1};
inline constexpr std::int32_t kEvCubeWideReady = 2;

// AIC side. Distinct HardEvents and a different core, so these may repeat the
// AIV's values.
//
//   kEvCubeMToMte1   M -> MTE1    L0A/L0B write-after-read: step k+1's MTE1
//                                 writes the buffers step k's Mmads read.
//                                 IMMEDIATE. It was briefly deferred one
//                                 iteration, to let the Cube run behind the
//                                 loader instead of draining on a
//                                 PipeBarrier<PIPE_M> every step; that cost
//                                 the accumulation and double-set the flag.
//                                 See CubeCloseAccumulationStep.
//   kEvCubeMte1ToM   MTE1 -> M    read-after-write into L0A/L0B. Immediate.
//   kEvCubeMToFix    M -> FIX     the contraction is complete; the Fixpipe may
//                                 read L0C. Immediate.
inline constexpr std::int32_t kEvCubeMToMte1 = 0;
inline constexpr std::int32_t kEvCubeMte1ToM = 1;
inline constexpr std::int32_t kEvCubeMToFix  = 2;

inline constexpr std::int32_t kEvCubeMaxUsableId = 5;
static_assert(kEvCubeWideReady <= kEvCubeMaxUsableId &&
                  kEvCubeMToFix <= kEvCubeMaxUsableId,
              "only event ids 0-5 are usable: 6 is reserved for internal "
              "system planning and 7 for TPipe's automatic synchronization");
static_assert(kCubeRingSlots == 2,
              "the per-slot event id tables above are sized for a depth-2 "
              "ring; a deeper ring needs more ids");
static_assert(kEvCubeSlotFull[0] != kEvCubeSlotFull[1] &&
                  kEvCubeSlotStage[0] != kEvCubeSlotStage[1] &&
                  kEvCubeSlotPushed[0] != kEvCubeSlotPushed[1],
              "a deferred fence needs a DISTINCT id per slot, or the second "
              "SetFlag lands on an unconsumed flag");

// --- workspace ------------------------------------------------------------
// NOTHING LIVES HERE ANY MORE. The 1 MiB reserve was the KfcServer's message
// queue, which REGIST_MATMUL_OBJ stood up at the head of the workspace; with
// the framework gone there is no server and no queue. The reserve is kept at
// its old size so the host launch wrapper's allocation and the operator's
// signature are unchanged, and because `aclrtMalloc` of zero bytes is not a
// thing this layer wants to special-case.
inline constexpr std::uint64_t kCubeKfcReserveBytes = 1u << 20;

// ===========================================================================
// LEGACY: THE FRAMEWORK PATH'S BUDGET. NOT REACHABLE FROM THE ENTRY POINT.
// ===========================================================================
// Everything below sized the Matmul-framework schedule this operator no longer
// runs. It is retained for ONE reason: five sim bisection probes textually
// include the operator's kernel source and allocate production's buffers so
// that "the configuration under test is production's" --
//
//     tests/sim/kernel/dsv4_moe_prefill_cube_dequant_unit.cpp
//     tests/sim/kernel/dsv4_moe_prefill_cube_inductive_kernel.cpp
//     tests/sim/kernel/test_subgemm_stage_kernel.cpp
//     tests/sim/kernel/test_dequant_micro_kernel.cpp
//     tests/sim/kernel/test_kfc_lifecycle_kernel.cpp
//
// -- and they back the registered gates Dsv4MoePrefillCubeInductiveSim,
// Dsv4MoePrefillCubeSingleTileSim, Dsv4PrefillCubeDequantMicroSim and
// KfcLifecycleSim. Deleting this section retires those gates, which is a
// decision about verification coverage and not a tidy-up; the probes that
// still test LIVE code (the MXFP4 dequantizer) want reworking onto the new
// path rather than deleting, and that is a separate change.
//
// NONE OF IT COSTS THE SHIPPING KERNEL ANYTHING. These are compile-time
// constants; the UB they used to describe is charged by the InitBuffer calls
// that named them, and the entry point makes none of those calls any more.
// The authority for what the kernel actually spends is kCubeUbBytes above.
//
// DO NOT SIZE NEW BUFFERS FROM THESE. The names that differ by one word --
// kCubeActBytes vs kCubeActPlaneBytes, kCubeWideBytes vs kCubeWideHalfBytes --
// differ because the new path bisects what the old one did whole, and crossing
// them silently halves or doubles a buffer.
inline constexpr std::uint32_t kCubeWeightSubElems = kCubeRSub * kCubeKSub;
static_assert(kCubeWeightSubElems == kCubeHSub * kCubeRSub2,
              "the legacy phase 1 and phase 2 weight sub-tiles must hold the "
              "same element count to share one dequant staging buffer");

inline constexpr std::uint32_t kCubeActBytes  = kCubeMTile * kCubeActInterCap * 2u;
inline constexpr std::uint32_t kCubeAcc1Bytes = kCubeMTile * kCubeRSub * 4u;
inline constexpr std::uint32_t kCubeAcc2Bytes = kCubeMTile * kCubeHSub * 4u;
inline constexpr std::uint32_t kCubeXF32Bytes = kCubeMTile * kCubeKSub * 4u;
inline constexpr std::uint32_t kCubeXSubBytes = kCubeMTile * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeASubBytes = kCubeMTile * kCubeRSub2 * 2u;
inline constexpr std::uint32_t kCubeNibBytes  = kCubeWeightSubElems / 2u;
inline constexpr std::uint32_t kCubeWideBytes = kCubeWeightSubElems * 2u;
static_assert(kCubeWideBytes == 32768, "the legacy dequant staging was 32,768 B");

// The TPosition::TSCM weight ring. MEASURED, dav-3510 CAModel, 2026-10-17, by
// the GetPhyAddr census in tests/sim/kernel/test_subgemm_stage_kernel.cpp: all
// four slots came back CONTIGUOUS with the sixteen UB buffers, 32,768 B apart,
// running 127,520 -> 258,592. They were NOT in the separate L1 pool that
// GetPhyType(TSCM) == Hardware::L1 and TPipe::TscmInitBuffer's top-of-L1
// allocation predict. That 131,072 B of UB is what made the production
// geometry over-commit by 45,600 B, and it is why the ring is now
// TBuf<A1>/TBuf<B1> (see kCubeL1Bytes).
inline constexpr std::uint32_t kCubeB1SlotBytes = kCubeRSub * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeB2SlotBytes = kCubeHSub * kCubeRSub2 * 2u;
static_assert(kCubeB1SlotBytes == 32768 && kCubeB2SlotBytes == 32768,
              "each legacy weight ring slot is 32,768 B");
inline constexpr std::uint32_t kCubeTscmRingBytes =
    2u * kCubeB1SlotBytes + 2u * kCubeB2SlotBytes;
static_assert(kCubeTscmRingBytes == 131072, "the legacy weight ring cost 131,072 B");

// The legacy event plan: EVENT_ID0..3, balanced 1:1 at every use so no token
// outlived a loop iteration. The new path's ids are above and may repeat these
// values -- the two schedules are reached through different entry points and
// never run in the same launch.
//
//   kEvCubeMte2ToV     a GM burst landed in UB; the Vector may read it
//   kEvCubeVToMte3     the Vector finished w_wide; MTE3 may push it to L1
//   kEvCubeMte3ToMte1  the L1 slot is written; the Cube's MTE1 may read it
//   kEvCubeMte3ToV     the L1 push drained; the Vector may overwrite w_wide
inline constexpr std::int32_t kEvCubeMte2ToV    = 0;
inline constexpr std::int32_t kEvCubeVToMte3    = 1;
inline constexpr std::int32_t kEvCubeMte3ToMte1 = 2;
inline constexpr std::int32_t kEvCubeMte3ToV    = 3;
static_assert(kEvCubeMte3ToV <= static_cast<std::int32_t>(kMaxEventId),
              "the legacy plan stayed inside arch35_common.hpp's conservative "
              "EVENT_ID0..3");

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube


#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
