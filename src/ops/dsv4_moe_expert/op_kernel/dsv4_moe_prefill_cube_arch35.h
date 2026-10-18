// dsv4_moe_prefill_cube_arch35.h -- UB / L1 budget and event plan for the
// Cube-accelerated batched prefill entry point (Ascend 950PR, arch35 /
// dav-3510, MIX AIC+AIV), TWO-PHASE and with no GM intermediate workspace.
//
// op_kernel/ is PRIVATE (CLAUDE.md §1.2): nothing outside this operator may
// depend on these numbers. The sub-tile extents the HOST must also know --
// because it validates the geometry and the tiling POD carries them -- live in
// op_host/dsv4_moe_prefill_cube_tiling.h instead, and the kernel TU
// static_asserts the two copies against each other so a drift is a compile
// error rather than a silently wrong buffer.
//
// PER-CORE CAPACITIES, from the toolkit's platform configs rather than the
// whitepaper's per-subsystem figures:
//
//     L1          524,288 B    shared with the paired Cube
//     L0A / L0B    65,536 B    Cube operand buffers, Matmul-managed
//     L0C         262,144 B    Cube accumulator, Matmul-managed
//     UB          253,952 B    PER VECTOR CORE (the paper's 512 KiB is per
//                              subsystem, and a subsystem holds two AIVs)
//
// L0A/L0B/L0C are not budgeted here: the Matmul template owns them, and its
// compile-time tiling (GetMatmulApiTiling) is what sizes them from the
// declared operand types and the per-sub-GEMM shape. What this file budgets is
// the UB the VECTOR side holds and the L1 it stages weights through.

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

inline constexpr std::uint32_t kCubeMTile = 16;    // tokens per token tile
inline constexpr std::uint32_t kCubeRSub = 128;    // phase 1 inter OUTPUT
inline constexpr std::uint32_t kCubeKSub = 128;    // phase 1 hidden CONTRACT
inline constexpr std::uint32_t kCubeHSub = 64;     // phase 2 hidden OUTPUT
inline constexpr std::uint32_t kCubeRSub2 = 256;   // phase 2 inter CONTRACT
inline constexpr std::uint32_t kCubeGroup = 32;    // MXFP4 group

inline constexpr std::uint32_t kCubeHiddenMax = 4096;
inline constexpr std::uint32_t kCubeInterMax = 2048;

// THE ACT PLANE'S CAPACITY IS NOT kCubeInterMax.
//
// kCubeInterMax is the MODEL's widest inter axis. kCubeActInterCap is how wide
// a `inter` the resident act plane is actually built to hold, and the two are
// only the same number in a production build. Sizing the plane from the model
// maximum charged 65,536 B whatever the run's geometry, and no CAModel fixture
// goes past inter = 512 -- so 49,152 B of that was capacity no access ever
// used.
//
// This is the host's constant, duplicated for the same reason kCubeMTile is:
// op_host/dsv4_moe_prefill_cube_tiling.h is what VALIDATES the geometry (§2.5)
// and op_kernel/ is private (§1.2), so each side carries its own copy and the
// kernel TU static_asserts them equal. The HOST side is authoritative: its
// admissibility predicate rejects `inter > cap`, and without that rejection an
// over-wide `inter` is a silent write past the end of the plane rather than an
// error -- the kernel's early return produces a zeroed output, not a fault.
#if defined(DSV4_TILING_TEST_OVERRIDE) && DSV4_TILING_TEST_OVERRIDE
inline constexpr std::uint32_t kCubeActInterCap = 512;
#else
inline constexpr std::uint32_t kCubeActInterCap = kCubeInterMax;
#endif

// Both phases' weight sub-tiles hold the same number of elements, which is why
// ONE staging pair (nibbles + BF16) serves both. Asserted rather than assumed:
// the two shapes are chosen independently and this coincidence is load-bearing.
inline constexpr std::uint32_t kCubeWeightSubElems = kCubeRSub * kCubeKSub;
static_assert(kCubeWeightSubElems == kCubeHSub * kCubeRSub2,
              "the phase 1 and phase 2 weight sub-tiles must hold the same "
              "element count to share one dequant staging buffer");

// ---------------------------------------------------------------------------
// THE UB TABLE, AND WHY M_TILE IS 16
// ---------------------------------------------------------------------------
// Resident across BOTH phases:
//
//     act          BF16  [16, cap]    32 B/col  the whole point of the
//                                                schedule: it carries the
//                                                intermediate from the last
//                                                gate/up GEMM to the first
//                                                down GEMM with no GM hop.
//                                                65,536 B at the production
//                                                cap of 2048, 16,384 B at
//                                                the sim cap of 512 -- see
//                                                kCubeActInterCap
//
// Phase 1 only:
//
//     gate_acc     FP32  [16, 128]     8,192 B   level-2 subtotal, summed
//     up_acc       FP32  [16, 128]     8,192 B   over the K_SUB blocks
//     c1           FP32  [16, 128]     8,192 B   one sub-GEMM's C, at VECIN
//     x_f32        FP32  [16, 128]     8,192 B   X sub-tile as the caller has
//                                                it, before the cast
//     x_sub        BF16  [16, 128]     4,096 B   X sub-tile as the Cube reads
//                                                it, at VECOUT
//
// Phase 2 only:
//
//     y_acc        FP32  [16, 64]      4,096 B   level-2 subtotal
//     c2           FP32  [16, 64]      4,096 B   one sub-GEMM's C, at VECIN
//     a_sub        BF16  [16, 256]     8,192 B   act's column slice, made
//                                                contiguous for the Cube's A
//
// Shared by both phases (the sub-tile element counts coincide):
//
//     w_nib        4-bit [16384]       8,192 B   packed nibbles, as read
//     w_wide       BF16  [16384]      32,768 B   expanded, before UB -> L1
//     w_scale      u8    [128 x 32]    4,096 B   E8M0 bytes, one 32 B burst
//                                                block per sub-tile row
//
// THE TABLE ABOVE IS NOT THE BUDGET. It lists the buffers once; the kernel
// allocates x_f32 and x_sub TWICE (ping and pong) and adds a one-row scale
// buffer and a 32 B scale temporary that the table never had. The authority
// is kCubeUbBytes below, which enumerates the InitBuffer calls one for one:
//
//     everything except act         111,136 B    kCubeUbBytesLessAct
//     + act at the production cap    65,536 B -> 176,672 B total
//     + act at the sim cap of 512    16,384 B -> 127,520 B total
//
// The phase-only segments are NOT overlaid even though the phases are disjoint
// in time: TPipe allocates statically, and overlaying them would mean aliasing
// two LocalTensors onto one TBuf, which is how a buffer-reuse bug becomes
// invisible.
//
// AT M_TILE = 32 every line except the three weight-staging ones doubles,
// since they are the only buffers whose extent does not carry M. At the
// production cap that is 176,672 + 131,072 = 307,744 B, past physical UB
// (262,144) on its own and nearly 2.4x the dynamic pool a TPipe on an AIV
// actually hands out. So 16 is forced, and the forcing term is the `act`
// residency -- the same argument as before the two-phase rewrite, for the
// same reason.
inline constexpr std::uint32_t kCubeActBytes = kCubeMTile * kCubeActInterCap * 2u;
inline constexpr std::uint32_t kCubeAcc1Bytes = kCubeMTile * kCubeRSub * 4u;
inline constexpr std::uint32_t kCubeAcc2Bytes = kCubeMTile * kCubeHSub * 4u;
inline constexpr std::uint32_t kCubeXF32Bytes = kCubeMTile * kCubeKSub * 4u;
inline constexpr std::uint32_t kCubeXSubBytes = kCubeMTile * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeASubBytes = kCubeMTile * kCubeRSub2 * 2u;
inline constexpr std::uint32_t kCubeNibBytes = kCubeWeightSubElems / 2u;
inline constexpr std::uint32_t kCubeWideBytes = kCubeWeightSubElems * 2u;

// The scale burst lands one 32-byte block per sub-tile row, because a row
// contributes only K_SUB / 32 = 4 scale bytes and a UB burst block is 32 B.
// The row stride in the staging buffer is therefore 32 ENTRIES, not 4, and the
// kernel indexes it that way.
inline constexpr std::uint32_t kCubeScaleRowStride = 32;
inline constexpr std::uint32_t kCubeScaleRowsMax =
    kCubeRSub > kCubeHSub ? kCubeRSub : kCubeHSub;
inline constexpr std::uint32_t kCubeScaleBytes =
    kCubeScaleRowsMax * kCubeScaleRowStride;

static_assert(kCubeActBytes == kCubeMTile * kCubeActInterCap * 2u &&
                  kCubeActBytes == (kCubeActInterCap == 2048u ? 65536u : 16384u),
              "the act plane is M_TILE x inter_cap in BF16: 65,536 B at the "
              "production cap of 2048, 16,384 B at the sim cap of 512");
static_assert(kCubeWideBytes == 32768, "the dequant staging must be 32,768 B");
// A sub-tile row's scale bytes must fit the 32 B block the burst reserves for
// it, in BOTH phases -- K_SUB / 32 = 4 and R_SUB2 / 32 = 8, so both do.
static_assert(kCubeKSub / kCubeGroup <= kCubeScaleRowStride &&
                  kCubeRSub2 / kCubeGroup <= kCubeScaleRowStride,
              "a sub-tile row's E8M0 scales must fit one 32-byte burst block");

// A ONE-ROW SCALE BUFFER, not an R_SUB x K_SUB PLANE.
//
// This was `kCubeRSub * kCubeKSub * 2` = 32,768 B: a full dequantized scale
// plane, built one row at a time by Broadcast and then consumed by ONE Mul
// over the whole sub-tile. The plane was never needed. Every row's scales are
// used only by that row's multiply, so a single row of the widest sub-tile
// serves, and the Mul moves inside the row loop.
//
// MEASURED CONSEQUENCE of the plane, on the dav-3510 CAModel: the Broadcast
// into scale_plane[out_base] was the access reported by
// `rvec_dec.cc judge_valid_addr ub_addr_overflow`, 1,280 diagnostics per
// DequantSubTileVector call (10 per row x 128 rows, a 256 B Broadcast split
// into 32 B bodies), in EVERY run that reached the dequantizer and none that
// did not.
//
// The widest row is phase 2's R_SUB2 contraction, not phase 1's K_SUB.
inline constexpr std::uint32_t kCubeScaleRowElems =
    kCubeKSub > kCubeRSub2 ? kCubeKSub : kCubeRSub2;
inline constexpr std::uint32_t kCubeScaleRowBytes = kCubeScaleRowElems * 2u;

// THE SUM MUST LIST EVERY InitBuffer, ONCE PER CALL.
//
// The previous formula did not, and the static_assert below was therefore
// vacuous -- it constrained a number that did not describe the kernel. It
// under-counted by 45,088 B in three ways: kCubeXF32Bytes and kCubeXSubBytes
// are each allocated TWICE (ping and pong) but were counted once, and the
// scale plane and the scale temporary were absent altogether. The comment
// above claiming "57,344 B of headroom" was computed from that wrong sum.
//
// Ordered to mirror the InitBuffer sequence in the kernel so the two can be
// diffed by eye. If a buffer is added there and not here, the assert stops
// meaning anything again.
inline constexpr std::uint32_t kCubeScaleTempBytes = 16u * 2u;

inline constexpr std::uint32_t kCubeUbBytes =
    kCubeActBytes +          // q_activation_resident
    kCubeAcc1Bytes +         // q_gate_accumulator
    kCubeAcc1Bytes +         // q_up_accumulator
    kCubeAcc1Bytes +         // q_subgemm_fwd
    kCubeXF32Bytes +         // q_tokens_f32_ping
    kCubeXF32Bytes +         // q_tokens_f32_pong
    kCubeXSubBytes +         // q_tokens_sub_bf16_ping
    kCubeXSubBytes +         // q_tokens_sub_bf16_pong
    kCubeAcc2Bytes +         // q_down_accumulator
    kCubeAcc2Bytes +         // q_subgemm_down
    kCubeASubBytes +         // q_activation_sub_bf16
    kCubeNibBytes +          // q_weight_nibbles
    kCubeWideBytes +         // q_weight_expanded
    kCubeScaleBytes +        // q_weight_scales
    kCubeScaleRowBytes +     // q_scale_plane  (was kCubeRSub * kCubeKSub * 2)
    kCubeScaleTempBytes;     // q_scale_temporary

// Everything EXCEPT the act plane. Stated separately because it is the part
// that does not move with the geometry, and because the act plane is the only
// line the inter cap can shrink -- so this is the floor any cap has to clear.
inline constexpr std::uint32_t kCubeUbBytesLessAct = kCubeUbBytes - kCubeActBytes;

static_assert(kCubeUbBytesLessAct == 111136,
              "every InitBuffer except the act plane sums to 111,136 B. This "
              "number is the ACTUAL total of the kernel's calls, verified "
              "against them one by one; the old 163,840 omitted the scale "
              "plane, the scale temporary and the second of each ping/pong "
              "pair. If this fires, re-derive it from the kernel rather than "
              "adjusting the literal");
static_assert(kCubeUbBytes == kCubeUbBytesLessAct + kCubeActBytes &&
                  kCubeUbBytes == (kCubeActInterCap == 2048u ? 176672u : 127520u),
              "the Cube prefill UB budget is 176,672 B at the production inter "
              "cap and 127,520 B at the sim cap of 512");

// kUsableUbBytes IS NOT THE BINDING CEILING, and this assert is kept because
// it is cheap, not because passing it means anything. kCubeUbBytes is not the
// whole UB charge either. Both statements are MEASURED, on the dav-3510
// CAModel, 2026-10-17, by the GetPhyAddr census in
// tests/sim/kernel/test_subgemm_stage_kernel.cpp.
//
// WHAT THE CENSUS READ, at the sim cap, stage 3, hidden 128 / inter 256:
//
//     TPipe dynamic pool base            0          NOT ~131,616
//     activation_resident                0 ..  16,384
//     ... the sixteen UB buffers ...
//     scale_temporary              127,488 .. 127,520
//     q_weight_fwd [0] (TSCM)      127,520 .. 160,288
//     q_weight_fwd [1] (TSCM)      160,288 .. 193,056
//     q_weight_down[0] (TSCM)      193,056 .. 225,824
//     q_weight_down[1] (TSCM)      225,824 .. 258,592
//
// TWO THINGS THAT WERE BELIEVED AND ARE NOT TRUE:
//
//   1. The pool does NOT start past a SIMT DataCache reservation. It starts
//      at 0. `TPipe::ResetPool()` on an AIV seeds it with
//      `GetDynamicMemStartPos<Hardware::UB>()` = `&dynamicStartUB[0]`
//      (impl/basic_api/kernel_tensor_impl.h, dav_3510 only), and in this
//      build that linker symbol resolves to zero. So the whole 262,144 B is
//      addressable, and both "buffers begin at ~131,616" and "the usable
//      ceiling is ~130,528" were readings of something else.
//
//   2. kCubeUbBytes IS NOT THE CHARGE. All four TPosition::TSCM ring slots
//      come back CONTIGUOUS with the sixteen UB buffers, 32,768 B apart,
//      running 127,520 -> 258,592. They are not in the separate L1 pool that
//      GetPhyType(TSCM) == Hardware::L1 and TPipe::TscmInitBuffer's
//      top-of-L1 allocation would predict. The ring is 131,072 B of UB.
//
// WHICH IS THE WHOLE BUG. The budget that faulted was
// 176,672 + 131,072 = 307,744 against 262,144 -- over by 45,600 B, the
// "~46 KB" the ub_addr_overflow diagnostics were reporting. The budget that
// does not is 127,520 + 131,072 = 258,592, which fits with 3,552 B to spare,
// and the fault count is zero.
//
// Measured, dav-3510 CAModel, 2026-10-17, by the GetPhyAddr census in
// tests/sim/kernel/test_subgemm_stage_kernel.cpp. Never run on hardware.
// The weight ring's slot sizes. Declared here rather than in the "L1" section
// below because the census says that is where they are SPENT: the ring is a
// line in the UB budget, and the section below is named for a pool it does
// not reach.
inline constexpr std::uint32_t kCubeB1SlotBytes = kCubeRSub * kCubeKSub * 2u;
inline constexpr std::uint32_t kCubeB2SlotBytes = kCubeHSub * kCubeRSub2 * 2u;
static_assert(kCubeB1SlotBytes == 32768 && kCubeB2SlotBytes == 32768,
              "each weight ring slot is 32,768 B");

// Two slots per queue, two queues, all four live at once -- confirmed by the
// census, which read them contiguous and 32,768 B apart.
inline constexpr std::uint32_t kCubeTscmRingBytes =
    2u * kCubeB1SlotBytes + 2u * kCubeB2SlotBytes;
static_assert(kCubeTscmRingBytes == 131072, "the weight ring costs 131,072 B");
inline constexpr std::uint32_t kCubeUbTotalBytes =
    kCubeUbBytes + kCubeTscmRingBytes;

// The largest act plane physical UB can hold, and hence the largest `inter`
// this schedule admits, derived rather than chosen:
//
//     262,144 - 111,136 (UB less act) - 131,072 (ring) = 19,936 B for act
//     19,936 / (M_TILE * 2) = 623 columns
//     rounded down to a whole phase 2 contraction sub-tile  -> 512
//
// So kCubeActInterCap = 512 is not a sim convenience, it is the ceiling.
inline constexpr std::uint32_t kCubeActInterCapMax =
    ((kUbTotalBytes - kCubeUbBytesLessAct - kCubeTscmRingBytes) /
     (kCubeMTile * 2u)) /
    kCubeRSub2 * kCubeRSub2;
static_assert(kCubeActInterCapMax == 512,
              "the act plane's ceiling is 512 inter columns at this layout");

// NOT ASSERTED, DELIBERATELY, AND THIS IS A KNOWN DEFECT RATHER THAN A GAP:
//
//     static_assert(kCubeUbTotalBytes <= kUbTotalBytes)
//
// would hold at the sim cap (258,592) and FAIL in a production build
// (307,744). The production geometry -- hidden 4096, inter 2048 -- does not
// fit this schedule's UB, so a production build today compiles a kernel that
// over-commits UB by 45,600 B and whose stores past the end the simulator
// drops silently. Writing the assert is the correct fix and it retires the
// operator's headline shape, which is a decision about what v1 supports and
// not a comment edit. Until it is taken, `kCubeActInterCap` at 2048 is known
// wrong and only the sim cap is verified.
//
// The assert below is kept because it is cheap, not because passing it means
// anything: kUsableUbBytes is 221,184 and the real ceiling is 262,144 against
// a charge this constant does not include.
static_assert(kCubeUbBytes <= kUsableUbBytes,
              "the Cube prefill UB budget exceeds the arch35 usable ceiling");

// ---------------------------------------------------------------------------
// The weight sub-tile staging the Cube reads B from
// ---------------------------------------------------------------------------
// THIS SECTION IS NAMED "L1" AND THE MEASUREMENT SAYS IT IS NOT.
//
// `TQue<TPosition::TSCM, 2>` is supposed to be an L1 allocation:
// GetPhyType(TSCM) is Hardware::L1, and TPipe::TscmInitBuffer fills it from
// the TOP of L1 downward, so the four slots should have reported addresses
// near 524,288 in a pool of their own. The GetPhyAddr census read them at
// 127,520 / 160,288 / 193,056 / 225,824 -- contiguous with the UB buffers and
// inside UB. See the census note on kCubeUbTotalBytes above; the 131,072 B
// they cost is charged there, not here.
//
// kCubeL1Bytes below is therefore NOT a measured L1 occupancy. It is kept
// because the emu golden's L1 charge is written against the same formula and
// the two are compared; it is not evidence about this kernel's L1 use, and
// the assert under it is checking an arithmetic identity rather than a
// hardware fact. WHY the slots land in UB is not yet explained -- the
// toolkit's own allocation path does not predict it -- and that is the next
// thing to understand, because 131,072 B is half this operator's budget.
//
// What the slot COUNT is for still holds, and still needs stating precisely
// because the obvious reading is wrong TODAY.
//
// Phase 1 stages W1 and W3 for the SAME (r0, k0) step -- they reduce against
// the same X sub-tile, which is the reuse the k-inner loop order exists for.
// The slot count is sized for both being live, and the emu golden's L1 charge
// (2 * R_SUB * K_SUB) assumes exactly that. But the driver as written
// SERIALIZES them: dequant W1, GEMM, free, dequant W3, GEMM. So only one slot
// is live at a time and the second is headroom, not a running overlap.
//
// It is not removed, for two reasons: the budget then stops matching the
// golden's, and overlapping the W3 dequantization with the gate GEMM needs
// only the asynchronous Iterate form rather than a new buffer -- which is the
// obvious next optimization and should not have to re-widen L1 to land.
// Phase 2's W2 sub-tile has no partner.
//
// 98,304 B on the formula the golden shares. Note it counts THREE slots while
// the ring allocates FOUR (two per queue), which is the other half of why
// this number does not describe the hardware charge -- kCubeTscmRingBytes
// does.
inline constexpr std::uint32_t kCubeL1Bytes = 2u * kCubeB1SlotBytes + kCubeB2SlotBytes;
static_assert(kCubeL1Bytes <= 524288u / 2u,
              "the formula the emu golden charges L1 by must leave at least "
              "half of L1 to the Matmul framework's A1 staging");

// --- KFC queue reserve ----------------------------------------------------
// REGIST_MATMUL_OBJ stands up a KfcServer whose message queue lives at the
// head of the workspace. It is now the ONLY thing in the workspace: the
// dequantized weight planes are gone, which is what the two-phase schedule
// bought. 1 MiB is the toolkit's own conservative reserve for the mix message
// queue plus its alignment.
inline constexpr std::uint64_t kCubeKfcReserveBytes = 1u << 20;

// Event ids. EVENT_ID0..3 only -- EVENT_ID4..7 are physically absent on
// arch35 -- and balanced 1:1 at every use so no token outlives a loop
// iteration.
//
//   kEvCubeMte2ToV     a GM burst landed in UB; the Vector may read it
//   kEvCubeVToMte3     the Vector finished w_wide; MTE3 may push it to L1
//   kEvCubeMte3ToMte1  the L1 slot is written; the Cube's MTE1 may read it
//   kEvCubeMte3ToV     the L1 push drained; the Vector may overwrite w_wide
inline constexpr std::int32_t kEvCubeMte2ToV = 0;
inline constexpr std::int32_t kEvCubeVToMte3 = 1;
inline constexpr std::int32_t kEvCubeMte3ToMte1 = 2;
inline constexpr std::int32_t kEvCubeMte3ToV = 3;

static_assert(kEvCubeMte3ToV <= static_cast<std::int32_t>(kMaxEventId),
              "arch35 has EVENT_ID0..3 only");

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube

#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
