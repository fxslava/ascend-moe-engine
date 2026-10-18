#ifndef DSV4_PREFILL_CUBE_DUALDST_H
#define DSV4_PREFILL_CUBE_DUALDST_H

// SIM ONLY -- shapes and wire format for the STEP 1 Fixpipe dual-destination
// probe, shared by the device kernel, the launch wrapper and the host checker
// so no two of them can disagree about a size.
//
// ACL-free and device-header-free: the host TU includes it with a stock g++.
//
// ---------------------------------------------------------------------------
// WHAT THIS PROBE IS FOR
// ---------------------------------------------------------------------------
// `dsv4_moe_prefill_cube` drives the Cube through the high-level Matmul
// framework (REGIST_MATMUL_OBJ / Iterate / GetTensorC). That framework cannot
// reach DUAL_DST_SPLIT_M with a UB destination: the mode requires
// `A_TYPE::ibShare && B_TYPE::ibShare` (copy_cube_out_fixpipe.h
// SplitMNParamsCheck), and with ibShare set `Iterate()` returns false
// immediately under USE_WORKSPACE, which is the mode this toolchain selects
// (matmul_client.h:716-721; KFC_C310_SSBUF is pre-defined to 0). IterateAll's
// LocalTensor overload is excluded twice over -- it wants C in L1 and forbids
// ibShare. So there is no surviving move-out call.
//
// This probe therefore bypasses the framework entirely and builds the Cube
// step out of Tensor API primitives, where the trait is ours to set:
//
//     GM(ND) -> L1    CopyGM2L1    A as NZ, B as ZN (ND2ZN is an inline
//                                  conversion this path supports)
//     L1     -> L0A   CopyL12L0A   non-transpose, NZ -> NZ
//     L1     -> L0B   CopyL12L0B   non-transpose, ZN -> ZN
//     L0A,L0B-> L0C   Mmad         bfloat16_t x bfloat16_t -> float
//     L0C    -> UB    CopyL0C2UB   DUAL_DST_SPLIT_M, float -> bfloat16_t
//
// The last step is the whole point: it splits M across the two AIVs' UBs and
// casts in-line, so no FP32 C ever lands in UB.
//
// ---------------------------------------------------------------------------
// THE TEST MATRICES
// ---------------------------------------------------------------------------
// A[m][k] = (k == 0) ? (m + 1)       : 0
// B[k][n] = (k == 0) ? ((n % 8) + 1) : 0
//   =>  C[m][n] = (m + 1) * ((n % 8) + 1)
//
// EVERY VALUE IS EXACT IN BF16, and that is a chosen property rather than a
// lucky one. BF16 keeps 8 significand bits, so it represents integers exactly
// only up to 256; the largest product here is 16 * 8 = 128. The obvious
// choice of (m+1)*(n+1) would NOT do: 15 * 31 = 465 needs 9 significand bits
// and would come back rounded, which turns a clean equality check into a
// tolerance argument about the thing being tested.
//
// The `% 8` on the N factor keeps the product inside that bound while leaving
// the value dependent on BOTH axes, so a transposed or mis-strided result
// still fails.
//
// Row identity is what the probe is really asserting: sub-block 0 must come
// back holding rows 0..7 and sub-block 1 rows 8..15, in that order. The M
// factor is (m + 1), so each row carries its own index and a swapped or
// duplicated half is visible by inspection.

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kDualDstM = 16;  // split 8 / 8 across the AIVs
inline constexpr std::uint32_t kDualDstN = 32;
inline constexpr std::uint32_t kDualDstK = 32;

static_assert(kDualDstM % 2 == 0,
              "DUAL_DST_SPLIT_M requires M to be a multiple of 2");

inline constexpr std::uint32_t kDualDstMHalf = kDualDstM / 2;

// Each AIV copies its half out to its own slice of the output buffer.
inline constexpr std::uint32_t kDualDstHalfElems = kDualDstMHalf * kDualDstN;
inline constexpr std::uint32_t kDualDstOutElems = 2u * kDualDstHalfElems;

// THE LANDING ZONE IS AT UB OFFSET 0, AND THAT IS A MEASURED FACT.
//
// The AIC issues the Fixpipe with a UB address that is interpreted in EACH
// AIV's own UB -- the two are separately addressed 0x00000..0x3FFFF -- so both
// sides have to agree on one offset. Offset 0 works because the AIV's TPipe
// dynamic pool starts at 0 on this toolchain: `TPipe::ResetPool()` seeds it
// with `GetDynamicMemStartPos<Hardware::UB>()` = `&dynamicStartUB[0]`, and
// that linker symbol resolves to zero here. That was measured by the
// GetPhyAddr census in test_subgemm_stage_kernel.cpp, not assumed -- and this
// probe re-checks it, reporting the AIV's first InitBuffer address in the
// status block so a toolchain that moves the base fails loudly instead of
// writing into the wrong buffer.
inline constexpr std::uint32_t kDualDstUbOffset = 0;

// Cross-core handshake: ONE flag, mode 2, set on PIPE_FIX, waited with the
// DEFAULT template arguments.
//
// The asymmetry is not a slip. The documented modes are: 0 = all AICs or all
// AIVs finished, 1 = among the AIVs of one AI Core, 2 = between the AIC and
// ALL AIVs of one AI Core, 4 = between the AIC and a SINGLE AIV of one AI
// Core (inter_core_synchronization_capability_overview.md). Mode 2 is what
// "the Cube finished, both vector subcores may read their half" means, so one
// flag serves both AIVs and there is no per-AIV id to get wrong.
//
// MEASURED, and the reason this is spelled out: setting AND waiting with
// mode 4 does not complete. That probe spun past 220,000 simulated cycles
// with the count still climbing and tripped a 300 s watchdog, while the same
// kernel with the AIV wait removed retires in 14.9 s -- so the Cube chain was
// never the slow part, the handshake simply never fired.
//
// The pairing used here is the one the shipping turboquant decode kernel uses
// on this hardware (csrc/attention/turboquant/op_kernel): AIC
// `CrossCoreSetFlag<kSubBlockSyncMode = 0x2, PIPE_FIX>(flag)`, AIV
// `CrossCoreWaitFlag(flag)` with no template arguments at all.
inline constexpr std::uint8_t  kDualDstSyncMode = 0x2;

// --- THE WAIT SIDE'S TEMPLATE ARGUMENTS ARE NOT COSMETIC ON 950PR --------
//
// `CrossCoreWaitFlag` is declared
//
//     template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
//
// so the bare `CrossCoreWaitFlag(flag)` spelling is <modeId 0, PIPE_S>, NOT
// mode 2. `CrossCoreSetFlag` by contrast declares NO defaults, which is why
// only the wait side can be written bare at all.
//
// On Ascend 950PR/950DT the spec says both template arguments are EFFECTIVE
// and "CrossCoreWaitFlag blocks the subsequent instructions of the SPECIFIED
// pipeline"; on A2/A3 the hardware ignores them and blocks ALL pipelines.
// So the bare spelling that is correct-by-accident on A2/A3 degenerates on
// 950PR into a PIPE_S-scoped block in the wrong mode, and PIPE_S is listed
// as NOT SUPPORTED for modes 0, 1 and 2.
//
// THE SPEC CONTRADICTS ITSELF HERE AND THE CONTRADICTION IS WORTH RECORDING:
// the constraint text forbids PIPE_S on the wait side for modes 0/1/2, while
// the normative calling example under "Full-Core Synchronization Between AIC
// and AIV" uses exactly the bare default --
//
//     if ASCEND_IS_AIV { CrossCoreSetFlag<2, PIPE_MTE3>(0); }
//     if ASCEND_IS_AIC { CrossCoreWaitFlag(0); }
//
// -- as do the mode 0 and mode 1 examples. The shipping turboquant kernel
// follows the examples. So "the default spelling is undefined behaviour" is
// NOT established; what IS established from the prototype is that the
// default is mode 0 + PIPE_S, which is a different mode and the one pipe
// scope the constraint list excludes.
//
// THE AIC's RENDEZVOUS WAIT KEEPS THE BARE SPELLING, because that is the
// spelling the passing configuration carries and the explicit one at that one
// site has never been measured green. See the flagId note below for the
// three-point bisection that assigned the blame, and note what it exonerates:
// an earlier reading of this probe's failure blamed the wait spelling, on two
// data points that differed in the flagIds as well. It was the flagIds.
//
// WHICH PIPE. The wait blocks one pipeline's issue, so the pipe must be the
// one that CONSUMES the data the flag protects. Picking any other pipe does
// not fail loudly -- it stops gating the consumer, which turns a deadlock
// into a silent wrong answer. Per site:
//
//   AIV waits kDualDstFlagReady    -> next AIV op is DataCopy(out, landing),
//                                     UB -> GM, so PIPE_MTE3.
//   AIC waits kDualDstFlagOperands -> next AIC op is CopyGM2L1, so PIPE_MTE2.
//
// PIPE_V would be wrong for the AIV: there is no vector instruction after
// the wait, so a PIPE_V-scoped block gates nothing and the MTE3 copy-out
// reads the landing zone before the Fixpipe has written it.
// The two pipe_t constants themselves live in the DEVICE TU
// (test_fixpipe_dualdst_kernel.cpp, kDualDstWaitPipeAiv/Aic). `pipe_t` is an
// AscendC device type and this header is included by the host checker with a
// stock g++, so naming it here would break that build.

// --- flagId allocation ---------------------------------------------------
//
// Modes 0/1/2 give each core 16 flagIds (0-15) and two ranges are reserved
// by other CANN machinery:
//
//   [0, 2N-1]  the Matmul high-level API, for N defined Matmul objects, so
//              [0, 7] at the maximum of 4 objects.
//   [11, 14]   the SyncAll hardware synchronisation API.
//
// This probe defines NO Matmul object and calls no SyncAll, so neither range
// is actually live here -- but it only needs two flags, so it takes them
// from outside both ranges and the question does not arise.
// --- MEASURED: THE FLAGIDS CANNOT BE MOVED FREELY, AND 8/9 DO NOT WORK ---
//
// These were briefly moved to 8 and 9 on the reasoning that [0, 2N-1] belongs
// to the Matmul API and [11, 14] to SyncAll, so 8/9 are "isolated". They are
// back at 0/1 because that is what the committed, passing configuration used
// -- NOT because 8/9 were shown to be bad. That attempt is worth recording
// precisely, because it is a lesson about this gate and not about flagIds:
//
//   dav-3510 CAModel, 2026-10-18, stage 4, in run order:
//     explicit AIC wait + 8/9   HANGS at the 200 s watchdog
//     bare AIC wait    + 0/1    RETIRES in 13.0 s, PASSES   (= committed)
//     bare AIC wait    + 8/9    HANGS
//     explicit AIC wait + 0/1   HANGS                       (== committed,
//                                                            behaviourally)
//
// The fourth run is the one that matters. It differs from the second only by
// an unused `constexpr pipe_t` and three static_asserts -- no codegen -- and
// it hung where the second passed. SO THIS PROBE IS NONDETERMINISTIC, and the
// two "8/9 hangs" rows carry no information: a flake explains them just as
// well. An earlier reading of rows 1-3 concluded that the flagIds were the
// cause; that conclusion was unsound and is retracted here.
//
// Nothing in the toolkit predicts a flagId restriction either: GetffstMsg
// masks the id with 0xf so 0..15 are representable, and the only documented
// reservations are the Matmul range and SyncAll's [11, 14] (SYNC_AIC_FLAG 11,
// SYNC_AIV_FLAG 12, SYNC_AIC_AIV_FLAG 13, SYNC_AIV_ONLY_ALL 14, in
// dav_3510/kernel_operator_sync_impl.h).
//
// WHAT THIS GATE NEEDS BEFORE IT CAN SETTLE ANYTHING: a repeat count. One run
// is not a result here, and the intermittency has to be characterised and
// explained before any single-run A/B over this probe is worth believing.
inline constexpr std::uint16_t kDualDstFlagReady    = 0;  // AIC -> both AIVs

// Stage selector, passed in the status buffer by the launcher.
//   1  AIC Cube chain only -- the AIVs record their UB address and exit
//      without waiting. Answers "does GM->L1->L0->Mmad->Fixpipe retire".
//   2  full probe, including the cross-core handshake and the copy-out.
//   3  as 2, but the AIV waits with the BARE DEFAULT template arguments,
//      which resolve to <modeId 0, PIPE_S>. Retained as the known-bad rung:
//      it is the spelling the turboquant kernel and the spec's own mode 2
//      example use, and the one this probe no longer relies on.
//   4  correctly scoped waits -- <mode 2, consuming pipe> at every site --
//      PLUS the AIV -> AIC rendezvous: the AIVs set a flag and the AIC waits
//      on it BEFORE the Cube runs. THIS IS THE WORKING CONFIGURATION and the
//      default.
//
// STAGES 3 AND 4 DIFFER IN TWO THINGS AT ONCE, DELIBERATELY. Stage 4 is the
// configuration under test; stage 3 exists to show that the bare default
// wait is not what carries it. A rung that changed only the spelling, with
// the rendezvous already in place, is the next bisection to run if stage 4
// ever regresses -- it is not pre-registered here because the ladder is
// `>=`-gated and adding it between 3 and 4 would renumber the gate.
//
// WHAT THE LADDER ESTABLISHED, because the result is not what the mode
// documentation on its own suggests:
//
//   stage 1   retires, 17.3 s   the Cube chain alone is fine
//   stage 2   HANGS             AIC->AIV flag, bare default wait
//   stage 3   HANGS             AIC->AIV flag, bare default wait (renumbered;
//                               the rung that measured HANGS carried the
//                               explicit <2, PIPE_MTE3> wait, which stage 4
//                               now carries instead)
//   stage 4   retires, 14.6 s   the same, once the AIVs ALSO signal the AIC
//
// READ THAT TABLE WITH ITS DATE. Rows 1, 2 and 4 were measured 2026-10-17
// against the PREVIOUS stage semantics, where stages 3 and 4 BOTH took the
// explicit <2, PIPE_MTE3> branch because the gate was `stage >= 3`. So the
// 14.6 s stage 4 result WAS an explicitly scoped wait, and what rows 2 and 3
// jointly establish is that no spelling of a ONE-WAY AIC -> AIV release
// fires. Stage 3 is now the bare-default rung and has not been re-measured
// under that meaning.
//
// So a one-way AIC -> AIV release does not fire, whichever way the wait is
// spelled. The AIV -> AIC direction has to be exercised first. Mode 2 is
// documented as a counter per (flagId, receiving core) with both directions
// independently valid, which would have made stage 2 work; it does not.
// The shipping turboquant decode kernel has the same rendezvous
// (kFlagOperandsReady set by both AIVs, waited by the AIC, before the first
// GEMM), which looked like a data dependency and is also a hard requirement
// of the handshake.
inline constexpr std::uint32_t kDualDstStageAicOnly   = 1;
inline constexpr std::uint32_t kDualDstStageFull      = 2;
inline constexpr std::uint32_t kDualDstStageWaitMode2 = 3;
inline constexpr std::uint32_t kDualDstStageRendezvous = 4;

// The AIV -> AIC direction, used only by stage 4.
inline constexpr std::uint16_t kDualDstFlagOperands = 1;  // both AIVs -> AIC

static_assert(kDualDstFlagReady <= 15 && kDualDstFlagOperands <= 15,
              "a mode 2 flagId is 4 bits; a larger value is silently "
              "truncated to its low 4 bits rather than rejected");
static_assert(kDualDstFlagReady != kDualDstFlagOperands,
              "the two directions must not share a counter");
static_assert((kDualDstFlagReady < 11 || kDualDstFlagReady > 14) &&
                  (kDualDstFlagOperands < 11 || kDualDstFlagOperands > 14),
              "flagIds 11-14 are reserved by the SyncAll hardware "
              "synchronisation API");

// --- status block -------------------------------------------------------
//
// ONE CACHE LINE PER WRITER, AND THAT IS LOAD-BEARING.
//
// Three cores report here, and each reports with SetValue followed by
// DataCacheCleanAndInvalid. A clean-and-invalidate writes back the whole
// LINE, so two cores touching different words of one line race: the second
// writeback carries the second core's stale copy of the first core's word and
// silently erases it.
//
// MEASURED: with all eight words packed together, a stage-1 run came back
// with AIV1's liveness word set but its UB-address word still at the host's
// 0xFFFFFFFF fill, and the AIC's liveness word erased although the run
// retired cleanly in 14.9 s. Both are impossible for a single writer and are
// exactly what the race produces.
//
// The regions are 128 B apart rather than 64 so the separation does not
// depend on which line size this part has.
inline constexpr std::uint32_t kDualDstStatusStride = 32;  // words, = 128 B
inline constexpr std::uint32_t kDualDstStatusWords = 4u * kDualDstStatusStride;

// Region 0: the AIC.          +0 liveness
// Region 1: AIV sub-block 0.  +0 liveness, +1 landing-zone UB address,
//                             +2 GetSubBlockNum()
// Region 2: AIV sub-block 1.  same
// Region 3: host input only.  +0 the stage selector; no core writes here, so
//                             no core's writeback can erase it.
inline constexpr std::uint32_t kDualDstAicRegion = 0;
inline constexpr std::uint32_t kDualDstAivRegion = 1;  // + sub-block index
inline constexpr std::uint32_t kDualDstHostRegion = 3;

inline constexpr std::uint32_t kDualDstSlotLive = 0;
inline constexpr std::uint32_t kDualDstSlotAddr = 1;
inline constexpr std::uint32_t kDualDstSlotSubs = 2;

inline constexpr std::uint32_t kDualDstStageWord =
    kDualDstHostRegion * kDualDstStatusStride;
inline constexpr std::uint32_t kDualDstAicDone = 0xA1CD09E5u;
inline constexpr std::uint32_t kDualDstAivDone = 0xA2D09E51u;

}  // namespace moe::kernels::arch35

#endif  // DSV4_PREFILL_CUBE_DUALDST_H
