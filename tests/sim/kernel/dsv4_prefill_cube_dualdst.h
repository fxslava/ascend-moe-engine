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
inline constexpr std::uint16_t kDualDstFlagReady = 0;

// Stage selector, passed in the status buffer by the launcher.
//   1  AIC Cube chain only -- the AIVs record their UB address and exit
//      without waiting. Answers "does GM->L1->L0->Mmad->Fixpipe retire".
//   2  full probe, including the cross-core handshake and the copy-out.
//   3  as 2, but the AIV waits with EXPLICIT <mode 2, PIPE_MTE3> instead of
//      the default template arguments.
//   4  as 3, plus the AIV -> AIC rendezvous: the AIVs set a flag and the AIC
//      waits on it BEFORE the Cube runs. THIS IS THE WORKING CONFIGURATION
//      and the default.
//
// WHAT THE LADDER ESTABLISHED, because the result is not what the mode
// documentation on its own suggests:
//
//   stage 1   retires, 17.3 s   the Cube chain alone is fine
//   stage 2   HANGS             AIC->AIV flag, default wait
//   stage 3   HANGS             AIC->AIV flag, explicit <2, PIPE_MTE3> wait
//   stage 4   retires, 14.6 s   the same, once the AIVs ALSO signal the AIC
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
inline constexpr std::uint16_t kDualDstFlagOperands = 1;

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
