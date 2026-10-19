/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- MINIMAL PROBE: does a mode-2 CrossCoreWaitFlag block when nobody
// has set the flag?
//
// ===========================================================================
// THE QUESTION
// ===========================================================================
// MEASURED on the dav-3510 CAModel, 2026-10-19, by the cycle-stamped progress
// trace of dsv4_moe_prefill_cube (tests/sim/kernel/
// dsv4_moe_prefill_cube_trace_kernel.cpp), at hidden 128 / inter 256:
//
//     60632  AIC   C p1 k=0 WAIT StepReady slot=0
//     66922  AIC   C p1 k=0 GOT StepReady      <-- cleared
//     73953  AIC   C p1 k=0 loaded L0A/L0B     <-- reads L1 slot 0
//     75853  AIV1  V p1 blk ib=0 steps=1       <-- AIVs only reach the block now
//     75946  AIV0  V p1 blk ib=0 steps=1
//
// GetSystemCycle() is device-wide, so that ordering is not a log artefact: the
// Cube's FIRST mode-2 wait on kCubeFlagStepReady cleared about 9,000 cycles
// before either vector core had entered the phase 1 block, and long before
// either could have issued CrossCoreSetFlag. Zero matching sets had occurred.
//
// Two mechanisms predict that, and they need different repairs:
//
//   RESIDUAL STATE -- the flag or its counter is nonzero when the kernel
//       starts, so the FIRST wait on each id clears for free and every later
//       one blocks normally. Repair: drain each id once at entry, or rendezvous
//       before the pipeline starts.
//   NON-BLOCKING WAIT -- a mode-2 CrossCoreWaitFlag with an unconfigured count
//       does not block AT ALL in this configuration. Repair: the handshake
//       cannot be built on this primitive as used, which is a design change,
//       not a drain.
//
// ===========================================================================
// THE EXPERIMENT, AND WHY ITS CHANNEL IS TERMINATION AND NOT PRINTING
// ===========================================================================
// One AIC issues N CONSECUTIVE mode-2 waits on kProbeFlagId, which NOTHING
// EVER SETS. N is read from GM so the same binary can be run at N = 1, 2, ...
// The AIVs enter, announce themselves, set nothing and return.
//
// The reading is WHETHER THE KERNEL RETURNS:
//
//   N = 1 returns              at least one wait cleared with no matching set
//   N = 2 returns              at least two cleared -> the wait does not block
//   N = 1 returns, N = 2 hangs EXACTLY ONE free clear per id -> RESIDUAL STATE
//   N = 1 hangs                the wait blocks correctly, and the cycle-stamped
//                              trace that motivated this probe needs another
//                              explanation entirely
//
// THE FIRST VERSION OF THIS PROBE COUNTED PRINTED RECORDS, AND THAT DID NOT
// WORK. MEASURED: at N = 4 the run emitted NOT ONE device record before the
// watchdog ended it at 120 s -- no "AIC begin", no "AIC WAIT 0" -- while the
// much larger progress-trace probe emits hundreds. The debug ring buffer is
// evidently drained on fill or at kernel end, so a kernel that prints a
// handful of records and then blocks never flushes any of them, and the
// watchdog's _Exit discards the buffer. Absence of records was therefore
// indistinguishable from "the wait blocked", which is the one distinction the
// probe exists to make.
//
// Termination needs no flush. The records below are kept because they cost
// nothing and are useful on the runs that DO return, but nothing is concluded
// from their absence.
//
// kProbeFlagId IS THE PRODUCTION ID AND THE MODE IS THE PRODUCTION MODE.
// Using some unrelated spare id would answer a question about that id instead:
// if the behaviour is per-id residue, only the id the operator actually waits
// on can show it. kCubeFlagStepReady and kCubeSyncMode are taken from the
// operator's own header for exactly that reason, and the wait is scoped to
// PIPE_MTE1 as RunCubeCore scopes it.

#include "kernel_operator.h"
#include "utils/debug/asc_printf.h"

#include "dsv4_moe_prefill_cube_arch35.h"

namespace moe::kernels::arch35 {

using namespace AscendC;

namespace {

using dsv4_moe_prefill_cube::kCubeAivPerAic;
using dsv4_moe_prefill_cube::kCubeFlagStepReady;
using dsv4_moe_prefill_cube::kCubeSyncMode;

inline constexpr std::uint16_t kProbeFlagId = kCubeFlagStepReady;

// Bounded so a corrupt or uninitialised GM word cannot turn the probe into an
// unbounded loop that looks like the hang it is trying to characterise.
inline constexpr std::uint32_t kProbeMaxWaits = 8;

// ---------------------------------------------------------------------------
// MODE 1: does a GM handshake actually make an AIV's write visible to the AIC?
// ---------------------------------------------------------------------------
// The startup rendezvous added to the operator is built on GM plus
// DataCacheCleanAndInvalid, deliberately avoiding the cross-core flag whose
// behaviour is unexplained. With that barrier in, the 4-step run produced
// EXACTLY the signature of a deadlock IN THE BARRIER: the four
// `execute_set_flag` collisions vanished AND the launch still did not retire.
// That pairing is what a stuck Cube predicts -- each AIV pushes, sets
// StepReady, signals, drains, then blocks on its Product wait, so it never
// reaches the LandingFree sets that were colliding.
//
// So the primitive itself has to be checked before anything is built on it.
// Mode 1 is the barrier, alone: both AIVs write their word and flush it, the
// AIC spins until it observes both. If the kernel RETURNS, GM-plus-dcci gives
// the Cube visibility of an AIV's write and the barrier is sound. If it HANGS,
// the barrier is the new deadlock and the operator's rendezvous must be built
// another way.
//
// The handshake words sit at 256 and 384 bytes, clear of the mode word at
// offset 0, each on its own 128-byte line so the two writers never share one.
inline constexpr std::uint32_t kProbeHandshakeBase = 256;
inline constexpr std::uint32_t kProbeHandshakeLine = 128;

__aicore__ inline void ProbeHandshakeWord(GM_ADDR scratch, std::uint32_t idx,
                                          GlobalTensor<int64_t>& word) {
    word.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ std::uint8_t*>(scratch) +
                                          kProbeHandshakeBase + idx * kProbeHandshakeLine),
        1);
}

// PRINTS ARE COMPILED OUT WHEN DSV4_PROBE_SILENT IS SET, AND THAT IS NOT A
// CONVENIENCE -- IT IS WHAT MAKES THIS PROBE SOUND.
//
// Both of this probe's modes print immediately BEFORE the operation they
// measure. Commit ab32d17 records that device printf needs
// `--cce-enable-print` and not merely ASCENDC_DUMP, and these objects are not
// compiled with that flag. If an unenabled printf HANGS rather than no-ops,
// then every "it blocked" reading from this probe was a stall in the PRINT
// PATH and not in the primitive -- the probe would have been measuring its own
// instrumentation. Compiling the prints out removes that confound and leaves
// TERMINATION as the only channel, which needs no flush, no flag and no print
// support.
#if defined(DSV4_PROBE_SILENT)
#define DSV4_PROBE_TRACE(fmt, ...) ((void)0)
#else
#define DSV4_PROBE_TRACE(fmt, ...)                                             \
    __asc_aicore::printf("@%u " fmt,                                           \
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(GetSystemCycle())\
                                   & 0xFFFFFFFFull),                           \
        ##__VA_ARGS__)
#endif

}  // namespace

extern "C" __global__ __aicore__ void dsv4_crosscore_wait_probe(GM_ADDR scratch) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (unlikely(scratch == nullptr)) {
        return;
    }

    // Host-written control word at offset 0. Values 1..kProbeMaxWaits select
    // MODE 0 (that many consecutive unmatched cross-core waits); the sentinel
    // selects MODE 1 (the GM handshake in isolation).
    constexpr std::uint32_t kProbeModeHandshake = 0xFFFFFFFFu;
    const std::uint32_t control = *reinterpret_cast<__gm__ std::uint32_t*>(scratch);

    if (control == kProbeModeHandshake) {
        // --- MODE 1: the GM handshake, alone -------------------------------
        if ASCEND_IS_AIV {
            const std::uint32_t sub = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());
            GlobalTensor<int64_t> word;
            ProbeHandshakeWord(scratch, sub, word);
            // Barrier-store-flush, the order kfc_comm_server_gm.h uses for its
            // own scalar GM writes (FreeUB: Barrier, raw store, dcci).
            __asm__ __volatile__("" ::: "memory");
            word.SetValue(0, 1);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(word);
            __asm__ __volatile__("" ::: "memory");
            DSV4_PROBE_TRACE("AIV sub=%u signalled the GM handshake\n", sub);
        }
        if ASCEND_IS_AIC {
            GlobalTensor<int64_t> word[kCubeAivPerAic];
            for (std::uint32_t s = 0; s < kCubeAivPerAic; ++s) {
                ProbeHandshakeWord(scratch, s, word[s]);
            }
            DSV4_PROBE_TRACE("AIC spinning on %u GM handshake words\n",
                             static_cast<std::uint32_t>(kCubeAivPerAic));
            // THE READ PATTERN IS COPIED FROM THE TOOLKIT'S OWN dav-3510 KFC
            // SERVER, kfc_comm_gm.h RcvMessageImpl, which is the AIC polling a
            // GM word an AIV wrote -- this exact problem:
            //
            //     Barrier();
            //     dcci(msg, SINGLE_CACHE_LINE, CACHELINE_OUT);
            //     Barrier();
            //     dc_preload((__gm__ uint64_t*)msg, 0);
            //     ... msg->head          // raw __gm__ dereference
            //
            // where Barrier() is `__asm__ __volatile__("" ::: "memory")`.
            //
            // THE COMPILER BARRIERS ARE THE PART THAT WAS MISSING. Without one
            // inside the loop, and with a non-volatile read, nothing stops the
            // compiler hoisting the load out of a `for (;;)` whose body it can
            // prove writes nothing -- the core then spins on a register copy
            // and can never observe the write, which is exactly the deadlock
            // measured. The dcci alone does not help: it is a cache operation,
            // not a sequence point for the optimiser.
            for (;;) {
                bool all_ready = true;
                for (std::uint32_t s = 0; s < kCubeAivPerAic; ++s) {
                    __asm__ __volatile__("" ::: "memory");
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE,
                                             DcciDst::CACHELINE_OUT>(word[s]);
                    __asm__ __volatile__("" ::: "memory");
                    const __gm__ volatile int64_t* raw =
                        reinterpret_cast<const __gm__ volatile int64_t*>(
                            reinterpret_cast<__gm__ std::uint8_t*>(scratch) +
                            kProbeHandshakeBase + s * kProbeHandshakeLine);
                    if (*raw == 0) {
                        all_ready = false;
                    }
                }
                if (all_ready) {
                    break;
                }
            }
            DSV4_PROBE_TRACE("AIC observed BOTH GM handshake words\n");
        }
        return;
    }

    // --- MODE 0: unmatched cross-core waits --------------------------------
    std::uint32_t waits = control;
    if (waits > kProbeMaxWaits) {
        waits = kProbeMaxWaits;
    }

    if ASCEND_IS_AIC {
        DSV4_PROBE_TRACE("AIC begin: %u consecutive mode-%u waits on flag %u, "
                         "nothing sets it\n",
                         waits, static_cast<std::uint32_t>(kCubeSyncMode),
                         static_cast<std::uint32_t>(kProbeFlagId));
        for (std::uint32_t i = 0; i < waits; ++i) {
            DSV4_PROBE_TRACE("AIC WAIT %u\n", i);
            CrossCoreWaitFlag<kCubeSyncMode, PIPE_MTE1>(kProbeFlagId);
            DSV4_PROBE_TRACE("AIC GOT %u (cleared with no matching set)\n", i);
        }
        DSV4_PROBE_TRACE("AIC end: all %u waits cleared for free\n", waits);
    }

    if ASCEND_IS_AIV {
        // Announce and leave. The AIVs exist only so the launch is a real
        // MIX_AIC_1_2 grid -- the same topology the measurement came from --
        // and they deliberately set NOTHING.
        const std::uint32_t sub = static_cast<std::uint32_t>(GetSubBlockIdx());
        DSV4_PROBE_TRACE("AIV sub=%u enter, sets nothing, exits\n", sub);
    }
}

}  // namespace moe::kernels::arch35
