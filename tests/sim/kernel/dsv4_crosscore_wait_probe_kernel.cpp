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

using dsv4_moe_prefill_cube::kCubeFlagStepReady;
using dsv4_moe_prefill_cube::kCubeSyncMode;

inline constexpr std::uint16_t kProbeFlagId = kCubeFlagStepReady;

// Bounded so a corrupt or uninitialised GM word cannot turn the probe into an
// unbounded loop that looks like the hang it is trying to characterise.
inline constexpr std::uint32_t kProbeMaxWaits = 8;

#define DSV4_PROBE_TRACE(fmt, ...)                                             \
    __asc_aicore::printf("@%u " fmt,                                           \
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(GetSystemCycle())\
                                   & 0xFFFFFFFFull),                           \
        ##__VA_ARGS__)

}  // namespace

extern "C" __global__ __aicore__ void dsv4_crosscore_wait_probe(GM_ADDR scratch) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (unlikely(scratch == nullptr)) {
        return;
    }

    // The wait count, written by the host at offset 0.
    std::uint32_t waits = *reinterpret_cast<__gm__ std::uint32_t*>(scratch);
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
