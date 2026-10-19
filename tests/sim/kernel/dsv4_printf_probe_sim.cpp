// =================================================================================================
// dsv4_printf_probe_sim.cpp -- the minimal dav-3510 CAModel probe for __asc_aicore::printf.
//
// Answers one question and nothing else: does a device printf built with ASCENDC_DUMP=1 actually
// produce visible output when the kernel EXECUTES on the simulator? Build and run it with
// scripts/run_printf_probe_sim.sh.
//
// -------------------------------------------------------------------------------------------------
// WHY IT IS NOT A CTEST GATE
// -------------------------------------------------------------------------------------------------
// It is deliberately NOT registered in tests/sim/CMakeLists.txt. A CAModel run costs ~200-320 s of
// which almost all is fixed simulator bring-up, so a second executable is another ~300 s of wall
// time on every sim invocation -- and tests/sim/CMakeLists.txt already records the rule that new
// checks belong in an existing micro gate rather than a new binary. This probe answers a TOOLCHAIN
// question, not a question about the operator, so it has no business running on every build. Run
// it by hand when the printf plumbing is in doubt.
//
// For the same reason it is ONE translation unit -- kernel, chevron launch and main() together --
// rather than the kernel/host split the registered sim drivers use. BiSheng compiles the host half
// with its own host compiler; there is no CMake target to keep the two halves in step with.
//
// -------------------------------------------------------------------------------------------------
// WHAT IT ESTABLISHES, AND WHAT IT CANNOT
// -------------------------------------------------------------------------------------------------
// MEASURED on the dav-3510 CAModel, 2026-10-19, with and without --cce-enable-print:
//
//     with flag     4/4 records, witness 1234, exit 0, 320 s
//     without flag  4/4 records, witness 1234, exit 0, 202 s
//
// Both printed every record correctly formatted and tagged [AIV Block 0/1], so printf works on
// CANN 9.2.0 with ASCENDC_DUMP=1 and ENABLE_KERNEL_PRINTF alone, and needs no host-side
// aclrtDbgInit. The wall times are ONE run each on a simulator whose fixed cost dominates; they are
// not a measurement of what the flag costs.
//
// This is the simulator, not hardware. It cannot establish that printf works on a real 950PR
// (CLAUDE.md section 5.7).
//
// -------------------------------------------------------------------------------------------------
// THE DESIGN, SUCH AS IT IS
// -------------------------------------------------------------------------------------------------
//   * AIV-only (KERNEL_TYPE_MIX_AIV_1_0). No Cube, no cross-core rendezvous, so there is nothing
//     that can deadlock and leave "printf produced nothing" indistinguishable from "the kernel
//     never retired".
//   * A GM WITNESS WORD, for the same reason. The host reads it back and reports it, so a silent
//     absence of output is always attributable to one of the two causes rather than to either.
//   * One block, four records, three format shapes (%d, %f/%u, a bare literal). Deliberately tiny:
//     the simulator's cost is in bring-up, so a bigger kernel buys nothing here.
// =================================================================================================

#include "kernel_operator.h"

#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
#include "utils/debug/asc_printf.h"
#define PROBE_TRACE(...) __asc_aicore::printf(__VA_ARGS__)
#else
#define PROBE_TRACE(...)
#endif

#include <acl/acl.h>

#include <cstdint>
#include <cstdio>

using namespace AscendC;

extern "C" __global__ __aicore__ void dsv4_printf_probe(GM_ADDR out) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    const std::int32_t block = static_cast<std::int32_t>(GetBlockIdx());
    const std::int32_t sub   = static_cast<std::int32_t>(GetSubBlockIdx());

    PROBE_TRACE("DEVPRINT marker A: block=%d sub=%d\n", block, sub);
    PROBE_TRACE("DEVPRINT marker B: float=%f uint=%u\n", 1.5F, 4242u);
    PROBE_TRACE("DEVPRINT marker C: plain literal, no arguments\n");

    GlobalTensor<std::int32_t> witness;
    witness.SetGlobalBuffer(reinterpret_cast<__gm__ std::int32_t*>(out), 8);
    witness.SetValue(0, 1234 + block);
    DataCacheCleanAndInvalid<std::int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        witness);

    PROBE_TRACE("DEVPRINT marker D: witness written\n");
}

namespace {

bool Check(const char* what, aclError rc) {
    if (rc != ACL_SUCCESS) {
        const char* msg = aclGetRecentErrMsg();
        std::printf("HOST FAIL %s aclError=%d msg=%s\n", what, static_cast<int>(rc),
                    (msg == nullptr || *msg == '\0') ? "<none>" : msg);
        return false;
    }
    return true;
}

}  // namespace

int main() {
#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
    std::printf("HOST: built with ENABLE_KERNEL_PRINTF=1\n");
#else
    std::printf("HOST: built WITHOUT ENABLE_KERNEL_PRINTF\n");
#endif
    std::fflush(stdout);

    if (!Check("aclInit", aclInit(nullptr))) {
        return 2;
    }
    if (!Check("aclrtSetDevice", aclrtSetDevice(0))) {
        return 2;
    }
    aclrtStream stream = nullptr;
    if (!Check("aclrtCreateStream", aclrtCreateStream(&stream))) {
        return 2;
    }

    void* out = nullptr;
    if (!Check("aclrtMalloc", aclrtMalloc(&out, 64, ACL_MEM_MALLOC_HUGE_FIRST))) {
        return 2;
    }
    const std::int32_t zeroed[16] = {};
    if (!Check("aclrtMemcpy(zero)",
               aclrtMemcpy(out, 64, zeroed, 64, ACL_MEMCPY_HOST_TO_DEVICE))) {
        return 2;
    }

    std::printf("HOST: launching dsv4_printf_probe, block_dim=1\n");
    std::fflush(stdout);

    dsv4_printf_probe<<<1, 0u, stream>>>(reinterpret_cast<std::uint8_t*>(out));

    if (!Check("aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
        return 2;
    }

    std::int32_t host[16] = {};
    if (!Check("aclrtMemcpy(D2H)", aclrtMemcpy(host, 64, out, 64, ACL_MEMCPY_DEVICE_TO_HOST))) {
        return 2;
    }
    std::printf("HOST: witness[0]=%d (expected 1234 => the kernel retired)\n", host[0]);

    (void)aclrtFree(out);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();

    const int rc = host[0] == 1234 ? 0 : 1;
    std::printf("HOST: done. rc=%d\n", rc);
    return rc;
}
