# Matmul contracts and error teardown

Reference: `Ascend Operator Library API Reference.pdf`, issue 1 (2026-09-29),
sections 4.7-4.9, 4.16, 4.31, 4.38-4.42 and 8.1.2.33. Chapter 5 points to
separate operator specifications rather than embedding their dtype tables.
The Doxygen declarations also use the installed CANN 9.2.0-beta.2 headers and
the public [Matmul](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnMatmul.md),
[Cast](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-math/aclnnCast.md),
[GroupedMatmulV5](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnGroupedMatmulV5.md),
and [FusedInferAttentionScoreV5](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScoreV5.md)
specifications. Notes explicitly distinguish the engine's selected dtype paths
from the APIs' wider, hardware-dependent support. They are not a certification
of every optional parameter combination on 950PR, 9589, or 910B.

## Why the previous Matmul plans were invalid

The error location in `PlanAclnnOp` was the common status check, not the root
cause. The router passed `[1,4096] x [256,4096]`; the LM head similarly supplied
checkpoint `[N,K]` weights to an API with no transpose attribute. The mock
explicitly accepted an imaginary implicit transpose. The combine then passed
FP32 routing weights and BF16 expert activations to ordinary Matmul.

Router and head checkpoint weights are now transposed, bit for bit, into
contiguous `[K,N]` ND storage during ingestion. Bounded pinned tiles avoid a
second full head matrix in RAM/HBM. Actual safetensors metadata must identify
BF16 `[N,K]` data; a same-size FP16 tensor is not silently reinterpreted.

All three dense Matmuls now use homogeneous BF16 A/B/C. Router logits are
explicitly cast to FP32 for softplus, square root and gating. Expanded FP32
routing weights are explicitly cast to BF16 before combination. **This chooses
BF16 rounding at the router output and combine weights.** It removes reliance
on implicit promotion, but numerical parity with an FP32-routing reference
still requires a hardware accuracy test; mock tests cannot establish it.

The dense planner checks shape, strides, format, zero offset, dtype and absolute
32-byte pointer alignment. Address rebinding also checks 32-byte alignment.
FP8 backbone projections continue to use QuantMatmulV5; FP4 routed experts use
the grouped quantization-aware path and its scales. Unexpected quantized router
or head checkpoints fail explicitly rather than guessing their scale scheme.

There are no direct `l0op::` calls in engine headers or sources. Public ACLNN
operators may use internal L0 implementations: exported symbols and a mock pass
do not establish kernel compatibility on a particular SoC. The undeclared FIA
maximum-workspace helper remains in the diagnostic inventory but is no longer
called through a guessed ABI. Workspace follows the public planner for the
fixed descriptor geometry; future variable-shape replanning needs its own
capacity/plan-lifetime design.

## Ownership and shutdown

- `AclStreamGuard`, `AclContextGuard`, `AclEventGuard`, `OpExecutorGuard`,
  `DeviceMemoryGuard` and `HostMemoryGuard` are noncopyable scoped owners.
- `ResourceScope` rolls back allocations, streams and events even when an
  engine constructor throws. Backend ownership provides an additional safety
  net for handles not yet transferred to a client scope.
- Runtime waits use `aclrtSynchronizeStreamWithTimeout` (30 seconds). Teardown
  drains streams with a 5-second timeout and force-destroys a failed stream.
  Exceptions carry `aclGetRecentErrMsg()` captured before cleanup overwrites it.
- Pipeline teardown drains work before freeing mailboxes or destroying
  repeatable executors. Router executors are explicitly released before the
  pipeline-owned descriptor arena. Device owners then release context and reset
  their device. Cleanup functions report errors without throwing.
- Tensor lists receive independent descriptor copies. Section 4.16 says that
  destroying a tensor list also destroys its tensor handles; sharing one handle
  across lists or subsequently destroying it separately is unsafe.
- A failed plan clears its executor output before calling the vendor planner
  and owns a partially returned executor during failure unwinding. Adoption owns
  the executor before repeatability setup or workspace bookkeeping can fail.
- Process-owned ACL/ACLNN initialization and finalization run once. External
  initialization is borrowed and remains the external caller's responsibility.
  Normal return and caught exceptions finalize ACLNN before ACL at process exit.

All executables have a top-level exception boundary. SIGINT/SIGTERM handlers
only set a cancellation flag; ordinary execution checks it at ACL/operator
boundaries and unwinds. SIGSEGV/SIGABRT and `std::terminate` write a minimal
diagnostic and use `_Exit`. **They do not call CANN, run destructors, or finalize
ACL from the signal handler.** Those operations are not async-signal-safe;
assertions and corrupted stacks cannot be made to unwind reliably. Fatal-signal
cleanup relies on process/driver reclamation. Driver lockups, uninterruptible
driver calls, force-destroy failures and SIGKILL cannot be guaranteed recoverable
by in-process RAII. A synchronization timeout bounds queued-work waiting, not
every possible kernel-driver call.

## Layout and documentation

`src/core`, `src/memory`, `src/pipeline`, and `src/diagnostics` match public
modules. Existing implementation names are retained; header-only abstractions
do not get empty `.cpp` files. `static_arena_manager.hpp` moves to `moe/memory`,
with a compatibility forwarding include at its previous path. CMake uses an
explicit source list. Generate the operator reference with `doxygen Doxyfile`;
output goes to ignored `build_docs/`, and documentation warnings are errors.

## Verification and remaining hardware gate

On the local x86_64 WSL host, CANN is installed but no Ascend device nodes are
exposed. The mock suite tests contracts, not NPU arithmetic. The dedicated
`mock_failure_teardown` test covers shape/dtype/alignment refusals, a non-square
bit-exact transpose with a partial final tile, partial construction, partial
executor plans, repeatability failure, stream timeout/force destruction,
pipeline launch failure, finalizer ordering, and signal exit paths.

```sh
cmake -S . -B build_mock -DENABLE_MOCK_RUNTIME=ON -DDSV4_WERROR=ON
cmake --build build_mock -j8
ctest --test-dir build_mock --output-on-failure

cmake -S . -B build_cann_guards -DENABLE_MOCK_RUNTIME=OFF -DDSV4_WERROR=ON
cmake --build build_cann_guards -j8
cmake --build build_cann_guards --target dsv4_verify_linkage
```

Driverless linkage/skip checks require the toolkit's x86_64 driver stubs on
`LD_LIBRARY_PATH`; use real driver libraries on the NPU server. With stubs,
all three NPU probes skip. This is **not** successful hardware execution.

On the actual device, use `--require-device` so absence or omitted coverage is
a failure. Choose coverage sufficient for the routing path being tested; a
small subset is only a bring-up configuration, and full coverage pins roughly
137 GiB of host memory. The coverage flag is deliberately not defaulted here.

```sh
./build_cann_guards/tests/npu_e2e_single_token --require-device --coverage <experts>
```

A real successful decode, supported quantized operator combinations, numerical
accuracy after the explicit casts, and driver recovery under actual NPU failure
remain hardware verification requirements.
