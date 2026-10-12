# Project rules: operators and tests

Rules for `ascend-moe-engine`. These are binding conventions, not suggestions.
Where a rule says *why*, the why is the rule — a change that keeps the letter
and breaks the reason is a violation.

Target hardware: Ascend 950PR, DaVinci v3 (`arch35` / `dav-3510`), CANN
9.2.0-beta.2, BiSheng. Host workstation has **no NPU**; see
[Verification gates](#verification-gates) for what that does and does not allow.

---

## 1. Operator layout

Every operator is one self-contained directory under `src/ops/<op_name>/`.
`src/ops/mhc_fused_sinkhorn_gemv` established the pattern; it is now
**deprecated** (§6) but remains the format reference.

```text
src/ops/<op_name>/
├── CMakeLists.txt                       # per-op targets, prefixed with <op_name>
├── cpu_ref/<op_name>_ref.{hpp,cpp}      # golden adapter onto moe::emu
├── op_host/<op_name>.cpp                # the Launch<Op>() wrapper
├── op_host/<op_name>_tiling.{h,cpp}     # tiling POD + validating builder
└── op_kernel/<op_name>_arch35.h         # device UB layout, op-private
    op_kernel/<op_name>_kernel.cpp       # the __global__ __aicore__ entry
```

Rules:

1. **No operator source outside its own directory.** A flat
   `src/ops/<op_name>.cpp` is not acceptable. If a file is shared by two
   operators it belongs in `include/moe/kernels/arch35/` or
   `src/kernels/arch35/`, not in one operator's directory.
2. **`op_kernel/` is private.** UB layouts, thread geometry and register
   budgets are implementation detail and live in `op_kernel/<op>_arch35.h`.
   Do not promote them to `include/` — nothing outside the operator may depend
   on them. A header under `include/` is a public contract.
3. **Includes are by bare filename** (`#include "<op>_tiling.h"`), resolved by
   the per-op include directories. Never reach across operators with a
   relative path like `../../other_op/...`.
4. **Target names are prefixed with the operator name.** A target called
   `cpu_ref` is a global-namespace collision as soon as a second operator
   follows the pattern; use `<op_name>_cpu_ref`.
5. One operator per directory, named exactly as the kernel entry symbol.

## 2. Tiling ABI

1. Every tiling struct is a **standard-layout, trivially-copyable POD of 32-bit
   fields only** — `std::uint32_t` and `float`. No pointers, no 64-bit fields,
   no bitfields, no nested types. Host and device TUs must see an identical
   layout with no pointer-width padding.
2. Assert it at the definition site with `MOE_ARCH35_ASSERT_TILING_POD(T, bytes)`
   from `include/moe/kernels/arch35/tiling_abi.hpp`. The byte size is the wire
   format; changing it is an ABI break.
3. First field is `abi_version`, set from `kTilingAbiVersion`. Kernels **must**
   reject an unknown version. `reserved` fields must be written zero.
4. `op_host/<op>_tiling.h` is included by **both** sides, so it must not
   include `<acl/acl.h>`. `Status` and the storage-bit typedefs come from
   `include/moe/kernels/arch35/status.hpp`, which is ACL-free for this reason.
5. **All shape validation lives in `Make<Op>Tiling()`**, host-side, before the
   enqueue. The kernel rejects bad tiling by returning early, which on device
   is a silent no-op producing a zeroed output — so a wrong shape that reaches
   the device is an undetectable wrong answer, not an error.
6. Distinguish the two rejections honestly: `invalid_argument` is a caller
   mistake (null pointer, aliasing violation); `unsupported_shape` is a
   geometry v1 does not implement. Do not collapse them.
7. Widen before multiplying any caller-supplied extent. `block_size * 128`
   overflows `uint32` above 2^25.

## 3. Launch layer

1. **Dispatch is the triple-chevron form only**:
   `kernel<<<block_dim, nullptr, stream>>>(args...)`, passing raw `uint8_t*`
   device pointers. BiSheng lowers it to
   `aclrtLaunchKernelWithHostArgs` / `aclrtBinaryLoadFromData` /
   `aclrtGetFuncBySymbol`.
2. **Never use `ACLRT_LAUNCH_KERNEL` / `aclrtlaunch_<kernel>` stubs, and never
   rely on `GEN_ACLRT`.** Verified against this toolkit: the ASC flow passes
   `-DGEN_ACLRT=<dir>` but nothing consumes it (the define appears only in
   `FindASC.cmake`, `ACLRT_LAUNCH_KERNEL` only in `extract_host_stub.py`), so
   no stub is emitted and the link fails with undefined symbols.
3. Consequence: `op_host/<op>.cpp` **must be compiled by BiSheng**, not the
   host compiler. Public headers stay plain C++ over `<acl/acl.h>` so host TUs
   built with g++ can call the wrappers and link against the BiSheng objects.
4. The caller owns every operand GM buffer. A wrapper owns only geometry
   validation, the tiling staging buffer, the dispatch, and its teardown. It
   does **not** allocate or copy operand data.
5. A chevron launch returns `void`, so no error is observable at the call. Every
   ACL failure must be recorded through `detail::RecordAclFailure()` and be
   readable via `LastLaunchFailure()`. **A bare `return Status::launch_error`
   with no record is a defect**: it makes a tiling malloc, a bad grid and a
   device-side fault indistinguishable, which is precisely the state that
   leaves an on-device failure undebuggable.
6. Zero a tiling staging allocation before the H2D copy. The allocation is
   rounded up to 32 B while the PODs are 32/40/48/64 B, so for most operators
   the trailing bytes would otherwise reach the device uninitialized.
7. The v1 wrappers synchronize the stream before releasing the staging buffer
   and are therefore **not safe to call concurrently**. Do not add a pipelined
   variant without giving the caller ownership of the tiling allocation.

## 4. Test suites

Four directories, four distinct jobs. A binary belongs to exactly one.

| Directory | Job | Links | Needs NPU |
| --- | --- | --- | --- |
| `tests/emu/` | **Golden/oracle.** Algebraic invariants of the CPU reference in `src/emu`; the source of truth later inference tests compare against. | `moe_emu` only | No |
| `tests/npu/` | **Device verification.** Real hardware through the `Launch*` wrappers, compared on the fly against the `tests/emu` golden. | `ascend_moe_engine` / `arch35_device_launch` + `libascendcl` | Yes |
| `tests/sim/` | Verification of host-side operator plumbing (tiling plans, reference adapters). | per-op `cpu_ref` | No |
| `tests/bench/` | **Timing only, never a gate.** Latency, GB/s, arithmetic intensity. | `moe_emu`, optionally the launch layer | Optional |

Rules:

1. **`tests/emu` must not link `libascendcl.so`, must not include any device
   header, and must build with a stock host `g++`/`clang++` with no CANN
   toolkit present.** It must pass on any machine. It is the CI gate.
2. **`tests/emu` must not skip.** It has no device gate and no conditional
   sections, so a skip means a case silently opted out — treat it as a failure.
3. **`tests/npu` must not fail for want of hardware.** Probe
   `aclrtGetDeviceCount` at entry; on zero devices or a failed init, print the
   skip banner and `exit(0)`. The banner must contain the phrase
   `Physical Ascend NPU not detected`, which is the
   `SKIP_REGULAR_EXPRESSION` the suite's ctest registration matches.
4. **Past the entry guard, there is a device, so every ACL error and every
   non-success `Status` is a hard FAIL** carrying its code, the recorded
   `LastLaunchFailure()` and `aclGetRecentErrMsg()`. A launch that fails on a
   machine that has an NPU is a real defect, never a skip.
   Keep "no device" and "device present but bring-up failed" apart: collapsing
   them makes the suite either cry wolf on every laptop or pass silently on a
   broken machine.
   Note the guard can only run where the **driver** is installed.
   `libascendcl` pulls in `libascend_hal.so`, which ships with the driver and
   not the toolkit, and the build links the driver stubs with `-rpath-link`
   (link-time only, deliberately not `-rpath`) so a stub can never be loaded
   at run time in place of a real driver. On a host with neither driver nor
   device the binary therefore fails to load before `main()` and exits 127.
   That is the correct outcome and must not be "fixed" by putting the stub
   directory on the runtime search path: that would let a fake driver satisfy
   the device probe.
5. **References are generated in memory, per run, from a printed seed.** No
   pre-baked disk fixtures: a stale asset must not be able to make a broken
   kernel pass. Print the seed so a failure is reproducible.
6. A benchmark is never a correctness gate. Timing thresholds on shared
   hardware are flake generators. `tests/bench` may be smoke-tested for a
   clean exit with iteration counts turned down, nothing more.
7. Tolerances must be justified by the format, not fitted to the observed
   error. State in a comment what sets the bound (e.g. "BF16 keeps an 8-bit
   mantissa, so one store carries up to 2^-9 relative error, and this identity
   combines four independently rounded outputs"). Keep cosine strict even where
   relative-L2 must be loose: rounding moves magnitude, structural bugs move
   direction.
8. A degeneracy guard must test the **reference**, not compare a value with
   itself. `CosineSimilarity(want, want) == 1.0` is not a non-degeneracy check:
   it is an exact float comparison of `dot/(sqrt(n)*sqrt(n))` against 1.0,
   which is false for ordinary inputs. Assert a nonzero norm and real variance
   in the golden instead.

## 5. Verification gates

What counts as verified, on a workstation with no NPU:

1. **The device compile gate is `scripts/device_compile_kernels.sh`**, which
   runs `bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang -c` over every
   operator inside the `ascend-c-dev:9.2.0-beta.2` container. Docker lives
   inside WSL, so drive it as:
   ```bash
   wsl -d Ubuntu-22.04 -- bash /mnt/c/ascend-moe-engine/scripts/device_compile_kernels.sh
   ```
   The image is a thin shell with no CANN in it; the toolkit must be
   bind-mounted from the Docker host (`-v /usr/local/Ascend:/usr/local/Ascend:ro`)
   or `bisheng` is not on `PATH`. This script must be updated whenever an
   operator is added or its paths change.
2. **Do not use host CPU debug / interpretation runtimes.** No `cpudebug`, no
   `tikicpulib`, no `libascendc_cpudebug.so`, for any SIMT kernel. Interpreting
   device SIMT code on x86 does not model `__simt_vf__` scheduling, warp
   primitives or DataCache behaviour, so a pass there means nothing. The claim
   is enforced against the linked artifacts by
   `tests/common/verify_no_cpudebug.cmake`, not by comment.
3. **The NPU build gate is `scripts/verify_npu_build.sh`**, which configures
   and builds `arch35_device_launch` and `test_custom_kernels_npu_device` in
   the same container, and asserts the linkage contract from section 4.1 both
   ways: the device suite links `libascendcl`, the CPU oracle does not. This
   answers "does the device side build" on a host with no NPU; it cannot
   answer "does it compute the right answer".
4. **Verify with: the BiSheng dav-3510 compile gate + the NPU build gate +
   the `tests/emu` oracle + `tests/npu` on real hardware.** In that order.
5. **Device logic that is pure integer/index arithmetic must be validated on
   the host, not left to the NPU.** A SIMT network's indexing is exactly the
   thing that is wrong the first time, and it needs no hardware to check: the
   tiered bitonic selection network in `fast_fp8_indexer_topk` is transcribed
   faithfully into `tests/emu` and compared against `std::sort`. Any
   equivalent rewrite must come with the same kind of host simulation --
   "it compiles for dav-3510" is not evidence that a sort sorts.
6. **Static analysis and simulation tools are out of scope for the current
   phase** — `ascend-kernel-analyzer`, `PipeSim`. Their AST parsers and rules
   predate the DaVinci v3 hybrid SIMD/SIMT primitives (`__simt_vf__`,
   `asc_vf_call`, `asc_shfl_xor`, L1 DataCache bypass via `asc_ldca` /
   `asc_ldcg`), so they produce false positives and pressure the kernels toward
   scalar fallbacks. Do not run them or treat their output as a gate until the
   rules are updated.
7. Never claim hardware behaviour that has not been observed on hardware. If a
   path has only been compiled, say "compiles and links; never executed".

## 6. Deprecated: `mhc_fused_sinkhorn_gemv`

`src/ops/mhc_fused_sinkhorn_gemv` is **deprecated**. It predates the arch35
contracts: its kernel is not one of the five `dav-3510` operators, it is not
covered by the device compile gate, and its CMake target name (`cpu_ref`)
violates §1.4.

- Keep it building; `tests/common` and `tests/sim` still depend on it.
- It remains the **format reference** for §1.
- Do not extend it, add features to it, or copy its target naming.
- New mHC/Sinkhorn work goes to `fused_mhc_pre_norm_sinkhorn` and
  `fused_mhc_post_residual`.

## 7. Skills

Ascend kernel skills are maintained in `C:\ascend-kernel-skills` and
**installed by copy** into `~/.claude/skills/`. Re-copy after any change
upstream; they are not symlinked.

For the 950PR / arch35 operators prefer the `ascend-950pr-*` packages over the
older generic ones — they are audited against CANN 9.1.0 source. Note the
conflict: `ascend-351x-simt-simd-hybrid` spells the SIMT launch `asc_call_vf`,
while the audited `ascend-950pr-aiv-simt` records it as **`asc_vf_call`**.
The audited spelling wins.
