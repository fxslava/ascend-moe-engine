# arch35 Stage-1 kernels — CANN 9.2 toolchain findings

Environment notes from implementing KER-01..KER-05 against CANN 9.2.0-beta.2
in container `cann92_dev` (image `b82c08f530f9`), target `dav-3510`. These are
CANN/environment behaviours, not gaps in PipeSim or ascend_analyzer; those are
recorded in each tool's own `ISSUES_AND_GAPS.md`.

## 1. `asc_vf_call` IS emitted under the bare compile line (an earlier note here was wrong)

`--npu-arch=dav-3510` predefines **`__DAV_VEC__`**, so the SIMT dispatch is
live with nothing but `bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang -c`:

```
$ bisheng --npu-arch=dav-3510 -dM -E - </dev/null | grep DAV
#define __DAV_C310_CUBE__ 1
#define __DAV_C310_VEC__ 1
#define __DAV_C310__ 1
#define __DAV_CUBE__ 1
#define __DAV_VEC__ 1
```

and preprocessing a kernel confirms the CANN-side gate opens too:

```
$ bisheng ... -dM -E ker05_mxfp4_expert_gemv.cpp | grep -E 'SPLIT_CORE_VEC|__DAV_VEC__'
#define SPLIT_CORE_VEC
#define __DAV_VEC__ 1
```

**Why this was briefly misdiagnosed, and the lesson.** Two `__DAV_VEC__` gates
have to open for a VF to exist: CANN's
`simt_api/common_functions.h` wraps `asc_vf_call` in
`#if ... defined(SPLIT_CORE_VEC) ...`, and the compiler's own
`__clang_cce_simt.h` puts the entire body of `cce::async_invoke` --
`__builtin_cce_store_vfsimt_info` and the `Func(Args...)` call -- inside
`#ifdef __DAV_VEC__`. Adding `-DSPLIT_CORE_VEC` produced a byte-identical
object, which was read as "the body is not emitted". The real reason is the
opposite: the header had **already** defined `SPLIT_CORE_VEC` off
`__DAV_VEC__`, so the command-line define was redundant. An unchanged object
proves a flag had no effect; it says nothing about whether the code was there
to begin with.

The test that actually answers the question is differential on the VF body --
perturb one constant *inside* the `__simt_vf__` and see whether the object
moves:

| kernel | original | VF constant perturbed | verdict |
|---|---|---|---|
| `ker01_mhc_pre_norm_sinkhorn` | `abe48ee7da7e3737` | `0a51bc84c9b0f3a1` | emitted |
| `ker05_mxfp4_expert_gemv` | `4648ee58fe40462b` | `5ddf70ffa57f1112` | emitted |

No split-core build wiring, `--cce-soc-core-type` or `--cce-soc-version` is
needed for the VFs to compile.

For the record, the soc-version route is not even available in this image:
`--cce-soc-version` accepts only the SKUs baked into this build, and every
950 string -- including the `Ascend950PR_9589` that appears in the binary's
own string table -- is rejected with *"Ascend soc_version: ... is not
supported"*. `--cce-soc-core-type` takes `CubeCore` or `VecCore` but requires
a soc_version, so it cannot be used standalone. The `dav-c310-vec`,
`dav-c310-simt` and `dav-c310-simd` arch names in the binary are likewise
rejected by `--npu-arch`; `dav-3510` is the one that works, and it already
defines both `__DAV_VEC__` and `__DAV_CUBE__`.

`--enable-simt` remains for *pure*-SIMT translation units and is incompatible
with `kernel_operator.h` (`#error`), so a hybrid SIMD+SIMT kernel must not use
it. `--run-mode` accepts `cpu` and `sim`, not `npu`.

## 2. `ReduceSum`'s count form requires a `TPipe` to exist

On dav-3510 `ReduceSum(dst, src, sharedTmp, count)` calls
`GetTPipePtr()->AllocEventID<HardEvent::V_S>()` internally. In a static-tensor
kernel with no `TPipe`, `GetTPipePtr()` is null and the reduction faults inside
the library. bisheng does not diagnose it. Every kernel using the count form
declares `TPipe pipe; (void)pipe;` at entry for this reason alone; no TQue is
ever allocated from it.

## 3. `ReducePairElem` validates its destination far beyond the bytes it writes

The CPU-debug checker demands roughly 512 B of destination per repeat:

| repeatTime | dstRepStride | bytes written | bytes demanded |
|---:|---:|---:|---:|
| 2 | 4 blocks | 256 | 640 |
| 4 | 4 blocks | 512 | 1664 |

Message: *"Failed to check dstLocal size in PairReduceSum when in normal mode,
tensor size needs to be at least 1664 bytes, while current tensor size is only
1024 bytes."* Destinations are padded accordingly in the layout generator.
`CheckFunVecReduceOther` is inside the closed `libcpudebug`, so the exact
formula was inferred from these two points rather than read.

## 4. BF16 must be `bfloat16_t`, never `uint16_t`, on a device tensor

The ABI typedef `Bf16 = std::uint16_t` is correct for describing GM storage,
but using it as a `LocalTensor` element type makes `Cast` perform an **integer**
conversion: `Cast<float, uint16_t>` turns the bit pattern 0x3F80 into 16256.0f
instead of 1.0f. bisheng compiles it; `Tuple<float, bfloat16_t>` is the pair in
the `cast_none` support set. Device tensors therefore use `bfloat16_t` and the
`uint16_t` typedef stays on the host side of the ABI.

## 5. CANN 9.2 intrinsic availability on dav-3510

* **Gone:** `WholeReduceSum`, `BlockReduceSum`. Use `ReduceSum`,
  `ReduceDataBlock`, `ReduceRepeat`, `ReducePairElem` with
  `ReduceType::SUM|MAX|MIN`.
* **No scalar libm in device scalar code:** `expf`, `exp`, `sqrtf`, `fmaxf`,
  `fminf` all fail to resolve. `sqrt(float)`, `/`, comparisons and
  `ScalarCast<...>` do work. Exponentials must go through vector `Exp` or a
  SIMT VF (`simt_api/math_functions.h` has `expf`, `exp2f`, `ldexpf`).
* **No scalar-operand `And`:** only the tensor-tensor form exists, so
  bit-masking a vector against a constant needs a `Duplicate`d tensor, a shift
  pair, or exact float arithmetic.
* **`CreateVecIndex` rejects `uint32_t`** (int8/int16/int32/half/float/int64
  only). Build as `int32_t` and `ReinterpretCast<uint32_t>()` for `Gather`.
* **`Gather` offsets are BYTES**, not elements — the impl divides by the
  element size (`"convert addr offset into B32 element index: divide by 4"`).
* **`float <- fp8_e4m3fn_t` and `bfloat16_t <- fp4x2_e2m1_t` are native
  `Cast`s.** Either removes the need for a decode table, though the fp4x2
  nibble order would have to be verified against `kMxfp4LowNibbleFirst` before
  being trusted.
* **SIMT verified working:** `__simt_vf__`, `__simt_callee__`, `threadIdx.x`,
  `blockDim.x`, `asc_shfl_xor(var, lane_mask, width)` (note: *not*
  `asc_warp_shfl_xor`), `expf`, `ldexpf`, direct `__gm__` loads and
  `__ubuf__` stores.

## 6. CPU interpreter: spurious SIGSEGV in `Cast<float, bfloat16_t>`

`tools/cpudebug` for `Ascend950PR_9599` raises a SIGSEGV backtrace inside
`plt_b32`/`write_sreg` on one core for every `Cast(float <- bfloat16_t)`, at
every count from 64 to 4096, while computing the correct values and exiting 0.
It is an interpreter artifact, not a kernel defect.
