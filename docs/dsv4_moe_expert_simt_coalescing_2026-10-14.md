# `dsv4_moe_expert` SIMT reduction: coalescing and vectorization

2026-10-14. Target: Ascend 950PR, `dav-3510` / `arch35`, `KERNEL_TYPE_MIX_AIV_1_0`,
32 AIVs, 512 SIMT threads per launch (32 rows x 16 lanes).

Scope: the inner reduction of `src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_expert_kernel.cpp`.
Everything below the "Measured" heading is measured; everything above it is
derived from the source and the access pattern. **No claim here is a claim
about hardware latency** — this operator has never run on an NPU
(CLAUDE.md §5.7).

---

## 1. What changed

Three changes, all inside the SIMT reduction. The tiling ABI, the host
contract, the two entry points, the grid-stride tiling, the UB layout and the
activation are untouched.

### 1.1 Group-to-lane map

A "group" is one E8M0 micro-scale's worth of weights: 32 logical columns = 16
MXFP4 pairs = **16 packed bytes**.

| | map | lane byte stride |
| --- | --- | --- |
| before | `g = lane * groups_per_lane + step` (contiguous block per lane) | `cols/32` B (**128 B** at `hidden`=4096) |
| after | `g = step * 16 + lane` (groups interleaved across lanes) | **16 B** |

### 1.2 Load width

One group is exactly one `uint4` (`uint32_t __attribute__((ext_vector_type(4)))`,
16 B), so a group's weights are now **one** `asc_ldca` instead of 16
single-byte `asc_ldca` calls. The four 32-bit words are sliced in-register and
the eight nibbles per word decoded with shifts and masks.

`asc_ldca(__gm__ uint4*)` is a real overload in this toolkit —
`simt_api/device_functions.h:248`.

### 1.3 Fused W1/W3 sweep

`Dsv4ExpertGateUpTileSimt` called `LaneRowPartial` twice, once per leg. It now
calls `LaneRowPartialGateUp` once: one loop, two live accumulators, and each
`x` element read from UB **once** and multiplied into both legs
(`WordDotGateUp` holds `x_even` / `x_odd` in registers across both).

---

## 2. Transaction accounting

Per **row** (the unit a warp of 16 lanes cooperates on), at `hidden` = 4096:
2048 packed weight bytes = 16 cache lines of 128 B.

| quantity | before | after | change |
| --- | --- | --- | --- |
| weight-load instructions, per lane | 128 (1 B) | 8 (16 B) | **16x fewer** |
| weight-load instructions, per warp | 2048 | 128 | **16x fewer** |
| 128 B lines touched by **one** warp-wide load | **16** | **2** | **8x fewer** |
| line-requests per warp, per row | 32,768 | 256 | **128x fewer** |
| bytes requested | 2048 | 2048 | unchanged |
| distinct lines touched, per row | 16 | 16 | unchanged |

Read the last two rows first, because they are the honest limit of this
change: **total weight traffic is identical.** Both geometries read every
weight byte exactly once, and a row is 16 lines either way. Nothing here
reduces DRAM or L2 volume.

What changes is how many line transactions the memory pipeline must service
to *retire one warp instruction*. Before, the 16 lanes of a warp sat 128 B
apart — exactly one line apart — so a single warp-wide byte load addressed 16
distinct lines and consumed 1 of the 128 bytes each one delivers. After, the
16 lanes span one contiguous, 128 B-aligned 256 B run, so the same
instruction addresses 2 lines and consumes every byte of both.

The 128x figure (32,768 -> 256) multiplies the two effects and is the count of
tag-lookup / arbitration events, i.e. the queue pressure the old pattern put
on the SIMT DataCache. It is a **static count from the access pattern**, not a
cycle count: it assumes one tag lookup per lane per uncoalesced load, which is
what "uncoalesced" means on this cache, and it deliberately does not model
hit/miss (after the first touch most of those 32,768 requests hit).

The 2-line result depends on the 256 B run being 128 B-aligned. It is, for
every admissible width: `packed_row_stride = cols/2`, and the host gate
requires `cols % 512 == 0`, so the stride is a multiple of 256 and therefore
128 B-aligned. Checked for 512 / 1024 / 2048 / 4096 in `tests/emu`
("every packed row base is 16 B aligned for the uint4 group load").

### 2.1 Scale stream

Minor, and worth stating because it was listed as a problem and is not one.
Scale bytes at `hidden` = 4096 are 128 B per row — **one line** — under both
geometries, so neither sweep can do better than one line per access. The lane
stride did tighten (8 B -> 1 B, so a warp's scale footprint per instruction
goes 120 B -> 16 B), but the transaction count was already 1.

### 2.2 Fusion

Per row of the gate/up pass:

| quantity | before (2 sweeps) | after (1 sweep) | change |
| --- | --- | --- | --- |
| weight-load instructions, per warp | 4096 | 256 | 16x fewer |
| `x` UB reads, per warp | 8192 | 4096 | **2x fewer** |
| E8M0 scale loads | 256 | 256 | unchanged |
| row address generations | 2 | 1 | 2x fewer |

The `x` halving is the fusion's actual contribution and it is structural, not
an optimizer hope: `WordDotGateUp` binds `x_even` and `x_odd` to locals and
uses each twice.

### 2.3 Arithmetic

Unchanged — same number of `DecodeFp4E2M1` calls, same multiplies and adds.
This change moves bytes, not FLOPs. PipeSim independently called the sibling
`fast_fp8_indexer_topk` kernel `VECTOR_BOUND` with secondary `MTE2`
(`docs/pipesim_audit_2026-10-12.md`); if that holds for this operator too,
the ceiling on the above is the vector unit, and only measured hardware
latency can rank it (CLAUDE.md §5.6, §5.7).

---

## 3. What was deliberately preserved

The oracle's grouping is the constraint that shaped the whole design.

A group is still owned **end to end by one lane**: its 16 pairs are summed in
ascending column order into a register partial, and the E8M0 scale is applied
**once** to that subtotal before it joins the lane total
(`total += partial * scale`). So each group's subtotal is **bit-identical** to
the previous geometry's, and the `__simt_vf__` / `asc_shfl_xor` butterfly
(masks 1, 2, 4, 8 over 16 lanes) is unchanged, still run once per leg.

This is why the load quantum is 16 B and not 4 B. A fully stride-4
`uint32`-per-lane pattern would put 4 lanes on each group, which makes
"sum the group, *then* scale" impossible without two extra in-loop
`asc_shfl_xor` per group — 64 extra shuffles per row against the 4 the
butterfly costs now. The 16 B quantum is the widest load that keeps the
oracle's grouping and still makes lane stride equal load width.

### 3.1 The one thing that is not bit-identical

Only **which lane holds which group** changed, which permutes the leaves of
the butterfly. FP32 addition is not associative, so the row total differs from
the old geometry's in the last bits — bounded by the standard
summation-reordering bound, `2 * G * 2^-24 * sum|group subtotal|` for `G`
groups, which is the bound `tests/emu` already checks the decomposition
against.

Neither geometry was ever bit-exact against the CPU oracle: the oracle sums
groups sequentially and the kernel sums them in a 16-way tree. That was true
before this change and is unchanged by it. The claim is bit-identical *group
subtotals*, not a bit-identical row total.

---

## 4. Verification

Per CLAUDE.md §5.4, in order.

1. **Host index simulation** (§5.5) — `tests/emu`, the CI gate, no NPU, stock
   `g++`, no CANN. The new map and the `uint4` word unpacking are transcribed
   and compared against `mxfp4_gemv_cpu` / `moe_expert_swiglu_mxfp4_cpu`. Two
   new checks assert the address pattern directly rather than inferring it
   from close numbers:
   - every scale group is owned by exactly one lane, exactly once (coverage —
     a dropped group is a silent wrong answer);
   - a warp step's 16 lanes read 16 B apart (the load-quantum-equals-stride
     property the whole change rests on). This one discriminates: under the
     old map, lane 1 at step 0 sits at byte 32, not 16.
2. **BiSheng `dav-3510` compile gate** (§5.1) — `scripts/device_compile_kernels.sh`.
3. **NPU build gate** (§5.3) — `scripts/verify_npu_build.sh`.
4. **`dav-3510` CAModel execution** — `Dsv4MoeExpertSim`, real kernel through
   the real `Launch*` wrapper against the CPU golden.
5. **`tests/npu` on real hardware** — not run. No NPU on this workstation.

See "Measured" below for results.

---

## 4a. The simulation gate had a blind spot at 512

Worth recording, because it produced a convincing false negative.

`groups_per_lane = (cols/32)/16`. At `cols` = 512 that is **1**, and with one
group per lane the new map

```
g = step * 16 + lane      ->   g = 0 * 16 + lane = lane
```

collapses to the same function as the old contiguous-block map

```
g = lane * groups_per_lane + step   ->   g = lane * 1 + 0 = lane
```

So a 512-wide reduction **cannot distinguish the two geometries**. Run the old
and new kernels through `Dsv4MoeExpertSim` at 512x512 and the diffs agree bit
for bit — 6.1035156e-05 activated, 0.0013427734 output, both — because at that
width the group-to-lane assignment is literally unchanged. That agreement
confirms the `uint4` word unpacking and the fused sweep (both of which DID
change, and would wreck the numbers if the word order or a shift were wrong),
but it is silent on the remap that is the point of the change.

512 is also the reduction quantum, i.e. the smallest admissible width, so the
degenerate case was the default.

Both gates were therefore widened to a width where the maps differ:

- `tests/emu`: 1024 x **1024** — `groups_per_lane` = 2 on the gate/up legs
  *and* the down leg. Host-side, so the extra width is free.
- `Dsv4MoeExpertSim`: **1024** x 512 — `groups_per_lane` = 2 on the gate/up
  legs, which carry the fused W1/W3 sweep, the most heavily rewritten path.
  `inter` stays 512 to keep CAModel wall time near 2x rather than 4x.

## 5. Note on the simulation gate

`Dsv4MoeExpertSim` did not execute this kernel before this work. The suite
builds at 512x512, but `MakeDsv4MoeExpertTiling` pinned `hidden == 4096 &&
inter == 2048`, so the launch returned `unsupported_shape` and the test failed
at its first `Require`. `tests/sim/CMakeLists.txt` passed
`-DDSV4_TILING_TEST_OVERRIDE=1` to both the BiSheng and host TUs, but **no
source consumed that define** — the same dead-define shape CLAUDE.md §3.2
records for `GEN_ACLRT`.

The define is now real. `Dsv4ExpertGeometryAdmissible()` in
`op_host/dsv4_moe_expert_tiling.h` is the single admissibility predicate,
consumed by both the host gate and the kernel's `TilingRejected()` backstop;
under the override it admits any width meeting the reduction quantum, and
otherwise keeps the 4096/2048 production pin. `Dsv4MoeExpertProductionTiling`
covers the unoverridden path.

It has to be a macro plus two same-name wrappers, one per execution space:
BiSheng partitions functions into `__host__` and `__aicore__` and will not let
one definition serve both, so a single plain function cannot be called by the
kernel and a single `__aicore__` function cannot be called by the host tiling
builder in the same ASC TU. The UB ceiling is **not** in the predicate — the
byte-count helpers are host-space functions a device wrapper cannot link
against, and the ceiling is already enforced in `LaunchDsv4MoeExpert`, the
only path that allocates.

---

## 6. Measured

### BiSheng `dav-3510` compile gate (§5.1)

`scripts/device_compile_kernels.sh`: **pass=24 fail=0**, `warnings=0` on every
TU, `dsv4_moe_expert_kernel.cpp` -> 50,128 B. Note this gate compiles the
**production** branch (no `DSV4_TILING_TEST_OVERRIDE`), so the 4096/2048 arm of
the shared predicate is compiled for device too, not just the override arm.

### NPU build gate (§5.3)

`scripts/verify_npu_build.sh`: **OK**, 0 warning lines. The §4.1 linkage
contract holds both ways — device suite links `libascendcl`: yes; CPU oracle
free of `ascendcl`/`cpudebug`: yes (also §5.2).

### `tests/emu` (CI gate, stock g++, no CANN, §4.1)

`checks: 42   failures: 0   RESULT: PASS`, zero skips (§4.2), at 1024 x 1024
(`groups_per_lane` = 2 on both legs).

### `Dsv4MoeExpertSim` (`dav-3510` CAModel, seed 0xD5A40001)

Tolerance: abs 1e-2, rel 1e-2.

At **512 x 512** — the degenerate width, where the two maps are the same
function (§4a), so this row pair measures the `uint4` unpacking and the fused
sweep and nothing about the remap:

| | activated max abs diff | output max abs diff |
| --- | --- | --- |
| before | 6.1035156e-05 | 0.0013427734 |
| after | 6.1035156e-05 | 0.0013427734 |

Bit-identical, as predicted: at `groups_per_lane` = 1 the group-to-lane
assignment is unchanged, and group subtotals are bit-identical by design
(§3), so there is nothing left to differ.

At **1024 x 512** — `groups_per_lane` = 2 on the gate/up legs, where the maps
genuinely differ. This is the configuration that actually exercises the
interleaving on device:

| | activated max abs diff | output max abs diff | wall |
| --- | --- | --- | --- |
| after (coalesced + fused) | 6.1035156e-05 | 0.0014648438 | 540.57 s |

`hidden=1024 inter=512 block_dim=32, gate/up tiles=16, down tiles=32`.
**PASS**, alongside `Dsv4MoeExpertProductionTiling` (2/2).

The output diff moves 0.0013427734 -> 0.0014648438 going from 512 to 1024
because the down leg now reduces twice as many signed terms, not because the
geometry changed; both are ~7x inside the 1e-2 bound.

The CAModel wall times (512x512: 276.81 s before, 259.96 s after; 370.31 s on
a contended re-run of the same binary) are **not** a performance measurement.
It is a functional simulator, the spread across identical runs is larger than
the difference between kernels, and CLAUDE.md §5.6 permits only measured
hardware latency to rank an optimization.

### Not run

`tests/npu` on real hardware — no NPU on this workstation. Per §5.7: this
operator **compiles, links and executes under the dav-3510 CAModel; it has
never executed on hardware**, and nothing here claims a hardware latency.
