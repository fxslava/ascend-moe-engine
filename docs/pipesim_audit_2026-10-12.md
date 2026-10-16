# PipeSim ↔ 950PR co-validation audit

Date: 2026-10-12. Target: `dav-3510` / CANN 9.2.0-beta.2.
Simulator: `ascend-pipesim` 0.2.0 at `C:\PipeSim`, installed into a throwaway
venv (`pip install -e`), profile `ascend950pr`.

Scope as agreed: PipeSim was run for **model characterisation only**. The
optimization roadmap in §4 is ranked by measured hardware latency, never by
simulator cycles — see CLAUDE.md §5.6.

Hardware figures referenced throughout are the user's 950PR run:
`mxfp4_expert_gemv` 208 µs, `fused_mhc_pre_norm_sinkhorn` 95 µs,
`fused_mhc_post_residual` 1.35 GB/s, `fast_fp8_indexer_topk` 10.6 ms at
`seq_len=2048`, `paged_compressor_rope` failed `aclError=507035` at step 0.

---

## 0. Corrections to the audit brief

Both were established before any conclusion was drawn, per the triangulation
rule.

| Brief says | Reality | Source |
| --- | --- | --- |
| PipeSim at `D:\Projects\PipeSim` | `C:\PipeSim`; no `D:` drive exists | `Get-PSDrive` |
| "160 KB usable UB ceiling per AIV" | **221,184 B (216 KiB)** | `ascend-950pr-aiv-simt` SKILL.md (`Static + Dynamic <= 216 KiB`); `kUsableUbBytes`; PipeSim `pool("UB").capacity_bytes = 221184` |
| `asc_shfl_xor` latency "1 cycle" | No published instruction latency | PipeSim: *"not measured instruction latencies"*; absent from whitepaper and audited skills |

The 160 KB figure matters: auditing PipeSim against it would have reported a
**correct** simulator as broken. The 1-cycle shuffle latency has no provenance
I could find in any of the five mandated sources, so nothing here is validated
against it.

---

## 1. PipeSim fidelity assessment

### Confirmed accurate

- **UB ceiling.** `pool("UB").capacity_bytes = 221184`, described as *"216 KiB
  tensor budget after 8 KiB compiler reserve and 32 KiB SIMT DataCache"*. This
  does real work: `ISSUES_AND_GAPS.md` records a `UB_OVERFLOW` that killed a
  KER-01 whole-row tiling at modeling time (`needs 262208 B ... only 221184 B`).
- **UB bank geometry.** `banks=8`, `bank_stride_bytes=32`, `alignment_bytes=32`,
  matching the skills' 8-bank 32 B interleave rule.
- **`asc_vf_call` accounting.** `vf_mode_switch_penalty_cycles = 64.0`
  (validated 30..80 window), counted by `vf_dispatches_per_phase` and priced
  inside the VECTOR stage. Observed as `simt.vf_mode_switch_cycles = 4096` for
  64 phases. Added 2026-10-11 — **CLAUDE.md §5.6's "predates the v3
  primitives" claim was stale for this primitive.**
- **`asc_shfl_xor`.** Modelled via `shuffle_ops_per_thread` as shared-ALU issue
  cycles with zero SRAM traffic (`simt.warp_shuffle_cycles = 1280`). That is
  the right *shape* for a register butterfly.
- **Warp geometry / register file.** `warp_size=32`, `warp_schedulers=4`,
  `register_file_bytes=131072`, `max_threads_per_core=2048`, with spill
  accounting (`register_limit_per_thread=64`, `spill_bytes=0` at 40 regs/thread).
- **Multi-core HBM contention.** Real and correctly shaped
  (`evaluator.py:2111`): `mte2_bpc = min(mte2_bpc, hbm_bpc / cores)`.

### False positives and modelling artifacts

- **`vector_gemv` forces an MTE2-staged weight tile even with
  `direct_gm_loads=True`.** Modelling the indexer's scoring with the real
  8192-element reduction produced
  `UB_OVERFLOW: needs 540928 B ... only 221184 B` — but the kernel stages no
  key tile at all; it streams through the SIMT DataCache. This is an artifact
  of the GEMV abstraction, not a kernel defect. Any `UB_OVERFLOW` on a
  `direct_gm_loads` flow should be re-derived by hand before it is believed.
- **`limiting_stage` and `hierarchical_roofline.binding_level` disagree**
  (`VECTOR_BOUND` vs `GM_MTE2`) on the same report. Two different bottleneck
  verdicts in one object; neither is labelled as subordinate.
- **Report accessors.** `report.to_dict()["total_cycles"]` does not exist;
  cycles live under `pipeline.total_cycles`. Reading the flat key silently
  yields `0`, i.e. a plausible-looking zero rather than an error. Already
  logged as `ISSUES_AND_GAPS.md` item 3 and still live.

### Missing 950PR features — the important gaps

1. **No static/dynamic UB region split.** The `UB` pool is a single capacity
   with a `datacache_min_bytes` floor. There is no representation of
   `TPipe`/`TBuf` (static) versus `LocalMemAllocator` (dynamic), and therefore
   none of `dyn_ub_size` as a launch-declared quantity.

   **Consequence: PipeSim is structurally blind to the error-341 / 507035
   class.** The `paged_compressor_rope` fault fixed in `0a36f51` was fully
   legal on aggregate footprint — 72,448 B against a 221,184 B ceiling — and
   was purely about *which region* ten SIMT-dereferenced buffers lived in. A
   capacity model cannot express that. Do not expect PipeSim to catch a
   repeat.

2. **DataCache capacity is pinned at the 32 KiB floor.** `datacache_bytes =
   32768`, a constant. The real value is
   `min(256 KiB − static − dynamic − 8 KiB, 128 KiB)`, so for the indexer's
   49,664 B footprint the hardware has **128 KiB — 4× what is modelled**. This
   is the dominant source of the error in §2.
3. **No `asc_syncthreads` barrier model.** Only a generic
   `synchronization.sync_handoff_cycles = 30` at architecture level. Cross-warp
   barrier stalls are not separately priced, which matters for the indexer's
   selection phase (20 barriers).
4. **Channel rates are assumptions, not measurements.** `128 B/cycle` for
   MTE1/2/3/FIXPIPE/NDDMA/CV is flagged in PipeSim's own profile notes as *"an
   analytical service-rate assumption"* with an explicit instruction to
   calibrate before trusting absolute predictions. The 1.65 GHz frequency is
   inferred from an aggregate TFLOP/s figure.

### Fidelity verdict

**Qualitative: trustworthy. Absolute: not usable.**

For `fast_fp8_indexer_topk` scoring, PipeSim predicted **269,163,581 cycles
≈ 163 ms** against **10.6 ms measured** — a **~15× overestimate**. The error is
traceable, not random: `simt.cache_latency_cycles = 168,034,744` of the total,
from `cache_miss_rate = 0.875` and `cache_fill_bytes = 3.76 GB` against a
working set of only 262 KB of distinct key bytes. The model thrashes a 32 KiB
cache that is really 128 KiB, and cannot represent the warp-local temporal
reuse of a 128 B key row across 64 heads.

Yet it independently reached `limiting_stage = VECTOR_BOUND`, secondary
`MTE2`, `vector_alu_utilization = 37.4%` — which **agrees with the
source-derived cost model**. That agreement on *which stage binds*, with a 15×
disagreement on *how long*, is the single most useful result of this audit and
is exactly why the roadmap below is ranked by hardware.

---

## 2. Kernel diagnostic matrix

Cycle counts are deliberately omitted where PipeSim is the only source: per
§1 they would be ~15× high. Bottleneck column is the qualitative verdict,
cross-checked against source analysis.

| Kernel | Measured | Arithmetic intensity | Bottleneck | Basis |
| --- | --- | --- | --- | --- |
| `mxfp4_expert_gemv` | 208 µs, ~21.5 GB/s | 2048×4096: 16.8 MFLOP / 4.48 MB ≈ **3.7 FLOP/B** | MTE2 / weight-stream bound | decode GEMV, weights read once; PipeSim `MTE2_STARVATION` for all decode GEMVs (its own item 4) |
| `fused_mhc_pre_norm_sinkhorn` | 95 µs | 0.8 MFLOP / 1.6 MB ≈ **0.5 FLOP/B** | MTE2, φ-stream bound | φ is [24,16384] FP32 = 1.5 MB of a 1.6 MB total |
| `fused_mhc_post_residual` | 1.35 GB/s | 0.66 MFLOP / 73.7 kB ≈ **9 FLOP/B** | **Host enqueue bound** — not a kernel property | see §3 |
| `fast_fp8_indexer_topk` | 10.6 ms | 33.5 MFLOP / 262 kB ≈ **128 FLOP/B** | **Vector ALU / issue bound** | PipeSim `VECTOR_BOUND` + source cost model |
| `paged_compressor_rope` | faulted | 16.8 MFLOP / 16.8 MB ≈ **1 FLOP/B** | MTE2 (projection weights) once it runs | two BF16 [1024,4096] matrices = 16.8 MB/step |

Note the indexer is the **only** compute-bound kernel in the set, at ~128
FLOP/B. Every other operator is a decode-shaped weight streamer under 4
FLOP/B. That single fact determines the roadmap.

---

## 3. Findings that did not need the simulator

- **`fused_mhc_post_residual` at 1.35 GB/s is host enqueue latency, not a GM
  stall.** 73,728 B per launch at 1.35 GB/s ⇒ **~54.6 µs per call**. The same
  traffic at even the modelled 211 GB/s would be ~0.35 µs, and `mxfp4` achieves
  21.5 GB/s on a 60× larger payload. So ~54 µs is per-launch cost — tiling
  `aclrtMalloc`, H2D, dispatch, the wrapper's own stream sync (CLAUDE.md §3.7),
  free. **No simulator can see this: the cost is in the ACL wrapper, outside
  the modelled pipeline.** Confirmation is already in the bench output — if
  GB/s is flat across N=1/8/64, it is overhead-bound, which is what the bench
  comment predicts.
- **Sinkhorn UB bound is proven at compile time, not by simulation.** The solve
  is register-resident across 16 SIMT threads; iteration count is
  `kHcSinkhornIters` and is not a UB axis. The layout header `static_assert`s
  against `kUsableUbBytes` and the DataCache floor, so a violation fails the
  build. Nothing to simulate.
- **The `gate_cur` ↔ `prev_a` hazard was real, not zero.** `stateGateGm_` was
  based at float 8,192 — exactly where `prev_kv_a` begins — so `gate_cur`
  covered the union of both prev-a regions while `[4096, 8192)` went untouched.
  In bounds, so it could never fault. Fixed in `0a36f51`. A pipeline simulator
  would not have found it: it is a GM addressing error, not a timing or
  capacity one.

---

## 4. Optimization roadmap — ranked by measured hardware

### 4.1 `fast_fp8_indexer_topk`: two-kernel multi-AIV split (10.6 ms → ~0.33 ms)

The dominant cost by two orders of magnitude. Scoring is
`seq_len × 64 × 128 = 16.7 M` decode+FMA pairs, `VECTOR_BOUND`, and perfectly
parallel over keys — it already writes to a GM workspace indexed by `j`, so
cores need no communication.

```
fast_fp8_indexer_score   block_dim = 32, grid-stride over keys by block_idx
fast_fp8_indexer_select  block_dim = 1, selection network byte-identical
```

Stream ordering between the two chevrons replaces any cross-core barrier, and
the verified selection network is untouched.

**HBM headroom, from PipeSim's contention model.** The per-core MTE2 rate is
clamped from 8 cores up, and at 32 cores to `969.7 / 32 = 30.3 B/cycle`
(≈50 GB/s), a 4.2× reduction from the 128 B/cycle nominal:

| cores | HBM share (B/cyc) | effective | regime |
| ---: | ---: | ---: | --- |
| 1 | 969.7 | 128.0 | MTE2-limited |
| 4 | 242.4 | 128.0 | MTE2-limited |
| 8 | 121.2 | 121.2 | clamped |
| 32 | 30.3 | 30.3 | clamped |

**It does not bind here.** Distinct key traffic is 262 kB (16.8 MB if the
DataCache fails to hold a row across the head loop). Spread over 32 cores
against a 0.33 ms target that is ~1.6 GB/s per core — **~30× under** the
clamped 50 GB/s share. So the answer to the brief's question is yes: MTE2 can
feed 32 AIVs for this kernel, because it is ALU-bound and its operand traffic
is orders of magnitude below the per-core HBM share.

### 4.2 `fast_fp8_indexer_topk`: cut redundant work in the inner loop (bit-exact)

Stacks on 4.1, no ABI change. `key_row[d]` is a GM **byte** load in the
innermost loop, issued 256× per lane per key for 128 distinct bytes, each
followed by a `DecodeFp8E4M3` with two data-dependent branches. Read the row as
32 `uint32` words and decode each byte once: **32 loads + 128 decodes** versus
256 + 256. Pure memoization of a pure function over an unchanged accumulation
order, so it is bit-exact by construction.

### 4.3 Do **not** implement the 2-pass radix select

The brief's first indexer item targets the wrong phase. Selection runs over
`kSortLength = kMaxSeqLen = 2048`, **fixed and independent of `seq_len`**: 66
levels × `kKeysPerThread = 4` ≈ 264 compare-exchanges ≈ **2.1 k issue slots per
thread**, against **~400 k** for scoring. The sort is **~0.5 %** of the kernel.
Make it free and 10.6 ms becomes ~10.55 ms. The proposed UB saving buys nothing
either: `sortKeys` is 8,192 B of a 49,664 B layout against a 221,184 B ceiling,
and UB is not the constraint.

This would replace a selection network that is *verified bit-exact against
`std::sort` in `tests/emu`* (CLAUDE.md §5.5) for a ≤1.005× ceiling.

**Decisive check, already in your data:** the bench sweeps
`seq_len ∈ {512, 1024, 2048}`. Scoring-bound ⇒ ~2.65 / 5.3 / 10.6 ms. Sort-bound
⇒ roughly flat. Two rows settle it.

### 4.4 `fused_mhc_post_residual`: give the caller the tiling allocation

At ~54.6 µs per launch of per-call overhead on a 73.7 kB operand, this kernel
is pure launch cost. The fix is the pipelined wrapper variant CLAUDE.md §3.7
describes — caller-owned tiling, no per-call `aclrtMalloc`/sync/free — not
anything inside the kernel. §3.7 forbids inventing it without giving the caller
that ownership, so this is an API change to propose, not a local edit.

### 4.5 Cube/Fixpipe for `mxfp4_expert_gemv`: not worth it

The brief asks about migrating to Cube. At 2048×4096 this is a **GEMV** —
`n = 1`. Cube wants fractal operands (K0 = 64 for MXFP4, 16×16 fractals in
L0A/L0B); a single-column right-hand side wastes 15/16 of every B fractal. The
kernel is already weight-stream bound at ~21.5 GB/s against a 1.6 TB/s part,
so the ceiling is MTE2, not MAC throughput. Cube migration pays off at batched
expert GEMM (several tokens per expert), not decode GEMV. Revisit if the
decode shape ever batches.

---

## 5. Reproducing this

```bash
python -m venv venv && venv/Scripts/python -m pip install -e C:\PipeSim
venv/Scripts/python probe_model.py        # model constants
venv/Scripts/python indexer_scaling.py    # core-count sweep
```

Scripts are in this session's scratchpad, not committed — they encode modelling
judgements (how a warp-per-key SIMT scorer maps onto `vector_gemv`) that should
be re-derived rather than inherited.

**Caveat on everything above:** no part of this audit ran on an NPU. The
hardware figures are the user's; the simulator figures came from an
analytically-calibrated model that this audit shows to be ~15× optimistic in
absolute terms. Only the measured latencies are evidence.
