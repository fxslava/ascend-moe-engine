# Audit: porting `mxfp4_expert_gemv` primitives into `dsv4_moe_expert`

**Date:** 2026-10-13
**Method:** static source inspection only. PipeSim and `ascend-kernel-analyzer`
were not run, per the task constraint. Measured figures are quoted from
`docs/pipesim_audit_2026-10-12.md`; nothing here was timed in this session.
**Target:** Ascend 950PR, arch35 / `dav-3510`.

---

## 0. Headline: the premise does not hold

The task asks me to extract optimizations from `mxfp4_expert_gemv` and port
them into `dsv4_moe_expert`. **There is nothing to port.** The MXFP4 GEMV
arithmetic core in the two operators is the same code.

`mxfp4_expert_gemv_kernel.cpp:72-84` and `dsv4_moe_expert_kernel.cpp:73-85`
are **byte-identical over all 13 lines** of the group/nibble loop, after
renaming only the two row-pointer locals (`prow`/`srow` →
`packed_row`/`scale_row`). Verified by normalized diff, not by eye:

```text
const float scale = DecodeE8M0(asc_ldca(S + lane_group_base + g));
float partial = 0.0f;
const std::uint32_t p0 = lane * pairs_per_lane + g * 16u;
for (std::uint32_t q = 0; q < 16u; ++q) {
  const std::uint32_t p = p0 + q;
  const std::uint32_t byte = asc_ldca(P + p);
  partial += DecodeFp4E2M1(byte & 0xFu) * x_ub[2u * p] +
             DecodeFp4E2M1(byte >> 4u) * x_ub[2u * p + 1u];
}
total += partial * scale;
```

The butterfly reduction is also the same four `asc_shfl_xor` masks
(1, 2, 4, 8) at width 16. The only structural difference is that
`dsv4_moe_expert` factors the row into a shared `__simt_callee__`
`LaneRowPartial()` + `ReduceRow()` pair, because it runs that row three times
(w1, w3, w2) across two entry points. Its own comment says so
(`dsv4_moe_expert_kernel.cpp:61-63`).

### 0.1 Every symbol attributed to `dsv4_moe_expert` is fictional

Each of the following, named in the task as existing code to be inspected or
replaced, has **zero occurrences** anywhere in the repository:

| Named in task | Occurrences |
| --- | --- |
| `DecodeCodes()`, `DecodeScales()` | 0 |
| `BalancedChunkCols`, `ReduceHalfBlock`, `SplitColumns` | 0 |
| `partRawBuf_`, `partHiBuf_` | 0 |
| `xFloat_`, `gateAcc_`, `upAcc_`, `downAcc_` | 0 |
| `BitAnd` | 0 |
| `dsv4_moe_expert_vector_compat.h` | file does not exist |

There is no SIMD bit-shift unpacking path in `dsv4_moe_expert`, or in any
operator in this repository. Across all nine kernel translation units,
`Select(` appears once (in `fast_fp8_indexer_select`, unrelated to MXFP4) and
`ShiftRight`/`ShiftLeft` appear **zero** times. Both MXFP4 kernels call the
same `DecodeFp4E2M1` / `DecodeE8M0` from the shared
`include/moe/kernels/arch35/arch35_common.hpp:81,102`.

Two further premises are also false:

- **`dav-c310`** is not this target. The string appears nowhere in the repo or
  in `C:\PipeSim\ASCEND_950_ARCHITECTURE.md`. The device gate compiles
  `--npu-arch=dav-3510` (`scripts/device_compile_kernels.sh:118`).
- **There is no Ascend 910B / `dav-c220` fallback** for these operators to
  regress. `dav-c220` appears nowhere; `910B` appears only in
  `third_party/ops_dsv4` vendor code and `include/moe/core/op_table.hpp`.
  The six arch35 operators are single-target `dav-3510`.

---

## 1. Side-by-side comparison matrix

| Feature | `mxfp4_expert_gemv` | `dsv4_moe_expert` |
| --- | --- | --- |
| Nibble decode | `DecodeFp4E2M1`, shared header | **same function** |
| Scale decode | `DecodeE8M0` (`ldexpf`), shared header | **same function** |
| Weight fetch | `asc_ldca` per byte, GM → SIMT DataCache, never UB | **same** |
| Scale application | once per 16-pair (32-col) group, post-subtotal | **same** |
| Row reduction | in-register `asc_shfl_xor` ×4, width 16 | **same**, factored into `ReduceRow()` |
| Lane geometry | 32 rows × 16 lanes = 512 threads | **same** |
| Grid | 32 AIVs, grid-stride over 32-row tiles | **same** |
| Kernel type | `KERNEL_TYPE_MIX_AIV_1_0` | **same** |
| UB mechanism | `LocalMemAllocator`, dynamic region, static = 0 B | **same**, same `-Wundefined-inline` rationale |
| UB footprint | 16,512 B (x 16,384 + tile 128) | gate_up **16,512 B**; down 8,320 B |
| Sync primitives | `MTE2_V` flag, `DataSyncBarrier<ALL>`, `V_MTE3` flag | **same sequence** |
| Live UB tensors | 2 (x, y-tile) | 2 (x *or* act, + tile) |
| Entry points | 1 | 2, stream-ordered |
| GM scratch | 0 B | `inter × 4` = 8,192 B (`activated`) |
| Fused epilogue | none | asymmetric clamp + silu + mul, in-register |

**Buffer count, register pressure and cycle estimates are not tabulated.**
CLAUDE.md §5.6 permits only measured hardware latency to rank kernel changes,
and §5.7 forbids claiming hardware behaviour that has not been observed. A
static cycle estimate for either kernel would be exactly the kind of number
the PipeSim audit found to be ~15× wrong. See §4.

---

## 2. What `mxfp4_expert_gemv` actually contains

For the record, since the audit was requested: its genuinely notable
properties, **all of which `dsv4_moe_expert` already has**.

1. **Weights bypass UB entirely.** `asc_ldca` streams packed bytes GM → SIMT
   DataCache → registers. UB holds only the activation vector and a 128 B
   output tile, which is why the footprint is 16,512 B against a 221,184 B
   ceiling (7.5%).
2. **Branch-free E2M1 decode.** `arch35_common.hpp:90` uses a select on
   `mag_code < 2u`, explicitly commented "select, not a branch on data", so
   warp divergence is not a function of weight values.
3. **Bit-exact group ordering.** The scale multiplies the 32-column subtotal
   *after* the subtotal, matching the oracle's grouping, so the device result
   is reproducible against `mxfp4_gemv_cpu` rather than merely close.
4. **Dense reduction in registers.** The header
   (`mxfp4_expert_gemv_kernel.cpp:27-30`) records *why*: a vector `ReduceSum`
   cannot express 32 independent 16-lane row reductions on this API
   generation (no sub-tensor views).
5. **Dynamic-UB allocation for every SIMT-visible buffer.** The error-341
   probe rationale (lines 32-41). `dsv4_moe_expert` inherited this.

`dsv4_moe_expert` adds, on top of all five: a shared `__simt_callee__` row
helper, the fused asymmetric clamp + silu + mul epilogue, and the two-phase
stream-ordered split.

---

## 3. Pillar-by-pillar findings

### Pillar A — unpacking micro-architecture
No contrast exists; both call the same shared function. No native vector
unpack intrinsic, LUT, or widen/shuffle path is used by either, and none is
used anywhere in the repo. **Cannot quantify stall cycles or instruction
savings** — §5.6/§5.7. What *is* statically countable, per lane-row at
`cols = 4096` (`pairs_per_lane = 128`, 8 groups):

- 128 `asc_ldca` byte loads
- 256 `DecodeFp4E2M1` calls → **256 `ldexpf`**
- 8 `DecodeE8M0` → 8 `ldexpf`
- **264 `ldexpf` total**

This is identical in both kernels. See §4.2 for why reducing it is
mis-targeted.

### Pillar B — reduction and accumulator staging
The named functions do not exist. There are **no intermediate staging
buffers** to eliminate: the accumulation is a single FP32 register `total`
per lane, and scale multiplication is *already* merged into the accumulation
stage (`total += partial * scale`). The request to "eliminate intermediate
buffers and merge scale multiplication into accumulation" describes the
code as it already is.

Tail handling: `rows` is 2048 or 4096 and `kRowsPerTile` is 32, so tiles
divide exactly and the `global_row < rows` guard plus the `lane == 0` padding
store are dead paths at the only admitted geometry. Lane alignment is exact —
`pairs_per_lane = (cols/2)/16` is a multiple of 16 for both 4096 and 2048, so
no lane straddles a scale group. This is the invariant
`tests/emu/test_custom_kernels_emu.cpp:778` exists to verify on the host.

### Pillar C — UB topology
The five concurrent buffers in the question do not exist. **Neither kernel
ever holds more than two UB tensors.** `gate_up` holds x (16,384 B) + a 128 B
tile; `down` holds `activated` (8,192 B) + a 128 B tile. There is no third
accumulator because gate and up are consumed into `act_tile` in the same VF
invocation that produces them.

Fragmentation is structurally impossible: allocation is a function-local
bump pointer (`LocalMemAllocator`), two `Alloc` calls, never freed, reset per
instance. Static UB is 0 B in both, so there is no static/dynamic overlap —
the error-341 / 507035 class CLAUDE.md §5.6 notes PipeSim is blind to.

Bank contention is the one open question I cannot resolve statically. The
arch doc gives an eight-bank, 32 B-interleave model; each lane reads a
contiguous 128 B span of x per group, spanning 4 banks, and all 32 rows of a
tile read *the same* x addresses (a broadcast). Whether that coalesces or
serializes is a hardware question. **It is identical in both kernels**, so it
is not a porting consideration either way.

Headroom: `gate_up` uses 16,512 of 221,184 B. There is no capacity obstacle
to any restructuring — capacity is not the binding constraint anywhere here.

### Pillar D — SIMD/SIMT hybrid potential
Both kernels are *already* hybrid: `KERNEL_TYPE_MIX_AIV_1_0`, with
`asc_vf_call` SIMT VFs doing unpack + FMA in registers while the SIMD side
owns only the MTE2 activation load and the MTE3 tile store.

The specific proposal — "decouple weight unpacking to scalar/SIMT threads
while reserving the Vector Core strictly for FP32 FMA" — **is not realizable
on this core.** `ASCEND_950_ARCHITECTURE.md:46-47` states SIMT and SIMD
instructions "contribute to one Vector stage", and lines 57-58 that every
`asc_vf_call` boundary "reconfigures the **physically shared** Vector ALU
between SIMD and SIMT warp issue", priced at
`vf_mode_switch_penalty_cycles = 64` (validated 30..80). Unpack-on-SIMT and
FMA-on-SIMD cannot overlap because they contend for one physical ALU;
splitting them adds mode-switch boundaries and buys no concurrency. Doing
both in one SIMT VF — what both kernels already do — is the correct shape.

---

## 4. The real finding, and the ranked roadmap

### 4.1 Both operators are weight-stream bound, not ALU bound

From the geometry, per expert per token:

| | bytes | |
| --- | --- | --- |
| w1 packed + scale | 4,194,304 + 262,144 | |
| w3 packed + scale | 4,194,304 + 262,144 | |
| w2 packed + scale | 4,194,304 + 262,144 | |
| **total** | **13,369,344 (12.75 MiB)** | |
| FLOP | 50,331,648 (50.3 MFLOP) | |
| **arithmetic intensity** | **3.76 FLOP/B** | |

That is the same 3.7 FLOP/B the 2026-10-12 audit records for
`mxfp4_expert_gemv`, which it classifies **MTE2 / weight-stream bound** from
a measured 208 µs / ~21.5 GB/s. Cross-check: 13.37 MB at 21.5 GB/s predicts
**622 µs/expert**, against **624 µs** for three measured `mxfp4_expert_gemv`
launches — agreement to 0.3%, which is strong evidence the weight stream, not
the decode, sets the time.

The audit's own conclusion applies directly: *"Every other operator is a
decode-shaped weight streamer under 4 FLOP/B. That single fact determines the
roadmap."* The indexer at ~128 FLOP/B is the only compute-bound kernel in the
set, and PipeSim put the Vector ALU at **37.4%** utilization even there.

### 4.2 Consequence: every optimization in the task brief is mis-targeted

Pillars A, B and D all propose reducing Vector-ALU work — fewer unpack
instructions, fused accumulation, moving unpack off the Vector core. At
3.76 FLOP/B with MTE2 starving, that attacks a stage that is not binding.
The 256-`ldexpf`-per-lane-row count in §Pillar A is real, and a 256-entry
byte → `{even, odd}` LUT would cut it to 8. The host oracle already does
exactly this (`src/emu/dsv4_quant_gemv.cpp:69-96`, `Fp4Pair` table, comment:
"instead of being branched on 67 million times per expert"). But on device
that trade is **ALU work for UB traffic**: a 2 KB table, with a
data-dependent gather per byte across an 8-bank/32 B-interleave UB, on a
kernel whose problem is already that `asc_ldca` DataCache fills and MTE2
"share bandwidth" (`ASCEND_950_ARCHITECTURE.md:46`). It plausibly makes
things worse. It is also a change to the *shared* `arch35_common.hpp`,
benefiting both MXFP4 kernels equally — not a port between them.

### 4.3 What would actually be worth trying, ranked

Ranked by expected effect on the binding stage. None of these is a port from
`mxfp4_expert_gemv`, because that kernel has no weight-stream trick
`dsv4_moe_expert` lacks.

1. **Share the weight stream across the six routed experts, or the token
   batch.** The weights are read exactly once per expert per token and never
   reused — this is the whole 12.75 MiB. At batch size 1 there is no reuse to
   find; across tokens routed to the same expert there is. This is a
   *scheduling* change above the operator, not a kernel change, and it is the
   only lever that changes the 3.76 FLOP/B ratio itself.
2. **Fold `w1` and `w3` into one interleaved weight matrix.** gate and up
   share `x` and are both `[inter, hidden]`. One packed buffer with
   interleaved rows would halve the number of distinct GM streams in
   `gate_up`, improving DataCache locality and MTE2 transaction shape without
   touching the arithmetic. This is a real, kernel-local change — and it is
   an *addition* to `dsv4_moe_expert`, with no counterpart in
   `mxfp4_expert_gemv`.
3. **Collapse the two dispatches into one.** Currently 6 `asc_vf_call` per
   core per expert (2 gate_up + 4 down) plus two kernel launches and an
   8,192 B GM round-trip for `activated`, re-read by all 32 cores (262,144 B
   of read traffic). The split is forced by the reduction shape — `down`
   reduces over all of `activated` while `gate_up` produces it tiled across
   cores — so this needs a cross-core barrier the launch layer deliberately
   avoids (`dsv4_moe_expert.cpp:6-11`). High effort, and the GM traffic saved
   is ~2% of the 12.75 MiB. **Low priority despite being the most obvious.**
4. **The decode LUT (§4.2).** Last, and only with hardware measurement on
   both sides. Do not land it on a static argument.

Note that items 1 and 2 concern memory, 3 concerns dispatch, and only 4
concerns the ALU — the inverse of the task brief's emphasis.

### 4.4 What `dsv4_moe_expert` already won

Worth stating, since the brief assumes it is the weaker kernel. Against the
composite path it replaced (three `LaunchMxfp4ExpertGemv` + `aclnnClamp` ×2 +
`aclnnSwiGlu`), it eliminates:

- 6 dispatches and 2 stream syncs per expert → 1 wrapper call
- the two aclnn clamp stages, whose argument order encoded the clamp
  asymmetry at the call site
- one redundant `x` UB load (gate and up now share one residency)
- two full GM round-trips of the `[inter]` gate and up vectors

It is strictly the better structure. The 622 µs floor is weight bandwidth,
and no inner-loop change reaches it.

---

## 5. Risk and compatibility

| Risk raised | Assessment |
| --- | --- |
| Ascend 910B / `dav-c220` fallback regression | **Not applicable.** No such path exists for these operators; single-target `dav-3510`. |
| UB memory pressure | **Not a constraint.** 16,512 / 221,184 B used. Any of §4.3 fits. |
| BiSheng compilation compatibility | Current state verified this session: device compile gate 21/21, 0 warnings; NPU build gate OK. Any change to `arch35_common.hpp` (the LUT) recompiles **all six** operators and must re-run both gates. |
| Bit-exactness against the oracle | The per-group scale-after-subtotal order is load-bearing. Items 2 and 3 preserve it; a LUT preserves values but **must** keep `kMxfp4LowNibbleFirst` baked in the same orientation, which is the one thing the all-ones GEMV cannot detect (`mxfp4_expert_gemv_kernel.cpp:206-207`). |
| Shared-header blast radius | `DecodeFp4E2M1` is used by both MXFP4 kernels; it is not op-private. Changing it is a six-operator change, not a `dsv4_moe_expert` change. |

---

## 6. Recommendation

Do not undertake this port: the two operators share the arithmetic core
byte-for-byte, and `dsv4_moe_expert` is the more fused of the two. The brief's
four pillars target the Vector ALU, which measured data puts at ~37%
utilization on the one compute-bound kernel in the set and which is not the
binding stage at 3.76 FLOP/B.

If `dsv4_moe_expert` is to be made faster, the next step is §4.3 item 2 (fuse
the w1/w3 streams), measured on hardware. Per CLAUDE.md §5.6 nothing here
ranks as an optimization until it is timed on a device; per §5.7, note that
`dsv4_moe_expert` has still never executed — it compiles and links only.
