# ascend-moe-engine

Standalone DeepSeek-V4 Flash decode engine for Ascend 950PR (ACLNN V5):
torch-free, python-free, no parent project.

Extracted from the vLLM Ascend pilot via `git subtree split -P csrc/standalone`;
this repository carries that subtree's full commit history (root-level here).

## Layout

```
apps/       executables (moe_runner, dsv4_layer0_emu_bench,
            dsv4_semi_e2e_bench)
include/    engine public headers (moe/...), including moe/emu/ -- the
            header-only host emulation API, safetensors reader and FP8 types
src/        engine library (static arena, exclusive staging, router engine,
            IRoutedMoeBlock + standard/lattice backends, config.json parsing)
            and custom operators in op_host/op_kernel/cpu_ref partitions
src/emu/    plain C/C++ host emulation of the three fused decode operators
mock/       symbolic libopapi_mock backend (zero-NPU contract validation)
tests/      common test utilities plus sim/, mock/, npu/ and emu/ suites
scripts/    download_dsv4_layer0.sh -- metadata + the shard holding Layer 0
            download_dsv4_semi_e2e_weights.sh + dsv4_fetch_tensors.py --
            per-tensor HTTP range fetch, repacked into one compact file
python/     torch-free tokenizer + benchmark harness (tokenizers-only)
```

## Build

```bash
# standalone host CPU emulation (no CANN, no Docker, no NPU, no device):
cmake -S . -B build_emu -DENABLE_HOST_EMU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_emu -j$(nproc)
ctest --test-dir build_emu --output-on-failure

# host-only symbolic verification (no CANN, no Docker, no NPU):
cmake -S . -B build_mock -DENABLE_MOCK_RUNTIME=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_mock -j$(nproc)
ctest --test-dir build_mock

# native against a CANN 9.2.0 toolkit (container or device host):
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build_cann -DCMAKE_BUILD_TYPE=Release
cmake --build build_cann -j$(nproc)
cmake --build build_cann --target dsv4_verify_linkage   # zero undefined symbols, no Py/Torch
```

## Run

```bash
./build_cann/apps/moe_runner --weights /mnt/c/models/DeepSeek-V4-Flash \
    --prompt-ids 7 --max-new-tokens 32 --stats-json stats.json
./build_cann/apps/moe_runner --config /mnt/c/models/DeepSeek-V4-Flash/config.json --dry-run
```

`--stats-json` emits the machine-readable run record (TTFT, TPOT, swap
bytes, launch/residency counters) that `python/benchmarks` consumes.

`--diag-json` exports structured inference diagnostics in both dry runs and
decode runs. TTFT includes prefill and the first synchronized token readback;
TPOT averages subsequent generated tokens (null when fewer than two). Entropy
and perplexity are null until measured. Attention is currently dense MLA, so
the observed sparsity ratio is zero, not the hypothetical Top-512 reduction.
KV block counts sum the reservations across all 43 layers; utilization is the
active context divided by the rounded reserved capacity per layer. MoE counters
exclude startup ingestion and count expert requests, HBM hits, promotions, and
evictions during routing.

```bash
./build_mock/apps/moe_runner \
    --config /mnt/c/models/DeepSeek-V4-Flash/config.json \
    --synthetic-weights --prompt-tokens 128 --max-new-tokens 1 \
    --diag-json test_diag.json --dry-run
```

The dry run builds and plans without decoding, so latencies remain null and
request counters remain zero. Omit `--dry-run` to exercise the full pipeline.
`--prompt-tokens` supplies dummy zero IDs and requires synthetic weights. The
mock build uses symbolic weights and seeded routing/token outputs; its timing
measures host simulation, not NPU performance or model quality.

## Host CPU emulation (`ENABLE_HOST_EMU`)

A plain C++17 emulation of the three fused decode operators, built with nothing
but a host compiler. It exists to replace minutes-per-iteration CAModel
simulation with milliseconds, to prove the CPU fallback semantics, and to be
the golden numerical reference the Ascend C kernels are diffed against.

| operator | upstream golden it is ported from |
| --- | --- |
| `mhc_fused_sinkhorn_gemv_cpu` | ops-transformer `mhc_pre_sinkhorn_premix` `golden.py::_compute` + `mhc_post` `golden.py::_mhc_post_compute` |
| `paged_compressor_decode_c4a` | ops-transformer `compressor_golden.py::cpu_compressor` + the vendored arch35 RMSNorm/RoPE epilog |
| `fast_fp8_indexer_topk_cpu` | ops-transformer `quant_lightning_indexer_golden.py::cal_atten_per_batch_fp8` |

`ENABLE_HOST_EMU=ON` makes the root `CMakeLists.txt` return before any toolkit
discovery, so the tree cannot acquire a CANN dependency even by accident;
`ctest` re-checks that against the built binary (`emu_standalone_linkage`).

### End-to-end Layer 0 pass on real weights

```bash
scripts/download_dsv4_layer0.sh              # metadata + 16 MiB of shard 2
scripts/download_dsv4_layer0.sh --shard 4 --slice-mib 40   # layer 2's compressor/indexer
./build_emu/apps/dsv4_layer0_emu_bench       # ~0.14 s, writes the golden dump
```

Nothing under `data/` is in version control, and no reference output ever will
be. The checkpoint slices are **inputs**, fetched on demand by the scripts
above; the references are computed in memory, per run, from a printed seed
(CLAUDE.md section 4.5). That is deliberate and it is the whole basis of the
device suite's claim: a pre-baked golden on disk can go stale and make a broken
kernel pass, which is exactly the failure mode no amount of care downstream can
detect. `test_custom_kernels_npu_device` prints `references: generated per run,
no disk fixtures` for the same reason.

So a remote NPU host needs **no fixture transfer and no Git LFS**. Build
`moe_emu` there -- it links nothing but libstdc++/libm and needs no CANN, so it
runs anywhere -- and it computes the golden alongside the device result in the
same process. Only the tests that read the real checkpoint
(`test_safetensors_loader`, `test_semi_e2e_pipeline`, `dsv4_layer0_emu_bench`)
want `data/`, they SKIP cleanly when it is absent, and the fix is to run the
same download script on that host rather than to commit the weights.

Three things about the checkpoint that the obvious assumption gets wrong, all
read out of `model.safetensors.index.json` rather than guessed:

- **Layer 0 is in shard 2, not shard 1.** Shard 1 maps to exactly one tensor,
  `embed.weight` BF16 `[129280, 4096]` (1.06 GB); all 1565 `layers.0.*`
  tensors are in shard 2. The download script fetches shard 2 by default.
- **The tensor names are flat, with no `model.` prefix and no `mhc_pre`
  submodule**: `layers.0.hc_attn_fn` F32 `[24, 16384]` is the mHC phi,
  `hc_attn_scale` `[3]` is alpha, `hc_attn_base` `[24]` is bias (and likewise
  `hc_ffn_*`). There is no `input_layernorm` and no `self_attn`.
- **Layer 0 has neither a compressor nor an indexer.** `compress_ratios`
  starts `[0, 0, 4, 128, ...]`, so layer 0 is an uncompressed (SWA) layer.
  Those weights first appear on layer 2, and
  `layers.2.attn.compressor.wkv.weight` BF16 `[1024, 4096]` is what pins the
  C4A configuration (coff 2, D 512, ratio 4). The bench prints a provenance
  line per phase saying whether each stage ran on layer 0's weights, layer 2's,
  or a synthetic stand-in; nothing is silently substituted.

The script's default **slice mode** fetches only the header plus a leading
window of a shard, because every tensor the Layer 0 pipeline reads lives in the
first ~11.5 MB of shard 2's data segment. `safetensors_view.hpp` treats such a
file as well-formed: the header parses in full and each `TensorView` carries an
`available` flag, so a tensor outside the window is reported rather than
faulted on. Use `--full` for the FP8 projections.

### Semi-end-to-end multi-layer harness

`apps/dsv4_semi_e2e_bench` threads one token through a representative slice of
the stack, so the fused operators are exercised with real inter-stage data
dependencies rather than in isolation:

```text
layer 0   mHC round -> shared-KV SWA attention (window 128) -> mHC fold-back
layer 2   mHC round -> SWA attention
          + 4:1 compressor cadence (HOLD x3, EMIT) + FP8 indexer top-512
          -> mHC fold-back
layer 1   mHC round -> MoE (hash router, 6 routed MXFP4 experts + 1 FP8
          shared) -> mHC fold-back
```

```bash
scripts/download_dsv4_semi_e2e_weights.sh     # 365 MB, 101 tensors
./build_emu/apps/dsv4_semi_e2e_bench
```

Measured: **228 ms** wall clock, **334 MB** peak RSS (budgets 1.5 s / 4 GB).
The weight file is mmap'd and every quantized matrix is dequantized one output
row at a time; materializing FP32 copies would cost 4x for FP8 and 8x for FP4.
`--no-dump`, `--decode-steps`, `--indexer-seq-len`, `--memory-budget-mb` and
`--time-budget-ms` are all adjustable. Every stage prints a fidelity tag —
`reference`, `reference*` (one convention assumed) or `synth-input` — so what
is faithful and what is substituted is never implicit.

**Selective tensor fetch.** A leading-window slice is useless here: in shard 3,
expert 0 starts at byte 344,923,480 and expert 7 at 3,150,912,856, so the
~100 MB of expert weights is spread across 3.4 GB. The script issues one HTTP
range request per tensor and repacks the result into a *new, valid* safetensors
file with regenerated offsets — not a sparse shard-shaped file, whose holes
would read as zeros and be indistinguishable from real weights.

**Layers 0–2 are hash layers, so "the first N experts" are the wrong ones.**
`num_hash_layers: 3`, and for those layers the routed experts are a pure
function of the token id — `indices = tid2eid[token_id]`, with
`e_score_correction_bias = None`. The checkpoint corroborates it exactly:
`ffn.gate.tid2eid` is present on precisely 3 layers and `ffn.gate.bias` on
precisely the other 40. The script therefore fetches `tid2eid` first, reads the
row for `--token-id`, and downloads exactly those six experts; token 3 routes to
`[30, 205, 192, 177, 242, 102]`, none of which is in the first eight. The id and
its expert row go into the file's `__metadata__`, and the bench refuses a
different token rather than routing to experts the file does not carry.

**Two quantization schemes that look alike.** Dense and shared-expert weights
are `F8_E4M3` with one `F8_E8M0` scale per **128×128 block**
(`wq_a [1024, 4096]` → `scale [8, 32]`). Routed experts are E2M1 nibbles with
one scale per **row, per 32-element group** (`w1` logical `[2048, 4096]` →
`scale [2048, 128]`). Reading one with the other's indexing yields finite,
plausibly-scaled garbage, so `test_semi_e2e_pipeline` perturbs a single scale
and asserts exactly which outputs move.

Two things worth knowing about the expert weights: the packed payload is spelled
`I8` in the header, not `F4_E2M1` — `expert_dtype: "fp4"` plus the scale-grid
width is what establishes that the bytes are nibble pairs, and this repo's
`src/core/weight_source.cpp:600` gates on `"F4_E2M1"`/`"MOE_F4"`, which would
not match. The checkpoint alone cannot settle nibble order by distributional
statistics: swapping adjacent values preserves each group’s multiset and scale.
The official DeepSeek-V4-Flash `inference/convert.py` decoder explicitly stacks
**low then high**, so `kMxfp4LowNibbleFirst` is `true`.
`scripts/verify_mxfp4_packing.py` checks this elementwise on expert 30 and
uses a nonuniform GEMV probe; an all-ones probe is invariant to pair swaps.

**Shared KV, not per-head.** `wkv` maps 4096 → 512, which is `head_dim`, *not*
`n_heads * head_dim`: with `num_key_value_heads: 1` all 64 query heads score
against the same 512-wide row per token. That is what the vendored
`kv_quant_sparse_attn_sharedkv` is named for, and why the window cache is
128×512 floats rather than 64× that.

### PyTorch reference check (WSL CPU)

After building the host-only `dsv4_semi_e2e_bench` target, rerun it to produce
version 3 of `golden_semi_e2e_ref.bin`. Version 3 preserves the version-1
prefix and appends the synthetic input, intermediate `h_in` vectors, MoE
expert outputs, compressed KV token, and stream bundles for stage-level checks. With CPU PyTorch
and `safetensors` installed, run:

```bash
./build_emu/apps/dsv4_semi_e2e_bench --memory-budget-mb 2500
python3 scripts/verify_mxfp4_packing.py
python3 scripts/run_pytorch_layer_ref.py --source-only
python3 scripts/run_pytorch_layer_ref.py
python3 scripts/run_pytorch_layer_ref.py --input source-embedding --attention-formula official
```

The source-offset flag range-fetches and hashes each of the 101 compact
tensors against its original shard bytes without opening whole shards. The
default PyTorch run reproduces the C++ semi-e2e schedule and compares all
saved tensors with relative L2 and cosine similarity. The real-embedding run
fetches only `embed.weight[3]` and uses the official one-token attention
formula; it cannot be compared numerically with the golden’s synthetic input.
The compact cache contains attention 0 and 2 plus MoE 1, so these checks are
for the available 0 → 2 → 1 slice, not complete decoder layers 0–2.
The benchmark computes a four-step layer-2 compressed token and a synthetic
indexer side path, but does not feed those keys into its attention output.

### A measured caveat on the 20-iteration Sinkhorn bound

With the **real** `layers.0.hc_attn_*` weights, 20 Sinkhorn iterations do *not*
produce a doubly stochastic `hRes`: the row sums land ~7.6e-2 off (column sums
converge to ~1e-6, since the cadence ends on a column normalization). This is a
property of the checkpoint, not a solver fault — `hc_attn_base` rows 8..23 span
roughly `-30.1 … +9.9`, and six of the matching `hc_attn_fn` rows have an L2
norm below 1e-4, leaving those logits equal to their bias. `exp` then covers
~e^-40, and Sinkhorn-Knopp converges slowly that close to degenerate: the same
logits reach 1e-6 by 200 iterations.

Since `config.json` pins `"hc_sinkhorn_iters": 20`, the 20-iteration result
**is** the contract the NPU kernel must reproduce. The emulation suite therefore
asserts the Birkhoff invariant only on fixtures that actually converge, and pins
the real-weight regime by its measured value instead.

## Tests (`tests/`)

| binary | what it proves | needs a device? |
| --- | --- | --- |
| `mock_contract_smoke` | slot layout, safetensors binding, hierarchy invariants, arena, model-config contract | no |
| `npu_backend_probe` | ACL device query (init / count / SoC / HBM) + the six required operator symbols via `dlsym` | yes |
| `npu_weights_ingest_probe` | shard-00001 header parse + one real pinned→HBM DMA path | yes (`DSV4_MODEL_DIR`/`--model-dir`; SKIPs otherwise) |
| `npu_e2e_single_token` | one full 43-layer decode step: zero deadlocks, zero stream errors, 0 in-step allocations | yes, explicit `--coverage` |
| `mock_pipeline_e2e` | the full symbolic suite (mock build only) | no |
| `mock_safetensors_index_probe` | headers, aliases, HF/HC topology, all local expert descriptors; no shard payloads | no |
| `test_mhc_sinkhorn_emu` | Birkhoff invariants, the golden stable fixture in closed form, mass conservation + L2 non-expansiveness over 100 cases, `B^T` orientation, the real-checkpoint logit regime | no (emu build) |
| `test_paged_compressor_emu` | columnar softmax, RMSNorm + half-mode RoPE on the last 64 lanes, HOLD/EMIT cadence and the coff=2 window overlap | no (emu build) |
| `test_fp8_indexer_emu` | E4M3FN/E5M2/E8M0 round trips, per-head-relu ordering, top-512 identical to a full-sort baseline as a ranked list | no (emu build) |
| `test_safetensors_loader` | offsets/dtypes/shapes, misaligned data segments, truncated slices, malformed headers; parses the real shard when present | no (emu build) |
| `test_semi_e2e_pipeline` | FP8 128x128 vs MXFP4 group-32 scale indexing, window-cache eviction, hash vs noaux_tc routing, the asymmetric SwiGLU clamp, and the assembled multi-layer pass | no (emu build) |
| `emu_standalone_linkage` | the emu bench links no CANN, Python or Torch library and has no undefined symbols | no (emu build) |

`ENABLE_MOCK_RUNTIME=ON` builds the mock suite and the standalone CPU
simulation; `OFF` builds the NPU suite and CPU simulation. The simulator target
`test_mhc_fused_sinkhorn_gemv_sim` links only `test_common` and `cpu_ref`, so it
does not require CANN libraries or an NPU. Binaries live in `build_*/tests/`,
and CTest labels are `mock`, `npu`, and `sim`.

The portable fused operator reference accepts `x=[1,4,4096]` BF16,
`phi=[24,16384]` FP32, `alpha=[3]`, `bias=[24]`, and `h_out=[1,4096]` BF16.
The projection rows are partitioned as `[h_in(4), h_post(4), h_res(4x4)]`;
the three alpha entries scale those three segments respectively, before bias.
Twenty row/column Sinkhorn normalization passes produce the residual mixing
matrix, and the output is `B^T*x + h_post[:,None]*h_out`, rounded to BF16.
The host C API exposes a CPU fallback entry point with explicit argument and
execution status codes for callers that select the portable path.

All probes exit 0 with `[ SKIP ]` when the capability they probe is genuinely
absent on the host (no weights dir, no NPU); they fail only when a present
capability misbehaves.

NPU probes print `[ SKIP ] Physical Ascend NPU not detected` when ACL cannot
initialize or reports no device. A driverless CANN container must still have
its shared libraries loadable before `main`: for skip-path testing only, add
the toolkit's `x86_64-linux/devlib/linux/x86_64` directory to `LD_LIBRARY_PATH`
alongside `x86_64-linux/lib64`. Use real driver libraries on the server.

Embedding lookup prefers `model.embed_tokens.weight`, then `embed.weight`,
then `embed_tokens.weight`. HC attention/FFN base keys are layout markers;
they never substitute for RMSNorm or projection weights. MTP keys are ignored
by backbone resolution. Passing the index probe does not certify shape/dtype
or folded-MLA execution compatibility; checkpoint conversion is still required
where the runner's absorbed projections are absent.

## Python harness (`python/`, zero torch)

```bash
pip install tokenizers                      # the only dependency (Rust-backed)

cd python
python -m tokenizer.dsv4_tokenizer "Hello DeepSeek"

# TTFT/TPOT + swap sweep across context lengths (engine as subprocess):
python -m benchmarks.bench_runner --engine ../build_cann/apps/moe_runner \
    --context-lengths 128,512,2048 --new-tokens 32 \
    --weights /mnt/c/models/DeepSeek-V4-Flash --out results.json

# operator timeline via msprof, summarized on the CLI:
python -m benchmarks.profile_cann --engine ../build_cann/apps/moe_runner \
    --engine-args "--synthetic-weights --prompt-ids 7 --max-new-tokens 4" \
    --output prof_out
```

## Model contract

The runner reads the checkpoint's `config.json` at startup and refuses to run
unless the router contract (`noaux_tc`, `sqrtsoftplus`, `norm_topk_prob`) and
the compiled graph geometry agree. Expert slots are 12.75 MiB FP4/E8M0
(hidden 4096, intermediate 2048); the exclusive RAM/VRAM hierarchy keeps
|Set_Device| + |Set_Host| = 11,008 experts with zero runtime disk reads.
