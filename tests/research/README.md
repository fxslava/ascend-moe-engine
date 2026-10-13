# CSA TurboQuant + lattice-VQ research probe

This directory is a deterministic, CPU-only experiment around the CSA cache
boundary. It does not import Torch-NPU, CANN, or any device library.

The repository reference path is `src/emu/semi_e2e_pipeline.cpp`: layer 2
drives the C4A compressor at a 4:1 HOLD/HOLD/HOLD/EMIT cadence, and the
lightning indexer reads a separate `PA_BSND [blocks, block_size, 1, 128]` FP8
key cache. The checkpoint-backed compressor itself emits 512 lanes. This probe
uses the indexer's 128-lane PA_BSND geometry because it is the cache that is
ranked by the attention retrieval probe.

`turbo_lattice_vq.py` contains:

- finite E4M3 block scaling;
- a deterministic scalar Lloyd-Max codebook;
- exact nearest-coset decoding for the 8D E8 lattice
  `D8 union (D8 + (1/2,...,1/2))`;
- a 24D mirrored-Golay Chase decoder for the integer embedding of the Leech
  lattice, with dimensions padded from 128 to 144;
- 3-bit or 4-bit packed Lloyd indices, packed signed-int16 lattice coordinates,
  and FP32 group scales;
- a paged K/V cache that quantizes only in `write()` and decodes only in
  `read()`/`attention()`;
- a standalone lane-wise CSA compressor model with ratio 4.

The synthetic compressor uses uniform gate logits by default. It models the
lane-wise reduction and cadence, not the checkpoint projections or C4A's
previous-a/current-b overlap. Those are already covered by the C++ emulator;
claiming them here without checkpoint weights would make the standalone probe
misleading.

Run focused tests:

```bash
python3 -m unittest discover -s tests/research -p 'test_*.py' -v
```

Run the requested NIAH grid:

```bash
python3 tests/research/niah_attention_regression.py
```

For a quick smoke run:

```bash
python3 tests/research/niah_attention_regression.py \
  --context-lengths 256,1024 --depths 10%,50%,90% --calibration-vectors 64
```

Each grid cell reports recall, the 1-based compressed-slot rank, Spearman rank
correlation between exact-compressed and decoded-cache scores, and the
requested dilution ratio. The indexer's original-token `top_k=512` is scaled
by the 4:1 compressor cadence, so the default compressed-cache window is 128.
Use `--csv path.csv` to save one row per case.

## Extreme contexts

`extreme_niah_benchmark.py` evaluates 8K, 32K, 256K, and 1M-token
contexts at depths 10%, 25%, 50%, 75%, and 90%. It requires NumPy and is
intended to run in the CANN Python 3.12 container. The 1M-token path never
allocates `[1M, 128]`: it generates 8,192-token chunks, compresses and packs
them, scores the decoded keys, and then releases the chunk. All depths share
one haystack pass per context.

The streaming indexer keeps only a top-129 heap (one spare entry permits
replacement of each needle slot), exact better-than-needle counters, and
4,096 deterministic score samples for Spearman rho. The default invocation
writes `tests/research/extreme_niah_results.csv`:

```bash
python3 tests/research/extreme_niah_benchmark.py
```

Use a smaller grid for a smoke run:

```bash
python3 tests/research/extreme_niah_benchmark.py \
  --context-lengths 8192 --depths 10%,50%,90% --no-csv
```

## Six-way ablation

The six configurations combine 3-bit/8-level or 4-bit/16-level analytic
standard-normal Lloyd-Max codebooks with no lattice, E8, or E24 residual
quantization. Run the complete deterministic sweep inside the CANN Python
3.12 container with:

```bash
bash tests/research/run_all_6_ablations.sh
```

The runner writes one 20-row CSV per configuration beside the scripts and
prints a final markdown comparison table. The E24 implementation is a
bounded-distance mirrored-Golay Chase decoder; it always emits a valid Leech
lattice point, but does not claim exhaustive maximum-likelihood decoding.

## Nine-way adversarial multi-needle ablation

`adversarial_multi_needle_benchmark.py` streams a 128-dimensional anisotropic
Gaussian haystack through the 4:1 synthetic CSA compressor. Its covariance has
eigenvalues `exp(-2.5*k/128)`; channels 0, 16, 64, and 96 are amplified 8x
before every token is normalized. Sixteen unit-norm needles are placed between
5% and 95% context depth, with one 9 dB SNR query and one paired distractor
whose cosine to that query is in `[0.70, 0.82]`.

The nine configurations use 2-, 3-, and 4-bit analytic standard-normal
Lloyd-Max codebooks, each with scalar-only, E8, and E24 residual codecs. E24
uses the existing mirrored-Golay Chase bounded-distance decoder into
`sqrt(8) * Lambda_24`; it is not an exhaustive nearest-point search. The
default 8,192-token chunk size bounds the dense token buffer. Top-128 metrics
are calculated against all 16 needle queries and their paired distractors.
The distractor false-positive rate is the fraction of each query's paired
distractors that land in its top-128 decoded-cache results.
`packed_key_stream_mib` reports key payload bytes (scales, packed indices, and
lattice coordinates), excluding values and runtime temporaries.
Mean exact needle ranks require a streamed score prepass that quantizes only
chunks containing needles; the reported stream latency includes that prepass.
Each completed row also enforces the 120 MiB process peak-RSS ceiling.

Before the 1M-token sweeps, the master script asserts the 2-bit codebook and
the exact four-indices-per-byte packing format:

```bash
bash tests/research/run_all_9_adversarial_ablations.sh
```

The runner writes one four-row CSV per configuration under `tests/research/`
and prints the requested markdown summary. A reduced smoke run for one
configuration is:

```bash
python3 tests/research/adversarial_multi_needle_benchmark.py \
  --config cfg1_tq2_lloydmax \
  --context-lengths 8192,32768 \
  --chunk-tokens 2048
```

## Faithful DSV4 MLA geometry (decoupled, unquantized RoPE)

The probes above are flat 128-lane: they conflate the positional and semantic
dimensions, quantize both together, and collapse 64 heads into one dot
product. `dsv4_mla_niah_benchmark.py` is the corrected geometry.

### What changed, and against which constant

| Property | Flat probe | Faithful | Source |
| --- | --- | --- | --- |
| Cached row | one 128-lane vector | 512 semantic + 64 rotary | `kDefaultKvLoraRank`, `kDefaultQkRopeHeadDim` |
| RoPE lanes | quantized with the key | **never quantized** | -- |
| Heads | 1 | 64, head-max aggregation | `kIndexerHeads` |
| Temperature | `1/sqrt(128)` | `1/sqrt(192)` | `kDefaultQkNopeHeadDim` + rope |
| RoPE theta | n/a | 160000, on the compressed index | `compress_rope_theta` |
| Compression | 4:1 | 4:1 | `kCompressRatioCsa` |

`turbo_lattice_vq.py` gains `DecoupledMlaCache` (packed 512-lane latent plus
raw float32 rotary lanes), `MultiHeadMlaIndexer` (the decomposed per-head
logit with the up-projection absorbed into the query), and `rms_norm`, which
mirrors `moe::emu::rms_norm_cpu` -- the engine spells it `rms_norm_cpu`, not
`rmsnorm_cpu`.

### Two layouts, both real

`kv_lora_rank + qk_rope_head_dim` is 576 and FIA V5 reads it as two tensors,
`key` [*, 512] and `keyRope` [*, 64]. The CSA compressor's own row is 512 wide
in **total**, with RoPE applied to its last 64 lanes
(`kRopeBaseOffset = 512 - 64 = 448`), leaving 448 semantic lanes. These are
different shapes and `DecoupledMlaCache` names which one it holds
(`layout="split"` or `"in_row"`); the benchmark uses the split layout. Keeping
the rotary lanes out of the quantizer is identical in both.

### Running it

```bash
bash tests/research/run_faithful_dsv4_ablations.sh
```

Nine configurations -- {2, 3, 4} bits x {scalar Lloyd-Max, E8, E24} -- over
8K/32K/256K/1M, writing one `faithful_dsv4_results_*.csv` each and printing a
markdown comparison. The dimension contracts run first, so a geometry change
stops the sweep before it produces numbers for the wrong shape. Budget a few
hours for the full grid; the E24 Chase decoder over 262,144 slots of 512 lanes
dominates. Cut it down with `CONTEXT_LENGTHS=8192,32768`.

Peak RSS stays under the 150 MiB ceiling (observed 77-96 MiB) by streaming
8,192-token chunks in 512-slot batches and generating token content from a
position-keyed hash, so any slot is reproducible on its own and none has to be
held between the two passes.

### Results (seed 20261013, full grid)

| Config | Bits | Lattice | Bits/lane | Recall@128 | Mean rank | Spearman rho | CF phase err | 1M latency |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| cfg1 | 2 | scalar | 3.00 | 0.7969 | 1010.2 | 0.899969 | 18.60 deg | 65.4 s |
| cfg2 | 2 | e8 | 19.00 | 0.8906 | 346.7 | 0.993521 | 8.13 deg | 62.0 s |
| cfg3 | 2 | e24 | 19.50 | 0.9062 | 337.0 | 0.991400 | 7.98 deg | 369.0 s |
| cfg4 | 3 | scalar | 4.00 | 0.8906 | 391.8 | 0.970886 | 9.92 deg | 65.3 s |
| cfg5 | 3 | e8 | 20.00 | 0.9062 | 334.5 | 0.996505 | 4.74 deg | 91.2 s |
| cfg6 | 3 | e24 | 20.50 | 0.9062 | 324.0 | 0.996867 | 4.57 deg | 341.6 s |
| cfg7 | 4 | scalar | 5.00 | 0.9062 | 298.5 | 0.992616 | 5.24 deg | 75.3 s |
| cfg8 | 4 | e8 | 21.00 | 0.9062 | 321.5 | 0.998777 | 2.83 deg | 78.6 s |
| cfg9 | 4 | e24 | 21.50 | 0.9062 | 307.5 | 0.998653 | 2.73 deg | 314.8 s |

Peak RSS across all 36 runs: 78.6-97.3 MiB, against the 150 MiB ceiling.
All 36 rows report `rope_quantized=False` at 512/64 geometry.

**The lattice rows are not iso-bitrate with the scalar rows, and the headline
framing of this ablation is misleading without that column.** `PackedVector`
stores every lattice coordinate as a raw `int16`, so an E8/E24 row carries 2
bytes per semantic lane on top of its 2-4 bit Lloyd index: ~19-21 bits/lane
against the scalar path's 3-5. "2-bit E8 matches 4-bit scalar in rho" is
therefore a statement about quality at roughly **four times** the storage, not
a rate-distortion win -- at equal footprint the scalar path is ahead. The fix
is to entropy-code the coset index (E8 needs ~8 bits per 8D block, about 1
bit/lane) rather than store doubled coordinates verbatim; that is a change to
the quantizer's wire format, which the existing suites depend on, so it is
deliberately not done here. The same inefficiency is present in the flat
128-lane probes, where it was never surfaced.

Where the configurations actually separate:

- **Recall@128** only discriminates at 1M. It is 1.0 for every configuration
  at 8K, 0.875 for all but one at 256K, and splits at 1M: 0.438 for 2-bit
  scalar against 0.688-0.750 for the rest.
- **Spearman rho** orders cleanly across the whole grid and is the metric to
  read at short contexts: 0.900 (2-bit scalar) < 0.971 (3-bit scalar) < 0.993
  (4-bit scalar) < 0.9988 (4-bit E8).
- **2-bit scalar is the only configuration that genuinely breaks**: it is also
  the only one whose `mean_dilution` falls materially below 1.0 (0.897, i.e.
  it erodes the needle's own logit by ~10%).
- **E24 costs 4-5x the latency of E8** for no measurable quality gain over it
  at these widths.

### Reading the metrics

`mean_spearman_rho` and `mean_needle_absolute_rank` are the discriminating
columns at 8K-32K; `mean_needle_recall_at_128` only separates once there is
enough background to rank against (it is 1.0 for most configurations at 8K and
starts falling at 256K, reaching 0.44 for 2-bit scalar at 1M).

Both the needle and its paired distractor are scored by **rank** -- the number
of slots that strictly beat them -- and counted as recalled or false-positive
when that rank is `< 128`. The earlier probes test membership of a top-K heap
instead, which requires the needle's pass-1 score to equal its pass-2 score
bit-for-bit; those are computed in batches of different shapes, which BLAS may
evaluate with different kernels, so the test silently misses hits. Switching
the 2-bit scalar configuration to rank counting moved its measured distractor
false-positive rate from 0.06 to 0.50. Note also that `rank <= 128` would
count the 129th-placed slot as recalled; the comparison here is strict.

`counterfactual_rope_phase_error_deg` is a **control, not a measurement of the
pipeline**: it applies the same codec to the rotary lanes and discards the
result, purely so the report can say what the decoupling buys. It runs
18.0 degrees at 2-bit scalar down to 2.8 degrees at 4-bit E8/E24. The shipped
path's rotary error is zero and that is structural, not measured --
`DecoupledMlaCache` has no code path that quantizes rope, which
`test_dsv4_mla_dimensions.py` asserts directly, alongside a degeneracy guard
that the semantic path on the same slot really is lossy.

`mean_rope_logit_share` should sit near the 64/192 ~ 0.33 the geometry implies
(observed ~0.20). Two fidelity details set it, and both were wrong in the
first draft:

- The row is RMSNorm'd **as one vector** before RoPE, which is what
  `rms_norm_cpu` does and what puts the 64 rotary lanes at `sqrt(64)` against
  the semantic lanes' `sqrt(512)`. Normalizing the halves separately left the
  rotary term at ~1% of the logit.
- `W_UK` is **learned**, so each head's row space is seeded with an
  orthonormal basis of the needle directions. A purely random `[128, 512]`
  projection retains only `sqrt(128/512) ~ 0.48` of any latent direction,
  discarding three quarters of the semantic energy while the rotary term loses
  none -- which inflated the rotary share to ~0.5 and buried the deepest
  needles for reasons unrelated to quantization.

### Tests

```bash
python3 -m unittest discover -s tests/research -p 'test_dsv4_*.py' -v
```

`test_dsv4_mla_dimensions.py` parses the engine's own headers and asserts the
Python constants agree, so editing `config.hpp` or `cpu_kernels.h` turns it
red. `test_dsv4_mla_niah_benchmark.py` checks the rotary and scoring index
arithmetic against scalar transcriptions of `BuildRopeTables` and the half-mode
branch of `paged_compressor_decode.cpp`, including that the rotary inner
product is a function of the position *difference* only -- the phase coherence
property the decoupling exists to preserve.
