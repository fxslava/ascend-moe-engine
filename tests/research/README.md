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
