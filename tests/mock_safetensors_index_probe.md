`mock_safetensors_index_probe` exercises the production header parser
(`ParseSafetensorsHeaderJson` / `SafetensorsWeightSource`) and the resolver used
by `StaticArenaManager::IngestBackbone`. Fixtures are generated inline; each
disk fixture is exactly 1024 bytes, with a little-endian size prefix, padded
JSON, zero-element shapes, and `[0, 0]` data offsets. No weights are downloaded.

Run on x86 WSL:

```sh
cmake -S . -B build_mock -DENABLE_MOCK_RUNTIME=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_mock --target mock_safetensors_index_probe -j$(nproc)
ctest --test-dir build_mock -R '^mock_safetensors_index_probe$' -V
```

The optional physical probe reads only
`${DSV4_MODEL_DIR:-/mnt/c/models/DeepSeek-V4-Flash}/model.safetensors.index.json`.
Missing indexes print `[ SKIP ]` and succeed after the synthetic checks.
Present but malformed or incomplete indexes fail. Shard paths are validated
as metadata targets; the files are never opened or required to exist.

Synthetic HF, HC and mixed indexes cover layers 0 through 42. Required roles
are embedding, final norm, head, per-layer input/post norms, router, and a
representative expert gate. Conventional attention requires Q and output
projections; HC topology requires both attention and FFN base keys. MTP names
are excluded, and canonical keys take precedence over aliases.
The optional local-index probe additionally resolves all three projections
and their scales for each of the 256 experts in all 43 layers. Embedding
fallback order is `model.embed_tokens.weight`, `embed.weight`, then
`embed_tokens.weight`. Missing indexes log `[ SKIP ] Local index file not found`.

This is a **key/topology contract**, not full inference compatibility: it does
not validate tensor shapes, dtypes, all expert projections, or numerical HC
execution. HC bases are never used as aliases for RMSNorm weights, nor are raw
attention projections substituted for the runner's required absorbed MLA
weights. Those conversion requirements remain enforced during ingestion.

The probe checks mock allocation counters (including symbolic and pinned
allocations) and exact header read counts. Ordinary host allocations for JSON
and key maps are expected; “0 MB” refers to tensor storage, not process RSS.
