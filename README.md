# ascend-moe-engine

Standalone DeepSeek-V4 Flash decode engine for Ascend 950PR (ACLNN V5):
torch-free, python-free, no parent project.

Extracted from the vLLM Ascend pilot via `git subtree split -P csrc/standalone`;
this repository carries that subtree's full commit history (root-level here).

## Layout

```
apps/       executables (moe_runner)
include/    engine public headers (moe/...)
src/        engine library (static arena, exclusive staging, router engine,
            IRoutedMoeBlock + standard/lattice backends, config.json parsing)
mock/       symbolic libopapi_mock backend (zero-NPU contract validation)
tests/      every test binary: contract smoke, mock suite, device probes
python/     torch-free tokenizer + benchmark harness (tokenizers-only)
```

## Build

```bash
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
./build_cann/moe_runner --weights /mnt/c/models/DeepSeek-V4-Flash \
    --prompt-ids 7 --max-new-tokens 32 --stats-json stats.json
./build_cann/moe_runner --config /mnt/c/models/DeepSeek-V4-Flash/config.json --dry-run
```

`--stats-json` emits the machine-readable run record (TTFT, TPOT, swap
bytes, launch/residency counters) that `python/benchmarks` consumes.

## Tests (`tests/`)

| binary | what it proves | needs a device? |
| --- | --- | --- |
| `moe_contract_smoke` | slot layout, safetensors binding, hierarchy invariants, arena, model-config contract | no |
| `test_cann_backend_probe` | ACL device query (init / count / SoC / HBM) + the six required operator symbols via `dlsym` | no (device section SKIPs) |
| `test_weights_ingest_probe` | shard-00001 header parse + one real pinned→HBM DMA path | yes (`DSV4_MODEL_DIR`/`--model-dir`; SKIPs otherwise) |
| `test_e2e_single_token` | one full 43-layer decode step: zero deadlocks, zero stream errors, 0 in-step allocations | mock: no; real: `--coverage` |
| `dsv4_mock_test` | the full symbolic suite (mock build only) | no |

All probes exit 0 with `[ SKIP ]` when the capability they probe is genuinely
absent on the host (no weights dir, no NPU); they fail only when a present
capability misbehaves.

## Python harness (`python/`, zero torch)

```bash
pip install tokenizers                      # the only dependency (Rust-backed)

cd python
python -m tokenizer.dsv4_tokenizer "Hello DeepSeek"

# TTFT/TPOT + swap sweep across context lengths (engine as subprocess):
python -m benchmarks.bench_runner --engine ../build_cann/moe_runner \
    --context-lengths 128,512,2048 --new-tokens 32 \
    --weights /mnt/c/models/DeepSeek-V4-Flash --out results.json

# operator timeline via msprof, summarized on the CLI:
python -m benchmarks.profile_cann --engine ../build_cann/moe_runner \
    --engine-args "--synthetic-weights --prompt-ids 7 --max-new-tokens 4" \
    --output prof_out
```

## Model contract

The runner reads the checkpoint's `config.json` at startup and refuses to run
unless the router contract (`noaux_tc`, `sqrtsoftplus`, `norm_topk_prob`) and
the compiled graph geometry agree. Expert slots are 12.75 MiB FP4/E8M0
(hidden 4096, intermediate 2048); the exclusive RAM/VRAM hierarchy keeps
|Set_Device| + |Set_Host| = 11,008 experts with zero runtime disk reads.
