# Pre-launch verification — 2026-10-08

Host: x86_64, Ubuntu 22.04 under WSL. Both builds used Debug and
`-DDSV4_WERROR=ON`. No physical NPU was exercised.

## Mock suite

```sh
cmake -S . -B build_mock -DENABLE_MOCK_RUNTIME=ON -DCMAKE_BUILD_TYPE=Debug -DDSV4_WERROR=ON
cmake --build build_mock -j$(nproc)
ctest --test-dir build_mock --output-on-failure --output-log build_mock/preflight-ctest.log
```

```text
1/3 Test #1: mock_contract_smoke .............. Passed
2/3 Test #2: mock_pipeline_e2e ................ Passed
3/3 Test #3: mock_safetensors_index_probe ..... Passed
100% tests passed, 0 tests failed out of 3
```

The local metadata probe found 69,187 index keys, mapped layers 0..42,
resolved all 66,048 routed-expert projection/scale descriptors, and excluded
1,575 MTP keys. It passed 66,941 checks with 0 MB weight allocation and zero
payload reads. With `DSV4_MODEL_DIR` set to a missing directory, synthetic
checks passed and the probe emitted `[ SKIP ] Local index file not found`.
Mock linkage contains the engine, `libopapi_mock`, and system libraries only.

## Synthetic runner

```sh
./build_mock/apps/moe_runner \
  --config /mnt/c/models/DeepSeek-V4-Flash/config.json \
  --synthetic-weights --prompt-tokens 128 --max-new-tokens 1 \
  --diag-json ./test_diag.json --dry-run
```

The command returned 0. The JSON was parsed and its fields checked; a copy is
in [examples/inference_diagnostics_dry_run.json](examples/inference_diagnostics_dry_run.json).
The dry run reserves 2,752 KV blocks across 43 layers and executes no tokens.
Unmeasured TTFT, TPOT, entropy and perplexity are null.

A separate run without `--dry-run`, with 128 prompt tokens and two new tokens,
completed 129 steps and 33,282 expert requests: 33,126 hits, 156 promotions,
and 156 evictions. Active KV context was 129 tokens; utilization was
0.0157470703125. TTFT and TPOT were positive host simulation timings.
A one-new-token run also verified TPOT remains null and no unused extra
decode step is performed. These are symbolic contract checks, not numerical
model-quality or hardware-performance measurements.

## CANN container

Image: `ascend-c-dev:9.2.0-beta.2`, with the installed CANN 9.2.0-beta.2
toolkit mounted read-only at `/usr/local/Ascend`. This is the locally
available prerelease, not a claim of verification against a final 9.2.0 GA.

```sh
docker run --rm \
  -v /usr/local/Ascend:/usr/local/Ascend:ro \
  -v /mnt/c/ascend-moe-engine:/workspace -w /workspace \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/lib64:/usr/local/Ascend/cann-9.2.0-beta.2/x86_64-linux/devlib/linux/x86_64 \
  --entrypoint bash ascend-c-dev:9.2.0-beta.2 -lc '
    cmake -S . -B build_cann_preflight -DENABLE_MOCK_RUNTIME=OFF \
      -DCMAKE_BUILD_TYPE=Debug -DDSV4_WERROR=ON \
      -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2 &&
    cmake --build build_cann_preflight -j8 &&
    cmake --build build_cann_preflight --target dsv4_verify_linkage &&
    ctest --test-dir build_cann_preflight -V'
```

All CANN targets compiled without warnings. Linkage verification reported
zero undefined symbols and no Python/Torch dependencies. Each of the three
`npu_*` probes returned 0 with `[ SKIP ] Physical Ascend NPU not detected`;
CTest correctly recorded them as skipped. The driver stubs are used only to
make skip-path verification possible in this driverless container. Without
them, the loader reports missing `libascend_hal.so` before reaching `main`.

HC keys remain layout markers. Index success does not establish HC numerical
execution support or substitute for the runner's required MLA absorption pass.
