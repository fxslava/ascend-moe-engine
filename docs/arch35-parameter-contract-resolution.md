# arch35 parameter contracts — 2026-10-09

The source audit found a runtime symbol collision and several differences
between the mock contracts and the vendored arch35 tilers. Device acceptance
still requires a 950PR with the matching custom OPP package.

## ABI resolution

CANN 9.2.0-beta.2 exports `aclnnCompressorGetWorkspaceSize` from `libopapi.so`
with a different prototype: it has no norm/RoPE parameters, and adds
`gradEnabled`, `softmaxScoreOut`, and `kvOut`. Previously `RTLD_DEFAULT`
selected that function before the engine's `libcust_opapi.so` implementation.
This is an ABI mismatch even if every tensor descriptor is otherwise valid.

The engine now resolves vendored operators explicitly from `libcust_opapi.so`
and rejects toolkit dependency symbols. An offline CANN loader probe confirmed
HcPre, Compressor, and VllmQuantLightningIndexer all resolve to the built
custom library. The hypotheses executable prints their providers before its
device-presence check.

## H5.1: HcPre

The existing descriptors and epsilon order already match both the wrapper and
`hc_pre_tiling_arch35.h`:

| Parameter | Contract |
|---|---|
| x | BF16 `[1,4,4096]`, strides `[16384,4096,1]` |
| hcFn | FP32 `[24,16384]` |
| hcScale / hcBase | FP32 `[3]` / `[24]` |
| attributes | `hcMult, hcSinkhornIters, hcEps, normEps` |
| y / post | BF16 `[1,4096]` / FP32 `[1,4]` |
| combFrag | FP32 `[1,4,4]`, strides `[16,4,1]` |

No rank change or epsilon swap is justified by these sources. Compile-time
signature checks already cover the HcPre function pointer. The test now starts
from initialized device storage, synchronizes replay before reporting success,
rejects nonfinite/negative Birkhoff values, and counts status 561103 as a
failure rather than an environmental skip. The specific on-device H5.1 refusal
is **not reproduced or proven fixed** here; a remaining refusal requires the
950PR runtime/OPP diagnostic log and provider/version comparison.

## H3: Compressor

`compressor_def.cpp` and `arch35/compressor_tiling.cpp` require, for the
engine's `coff=1`, single-sequence TH window:

| Parameter | Contract |
|---|---|
| x | BF16 `[ratio,4096]`, ratio 4 or 128 |
| wkv and wgate | BF16 `[512,4096]` each |
| stateCache | FP32 `[4,8,1024]`; last axis stores both KV and scores |
| norm / ape | FP32 `[512]` / `[ratio,512]` |
| cuSeqlens / seqused / startPos | INT32 `[0,ratio]` / `[ratio]` / `[window_start]` |
| sin and cos | FP32 `[2,64]` |
| cmpKvOut storage | BF16 `[2,512]`; one emitted row plus TH padding |
| rotaryMode / cacheMode | 1 (half) / 2 (cyclic) |

The tiler sizes TH rows as `min(T,T/ratio+B)`, which is two for one full
window and B=1. Therefore the proposed `[1,512]` output allocation and
`[4,8,512]` combined state are too small for this vendored implementation.
The pipeline exposes the first output row as `[1,512]` to downstream stages,
while reserving the padded storage required by the compressor.

HOLD steps perform only the existing device-to-device window copy. They
create no zero-row descriptors, plan, or launch. Full windows replay a
retained executor. RoPE is cast from the closing token's BF16 table row into
preallocated FP32 storage. Projection/state reservations and memory budgeting
were enlarged to match the tiler. Existing scalar-gate checkpoints must be
converted to the full `[512,4096]` projection before use.

H3 seeds valid metadata, replays two full windows, and checks the zero-input
compressed row on a real device. The symbolic backend checks the corrected
shapes and cadence; it cannot establish kernel numerics.

## H4: VllmQuantLightningIndexer

The tiler explicitly reads block size from key axis 1 and head count from
axis 2. Keep PA_BSND `[blocks,block_size,1,128]`; do not transpose it to
`[blocks,1,block_size,128]`. At block size 128 the contiguous strides are
`[16384,128,128,1]`. The dtype is `ACL_FLOAT8_E4M3FN`.

The failures found in H4 were elsewhere: arch35 requires **FP32 scoring
weights**, FP32 query/key scales, and non-null **INT32 `[1024]` scheduling
metadata**. The test formerly used BF16 weights and `[1]` metadata; the
pipeline supplied null metadata. Key lengths are expressed in original token
units: the kernel divides them by `cmpRatio` internally.

The pipeline now casts scoring weights into preallocated FP32 storage and
binds a separate original-length view. A static single-sequence schedule
assigns the half-open BN2 interval `[0,1)` to LI core 0, with other LI/LD cores
disabled. `QLIPreload::SplitCoreByAICPU` derives the terminal M/S2 blocks from
live lengths. This conservative schedule needs device validation and trades
parallelism for replayability; it is not a performance claim.

H4 initializes scales, original lengths, block tables, and scheduling data
before its first launch, replays after input rebinding, and rejects the old
BF16-weight and one-element-metadata descriptors. Existing pipeline unit-scale
quantization limitations remain documented in its stage report.

## Validation

Commands run in Ubuntu 22.04 WSL using the installed CANN 9.2 toolkit:

```sh
cmake -S . -B build_mock -DENABLE_MOCK_RUNTIME=ON -DCMAKE_BUILD_TYPE=Debug -DDSV4_WERROR=ON
cmake --build build_mock -j8
ctest --test-dir build_mock --output-on-failure
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build_cann -DENABLE_MOCK_RUNTIME=OFF -DCMAKE_BUILD_TYPE=Release -DDSV4_WERROR=ON
cmake --build build_cann -j8
./build_cann/tests/dsv4_operator_hypotheses_test --require-device
```

See `arch35-parameter-contract-validation.txt` for the final local results.
No device pass is inferred from compilation or symbolic tests.
