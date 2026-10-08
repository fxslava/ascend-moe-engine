# Vendored DeepSeek-V4-Flash operators (Ascend 950PR / arch35)

This directory consolidates the nine arch35 operators the DeepSeek-V4-Flash
decode path depends on -- the manifold-hyper-connection (mHC) residual chain
and the compressed-KV sparse-attention front end -- so the engine carries no
multi-repo binary dependency for them:

| Operator | ACLNN entry points | Engine | Source tree | Upstream |
|---|---|---|---|---|
| `mhc_pre` | `aclnnMhcPreGetWorkspaceSize` / `aclnnMhcPre` | AIC+AIV | `mhc/mhc_pre/` | ops-transformer |
| `mhc_sinkhorn` | `aclnnMhcSinkhornGetWorkspaceSize` / `aclnnMhcSinkhorn` | AIV | `mhc/mhc_sinkhorn/` | ops-transformer |
| `mhc_post` | `aclnnMhcPostGetWorkspaceSize` / `aclnnMhcPost` | AIV | `mhc/mhc_post/` | ops-transformer |
| `quant_lightning_indexer` | `aclnnQuantLightningIndexerGetWorkspaceSize` / `aclnnQuantLightningIndexer` | AIC+AIV | `attention/quant_lightning_indexer/` | ops-transformer |
| `compressor` | `aclnnCompressorGetWorkspaceSize` / `aclnnCompressor` | AIC+AIV | `attention/compressor/` | vllm-ascend |
| `vllm_quant_lightning_indexer` | `aclnnVllmQuantLightningIndexerGetWorkspaceSize` / `aclnnVllmQuantLightningIndexer` | AIC+AIV | `attention/vllm_quant_lightning_indexer/` | vllm-ascend |
| `kv_quant_sparse_attn_sharedkv` | `aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize` / `aclnnKvQuantSparseAttnSharedkv` | AIC+AIV | `attention/kv_quant_sparse_attn_sharedkv/` | vllm-ascend |
| `kv_compress_epilog` | `aclnnKvCompressEpilogGetWorkspaceSize` / `aclnnKvCompressEpilog` | AIV | `attention/kv_compress_epilog/` | vllm-ascend |
| `indexer_compress_epilog_v2` | `aclnnIndexerCompressEpilogV2GetWorkspaceSize` / `aclnnIndexerCompressEpilogV2` | AIV | `attention/indexer_compress_epilog_v2/` | vllm-ascend |

The two indexers are distinct operators, not duplicates: the vllm-ascend one
adds a scheduling-metadata input, an explicit key compression ratio, the
paged-key strides as attributes and an optional FP32 score output. Both are
registered.

## The hand-written op_api layer

The four ops-transformer trees arrive with their own `op_host/op_api/` layer.
The five vllm-ascend trees do **not**: upstream reaches them through torch-npu
`EXEC_NPU_CMD`, which dlsyms the `aclnn<Op>GetWorkspaceSize` / `aclnn<Op>`
pair that the CANN custom-op package generator emits from
`op_host/<op>_def.cpp`. This repository is torch-free, so their `op_api` layer
is written here by hand, transcribed from the operator definitions (input
order, attribute order and types, output order) and from the upstream call
sites, in the same shape as the ops-transformer ones.

Two consequences worth knowing before editing them:

* The l0op builders size their outputs from the **caller** descriptors rather
  than calling `INFER_SHAPE`. The proto and tiling registrations live in the
  deployed OPP vendor package, not in this library, and the engine sizes its
  own static arena slots anyway.
* Each `op_api/aclnn_<op>.cpp` keeps its file-scope unnamed namespace
  **outside** the `extern "C"` block. A function declared inside `extern "C"`
  gets C language linkage, which is external even in an unnamed namespace, so
  helpers named `CheckParams` or `StageContiguous` would otherwise collide at
  link time across the nine operators.

## Repeatability (operator-library manual 4.31)

A same-address `ViewCopy` makes an executor non-reusable, which this engine
cannot accept: the static op-slot table plans each vendored call once and
relaunches it per token with `aclSetTensorAddr`. Every wrapper here follows one
rule:

* Genuine outputs are written through a distinct executor-owned tensor, so the
  trailing `ViewCopy` always has `src != dst`.
* REF parameters -- `Compressor.stateCacheRef`,
  `KvCompressEpilog.kvCompressCacheRef`,
  `IndexerCompressEpilogV2.indexerCompressCacheRef` -- are the operator's input
  *and* its output, are written in place, and get no copy stage at all.
* `mhc_sinkhorn` is **patched** for this. Upstream copies unconditionally, and
  for a contiguous output `l0op::Contiguous` is the identity, so the kernel
  returns the caller tensor and the copy lands on its own address. The patched
  wrapper skips the copy when `kernelOut == output`; a non-contiguous output
  still takes it, where `Contiguous` allocated a distinct temp.

The mock suite holds this as an invariant: see `MockVendorSelfCopyHazards()`
(must stay 0), `MockSinkhornSelfCopyElisions()` and `MockRefOutputPlans()` in
`mock/mock_ops_api.hpp`, asserted by `tests/vendor_ops_table_test.cpp`.

## Layout

* `mhc/`, `attention/` -- the operator trees, preserving their upstream
  `op_host/` (proto defs, infershape, tiling, parameter checks, and the
  `op_api/` aclnn layer) and `op_kernel/` (Ascend C device code, arch35 headers
  under `op_kernel/arch35/`) structure. Upstream unit tests, torch-extension
  examples, ONNX `op_graph/` protos and per-op build files were pruned at
  vendoring time, as were the `arch32` (910B) kernel and tiling variants --
  this engine targets 950PR only.
* `common/` -- the shared host headers/tiling-base sources the operator trees
  include (`external/aclnn_kernels/*`, `external/aclnn_util.h`,
  `err/ops_err.h`, `tiling_base/*`, ...).
* `LICENSE` -- the upstream repository license (upstream files keep their
  per-file Huawei license headers).

## Build

See `CMakeLists.txt` here; `DSV4_VENDOR_OPS` is the single list the sources,
the per-op include roots and the kernel entry files all derive from, so adding
an operator is one line plus its `op_api/` pair. In a CANN build the
`cust_opapi` target compiles the nine `op_api` layers into `libcust_opapi.so`
against the same toolkit the engine uses (with `_GLIBCXX_USE_CXX11_ABI=0`, which
the toolkit's own C++ surface requires); the engine link-loads it and `OpTable`
resolves the eighteen entry points from it. In a mock build
(`-DENABLE_MOCK_RUNTIME=ON`) this directory contributes nothing and the mock
stubs in the engine's `mock/mock_ops.cpp` provide the symbols inside
`libopapi_mock.so`.

The `op_host` proto/tiling binaries and the arch35 Ascend C kernels deploy
through the opp/vendor package flow over these same sources (or in-tree with
`-DDSV4_VENDOR_ASCENDC_KERNELS=ON` on a toolkit whose op-project templates ship
`ascendc_library()`).

## Updating

Re-copy the trees from the upstream repositories and keep the pruning policy
above (no tests, no examples, no torch_extension, no op_graph, no arch32). For
the vllm-ascend operators, re-read `op_host/<op>_def.cpp` and re-check the
hand-written `op_api/` signatures against it -- a changed input, attribute or
output order is a silent ABI break. `src/core/aclnn_signature_check.cpp`
static_asserts every vendored header against the engine's transcribed typedefs
in `include/moe/core/op_table.hpp`, so a drift between those two fails the CANN
build instead of misreading registers at run time. The C API surface is
declared in `include/moe/ops/aclnn_dsv4_vendor_ops.h` and the engine contract
constants that mirror these operators live in `mock/mock_ops.cpp`.
