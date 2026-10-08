# Vendored ops-transformer operators (Ascend 950PR / arch35)

This directory vendors the four arch35 operators the DeepSeek-V4 Flash
manifold-hyper-connection (mHC) decode path depends on, out of the
experimental `ops-transformer` repository, so the engine no longer carries a
multi-repo binary dependency for them:

| Operator | ACLNN entry points | Engine | Source tree |
|---|---|---|---|
| `mhc_pre` | `aclnnMhcPreGetWorkspaceSize` / `aclnnMhcPre` | AIC+AIV | `mhc/mhc_pre/` |
| `mhc_sinkhorn` | `aclnnMhcSinkhornGetWorkspaceSize` / `aclnnMhcSinkhorn` | AIV | `mhc/mhc_sinkhorn/` |
| `mhc_post` | `aclnnMhcPostGetWorkspaceSize` / `aclnnMhcPost` | AIV | `mhc/mhc_post/` |
| `quant_lightning_indexer` | `aclnnQuantLightningIndexerGetWorkspaceSize` / `aclnnQuantLightningIndexer` | AIC+AIV | `attention/quant_lightning_indexer/` |

## Layout

* `mhc/`, `attention/` -- the operator trees, preserving their upstream
  `op_host/` (proto defs, infershape, tiling, and the `op_api/` aclnn layer)
  and `op_kernel/` (Ascend C device code, arch35 headers under
  `op_kernel/arch35/`) structure. Upstream unit tests, torch-extension
  examples and ONNX `op_graph/` protos were pruned at vendoring time.
* `common/` -- the shared host headers/tiling-base sources the operator trees
  include (`external/aclnn_kernels/*`, `external/aclnn_util.h`,
  `err/ops_err.h`, `tiling_base/*`, ...).
* `LICENSE` -- the upstream repository license (upstream files keep their
  per-file Huawei license headers).

## Build

See `CMakeLists.txt` here. In a CANN build the `cust_opapi` target compiles
the four `op_api` layers into `libcust_opapi.so` against the same toolkit the
engine uses; the engine link-loads it and `OpTable` resolves the eight entry
points from it. In a mock build (`-DENABLE_MOCK_RUNTIME=ON`) this directory
contributes nothing and the mock stubs in the engine's `mock/mock_ops.cpp`
provide the symbols inside `libopapi_mock.so`.

The `op_host` proto/tiling binaries and the arch35 Ascend C kernels deploy
through the ops-transformer opp/vendor package flow over these same sources
(or in-tree with `-DDSV4_VENDOR_ASCENDC_KERNELS=ON` on a toolkit whose
op-project templates ship `ascendc_library()`).

## Updating

Re-copy the trees from the upstream repository and keep the pruning policy
above (no tests, no examples, no torch_extension, no op_graph). The engine
contract constants that mirror these operators live in `mock/mock_ops.cpp`
and the C API surface is declared in `include/moe/ops/aclnn_dsv4_vendor_ops.h`.
