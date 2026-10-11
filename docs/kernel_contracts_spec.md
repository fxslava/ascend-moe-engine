# DeepSeek V4 Flash decode kernel contracts (arch35)

**Status:** proposed, correctness-first C++/Ascend C ABI. No kernel in this document is claimed to be implemented or device-validated. This is the contract to implement and test before replacing any existing graph stage. Scope is one token, one sequence, BF16 checkpoint activations, Ascend 950PR/arch35. The reference is the current `moe::emu` implementation and `data/dsv4_layer0/golden_semi_e2e_ref.bin` **v3**.

## Provenance and branch audit

`git branch -a` was inspected at `feat/host-cpu-emulation-layer` (`c82acaa`). Local branches are `main`, `feat/consolidate-dsv4-operators`, `feat/consolidate-dsv4-operators-kernels`, `feat/consolidate-dsv4-operators-kernels-pipesim`, `feat/consolidate-dsv4-operators-simt`, and the current branch; remote refs are `origin/main` and `origin/feat/consolidate-dsv4-operators`. These are local Git objects, not a claim about un-fetched PR refs. Relevant history was examined with `git log --all`, `git show`, and tree/diff inspection, without changing branches.

| Proposal / source | Declared interface and memory plan | Finding against the reference |
|---|---|---|
| `1758b65`, `4c55f7d`, `c6f6545`, `d0bebf5`, `35cbaf8`, `8bd5075`: `src/ops/mhc_fused_sinkhorn_gemv/` on the kernel and PipeSim branches | `x` BF16 `[1,4,4096]`, `phi` FP32 `[24,16384]`, `alpha` FP32 `[3]`, `bias` FP32 `[24]`, **input** `h_out` BF16 `[1,4096]`, `y` BF16 `[1,4,4096]`. Early draft used a 1-core vector path with zero GM workspace, UB queues and scalar Sinkhorn. `4c55f7d` added a 64-byte tiling plan, 512-element phi/feature tiles, 2048-element input tiles, queue depth 2 and a 128 KiB UB budget. `c6f6545` changed to 4096-element phi chunks, skewed UB banks and explicit MTE2/V/S/MTE3 events; `h_in` remained unmaterialized. `35cbaf8`/PipeSim explored 16 AIVs, 128-byte partial slots (2048-byte GM workspace) and `SyncAll<true>` with a hard 16-block launch. | The whole-launch API cannot represent a decoder round: attention/FFN must run **between** pre and post on `h_in`. The inspected `c6f6545` header directly scales all 24 projection rows by alpha and bias, uses raw first four rows as an unused `h_in`, uses un-sigmoided post rows, and normalizes the residual matrix without the reference's initial-softmax `+eps` cadence. Thus its arithmetic is not the checkpoint mHC formula. Its zero workspace/UB or 16-core/2048-byte workspace figures belong to those **old** kernels, not to the APIs below. The 16-core draft's synchronization is valid only with every block participating and a real shared-workspace lifetime barrier. |
| `cba014a`, `8f7e694`, `7eb193f`, `001abe3`, `0ad1bbe`, `fd9604d`: `third_party/ops_dsv4/mhc/` | Vendored `mhc_pre` → `mhc_sinkhorn` → `mhc_post`, plus `hc_pre`, `hc_pre_inv_rms`, `hc_pre_sinkhorn`, `hc_post`. `aclnnHcPre` fuses bundle inv-RMS, projection and Sinkhorn and outputs `y/h_in`, `post`, `combFrag/B`; `aclnnHcPost` consumes sub-block output and residual in a BSHD view. The staged `mhc_sinkhorn` works on a GM matrix and has a wrapper REF/copy issue documented in `001abe3`; `hc_pre` keeps its 4×4 Sinkhorn in registers. Host tilers and opapi wrappers return their own workspace sizes. | `hc_pre`/`hc_post` is the closest usable split. Its `y` must pass through **attn_norm/ffn_norm after** pre and before the function; the bundle inv-RMS in `hc_pre` is a different normalization. `hc_post` uses unit sequence axes over the same bytes, so the view/stride mapping must be explicit. `7eb193f` selected this two-launch path; it did not fuse pre and post into one launch. The `mhc_sinkhorn` repeatability and same-address copy concerns are wrapper-specific, not a reason to conflate the two math stages. |
| `cba014a`, `001abe3`, `fd9604d`: `third_party/ops_dsv4/attention/` | Six vendored trees: `compressor`, `quant_lightning_indexer`, `vllm_quant_lightning_indexer`, `kv_quant_sparse_attn_sharedkv`, `kv_compress_epilog`, `indexer_compress_epilog_v2`. The compressor OpDef accepts `x,wkv,wgate,state_cache,ape,norm_weight,rope_sin,rope_cos`, optional block/sequence metadata, and emits `cmp_kv` while updating state in place. Its arch35 tiler carries core splits, `cmpRatio`, `coff`, page dimensions, double-buffered matmul/vec workspaces and an explicit `vec1TailCacheSize` when `coff==2`. The two indexers accept FP8 query/key, weights and separate dequant scales, paged metadata and top-k count. Epi operators quantize/scatter; sparse attention consumes the two cache streams. | `compressor_def.cpp` defaults `coff=1`; DSV4 weights require **2**. The vendored compressor workspace is platform/tile dependent and includes a library reserve; do not substitute the proposed zero-GM-workspace direct kernel below for its `GetWorkspaceSize` return. The 604-byte **packed cache entry** discussed in `001abe3` is the KV epilog output, not the compressor's 1024-byte BF16 token. The indexer tilers' metadata/stride inputs are real ABI constraints; a merged `dequant_scales` buffer below is a proposed direct-kernel adapter, not their aclnn signature. |
| `faee654`, `c82acaa`: `src/emu/`, `include/moe/emu/` | Split `mhc_pre_sinkhorn_cpu` / `mhc_post_cpu`, `paged_compressor_decode_c4a`, `fast_fp8_indexer_topk_cpu`, and MXFP4 GEMV. The semi-E2E writer emits v3. | This is the numerical oracle. The semi-E2E path is **layer 0 attention → layer 2 attention and side-path compressor/indexer → layer 1 MoE**, not a contiguous model layer 0→1→2 forward. Layer 1 attention is omitted. The indexer's 2048 historical FP8 keys are synthetic; its output is not in v3. |

The existing 13 vendored operator trees and their declared GM-level boundaries are inventoried below. The exact exported aclnn parameter types/order are in `include/moe/ops/aclnn_dsv4_vendor_ops.h` and the corresponding `third_party/ops_dsv4/**/op_host/*_def.cpp`. Their host tilers own tiling keys, core counts and workspace returns; the direct PODs specified later in this document do **not** overwrite those vendor ABIs.

| Existing tree | Declared input → output boundary | Existing tiling / residency / synchronization / workspace |
|---|---|---|
| `mhc/mhc_pre` | `x,phi,alpha,bias,gamma?` → `hIn,hPost,hRes` and optional inverse-RMS/mix/pre traces | GM tensors with UB/vector staging; per-op arch35 tiler and optional outputs. Returns its aclnn workspace; `hRes` is still raw for the separate Sinkhorn call. |
| `mhc/mhc_sinkhorn` | raw FP32 `[4,4]` → normalized FP32 `[4,4]` | A separate launch over GM, with per-op tiler; wrapper in-place/REF and repeatability limitations are recorded in `001abe3`/`0ad1bbe`. |
| `mhc/mhc_post` | `x,hRes,hOut,hPost` → residual streams | GM inputs/output with arch35 post tiling data and UB staging; executes after the function output exists. |
| `mhc/hc_pre` | `x,hc_fn,hc_scale,hc_base` → `y,post,comb_frag` | Fused prologue with register-resident Sinkhorn, UB vector staging and, on split paths, AIC/AIV cross-core flags/`SyncAll`; its tiler supplies workspace and launch geometry. It does not take activation norm weights. |
| `mhc/hc_pre_inv_rms` | residual streams → FP32 `rsqrt` | Separate GM output feeding the staged path; host tiler/workspace returned by aclnn wrapper. |
| `mhc/hc_pre_sinkhorn` | `mixes,rsqrt,hc_scale,hc_base,x` → `y,post,comb_frag` | Fuses mixing epilogue and register Sinkhorn but requires the preceding inv-RMS and mixing GEMM launches; host tiler/workspace returned by aclnn wrapper. |
| `mhc/hc_post` | function `x`, residual, post, comb → `y` | BSHD view over the same token bytes; GM boundary, UB/vector combine and separate launch after the sub-block. |
| `attention/compressor` | `x,wkv,wgate,state_cache,ape,norm_weight,rope_sin,rope_cos`, optional page/sequence metadata → `cmp_kv`, updated state | GM state is REF; arch35 `CompressorTilingData` records page shape, `cmpRatio`, `coff`, split-core ranges and four workspace-size components. It uses AIC/AIV and UB/L1/L0 private buffers; tiler requests double-buffered matmul/vec workspace plus library reserve. `vec1TailCacheSize=ratio*D` only for `coff=2`. |
| `attention/quant_lightning_indexer` | FP8 query/key, weights, query/key scales, lengths/page table → INT32 sparse indices | Separate GM top-k output; arch35 tiler chooses core/sort geometry and workspace. A row/key layout and sparse-count attribute are part of its aclnn ABI. |
| `attention/vllm_quant_lightning_indexer` | same plus metadata, compression ratio and explicit key/scale strides → indices and optional scores | GM FP8 page cache and INT32/FP32 outputs; vendor tiler and mixed Cube/Vector stages own UB/L1/L0 scheduling and workspace. The `stride` and `scaleStride` parameters must match bound descriptors. |
| `attention/kv_quant_sparse_attn_sharedkv` | query, original/compressed KV, both sparse-index sets and block tables, sequence metadata and sinks → attention output, optional log-sum-exp | GM hybrid-cache interface with distinct original/compressed cache strides; arch35 tile size, masks and core/UB/Cube scheduling belong to the vendor tiler. This is downstream of KER-04 and does not exist in the semi-E2E golden output path. |
| `attention/kv_compress_epilog` | compressed BF16 row, slot mapping, REF KV cache → quantized paged cache | Quantization/scatter launch with `quant_group_size`, `layout`, `block_stride`; GM cache mutation, private UB packing and vendor workspace. The 604-byte entry is this output, not compressor `cmp_kv`. |
| `attention/indexer_compress_epilog_v2` | compressed BF16 index row, slot mapping, REF index cache → FP8 paged cache | GM cache mutation with `layout`/`block_stride`, private UB packing and vendor workspace; its address geometry must match the indexer key view. |

`third_party/ops_transformer/attention/quant_lightning_indexer` and its earlier `mhc_*` copies are ancestors of the vendored tree; they are not additional independent fusion proposals. `feat/consolidate-dsv4-operators-simt` adds a portable Sinkhorn CPU fallback rather than a new device math contract. No MXFP4 Ascend C expert GEMV implementation was found in these branch trees; KER-05 below is a required new primitive contract.

## Universal ABI and numerical rules

The following are **new proposed direct-kernel signatures** in `namespace moe::kernels::arch35`, not current exported symbols and not replacements for the vendored `aclnn...GetWorkspaceSize` signatures. All pointer arguments address **device global memory (GM)**, even if the implementation stages values in unified buffer (UB). `aclrtStream` denotes the ordinary CANN stream type. A non-null `workspace` is required only when `workspace_bytes > 0`; no hidden cross-launch UB state is allowed. The host owns allocation, lifetime and stream ordering. Inputs and outputs must be contiguous except for the explicitly named strides. Strides are in **elements of that pointer's type**, never bytes. Values are little-endian in the golden fixture. A launch validates exact dimensions, stride minima, output capacity, page-table bounds and workspace size before enqueueing; no silently clipped shape is permitted.

```cpp
#include <cstddef>
#include <cstdint>
#include <acl/acl.h>
namespace moe::kernels::arch35 {
using Bf16 = std::uint16_t;       // storage bits, round-to-nearest-even on store
using Fp8E4M3 = std::uint8_t;    // storage bits
using Mxfp4Pair = std::uint8_t;  // low nibble = even logical column
using E8M0 = std::uint8_t;       // one exponent scale code
enum class Status : std::uint32_t { success, invalid_argument, unsupported_shape, launch_error };
}
```

Fixed token geometry: mHC streams `N=4`, hidden `H=4096`, projection rows `N*N+2*N=24`, Sinkhorn iterations `20`, `hc_eps=1e-6`, bundle and activation RMS epsilon `1e-6`; compressor `D=512`, `coff=2`, `ratio=4`, RoPE width `64`; indexer `64×128`, top-k `512`. FP32 is used for GEMV accumulation, Sinkhorn, gate arithmetic, softmax and top-k scores. No L1/L0A/L0B/L0C residency is part of these proposed vector-kernel interfaces; an implementation using Cube/L1/L0 must preserve the GM-visible contract and document its private tiling separately. GM outputs are distinct from all GM inputs unless an operator explicitly says otherwise. The host must sequence the two mHC launches and the sub-block on the **same stream** or establish equivalent events; a launch boundary is the producer/consumer ordering point. No implicit cross-core `SyncAll` is required in the single-core plans below.

Every `TilingData` definition here is a versioned, fixed-width, standard-layout host/device POD copied into the device tiling buffer; it contains **no pointers**. Pointer addresses are kernel `GM_ADDR` arguments. All `reserved` fields are zero; reject unknown `abi_version`. The numerical order, not a particular private UB layout, is normative. The UB formulas are explicit correctness-first implementation budgets; a future tuned kernel may replace them only after documenting the new bound and passing the same fixture.

## KER-01 — `fused_mhc_pre_norm_sinkhorn`

**Dependency and math.** `x` is four residual streams, not a pre-normalized vector. Compute `r=rsqrt(mean(flatten(x)^2)+1e-6)`, `p=phi @ flatten(x)`, `u=p*r`, `pre[i]=sigmoid(u[i]*alpha[0]+bias[i])+1e-6`, `post[i]=2*sigmoid(u[4+i]*alpha[1]+bias[4+i])`, `logits[i,j]=u[8+4*i+j]*alpha[2]+bias[8+4*i+j]`, and `h_raw[d]=sum_i pre[i]*x[i,d]`. For `B`, do row softmax of logits with a separate maximum for each row, add `hc_eps` **once** to each softmax entry, perform the first column normalization by `(sum+hc_eps)`, then 19 more row/column pairs each with denominator `(sum+hc_eps)`; see `sinkhorn_knopp_4x4`. Finally `h_norm[d]=BF16(gamma[d]*h_raw[d]*rsqrt(mean(h_raw^2)+1e-6))`. This fuses the **second** RMSNorm with pre, after `h_raw`; it does not move it ahead of the four-stream reduction. Attention/FFN consumes `h_norm` and produces `h_out` for KER-02.

```cpp
Status LaunchMhcPreNormSinkhorn(
    const Bf16* x_gm, std::uint32_t x_stream_stride,       // 4096
    const float* phi_gm, std::uint32_t phi_row_stride,     // 16384
    const float* alpha_gm, const float* bias_gm,
    const Bf16* gamma_gm,                                 // attn_norm or ffn_norm
    float* h_raw_gm, Bf16* h_norm_gm, float* post_gm,
    float* birkhoff_gm,                                   // row-major B[i,j]
    void* workspace, std::size_t workspace_bytes, aclrtStream stream);
struct MhcPreNormTilingData {
    std::uint32_t abi_version;          // 1
    std::uint32_t block_dim;            // 1 AIV
    std::uint32_t hidden;               // 4096
    std::uint32_t streams;              // 4
    std::uint32_t projection_rows;      // 24
    std::uint32_t projection_tile;      // 4096 FP32 elements
    std::uint32_t projection_chunks;    // 4 per row
    std::uint32_t queue_depth;          // 2
    std::uint32_t sinkhorn_iterations;  // 20
    std::uint32_t x_stream_stride;      // 4096 BF16 elements
    std::uint32_t phi_row_stride;       // 16384 FP32 elements
    std::uint32_t reserved;             // 0
    float hc_eps;                       // 1e-6
    float bundle_norm_eps;              // 1e-6
    float activation_norm_eps;          // 1e-6
    float reserved_fp32;                // 0
};
```

GM byte extents: `x` at least `(3*x_stream_stride+4096)*2` (32,768 at contract stride), `phi` at least `(23*phi_row_stride+16384)*4` (1,572,864), `alpha` 12, `bias` 96, `gamma` 8,192, `h_raw` 16,384, `h_norm` 8,192, `post` 16, `B` 64. `h_raw` is a diagnostic/FP32 oracle output; `h_norm` is the BF16 device-graph input. All outputs reside in GM and are non-aliasing. **GM scratch workspace = 0 bytes.** A single AIV holds a BF16 input tile (8,192 B), widened `x` (65,536), two FP32 phi tiles (2×16,384), product/accumulator/`h_raw`/norm FP32 vectors (4×16,384), a BF16 norm output tile (8,192), reduction scratch (2,048), and 128+64 B padded small state: `UB ≤ 182,464 B`. Tile sizes and queue depth are part of v1; reject a device whose available UB is smaller. MTE2→V and V→MTE3 queue events must be paired; scalar reads of vector reductions need V→S completion and scalar-produced factors need S→V completion. No core barrier or GM partials.

Golden: raw `h_in0` at byte **131296** (layer 0 attention), `h_in2` at **147680**, `moe_h_in` at **164064**, each FP32 `[4096]`; `B0` at **32816**, `B2` at **32880**, `Bmoe` at **180448**, each FP32 `[4,4]`. `h_norm` and `post` are **not serialized**: derive expected `h_norm` from golden `h_in` and the corresponding checkpoint norm weight, and use `mhc_pre_sinkhorn_cpu` for `post`. The source stream for layer 0 is `x_input` at 32992; for layer 2 it is `x_after_layer0` at 65760; for the layer 1 MoE round it is `x_after_layer2` at 98528. This mapping follows the semi-E2E execution order stated above.

## KER-02 — `fused_mhc_post_residual`

**Math.** After the attention/FFN function returns its **function output alone**, `acc=post[j]*BF16_to_FP32(h_out[d]); for i=0..3: acc+=B[i,j]*BF16_to_FP32(x[i,d]); y[j,d]=BF16(acc)`. `B` is row-major and is transposed by its indexing, not by physically transposing memory. There is no extra residual add. `h_out` is unavailable at KER-01 launch time, which is why this is a separate launch.

```cpp
Status LaunchMhcPostResidual(
    const Bf16* x_gm, std::uint32_t x_stream_stride,       // 4096
    const float* birkhoff_gm, std::uint32_t b_row_stride,  // 4
    const float* post_gm, const Bf16* h_out_gm,
    Bf16* y_gm, std::uint32_t y_stream_stride,             // 4096
    void* workspace, std::size_t workspace_bytes, aclrtStream stream);
struct MhcPostTilingData {
    std::uint32_t abi_version;       // 1
    std::uint32_t block_dim;         // 4 AIV, one output stream per block
    std::uint32_t hidden;            // 4096
    std::uint32_t streams;           // 4
    std::uint32_t feature_tile;      // 512 elements, 8 tiles per stream
    std::uint32_t x_stream_stride;   // 4096
    std::uint32_t y_stream_stride;   // 4096
    std::uint32_t b_row_stride;      // 4
};
```

GM extents: `x` 32,768 B, `B` 64 B, `post` 16 B, `h_out` 8,192 B, `y` 32,768 B at contract strides. All five pointers are distinct; **in-place `x==y` is forbidden** because parallel output streams can overwrite another block's source. GM scratch = **0 B**. Per-core UB v1 budget: four widened 512-lane source tiles (8,192 B), widened `h_out` tile (2,048), FP32 accumulator (2,048), BF16 output tile (1,024), padded coefficients (128): **13,440 B**. Four blocks have disjoint `y` rows; no inter-core barrier. MTE2 must complete before vector reads and V→MTE3 before each store/UB reuse. The previous KER-01 output and `h_out` must be complete on the stream.

Golden: the layer 0 round's `y` is `x_after_layer0` at **65760**; the layer 2 round's is `x_after_layer2` at **98528**; the layer 1 MoE round's `y` is `x_final` at **48**. All are BF16 `[4,4096]`. v3 does not serialize standalone `h_out` or `post`, so isolated KER-02 testing must reconstruct them with the C++/PyTorch stage math or append a separate input fixture. The comparator is cosine ≥0.9995 plus reported relative L2; no acceptance from cosine alone for an all-zero output.

## KER-03 — `fused_paged_compressor_rope` (C4A)

**Math and state.** Each call projects `x` through `wkv,wgate` (`[1024,4096]` BF16), adds `ape[step%4,:]` to the gate projection and stores two FP32 rows in persistent GM state. Steps `step%4 = 0,1,2` are **HOLD**: never write `cache_slot_gm`. At `step%4=3`, form an `[8,512]` window: previous group's four **a** halves then current group's four **b** halves. Before the first group, prior a scores are `-inf` and prior a values zero. Column-softmax the eight scores, weight the eight KV rows, sum to 512, apply weighted RMSNorm, and half-mode RoPE to lanes `[448,512)` using tables for compressed position `step/4` and `compress_rope_theta=160000`. Then store one BF16 `[512]` token, set `emitted=1`, and copy this group's a halves into previous state. This is exactly the `coff=2` branch of `paged_compressor_decode_c4a`; a four-row-only window is wrong after the first emission.

```cpp
Status LaunchPagedCompressorRope(
    const Bf16* x_gm, const Bf16* wkv_gm, std::uint32_t wkv_row_stride,
    const Bf16* wgate_gm, std::uint32_t wgate_row_stride,
    const float* ape_gm, std::uint32_t ape_row_stride,
    const Bf16* norm_weight_gm,
    const float* rope_cos_gm, const float* rope_sin_gm,
    std::uint32_t step,
    float* state_gm, std::uint32_t* state_meta_gm,
    Bf16* cache_slot_gm, std::uint32_t* emitted_gm,
    void* workspace, std::size_t workspace_bytes, aclrtStream stream);
struct CompressorTilingData {
    std::uint32_t abi_version;       // 1
    std::uint32_t block_dim;         // 1 AIV
    std::uint32_t hidden;            // 4096
    std::uint32_t head_dim;          // 512
    std::uint32_t coff;              // 2, reject 1
    std::uint32_t ratio;             // 4
    std::uint32_t reduction_rows;    // 8
    std::uint32_t rope_head_dim;     // 64
    std::uint32_t rotary_mode;       // 1 = half, not interleave
    std::uint32_t projection_row_tile; // 1 row at a time
    std::uint32_t wkv_row_stride;    // 4096 BF16 elements
    std::uint32_t wgate_row_stride;  // 4096 BF16 elements
    std::uint32_t ape_row_stride;    // 1024 FP32 elements
    std::uint32_t reserved;          // 0
    float norm_eps;                  // 1e-6
    float reserved_fp32;             // 0
};
```

The **state GM ABI** is `state_gm[12288]` FP32 (49,152 B): offsets `[0,4096)` current KV `[4,1024]`; `[4096,8192)` current gate `[4,1024]`; `[8192,10240)` previous a KV `[4,512]`; `[10240,12288)` previous a gate `[4,512]`. `state_meta_gm[4]` (16 B): `[0]=have_previous` (0/1), `[1]=tokens_seen`, `[2:4]=0`. Initialize both allocations once, with meta zero; call only with `step==tokens_seen` on the same stream for a given state. Different sequences get disjoint state. The output `cache_slot_gm` is the caller-resolved PA_BSND slot `[512]` BF16 (1,024 B); its address must be valid even on HOLD, but HOLD leaves all its bytes untouched. `emitted_gm[0]` is written 0 or 1 every call. For paged caches the caller resolves `physical_block=block_table[compressed_index/block_size]` and `slot=physical_block*block_size+compressed_index%block_size`; the kernel does not infer page geometry. The 604-byte quantized epilog entry is outside this API.

GM input extents at contract strides: `x` 8,192 B, each weight 8,388,608 B, `ape` 16,384 B, norm weight 1,024 B, each RoPE table 256 B. All mutable buffers are non-aliasing except state is intentionally read/write. **GM scratch = 0 B**; persistent state 49,168 B is an input/output allocation, not scratch. A correctness-first single-core kernel streams one BF16 4096-element weight row at a time against a resident FP32 `x`; the larger emit phase needs at most two `[8,512]` FP32 matrices plus a same-size softmax matrix, `x` and two streamed BF16 rows, and a `[512]` FP32 output. UB upper bound `3*(8*512*4) + 4096*4 + 2*(4096*2) + 512*4 = **83,968 B**` (plus 1 KiB for padded scalars/events: **84,992 B**). No Cube/L1/L0 or cross-core barrier. Sequential steps and MTE2/V/MTE3 events must complete before `state_meta[1]` is advanced.

Golden: `layer2_compressed_tokens` BF16 `[1,512]` at byte **311584** (1,024 B). It is the first emission after four synthetic decode tokens based on layer 2 normalized `h_in`; v3 does not store the four token inputs or the state after each HOLD. Build them as `src/emu/semi_e2e_pipeline.cpp` does and compare the emitted row. A separate two-group fixture is required to certify the cross-boundary previous-a path. **Vendor caveat:** `compressor_def.cpp` defaults `coff=1` and its public README lists a restricted rotary-mode set; the proposed half-mode/`coff=2` direct ABI follows the checkpoint C++ emulation. Do not claim that merely passing these values to the vendored wrapper is supported until its arch35 tiler/kernel and device probe accept them.

## KER-04 — `fused_fp8_indexer_topk`

**Math.** For logical key `j`, resolve `physical_block=block_table[j/block_size]`, read FP8 row `(physical_block*block_size+j%block_size)*128`, then `score[j]=key_scale[j]*sum_{h=0}^{63} head_scale[h]*max(dot(q[h,:],key[j,:]),0)`. `head_scale` already folds query dequant and per-head gate; the semi-E2E harness takes the absolute projected gate before folding. Convert each FP32 score to BF16 **before** selection. Sort by score descending, then logical index ascending on exact BF16 ties. Fill all unused `[seq_len,512)` output lanes with `-1`. ReLU is per-head, before the weighted head sum.

```cpp
Status LaunchFp8IndexerTopK(
    const Fp8E4M3* query_gm, std::uint32_t query_head_stride, // 128
    const Fp8E4M3* key_cache_gm, std::uint32_t key_block_stride,
    const float* dequant_scales_gm,                         // [64 | seq_len]
    const std::int32_t* block_table_gm, std::uint32_t block_table_len,
    std::uint32_t num_physical_blocks, std::uint32_t block_size,
    std::uint32_t seq_len, std::int32_t* indices_gm,
    void* workspace, std::size_t workspace_bytes, aclrtStream stream);
struct IndexerTilingData {
    std::uint32_t abi_version;       // 1
    std::uint32_t block_dim;         // 1 AIV, deterministic total order
    std::uint32_t heads;             // 64
    std::uint32_t head_dim;          // 128
    std::uint32_t top_k;             // 512
    std::uint32_t query_head_stride; // 128 FP8 elements
    std::uint32_t key_block_stride;  // block_size*128 FP8 elements
    std::uint32_t block_size;        // 128 in v3 semi-E2E
    std::uint32_t seq_len;           // 2048 in v3 semi-E2E
    std::uint32_t block_table_len;   // >= ceil(seq_len/block_size)
    std::uint32_t num_physical_blocks;
    std::uint32_t score_tile;        // 512 FP32 elements
};
```

GM extents at contract strides: query 8,192 B; key cache `num_physical_blocks*key_block_stride` B; fused scales `(64+seq_len)*4` B; table `block_table_len*4` B; output `512*4=2,048` B. Every table entry used must be in `[0,num_physical_blocks)`, and `key_block_stride>=block_size*128`. `seq_len` may be 0; `top_k` remains 512. The one-core v1 writes FP32 scores to GM scratch, then reads/rounds them for deterministic selection: **workspace = align_up(4*seq_len,32) B**, 8,192 B for `seq_len=2048`. This workspace must not alias inputs or output. UB cap: FP8 query 8,192 + widened query 32,768 + FP8 key tile 256 padded + widened key 512 + head partials 256 + score scan 2,048 + scalar 32 = **44,064 B**. One core means no cross-core top-k merge or barrier; all scores must be complete before selection begins.

Golden: **no indexer scores or indices are serialized in v3**. The header's `indexer_top_k=512` is metadata, not an output tensor. The harness uses real `weights_proj` but synthetic FP8 query and 2048 historical key rows; accept KER-04 only against a separate fixture containing query/key bytes, scales, page table, and CPU-ranked indices. `x_final` is not evidence for indexer correctness because this side path does not feed the semi-E2E attention output.

## KER-05 — MXFP4 expert GEMV prerequisite

No preceding branch supplies this direct kernel. It must be fixed before fusing routed-expert SwiGLU. `w1,w3` are logical `[2048,4096]`, packed `[2048,2048]` bytes and scales `[2048,128]`; `w2` is logical `[4096,2048]`, packed `[4096,1024]` bytes and scales `[4096,64]`. One scale covers **32 logical input values per row**. `packed[row*packed_row_stride+col/2]` holds **logical even `col` in bits 0..3** and odd `col+1` in bits 4..7; decode OCP E2M1 and multiply by the E8M0 scale at `scale[row*scale_row_stride+col/32]`. Require `packed_row_stride=cols/2`, `scale_row_stride=cols/32` (128 or 64, both 32-byte aligned) for v1; never infer a shared FP8-style 128×128 scale grid. The authoritative converter is `inference/convert.py`, and `scripts/verify_mxfp4_packing.py` demonstrates why an all-ones vector alone cannot distinguish a within-byte swap.

```cpp
Status LaunchMxfp4ExpertGemv(
    const Mxfp4Pair* packed_gm, std::uint32_t packed_row_stride,
    const E8M0* scale_gm, std::uint32_t scale_row_stride,
    const float* x_gm, std::uint32_t rows, std::uint32_t cols,
    float* y_gm, void* workspace, std::size_t workspace_bytes,
    aclrtStream stream);
struct Mxfp4GemvTilingData {
    std::uint32_t abi_version;       // 1
    std::uint32_t block_dim;         // 32 AIVs; rows assigned in 32-row tiles
    std::uint32_t rows;              // 2048 or 4096
    std::uint32_t cols;              // 4096 or 2048
    std::uint32_t rows_per_block;    // 32
    std::uint32_t group_size;        // 32
    std::uint32_t packed_row_stride; // cols/2 bytes
    std::uint32_t scale_row_stride;  // cols/32 bytes
    std::uint32_t low_nibble_first;  // 1; reject 0
    std::uint32_t reserved;          // 0
};
```

Inputs/outputs are GM, with FP32 `x[cols]` and `y[rows]`; GM scratch **0 B**. Per core, process one row at a time with at most `cols*4 + cols/2 + cols/32 + 32*4` UB bytes: **18,688 B** at `cols=4096`; pad to **19,456 B** for 32-byte transfers and local scale/decode scratch. Each core owns disjoint 32-row tiles with a grid-stride loop, so no inter-core synchronization or L1/L0 contract. Full expert math is `w2 @ (silu(clamp(w1 @ x, max=10))*clamp(w3 @ x, -10, 10))` with FP32 intermediate accumulations and routing weights as implemented in `src/emu/moe_block.cpp`; expert 30 is not a substitute for all six active experts. Golden v3 stores **six complete expert outputs**, FP32 `[6,4096]`, in routing order `[30,205,192,177,242,102]` at **180512**. It does **not** store isolated `w1/w3/w2` GEMV outputs, so KER-05 alone requires a separate per-GEMV fixture; its integrated acceptance uses those six expert outputs, routed sum at **278816**, shared sum at **295200**, and final stream at **48**.

## Golden v3 byte directory and acceptance

The proposed tiling buffer ABI is fixed-size: `sizeof(MhcPreNormTilingData)==64`, `sizeof(MhcPostTilingData)==32`, `sizeof(CompressorTilingData)==64`, `sizeof(IndexerTilingData)==48`, and `sizeof(Mxfp4GemvTilingData)==40`. Host and device translation units must compile matching `static_assert`s for these sizes and for standard-layout/trivial-copyability. All fields are 32-bit, so no host/device pointer-width padding is permitted.

The writer is `apps/dsv4_semi_e2e_bench.cpp` and the file was checked to be exactly **312,608 bytes**. Header is 48 bytes: ASCII `DSV4SE2E`, u32 version **3**, u32 BF16 code **1**, i32 streams **4**, i32 hidden **4096**, i64 token ID **3**, i32 routed count **6**, i32 Sinkhorn n **4**, i32 emitted compressed tokens **1**, i32 indexer top-k **512**. It is native little-endian C++ layout; this platform's header fields have the stated 48-byte layout. Reject a file with any other version, length or geometry.

| Offset (bytes) | Length | Tensor / type / shape |
|---:|---:|---|
| 48 | 32768 | `x_final` BF16 `[4,4096]` |
| 32816 | 64 | layer-0 `B` FP32 `[4,4]` |
| 32880 | 64 | layer-2 `B` FP32 `[4,4]` |
| 32944 | 24 | routed expert IDs INT32 `[6]` |
| 32968 | 24 | routing weights FP32 `[6]` |
| 32992 | 32768 | `x_input` BF16 `[4,4096]` |
| 65760 | 32768 | `x_after_layer0` BF16 `[4,4096]` |
| 98528 | 32768 | `x_after_layer2` BF16 `[4,4096]` |
| 131296 | 16384 | layer-0 raw `h_in` FP32 `[4096]` |
| 147680 | 16384 | layer-2 raw `h_in` FP32 `[4096]` |
| 164064 | 16384 | layer-1 MoE raw `h_in` FP32 `[4096]` |
| 180448 | 64 | layer-1 MoE `B` FP32 `[4,4]` |
| 180512 | 98304 | six expert outputs FP32 `[6,4096]` |
| 278816 | 16384 | routed sum FP32 `[4096]` |
| 295200 | 16384 | shared sum FP32 `[4096]` |
| 311584 | 1024 | layer-2 compressed token BF16 `[1,512]` |

The `moe_h_in` row at 164064 is **16,384 bytes**; the following offset 180448 is authoritative. Expert rows follow the serialized IDs, not numerical expert order. The file is a C++ emulation fixture, independently cross-checked by the standalone PyTorch reference at ≥0.9995 cosine for its serialized activation comparisons; it is not a device output. For each applicable kernel compare shape, dtype, finite values, cosine and relative L2, plus structural invariants: `B` row/column sums, HOLD slot untouched, first and second compressor emissions, top-k tie order, and FP4 nibble order. Use `scripts/run_pytorch_layer_ref.py` for the independent layer comparisons. A missing v3 field is explicitly **not** a passing golden check; add a supplementary fixture before claiming isolated acceptance of that output.

## Implementation gate

1. Implement KER-01 and KER-02 as separate launches. Verify the exact mHC gate transforms, two distinct RMSNorm locations, Sinkhorn epsilon cadence, and `B[i,j]` orientation before tuning occupancy.
2. Implement KER-03 with `coff=2` state and a two-group fixture. Validate HOLD leaves the output slot untouched and the second group uses prior a halves. Check the vendored rotary-mode limitation before trying to substitute its aclnn wrapper.
3. Record the synthetic FP8 indexer inputs/expected indices in a separate fixture, then implement KER-04 with BF16-before-sort and lower-index tie break.
4. Validate KER-05 using nonuniform inputs and all six active expert outputs. Reject any scale-stride or nibble-order mismatch before integrating SwiGLU.
5. Only then consider multi-core projection reductions or fusion with cache epilogs. Any such revision needs a new tiling ABI version, workspace equation, synchronization proof, and the same numerical fixtures.
