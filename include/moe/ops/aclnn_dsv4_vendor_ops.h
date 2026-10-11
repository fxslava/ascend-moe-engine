/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The C-API surface of the vendored arch35 (Ascend 950PR) DeepSeek-V4-Flash
// operators from third_party/ops_dsv4: the mHC residual chain, both lightning
// indexers, the token-level KV compressor, the shared-KV sparse attention core
// and the two cache epilogs.
//
// The prototypes are transcribed from the vendored trees' own
// op_host/op_api/aclnn_<op>.h headers, which this repository now builds into
// libcust_opapi.so (CANN builds) or provides through libopapi_mock.so (mock
// builds). OpTable resolves and calls them with dlsym + reinterpret_cast, so
// a prototype that drifted from the vendored header would be a silent ABI
// mismatch; keep this file and the vendored headers in lockstep when
// re-vendoring.
//
// Semantics (engine geometry: n_hc = 4, hidden = 4096, TND token layout):
//   * MhcPre maps the stacked mHC states x [T, n, D] through the mixing
//     weights phi [n^2+2n, nD] into the layer input hIn [T, D] and the
//     routing state hPost [T, n] / hRes [T, n, n].
//   * MhcSinkhorn normalizes hRes into a doubly-stochastic matrix in place
//     (output [T, n, n], 1 <= numIters <= 100, n in {4, 6, 8}).
//   * MhcPost applies x_next = (hRes)^T @ x + hOut * hPost after the layer.
//   * HcPre / HcPreInvRms / HcPreSinkhorn / HcPost are the vllm-ascend
//     decomposition of that same mHC mapping, kept alongside the trio above.
//     HcPre fuses the whole prologue (reciprocal RMS, mixing projection and
//     Sinkhorn) into one launch; HcPreInvRms + a mixing GEMM + HcPreSinkhorn
//     is the staged alternative; HcPost is the BSHD residual combine.
//   * QuantLightningIndexer selects the sparse tokens and quantizes
//     query/key for the sparse-flash front end (out is INT32 indices).
//   * Compressor pools cmpRatio tokens (4 = CSA, 128 = HCA) into one
//     compressed-KV row and advances the recurrent pooling state in place.
//   * VllmQuantLightningIndexer is the shared-KV form of the indexer: same
//     top-k selection plus a scheduling-metadata input, an explicit key
//     compression ratio and an optional score output.
//   * KvQuantSparseAttnSharedkv is the MQA core attention over the hybrid
//     cache: one query stream against both the uncompressed and the
//     compressed KV streams, with attention-sink support.
//   * KvCompressEpilog and IndexerCompressEpilogV2 quantize and scatter the
//     two compressed streams into their paged caches.
//
// REF PARAMETERS AND REPEATABILITY
//   Compressor.stateCacheRef, KvCompressEpilog.kvCompressCacheRef and
//   IndexerCompressEpilogV2.indexerCompressCacheRef are written in place and
//   receive no ViewCopy, and every genuine output is staged through a distinct
//   executor-owned tensor. So no vendored wrapper here can issue a
//   same-address self-copy, and every executor they return is reusable with
//   aclSetAclOpExecutorRepeatable + aclSetTensorAddr -- except
//   aclnnMhcSinkhorn, whose upstream copy stage is patched to the same rule
//   (see third_party/ops_dsv4/mhc/mhc_sinkhorn/op_host/op_api).
//
//   The four hc_* operators declare no REF parameter at all -- no output name
//   matches an input name in their OpDefs -- and the Sinkhorn normalization
//   they perform never leaves the kernel as a tensor. So unlike
//   aclnnMhcSinkhorn they need no patch: the same-address copy they would have
//   to elide does not exist in their wrappers.

#pragma once

#include <stdint.h>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

// The real aclnn_base.h defines ACLNN_API as a visibility attribute; the
// engine's minimal mock copy does not, so the declaration stays portable.
#ifndef ACLNN_API
#define ACLNN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief aclnnMhcPre: plan phase (vendored ops-transformer arch35).
 * @param[in] x Stacked mHC states, [B,S,N,D] or [T,N,D], BF16/FP16.
 * @param[in] phi Mixing weights, [N^2+2N, N*D], FP32.
 * @param[in] alpha Gain triple, [3], FP32.
 * @param[in] bias Mixing bias, [N^2+2N], FP32.
 * @param[in] gammaOptional Per-stream normalizer, [N, D], FP32; may be null.
 * @param[in] normEps RMS normalization epsilon.
 * @param[in] hcEps Hyper-connection stability epsilon.
 * @param[out] hIn Layer input, [B,S,D] or [T,D], dtype of x.
 * @param[out] hPost Post-mapping state, [B,S,N] or [T,N], FP32.
 * @param[out] hRes Residual mapping matrix, [B,S,N,N] or [T,N,N], FP32.
 * @param[out] invRmsOptional Reciprocal RMS, [B,S] or [T], FP32; may be null.
 * @param[out] hMixOptional Pre-norm mixing state, [B,S,N^2+2N] or [T,N^2+2N],
 * FP32; may be null.
 * @param[out] hPreOptional Pre-mapping state, [B,S,N] or [T,N], FP32; may be
 * null.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus: 0 on success; 161001 null pointer; 161002 invalid
 * shape/dtype/format; 361001 runtime failure.
 */
ACLNN_API aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor* x, const aclTensor* phi, const aclTensor* alpha,
                                                  const aclTensor* bias, const aclTensor* gammaOptional,
                                                  double normEps, double hcEps, aclTensor* hIn, aclTensor* hPost,
                                                  aclTensor* hRes, aclTensor* invRmsOptional,
                                                  aclTensor* hMixOptional, aclTensor* hPreOptional,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMhcPre(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream);

/*
 * @brief aclnnMhcSinkhorn: plan phase (vendored ops-transformer arch35).
 * @param[in] x Square matrices to normalize, [B,S,N,N] or [T,N,N], FP32.
 * @param[in] eps Sinkhorn division guard.
 * @param[in] numIters Iteration count, in [1, 100].
 * @param[out] output Doubly-stochastic matrices, same shape and dtype as x.
 * @param[out] normOut Optional norm state, FP32; bound together with sumOut.
 * @param[out] sumOut Optional row-sum state, FP32; bound together with normOut.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 * @note N must be 4, 6 or 8 and the last two dimensions equal. When either
 * optional output is null the operator runs with outFlag 0.
 */
ACLNN_API aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor* x, float eps, int64_t numIters,
                                                       aclTensor* output, aclTensor* normOut, aclTensor* sumOut,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMhcSinkhorn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

/*
 * @brief aclnnMhcPost: plan phase (vendored ops-transformer arch35).
 * @details x_next = (hRes)^T @ x + hOut * hPost.
 * @param[in] x Layer input state, [B,S,N,D] or [T,N,D], BF16/FP16.
 * @param[in] hRes Doubly-stochastic mapping, [B,S,N,N] or [T,N,N], FP32.
 * @param[in] hOut Layer output, [B,S,D] or [T,D], dtype of x.
 * @param[in] hPost Post-mapping state, [B,S,N] or [T,N], FP32.
 * @param[out] out Next layer input, same shape and dtype as x.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor* x, const aclTensor* hRes, const aclTensor* hOut,
                                                   const aclTensor* hPost, aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMhcPost(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

/*
 * THE FUSED hc_* mHC FAMILY (vendored vllm-ascend csrc/moe arch35)
 *
 * The same mHC mapping as the mhc_* trio above, decomposed differently and
 * kept alongside it. The decisive difference is the Sinkhorn stage:
 * aclnnMhcSinkhorn is a standalone in-place operator, and that in-place shape
 * is what creates the manual-4.31 same-address ViewCopy its wrapper has to
 * elide. In this family the normalization is interior to aclnnHcPre and
 * aclnnHcPreSinkhorn and never becomes a tensor at the aclnn layer, so no
 * copy stage can have src == dst. None of the four declares a REF parameter
 * either -- no output name matches an input name -- so every output is staged
 * through a distinct executor-owned tensor.
 *
 * Launch counts per layer:
 *   aclnnHcPre -> aclnnHcPost                                           2
 *   aclnnMhcPre -> aclnnMhcSinkhorn -> aclnnMhcPost                      3
 *   aclnnHcPreInvRms -> aclnnMatmul -> aclnnHcPreSinkhorn -> aclnnHcPost  4
 */

/*
 * @brief aclnnHcPre: plan phase (vendored vllm-ascend arch35).
 * @details The fully fused mHC prologue: reciprocal RMS, the mixing
 * projection x.flatten(-2) @ hcFn^T and the Sinkhorn normalization in ONE
 * launch, from the stacked states and mixing weights alone.
 * @param[in] x Stacked mHC states, [bs,hc,d] or [b,s,hc,d], BF16; hc = hcMult
 * = 4, d = 4096 or 7168.
 * @param[in] hcFn Mixing weights, [hcMult^2+2*hcMult, hc*d] = [24,16384], FP32.
 * @param[in] hcScale Gain triple, [3], FP32.
 * @param[in] hcBase Mixing bias, [24], FP32.
 * @param[in] hcMult Hyper-connection multiplicity; 4 only.
 * @param[in] hcSinkhornIters Sinkhorn iteration count, in [1, 100].
 * @param[in] hcEps Sinkhorn division guard.
 * @param[in] normEps RMS normalization epsilon. NOTE the order: hcEps comes
 * first, the reverse of aclnnMhcPre's (normEps, hcEps).
 * @param[out] yOut Layer input, [bs,d] or [b,s,d], BF16.
 * @param[out] postOut Post-mapping state, [..,hcMult], FP32.
 * @param[out] combFragOut Combine fragment, [..,hcMult,hcMult], FP32.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnHcPreGetWorkspaceSize(const aclTensor* x, const aclTensor* hcFn,
                                                 const aclTensor* hcScale, const aclTensor* hcBase, int64_t hcMult,
                                                 int64_t hcSinkhornIters, double hcEps, double normEps,
                                                 const aclTensor* yOut, const aclTensor* postOut,
                                                 const aclTensor* combFragOut, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnHcPre(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                 aclrtStream stream);

/*
 * @brief aclnnHcPreInvRms: plan phase (vendored vllm-ascend arch35).
 * @details Reduce the stacked mHC states over their last two axes into
 * rsqrt = 1 / sqrt(mean(x^2) + epsilon). This is a REQUIRED prologue of the
 * staged path, not an optional helper: it is the only producer of
 * aclnnHcPreSinkhorn's rsqrt input, and this engine is torch-free so there is
 * no host-side fallback.
 * @param[in] x Stacked mHC states, FP32/FP16/BF16, rank >= 2.
 * @param[in] epsilon Variance floor.
 * @param[out] yOut Reciprocal RMS scale: x's leading axes with a trailing 1,
 * so [bs,1] from [bs,hc,d] and [b,s,1] from [b,s,hc,d]. FP32.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnHcPreInvRmsGetWorkspaceSize(const aclTensor* x, double epsilon, const aclTensor* yOut,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnHcPreInvRms(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

/*
 * @brief aclnnHcPreSinkhorn: plan phase (vendored vllm-ascend arch35).
 * @details The mixing projection fused with the Sinkhorn normalization. NOT a
 * rename of aclnnMhcSinkhorn: that operator normalizes a square matrix in
 * place, which is the source of its manual-4.31 hazard; here the
 * normalization never leaves the kernel.
 * @param[in] mixes Mixing state, [..,hcMult^2+2*hcMult] = [..,24], FP32.
 * Upstream produces it with at::linear(x.flatten(-2), hcFn); the engine
 * produces the same rows with aclnnMatmul under its [K, N] contract.
 * @param[in] rsqrt Reciprocal RMS scale from aclnnHcPreInvRms, [..,1], FP32.
 * @param[in] hcScale Gain triple, [3], FP32.
 * @param[in] hcBase Mixing bias, [24], FP32.
 * @param[in] x Stacked mHC states, [bs,hc,d] or [b,s,hc,d], BF16.
 * @param[in] hcMult Hyper-connection multiplicity; 4 only.
 * @param[in] hcSinkhornIters Sinkhorn iteration count, in [1, 100].
 * @param[in] hcEps Sinkhorn division guard.
 * @param[out] yOut Layer input, [bs,d] or [b,s,d], BF16.
 * @param[out] postOut Post-mapping state, [..,hcMult], FP32.
 * @param[out] combFragOut Combine fragment, [..,hcMult,hcMult], FP32.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnHcPreSinkhornGetWorkspaceSize(const aclTensor* mixes, const aclTensor* rsqrt,
                                                         const aclTensor* hcScale, const aclTensor* hcBase,
                                                         const aclTensor* x, int64_t hcMult,
                                                         int64_t hcSinkhornIters, double hcEps,
                                                         const aclTensor* yOut, const aclTensor* postOut,
                                                         const aclTensor* combFragOut, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnHcPreSinkhorn(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

/*
 * @brief aclnnHcPost: plan phase (vendored vllm-ascend arch35).
 * @details y = comb^T @ residual + x * post -- the same residual combine as
 * aclnnMhcPost, in the BSHD layout instead of TND, and with NO attributes.
 * A TND token stream maps on as b = T, s = 1.
 * @param[in] x Layer output, [b,s,d], FP32/FP16/BF16.
 * @param[in] residual Residual streams, [b,s,hc,d], dtype of x.
 * @param[in] post Post-mapping state, [b,s,hc], FP32/FP16/BF16.
 * @param[in] comb Combine matrix, [b,s,hc,hc], dtype of post.
 * @param[out] yOut Next layer input, shape and dtype of residual.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnHcPostGetWorkspaceSize(const aclTensor* x, const aclTensor* residual,
                                                  const aclTensor* post, const aclTensor* comb,
                                                  const aclTensor* yOut, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnHcPost(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  aclrtStream stream);

/*
 * @brief aclnnQuantLightningIndexer: plan phase (vendored ops-transformer
 * arch35).
 * @details Sparse-flash front end: top-k token selection with quantized
 * query/key correlation scoring.
 * @param[in] query Index query, [B,S1,N1,D] or [T1,N1,D]; D=128, N1 in
 * {16, 24, 32, 64} on 950PR; FP8 E4M3 / HiFloat8.
 * @param[in] key Index key, layout per layoutKeyOptional (BSND / TND /
 * PA_BSND), N2=1, D=128; dtype of query.
 * @param[in] weights Scoring weights, [B,S1,N1] or [T,N1], BF16/FP16.
 * @param[in] queryDequantScale Query dequant scales, [B,S1,N1] or [T,N1],
 * FP32 (BF16 weights) or FP16.
 * @param[in] keyDequantScale Key dequant scales, layout of key with D
 * dropped, FP32/FP16.
 * @param[in] actualSeqLengthsQueryOptional Per-batch cumulative query
 * lengths, INT32; required for TND.
 * @param[in] actualSeqLengthsKeyOptional Per-batch cumulative key lengths,
 * INT32; required for TND / PA_BSND.
 * @param[in] blockTableOptional Paged-KV block mapping, INT32 [B, >=maxBlocks].
 * @param[in] queryQuantMode Quantization mode; 0 = per-token-head.
 * @param[in] keyQuantMode Quantization mode; 0 = per-token-head.
 * @param[in] layoutQueryOptional Host layout string, "BSND" or "TND".
 * @param[in] layoutKeyOptional Host layout string, "BSND", "TND" or "PA_BSND".
 * @param[in] sparseCount Top-k blocks retained, in [1, 2048].
 * @param[in] sparseMode 0 dense mask, 3 right-down causal.
 * @param[in] preTokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] nextTokens Sparse window; INT64_MAX = unrestricted.
 * @param[out] out Selected sparse indices, INT32, [B,S1,N2,k] or [T,N2,k].
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* weights, const aclTensor* queryDequantScale,
    const aclTensor* keyDequantScale, const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional, const aclTensor* blockTableOptional, int64_t queryQuantMode,
    int64_t keyQuantMode, char* layoutQueryOptional, char* layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnQuantLightningIndexer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream);

/*
 * @brief aclnnCompressor: plan phase (vendored vllm-ascend arch35).
 * @details The token-level KV compressor: project a token window through
 * wkv/wgate, add the positional bias ape, pool with a softmax-weighted
 * reduction over cmpRatio tokens, RMSNorm with normWeight and rotate by the
 * partial RoPE in ropeSin/ropeCos. cmpRatio 4 selects CSA, 128 selects HCA.
 * @param[in] x Token states, [T, H] or [B, S, H], BF16/FP16.
 * @param[in] wkv Compressed-KV projection [coff*D,H], dtype of x.
 * @param[in] wgate Pooling-gate projection [coff*D,H], dtype of x.
 * @param[in,out] stateCacheRef Recurrent pooling state, [blocks, blockSize, 2*coff*D]
 * FP32; a REF parameter, UPDATED IN PLACE and never ViewCopied.
 * @param[in] ape Absolute positional bias, FP32.
 * @param[in] normWeight RMSNorm gain, 1-D.
 * @param[in] ropeSin RoPE sin table, rank of x.
 * @param[in] ropeCos RoPE cos table, rank of x.
 * @param[in] stateBlockTableOptional Paged state-cache block map, INT32.
 * @param[in] cuSeqlensOptional Cumulative sequence lengths, INT32.
 * @param[in] sequsedOptional Used length per batch, INT32.
 * @param[in] startPosOptional Window start per batch, INT32.
 * @param[in] ropeHeadDim RoPE head dimension (64).
 * @param[in] cmpRatio Compression ratio; 4 (CSA) or 128 (HCA).
 * @param[in] coff Output-channel multiplier.
 * @param[in] normEps RMSNorm epsilon.
 * @param[in] rotaryMode RoPE interleaving mode.
 * @param[in] cacheMode State-cache addressing mode.
 * @param[in] stateCacheStrideDim0 Element stride of the state-cache axis 0.
 * @param[out] cmpKvOut Compressed KV rows, dtype of x.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor* x, const aclTensor* wkv, const aclTensor* wgate, aclTensor* stateCacheRef, const aclTensor* ape,
    const aclTensor* normWeight, const aclTensor* ropeSin, const aclTensor* ropeCos,
    const aclTensor* stateBlockTableOptional, const aclTensor* cuSeqlensOptional, const aclTensor* sequsedOptional,
    const aclTensor* startPosOptional, int64_t ropeHeadDim, int64_t cmpRatio, int64_t coff, double normEps,
    int64_t rotaryMode, int64_t cacheMode, int64_t stateCacheStrideDim0, const aclTensor* cmpKvOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnCompressor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

/*
 * @brief aclnnVllmQuantLightningIndexer: plan phase (vendored vllm-ascend
 * arch35).
 * @details The quantized lightning indexer in its shared-KV form: a mixed
 * Cube/Vector kernel that scores FP8 E4M3FN query against key under
 * their dequant scales and emits the top-sparseCount block indices. Against
 * aclnnQuantLightningIndexer this adds the metadata input, the cmpRatio /
 * returnValues / stride / scaleStride attributes and the sparseValues output.
 * @param[in] query Index query, [B,S1,N1,D] or [T1,N1,D]; D=128.
 * @param[in] key Index key per layoutKeyOptional, N2=1, D=128, dtype of query.
 * @param[in] weights Scoring weights, [B,S1,N1] or [T,N1], FP32.
 * @param[in] queryDequantScale Query dequant scales, [B,S1,N1] or [T,N1].
 * @param[in] keyDequantScale Key dequant scales, layout of key without D.
 * @param[in] actualSeqLengthsQueryOptional Cumulative query lengths, INT32.
 * @param[in] actualSeqLengthsKeyOptional Cumulative key lengths, INT32.
 * @param[in] blockTableOptional Paged-KV block map, INT32.
 * @param[in] metadataOptional Precomputed scheduling metadata, INT32.
 * @param[in] queryQuantMode Quantization mode; 0 = per-token-head.
 * @param[in] keyQuantMode Quantization mode; 0 = per-token-head.
 * @param[in] layoutQueryOptional Host layout string, "BSND" or "TND".
 * @param[in] layoutKeyOptional Host layout string, "BSND", "TND", "PA_BSND".
 * @param[in] sparseCount Top-k blocks retained, in [1, 2048].
 * @param[in] sparseMode 0 dense mask, 3 right-down causal.
 * @param[in] preTokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] nextTokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] cmpRatio Compression ratio of the key stream.
 * @param[in] returnValues true to fill sparseValuesOut.
 * @param[in] stride Element stride of the key axis 0.
 * @param[in] scaleStride Element stride of the keyDequantScale axis 0.
 * @param[out] sparseIndicesOut Selected sparse indices, INT32.
 * @param[out] sparseValuesOut Selected scores, FP32; pass a [0] tensor when
 * returnValues is false.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnVllmQuantLightningIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* weights, const aclTensor* queryDequantScale,
    const aclTensor* keyDequantScale, const aclTensor* actualSeqLengthsQueryOptional,
    const aclTensor* actualSeqLengthsKeyOptional, const aclTensor* blockTableOptional,
    const aclTensor* metadataOptional, int64_t queryQuantMode, int64_t keyQuantMode, char* layoutQueryOptional,
    char* layoutKeyOptional, int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    int64_t cmpRatio, bool returnValues, int64_t stride, int64_t scaleStride, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnVllmQuantLightningIndexer(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);

/*
 * @brief aclnnKvQuantSparseAttnSharedkv: plan phase (vendored vllm-ascend
 * arch35).
 * @details The shared-KV sparse attention core: a dedicated MQA kernel in
 * which one BF16 query stream attends to BOTH halves of the quantized hybrid
 * KV cache -- the uncompressed (ori) stream and the compressor-produced
 * compressed (cmp) stream -- each at its own sparse indices, block table and
 * mask mode, with optional attention sinks folded into the softmax
 * denominator.
 * @note At least one of oriKvOptional and cmpKvOptional must be bound.
 * @param[in] q Query, BF16, layout per layoutQOptional.
 * @param[in] oriKvOptional Uncompressed KV cache, FP8 E4M3.
 * @param[in] cmpKvOptional Compressed KV cache, FP8 E4M3.
 * @param[in] oriSparseIndicesOptional INT32 indices into the ori stream.
 * @param[in] cmpSparseIndicesOptional INT32 indices into the cmp stream.
 * @param[in] oriBlockTableOptional Paged block map for ori, INT32.
 * @param[in] cmpBlockTableOptional Paged block map for cmp, INT32.
 * @param[in] cuSeqlensQOptional Cumulative query lengths, INT32.
 * @param[in] cuSeqlensOriKvOptional Cumulative ori KV lengths, INT32.
 * @param[in] cuSeqlensCmpKvOptional Cumulative cmp KV lengths, INT32.
 * @param[in] sequsedQOptional Used query length per batch, INT32.
 * @param[in] sequsedKvOptional Used KV length per batch, INT32.
 * @param[in] sinksOptional Attention sink logits, FP32.
 * @param[in] metadataOptional Precomputed scheduling metadata, INT32.
 * @param[in] kvQuantMode KV quantization mode.
 * @param[in] tileSize KV tile size (default 64).
 * @param[in] ropeHeadDim RoPE head dimension (default 64).
 * @param[in] softmaxScale Logit scale.
 * @param[in] cmpRatio Compression ratio of the cmp stream; 4 or 128.
 * @param[in] oriMaskMode Mask mode for the ori stream (default 4).
 * @param[in] cmpMaskMode Mask mode for the cmp stream (default 3).
 * @param[in] oriWinLeft Sliding-window left bound.
 * @param[in] oriWinRight Sliding-window right bound.
 * @param[in] layoutQOptional Host layout string for q (default "BSND").
 * @param[in] layoutKvOptional Host layout string for KV (default "PA_ND").
 * @param[in] oriKvStride0 Element stride of the ori cache axis 0.
 * @param[in] cmpKvStride0 Element stride of the cmp cache axis 0.
 * @param[in] returnSoftmaxLse true to fill softmaxLseOut.
 * @param[out] attnOut Attention output, BF16, shape of q.
 * @param[out] softmaxLseOut Log-sum-exp, FP32; pass a [0] tensor when
 * returnSoftmaxLse is false.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor* q, const aclTensor* oriKvOptional, const aclTensor* cmpKvOptional,
    const aclTensor* oriSparseIndicesOptional, const aclTensor* cmpSparseIndicesOptional,
    const aclTensor* oriBlockTableOptional, const aclTensor* cmpBlockTableOptional,
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensOriKvOptional,
    const aclTensor* cuSeqlensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    const aclTensor* sinksOptional, const aclTensor* metadataOptional, int64_t kvQuantMode, int64_t tileSize,
    int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, char* layoutQOptional, char* layoutKvOptional, int64_t oriKvStride0,
    int64_t cmpKvStride0, bool returnSoftmaxLse, const aclTensor* attnOut, const aclTensor* softmaxLseOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnKvQuantSparseAttnSharedkv(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);

/*
 * @brief aclnnKvCompressEpilog: plan phase (vendored vllm-ascend arch35).
 * @details Quantize the compressed-KV rows x and scatter them into the paged
 * FP8 cache at the slots slotMapping names.
 * @param[in,out] kvCompressCacheRef Paged FP8 E5M2/E4M3 cache; a REF
 * parameter, WRITTEN IN PLACE and never ViewCopied, so the returned executor
 * is reusable under manual 4.31.
 * @param[in] x Compressed KV rows to store, BF16.
 * @param[in] slotMapping Destination slot per row, INT32/INT64.
 * @param[in] quantGroupSize Elements per quant group (128 on 950PR).
 * @param[in] quantMode Quantization mode.
 * @param[in] roundScale 1 to round the scale to a power of two.
 * @param[in] layout Cache layout selector.
 * @param[in] blockStride Element stride of the cache block axis.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(aclTensor* kvCompressCacheRef, const aclTensor* x,
                                                            const aclTensor* slotMapping, int64_t quantGroupSize,
                                                            int64_t quantMode, int64_t roundScale, int64_t layout,
                                                            int64_t blockStride, uint64_t* workspaceSize,
                                                            aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnKvCompressEpilog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream);

/*
 * @brief aclnnIndexerCompressEpilogV2: plan phase (vendored vllm-ascend
 * arch35).
 * @details Scatter the indexer-side compressed rows x into the paged UINT8
 * indexer cache at the slots slotMapping names.
 * @param[in,out] indexerCompressCacheRef Paged UINT8 indexer cache; a REF
 * parameter, WRITTEN IN PLACE and never ViewCopied, so the returned executor
 * is reusable under manual 4.31.
 * @param[in] x Rows to store, FP16 or BF16.
 * @param[in] slotMapping Destination slot per row, INT32.
 * @param[in] layout Cache layout selector.
 * @param[in] blockStride Element stride of the cache block axis.
 * @param[out] workspaceSize Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return aclnnStatus as for aclnnMhcPreGetWorkspaceSize.
 */
ACLNN_API aclnnStatus aclnnIndexerCompressEpilogV2GetWorkspaceSize(aclTensor* indexerCompressCacheRef,
                                                                   const aclTensor* x, const aclTensor* slotMapping,
                                                                   int64_t layout, int64_t blockStride,
                                                                   uint64_t* workspaceSize,
                                                                   aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnIndexerCompressEpilogV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}  // extern "C"
#endif
