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

// The C-API surface of the vendored arch35 (Ascend 950PR) operators from
// third_party/ops_transformer -- the four-operator mHC chain plus the
// lightning indexer.
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
//   * QuantLightningIndexer selects the sparse tokens and quantizes
//     query/key for the sparse-flash front end (out is INT32 indices).

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

#ifdef __cplusplus
}  // extern "C"
#endif
