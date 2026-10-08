/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

// The ACLNN V5 operator surface the DSV4 pipeline drives.
//
// Every `GetWorkspaceSize` prototype below is transcribed from the CANN
// 9.2.0-beta.2 header of the same name, and the audited subset matches
// `csrc/tests/common/aclnn_ops_950pr.hpp` argument for argument.
// `dsv4_aclnn_signature_check.cpp` includes the real headers and static_asserts
// each typedef against the declared function, so a toolkit that changes one
// breaks the build instead of the dlsym'd call reading the wrong registers.
//
// WHY THE ENTRY POINTS ARE RESOLVED, NOT LINK-BOUND
// -------------------------------------------------
// The CANN libraries are on the link line (see CMakeLists.txt), which is what
// makes `ldd -r`'s "zero undefined symbols" check meaningful for the ACL
// runtime, `aclCreateTensor`, `aclSetTensorAddr` and friends. The *operator*
// entry points are then taken out of those same already-loaded libraries with
// `dlsym(RTLD_DEFAULT, ...)` -- no `dlopen`, no second copy -- for one concrete
// reason: the op set is not the same across the toolkits this binary must
// survive. Checked here:
//
//   CANN 9.2.0-beta.2 x86_64   exports all the ops below.
//   CANN 9.1.0 aarch64         does not export `aclnnMoeGatingTopKV2` or
//                              `aclnnMoeInitRoutingV4` at all.
//
// A link-time dependency on those two would make the binary refuse to start on
// 9.1.0 with a loader error naming a symbol, instead of starting and reporting
// which operator its toolkit is missing. `aclnnFusedInferAttentionScoreV5-
// GetMaxWorkspaceSize` is a third case: it is *exported* by both toolkits and
// declared by neither header, so it can only be reached this way.

#pragma once
#include <type_traits>

#include <cstdint>
#include <string>
#include <vector>

#include "moe/core/error.hpp"
#include "moe/core/acl_guard.hpp"
#include "moe/core/device_ops.hpp"

// Forward declarations matching acl/aclnn/acl_meta.h, so this header does not
// drag the whole toolkit into every translation unit.
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclTensorList aclTensorList;

namespace ascend_moe {

enum class OpId {
  // Dense / backbone
  kRmsNorm,
  kRmsNormDynamicMxQuant,
  kDynamicMxQuant,
  kMatmul,
  kCast,
  kQuantMatmulV5,
  kApplyRotaryPosEmbV2,
  kScatterPaKvCache,
  kFusedInferAttentionScoreV5,
  kFiaV5GetMaxWorkspace,
  kSigmoid,
  kMul,
  kInplaceAdd,
  kSwiGlu,
  kArgMax,
  // Decomposed sqrtsoftplus router scoring (DSV4 `scoring_func`)
  kSoftplus,
  kSqrt,
  // MoE
  kMoeGatingTopKV2,
  kMoeInitRoutingV4,
  kGroupedMatmulV5,
  kSwigluMxQuant,
  kMoeTokenUnpermute,
  kGroupedMatmulSwigluQuantV2,
  kGroupedMatmulFinalizeRoutingV3,
  kOpCount,
};

struct ResolvedOp {
  const char* name = nullptr;
  const char* role = nullptr;
  void* plan = nullptr;    // aclnn<Op>GetWorkspaceSize
  void* launch = nullptr;  // aclnn<Op>
  std::string provider;    // the shared object dlsym found it in
  bool required = true;    // an optional op may legitimately be absent

  bool available() const { return plan != nullptr && launch != nullptr; }
};

// ---------------------------------------------------------------------------
// Plan prototypes (CANN 9.2.0-beta.2 headers, verbatim)
// ---------------------------------------------------------------------------

/**
 * @defgroup aclnn_contract ACLNN two-phase operator contract
 * @brief Imported NN, Math and Transformer operators and their ownership rules.
 * @details Source: Ascend Operator Library API Reference, issue 1, 2026-09-29,
 * sections 4.7-4.9, 4.16, 4.31, 4.38-4.42 and 8.1.2.33. Chapter 5 links
 * to separate operator specifications; it does not contain their dtype tables.
 * Per-operator semantics below also use the installed CANN 9.2.0-beta.2 headers.
 * Dtype notes distinguish the engine's selected path from the wider API surface.
 * @note Call GetWorkspaceSize first, allocate at least workspace_size device
 * bytes, then invoke the matching launch entry on an initialized ACL stream.
 * A normal executor is consumed by launch. StaticOpSlot immediately enables
 * repeatability; retained executors must be explicitly destroyed after draining
 * streams and before destroying descriptors, workspace or tensor storage.
 * Shape, dtype and format are frozen: only addresses may change between launches.
 * @warning Exported symbols do not prove hardware support. The engine targets
 * Ascend 950PR; FP8/FP4/MX paths are not portable to 910B. Validate the exact
 * toolkit and SoC combination on hardware; a numeric device identifier such as
 * 9589 is not a substitute for aclrtGetSocName(). No l0op APIs are imported.
 * Internal L0 copies may prohibit repeatability (section 4.31); always check
 * aclSetAclOpExecutorRepeatable. Do not invoke the undeclared FIA maximum-
 * workspace helper: its presence in the inventory is diagnostic only.
 * @return ACLNN_SUCCESS (0): planning succeeded or launch was enqueued.
 * @return ACLNN_ERR_PARAM_NULLPTR (161001): a required argument was null.
 * @return ACLNN_ERR_PARAM_INVALID (161002): invalid shape, dtype, format or attribute.
 * @return ACLNN_ERR_RUNTIME_ERROR (361001): runtime memory/API failure.
 * Other runtime/internal errors are possible; capture aclGetRecentErrMsg before
 * cleanup and check synchronization as well as launch for asynchronous errors.
 */

/**
 * @brief aclnnRmsNorm: plan phase (NN).
 * @details RMS normalization: y = x * gamma / sqrt(mean(x*x) + epsilon).
 * @note Engine: BF16 x/gamma/y, FP32 reciprocal-standard-deviation output; no mixed-input promotion is
 * assumed.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] gamma Per-channel normalization multiplier.
 * @param[in] epsilon Positive normalization stability constant.
 * @param[out] y_out Output values in the selected operation's output dtype/shape.
 * @param[out] rstd_out FP32 reciprocal standard deviation of each normalized row.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using RmsNormPlanFn = int (*)(const aclTensor* x, const aclTensor* gamma, double epsilon, const aclTensor* y_out,
                              const aclTensor* rstd_out, uint64_t* workspace_size, aclOpExecutor** executor);

// RMSNorm fused with OCP block-32 MX quantization: the activation quantizer
// for both the dense FP8 GEMMs and the routed FP4 expert GEMM.
/**
 * @brief aclnnRmsNormDynamicMxQuant: plan phase (NN).
 * @details RMS normalization followed by block MX quantization.
 * @note Engine: BF16 x/gamma, FP8 E4M3 output and E8M0 scales; rstd is optional. This MX path requires
 * 950-class support, not 910B.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] gamma Per-channel normalization multiplier.
 * @param[in] beta Optional normalization offset tensor; null when no offset is needed.
 * @param[in] epsilon Positive normalization stability constant.
 * @param[in] scale_alg Scale-selection algorithm; engine selects OCP MX.
 * @param[in] round_mode Host rounding-mode string; engine uses rint.
 * @param[in] dst_type ACL destination dtype code for quantized values.
 * @param[in] output_rstd Whether to write reciprocal standard deviations.
 * @param[out] y_out Output values in the selected operation's output dtype/shape.
 * @param[out] mxscale_out Output E8M0 microscales associated with the quantized blocks.
 * @param[out] rstd_out Output reciprocal standard deviation; optional when output_rstd is false.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using RmsNormDynamicMxQuantPlanFn = int (*)(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                            double epsilon, int64_t scale_alg, char* round_mode, int64_t dst_type,
                                            bool output_rstd, aclTensor* y_out, aclTensor* mxscale_out,
                                            aclTensor* rstd_out, uint64_t* workspace_size,
                                            aclOpExecutor** executor);

// Standalone OCP block-32 MX quantization, for the two places an activation has
// to become FP8 without a fused RMSNorm in front of it: the MLA attention
// output and the shared expert's SwiGLU output.
/**
 * @brief aclnnDynamicMxQuant: plan phase (Math).
 * @details Quantize blocks along an axis using shared power-of-two microscales.
 * @note Engine: BF16 input to FP8 E4M3 and E8M0 scales, blocksize=32, round mode rint. FP8/FP4 formats
 * require a supporting SoC; do not assume 910B support.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] axis Quantization axis; -1 selects the final dimension.
 * @param[in] round_mode_optional Optional host rounding-mode string; engine supplies rint.
 * @param[in] dst_type ACL destination dtype code for quantized values.
 * @param[in] blocksize Elements sharing a quantization scale; engine uses 32.
 * @param[in] scale_alg Scale-selection algorithm; engine selects OCP MX.
 * @param[out] y_out Output values in the selected operation's output dtype/shape.
 * @param[out] mxscale_out Output E8M0 microscales associated with the quantized blocks.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using DynamicMxQuantPlanFn = int (*)(const aclTensor* x, int64_t axis, char* round_mode_optional, int64_t dst_type,
                                     int64_t blocksize, int64_t scale_alg, const aclTensor* y_out,
                                     const aclTensor* mxscale_out, uint64_t* workspace_size,
                                     aclOpExecutor** executor);

/**
 * @brief aclnnCast: plan phase (Math).
 * @details Convert every element to an explicitly selected dtype.
 * @note Used for BF16 to FP32 router scores and FP32 to BF16 combine weights. Shapes agree; conversion
 * can round. These conversions are supported on A2/910B and 950PR; other formats have SoC-specific
 * restrictions.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[in] dtype Output ACL dtype code; must agree with the output descriptor.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using CastPlanFn = int (*)(const aclTensor* self, aclDataType dtype, aclTensor* out,
                           uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnMatmul: plan phase (NN).
 * @details Matrix multiplication C = A B; the engine uses [M,K] x [K,N].
 * @note API supports BF16/FP16/FP32 with hardware-specific deduction. Engine deliberately requires
 * homogeneous BF16 or FP16 A/B/C, ND, contiguous strides, zero offset and 32-byte-aligned pointers,
 * with cube_math_type=0. No implicit transpose; checkpoint [N,K] weights are materialized as [K,N].
 * Quantized projections use QuantMatmulV5 or GroupedMatmulV5 instead. BF16 requires a capable SoC such
 * as 910B/950PR.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[in] mat2 Right matrix [K,N]; this API has no transpose attribute.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[in] cube_math_type Cube arithmetic policy; engine requires KEEP_DTYPE (0).
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using MatmulPlanFn = int (*)(const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cube_math_type,
                             uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulV5: plan phase (NN).
 * @details Scaled matrix product with input/output quantization parameters and optional bias.
 * @note Engine: FP8 E4M3 activations/weights with E8M0 scales and BF16 output; transpose_x2=true
 * consumes [N,K] weights. Scales and group_size must match the quantization scheme; this is not
 * ordinary Matmul type promotion. The selected FP8/MX path is 950-specific, not a 910B fallback.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x1 Left input matrix.
 * @param[in] x2 Right input matrix/weight tensor.
 * @param[in] x1_scale Activation quantization scales, required by the selected scheme.
 * @param[in] x2_scale Weight quantization scales, required by the selected scheme.
 * @param[in] y_scale Optional output quantization scale; null for the selected BF16 output.
 * @param[in] x1_offset Optional activation zero point; null for the selected symmetric scheme.
 * @param[in] x2_offset Optional weight zero point; null for the selected symmetric scheme.
 * @param[in] y_offset Optional output zero point; null for the selected BF16 output.
 * @param[in] bias Optional additive bias, subject to the selected quantization mode.
 * @param[in] transpose_x1 Whether to transpose the left matrix's last two axes.
 * @param[in] transpose_x2 Whether to transpose the right matrix's last two axes.
 * @param[in] group_size Packed M/N/K quantization block sizes; must match stored scales.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using QuantMatmulV5PlanFn = int (*)(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1_scale,
                                    const aclTensor* x2_scale, const aclTensor* y_scale, const aclTensor* x1_offset,
                                    const aclTensor* x2_offset, const aclTensor* y_offset, const aclTensor* bias,
                                    bool transpose_x1, bool transpose_x2, int64_t group_size, aclTensor* out,
                                    uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnApplyRotaryPosEmbV2: plan phase (Transformer).
 * @details Apply rotary positional embedding to query and key in place.
 * @note Engine uses BF16 query/key/cos/sin and half rotation; layout and rotary dimension must match
 * the descriptor geometry. Compatibility of each layout is toolkit/SoC-specific; symbol resolution
 * alone is insufficient.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in,out] query_ref Query tensor updated in place by rotary embedding.
 * @param[in,out] key_ref Key tensor updated in place by rotary embedding.
 * @param[in] cos Cosine table broadcast over the rotary slices.
 * @param[in] sin Sine table broadcast over the rotary slices.
 * @param[in] layout Integer layout selector describing query and key axes.
 * @param[in] rotary_mode Host string selecting the rotary pairing convention; engine uses half.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using ApplyRotaryPosEmbV2PlanFn = int (*)(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                          const aclTensor* sin, int64_t layout, char* rotary_mode,
                                          uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnScatterPaKvCache: plan phase (Transformer).
 * @details Scatter current key/value rows into a paged attention cache using slot indices.
 * @note Engine uses BF16 cache/token data and INT32 slot mapping. Cache layout, stride and compression
 * modes must match storage. Optional compression paths are not validated by the standard decode path.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] key Input key tensor, or list of key cache tensors for attention.
 * @param[in,out] key_cache_ref Paged key cache updated in place.
 * @param[in] slot_mapping Integer destination slot for each token row.
 * @param[in] value Input value tensor, or list of value cache tensors for attention.
 * @param[in,out] value_cache_ref Paged value cache updated in place.
 * @param[in] compress_lens_optional Optional compressed lengths, required by compression modes.
 * @param[in] compress_seq_offset_optional Optional offsets into compressed sequences.
 * @param[in] seq_lens_optional Optional valid sequence lengths for cache scattering.
 * @param[in] cache_mode_optional Host cache-mode string; engine uses Norm.
 * @param[in] scatter_mode_optional Optional host scatter-mode string.
 * @param[in] strides_optional Optional host integer array of cache strides.
 * @param[in] offsets_optional Optional host integer array of cache offsets.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using ScatterPaKvCachePlanFn = int (*)(const aclTensor* key, aclTensor* key_cache_ref, const aclTensor* slot_mapping,
                                       const aclTensor* value, aclTensor* value_cache_ref,
                                       const aclTensor* compress_lens_optional,
                                       const aclTensor* compress_seq_offset_optional,
                                       const aclTensor* seq_lens_optional, char* cache_mode_optional,
                                       char* scatter_mode_optional, const aclIntArray* strides_optional,
                                       const aclIntArray* offsets_optional, uint64_t* workspace_size,
                                       aclOpExecutor** executor);

// Signature checked against the installed CANN header at build time.
/**
 * @brief aclnnFusedInferAttentionScoreV5: plan phase (Transformer).
 * @details Fused scaled dot-product attention: softmax(scale Q K^T + mask/position terms) V.
 * @note Engine uses contiguous ND BF16 Q/K/V and output, FP32 LSE, TND layout and paged MLA. Quantized
 * query/KV variants require matching scale tensors; no automatic promotion is assumed. 950PR and 910B
 * have different mode, head-size and quantization constraints; verify the selected MLA geometry.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] query Attention query tensor with axes defined by input_layout.
 * @param[in] key Input key tensor, or list of key cache tensors for attention.
 * @param[in] value Input value tensor, or list of value cache tensors for attention.
 * @param[in] pse_shift Optional position-encoding term; null when unused.
 * @param[in] atten_mask Optional attention mask, shaped for the selected sparse mode.
 * @param[in] actual_seq_lengths Optional host array of valid query lengths.
 * @param[in] actual_seq_lengths_kv Optional host array of valid key/value lengths.
 * @param[in] deq_scale1 Optional first matrix-product dequantization scale; null on BF16 path.
 * @param[in] quant_scale1 Optional intermediate quantization scale; null on BF16 path.
 * @param[in] deq_scale2 Optional second matrix-product dequantization scale; null on BF16 path.
 * @param[in] quant_scale2 Optional attention-output quantization scale; null on BF16 path.
 * @param[in] quant_offset2 Optional attention-output quantization offset; null on BF16 path.
 * @param[in] antiquant_scale Optional joint KV antiquantization scale; null on BF16 path.
 * @param[in] antiquant_offset Optional joint KV antiquantization offset; null on BF16 path.
 * @param[in] block_table Device INT32 table mapping sequences to physical cache blocks.
 * @param[in] query_padding_size Optional left-padding sizes for query sequences.
 * @param[in] kv_padding_size Optional left-padding sizes for KV sequences.
 * @param[in] key_antiquant_scale Optional separate key dequantization scales.
 * @param[in] key_antiquant_offset Optional separate key dequantization offsets.
 * @param[in] value_antiquant_scale Optional separate value dequantization scales.
 * @param[in] value_antiquant_offset Optional separate value dequantization offsets.
 * @param[in] key_shared_prefix Optional shared-prefix key tensor.
 * @param[in] value_shared_prefix Optional shared-prefix value tensor.
 * @param[in] actual_shared_prefix_len Optional host array of valid shared-prefix lengths.
 * @param[in] query_rope Optional MLA query rotary components.
 * @param[in] key_rope Optional MLA key rotary components.
 * @param[in] key_rope_antiquant_scale Reserved rotary-key scale; pass null for this version.
 * @param[in] dequant_scale_query Optional quantized-query dequantization scale.
 * @param[in] learnable_sink Optional learned attention-sink values; unused by this graph.
 * @param[in] q_start_idx Optional host query start-position array.
 * @param[in] kv_start_idx Optional host KV start-position array.
 * @param[in] num_heads Number of query heads.
 * @param[in] scale_value Multiplier applied to attention logits before softmax.
 * @param[in] pre_tokens Left attention window size.
 * @param[in] next_tokens Right attention window size.
 * @param[in] input_layout Host axis-layout string; engine uses TND.
 * @param[in] num_key_value_heads Number of key/value heads for MHA/GQA/MLA mode.
 * @param[in] sparse_mode Sparse-mask interpretation; engine selects dense attention.
 * @param[in] inner_precise Internal attention precision policy.
 * @param[in] block_size Tokens per physical paged-cache block.
 * @param[in] antiquant_mode Joint KV antiquantization granularity selector.
 * @param[in] softmax_lse_flag Whether to emit the log-sum-exp output.
 * @param[in] key_antiquant_mode Separate key antiquantization granularity selector.
 * @param[in] value_antiquant_mode Separate value antiquantization granularity selector.
 * @param[in] query_quant_mode Query quantization granularity selector.
 * @param[in] pse_type Position-encoding interpretation selector.
 * @param[out] attention_out Attention result; value feature width and query token/head axes.
 * @param[out] softmax_lse FP32 log-sum-exp output; [T,N,1] for TND when enabled.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using FusedInferAttentionScoreV5PlanFn = int (*)(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pse_shift,
    const aclTensor* atten_mask, const aclIntArray* actual_seq_lengths, const aclIntArray* actual_seq_lengths_kv,
    const aclTensor* deq_scale1, const aclTensor* quant_scale1, const aclTensor* deq_scale2,
    const aclTensor* quant_scale2, const aclTensor* quant_offset2, const aclTensor* antiquant_scale,
    const aclTensor* antiquant_offset, const aclTensor* block_table, const aclTensor* query_padding_size,
    const aclTensor* kv_padding_size, const aclTensor* key_antiquant_scale, const aclTensor* key_antiquant_offset,
    const aclTensor* value_antiquant_scale, const aclTensor* value_antiquant_offset,
    const aclTensor* key_shared_prefix, const aclTensor* value_shared_prefix,
    const aclIntArray* actual_shared_prefix_len, const aclTensor* query_rope, const aclTensor* key_rope,
    const aclTensor* key_rope_antiquant_scale, const aclTensor* dequant_scale_query, const aclTensor* learnable_sink,
    const aclIntArray* q_start_idx, const aclIntArray* kv_start_idx, int64_t num_heads, double scale_value,
    int64_t pre_tokens, int64_t next_tokens, char* input_layout, int64_t num_key_value_heads, int64_t sparse_mode,
    int64_t inner_precise, int64_t block_size, int64_t antiquant_mode, bool softmax_lse_flag,
    int64_t key_antiquant_mode, int64_t value_antiquant_mode, int64_t query_quant_mode, int64_t pse_type,
    const aclTensor* attention_out, const aclTensor* softmax_lse, uint64_t* workspace_size,
    aclOpExecutor** executor);

/**
 * @brief aclnnSigmoid / aclnnSqrt: plan phase (Math).
 * @details Elementwise sigmoid 1/(1+exp(-x)) or principal square root, selected by OpId.
 * @note Router sqrt uses FP32 input/output of identical shape; negative inputs yield non-real/NaN
 * results. Floating FP16/BF16/FP32 availability depends on the selected API/SoC. Sigmoid is optional
 * and is not used by this graph.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using UnaryPlanFn = int (*)(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                            aclOpExecutor** executor);

/**
 * @brief aclnnMul: plan phase (Math).
 * @details Elementwise multiplication with broadcast-compatible inputs.
 * @note API promotion follows the selected dtype pair and requires a compatible output; the engine
 * must not infer Matmul promotion from these rules. This optional operation is not used by the DSV4
 * graph.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[in] other Second device operand, broadcast-compatible with the first.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using BinaryPlanFn = int (*)(const aclTensor* self, const aclTensor* other, aclTensor* out,
                             uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAdd: plan phase (Math).
 * @details Update self_ref = self_ref + alpha * other with broadcast rules.
 * @note Engine residual tensors are BF16; promotion must be writable back to self_ref without changing
 * its dtype. alpha is a host scalar; nullptr selects the API default. BF16 support is SoC-dependent.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in,out] self_ref Device accumulator updated in place.
 * @param[in] other Second device operand, broadcast-compatible with the first.
 * @param[in] alpha Host scalar multiplying other; null selects the API default.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using InplaceAddPlanFn = int (*)(const aclTensor* self_ref, const aclTensor* other, const aclScalar* alpha,
                                 uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnSwiGlu: plan phase (NN).
 * @details Split x into two equal parts along dim and multiply SiLU of the first by the second.
 * @note Engine input/output are BF16; the split dimension is even and output halves it. No FP8/FP4
 * implicit dequantization. Check the installed SoC kernel before using an alternate dtype.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] dim Reduction/split axis; negative values count from the final axis.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using SwiGluPlanFn = int (*)(const aclTensor* x, int64_t dim, const aclTensor* out, uint64_t* workspace_size,
                             aclOpExecutor** executor);

/**
 * @brief aclnnArgMax: plan phase (Math).
 * @details Return indices of maximum values along dim.
 * @note Engine uses BF16 logits and INT64 indices; output shape removes dim unless keepdim is true.
 * Index dtype is intentionally different from the floating input.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[in] dim Reduction/split axis; negative values count from the final axis.
 * @param[in] keepdim Keep the reduced dimension with size one.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using ArgMaxPlanFn = int (*)(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                              uint64_t* workspace_size, aclOpExecutor** executor);

// aclnnSoftplus: stage 1 of the decomposed sqrtsoftplus router scoring. beta
// and threshold are HOST aclScalars (FP32, created once by the arena); the
// elementwise result feeds aclnnSqrt (UnaryPlanFn) and then the pre-normalized
// gating stage. SoftplusConfig in v5_ops_moe.py pins the same two values.
/**
 * @brief aclnnSoftplus: plan phase (NN).
 * @details Compute log(1 + exp(beta*x))/beta, using the linear branch above threshold.
 * @note Engine uses FP32 input/output and FP32 host beta/threshold scalars. beta is nonzero; output
 * preserves shape. This feeds FP32 Sqrt and gating without implicit dtype conversion.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] self Input device tensor.
 * @param[in] beta Host FP32 scalar controlling the exponential slope; nonzero.
 * @param[in] threshold Host scalar selecting softplus's linear branch.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using SoftplusPlanFn = int (*)(const aclTensor* self, const aclScalar* beta, const aclScalar* threshold,
                               aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnMoeGatingTopKV2: plan phase (Transformer).
 * @details Select top-k experts, optionally group-filter and normalize/scalewise adjust routing
 * weights.
 * @note Engine uses FP32 scores/bias/weights and INT32 expert indices, k=6 over 256 experts.
 * norm_type=-1 is the selected pre-normalized mode; verify it on the installed kernel. This version
 * may be absent from older toolkits; 910B support must not be inferred from the symbol.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] bias_optional Optional selection correction or GEMM bias, as specified by the operator.
 * @param[in] input_ids_optional Optional token IDs used by expert-selection mapping.
 * @param[in] tid2eid_optional Optional token-to-expert mapping tensor.
 * @param[in] k Number of experts selected for each token.
 * @param[in] k_group Number of selected expert groups.
 * @param[in] group_count Number of expert groups.
 * @param[in] group_select_mode Group-score selection algorithm.
 * @param[in] renorm Selected-weight renormalization policy; engine requests L1.
 * @param[in] norm_type Score normalization selector; engine supplies pre-normalized scores.
 * @param[in] out_flag Whether to write the auxiliary full-score output.
 * @param[in] routed_scaling_factor Multiplier applied to selected expert probabilities.
 * @param[in] eps Stability constant used by routing normalization.
 * @param[out] y_out Output values in the selected operation's output dtype/shape.
 * @param[out] expert_idx_out Output expert IDs for each token's top-k selection.
 * @param[out] out_out Optional auxiliary score output, controlled by out_flag.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using MoeGatingTopKV2PlanFn = int (*)(const aclTensor* x, const aclTensor* bias_optional,
                                      const aclTensor* input_ids_optional, const aclTensor* tid2eid_optional,
                                      int64_t k, int64_t k_group, int64_t group_count, int64_t group_select_mode,
                                      int64_t renorm, int64_t norm_type, bool out_flag,
                                      double routed_scaling_factor, double eps, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out, const aclTensor* out_out,
                                      uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnMoeInitRoutingV4: plan phase (Transformer).
 * @details Expand and sort tokens by expert index, emitting reverse indices and per-expert counts.
 * @note Engine uses FP8 E4M3 tokens, INT32 expert/row indices, INT64 group counts, E8M0 scales and
 * FP32 top-k weights. Selected mode is dropless, quant_mode=0 and cumsum group counts. These typed
 * outputs are distinct, not promoted inputs; FP8/MX requires 950 support.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] expert_idx Integer expert IDs for every token/top-k pair.
 * @param[in] scale_optional Optional quantization scales or grouped weight dequantization scales.
 * @param[in] offset_optional Optional quantization offsets/zero points.
 * @param[in] active_num_optional Optional upper bound on expanded rows; defaults to rows times k.
 * @param[in] topk_weight_optional Optional routing weights to permute with dispatched rows.
 * @param[in] expert_capacity Per-expert token capacity in drop/pad mode.
 * @param[in] expert_num Expert count in the local dispatch domain.
 * @param[in] drop_pad_mode 0 for dropless routing; 1 for capacity-based drop/pad.
 * @param[in] expert_tokens_num_type 0 cumsum, 1 counts, 2 key/value expert statistics.
 * @param[in] expert_tokens_num_flag Whether to output expert-token statistics.
 * @param[in] quant_mode Activation quantization mode; must agree with scales and output dtype.
 * @param[in] active_expert_range_optional Optional host [start,end) expert range.
 * @param[in] row_idx_type 0 gather mapping, 1 scatter mapping.
 * @param[out] expanded_x_out Expanded tokens sorted by expert.
 * @param[out] expanded_row_idx_out Integer mapping between original and expanded token rows.
 * @param[out] expert_tokens_count_or_cumsum_out INT64 expert statistics, interpreted by
 * expert_tokens_num_type.
 * @param[out] expanded_scale_out Quantization scales permuted/generated alongside expanded tokens.
 * @param[out] expanded_topk_weight_out Optional top-k weights in expanded-row order.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using MoeInitRoutingV4PlanFn = int (*)(const aclTensor* x, const aclTensor* expert_idx,
                                       const aclTensor* scale_optional, const aclTensor* offset_optional,
                                       const aclTensor* active_num_optional, const aclTensor* topk_weight_optional,
                                       int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode,
                                       int64_t expert_tokens_num_type, bool expert_tokens_num_flag,
                                       int64_t quant_mode, const aclIntArray* active_expert_range_optional,
                                       int64_t row_idx_type, const aclTensor* expanded_x_out,
                                       const aclTensor* expanded_row_idx_out,
                                       const aclTensor* expert_tokens_count_or_cumsum_out,
                                       const aclTensor* expanded_scale_out,
                                       const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size,
                                       aclOpExecutor** executor);

// 22 arguments.
/**
 * @brief aclnnGroupedMatmulV5: plan phase (Transformer).
 * @details Compute grouped matrix products, with optional bias, quantization and activation.
 * @note Engine requests FP8 E4M3 x, FP4 E2M1 weights, E8M0 activation/weight scales and BF16 outputs.
 * This specific mixed-precision tuple must be checked against the installed 950 kernel; listed
 * individual dtypes do not guarantee every combination. 910B lacks this FP4/MX path. Grouping uses
 * split_item=3, group_type=0, group_list_type=0.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] weight Expert weight tensor list; no automatic dense reinterpretation of FP4.
 * @param[in] bias_optional Optional selection correction or GEMM bias, as specified by the operator.
 * @param[in] scale_optional Optional quantization scales or grouped weight dequantization scales.
 * @param[in] offset_optional Optional quantization offsets/zero points.
 * @param[in] antiquant_scale_optional Optional weight-only dequantization scale list/tensor.
 * @param[in] antiquant_offset_optional Optional weight-only dequantization offset list/tensor.
 * @param[in] per_token_scale_optional Activation scale list for grouped GEMM, aligned with x.
 * @param[in] group_list_optional Device INT64 group boundaries/counts for the selected
 * group_list_type.
 * @param[in] activation_input_optional Optional activation backward input list.
 * @param[in] activation_quant_scale_optional Reserved activation scale list; pass null in this graph.
 * @param[in] activation_quant_offset_optional Reserved activation offset list; pass null in this
 * graph.
 * @param[in] split_item Output grouping selector; engine uses 3 for a single concatenated output.
 * @param[in] group_type Grouping axis: -1 none, 0 M, 2 K; N grouping is unsupported by the installed
 * header.
 * @param[in] group_list_type Group encoding: 0 cumulative ends, 1 lengths, 2 sparse key/value pairs.
 * @param[in] act_type Activation selector; engine uses 0 (no fused activation).
 * @param[in] tuning_config_optional Optional host tiling hints, including expected tokens per expert.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] activation_feature_out_optional Optional pre-activation intermediate output list.
 * @param[out] dyn_quant_scale_out_optional Optional scales for dynamically quantized activated output.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using GroupedMatmulV5PlanFn = int (*)(const aclTensorList* x, const aclTensorList* weight,
                                      const aclTensorList* bias_optional, const aclTensorList* scale_optional,
                                      const aclTensorList* offset_optional,
                                      const aclTensorList* antiquant_scale_optional,
                                      const aclTensorList* antiquant_offset_optional,
                                      const aclTensorList* per_token_scale_optional,
                                      const aclTensor* group_list_optional,
                                      const aclTensorList* activation_input_optional,
                                      const aclTensorList* activation_quant_scale_optional,
                                      const aclTensorList* activation_quant_offset_optional, int64_t split_item,
                                      int64_t group_type, int64_t group_list_type, int64_t act_type,
                                      aclIntArray* tuning_config_optional, aclTensorList* out,
                                      aclTensorList* activation_feature_out_optional,
                                      aclTensorList* dyn_quant_scale_out_optional, uint64_t* workspace_size,
                                      aclOpExecutor** executor);

/**
 * @brief aclnnSwigluMxQuant: plan phase (Transformer).
 * @details Apply optional clamping and SwiGLU, then produce MX-quantized activation blocks.
 * @note Engine uses BF16 input, FP8 E4M3 output and E8M0 scales with block-32 MX. Clamping and scale
 * selection are explicit attributes. The selected FP8/MX mode is not a 910B-compatible path.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] group_index_optional Optional group boundaries for grouped activation processing.
 * @param[in] activate_dim Axis containing the two SwiGLU halves.
 * @param[in] activate_left Whether to activate the left half before multiplication.
 * @param[in] swiglu_mode SwiGLU variant selector.
 * @param[in] clamp_limit Magnitude limit applied by the selected clamped SwiGLU variant.
 * @param[in] glu_alpha Activation slope coefficient.
 * @param[in] glu_bias Additive coefficient in the selected SwiGLU variant.
 * @param[in] group_mode Interpretation of group_index_optional.
 * @param[in] axis Quantization axis; -1 selects the final dimension.
 * @param[in] dst_type ACL destination dtype code for quantized values.
 * @param[in] round_mode_optional Optional host rounding-mode string; engine supplies rint.
 * @param[in] scale_alg Scale-selection algorithm; engine selects OCP MX.
 * @param[in] max_dtype_value Maximum representable target magnitude used by quantization.
 * @param[out] y_out Output values in the selected operation's output dtype/shape.
 * @param[out] mxscale_out Output E8M0 microscales associated with the quantized blocks.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using SwigluMxQuantPlanFn = int (*)(const aclTensor* x, const aclTensor* group_index_optional, int64_t activate_dim,
                                    bool activate_left, int64_t swiglu_mode, double clamp_limit, double glu_alpha,
                                    double glu_bias, int64_t group_mode, int64_t axis, int64_t dst_type,
                                    char* round_mode_optional, int64_t scale_alg, double max_dtype_value,
                                    const aclTensor* y_out, const aclTensor* mxscale_out, uint64_t* workspace_size,
                                    aclOpExecutor** executor);

/**
 * @brief aclnnMoeTokenUnpermute: plan phase (Transformer).
 * @details Restore token order and optionally sum weighted expert contributions.
 * @note Optional alternative: floating permuted tokens and probabilities with integer permutation
 * indices. Probability dtype/shape and padded_mode must match the selected implementation. This entry
 * is resolved but the single-token path uses explicit cast plus Matmul.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] permuted_tokens Expert outputs in routed/permuted token order.
 * @param[in] sorted_indices Integer inverse-permutation metadata.
 * @param[in] probs_optional Optional routing probabilities for weighted accumulation.
 * @param[in] padded_mode Whether permutation included padding to expert capacities.
 * @param[in] restore_shape_optional Optional host shape of the restored token tensor.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using MoeTokenUnpermutePlanFn = int (*)(const aclTensor* permuted_tokens, const aclTensor* sorted_indices,
                                        const aclTensor* probs_optional, bool padded_mode,
                                        const aclIntArray* restore_shape_optional, aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantV2: plan phase (Transformer).
 * @details Fuse grouped expert GEMM, SwiGLU and activation quantization.
 * @note Engine supplies FP8 activations, FP4 weights, E8M0 scales and FP8/E8M0 outputs. dequant_mode,
 * dequant_dtype and quant_mode must describe the same scheme. This 950-class path cannot be
 * substituted unchanged on 910B.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x Input activation tensor (or tensor list for grouped GEMM).
 * @param[in] weight Expert weight tensor list; no automatic dense reinterpretation of FP4.
 * @param[in] weight_scale Expert weight scale tensor list.
 * @param[in] weight_assist_matrix Optional weight-assist tensor list for the selected quantization
 * scheme.
 * @param[in] bias Optional additive bias, subject to the selected quantization mode.
 * @param[in] x_scale Activation quantization scales.
 * @param[in] smooth_scale Optional activation smoothing scale.
 * @param[in] group_list Device expert group boundaries/counts.
 * @param[in] dequant_mode Dequantization scheme selector; engine requests MX mode.
 * @param[in] dequant_dtype Dtype of the dequantized intermediate.
 * @param[in] quant_mode Activation quantization mode; must agree with scales and output dtype.
 * @param[in] group_list_type Group encoding: 0 cumulative ends, 1 lengths, 2 sparse key/value pairs.
 * @param[in] tuning_config_optional Optional host tiling hints, including expected tokens per expert.
 * @param[out] output Quantized fused-GEMM/SwiGLU output.
 * @param[out] output_scale Scales associated with the quantized output.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using GroupedMatmulSwigluQuantV2PlanFn =
    int (*)(const aclTensor* x, const aclTensorList* weight, const aclTensorList* weight_scale,
            const aclTensorList* weight_assist_matrix, const aclTensor* bias, const aclTensor* x_scale,
            const aclTensor* smooth_scale, const aclTensor* group_list, int64_t dequant_mode, int64_t dequant_dtype,
            int64_t quant_mode, int64_t group_list_type, const aclIntArray* tuning_config_optional,
            aclTensor* output, aclTensor* output_scale, uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingV3: plan phase (Transformer).
 * @details Fuse grouped expert product with weighted routing accumulation and optional shared input.
 * @note Optional entry not used: x2 is a single weight tensor, incompatible with scattered expert
 * slots. Dtype/scale permutations are mode-specific; integer/FP4 storage is never a dense BF16 alias.
 * Requires independent SoC validation.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract apply.
 * No descriptor or buffer may be released while queued work can still reference it.
 * @param[in] x1 Left input matrix.
 * @param[in] x2 Right input matrix/weight tensor.
 * @param[in] scale_optional Optional quantization scales or grouped weight dequantization scales.
 * @param[in] bias_optional Optional selection correction or GEMM bias, as specified by the operator.
 * @param[in] offset_optional Optional quantization offsets/zero points.
 * @param[in] antiquant_scale_optional Optional weight-only dequantization scale list/tensor.
 * @param[in] antiquant_offset_optional Optional weight-only dequantization offset list/tensor.
 * @param[in] pertoken_scale_optional Optional per-token activation dequantization scales.
 * @param[in] group_list_optional Device INT64 group boundaries/counts for the selected
 * group_list_type.
 * @param[in] shared_input_optional Optional shared-expert contribution to final routing output.
 * @param[in] logit_optional Optional per-expert routing weights.
 * @param[in] row_index_optional Optional mapping to original token rows.
 * @param[in] dtype Output ACL dtype code; must agree with the output descriptor.
 * @param[in] shared_input_weight Multiplier for the shared-expert contribution.
 * @param[in] shared_input_offset Offset used when combining the shared-expert input.
 * @param[in] transpose_x1 Whether to transpose the left matrix's last two axes.
 * @param[in] transpose_x2 Whether to transpose the right matrix's last two axes.
 * @param[in] group_list_type Group encoding: 0 cumulative ends, 1 lengths, 2 sparse key/value pairs.
 * @param[in] tuning_config_optional Optional host tiling hints, including expected tokens per expert.
 * @param[out] out Output device tensor or grouped output tensor list.
 * @param[out] workspace_size Returned required device workspace bytes; allocate before launch.
 * @param[out] executor Returned opaque execution plan; transfer to StaticOpSlot before launch.
 * @return 0 (ACLNN_SUCCESS); 161001 (ACLNN_ERR_PARAM_NULLPTR);
 * 161002 (ACLNN_ERR_PARAM_INVALID); 361001 (ACLNN_ERR_RUNTIME_ERROR).
 * See @ref aclnn_contract for meanings and asynchronous error handling.
 * @see AclnnLaunchFn
 */
using GroupedMatmulFinalizeRoutingV3PlanFn =
    int (*)(const aclTensor* x1, aclTensor* x2, const aclTensor* scale_optional, const aclTensor* bias_optional,
            const aclTensor* offset_optional, const aclTensor* antiquant_scale_optional,
            const aclTensor* antiquant_offset_optional, const aclTensor* pertoken_scale_optional,
            const aclTensor* group_list_optional, const aclTensor* shared_input_optional,
            const aclTensor* logit_optional, const aclTensor* row_index_optional, int64_t dtype,
            float shared_input_weight, int64_t shared_input_offset, bool transpose_x1, bool transpose_x2,
            int64_t group_list_type, const aclIntArray* tuning_config_optional, aclTensor* out,
            uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief Shared second-phase ABI for every imported ACLNN operator.
 * @details Enqueues the computation prepared by the corresponding PlanFn.
 * @param[in] workspace Device workspace base; null only when workspace_size is zero.
 * @param[in] workspace_size Allocated bytes, at least the size returned by the plan.
 * @param[in] executor Matching plan; StaticOpSlot owns its repeatable executor.
 * @param[in] stream ACL stream in the tensors' device context.
 * @note Return success means enqueue succeeded; synchronization can fail later.
 * @warning Keep workspace, tensors and executor alive until the stream drains.
 * Normal executors are consumed on launch; repeatable ones require explicit destruction.
 * @return 0 (ACLNN_SUCCESS), 161001 (ACLNN_ERR_PARAM_NULLPTR),
 * 161002 (ACLNN_ERR_PARAM_INVALID), 361001 (ACLNN_ERR_RUNTIME_ERROR), or runtime status.
 * @see aclnn_contract
 */
using AclnnLaunchFn = int (*)(void* workspace, uint64_t workspace_size, aclOpExecutor* executor, void* stream);

// ---------------------------------------------------------------------------
// OpTable
// ---------------------------------------------------------------------------

class OpTable {
 public:
  // Resolves every entry once. Does not throw when an op is missing; the
  // pipeline decides which absences are fatal for the path it selected.
  OpTable();

  const ResolvedOp& op(OpId id) const;
  bool available(OpId id) const { return op(id).available(); }

  // Throws naming every required op this toolkit does not export.
  void RequireAll(const std::vector<OpId>& ids) const;

  template <typename PlanFn>
  PlanFn plan_fn(OpId id) const {
    DSV4_REQUIRE(id != OpId::kFiaV5GetMaxWorkspace, "FIA maximum workspace helper has no public checked ABI");
    RequireAvailable(id);
    return reinterpret_cast<PlanFn>(op(id).plan);
  }

  AclnnLaunchFn launch_fn(OpId id) const;

  std::string DescribeInventory() const;

  // True when the aclnn libraries could be reached at all. False means the
  // binary is running without the CANN runtime on its loader path, which is a
  // different failure from a missing operator.
  bool runtime_reachable() const { return runtime_reachable_; }

 private:
  void Resolve(OpId id, const char* name, const char* role, bool required);
  void ResolveBare(OpId id, const char* name, const char* role, bool required);
  void RequireAvailable(OpId id) const;

  ResolvedOp ops_[static_cast<size_t>(OpId::kOpCount)];
  bool runtime_reachable_ = false;
};

// ---------------------------------------------------------------------------
// One pre-planned, address-swappable operator invocation
// ---------------------------------------------------------------------------
//
// The two-call ACLNN protocol normally consumes its executor on launch. With
// `aclSetAclOpExecutorRepeatable` the executor survives, and the only thing a
// later launch needs is the new addresses -- which is what makes a decode loop
// with zero descriptor creations and zero re-planning possible.
class StaticOpSlot {
 public:
  StaticOpSlot() = default;
  ~StaticOpSlot();
  void Reset() noexcept { executor_guard_.reset(); executor_ = nullptr; }

  StaticOpSlot(const StaticOpSlot&) = delete;
  StaticOpSlot& operator=(const StaticOpSlot&) = delete;

  // `PlanAclnnOp` has already run; this records the result and makes the
  // executor repeatable.
  void Adopt(OpId id, const char* label, uint64_t workspace_size, aclOpExecutor* executor);

  // Repoint input/output slot `index` of the retained executor at `address`.
  void SetAddress(size_t index, aclTensor* tensor, void* address) const;
  void SetTensorListAddress(size_t ir_index, size_t relative_index, aclTensorList* tensors, void* address) const;

  void Launch(const OpTable& table, void* workspace, void* stream) const;

  uint64_t workspace_size() const { return workspace_size_; }
  aclOpExecutor* executor() const { return executor_; }
  bool planned() const { return executor_ != nullptr; }
  const char* label() const { return label_; }
  OpId id() const { return id_; }

 private:
  OpId id_ = OpId::kOpCount;
  const char* label_ = "<unplanned>";
  uint64_t workspace_size_ = 0;
  aclOpExecutor* executor_ = nullptr;
  OpExecutorGuard executor_guard_;
};

const char* OpName(OpId id);

/// Enforce this engine's dense 2-D ND, homogeneous FP16/BF16 Matmul contract.
void ValidateDenseMatmul(const aclTensor* self, const aclTensor* mat2, aclTensor* out, int8_t cube_math_type);

// Runs a plan function and returns the workspace size, throwing with the op's
// name on a non-zero status. Variadic so every op keeps its real signature; the
// two protocol arguments are appended here, never at the call site.
template <typename PlanFn, typename... Args>
uint64_t PlanAclnnOp(const OpTable& table, OpId id, aclOpExecutor** executor, Args... args) {
  CheckForInterrupt();
  DSV4_REQUIRE(executor != nullptr, "null executor output parameter");
  *executor = nullptr;
  if constexpr (std::is_same_v<PlanFn, MatmulPlanFn>) ValidateDenseMatmul(args...);
  PlanFn plan = table.plan_fn<PlanFn>(id);
  uint64_t workspace_size = 0;
  const int status = plan(args..., &workspace_size, executor);
  if (status != 0) {
    OpExecutorGuard partial(*executor);
    *executor = nullptr;
    AclError error(OpName(id), __FILE__, __LINE__, status);
    throw error;
  }
  if (*executor == nullptr) {
    throw Dsv4Error(std::string(OpName(id)) + "GetWorkspaceSize returned no executor");
  }
  return workspace_size;
}

// One planned, address-swappable operator invocation. Shared by the pipeline
// orchestrator and the router engine; the stage vector holding these is born
// at full capacity because StaticOpSlot is not movable.
struct PipelineStage {
  const char* name = nullptr;
  OpId op = OpId::kOpCount;
  StaticOpSlot slot;
};

}  // namespace ascend_moe
