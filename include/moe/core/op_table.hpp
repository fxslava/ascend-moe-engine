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
  // Vendored arch35 (Ascend 950PR) DSV4 operators from third_party/ops_dsv4:
  // the mHC residual chain, the sparse-attention indexer front end in both
  // its forms, the token-level KV compressor, the shared-KV attention core
  // and the two cache epilogs. They resolve from libcust_opapi.so (CANN
  // builds) or libopapi_mock.so (mock builds), never from the toolkit's own
  // libopapi.
  kMhcPre,
  kMhcSinkhorn,
  kMhcPost,
  // The vllm-ascend decomposition of the SAME mHC mapping, kept alongside the
  // ops-transformer trio rather than replacing it: kHcPre is the fully fused
  // prologue (reciprocal RMS + mixing projection + Sinkhorn in one launch),
  // kHcPreInvRms + a mixing GEMM + kHcPreSinkhorn is the staged alternative,
  // and kHcPost is the BSHD residual combine.
  kHcPre,
  kHcPreInvRms,
  kHcPreSinkhorn,
  kHcPost,
  kQuantLightningIndexer,
  kCompressor,
  kVllmQuantLightningIndexer,
  kKvQuantSparseAttnSharedkv,
  kKvCompressEpilog,
  kIndexerCompressEpilogV2,
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

// ---------------------------------------------------------------------------
// Vendored ops-transformer arch35 operators (third_party/ops_dsv4).
// Prototypes: include/moe/ops/aclnn_dsv4_vendor_ops.h, transcribed from the
// vendored op_host/op_api/aclnn_<op>.h headers.
// ---------------------------------------------------------------------------

/**
 * @brief aclnnMhcPre: plan phase (vendored arch35 custom op).
 * @details Fold the stacked mHC states x [T,N,D] through the mixing weights
 * phi [N^2+2N, ND] into hIn [T,D] plus the routing state hPost [T,N] and the
 * residual mapping hRes [T,N,N].
 * @note Engine geometry: N=4, D=4096, TND layout, BF16 x/hIn, FP32 state.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Stacked mHC states, [T,N,D], BF16/FP16.
 * @param[in] phi Mixing weights, [N^2+2N, N*D], FP32.
 * @param[in] alpha Gain triple [3], FP32.
 * @param[in] bias Mixing bias [N^2+2N], FP32.
 * @param[in] gamma_optional Per-stream normalizer [N,D], FP32; null allowed.
 * @param[in] norm_eps RMS normalization epsilon.
 * @param[in] hc_eps Hyper-connection stability epsilon.
 * @param[out] h_in Layer input [T,D], dtype of x.
 * @param[out] h_post Post-mapping state [T,N], FP32.
 * @param[out] h_res Residual mapping matrix [T,N,N], FP32.
 * @param[out] inv_rms_optional Reciprocal RMS [T], FP32; null allowed.
 * @param[out] h_mix_optional Pre-norm mixing state [T,N^2+2N], FP32; null allowed.
 * @param[out] h_pre_optional Pre-mapping state [T,N], FP32; null allowed.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using MhcPrePlanFn = int (*)(const aclTensor* x, const aclTensor* phi, const aclTensor* alpha, const aclTensor* bias,
                             const aclTensor* gamma_optional, double norm_eps, double hc_eps, aclTensor* h_in,
                             aclTensor* h_post, aclTensor* h_res, aclTensor* inv_rms_optional,
                             aclTensor* h_mix_optional, aclTensor* h_pre_optional, uint64_t* workspace_size,
                             aclOpExecutor** executor);

/**
 * @brief aclnnMhcSinkhorn: plan phase (vendored arch35 custom op).
 * @details Row/column-normalize hRes [T,N,N] into a doubly-stochastic matrix.
 * @note N in {4, 6, 8}; 1 <= num_iters <= 100; FP32 only. When either optional
 * output is null the op runs with outFlag 0. Repeatability hazard (manual
 * 4.31): the aclnn layer runs Contiguous(output), MhcSinkhorn writes into and
 * returns that tensor, then ViewCopy(kernelOut, output) -- for a contiguous
 * output that is a same-address self-copy and the executor may NOT be reused.
 * Handing a non-contiguous output view is the workaround: Contiguous then
 * allocates a distinct temp (src != dst) at the cost of one extra copy
 * launch; the planned aclnnMhcPreSinkhorn fusion removes the hazard entirely.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Square matrices to normalize, [T,N,N], FP32.
 * @param[in] eps Sinkhorn division guard.
 * @param[in] num_iters Iteration count in [1, 100].
 * @param[out] output Doubly-stochastic matrices, same shape/dtype as x.
 * @param[out] norm_out Optional norm state, FP32; bound together with sum_out.
 * @param[out] sum_out Optional row-sum state, FP32; bound together with norm_out.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using MhcSinkhornPlanFn = int (*)(const aclTensor* x, float eps, int64_t num_iters, aclTensor* output,
                                  aclTensor* norm_out, aclTensor* sum_out, uint64_t* workspace_size,
                                  aclOpExecutor** executor);

/**
 * @brief aclnnMhcPost: plan phase (vendored arch35 custom op).
 * @details x_next = (hRes)^T @ x + hOut * hPost after the attention/MLP layer.
 * @note Engine geometry: x/hOut/out [T,4,4096]/[T,4096] BF16, hRes [T,4,4] and
 * hPost [T,4] FP32, TND layout.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Layer input state, [T,N,D], BF16/FP16.
 * @param[in] h_res Doubly-stochastic mapping, [T,N,N], FP32.
 * @param[in] h_out Layer output, [T,D], dtype of x.
 * @param[in] h_post Post-mapping state, [T,N], FP32.
 * @param[out] out Next layer input, same shape/dtype as x.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using MhcPostPlanFn = int (*)(const aclTensor* x, const aclTensor* h_res, const aclTensor* h_out,
                              const aclTensor* h_post, aclTensor* out, uint64_t* workspace_size,
                              aclOpExecutor** executor);

// ---------------------------------------------------------------------------
// Vendored vllm-ascend arch35 fused mHC family (third_party/ops_dsv4/mhc/hc_*).
// Prototypes: include/moe/ops/aclnn_dsv4_vendor_ops.h, transcribed from the
// vendored op_host/op_api/aclnn_hc_*.h headers.
//
// The SAME mHC mapping as the mhc_* trio above, decomposed differently. The
// decisive structural difference is the Sinkhorn stage: MhcSinkhorn is a
// standalone in-place operator, which is what creates the manual-4.31
// same-address ViewCopy its wrapper has to elide. In this family the
// normalization is interior to HcPre / HcPreSinkhorn and never becomes a
// tensor at the aclnn layer, so the hazard cannot arise at all.
//
// Launch counts per layer, for choosing between them:
//   HcPre -> HcPost                                      2
//   MhcPre -> MhcSinkhorn -> MhcPost                      3
//   HcPreInvRms -> Matmul -> HcPreSinkhorn -> HcPost       4
// ---------------------------------------------------------------------------

/**
 * @brief aclnnHcPre: plan phase (vendored arch35 custom op).
 * @details The fully fused mHC prologue: the reciprocal-RMS reduction, the
 * mixing projection x.flatten(-2) @ hc_fn^T and the Sinkhorn normalization in
 * ONE launch, from the stacked states and the mixing weights alone.
 * @note Engine geometry: hc = hc_mult = 4, d = 4096 (7168 also accepted), BF16
 * x/y, FP32 weights and state. x is [bs,hc,d] or [b,s,hc,d].
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @warning Attr order: hc_eps precedes norm_eps, the reverse of the order the
 * two epsilons appear in aclnnMhcPre.
 * @param[in] x Stacked mHC states, [bs,hc,d] or [b,s,hc,d], BF16.
 * @param[in] hc_fn Mixing weights, [hc_mult^2+2*hc_mult, hc*d] = [24,16384], FP32.
 * @param[in] hc_scale Gain triple [3], FP32.
 * @param[in] hc_base Mixing bias [24], FP32.
 * @param[in] hc_mult Hyper-connection multiplicity; 4 only.
 * @param[in] hc_sinkhorn_iters Sinkhorn iteration count, in [1, 100].
 * @param[in] hc_eps Sinkhorn division guard.
 * @param[in] norm_eps RMS normalization epsilon.
 * @param[out] y Layer input, [bs,d] or [b,s,d], BF16.
 * @param[out] post Post-mapping state, [bs,hc_mult] or [b,s,hc_mult], FP32.
 * @param[out] comb_frag Combine fragment, [..,hc_mult,hc_mult], FP32.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using HcPrePlanFn = int (*)(const aclTensor* x, const aclTensor* hc_fn, const aclTensor* hc_scale,
                            const aclTensor* hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters, double hc_eps,
                            double norm_eps, const aclTensor* y, const aclTensor* post, const aclTensor* comb_frag,
                            uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnHcPreInvRms: plan phase (vendored arch35 custom op).
 * @details The prologue of the staged path: reduce the stacked states over
 * their last two axes into rsqrt = 1 / sqrt(mean(x^2) + epsilon), the FP32
 * normalizer aclnnHcPreSinkhorn consumes.
 * @note Required, not optional: it is the only producer of HcPreSinkhorn's
 * `rsqrt` input, and this engine is torch-free so there is no fallback.
 * The output drops the two reduced axes and keeps a trailing 1.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Stacked mHC states, FP32/FP16/BF16.
 * @param[in] epsilon Variance floor.
 * @param[out] y Reciprocal RMS scale, [bs,1] or [b,s,1], FP32.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using HcPreInvRmsPlanFn = int (*)(const aclTensor* x, double epsilon, const aclTensor* y, uint64_t* workspace_size,
                                  aclOpExecutor** executor);

/**
 * @brief aclnnHcPreSinkhorn: plan phase (vendored arch35 custom op).
 * @details The mixing projection fused with the Sinkhorn normalization: takes
 * the already-projected `mixes`, the `rsqrt` scale from aclnnHcPreInvRms, the
 * gain triple, the bias and the stacked states, and emits the layer input, the
 * post-mapping state and the combine fragment in one launch.
 * @note NOT a rename of aclnnMhcSinkhorn. MhcSinkhorn normalizes a square
 * matrix in place, and that in-place shape is exactly what makes its trailing
 * ViewCopy a potential manual-4.31 same-address self-copy. Here the
 * normalization never leaves the kernel, so no such copy stage exists --
 * the hazard is removed by construction, not elided.
 * @note `mixes` is at::linear(x.flatten(-2), hc_fn) upstream; the engine
 * produces the same [.., 24] rows with aclnnMatmul under its [K, N] contract.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] mixes Mixing state, [..,hc_mult^2+2*hc_mult] = [..,24], FP32.
 * @param[in] rsqrt Reciprocal RMS scale, [..,1], FP32.
 * @param[in] hc_scale Gain triple [3], FP32.
 * @param[in] hc_base Mixing bias [24], FP32.
 * @param[in] x Stacked mHC states, [bs,hc,d] or [b,s,hc,d], BF16.
 * @param[in] hc_mult Hyper-connection multiplicity; 4 only.
 * @param[in] hc_sinkhorn_iters Sinkhorn iteration count, in [1, 100].
 * @param[in] hc_eps Sinkhorn division guard.
 * @param[out] y Layer input, [bs,d] or [b,s,d], BF16.
 * @param[out] post Post-mapping state, [..,hc_mult], FP32.
 * @param[out] comb_frag Combine fragment, [..,hc_mult,hc_mult], FP32.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using HcPreSinkhornPlanFn = int (*)(const aclTensor* mixes, const aclTensor* rsqrt, const aclTensor* hc_scale,
                                    const aclTensor* hc_base, const aclTensor* x, int64_t hc_mult,
                                    int64_t hc_sinkhorn_iters, double hc_eps, const aclTensor* y,
                                    const aclTensor* post, const aclTensor* comb_frag, uint64_t* workspace_size,
                                    aclOpExecutor** executor);

/**
 * @brief aclnnHcPost: plan phase (vendored arch35 custom op).
 * @details y = comb^T @ residual + x * post -- the same residual combine as
 * aclnnMhcPost, carried in the BSHD layout instead of TND, and with no
 * attributes at all.
 * @note Layout: x [b,s,d], residual/y [b,s,hc,d], post [b,s,hc], comb
 * [b,s,hc,hc]. A TND token stream maps on as b = T, s = 1.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Layer output, [b,s,d], FP32/FP16/BF16.
 * @param[in] residual Residual streams, [b,s,hc,d], dtype of x.
 * @param[in] post Post-mapping state, [b,s,hc], FP32/FP16/BF16.
 * @param[in] comb Combine matrix, [b,s,hc,hc], dtype of post.
 * @param[out] y Next layer input, shape and dtype of residual.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using HcPostPlanFn = int (*)(const aclTensor* x, const aclTensor* residual, const aclTensor* post,
                             const aclTensor* comb, const aclTensor* y, uint64_t* workspace_size,
                             aclOpExecutor** executor);

/**
 * @brief aclnnQuantLightningIndexer: plan phase (vendored arch35 custom op).
 * @details Sparse-flash front end: top-k token selection over quantized
 * query/key correlation scores, emitting INT32 sparse indices.
 * @note Engine geometry: D=128, N1 in {16,24,32,64} (950PR), FP8 E4M3 or
 * HiFloat8 query/key with BF16 weights + FP32 dequant scales; PA_BSND key
 * layout with a paged block table. quant modes are 0 (per-token-head).
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] query Index query, [B,S1,N1,D] or [T1,N1,D].
 * @param[in] key Index key, layout per layout_key_optional, N2=1, D=128.
 * @param[in] weights Scoring weights, [B,S1,N1] or [T,N1], BF16/FP16.
 * @param[in] query_dequant_scale Query dequant scales, [B,S1,N1] or [T,N1].
 * @param[in] key_dequant_scale Key dequant scales, layout of key without D.
 * @param[in] actual_seq_lengths_query_optional Per-batch cumulative query
 * lengths, INT32; required for TND.
 * @param[in] actual_seq_lengths_key_optional Per-batch cumulative key lengths,
 * INT32; required for TND / PA_BSND.
 * @param[in] block_table_optional Paged-KV block mapping, INT32 [B, blocks].
 * @param[in] query_quant_mode Quantization mode; 0 = per-token-head.
 * @param[in] key_quant_mode Quantization mode; 0 = per-token-head.
 * @param[in] layout_query_optional Host layout string, "BSND" or "TND".
 * @param[in] layout_key_optional Host layout string, "BSND", "TND", "PA_BSND".
 * @param[in] sparse_count Top-k blocks retained, in [1, 2048].
 * @param[in] sparse_mode 0 dense mask, 3 right-down causal.
 * @param[in] pre_tokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] next_tokens Sparse window; INT64_MAX = unrestricted.
 * @param[out] out Selected sparse indices, INT32, [B,S1,N2,k] or [T,N2,k].
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using QuantLightningIndexerPlanFn = int (*)(const aclTensor* query, const aclTensor* key, const aclTensor* weights,
                                            const aclTensor* query_dequant_scale, const aclTensor* key_dequant_scale,
                                            const aclTensor* actual_seq_lengths_query_optional,
                                            const aclTensor* actual_seq_lengths_key_optional,
                                            const aclTensor* block_table_optional, int64_t query_quant_mode,
                                            int64_t key_quant_mode, char* layout_query_optional,
                                            char* layout_key_optional, int64_t sparse_count, int64_t sparse_mode,
                                            int64_t pre_tokens, int64_t next_tokens, const aclTensor* out,
                                            uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnCompressor: plan phase (vendored arch35 custom op).
 * @details Token-level KV compressor: project a window through wkv/wgate, add
 * the positional bias, pool cmpRatio tokens with a softmax-weighted reduction,
 * RMSNorm and apply the partial RoPE. cmp_ratio 4 is CSA, 128 is HCA.
 * @note state_cache_ref is a REF parameter and is updated in place; it gets no
 * ViewCopy, and cmp_kv_out is written through a distinct executor-owned
 * tensor, so the returned executor is reusable (see @ref aclnn_contract).
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] x Token states, [T,H] or [B,S,H], BF16/FP16.
 * @param[in] wkv Compressed-KV projection weight, dtype of x.
 * @param[in] wgate Pooling-gate projection weight, dtype of x.
 * @param[in,out] state_cache_ref Recurrent pooling state [blocks, blockSize, D]
 * FP32; updated in place.
 * @param[in] ape Absolute positional bias, FP32.
 * @param[in] norm_weight RMSNorm gain, 1-D.
 * @param[in] rope_sin RoPE sin table, rank of x.
 * @param[in] rope_cos RoPE cos table, rank of x.
 * @param[in] state_block_table_optional Paged state-cache map, INT32.
 * @param[in] cu_seqlens_optional Cumulative sequence lengths, INT32.
 * @param[in] seqused_optional Used length per batch, INT32.
 * @param[in] start_pos_optional Window start per batch, INT32.
 * @param[in] rope_head_dim RoPE head dimension (64).
 * @param[in] cmp_ratio Compression ratio; 4 or 128.
 * @param[in] coff Output-channel multiplier.
 * @param[in] norm_eps RMSNorm epsilon.
 * @param[in] rotary_mode RoPE interleaving mode.
 * @param[in] cache_mode State-cache addressing mode.
 * @param[in] state_cache_stride_dim0 Element stride of state-cache axis 0.
 * @param[out] cmp_kv_out Compressed KV rows, dtype of x.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using CompressorPlanFn = int (*)(const aclTensor* x, const aclTensor* wkv, const aclTensor* wgate,
                                 aclTensor* state_cache_ref, const aclTensor* ape, const aclTensor* norm_weight,
                                 const aclTensor* rope_sin, const aclTensor* rope_cos,
                                 const aclTensor* state_block_table_optional, const aclTensor* cu_seqlens_optional,
                                 const aclTensor* seqused_optional, const aclTensor* start_pos_optional,
                                 int64_t rope_head_dim, int64_t cmp_ratio, int64_t coff, double norm_eps,
                                 int64_t rotary_mode, int64_t cache_mode, int64_t state_cache_stride_dim0,
                                 const aclTensor* cmp_kv_out, uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnVllmQuantLightningIndexer: plan phase (vendored arch35 custom
 * op).
 * @details The shared-KV form of the quantized lightning indexer. Against
 * aclnnQuantLightningIndexer it adds the scheduling-metadata input, the
 * cmp_ratio / return_values attributes, the explicit key strides and the
 * optional FP32 score output.
 * @note Engine geometry: D=128, FP8 E4M3 or HiFloat8 query/key with BF16
 * weights and FP32 dequant scales, PA_BSND key layout with a paged block
 * table. Both outputs are staged through distinct executor-owned tensors, so
 * the returned executor is reusable.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] query Index query, [B,S1,N1,D] or [T1,N1,D].
 * @param[in] key Index key per layout_key_optional, N2=1, D=128.
 * @param[in] weights Scoring weights, [B,S1,N1] or [T,N1], BF16/FP16.
 * @param[in] query_dequant_scale Query dequant scales.
 * @param[in] key_dequant_scale Key dequant scales, layout of key without D.
 * @param[in] actual_seq_lengths_query_optional Cumulative query lengths, INT32.
 * @param[in] actual_seq_lengths_key_optional Cumulative key lengths, INT32.
 * @param[in] block_table_optional Paged-KV block map, INT32.
 * @param[in] metadata_optional Precomputed scheduling metadata, INT32.
 * @param[in] query_quant_mode Quantization mode; 0 = per-token-head.
 * @param[in] key_quant_mode Quantization mode; 0 = per-token-head.
 * @param[in] layout_query_optional Host layout string, "BSND" or "TND".
 * @param[in] layout_key_optional Host layout string, "BSND"/"TND"/"PA_BSND".
 * @param[in] sparse_count Top-k blocks retained, in [1, 2048].
 * @param[in] sparse_mode 0 dense mask, 3 right-down causal.
 * @param[in] pre_tokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] next_tokens Sparse window; INT64_MAX = unrestricted.
 * @param[in] cmp_ratio Compression ratio of the key stream.
 * @param[in] return_values true to fill sparse_values_out.
 * @param[in] stride Element stride of the key axis 0.
 * @param[in] scale_stride Element stride of the key_dequant_scale axis 0.
 * @param[out] sparse_indices_out Selected sparse indices, INT32.
 * @param[out] sparse_values_out Selected scores, FP32; a [0] tensor when
 * return_values is false.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using VllmQuantLightningIndexerPlanFn =
    int (*)(const aclTensor* query, const aclTensor* key, const aclTensor* weights,
            const aclTensor* query_dequant_scale, const aclTensor* key_dequant_scale,
            const aclTensor* actual_seq_lengths_query_optional, const aclTensor* actual_seq_lengths_key_optional,
            const aclTensor* block_table_optional, const aclTensor* metadata_optional, int64_t query_quant_mode,
            int64_t key_quant_mode, char* layout_query_optional, char* layout_key_optional, int64_t sparse_count,
            int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_values,
            int64_t stride, int64_t scale_stride, const aclTensor* sparse_indices_out,
            const aclTensor* sparse_values_out, uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnKvQuantSparseAttnSharedkv: plan phase (vendored arch35 custom
 * op).
 * @details The shared-KV sparse attention core: one BF16 MQA query stream
 * attends to both halves of the quantized hybrid KV cache -- the uncompressed
 * (ori) stream and the compressor-produced compressed (cmp) stream -- each at
 * its own sparse indices, block table and mask mode, with optional attention
 * sinks folded into the softmax denominator.
 * @note At least one of ori_kv_optional and cmp_kv_optional must be bound. The
 * two paged caches are passed uncontiguized and their axis-0 strides travel as
 * ori_kv_stride0 / cmp_kv_stride0. Both outputs are staged through distinct
 * executor-owned tensors, so the returned executor is reusable.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in] q Query, BF16, layout per layout_q_optional.
 * @param[in] ori_kv_optional Uncompressed KV cache, FP8 E4M3.
 * @param[in] cmp_kv_optional Compressed KV cache, FP8 E4M3.
 * @param[in] ori_sparse_indices_optional INT32 indices into the ori stream.
 * @param[in] cmp_sparse_indices_optional INT32 indices into the cmp stream.
 * @param[in] ori_block_table_optional Paged block map for ori, INT32.
 * @param[in] cmp_block_table_optional Paged block map for cmp, INT32.
 * @param[in] cu_seqlens_q_optional Cumulative query lengths, INT32.
 * @param[in] cu_seqlens_ori_kv_optional Cumulative ori KV lengths, INT32.
 * @param[in] cu_seqlens_cmp_kv_optional Cumulative cmp KV lengths, INT32.
 * @param[in] seqused_q_optional Used query length per batch, INT32.
 * @param[in] seqused_kv_optional Used KV length per batch, INT32.
 * @param[in] sinks_optional Attention sink logits, FP32.
 * @param[in] metadata_optional Precomputed scheduling metadata, INT32.
 * @param[in] kv_quant_mode KV quantization mode.
 * @param[in] tile_size KV tile size (default 64).
 * @param[in] rope_head_dim RoPE head dimension (default 64).
 * @param[in] softmax_scale Logit scale.
 * @param[in] cmp_ratio Compression ratio of the cmp stream; 4 or 128.
 * @param[in] ori_mask_mode Mask mode for the ori stream (default 4).
 * @param[in] cmp_mask_mode Mask mode for the cmp stream (default 3).
 * @param[in] ori_win_left Sliding-window left bound.
 * @param[in] ori_win_right Sliding-window right bound.
 * @param[in] layout_q_optional Host layout string for q (default "BSND").
 * @param[in] layout_kv_optional Host layout string for KV (default "PA_ND").
 * @param[in] ori_kv_stride0 Element stride of the ori cache axis 0.
 * @param[in] cmp_kv_stride0 Element stride of the cmp cache axis 0.
 * @param[in] return_softmax_lse true to fill softmax_lse_out.
 * @param[out] attn_out Attention output, BF16, shape of q.
 * @param[out] softmax_lse_out Log-sum-exp, FP32; a [0] tensor when
 * return_softmax_lse is false.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using KvQuantSparseAttnSharedkvPlanFn =
    int (*)(const aclTensor* q, const aclTensor* ori_kv_optional, const aclTensor* cmp_kv_optional,
            const aclTensor* ori_sparse_indices_optional, const aclTensor* cmp_sparse_indices_optional,
            const aclTensor* ori_block_table_optional, const aclTensor* cmp_block_table_optional,
            const aclTensor* cu_seqlens_q_optional, const aclTensor* cu_seqlens_ori_kv_optional,
            const aclTensor* cu_seqlens_cmp_kv_optional, const aclTensor* seqused_q_optional,
            const aclTensor* seqused_kv_optional, const aclTensor* sinks_optional,
            const aclTensor* metadata_optional, int64_t kv_quant_mode, int64_t tile_size, int64_t rope_head_dim,
            double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
            int64_t ori_win_left, int64_t ori_win_right, char* layout_q_optional, char* layout_kv_optional,
            int64_t ori_kv_stride0, int64_t cmp_kv_stride0, bool return_softmax_lse, const aclTensor* attn_out,
            const aclTensor* softmax_lse_out, uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnKvCompressEpilog: plan phase (vendored arch35 custom op).
 * @details Quantize the compressed-KV rows x and scatter them into the paged
 * FP8 cache at the slots slot_mapping names.
 * @note kv_compress_cache_ref is a REF parameter: it is the input AND the
 * output, is written in place, and receives no ViewCopy -- so the returned
 * executor is reusable under @ref aclnn_contract.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in,out] kv_compress_cache_ref Paged FP8 E5M2/E4M3 cache.
 * @param[in] x Compressed KV rows to store, BF16.
 * @param[in] slot_mapping Destination slot per row, INT32/INT64.
 * @param[in] quant_group_size Elements per quant group (128 on 950PR).
 * @param[in] quant_mode Quantization mode.
 * @param[in] round_scale 1 to round the scale to a power of two.
 * @param[in] layout Cache layout selector.
 * @param[in] block_stride Element stride of the cache block axis.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using KvCompressEpilogPlanFn = int (*)(aclTensor* kv_compress_cache_ref, const aclTensor* x,
                                       const aclTensor* slot_mapping, int64_t quant_group_size, int64_t quant_mode,
                                       int64_t round_scale, int64_t layout, int64_t block_stride,
                                       uint64_t* workspace_size, aclOpExecutor** executor);

/**
 * @brief aclnnIndexerCompressEpilogV2: plan phase (vendored arch35 custom op).
 * @details Scatter the indexer-side compressed rows x into the paged UINT8
 * indexer cache at the slots slot_mapping names.
 * @note indexer_compress_cache_ref is a REF parameter: it is the input AND the
 * output, is written in place, and receives no ViewCopy -- so the returned
 * executor is reusable under @ref aclnn_contract.
 * @warning The two-phase ownership and SoC constraints in @ref aclnn_contract
 * apply. Resolves from libcust_opapi.so, not the toolkit's libopapi.
 * @param[in,out] indexer_compress_cache_ref Paged UINT8 indexer cache.
 * @param[in] x Rows to store, FP16 or BF16.
 * @param[in] slot_mapping Destination slot per row, INT32.
 * @param[in] layout Cache layout selector.
 * @param[in] block_stride Element stride of the cache block axis.
 * @param[out] workspace_size Returned device workspace bytes.
 * @param[out] executor Returned execution plan.
 * @return 0 (ACLNN_SUCCESS); 161001 (NULLPTR); 161002 (INVALID); 361001
 * (RUNTIME_ERROR).
 * @see AclnnLaunchFn
 */
using IndexerCompressEpilogV2PlanFn = int (*)(aclTensor* indexer_compress_cache_ref, const aclTensor* x,
                                              const aclTensor* slot_mapping, int64_t layout, int64_t block_stride,
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
