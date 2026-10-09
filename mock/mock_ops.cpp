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

// The aclnn operator surface of libopapi_mock.
//
// Every GetWorkspaceSize stub is a CONTRACT VALIDATOR: it asserts the shapes,
// dtypes and scalar flags the DeepSeek-V4 Flash decode graph must produce,
// against the constants pinned by the checkpoint's config.json
// (F:\AI\models\DeepSeek-V4-Flash) and the operator semantics in
// F:\ops-transformer:
//
//   * moe_gating_top_k: y = [rows, k], expertIdx = [rows, k] (infershape)
//   * moe_init_routing_v4: the expert-token cumsum/count output is DT_INT64
//   * grouped_matmul: splitItem=3 (single output), groupType=0 (M groups)
//
// Execution stubs are NO-OPS returning ACL_SUCCESS: no buffer is touched, so
// the mock validates the graph's form, not its numerics. Numerical oracles
// live in tools/dsv4_moe_runtime (host references) and on the device.

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"  // aclrtStream for the launch stubs

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"
#include "runtime_faults.hpp"

// The dsv4 headers are NOT included here: this library is standalone, so the
// contract constants below are the mock's own transcription of
// DeepSeek-V4-Flash config.json. A constant that drifts from the checkpoint
// is a bug in this file, and the test suite cross-checks the C++ side
// separately.
namespace ascend_moe {
namespace mock {
namespace {

// DeepSeek-V4 Flash topology (config.json).
constexpr int64_t kExpertNum = 256;         // n_routed_experts
constexpr int64_t kTopK = 6;                // num_experts_per_tok
constexpr int64_t kHidden = 4096;           // hidden_size
constexpr int64_t kIntermediate = 2048;     // moe_intermediate_size
constexpr double kRoutedScaling = 1.5;      // routed_scaling_factor
constexpr int64_t kNormTypePreNormalized = -1;  // sqrtsoftplus runs decomposed
constexpr int64_t kRenormL1 = 1;                // norm_topk_prob = true
constexpr int64_t kNumHeads = 64;           // num_attention_heads
constexpr int64_t kRoutedScaleBlock = 32;   // OCP microscale block

constexpr uint64_t kWorkspaceElementwise = 32u << 10;
constexpr uint64_t kWorkspaceNorm = 16u << 10;
constexpr uint64_t kWorkspaceGating = 64u << 10;
constexpr uint64_t kWorkspaceRouting = 64u << 10;
constexpr uint64_t kWorkspaceAttention = 1u << 20;
constexpr uint64_t kWorkspaceAttentionMax = 8u << 20;
constexpr uint64_t kWorkspaceGmmBase = 256u << 10;
constexpr uint64_t kWorkspaceMhcBase = 64u << 10;
constexpr uint64_t kWorkspaceIndexerBase = 128u << 10;

// DeepSeek-V4 Flash mHC geometry: four hyper-connection streams over the
// 4096-wide hidden state (kHidden above), TND token layout.
constexpr int64_t kNhcStreams = 4;

uint64_t Align4k(uint64_t bytes) { return (bytes + 4095) & ~4095ull; }

// Row-major contiguity of a mock tensor's view, the property l0op::ViewCopy
// and l0op::Contiguous branch on in the real aclnn layer.
bool IsContiguous(const MockAclTensor* tensor) {
  if (tensor->strides.size() != tensor->shape.size() || tensor->shape.empty()) {
    return false;
  }
  int64_t expected = 1;
  for (size_t index = tensor->shape.size(); index-- > 0;) {
    if (tensor->strides[index] != expected) {
      return false;
    }
    expected *= tensor->shape[index];
  }
  return true;
}

// ---------------------------------------------------------------------------
// The vendored operators' manual-4.31 repeatability ledger
// ---------------------------------------------------------------------------
// Operator-library manual 4.31 makes an executor non-reusable if it holds a
// same-address ViewCopy, which matters here because the static op-slot table
// plans each vendored call once and relaunches it per token with
// aclSetTensorAddr. Every vendored wrapper in third_party/ops_dsv4 is written
// (or, for mhc_sinkhorn, patched) to the same rule, and these counters are how
// the mock suite holds that rule:
//
//   g_vendor_selfcopy_hazards   a plan that WOULD issue a same-address copy.
//                               The invariant is that this stays 0 across the
//                               whole vendored set.
//   g_sinkhorn_selfcopy_elided  the mhc_sinkhorn patch firing. Upstream copies
//                               unconditionally, and for a CONTIGUOUS output
//                               l0op::Contiguous is the identity, so
//                               MhcSinkhorn returns the caller tensor and the
//                               trailing ViewCopy(kernelOut, output) is a
//                               same-address self-copy. The patched wrapper
//                               skips the copy when kernelOut == output, which
//                               is counted here. A NON-contiguous output still
//                               takes the copy, where Contiguous allocated a
//                               distinct temp and src != dst already holds.
//   g_ref_output_plans          plans of an operator whose output IS one of
//                               its inputs (Compressor.stateCache and the two
//                               cache epilogs). These never stage a copy at
//                               all, so they cannot contribute a hazard; the
//                               tally only proves the tests exercised them.
int g_vendor_selfcopy_hazards = 0;
int g_sinkhorn_selfcopy_elided = 0;
int g_ref_output_plans = 0;

void RecordRefOutputPlan() { ++g_ref_output_plans; }

// Captures tensors into a fresh executor in IR order. Null tensor arguments
// still occupy their slot (ACLNN numbers optional tensors that were bound as
// null out of the IR; a later aclSetTensorAddr on such a slot is a product
// bug the strict path reports). List arguments pass their list handle: the
// slot holds it, and aclSetDynamicTensorAddr works through the caller's own
// list handle, so only the plain-index tally reads these.
aclOpExecutor* NewExecutor(const char* name, std::initializer_list<const void*> tensors) {
  auto* executor = new MockAclOpExecutor();
  executor->op_name = name;
  executor->tensors.reserve(tensors.size());
  for (const void* tensor : tensors) {
    executor->tensors.push_back(reinterpret_cast<aclTensor*>(const_cast<void*>(tensor)));
  }
  return reinterpret_cast<aclOpExecutor*>(executor);
}

// -- validator helpers -------------------------------------------------------

#define MOCK_REQUIRE(condition, message)      \
  do {                                        \
    if (!(condition)) {                       \
      return MockContractFailure(message);    \
    }                                         \
  } while (false)

bool Is2D(const MockAclTensor* tensor) { return tensor->shape.size() == 2; }

bool IsFloat(const MockAclTensor* tensor) {
  return tensor->dtype == ACL_FLOAT32 || tensor->dtype == ACL_BF16 || tensor->dtype == ACL_FLOAT16;
}

std::string ShapeOf(const MockAclTensor* tensor) {
  std::ostringstream text;
  text << "[";
  for (size_t index = 0; index < tensor->shape.size(); ++index) {
    if (index != 0) {
      text << ", ";
    }
    text << tensor->shape[index];
  }
  text << "]";
  return text.str();
}

// The five DeepSeek-V4 Flash contract validators -----------------------------

// aclnnSoftplus / aclnnSqrt: scores over [tokens, 256], float dtype.
aclnnStatus ValidateScoringStage(const char* name, const aclTensor* self, const aclTensor* out,
                                 uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* y = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && y != nullptr, std::string(name) + ": bad tensor handle");
  MOCK_REQUIRE(Is2D(x) && x->dim(1) == kExpertNum,
               std::string(name) + ": x must be [tokens, 256], got " + ShapeOf(x));
  MOCK_REQUIRE(IsFloat(x), std::string(name) + ": x must be FP32/BF16/FP16");
  MOCK_REQUIRE(y->same_shape_as(*x), std::string(name) + ": out must match x shape " + ShapeOf(x) +
                                          ", got " + ShapeOf(y));
  MOCK_REQUIRE(y->dtype == x->dtype, std::string(name) + ": out dtype must equal x dtype");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor(name, {self, out});
  return 0;
}

}  // namespace

// exposed for the test
aclnnStatus MockValidateGatingForTest(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                                      int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                      int64_t norm_type, double routed_scaling_factor, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out);
aclnnStatus MockValidateRoutingForTest(const aclTensor* expert_idx, int64_t expert_num,
                                       const aclTensor* group_list_out);
aclnnStatus MockValidateGmmForTest(const aclTensorList* weight, const aclTensorList* scale_optional,
                                   int64_t split_item, int64_t group_type);

namespace {

// aclnnMoeGatingTopKV2 over pre-normalized sqrtsoftplus scores.
aclnnStatus ValidateGating(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                           int64_t group_count, int64_t group_select_mode, int64_t renorm, int64_t norm_type,
                           double routed_scaling_factor, const aclTensor* y_out, const aclTensor* expert_idx_out) {
  const MockAclTensor* scores = AsMockTensor(x);
  const MockAclTensor* bias = bias_optional == nullptr ? nullptr : AsMockTensor(bias_optional);
  const MockAclTensor* y = AsMockTensor(y_out);
  const MockAclTensor* idx = AsMockTensor(expert_idx_out);
  MOCK_REQUIRE(scores != nullptr && y != nullptr && idx != nullptr, "MoeGatingTopKV2: bad tensor handle");
  MOCK_REQUIRE(Is2D(scores) && scores->dim(1) == kExpertNum,
               "MoeGatingTopKV2: x must be [tokens, 256] (pre-normalized scores), got " + ShapeOf(scores));
  MOCK_REQUIRE(IsFloat(scores), "MoeGatingTopKV2: x must be FP32/BF16");
  MOCK_REQUIRE(k == kTopK, "MoeGatingTopKV2: k must be 6 (num_experts_per_tok), got " + std::to_string(k));
  MOCK_REQUIRE(group_count == 1,
               "MoeGatingTopKV2: groupCount must be 1 (noaux_tc is not group-constrained), got " +
                   std::to_string(group_count));
  MOCK_REQUIRE(norm_type == kNormTypePreNormalized,
               "MoeGatingTopKV2: normType must be -1 (scores arrive pre-normalized from the decomposed "
               "sqrtsoftplus chain), got " +
                   std::to_string(norm_type));
  MOCK_REQUIRE(renorm == kRenormL1,
               "MoeGatingTopKV2: renorm must be 1 (norm_topk_prob: top-k weights sum 1.0 before scaling), got " +
                   std::to_string(renorm));
  MOCK_REQUIRE(routed_scaling_factor == kRoutedScaling, "MoeGatingTopKV2: routedScalingFactor must be 1.5, got " +
                                                            std::to_string(routed_scaling_factor));
  const int64_t tokens = scores->dim(0);
  MOCK_REQUIRE(Is2D(y) && y->dim(0) == tokens && y->dim(1) == k,
               "MoeGatingTopKV2: yOut must be [tokens, 6], got " + ShapeOf(y));
  MOCK_REQUIRE(Is2D(idx) && idx->dim(0) == tokens && idx->dim(1) == k,
               "MoeGatingTopKV2: expertIdxOut must be [tokens, 6], got " + ShapeOf(idx));
  MOCK_REQUIRE(idx->dtype == ACL_INT32 || idx->dtype == ACL_INT64,
               "MoeGatingTopKV2: expertIdxOut must be INT32 or INT64");
  MOCK_REQUIRE(bias == nullptr || (Is2D(bias) && bias->dim(0) == kExpertNum && bias->dim(1) == 1) ||
                   (bias->shape.size() == 1 && bias->dim(0) == kExpertNum),
               "MoeGatingTopKV2: bias must cover 256 experts, got " + ShapeOf(bias));
  (void)k_group;
  (void)group_select_mode;
  return 0;
}

// aclnnMoeInitRoutingV4: the device cumsum groupList and its non-aliasing.
aclnnStatus ValidateRouting(const aclTensor* expert_idx, int64_t expert_num,
                            const aclTensor* group_list_out) {
  const MockAclTensor* idx = AsMockTensor(expert_idx);
  const MockAclTensor* cumsum = AsMockTensor(group_list_out);
  MOCK_REQUIRE(idx != nullptr && cumsum != nullptr, "MoeInitRoutingV4: bad tensor handle");
  MOCK_REQUIRE(cumsum->shape.size() == 1 && cumsum->dim(0) == expert_num,
               "MoeInitRoutingV4: groupListOut must be [expert_num] in cumsum mode (expertTokensNumType 0), got " +
                   ShapeOf(cumsum) + " for expert_num " + std::to_string(expert_num));
  MOCK_REQUIRE(cumsum->dtype == ACL_INT64, "MoeInitRoutingV4: groupListOut must be INT64 (the op infershape forces "
                                           "DT_INT64), got dtype code " +
                                               std::to_string(static_cast<int>(cumsum->dtype)));
  // Non-aliasing: the expert ids the router produced and the cumsum the
  // dispatcher writes must not share a single byte.
  if (idx->device_addr != nullptr && cumsum->device_addr != nullptr) {
    const uintptr_t idx_base = reinterpret_cast<uintptr_t>(idx->device_addr);
    const uintptr_t sum_base = reinterpret_cast<uintptr_t>(cumsum->device_addr);
    MOCK_REQUIRE(!MockPartiallyOverlaps(idx_base, idx->total_bytes, sum_base, cumsum->total_bytes),
                 "MoeInitRoutingV4: expertIdx and groupListOut alias the same bytes");
  }
  return 0;
}

// aclnnGroupedMatmulV5: the FP4/UE8M0 expert GEMM contract.
aclnnStatus ValidateGmm(const aclTensorList* weight, const aclTensorList* scale_optional, int64_t split_item,
                        int64_t group_type) {
  const MockAclTensorList* weights = AsMockTensorList(weight);
  const MockAclTensorList* scales = scale_optional == nullptr ? nullptr : AsMockTensorList(scale_optional);
  MOCK_REQUIRE(weights != nullptr, "GroupedMatmulV5: bad weight list handle");
  MOCK_REQUIRE(split_item == 3, "GroupedMatmulV5: splitItem must be 3 (one output tensor spanning all groups), got " +
                                    std::to_string(split_item));
  MOCK_REQUIRE(group_type == 0, "GroupedMatmulV5: groupType must be 0 (groups on the M/token axis), got " +
                                    std::to_string(group_type));
  MOCK_REQUIRE(!weights->items.empty(), "GroupedMatmulV5: the weight list is empty");
  for (size_t index = 0; index < weights->items.size(); ++index) {
    const MockAclTensor* w = AsMockTensor(weights->items[index]);
    MOCK_REQUIRE(w != nullptr, "GroupedMatmulV5: bad weight handle at list index " + std::to_string(index));
    MOCK_REQUIRE(w->dtype == ACL_FP4X2_E2M1,
                 "GroupedMatmulV5: weights must be ACL_FLOAT4_E2M1 (40), got dtype code " +
                     std::to_string(static_cast<int>(w->dtype)));
    MOCK_REQUIRE(w->total_bytes == static_cast<size_t>(w->dim(0)) * static_cast<size_t>((w->dim(1) + 1) / 2),
                 "GroupedMatmulV5: weight storage must be rows x cols/2 packed nibbles");
    if (scales != nullptr) {
      const MockAclTensor* s = AsMockTensor(scales->items[index]);
      MOCK_REQUIRE(s != nullptr, "GroupedMatmulV5: bad scale handle at list index " + std::to_string(index));
      MOCK_REQUIRE(s->dtype == ACL_FLOAT8_E8M0,
                   "GroupedMatmulV5: scales must be ACL_FLOAT8_E8M0 (37), got dtype code " +
                       std::to_string(static_cast<int>(s->dtype)));
      MOCK_REQUIRE(s->dim(1) == (w->dim(1) + kRoutedScaleBlock - 1) / kRoutedScaleBlock,
                   "GroupedMatmulV5: scale columns must be one E8M0 byte per block-32 of the reduction axis");
    }
  }
  return 0;
}

}  // namespace

}  // namespace mock
}  // namespace ascend_moe

// ---------------------------------------------------------------------------
// C entry points. Signatures are transcribed verbatim from
// csrc/standalone/dsv4_aclnn_v5.hpp (which transcribes the CANN 9.2.0-beta.2
// headers), because dsv4_aclnn_v5.cpp dlsym-casts these symbols.
// ---------------------------------------------------------------------------

extern "C" {

using namespace ascend_moe::mock;

// -- dense / backbone --------------------------------------------------------

aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, double epsilon,
                                         const aclTensor* y_out, const aclTensor* rstd_out,
                                         uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mg = AsMockTensor(gamma);
  const MockAclTensor* my = AsMockTensor(y_out);
  MOCK_REQUIRE(mx != nullptr && mg != nullptr && my != nullptr, "RmsNorm: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "RmsNorm: y must match x shape");
  MOCK_REQUIRE(mg->elements() == mx->dim(mx->shape.size() - 1), "RmsNorm: gamma covers the last dimension");
  MOCK_REQUIRE(epsilon > 0.0, "RmsNorm: epsilon must be positive");
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnRmsNorm", {x, gamma, y_out, rstd_out});
  return 0;
}

aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma,
                                                       const aclTensor* beta, double epsilon, int64_t scale_alg,
                                                       char* round_mode, int64_t dst_type, bool output_rstd,
                                                       aclTensor* y_out, aclTensor* mxscale_out, aclTensor* rstd_out,
                                                       uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "RmsNormDynamicMxQuant: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "RmsNormDynamicMxQuant: y must match x shape");
  MOCK_REQUIRE(my->dtype == static_cast<aclDataType>(dst_type),
               "RmsNormDynamicMxQuant: y dtype must equal dstType");
  MOCK_REQUIRE(ms->shape.size() == mx->shape.size(), "RmsNormDynamicMxQuant: mxscale rank matches x");
  MOCK_REQUIRE(ms->dtype == ACL_FLOAT8_E8M0, "RmsNormDynamicMxQuant: mxscale must be E8M0");
  (void)gamma;
  (void)beta;
  (void)epsilon;
  (void)scale_alg;
  (void)round_mode;
  (void)output_rstd;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnRmsNormDynamicMxQuant", {x, gamma, beta, y_out, mxscale_out, rstd_out});
  return 0;
}


aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor* x, int64_t axis, char* round_mode_optional,
                                                int64_t dst_type, int64_t blocksize, int64_t scale_alg,
                                                const aclTensor* y_out, const aclTensor* mxscale_out,
                                                uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "DynamicMxQuant: bad tensor handle");
  MOCK_REQUIRE(my->same_shape_as(*mx), "DynamicMxQuant: y must match x shape");
  MOCK_REQUIRE(blocksize == 32, "DynamicMxQuant: the DSV4 microscale block is 32");
  (void)axis;
  (void)round_mode_optional;
  (void)dst_type;
  (void)scale_alg;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnDynamicMxQuant", {x, y_out, mxscale_out});
  return 0;
}

aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor* self, aclDataType dtype, aclTensor* out,
                                     uint64_t* workspace_size, aclOpExecutor** executor) {
  const auto* a = AsMockTensor(self);
  const auto* b = AsMockTensor(out);
  MOCK_REQUIRE(a && b && a->shape == b->shape && b->dtype == dtype, "Cast: shape/target dtype mismatch");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnCast", {self, out});
  return 0;
}

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
                                        int8_t cube_math_type, uint64_t* workspace_size,
                                        aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self);
  const MockAclTensor* b = AsMockTensor(mat2);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "Matmul: bad tensor handle");
  MOCK_REQUIRE(a->shape.size() == 2 && b->shape.size() == 2 && o->shape.size() == 2,
               "Matmul: 2-D operands expected");
  MOCK_REQUIRE(a->dim(1) == b->dim(0), "Matmul: expected [M,K] x [K,N]; no implicit transpose");
  MOCK_REQUIRE(o->dim(0) == a->dim(0) && o->dim(1) == b->dim(1), "Matmul: output must be [M,N]");
  MOCK_REQUIRE(a->dtype == b->dtype && a->dtype == o->dtype &&
               (a->dtype == ACL_BF16 || a->dtype == ACL_FLOAT16), "Matmul: homogeneous BF16/FP16 required");
  for (const auto* t : {a, b, o}) {
    MOCK_REQUIRE(t->format == ACL_FORMAT_ND && t->strides == std::vector<int64_t>({t->dim(1), 1}),
                 "Matmul: contiguous ND required");
    MOCK_REQUIRE(reinterpret_cast<uintptr_t>(t->device_addr) % 32 == 0, "Matmul: 32-byte alignment required");
  }
  (void)cube_math_type;
  *workspace_size = Align4k(static_cast<uint64_t>(a->dim(0)) * static_cast<uint64_t>(o->dim(1)) * 2) + (64u << 10);
  *executor = NewExecutor("aclnnMatmul", {self, mat2, out});
  return 0;
}

aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1_scale,
                                               const aclTensor* x2_scale, const aclTensor* y_scale,
                                               const aclTensor* x1_offset, const aclTensor* x2_offset,
                                               const aclTensor* y_offset, const aclTensor* bias, bool transpose_x1,
                                               bool transpose_x2, int64_t group_size, aclTensor* out,
                                               uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(x1);
  const MockAclTensor* b = AsMockTensor(x2);
  const MockAclTensor* o = AsMockTensor(out);
  const MockAclTensor* as = AsMockTensor(x1_scale);
  const MockAclTensor* bs = AsMockTensor(x2_scale);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "QuantMatmulV5: bad tensor handle");
  MOCK_REQUIRE(a->shape.size() == 2 && b->shape.size() == 2 && o->shape.size() == 2,
               "QuantMatmulV5: 2-D operands expected");
  MOCK_REQUIRE(transpose_x2, "QuantMatmulV5: the DSV4 weights are [N, K] with transposeX2");
  MOCK_REQUIRE(a->dim(1) == b->dim(1), "QuantMatmulV5: reduction (K) dimensions must agree, got " +
                                           std::to_string(a->dim(1)) + " vs " + std::to_string(b->dim(1)));
  MOCK_REQUIRE(o->dim(0) == a->dim(0) && o->dim(1) == b->dim(0), "QuantMatmulV5: out must be [m, n]");
  if (as != nullptr) {
    MOCK_REQUIRE(as->dtype == ACL_FLOAT8_E8M0, "QuantMatmulV5: activation scales are E8M0");
  }
  if (bs != nullptr) {
    MOCK_REQUIRE(bs->dtype == ACL_FLOAT8_E8M0, "QuantMatmulV5: weight scales are E8M0");
  }
  (void)y_scale;
  (void)x1_offset;
  (void)x2_offset;
  (void)y_offset;
  (void)bias;
  (void)transpose_x1;
  (void)group_size;
  *workspace_size = Align4k(static_cast<uint64_t>(a->dim(0)) * static_cast<uint64_t>(b->dim(0)) * 2) + (128u << 10);
  *executor = NewExecutor("aclnnQuantMatmulV5", {x1, x2, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset,
                                                 bias, out});
  return 0;
}

aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(aclTensor* query_ref, aclTensor* key_ref, const aclTensor* cos,
                                                     const aclTensor* sin, int64_t layout, char* rotary_mode,
                                                     uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* q = AsMockTensor(query_ref);
  const MockAclTensor* c = AsMockTensor(cos);
  const MockAclTensor* s = AsMockTensor(sin);
  MOCK_REQUIRE(q != nullptr && c != nullptr && s != nullptr, "ApplyRotaryPosEmbV2: bad tensor handle");
  MOCK_REQUIRE(c->same_shape_as(*s), "ApplyRotaryPosEmbV2: cos and sin must match");
  (void)key_ref;
  (void)layout;
  (void)rotary_mode;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnApplyRotaryPosEmbV2", {query_ref, key_ref, cos, sin});
  return 0;
}

aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
    const aclTensor* key, aclTensor* key_cache_ref, const aclTensor* slot_mapping, const aclTensor* value,
    aclTensor* value_cache_ref, const aclTensor* compress_lens_optional, const aclTensor* compress_seq_offset_optional,
    const aclTensor* seq_lens_optional, char* cache_mode_optional, char* scatter_mode_optional,
    const aclIntArray* strides_optional, const aclIntArray* offsets_optional, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  const MockAclTensor* k = AsMockTensor(key);
  const MockAclTensor* cache = AsMockTensor(key_cache_ref);
  const MockAclTensor* slots = AsMockTensor(slot_mapping);
  MOCK_REQUIRE(k != nullptr && cache != nullptr && slots != nullptr, "ScatterPaKvCache: bad tensor handle");
  MOCK_REQUIRE(slots->dtype == ACL_INT32 || slots->dtype == ACL_INT64, "ScatterPaKvCache: slot mapping is integral");
  MOCK_REQUIRE(cache->shape.size() >= 2, "ScatterPaKvCache: the cache is paged [blocks, block, ...]");
  (void)value;
  (void)value_cache_ref;
  (void)compress_lens_optional;
  (void)compress_seq_offset_optional;
  (void)seq_lens_optional;
  (void)cache_mode_optional;
  (void)scatter_mode_optional;
  (void)strides_optional;
  (void)offsets_optional;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnScatterPaKvCache",
                          {key, key_cache_ref, slot_mapping, value, value_cache_ref});
  return 0;
}

aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
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
    aclOpExecutor** executor) {
  const MockAclTensor* q = AsMockTensor(query);
  const MockAclTensorList* keys = AsMockTensorList(key);
  const MockAclTensorList* values = AsMockTensorList(value);
  const MockAclTensor* out = AsMockTensor(attention_out);
  MOCK_REQUIRE(q != nullptr && keys != nullptr && values != nullptr && out != nullptr,
               "FusedInferAttentionScoreV5: bad tensor handle");
  MOCK_REQUIRE(!keys->items.empty() && !values->items.empty(), "FusedInferAttentionScoreV5: empty key/value lists");
  MOCK_REQUIRE(num_heads == kNumHeads, "FusedInferAttentionScoreV5: 64 attention heads expected");
  MOCK_REQUIRE(scale_value > 0.0, "FusedInferAttentionScoreV5: a positive softmax scale is required");
  (void)pse_shift;
  (void)atten_mask;
  (void)actual_seq_lengths;
  (void)actual_seq_lengths_kv;
  (void)deq_scale1;
  (void)quant_scale1;
  (void)deq_scale2;
  (void)quant_scale2;
  (void)quant_offset2;
  (void)antiquant_scale;
  (void)antiquant_offset;
  (void)block_table;
  (void)query_padding_size;
  (void)kv_padding_size;
  (void)key_antiquant_scale;
  (void)key_antiquant_offset;
  (void)value_antiquant_scale;
  (void)value_antiquant_offset;
  (void)key_shared_prefix;
  (void)value_shared_prefix;
  (void)actual_shared_prefix_len;
  (void)query_rope;
  (void)key_rope;
  (void)key_rope_antiquant_scale;
  (void)dequant_scale_query;
  (void)learnable_sink;
  (void)q_start_idx;
  (void)kv_start_idx;
  (void)pre_tokens;
  (void)next_tokens;
  (void)input_layout;
  (void)num_key_value_heads;
  (void)sparse_mode;
  (void)inner_precise;
  (void)block_size;
  (void)antiquant_mode;
  (void)softmax_lse_flag;
  (void)key_antiquant_mode;
  (void)value_antiquant_mode;
  (void)query_quant_mode;
  (void)pse_type;
  (void)softmax_lse;
  *workspace_size = kWorkspaceAttention;
  // Every tensor argument is captured, nulls included, so the IR indices the
  // pipeline derives for aclSetTensorAddr (query=0, key list=1, value list=2,
  // ... keyRope late in the list) all land inside the captured range. The
  // rope index convention is a documented derive-and-verify item in the
  // product; the mock counts mismatches instead of failing on them.
  *executor = NewExecutor("aclnnFusedInferAttentionScoreV5",
                          {query, key, value, pse_shift, atten_mask, deq_scale1, quant_scale1, deq_scale2,
                           quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table,
                           query_padding_size, kv_padding_size, key_antiquant_scale, key_antiquant_offset,
                           value_antiquant_scale, value_antiquant_offset, key_shared_prefix, value_shared_prefix,
                           query_rope, key_rope, key_rope_antiquant_scale, dequant_scale_query, learnable_sink,
                           attention_out, softmax_lse});
  return 0;
}

aclnnStatus aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize(const aclTensor* query, const aclTensorList* key,
                                                                const aclTensorList* value, const aclTensor* pse_shift,
                                                                const aclTensor* atten_mask,
                                                                const aclIntArray* actual_seq_lengths,
                                                                const aclIntArray* actual_seq_lengths_kv,
                                                                const aclTensor* deq_scale1,
                                                                const aclTensor* quant_scale1,
                                                                const aclTensor* deq_scale2,
                                                                const aclTensor* quant_scale2,
                                                                const aclTensor* quant_offset2,
                                                                const aclTensor* antiquant_scale,
                                                                const aclTensor* antiquant_offset,
                                                                const aclTensor* block_table,
                                                                const aclTensor* query_padding_size,
                                                                const aclTensor* kv_padding_size,
                                                                const aclTensor* key_antiquant_scale,
                                                                const aclTensor* key_antiquant_offset,
                                                                const aclTensor* value_antiquant_scale,
                                                                const aclTensor* value_antiquant_offset,
                                                                const aclTensor* key_shared_prefix,
                                                                const aclTensor* value_shared_prefix,
                                                                const aclIntArray* actual_shared_prefix_len,
                                                                const aclTensor* query_rope,
                                                                const aclTensor* key_rope,
                                                                const aclTensor* key_rope_antiquant_scale,
                                                                const aclTensor* dequant_scale_query,
                                                                const aclTensor* learnable_sink,
                                                                const aclIntArray* q_start_idx,
                                                                const aclIntArray* kv_start_idx, int64_t num_heads,
                                                                double scale_value, int64_t pre_tokens,
                                                                int64_t next_tokens, char* input_layout,
                                                                int64_t num_key_value_heads, int64_t sparse_mode,
                                                                int64_t inner_precise, int64_t block_size,
                                                                int64_t antiquant_mode, bool softmax_lse_flag,
                                                                int64_t key_antiquant_mode,
                                                                int64_t value_antiquant_mode,
                                                                int64_t query_quant_mode, int64_t pse_type,
                                                                const aclTensor* attention_out,
                                                                const aclTensor* softmax_lse,
                                                                uint64_t* workspace_size,
                                                                aclOpExecutor** executor) {
  // The workspace upper-bound helper: same contract, bigger number. The
  // pipeline only reserves the max and destroys the executor, so the capture
  // is minimal and no parameter is inspected.
  (void)key;
  (void)value;
  (void)pse_shift;
  (void)atten_mask;
  (void)actual_seq_lengths;
  (void)actual_seq_lengths_kv;
  (void)deq_scale1;
  (void)quant_scale1;
  (void)deq_scale2;
  (void)quant_scale2;
  (void)quant_offset2;
  (void)antiquant_scale;
  (void)antiquant_offset;
  (void)block_table;
  (void)query_padding_size;
  (void)kv_padding_size;
  (void)key_antiquant_scale;
  (void)key_antiquant_offset;
  (void)value_antiquant_scale;
  (void)value_antiquant_offset;
  (void)key_shared_prefix;
  (void)value_shared_prefix;
  (void)actual_shared_prefix_len;
  (void)query_rope;
  (void)key_rope;
  (void)key_rope_antiquant_scale;
  (void)dequant_scale_query;
  (void)learnable_sink;
  (void)q_start_idx;
  (void)kv_start_idx;
  (void)num_heads;
  (void)scale_value;
  (void)pre_tokens;
  (void)next_tokens;
  (void)input_layout;
  (void)num_key_value_heads;
  (void)sparse_mode;
  (void)inner_precise;
  (void)block_size;
  (void)antiquant_mode;
  (void)softmax_lse_flag;
  (void)key_antiquant_mode;
  (void)value_antiquant_mode;
  (void)query_quant_mode;
  (void)pse_type;
  (void)attention_out;
  (void)softmax_lse;
  *workspace_size = kWorkspaceAttentionMax;
  *executor = NewExecutor("aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize", {query, key, value});
  return 0;
}

// -- elementwise --------------------------------------------------------------

aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                         aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* y = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && y != nullptr && y->same_shape_as(*x), "Sigmoid: out must match x");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnSigmoid", {self, out});
  return 0;
}

aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspace_size,
                                      aclOpExecutor** executor) {
  return ValidateScoringStage("aclnnSqrt", self, out, workspace_size, executor);
}

aclnnStatus aclnnSoftplusGetWorkspaceSize(const aclTensor* self, const aclScalar* beta, const aclScalar* threshold,
                                          aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclScalar* b = AsMockScalar(beta);
  const MockAclScalar* t = AsMockScalar(threshold);
  MOCK_REQUIRE(b != nullptr && t != nullptr, "aclnnSoftplus: beta/threshold scalars are required");
  MOCK_REQUIRE(b->as_f64() == 1.0, "aclnnSoftplus: the DSV4 sqrtsoftplus uses beta 1.0");
  MOCK_REQUIRE(t->as_f64() == 20.0, "aclnnSoftplus: the DSV4 sqrtsoftplus uses threshold 20.0");
  return ValidateScoringStage("aclnnSoftplus", self, out, workspace_size, executor);
}

aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                     uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self);
  const MockAclTensor* b = AsMockTensor(other);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && b != nullptr && o != nullptr, "Mul: bad tensor handle");
  MOCK_REQUIRE(o->same_shape_as(*a) && o->same_shape_as(*b), "Mul: out must match both operands");
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnMul", {self, other, out});
  return 0;
}

aclnnStatus aclnnInplaceAddGetWorkspaceSize(const aclTensor* self_ref, const aclTensor* other,
                                            const aclScalar* alpha, uint64_t* workspace_size,
                                            aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(self_ref);
  const MockAclTensor* b = AsMockTensor(other);
  MOCK_REQUIRE(a != nullptr && b != nullptr, "InplaceAdd: bad tensor handle");
  MOCK_REQUIRE(a->same_shape_as(*b), "InplaceAdd: operands must agree in shape");
  (void)alpha;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnInplaceAdd", {self_ref, other});
  return 0;
}

aclnnStatus aclnnSwiGluGetWorkspaceSize(const aclTensor* x, int64_t dim, const aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr, "SwiGlu: bad tensor handle");
  MOCK_REQUIRE(mx->dim(dim < 0 ? static_cast<size_t>(dim + static_cast<int64_t>(mx->shape.size()))
                               : static_cast<size_t>(dim)) % 2 == 0,
               "SwiGlu: the fused gate/up axis must be even");
  (void)my;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnSwiGlu", {x, out});
  return 0;
}

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                                        uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* x = AsMockTensor(self);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(x != nullptr && o != nullptr, "ArgMax: bad tensor handle");
  MOCK_REQUIRE(o->elements() == 1, "ArgMax: one greedy index out");
  (void)dim;
  (void)keepdim;
  *workspace_size = kWorkspaceNorm;
  *executor = NewExecutor("aclnnArgMax", {self, out});
  return 0;
}

// -- MoE ----------------------------------------------------------------------

aclnnStatus aclnnMoeGatingTopKV2GetWorkspaceSize(const aclTensor* x, const aclTensor* bias_optional,
                                                 const aclTensor* input_ids_optional,
                                                 const aclTensor* tid2eid_optional, int64_t k, int64_t k_group,
                                                 int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                                 int64_t norm_type, bool out_flag, double routed_scaling_factor,
                                                 double eps, const aclTensor* y_out, const aclTensor* expert_idx_out,
                                                 const aclTensor* out_out, uint64_t* workspace_size,
                                                 aclOpExecutor** executor) {
  const aclnnStatus status =
      ValidateGating(x, bias_optional, k, k_group, group_count, group_select_mode, renorm, norm_type,
                     routed_scaling_factor, y_out, expert_idx_out);
  if (status != 0) {
    return status;
  }
  MOCK_REQUIRE(eps > 0.0, "MoeGatingTopKV2: the renorm denominator needs a positive eps guard");
  MOCK_REQUIRE(!out_flag, "MoeGatingTopKV2: the full softmax output is not requested by the DSV4 graph");
  (void)input_ids_optional;
  (void)tid2eid_optional;
  (void)out_out;
  *workspace_size = kWorkspaceGating;
  *executor = NewExecutor("aclnnMoeGatingTopKV2", {x, bias_optional, input_ids_optional, tid2eid_optional, y_out,
                                                   expert_idx_out, out_out});
  return 0;
}

aclnnStatus aclnnMoeInitRoutingV4GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expert_idx, const aclTensor* scale_optional,
    const aclTensor* offset_optional, const aclTensor* active_num_optional, const aclTensor* topk_weight_optional,
    int64_t expert_capacity, int64_t expert_num, int64_t drop_pad_mode, int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag, int64_t quant_mode, const aclIntArray* active_expert_range_optional,
    int64_t row_idx_type, const aclTensor* expanded_x_out, const aclTensor* expanded_row_idx_out,
    const aclTensor* expert_tokens_count_or_cumsum_out, const aclTensor* expanded_scale_out,
    const aclTensor* expanded_topk_weight_out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* idx = AsMockTensor(expert_idx);
  const MockAclTensor* ex = AsMockTensor(expanded_x_out);
  const MockAclTensor* cumsum = AsMockTensor(expert_tokens_count_or_cumsum_out);
  MOCK_REQUIRE(mx != nullptr && idx != nullptr && ex != nullptr && cumsum != nullptr,
               "MoeInitRoutingV4: bad tensor handle");
  const aclnnStatus contract = ValidateRouting(expert_idx, expert_num, expert_tokens_count_or_cumsum_out);
  if (contract != 0) {
    return contract;
  }
  MOCK_REQUIRE(expert_capacity == 0, "MoeInitRoutingV4: dropless dispatch (expertCapacity 0)");
  MOCK_REQUIRE(drop_pad_mode == 0, "MoeInitRoutingV4: dropPadMode 0");
  MOCK_REQUIRE(expert_tokens_num_type == 0, "MoeInitRoutingV4: cumsum token counts (type 0) feed the GMM groupList");
  MOCK_REQUIRE(expert_tokens_num_flag, "MoeInitRoutingV4: the token-count output must be requested");
  const int64_t tokens = mx->dim(0);
  const int64_t top_k = idx->shape.size() == 2 ? idx->dim(1) : 1;
  MOCK_REQUIRE(ex->shape.size() == 2 && ex->dim(0) == tokens * top_k && ex->dim(1) == mx->dim(1),
               "MoeInitRoutingV4: expandedX must be [tokens*k, hidden]");
  (void)offset_optional;
  (void)active_num_optional;
  (void)topk_weight_optional;
  (void)quant_mode;
  (void)active_expert_range_optional;
  (void)row_idx_type;
  (void)expanded_row_idx_out;
  (void)expanded_scale_out;
  (void)expanded_topk_weight_out;
  *workspace_size = kWorkspaceRouting + static_cast<uint64_t>(expert_num) * sizeof(int64_t);
  *executor = NewExecutor("aclnnMoeInitRoutingV4",
                          {x, expert_idx, scale_optional, offset_optional, active_num_optional,
                           topk_weight_optional, expanded_x_out, expanded_row_idx_out,
                           expert_tokens_count_or_cumsum_out, expanded_scale_out, expanded_topk_weight_out});
  return 0;
}

aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight, const aclTensorList* bias_optional,
    const aclTensorList* scale_optional, const aclTensorList* offset_optional,
    const aclTensorList* antiquant_scale_optional, const aclTensorList* antiquant_offset_optional,
    const aclTensorList* per_token_scale_optional, const aclTensor* group_list_optional,
    const aclTensorList* activation_input_optional, const aclTensorList* activation_quant_scale_optional,
    const aclTensorList* activation_quant_offset_optional, int64_t split_item, int64_t group_type,
    int64_t group_list_type, int64_t act_type, aclIntArray* tuning_config_optional, aclTensorList* out,
    aclTensorList* activation_feature_out_optional, aclTensorList* dyn_quant_scale_out_optional,
    uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensorList* xs = AsMockTensorList(x);
  const MockAclTensor* groups = group_list_optional == nullptr ? nullptr : AsMockTensor(group_list_optional);
  MOCK_REQUIRE(xs != nullptr && !xs->items.empty(), "GroupedMatmulV5: empty x list");
  const aclnnStatus contract = ValidateGmm(weight, scale_optional, split_item, group_type);
  if (contract != 0) {
    return contract;
  }
  if (groups != nullptr) {
    MOCK_REQUIRE(groups->dtype == ACL_INT64, "GroupedMatmulV5: the groupList cumsum is INT64");
  }
  MOCK_REQUIRE(group_list_type == 0, "GroupedMatmulV5: groupListType 0 (per-expert token cumsum)");
  MOCK_REQUIRE(act_type == 0, "GroupedMatmulV5: actType NONE (the clamped SwiGLU is a separate op)");
  (void)bias_optional;
  (void)offset_optional;
  (void)antiquant_scale_optional;
  (void)antiquant_offset_optional;
  (void)per_token_scale_optional;
  (void)activation_input_optional;
  (void)activation_quant_scale_optional;
  (void)activation_quant_offset_optional;
  (void)tuning_config_optional;
  (void)activation_feature_out_optional;
  (void)dyn_quant_scale_out_optional;
  uint64_t rows = 0;
  uint64_t cols = 0;
  const MockAclTensor* x0 = AsMockTensor(xs->items[0]);
  const MockAclTensorList* outs = AsMockTensorList(out);
  const MockAclTensor* out0 = (outs != nullptr && !outs->items.empty()) ? AsMockTensor(outs->items[0]) : nullptr;
  if (x0 != nullptr && out0 != nullptr) {
    rows = static_cast<uint64_t>(x0->dim(0)) * xs->items.size();
    cols = static_cast<uint64_t>(out0->dim(1));
    // The tiling-formula stand-in: one aligned tile buffer per output matrix
    // plus the base, mirroring the shape of gmm tiling workspace formulas.
    *workspace_size = Align4k(rows * cols * 2) + kWorkspaceGmmBase;
  } else {
    *workspace_size = kWorkspaceGmmBase;
  }
  *executor = NewExecutor("aclnnGroupedMatmulV5", {x, weight, bias_optional, scale_optional, offset_optional,
                                                   antiquant_scale_optional, antiquant_offset_optional,
                                                   per_token_scale_optional, group_list_optional,
                                                   activation_input_optional, activation_quant_scale_optional,
                                                   activation_quant_offset_optional, out,
                                                   activation_feature_out_optional, dyn_quant_scale_out_optional});
  return 0;
}

aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* group_index_optional,
                                               int64_t activate_dim, bool activate_left, int64_t swiglu_mode,
                                               double clamp_limit, double glu_alpha, double glu_bias,
                                               int64_t group_mode, int64_t axis, int64_t dst_type,
                                               char* round_mode_optional, int64_t scale_alg,
                                               double max_dtype_value, const aclTensor* y_out,
                                               const aclTensor* mxscale_out, uint64_t* workspace_size,
                                               aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y_out);
  const MockAclTensor* ms = AsMockTensor(mxscale_out);
  MOCK_REQUIRE(mx != nullptr && my != nullptr && ms != nullptr, "SwigluMxQuant: bad tensor handle");
  MOCK_REQUIRE(mx->dim(1) == 2 * my->dim(1), "SwigluMxQuant: the gate/up input is twice the activation width");
  MOCK_REQUIRE(clamp_limit == 10.0, "SwigluMxQuant: the DSV4 swiglu clamp is 10.0");
  MOCK_REQUIRE(max_dtype_value == 448.0, "SwigluMxQuant: the FP8 E4M3 maximum is 448");
  (void)group_index_optional;
  (void)activate_dim;
  (void)activate_left;
  (void)swiglu_mode;
  (void)glu_alpha;
  (void)glu_bias;
  (void)group_mode;
  (void)axis;
  (void)dst_type;
  (void)round_mode_optional;
  (void)scale_alg;
  *workspace_size = kWorkspaceElementwise;
  *executor = NewExecutor("aclnnSwigluMxQuant", {x, group_index_optional, y_out, mxscale_out});
  return 0;
}

aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor* permuted_tokens,
                                                   const aclTensor* sorted_indices, const aclTensor* probs_optional,
                                                   bool padded_mode, const aclIntArray* restore_shape_optional,
                                                   aclTensor* out, uint64_t* workspace_size,
                                                   aclOpExecutor** executor) {
  const MockAclTensor* t = AsMockTensor(permuted_tokens);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(t != nullptr && o != nullptr, "MoeTokenUnpermute: bad tensor handle");
  (void)sorted_indices;
  (void)probs_optional;
  (void)padded_mode;
  (void)restore_shape_optional;
  *workspace_size = kWorkspaceRouting;
  *executor = NewExecutor("aclnnMoeTokenUnpermute", {permuted_tokens, sorted_indices, probs_optional, out});
  return 0;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weight_scale,
    const aclTensorList* weight_assist_matrix, const aclTensor* bias, const aclTensor* x_scale,
    const aclTensor* smooth_scale, const aclTensor* group_list, int64_t dequant_mode, int64_t dequant_dtype,
    int64_t quant_mode, int64_t group_list_type, const aclIntArray* tuning_config_optional, aclTensor* output,
    aclTensor* output_scale, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensorList* ws = AsMockTensorList(weight_scale);
  const MockAclTensor* ms = AsMockTensor(x_scale);
  const MockAclTensor* my = AsMockTensor(output);
  const MockAclTensor* mos = AsMockTensor(output_scale);
  const MockAclTensor* groups = group_list == nullptr ? nullptr : AsMockTensor(group_list);
  MOCK_REQUIRE(mx != nullptr && ws != nullptr && ms != nullptr && my != nullptr && mos != nullptr,
               "GroupedMatmulSwigluQuantV2: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 2 && my->shape.size() == 2 && mx->dim(1) == 2 * my->dim(1),
               "GroupedMatmulSwigluQuantV2: input is [tokens, hidden], output [tokens, intermediate]");
  for (size_t index = 0; index < ws->items.size(); ++index) {
    const MockAclTensor* w = AsMockTensor(ws->items[index]);
    MOCK_REQUIRE(w != nullptr && w->dtype == ACL_FLOAT8_E8M0,
                 "GroupedMatmulSwigluQuantV2: weight scales must be E8M0");
  }
  MOCK_REQUIRE(ms->dtype == ACL_FLOAT8_E8M0, "GroupedMatmulSwigluQuantV2: activation scales are E8M0");
  if (groups != nullptr) {
    MOCK_REQUIRE(groups->dtype == ACL_INT64, "GroupedMatmulSwigluQuantV2: the groupList cumsum is INT64");
  }
  MOCK_REQUIRE(dequant_mode == 1, "GroupedMatmulSwigluQuantV2: dequantMode 1 (MX block scales)");
  MOCK_REQUIRE(dequant_dtype == ACL_FLOAT8_E4M3FN, "GroupedMatmulSwigluQuantV2: dequant to FP8 E4M3");
  MOCK_REQUIRE(group_list_type == 0, "GroupedMatmulSwigluQuantV2: groupListType 0 (cumsum)");
  (void)weight;
  (void)weight_assist_matrix;
  (void)bias;
  (void)smooth_scale;
  (void)quant_mode;
  (void)tuning_config_optional;
  *workspace_size = Align4k(static_cast<uint64_t>(mx->dim(0)) * static_cast<uint64_t>(my->dim(1)) * 2) +
                    kWorkspaceGmmBase + (64u << 10);
  *executor = NewExecutor("aclnnGroupedMatmulSwigluQuantV2",
                          {x, weight, weight_scale, weight_assist_matrix, bias, x_scale, smooth_scale, group_list,
                           output, output_scale});
  return 0;
}

aclnnStatus aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize(
    const aclTensor* x1, aclTensor* x2, const aclTensor* scale_optional, const aclTensor* bias_optional,
    const aclTensor* offset_optional, const aclTensor* antiquant_scale_optional,
    const aclTensor* antiquant_offset_optional, const aclTensor* pertoken_scale_optional,
    const aclTensor* group_list_optional, const aclTensor* shared_input_optional, const aclTensor* logit_optional,
    const aclTensor* row_index_optional, int64_t dtype, float shared_input_weight, int64_t shared_input_offset,
    bool transpose_x1, bool transpose_x2, int64_t group_list_type, const aclIntArray* tuning_config_optional,
    aclTensor* out, uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* a = AsMockTensor(x1);
  const MockAclTensor* o = AsMockTensor(out);
  MOCK_REQUIRE(a != nullptr && o != nullptr, "GroupedMatmulFinalizeRoutingV3: bad tensor handle");
  (void)x2;
  (void)scale_optional;
  (void)bias_optional;
  (void)offset_optional;
  (void)antiquant_scale_optional;
  (void)antiquant_offset_optional;
  (void)pertoken_scale_optional;
  (void)group_list_optional;
  (void)shared_input_optional;
  (void)logit_optional;
  (void)row_index_optional;
  (void)dtype;
  (void)shared_input_weight;
  (void)shared_input_offset;
  (void)transpose_x1;
  (void)transpose_x2;
  (void)group_list_type;
  (void)tuning_config_optional;
  *workspace_size = kWorkspaceRouting;
  *executor = NewExecutor("aclnnGroupedMatmulFinalizeRoutingV3", {x1, x2, out});
  return 0;
}

// -- vendored ops-transformer arch35 operators ---------------------------------
//
// Contracts transcribed from the vendored trees' aclnn_*.cpp checks
// (third_party/ops_dsv4) at the DSV4 geometry: n_hc = 4 streams over a
// 4096-wide hidden state, TND layout, FP32 mHC state, BF16 activations.

// aclnnMhcPre: fold [T, 4, 4096] states through phi [24, 16384].
aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor* x, const aclTensor* phi, const aclTensor* alpha,
                                        const aclTensor* bias, const aclTensor* gamma_optional, double norm_eps,
                                        double hc_eps, aclTensor* h_in, aclTensor* h_post, aclTensor* h_res,
                                        aclTensor* inv_rms_optional, aclTensor* h_mix_optional,
                                        aclTensor* h_pre_optional, uint64_t* workspace_size,
                                        aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mp = AsMockTensor(phi);
  const MockAclTensor* ma = AsMockTensor(alpha);
  const MockAclTensor* mb = AsMockTensor(bias);
  const MockAclTensor* mg = gamma_optional == nullptr ? nullptr : AsMockTensor(gamma_optional);
  const MockAclTensor* min = AsMockTensor(h_in);
  const MockAclTensor* mpost = AsMockTensor(h_post);
  const MockAclTensor* mres = AsMockTensor(h_res);
  MOCK_REQUIRE(mx != nullptr && mp != nullptr && ma != nullptr && mb != nullptr && min != nullptr &&
                   mpost != nullptr && mres != nullptr,
               "MhcPre: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 3 && mx->dim(1) == kNhcStreams && mx->dim(2) == kHidden,
               "MhcPre: x must be [T, 4, 4096] (TND stacked mHC states), got " + ShapeOf(mx));
  MOCK_REQUIRE(mx->dtype == ACL_BF16 || mx->dtype == ACL_FLOAT16, "MhcPre: x must be BF16/FP16");
  const int64_t tokens = mx->dim(0);
  const int64_t mix_rows = kNhcStreams * kNhcStreams + 2 * kNhcStreams;  // n^2 + 2n = 24
  MOCK_REQUIRE(mp->shape.size() == 2 && mp->dim(0) == mix_rows &&
                   mp->dim(1) == kNhcStreams * kHidden,
               "MhcPre: phi must be [n^2+2n, nD] = [24, 16384] FP32, got " + ShapeOf(mp));
  MOCK_REQUIRE(mp->dtype == ACL_FLOAT32, "MhcPre: phi must be FP32");
  MOCK_REQUIRE(ma->elements() == 3 && ma->dtype == ACL_FLOAT32, "MhcPre: alpha must be [3] FP32");
  MOCK_REQUIRE(mb->elements() == mix_rows && mb->dtype == ACL_FLOAT32, "MhcPre: bias must be [24] FP32");
  MOCK_REQUIRE(mg == nullptr || (mg->shape.size() == 2 && mg->dim(0) == kNhcStreams && mg->dim(1) == kHidden &&
                                  mg->dtype == ACL_FLOAT32),
               "MhcPre: gammaOptional must be [4, 4096] FP32 or null");
  MOCK_REQUIRE(norm_eps > 0.0 && hc_eps > 0.0, "MhcPre: normEps and hcEps must be positive");
  MOCK_REQUIRE(min->shape.size() == 2 && min->dim(0) == tokens && min->dim(1) == kHidden &&
                   min->dtype == mx->dtype,
               "MhcPre: hIn must be [T, 4096] in the dtype of x");
  MOCK_REQUIRE(mpost->shape.size() == 2 && mpost->dim(0) == tokens && mpost->dim(1) == kNhcStreams &&
                   mpost->dtype == ACL_FLOAT32,
               "MhcPre: hPost must be [T, 4] FP32");
  MOCK_REQUIRE(mres->shape.size() == 3 && mres->dim(0) == tokens && mres->dim(1) == kNhcStreams &&
                   mres->dim(2) == kNhcStreams && mres->dtype == ACL_FLOAT32,
               "MhcPre: hRes must be [T, 4, 4] FP32");
  const MockAclTensor* mrms = inv_rms_optional == nullptr ? nullptr : AsMockTensor(inv_rms_optional);
  const MockAclTensor* mmix = h_mix_optional == nullptr ? nullptr : AsMockTensor(h_mix_optional);
  const MockAclTensor* mpre = h_pre_optional == nullptr ? nullptr : AsMockTensor(h_pre_optional);
  MOCK_REQUIRE(mrms == nullptr || (mrms->elements() == tokens && mrms->dtype == ACL_FLOAT32),
               "MhcPre: invRmsOptional must be [T] FP32 or null");
  MOCK_REQUIRE(mmix == nullptr || (mmix->shape.size() == 2 && mmix->dim(0) == tokens &&
                                   mmix->dim(1) == mix_rows && mmix->dtype == ACL_FLOAT32),
               "MhcPre: hMixOptional must be [T, 24] FP32 or null");
  MOCK_REQUIRE(mpre == nullptr || (mpre->shape.size() == 2 && mpre->dim(0) == tokens &&
                                   mpre->dim(1) == kNhcStreams && mpre->dtype == ACL_FLOAT32),
               "MhcPre: hPreOptional must be [T, 4] FP32 or null");
  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(kNhcStreams) *
                            static_cast<uint64_t>(kHidden) * 4) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnMhcPre",
                          {x, phi, alpha, bias, gamma_optional, h_in, h_post, h_res, inv_rms_optional,
                           h_mix_optional, h_pre_optional});
  return 0;
}

// aclnnMhcSinkhorn: doubly-stochastic normalization of [T, n, n].
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor* x, float eps, int64_t num_iters, aclTensor* output,
                                             aclTensor* norm_out, aclTensor* sum_out, uint64_t* workspace_size,
                                             aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(output);
  MOCK_REQUIRE(mx != nullptr && my != nullptr, "MhcSinkhorn: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 3, "MhcSinkhorn: x must be [T, n, n] (TND), got " + ShapeOf(mx));
  const int64_t n0 = mx->dim(1);
  const int64_t n1 = mx->dim(2);
  MOCK_REQUIRE(n0 == n1 && (n0 == 4 || n0 == 6 || n0 == 8),
               "MhcSinkhorn: n must be square and one of {4, 6, 8}, got " + ShapeOf(mx));
  MOCK_REQUIRE(mx->dtype == ACL_FLOAT32, "MhcSinkhorn: x must be FP32");
  MOCK_REQUIRE(eps > 0.0f, "MhcSinkhorn: eps must be positive");
  MOCK_REQUIRE(num_iters >= 1 && num_iters <= 100,
               "MhcSinkhorn: numIters must be in [1, 100], got " + std::to_string(num_iters));
  MOCK_REQUIRE(my->same_shape_as(*mx) && my->dtype == ACL_FLOAT32, "MhcSinkhorn: output must match x [T, n, n] FP32");
  // The optional norm/sum pair binds together: either both are present or the
  // operator runs with outFlag 0.
  const MockAclTensor* mnorm = norm_out == nullptr ? nullptr : AsMockTensor(norm_out);
  const MockAclTensor* msum = sum_out == nullptr ? nullptr : AsMockTensor(sum_out);
  MOCK_REQUIRE((mnorm == nullptr) == (msum == nullptr),
               "MhcSinkhorn: normOut and sumOut must be bound together (outFlag is 0 or 1)");
  if (mnorm != nullptr) {
    MOCK_REQUIRE(mnorm->dtype == ACL_FLOAT32 && msum->dtype == ACL_FLOAT32,
                 "MhcSinkhorn: normOut/sumOut must be FP32");
  }
  // The patched copy stage (see the repeatability ledger above). A contiguous
  // output is the case upstream got wrong: Contiguous is the identity, so
  // MhcSinkhorn returns the caller tensor and the unconditional
  // ViewCopy(kernelOut, output) would be a manual-4.31 same-address self-copy.
  // The vendored wrapper now skips the copy in exactly that case, so the
  // executor stays reusable on both paths and no hazard is recorded.
  if (IsContiguous(my)) {
    ++g_sinkhorn_selfcopy_elided;
  }
  *workspace_size = Align4k(static_cast<uint64_t>(mx->dim(0)) * static_cast<uint64_t>(n0) *
                            static_cast<uint64_t>(n1) * 4) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnMhcSinkhorn", {x, output, norm_out, sum_out});
  return 0;
}

// aclnnMhcPost: x_next = (hRes)^T @ x + hOut * hPost.
aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor* x, const aclTensor* h_res, const aclTensor* h_out,
                                         const aclTensor* h_post, aclTensor* out, uint64_t* workspace_size,
                                         aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mr = AsMockTensor(h_res);
  const MockAclTensor* mo = AsMockTensor(h_out);
  const MockAclTensor* mp = AsMockTensor(h_post);
  const MockAclTensor* my = AsMockTensor(out);
  MOCK_REQUIRE(mx != nullptr && mr != nullptr && mo != nullptr && mp != nullptr && my != nullptr,
               "MhcPost: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 3 && mx->dim(1) == kNhcStreams && mx->dim(2) == kHidden,
               "MhcPost: x must be [T, 4, 4096] (TND), got " + ShapeOf(mx));
  MOCK_REQUIRE(mx->dtype == ACL_BF16 || mx->dtype == ACL_FLOAT16, "MhcPost: x must be BF16/FP16");
  const int64_t tokens = mx->dim(0);
  MOCK_REQUIRE(mr->shape.size() == 3 && mr->dim(0) == tokens && mr->dim(1) == kNhcStreams &&
                   mr->dim(2) == kNhcStreams && mr->dtype == ACL_FLOAT32,
               "MhcPost: hRes must be [T, 4, 4] FP32");
  MOCK_REQUIRE(mo->shape.size() == 2 && mo->dim(0) == tokens && mo->dim(1) == kHidden && mo->dtype == mx->dtype,
               "MhcPost: hOut must be [T, 4096] in the dtype of x");
  MOCK_REQUIRE(mp->shape.size() == 2 && mp->dim(0) == tokens && mp->dim(1) == kNhcStreams &&
                   mp->dtype == ACL_FLOAT32,
               "MhcPost: hPost must be [T, 4] FP32");
  MOCK_REQUIRE(my->same_shape_as(*mx) && my->dtype == mx->dtype, "MhcPost: out must match x [T, 4, 4096]");
  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(kHidden) * 2) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnMhcPost", {x, h_res, h_out, h_post, out});
  return 0;
}

// -- vendored vllm-ascend arch35 fused mHC family (hc_*) ----------------------
//
// The same mHC mapping as the mhc_* trio above, decomposed differently; both
// families are registered and both are exercised. Geometry transcribed from the
// vendored wrappers' own checks and from vllm-ascend's
// check_hc_pre_shape_and_dtype: hc = hc_mult = 4 streams over the 4096-wide
// hidden state (7168 also accepted upstream), BF16 states, FP32 mixing weights
// and routing state, Sinkhorn iterations in [1, 100].
//
// THE REPEATABILITY LEDGER AND THIS FAMILY
//   None of the four stubs below calls RecordRefOutputPlan(), and none can
//   increment g_sinkhorn_selfcopy_elided either. Both omissions are the point:
//   the hc_* OpDefs declare NO REF parameter (no output name matches an input
//   name), and the Sinkhorn normalization is interior to HcPre / HcPreSinkhorn
//   rather than being a standalone in-place operator. So there is no tensor
//   that is both an input and an output, and no copy stage whose source and
//   destination could coincide -- the hazard aclnnMhcSinkhorn has to elide does
//   not arise here at all. The invariant these stubs hold is therefore the
//   plain one: g_vendor_selfcopy_hazards never moves.

constexpr int64_t kHcMult = 4;                                   // hc_mult, HC_PRE_HC_LIMIT upstream
constexpr int64_t kHcMixRows = kHcMult * kHcMult + 2 * kHcMult;   // n^2 + 2n = 24, HC_PRE_MIX_HC_LIMIT
constexpr int64_t kHcScaleElements = 3;                          // HC_SCALE_SIZE upstream
constexpr int64_t kHiddenExtended = 7168;                        // HC_PRE_D_LIMIT_EXTEND

bool IsHcHidden(int64_t d) { return d == kHidden || d == kHiddenExtended; }

// Every hc_pre / hc_pre_sinkhorn tensor is shaped over the same leading axes as
// x -- the [bs] or [b, s] prefix left once x's trailing hc and d axes drop.
bool HcLeadingAxesMatch(const MockAclTensor* tensor, const MockAclTensor* x, size_t leading) {
  for (size_t index = 0; index < leading; ++index) {
    if (tensor->dim(static_cast<int>(index)) != x->dim(static_cast<int>(index))) {
      return false;
    }
  }
  return true;
}

// aclnnHcPre: the whole prologue fused -- inv-RMS, mixing projection through
// hcFn [24, 16384] and Sinkhorn -- in one launch.
aclnnStatus aclnnHcPreGetWorkspaceSize(const aclTensor* x, const aclTensor* hc_fn, const aclTensor* hc_scale,
                                       const aclTensor* hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters,
                                       double hc_eps, double norm_eps, const aclTensor* y, const aclTensor* post,
                                       const aclTensor* comb_frag, uint64_t* workspace_size,
                                       aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mfn = AsMockTensor(hc_fn);
  const MockAclTensor* mscale = AsMockTensor(hc_scale);
  const MockAclTensor* mbase = AsMockTensor(hc_base);
  const MockAclTensor* my = AsMockTensor(y);
  const MockAclTensor* mpost = AsMockTensor(post);
  const MockAclTensor* mcomb = AsMockTensor(comb_frag);
  MOCK_REQUIRE(mx != nullptr && mfn != nullptr && mscale != nullptr && mbase != nullptr && my != nullptr &&
                   mpost != nullptr && mcomb != nullptr,
               "HcPre: bad tensor handle");
  MOCK_REQUIRE(hc_mult == kHcMult, "HcPre: hcMult only supports 4, got " + std::to_string(hc_mult));
  MOCK_REQUIRE(hc_sinkhorn_iters >= 1 && hc_sinkhorn_iters <= 100,
               "HcPre: hcSinkhornIters must be in [1, 100], got " + std::to_string(hc_sinkhorn_iters));
  MOCK_REQUIRE(hc_eps > 0.0 && norm_eps > 0.0, "HcPre: hcEps and normEps must be positive");
  MOCK_REQUIRE(mx->shape.size() == 3 || mx->shape.size() == 4,
               "HcPre: x must be [bs, hc, d] or [b, s, hc, d], got " + ShapeOf(mx));
  const size_t rank = mx->shape.size();
  const size_t leading = rank - 2;
  const int64_t hc = mx->dim(static_cast<int>(rank) - 2);
  const int64_t d = mx->dim(static_cast<int>(rank) - 1);
  MOCK_REQUIRE(hc == hc_mult, "HcPre: x's hc axis must equal hcMult (4), got " + std::to_string(hc));
  MOCK_REQUIRE(IsHcHidden(d), "HcPre: x's d must be 4096 or 7168, got " + std::to_string(d));
  MOCK_REQUIRE(mx->dtype == ACL_BF16, "HcPre: x must be BF16");
  MOCK_REQUIRE(mfn->shape.size() == 2 && mfn->dim(0) == kHcMixRows && mfn->dim(1) == hc * d &&
                   mfn->dtype == ACL_FLOAT32,
               "HcPre: hcFn must be [24, hc*d] FP32, got " + ShapeOf(mfn));
  MOCK_REQUIRE(mscale->elements() == kHcScaleElements && mscale->dtype == ACL_FLOAT32,
               "HcPre: hcScale must be [3] FP32");
  MOCK_REQUIRE(mbase->elements() == kHcMixRows && mbase->dtype == ACL_FLOAT32, "HcPre: hcBase must be [24] FP32");
  MOCK_REQUIRE(my->shape.size() == leading + 1 && my->dim(static_cast<int>(leading)) == d &&
                   HcLeadingAxesMatch(my, mx, leading) && my->dtype == ACL_BF16,
               "HcPre: y must carry x's leading axes and a trailing d, BF16, got " + ShapeOf(my));
  MOCK_REQUIRE(mpost->shape.size() == leading + 1 && mpost->dim(static_cast<int>(leading)) == hc_mult &&
                   HcLeadingAxesMatch(mpost, mx, leading) && mpost->dtype == ACL_FLOAT32,
               "HcPre: post must carry x's leading axes and a trailing hcMult, FP32, got " + ShapeOf(mpost));
  MOCK_REQUIRE(mcomb->shape.size() == leading + 2 && mcomb->dim(static_cast<int>(leading)) == hc_mult &&
                   mcomb->dim(static_cast<int>(leading) + 1) == hc_mult && HcLeadingAxesMatch(mcomb, mx, leading) &&
                   mcomb->dtype == ACL_FLOAT32,
               "HcPre: combFrag must carry x's leading axes and a trailing [hcMult, hcMult], FP32, got " +
                   ShapeOf(mcomb));
  // The fused kernel keeps the mixing state and the Sinkhorn matrix internal,
  // so its workspace carries both where the staged path would materialize them.
  *workspace_size = Align4k(static_cast<uint64_t>(mx->elements()) * 4) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnHcPre", {x, hc_fn, hc_scale, hc_base, y, post, comb_frag});
  return 0;
}

// aclnnHcPreInvRms: rsqrt = 1 / sqrt(mean(x^2) + epsilon) over x's last two
// axes -- the only producer of HcPreSinkhorn's `rsqrt` input.
aclnnStatus aclnnHcPreInvRmsGetWorkspaceSize(const aclTensor* x, double epsilon, const aclTensor* y,
                                             uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y);
  MOCK_REQUIRE(mx != nullptr && my != nullptr, "HcPreInvRms: bad tensor handle");
  MOCK_REQUIRE(IsFloat(mx), "HcPreInvRms: x must be FP32/FP16/BF16");
  MOCK_REQUIRE(my->dtype == ACL_FLOAT32, "HcPreInvRms: y must be FP32");
  MOCK_REQUIRE(epsilon >= 0.0, "HcPreInvRms: epsilon must not be negative");
  MOCK_REQUIRE(mx->shape.size() >= 2, "HcPreInvRms: x must have at least 2 axes to reduce over, got " + ShapeOf(mx));
  const size_t leading = mx->shape.size() - 2;
  MOCK_REQUIRE(my->shape.size() == leading + 1 && my->dim(static_cast<int>(leading)) == 1 &&
                   HcLeadingAxesMatch(my, mx, leading),
               "HcPreInvRms: y must carry x's leading axes and a trailing 1, got " + ShapeOf(my));
  *workspace_size = Align4k(static_cast<uint64_t>(my->elements()) * 4) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnHcPreInvRms", {x, y});
  return 0;
}

// aclnnHcPreSinkhorn: the mixing projection fused with the Sinkhorn
// normalization. Unlike aclnnMhcSinkhorn there is no in-place square matrix
// here, so no copy stage to elide.
aclnnStatus aclnnHcPreSinkhornGetWorkspaceSize(const aclTensor* mixes, const aclTensor* rsqrt,
                                               const aclTensor* hc_scale, const aclTensor* hc_base,
                                               const aclTensor* x, int64_t hc_mult, int64_t hc_sinkhorn_iters,
                                               double hc_eps, const aclTensor* y, const aclTensor* post,
                                               const aclTensor* comb_frag, uint64_t* workspace_size,
                                               aclOpExecutor** executor) {
  const MockAclTensor* mmix = AsMockTensor(mixes);
  const MockAclTensor* mrsqrt = AsMockTensor(rsqrt);
  const MockAclTensor* mscale = AsMockTensor(hc_scale);
  const MockAclTensor* mbase = AsMockTensor(hc_base);
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* my = AsMockTensor(y);
  const MockAclTensor* mpost = AsMockTensor(post);
  const MockAclTensor* mcomb = AsMockTensor(comb_frag);
  MOCK_REQUIRE(mmix != nullptr && mrsqrt != nullptr && mscale != nullptr && mbase != nullptr && mx != nullptr &&
                   my != nullptr && mpost != nullptr && mcomb != nullptr,
               "HcPreSinkhorn: bad tensor handle");
  MOCK_REQUIRE(hc_mult == kHcMult, "HcPreSinkhorn: hcMult only supports 4, got " + std::to_string(hc_mult));
  MOCK_REQUIRE(hc_sinkhorn_iters >= 1 && hc_sinkhorn_iters <= 100,
               "HcPreSinkhorn: hcSinkhornIters must be in [1, 100], got " + std::to_string(hc_sinkhorn_iters));
  MOCK_REQUIRE(hc_eps > 0.0, "HcPreSinkhorn: hcEps must be positive");
  MOCK_REQUIRE(mx->shape.size() == 3 || mx->shape.size() == 4,
               "HcPreSinkhorn: x must be [bs, hc, d] or [b, s, hc, d], got " + ShapeOf(mx));
  const size_t rank = mx->shape.size();
  const size_t leading = rank - 2;
  const int64_t hc = mx->dim(static_cast<int>(rank) - 2);
  const int64_t d = mx->dim(static_cast<int>(rank) - 1);
  MOCK_REQUIRE(hc == hc_mult, "HcPreSinkhorn: x's hc axis must equal hcMult (4), got " + std::to_string(hc));
  MOCK_REQUIRE(IsHcHidden(d), "HcPreSinkhorn: x's d must be 4096 or 7168, got " + std::to_string(d));
  MOCK_REQUIRE(mx->dtype == ACL_BF16, "HcPreSinkhorn: x must be BF16");
  MOCK_REQUIRE(mmix->shape.size() == leading + 1 && mmix->dim(static_cast<int>(leading)) == kHcMixRows &&
                   HcLeadingAxesMatch(mmix, mx, leading) && mmix->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: mixes must carry x's leading axes and a trailing 24, FP32, got " + ShapeOf(mmix));
  MOCK_REQUIRE(mrsqrt->shape.size() == leading + 1 && mrsqrt->dim(static_cast<int>(leading)) == 1 &&
                   HcLeadingAxesMatch(mrsqrt, mx, leading) && mrsqrt->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: rsqrt must carry x's leading axes and a trailing 1, FP32, got " + ShapeOf(mrsqrt));
  MOCK_REQUIRE(mscale->elements() == kHcScaleElements && mscale->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: hcScale must be [3] FP32");
  MOCK_REQUIRE(mbase->elements() == kHcMixRows && mbase->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: hcBase must be [24] FP32");
  MOCK_REQUIRE(my->shape.size() == leading + 1 && my->dim(static_cast<int>(leading)) == d &&
                   HcLeadingAxesMatch(my, mx, leading) && my->dtype == ACL_BF16,
               "HcPreSinkhorn: y must carry x's leading axes and a trailing d, BF16, got " + ShapeOf(my));
  MOCK_REQUIRE(mpost->shape.size() == leading + 1 && mpost->dim(static_cast<int>(leading)) == hc_mult &&
                   HcLeadingAxesMatch(mpost, mx, leading) && mpost->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: post must carry x's leading axes and a trailing hcMult, FP32, got " + ShapeOf(mpost));
  MOCK_REQUIRE(mcomb->shape.size() == leading + 2 && mcomb->dim(static_cast<int>(leading)) == hc_mult &&
                   mcomb->dim(static_cast<int>(leading) + 1) == hc_mult && HcLeadingAxesMatch(mcomb, mx, leading) &&
                   mcomb->dtype == ACL_FLOAT32,
               "HcPreSinkhorn: combFrag must carry x's leading axes and a trailing [hcMult, hcMult], FP32, got " +
                   ShapeOf(mcomb));
  *workspace_size = Align4k(static_cast<uint64_t>(mmix->elements()) * 4) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnHcPreSinkhorn", {mixes, rsqrt, hc_scale, hc_base, x, y, post, comb_frag});
  return 0;
}

// aclnnHcPost: y = comb^T @ residual + x * post, BSHD layout, no attributes.
aclnnStatus aclnnHcPostGetWorkspaceSize(const aclTensor* x, const aclTensor* residual, const aclTensor* post,
                                        const aclTensor* comb, const aclTensor* y, uint64_t* workspace_size,
                                        aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mres = AsMockTensor(residual);
  const MockAclTensor* mpost = AsMockTensor(post);
  const MockAclTensor* mcomb = AsMockTensor(comb);
  const MockAclTensor* my = AsMockTensor(y);
  MOCK_REQUIRE(mx != nullptr && mres != nullptr && mpost != nullptr && mcomb != nullptr && my != nullptr,
               "HcPost: bad tensor handle");
  MOCK_REQUIRE(mx->shape.size() == 3, "HcPost: x must be [b, s, d] (BSHD, not TND), got " + ShapeOf(mx));
  MOCK_REQUIRE(IsFloat(mx), "HcPost: x must be FP32/FP16/BF16");
  const int64_t batch = mx->dim(0);
  const int64_t sequence = mx->dim(1);
  const int64_t d = mx->dim(2);
  MOCK_REQUIRE(batch > 0 && sequence > 0 && IsHcHidden(d),
               "HcPost: x must be [b, s, d] with d 4096 or 7168, got " + ShapeOf(mx));
  MOCK_REQUIRE(mres->shape.size() == 4 && mres->dim(0) == batch && mres->dim(1) == sequence &&
                   mres->dim(2) == kHcMult && mres->dim(3) == d && mres->dtype == mx->dtype,
               "HcPost: residual must be [b, s, 4, d] in the dtype of x, got " + ShapeOf(mres));
  MOCK_REQUIRE(mpost->shape.size() == 3 && mpost->dim(0) == batch && mpost->dim(1) == sequence &&
                   mpost->dim(2) == kHcMult && IsFloat(mpost),
               "HcPost: post must be [b, s, 4], got " + ShapeOf(mpost));
  MOCK_REQUIRE(mcomb->shape.size() == 4 && mcomb->dim(0) == batch && mcomb->dim(1) == sequence &&
                   mcomb->dim(2) == kHcMult && mcomb->dim(3) == kHcMult && mcomb->dtype == mpost->dtype,
               "HcPost: comb must be [b, s, 4, 4] in the dtype of post, got " + ShapeOf(mcomb));
  MOCK_REQUIRE(my->same_shape_as(*mres) && my->dtype == mres->dtype,
               "HcPost: y must match residual [b, s, 4, d]");
  *workspace_size = Align4k(static_cast<uint64_t>(mx->elements()) * 2) + kWorkspaceMhcBase;
  *executor = NewExecutor("aclnnHcPost", {x, residual, post, comb, y});
  return 0;
}

// aclnnQuantLightningIndexer: sparse top-k selection + query/key quantization.
aclnnStatus aclnnQuantLightningIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* weights, const aclTensor* query_dequant_scale,
    const aclTensor* key_dequant_scale, const aclTensor* actual_seq_lengths_query_optional,
    const aclTensor* actual_seq_lengths_key_optional, const aclTensor* block_table_optional, int64_t query_quant_mode,
    int64_t key_quant_mode, char* layout_query_optional, char* layout_key_optional, int64_t sparse_count,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, const aclTensor* out, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  const MockAclTensor* mq = AsMockTensor(query);
  const MockAclTensor* mk = AsMockTensor(key);
  const MockAclTensor* mw = AsMockTensor(weights);
  const MockAclTensor* mqs = AsMockTensor(query_dequant_scale);
  const MockAclTensor* mks = AsMockTensor(key_dequant_scale);
  const MockAclTensor* my = AsMockTensor(out);
  MOCK_REQUIRE(mq != nullptr && mk != nullptr && mw != nullptr && mqs != nullptr && mks != nullptr &&
                   my != nullptr,
               "QuantLightningIndexer: bad tensor handle");
  const char* layout_query = layout_query_optional != nullptr ? layout_query_optional : "BSND";
  const char* layout_key = layout_key_optional != nullptr ? layout_key_optional : "BSND";
  // TND query: [T, N1, D] with D = 128 and N1 in the 950PR set; BSND adds a
  // leading [B, S] pair over the same N1/D geometry.
  MOCK_REQUIRE(mq->shape.size() == 3 || mq->shape.size() == 4,
               "QuantLightningIndexer: query must be TND [T,N1,D] or BSND [B,S,N1,D]");
  const bool query_tnd = mq->shape.size() == 3;
  const size_t n1_index = query_tnd ? 1 : 2;
  const int64_t heads = mq->dim(n1_index);
  const int64_t head_dim = mq->dim(n1_index + 1);
  MOCK_REQUIRE(head_dim == 128, "QuantLightningIndexer: the indexer head dim is 128, got " +
                                    std::to_string(head_dim));
  MOCK_REQUIRE(heads == 16 || heads == 24 || heads == 32 || heads == 64,
               "QuantLightningIndexer: N1 must be one of {16, 24, 32, 64} on 950PR, got " + std::to_string(heads));
  MOCK_REQUIRE(mq->dtype == ACL_FLOAT8_E4M3FN, "QuantLightningIndexer: query must be FP8 E4M3 (950PR path)");
  MOCK_REQUIRE(IsContiguous(mq) && IsContiguous(mk),
               "QuantLightningIndexer: query and key do not support non-contiguous views");
  const int64_t tokens = query_tnd ? mq->dim(0) : mq->dim(0) * mq->dim(1);
  // Key: N2 = 1, D = 128; PA_BSND pages it as [block_count, block_size, 1, 128].
  MOCK_REQUIRE(mk->dtype == mq->dtype, "QuantLightningIndexer: key dtype must equal query dtype");
  const bool key_paged = std::string(layout_key) == "PA_BSND";
  if (key_paged) {
    MOCK_REQUIRE(mk->shape.size() == 4, "QuantLightningIndexer: PA_BSND key must be [block_count, block_size, 1, 128]");
    MOCK_REQUIRE(mk->dim(2) == 1 && mk->dim(3) == 128,
                 "QuantLightningIndexer: PA_BSND key must carry N2=1 and D=128");
    MOCK_REQUIRE(mk->dim(1) % 16 == 0 && mk->dim(1) <= 1024,
                 "QuantLightningIndexer: block_size must be a multiple of 16 and at most 1024");
  } else {
    MOCK_REQUIRE(mk->shape.size() == 3 || mk->shape.size() == 4,
                 "QuantLightningIndexer: key must be TND [T,N2,D] or BSND [B,S,N2,D]");
    const size_t n2_index = mk->shape.size() == 3 ? 1 : 2;
    MOCK_REQUIRE(mk->dim(n2_index) == 1, "QuantLightningIndexer: N2 is 1");
    MOCK_REQUIRE(mk->dim(n2_index + 1) == 128, "QuantLightningIndexer: key head dim is 128");
  }
  // The BF16-weights path carries FP32 dequant scales (950PR FP8 tuple).
  MOCK_REQUIRE(mw->shape.size() == (query_tnd ? 2 : 3) && mw->dim(mw->shape.size() - 1) == heads &&
                   (mw->dtype == ACL_BF16 || mw->dtype == ACL_FLOAT16),
               "QuantLightningIndexer: weights must be [T,N1]/[B,S,N1] BF16/FP16");
  MOCK_REQUIRE(mqs->shape == mw->shape && mqs->dtype == ACL_FLOAT32,
               "QuantLightningIndexer: queryDequantScale must match weights shape in FP32");
  if (key_paged) {
    MOCK_REQUIRE(mks->shape.size() == 3 && mks->dim(0) == mk->dim(0) && mks->dim(1) == mk->dim(1) &&
                     mks->dim(2) == 1 && mks->dtype == ACL_FLOAT32,
                 "QuantLightningIndexer: PA keyDequantScale must be [block_count, block_size, 1] FP32");
  } else {
    MOCK_REQUIRE(mks->dtype == ACL_FLOAT32, "QuantLightningIndexer: keyDequantScale must be FP32");
  }
  // Scalar attributes: per-token-head quantization, top-k window and mask.
  MOCK_REQUIRE(query_quant_mode == 0 && key_quant_mode == 0,
               "QuantLightningIndexer: quant modes must be 0 (per-token-head)");
  MOCK_REQUIRE(sparse_count >= 1 && sparse_count <= 2048,
               "QuantLightningIndexer: sparseCount must be in [1, 2048]");
  MOCK_REQUIRE(sparse_mode == 0 || sparse_mode == 3, "QuantLightningIndexer: sparseMode must be 0 or 3");
  (void)pre_tokens;
  (void)next_tokens;
  // TND query needs the cumulative query lengths; paged keys need the key
  // lengths and the block table.
  const MockAclTensor* maslq =
      actual_seq_lengths_query_optional == nullptr ? nullptr : AsMockTensor(actual_seq_lengths_query_optional);
  const MockAclTensor* maslk =
      actual_seq_lengths_key_optional == nullptr ? nullptr : AsMockTensor(actual_seq_lengths_key_optional);
  const MockAclTensor* mblocks =
      block_table_optional == nullptr ? nullptr : AsMockTensor(block_table_optional);
  if (std::string(layout_query) == "TND" || query_tnd) {
    MOCK_REQUIRE(maslq != nullptr && maslq->dtype == ACL_INT32,
                 "QuantLightningIndexer: TND query requires INT32 actualSeqLengthsQuery (cumulative)");
  }
  if (key_paged) {
    MOCK_REQUIRE(maslk != nullptr && maslk->dtype == ACL_INT32,
                 "QuantLightningIndexer: PA_BSND key requires INT32 actualSeqLengthsKey");
    MOCK_REQUIRE(mblocks != nullptr && mblocks->dtype == ACL_INT32 && mblocks->shape.size() == 2,
                 "QuantLightningIndexer: PA_BSND key requires a 2-D INT32 blockTable");
  }
  // Output: INT32 sparse indices with N2 = 1 on the head axis.
  MOCK_REQUIRE(my->dtype == ACL_INT32, "QuantLightningIndexer: out must be INT32");
  MOCK_REQUIRE(my->shape.size() == (query_tnd ? 3 : 4), "QuantLightningIndexer: out must be [T,1,k] or [B,S,1,k]");
  MOCK_REQUIRE(my->dim(my->shape.size() - 2) == 1, "QuantLightningIndexer: out carries N2=1");
  MOCK_REQUIRE(my->dim(my->shape.size() - 1) == sparse_count,
               "QuantLightningIndexer: out's last axis is the retained top-k count");
  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(heads) * 8) +
                    kWorkspaceIndexerBase;
  *executor = NewExecutor("aclnnQuantLightningIndexer",
                          {query, key, weights, query_dequant_scale, key_dequant_scale,
                           actual_seq_lengths_query_optional, actual_seq_lengths_key_optional,
                           block_table_optional, out});
  return 0;
}

// aclnnCompressor: pool cmp_ratio tokens into one compressed-KV row and
// advance the recurrent pooling state in place.
aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor* x, const aclTensor* wkv, const aclTensor* wgate, aclTensor* state_cache_ref,
    const aclTensor* ape, const aclTensor* norm_weight, const aclTensor* rope_sin, const aclTensor* rope_cos,
    const aclTensor* state_block_table_optional, const aclTensor* cu_seqlens_optional,
    const aclTensor* seqused_optional, const aclTensor* start_pos_optional, int64_t rope_head_dim,
    int64_t cmp_ratio, int64_t coff, double norm_eps, int64_t rotary_mode, int64_t cache_mode,
    int64_t state_cache_stride_dim0, const aclTensor* cmp_kv_out, uint64_t* workspace_size,
    aclOpExecutor** executor) {
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* mwkv = AsMockTensor(wkv);
  const MockAclTensor* mwg = AsMockTensor(wgate);
  const MockAclTensor* msc = AsMockTensor(state_cache_ref);
  const MockAclTensor* mape = AsMockTensor(ape);
  const MockAclTensor* mnw = AsMockTensor(norm_weight);
  const MockAclTensor* msin = AsMockTensor(rope_sin);
  const MockAclTensor* mcos = AsMockTensor(rope_cos);
  const MockAclTensor* mkv = AsMockTensor(cmp_kv_out);
  MOCK_REQUIRE(mx != nullptr && mwkv != nullptr && mwg != nullptr && msc != nullptr && mape != nullptr &&
                   mnw != nullptr && msin != nullptr && mcos != nullptr && mkv != nullptr,
               "Compressor: bad tensor handle");

  // x is [T, H] (TND, the decode path) or [B, S, H].
  MOCK_REQUIRE(mx->shape.size() == 2 || mx->shape.size() == 3,
               "Compressor: x must be [T, H] or [B, S, H], got " + ShapeOf(mx));
  MOCK_REQUIRE(mx->dtype == ACL_BF16 || mx->dtype == ACL_FLOAT16, "Compressor: x must be BF16/FP16");
  const bool x_tnd = mx->shape.size() == 2;
  const int64_t tokens = x_tnd ? mx->dim(0) : mx->dim(0) * mx->dim(1);
  MOCK_REQUIRE(mx->dim(mx->shape.size() - 1) == kHidden,
               "Compressor: the DSV4 hidden size is 4096, got " + std::to_string(mx->dim(mx->shape.size() - 1)));
  MOCK_REQUIRE(mwkv->dtype == mx->dtype && mwg->dtype == mx->dtype,
               "Compressor: wkv and wgate must share the dtype of x");
  MOCK_REQUIRE(mkv->dtype == mx->dtype, "Compressor: cmpKvOut must share the dtype of x");

  // The two compression ratios DSV4-Flash deploys: 4 selects CSA, 128 HCA.
  MOCK_REQUIRE(cmp_ratio == 4 || cmp_ratio == 128,
               "Compressor: cmpRatio must be 4 (CSA) or 128 (HCA), got " + std::to_string(cmp_ratio));
  MOCK_REQUIRE(rope_head_dim > 0 && rope_head_dim % 2 == 0,
               "Compressor: ropeHeadDim must be positive and even, got " + std::to_string(rope_head_dim));
  MOCK_REQUIRE(norm_eps > 0.0, "Compressor: normEps must be positive");
  MOCK_REQUIRE(coff >= 1, "Compressor: coff must be at least 1");
  MOCK_REQUIRE(rotary_mode == 0 || rotary_mode == 1, "Compressor: rotaryMode must be 0 or 1");

  // The recurrent pooling state is a paged [blocks, blockSize, D] FP32 cache
  // and is a REF parameter: the operator updates it in place.
  MOCK_REQUIRE(msc->shape.size() == 3 && msc->dtype == ACL_FLOAT32,
               "Compressor: stateCacheRef must be 3-D [blocks, blockSize, D] FP32, got " + ShapeOf(msc));
  // state_cache_stride_dim0 must agree with the view the caller handed over:
  // the kernel addresses the paged cache through it, so a stale value writes
  // the wrong block.
  MOCK_REQUIRE(state_cache_stride_dim0 == msc->strides.at(0),
               "Compressor: stateCacheStrideDim0 (" + std::to_string(state_cache_stride_dim0) +
                   ") must equal the stateCacheRef axis-0 stride (" + std::to_string(msc->strides.at(0)) + ")");
  MOCK_REQUIRE(cache_mode == 0 || cache_mode == 1, "Compressor: cacheMode must be 0 or 1");

  MOCK_REQUIRE(mape->dtype == ACL_FLOAT32, "Compressor: ape must be FP32");
  MOCK_REQUIRE(mnw->shape.size() == 1, "Compressor: normWeight must be 1-D, got " + ShapeOf(mnw));
  MOCK_REQUIRE(msin->shape.size() == mx->shape.size() && mcos->shape.size() == mx->shape.size(),
               "Compressor: ropeSin and ropeCos must carry the rank of x");
  MOCK_REQUIRE(msin->shape == mcos->shape && msin->dtype == mcos->dtype,
               "Compressor: ropeSin and ropeCos must agree in shape and dtype");

  // cmp_kv_out holds one pooled row per cmp_ratio input tokens, widened by
  // coff on the channel axis.
  MOCK_REQUIRE(mkv->shape.size() == 2, "Compressor: cmpKvOut must be [cmpS, D], got " + ShapeOf(mkv));
  MOCK_REQUIRE(mkv->dim(0) == tokens / cmp_ratio,
               "Compressor: cmpKvOut rows must be T/cmpRatio = " + std::to_string(tokens / cmp_ratio) + ", got " +
                   std::to_string(mkv->dim(0)));
  MOCK_REQUIRE(mkv->dim(1) == mnw->dim(0) * coff,
               "Compressor: cmpKvOut channels must be normWeight[0]*coff = " +
                   std::to_string(mnw->dim(0) * coff) + ", got " + std::to_string(mkv->dim(1)));

  const MockAclTensor* mbt =
      state_block_table_optional == nullptr ? nullptr : AsMockTensor(state_block_table_optional);
  if (mbt != nullptr) {
    MOCK_REQUIRE(mbt->dtype == ACL_INT32 && mbt->shape.size() == 2,
                 "Compressor: stateBlockTableOptional must be a 2-D INT32 map");
  }
  for (const aclTensor* lengths : {cu_seqlens_optional, seqused_optional, start_pos_optional}) {
    if (lengths != nullptr) {
      MOCK_REQUIRE(AsMockTensor(lengths)->dtype == ACL_INT32,
                   "Compressor: cuSeqlens/seqused/startPos must be INT32 when bound");
    }
  }

  // The REF parameter takes no copy stage, so nothing here can be a
  // same-address self-copy (see g_ref_selfcopy_hazards).
  RecordRefOutputPlan();
  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(kHidden) * 4) +
                    kWorkspaceIndexerBase;
  *executor = NewExecutor("aclnnCompressor",
                          {x, wkv, wgate, state_cache_ref, ape, norm_weight, rope_sin, rope_cos,
                           state_block_table_optional, cu_seqlens_optional, seqused_optional, start_pos_optional,
                           cmp_kv_out});
  return 0;
}

// aclnnVllmQuantLightningIndexer: the shared-KV form of the indexer.
aclnnStatus aclnnVllmQuantLightningIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* weights, const aclTensor* query_dequant_scale,
    const aclTensor* key_dequant_scale, const aclTensor* actual_seq_lengths_query_optional,
    const aclTensor* actual_seq_lengths_key_optional, const aclTensor* block_table_optional,
    const aclTensor* metadata_optional, int64_t query_quant_mode, int64_t key_quant_mode,
    char* layout_query_optional, char* layout_key_optional, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, int64_t cmp_ratio, bool return_values, int64_t stride,
    int64_t scale_stride, const aclTensor* sparse_indices_out, const aclTensor* sparse_values_out,
    uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mq = AsMockTensor(query);
  const MockAclTensor* mk = AsMockTensor(key);
  const MockAclTensor* mw = AsMockTensor(weights);
  const MockAclTensor* mqs = AsMockTensor(query_dequant_scale);
  const MockAclTensor* mks = AsMockTensor(key_dequant_scale);
  const MockAclTensor* mi = AsMockTensor(sparse_indices_out);
  const MockAclTensor* mv = AsMockTensor(sparse_values_out);
  MOCK_REQUIRE(mq != nullptr && mk != nullptr && mw != nullptr && mqs != nullptr && mks != nullptr &&
                   mi != nullptr && mv != nullptr,
               "VllmQuantLightningIndexer: bad tensor handle");

  const char* layout_query = layout_query_optional != nullptr ? layout_query_optional : "BSND";
  const char* layout_key = layout_key_optional != nullptr ? layout_key_optional : "BSND";
  MOCK_REQUIRE(mq->shape.size() == 3 || mq->shape.size() == 4,
               "VllmQuantLightningIndexer: query must be TND [T,N1,D] or BSND [B,S,N1,D]");
  const bool query_tnd = mq->shape.size() == 3;
  const size_t n1_index = query_tnd ? 1 : 2;
  const int64_t heads = mq->dim(n1_index);
  MOCK_REQUIRE(mq->dim(n1_index + 1) == 128,
               "VllmQuantLightningIndexer: the indexer head dim is 128, got " +
                   std::to_string(mq->dim(n1_index + 1)));
  MOCK_REQUIRE(heads == 16 || heads == 24 || heads == 32 || heads == 64,
               "VllmQuantLightningIndexer: N1 must be one of {16, 24, 32, 64} on 950PR, got " +
                   std::to_string(heads));
  // 950PR selected path: FP8 E4M3. (The operator also admits HiFloat8, which
  // this mock's dtype enum does not model.)
  MOCK_REQUIRE(mq->dtype == ACL_FLOAT8_E4M3FN,
               "VllmQuantLightningIndexer: query must be FP8 E4M3 (950PR path)");
  MOCK_REQUIRE(mk->dtype == mq->dtype, "VllmQuantLightningIndexer: key dtype must equal query dtype");
  const int64_t tokens = query_tnd ? mq->dim(0) : mq->dim(0) * mq->dim(1);

  const bool key_paged = std::string(layout_key) == "PA_BSND";
  if (key_paged) {
    MOCK_REQUIRE(mk->shape.size() == 4,
                 "VllmQuantLightningIndexer: PA_BSND key must be [block_count, block_size, 1, 128]");
    MOCK_REQUIRE(mk->dim(2) == 1 && mk->dim(3) == 128,
                 "VllmQuantLightningIndexer: PA_BSND key must carry N2=1 and D=128");
  } else {
    MOCK_REQUIRE(mk->shape.size() == 3 || mk->shape.size() == 4,
                 "VllmQuantLightningIndexer: key must be TND [T,N2,D] or BSND [B,S,N2,D]");
  }
  // The strides are explicit attributes precisely because the paged key and
  // its scales are handed over uncontiguized; a stale value reads the wrong
  // block.
  MOCK_REQUIRE(stride == mk->strides.at(0),
               "VllmQuantLightningIndexer: stride (" + std::to_string(stride) +
                   ") must equal the key axis-0 stride (" + std::to_string(mk->strides.at(0)) + ")");
  MOCK_REQUIRE(scale_stride == mks->strides.at(0),
               "VllmQuantLightningIndexer: scaleStride (" + std::to_string(scale_stride) +
                   ") must equal the keyDequantScale axis-0 stride (" + std::to_string(mks->strides.at(0)) + ")");

  MOCK_REQUIRE(mw->shape.size() == (query_tnd ? 2u : 3u) && mw->dim(mw->shape.size() - 1) == heads &&
                   (mw->dtype == ACL_BF16 || mw->dtype == ACL_FLOAT16),
               "VllmQuantLightningIndexer: weights must be [T,N1]/[B,S,N1] BF16/FP16");
  MOCK_REQUIRE(mqs->shape == mw->shape && mqs->dtype == ACL_FLOAT32,
               "VllmQuantLightningIndexer: queryDequantScale must match weights shape in FP32");
  MOCK_REQUIRE(mks->dtype == ACL_FLOAT32, "VllmQuantLightningIndexer: keyDequantScale must be FP32");

  MOCK_REQUIRE(query_quant_mode == 0 && key_quant_mode == 0,
               "VllmQuantLightningIndexer: quant modes must be 0 (per-token-head)");
  MOCK_REQUIRE(sparse_count >= 1 && sparse_count <= 2048,
               "VllmQuantLightningIndexer: sparseCount must be in [1, 2048]");
  MOCK_REQUIRE(sparse_mode == 0 || sparse_mode == 3, "VllmQuantLightningIndexer: sparseMode must be 0 or 3");
  // This indexer scores the compressed key stream, so cmpRatio tracks the
  // compressor; 1 means the stream is uncompressed.
  MOCK_REQUIRE(cmp_ratio == 1 || cmp_ratio == 4 || cmp_ratio == 128,
               "VllmQuantLightningIndexer: cmpRatio must be 1, 4 (CSA) or 128 (HCA), got " +
                   std::to_string(cmp_ratio));
  (void)pre_tokens;
  (void)next_tokens;

  if (std::string(layout_query) == "TND" || query_tnd) {
    const MockAclTensor* maslq =
        actual_seq_lengths_query_optional == nullptr ? nullptr : AsMockTensor(actual_seq_lengths_query_optional);
    MOCK_REQUIRE(maslq != nullptr && maslq->dtype == ACL_INT32,
                 "VllmQuantLightningIndexer: TND query requires INT32 actualSeqLengthsQuery (cumulative)");
  }
  if (key_paged) {
    const MockAclTensor* maslk =
        actual_seq_lengths_key_optional == nullptr ? nullptr : AsMockTensor(actual_seq_lengths_key_optional);
    const MockAclTensor* mblocks = block_table_optional == nullptr ? nullptr : AsMockTensor(block_table_optional);
    MOCK_REQUIRE(maslk != nullptr && maslk->dtype == ACL_INT32,
                 "VllmQuantLightningIndexer: PA_BSND key requires INT32 actualSeqLengthsKey");
    MOCK_REQUIRE(mblocks != nullptr && mblocks->dtype == ACL_INT32 && mblocks->shape.size() == 2,
                 "VllmQuantLightningIndexer: PA_BSND key requires a 2-D INT32 blockTable");
  }
  if (metadata_optional != nullptr) {
    MOCK_REQUIRE(AsMockTensor(metadata_optional)->dtype == ACL_INT32,
                 "VllmQuantLightningIndexer: metadataOptional must be INT32");
  }

  MOCK_REQUIRE(mi->dtype == ACL_INT32, "VllmQuantLightningIndexer: sparseIndicesOut must be INT32");
  MOCK_REQUIRE(mi->shape.size() == (query_tnd ? 3u : 4u),
               "VllmQuantLightningIndexer: sparseIndicesOut must be [T,N2,k] or [B,S,N2,k]");
  MOCK_REQUIRE(mi->dim(mi->shape.size() - 1) == sparse_count,
               "VllmQuantLightningIndexer: sparseIndicesOut last axis is the retained top-k count");
  MOCK_REQUIRE(mv->dtype == ACL_FLOAT32, "VllmQuantLightningIndexer: sparseValuesOut must be FP32");
  // returnValues false is signalled by a [0] placeholder, exactly as the
  // vendored wrapper expects: it then issues no copy for that output.
  if (return_values) {
    MOCK_REQUIRE(mv->shape == mi->shape,
                 "VllmQuantLightningIndexer: returnValues=true requires sparseValuesOut to match "
                 "sparseIndicesOut");
  } else {
    MOCK_REQUIRE(mv->elements() == 0,
                 "VllmQuantLightningIndexer: returnValues=false requires a [0] sparseValuesOut placeholder");
  }

  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(heads) * 8) +
                    kWorkspaceIndexerBase;
  *executor = NewExecutor("aclnnVllmQuantLightningIndexer",
                          {query, key, weights, query_dequant_scale, key_dequant_scale,
                           actual_seq_lengths_query_optional, actual_seq_lengths_key_optional,
                           block_table_optional, metadata_optional, sparse_indices_out, sparse_values_out});
  return 0;
}

// aclnnKvQuantSparseAttnSharedkv: MQA core attention over the hybrid cache.
aclnnStatus aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor* q, const aclTensor* ori_kv_optional, const aclTensor* cmp_kv_optional,
    const aclTensor* ori_sparse_indices_optional, const aclTensor* cmp_sparse_indices_optional,
    const aclTensor* ori_block_table_optional, const aclTensor* cmp_block_table_optional,
    const aclTensor* cu_seqlens_q_optional, const aclTensor* cu_seqlens_ori_kv_optional,
    const aclTensor* cu_seqlens_cmp_kv_optional, const aclTensor* seqused_q_optional,
    const aclTensor* seqused_kv_optional, const aclTensor* sinks_optional, const aclTensor* metadata_optional,
    int64_t kv_quant_mode, int64_t tile_size, int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    char* layout_q_optional, char* layout_kv_optional, int64_t ori_kv_stride0, int64_t cmp_kv_stride0,
    bool return_softmax_lse, const aclTensor* attn_out, const aclTensor* softmax_lse_out,
    uint64_t* workspace_size, aclOpExecutor** executor) {
  const MockAclTensor* mq = AsMockTensor(q);
  const MockAclTensor* mo = AsMockTensor(attn_out);
  const MockAclTensor* mlse = AsMockTensor(softmax_lse_out);
  MOCK_REQUIRE(mq != nullptr && mo != nullptr && mlse != nullptr,
               "KvQuantSparseAttnSharedkv: bad tensor handle");

  MOCK_REQUIRE(mq->dtype == ACL_BF16, "KvQuantSparseAttnSharedkv: q must be BF16");
  MOCK_REQUIRE(mq->shape.size() == 3 || mq->shape.size() == 4,
               "KvQuantSparseAttnSharedkv: q must be TND [T,N,D] or BSND [B,S,N,D], got " + ShapeOf(mq));
  const bool q_tnd = mq->shape.size() == 3;
  const int64_t tokens = q_tnd ? mq->dim(0) : mq->dim(0) * mq->dim(1);
  const int64_t q_heads = mq->dim(q_tnd ? 1 : 2);
  const int64_t head_dim = mq->dim(mq->shape.size() - 1);
  // DSV4-Flash MLA geometry: 512 latent channels carrying a 64-wide RoPE
  // slice, over the 64-head query.
  MOCK_REQUIRE(head_dim == 512,
               "KvQuantSparseAttnSharedkv: the DSV4 total head dim is 512, got " + std::to_string(head_dim));
  MOCK_REQUIRE(q_heads == kNumHeads,
               "KvQuantSparseAttnSharedkv: the DSV4 query head count is 64, got " + std::to_string(q_heads));
  MOCK_REQUIRE(rope_head_dim > 0 && rope_head_dim < head_dim,
               "KvQuantSparseAttnSharedkv: ropeHeadDim must be positive and under the total head dim");
  MOCK_REQUIRE(mo->same_shape_as(*mq) && mo->dtype == ACL_BF16,
               "KvQuantSparseAttnSharedkv: attnOut must match q in shape and BF16 dtype");

  // The hybrid cache is the point of this operator: at least one half must be
  // bound, and each bound half brings its own indices and stride attribute.
  const MockAclTensor* mori = ori_kv_optional == nullptr ? nullptr : AsMockTensor(ori_kv_optional);
  const MockAclTensor* mcmp = cmp_kv_optional == nullptr ? nullptr : AsMockTensor(cmp_kv_optional);
  MOCK_REQUIRE(mori != nullptr || mcmp != nullptr,
               "KvQuantSparseAttnSharedkv: at least one of oriKv and cmpKv must be bound");
  if (mori != nullptr) {
    MOCK_REQUIRE(mori->dtype == ACL_FLOAT8_E4M3FN, "KvQuantSparseAttnSharedkv: oriKv must be FP8 E4M3");
    MOCK_REQUIRE(ori_kv_stride0 == mori->strides.at(0),
                 "KvQuantSparseAttnSharedkv: oriKvStride0 (" + std::to_string(ori_kv_stride0) +
                     ") must equal the oriKv axis-0 stride (" + std::to_string(mori->strides.at(0)) + ")");
    MOCK_REQUIRE(ori_sparse_indices_optional != nullptr &&
                     AsMockTensor(ori_sparse_indices_optional)->dtype == ACL_INT32,
                 "KvQuantSparseAttnSharedkv: a bound oriKv requires INT32 oriSparseIndices");
  }
  if (mcmp != nullptr) {
    MOCK_REQUIRE(mcmp->dtype == ACL_FLOAT8_E4M3FN, "KvQuantSparseAttnSharedkv: cmpKv must be FP8 E4M3");
    MOCK_REQUIRE(cmp_kv_stride0 == mcmp->strides.at(0),
                 "KvQuantSparseAttnSharedkv: cmpKvStride0 (" + std::to_string(cmp_kv_stride0) +
                     ") must equal the cmpKv axis-0 stride (" + std::to_string(mcmp->strides.at(0)) + ")");
    MOCK_REQUIRE(cmp_sparse_indices_optional != nullptr &&
                     AsMockTensor(cmp_sparse_indices_optional)->dtype == ACL_INT32,
                 "KvQuantSparseAttnSharedkv: a bound cmpKv requires INT32 cmpSparseIndices");
    MOCK_REQUIRE(cmp_ratio == 4 || cmp_ratio == 128,
                 "KvQuantSparseAttnSharedkv: cmpRatio must be 4 (CSA) or 128 (HCA) when cmpKv is bound, got " +
                     std::to_string(cmp_ratio));
  }

  MOCK_REQUIRE(tile_size > 0 && tile_size % 16 == 0,
               "KvQuantSparseAttnSharedkv: tileSize must be a positive multiple of 16, got " +
                   std::to_string(tile_size));
  MOCK_REQUIRE(softmax_scale > 0.0, "KvQuantSparseAttnSharedkv: softmaxScale must be positive");
  MOCK_REQUIRE(kv_quant_mode >= 0, "KvQuantSparseAttnSharedkv: kvQuantMode must be non-negative");
  MOCK_REQUIRE(ori_mask_mode >= 0 && cmp_mask_mode >= 0,
               "KvQuantSparseAttnSharedkv: the mask modes must be non-negative");
  MOCK_REQUIRE(ori_win_left >= 0 && ori_win_right >= 0,
               "KvQuantSparseAttnSharedkv: the sliding-window bounds must be non-negative");
  (void)layout_q_optional;
  (void)layout_kv_optional;

  for (const aclTensor* table : {ori_block_table_optional, cmp_block_table_optional}) {
    if (table != nullptr) {
      const MockAclTensor* mt = AsMockTensor(table);
      MOCK_REQUIRE(mt->dtype == ACL_INT32 && mt->shape.size() == 2,
                   "KvQuantSparseAttnSharedkv: block tables must be 2-D INT32");
    }
  }
  for (const aclTensor* lengths : {cu_seqlens_q_optional, cu_seqlens_ori_kv_optional, cu_seqlens_cmp_kv_optional,
                                   seqused_q_optional, seqused_kv_optional, metadata_optional}) {
    if (lengths != nullptr) {
      MOCK_REQUIRE(AsMockTensor(lengths)->dtype == ACL_INT32,
                   "KvQuantSparseAttnSharedkv: the length and metadata vectors must be INT32");
    }
  }
  if (sinks_optional != nullptr) {
    const MockAclTensor* ms = AsMockTensor(sinks_optional);
    MOCK_REQUIRE(ms->dtype == ACL_FLOAT32, "KvQuantSparseAttnSharedkv: sinks must be FP32");
    MOCK_REQUIRE(ms->elements() == q_heads,
                 "KvQuantSparseAttnSharedkv: sinks carries one logit per query head");
  }

  MOCK_REQUIRE(mlse->dtype == ACL_FLOAT32, "KvQuantSparseAttnSharedkv: softmaxLseOut must be FP32");
  if (return_softmax_lse) {
    MOCK_REQUIRE(mlse->elements() > 0,
                 "KvQuantSparseAttnSharedkv: returnSoftmaxLse=true requires a sized softmaxLseOut");
  } else {
    MOCK_REQUIRE(mlse->elements() == 0,
                 "KvQuantSparseAttnSharedkv: returnSoftmaxLse=false requires a [0] softmaxLseOut placeholder");
  }

  *workspace_size = Align4k(static_cast<uint64_t>(tokens) * static_cast<uint64_t>(q_heads) *
                            static_cast<uint64_t>(tile_size) * 4) + kWorkspaceAttention;
  *executor = NewExecutor("aclnnKvQuantSparseAttnSharedkv",
                          {q, ori_kv_optional, cmp_kv_optional, ori_sparse_indices_optional,
                           cmp_sparse_indices_optional, ori_block_table_optional, cmp_block_table_optional,
                           cu_seqlens_q_optional, cu_seqlens_ori_kv_optional, cu_seqlens_cmp_kv_optional,
                           seqused_q_optional, seqused_kv_optional, sinks_optional, metadata_optional, attn_out,
                           softmax_lse_out});
  return 0;
}

// The two cache epilogs share their whole contract apart from the cache dtype
// and the quantization attributes: three tensors, the first of which is a REF
// parameter the kernel scatters into in place.
namespace {

// static: this unnamed namespace is nested inside the file's extern "C" block,
// where C language linkage would otherwise make the name external.
static aclnnStatus ValidateCompressEpilog(const char* op, const aclTensor* cache_ref, const aclTensor* x,
                                          const aclTensor* slot_mapping, int64_t layout, int64_t block_stride) {
  const MockAclTensor* mc = AsMockTensor(cache_ref);
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* ms = AsMockTensor(slot_mapping);
  MOCK_REQUIRE(mc != nullptr && mx != nullptr && ms != nullptr, std::string(op) + ": bad tensor handle");
  MOCK_REQUIRE(mc->shape.size() == 3 || mc->shape.size() == 4,
               std::string(op) + ": the cache must be 3-D or 4-D paged, got " + ShapeOf(mc));
  // The cache is reached through block_stride, so a stale value scatters into
  // the wrong block.
  MOCK_REQUIRE(block_stride == mc->strides.at(0),
               std::string(op) + ": blockStride (" + std::to_string(block_stride) +
                   ") must equal the cache axis-0 stride (" + std::to_string(mc->strides.at(0)) + ")");
  MOCK_REQUIRE(layout >= 0, std::string(op) + ": layout must be non-negative");
  MOCK_REQUIRE(mx->shape.size() == 2, std::string(op) + ": x must be [rows, D], got " + ShapeOf(mx));
  MOCK_REQUIRE(ms->shape.size() == 1 && ms->dim(0) == mx->dim(0),
               std::string(op) + ": slotMapping needs one destination slot per row of x");
  return 0;
}

}  // namespace

// aclnnKvCompressEpilog: quantize compressed KV rows and scatter them.
aclnnStatus aclnnKvCompressEpilogGetWorkspaceSize(aclTensor* kv_compress_cache_ref, const aclTensor* x,
                                                  const aclTensor* slot_mapping, int64_t quant_group_size,
                                                  int64_t quant_mode, int64_t round_scale, int64_t layout,
                                                  int64_t block_stride, uint64_t* workspace_size,
                                                  aclOpExecutor** executor) {
  if (int error = ValidateCompressEpilog("KvCompressEpilog", kv_compress_cache_ref, x, slot_mapping, layout,
                                         block_stride)) {
    return error;
  }
  const MockAclTensor* mc = AsMockTensor(kv_compress_cache_ref);
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* ms = AsMockTensor(slot_mapping);
  MOCK_REQUIRE(mc->dtype == ACL_FLOAT8_E5M2 || mc->dtype == ACL_FLOAT8_E4M3FN,
               "KvCompressEpilog: the cache must be FP8 E5M2 or FP8 E4M3");
  MOCK_REQUIRE(mx->dtype == ACL_BF16, "KvCompressEpilog: x must be BF16");
  MOCK_REQUIRE(ms->dtype == ACL_INT32 || ms->dtype == ACL_INT64,
               "KvCompressEpilog: slotMapping must be INT32 or INT64");
  // 950PR microscale quantization: block-128 groups over the channel axis.
  MOCK_REQUIRE(quant_group_size == 128 || quant_group_size == kRoutedScaleBlock,
               "KvCompressEpilog: quantGroupSize must be 128 or 32, got " + std::to_string(quant_group_size));
  MOCK_REQUIRE(mx->dim(1) % quant_group_size == 0,
               "KvCompressEpilog: the x channel axis must be a whole number of quant groups");
  MOCK_REQUIRE(quant_mode >= 0, "KvCompressEpilog: quantMode must be non-negative");
  MOCK_REQUIRE(round_scale == 0 || round_scale == 1, "KvCompressEpilog: roundScale must be 0 or 1");

  RecordRefOutputPlan();
  *workspace_size = Align4k(static_cast<uint64_t>(mx->elements()) * 4) + kWorkspaceElementwise;
  *executor = NewExecutor("aclnnKvCompressEpilog", {kv_compress_cache_ref, x, slot_mapping});
  return 0;
}

// aclnnIndexerCompressEpilogV2: scatter indexer-side compressed rows.
aclnnStatus aclnnIndexerCompressEpilogV2GetWorkspaceSize(aclTensor* indexer_compress_cache_ref, const aclTensor* x,
                                                         const aclTensor* slot_mapping, int64_t layout,
                                                         int64_t block_stride, uint64_t* workspace_size,
                                                         aclOpExecutor** executor) {
  if (int error = ValidateCompressEpilog("IndexerCompressEpilogV2", indexer_compress_cache_ref, x, slot_mapping,
                                         layout, block_stride)) {
    return error;
  }
  const MockAclTensor* mc = AsMockTensor(indexer_compress_cache_ref);
  const MockAclTensor* mx = AsMockTensor(x);
  const MockAclTensor* ms = AsMockTensor(slot_mapping);
  MOCK_REQUIRE(mc->dtype == ACL_UINT8, "IndexerCompressEpilogV2: the indexer cache must be UINT8");
  MOCK_REQUIRE(mx->dtype == ACL_BF16 || mx->dtype == ACL_FLOAT16,
               "IndexerCompressEpilogV2: x must be BF16 or FP16");
  MOCK_REQUIRE(ms->dtype == ACL_INT32, "IndexerCompressEpilogV2: slotMapping must be INT32");
  // The indexer stream is the 128-wide quantized query/key the lightning
  // indexer scores.
  MOCK_REQUIRE(mx->dim(1) == 128,
               "IndexerCompressEpilogV2: the indexer channel width is 128, got " + std::to_string(mx->dim(1)));

  RecordRefOutputPlan();
  *workspace_size = Align4k(static_cast<uint64_t>(mx->elements()) * 2) + kWorkspaceElementwise;
  *executor = NewExecutor("aclnnIndexerCompressEpilogV2", {indexer_compress_cache_ref, x, slot_mapping});
  return 0;
}

// -- execution: every launch is a validated no-op -----------------------------

#define MOCK_NOOP_LAUNCH(name)                                                             \
  aclnnStatus name(void* workspace, uint64_t workspace_size, aclOpExecutor* executor,      \
                   aclrtStream stream) {                                                   \
    if (int error = RecordRuntimeCall(#name)) return error;                               \
    (void)workspace;                                                                       \
    (void)workspace_size;                                                                  \
    (void)executor;                                                                        \
    (void)stream;                                                                          \
    return 0;                                                                              \
  }

MOCK_NOOP_LAUNCH(aclnnRmsNorm)
MOCK_NOOP_LAUNCH(aclnnRmsNormDynamicMxQuant)
MOCK_NOOP_LAUNCH(aclnnDynamicMxQuant)
MOCK_NOOP_LAUNCH(aclnnMatmul)
MOCK_NOOP_LAUNCH(aclnnCast)
MOCK_NOOP_LAUNCH(aclnnQuantMatmulV5)
MOCK_NOOP_LAUNCH(aclnnApplyRotaryPosEmbV2)
MOCK_NOOP_LAUNCH(aclnnScatterPaKvCache)
MOCK_NOOP_LAUNCH(aclnnFusedInferAttentionScoreV5)
MOCK_NOOP_LAUNCH(aclnnSigmoid)
MOCK_NOOP_LAUNCH(aclnnSqrt)
MOCK_NOOP_LAUNCH(aclnnSoftplus)
MOCK_NOOP_LAUNCH(aclnnMul)
MOCK_NOOP_LAUNCH(aclnnInplaceAdd)
MOCK_NOOP_LAUNCH(aclnnSwiGlu)
MOCK_NOOP_LAUNCH(aclnnArgMax)
MOCK_NOOP_LAUNCH(aclnnMoeGatingTopKV2)
MOCK_NOOP_LAUNCH(aclnnMoeInitRoutingV4)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulV5)
MOCK_NOOP_LAUNCH(aclnnSwigluMxQuant)
MOCK_NOOP_LAUNCH(aclnnMoeTokenUnpermute)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulSwigluQuantV2)
MOCK_NOOP_LAUNCH(aclnnGroupedMatmulFinalizeRoutingV3)
MOCK_NOOP_LAUNCH(aclnnMhcPre)
MOCK_NOOP_LAUNCH(aclnnMhcSinkhorn)
MOCK_NOOP_LAUNCH(aclnnMhcPost)
MOCK_NOOP_LAUNCH(aclnnHcPre)
MOCK_NOOP_LAUNCH(aclnnHcPreInvRms)
MOCK_NOOP_LAUNCH(aclnnHcPreSinkhorn)
MOCK_NOOP_LAUNCH(aclnnHcPost)
MOCK_NOOP_LAUNCH(aclnnQuantLightningIndexer)
MOCK_NOOP_LAUNCH(aclnnCompressor)
MOCK_NOOP_LAUNCH(aclnnVllmQuantLightningIndexer)
MOCK_NOOP_LAUNCH(aclnnKvQuantSparseAttnSharedkv)
MOCK_NOOP_LAUNCH(aclnnKvCompressEpilog)
MOCK_NOOP_LAUNCH(aclnnIndexerCompressEpilogV2)

#undef MOCK_NOOP_LAUNCH

}  // extern "C"

namespace ascend_moe {
namespace mock {

// Direct entry points for the test's negative cases (no executor plumbing).
aclnnStatus MockValidateGatingForTest(const aclTensor* x, const aclTensor* bias_optional, int64_t k, int64_t k_group,
                                      int64_t group_count, int64_t group_select_mode, int64_t renorm,
                                      int64_t norm_type, double routed_scaling_factor, const aclTensor* y_out,
                                      const aclTensor* expert_idx_out) {
  return ValidateGating(x, bias_optional, k, k_group, group_count, group_select_mode, renorm, norm_type,
                        routed_scaling_factor, y_out, expert_idx_out);
}

aclnnStatus MockValidateRoutingForTest(const aclTensor* expert_idx, int64_t expert_num,
                                       const aclTensor* group_list_out) {
  return ValidateRouting(expert_idx, expert_num, group_list_out);
}

aclnnStatus MockValidateGmmForTest(const aclTensorList* weight, const aclTensorList* scale_optional,
                                   int64_t split_item, int64_t group_type) {
  return ValidateGmm(weight, scale_optional, split_item, group_type);
}

// Plans across the whole vendored operator set that would issue a same-address
// ViewCopy -- the manual-4.31 condition that makes an executor non-reusable.
// Every wrapper in third_party/ops_dsv4 is written to avoid it, so the
// invariant this exists to hold is that it stays 0.
int MockVendorSelfCopyHazards() { return g_vendor_selfcopy_hazards; }

// Plans where the patched aclnnMhcSinkhorn wrapper elided its trailing
// ViewCopy because the kernel had already written the caller tensor -- the
// contiguous-output case upstream copied onto its own address.
int MockSinkhornSelfCopyElisions() { return g_sinkhorn_selfcopy_elided; }

// Plans of a vendored operator whose output IS one of its inputs
// (Compressor.stateCache, aclnnKvCompressEpilog, aclnnIndexerCompressEpilogV2).
int MockRefOutputPlans() { return g_ref_output_plans; }

}  // namespace mock
}  // namespace ascend_moe
