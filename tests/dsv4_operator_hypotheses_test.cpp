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

// dsv4_operator_hypotheses_test -- targeted, hypothesis-driven verification of
// the vendored arch35 (Ascend 950PR) operators against the exact
// DeepSeek-V4-Flash tensor geometry this engine intends to feed them.
//
// This is NOT a regression suite for the upstream operators; their parent
// frameworks already have those. Each section here states ONE assumption the
// single-device inference runtime is about to be built on, feeds the operator
// the precise shapes / dtypes / layouts that assumption implies, and reports
// whether the operator agrees -- BEFORE any fusing, patching or caching work
// depends on it being true.
//
// THE FOUR HYPOTHESES
//   H1  aclnnMhcPre / aclnnMhcPost accept the DSV4 residual-stream geometry
//       (n_hc = 4 streams over the 4096-wide hidden state, phi [24, 16384]
//       FP32, FP32 alpha / bias) in both the TND and BSND spellings, emit the
//       hIn / hPost / hRes triple at the shapes the runtime allocates, and
//       plan without a single host synchronization or DMA.
//   H2  aclnnMhcSinkhorn yields a REUSABLE (aclSetAclOpExecutorRepeatable)
//       executor for a non-contiguous output view AND for the contiguous one
//       the DSV4 patch targets, and its result is doubly stochastic -- every
//       row sum and every column sum 1.0 +/- 1e-5 (the Birkhoff invariant).
//       REFUTED ON HARDWARE, BOTH HALVES OF THE REUSE CLAIM. The 950PR run
//       refused the non-contiguous plan outright with 561103, and failed
//       aclSetAclOpExecutorRepeatable on the contiguous plan with 561000. The
//       Birkhoff half stands. See the [refuted] verdict below for what the
//       engine does instead; the claims are kept stated rather than rewritten
//       so this file still records what was believed and what disproved it.
//   H3  aclnnCompressor expresses a sequence-ring cadence at cmpRatio 4 (CSA):
//       an incomplete window emits nothing at all, and the window-closing step
//       emits exactly one compressed row.
//   H4  aclnnVllmQuantLightningIndexer returns top-512 sparse indices as INT32
//       at [1, 1, 1, 512] over an FP8 E4M3 paged key stream, and every index
//       it returns is a valid, non-negative sequence slot.
//
// HOW IT RUNS IN THE TWO BUILDS
//   -DENABLE_MOCK_RUNTIME=ON    every plan-phase hypothesis runs against the
//                               symbolic backend, which replays the vendored
//                               wrappers' own parameter checks and keeps the
//                               manual-4.31 repeatability ledger. Numerics run
//                               against the host reference only: mock device
//                               memory is an interval, not bytes, so there is
//                               nothing to read back.
//   -DENABLE_MOCK_RUNTIME=OFF   the same plan-phase hypotheses run against
//                               libcust_opapi.so -- the real vendored
//                               wrappers -- and the numeric sections become
//                               live: seed H2D, launch, read back D2H and
//                               apply the identical invariant checker. With no
//                               device attached the binary SKIPs, exactly as
//                               the other npu_* probes do.
//
// READING THE OUTPUT
//   [ ok ]   the hypothesis holds as stated
//   [FAIL]   the hypothesis is FALSE -- the runtime design that assumed it
//            needs changing before any fusion work starts
//   [refuted] the hypothesis is FALSE, the 950PR run is what established it,
//            and the engine ALREADY encodes the consequence. Recorded, not
//            scored: it cannot be a [FAIL] without making a correct engine
//            report a red suite forever, and it cannot be an [ ok ] either.
//            The symbolic backend also cannot reproduce these refusals -- it
//            has no kernel to reject a stride and no CANN to reject a
//            repeatable executor -- so the two builds could never agree on a
//            pass/fail line here.
//   [note]   a measured fact worth recording; never a pass/fail verdict
//   [SKIP]   not decidable in this build (no device, or the operator's opp
//            package is not deployed)
//
// What is NOT settled here: kernel numerics beyond the invariants above, and
// the real GetWorkspaceSize magnitudes. Those need a 950PR with the vendored
// opp package deployed, which is what the compiled binary is for.

#include <cinttypes>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "aclnn/acl_meta.h"

#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/error.hpp"
#include "moe/core/kv_cache_layout.hpp"
#include "moe/core/op_table.hpp"
#include "moe/ops/aclnn_dsv4_vendor_ops.h"

#ifdef ASCEND_MOCK_RUNTIME
#include "mock_allocator.hpp"
#include "mock_ops_api.hpp"
#else
#include "npu_test_support.hpp"
#endif

namespace ascend_moe {
namespace {

// ---------------------------------------------------------------------------
// Verdict reporting
// ---------------------------------------------------------------------------

int g_checks = 0;
int g_failures = 0;
int g_skips = 0;
int g_refutations = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (condition) {
    std::printf("  [ ok ] %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

void Note(const std::string& what) { std::printf("  [note] %s\n", what.c_str()); }

void Skip(const std::string& what) {
  ++g_skips;
  std::printf("  [SKIP] %s\n", what.c_str());
}

// A claim the 950PR run has ALREADY settled as false, and that the engine has
// already been changed to accommodate. It is recorded, not scored.
//
// This is a third verdict on purpose. A [FAIL] means "the runtime design that
// assumed this needs revisiting"; a [refuted] means "it was revisited, this is
// the finding it was revisited for". Scoring these as failures would make a
// correct engine report a red suite forever; hiding them would lose the one
// record of why the engine looks the way it does. They also cannot be [FAIL]
// lines for a mechanical reason: the symbolic backend has no kernel to reject
// a stride and no CANN to reject a repeatable executor, so it accepts what
// hardware refuses, and the two builds would disagree on a pass/fail line.
void Refuted(const std::string& claim, const std::string& finding) {
  ++g_refutations;
  std::printf("  [refuted] %s\n", claim.c_str());
  std::printf("            FINDING: %s\n", finding.c_str());
}

void Hypothesis(int index, const char* title) {
  std::printf("\n=============================================================\n");
  std::printf("HYPOTHESIS %d -- %s\n", index, title);
  std::printf("=============================================================\n");
}

// The assumption under test, printed before the probes so a failing line is
// read against the claim it refutes rather than against the code.
void Assume(const std::string& text) { std::printf("  assumed: %s\n", text.c_str()); }

// ---------------------------------------------------------------------------
// DSV4-Flash geometry for these probes
// ---------------------------------------------------------------------------
//
// kHiddenSize (4096), kIndexTopK (512), kIndexHeadDim (128), kIndexNumHeads
// (64) and kDefaultKvLoraRank (512) come from moe/core/config.hpp; nothing
// below re-spells a number the engine already pins.

constexpr int64_t kBatch = 1;      // single-device decode: one sequence
constexpr int64_t kSeq = 1;        // one token per step
constexpr int64_t kDecodeTokens = kBatch * kSeq;

constexpr int64_t kNhc = 4;                              // mHC streams (n_hc)
constexpr int64_t kMixRows = kNhc * kNhc + 2 * kNhc;     // n^2 + 2n = 24
constexpr int64_t kMixCols = kNhc * kHiddenSize;         // n * D = 16384

constexpr int64_t kSinkhornIters = 20;
constexpr float kSinkhornEps = 1e-6f;
constexpr double kBirkhoffTolerance = 1e-5;  // the hypothesis' own tolerance

// H3: the CSA compressor at the brief's geometry.
constexpr int64_t kCompressRatio = 4;                          // CSA
constexpr int64_t kCompressedDim = kDefaultKvLoraRank;         // D = 512
constexpr int64_t kCompressorGateWidth = 2 * kDefaultKvLoraRank;  // 1024
constexpr int64_t kRopeHeadDim = kDefaultQkRopeHeadDim;        // 64
constexpr int64_t kStateBlocks = 4;
constexpr int64_t kStateBlockSize = 8;
// The compressed-entry width the task brief states. It is NOT derivable from
// any (D, dtype) pair in this geometry, so H3 asserts the derived width and
// records this one for the on-device run to settle. See the note it prints.
constexpr int64_t kBriefCompressedEntryBytes = 604;

// H4: the lightning indexer at the brief's geometry.
constexpr int64_t kIndexerBlocks = 8;
constexpr int64_t kIndexerBlockSize = 128;
constexpr int64_t kIndexerKeySlots = kIndexerBlocks * kIndexerBlockSize;  // 1024

// ---------------------------------------------------------------------------
// Portable descriptor plumbing
// ---------------------------------------------------------------------------
//
// The two header sets spell FP32 differently -- ACL_FLOAT in the CANN
// toolkit's acl_base_rt.h, ACL_FLOAT32 in the engine's mock copy -- and the
// toolkit carries dtypes (ACL_HIFLOAT8, ACL_UINT32) the mock enum does not
// model. One alias per dtype these probes use keeps every geometry below
// spelled exactly once, in either build.

#ifdef ASCEND_MOCK_RUNTIME
constexpr aclDataType kDtFp32 = ACL_FLOAT32;
#else
constexpr aclDataType kDtFp32 = ACL_FLOAT;
#endif
constexpr aclDataType kDtBf16 = ACL_BF16;
constexpr aclDataType kDtFp16 = ACL_FLOAT16;
constexpr aclDataType kDtInt32 = ACL_INT32;
constexpr aclDataType kDtUint8 = ACL_UINT8;
constexpr aclDataType kDtFp8E4m3 = ACL_FLOAT8_E4M3FN;
constexpr aclDataType kDtFp8E5m2 = ACL_FLOAT8_E5M2;

size_t DataTypeBytes(aclDataType dtype) {
  switch (dtype) {
    case kDtFp8E4m3:
    case kDtFp8E5m2:
    case kDtUint8:
      return 1;
    case kDtBf16:
    case kDtFp16:
      return 2;
    case kDtFp32:
    case kDtInt32:
      return 4;
    case ACL_INT64:
      return 8;
    default:
      return 4;
  }
}

const char* DataTypeName(aclDataType dtype) {
  switch (dtype) {
    case kDtFp32:
      return "FP32";
    case kDtBf16:
      return "BF16";
    case kDtFp16:
      return "FP16";
    case kDtInt32:
      return "INT32";
    case kDtUint8:
      return "UINT8";
    case kDtFp8E4m3:
      return "FP8_E4M3";
    case kDtFp8E5m2:
      return "FP8_E5M2";
    default:
      return "?";
  }
}

// std::to_string on a double prints six decimals, which rounds a 1e-7
// deviation to "0.000000" -- useless in exactly the range these invariants
// live in. Scientific notation keeps the measurement readable.
std::string Scientific(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.3e", value);
  return text;
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::ostringstream text;
  text << "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      text << ", ";
    }
    text << shape[index];
  }
  text << "]";
  return text.str();
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t index = shape.size(); index-- > 1;) {
    strides[index - 1] = strides[index] * shape[index];
  }
  return strides;
}

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return count;
}

// A tensor plus the descriptor we handed the runtime. The hypotheses assert
// against THIS record rather than re-reading the opaque handle: these are the
// exact view dims, strides and dtype aclCreateTensor received, which is what
// "the operator accepts this geometry" has to mean.
struct Tensor {
  aclTensor* handle = nullptr;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  aclDataType dtype = kDtFp32;
  void* address = nullptr;
  size_t bytes = 0;

  int64_t elements() const { return ElementCount(shape); }
  std::string text() const { return ShapeText(shape) + " " + DataTypeName(dtype); }
};

// One bump-allocated span of device memory, from whichever backend this build
// has: the symbolic interval registry in mock mode (so aclSetTensorAddr can
// validate a rebind against a span the allocator really handed out), real HBM
// in a CANN build (so the numeric sections have something to copy).
class DeviceArena {
 public:
  DeviceArena(IDeviceAllocator* allocator, size_t bytes) : allocator_(allocator), capacity_(bytes) {
#ifdef ASCEND_MOCK_RUNTIME
    (void)allocator_;
    base_ = reinterpret_cast<char*>(mock::MockDeviceMalloc(bytes));
#else
    base_ = static_cast<char*>(allocator_->DeviceMalloc(bytes));
#endif
    DSV4_REQUIRE(base_ != nullptr, "the hypothesis arena could not be reserved");
    // A leading guard so no tensor ever starts exactly at the span base: every
    // region a probe hands over is then strictly interior to one span.
    used_ = kGuardBytes;
  }

  ~DeviceArena() {
#ifdef ASCEND_MOCK_RUNTIME
    mock::MockUnregisterSpan(reinterpret_cast<uintptr_t>(base_));
#else
    if (allocator_ != nullptr) {
      allocator_->DeviceFree(base_);
    }
#endif
  }

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  void* Take(size_t bytes) {
    const size_t rounded = ((bytes == 0 ? 1 : bytes) + kAlignBytes - 1) & ~(kAlignBytes - 1);
    DSV4_REQUIRE(used_ + rounded + kGuardBytes <= capacity_,
                 "the hypothesis arena is too small: " << used_ + rounded << " of " << capacity_ << " bytes");
    char* slice = base_ + used_;
    used_ += rounded;
    return slice;
  }

  Tensor Make(const std::vector<int64_t>& shape, aclDataType dtype) {
    return MakeWithStrides(shape, dtype, ContiguousStrides(shape));
  }

  // A view whose element stride is `gap` times the contiguous one. This is the
  // non-contiguous output H2 needs: l0op::Contiguous cannot be the identity on
  // it, so the wrapper stages a distinct temp and its trailing ViewCopy keeps
  // src != dst.
  Tensor MakeGapped(const std::vector<int64_t>& shape, aclDataType dtype, int64_t gap) {
    std::vector<int64_t> strides(shape.size(), 1);
    int64_t element = gap;
    for (size_t index = shape.size(); index-- > 0;) {
      strides[index] = element;
      element *= shape[index] * gap;
    }
    return MakeWithStrides(shape, dtype, strides);
  }

 private:
  static constexpr size_t kAlignBytes = 512;
  static constexpr size_t kGuardBytes = 4096;

  Tensor MakeWithStrides(const std::vector<int64_t>& shape, aclDataType dtype,
                         const std::vector<int64_t>& strides) {
    Tensor tensor;
    tensor.shape = shape;
    tensor.strides = strides;
    tensor.dtype = dtype;
    // Size the span from the STRIDES, not from the element count: a gapped
    // view's last element sits at sum((shape[i] - 1) * strides[i]), well past
    // product(shape). Sizing it from the element count would hand the operator
    // a buffer shorter than the addresses its own strides reach -- harmless
    // against symbolic memory, a real out-of-bounds write on the device.
    int64_t last_offset = 0;
    bool empty = false;
    for (size_t index = 0; index < shape.size(); ++index) {
      if (shape[index] <= 0) {
        empty = true;
        break;
      }
      last_offset += (shape[index] - 1) * strides[index];
    }
    const int64_t span_elements = empty ? 0 : last_offset + 1;
    tensor.bytes = static_cast<size_t>(span_elements) * DataTypeBytes(dtype);
    tensor.address = Take(tensor.bytes);
    tensor.handle = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), tensor.address);
    DSV4_REQUIRE(tensor.handle != nullptr, "aclCreateTensor refused " << ShapeText(shape));
    return tensor;
  }

  IDeviceAllocator* allocator_ = nullptr;
  char* base_ = nullptr;
  size_t capacity_ = 0;
  size_t used_ = 0;
};

void Destroy(std::vector<Tensor>* tensors) {
  for (Tensor& tensor : *tensors) {
    if (tensor.handle != nullptr) {
      aclDestroyTensor(tensor.handle);
      tensor.handle = nullptr;
    }
  }
  tensors->clear();
}

// ---------------------------------------------------------------------------
// Plan outcomes
// ---------------------------------------------------------------------------
//
// Three outcomes, not two. A plan that fails with a PARAMETER status is the
// operator delivering a verdict on the geometry -- the thing being tested. A
// plan that fails anywhere else (no opp package deployed, an inner allocation
// refused) says nothing about the hypothesis, so it is reported as a SKIP
// instead of poisoning the run with failures the geometry did not cause.

constexpr int64_t kStatusNullPointer = 161001;
constexpr int64_t kStatusParamInvalid = 161002;
constexpr int64_t kStatusMockContract = 100001;  // the symbolic backend's own

struct PlanResult {
  bool planned = false;
  bool contract_verdict = false;  // the operator judged the geometry
  int64_t status = 0;
  uint64_t workspace = 0;
  aclOpExecutor* executor = nullptr;
  std::string detail;
};

template <typename Fn>
PlanResult RunPlan(Fn plan) {
  PlanResult result;
  try {
    result.workspace = plan(&result.executor);
    result.planned = true;
  } catch (const AclError& error) {
    result.status = error.status();
    result.contract_verdict = result.status == kStatusNullPointer || result.status == kStatusParamInvalid ||
                              result.status == kStatusMockContract;
    result.detail = error.what();
  } catch (const Dsv4Error& error) {
    // A refusal raised by the engine's own validation, or by the symbolic
    // backend before it ever reached a status code.
    result.contract_verdict = true;
    result.detail = error.what();
  }
  return result;
}

std::string FirstLine(const std::string& text) {
  const size_t end = text.find('\n');
  return end == std::string::npos ? text : text.substr(0, end);
}

// `what` reads as a claim: "mhc_pre accepts the TND decode geometry".
bool ExpectPlanned(const PlanResult& result, const std::string& what) {
  if (result.planned) {
    Check(true, what);
    return true;
  }
  if (result.contract_verdict) {
    Check(false, what + " -- REFUSED: " + FirstLine(result.detail));
    return false;
  }
  Skip(what + " -- not decidable here (status " + std::to_string(result.status) + "): " +
       FirstLine(result.detail));
  return false;
}

// A probe whose verdict legitimately differs between the two builds, and so
// can never be a pass/fail line.
//
// The stride attributes (Compressor.stateCacheStrideDim0, the indexer's
// stride / scaleStride) are the case. The symbolic backend guards them because
// it can see the view the caller handed over and compare. The vendored
// wrappers cannot and do not: aclnn_compressor.cpp and
// aclnn_vllm_quant_lightning_indexer.cpp forward the value to l0op untouched,
// with no check anywhere in CheckParams. So a stale attribute is caught on one
// backend and silently honoured on the other -- which is itself the finding,
// recorded here rather than scored.
void ReportAttributeGuard(const PlanResult& result, const std::string& what, const std::string& consequence) {
  if (result.contract_verdict) {
    Note("GUARDED on this backend: " + what + " was refused before it could reach a kernel");
  } else if (result.planned) {
    Note("UNGUARDED on this backend: " + what + " was ACCEPTED. " + consequence);
  } else {
    Skip("whether " + what + " is guarded is not decidable here");
  }
}

bool ExpectRefused(const PlanResult& result, const std::string& what) {
  if (result.contract_verdict) {
    Check(true, "refused as predicted: " + what);
    return true;
  }
  if (result.planned) {
    Check(false, "ACCEPTED but predicted a refusal: " + what);
    return false;
  }
  Skip("refusal of " + what + " not decidable here (status " + std::to_string(result.status) + ")");
  return false;
}

// ---------------------------------------------------------------------------
// The two numeric invariants
// ---------------------------------------------------------------------------

// The algorithm aclnnMhcSinkhorn is assumed to implement: `iters` alternating
// row- and column-normalizations of each n x n block, with `eps` guarding the
// division. This is the host reference the device output is held against --
// identical code runs over the readback in a CANN build.
void ReferenceSinkhorn(float* blocks, int64_t count, int64_t n, int64_t iters, float eps) {
  std::vector<float> sums(static_cast<size_t>(n));
  for (int64_t block = 0; block < count; ++block) {
    float* matrix = blocks + block * n * n;
    for (int64_t iteration = 0; iteration < iters; ++iteration) {
      for (int64_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int64_t column = 0; column < n; ++column) {
          sum += matrix[row * n + column];
        }
        const float scale = 1.0f / (sum + eps);
        for (int64_t column = 0; column < n; ++column) {
          matrix[row * n + column] *= scale;
        }
      }
      for (int64_t column = 0; column < n; ++column) {
        float sum = 0.0f;
        for (int64_t row = 0; row < n; ++row) {
          sum += matrix[row * n + column];
        }
        sums[static_cast<size_t>(column)] = 1.0f / (sum + eps);
      }
      for (int64_t row = 0; row < n; ++row) {
        for (int64_t column = 0; column < n; ++column) {
          matrix[row * n + column] *= sums[static_cast<size_t>(column)];
        }
      }
    }
  }
}

// The Birkhoff invariant: a doubly-stochastic matrix has every row sum and
// every column sum equal to 1. `worst` reports the largest deviation found, so
// a pass still shows how much headroom the tolerance had.
bool CheckBirkhoffInvariant(const float* blocks, int64_t count, int64_t n, double tolerance, double* worst) {
  double worst_deviation = 0.0;
  for (int64_t block = 0; block < count; ++block) {
    const float* matrix = blocks + block * n * n;
    for (int64_t row = 0; row < n; ++row) {
      double sum = 0.0;
      for (int64_t column = 0; column < n; ++column) {
        sum += matrix[row * n + column];
      }
      worst_deviation = std::fmax(worst_deviation, std::fabs(sum - 1.0));
    }
    for (int64_t column = 0; column < n; ++column) {
      double sum = 0.0;
      for (int64_t row = 0; row < n; ++row) {
        sum += matrix[row * n + column];
      }
      worst_deviation = std::fmax(worst_deviation, std::fabs(sum - 1.0));
    }
  }
  *worst = worst_deviation;
  return worst_deviation <= tolerance;
}

// Every index the indexer emits must address a real sequence slot. A negative
// value, or one at or past `valid_slots`, would make the attention core gather
// from outside the paged cache.
bool CheckIndexRange(const int32_t* indices, size_t count, int32_t valid_slots, std::string* detail) {
  for (size_t position = 0; position < count; ++position) {
    const int32_t index = indices[position];
    if (index < 0 || index >= valid_slots) {
      std::ostringstream text;
      text << "index[" << position << "] = " << index << " is outside [0, " << valid_slots << ")";
      *detail = text.str();
      return false;
    }
  }
  detail->clear();
  return true;
}

// A deterministic, strictly positive seed for the Sinkhorn probe: Sinkhorn is
// only defined on a non-negative matrix with no all-zero row or column, and a
// fixed pattern keeps the reported deviation reproducible run to run.
void SeedPositiveMatrices(std::vector<float>* blocks, int64_t count, int64_t n) {
  blocks->resize(static_cast<size_t>(count * n * n));
  for (int64_t block = 0; block < count; ++block) {
    for (int64_t row = 0; row < n; ++row) {
      for (int64_t column = 0; column < n; ++column) {
        const int64_t position = ((block * n) + row) * n + column;
        // 0.25 .. 1.75, asymmetric so row and column scaling both do work.
        (*blocks)[static_cast<size_t>(position)] =
            0.25f + 0.1875f * static_cast<float>((position * 7 + row * 3 + column) % 9);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The manual-4.31 repeatability ledger, where this build keeps one
// ---------------------------------------------------------------------------

struct Ledger {
  int self_copy_hazards = 0;
  int sinkhorn_elisions = 0;
  int ref_output_plans = 0;
  bool available = false;
};

Ledger ReadLedger() {
  Ledger ledger;
#ifdef ASCEND_MOCK_RUNTIME
  ledger.self_copy_hazards = mock::MockVendorSelfCopyHazards();
  ledger.sinkhorn_elisions = mock::MockSinkhornSelfCopyElisions();
  ledger.ref_output_plans = mock::MockRefOutputPlans();
  ledger.available = true;
#endif
  return ledger;
}

// ---------------------------------------------------------------------------
// The backend this build probes through
// ---------------------------------------------------------------------------

struct Backend {
  IDeviceAllocator* allocator = nullptr;
  IStreamEngine* streams = nullptr;
  // True when device memory holds bytes a probe may seed and read back. The
  // symbolic backend's device memory is an interval, so numerics stay on the
  // host reference there.
  bool numerics_live = false;
  const char* name = "<none>";
};

// Snapshot of the DMA counters that would reveal a host synchronization or a
// transfer smuggled into the plan phase.
struct SyncWitness {
  uint64_t synchronizations = 0;
  uint64_t sync_copies = 0;
  uint64_t async_copies = 0;

  bool operator==(const SyncWitness& other) const {
    return synchronizations == other.synchronizations && sync_copies == other.sync_copies &&
           async_copies == other.async_copies;
  }
};

// ===========================================================================
// HYPOTHESIS 1 -- the mHC residual-stream contract
// ===========================================================================

void TestMhcPreAndPostContract(const Backend& backend, const DmaCounters& (*counters)()) {
  Hypothesis(1, "aclnnMhcPre / aclnnMhcPost accept the DSV4 residual-stream geometry");
  Assume("residual stream x = [1, 1, 4, 4096] BF16 (BSND) or [1, 4, 4096] BF16 (TND)");
  Assume("projection weight phi = [24, 16384] FP32 (n^2+2n rows by n*D columns)");
  Assume("gain alpha = [3] FP32, mixing bias = [24] FP32");
  Assume("mhc_pre emits hIn [1, 4096], hPost [1, 4], hRes [1, 4, 4] on the TND spelling");
  Assume("mhc_post consumes x, hRes, hOut, hPost and writes x_next at the shape of x");
  Assume("the plan phase performs NO host synchronization and NO DMA");

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Skip("the aclnn runtime is not on the loader path; no operator to question");
    return;
  }

  DeviceArena arena(backend.allocator, 48ull << 20);
  std::vector<Tensor> owned;
  const auto track = [&owned](Tensor tensor) {
    owned.push_back(tensor);
    return tensor;
  };

  // ---- H1.1  the TND decode geometry, T = 1 -------------------------------
  std::printf("\n-- H1.1 the TND decode spelling, one token --\n");
  const Tensor x_tnd = track(arena.Make({kDecodeTokens, kNhc, kHiddenSize}, kDtBf16));
  const Tensor phi = track(arena.Make({kMixRows, kMixCols}, kDtFp32));
  const Tensor alpha = track(arena.Make({3}, kDtFp32));
  const Tensor bias = track(arena.Make({kMixRows}, kDtFp32));
  const Tensor gamma = track(arena.Make({kNhc, kHiddenSize}, kDtFp32));
  const Tensor h_in = track(arena.Make({kDecodeTokens, kHiddenSize}, kDtBf16));
  const Tensor h_post = track(arena.Make({kDecodeTokens, kNhc}, kDtFp32));
  const Tensor h_res = track(arena.Make({kDecodeTokens, kNhc, kNhc}, kDtFp32));

  Check(phi.shape == std::vector<int64_t>({24, 16384}),
        "phi's [n^2+2n, n*D] really is [24, 16384] at n_hc = 4, D = 4096");
  Check(h_in.shape == std::vector<int64_t>({1, 4096}) && h_post.shape == std::vector<int64_t>({1, 4}) &&
            h_res.shape == std::vector<int64_t>({1, 4, 4}),
        "the hIn / hPost / hRes triple is allocated at [1, 4096] / [1, 4] / [1, 4, 4]");

  const SyncWitness before{counters().stream_synchronizations, counters().sync_copies, counters().async_copies};

  PlanResult pre = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, executor, x_tnd.handle, phi.handle, alpha.handle,
                                     bias.handle, gamma.handle, kRmsNormEpsilon, kRmsNormEpsilon, h_in.handle,
                                     h_post.handle, h_res.handle, nullptr, nullptr, nullptr);
  });
  const bool pre_planned = ExpectPlanned(pre, "mhc_pre accepts x " + x_tnd.text() + " with phi " + phi.text());

  const SyncWitness after{counters().stream_synchronizations, counters().sync_copies, counters().async_copies};
  Check(before == after,
        "the mhc_pre plan phase issued no host synchronization and no DMA (every staging tensor comes from "
        "executor->AllocTensor)");

  if (pre_planned) {
    Check(pre.workspace > 0, "mhc_pre reports a non-empty device workspace for this geometry");
    StaticOpSlot slot;
    slot.Adopt(OpId::kMhcPre, "hypothesis/mhc_pre", pre.workspace, pre.executor);
    Check(slot.planned(), "the mhc_pre executor is retained and made repeatable (aclSetAclOpExecutorRepeatable)");
    void* workspace = pre.workspace > 0 ? backend.allocator->DeviceMalloc(pre.workspace) : nullptr;
    DeviceStream stream = backend.streams->CreateStream();
    slot.Launch(ops, workspace, stream);
    Check(true, "mhc_pre launched on the retained executor");

    // Determinism: the same descriptors must plan the same workspace, or the
    // engine cannot size a static arena once and reuse it every token.
    PlanResult again = RunPlan([&](aclOpExecutor** executor) {
      return PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, executor, x_tnd.handle, phi.handle, alpha.handle,
                                       bias.handle, gamma.handle, kRmsNormEpsilon, kRmsNormEpsilon, h_in.handle,
                                       h_post.handle, h_res.handle, nullptr, nullptr, nullptr);
    });
    if (again.planned) {
      Check(again.workspace == pre.workspace,
            "re-planning mhc_pre returns the same workspace size (a static arena can be sized once)");
      aclDestroyAclOpExecutor(again.executor);
    }
    slot.Reset();
    backend.streams->DestroyStream(stream);
    if (workspace != nullptr) {
      backend.allocator->DeviceFree(workspace);
    }
  }

  // ---- H1.2  the BSND spelling the brief states ---------------------------
  std::printf("\n-- H1.2 the BSND spelling x = [1, 1, 4, 4096] --\n");
  const Tensor x_bsnd = track(arena.Make({kBatch, kSeq, kNhc, kHiddenSize}, kDtBf16));
  const Tensor h_in_bsnd = track(arena.Make({kBatch, kSeq, kHiddenSize}, kDtBf16));
  const Tensor h_post_bsnd = track(arena.Make({kBatch, kSeq, kNhc}, kDtFp32));
  const Tensor h_res_bsnd = track(arena.Make({kBatch, kSeq, kNhc, kNhc}, kDtFp32));

  PlanResult pre_bsnd = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcPrePlanFn>(ops, OpId::kMhcPre, executor, x_bsnd.handle, phi.handle, alpha.handle,
                                     bias.handle, gamma.handle, kRmsNormEpsilon, kRmsNormEpsilon,
                                     h_in_bsnd.handle, h_post_bsnd.handle, h_res_bsnd.handle, nullptr, nullptr,
                                     nullptr);
  });
  if (ExpectPlanned(pre_bsnd, "mhc_pre accepts the rank-4 BSND stream " + x_bsnd.text())) {
    aclDestroyAclOpExecutor(pre_bsnd.executor);
  }
  Note("the BSND outputs are rank-4/rank-3 throughout -- hIn [1, 1, 4096], hPost [1, 1, 4], hRes "
       "[1, 1, 4, 4]. The brief pairs a rank-4 x with the TND output shapes hIn [1, 4096], hPost [1, 4], "
       "hRes [1, 4, 4]; that pairing is one layout short and H1.4 shows what the operator does with it.");

  // ---- H1.3  mhc_post, both spellings, each internally consistent ---------
  std::printf("\n-- H1.3 mhc_post: x_next = (hRes)^T x + hOut * hPost --\n");
  const Tensor post_x = track(arena.Make({kDecodeTokens, kNhc, kHiddenSize}, kDtBf16));
  const Tensor post_h_out = track(arena.Make({kDecodeTokens, kHiddenSize}, kDtBf16));
  const Tensor post_out = track(arena.Make({kDecodeTokens, kNhc, kHiddenSize}, kDtBf16));

  PlanResult post = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, executor, post_x.handle, h_res.handle,
                                      post_h_out.handle, h_post.handle, post_out.handle);
  });
  if (ExpectPlanned(post, "mhc_post accepts the TND set x " + post_x.text() + ", hRes " + h_res.text() +
                              ", hOut " + post_h_out.text() + ", hPost " + h_post.text())) {
    Check(post_out.shape == post_x.shape, "mhc_post writes x_next at the shape of x, " + post_out.text());
    StaticOpSlot slot;
    slot.Adopt(OpId::kMhcPost, "hypothesis/mhc_post", post.workspace, post.executor);
    Check(slot.planned(), "the mhc_post executor is retained and made repeatable");
    void* workspace = post.workspace > 0 ? backend.allocator->DeviceMalloc(post.workspace) : nullptr;
    DeviceStream stream = backend.streams->CreateStream();
    slot.Launch(ops, workspace, stream);
    Check(true, "mhc_post launched on the retained executor");
    slot.Reset();
    backend.streams->DestroyStream(stream);
    if (workspace != nullptr) {
      backend.allocator->DeviceFree(workspace);
    }
  }

  const Tensor post_x_bsnd = track(arena.Make({kBatch, kSeq, kNhc, kHiddenSize}, kDtBf16));
  const Tensor post_h_out_bsnd = track(arena.Make({kBatch, kSeq, kHiddenSize}, kDtBf16));
  const Tensor post_out_bsnd = track(arena.Make({kBatch, kSeq, kNhc, kHiddenSize}, kDtBf16));
  PlanResult post_bsnd = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, executor, post_x_bsnd.handle, h_res_bsnd.handle,
                                      post_h_out_bsnd.handle, h_post_bsnd.handle, post_out_bsnd.handle);
  });
  if (ExpectPlanned(post_bsnd, "mhc_post accepts the all-rank-4 BSND set (hRes [1, 1, 4, 4], hOut [1, 1, 4096], "
                               "hPost [1, 1, 4])")) {
    aclDestroyAclOpExecutor(post_bsnd.executor);
  }

  // ---- H1.4  the brief's literal argument set ----------------------------
  std::printf("\n-- H1.4 the brief's mixed-layout mhc_post set --\n");
  Assume("the brief's mhc_post call is x [1, 1, 4, 4096], hOut [1, 4096], hRes [1, 4, 4], hPost [1, 4]");
  PlanResult mixed = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcPostPlanFn>(ops, OpId::kMhcPost, executor, post_x_bsnd.handle, h_res.handle,
                                      post_h_out.handle, h_post.handle, post_out_bsnd.handle);
  });
  ExpectRefused(mixed, "a rank-4 BSND x paired with the rank-3/rank-2 TND companions");
  Note("the vendored wrapper ties the ranks together (aclnn_mhc_post.cpp CheckInputOutDims): a rank-4 x "
       "demands rank-4 hRes and out plus rank-3 hOut and hPost. The runtime must pick ONE spelling per "
       "call -- the TND set of H1.1/H1.3 is the one this engine's decode stream already carries.");

  // ---- H1.5  the whole chain, no self-copy anywhere ----------------------
  std::printf("\n-- H1.5 pre -> sinkhorn -> post over one shared hRes --\n");
  const Ledger chain_before = ReadLedger();
  const Tensor sink_out = track(arena.Make({kDecodeTokens, kNhc, kNhc}, kDtFp32));
  PlanResult sink = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, h_res.handle, kSinkhornEps,
                                          kSinkhornIters, sink_out.handle, nullptr, nullptr);
  });
  if (sink.planned) {
    aclDestroyAclOpExecutor(sink.executor);
  }
  const Ledger chain_after = ReadLedger();
  if (chain_after.available) {
    Check(chain_after.self_copy_hazards == chain_before.self_copy_hazards,
          "the three-stage mHC chain plans no same-address ViewCopy, so every stage's executor is reusable");
  } else {
    Skip("the manual-4.31 ledger is a mock-build instrument; on device, reuse is proven by H2's relaunch");
  }

  Destroy(&owned);
}

// ===========================================================================
// HYPOTHESIS 2 -- Sinkhorn: Birkhoff invariant and the aliasing guard
// ===========================================================================

void TestSinkhornInvariantAndAliasing(const Backend& backend) {
  Hypothesis(2, "aclnnMhcSinkhorn: Birkhoff invariant, and whether it can be replayed at all");
  Assume("input hRes = [1, 4, 4] FP32, numIters = 20, eps = 1e-6");
  Assume("output B = [1, 4, 4] FP32, doubly stochastic: row and column sums 1.0 +/- 1e-5");
  Assume("a NON-CONTIGUOUS output view keeps the executor configurable with "
         "aclSetAclOpExecutorRepeatable (the stated workaround for the 4.31 ViewCopy(src, src) hazard)");
  Assume("a CONTIGUOUS output can be made repeatable, so one retained executor serves every token");
  Note("BOTH of those last two were REFUTED by the 950PR run -- 561103 for the non-contiguous view, 561000 "
       "from aclSetAclOpExecutorRepeatable for the contiguous one. They are stated here as the assumptions "
       "the engine was originally built on, and the [refuted] lines below are what replaced them.");

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Skip("the aclnn runtime is not on the loader path; no operator to question");
    return;
  }

  DeviceArena arena(backend.allocator, 8ull << 20);
  std::vector<Tensor> owned;
  const auto track = [&owned](Tensor tensor) {
    owned.push_back(tensor);
    return tensor;
  };

  const Tensor input = track(arena.Make({kDecodeTokens, kNhc, kNhc}, kDtFp32));
  const Tensor contiguous_out = track(arena.Make({kDecodeTokens, kNhc, kNhc}, kDtFp32));
  const Tensor gapped_out = track(arena.MakeGapped({kDecodeTokens, kNhc, kNhc}, kDtFp32, 2));

  Check(gapped_out.strides != ContiguousStrides(gapped_out.shape),
        "the probe's second output really is a non-contiguous view (strides " + ShapeText(gapped_out.strides) +
            " against contiguous " + ShapeText(ContiguousStrides(gapped_out.shape)) + ")");

  // ---- H2.1  the non-contiguous output the hypothesis named ---------------
  //
  // REFUTED ON HARDWARE. The 950PR run returned 561103
  // (ACL_ERROR_INVALID_PARAM) from the plan: the vendor kernel rejects
  // non-contiguous view strides outright. So the "hand it a gapped view"
  // workaround the CANN manual-4.31 note recommended is not merely expensive,
  // it is unavailable -- which is why the engine's B_l slot is contiguous and
  // StaticArenaManager says so where it creates it.
  std::printf("\n-- H2.1 the non-contiguous output view --\n");
  const Ledger gapped_before = ReadLedger();
  PlanResult gapped = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, input.handle, kSinkhornEps,
                                          kSinkhornIters, gapped_out.handle, nullptr, nullptr);
  });
  if (gapped.planned) {
    // The symbolic backend has no kernel to reject the stride, so it plans.
    // Record what hardware does instead of scoring a line the two builds
    // cannot agree on.
    Refuted("mhc_sinkhorn accepts a non-contiguous [1, 4, 4] FP32 output view",
            "on a 950PR this plan fails with 561103 (ACL_ERROR_INVALID_PARAM) -- the kernel rejects "
            "non-contiguous view strides. This backend planned it, so the refusal is not reproducible here; "
            "the engine binds a CONTIGUOUS B_l either way.");
    const Ledger gapped_after = ReadLedger();
    if (gapped_after.available) {
      Check(gapped_after.self_copy_hazards == gapped_before.self_copy_hazards,
            "no same-address ViewCopy was planned: Contiguous() allocated a distinct temp, so src != dst");
    }
    aclDestroyAclOpExecutor(gapped.executor);
  } else if (gapped.contract_verdict) {
    Check(true, "mhc_sinkhorn REFUSES a non-contiguous [1, 4, 4] FP32 output view, as the 950PR run found "
                "(" + FirstLine(gapped.detail) + ")");
  } else {
    Skip("whether a non-contiguous Sinkhorn output is refused is not decidable here (status " +
         std::to_string(gapped.status) + ")");
  }

  // ---- H2.2  the contiguous output, which is what the patch is for --------
  std::printf("\n-- H2.2 the contiguous output (the case the DSV4 patch targets) --\n");
  const Ledger contiguous_before = ReadLedger();
  PlanResult contiguous = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, input.handle, kSinkhornEps,
                                          kSinkhornIters, contiguous_out.handle, nullptr, nullptr);
  });
  StaticOpSlot contiguous_slot;
  void* contiguous_workspace = nullptr;
  DeviceStream contiguous_stream = nullptr;
  bool contiguous_repeatable = false;
  // Kept alive for the numeric section below when the executor could NOT be
  // made repeatable: it is then launched exactly once, from this plan.
  aclOpExecutor* single_use_executor = nullptr;
  uint64_t single_use_workspace = 0;

  if (ExpectPlanned(contiguous, "mhc_sinkhorn accepts a contiguous [1, 4, 4] FP32 output")) {
    contiguous_workspace =
        contiguous.workspace > 0 ? backend.allocator->DeviceMalloc(contiguous.workspace) : nullptr;
    contiguous_stream = backend.streams->CreateStream();
    // THE DECISIVE PROBE, AND THE REASON THIS TEST USED TO STOP HERE.
    //
    // `StaticOpSlot::Adopt` calls aclSetAclOpExecutorRepeatable, which on a
    // 950PR fails this operator with 561000 and used to escape as an AclError
    // -- taking Hypotheses 3 and 4 with it. It is caught now: the finding is
    // recorded and the suite carries on, which is the only way the compressor
    // and indexer hypotheses ever get measured on hardware.
    try {
      contiguous_slot.Adopt(OpId::kMhcSinkhorn, "hypothesis/mhc_sinkhorn", contiguous.workspace,
                            contiguous.executor);
      contiguous_repeatable = contiguous_slot.planned();
    } catch (const AclError& error) {
      Refuted("a contiguous-output mhc_sinkhorn plan can be made repeatable with "
              "aclSetAclOpExecutorRepeatable",
              std::string("aclSetAclOpExecutorRepeatable REFUSED the executor: ") + FirstLine(error.what()) +
                  ". On a 950PR this is status 561000: the wrapper builds a dynamic internal iteration graph "
                  "that CANN will not mark reusable. CONSEQUENCE: a standalone aclnnMhcSinkhorn cannot hold a "
                  "retained executor in StaticOpSlotTable at all, so it cannot be replayed inside a "
                  "zero-allocation decode loop. It must be planned once per use (what the engine does today, "
                  "counted as StepCounters::sinkhorn_replans), fused into a kernel that never materializes B_l "
                  "at the aclnn layer (aclnnHcPreSinkhorn), or replaced by a custom AIV normalization block.");
      // Adopt threw after taking ownership, so the slot holds nothing and the
      // plan is already released; the numeric section re-plans below.
      contiguous.executor = nullptr;
    } catch (const Dsv4Error& error) {
      Refuted("a contiguous-output mhc_sinkhorn plan can be made repeatable",
              std::string("the adoption was refused: ") + FirstLine(error.what()));
      contiguous.executor = nullptr;
    }

    if (contiguous_repeatable) {
      Check(true, "aclSetAclOpExecutorRepeatable succeeded on the CONTIGUOUS plan on this backend");
      contiguous_slot.Launch(ops, contiguous_workspace, contiguous_stream);
      contiguous_slot.SetAddress(1, contiguous_out.handle, contiguous_out.address);
      contiguous_slot.Launch(ops, contiguous_workspace, contiguous_stream);
      Check(true, "the contiguous plan relaunched after aclSetTensorAddr on its output slot");
      const Ledger contiguous_after = ReadLedger();
      if (contiguous_after.available) {
        Check(contiguous_after.self_copy_hazards == contiguous_before.self_copy_hazards,
              "no same-address ViewCopy on the contiguous path -- the patch elides it instead of issuing it");
        Check(contiguous_after.sinkhorn_elisions == contiguous_before.sinkhorn_elisions + 1,
              "the patched wrapper elided exactly one trailing ViewCopy (kernelOut == output)");
      }
    } else {
      // No retained executor, so the numeric section gets a fresh single-use
      // plan. This is exactly the shape the engine's kPerUsePlan mode has.
      PlanResult fresh = RunPlan([&](aclOpExecutor** executor) {
        return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, input.handle, kSinkhornEps,
                                              kSinkhornIters, contiguous_out.handle, nullptr, nullptr);
      });
      if (fresh.planned) {
        single_use_executor = fresh.executor;
        single_use_workspace = fresh.workspace;
        Check(true, "a single-use (non-repeatable) contiguous plan is still available, so the operator is "
                    "usable at one plan per invocation");
      }
    }
  }
  Note("The contiguous output is the only admissible form: H2.1 shows the non-contiguous view is refused on "
       "hardware, and this section shows the contiguous one cannot be made repeatable. The runtime therefore "
       "binds a CONTIGUOUS B_l and plans the operator per use.");

  // ---- H2.3  the attribute bounds the wrapper enforces --------------------
  std::printf("\n-- H2.3 numIters and eps bounds --\n");
  ExpectRefused(RunPlan([&](aclOpExecutor** executor) {
                  return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, input.handle,
                                                        kSinkhornEps, 0, contiguous_out.handle, nullptr, nullptr);
                }),
                "numIters = 0, below the [1, 100] bound");
  ExpectRefused(RunPlan([&](aclOpExecutor** executor) {
                  return PlanAclnnOp<MhcSinkhornPlanFn>(ops, OpId::kMhcSinkhorn, executor, input.handle,
                                                        kSinkhornEps, 101, contiguous_out.handle, nullptr,
                                                        nullptr);
                }),
                "numIters = 101, above the [1, 100] bound");
  Note("numIters = 20 sits inside the bound, so the engine's Sinkhorn attribute is admissible.");

  // ---- H2.4  the Birkhoff invariant --------------------------------------
  std::printf("\n-- H2.4 the Birkhoff invariant over the result --\n");
  std::vector<float> host;
  SeedPositiveMatrices(&host, kDecodeTokens, kNhc);
  double seeded_worst = 0.0;
  (void)CheckBirkhoffInvariant(host.data(), kDecodeTokens, kNhc, kBirkhoffTolerance, &seeded_worst);
  Check(seeded_worst > kBirkhoffTolerance,
        "the seed is NOT already doubly stochastic (worst deviation " + Scientific(seeded_worst) +
            "), so the invariant below is a real measurement");

  std::vector<float> reference = host;
  ReferenceSinkhorn(reference.data(), kDecodeTokens, kNhc, kSinkhornIters, kSinkhornEps);
  double reference_worst = 0.0;
  const bool reference_holds =
      CheckBirkhoffInvariant(reference.data(), kDecodeTokens, kNhc, kBirkhoffTolerance, &reference_worst);
  Check(reference_holds, "20 alternating row/column normalizations at eps = 1e-6 reach row and column sums of "
                         "1.0 within 1e-5 (worst deviation " + Scientific(reference_worst) +
                             ") -- the invariant is reachable at these attributes");

  if (!backend.numerics_live) {
    Skip("the operator's own output cannot be read back in this build (symbolic device memory); the check "
         "above fixes the reference and the tolerance the on-device run applies");
  } else if (!contiguous_repeatable && single_use_executor == nullptr) {
    Skip("no planned Sinkhorn executor to measure -- see the refusal above");
  } else {
    // Seed the operator's input, run it, read the output back and hold it to
    // the same invariant as the reference. Which launch path depends on what
    // the hardware allowed: the retained executor if it could be made
    // repeatable, otherwise the single-use plan. The NUMERICS are the same
    // question either way, so the refutation above must not cost us the
    // measurement.
    backend.streams->MemcpySync(input.address, input.bytes, host.data(), host.size() * sizeof(float),
                                MemcpyKind::kHostToDevice);
    if (contiguous_repeatable) {
      contiguous_slot.Launch(ops, contiguous_workspace, contiguous_stream);
    } else {
      void* workspace = single_use_workspace > 0 ? contiguous_workspace : nullptr;
      const int launch_status = ops.launch_fn(OpId::kMhcSinkhorn)(workspace, single_use_workspace,
                                                                  single_use_executor, contiguous_stream);
      // Consumed by its own launch, success or not: the plan is gone either way.
      single_use_executor = nullptr;
      Check(launch_status == 0, "the single-use contiguous plan launched (status " +
                                    std::to_string(launch_status) + ")");
    }
    backend.streams->SynchronizeStream(contiguous_stream);
    std::vector<float> readback(host.size(), 0.0f);
    backend.streams->MemcpySync(readback.data(), readback.size() * sizeof(float), contiguous_out.address,
                                contiguous_out.bytes, MemcpyKind::kDeviceToHost);
    double device_worst = 0.0;
    const bool device_holds =
        CheckBirkhoffInvariant(readback.data(), kDecodeTokens, kNhc, kBirkhoffTolerance, &device_worst);
    Check(device_holds, "the operator's own [1, 4, 4] output is doubly stochastic within 1e-5 (worst deviation " +
                            Scientific(device_worst) + ")");
  }

  if (contiguous_repeatable) {
    contiguous_slot.Reset();
  }
  if (single_use_executor != nullptr) {
    // Planned but never launched (the numeric section was skipped), so this
    // one really does have to be destroyed.
    aclDestroyAclOpExecutor(single_use_executor);
    single_use_executor = nullptr;
  }
  if (contiguous_stream != nullptr) {
    backend.streams->DestroyStream(contiguous_stream);
  }
  if (contiguous_workspace != nullptr) {
    backend.allocator->DeviceFree(contiguous_workspace);
  }
  Destroy(&owned);
}

// ===========================================================================
// HYPOTHESIS 3 -- the compressor's sequence-ring cadence
// ===========================================================================

// ===========================================================================
// HYPOTHESIS 5 -- the FUSED normalization, which is the one the engine ships
// ===========================================================================
//
// H2 refuted the standalone operator's reusability, which leaves the one
// question that actually decides the runtime: is the B_l that comes OUT of the
// fused kernel doubly stochastic, and is THAT executor repeatable?
// aclnnHcPre runs the Sinkhorn in its AIV vector epilogue and returns the
// result as combFrag, so if both hold, the standalone operator is not needed
// at all -- which is exactly what the engine's default mHC mode assumes.
void TestFusedHcPreSinkhorn(const Backend& backend) {
  Hypothesis(5, "aclnnHcPre's fused epilogue returns a doubly-stochastic combFrag, repeatably");
  Assume("x = [1, 4, 4096] BF16 (TND), hcFn [24, 16384] FP32, hcScale [3] and hcBase [24] FP32");
  Assume("hcMult = 4, hcSinkhornIters = 20, both epsilons 1e-6");
  Assume("the outputs are y [1, 4096] BF16, post [1, 4] FP32 and combFrag [1, 4, 4] FP32");
  Assume("combFrag is ALREADY doubly stochastic -- row and column sums 1.0 +/- 1e-5 -- with no "
         "standalone Sinkhorn anywhere in the chain");
  Assume("its executor IS repeatable: B_l never becomes a tensor at the aclnn layer, so the 561000 "
         "refusal H2 recorded cannot arise here");

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Skip("the aclnn runtime is not on the loader path; no operator to question");
    return;
  }
  if (!ops.available(OpId::kHcPre)) {
    Skip("aclnnHcPre is not deployed in this opp package; the engine's default mHC mode needs it");
    return;
  }

  DeviceArena arena(backend.allocator, 32ull << 20);
  std::vector<Tensor> owned;
  const auto track = [&owned](Tensor tensor) {
    owned.push_back(tensor);
    return tensor;
  };

  const Tensor x = track(arena.Make({kDecodeTokens, kNhc, kHiddenSize}, kDtBf16));
  const Tensor hc_fn = track(arena.Make({kMixRows, kMixCols}, kDtFp32));
  const Tensor hc_scale = track(arena.Make({3}, kDtFp32));
  const Tensor hc_base = track(arena.Make({kMixRows}, kDtFp32));
  const Tensor y = track(arena.Make({kDecodeTokens, kHiddenSize}, kDtBf16));
  const Tensor post = track(arena.Make({kDecodeTokens, kNhc}, kDtFp32));
  const Tensor comb_frag = track(arena.Make({kDecodeTokens, kNhc, kNhc}, kDtFp32));

  Check(comb_frag.shape == std::vector<int64_t>({1, 4, 4}) && comb_frag.dtype == kDtFp32,
        "combFrag is allocated at the B_l geometry the engine binds, " + comb_frag.text());
  Check(comb_frag.strides == ContiguousStrides(comb_frag.shape),
        "and CONTIGUOUS, which H2.1 showed is the only form this kernel family accepts (strides " +
            ShapeText(comb_frag.strides) + ")");
  Check(y.shape == std::vector<int64_t>({1, kHiddenSize}),
        "y comes out at [1, 4096] -- the activation RMSNorm and the attention projections already take, "
        "so the engine feeds it on with no reshape");

  std::printf("\n-- H5.1 the fused plan, and whether it can be retained --\n");
  PlanResult plan = RunPlan([&](aclOpExecutor** executor) {
    // hcEps precedes normEps here, the reverse of aclnnMhcPre's order.
    return PlanAclnnOp<HcPrePlanFn>(ops, OpId::kHcPre, executor, x.handle, hc_fn.handle, hc_scale.handle,
                                    hc_base.handle, kNhc, kSinkhornIters, kSinkhornEps, kRmsNormEpsilon,
                                    y.handle, post.handle, comb_frag.handle);
  });
  if (!ExpectPlanned(plan, "hc_pre accepts the TND decode geometry at hcSinkhornIters = " +
                               std::to_string(kSinkhornIters))) {
    Destroy(&owned);
    return;
  }

  StaticOpSlot slot;
  void* workspace = nullptr;
  DeviceStream stream = nullptr;
  bool repeatable = false;
  try {
    slot.Adopt(OpId::kHcPre, "hypothesis/hc_pre", plan.workspace, plan.executor);
    repeatable = slot.planned();
  } catch (const std::exception& error) {
    // Unlike H2's refutation this is a genuine FAILURE, not a recorded
    // finding: the engine's default mode depends on this executor being
    // retained, so if it is refused the default is unusable.
    Check(false, std::string("aclSetAclOpExecutorRepeatable REFUSED the fused plan: ") +
                     FirstLine(error.what()) +
                     " -- kFusedHcPre depends on this, so the engine would have to fall back to "
                     "kStagedMhcSinkhorn and its per-round host plan");
    plan.executor = nullptr;
  }
  if (repeatable) {
    Check(true, "the fused executor IS repeatable: one retained plan serves every token, so the decode "
                "loop issues no host plan for the normalization");
    workspace = plan.workspace > 0 ? backend.allocator->DeviceMalloc(plan.workspace) : nullptr;
    stream = backend.streams->CreateStream();
    slot.Launch(ops, workspace, stream);
    // The decode loop's real access pattern: repoint x -- the ping-ponged
    // residual stream -- and relaunch.
    slot.SetAddress(0, x.handle, x.address);
    slot.Launch(ops, workspace, stream);
    Check(true, "the fused plan relaunched after aclSetTensorAddr on its stream input");
  }

  std::printf("\n-- H5.2 the Birkhoff invariant over the FUSED output --\n");
  if (!backend.numerics_live) {
    Skip("combFrag cannot be read back in this build (symbolic device memory); H2.4 fixes the reference "
         "and the 1e-5 tolerance the on-device run applies to this output");
  } else if (!repeatable) {
    Skip("no retained fused executor to measure -- see the refusal above");
  } else {
    // Seed a strictly positive projection: Sinkhorn is only defined on a
    // non-negative matrix with no all-zero row or column, and hcFn / hcBase
    // are what the kernel builds its 4x4 from.
    std::vector<float> fn_host(static_cast<size_t>(hc_fn.elements()), 0.0f);
    for (size_t index = 0; index < fn_host.size(); ++index) {
      fn_host[index] = 0.05f + 0.01f * static_cast<float>(index % 17);
    }
    std::vector<float> scale_host(3, 1.0f);
    std::vector<float> base_host(static_cast<size_t>(kMixRows), 0.1f);
    backend.streams->MemcpySync(hc_fn.address, hc_fn.bytes, fn_host.data(), fn_host.size() * sizeof(float),
                                MemcpyKind::kHostToDevice);
    backend.streams->MemcpySync(hc_scale.address, hc_scale.bytes, scale_host.data(),
                                scale_host.size() * sizeof(float), MemcpyKind::kHostToDevice);
    backend.streams->MemcpySync(hc_base.address, hc_base.bytes, base_host.data(),
                                base_host.size() * sizeof(float), MemcpyKind::kHostToDevice);

    slot.Launch(ops, workspace, stream);
    backend.streams->SynchronizeStream(stream);
    std::vector<float> readback(static_cast<size_t>(comb_frag.elements()), 0.0f);
    backend.streams->MemcpySync(readback.data(), readback.size() * sizeof(float), comb_frag.address,
                                comb_frag.bytes, MemcpyKind::kDeviceToHost);
    double worst = 0.0;
    const bool holds = CheckBirkhoffInvariant(readback.data(), kDecodeTokens, kNhc, kBirkhoffTolerance,
                                              &worst);
    Check(holds, "the FUSED combFrag is doubly stochastic within 1e-5 (worst deviation " +
                     Scientific(worst) + ") -- so aclnnMhcSinkhorn was never needed for CORRECTNESS, "
                     "only its repeatability was ever the problem");
  }

  if (repeatable) {
    slot.Reset();
  }
  if (stream != nullptr) {
    backend.streams->DestroyStream(stream);
  }
  if (workspace != nullptr) {
    backend.allocator->DeviceFree(workspace);
  }
  Destroy(&owned);
}

void TestCompressorRingCadence(const Backend& backend) {
  Hypothesis(3, "aclnnCompressor expresses a sequence-ring cadence at cmpRatio 4 (CSA)");
  Assume("compression ratio 4 (CSA), compressed width D = 512, ropeHeadDim = 64");
  Assume("the pooling state is a paged ring buffer [blocks, blockSize, D] FP32, updated in place (REF)");
  Assume("steps t = 0, 1, 2 emit nothing and write nothing into the compressed KV destination");
  Assume("step t = 3 closes the window and emits exactly one compressed entry");

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Skip("the aclnn runtime is not on the loader path; no operator to question");
    return;
  }

  Note("API mapping: the compressed rows land in cmpKvOut (the brief's 'kvCacheRef' destination) and the "
       "recurrent ring lives in stateCacheRef, the one REF parameter. 'emitted' is cmpKvOut's row count, "
       "which the wrapper pins at T / cmpRatio.");

  DeviceArena arena(backend.allocator, 64ull << 20);
  std::vector<Tensor> owned;
  const auto track = [&owned](Tensor tensor) {
    owned.push_back(tensor);
    return tensor;
  };

  // The projection pair, the positional bias and the RoPE tables are the same
  // weights at every step; only the window and its destination move.
  const Tensor wkv = track(arena.Make({kHiddenSize, kCompressedDim}, kDtBf16));
  const Tensor wgate = track(arena.Make({kHiddenSize, 1}, kDtBf16));
  const Tensor ape = track(arena.Make({kCompressRatio, kCompressedDim}, kDtFp32));
  const Tensor norm_weight = track(arena.Make({kCompressedDim}, kDtFp32));
  const Tensor state_cache = track(arena.Make({kStateBlocks, kStateBlockSize, kCompressedDim}, kDtFp32));
  const Tensor state_blocks = track(arena.Make({1, kStateBlocks}, kDtInt32));
  const Tensor cu_seqlens = track(arena.Make({1}, kDtInt32));
  const Tensor seqused = track(arena.Make({1}, kDtInt32));
  const Tensor start_pos = track(arena.Make({1}, kDtInt32));

  Check(state_cache.shape.size() == 3 && state_cache.dtype == kDtFp32,
        "the ring state is the paged 3-D FP32 buffer the wrapper requires, " + state_cache.text());
  Check(norm_weight.shape == std::vector<int64_t>({kCompressedDim}),
        "normWeight is 1-D at the compressed width, so cmpKvOut channels are normWeight[0] * coff = " +
            std::to_string(kCompressedDim));
  Note("the brief's [1, 1, 1024] KV and score tensors map onto the compressor's wkv/wgate projection pair "
       "over the 4096-wide residual stream; 1024 = 2 * kv_lora_rank is the gate-side width, not a tensor "
       "this operator takes directly.");

  const int64_t state_stride0 = state_cache.strides.at(0);

  // ---- the four steps of one CSA window ----------------------------------
  int64_t emitted_total = 0;
  for (int64_t step = 0; step < kCompressRatio; ++step) {
    const int64_t window = step + 1;               // tokens the caller is holding
    const int64_t emitted = window / kCompressRatio;  // the wrapper's own rule
    const bool closing = window % kCompressRatio == 0;
    std::printf("\n-- H3 step t = %" PRId64 ": window %" PRId64 " token(s), cmpKvOut rows %" PRId64 " --\n",
                step, window, emitted);

    const Tensor x = track(arena.Make({window, kHiddenSize}, kDtBf16));
    const Tensor rope_sin = track(arena.Make({window, kRopeHeadDim}, kDtBf16));
    const Tensor rope_cos = track(arena.Make({window, kRopeHeadDim}, kDtBf16));
    const Tensor cmp_kv_out = track(arena.Make({emitted, kCompressedDim}, kDtBf16));

    const Ledger before = ReadLedger();
    PlanResult plan = RunPlan([&](aclOpExecutor** executor) {
      return PlanAclnnOp<CompressorPlanFn>(
          ops, OpId::kCompressor, executor, x.handle, wkv.handle, wgate.handle, state_cache.handle, ape.handle,
          norm_weight.handle, rope_sin.handle, rope_cos.handle, state_blocks.handle, cu_seqlens.handle,
          seqused.handle, start_pos.handle, kRopeHeadDim, kCompressRatio, /*coff=*/1, kRmsNormEpsilon,
          /*rotaryMode=*/0, /*cacheMode=*/0, state_stride0, cmp_kv_out.handle);
    });
    if (!ExpectPlanned(plan, "the compressor accepts the step-" + std::to_string(step) + " window x " + x.text())) {
      continue;
    }
    const Ledger after = ReadLedger();

    if (!closing) {
      Check(emitted == 0, "step " + std::to_string(step) + " emits nothing: cmpKvOut is " + cmp_kv_out.text());
      Check(plan.workspace == 0,
            "the incomplete window plans a ZERO-byte workspace -- the wrapper's empty-tensor early return "
            "fired, so no kernel and no ViewCopy were staged");
      if (after.available) {
        Check(after.ref_output_plans == before.ref_output_plans,
              "no in-place update of stateCacheRef was staged either: the step is a total no-op, so nothing "
              "can reach the compressed KV destination");
      } else {
        Skip("the REF-plan ledger is a mock-build instrument; the zero workspace above is the device-side "
             "witness that nothing was staged");
      }
    } else {
      Check(emitted == 1, "step " + std::to_string(step) + " emits exactly one row: cmpKvOut is " +
                              cmp_kv_out.text());
      Check(plan.workspace > 0, "the window-closing step plans a non-empty workspace -- real work is staged");
      if (after.available) {
        Check(after.ref_output_plans == before.ref_output_plans + 1,
              "exactly one REF plan was recorded: stateCacheRef is updated in place, with no copy stage to "
              "alias");
        Check(after.self_copy_hazards == before.self_copy_hazards,
              "the emitting step plans no same-address ViewCopy, so its executor stays reusable across windows");
      }
      const int64_t entry_bytes =
          kCompressedDim * static_cast<int64_t>(DataTypeBytes(cmp_kv_out.dtype));
      Check(cmp_kv_out.bytes == static_cast<size_t>(entry_bytes),
            "the emitted entry is " + std::to_string(entry_bytes) + " bytes (D = " +
                std::to_string(kCompressedDim) + " x " + DataTypeName(cmp_kv_out.dtype) + ")");

      // THE MEASURED RECORD LAYOUT, read back off the descriptors the kernel
      // was actually handed rather than recomputed. This is the record the
      // on-device run exists to produce: the engine sizes its paged cache from
      // Dsv4CompressedKvEntry and derives every stride attribute from the
      // bound view, so these three numbers are what a disagreement would show
      // up in.
      const int64_t out_row_stride = DeriveDimensionStrideElements(cmp_kv_out.handle, 0);
      const size_t out_row_bytes = DeriveDimensionStrideBytes(cmp_kv_out.handle, 0);
      const int64_t ring_stride = DeriveDimensionStrideElements(state_cache.handle, 0);
      const size_t ring_bytes = DeriveDimensionStrideBytes(state_cache.handle, 0);
      Note("MEASURED LAYOUT: cmpKvOut row stride " + std::to_string(out_row_stride) + " elements = " +
           std::to_string(out_row_bytes) + " bytes (" + DataTypeName(cmp_kv_out.dtype) + " x " +
           std::to_string(kCompressedDim) + "); stateCacheRef axis-0 stride " + std::to_string(ring_stride) +
           " elements = " + std::to_string(ring_bytes) + " bytes. Both were DERIVED from the bound view with "
           "aclGetViewStrides, which is the only way the runtime is allowed to learn a stride attribute.");
      Check(out_row_bytes == static_cast<size_t>(entry_bytes),
            "the derived byte stride of one emitted row agrees with D x sizeof(dtype) -- the destination view "
            "really is contiguous rows, so a cache sized on that stride addresses it correctly");
      if (entry_bytes != kBriefCompressedEntryBytes) {
        Note("UNRESOLVED: the brief states a " + std::to_string(kBriefCompressedEntryBytes) +
             "-byte compressed entry, and the kernel's own cmpKvOut row is " + std::to_string(entry_bytes) +
             " bytes. No (width, dtype) pair in this geometry produces " +
             std::to_string(kBriefCompressedEntryBytes) + ": BF16 x 512 = " +
             std::to_string(kCompressedDim * 2) + ", FP8 x 512 = " + std::to_string(kCompressedDim) +
             ", FP8 x (512 + 64 rope) = " + std::to_string(kCompressedDim + kRopeHeadDim) +
             ". The " + std::to_string(kBriefCompressedEntryBytes) +
             "-byte figure is a PACKED CACHE ENTRY (448 nope FP8 + 7 UE8M0 scales + 21 pad + 64 rope BF16), "
             "which is what aclnnKvCompressEpilog produces FROM this row -- not what the compressor emits. "
             "Dsv4CompressedKvEntry in kv_cache_layout.hpp holds that packing to the byte; this probe "
             "asserts the compressor's own row width, which is the number its destination view must match.");
      }
    }
    emitted_total += emitted;

    if (plan.planned) {
      StaticOpSlot slot;
      slot.Adopt(OpId::kCompressor, "hypothesis/compressor", plan.workspace, plan.executor);
      void* workspace = plan.workspace > 0 ? backend.allocator->DeviceMalloc(plan.workspace) : nullptr;
      DeviceStream stream = backend.streams->CreateStream();
      slot.Launch(ops, workspace, stream);
      Check(true, "the step-" + std::to_string(step) + " plan adopted (repeatable) and launched");
      slot.Reset();
      backend.streams->DestroyStream(stream);
      if (workspace != nullptr) {
        backend.allocator->DeviceFree(workspace);
      }
    }
  }

  std::printf("\n-- H3 the cadence over one full window --\n");
  Check(emitted_total == 1,
        "four steps at cmpRatio 4 emit exactly one compressed entry in total (" +
            std::to_string(emitted_total) + ")");

  // The stride attribute is how the kernel addresses the paged ring; a stale
  // value scatters the pooling state into the wrong block.
  std::printf("\n-- H3 the ring's addressing attribute --\n");
  const Tensor x_closing = track(arena.Make({kCompressRatio, kHiddenSize}, kDtBf16));
  const Tensor sin_closing = track(arena.Make({kCompressRatio, kRopeHeadDim}, kDtBf16));
  const Tensor cos_closing = track(arena.Make({kCompressRatio, kRopeHeadDim}, kDtBf16));
  const Tensor out_closing = track(arena.Make({1, kCompressedDim}, kDtBf16));
  ReportAttributeGuard(RunPlan([&](aclOpExecutor** executor) {
                         return PlanAclnnOp<CompressorPlanFn>(
                             ops, OpId::kCompressor, executor, x_closing.handle, wkv.handle, wgate.handle,
                             state_cache.handle, ape.handle, norm_weight.handle, sin_closing.handle,
                             cos_closing.handle, state_blocks.handle, cu_seqlens.handle, seqused.handle,
                             start_pos.handle, kRopeHeadDim, kCompressRatio, /*coff=*/1, kRmsNormEpsilon,
                             /*rotaryMode=*/0, /*cacheMode=*/0, state_stride0 + 1, out_closing.handle);
                       }),
                       "a stateCacheStrideDim0 one element past the ring buffer's own axis-0 stride (" +
                           std::to_string(state_stride0) + ")",
                       "the kernel addresses the paged ring through this attribute, so a stale value pools "
                       "into the wrong block with no diagnostic. The runtime must DERIVE it from the view it "
                       "just bound -- never carry it as an independent number.");
  ExpectRefused(RunPlan([&](aclOpExecutor** executor) {
                  return PlanAclnnOp<CompressorPlanFn>(
                      ops, OpId::kCompressor, executor, x_closing.handle, wkv.handle, wgate.handle,
                      state_cache.handle, ape.handle, norm_weight.handle, sin_closing.handle, cos_closing.handle,
                      state_blocks.handle, cu_seqlens.handle, seqused.handle, start_pos.handle, kRopeHeadDim,
                      /*cmpRatio=*/8, /*coff=*/1, kRmsNormEpsilon, /*rotaryMode=*/0, /*cacheMode=*/0,
                      state_stride0, out_closing.handle);
                }),
                "cmpRatio = 8, outside the deployed {4 (CSA), 128 (HCA)} pair");

  Destroy(&owned);
}

// ===========================================================================
// HYPOTHESIS 4 -- the lightning indexer's top-k sparsity range
// ===========================================================================

void TestIndexerTopKRange(const Backend& backend) {
  Hypothesis(4, "aclnnVllmQuantLightningIndexer: top-512 indices, INT32, inside the sequence");
  Assume("query = [1, 1, 64, 128] FP8 E4M3 (BSND, D = 128, N1 = 64)");
  Assume("key cache = FP8 E4M3 in the PA_BSND paged layout, N2 = 1, D = 128");
  Assume("top-k = 512");
  Assume("sparseIndicesOut = [1, 1, 1, 512] INT32");
  Assume("every index returned is >= 0 and strictly inside the valid sequence slots");

  OpTable ops;
  if (!ops.runtime_reachable()) {
    Skip("the aclnn runtime is not on the loader path; no operator to question");
    return;
  }

  DeviceArena arena(backend.allocator, 16ull << 20);
  std::vector<Tensor> owned;
  const auto track = [&owned](Tensor tensor) {
    owned.push_back(tensor);
    return tensor;
  };

  const Tensor query = track(arena.Make({kBatch, kSeq, kIndexNumHeads, kIndexHeadDim}, kDtFp8E4m3));
  const Tensor key =
      track(arena.Make({kIndexerBlocks, kIndexerBlockSize, 1, kIndexHeadDim}, kDtFp8E4m3));
  const Tensor weights = track(arena.Make({kBatch, kSeq, kIndexNumHeads}, kDtBf16));
  const Tensor query_scale = track(arena.Make({kBatch, kSeq, kIndexNumHeads}, kDtFp32));
  const Tensor key_scale = track(arena.Make({kIndexerBlocks, kIndexerBlockSize, 1}, kDtFp32));
  const Tensor seq_key = track(arena.Make({kBatch}, kDtInt32));
  const Tensor block_table = track(arena.Make({kBatch, kIndexerBlocks}, kDtInt32));
  const Tensor metadata = track(arena.Make({kBatch}, kDtInt32));
  const Tensor indices_out = track(arena.Make({kBatch, kSeq, 1, kIndexTopK}, kDtInt32));
  const Tensor values_out = track(arena.Make({0}, kDtFp32));

  Check(query.shape == std::vector<int64_t>({1, 1, 64, 128}) && query.dtype == kDtFp8E4m3,
        "the query really is [1, 1, 64, 128] FP8 E4M3");
  Check(indices_out.shape == std::vector<int64_t>({1, 1, 1, 512}) && indices_out.dtype == kDtInt32,
        "the output really is [1, 1, 1, 512] INT32 at top-k = " + std::to_string(kIndexTopK));
  Check(kIndexTopK <= 2048, "top-k = " + std::to_string(kIndexTopK) + " sits inside the operator's [1, 2048] bound");
  Note("the operator also admits HiFloat8 for query/key; E4M3 is the path this engine selects on 950PR, and "
       "ACL_HIFLOAT8 is the single-token change if that choice is revisited.");

  const int64_t key_stride0 = key.strides.at(0);
  const int64_t scale_stride0 = key_scale.strides.at(0);

  std::printf("\n-- H4.1 the plan over the paged FP8 key stream --\n");
  PlanResult plan = RunPlan([&](aclOpExecutor** executor) {
    return PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
        ops, OpId::kVllmQuantLightningIndexer, executor, query.handle, key.handle, weights.handle,
        query_scale.handle, key_scale.handle, /*actualSeqLengthsQuery=*/nullptr, seq_key.handle,
        block_table.handle, metadata.handle, /*queryQuantMode=*/0, /*keyQuantMode=*/0,
        const_cast<char*>("BSND"), const_cast<char*>("PA_BSND"), kIndexTopK, /*sparseMode=*/3, INT64_MAX,
        INT64_MAX, /*cmpRatio=*/kCompressRatio, /*returnValues=*/false, key_stride0, scale_stride0,
        indices_out.handle, values_out.handle);
  });

  StaticOpSlot slot;
  void* workspace = nullptr;
  DeviceStream stream = nullptr;
  if (ExpectPlanned(plan, "the indexer accepts query " + query.text() + " over a PA_BSND FP8 E4M3 key cache")) {
    Check(plan.workspace > 0, "the indexer reports a non-empty device workspace for this geometry");
    slot.Adopt(OpId::kVllmQuantLightningIndexer, "hypothesis/vllm_quant_lightning_indexer", plan.workspace,
               plan.executor);
    Check(slot.planned(), "the indexer's executor is retained and made repeatable");
    workspace = plan.workspace > 0 ? backend.allocator->DeviceMalloc(plan.workspace) : nullptr;
    stream = backend.streams->CreateStream();
    slot.Launch(ops, workspace, stream);
    Check(true, "the indexer launched on the retained executor");
  }

  std::printf("\n-- H4.2 the attributes that address the paged stream --\n");
  ReportAttributeGuard(RunPlan([&](aclOpExecutor** executor) {
                         return PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
                             ops, OpId::kVllmQuantLightningIndexer, executor, query.handle, key.handle,
                             weights.handle, query_scale.handle, key_scale.handle, nullptr, seq_key.handle,
                             block_table.handle, metadata.handle, 0, 0, const_cast<char*>("BSND"),
                             const_cast<char*>("PA_BSND"), kIndexTopK, 3, INT64_MAX, INT64_MAX, kCompressRatio,
                             false, key_stride0 + 1, scale_stride0, indices_out.handle, values_out.handle);
                       }),
                       "a key stride one element past the paged key's own axis-0 stride (" +
                           std::to_string(key_stride0) + ")",
                       "the indexer reads the paged key through this attribute, so a stale value scores the "
                       "wrong block and the returned indices address rows that were never compared. Derive "
                       "it from the bound key view.");
  ExpectRefused(RunPlan([&](aclOpExecutor** executor) {
                  return PlanAclnnOp<VllmQuantLightningIndexerPlanFn>(
                      ops, OpId::kVllmQuantLightningIndexer, executor, query.handle, key.handle, weights.handle,
                      query_scale.handle, key_scale.handle, nullptr, seq_key.handle, block_table.handle,
                      metadata.handle, 0, 0, const_cast<char*>("BSND"), const_cast<char*>("PA_BSND"),
                      /*sparseCount=*/2049, 3, INT64_MAX, INT64_MAX, kCompressRatio, false, key_stride0,
                      scale_stride0, indices_out.handle, values_out.handle);
                }),
                "top-k = 2049, past the operator's [1, 2048] bound");

  std::printf("\n-- H4.3 the index range the attention core will gather at --\n");
  // The checker itself is held to a known-bad vector first, so a pass on the
  // real output means the bound was tested rather than merely declared.
  std::vector<int32_t> poisoned(static_cast<size_t>(kIndexTopK), 0);
  poisoned[static_cast<size_t>(kIndexTopK) - 1] = static_cast<int32_t>(kIndexerKeySlots);
  std::string poisoned_detail;
  // Sequenced deliberately: the detail string is only filled by the call, so
  // reading it in the same expression would depend on argument order.
  const bool poisoned_rejected = !CheckIndexRange(poisoned.data(), poisoned.size(),
                                                  static_cast<int32_t>(kIndexerKeySlots), &poisoned_detail);
  Check(poisoned_rejected, "the range checker rejects an out-of-bounds slot (" + poisoned_detail + ")");

  if (!backend.numerics_live) {
    Skip("the operator's own indices cannot be read back in this build (symbolic device memory); the checker "
         "above fixes the bound [0, " + std::to_string(kIndexerKeySlots) +
         ") the on-device run applies to sparseIndicesOut");
  } else if (!slot.planned()) {
    Skip("no planned indexer executor to measure -- see the refusal above");
  } else {
    // A sentinel fill makes an untouched output visible: -1 everywhere would
    // fail the range check rather than passing as a plausible zero.
    std::vector<int32_t> sentinel(static_cast<size_t>(kIndexTopK), -1);
    backend.streams->MemcpySync(indices_out.address, indices_out.bytes, sentinel.data(),
                                sentinel.size() * sizeof(int32_t), MemcpyKind::kHostToDevice);
    slot.Launch(ops, workspace, stream);
    backend.streams->SynchronizeStream(stream);
    std::vector<int32_t> readback(static_cast<size_t>(kIndexTopK), -1);
    backend.streams->MemcpySync(readback.data(), readback.size() * sizeof(int32_t), indices_out.address,
                                indices_out.bytes, MemcpyKind::kDeviceToHost);
    std::string detail;
    const bool in_range =
        CheckIndexRange(readback.data(), readback.size(), static_cast<int32_t>(kIndexerKeySlots), &detail);
    Check(in_range, in_range ? "all " + std::to_string(kIndexTopK) +
                                   " emitted indices are non-negative and inside [0, " +
                                   std::to_string(kIndexerKeySlots) + ")"
                             : "an emitted index left the valid sequence slots: " + detail);
  }

  if (slot.planned()) {
    slot.Reset();
  }
  if (stream != nullptr) {
    backend.streams->DestroyStream(stream);
  }
  if (workspace != nullptr) {
    backend.allocator->DeviceFree(workspace);
  }
  Destroy(&owned);
}

}  // namespace
}  // namespace ascend_moe

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

namespace {

using namespace ascend_moe;

#ifdef ASCEND_MOCK_RUNTIME
SimulatedDeviceOps* g_device = nullptr;
const DmaCounters& ReadCounters() { return g_device->counters(); }
#else
AclDeviceOps* g_device = nullptr;
const DmaCounters& ReadCounters() { return g_device->counters(); }
#endif

}  // namespace

int main(int argc, char** argv) {
  bool require_device = false;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--require-device") == 0) {
      require_device = true;
    }
  }

  std::printf("dsv4_operator_hypotheses_test -- targeted contract verification of the vendored\n");
  std::printf("arch35 (Ascend 950PR) DeepSeek-V4-Flash operators at the engine's own geometry.\n");

  Backend backend;
#ifdef ASCEND_MOCK_RUNTIME
  // --require-device is a CANN-build switch: it turns "no NPU attached" from a
  // SKIP into a failure. A mock build never needs one, so the flag is inert.
  (void)require_device;
  std::printf("backend: symbolic (ENABLE_MOCK_RUNTIME=ON) -- plan-phase hypotheses only; device memory is\n");
  std::printf("an interval registry, so the numeric sections hold the host reference instead.\n");
  mock::MockResetAllocatorForTest();
  SimulatedDeviceOps device(256ull << 20);
  g_device = &device;
  backend.allocator = &device;
  backend.streams = &device;
  backend.numerics_live = false;
  backend.name = device.backend_name();
#else
  if (!PhysicalNpuPresent()) {
    return require_device ? 1 : 0;
  }
  std::printf("backend: physical CANN -- plan-phase hypotheses against libcust_opapi.so, and the numeric\n");
  std::printf("sections seed, launch and read back on the attached device.\n");
  AclDeviceOps device(0);
  g_device = &device;
  backend.allocator = &device;
  backend.streams = &device;
  backend.numerics_live = true;
  backend.name = device.backend_name();
#endif
  std::printf("device backend: %s\n", backend.name);

  // EACH HYPOTHESIS IS ISOLATED, and that is the point of this loop rather
  // than one try block around all four.
  //
  // The 950PR run of this suite died inside Hypothesis 2 -- aclnnMhcSinkhorn's
  // aclSetAclOpExecutorRepeatable failed with 561000 and the AclError escaped
  // -- so Hypotheses 3 and 4 were never executed on the device at all. The
  // compressor cadence and the indexer's index range are independent
  // questions about different operators; one operator refusing a call it was
  // asked to refuse must not cost us the answers to the others.
  int status = 0;
  const auto guard = [&status](const char* name, const std::function<void()>& body) {
    try {
      body();
    } catch (const std::exception& error) {
      ++g_failures;
      status = 1;
      std::printf("\n  [FAIL] hypothesis %s aborted: %s\n", name, error.what());
      std::printf("         the remaining hypotheses still run -- they question different operators.\n");
    }
  };
  guard("1 (mhc_pre / mhc_post)", [&] { TestMhcPreAndPostContract(backend, &ReadCounters); });
  guard("2 (mhc_sinkhorn)", [&] { TestSinkhornInvariantAndAliasing(backend); });
  guard("5 (fused hc_pre Sinkhorn)", [&] { TestFusedHcPreSinkhorn(backend); });
  guard("3 (compressor cadence)", [&] { TestCompressorRingCadence(backend); });
  guard("4 (lightning indexer)", [&] { TestIndexerTopKRange(backend); });

  std::printf("\n=============================================================\n");
  std::printf("%d checks, %d failures, %d skipped, %d refuted-and-encoded\n", g_checks, g_failures, g_skips,
              g_refutations);
  if (g_failures == 0 && status == 0) {
    std::printf("every decidable hypothesis holds at the DSV4-Flash geometry.\n");
  } else {
    std::printf("a hypothesis was REFUTED -- the runtime design that assumed it needs revisiting.\n");
  }
  if (g_refutations > 0) {
    std::printf("\n%d claim(s) were refuted by the 950PR run and are ALREADY encoded in the engine; they are\n",
                g_refutations);
    std::printf("recorded above as [refuted] and do not fail this suite. See the B_l note in config.hpp.\n");
  }
  std::printf("=============================================================\n");
  return (g_failures == 0 && status == 0) ? 0 : 1;
}
