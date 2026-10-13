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

// dsv4_semi_e2e_bench_npu -- the device twin of apps/dsv4_semi_e2e_bench.cpp.
//
// One token through layer 0, layer 2 and layer 1's MoE on real
// DeepSeek-V4-Flash weights, executing entirely on device 0. There is no host
// fallback: every arithmetic stage is either one of the five arch35 operators
// or a stock CANN 9.2 aclnn operator. The host reads the checkpoint, stages it
// into HBM, plans the aclnn graph once, and then does nothing but enqueue.
//
//   layer 0   fused_mhc_pre_norm_sinkhorn
//             -> shared-KV SWA attention (aclnn)
//             -> fused_mhc_post_residual
//   layer 2   fused_mhc_pre_norm_sinkhorn
//             -> the same attention core
//             + paged_compressor_rope over the 4:1 cadence (HOLD x3, EMIT)
//             + fast_fp8_indexer_topk, top-512 over the key cache
//             -> fused_mhc_post_residual
//   layer 1   fused_mhc_pre_norm_sinkhorn
//             -> MoE: aclnn MoeGatingTopKV2 hash route, dsv4_moe_expert per
//                routed expert (gate/up GEMV + clamped SwiGLU + down GEMV in
//                one wrapper call), QuantMatmulV5 shared expert
//             -> fused_mhc_post_residual
//
// ---------------------------------------------------------------------------
// VERIFICATION STATUS -- READ BEFORE QUOTING ANY NUMBER THIS PRINTS
// ---------------------------------------------------------------------------
// NEVER EXECUTED. The development workstation has no Ascend NPU (no
// /dev/davinci*, no driver), so what has been verified is that this COMPILES
// and LINKS against CANN 9.2.0-beta.2 and the BiSheng-built arch35 launch
// layer. Per CLAUDE.md section 5.7 nothing here may be described as hardware
// behaviour until it has run on hardware.
//
// Decisions that only a device can settle are marked UNVERIFIED at their site.
//
// The FP8 projection path is bound to a binary that was confirmed present in
// this toolkit rather than assumed:
//   opp/.../kernel/ascend950/ops_nn/quant_batch_matmul_v3/
//     QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance.{o,json}
// whose supportInfo declares x1/x2 float8_e4m3fn, scale and pertoken_scale
// float32, y bfloat16, all ND and dynamic-rank. The checkpoint's 128x128 E8M0
// scale grid is rewritten to FP32 at staging to reach it -- see Fp8Matrix for
// why the mxfp8/E8M0 binaries are the wrong granularity despite matching on
// dtype -- and `groupSize` carries the 128x128 block geometry.
//
// Note this bench and src/pipeline/pipeline.cpp now differ deliberately:
// RuntimeConfig::dense_group_size still defaults to 0 ("ask the operator")
// because that path feeds the MLA decode, while this one passes the packed
// 128|128<<16 value. Neither has run on hardware; --dense-group-size exists
// so the two can be A/B'd against each other once one can.
//
// ---------------------------------------------------------------------------
// WHERE THIS DIVERGES FROM THE HOST BENCH, BY CONSTRUCTION
// ---------------------------------------------------------------------------
// The host emulation dequantizes each FP8 block matrix and does the GEMV in
// FP32 (semi-E2E investigation, section 2.2). A device FP8 GEMV cannot:
// aclnnQuantMatmulV5 consumes a QUANTIZED activation, so every projection
// here quantizes its input first. That is closer to the upstream model's own
// quantized-activation path than the host reference is, but it means a
// systematic delta against golden_semi_e2e_ref.bin that is not a bug. The
// parity gate below is therefore cosine-only, with the bound justified by the
// format at its definition.
//
// The four BF16 quantizer sites now use the arch35 bf16_block_quant_fp8
// operator instead of aclnnDynamicMxQuant + aclnnCast: one dispatch emitting
// E4M3 codes and FP32 block scales directly. That ALSO moves the parity
// baseline, in a second way and deliberately -- DynamicMxQuant emitted OCP
// E8M0 power-of-two scales, and these are arbitrary FP32, which uses the
// E4M3 range strictly better. Quantization error goes down; the numbers
// differ from any figure taken before 2026-10-13.
//
// The FIFTH site (shared expert, "quantize hidden") still runs
// aclnnDynamicMxQuant, because its input t_hidden is FP32 and the arch35
// operator takes BF16. See the note at that site.
//
// ---------------------------------------------------------------------------
// ZERO-ALLOCATION: WHAT IS AND IS NOT TRUE
// ---------------------------------------------------------------------------
// This bench performs zero aclrtMalloc in its timed region: every activation,
// weight, workspace and cache buffer is a window into one up-front arena, and
// every aclnn stage holds a repeatable executor planned before the clock
// starts.
//
// The five arch35 wrappers do NOT honour that. Each allocates and frees its
// own tiling staging buffer per call and synchronizes the stream before the
// free (launch.hpp, "STREAM AND TILING LIFETIME"). Removing that needs a
// caller-owned-tiling launch variant, which CLAUDE.md section 3.7 explicitly
// gates. Two consequences, both reported in [5] rather than hidden:
//   * the malloc count in the timed region is one per arch35 launch, not zero,
//     and the bench counts them instead of claiming otherwise;
//   * every arch35 per-stage figure is SYNC-INCLUSIVE -- a host-observed round
//     trip, not a kernel time. aclnn stages only enqueue, so they are timed as
//     groups bracketed by one explicit sync.
//
// USAGE
//   scripts/download_dsv4_semi_e2e_weights.sh
//   ./build/apps/dsv4_semi_e2e_bench_npu [--data-dir DIR] [--token-id N]
//       [--decode-steps N] [--indexer-seq-len N] [--device N]
//       [--dense-group-size N] [--memory-budget-mb N] [--time-budget-ms N]
//       [--no-golden]

#include "moe/core/acl_guard.hpp"
#include "moe/core/config.hpp"
#include "moe/core/error.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/weight_source.hpp"
#include "moe/kernels/arch35/launch.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ascend_moe::OpId;
using ascend_moe::OpTable;
using ascend_moe::PipelineStage;
using ascend_moe::PlanAclnnOp;

// The audited operator-argument constants. Reusing the engine's own selectors
// rather than restating them keeps this bench and src/pipeline/pipeline.cpp
// from drifting apart on, say, which scale algorithm OCP means.
using ascend_moe::kAclBf16;
using ascend_moe::kAclFloat32;
using ascend_moe::kAclFloat8E4m3Fn;
using ascend_moe::kAclFloat8E8m0;
using ascend_moe::kAclInt32;
using ascend_moe::kCubeMathTypeKeepDtype;
using ascend_moe::kFiaAntiquantModeNone;
using ascend_moe::kFiaInnerPreciseHighPrecision;
using ascend_moe::kFiaNextTokensCausal;
using ascend_moe::kFiaPreTokensAll;
using ascend_moe::kFiaPseTypeNone;
using ascend_moe::kFiaQueryQuantModeNone;
using ascend_moe::kFiaSparseModeBand;
using ascend_moe::kGatingEps;
using ascend_moe::kGatingGroupCount;
using ascend_moe::kGatingGroupSelectMode;
using ascend_moe::kGatingKGroup;
using ascend_moe::kGatingNormTypePreNormalized;
using ascend_moe::kGatingRenormL1;
using ascend_moe::kRotaryLayoutBsnd;
using ascend_moe::kRoutedScaleBlock;
using ascend_moe::kSoftplusBeta;
using ascend_moe::kSoftplusThreshold;
using ascend_moe::kSwigluAxisLast;
using ascend_moe::kSwigluScaleAlgOcp;

namespace arch35 = moe::kernels::arch35;

using Clock = std::chrono::steady_clock;

double MicrosecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// aclnn takes these as `char*`, so they cannot be string literals.
char kRoundModeRint[] = "rint";
char kRotaryModeHalf[] = "half";
char kFiaLayout[] = "TND";

// ---------------------------------------------------------------------------
// Geometry. Mirrors include/moe/emu/dsv4_layers.h, restated because that
// header belongs to the host emulation tree while this binary is built in the
// engine configuration where moe_emu does not exist. Every value is checked
// against config.json at startup, so a divergence is an error, never a silent
// re-plan at the wrong stride.
// ---------------------------------------------------------------------------
constexpr int kHiddenSize = 4096;
constexpr int kHcStreams = 4;
constexpr int kHcSinkhornIters = 20;
constexpr int kHcPhiRows = 24;
constexpr int kHcPhiCols = kHcStreams * kHiddenSize;  // 16384
constexpr int kHcAlphaElements = 3;
constexpr int kHcProjectionRows = 24;

constexpr int kAttnHeads = 64;
constexpr int kAttnHeadDim = 512;
constexpr int kQLoraRank = 1024;
constexpr int kOLoraRank = 1024;
constexpr int kOGroups = 8;
constexpr int kAttnRopeHeadDim = 64;
constexpr int kSlidingWindow = 128;
constexpr int kWoAGroupInput = kAttnHeads * kAttnHeadDim / kOGroups;  // 4096
constexpr int kWoAOutput = kOGroups * kOLoraRank;                     // 8192
constexpr int kAttnQkvWidth = kAttnHeads * kAttnHeadDim;              // 32768

constexpr int kNumRoutedExperts = 256;
constexpr int kNumExpertsPerTok = 6;
constexpr int kMoeIntermediate = 2048;
constexpr float kRoutedScalingFactor = 1.5F;
constexpr float kSwigluClampLimit = 10.0F;

constexpr int kFp8BlockSize = 128;
constexpr int kMxfp4GroupSize = 32;

constexpr int kCompressorRatio = 4;
constexpr int kCompressorCoff = 2;
constexpr int kCompressorHeadDim = 512;
constexpr int kCompressorStateWidth = kCompressorCoff * kCompressorHeadDim;  // 1024
constexpr int kIndexerHeads = 64;
constexpr int kIndexerHeadDim = 128;
constexpr int kIndexerTopK = 512;
constexpr int kIndexerBlockSize = 128;

constexpr double kRmsEpsilon = 1e-6;
constexpr double kRopeThetaAttention = 10000.0;
constexpr double kRopeThetaCompressor = 160000.0;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct Options {
  std::string data_dir = "data/dsv4_layer0";
  std::int64_t token_id = 3;
  int decode_steps = 4;
  std::int32_t indexer_seq_len = 2048;
  std::int32_t device_id = 0;
  int layer0 = 0;
  int layer2 = 2;
  int moe_layer = 1;
  int warmup_window_tokens = 8;
  // aclnnQuantMatmulV5's `groupSize` packs the three per-axis block sizes:
  //   groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
  // documented in aclnnop/aclnn_quant_grouped_matmul_inplace_add.h:37 (the
  // quant_matmul_v5 header itself only says "int64"). The checkpoint's dense
  // FP8 scale grid is 128x128, so K = N = 128 and M is left at 0 for this
  // one-token GEMV: 128 | (128 << 16) == 8388736.
  //
  // UNVERIFIED: the formula is documented for the grouped operator and the
  // dense one takes the same scalar, but no device has confirmed that the
  // dense path reads it identically, nor what groupSizeM means at M = 1.
  // --dense-group-size overrides it for a hardware A/B; 0 asks the operator
  // for its own default, which is what RuntimeConfig::dense_group_size still
  // uses in src/pipeline.
  std::int64_t dense_group_size = (128 | (static_cast<std::int64_t>(128) << 16));
  double memory_budget_mb = 4096.0;
  double time_budget_ms = 1500.0;
  bool golden = true;
};

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("option ") + flag + " needs a value");
      return argv[++i];
    };
    if (argument == "--data-dir") {
      options.data_dir = next("--data-dir");
    } else if (argument == "--token-id") {
      options.token_id = std::stoll(next("--token-id"));
    } else if (argument == "--decode-steps") {
      options.decode_steps = std::stoi(next("--decode-steps"));
    } else if (argument == "--indexer-seq-len") {
      options.indexer_seq_len = static_cast<std::int32_t>(std::stol(next("--indexer-seq-len")));
    } else if (argument == "--device") {
      options.device_id = static_cast<std::int32_t>(std::stol(next("--device")));
    } else if (argument == "--dense-group-size") {
      options.dense_group_size = std::stoll(next("--dense-group-size"));
    } else if (argument == "--memory-budget-mb") {
      options.memory_budget_mb = std::stod(next("--memory-budget-mb"));
    } else if (argument == "--time-budget-ms") {
      options.time_budget_ms = std::stod(next("--time-budget-ms"));
    } else if (argument == "--no-golden") {
      options.golden = false;
    } else if (argument == "-h" || argument == "--help") {
      std::printf(
          "usage: dsv4_semi_e2e_bench_npu [--data-dir DIR] [--token-id N]\n"
          "       [--decode-steps N] [--indexer-seq-len N] [--device N]\n"
          "       [--dense-group-size N] [--memory-budget-mb N]\n"
          "       [--time-budget-ms N] [--no-golden]\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.decode_steps < 1 || options.indexer_seq_len < 1) {
    throw std::runtime_error("--decode-steps and --indexer-seq-len must be >= 1");
  }
  return options;
}

bool FileExists(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  return probe.good();
}

std::string ReadWholeFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------
// BF16, round-to-nearest-even, matching moe::emu::FloatToBf16 so the staged
// inputs and the golden comparison agree with the host bench on the
// conversion itself.
// ---------------------------------------------------------------------------
std::uint16_t FloatToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if (((bits >> 23) & 0xFFU) == 0xFFU) return static_cast<std::uint16_t>(bits >> 16);
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float Bf16ToFloat(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// E4M3 encode, matching moe::emu::EncodeFp8E4M3. Used only to synthesize the
// indexer's Q/K activation bytes, which the host bench also synthesizes.
std::uint8_t EncodeFp8E4M3(float value) {
  if (!std::isfinite(value)) return 0x7FU;
  const std::uint8_t sign = value < 0.0F ? 0x80U : 0x00U;
  float magnitude = std::fabs(value);
  if (magnitude > 448.0F) magnitude = 448.0F;
  if (magnitude < 1.0F / 512.0F) return sign;
  int exponent = 0;
  const float fraction = std::frexp(magnitude, &exponent);  // [0.5, 1)
  int unbiased = exponent - 1;
  float mantissa = fraction * 2.0F - 1.0F;  // [0, 1)
  if (unbiased < -6) {  // subnormal
    const int quantized = static_cast<int>(std::lround(magnitude * 512.0F));
    return static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(std::min(quantized, 7)));
  }
  int mantissa_bits = static_cast<int>(std::lround(mantissa * 8.0F));
  if (mantissa_bits == 8) {
    mantissa_bits = 0;
    ++unbiased;
  }
  const int biased = unbiased + 7;
  if (biased > 15) return static_cast<std::uint8_t>(sign | 0x7EU);
  return static_cast<std::uint8_t>(sign | (biased << 3) | mantissa_bits);
}

// ---------------------------------------------------------------------------
// A one-shot device arena.
//
// Every device buffer is a window into a single aclrtMalloc, so the timed
// region issues none of its own. 512-byte alignment covers the 32 B the
// arch35 tiling ABI needs and the base alignment an aclnn ND tensor wants,
// and keeps stages off each other's cache lines.
// ---------------------------------------------------------------------------
class DeviceArena {
 public:
  explicit DeviceArena(std::size_t capacity) : capacity_(capacity) {
    DSV4_ACL_CHECK(aclrtMalloc(&base_, capacity_, ACL_MEM_MALLOC_HUGE_FIRST));
    DSV4_ACL_CHECK(aclrtMemset(base_, capacity_, 0, capacity_));
  }
  ~DeviceArena() {
    if (base_ != nullptr) ascend_moe::ReportAclCleanup("aclrtFree(arena)", aclrtFree(base_));
  }
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  void* Take(std::size_t bytes, const char* label) {
    const std::size_t aligned = (bytes + 511U) & ~static_cast<std::size_t>(511U);
    if (used_ + aligned > capacity_) {
      throw ascend_moe::Dsv4Error(std::string("device arena exhausted allocating '") + label + "': needs " +
                                  std::to_string(aligned) + " B, " + std::to_string(capacity_ - used_) +
                                  " B left of " + std::to_string(capacity_) +
                                  " B. Raise --memory-budget-mb.");
    }
    void* pointer = static_cast<std::uint8_t*>(base_) + used_;
    used_ += aligned;
    return pointer;
  }

  void* Stage(const void* host, std::size_t bytes, const char* label) {
    void* device = Take(bytes, label);
    DSV4_ACL_CHECK(aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    return device;
  }

  std::size_t used() const { return used_; }
  std::size_t capacity() const { return capacity_; }

 private:
  void* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
};

// ---------------------------------------------------------------------------
// Descriptor ownership. Descriptors must outlive every launch referencing
// them, so they are all parented on one bag destroyed after the final drain.
// ---------------------------------------------------------------------------
class DescriptorBag {
 public:
  ~DescriptorBag() {
    for (aclTensor* t : tensors_) ascend_moe::ReportAclCleanup("aclDestroyTensor", aclDestroyTensor(t));
    for (aclTensorList* l : lists_) ascend_moe::ReportAclCleanup("aclDestroyTensorList", aclDestroyTensorList(l));
    for (aclScalar* s : scalars_) ascend_moe::ReportAclCleanup("aclDestroyScalar", aclDestroyScalar(s));
    for (aclIntArray* a : arrays_) ascend_moe::ReportAclCleanup("aclDestroyIntArray", aclDestroyIntArray(a));
  }

  // A contiguous ND view. `storage_elements` lets a view address a window of a
  // larger allocation (the SwiGLU halves, the o-projection groups) without a
  // second allocation.
  aclTensor* Tensor(std::vector<std::int64_t> dims, int32_t dtype, void* address, const char* label,
                    std::int64_t storage_elements = 0) {
    std::vector<std::int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) strides[i - 1] = strides[i] * dims[i];
    std::int64_t elements = 1;
    for (std::int64_t dim : dims) elements *= dim;
    std::int64_t storage = storage_elements > 0 ? storage_elements : elements;
    aclTensor* tensor =
        aclCreateTensor(dims.data(), static_cast<std::uint64_t>(dims.size()), static_cast<aclDataType>(dtype),
                        strides.data(), 0, static_cast<aclFormat>(ascend_moe::kAclFormatNd), &storage, 1, address);
    DSV4_REQUIRE(tensor != nullptr, label << ": aclCreateTensor returned null");
    tensors_.push_back(tensor);
    return tensor;
  }

  // A transposed 2-D view: same bytes, swapped extents and strides. The router
  // GEMM needs gate.weight as [K, N] while the checkpoint stores [N, K].
  aclTensor* Transposed2d(std::int64_t rows, std::int64_t cols, int32_t dtype, void* address, const char* label) {
    std::int64_t dims[2] = {cols, rows};
    std::int64_t strides[2] = {1, cols};
    std::int64_t storage = rows * cols;
    aclTensor* tensor = aclCreateTensor(dims, 2, static_cast<aclDataType>(dtype), strides, 0,
                                        static_cast<aclFormat>(ascend_moe::kAclFormatNd), &storage, 1, address);
    DSV4_REQUIRE(tensor != nullptr, label << ": aclCreateTensor (transposed) returned null");
    tensors_.push_back(tensor);
    return tensor;
  }

  aclTensorList* List(const std::vector<aclTensor*>& members, const char* label) {
    aclTensorList* list = aclCreateTensorList(members.data(), static_cast<std::uint64_t>(members.size()));
    DSV4_REQUIRE(list != nullptr, label << ": aclCreateTensorList returned null");
    lists_.push_back(list);
    return list;
  }

  aclScalar* Scalar(float value, const char* label) {
    aclScalar* scalar = aclCreateScalar(&value, static_cast<aclDataType>(kAclFloat32));
    DSV4_REQUIRE(scalar != nullptr, label << ": aclCreateScalar returned null");
    scalars_.push_back(scalar);
    return scalar;
  }

  aclIntArray* IntArray(const std::vector<std::int64_t>& values, const char* label) {
    aclIntArray* array = aclCreateIntArray(values.data(), values.size());
    DSV4_REQUIRE(array != nullptr, label << ": aclCreateIntArray returned null");
    arrays_.push_back(array);
    return array;
  }

 private:
  std::vector<aclTensor*> tensors_;
  std::vector<aclTensorList*> lists_;
  std::vector<aclScalar*> scalars_;
  std::vector<aclIntArray*> arrays_;
};

// ---------------------------------------------------------------------------
// Checkpoint staging
//
// Every tensor is checked against the compiled geometry BEFORE the H2D copy.
// A shape error caught here is a message; the same error reaching the device
// is a wrong answer, because the arch35 kernels reject bad tiling by
// returning early into a zeroed output (CLAUDE.md section 2.5).
// ---------------------------------------------------------------------------
class Checkpoint {
 public:
  Checkpoint(ascend_moe::SafetensorsWeightSource& source, DeviceArena& arena) : source_(source), arena_(arena) {}

  void* Stage(const std::string& name, const char* dtype, const std::vector<std::int64_t>& shape) {
    const auto cached = staged_.find(name);
    if (cached != staged_.end()) return cached->second;
    RequireDtype(name, dtype);
    RequireShape(name, shape);
    const std::size_t bytes = source_.NamedByteSize(name);
    std::vector<std::uint8_t> host(bytes);
    source_.ReadNamed(name, host.data(), host.size(), 0, bytes);
    void* device = arena_.Stage(host.data(), bytes, name.c_str());
    staged_.emplace(name, device);
    bytes_ += bytes;
    return device;
  }

  // phi and ape are F32 in a repacked file whose data segment is not 4-byte
  // aligned, so they are copied out rather than aliased: reading them through
  // a const float* cast would be undefined behaviour.
  float* StageAsFloat(const std::string& name, const std::vector<std::int64_t>& shape) {
    const auto cached = staged_float_.find(name);
    if (cached != staged_float_.end()) return cached->second;
    RequireShape(name, shape);
    const std::string dtype = source_.NamedDtype(name);
    std::size_t elements = 1;
    for (std::int64_t dim : shape) elements *= static_cast<std::size_t>(dim);
    const std::size_t bytes = source_.NamedByteSize(name);
    std::vector<std::uint8_t> host(bytes);
    source_.ReadNamed(name, host.data(), host.size(), 0, bytes);
    std::vector<float> values(elements);
    if (dtype == "F32") {
      DSV4_REQUIRE(bytes == elements * 4, name << ": F32 payload is " << bytes << " B, expected " << elements * 4);
      std::memcpy(values.data(), host.data(), bytes);
    } else if (dtype == "BF16") {
      DSV4_REQUIRE(bytes == elements * 2, name << ": BF16 payload is " << bytes << " B, expected " << elements * 2);
      for (std::size_t i = 0; i < elements; ++i) {
        std::uint16_t raw = 0;
        std::memcpy(&raw, host.data() + i * 2, 2);
        values[i] = Bf16ToFloat(raw);
      }
    } else {
      throw ascend_moe::Dsv4Error(name + " is " + dtype + ", expected F32 or BF16");
    }
    auto* device = static_cast<float*>(arena_.Stage(values.data(), values.size() * sizeof(float), name.c_str()));
    staged_float_.emplace(name, device);
    bytes_ += bytes;
    return device;
  }

  // Read a tensor to the host without staging it. Used for tid2eid, whose
  // route row the host must know to decide which expert weights to stage.
  std::vector<std::uint8_t> ReadHost(const std::string& name) {
    DSV4_REQUIRE(source_.HasNamed(name), "the weight file has no tensor '" << name << "'");
    const std::size_t bytes = source_.NamedByteSize(name);
    std::vector<std::uint8_t> host(bytes);
    source_.ReadNamed(name, host.data(), host.size(), 0, bytes);
    return host;
  }

  // Assert a tensor's declared dtype and shape without staging it. Used by
  // callers that read the payload themselves to reformat it.
  void Require(const std::string& name, const char* dtype, const std::vector<std::int64_t>& shape) {
    RequireDtype(name, dtype);
    RequireShape(name, shape);
  }

  bool Has(const std::string& name) const { return source_.HasNamed(name); }
  std::string Dtype(const std::string& name) const { return source_.NamedDtype(name); }
  std::vector<std::int64_t> Shape(const std::string& name) const { return source_.NamedShape(name); }
  std::size_t bytes() const { return bytes_; }
  std::size_t count() const { return staged_.size() + staged_float_.size(); }

 private:
  void RequireDtype(const std::string& name, const char* expect) {
    const std::string dtype = source_.NamedDtype(name);
    DSV4_REQUIRE(dtype == expect, name << " is " << dtype << ", expected " << expect);
  }

  void RequireShape(const std::string& name, const std::vector<std::int64_t>& expect) {
    DSV4_REQUIRE(source_.HasNamed(name), "the weight file has no tensor '" << name << "'");
    const std::vector<std::int64_t> shape = source_.NamedShape(name);
    bool same = shape.size() == expect.size();
    for (std::size_t i = 0; same && i < shape.size(); ++i) same = shape[i] == expect[i];
    if (!same) {
      std::string got;
      for (std::int64_t d : shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
      std::string want;
      for (std::int64_t d : expect) want += (want.empty() ? "" : ", ") + std::to_string(d);
      throw ascend_moe::Dsv4Error("tensor '" + name + "' is [" + got + "] but this build expects [" + want + "]");
    }
  }

  ascend_moe::SafetensorsWeightSource& source_;
  DeviceArena& arena_;
  std::map<std::string, void*> staged_;
  std::map<std::string, float*> staged_float_;
  std::size_t bytes_ = 0;
};

// E8M0 is a bare power-of-two exponent with bias 127; 0xFF is the NaN code.
// The conversion to FP32 is therefore EXACT -- every representable E8M0 value
// is a normal float -- which is what makes the staging-time rewrite below a
// reformat rather than a precision loss.
float DecodeE8M0(std::uint8_t code) {
  if (code == 0xFFU) return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

// An FP8 block-scaled matrix staged on device.
//
// WHY THE SCALE GRID IS REWRITTEN TO FP32
// The checkpoint stores a 128x128 E8M0 block grid. On ascend950,
// QuantBatchMatmulV3's compiled binaries split into two families and neither
// takes that combination directly:
//
//   ND_ND_fp8e4m3_fp8e4m3_fp32_bf16      scale fp32,        pertoken_scale fp32
//   ND_ND_mxfp8e4m3_mxfp8e4m3_e8m0_bf16  scale float8_e8m0, pertoken_scale e8m0
//
// Every E8M0-scaled binary in that directory is an `mx*` one, i.e. OCP
// microscaling at block-32 along K -- NOT a 128x128 dense grid. So the MX
// family matches our dtype and not our granularity. The FP32 family matches
// our granularity (expressed through `groupSize`) and not our scale dtype.
//
// Converting E8M0 -> FP32 once at staging costs one float per 128x128 block
// (a 4096x4096 matrix needs 32x32 floats = 4 KB) and binds us to the family
// whose granularity is right. Verified present:
//   opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/
//     quant_batch_matmul_v3/
//     QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance.{o,json}
struct Fp8Matrix {
  void* weight = nullptr;
  float* scale = nullptr;  // FP32, rewritten from the checkpoint's E8M0 grid
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t scale_rows = 0;
  std::int64_t scale_cols = 0;
};

Fp8Matrix StageFp8(Checkpoint& checkpoint, DeviceArena& arena, const std::string& stem, std::int64_t rows,
                   std::int64_t cols) {
  Fp8Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.scale_rows = (rows + kFp8BlockSize - 1) / kFp8BlockSize;
  matrix.scale_cols = (cols + kFp8BlockSize - 1) / kFp8BlockSize;
  matrix.weight = checkpoint.Stage(stem + ".weight", "F8_E4M3", {rows, cols});

  const std::string scale_name = stem + ".scale";
  checkpoint.Require(scale_name, "F8_E8M0", {matrix.scale_rows, matrix.scale_cols});
  const std::vector<std::uint8_t> codes = checkpoint.ReadHost(scale_name);
  const std::size_t elements = static_cast<std::size_t>(matrix.scale_rows) * matrix.scale_cols;
  DSV4_REQUIRE(codes.size() == elements,
               scale_name << ": E8M0 payload is " << codes.size() << " B, expected " << elements);
  std::vector<float> decoded(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    decoded[i] = DecodeE8M0(codes[i]);
    DSV4_REQUIRE(std::isfinite(decoded[i]),
                 scale_name << "[" << i << "] is the E8M0 NaN code (0xFF); the checkpoint grid is corrupt");
  }
  matrix.scale = static_cast<float*>(arena.Stage(decoded.data(), decoded.size() * sizeof(float), scale_name.c_str()));
  return matrix;
}

// A routed-expert MXFP4 matrix. `cols` is LOGICAL; the stored payload is half
// as wide because two E2M1 nibbles share a byte, and the scale grid is one
// column per 32 logical elements.
struct Mxfp4Matrix {
  void* packed = nullptr;
  void* scale = nullptr;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
};

Mxfp4Matrix StageMxfp4(Checkpoint& checkpoint, const std::string& stem, std::int64_t rows, std::int64_t cols) {
  Mxfp4Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  // The checkpoint spells the packed payload I8 -- a byte container -- not
  // F4_E2M1. config.json's "expert_dtype": "fp4" plus the scale grid width is
  // what establishes that these bytes are nibble pairs.
  const std::string dtype = checkpoint.Dtype(stem + ".weight");
  DSV4_REQUIRE(dtype == "I8" || dtype == "U8" || dtype == "F4_E2M1",
               stem << ".weight is " << dtype << ", expected the packed-FP4 byte container (I8/U8/F4_E2M1)");
  matrix.packed = checkpoint.Stage(stem + ".weight", dtype.c_str(), {rows, cols / 2});
  matrix.scale = checkpoint.Stage(stem + ".scale", "F8_E8M0", {rows, cols / kMxfp4GroupSize});
  return matrix;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
struct StageTiming {
  std::string name;
  double microseconds = 0.0;
  int launches = 0;
  bool sync_inclusive = false;
};

class Timeline {
 public:
  void Add(const std::string& name, double microseconds, int launches, bool sync_inclusive) {
    for (StageTiming& stage : stages_) {
      if (stage.name == name) {
        stage.microseconds += microseconds;
        stage.launches += launches;
        return;
      }
    }
    stages_.push_back(StageTiming{name, microseconds, launches, sync_inclusive});
  }
  const std::vector<StageTiming>& stages() const { return stages_; }
  double total() const {
    double sum = 0.0;
    for (const StageTiming& stage : stages_) sum += stage.microseconds;
    return sum;
  }

 private:
  std::vector<StageTiming> stages_;
};

int g_arch35_launches = 0;

// Past the device guard there is a device, so a non-success Status is a hard
// error carrying its code, the recorded LastLaunchFailure() and
// aclGetRecentErrMsg() -- never a skip (CLAUDE.md section 4.4).
void CheckArch35(arch35::Status status, const char* what) {
  if (status == arch35::Status::success) return;
  const arch35::LaunchFailure& failure = arch35::LastLaunchFailure();
  std::string message = std::string(what) + " failed: " + arch35::ToString(status);
  if (failure.op != nullptr) {
    message += std::string(" (") + failure.op + " / " + (failure.call != nullptr ? failure.call : "?") +
               ", aclError " + std::to_string(failure.acl_error) + ")";
  }
  const char* recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') message += std::string("; aclGetRecentErrMsg: ") + recent;
  throw ascend_moe::Dsv4Error(message);
}

template <typename Fn>
void TimedArch35(Timeline& timeline, const char* name, Fn&& launch) {
  const auto start = Clock::now();
  CheckArch35(launch(), name);
  timeline.Add(name, MicrosecondsSince(start), 1, true);
  ++g_arch35_launches;
}

// ---------------------------------------------------------------------------
// The aclnn stage graph.
//
// Stages are planned once before the timed region and hold repeatable
// executors, so a launch is an enqueue against the shared workspace and
// nothing more.
//
// The stage vector is BORN at full size rather than reserved, the same way
// MoeRouterEngine and StaticOpSlotTable do it: StaticOpSlot is neither
// copyable nor movable, so `reserve` does not even compile and a
// reallocation would destroy live executors.
// ---------------------------------------------------------------------------
constexpr std::size_t kMaxAclnnStages = 128;

class AclnnGraph {
 public:
  explicit AclnnGraph(const OpTable& ops) : ops_(ops) {}

  template <typename PlanFn, typename... Args>
  PipelineStage& Plan(const char* name, OpId op, Args... args) {
    DSV4_REQUIRE(used_ < stages_.size(),
                 "AclnnGraph holds at most " << stages_.size() << " stages; raise kMaxAclnnStages");
    PipelineStage& stage = stages_[used_++];
    stage.name = name;
    stage.op = op;
    aclOpExecutor* executor = nullptr;
    const std::uint64_t workspace = PlanAclnnOp<PlanFn>(ops_, op, &executor, args...);
    workspace_high_water_ = std::max(workspace_high_water_, workspace);
    stage.slot.Adopt(op, name, workspace, executor);
    return stage;
  }

  void Launch(PipelineStage& stage, aclrtStream stream) const { stage.slot.Launch(ops_, workspace_, stream); }

  std::uint64_t workspace_high_water() const { return workspace_high_water_; }
  void set_workspace(void* workspace) { workspace_ = workspace; }
  std::size_t size() const { return used_; }

  // Repeatable executors must be destroyed after the last launch has drained
  // and before the ACL runtime goes away.
  void Reset() {
    for (std::size_t i = 0; i < used_; ++i) stages_[i].slot.Reset();
  }

 private:
  const OpTable& ops_;
  std::vector<PipelineStage> stages_ = std::vector<PipelineStage>(kMaxAclnnStages);
  std::size_t used_ = 0;
  std::uint64_t workspace_high_water_ = 0;
  void* workspace_ = nullptr;
};

// ---------------------------------------------------------------------------
// Golden parity -- golden_semi_e2e_ref.bin v3, written by the host bench.
// ---------------------------------------------------------------------------
struct Golden {
  bool present = false;
  std::int64_t token_id = 0;
  std::vector<std::uint16_t> x_next;
  std::vector<std::uint16_t> x_after0;
  std::vector<std::uint16_t> x_after2;
};

Golden ReadGolden(const std::string& path) {
  Golden golden;
  if (!FileExists(path)) return golden;
  const std::string blob = ReadWholeFile(path);

  struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t dtype_code;
    std::int32_t streams;
    std::int32_t hidden;
    std::int64_t token_id;
    std::int32_t routed_count;
    std::int32_t sinkhorn_n;
    std::int32_t compressed_tokens;
    std::int32_t indexer_top_k;
  };
  if (blob.size() < sizeof(Header)) throw ascend_moe::Dsv4Error(path + " is shorter than its header");
  Header header{};
  std::memcpy(&header, blob.data(), sizeof(header));
  if (std::memcmp(header.magic, "DSV4SE2E", 8) != 0) throw ascend_moe::Dsv4Error(path + " has the wrong magic");
  if (header.version != 3) {
    throw ascend_moe::Dsv4Error(path + " is version " + std::to_string(header.version) +
                                ", this bench reads version 3. Regenerate with dsv4_semi_e2e_bench.");
  }
  DSV4_REQUIRE(header.streams == kHcStreams && header.hidden == kHiddenSize,
               path << " carries a [" << header.streams << ", " << header.hidden << "] bundle, expected ["
                    << kHcStreams << ", " << kHiddenSize << "]");

  const std::size_t bundle = static_cast<std::size_t>(kHcStreams) * kHiddenSize;
  const std::size_t bundle_bytes = bundle * sizeof(std::uint16_t);
  std::size_t cursor = sizeof(Header);
  const auto take = [&](std::vector<std::uint16_t>* out) {
    if (cursor + bundle_bytes > blob.size()) throw ascend_moe::Dsv4Error(path + " is truncated");
    out->resize(bundle);
    std::memcpy(out->data(), blob.data() + cursor, bundle_bytes);
    cursor += bundle_bytes;
  };

  take(&golden.x_next);
  cursor += 2 * static_cast<std::size_t>(kHcStreams) * kHcStreams * sizeof(float);   // B0, B2
  cursor += static_cast<std::size_t>(kNumExpertsPerTok) * sizeof(std::int32_t);      // expert ids
  cursor += static_cast<std::size_t>(kNumExpertsPerTok) * sizeof(float);             // routing weights
  cursor += bundle_bytes;                                                            // x_input
  take(&golden.x_after0);
  take(&golden.x_after2);

  golden.token_id = header.token_id;
  golden.present = true;
  return golden;
}

double CosineSimilarity(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
  DSV4_REQUIRE(a.size() == b.size() && !a.empty(), "cosine over mismatched or empty vectors");
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = Bf16ToFloat(a[i]);
    const double y = Bf16ToFloat(b[i]);
    dot += x * y;
    norm_a += x * x;
    norm_b += y * y;
  }
  // A degeneracy guard that tests the REFERENCE, not a value against itself
  // (CLAUDE.md section 4.8): a zeroed golden or a zeroed device output would
  // otherwise divide by zero rather than fail honestly.
  DSV4_REQUIRE(norm_a > 0.0 && norm_b > 0.0,
               "cosine against an all-zero vector: the comparison would be degenerate");
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

double L2Norm(const std::vector<std::uint16_t>& values) {
  double total = 0.0;
  for (std::uint16_t raw : values) {
    const double value = Bf16ToFloat(raw);
    total += value * value;
  }
  return std::sqrt(total);
}

// Flat "key": <integer> lookup. config.json is machine-generated and flat at
// the level this needs; this is not a JSON parser and refuses rather than
// guessing at anything else.
std::int64_t ConfigInt(const std::string& text, const char* key) {
  const std::string needle = std::string("\"") + key + "\"";
  const std::size_t at = text.find(needle);
  if (at == std::string::npos) throw ascend_moe::Dsv4Error(std::string("config.json has no \"") + key + "\"");
  std::size_t cursor = text.find(':', at + needle.size());
  if (cursor == std::string::npos) throw ascend_moe::Dsv4Error(std::string("config.json: \"") + key + "\" has no value");
  ++cursor;
  while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])) != 0) ++cursor;
  const std::size_t begin = cursor;
  if (cursor < text.size() && text[cursor] == '-') ++cursor;
  while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) ++cursor;
  if (cursor == begin) throw ascend_moe::Dsv4Error(std::string("config.json: \"") + key + "\" is not an integer");
  return std::stoll(text.substr(begin, cursor - begin));
}

// Interleaved-half RoPE tables, matching moe::emu::BuildRopeTables.
void BuildRopeTables(int position, double theta, int rotary_dim, std::vector<std::uint16_t>* cos_table,
                     std::vector<std::uint16_t>* sin_table) {
  const int half = rotary_dim / 2;
  cos_table->resize(static_cast<std::size_t>(rotary_dim));
  sin_table->resize(static_cast<std::size_t>(rotary_dim));
  for (int i = 0; i < half; ++i) {
    const double inverse = 1.0 / std::pow(theta, static_cast<double>(2 * i) / rotary_dim);
    const double angle = static_cast<double>(position) * inverse;
    const std::uint16_t c = FloatToBf16(static_cast<float>(std::cos(angle)));
    const std::uint16_t s = FloatToBf16(static_cast<float>(std::sin(angle)));
    (*cos_table)[static_cast<std::size_t>(i)] = c;
    (*cos_table)[static_cast<std::size_t>(i + half)] = c;
    (*sin_table)[static_cast<std::size_t>(i)] = s;
    (*sin_table)[static_cast<std::size_t>(i + half)] = s;
  }
}

}  // namespace

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
  return ascend_moe::GuardedMain([&]() -> int {
    const Options options = ParseOptions(argc, argv);

    std::printf(
        "=================================================================\n"
        " DeepSeek-V4-Flash semi-end-to-end decode -- Ascend 950PR, device %d\n"
        " layer 0 (SWA) -> layer 2 (compressed + indexer) -> layer 1 (MoE)\n"
        " five arch35 operators + stock CANN 9.2 aclnn, no host fallback\n"
        "=================================================================\n",
        options.device_id);

    // -----------------------------------------------------------------------
    // Device guard (CLAUDE.md section 4.3). Probe first; on zero devices print
    // the phrase the ctest registration matches and exit 0. Past this point a
    // device exists, so every failure below is a hard failure.
    //
    // This guard can only run where the DRIVER is installed: libascendcl pulls
    // in libascend_hal.so, so on a host with neither driver nor device the
    // binary fails to load before main() and exits 127. That is correct and
    // must not be "fixed" by putting the link-time driver stubs on the runtime
    // search path -- that would let a fake driver satisfy the probe.
    // -----------------------------------------------------------------------
    std::uint32_t device_count = 0;
    const aclError probe = aclrtGetDeviceCount(&device_count);
    if (probe != ACL_SUCCESS || device_count == 0) {
      std::printf(
          "\nPhysical Ascend NPU not detected (aclrtGetDeviceCount -> %d, %u devices).\n"
          "This bench executes on hardware only; the host twin is\n"
          "  ./build_emu/apps/dsv4_semi_e2e_bench\n",
          static_cast<int>(probe), device_count);
      return 0;
    }

    ascend_moe::EnsureAclRuntime();
    DSV4_ACL_CHECK(aclrtSetDevice(options.device_id));
    aclrtStream raw_stream = nullptr;
    DSV4_ACL_CHECK(aclrtCreateStream(&raw_stream));
    ascend_moe::AclStreamGuard stream_guard(raw_stream);
    const aclrtStream stream = stream_guard.get();

    const std::string weights_path = options.data_dir + "/dsv4_semi_e2e.safetensors";
    const std::string config_path = options.data_dir + "/config.json";
    const std::string golden_path = options.data_dir + "/golden_semi_e2e_ref.bin";
    for (const std::string& path : {config_path, weights_path}) {
      if (!FileExists(path)) {
        std::fprintf(stderr,
                     "error: %s not found.\n"
                     "       Run scripts/download_dsv4_semi_e2e_weights.sh first.\n",
                     path.c_str());
        return 2;
      }
    }

    // =======================================================================
    // [0] weights
    // =======================================================================
    const std::string config_text = ReadWholeFile(config_path);
    const std::size_t arena_bytes = static_cast<std::size_t>(options.memory_budget_mb) * 1024U * 1024U;
    DeviceArena arena(arena_bytes);

    ascend_moe::SafetensorsWeightSource source(weights_path, ascend_moe::ExpertSlotLayout::ForDeepSeekV4Flash(),
                                               ascend_moe::CheckpointNaming::kDsv4Flat,
                                               /*num_layers=*/64, /*num_experts=*/kNumRoutedExperts);
    Checkpoint checkpoint(source, arena);

    std::printf("\n[0] weights: %s\n", weights_path.c_str());
    std::printf("    %zu tensors in the file; HBM arena %.1f MB reserved on device %d\n", source.tensor_count(),
                static_cast<double>(arena_bytes) / 1048576.0, options.device_id);

    // =======================================================================
    // [1] geometry
    // =======================================================================
    struct Check {
      const char* field;
      std::int64_t expected;
    };
    const Check checks[] = {
        {"hidden_size", kHiddenSize},
        {"hc_mult", kHcStreams},
        {"hc_sinkhorn_iters", kHcSinkhornIters},
        {"num_attention_heads", kAttnHeads},
        {"head_dim", kAttnHeadDim},
        {"q_lora_rank", kQLoraRank},
        {"o_lora_rank", kOLoraRank},
        {"o_groups", kOGroups},
        {"qk_rope_head_dim", kAttnRopeHeadDim},
        {"sliding_window", kSlidingWindow},
        {"n_routed_experts", kNumRoutedExperts},
        {"num_experts_per_tok", kNumExpertsPerTok},
        {"moe_intermediate_size", kMoeIntermediate},
        {"index_topk", kIndexerTopK},
    };
    for (const Check& check : checks) {
      const std::int64_t actual = ConfigInt(config_text, check.field);
      DSV4_REQUIRE(actual == check.expected, "config.json \"" << check.field << "\" is " << actual
                                                              << " but this build is compiled for "
                                                              << check.expected);
    }
    std::printf("\n[1] geometry checked against the compiled contract: %zu fields match config.json\n",
                sizeof(checks) / sizeof(checks[0]));

    // Operator inventory up front, so a toolkit gap is one message and not a
    // failure part-way through a half-staged pipeline.
    OpTable ops;
    DSV4_REQUIRE(ops.runtime_reachable(),
                 "the aclnn runtime is not on this process's loader path; the operator entry points "
                 "are resolved out of the already-linked CANN libraries");
    // kDynamicMxQuant is still required by exactly ONE site: the shared
    // expert's "quantize hidden", whose input is FP32 rather than BF16 (see
    // that site). The four BF16 attention quantizers no longer use it. It
    // cannot be dropped from this list until that site is handled, and on
    // ascend950 it is the operator that fails with EZ1009 -- so this list is
    // currently what will stop a run at the shared expert.
    ops.RequireAll({OpId::kDynamicMxQuant, OpId::kQuantMatmulV5, OpId::kRmsNorm, OpId::kApplyRotaryPosEmbV2,
                    OpId::kFusedInferAttentionScoreV5, OpId::kMoeGatingTopKV2, OpId::kSwiGlu, OpId::kClamp,
                    OpId::kCast, OpId::kMatmul, OpId::kMul, OpId::kInplaceAdd, OpId::kSoftplus, OpId::kSqrt});

    AclnnGraph graph(ops);
    DescriptorBag bag;
    Timeline timeline;

    const std::size_t bundle = static_cast<std::size_t>(kHcStreams) * kHiddenSize;
    const auto layer_name = [](int layer, const std::string& suffix) {
      return "layers." + std::to_string(layer) + "." + suffix;
    };

    // =======================================================================
    // Activation buffers. Allocated once; nothing below allocates again.
    // =======================================================================
    auto* x_ping = static_cast<arch35::Bf16*>(arena.Take(bundle * 2, "x_ping"));
    auto* x_pong = static_cast<arch35::Bf16*>(arena.Take(bundle * 2, "x_pong"));
    auto* h_raw = static_cast<float*>(arena.Take(kHiddenSize * 4, "h_raw"));
    auto* h_norm = static_cast<arch35::Bf16*>(arena.Take(kHiddenSize * 2, "h_norm"));
    auto* h_norm_f32 = static_cast<float*>(arena.Take(kHiddenSize * 4, "h_norm_f32"));
    auto* post_gates = static_cast<float*>(arena.Take(kHcStreams * 4, "post_gates"));
    auto* birkhoff = static_cast<float*>(arena.Take(kHcStreams * kHcStreams * 4, "birkhoff"));
    auto* h_out = static_cast<arch35::Bf16*>(arena.Take(kHiddenSize * 2, "h_out"));

    aclTensor* t_h_norm = bag.Tensor({1, kHiddenSize}, kAclBf16, h_norm, "h_norm");
    aclTensor* t_h_norm_f32 = bag.Tensor({1, kHiddenSize}, kAclFloat32, h_norm_f32, "h_norm_f32");
    aclTensor* t_h_out = bag.Tensor({1, kHiddenSize}, kAclBf16, h_out, "h_out");

    // h_norm arrives BF16 from the mHC prologue; mxfp4_expert_gemv and the
    // router both want FP32, so one Cast feeds both.
    PipelineStage& cast_h_norm =
        graph.Plan<ascend_moe::CastPlanFn>("h_norm -> f32", OpId::kCast, t_h_norm,
                                           static_cast<aclDataType>(kAclFloat32), t_h_norm_f32);

    // =======================================================================
    // The synthesized input bundle, generated exactly as the host bench does
    // so both runs start from the same x_input: mt19937(0x9E3779B9 ^ token),
    // N(0, initializer_range = 0.02), one hidden-width row broadcast into all
    // four streams after the BF16 conversion.
    // =======================================================================
    {
      std::vector<std::uint16_t> host_bundle(bundle);
      std::mt19937 engine(0x9E3779B9U ^ static_cast<std::uint32_t>(options.token_id));
      std::normal_distribution<float> distribution(0.0F, 0.02F);
      std::vector<std::uint16_t> row(kHiddenSize);
      for (int i = 0; i < kHiddenSize; ++i) row[static_cast<std::size_t>(i)] = FloatToBf16(distribution(engine));
      for (int s = 0; s < kHcStreams; ++s) {
        std::copy(row.begin(), row.end(), host_bundle.begin() + static_cast<std::ptrdiff_t>(s) * kHiddenSize);
      }
      DSV4_ACL_CHECK(aclrtMemcpy(x_ping, bundle * 2, host_bundle.data(), bundle * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // =======================================================================
    // The mHC prologue / epilogue pair.
    //
    // The prologue kernel already fuses the activation RMSNorm with the
    // stage's norm weight and emits h_norm in BF16, so there is deliberately
    // NO separate aclnnRmsNorm between it and the sub-block: adding one would
    // normalize twice. Note where that norm sits -- on h_in, the [1, D] mHC
    // output, not on the [1, n, D] bundle.
    // =======================================================================
    struct MhcWeights {
      const float* phi = nullptr;
      const float* alpha = nullptr;
      const float* bias = nullptr;
      const arch35::Bf16* gamma = nullptr;
    };
    const auto load_mhc = [&](int layer, const char* stem, const char* norm) {
      const std::string prefix = layer_name(layer, std::string("hc_") + stem);
      MhcWeights weights;
      weights.phi = checkpoint.StageAsFloat(prefix + "_fn", {kHcPhiRows, kHcPhiCols});
      weights.alpha = checkpoint.StageAsFloat(prefix + "_scale", {kHcAlphaElements});
      weights.bias = checkpoint.StageAsFloat(prefix + "_base", {kHcProjectionRows});
      weights.gamma = static_cast<const arch35::Bf16*>(
          checkpoint.Stage(layer_name(layer, std::string(norm) + ".weight"), "BF16", {kHiddenSize}));
      return weights;
    };

    const auto mhc_prologue = [&](const MhcWeights& w, const arch35::Bf16* x) {
      TimedArch35(timeline, "fused_mhc_pre_norm_sinkhorn", [&] {
        return arch35::LaunchMhcPreNormSinkhorn(x, kHiddenSize, w.phi, kHcPhiCols, w.alpha, w.bias, w.gamma, h_raw,
                                                h_norm, post_gates, birkhoff, nullptr,
                                                arch35::kMhcPreNormWorkspaceBytes, stream);
      });
    };
    const auto mhc_epilogue = [&](const arch35::Bf16* x, arch35::Bf16* y) {
      TimedArch35(timeline, "fused_mhc_post_residual", [&] {
        return arch35::LaunchMhcPostResidual(x, kHiddenSize, birkhoff, kHcStreams, post_gates, h_out, y,
                                             kHiddenSize, nullptr, arch35::kMhcPostWorkspaceBytes, stream);
      });
    };

    // Replaces the aclnnDynamicMxQuant + aclnnCast pair that every BF16
    // activation quantizer site used to run. DynamicMxQuant emitted E8M0
    // scales while the ascend950 QuantBatchMatmulV3 binary takes
    // pertoken_scale in FP32, so each site cost two launches and a scale
    // round trip -- and on hardware the pair failed with EZ1009. This is one
    // dispatch producing E4M3 codes and FP32 block scales directly, so the
    // a_*_scale_f32 windows are now written at source rather than cast into.
    //
    // Sync-inclusive like every arch35 row (the wrapper synchronizes before
    // releasing its tiling staging buffer, CLAUDE.md §3.7), where the two
    // aclnn stages it replaces were enqueue-only and timed in a group. The
    // launch COUNT goes down by two per site; the per-site wall time is not
    // comparable to the old grouped figure.
    const auto block_quant = [&](const char* name, const void* src_bf16,
                                 std::uint32_t width, void* dst_fp8,
                                 void* dst_scale_f32) {
      TimedArch35(timeline, name, [&] {
        return arch35::LaunchBf16BlockQuantFp8(
            static_cast<const arch35::Bf16*>(src_bf16), width,
            static_cast<arch35::Fp8E4M3*>(dst_fp8),
            static_cast<float*>(dst_scale_f32), nullptr,
            arch35::kBlockQuantFp8WorkspaceBytes, stream);
      });
    };

    // =======================================================================
    // The shared-KV SWA attention core, planned once per layer.
    //
    // UNVERIFIED: every aclnn geometry decision here. The projections are
    // activation-quantized FP8 against the checkpoint's 128x128 E8M0 grid,
    // which QuantMatmulV5 is told about through `groupSize` (see the file
    // header). `transposeX2` is true because the checkpoint stores [N, K].
    // FIA V5 runs over one shared KV head with the per-head attn_sink as
    // `learnableSink`. Each of these is the documented contract; none has
    // been executed.
    // =======================================================================
    struct AttentionPlan {
      PipelineStage* q_a = nullptr;
      PipelineStage* q_norm = nullptr;
      PipelineStage* q_b = nullptr;
      PipelineStage* kv_proj = nullptr;
      PipelineStage* kv_norm = nullptr;
      PipelineStage* rope = nullptr;
      PipelineStage* attention = nullptr;
      PipelineStage* o_group[kOGroups] = {};
      PipelineStage* o_b = nullptr;
      // The four activation quantizers and their E8M0 -> FP32 casts are gone:
      // bf16_block_quant_fp8 replaces each pair with one arch35 dispatch that
      // writes the FP32 scales directly (see `block_quant` above), and an
      // arch35 launch is a direct call rather than a planned aclnn stage.
      void* kv_cache = nullptr;
      void* kv_normed = nullptr;
      std::size_t kv_row_bytes = 0;
    };

    // Attention scratch, shared by both layers: the two never overlap in time.
    void* a_h_fp8 = arena.Take(kHiddenSize, "attn.h_fp8");
    void* a_q_a = arena.Take(kQLoraRank * 2, "attn.q_a");
    void* a_q_normed = arena.Take(kQLoraRank * 2, "attn.q_normed");
    void* a_q_fp8 = arena.Take(kQLoraRank, "attn.q_fp8");
    void* a_q_heads = arena.Take(static_cast<std::size_t>(kAttnQkvWidth) * 2, "attn.q_heads");
    void* a_kv = arena.Take(static_cast<std::size_t>(kAttnHeadDim) * 2, "attn.kv");
    void* a_attn_out = arena.Take(static_cast<std::size_t>(kAttnQkvWidth) * 2, "attn.out");
    void* a_attn_fp8 = arena.Take(kAttnQkvWidth, "attn.out_fp8");
    void* a_o_mid = arena.Take(static_cast<std::size_t>(kWoAOutput) * 2, "attn.o_mid");
    void* a_o_fp8 = arena.Take(kWoAOutput, "attn.o_fp8");
    // Per-block activation scales, FP32 -- the dtype the ascend950 binary
    // takes for `pertoken_scale` (see the Fp8Matrix comment). These used to be
    // FP32 twins written by an aclnnCast from an E8M0 original;
    // bf16_block_quant_fp8 writes them directly, so the E8M0 buffers and the
    // four casts are gone.
    void* a_h_scale_f32 = arena.Take((kHiddenSize / kRoutedScaleBlock) * 4, "attn.h_scale_f32");
    void* a_q_scale_f32 = arena.Take((kQLoraRank / kRoutedScaleBlock) * 4, "attn.q_scale_f32");
    void* a_attn_scale_f32 = arena.Take((kAttnQkvWidth / kRoutedScaleBlock) * 4, "attn.out_scale_f32");
    void* a_o_scale_f32 = arena.Take((kWoAOutput / kRoutedScaleBlock) * 4, "attn.o_scale_f32");
    void* a_rstd = arena.Take(4 * 8, "attn.rstd");
    void* a_rope_cos = arena.Take(static_cast<std::size_t>(kAttnRopeHeadDim) * 2, "attn.rope_cos");
    void* a_rope_sin = arena.Take(static_cast<std::size_t>(kAttnRopeHeadDim) * 2, "attn.rope_sin");
    void* a_block_table = arena.Take(sizeof(std::int32_t), "attn.block_table");
    void* a_softmax_lse = arena.Take(static_cast<std::size_t>(kAttnHeads) * 4, "attn.softmax_lse");

    aclTensor* t_h_fp8 = bag.Tensor({1, kHiddenSize}, kAclFloat8E4m3Fn, a_h_fp8, "attn.h_fp8");
    aclTensor* t_h_scale_f32 =
        bag.Tensor({1, kHiddenSize / kRoutedScaleBlock}, kAclFloat32, a_h_scale_f32, "attn.h_scale_f32");
    aclTensor* t_q_a = bag.Tensor({1, kQLoraRank}, kAclBf16, a_q_a, "attn.q_a");
    aclTensor* t_q_normed = bag.Tensor({1, kQLoraRank}, kAclBf16, a_q_normed, "attn.q_normed");
    aclTensor* t_q_fp8 = bag.Tensor({1, kQLoraRank}, kAclFloat8E4m3Fn, a_q_fp8, "attn.q_fp8");
    aclTensor* t_q_scale_f32 =
        bag.Tensor({1, kQLoraRank / kRoutedScaleBlock}, kAclFloat32, a_q_scale_f32, "attn.q_scale_f32");
    aclTensor* t_q_flat = bag.Tensor({1, kAttnQkvWidth}, kAclBf16, a_q_heads, "attn.q_flat");
    aclTensor* t_q_bsnd = bag.Tensor({1, 1, kAttnHeads, kAttnHeadDim}, kAclBf16, a_q_heads, "attn.q_bsnd");
    aclTensor* t_kv = bag.Tensor({1, kAttnHeadDim}, kAclBf16, a_kv, "attn.kv");
    aclTensor* t_attn_bsnd = bag.Tensor({1, 1, kAttnHeads, kAttnHeadDim}, kAclBf16, a_attn_out, "attn.out_bsnd");
    // No whole-tensor views of attn_fp8 / attn_scale_f32: the eight wo_a
    // groups build their own offset sub-views from the arena pointers, and
    // the block quantizer takes raw pointers rather than aclTensors, so a
    // dense [1, 32768] view has no consumer left.
    aclTensor* t_o_fp8 = bag.Tensor({1, kWoAOutput}, kAclFloat8E4m3Fn, a_o_fp8, "attn.o_fp8");
    aclTensor* t_o_scale_f32 =
        bag.Tensor({1, kWoAOutput / kRoutedScaleBlock}, kAclFloat32, a_o_scale_f32, "attn.o_scale_f32");
    aclTensor* t_rstd = bag.Tensor({1, 1}, kAclFloat32, a_rstd, "attn.rstd");
    aclTensor* t_rope_cos = bag.Tensor({1, 1, 1, kAttnRopeHeadDim}, kAclBf16, a_rope_cos, "attn.rope_cos");
    aclTensor* t_rope_sin = bag.Tensor({1, 1, 1, kAttnRopeHeadDim}, kAclBf16, a_rope_sin, "attn.rope_sin");
    aclTensor* t_block_table = bag.Tensor({1, 1}, kAclInt32, a_block_table, "attn.block_table");
    aclTensor* t_softmax_lse = bag.Tensor({1, kAttnHeads, 1}, kAclFloat32, a_softmax_lse, "attn.softmax_lse");

    // RoPE rides the final 64 lanes of each 512-wide row, in place.
    aclTensor* t_q_rope = bag.Tensor({1, 1, kAttnHeads, kAttnRopeHeadDim}, kAclBf16,
                                     static_cast<std::uint16_t*>(a_q_heads) + (kAttnHeadDim - kAttnRopeHeadDim),
                                     "attn.q_rope", kAttnQkvWidth);
    aclTensor* t_k_rope = bag.Tensor({1, 1, 1, kAttnRopeHeadDim}, kAclBf16,
                                     static_cast<std::uint16_t*>(a_kv) + (kAttnHeadDim - kAttnRopeHeadDim),
                                     "attn.k_rope", kAttnHeadDim);

    const auto plan_attention = [&](int layer, AttentionPlan* plan) {
      const std::string prefix = layer_name(layer, "attn");
      const Fp8Matrix wq_a = StageFp8(checkpoint, arena, prefix + ".wq_a", kQLoraRank, kHiddenSize);
      const Fp8Matrix wq_b = StageFp8(checkpoint, arena, prefix + ".wq_b", kAttnQkvWidth, kQLoraRank);
      const Fp8Matrix wkv = StageFp8(checkpoint, arena, prefix + ".wkv", kAttnHeadDim, kHiddenSize);
      const Fp8Matrix wo_a = StageFp8(checkpoint, arena, prefix + ".wo_a", kWoAOutput, kWoAGroupInput);
      const Fp8Matrix wo_b = StageFp8(checkpoint, arena, prefix + ".wo_b", kHiddenSize, kWoAOutput);
      void* q_norm_w = checkpoint.Stage(prefix + ".q_norm.weight", "BF16", {kQLoraRank});
      void* kv_norm_w = checkpoint.Stage(prefix + ".kv_norm.weight", "BF16", {kAttnHeadDim});
      void* attn_sink = checkpoint.Stage(prefix + ".attn_sink", "F32", {kAttnHeads});

      // The 128-row shared-KV ring, one per layer: layer 0's window must not
      // be clobbered by layer 2's.
      plan->kv_row_bytes = static_cast<std::size_t>(kAttnHeadDim) * 2;
      plan->kv_cache = arena.Take(static_cast<std::size_t>(kSlidingWindow) * plan->kv_row_bytes, "attn.kv_cache");
      plan->kv_normed = a_kv;

      aclTensor* t_wq_a = bag.Tensor({kQLoraRank, kHiddenSize}, kAclFloat8E4m3Fn, wq_a.weight, "wq_a");
      aclTensor* t_wq_a_s =
          bag.Tensor({wq_a.scale_rows, wq_a.scale_cols}, kAclFloat32, wq_a.scale, "wq_a.scale");
      aclTensor* t_wq_b = bag.Tensor({kAttnQkvWidth, kQLoraRank}, kAclFloat8E4m3Fn, wq_b.weight, "wq_b");
      aclTensor* t_wq_b_s =
          bag.Tensor({wq_b.scale_rows, wq_b.scale_cols}, kAclFloat32, wq_b.scale, "wq_b.scale");
      aclTensor* t_wkv = bag.Tensor({kAttnHeadDim, kHiddenSize}, kAclFloat8E4m3Fn, wkv.weight, "wkv");
      aclTensor* t_wkv_s = bag.Tensor({wkv.scale_rows, wkv.scale_cols}, kAclFloat32, wkv.scale, "wkv.scale");
      aclTensor* t_wo_b = bag.Tensor({kHiddenSize, kWoAOutput}, kAclFloat8E4m3Fn, wo_b.weight, "wo_b");
      aclTensor* t_wo_b_s =
          bag.Tensor({wo_b.scale_rows, wo_b.scale_cols}, kAclFloat32, wo_b.scale, "wo_b.scale");
      aclTensor* t_q_norm_w = bag.Tensor({kQLoraRank}, kAclBf16, q_norm_w, "q_norm.weight");
      aclTensor* t_kv_norm_w = bag.Tensor({kAttnHeadDim}, kAclBf16, kv_norm_w, "kv_norm.weight");
      aclTensor* t_sink = bag.Tensor({kAttnHeads}, kAclFloat32, attn_sink, "attn_sink");
      aclTensor* t_kv_cache =
          bag.Tensor({1, kSlidingWindow, 1, kAttnHeadDim}, kAclBf16, plan->kv_cache, "attn.kv_cache");
      // Shared KV: all 64 query heads score the same 512-wide row, so key and
      // value are the same cache. num_key_value_heads = 1 is what makes that
      // legal rather than an aliasing bug.
      aclTensorList* kv_list = bag.List({t_kv_cache}, "attn.kv_list");

      // h_norm -> (h_fp8, h_scale_f32) is now block_quant(), called from the
      // execution path below rather than planned here.
      plan->q_a = &graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
          "attn: wq_a", OpId::kQuantMatmulV5, t_h_fp8, t_wq_a, t_h_scale_f32, t_wq_a_s, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, true, options.dense_group_size, t_q_a);
      plan->q_norm = &graph.Plan<ascend_moe::RmsNormPlanFn>("attn: q_norm", OpId::kRmsNorm, t_q_a, t_q_norm_w,
                                                            kRmsEpsilon, t_q_normed, t_rstd);
      plan->q_b = &graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
          "attn: wq_b", OpId::kQuantMatmulV5, t_q_fp8, t_wq_b, t_q_scale_f32, t_wq_b_s, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, true, options.dense_group_size, t_q_flat);
      plan->kv_proj = &graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
          "attn: wkv", OpId::kQuantMatmulV5, t_h_fp8, t_wkv, t_h_scale_f32, t_wkv_s, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, true, options.dense_group_size, t_kv);
      plan->kv_norm = &graph.Plan<ascend_moe::RmsNormPlanFn>("attn: kv_norm", OpId::kRmsNorm, t_kv, t_kv_norm_w,
                                                             kRmsEpsilon, t_kv, t_rstd);
      plan->rope = &graph.Plan<ascend_moe::ApplyRotaryPosEmbV2PlanFn>(
          "attn: rope", OpId::kApplyRotaryPosEmbV2, t_q_rope, t_k_rope, t_rope_cos, t_rope_sin, kRotaryLayoutBsnd,
          kRotaryModeHalf);

      // A learned per-head sink logit contributes to the softmax denominator
      // but has no value vector, so real-token probabilities sum to at most
      // one. That is what `learnableSink` is for; the host reference carries
      // the same term as an extra sinkless column.
      const double scale = 1.0 / std::sqrt(static_cast<double>(kAttnHeadDim));
      plan->attention = &graph.Plan<ascend_moe::FusedInferAttentionScoreV5PlanFn>(
          "attn: FusedInferAttentionScoreV5", OpId::kFusedInferAttentionScoreV5, t_q_bsnd, kv_list, kv_list,
          nullptr, nullptr, bag.IntArray({1}, "actual_seq_q"), bag.IntArray({kSlidingWindow}, "actual_seq_kv"),
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t_block_table, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, t_sink,
          nullptr, nullptr, static_cast<std::int64_t>(kAttnHeads), scale, kFiaPreTokensAll, kFiaNextTokensCausal,
          kFiaLayout, /*num_key_value_heads=*/1, kFiaSparseModeBand, kFiaInnerPreciseHighPrecision,
          static_cast<std::int64_t>(kSlidingWindow), kFiaAntiquantModeNone, /*softmax_lse_flag=*/false,
          kFiaAntiquantModeNone, kFiaAntiquantModeNone, kFiaQueryQuantModeNone, kFiaPseTypeNone, t_attn_bsnd,
          t_softmax_lse);

      // wo_a is block-diagonal: 8 groups of 4096 attention lanes each produce
      // 1024 of the 8192-wide intermediate. One QuantMatmulV5 per group over
      // offset views, rather than pretending it is a dense [8192, 32768].
      static const char* const kGroupNames[kOGroups] = {
          "attn: wo_a group 0", "attn: wo_a group 1", "attn: wo_a group 2", "attn: wo_a group 3",
          "attn: wo_a group 4", "attn: wo_a group 5", "attn: wo_a group 6", "attn: wo_a group 7"};
      for (int g = 0; g < kOGroups; ++g) {
        aclTensor* x_group =
            bag.Tensor({1, kWoAGroupInput}, kAclFloat8E4m3Fn,
                       static_cast<std::uint8_t*>(a_attn_fp8) + static_cast<std::size_t>(g) * kWoAGroupInput,
                       "attn.o_x_group", kAttnQkvWidth);
        aclTensor* x_group_scale = bag.Tensor(
            {1, kWoAGroupInput / kRoutedScaleBlock}, kAclFloat32,
            static_cast<float*>(a_attn_scale_f32) + static_cast<std::size_t>(g) * (kWoAGroupInput / kRoutedScaleBlock),
            "attn.o_x_group_scale", kAttnQkvWidth / kRoutedScaleBlock);
        aclTensor* w_group =
            bag.Tensor({kOLoraRank, kWoAGroupInput}, kAclFloat8E4m3Fn,
                       static_cast<std::uint8_t*>(wo_a.weight) +
                           static_cast<std::size_t>(g) * kOLoraRank * kWoAGroupInput,
                       "attn.wo_a_group", wo_a.rows * wo_a.cols);
        const std::int64_t group_scale_rows = kOLoraRank / kFp8BlockSize;
        aclTensor* w_group_scale =
            bag.Tensor({group_scale_rows, wo_a.scale_cols}, kAclFloat32,
                       wo_a.scale + static_cast<std::size_t>(g) * group_scale_rows * wo_a.scale_cols,
                       "attn.wo_a_group_scale", wo_a.scale_rows * wo_a.scale_cols);
        aclTensor* out_group =
            bag.Tensor({1, kOLoraRank}, kAclBf16,
                       static_cast<std::uint16_t*>(a_o_mid) + static_cast<std::size_t>(g) * kOLoraRank,
                       "attn.o_mid_group", kWoAOutput);
        plan->o_group[g] = &graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
            kGroupNames[g], OpId::kQuantMatmulV5, x_group, w_group, x_group_scale, w_group_scale, nullptr, nullptr,
            nullptr, nullptr, nullptr, false, true, options.dense_group_size, out_group);
      }

      plan->o_b = &graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
          "attn: wo_b", OpId::kQuantMatmulV5, t_o_fp8, t_wo_b, t_o_scale_f32, t_wo_b_s, nullptr, nullptr, nullptr,
          nullptr, nullptr, false, true, options.dense_group_size, t_h_out);
    };

    // One decode position through the attention core. `warm` writes the KV
    // ring only: a warmup token exists to populate the window, and the query
    // and output projections it does not need are most of the cost.
    const auto run_attention = [&](AttentionPlan& plan, int position, bool warm) {
      std::vector<std::uint16_t> cos_table;
      std::vector<std::uint16_t> sin_table;
      BuildRopeTables(position, kRopeThetaAttention, kAttnRopeHeadDim, &cos_table, &sin_table);
      const std::size_t rope_bytes = cos_table.size() * 2;
      DSV4_ACL_CHECK(aclrtMemcpyAsync(a_rope_cos, rope_bytes, cos_table.data(), rope_bytes,
                                      ACL_MEMCPY_HOST_TO_DEVICE, stream));
      DSV4_ACL_CHECK(aclrtMemcpyAsync(a_rope_sin, rope_bytes, sin_table.data(), rope_bytes,
                                      ACL_MEMCPY_HOST_TO_DEVICE, stream));

      // TIMING, NOW THAT THE QUANTIZERS ARE arch35 LAUNCHES
      // Each block_quant() records its own sync-inclusive row through
      // TimedArch35, and the ones below sit INSIDE the span `start` brackets
      // because stream order requires it (q_b consumes q_fp8, so the
      // quantizer cannot be hoisted out of the sequence). The aclnn group row
      // is therefore inclusive of the interleaved arch35 rows: the two
      // OVERLAP and must not be summed. The aclnn launch counts passed to
      // timeline.Add() drop accordingly -- two fewer enqueues per site, since
      // each quantizer replaced a DynamicMxQuant plus its scale Cast.
      const auto start = Clock::now();
      block_quant("bf16_block_quant_fp8: h_norm", h_norm, kHiddenSize, a_h_fp8, a_h_scale_f32);
      graph.Launch(*plan.kv_proj, stream);
      graph.Launch(*plan.kv_norm, stream);
      if (!warm) {
        graph.Launch(*plan.q_a, stream);
        graph.Launch(*plan.q_norm, stream);
        block_quant("bf16_block_quant_fp8: q_normed", a_q_normed, kQLoraRank, a_q_fp8, a_q_scale_f32);
        graph.Launch(*plan.q_b, stream);
      }
      graph.Launch(*plan.rope, stream);

      // The ring append: position p overwrites row p mod 128. A straight D2D
      // copy of the normed, rotated KV row -- no aclnnScatterPaKvCache,
      // because that operator writes a paged MLA pair and this cache is one
      // shared row per token.
      void* slot = static_cast<std::uint8_t*>(plan.kv_cache) +
                   static_cast<std::size_t>(position % kSlidingWindow) * plan.kv_row_bytes;
      DSV4_ACL_CHECK(aclrtMemcpyAsync(slot, plan.kv_row_bytes, plan.kv_normed, plan.kv_row_bytes,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream));

      if (!warm) {
        graph.Launch(*plan.attention, stream);
        block_quant("bf16_block_quant_fp8: attn_out", a_attn_out, kAttnQkvWidth, a_attn_fp8, a_attn_scale_f32);
        for (PipelineStage* group : plan.o_group) graph.Launch(*group, stream);
        block_quant("bf16_block_quant_fp8: o_mid", a_o_mid, kWoAOutput, a_o_fp8, a_o_scale_f32);
        graph.Launch(*plan.o_b, stream);
      }
      DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
      // aclnn enqueues only: 3 on a warm token (kv_proj, kv_norm, rope) and
      // 7 + kOGroups on a full one (+ q_a, q_norm, q_b, attention, o_b). The
      // four quantizers that used to contribute 8 of these enqueues are now
      // arch35 launches with their own rows.
      timeline.Add(warm ? "attention: KV warmup (aclnn)" : "attention: shared-KV SWA (aclnn)",
                   MicrosecondsSince(start), warm ? 3 : 7 + kOGroups, false);
    };

    // =======================================================================
    // [2] run
    // =======================================================================
    std::printf("\n[2] running the pipeline on NPU (device %d)\n", options.device_id);

    AttentionPlan layer0_attention;
    AttentionPlan layer2_attention;
    const MhcWeights mhc_layer0 = load_mhc(options.layer0, "attn", "attn_norm");
    const MhcWeights mhc_layer2 = load_mhc(options.layer2, "attn", "attn_norm");
    const MhcWeights mhc_moe = load_mhc(options.moe_layer, "ffn", "ffn_norm");
    plan_attention(options.layer0, &layer0_attention);
    plan_attention(options.layer2, &layer2_attention);

    // --- the compressor's weights and persistent state ---------------------
    const std::string compressor_prefix = layer_name(options.layer2, "attn.compressor");
    const auto* c_wkv = static_cast<const arch35::Bf16*>(
        checkpoint.Stage(compressor_prefix + ".wkv.weight", "BF16", {kCompressorStateWidth, kHiddenSize}));
    const auto* c_wgate = static_cast<const arch35::Bf16*>(
        checkpoint.Stage(compressor_prefix + ".wgate.weight", "BF16", {kCompressorStateWidth, kHiddenSize}));
    const float* c_ape = checkpoint.StageAsFloat(compressor_prefix + ".ape", {kCompressorRatio, kCompressorStateWidth});
    const auto* c_norm = static_cast<const arch35::Bf16*>(
        checkpoint.Stage(compressor_prefix + ".norm.weight", "BF16", {kCompressorHeadDim}));
    auto* c_state = static_cast<float*>(arena.Take(arch35::kCompressorStateBytes, "compressor.state"));
    auto* c_state_meta =
        static_cast<std::uint32_t*>(arena.Take(arch35::kCompressorStateMetaBytes, "compressor.state_meta"));
    auto* c_slot = static_cast<arch35::Bf16*>(
        arena.Take(static_cast<std::size_t>(kCompressorHeadDim) * 2, "compressor.cache_slot"));
    auto* c_emitted = static_cast<std::uint32_t*>(arena.Take(sizeof(std::uint32_t), "compressor.emitted"));
    auto* c_token = static_cast<arch35::Bf16*>(arena.Take(static_cast<std::size_t>(kHiddenSize) * 2, "compressor.token"));
    auto* c_rope_cos = static_cast<float*>(arena.Take(static_cast<std::size_t>(kAttnRopeHeadDim) * 4, "compressor.cos"));
    auto* c_rope_sin = static_cast<float*>(arena.Take(static_cast<std::size_t>(kAttnRopeHeadDim) * 4, "compressor.sin"));

    // --- the indexer's head gate and its synthesized activation history ----
    const std::string indexer_prefix = layer_name(options.layer2, "attn.indexer");
    void* i_weights_proj =
        checkpoint.Stage(indexer_prefix + ".weights_proj.weight", "BF16", {kIndexerHeads, kHiddenSize});
    const std::int32_t indexer_blocks = (options.indexer_seq_len + kIndexerBlockSize - 1) / kIndexerBlockSize;
    auto* i_head_weights = static_cast<float*>(arena.Take(static_cast<std::size_t>(kIndexerHeads) * 4, "indexer.gate"));
    auto* i_query = static_cast<arch35::Fp8E4M3*>(
        arena.Take(static_cast<std::size_t>(kIndexerHeads) * kIndexerHeadDim, "indexer.query"));
    auto* i_keys = static_cast<arch35::Fp8E4M3*>(arena.Take(
        static_cast<std::size_t>(indexer_blocks) * kIndexerBlockSize * kIndexerHeadDim, "indexer.key_cache"));
    auto* i_scales = static_cast<float*>(
        arena.Take((static_cast<std::size_t>(kIndexerHeads) + options.indexer_seq_len) * 4, "indexer.scales"));
    auto* i_block_table =
        static_cast<std::int32_t*>(arena.Take(static_cast<std::size_t>(indexer_blocks) * 4, "indexer.block_table"));
    auto* i_indices = static_cast<std::int32_t*>(arena.Take(static_cast<std::size_t>(kIndexerTopK) * 4, "indexer.out"));
    const std::size_t indexer_workspace_bytes =
        arch35::Fp8IndexerWorkspaceBytes(static_cast<std::uint32_t>(options.indexer_seq_len));
    void* i_workspace = arena.Take(indexer_workspace_bytes, "indexer.workspace");

    aclTensor* t_weights_proj =
        bag.Transposed2d(kIndexerHeads, kHiddenSize, kAclBf16, i_weights_proj, "indexer.weights_proj");
    aclTensor* t_head_weights_bf16 = nullptr;
    void* i_head_weights_bf16 = arena.Take(static_cast<std::size_t>(kIndexerHeads) * 2, "indexer.gate_bf16");
    t_head_weights_bf16 = bag.Tensor({1, kIndexerHeads}, kAclBf16, i_head_weights_bf16, "indexer.gate_bf16");
    aclTensor* t_head_weights = bag.Tensor({1, kIndexerHeads}, kAclFloat32, i_head_weights, "indexer.gate");
    PipelineStage& indexer_gate =
        graph.Plan<ascend_moe::MatmulPlanFn>("indexer: head gate", OpId::kMatmul, t_h_norm, t_weights_proj,
                                             t_head_weights_bf16, kCubeMathTypeKeepDtype);
    PipelineStage& indexer_gate_cast = graph.Plan<ascend_moe::CastPlanFn>(
        "indexer: gate -> f32", OpId::kCast, t_head_weights_bf16, static_cast<aclDataType>(kAclFloat32),
        t_head_weights);

    // --- the MoE block -----------------------------------------------------
    const std::string moe_prefix = layer_name(options.moe_layer, "ffn");
    void* gate_weight = checkpoint.Stage(moe_prefix + ".gate.weight", "BF16", {kNumRoutedExperts, kHiddenSize});

    // The route row. This is a HASH layer, so the six experts are a pure
    // function of the token id -- which the host must know before it can
    // decide WHICH expert weights to stage. The device router below is still
    // run, and its expert ids are checked against this row after the run, so
    // the host read is a staging decision and not a shortcut around the
    // device routing.
    const std::string tid2eid_name = moe_prefix + ".gate.tid2eid";
    DSV4_REQUIRE(checkpoint.Has(tid2eid_name),
                 "layer " << options.moe_layer << " has no tid2eid, so it is not a hash layer; this bench "
                          << "implements the hash route only");
    const std::vector<std::int64_t> tid2eid_shape = checkpoint.Shape(tid2eid_name);
    DSV4_REQUIRE(tid2eid_shape.size() == 2 && tid2eid_shape[1] == kNumExpertsPerTok,
                 tid2eid_name << " is not [vocab, " << kNumExpertsPerTok << "]");
    DSV4_REQUIRE(options.token_id >= 0 && options.token_id < tid2eid_shape[0],
                 "token id " << options.token_id << " is outside [0, " << tid2eid_shape[0] << ")");
    const std::vector<std::uint8_t> tid2eid_bytes = checkpoint.ReadHost(tid2eid_name);
    const std::string tid2eid_dtype = checkpoint.Dtype(tid2eid_name);
    const std::size_t tid2eid_element = tid2eid_bytes.size() / (static_cast<std::size_t>(tid2eid_shape[0]) *
                                                                static_cast<std::size_t>(tid2eid_shape[1]));
    std::vector<std::int32_t> host_experts(kNumExpertsPerTok);
    for (int i = 0; i < kNumExpertsPerTok; ++i) {
      const std::size_t at =
          (static_cast<std::size_t>(options.token_id) * kNumExpertsPerTok + static_cast<std::size_t>(i)) *
          tid2eid_element;
      std::int64_t value = 0;
      std::memcpy(&value, tid2eid_bytes.data() + at, std::min<std::size_t>(tid2eid_element, sizeof(value)));
      host_experts[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value);
    }
    void* d_tid2eid = arena.Stage(tid2eid_bytes.data(), tid2eid_bytes.size(), "ffn.gate.tid2eid");
    std::int32_t host_token_id = static_cast<std::int32_t>(options.token_id);
    void* d_input_ids = arena.Stage(&host_token_id, sizeof(host_token_id), "ffn.input_ids");

    // Router scratch.
    void* r_logits_bf16 = arena.Take(kNumRoutedExperts * 2, "router.logits_bf16");
    void* r_logits = arena.Take(kNumRoutedExperts * 4, "router.logits");
    void* r_softplus = arena.Take(kNumRoutedExperts * 4, "router.softplus");
    void* r_scores = arena.Take(kNumRoutedExperts * 4, "router.scores");
    void* r_weights = arena.Take(kNumExpertsPerTok * 4, "router.weights");
    void* r_experts = arena.Take(kNumExpertsPerTok * 4, "router.experts");

    aclTensor* t_gate_w = bag.Transposed2d(kNumRoutedExperts, kHiddenSize, kAclBf16, gate_weight, "ffn.gate.weight");
    aclTensor* t_logits_bf16 = bag.Tensor({1, kNumRoutedExperts}, kAclBf16, r_logits_bf16, "router.logits_bf16");
    aclTensor* t_logits = bag.Tensor({1, kNumRoutedExperts}, kAclFloat32, r_logits, "router.logits");
    aclTensor* t_softplus = bag.Tensor({1, kNumRoutedExperts}, kAclFloat32, r_softplus, "router.softplus");
    aclTensor* t_scores = bag.Tensor({1, kNumRoutedExperts}, kAclFloat32, r_scores, "router.scores");
    aclTensor* t_weights_out = bag.Tensor({1, kNumExpertsPerTok}, kAclFloat32, r_weights, "router.weights");
    aclTensor* t_experts_out = bag.Tensor({1, kNumExpertsPerTok}, kAclInt32, r_experts, "router.experts");
    aclTensor* t_tid2eid =
        bag.Tensor({tid2eid_shape[0], kNumExpertsPerTok}, kAclInt32, d_tid2eid, "router.tid2eid");
    aclTensor* t_input_ids = bag.Tensor({1}, kAclInt32, d_input_ids, "router.input_ids");

    // router_logits = F.linear(h_norm.float(), gate.weight), then the DSV4
    // scoring function s_i = sqrt(softplus(g_i)).
    PipelineStage& router_matmul = graph.Plan<ascend_moe::MatmulPlanFn>(
        "router: gate matmul", OpId::kMatmul, t_h_norm, t_gate_w, t_logits_bf16, kCubeMathTypeKeepDtype);
    PipelineStage& router_cast = graph.Plan<ascend_moe::CastPlanFn>(
        "router: logits -> f32", OpId::kCast, t_logits_bf16, static_cast<aclDataType>(kAclFloat32), t_logits);
    PipelineStage& router_softplus = graph.Plan<ascend_moe::SoftplusPlanFn>(
        "router: softplus", OpId::kSoftplus, t_logits, bag.Scalar(static_cast<float>(kSoftplusBeta), "beta"),
        bag.Scalar(static_cast<float>(kSoftplusThreshold), "threshold"), t_softplus);
    PipelineStage& router_sqrt =
        graph.Plan<ascend_moe::UnaryPlanFn>("router: sqrt", OpId::kSqrt, t_softplus, t_scores);
    // The hash branch: tid2eid[token_id] fixes membership and the scores only
    // set the weights. routed_scaling_factor rides here because the top-k
    // weights are normalized first -- folding 1.5 in at this point scales the
    // routed sum exactly as DeepseekV4MoE.forward does, and leaves the shared
    // expert at unit weight.
    PipelineStage& router_gating = graph.Plan<ascend_moe::MoeGatingTopKV2PlanFn>(
        "router: MoeGatingTopKV2 (hash)", OpId::kMoeGatingTopKV2, t_scores, nullptr, t_input_ids, t_tid2eid,
        static_cast<std::int64_t>(kNumExpertsPerTok), kGatingKGroup, kGatingGroupCount, kGatingGroupSelectMode,
        kGatingRenormL1, kGatingNormTypePreNormalized, false, static_cast<double>(kRoutedScalingFactor),
        kGatingEps, t_weights_out, t_experts_out, nullptr);

    // Expert scratch. One [4096] buffer holds [gate | up] back to back so
    // aclnnSwiGlu's split along the last axis is exactly the gate/up split.
    void* e_gate_up = arena.Take(static_cast<std::size_t>(2 * kMoeIntermediate) * 4, "expert.gate_up");
    void* e_hidden = arena.Take(static_cast<std::size_t>(kMoeIntermediate) * 4, "expert.hidden");
    void* e_out = arena.Take(static_cast<std::size_t>(kHiddenSize) * 4, "expert.out");
    void* e_routed = arena.Take(static_cast<std::size_t>(kHiddenSize) * 4, "expert.routed_sum");
    void* e_scaled = arena.Take(static_cast<std::size_t>(kHiddenSize) * 4, "expert.scaled");
    void* e_weight_broadcast = arena.Take(static_cast<std::size_t>(kHiddenSize) * 4, "expert.weight_broadcast");
    // dsv4_moe_expert's `activated`, written by its gate_up pass and read by
    // its down pass. Unlike the other arch35 operators' workspaces this is a
    // real inter-stage operand, not scratch, and it must not alias x or y --
    // the wrapper refuses if it does.
    const std::size_t e_expert_workspace_bytes =
        arch35::Dsv4MoeExpertWorkspaceBytes(static_cast<std::uint32_t>(kMoeIntermediate));
    void* e_expert_workspace = arena.Take(e_expert_workspace_bytes, "expert.activated");

    auto* e_gate_f32 = static_cast<float*>(e_gate_up);
    auto* e_up_f32 = static_cast<float*>(e_gate_up) + kMoeIntermediate;

    aclTensor* t_gate_half =
        bag.Tensor({1, kMoeIntermediate}, kAclFloat32, e_gate_f32, "expert.gate", 2 * kMoeIntermediate);
    aclTensor* t_up_half =
        bag.Tensor({1, kMoeIntermediate}, kAclFloat32, e_up_f32, "expert.up", 2 * kMoeIntermediate);
    aclTensor* t_gate_up = bag.Tensor({1, 2 * kMoeIntermediate}, kAclFloat32, e_gate_up, "expert.gate_up");
    aclTensor* t_hidden = bag.Tensor({1, kMoeIntermediate}, kAclFloat32, e_hidden, "expert.hidden");
    aclTensor* t_expert_out = bag.Tensor({1, kHiddenSize}, kAclFloat32, e_out, "expert.out");
    aclTensor* t_routed = bag.Tensor({1, kHiddenSize}, kAclFloat32, e_routed, "expert.routed_sum");
    aclTensor* t_scaled = bag.Tensor({1, kHiddenSize}, kAclFloat32, e_scaled, "expert.scaled");
    aclTensor* t_weight_broadcast =
        bag.Tensor({1, kHiddenSize}, kAclFloat32, e_weight_broadcast, "expert.weight_broadcast");

    // DSV4's clamp is ASYMMETRIC: gate = clamp(gate, max=10) with no lower
    // bound, up = clamp(up, -10, 10). aclnnSwiGlu cannot express that, which
    // is why the two halves are clamped separately before the split.
    PipelineStage& clamp_gate = graph.Plan<ascend_moe::ClampPlanFn>(
        "expert: clamp gate (max only)", OpId::kClamp, t_gate_half, nullptr,
        bag.Scalar(kSwigluClampLimit, "swiglu.limit"), t_gate_half);
    PipelineStage& clamp_up = graph.Plan<ascend_moe::ClampPlanFn>(
        "expert: clamp up (two-sided)", OpId::kClamp, t_up_half, bag.Scalar(-kSwigluClampLimit, "swiglu.-limit"),
        bag.Scalar(kSwigluClampLimit, "swiglu.limit"), t_up_half);
    PipelineStage& expert_swiglu = graph.Plan<ascend_moe::SwiGluPlanFn>("expert: SwiGLU", OpId::kSwiGlu, t_gate_up,
                                                                        kSwigluAxisLast, t_hidden);
    PipelineStage& expert_weight_mul = graph.Plan<ascend_moe::BinaryPlanFn>(
        "expert: scale by route weight", OpId::kMul, t_expert_out, t_weight_broadcast, t_scaled);
    PipelineStage& expert_accumulate = graph.Plan<ascend_moe::InplaceAddPlanFn>(
        "expert: accumulate", OpId::kInplaceAdd, t_routed, t_scaled, nullptr);

    // Stage the six routed experts' MXFP4 matrices.
    struct RoutedExpert {
      Mxfp4Matrix w1;
      Mxfp4Matrix w2;
      Mxfp4Matrix w3;
    };
    std::vector<RoutedExpert> routed_experts;
    routed_experts.reserve(kNumExpertsPerTok);
    for (int i = 0; i < kNumExpertsPerTok; ++i) {
      const std::string stem = moe_prefix + ".experts." + std::to_string(host_experts[static_cast<std::size_t>(i)]);
      DSV4_REQUIRE(checkpoint.Has(stem + ".w1.weight"),
                   "the router selected expert " << host_experts[static_cast<std::size_t>(i)]
                                                 << " but the weight file does not carry it. Re-fetch with "
                                                 << "--token-id " << options.token_id << ".");
      RoutedExpert expert;
      expert.w1 = StageMxfp4(checkpoint, stem + ".w1", kMoeIntermediate, kHiddenSize);
      expert.w3 = StageMxfp4(checkpoint, stem + ".w3", kMoeIntermediate, kHiddenSize);
      expert.w2 = StageMxfp4(checkpoint, stem + ".w2", kHiddenSize, kMoeIntermediate);
      routed_experts.push_back(expert);
    }

    // The shared expert is FP8, not MXFP4, in this slice.
    const Fp8Matrix shared_w1 = StageFp8(checkpoint, arena, moe_prefix + ".shared_experts.w1", kMoeIntermediate, kHiddenSize);
    const Fp8Matrix shared_w3 = StageFp8(checkpoint, arena, moe_prefix + ".shared_experts.w3", kMoeIntermediate, kHiddenSize);
    const Fp8Matrix shared_w2 = StageFp8(checkpoint, arena, moe_prefix + ".shared_experts.w2", kHiddenSize, kMoeIntermediate);

    void* s_gate_up_bf16 = arena.Take(static_cast<std::size_t>(2 * kMoeIntermediate) * 2, "shared.gate_up_bf16");
    void* s_hidden_fp8 = arena.Take(kMoeIntermediate, "shared.hidden_fp8");
    void* s_hidden_scale = arena.Take(kMoeIntermediate / kRoutedScaleBlock, "shared.hidden_scale");
    void* s_hidden_scale_f32 =
        arena.Take((kMoeIntermediate / kRoutedScaleBlock) * 4, "shared.hidden_scale_f32");
    void* s_out_bf16 = arena.Take(static_cast<std::size_t>(kHiddenSize) * 2, "shared.out_bf16");

    aclTensor* t_s_gate_bf16 = bag.Tensor({1, kMoeIntermediate}, kAclBf16, s_gate_up_bf16, "shared.gate_bf16",
                                          2 * kMoeIntermediate);
    aclTensor* t_s_up_bf16 =
        bag.Tensor({1, kMoeIntermediate}, kAclBf16, static_cast<std::uint16_t*>(s_gate_up_bf16) + kMoeIntermediate,
                   "shared.up_bf16", 2 * kMoeIntermediate);
    aclTensor* t_s_gate_f32 = t_gate_half;  // reuse the FP32 gate/up pair
    aclTensor* t_s_up_f32 = t_up_half;
    aclTensor* t_s_hidden_fp8 = bag.Tensor({1, kMoeIntermediate}, kAclFloat8E4m3Fn, s_hidden_fp8, "shared.hidden_fp8");
    aclTensor* t_s_hidden_scale = bag.Tensor({1, kMoeIntermediate / kRoutedScaleBlock}, kAclFloat8E8m0,
                                             s_hidden_scale, "shared.hidden_scale");
    aclTensor* t_s_hidden_scale_f32 = bag.Tensor({1, kMoeIntermediate / kRoutedScaleBlock}, kAclFloat32,
                                                 s_hidden_scale_f32, "shared.hidden_scale_f32");
    aclTensor* t_s_out_bf16 = bag.Tensor({1, kHiddenSize}, kAclBf16, s_out_bf16, "shared.out_bf16");

    aclTensor* t_sw1 = bag.Tensor({kMoeIntermediate, kHiddenSize}, kAclFloat8E4m3Fn, shared_w1.weight, "shared.w1");
    aclTensor* t_sw1_s =
        bag.Tensor({shared_w1.scale_rows, shared_w1.scale_cols}, kAclFloat32, shared_w1.scale, "shared.w1.scale");
    aclTensor* t_sw3 = bag.Tensor({kMoeIntermediate, kHiddenSize}, kAclFloat8E4m3Fn, shared_w3.weight, "shared.w3");
    aclTensor* t_sw3_s =
        bag.Tensor({shared_w3.scale_rows, shared_w3.scale_cols}, kAclFloat32, shared_w3.scale, "shared.w3.scale");
    aclTensor* t_sw2 = bag.Tensor({kHiddenSize, kMoeIntermediate}, kAclFloat8E4m3Fn, shared_w2.weight, "shared.w2");
    aclTensor* t_sw2_s =
        bag.Tensor({shared_w2.scale_rows, shared_w2.scale_cols}, kAclFloat32, shared_w2.scale, "shared.w2.scale");

    PipelineStage& shared_gate = graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
        "shared expert: w1", OpId::kQuantMatmulV5, t_h_fp8, t_sw1, t_h_scale_f32, t_sw1_s, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, options.dense_group_size, t_s_gate_bf16);
    PipelineStage& shared_up = graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
        "shared expert: w3", OpId::kQuantMatmulV5, t_h_fp8, t_sw3, t_h_scale_f32, t_sw3_s, nullptr, nullptr, nullptr,
        nullptr, nullptr, false, true, options.dense_group_size, t_s_up_bf16);
    PipelineStage& shared_gate_cast = graph.Plan<ascend_moe::CastPlanFn>(
        "shared expert: gate -> f32", OpId::kCast, t_s_gate_bf16, static_cast<aclDataType>(kAclFloat32),
        t_s_gate_f32);
    PipelineStage& shared_up_cast = graph.Plan<ascend_moe::CastPlanFn>(
        "shared expert: up -> f32", OpId::kCast, t_s_up_bf16, static_cast<aclDataType>(kAclFloat32), t_s_up_f32);
    // THE ONE REMAINING aclnnDynamicMxQuant.
    // t_hidden is FP32 (kAclFloat32 over e_hidden), not BF16, so
    // bf16_block_quant_fp8 cannot be substituted here: handing it this buffer
    // would reinterpret every FP32 as two BF16 values. The arch35 operator
    // needs an FP32 input mode before this site can move, which is a kernel
    // change (a tiling flag plus a branch on the staging load), not a
    // rewiring. Left on the aclnn path rather than silently misreading the
    // bytes -- and this is the site that will still hit EZ1009 on ascend950.
    PipelineStage& shared_quant = graph.Plan<ascend_moe::DynamicMxQuantPlanFn>(
        "shared expert: quantize hidden", OpId::kDynamicMxQuant, t_hidden, kSwigluAxisLast, kRoundModeRint,
        static_cast<std::int64_t>(kAclFloat8E4m3Fn), kRoutedScaleBlock, kSwigluScaleAlgOcp, t_s_hidden_fp8,
        t_s_hidden_scale);
    PipelineStage& shared_hidden_scale_cast = graph.Plan<ascend_moe::CastPlanFn>(
        "shared expert: hidden scale -> f32", OpId::kCast, t_s_hidden_scale,
        static_cast<aclDataType>(kAclFloat32), t_s_hidden_scale_f32);
    PipelineStage& shared_down = graph.Plan<ascend_moe::QuantMatmulV5PlanFn>(
        "shared expert: w2", OpId::kQuantMatmulV5, t_s_hidden_fp8, t_sw2, t_s_hidden_scale_f32, t_sw2_s, nullptr,
        nullptr, nullptr, nullptr, nullptr, false, true, options.dense_group_size, t_s_out_bf16);

    // ffn_out = 1.5 * routed_sum + shared, in BF16 for the mHC fold-back. The
    // 1.5 already rode in through the gating's routed_scaling_factor.
    PipelineStage& routed_to_bf16 = graph.Plan<ascend_moe::CastPlanFn>(
        "MoE: routed -> bf16", OpId::kCast, t_routed, static_cast<aclDataType>(kAclBf16), t_h_out);
    PipelineStage& add_shared = graph.Plan<ascend_moe::InplaceAddPlanFn>("MoE: add shared expert",
                                                                         OpId::kInplaceAdd, t_h_out, t_s_out_bf16,
                                                                         nullptr);

    // One shared workspace, sized by the high-water mark over every planned
    // stage. Taken after planning, which is why nothing in the timed region
    // needs to allocate.
    void* aclnn_workspace = nullptr;
    if (graph.workspace_high_water() > 0) {
      aclnn_workspace = arena.Take(static_cast<std::size_t>(graph.workspace_high_water()), "aclnn.workspace");
    }
    graph.set_workspace(aclnn_workspace);

    std::printf("    %zu aclnn stages planned, %zu tensors staged (%.1f MB read)\n", graph.size(),
                checkpoint.count(), static_cast<double>(checkpoint.bytes()) / 1048576.0);
    std::printf("    routed experts (hash row for token %lld):", static_cast<long long>(options.token_id));
    for (std::int32_t expert : host_experts) std::printf(" %d", expert);
    std::printf("\n");

    // =======================================================================
    // Execute. Everything below is enqueue-only except the arch35 wrappers'
    // own internal syncs and the explicit group boundaries.
    // =======================================================================
    const auto wall_start = Clock::now();
    std::vector<std::uint16_t> x_after0(bundle);
    std::vector<std::uint16_t> x_after2(bundle);

    const auto run_attention_layer = [&](const MhcWeights& mhc, AttentionPlan& plan, const arch35::Bf16* x,
                                         arch35::Bf16* y) {
      mhc_prologue(mhc, x);
      // Prime the sliding window, then the measured step. Without this the
      // window holds one token and the softmax is a no-op, which would hide a
      // window or RoPE error.
      for (int warm = 0; warm < options.warmup_window_tokens; ++warm) run_attention(plan, warm, true);
      run_attention(plan, options.warmup_window_tokens, false);
      mhc_epilogue(x, y);
    };

    // ===== layer 0 =========================================================
    run_attention_layer(mhc_layer0, layer0_attention, x_ping, x_pong);
    DSV4_ACL_CHECK(aclrtMemcpy(x_after0.data(), bundle * 2, x_pong, bundle * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // ===== layer 2 =========================================================
    run_attention_layer(mhc_layer2, layer2_attention, x_pong, x_ping);

    // --- the 4:1 compressor cadence, driven by layer 2's h_norm ------------
    {
      std::vector<std::uint16_t> host_h_norm(kHiddenSize);
      DSV4_ACL_CHECK(aclrtMemcpy(host_h_norm.data(), kHiddenSize * 2, h_norm, kHiddenSize * 2,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
      std::vector<std::uint16_t> token(kHiddenSize);
      std::vector<std::uint16_t> cos_bf16;
      std::vector<std::uint16_t> sin_bf16;
      std::vector<float> cos_f32(kAttnRopeHeadDim);
      std::vector<float> sin_f32(kAttnRopeHeadDim);
      int emitted_total = 0;
      for (int step = 0; step < options.decode_steps; ++step) {
        const float drift = 1.0F + 0.01F * static_cast<float>(step);
        for (int i = 0; i < kHiddenSize; ++i) {
          token[static_cast<std::size_t>(i)] =
              FloatToBf16(Bf16ToFloat(host_h_norm[static_cast<std::size_t>(i)]) * drift);
        }
        DSV4_ACL_CHECK(aclrtMemcpy(c_token, kHiddenSize * 2, token.data(), kHiddenSize * 2,
                                   ACL_MEMCPY_HOST_TO_DEVICE));
        // The compressor path uses "compress_rope_theta": 160000, NOT
        // "rope_theta": 10000, and the table is for the COMPRESSED token
        // index, not the decode step.
        BuildRopeTables(step / kCompressorRatio, kRopeThetaCompressor, kAttnRopeHeadDim, &cos_bf16, &sin_bf16);
        for (int i = 0; i < kAttnRopeHeadDim; ++i) {
          cos_f32[static_cast<std::size_t>(i)] = Bf16ToFloat(cos_bf16[static_cast<std::size_t>(i)]);
          sin_f32[static_cast<std::size_t>(i)] = Bf16ToFloat(sin_bf16[static_cast<std::size_t>(i)]);
        }
        DSV4_ACL_CHECK(aclrtMemcpy(c_rope_cos, kAttnRopeHeadDim * 4, cos_f32.data(), kAttnRopeHeadDim * 4,
                                   ACL_MEMCPY_HOST_TO_DEVICE));
        DSV4_ACL_CHECK(aclrtMemcpy(c_rope_sin, kAttnRopeHeadDim * 4, sin_f32.data(), kAttnRopeHeadDim * 4,
                                   ACL_MEMCPY_HOST_TO_DEVICE));

        TimedArch35(timeline, "paged_compressor_rope", [&] {
          return arch35::LaunchPagedCompressorRope(
              c_token, c_wkv, kHiddenSize, c_wgate, kHiddenSize, c_ape, kCompressorStateWidth, c_norm, c_rope_cos,
              c_rope_sin, static_cast<std::uint32_t>(step), c_state, c_state_meta, c_slot, c_emitted, nullptr,
              arch35::kPagedCompressorWorkspaceBytes, stream);
        });
        std::uint32_t emitted = 0;
        DSV4_ACL_CHECK(aclrtMemcpy(&emitted, sizeof(emitted), c_emitted, sizeof(emitted),
                                   ACL_MEMCPY_DEVICE_TO_HOST));
        if (emitted != 0) ++emitted_total;
      }
      std::printf("    compressor: %d decode steps -> %d emitted token(s) at %d:1\n", options.decode_steps,
                  emitted_total, kCompressorRatio);
    }

    // --- the FP8 lightning indexer, top-512 --------------------------------
    {
      // The per-head gate is real: w = weights_proj @ h_norm, folded into the
      // query dequant scale exactly as the host golden's
      // `cur_w = cur_wt * cur_q_scale` does. Magnitudes only, because the
      // scale multiplies a relu'd score.
      const auto gate_start = Clock::now();
      graph.Launch(indexer_gate, stream);
      graph.Launch(indexer_gate_cast, stream);
      DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
      timeline.Add("indexer: head gate (aclnn)", MicrosecondsSince(gate_start), 2, false);

      // SYNTHETIC INPUT, exactly as the host bench does it and for the same
      // reason: the indexer's FP8 query and key cache are runtime ACTIVATIONS
      // that need indexer_seq_len historical compressed tokens, which a single
      // decode step has not produced. The geometry and the head gate are real;
      // the Q/K bytes are not. Seeded from the token id and printed, so a
      // failure is reproducible.
      const std::uint32_t indexer_seed = 0xC0FFEEu ^ static_cast<std::uint32_t>(options.token_id);
      std::mt19937 engine(indexer_seed);
      std::uniform_real_distribution<float> distribution(-2.0F, 2.0F);
      std::vector<std::uint8_t> query(static_cast<std::size_t>(kIndexerHeads) * kIndexerHeadDim);
      for (std::uint8_t& byte : query) byte = EncodeFp8E4M3(distribution(engine));
      std::vector<std::uint8_t> keys(static_cast<std::size_t>(indexer_blocks) * kIndexerBlockSize * kIndexerHeadDim);
      for (std::uint8_t& byte : keys) byte = EncodeFp8E4M3(distribution(engine));

      std::vector<float> host_scales(static_cast<std::size_t>(kIndexerHeads) + options.indexer_seq_len);
      std::vector<float> head_gate(kIndexerHeads);
      DSV4_ACL_CHECK(aclrtMemcpy(head_gate.data(), kIndexerHeads * 4, i_head_weights, kIndexerHeads * 4,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
      for (int head = 0; head < kIndexerHeads; ++head) {
        host_scales[static_cast<std::size_t>(head)] = std::fabs(head_gate[static_cast<std::size_t>(head)]);
      }
      for (std::int32_t j = 0; j < options.indexer_seq_len; ++j) {
        host_scales[static_cast<std::size_t>(kIndexerHeads + j)] = 0.5F + distribution(engine) * 0.1F;
      }
      std::vector<std::int32_t> host_block_table(static_cast<std::size_t>(indexer_blocks));
      for (std::int32_t b = 0; b < indexer_blocks; ++b) host_block_table[static_cast<std::size_t>(b)] = b;

      DSV4_ACL_CHECK(aclrtMemcpy(i_query, query.size(), query.data(), query.size(), ACL_MEMCPY_HOST_TO_DEVICE));
      DSV4_ACL_CHECK(aclrtMemcpy(i_keys, keys.size(), keys.data(), keys.size(), ACL_MEMCPY_HOST_TO_DEVICE));
      DSV4_ACL_CHECK(aclrtMemcpy(i_scales, host_scales.size() * 4, host_scales.data(), host_scales.size() * 4,
                                 ACL_MEMCPY_HOST_TO_DEVICE));
      DSV4_ACL_CHECK(aclrtMemcpy(i_block_table, host_block_table.size() * 4, host_block_table.data(),
                                 host_block_table.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));

      TimedArch35(timeline, "fast_fp8_indexer_topk", [&] {
        return arch35::LaunchFp8IndexerTopK(i_query, kIndexerHeadDim, i_keys, kIndexerBlockSize * kIndexerHeadDim,
                                            i_scales, i_block_table, static_cast<std::uint32_t>(indexer_blocks),
                                            static_cast<std::uint32_t>(indexer_blocks),
                                            static_cast<std::uint32_t>(kIndexerBlockSize),
                                            static_cast<std::uint32_t>(options.indexer_seq_len), i_indices,
                                            i_workspace, indexer_workspace_bytes, stream);
      });
      std::vector<std::int32_t> indices(kIndexerTopK);
      DSV4_ACL_CHECK(aclrtMemcpy(indices.data(), kIndexerTopK * 4, i_indices, kIndexerTopK * 4,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
      int selected = 0;
      for (std::int32_t index : indices) {
        if (index >= 0) ++selected;
      }
      std::printf("    indexer: %d of top-%d selected over %d keys (synthetic Q/K, seed 0x%08X)\n", selected,
                  kIndexerTopK, options.indexer_seq_len, indexer_seed);
    }

    DSV4_ACL_CHECK(aclrtMemcpy(x_after2.data(), bundle * 2, x_ping, bundle * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // ===== layer 1: mHC round + MoE ========================================
    {
      mhc_prologue(mhc_moe, x_ping);

      const auto moe_start = Clock::now();
      graph.Launch(cast_h_norm, stream);
      // h_norm -> FP8 for the shared expert's w1/w3. Same arch35 quantizer as
      // the attention sites; the layer-2 AttentionPlan no longer owns a
      // quant_h stage to borrow.
      block_quant("bf16_block_quant_fp8: h_norm (shared)", h_norm, kHiddenSize, a_h_fp8, a_h_scale_f32);
      graph.Launch(router_matmul, stream);
      graph.Launch(router_cast, stream);
      graph.Launch(router_softplus, stream);
      graph.Launch(router_sqrt, stream);
      graph.Launch(router_gating, stream);
      DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
      // 6 aclnn enqueues: the h_norm quantizer and its scale cast are now one
      // arch35 launch with its own row (and overlap this span, as above).
      timeline.Add("router: hash route (aclnn)", MicrosecondsSince(moe_start), 6, false);

      // The device router's expert ids must agree with the hash row the host
      // staged from. A mismatch means the six staged weight matrices are not
      // the six the device selected, which would silently compute the wrong
      // experts -- so it is a hard failure, checked once rather than trusted.
      std::vector<std::int32_t> device_experts(kNumExpertsPerTok);
      DSV4_ACL_CHECK(aclrtMemcpy(device_experts.data(), kNumExpertsPerTok * 4, r_experts, kNumExpertsPerTok * 4,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
      std::vector<float> route_weights(kNumExpertsPerTok);
      DSV4_ACL_CHECK(aclrtMemcpy(route_weights.data(), kNumExpertsPerTok * 4, r_weights, kNumExpertsPerTok * 4,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
      for (int i = 0; i < kNumExpertsPerTok; ++i) {
        DSV4_REQUIRE(device_experts[static_cast<std::size_t>(i)] == host_experts[static_cast<std::size_t>(i)],
                     "aclnnMoeGatingTopKV2 selected expert "
                         << device_experts[static_cast<std::size_t>(i)] << " at slot " << i
                         << " but tid2eid[" << options.token_id << "] says "
                         << host_experts[static_cast<std::size_t>(i)]
                         << ". The staged expert weights would not match the device route.");
      }

      DSV4_ACL_CHECK(aclrtMemsetAsync(e_routed, static_cast<std::size_t>(kHiddenSize) * 4, 0,
                                      static_cast<std::size_t>(kHiddenSize) * 4, stream));

      // --- the six routed experts, MXFP4 ----------------------------------
      for (int i = 0; i < kNumExpertsPerTok; ++i) {
        const RoutedExpert& expert = routed_experts[static_cast<std::size_t>(i)];
        // One launch for the whole expert: gate GEMV, up GEMV, the model's
        // asymmetric clamped SwiGLU and the down GEMV. This replaces three
        // LaunchMxfp4ExpertGemv calls and the aclnn Clamp/Clamp/SwiGlu trio
        // that used to sit between them -- six dispatches and two stream
        // synchronizations per expert, down to one wrapper call.
        //
        // The clamp now lives inside the operator, where tiling carries its
        // bound, rather than in two aclnn stages whose argument order
        // encoded the asymmetry at the call site. The shared expert below
        // still uses the aclnn path because it is FP8, not MXFP4.
        TimedArch35(timeline, "dsv4_moe_expert", [&] {
          return arch35::LaunchDsv4MoeExpert(
              static_cast<const arch35::Mxfp4Pair*>(expert.w1.packed),
              static_cast<const arch35::E8M0*>(expert.w1.scale),
              static_cast<const arch35::Mxfp4Pair*>(expert.w3.packed),
              static_cast<const arch35::E8M0*>(expert.w3.scale),
              static_cast<const arch35::Mxfp4Pair*>(expert.w2.packed),
              static_cast<const arch35::E8M0*>(expert.w2.scale),
              static_cast<std::uint32_t>(kHiddenSize / 2),
              static_cast<std::uint32_t>(kHiddenSize / kMxfp4GroupSize),
              static_cast<std::uint32_t>(kMoeIntermediate / 2),
              static_cast<std::uint32_t>(kMoeIntermediate / kMxfp4GroupSize), h_norm_f32,
              static_cast<std::uint32_t>(kHiddenSize), static_cast<std::uint32_t>(kMoeIntermediate),
              static_cast<float*>(e_out), e_expert_workspace, e_expert_workspace_bytes, stream);
        });

        // routed_sum += route_weight[i] * expert_out. aclnnMul needs the
        // scalar as a tensor, so the weight is broadcast once per expert.
        std::vector<float> broadcast(kHiddenSize, route_weights[static_cast<std::size_t>(i)]);
        DSV4_ACL_CHECK(aclrtMemcpyAsync(e_weight_broadcast, static_cast<std::size_t>(kHiddenSize) * 4,
                                        broadcast.data(), static_cast<std::size_t>(kHiddenSize) * 4,
                                        ACL_MEMCPY_HOST_TO_DEVICE, stream));
        const auto combine_start = Clock::now();
        graph.Launch(expert_weight_mul, stream);
        graph.Launch(expert_accumulate, stream);
        DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
        timeline.Add("expert: weighted accumulate (aclnn)", MicrosecondsSince(combine_start), 2, false);
      }

      // --- the shared expert, FP8 ------------------------------------------
      const auto shared_start = Clock::now();
      graph.Launch(shared_gate, stream);
      graph.Launch(shared_up, stream);
      graph.Launch(shared_gate_cast, stream);
      graph.Launch(shared_up_cast, stream);
      graph.Launch(clamp_gate, stream);
      graph.Launch(clamp_up, stream);
      graph.Launch(expert_swiglu, stream);
      graph.Launch(shared_quant, stream);
      graph.Launch(shared_hidden_scale_cast, stream);
      graph.Launch(shared_down, stream);
      graph.Launch(routed_to_bf16, stream);
      graph.Launch(add_shared, stream);
      DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
      timeline.Add("shared expert + combine (aclnn)", MicrosecondsSince(shared_start), 12, false);

      mhc_epilogue(x_ping, x_pong);
    }

    DSV4_ACL_CHECK(aclrtSynchronizeStream(stream));
    const double total_microseconds = MicrosecondsSince(wall_start);

    std::vector<std::uint16_t> x_next(bundle);
    DSV4_ACL_CHECK(aclrtMemcpy(x_next.data(), bundle * 2, x_pong, bundle * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // =======================================================================
    // [3] stages
    // =======================================================================
    std::printf("\n[3] stages\n");
    for (const StageTiming& stage : timeline.stages()) {
      std::printf("  %12.1f us  %-34s %3d launch%-3s%s\n", stage.microseconds, stage.name.c_str(), stage.launches,
                  stage.launches == 1 ? "" : "es", stage.sync_inclusive ? " (sync-inclusive)" : "");
    }
    std::printf(
        "\n  sync-inclusive = an arch35 wrapper that synchronizes the stream inside\n"
        "  itself, so the figure is a host-observed round trip and NOT a kernel time.\n"
        "  The aclnn rows are groups bracketed by one explicit sync.\n");

    // =======================================================================
    // [4] activations
    // =======================================================================
    std::printf("\n[4] activations (L2 of the [1, 4, 4096] stream bundle)\n");
    std::printf("    after layer 0           %12.4f\n", L2Norm(x_after0));
    std::printf("    after layer 2           %12.4f\n", L2Norm(x_after2));
    std::printf("    after the MoE layer     %12.4f\n", L2Norm(x_next));

    bool all_finite = true;
    for (const std::vector<std::uint16_t>* stageptr : {&x_after0, &x_after2, &x_next}) {
      for (std::uint16_t raw : *stageptr) {
        if (!std::isfinite(Bf16ToFloat(raw))) all_finite = false;
      }
    }
    std::printf("    all values finite       %s\n", all_finite ? "yes" : "NO");

    bool parity_ok = true;
    if (options.golden) {
      const Golden golden = ReadGolden(golden_path);
      if (!golden.present) {
        std::printf("    golden parity skipped: %s not present (run the host bench to write it)\n",
                    golden_path.c_str());
      } else {
        DSV4_REQUIRE(golden.token_id == options.token_id,
                     "the golden was written for token " << golden.token_id << " but this run used "
                                                         << options.token_id);
        // TOLERANCE. Cosine only, at 0.99 rather than the host-to-host 0.9995.
        // The device path activation-quantizes every FP8 projection input to
        // E4M3, which keeps a 3-bit mantissa and so carries up to 2^-4
        // relative error per element before its block scale, while the host
        // reference dequantizes the weights and works in FP32. Layer 0 alone
        // puts five such projections in series. Relative-L2 is deliberately
        // NOT gated: quantization moves magnitude, and this bound exists to
        // catch a structural error -- a transposed weight, a wrong layout, a
        // mis-ordered stream -- which moves direction.
        struct Compare {
          const char* label;
          const std::vector<std::uint16_t>* got;
          const std::vector<std::uint16_t>* want;
        };
        const Compare comparisons[] = {
            {"x after layer 0", &x_after0, &golden.x_after0},
            {"x after layer 2", &x_after2, &golden.x_after2},
            {"x_next         ", &x_next, &golden.x_next},
        };
        for (const Compare& comparison : comparisons) {
          const double cosine = CosineSimilarity(*comparison.got, *comparison.want);
          const bool stage_ok = cosine > 0.99;
          parity_ok = parity_ok && stage_ok;
          std::printf("    %s cosine vs golden %10.6f  %s\n", comparison.label, cosine, stage_ok ? "OK" : "FAIL");
        }
      }
    }

    // =======================================================================
    // [5] budgets
    // =======================================================================
    std::printf("\n[5] budgets\n");
    const double milliseconds = total_microseconds / 1000.0;
    const bool time_ok = milliseconds < options.time_budget_ms;
    std::printf("  NPU wall clock        %10.1f ms   budget %7.1f ms   %s\n", milliseconds, options.time_budget_ms,
                time_ok ? "OK" : "EXCEEDED");
    std::printf("  peak HBM (arena)      %10.1f MB   reserved %7.1f MB\n",
                static_cast<double>(arena.used()) / 1048576.0, static_cast<double>(arena.capacity()) / 1048576.0);
    std::printf("  aclnn workspace       %10.1f KB   (high water, shared by all %zu stages)\n",
                static_cast<double>(graph.workspace_high_water()) / 1024.0, graph.size());
    std::printf("  aclrtMalloc in the timed region: %d\n", g_arch35_launches);
    std::printf(
        "    This bench allocates nothing after planning. The count above is the\n"
        "    arch35 launch layer's own per-call tiling staging buffer, which every\n"
        "    wrapper mallocs, copies into, synchronizes on and frees (launch.hpp,\n"
        "    \"STREAM AND TILING LIFETIME\"). Removing it needs a caller-owned\n"
        "    tiling variant, which CLAUDE.md section 3.7 gates.\n");

    // Repeatable executors must go before the runtime does.
    graph.Reset();

    const bool ok = all_finite && parity_ok && time_ok;
    std::printf("\n  semi-end-to-end multi-layer decode on NPU: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 6;
  });
}
