// ---------------------------------------------------------------------------
// WHAT THIS BINARY IS FOR, AND WHAT IT DELIBERATELY DOES NOT COVER
// ---------------------------------------------------------------------------
// An isolated smoke/parity harness for the VENDOR aclnn operators the decode
// pipeline drives: each case plans and launches one operator (or one short
// chain) on real hardware and compares the result against a host reference.
//
// It is not the device verification suite for this repo's own kernels. The
// five arch35 operators are covered by tests/npu's
// test_custom_kernels_npu_device through the Launch* wrappers, and nothing
// here duplicates that.
//
// Two cases were removed rather than fixed, because the operator binary is
// absent from the stock CANN 9.2 OPP for ascend950 and the plan dies with
// EZ1013 before any descriptor is read -- a packaging fact, not a defect this
// harness can express:
//   * aclnnHcPre                      (the fused mHC prologue; the staged
//                                      aclnnMhcPre/Sinkhorn/Post trio below
//                                      covers the same mapping)
//   * aclnnVllmQuantLightningIndexer  (the shared-KV indexer front end)
// The tiered selection network they would have exercised is validated on the
// host instead -- CLAUDE.md section 5.5 -- so removing them costs no coverage
// of the indexing arithmetic.
//
// Per CLAUDE.md section 4.5 every reference is generated in memory from a
// printed seed; there are no disk fixtures, so a stale asset cannot make a
// broken operator pass.
// ---------------------------------------------------------------------------

#include "moe/core/arch35_contract.hpp"
#include "moe/core/config.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/op_table.hpp"
#include "moe/emu/cpu_kernels.h"
#include "npu_test_support.hpp"

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ascend_moe;

constexpr int kN = moe::emu::kHcStreams;
constexpr int kD = moe::emu::kHiddenSize;
constexpr int kRows = moe::emu::kHcProjectionRows;
constexpr int kFlat = moe::emu::kHcFlatWidth;
constexpr float kHcEps = moe::emu::kHcEpsilon;
constexpr float kNormEps = moe::emu::kRmsNormEpsilon;
constexpr int kSinkhornIters = moe::emu::kHcSinkhornIters;
constexpr std::uint32_t kSeed = 0x5EEDu;

class Report {
 public:
  void Check(bool ok, const std::string& label, const std::string& detail = {}) {
    ++checks_;
    std::printf("  [%s] %s", ok ? "PASS" : "FAIL", label.c_str());
    if (!detail.empty()) std::printf(": %s", detail.c_str());
    std::putchar('\n');
    if (!ok) ++failures_;
  }

  int exit_code() const {
    std::printf("%d checks, %d failures\n", checks_, failures_);
    return failures_ == 0 ? 0 : 1;
  }

 private:
  int checks_ = 0;
  int failures_ = 0;
};

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t i = shape.size(); i-- > 1;) strides[i - 1] = strides[i] * shape[i];
  return strides;
}

size_t ElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (const int64_t dim : shape) count *= static_cast<size_t>(dim);
  return count;
}

size_t DTypeBytes(aclDataType dtype) {
  if (dtype == ACL_BF16 || dtype == ACL_FLOAT16) return 2;
  if (dtype == ACL_INT32 || dtype == ACL_FLOAT) return 4;
  throw std::invalid_argument("unsupported test tensor dtype");
}

class DeviceTensor {
 public:
  DeviceTensor(AclDeviceOps& device, std::vector<int64_t> shape, aclDataType dtype,
               const void* host_data = nullptr)
      : device_(device), shape_(std::move(shape)), strides_(ContiguousStrides(shape_)),
        bytes_(ElementCount(shape_) * DTypeBytes(dtype)) {
    storage_ = device_.DeviceMalloc(bytes_ == 0 ? 1 : bytes_);
    if (storage_ == nullptr) throw std::runtime_error("aclrtMalloc returned a null tensor buffer");
    handle_ = aclCreateTensor(shape_.data(), shape_.size(), dtype, strides_.data(), 0, ACL_FORMAT_ND,
                              shape_.data(), shape_.size(), storage_);
    if (handle_ == nullptr) {
      const char* recent = aclGetRecentErrMsg();
      const std::string error =
          std::string("aclCreateTensor failed") +
          (recent != nullptr && recent[0] != '\0' ? ": " + std::string(recent) : "");
      device_.DeviceFree(storage_);
      storage_ = nullptr;
      throw std::runtime_error(error);
    }
    try {
      if (host_data != nullptr && bytes_ != 0) {
        device_.MemcpySync(storage_, bytes_, host_data, bytes_, MemcpyKind::kHostToDevice);
      } else if (bytes_ != 0) {
        device_.DeviceMemset(storage_, bytes_, 0, bytes_);
      }
    } catch (...) {
      ReportAclCleanup("aclDestroyTensor", aclDestroyTensor(handle_));
      handle_ = nullptr;
      device_.DeviceFree(storage_);
      storage_ = nullptr;
      throw;
    }
  }

  ~DeviceTensor() {
    if (handle_ != nullptr) ReportAclCleanup("aclDestroyTensor", aclDestroyTensor(handle_));
    if (storage_ != nullptr) device_.DeviceFree(storage_);
  }

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  aclTensor* get() const { return handle_; }

  void Read(void* host, size_t bytes) const {
    if (bytes > bytes_) throw std::invalid_argument("tensor read exceeds its storage");
    // `bytes` is the transfer length, not bytes_: aclrtMemcpy's second argument
    // is the DESTINATION capacity, so passing the full tensor size as the count
    // is what would make a partial read overrun the host buffer.
    if (bytes != 0) device_.MemcpySync(host, bytes, storage_, bytes, MemcpyKind::kDeviceToHost);
  }

 private:
  AclDeviceOps& device_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  size_t bytes_ = 0;
  void* storage_ = nullptr;
  aclTensor* handle_ = nullptr;
};

class Workspace {
 public:
  Workspace(AclDeviceOps& device, uint64_t bytes) : device_(device) {
    if (bytes > 0) data_ = device_.DeviceMalloc(static_cast<size_t>(bytes));
    if (bytes > 0 && data_ == nullptr) throw std::runtime_error("failed to allocate ACLNN workspace");
  }
  ~Workspace() {
    if (data_ != nullptr) device_.DeviceFree(data_);
  }
  void* get() const { return data_; }

 private:
  AclDeviceOps& device_;
  void* data_ = nullptr;
};

using ScalarGuard = AclHandleGuard<aclScalar*, aclDestroyScalar>;

aclScalar* MakeFloatScalar(float value) {
  // aclCreateScalar copies the bytes, so a local is a legal source.
  aclScalar* scalar = aclCreateScalar(&value, ACL_FLOAT);
  if (scalar == nullptr) throw std::runtime_error("aclCreateScalar returned null");
  return scalar;
}

template <typename PlanFn, typename... Args>
void RunAclnn(const OpTable& ops, AclDeviceOps& device, DeviceStream stream, OpId op, Args... args) {
  aclOpExecutor* executor = nullptr;
  const uint64_t workspace_bytes = PlanAclnnOp<PlanFn>(ops, op, &executor, args...);
  OpExecutorGuard executor_guard(executor);
  Workspace workspace(device, workspace_bytes);
  const int status = ops.launch_fn(op)(workspace.get(), workspace_bytes, executor, stream);
  if (status != ACL_SUCCESS) throw AclError(OpName(op), __FILE__, __LINE__, status);
  (void)executor_guard.release();
  device.SynchronizeStream(stream);
}

struct Inputs {
  std::vector<uint16_t> x;
  std::vector<float> phi;
  float alpha[3] = {};
  float bias[kRows] = {};
  std::vector<uint16_t> h_out;
};

Inputs MakeInputs() {
  Inputs inputs;
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<float> activation(-1.0F, 1.0F);
  std::uniform_real_distribution<float> weight(-1.0F / std::sqrt(static_cast<float>(kFlat)),
                                                1.0F / std::sqrt(static_cast<float>(kFlat)));
  std::uniform_real_distribution<float> gain(0.1F, 1.0F);
  std::uniform_real_distribution<float> bias(-0.5F, 0.5F);
  inputs.x.resize(static_cast<size_t>(kN) * kD);
  for (auto& value : inputs.x) value = moe::emu::FloatToBf16(activation(rng));
  inputs.phi.resize(static_cast<size_t>(kRows) * kFlat);
  for (float& value : inputs.phi) value = weight(rng);
  for (float& value : inputs.alpha) value = gain(rng);
  for (float& value : inputs.bias) value = bias(rng);
  inputs.h_out.resize(kD);
  for (auto& value : inputs.h_out) value = moe::emu::FloatToBf16(activation(rng));
  return inputs;
}

bool AllClose(const float* actual, const float* expected, size_t count, double atol, double rtol,
              double* worst_abs) {
  *worst_abs = 0.0;
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) return false;
    const double error = std::fabs(static_cast<double>(actual[i]) - expected[i]);
    *worst_abs = std::max(*worst_abs, error);
    if (error > atol + rtol * std::fabs(static_cast<double>(expected[i]))) return false;
  }
  return true;
}

bool CheckBf16(Report& report, const std::string& label, const std::vector<uint16_t>& actual,
               const float* expected, size_t count) {
  std::vector<float> actual_f32(count);
  std::vector<float> rounded_expected(count);
  for (size_t i = 0; i < count; ++i) {
    actual_f32[i] = moe::emu::Bf16ToFloat(actual[i]);
    rounded_expected[i] = moe::emu::Bf16ToFloat(moe::emu::FloatToBf16(expected[i]));
  }
  double worst = 0.0;
  // BF16 unit roundoff is 2^-8; the small additional margin covers FP32
  // reduction-order differences around zero.
  const bool ok = AllClose(actual_f32.data(), rounded_expected.data(), count, 1e-2, 1e-2, &worst);
  char detail[96];
  std::snprintf(detail, sizeof(detail), "max abs error %.3e (atol=1e-2, rtol=1e-2)", worst);
  report.Check(ok, label, detail);
  return ok;
}

std::string DescribeInt32(const std::vector<int32_t>& values) {
  std::string text = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) text += ", ";
    text += std::to_string(values[i]);
  }
  return text + "]";
}

// ---------------------------------------------------------------------------
// aclnnMhcPre -> aclnnMhcSinkhorn -> aclnnMhcPost
// ---------------------------------------------------------------------------

void TestMhcChain(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops,
                  const Inputs& inputs, const moe::emu::MhcSinkhornTrace& trace) {
  DeviceTensor x(device, {1, kN, kD}, ACL_BF16, inputs.x.data());
  DeviceTensor phi(device, {kRows, kFlat}, ACL_FLOAT, inputs.phi.data());
  DeviceTensor alpha(device, {3}, ACL_FLOAT, inputs.alpha);
  DeviceTensor bias(device, {kRows}, ACL_FLOAT, inputs.bias);
  // gamma is OPTIONAL in MhcPre's op definition, but the vendored aclnn
  // wrapper hands it to INFER_SHAPE unconditionally and a null one is rejected
  // with 161001 (EZ1001, "Expected a proper Tensor but got null"). So it has to
  // be bound -- and bound the way the wrapper's own checks demand, which is
  // 2-D [N, D] and FP32: CheckInputOutDims refuses any rank but 2 and
  // CheckDtypeValid refuses BF16, so a 1-D BF16 gain vector would only trade
  // the null error for 161002.
  //
  // All ones is exactly the identity the CPU golden assumes, so the reference
  // stays valid: VFDoV0ProcessXIn's hasGamma=false branch is xMul = x and its
  // hasGamma=true branch is xMul = gamma * x.
  const std::vector<float> gamma_ones(static_cast<size_t>(kN) * kD, 1.0F);
  DeviceTensor gamma(device, {kN, kD}, ACL_FLOAT, gamma_ones.data());
  DeviceTensor h_in(device, {1, kD}, ACL_BF16);
  DeviceTensor h_post(device, {1, kN}, ACL_FLOAT);
  DeviceTensor h_res(device, {1, kN, kN}, ACL_FLOAT);

  RunAclnn<MhcPrePlanFn>(ops, device, stream, OpId::kMhcPre, x.get(), phi.get(), alpha.get(), bias.get(),
                         gamma.get(), static_cast<double>(kNormEps), static_cast<double>(kHcEps),
                         h_in.get(), h_post.get(), h_res.get(), static_cast<aclTensor*>(nullptr),
                         static_cast<aclTensor*>(nullptr), static_cast<aclTensor*>(nullptr));

  std::vector<uint16_t> actual_h_in(kD);
  std::vector<float> actual_h_post(kN), actual_h_res(kN * kN);
  h_in.Read(actual_h_in.data(), actual_h_in.size() * sizeof(uint16_t));
  h_post.Read(actual_h_post.data(), actual_h_post.size() * sizeof(float));
  h_res.Read(actual_h_res.data(), actual_h_res.size() * sizeof(float));
  CheckBf16(report, "aclnnMhcPre hIn matches CPU golden", actual_h_in, trace.h_in, kD);
  double worst = 0.0;
  bool ok = AllClose(actual_h_post.data(), trace.h_post, kN, 1e-5, 1e-4, &worst);
  // FP32's unit roundoff is ~6e-8; these bounds leave headroom for reduction
  // ordering and the nonlinear gate while staying well below BF16 precision.
  report.Check(ok, "aclnnMhcPre hPost matches CPU golden",
               "max abs error " + std::to_string(worst) + " (atol=1e-5, rtol=1e-4)");
  ok = AllClose(actual_h_res.data(), trace.residual_logits, kN * kN, 1e-5, 1e-4, &worst);
  report.Check(ok, "aclnnMhcPre unnormalized hRes matches CPU golden",
               "max abs error " + std::to_string(worst) + " (atol=1e-5, rtol=1e-4)");

  DeviceTensor sinkhorn_out(device, {1, kN, kN}, ACL_FLOAT);
  RunAclnn<MhcSinkhornPlanFn>(ops, device, stream, OpId::kMhcSinkhorn, h_res.get(), kHcEps,
                              static_cast<int64_t>(kSinkhornIters), sinkhorn_out.get(),
                              static_cast<aclTensor*>(nullptr), static_cast<aclTensor*>(nullptr));
  std::vector<float> actual_birkhoff(kN * kN);
  sinkhorn_out.Read(actual_birkhoff.data(), actual_birkhoff.size() * sizeof(float));
  ok = AllClose(actual_birkhoff.data(), trace.birkhoff, kN * kN, 2e-5, 2e-4, &worst);
  // Twenty row/column passes accumulate FP32 rounding at each normalization.
  report.Check(ok, "aclnnMhcSinkhorn matches CPU golden",
               "max abs error " + std::to_string(worst) + " (atol=2e-5, rtol=2e-4)");

  DeviceTensor h_out(device, {1, kD}, ACL_BF16, inputs.h_out.data());
  DeviceTensor x_next(device, {1, kN, kD}, ACL_BF16);
  RunAclnn<MhcPostPlanFn>(ops, device, stream, OpId::kMhcPost, x.get(), sinkhorn_out.get(), h_out.get(),
                          h_post.get(), x_next.get());
  std::vector<uint16_t> expected_x_next(static_cast<size_t>(kN) * kD);
  std::vector<uint16_t> actual_x_next(expected_x_next.size());
  moe::emu::mhc_post_cpu(inputs.x.data(), trace.birkhoff, trace.h_post, inputs.h_out.data(),
                         expected_x_next.data());
  std::vector<float> expected_x_next_f32(expected_x_next.size());
  for (size_t i = 0; i < expected_x_next.size(); ++i) {
    expected_x_next_f32[i] = moe::emu::Bf16ToFloat(expected_x_next[i]);
  }
  x_next.Read(actual_x_next.data(), actual_x_next.size() * sizeof(uint16_t));
  CheckBf16(report, "aclnnMhcPost matches CPU golden", actual_x_next, expected_x_next_f32.data(),
            expected_x_next.size());
}

// ---------------------------------------------------------------------------
// aclnnCompressor (C4A, cmp_ratio 4)
// ---------------------------------------------------------------------------

void TestCompressor(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops) {
  constexpr int ratio = moe::emu::kCompressorRatio;          // cmp_ratio, 4 == CSA
  constexpr int hidden = moe::emu::kHiddenSize;              // H
  constexpr int dim = moe::emu::kCompressorHeadDim;          // D, and normWeight[0]
  constexpr int coff = moe::emu::kCompressorCoff;            // 2
  constexpr int channels = moe::emu::kCompressorStateWidth;  // coff*D: the wkv/wgate row count
  // The paged state cache carries BOTH the KV and the score channels, so its
  // last axis is 2*coff*D -- CompressorTiling::CheckShapeConsistency checks
  // that literally ("stateCache ... 2*coff*headDim"), and stateCacheStrideDim0
  // must be the axis-0 stride of the contiguous view the caller owns.
  constexpr int state_dim = 2 * channels;
  constexpr int state_blocks = 4;
  constexpr int state_block_size = 8;
  constexpr int64_t state_stride_dim0 = static_cast<int64_t>(state_block_size) * state_dim;
  // cacheMode 2 is CACHE_MODE::CYCLE (compressor_tiling.h); in that mode the
  // tiler does not tie stateCache's leading axes to blockNum/blockSize.
  constexpr int64_t cache_mode_cycle = 2;
  // TH reserves min(T, T/ratio+B) rows; for one complete window and B=1 that is
  // two, of which only the first is emitted.
  constexpr int out_rows = static_cast<int>(kCompressorWindowOutputRows);

  std::mt19937 rng(kSeed + 2);
  std::uniform_real_distribution<float> activation(-0.5F, 0.5F);
  std::uniform_real_distribution<float> projection_weight(-0.02F, 0.02F);
  std::uniform_real_distribution<float> positional_bias(-0.1F, 0.1F);

  std::vector<uint16_t> x(static_cast<size_t>(ratio) * hidden);
  std::vector<uint16_t> wkv(static_cast<size_t>(channels) * hidden);
  std::vector<uint16_t> wgate(wkv.size());
  // arch35 takes an FP32 norm gain and FP32 RoPE tables -- the aclnn wrapper
  // rejects BF16 for all three outright ("arch35 requires FP32 norm/RoPE and
  // rotary/cache/coff in {1,2}") -- so there is no quantization step for the
  // oracle to mirror and both sides read the same FP32 values. The golden's
  // gain argument is BF16 by signature; all ones is exact in both formats, so
  // it is the same gain.
  std::vector<uint16_t> norm_bf16(dim, moe::emu::FloatToBf16(1.0F));
  std::vector<float> norm_fp32(dim, 1.0F);
  std::vector<float> ape(static_cast<size_t>(ratio) * channels);
  std::vector<float> rope_sin(static_cast<size_t>(out_rows) * moe::emu::kRopeHeadDim);
  std::vector<float> rope_cos(rope_sin.size());
  for (auto& value : x) value = moe::emu::FloatToBf16(activation(rng));
  for (auto& value : wkv) value = moe::emu::FloatToBf16(projection_weight(rng));
  for (auto& value : wgate) value = moe::emu::FloatToBf16(projection_weight(rng));
  for (float& value : ape) value = positional_bias(rng);
  for (size_t i = 0; i < rope_sin.size(); ++i) {
    const float angle = static_cast<float>(i) * 0.01F;
    rope_sin[i] = std::sin(angle);
    rope_cos[i] = std::cos(angle);
  }

  moe::emu::CompressorRingState emu_state;
  std::vector<uint16_t> expected(dim, 0);
  for (int step = 0; step < ratio; ++step) {
    const auto result = moe::emu::paged_compressor_decode_c4a(
        x.data() + static_cast<size_t>(step) * hidden, wkv.data(), wgate.data(), ape.data(),
        norm_bf16.data(), step, &emu_state, rope_cos.data(), rope_sin.data(), expected.data(), hidden,
        dim, ratio, moe::emu::kRmsNormEpsilon, moe::emu::RotaryMode::kHalf);
    if (step + 1 == ratio && !result.emitted) {
      throw std::runtime_error("CPU compressor golden did not emit at the end of a complete window");
    }
  }
  std::vector<float> expected_f32(dim);
  for (size_t i = 0; i < expected.size(); ++i) expected_f32[i] = moe::emu::Bf16ToFloat(expected[i]);

  const std::array<int32_t, state_blocks> block_ids = {0, 1, 2, 3};
  const std::array<int32_t, 2> cu_seqlens = {0, ratio};
  const int32_t used_rows = ratio;
  const int32_t start_pos = 0;
  DeviceTensor d_x(device, {ratio, hidden}, ACL_BF16, x.data());
  DeviceTensor d_wkv(device, {channels, hidden}, ACL_BF16, wkv.data());
  DeviceTensor d_wgate(device, {channels, hidden}, ACL_BF16, wgate.data());
  DeviceTensor d_state(device, {state_blocks, state_block_size, state_dim}, ACL_FLOAT);
  DeviceTensor d_ape(device, {ratio, channels}, ACL_FLOAT, ape.data());
  DeviceTensor d_norm(device, {dim}, ACL_FLOAT, norm_fp32.data());
  DeviceTensor d_sin(device, {out_rows, moe::emu::kRopeHeadDim}, ACL_FLOAT, rope_sin.data());
  DeviceTensor d_cos(device, {out_rows, moe::emu::kRopeHeadDim}, ACL_FLOAT, rope_cos.data());
  DeviceTensor d_blocks(device, {1, state_blocks}, ACL_INT32, block_ids.data());
  DeviceTensor d_cu(device, {2}, ACL_INT32, cu_seqlens.data());
  DeviceTensor d_used(device, {1}, ACL_INT32, &used_rows);
  DeviceTensor d_start(device, {1}, ACL_INT32, &start_pos);
  DeviceTensor d_output(device, {out_rows, dim}, ACL_BF16);

  RunAclnn<CompressorPlanFn>(
      ops, device, stream, OpId::kCompressor, d_x.get(), d_wkv.get(), d_wgate.get(), d_state.get(),
      d_ape.get(), d_norm.get(), d_sin.get(), d_cos.get(), d_blocks.get(), d_cu.get(), d_used.get(),
      d_start.get(), static_cast<int64_t>(moe::emu::kRopeHeadDim), static_cast<int64_t>(ratio),
      static_cast<int64_t>(coff), static_cast<double>(moe::emu::kRmsNormEpsilon),
      static_cast<int64_t>(moe::emu::RotaryMode::kHalf), cache_mode_cycle, state_stride_dim0,
      d_output.get());
  std::vector<uint16_t> actual(static_cast<size_t>(out_rows) * dim);
  d_output.Read(actual.data(), actual.size() * sizeof(uint16_t));
  std::vector<uint16_t> actual_first(actual.begin(), actual.begin() + dim);
  CheckBf16(report, "aclnnCompressor first emitted row matches CPU golden", actual_first,
            expected_f32.data(), dim);
}

// ---------------------------------------------------------------------------
// aclnnSoftplus -> aclnnSqrt: DSV4's `scoring_func = "sqrtsoftplus"`
// ---------------------------------------------------------------------------

void TestRouterScoring(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops) {
  constexpr int experts = static_cast<int>(kNumRoutedExperts);
  const float beta = static_cast<float>(kSoftplusBeta);
  const float threshold = static_cast<float>(kSoftplusThreshold);

  std::mt19937 rng(kSeed + 3);
  // Well inside the threshold, so the comparison exercises the log1p branch the
  // router actually runs in; the reference below still models the linear branch,
  // which is part of the operator's contract.
  std::uniform_real_distribution<float> logit(-8.0F, 8.0F);
  std::vector<float> logits(experts);
  for (float& value : logits) value = logit(rng);

  DeviceTensor d_logits(device, {1, experts}, ACL_FLOAT, logits.data());
  DeviceTensor d_softplus(device, {1, experts}, ACL_FLOAT);
  DeviceTensor d_scores(device, {1, experts}, ACL_FLOAT);
  ScalarGuard d_beta(MakeFloatScalar(beta));
  ScalarGuard d_threshold(MakeFloatScalar(threshold));

  RunAclnn<SoftplusPlanFn>(ops, device, stream, OpId::kSoftplus, d_logits.get(), d_beta.get(),
                           d_threshold.get(), d_softplus.get());
  RunAclnn<UnaryPlanFn>(ops, device, stream, OpId::kSqrt, d_softplus.get(), d_scores.get());

  std::vector<float> expected(experts);
  for (int i = 0; i < experts; ++i) {
    const float logit_value = logits[static_cast<size_t>(i)];
    const float scaled = beta * logit_value;
    const float softplus = scaled > threshold ? logit_value : std::log1p(std::exp(scaled)) / beta;
    expected[static_cast<size_t>(i)] = std::sqrt(softplus);
  }
  std::vector<float> actual(experts);
  d_scores.Read(actual.data(), actual.size() * sizeof(float));
  double worst = 0.0;
  // FP32 keeps a 24-bit mantissa (unit roundoff 6e-8) and this chain rounds
  // three times -- exp, log1p, sqrt -- through device approximations that need
  // not be the host libm's polynomials. Over logits in [-8, 8] the outputs span
  // [0.018, 2.83], so atol=1e-5 is the binding bound at the small end (~5e-4
  // relative) and rtol=1e-4 (~400 ulp) at the large end: loose enough for a
  // different transcendental implementation, three decades tighter than a BF16
  // round-trip would need.
  const bool ok = AllClose(actual.data(), expected.data(), experts, 1e-5, 1e-4, &worst);
  report.Check(ok, "aclnnSoftplus -> aclnnSqrt matches the host sqrtsoftplus",
               "max abs error " + std::to_string(worst) + " (atol=1e-5, rtol=1e-4)");

  // Non-degeneracy on the REFERENCE, per CLAUDE.md section 4.8: a constant or
  // all-zero golden would make the comparison above vacuous.
  float lowest = expected[0];
  float highest = expected[0];
  for (const float value : expected) {
    lowest = std::min(lowest, value);
    highest = std::max(highest, value);
  }
  report.Check(lowest > 0.0F && highest - lowest > 1.0F,
               "the sqrtsoftplus golden is strictly positive and has real spread",
               "range [" + std::to_string(lowest) + ", " + std::to_string(highest) + "]");
}

// ---------------------------------------------------------------------------
// aclnnMoeGatingTopKV2, hash branch
// ---------------------------------------------------------------------------

void TestGatingTopKHash(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops) {
  constexpr int experts = static_cast<int>(kNumRoutedExperts);
  constexpr int top_k = static_cast<int>(kNumExpertsPerTok);
  constexpr int table_rows = 16;
  constexpr int32_t token_id = 0;

  std::mt19937 rng(kSeed + 4);
  // Strictly positive, like the sqrtsoftplus scores this stage consumes; an L1
  // renormalization over signed scores would not be meaningful.
  std::uniform_real_distribution<float> score(0.25F, 2.0F);
  std::vector<float> scores(experts);
  for (float& value : scores) value = score(rng);

  // tid2eid is the hash branch's membership table: row `token_id` names the
  // token's experts outright, so selection does not consult the scores at all
  // and the scores only set the weights. The other rows hold a different set,
  // so a kernel that indexed the wrong row would not accidentally agree.
  const std::array<int32_t, top_k> membership = {3, 17, 64, 128, 200, 255};
  std::vector<int32_t> tid2eid(static_cast<size_t>(table_rows) * top_k);
  std::copy(membership.begin(), membership.end(), tid2eid.begin());
  for (int row = 1; row < table_rows; ++row) {
    for (int slot = 0; slot < top_k; ++slot) {
      tid2eid[static_cast<size_t>(row) * top_k + slot] = (row * top_k + slot) % experts;
    }
  }

  DeviceTensor d_scores(device, {1, experts}, ACL_FLOAT, scores.data());
  DeviceTensor d_input_ids(device, {1}, ACL_INT32, &token_id);
  DeviceTensor d_tid2eid(device, {table_rows, top_k}, ACL_INT32, tid2eid.data());
  DeviceTensor d_weights(device, {1, top_k}, ACL_FLOAT);
  DeviceTensor d_experts(device, {1, top_k}, ACL_INT32);
  // outOut is NOT optional even with outFlag false: the wrapper's check is
  // CheckNotNull(x, yOut, expertIdxOut, outOut) and runs before the flag is
  // read, so a null one fails the plan with 161001 either way. The ops-info
  // declares it FP32 with x's shape and the tiling reads "out dim[0]" / "out
  // dim[1]", so it is the normalized full-score row [1, 256] -- not the top-k
  // width. Commit 33cf49f bound it on the bench for exactly this reason.
  DeviceTensor d_norm_scores(device, {1, experts}, ACL_FLOAT);

  RunAclnn<MoeGatingTopKV2PlanFn>(
      ops, device, stream, OpId::kMoeGatingTopKV2, d_scores.get(), static_cast<const aclTensor*>(nullptr),
      d_input_ids.get(), d_tid2eid.get(), static_cast<int64_t>(top_k), kGatingKGroup, kGatingGroupCount,
      kGatingGroupSelectMode, kGatingRenormL1, kGatingNormTypePreNormalized, false, kRoutedScalingFactor,
      kGatingEps, d_weights.get(), d_experts.get(), d_norm_scores.get());

  std::vector<int32_t> actual_experts(top_k);
  std::vector<float> actual_weights(top_k);
  d_experts.Read(actual_experts.data(), actual_experts.size() * sizeof(int32_t));
  d_weights.Read(actual_weights.data(), actual_weights.size() * sizeof(float));

  bool in_range = true;
  for (const int32_t expert : actual_experts) {
    if (expert < 0 || expert >= experts) in_range = false;
  }
  report.Check(in_range, "aclnnMoeGatingTopKV2 expert indices are inside [0, 256)",
               DescribeInt32(actual_experts));

  // Membership is the documented contract; the slot ORDER inside the row is
  // not, so the SET is compared and each weight is then checked against the
  // expert its own index names. That is strictly stronger than assuming an
  // order: a permutation passes, a wrong expert does not.
  std::vector<int32_t> want_sorted(membership.begin(), membership.end());
  std::vector<int32_t> got_sorted(actual_experts);
  std::sort(want_sorted.begin(), want_sorted.end());
  std::sort(got_sorted.begin(), got_sorted.end());
  report.Check(want_sorted == got_sorted, "aclnnMoeGatingTopKV2 selects tid2eid[token_id]",
               "want " + DescribeInt32(want_sorted) + ", got " + DescribeInt32(got_sorted));

  // renorm=L1 divides the selected scores by their sum (denominator guarded by
  // eps) before routedScalingFactor is applied.
  double denominator = 0.0;
  for (const int32_t expert : membership) denominator += scores[static_cast<size_t>(expert)];
  denominator = std::max(denominator, kGatingEps);
  std::vector<float> expected(top_k);
  for (int i = 0; i < top_k; ++i) {
    const int32_t expert = actual_experts[static_cast<size_t>(i)];
    // An out-of-range index has no reference weight; NaN makes AllClose's
    // finiteness test fail the check rather than silently skipping it.
    expected[static_cast<size_t>(i)] =
        (expert >= 0 && expert < experts)
            ? static_cast<float>(scores[static_cast<size_t>(expert)] / denominator * kRoutedScalingFactor)
            : std::numeric_limits<float>::quiet_NaN();
  }
  double worst = 0.0;
  // One FP32 reciprocal and one multiply over O(1) values is about two ulp
  // (1.2e-7 relative); the six-term denominator may be summed in a different
  // order on device, worth a few more. atol=1e-6 / rtol=1e-5 is two decades
  // above that floor and still far tighter than the format's own limit.
  const bool ok = AllClose(actual_weights.data(), expected.data(), top_k, 1e-6, 1e-5, &worst);
  report.Check(ok, "aclnnMoeGatingTopKV2 weights are L1-normalized scores scaled by 1.5",
               "max abs error " + std::to_string(worst) + " (atol=1e-6, rtol=1e-5)");
}

// ---------------------------------------------------------------------------
// aclnnMatmul with a contiguous row-major B
// ---------------------------------------------------------------------------

void TestDenseMatmulContiguousB(Report& report, AclDeviceOps& device, DeviceStream stream,
                                const OpTable& ops) {
  constexpr int m = 1;
  constexpr int k = static_cast<int>(kHiddenSize);        // 4096
  constexpr int n = static_cast<int>(kNumRoutedExperts);  // 256, the router gate's shape

  std::mt19937 rng(kSeed + 5);
  std::uniform_real_distribution<float> activation(-1.0F, 1.0F);
  // B scaled by 1/sqrt(K) so the 4096-term reduction lands at O(1) instead of
  // somewhere the comparison would be dominated by absolute tolerance.
  std::uniform_real_distribution<float> weight(-1.0F / std::sqrt(static_cast<float>(k)),
                                                1.0F / std::sqrt(static_cast<float>(k)));
  std::vector<uint16_t> a(static_cast<size_t>(m) * k);
  std::vector<uint16_t> b(static_cast<size_t>(k) * n);
  for (auto& value : a) value = moe::emu::FloatToBf16(activation(rng));
  for (auto& value : b) value = moe::emu::FloatToBf16(weight(rng));

  // B is [K, N] with contiguous row-major strides {N, 1}. aclnnMatmul carries
  // no transpose attribute, so ValidateDenseMatmul refuses a transposed view of
  // an [N, K] checkpoint weight outright -- which is why the bench repacks in
  // host staging. DeviceTensor only ever builds contiguous strides, so this
  // case is the positive half of that rule.
  DeviceTensor d_a(device, {m, k}, ACL_BF16, a.data());
  DeviceTensor d_b(device, {k, n}, ACL_BF16, b.data());
  DeviceTensor d_c(device, {m, n}, ACL_BF16);

  RunAclnn<MatmulPlanFn>(ops, device, stream, OpId::kMatmul, d_a.get(), d_b.get(), d_c.get(),
                         kCubeMathTypeKeepDtype);
  report.Check(true, "aclnnMatmul plans and launches [1,4096] x [4096,256] past the dense contract",
               "homogeneous BF16, ND, strides {256, 1}, cubeMathType=KEEP_DTYPE");

  // "It did not abort" would hold just as well for an all-zero output, and a B
  // read in the wrong orientation is the exact defect the stride rule exists to
  // catch -- so the row is compared against the host reduction too.
  std::vector<float> expected(static_cast<size_t>(m) * n);
  for (int col = 0; col < n; ++col) {
    float sum = 0.0F;
    for (int i = 0; i < k; ++i) {
      sum += moe::emu::Bf16ToFloat(a[static_cast<size_t>(i)]) *
             moe::emu::Bf16ToFloat(b[static_cast<size_t>(i) * n + col]);
    }
    expected[static_cast<size_t>(col)] = sum;
  }
  std::vector<uint16_t> actual(static_cast<size_t>(m) * n);
  d_c.Read(actual.data(), actual.size() * sizeof(uint16_t));
  CheckBf16(report, "aclnnMatmul row matches the host [M,K] x [K,N] reduction", actual, expected.data(),
            expected.size());
}

template <typename Fn>
void RunCase(Report& report, const char* name, Fn&& body) {
  try {
    std::printf("\n== %s ==\n", name);
    body();
  } catch (const std::exception& error) {
    report.Check(false, name, error.what());
  }
}

}  // namespace

int main() {
  std::printf("aclnn parity against tests/emu CPU goldens\n");
  std::printf("  base seed 0x%X; cases draw from base+0 (mHC), +2 (compressor), +3 (router scoring),\n"
              "  +4 (gating top-k), +5 (dense matmul)\n",
              kSeed);
  if (!PhysicalNpuPresent()) return 0;

  return GuardedMain([&]() {
    OpTable ops;
    ops.RequireAll({OpId::kMhcPre, OpId::kMhcSinkhorn, OpId::kMhcPost, OpId::kCompressor,
                    OpId::kSoftplus, OpId::kSqrt, OpId::kMoeGatingTopKV2, OpId::kMatmul});
    AclDeviceOps device(0);
    const DeviceStream stream = device.CreateStream();
    Report report;
    const Inputs inputs = MakeInputs();
    moe::emu::MhcSinkhornTrace trace;
    moe::emu::mhc_pre_sinkhorn_cpu(inputs.x.data(), inputs.phi.data(), inputs.alpha, inputs.bias, &trace);

    RunCase(report, "mHC split chain", [&] {
      TestMhcChain(report, device, stream, ops, inputs, trace);
    });
    RunCase(report, "C4A compressor", [&] {
      TestCompressor(report, device, stream, ops);
    });
    RunCase(report, "router scoring (softplus -> sqrt)", [&] {
      TestRouterScoring(report, device, stream, ops);
    });
    RunCase(report, "MoeGatingTopKV2 hash branch", [&] {
      TestGatingTopKHash(report, device, stream, ops);
    });
    RunCase(report, "dense matmul, contiguous B", [&] {
      TestDenseMatmulContiguousB(report, device, stream, ops);
    });
    device.DestroyStream(stream);
    return report.exit_code();
  });
}
