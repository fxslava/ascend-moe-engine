#include "moe/core/arch35_contract.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/op_table.hpp"
#include "moe/emu/cpu_kernels.h"
#include "moe/emu/fp8_types.hpp"
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
  if (dtype == ACL_FLOAT8_E4M3FN) return 1;
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
    if (bytes != 0) device_.MemcpySync(host, bytes, storage_, bytes_, MemcpyKind::kDeviceToHost);
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

void TestMhcChain(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops,
                  const Inputs& inputs, const moe::emu::MhcSinkhornTrace& trace) {
  DeviceTensor x(device, {1, kN, kD}, ACL_BF16, inputs.x.data());
  DeviceTensor phi(device, {kRows, kFlat}, ACL_FLOAT, inputs.phi.data());
  DeviceTensor alpha(device, {3}, ACL_FLOAT, inputs.alpha);
  DeviceTensor bias(device, {kRows}, ACL_FLOAT, inputs.bias);
  DeviceTensor h_in(device, {1, kD}, ACL_BF16);
  DeviceTensor h_post(device, {1, kN}, ACL_FLOAT);
  DeviceTensor h_res(device, {1, kN, kN}, ACL_FLOAT);

  RunAclnn<MhcPrePlanFn>(ops, device, stream, OpId::kMhcPre, x.get(), phi.get(), alpha.get(), bias.get(),
                         static_cast<const aclTensor*>(nullptr), static_cast<double>(kNormEps),
                         static_cast<double>(kHcEps), h_in.get(), h_post.get(), h_res.get(),
                         static_cast<aclTensor*>(nullptr), static_cast<aclTensor*>(nullptr),
                         static_cast<aclTensor*>(nullptr));

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

void TestFusedMhc(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops,
                  const Inputs& inputs, const moe::emu::MhcSinkhornTrace& trace) {
  DeviceTensor x(device, {1, kN, kD}, ACL_BF16, inputs.x.data());
  DeviceTensor phi(device, {kRows, kFlat}, ACL_FLOAT, inputs.phi.data());
  DeviceTensor alpha(device, {3}, ACL_FLOAT, inputs.alpha);
  DeviceTensor bias(device, {kRows}, ACL_FLOAT, inputs.bias);
  DeviceTensor y(device, {1, kD}, ACL_BF16);
  DeviceTensor post(device, {1, kN}, ACL_FLOAT);
  DeviceTensor comb_frag(device, {1, kN, kN}, ACL_FLOAT);

  RunAclnn<HcPrePlanFn>(ops, device, stream, OpId::kHcPre, x.get(), phi.get(), alpha.get(), bias.get(),
                        static_cast<int64_t>(kN), static_cast<int64_t>(kSinkhornIters),
                        static_cast<double>(kHcEps), static_cast<double>(kNormEps), y.get(),
                        post.get(), comb_frag.get());
  std::vector<uint16_t> actual_y(kD);
  std::vector<float> actual_post(kN), actual_comb_frag(kN * kN);
  y.Read(actual_y.data(), actual_y.size() * sizeof(uint16_t));
  post.Read(actual_post.data(), actual_post.size() * sizeof(float));
  comb_frag.Read(actual_comb_frag.data(), actual_comb_frag.size() * sizeof(float));
  CheckBf16(report, "aclnnHcPre y matches CPU golden", actual_y, trace.h_in, kD);
  double worst = 0.0;
  bool ok = AllClose(actual_post.data(), trace.h_post, kN, 1e-5, 1e-4, &worst);
  report.Check(ok, "aclnnHcPre post matches CPU golden",
               "max abs error " + std::to_string(worst) + " (atol=1e-5, rtol=1e-4)");
  ok = AllClose(actual_comb_frag.data(), trace.birkhoff, kN * kN, 2e-5, 2e-4, &worst);
  report.Check(ok, "aclnnHcPre combFrag matches CPU golden",
               "max abs error " + std::to_string(worst) + " (atol=2e-5, rtol=2e-4)");
}

void TestFp8Indexer(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops) {
  constexpr int heads = moe::emu::kIndexerHeads;
  constexpr int head_dim = moe::emu::kIndexerHeadDim;
  constexpr int blocks_count = 8;
  constexpr int block_size = 128;
  constexpr int seq_len = blocks_count * block_size;
  constexpr int top_k = 512;
  std::mt19937 rng(kSeed + 1);
  std::uniform_real_distribution<float> fp8_value(-1.0F, 1.0F);
  std::uniform_real_distribution<float> positive(0.75F, 1.25F);

  std::vector<uint8_t> query(static_cast<size_t>(heads) * head_dim);
  std::vector<uint8_t> key(static_cast<size_t>(seq_len) * head_dim);
  for (uint8_t& value : query) value = moe::emu::EncodeFp8E4M3(fp8_value(rng));
  for (uint8_t& value : key) value = moe::emu::EncodeFp8E4M3(fp8_value(rng));
  std::vector<float> weights(heads), query_scale(heads), key_scale(seq_len);
  for (float& value : weights) value = positive(rng);
  for (float& value : query_scale) value = positive(rng);
  for (float& value : key_scale) value = positive(rng);
  std::vector<float> emu_scales(static_cast<size_t>(heads) + seq_len);
  for (int h = 0; h < heads; ++h) emu_scales[h] = weights[h] * query_scale[h];
  std::copy(key_scale.begin(), key_scale.end(), emu_scales.begin() + heads);
  std::vector<int32_t> block_table(blocks_count);
  for (int i = 0; i < blocks_count; ++i) block_table[i] = i;
  const auto metadata = DecodeIndexerMetadata();
  std::vector<int32_t> expected(top_k), actual(top_k, -1);
  moe::emu::fast_fp8_indexer_topk_cpu(query.data(), key.data(), emu_scales.data(), block_table.data(),
                                      seq_len, top_k, expected.data(), block_size, heads, head_dim);

  DeviceTensor d_query(device, {1, 1, heads, head_dim}, ACL_FLOAT8_E4M3FN, query.data());
  DeviceTensor d_key(device, {blocks_count, block_size, 1, head_dim}, ACL_FLOAT8_E4M3FN, key.data());
  DeviceTensor d_weights(device, {1, 1, heads}, ACL_FLOAT, weights.data());
  DeviceTensor d_query_scale(device, {1, 1, heads}, ACL_FLOAT, query_scale.data());
  DeviceTensor d_key_scale(device, {blocks_count, block_size, 1}, ACL_FLOAT, key_scale.data());
  const int32_t actual_key_len = seq_len;
  DeviceTensor d_key_len(device, {1}, ACL_INT32, &actual_key_len);
  DeviceTensor d_blocks(device, {1, blocks_count}, ACL_INT32, block_table.data());
  DeviceTensor d_metadata(device, {kIndexerMetadataElements}, ACL_INT32, metadata.data());
  DeviceTensor d_indices(device, {1, 1, 1, top_k}, ACL_INT32);
  DeviceTensor d_values(device, {0}, ACL_FLOAT);
  char query_layout[] = "BSND";
  char key_layout[] = "PA_BSND";
  RunAclnn<VllmQuantLightningIndexerPlanFn>(
      ops, device, stream, OpId::kVllmQuantLightningIndexer, d_query.get(), d_key.get(), d_weights.get(),
      d_query_scale.get(), d_key_scale.get(), static_cast<const aclTensor*>(nullptr), d_key_len.get(),
      d_blocks.get(), d_metadata.get(), 0, 0, query_layout, key_layout, top_k, 0,
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 1, false,
      block_size * head_dim, block_size, d_indices.get(), d_values.get());
  d_indices.Read(actual.data(), actual.size() * sizeof(int32_t));
  const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
  report.Check(mismatch.first == expected.end(), "aclnnVllmQuantLightningIndexer top-k matches CPU golden",
               mismatch.first == expected.end()
                   ? "exact INT32 match over FP8 E4M3 inputs"
                   : "first mismatch at " + std::to_string(mismatch.first - expected.begin()) + ": expected " +
                         std::to_string(*mismatch.first) + ", got " + std::to_string(*mismatch.second));
}

void TestCompressor(Report& report, AclDeviceOps& device, DeviceStream stream, const OpTable& ops) {
  constexpr int ratio = moe::emu::kCompressorRatio;
  constexpr int hidden = moe::emu::kHiddenSize;
  constexpr int dim = moe::emu::kCompressorHeadDim;
  constexpr int state_width = 2 * dim;
  std::mt19937 rng(kSeed + 2);
  std::uniform_real_distribution<float> activation(-0.5F, 0.5F);
  std::uniform_real_distribution<float> projection_weight(-0.02F, 0.02F);
  std::uniform_real_distribution<float> positional_bias(-0.1F, 0.1F);

  std::vector<uint16_t> x(static_cast<size_t>(ratio) * hidden);
  std::vector<uint16_t> wkv(static_cast<size_t>(state_width) * hidden);
  std::vector<uint16_t> wgate(wkv.size());
  std::vector<uint16_t> norm(dim, moe::emu::FloatToBf16(1.0F));
  std::vector<float> ape(static_cast<size_t>(ratio) * state_width);
  std::vector<float> rope_sin(2 * moe::emu::kRopeHeadDim);
  std::vector<float> rope_cos(2 * moe::emu::kRopeHeadDim);
  for (auto& value : x) value = moe::emu::FloatToBf16(activation(rng));
  for (auto& value : wkv) value = moe::emu::FloatToBf16(projection_weight(rng));
  for (auto& value : wgate) value = moe::emu::FloatToBf16(projection_weight(rng));
  for (float& value : ape) value = positional_bias(rng);
  for (size_t i = 0; i < rope_sin.size(); ++i) {
    const float angle = static_cast<float>(i) * 0.01F;
    rope_sin[i] = std::sin(angle);
    rope_cos[i] = std::cos(angle);
  }

  std::vector<uint16_t> sin_bf16(rope_sin.size()), cos_bf16(rope_cos.size());
  for (size_t i = 0; i < sin_bf16.size(); ++i) {
    sin_bf16[i] = moe::emu::FloatToBf16(rope_sin[i]);
    cos_bf16[i] = moe::emu::FloatToBf16(rope_cos[i]);
  }
  // The operator accepts BF16 RoPE tables; the oracle receives their widened
  // values so both implementations use the same quantized inputs.
  for (size_t i = 0; i < rope_sin.size(); ++i) {
    rope_sin[i] = moe::emu::Bf16ToFloat(sin_bf16[i]);
    rope_cos[i] = moe::emu::Bf16ToFloat(cos_bf16[i]);
  }
  moe::emu::CompressorRingState emu_state;
  std::vector<uint16_t> expected(dim, 0);
  for (int step = 0; step < ratio; ++step) {
    const auto result = moe::emu::paged_compressor_decode_c4a(
        x.data() + static_cast<size_t>(step) * hidden, wkv.data(), wgate.data(), ape.data(), norm.data(),
        step, &emu_state, rope_cos.data(), rope_sin.data(), expected.data(), hidden, dim, ratio,
        moe::emu::kRmsNormEpsilon, moe::emu::RotaryMode::kHalf);
    if (step + 1 == ratio && !result.emitted) {
      throw std::runtime_error("CPU compressor golden did not emit with BF16 RoPE tables");
    }
  }
  std::vector<float> expected_f32(dim);
  for (size_t i = 0; i < expected.size(); ++i) expected_f32[i] = moe::emu::Bf16ToFloat(expected[i]);

  const std::array<int32_t, 4> block_ids = {0, 1, 2, 3};
  const std::array<int32_t, 2> cu_seqlens = {0, ratio};
  const int32_t used_rows = ratio;
  const int32_t start_pos = 0;
  DeviceTensor d_x(device, {ratio, hidden}, ACL_BF16, x.data());
  DeviceTensor d_wkv(device, {state_width, hidden}, ACL_BF16, wkv.data());
  DeviceTensor d_wgate(device, {state_width, hidden}, ACL_BF16, wgate.data());
  DeviceTensor d_state(device, {4, 8, state_width}, ACL_FLOAT);
  DeviceTensor d_ape(device, {ratio, state_width}, ACL_FLOAT, ape.data());
  DeviceTensor d_norm(device, {dim}, ACL_BF16, norm.data());
  DeviceTensor d_sin(device, {2, moe::emu::kRopeHeadDim}, ACL_BF16, sin_bf16.data());
  DeviceTensor d_cos(device, {2, moe::emu::kRopeHeadDim}, ACL_BF16, cos_bf16.data());
  DeviceTensor d_blocks(device, {1, 4}, ACL_INT32, block_ids.data());
  DeviceTensor d_cu(device, {2}, ACL_INT32, cu_seqlens.data());
  DeviceTensor d_used(device, {1}, ACL_INT32, &used_rows);
  DeviceTensor d_start(device, {1}, ACL_INT32, &start_pos);
  DeviceTensor d_output(device, {2, dim}, ACL_BF16);

  RunAclnn<CompressorPlanFn>(
      ops, device, stream, OpId::kCompressor, d_x.get(), d_wkv.get(), d_wgate.get(), d_state.get(),
      d_ape.get(), d_norm.get(), d_sin.get(), d_cos.get(), d_blocks.get(), d_cu.get(), d_used.get(),
      d_start.get(), moe::emu::kRopeHeadDim, ratio, moe::emu::kCompressorCoff,
      static_cast<double>(moe::emu::kRmsNormEpsilon), static_cast<int64_t>(moe::emu::RotaryMode::kHalf),
      2, static_cast<int64_t>(8 * state_width), d_output.get());
  std::vector<uint16_t> actual(2 * dim);
  d_output.Read(actual.data(), actual.size() * sizeof(uint16_t));
  std::vector<uint16_t> actual_first(actual.begin(), actual.begin() + dim);
  CheckBf16(report, "aclnnCompressor first emitted row matches CPU golden", actual_first,
            expected_f32.data(), dim);
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
  std::printf("aclnn parity against tests/emu CPU goldens "
              "(seeds: mHC=0x%X, indexer=0x%X, compressor=0x%X)\n",
              kSeed, kSeed + 1, kSeed + 2);
  if (!PhysicalNpuPresent()) return 0;

  return GuardedMain([&]() {
    OpTable ops;
    ops.RequireAll({OpId::kMhcPre, OpId::kMhcSinkhorn, OpId::kMhcPost, OpId::kHcPre,
                    OpId::kCompressor, OpId::kVllmQuantLightningIndexer});
    AclDeviceOps device(0);
    const DeviceStream stream = device.CreateStream();
    Report report;
    const Inputs inputs = MakeInputs();
    moe::emu::MhcSinkhornTrace trace;
    moe::emu::mhc_pre_sinkhorn_cpu(inputs.x.data(), inputs.phi.data(), inputs.alpha, inputs.bias, &trace);

    RunCase(report, "mHC split chain", [&] {
      TestMhcChain(report, device, stream, ops, inputs, trace);
    });
    RunCase(report, "mHC fused pre", [&] {
      TestFusedMhc(report, device, stream, ops, inputs, trace);
    });
    RunCase(report, "FP8 lightning indexer", [&] {
      TestFp8Indexer(report, device, stream, ops);
    });
    RunCase(report, "C4A compressor", [&] {
      TestCompressor(report, device, stream, ops);
    });
    device.DestroyStream(stream);
    return report.exit_code();
  });
}
