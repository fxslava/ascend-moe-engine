#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstring>

#include "mock_acl_tensor.hpp"
#include "mock_allocator.hpp"
#include "mock_weight_source.hpp"
#include "moe/core/acl_guard.hpp"
#include "moe/core/device_ops.hpp"
#include "moe/core/op_table.hpp"
#include "moe/core/resource_scope.hpp"
#include "moe/memory/static_arena.hpp"
#include "moe/memory/weight_transpose.hpp"
#include "moe/pipeline/pipeline.hpp"
#include "runtime_faults.hpp"

using namespace ascend_moe;
using namespace ascend_moe::mock;

namespace {
size_t Count(const char* name) { return std::count(RuntimeTrace().begin(), RuntimeTrace().end(), name); }
size_t Position(const char* name) {
  return static_cast<size_t>(std::find(RuntimeTrace().begin(), RuntimeTrace().end(), name) -
                             RuntimeTrace().begin());
}
template <typename F>
void MustThrow(F run, const char* part) {
  bool failed = false;
  try {
    run();
  } catch (const std::exception& error) {
    failed = std::strstr(error.what(), part) != nullptr;
  }
  DSV4_REQUIRE(failed, "expected exception containing " << part);
}
void TestResources() {
  EnsureAclRuntime();
  ClearRuntimeTrace();
  FailRuntimeCall("aclrtSetCurrentContext");
  MustThrow([] { AclDeviceOps device(0); }, "injected failure");
  DSV4_REQUIRE(Count("aclrtDestroyContext") == 1 && Count("aclrtResetDevice") == 1,
               "partial device construction must destroy its context and reset device");
  DSV4_REQUIRE(Position("aclrtDestroyContext") < Position("aclrtResetDevice"), "context before reset");
  ClearRuntimeTrace();
  {
    AclDeviceOps device(0);
    struct PartiallyBuilt {
      ResourceScope resources;
      explicit PartiallyBuilt(AclDeviceOps& device) : resources(device, device) {
        resources.CreateStream();
        resources.DeviceMalloc(4096);
        resources.HostPinnedMalloc(4096);
        resources.DeviceMalloc(4096);
      }
    };
    FailRuntimeCall("aclrtMalloc", 1);
    MustThrow([&] { PartiallyBuilt value(device); }, "injected failure");
    DSV4_REQUIRE(Count("aclrtDestroyStream") == 1 && Count("aclrtFreeHost") == 1,
                 "constructor rollback must release streams and pinned allocations");
    DSV4_REQUIRE(Position("aclrtSynchronizeStreamWithTimeout") < Position("aclrtFree"), "drain before free");
    // Unclaimed raw handles remain owned by the backend as a final safety net.
    device.CreateStream();
    device.DeviceMalloc(1024);
    device.HostPinnedMalloc(1024);
    FailRuntimeCall("aclrtSynchronizeStreamWithTimeout");
  }
  DSV4_REQUIRE(Count("aclrtDestroyStreamForce") == 1, "timed-out drain must force-destroy stream");
  DSV4_REQUIRE(Count("aclrtDestroyContext") == 1 && Count("aclrtResetDevice") == 1,
               "device teardown must happen exactly once");
  DSV4_REQUIRE(Position("aclrtDestroyStreamForce") < Position("aclrtResetDevice"), "stream before reset");
}

int PartialPlan(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor** executor) {
  *executor = reinterpret_cast<aclOpExecutor*>(new MockAclOpExecutor());
  return 161002;
}
void TestExecutorsAndMatmul() {
  AclDeviceOps device(0);
  StaticMemoryArena arena(device);
  auto storage = arena.Reserve("test", 4096);
  arena.Commit();
  auto tensor = [&](const char* name, std::vector<int64_t> dims, int dtype, size_t offset = 0) {
    return arena.CreateTensor(name, dims, dtype, static_cast<char*>(arena.Address(storage)) + offset);
  };
  auto* a = tensor("A", {2, 3}, ACL_BF16);
  auto* b = tensor("B", {3, 4}, ACL_BF16, 32);
  auto* c = tensor("C", {2, 4}, ACL_BF16, 64);
  ValidateDenseMatmul(a, b, c, 0);
  auto* original_b = tensor("checkpoint B", {4, 3}, ACL_BF16, 32);
  MustThrow([&] { ValidateDenseMatmul(a, original_b, c, 0); }, "no implicit weight transpose");
  auto* fp32 = tensor("FP32 routing", {2, 3}, ACL_FLOAT32);
  MustThrow([&] { ValidateDenseMatmul(fp32, b, c, 0); }, "BF16/FP16");
  auto* fp16 = tensor("FP16 routing", {2, 3}, ACL_FLOAT16);
  MustThrow([&] { ValidateDenseMatmul(fp16, b, c, 0); }, "dtypes must match");
  auto* quantized = tensor("FP8 weight", {3, 4}, ACL_FLOAT8_E4M3FN, 32);
  MustThrow([&] { ValidateDenseMatmul(a, quantized, c, 0); }, "quantized weights");
  auto* misaligned = tensor("bad address", {3, 4}, ACL_BF16, 2);
  MustThrow([&] { ValidateDenseMatmul(a, misaligned, c, 0); }, "32-byte-aligned");
  // List destruction owns distinct copies even when tensors occur in two lists.
  arena.CreateTensorList("K", {b});
  arena.CreateTensorList("V", {b});
  OpTable ops;
  aclOpExecutor* executor = nullptr;
  auto workspace = PlanAclnnOp<MatmulPlanFn>(ops, OpId::kMatmul, &executor, a, b, c, int8_t{0});
  ClearRuntimeTrace();
  {
    StaticOpSlot slot;
    FailRuntimeCall("aclSetAclOpExecutorRepeatable");
    MustThrow([&] { slot.Adopt(OpId::kMatmul, "test", workspace, executor); }, "injected failure");
  }
  DSV4_REQUIRE(Count("aclDestroyAclOpExecutor") == 1, "failed adoption must release its executor once");
  // A failed plan must not destroy a previous successful plan's stale out value.
  const_cast<ResolvedOp&>(ops.op(OpId::kSqrt)).plan = reinterpret_cast<void*>(&PartialPlan);
  executor = reinterpret_cast<aclOpExecutor*>(uintptr_t{0x1234});
  ClearRuntimeTrace();
  MustThrow([&] { PlanAclnnOp<UnaryPlanFn>(ops, OpId::kSqrt, &executor, a, c); }, "161002");
  DSV4_REQUIRE(executor == nullptr && Count("aclDestroyAclOpExecutor") == 1,
               "failed plan must release only the partial executor and clear the output");
}

class MatrixSource : public WeightByteSource {
 public:
  std::vector<uint16_t> data;
  MatrixSource(size_t n, size_t k) : data(n * k) {
    for (size_t i = 0; i < data.size(); ++i) data[i] = i;
  }
  const char* source_name() const override { return "matrix fixture"; }
  bool Contains(int32_t, int32_t) const override { return false; }
  void ReadExpertSlotRange(uint8_t*, size_t, size_t, size_t, int32_t, int32_t) override {}
  bool HasNamed(const std::string&) const override { return true; }
  void ReadNamed(const std::string&, uint8_t* dest, size_t capacity, size_t offset, size_t count) override {
    DSV4_REQUIRE(count <= capacity && offset + count <= data.size() * 2, "fixture bounds");
    std::memcpy(dest, reinterpret_cast<uint8_t*>(data.data()) + offset, count);
  }
  size_t NamedByteSize(const std::string&) const override { return data.size() * 2; }
  void Close() override { closed_ = true; }
};
void TestTranspose() {
  // Non-square and ragged at the 512-row tile boundary; verify actual bytes.
  constexpr size_t n = 515, k = 7;
  SimulatedDeviceOps device(1 << 20);
  ResourceScope memory(device, device);
  auto* output = static_cast<uint16_t*>(memory.DeviceMalloc(n * k * 2));
  MatrixSource source(n, k);
  IngestTransposedBf16(source, "weight", output, n, k, device, device);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < k; ++j)
      DSV4_REQUIRE(output[j * n + i] == source.data[i * k + j], "transpose changed matrix elements");
}
void TestPipelineFailure() {
  ClearRuntimeTrace();
  MustThrow(
      [] {
        AclDeviceOps device(0);
        const auto layout = ExpertSlotLayout::ForDeepSeekV4Flash();
        ExclusiveExpertManager::Options options;
        options.routed_coverage = kNumExpertsPerTok;
        options.device_slots = kNumExpertsPerTok;
        options.transfer_chunk_bytes = 512 * 1024;
        ExclusiveExpertManager experts(device, device, layout, options);
        OpTable ops;
        MoeRouterEngine router(device, device);
        RuntimeConfig config;
        config.synthetic_weights = true;
        config.max_context_len = 256;
        Dsv4Pipeline pipeline(device, device, ops, experts, router, config);
        SymbolicWeightSource source(layout, kNumLayers, kNumRoutedExperts);
        pipeline.Build(source);
        experts.Ingest(source, {});
        ClearRuntimeTrace();
        FailRuntimeCall("aclnnQuantMatmulV5");
        pipeline.DecodeStep(7, 0);
      },
      "injected failure");
  DSV4_REQUIRE(Count("aclDestroyAclOpExecutor") > 20, "all planned executors released after launch failure");
  DSV4_REQUIRE(Position("aclrtSynchronizeStreamWithTimeout") < Position("aclDestroyAclOpExecutor"),
               "launched streams must drain before executor destruction");
  DSV4_REQUIRE(Count("aclrtResetDevice") == 1, "failed pipeline resets its device");
  const auto& memory = MockMemoryStatistics();
  DSV4_REQUIRE(memory.active_spans == 0, "failed pipeline leaked device or pinned spans");
}

void TestSignals() {
  for (int signal : {SIGINT, SIGABRT, SIGSEGV}) {
    const pid_t child = fork();
    DSV4_REQUIRE(child >= 0, "fork failed");
    if (child == 0) {
      const int result = GuardedMain([&] {
        AclDeviceOps device(0);
        device.CreateStream();
        std::raise(signal);
        CheckForInterrupt();
        return 0;
      });
      std::exit(result);
    }
    int status = 0;
    DSV4_REQUIRE(waitpid(child, &status, 0) == child, "waitpid failed");
    DSV4_REQUIRE(WIFEXITED(status) && WEXITSTATUS(status) == (signal == SIGINT ? 1 : 128 + signal),
                 "signal guard failed to exit promptly");
  }
}
}  // namespace

int main() {
  // Registered before RuntimeLifetime: observe both finalizers after its destruction.
  std::atexit([] {
    if (Count("aclnnFinalize") != 1 || Count("aclFinalize") != 1 ||
        Position("aclnnFinalize") > Position("aclFinalize")) {
      std::fputs("FAIL: process runtime finalization order/count\n", stderr);
      std::_Exit(1);
    }
  });
  return GuardedMain([] {
    TestResources();
    TestExecutorsAndMatmul();
    TestTranspose();
    TestPipelineFailure();
    TestSignals();
    std::puts(
        "PASS: Matmul contracts, transpose bytes, partial construction, executors, timeout and signals");
    return 0;
  });
}
