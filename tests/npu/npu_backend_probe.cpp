#include "moe/core/acl_guard.hpp"
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

// npu_backend_probe -- backend diagnostics without npu-smi.
//
// Two sections, both read-only:
//
//   1. DEVICE: aclInit -> aclrtGetDeviceCount -> aclrtSetDevice(0) ->
//      aclrtGetSocName -> aclrtGetMemInfo(ACL_HBM_MEM). Purely through the
//      ACL C-APIs. No NPU attached (a build container, an x86 host) is a
//      SKIP, not a failure -- the probe simply does not apply.
//
//   2. OPERATOR SYMBOLS: dlsym(RTLD_DEFAULT, ...) for the six GetWorkspaceSize
//      entry points the decode graph plans against, with dladdr naming the
//      providing shared object. This needs no device at all -- only that the
//      toolkit is on the link line, which it is whenever
//      this binary exists. A missing symbol is the difference between "this
//      toolkit cannot run the model" and a confusing mid-run dlsym failure,
//      which is exactly what this probe exists to surface early. No operator
//      is invoked; nothing but symbol lookup happens.
//
// Exit status: 0 if every REQUIRED symbol resolved and no device claim was
// false; 1 otherwise. A device-less host exits 0 before operator lookup.

#include <acl/acl.h>
#include "moe/core/device_ops.hpp"
#include "npu_test_support.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  std::printf("  [ %s ] %s\n", condition ? "ok" : "FAIL", what.c_str());
  if (!condition) {
    ++g_failures;
  }
}

void Skip(const std::string& what) { std::printf("  [ SKIP ] %s\n", what.c_str()); }

void Section(const char* title) { std::printf("\n== %s ==\n", title); }

// The GetWorkspaceSize entry points the DSV4 decode graph cannot plan without.
// The launch entry (same name minus the suffix) is checked alongside each.
struct OperatorSymbol {
  const char* plan;         // aclnn<Op>GetWorkspaceSize
  const char* role;
};

const OperatorSymbol kRequiredSymbols[] = {
    {"aclnnSoftplusGetWorkspaceSize", "sqrtsoftplus router scoring, stage 1"},
    {"aclnnSqrtGetWorkspaceSize", "sqrtsoftplus router scoring, stage 2"},
    {"aclnnMoeGatingTopKV2GetWorkspaceSize", "noaux_tc gating over pre-normalized scores"},
    {"aclnnMoeInitRoutingV4GetWorkspaceSize", "dropless dispatch, device cumsum groupList"},
    {"aclnnGroupedMatmulV5GetWorkspaceSize", "FP8xFP4 M-grouped expert GEMM"},
    {"aclnnFusedInferAttentionScoreV5GetWorkspaceSize", "paged MLA decode attention"},
};

void ProbeOperatorSymbols() {
  Section("operator symbols: dlsym(RTLD_DEFAULT), no operator is invoked");
  for (const OperatorSymbol& symbol : kRequiredSymbols) {
    void* plan = dlsym(RTLD_DEFAULT, symbol.plan);
    if (plan != nullptr) {
      Dl_info info{};
      const char* provider = "<unknown>";
      if (dladdr(plan, &info) != 0 && info.dli_fname != nullptr) {
        provider = info.dli_fname;
      }
      Check(true, std::string("found ") + symbol.plan + " (" + symbol.role + ")\n"
                       "           from " + provider);
    } else {
      Check(false, std::string("MISSING ") + symbol.plan + " (" + symbol.role + "): " + dlerror());
    }
  }
}

bool ProbeDevice() {
  ascend_moe::AclDeviceOps device(0);
  Section("device: aclInit -> device count -> SoC -> HBM info");

  // aclInit is per process; ACL_ERROR_REPEAT_INITIALIZE means somebody (the
  // engine's device ops, a previous section) already initialized it, which is
  // fine for a probe. Any OTHER status on a driver-less container (500000 on
  // a host with no CANN driver, for instance) is the runtime itself saying
  // "no backend here" -- a SKIP for this section, not a symbol failure.
  const aclError init_status = aclInit(nullptr);
  if (init_status != 0 && init_status != ACL_ERROR_REPEAT_INITIALIZE) {
    Skip("aclInit returned " + std::to_string(init_status) + ": no usable CANN backend on this host; "
             "device section skipped (symbol section still ran)");
    return false;
  }
  Check(init_status == 0 ? true : init_status == ACL_ERROR_REPEAT_INITIALIZE,
        init_status == 0 ? "aclInit initialized the ACL runtime"
                         : "aclInit answered repeat-initialize (already initialized), accepted");

  uint32_t device_count = 0;
  const aclError count_status = aclrtGetDeviceCount(&device_count);
  if (count_status != 0) {
    Check(false, "aclrtGetDeviceCount failed with status " + std::to_string(count_status));
    return true;
  }
  if (device_count == 0) {
    Skip("aclrtGetDeviceCount reports 0 devices: no NPU on this host; "
         "device section skipped (symbol section still ran)");
    return false;
  }
  Check(device_count > 0, "aclrtGetDeviceCount reports " + std::to_string(device_count) + " device(s)");

  const aclError set_status = aclrtSetDevice(0);
  Check(set_status == 0, "aclrtSetDevice(0) returned " + std::to_string(set_status));

  const char* soc = aclrtGetSocName();
  Check(soc != nullptr && soc[0] != '\0',
        std::string("aclrtGetSocName: ") + (soc != nullptr ? soc : "<null>"));

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  const aclError mem_status = aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
  Check(mem_status == 0, "aclrtGetMemInfo(ACL_HBM_MEM) returned " + std::to_string(mem_status));
  if (mem_status == 0) {
    Check(total_bytes > 0 && free_bytes <= total_bytes,
          "HBM: " + std::to_string(free_bytes >> 20) + " MiB free of " + std::to_string(total_bytes >> 20) +
              " MiB (free <= total, total > 0)");
  }
  return true;
}

}  // namespace

int RunMain(int argc, char** argv) {
  std::printf("npu_backend_probe -- ACL device + operator-symbol diagnostics (no npu-smi)\n");
  const bool require_device = argc > 1 && std::strcmp(argv[1], "--require-device") == 0;
  if (!PhysicalNpuPresent()) return require_device ? 1 : 0;

  int failures_before_device = g_failures;
  const bool device_present = ProbeDevice();
  ProbeOperatorSymbols();

  if (!device_present && require_device) {
    std::printf("\n[FAIL] --require-device was given but no device is present\n");
    return 1;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  const bool symbol_failures = g_failures > failures_before_device;
  if (!device_present) {
    std::printf("verdict: %s (device absent is a SKIP; symbols are the gate)\n",
                symbol_failures ? "FAIL" : "PASS");
  }
  return symbol_failures ? 1 : 0;
}

int main(int argc, char** argv) {
  return ascend_moe::GuardedMain([&] { return RunMain(argc, argv); });
}
