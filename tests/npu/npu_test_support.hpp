#pragma once

#include <acl/acl.h>
#include "moe/core/acl_guard.hpp"
#include <cstdio>

// Gate live probes before reading any model data or planning operators.
inline bool PhysicalNpuPresent() {
  try { ascend_moe::EnsureAclRuntime(); } catch (const std::exception&) {
    std::printf("[ SKIP ] Physical Ascend NPU not detected\n");
    return false;
  }
  uint32_t count = 0;
  if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
    std::printf("[ SKIP ] Physical Ascend NPU not detected\n");
    return false;
  }
  return true;
}
