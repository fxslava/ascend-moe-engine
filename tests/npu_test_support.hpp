#pragma once

#include <acl/acl.h>
#include <cstdio>

// Gate live probes before reading any model data or planning operators.
inline bool PhysicalNpuPresent() {
  const aclError status = aclInit(nullptr);
  if (status != ACL_SUCCESS && status != ACL_ERROR_REPEAT_INITIALIZE) {
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
