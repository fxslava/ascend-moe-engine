// device_session.hpp -- ACL device bring-up and RAII GM buffers, shared by the
// suites that actually touch hardware.
//
// WHY THIS IS A SHARED HEADER
// tests/npu verifies correctness on device and tests/bench times it. Both need
// the same three things -- open a device, allocate GM, move data -- and both
// need the same distinction between "there is no NPU here" and "there is one
// and bring-up failed" (CLAUDE.md section 4.4). Having that logic twice is how
// the two suites drift into disagreeing about what a missing device means.
//
// EVERYTHING HERE IS COMPILED OUT WITHOUT THE LAUNCH LAYER.
// The whole header is behind MOE_KERNEL_LAUNCH_LAYER_AVAILABLE, which the
// arch35_device_launch target carries along with the CANN include dirs and the
// AscendCL link interface. A host-emu build defines none of it and must not
// see <acl/acl.h>, so a translation unit that includes this header still
// compiles there -- it just gets nothing.
//
// NOTE on loading, not just linking (CLAUDE.md section 4.4): a binary linked
// against libascendcl pulls in libascend_hal.so, which ships with the DRIVER
// and not the toolkit, and the build deliberately uses -rpath-link rather than
// -rpath so a stub can never satisfy it at run time. On a host with neither
// driver nor device such a binary fails to load before main() and exits 127.
// That is the correct outcome; it must not be "fixed" by putting the stub
// directory on the runtime search path.

#ifndef TESTS_COMMON_DEVICE_SESSION_HPP
#define TESTS_COMMON_DEVICE_SESSION_HPP

#if defined(MOE_KERNEL_LAUNCH_LAYER_AVAILABLE)

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace moe::kerntest::device {

// ---------------------------------------------------------------------------
// RAII GM buffer
// ---------------------------------------------------------------------------
// The caller owns every operand buffer (CLAUDE.md section 3.4); the Launch*
// wrappers allocate nothing but their own tiling staging. This is that owner.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { Free(); }

  aclError Alloc(std::size_t bytes) {
    Free();
    bytes_ = bytes;
    if (bytes == 0) {
      return ACL_SUCCESS;  // a legitimately empty operand stays null
    }
    const aclError rc = aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
      ptr_ = nullptr;
      bytes_ = 0;
    }
    return rc;
  }

  aclError Upload(const void* host, std::size_t bytes) {
    if (bytes == 0) return ACL_SUCCESS;
    return aclrtMemcpy(ptr_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  }

  aclError Download(void* host, std::size_t bytes) const {
    if (bytes == 0) return ACL_SUCCESS;
    return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  }

  aclError Zero() {
    if (bytes_ == 0) return ACL_SUCCESS;
    return aclrtMemset(ptr_, bytes_, 0, bytes_);
  }

  void Free() noexcept {
    if (ptr_ != nullptr) {
      (void)aclrtFree(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  void* get() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return bytes_; }
  template <typename T>
  T* as() const noexcept {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------
// Device session
// ---------------------------------------------------------------------------
// Two outcomes are deliberately kept apart (CLAUDE.md section 4.4):
//
//   kNoDevice -- there is no NPU here, or no CANN runtime to ask. An
//                environment fact, not a defect. tests/npu turns it into the
//                skip banner and exit 0; tests/bench turns it into a
//                "NOT MEASURED" row and keeps going.
//   kFailed   -- hardware EXISTS but device, context or stream bring-up
//                failed. A real fault, reported as one, with its aclError.
//
// Collapsing the two is how a suite ends up either crying wolf on every
// laptop or passing silently on a broken machine.
enum class OpenResult { kOk, kNoDevice, kFailed };

inline std::string AclFail(const char* what, aclError rc) {
  return std::string(what) + " failed, aclError=" + std::to_string(rc);
}

class AclSession {
 public:
  AclSession() = default;
  AclSession(const AclSession&) = delete;
  AclSession& operator=(const AclSession&) = delete;

  OpenResult Open(std::string* why) {
    const aclError init_rc = aclInit(nullptr);
    if (init_rc != ACL_SUCCESS) {
      // No runtime to ask means no device to find. ACL_ERROR_REPEAT_INITIALIZE
      // is not an error here: something else in the process already did it.
      if (init_rc != ACL_ERROR_REPEAT_INITIALIZE) {
        *why = "aclInit failed (no CANN runtime available), aclError=" +
               std::to_string(init_rc);
        return OpenResult::kNoDevice;
      }
    } else {
      initialized_ = true;
    }

    std::uint32_t count = 0;
    const aclError count_rc = aclrtGetDeviceCount(&count);
    if (count_rc != ACL_SUCCESS) {
      *why = "aclrtGetDeviceCount failed, aclError=" + std::to_string(count_rc);
      return OpenResult::kNoDevice;
    }
    if (count == 0) {
      *why = "aclrtGetDeviceCount reported 0 devices";
      return OpenResult::kNoDevice;
    }
    device_count_ = count;

    // Past this point hardware exists, so every failure is a real one.
    const aclError set_rc = aclrtSetDevice(0);
    if (set_rc != ACL_SUCCESS) {
      *why = AclFail("aclrtSetDevice(0)", set_rc);
      return OpenResult::kFailed;
    }
    device_set_ = true;

    const aclError stream_rc = aclrtCreateStream(&stream_);
    if (stream_rc != ACL_SUCCESS) {
      *why = AclFail("aclrtCreateStream", stream_rc);
      return OpenResult::kFailed;
    }
    return OpenResult::kOk;
  }

  aclrtStream stream() const noexcept { return stream_; }
  std::uint32_t device_count() const noexcept { return device_count_; }

  ~AclSession() {
    if (stream_ != nullptr) (void)aclrtDestroyStream(stream_);
    if (device_set_) (void)aclrtResetDevice(0);
    if (initialized_) (void)aclFinalize();
  }

 private:
  aclrtStream stream_ = nullptr;
  std::uint32_t device_count_ = 0;
  bool initialized_ = false;
  bool device_set_ = false;
};

}  // namespace moe::kerntest::device

#endif  // MOE_KERNEL_LAUNCH_LAYER_AVAILABLE
#endif  // TESTS_COMMON_DEVICE_SESSION_HPP
