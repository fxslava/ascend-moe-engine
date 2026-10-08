#pragma once
#include <vector>

#include "moe/core/device_allocator.hpp"
#include "moe/core/stream_engine.hpp"

namespace ascend_moe {
/** @brief Roll back partially constructed backend resources and drain before freeing.
 * Members using this scope must declare it after their allocator/stream references.
 * Reset before destroying executors, descriptors, or buffers borrowed from another owner.
 */
class ResourceScope {
 public:
  ResourceScope(IDeviceAllocator& allocator, IStreamEngine& streams)
      : allocator_(allocator), streams_(streams) {}
  ~ResourceScope() { Reset(); }
  ResourceScope(const ResourceScope&) = delete;
  ResourceScope& operator=(const ResourceScope&) = delete;
  void* DeviceMalloc(size_t bytes) { return Allocate(bytes, false); }
  void* HostPinnedMalloc(size_t bytes) { return Allocate(bytes, true); }
  DeviceStream CreateStream() {
    auto value = streams_.CreateStream();
    try {
      streams_owned_.push_back(value);
    } catch (...) {
      streams_.DestroyStream(value);
      throw;
    }
    return value;
  }
  DeviceEvent CreateEvent() {
    auto value = streams_.CreateEvent();
    try {
      events_.push_back(value);
    } catch (...) {
      streams_.DestroyEvent(value);
      throw;
    }
    return value;
  }
  void Reset() noexcept {
    // DestroyStream on ACL drains with a timeout and force-destroys on error.
    for (auto value : streams_owned_) {
      try {
        streams_.DestroyStream(value);
      } catch (...) {
      }
    }
    streams_owned_.clear();
    for (auto value : events_) {
      try {
        streams_.DestroyEvent(value);
      } catch (...) {
      }
    }
    events_.clear();
    for (auto value : memory_) {
      try {
        if (value.host)
          allocator_.HostPinnedFree(value.pointer);
        else
          allocator_.DeviceFree(value.pointer);
      } catch (...) {
      }
    }
    memory_.clear();
  }

 private:
  void* Allocate(size_t bytes, bool host) {
    void* value = host ? allocator_.HostPinnedMalloc(bytes) : allocator_.DeviceMalloc(bytes);
    try {
      memory_.push_back({value, host});
    } catch (...) {
      if (host)
        allocator_.HostPinnedFree(value);
      else
        allocator_.DeviceFree(value);
      throw;
    }
    return value;
  }
  IDeviceAllocator& allocator_;
  IStreamEngine& streams_;
  struct Memory {
    void* pointer;
    bool host;
  };
  std::vector<Memory> memory_;
  std::vector<DeviceStream> streams_owned_;
  std::vector<DeviceEvent> events_;
};
}  // namespace ascend_moe
