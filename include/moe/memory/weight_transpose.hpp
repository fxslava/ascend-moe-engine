#pragma once
#include "moe/core/device_allocator.hpp"
#include "moe/core/stream_engine.hpp"
#include "moe/core/weight_source.hpp"

namespace ascend_moe {
/// Load BF16 checkpoint [N,K] bytes into contiguous ND [K,N] device storage.
/// Transposition is bit-preserving and uses bounded startup-only pinned tiles.
void IngestTransposedBf16(WeightByteSource& source, const std::string& name, void* destination, size_t n,
                          size_t k, IDeviceAllocator& allocator, IStreamEngine& streams);
}  // namespace ascend_moe
