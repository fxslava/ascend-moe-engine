#include "moe/memory/weight_transpose.hpp"

#include <algorithm>

#include "moe/core/error.hpp"
#include "moe/core/resource_scope.hpp"

namespace ascend_moe {
void IngestTransposedBf16(WeightByteSource& source, const std::string& name, void* destination, size_t n,
                          size_t k, IDeviceAllocator& allocator, IStreamEngine& streams) {
  DSV4_REQUIRE(n && k, "empty BF16 projection");
  const auto dtype = source.NamedDtype(name);
  DSV4_REQUIRE(dtype.empty() || dtype == "BF16",
               name << ": dense projection requires BF16 checkpoint bytes, got " << dtype
                    << "; quantized weights need their scale tensors and a quantization-aware GEMM");
  const auto shape = source.NamedShape(name);
  DSV4_REQUIRE(
      shape.empty() || shape == std::vector<int64_t>({static_cast<int64_t>(n), static_cast<int64_t>(k)}),
      name << ": expected checkpoint shape [N,K]=[" << n << ',' << k << ']');
  constexpr size_t tile_rows = 512;
  const size_t tile_bytes = std::min(n, tile_rows) * k * sizeof(uint16_t);
  ResourceScope staging(allocator, streams);
  auto* input = static_cast<uint16_t*>(staging.HostPinnedMalloc(tile_bytes));
  auto* output = static_cast<uint16_t*>(staging.HostPinnedMalloc(tile_bytes));
  auto* target = static_cast<uint16_t*>(destination);
  for (size_t row = 0; row < n; row += tile_rows) {
    CheckForInterrupt();
    const size_t rows = std::min(tile_rows, n - row);
    source.ReadNamed(name, reinterpret_cast<uint8_t*>(input), tile_bytes, row * k * 2, rows * k * 2);
    for (size_t col = 0; col < k; ++col) {
      for (size_t r = 0; r < rows; ++r) output[col * rows + r] = input[r * k + col];
      streams.MemcpySync(target + col * n + row, (n - row) * 2, output + col * rows, rows * 2,
                         MemcpyKind::kHostToDevice);
    }
  }
}
}  // namespace ascend_moe
