#include "dsv4_moe_expert_tiling.h"

#include <cstdint>
#include <cstdio>

int main() {
  using moe::kernels::arch35::Dsv4MoeExpertTilingData;
  using moe::kernels::arch35::MakeDsv4MoeExpertTiling;
  using moe::kernels::arch35::Status;

  Dsv4MoeExpertTilingData tiling{};
  const Status scaled = MakeDsv4MoeExpertTiling(
      512u, 512u, 256u, 16u, 256u, 16u, &tiling);
  const Status production = MakeDsv4MoeExpertTiling(
      4096u, 2048u, 2048u, 128u, 1024u, 64u, &tiling);
  if (scaled != Status::unsupported_shape ||
      production != Status::success || tiling.block_dim != 32u) {
    std::fprintf(stderr,
                 "[FAIL] production tiler did not preserve 4096x2048/32-AIV "
                 "contract\n");
    return 1;
  }
  std::puts("[PASS] production tiler rejects scaled geometry and retains "
            "4096x2048/32-AIV geometry");
  return 0;
}
