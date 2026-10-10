#include "mhc_fused_sinkhorn_gemv_tiling.h"

namespace moe::ops::mhc_fused_sinkhorn_gemv {

TilingPlan MakeTilingPlan() {
  return {4096, 4, kTilingSinkhornIterations, 0, 0};
}

}  // namespace moe::ops::mhc_fused_sinkhorn_gemv
