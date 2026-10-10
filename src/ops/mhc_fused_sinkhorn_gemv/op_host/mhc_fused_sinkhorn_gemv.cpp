#include "moe/ops/mhc_fused_sinkhorn_gemv.hpp"
#include "mhc_fused_sinkhorn_gemv_tiling.h"

#include <cstdint>

namespace moe::ops {

extern "C" std::uint32_t MhcFusedSinkhornGemvGetTilingPlan(
    MhcFusedSinkhornGemvTilingPlan* plan) {
  if (plan == nullptr) {
    return 1;
  }
  const auto internal_plan = mhc_fused_sinkhorn_gemv::MakeTilingPlan();
  plan->hidden_size = internal_plan.hidden_size;
  plan->head_count = internal_plan.head_count;
  plan->sinkhorn_iterations = internal_plan.sinkhorn_iterations;
  plan->reserved = internal_plan.reserved;
  plan->workspace_bytes = internal_plan.workspace_bytes;
  return 0;
}

}  // namespace moe::ops
