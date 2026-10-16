#include "kernel_operator.h"
#include "mhc_fused_sinkhorn_gemv_arch35.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void mhc_fused_sinkhorn_gemv_kernel(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR h_out,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  (void)tiling;
  const moe::ops::mhc_fused_sinkhorn_gemv::TilingPlan plan =
      moe::ops::mhc_fused_sinkhorn_gemv::MakeTilingPlan();
  TPipe pipe;
  (void)pipe;
  MhcFusedSinkhornGemvArch35 op(plan);
  op.Init(x, phi, alpha, bias, h_out, y);
  op.Process();
}
