// -----------------------------------------------------------------------------
// paged_compressor_rope.cpp -- host-side launch wrapper.
// Compiled by BiSheng as an ASC TU; see CLAUDE.md §3.
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "paged_compressor_rope_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void paged_compressor_rope(
    GM_ADDR x, GM_ADDR wkv, GM_ADDR wgate, GM_ADDR ape, GM_ADDR norm_weight,
    GM_ADDR rope_cos, GM_ADDR rope_sin, GM_ADDR state, GM_ADDR state_meta,
    GM_ADDR cache_slot, GM_ADDR emitted, std::uint32_t step, GM_ADDR workspace,
    GM_ADDR tiling);

constexpr const char* kOp = "paged_compressor_rope";

}  // namespace

Status LaunchPagedCompressorRope(
    const Bf16* x_gm, const Bf16* wkv_gm, std::uint32_t wkv_row_stride,
    const Bf16* wgate_gm, std::uint32_t wgate_row_stride, const float* ape_gm,
    std::uint32_t ape_row_stride, const Bf16* norm_weight_gm,
    const float* rope_cos_gm, const float* rope_sin_gm, std::uint32_t step,
    float* state_gm, std::uint32_t* state_meta_gm, Bf16* cache_slot_gm,
    std::uint32_t* emitted_gm, void* workspace, std::size_t workspace_bytes,
    aclrtStream stream) {
  detail::ClearLaunchFailure();

  if (x_gm == nullptr || wkv_gm == nullptr || wgate_gm == nullptr ||
      ape_gm == nullptr || norm_weight_gm == nullptr ||
      rope_cos_gm == nullptr || rope_sin_gm == nullptr ||
      state_gm == nullptr || state_meta_gm == nullptr ||
      emitted_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // "its address must be valid even on HOLD, but HOLD leaves all its bytes
  // untouched" -- so cache_slot is required on every call, hold or emit.
  if (cache_slot_gm == nullptr) {
    return Status::invalid_argument;
  }
  if (workspace_bytes != kPagedCompressorWorkspaceBytes) {
    return Status::invalid_argument;
  }

  CompressorTilingData tiling{};
  const Status shape = MakeCompressorTiling(wkv_row_stride, wgate_row_stride,
                                            ape_row_stride, &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // The second chevron slot is dyn_ub_size (uint32 BYTES), not an l2ctrl
  // pointer: the 950PR hybrid launch is
  // kernel<<<block_num, dyn_ub_size, stream>>>. This kernel hands ten UB
  // pointers to two SIMT VFs, and all ten now come from the dynamic region,
  // so the size has to be declared here. Passing nullptr -- which is what
  // this call did -- declares ZERO dynamic bytes, which put every SIMT UB
  // access outside the reserved window and faulted on device with
  // aclError=507035 (ACL_ERROR_RT_VECTOR_CORE_EXCEPTION) at step 0.
  //
  // CLAUDE.md section 3.1 prescribes nullptr for this slot; the dynamic-UB
  // argument is the granted exception, and it is the same exception
  // mxfp4_expert_gemv, fused_mhc_pre_norm_sinkhorn and fast_fp8_indexer_topk
  // already rely on -- those three declare a real size and run on hardware,
  // this one did not and did not. The byte count is a compile-time constant
  // shared with the kernel's UB layout, so the ceiling is enforced by
  // static_assert in the tiling header rather than branched on here.
  paged_compressor_rope<<<tiling.block_dim, kCompressorDynUbBytes, stream>>>(
      detail::Gm(x_gm), detail::Gm(wkv_gm), detail::Gm(wgate_gm),
      detail::Gm(ape_gm), detail::Gm(norm_weight_gm), detail::Gm(rope_cos_gm),
      detail::Gm(rope_sin_gm), detail::Gm(state_gm),
      detail::Gm(state_meta_gm), detail::Gm(cache_slot_gm),
      detail::Gm(emitted_gm), step, detail::Gm(workspace),
      detail::Gm(staging.get()));
  return detail::FinishLaunch(kOp, stream);
}

}  // namespace moe::kernels::arch35
