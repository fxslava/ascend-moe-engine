// -----------------------------------------------------------------------------
// dsv4_moe_prefill_cube.cpp -- host-side launch wrapper for the
// Cube-accelerated batched prefill entry point. Compiled by BiSheng as an ASC
// TU; see CLAUDE.md §3.
//
// ONE DISPATCH. The gate/up GEMMs, the SwiGLU and the down GEMM all run inside
// one launch: the Matmul objects' KfcServer carries the AIV -> AIC traffic, so
// there is no host-visible phase boundary to order and no second enqueue.
//
// THIS WRAPPER HAS A WORKSPACE, unlike the AIV block wrappers, and it is
// caller-owned. Two reasons stack:
//   * REGIST_MATMUL_OBJ puts the KFC message queue at the head of it, so a
//     MIX Matmul kernel cannot run without one at all;
//   * the dequantized BF16 weight planes live there too, because one expert's
//     W1 in BF16 is 16 MiB at the production geometry against 512 KiB of L1.
// Both are explained where they are sized, on
// Dsv4MoePrefillCubeWorkspaceBytes().
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35 {
namespace {

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR tiling);

constexpr const char* kCubeOp = "dsv4_moe_prefill_cube";

}  // namespace

Status LaunchDsv4MoePrefillCube(
    const float* x_tokens_gm, const Mxfp4Pair* w1_packed_gm,
    const E8M0* w1_scale_gm, const Mxfp4Pair* w3_packed_gm,
    const E8M0* w3_scale_gm, const Mxfp4Pair* w2_packed_gm,
    const E8M0* w2_scale_gm, const std::int32_t* token_to_expert_gm,
    std::uint32_t hidden, std::uint32_t inter, std::uint32_t num_experts,
    std::uint32_t total_tokens, float* y_tokens_gm, void* workspace_gm,
    aclrtStream stream) {
  if (x_tokens_gm == nullptr || w1_packed_gm == nullptr ||
      w1_scale_gm == nullptr || w3_packed_gm == nullptr ||
      w3_scale_gm == nullptr || w2_packed_gm == nullptr ||
      w2_scale_gm == nullptr || token_to_expert_gm == nullptr ||
      y_tokens_gm == nullptr || workspace_gm == nullptr || stream == nullptr) {
    return Status::invalid_argument;
  }
  // Each token's output row is written while its input row is still being read
  // through the workspace's BF16 copy, so in-place is not supported.
  if (static_cast<const void*>(y_tokens_gm) ==
      static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }
  // The workspace holds the BF16 copy of X and the dequantized weights; an
  // overlap with either operand is a self-corrupting launch.
  if (workspace_gm == static_cast<const void*>(x_tokens_gm) ||
      workspace_gm == static_cast<const void*>(y_tokens_gm)) {
    return Status::invalid_argument;
  }

  Dsv4MoePrefillCubeTilingData tiling{};
  const Status shape = MakeDsv4MoePrefillCubeTiling(
      hidden, inter, num_experts, total_tokens, detail::QueryAivCoreCount(),
      &tiling);
  if (shape != Status::success) {
    return shape;
  }

  detail::TilingStaging staging;
  const Status staged = staging.Stage(kCubeOp, &tiling, sizeof(tiling));
  if (staged != Status::success) {
    return staged;
  }

  // dyn_ub_size is ZERO here, and that is not an oversight. The AIV block
  // kernels hand __ubuf__ pointers to a SIMT VF, which must come from the
  // DYNAMIC region the chevron declares. This kernel has no VF: its UB is
  // TQue/TBuf, which the Matmul framework and TPipe allocate from the STATIC
  // region. Declaring dynamic bytes nobody addresses would reserve UB that the
  // static allocator then cannot use.
  dsv4_moe_prefill_cube<<<tiling.block_dim, 0u, stream>>>(
      detail::Gm(x_tokens_gm), detail::Gm(w1_packed_gm),
      detail::Gm(w1_scale_gm), detail::Gm(w3_packed_gm),
      detail::Gm(w3_scale_gm), detail::Gm(w2_packed_gm),
      detail::Gm(w2_scale_gm), detail::Gm(token_to_expert_gm),
      detail::Gm(y_tokens_gm), detail::Gm(workspace_gm),
      detail::Gm(staging.get()));

  return detail::FinishLaunch(kCubeOp, stream);
}

}  // namespace moe::kernels::arch35
