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
// caller-owned -- but for ONE reason now, not two. REGIST_MATMUL_OBJ puts the
// KFC message queue at the head of it, so a MIX Matmul kernel cannot run
// without one at all. The second reason is gone: the dequantized BF16 weight
// planes used to live there because a TPosition::TSCM B operand must be the
// whole single-core B and one expert's W1 is 16 MiB in BF16. The two-phase
// kernel dequantizes per SUB-TILE into UB and pushes UB -> L1 instead, so
// nothing dequantized reaches GM. See Dsv4MoePrefillCubeWorkspaceBytes().
// -----------------------------------------------------------------------------

#include "kernel_operator.h"

#include "launch_common.hpp"

#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include <cstring>

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
  // Y ALIASING X IS STILL REJECTED, and the reason changed with the schedule,
  // so it is restated rather than left to read as stale. There is no longer a
  // BF16 copy of X in the workspace -- X is cast per sub-tile in UB -- but
  // phase 2 writes y[row0 .. row0+rows] while phase 1 of a LATER r0 step on
  // the same core has already finished reading x for those rows and a later
  // TILE has not. In-place would therefore let one core's output land on
  // another core's unread input, with no ordering between them.
  if (static_cast<const void*>(y_tokens_gm) ==
      static_cast<const void*>(x_tokens_gm)) {
    return Status::invalid_argument;
  }
  // The workspace is the Matmul KFC message queue. An overlap with either
  // operand is a self-corrupting launch -- the queue is written by both the
  // AIV client and the AIC server throughout.
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

  // THE STAGING BUFFER CARRIES THE OPERATOR POD **AND** THE TWO CUBE TILINGS.
  //
  // The 72-byte Dsv4MoePrefillCubeTilingData is unchanged and still at offset
  // 0; the Matmul framework's two TCubeTilings follow it. They have to reach
  // the device somehow, and REGIST_MATMUL_OBJ needs them as
  // `const __gm__ TCubeTiling*` -- see dsv4_moe_prefill_cube_cube_tiling.h for
  // what passing no tiling at all did instead (a null-pointer read storm on
  // the Cube that never terminated).
  //
  // This is NOT an operator ABI break: the published POD keeps its size and
  // its MOE_ARCH35_ASSERT_TILING_POD. The staging buffer is wrapper-private.
  alignas(8) unsigned char staging_blob[kPrefillCubeStagingBytes] = {};
  std::memcpy(staging_blob, &tiling, sizeof(tiling));
  const Status gate_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::gate_up, tiling.m_tile,
      staging_blob + kPrefillCubeGateTilingOffset);
  if (gate_tiling != Status::success) {
    return gate_tiling;
  }
  const Status down_tiling = MakeDsv4MoePrefillCubeGemmTiling(
      PrefillCubeGemm::down, tiling.m_tile,
      staging_blob + kPrefillCubeDownTilingOffset);
  if (down_tiling != Status::success) {
    return down_tiling;
  }

  detail::TilingStaging staging;
  const Status staged =
      staging.Stage(kCubeOp, staging_blob, sizeof(staging_blob));
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
