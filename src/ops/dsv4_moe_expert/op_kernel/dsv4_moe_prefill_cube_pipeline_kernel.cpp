// =================================================================================================
// dsv4_moe_prefill_cube_pipeline_kernel.cpp
//
// Batched DeepSeek-V4 MoE prefill with gate (W1), up (W3) and down (W2) contracted on the Cube in
// BF16. Target Ascend 950PR / arch35 / dav-3510, CANN 9.2, task mode KERNEL_TYPE_MIX_AIC_1_2
// (1 AIC + 2 AIV). No Matmul framework, no KfcServer, no GM workspace for intermediates: the
// activation plane is Unified-Buffer resident across the phase boundary.
//
// -------------------------------------------------------------------------------------------------
// STRUCTURE
// -------------------------------------------------------------------------------------------------
// Every routine below the state declarations is a PURE asynchronous execution body: it issues work
// on exactly one hardware pipeline and contains no SetFlag, WaitFlag, CrossCoreSetFlag,
// CrossCoreWaitFlag or inter-stage PipeBarrier. All pipeline synchronisation lives in the two outer
// schedulers, RunVectorCore and RunCubeCore, so the handshake protocol is readable in one place.
//
//     MTE2   AsyncLoadTokensChunk      AsyncLoadWeightsSubTile
//     V      CastTokensToBf16          DequantSubTileVector       ApplySwiGluVector
//     MTE3   PushUbTileToL1Nz          StoreSwiGluToActPlane      StoreLandingToTokensOut
//     MTE1   CubeLoadTokensToL0A       CubeLoadWeightsToL0B
//     M      CubeMmadAccumulate
//     FIX    CubeMoveOutDualDst
//
// The one exception is DequantSubTileVector, which keeps its four PipeBarrier<PIPE_V> fences: its
// scale expansion writes a uint16 view and reads a bfloat16 ReinterpretCast of the same scratch,
// and an alias the compiler's automatic synchronisation cannot see through must not be left to it.
// Those barriers order two instructions inside one routine on one pipe; they carry no pipeline
// stage.
//
// -------------------------------------------------------------------------------------------------
// LAYOUT MATH
// -------------------------------------------------------------------------------------------------
// Sub-tiles (op_kernel/dsv4_moe_prefill_cube_arch35.h):
//     phase 1   M = 16   N = R_SUB  = 128   K = K_SUB  = 128
//     phase 2   M = 16   N = H_SUB  =  64   K = R_SUB2 = 256
//
// Each of the two AIVs owns one CONTRACTION HALF of every step: K_SUB/2 = 64 for phase 1,
// R_SUB2/2 = 128 for phase 2. Both halves are whole multiples of the BF16 C0 block (16) and of the
// MXFP4 group (32), so the two cores' MTE3 bursts never share a C0 block and never split a group.
//
// NZ(nz_rows, C) is C/16 blocks of a dense [nz_rows][16] row-major run, blocks strided 16*nz_rows
// apart, so one block is nz_rows consecutive 32-byte units. PushUbTileToL1Nz issues one strided
// burst per block, which is what lets a core write either a COLUMN range (a contraction half) or a
// ROW range (a token half) of the same slot. The Nd2NzParams overload can express only the former.
//
// NZ(N, K) is bit-identical to ZN(K, N). The AIVs write a weight sub-tile as NZ(n_extent, k_extent)
// because the weights are stored [N][K] and the MXFP4 groups run along K; declaring that same
// memory as ZN(k_extent, n_extent) in CubeLoadWeightsToL0B therefore supplies B = W^T for no
// instruction, and CopyL12L0B takes its noTrans route.
//
// L1 ring: 2 slots x (A 4 KiB + gate 32 KiB + up 32 KiB) + 2 slots x (A2 8 KiB + down 32 KiB)
//          = 221,184 B, against 512 KiB of L1.
// UB:      125,472 B of staging plus the M/2 x inter_cap BF16 activation plane.
//
// -------------------------------------------------------------------------------------------------
// HARDWARE CONSTRAINTS ENCODED BELOW
// -------------------------------------------------------------------------------------------------
//   * An AIV's GetBlockIdx() counts AI VECTOR CORES, not AI Core groups. The group index is
//     GetBlockIdx() / GetSubBlockNum(); the raw index leaves the second AIV of a one-group grid
//     outside the tile loop, which deadlocks the mode 2 StepReady rendezvous. The AIC's
//     GetSubBlockNum() is 1, so its plain GetBlockIdx() is already the group index.
//   * Intra-core SetFlag/WaitFlag are BINARY flags, not counting semaphores. One event id per ring
//     slot, and one flag per group of in-order MTE2 bursts.
//   * Accumulating Mmad into one L0C needs PipeBarrier<PIPE_M> between consecutive calls whenever
//     (m/16)*(n/16) < 10. Phase 1 gives 1*8 and phase 2 gives 1*4, so the barrier is unconditional.
//   * The AIC -> AIV release is mode 4 point-to-point, flagId + sub * INTRA_NUM. Mode 2 for the
//     same release intermittently hangs.
//   * DUAL_DST_SPLIT_M bisects M across the two AIVs' separately addressed UBs, and the Fixpipe
//     destination layout must carry the FULL M, not M/2. The landing zone is UB offset 0 in each
//     AIV's own address space, so q_landing must be the first InitBuffer on the vector side.
//   * PIPE_FIX is unresolvable at namespace scope once tensor_api/tensor.h is included, so it is
//     named only inside function bodies.
//   * The first kCubeStartupBytes of the shared workspace must be ZERO at launch: the AIC spins on
//     them until both paired AIVs have signalled that ring slot 0 holds their operands. The host
//     wrapper zeroes them with a synchronous H2D copy.
// =================================================================================================

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

#if defined(ENABLE_KERNEL_PRINTF) && ENABLE_KERNEL_PRINTF
#include "utils/debug/asc_printf.h"
#define DSV4_PIPELINE_TRACE(...) __asc_aicore::printf(__VA_ARGS__)
#else
#define DSV4_PIPELINE_TRACE(...)
#endif

namespace moe::kernels::arch35::dsv4_moe_prefill_cube_pipeline {

using namespace AscendC;

namespace Te = AscendC::Te;

using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeAccBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeActInterCap;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeActPlaneBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeActSubBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeAivPerAic;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeC0;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeFlagLandingFree;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeFlagProduct;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeFlagStepFree;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeFlagStepReady;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeGroup;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeHSub;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeIntraFlagStride;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeIntraSyncMode;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeKHalf;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeKSub;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL0AElems;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL0BElems;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL0CElems;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL1A2SlotBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL1ASlotBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL1W2SlotBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeL1WSlotBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeLandingBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeLandingUbOffset;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeMHalf;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeMTile;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeNibHalfBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeRHalf2;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeRingSlots;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeRSub;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeRSub2;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeScaleBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeScaleRowBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeScaleRowStride;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeScaleTempBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeSyncMode;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeTokenF32Bytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeTokenSubBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeWeightStreams;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kCubeWideHalfBytes;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeMte1ToM;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeMToFix;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeMToMte1;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeSlotFull;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeSlotPushed;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeSlotStage;
using moe::kernels::arch35::dsv4_moe_prefill_cube::kEvCubeWideReady;

static_assert(kCubeMTile == kPrefillCubeMTile,          "token tile disagrees with the validated tiling header");
static_assert(kCubeRSub  == kPrefillCubeRSub,           "phase 1 inter output sub-tile disagrees");
static_assert(kCubeKSub  == kPrefillCubeKSub,           "phase 1 hidden contraction sub-tile disagrees");
static_assert(kCubeHSub  == kPrefillCubeHSub,           "phase 2 hidden output sub-tile disagrees");
static_assert(kCubeRSub2 == kPrefillCubeRSub2,          "phase 2 inter contraction sub-tile disagrees");
static_assert(kCubeGroup == kPrefillCubeGroupSize,      "MXFP4 group size disagrees");
static_assert(kCubeActInterCap == kPrefillCubeInterCap, "act plane capacity disagrees with the validated tiling header");

inline constexpr std::uint32_t kStreamGate          = 0;
inline constexpr std::uint32_t kStreamUp            = 1;
inline constexpr std::uint32_t kStreamDown          = 0;
inline constexpr std::uint32_t kStartupLineBytes    = 128;
inline constexpr std::uint32_t kCubeStartupBytes    = kCubeAivPerAic * kStartupLineBytes;

namespace {

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR ptr) {
    return reinterpret_cast<__gm__ std::uint8_t*>(ptr);
}

struct Geometry {
    std::uint32_t block_dim;
    std::uint32_t hidden;
    std::uint32_t inter;
    std::uint32_t num_experts;
    std::uint32_t total_tokens;
    std::uint32_t token_tile;
    std::uint32_t gate_up_packed_stride;
    std::uint32_t gate_up_scale_stride;
    std::uint32_t down_packed_stride;
    std::uint32_t down_scale_stride;
    float         swiglu_limit;
};

struct ExpertWeights {
    __gm__ std::uint8_t* gate_packed;
    __gm__ std::uint8_t* gate_scales;
    __gm__ std::uint8_t* up_packed;
    __gm__ std::uint8_t* up_scales;
    __gm__ std::uint8_t* down_packed;
    __gm__ std::uint8_t* down_scales;
};

struct WeightStage {
    LocalTensor<fp4x2_e2m1_t> weight_nibbles;
    LocalTensor<bfloat16_t>   weight_expanded;
    LocalTensor<uint8_t>      weight_scales;
    LocalTensor<bfloat16_t>   scale_plane;
    LocalTensor<uint16_t>     scale_temporary;
};

struct AivBuffers {
    LocalTensor<float>      landing;
    LocalTensor<bfloat16_t> act;
    LocalTensor<float>      gate_acc;
    LocalTensor<float>      up_acc;
    LocalTensor<bfloat16_t> act_sub;
    LocalTensor<float>      tokens_f32[kCubeRingSlots];
    LocalTensor<bfloat16_t> tokens_bf16[kCubeRingSlots];
    WeightStage             weights[kCubeRingSlots][kCubeWeightStreams];
};

struct AivRingState {
    bool stage_live[kCubeRingSlots];
    bool pushed_live[kCubeRingSlots];
};

__aicore__ inline Geometry LoadGeometry(const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data) {
    Geometry geom;
    geom.block_dim             = tiling_data->block_dim;
    geom.hidden                = tiling_data->hidden;
    geom.inter                 = tiling_data->inter;
    geom.num_experts           = tiling_data->num_experts;
    geom.total_tokens          = tiling_data->total_tokens;
    geom.token_tile            = tiling_data->m_tile;
    geom.gate_up_packed_stride = tiling_data->w13_packed_expert_stride;
    geom.gate_up_scale_stride  = tiling_data->w13_scale_expert_stride;
    geom.down_packed_stride    = tiling_data->w2_packed_expert_stride;
    geom.down_scale_stride     = tiling_data->w2_scale_expert_stride;
    geom.swiglu_limit          = tiling_data->swiglu_limit;
    return geom;
}

__aicore__ inline bool TilingRejected(const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data) {
    const std::uint32_t hidden = tiling_data->hidden;
    const std::uint32_t inter  = tiling_data->inter;
    return tiling_data->abi_version      != kTilingAbiVersion ||
           tiling_data->group_size       != kCubeGroup        ||
           tiling_data->low_nibble_first != 1u                ||
           tiling_data->num_experts      == 0u                ||
           tiling_data->total_tokens     == 0u                ||
           tiling_data->m_tile           != kCubeMTile        ||
           tiling_data->r_sub            != kCubeRSub         ||
           tiling_data->k_sub            != kCubeKSub         ||
           tiling_data->h_sub            != kCubeHSub         ||
           tiling_data->r_sub2           != kCubeRSub2        ||
           !Dsv4MoePrefillCubeAdmissible(hidden, inter);
}

__aicore__ inline bool TileAdmitted(GM_ADDR token_to_expert_lut, std::uint32_t token_row_start,
                                    std::uint32_t num_experts, std::uint32_t& expert_id_out) {
    const std::uint32_t expert_id = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert_lut)[token_row_start]);
    expert_id_out = expert_id;
    return expert_id < num_experts;
}

__aicore__ inline ExpertWeights ResolveExpert(GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
                                              GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
                                              GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
                                              std::uint32_t expert_id, const Geometry& geom) {
    const std::uint64_t expert = static_cast<std::uint64_t>(expert_id);
    ExpertWeights w;
    w.gate_packed = Bytes(gate_weights_packed) + expert * geom.gate_up_packed_stride;
    w.gate_scales = Bytes(gate_weight_scales)  + expert * geom.gate_up_scale_stride;
    w.up_packed   = Bytes(up_weights_packed)   + expert * geom.gate_up_packed_stride;
    w.up_scales   = Bytes(up_weight_scales)    + expert * geom.gate_up_scale_stride;
    w.down_packed = Bytes(down_weights_packed) + expert * geom.down_packed_stride;
    w.down_scales = Bytes(down_weight_scales)  + expert * geom.down_scale_stride;
    return w;
}

__aicore__ inline void AivResetRing(AivRingState& ring) {
    for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
        ring.stage_live[slot]  = false;
        ring.pushed_live[slot] = false;
    }
}

// ---------------------------------------------------------------------------------------------
// MTE2
// ---------------------------------------------------------------------------------------------
__aicore__ inline void AsyncLoadTokensChunk(
    const GlobalTensor<float>& tokens_in_gm, const LocalTensor<float>& dst,
    std::uint32_t token_row_start, std::uint32_t col_start, std::uint32_t col_count,
    std::uint32_t valid_rows, std::uint32_t matrix_cols) {
    DataCopyParams burst;
    burst.blockCount = static_cast<std::uint16_t>(valid_rows);
    burst.blockLen   = static_cast<std::uint16_t>((col_count * 4u) / 32u);
    burst.srcStride  = static_cast<std::uint16_t>(((matrix_cols - col_count) * 4u) / 32u);
    burst.dstStride  = 0;
    DataCopy(dst, tokens_in_gm[(static_cast<std::uint64_t>(token_row_start) * matrix_cols) + col_start], burst);
}

__aicore__ inline void AsyncLoadWeightsSubTile(
    __gm__ std::uint8_t* packed_weights, __gm__ std::uint8_t* scale_factors,
    std::uint32_t row_start, std::uint32_t col_start, std::uint32_t row_count,
    std::uint32_t col_count, std::uint32_t matrix_cols, const WeightStage& stage) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t row_groups     = matrix_cols / kCubeGroup;

    const std::uint64_t packed_offset = static_cast<std::uint64_t>(row_start) * (matrix_cols / 2u) + (col_start / 2u);
    GlobalTensor<fp4x2_e2m1_t> packed_gm;
    packed_gm.SetGlobalBuffer(reinterpret_cast<__gm__ fp4x2_e2m1_t*>(packed_weights + packed_offset),
                              static_cast<std::uint64_t>(row_count) * col_count);
    DataCopyParams packed_burst;
    packed_burst.blockCount = static_cast<std::uint16_t>(row_count);
    packed_burst.blockLen   = static_cast<std::uint16_t>((col_count / 2u) / 32u);
    packed_burst.srcStride  = static_cast<std::uint16_t>(((matrix_cols - col_count) / 2u) / 32u);
    packed_burst.dstStride  = 0;
    DataCopy(stage.weight_nibbles, packed_gm, packed_burst);

    const std::uint64_t scale_offset = static_cast<std::uint64_t>(row_start) * row_groups + (col_start / kCubeGroup);
    GlobalTensor<uint8_t> scale_gm;
    scale_gm.SetGlobalBuffer(scale_factors + scale_offset,
                             static_cast<std::uint64_t>(row_count) * groups_per_row);
    DataCopyExtParams scale_burst;
    scale_burst.blockCount = static_cast<std::uint16_t>(row_count);
    scale_burst.blockLen   = groups_per_row;
    scale_burst.srcStride  = row_groups - groups_per_row;
    scale_burst.dstStride  = 0;
    scale_burst.rsv        = 0;
    DataCopyPadExtParams<uint8_t> scale_pad;
    scale_pad.isPad        = false;
    scale_pad.leftPadding  = 0;
    scale_pad.rightPadding = 0;
    scale_pad.paddingValue = 0;
    DataCopyPad(stage.weight_scales, scale_gm, scale_burst, scale_pad);
}

// ---------------------------------------------------------------------------------------------
// V
// ---------------------------------------------------------------------------------------------
__aicore__ inline void DequantSubTileVector(
    std::uint32_t row_count, std::uint32_t col_count, const WeightStage& stage) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t total_elements = row_count * col_count;

    Cast(stage.weight_expanded, stage.weight_nibbles, RoundMode::CAST_NONE, total_elements);

    LocalTensor<uint16_t>   row_scales_u16  = stage.scale_temporary;
    LocalTensor<bfloat16_t> row_scales_bf16 = stage.scale_temporary.ReinterpretCast<bfloat16_t>();

    const uint32_t dst_shape[2] = {groups_per_row, kCubeGroup};
    const uint32_t src_shape[2] = {groups_per_row, 1};

    for (std::uint32_t row = 0; row < row_count; ++row) {
        const std::uint32_t scale_base = row * kCubeScaleRowStride;
        const std::uint32_t out_base   = row * col_count;

        Cast(row_scales_u16, stage.weight_scales[scale_base], RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        ShiftLeft(row_scales_u16, row_scales_u16, static_cast<uint16_t>(7), 16);
        PipeBarrier<PIPE_V>();
        Broadcast<bfloat16_t, 2, 1>(stage.scale_plane, row_scales_bf16, dst_shape, src_shape);
        PipeBarrier<PIPE_V>();
        Mul(stage.weight_expanded[out_base], stage.weight_expanded[out_base], stage.scale_plane, col_count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void CastTokensToBf16(const LocalTensor<bfloat16_t>& dst,
                                        const LocalTensor<float>& src, std::uint32_t count) {
    Cast(dst, src, RoundMode::CAST_RINT, count);
}

__aicore__ inline void ApplySwiGluVector(const AivBuffers& buf, std::uint32_t elems) {
    Silu(buf.gate_acc, buf.gate_acc, elems);
    Mul(buf.gate_acc, buf.gate_acc, buf.up_acc, elems);
    Cast(buf.act_sub, buf.gate_acc, RoundMode::CAST_RINT, elems);
}

// ---------------------------------------------------------------------------------------------
// MTE3
// ---------------------------------------------------------------------------------------------
__aicore__ inline void PushUbTileToL1Nz(
    const LocalTensor<bfloat16_t>& l1_slot, const LocalTensor<bfloat16_t>& ub_tile,
    std::uint32_t rows, std::uint32_t cols, std::uint32_t src_row_stride,
    std::uint32_t nz_rows, std::uint32_t col_base, std::uint32_t row_base) {
    DataCopyParams burst;
    burst.blockCount = static_cast<std::uint16_t>(rows);
    burst.blockLen   = 1;
    burst.srcStride  = static_cast<std::uint16_t>(src_row_stride / kCubeC0 - 1u);
    burst.dstStride  = 0;

    const std::uint32_t blocks = cols / kCubeC0;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        DataCopy(l1_slot[(col_base + block * kCubeC0) * nz_rows + row_base * kCubeC0],
                 ub_tile[block * kCubeC0], burst);
    }
}

__aicore__ inline void StoreSwiGluToActPlane(const AivBuffers& buf, std::uint32_t inter_base,
                                             std::uint32_t inter) {
    DataCopyParams burst;
    burst.blockCount = static_cast<std::uint16_t>(kCubeMHalf);
    burst.blockLen   = static_cast<std::uint16_t>((kCubeRSub * 2u) / 32u);
    burst.srcStride  = 0;
    burst.dstStride  = static_cast<std::uint16_t>(((inter - kCubeRSub) * 2u) / 32u);
    DataCopy(buf.act[inter_base], buf.act_sub, burst);
}

__aicore__ inline void StoreLandingToTokensOut(
    const GlobalTensor<float>& tokens_out_gm, const AivBuffers& buf,
    std::uint32_t token_row_start, std::uint32_t row_base, std::uint32_t hidden_base,
    std::uint32_t store_rows, std::uint32_t hidden) {
    DataCopyParams burst;
    burst.blockCount = static_cast<std::uint16_t>(store_rows);
    burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
    burst.srcStride  = 0;
    burst.dstStride  = static_cast<std::uint16_t>(((hidden - kCubeHSub) * 4u) / 32u);
    DataCopy(tokens_out_gm[(static_cast<std::uint64_t>(token_row_start + row_base) * hidden) + hidden_base],
             buf.landing, burst);
}

// ---------------------------------------------------------------------------------------------
// MTE1, M and FIX
// ---------------------------------------------------------------------------------------------
constexpr Te::CopyL0C2UBTrait kCubeL0c2UbTrait = {
    Te::RoundMode::DEFAULT,
    false,
    false,
    Te::DUAL_DST_SPLIT_M
};

struct CubeL0c2UbTraitDualDstM {
    using TraitType = Te::CopyL0C2UBTrait;
    static constexpr const TraitType value = kCubeL0c2UbTrait;
};

constexpr pipe_t kCubeWaitAicStep   = PIPE_MTE1;
constexpr pipe_t kCubeWaitAivStep   = PIPE_MTE3;
constexpr pipe_t kCubeWaitAivVector = PIPE_V;
constexpr pipe_t kCubeWaitAivStore  = PIPE_MTE3;

__aicore__ inline void CubeLoadTokensToL0A(
    __ca__ bfloat16_t* l0a_mem, const LocalTensor<bfloat16_t>& l1_tokens,
    std::uint32_t m_extent, std::uint32_t k_extent) {
    auto l1A = Te::MakeTensor(Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1_tokens.GetPhyAddr())),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    auto l0A = Te::MakeTensor(Te::MakeMemPtr(l0a_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0A{}, Te::CopyL12L0ATraitDefault{}), l0A, l1A);
}

__aicore__ inline void CubeLoadWeightsToL0B(
    __cb__ bfloat16_t* l0b_mem, const LocalTensor<bfloat16_t>& l1_weights,
    std::uint32_t k_extent, std::uint32_t n_extent) {
    auto l1W = Te::MakeTensor(Te::MakeMemPtr(reinterpret_cast<__cbuf__ bfloat16_t*>(l1_weights.GetPhyAddr())),
                              Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    auto l0B = Te::MakeTensor(Te::MakeMemPtr(l0b_mem),
                              Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    Te::Copy(Te::MakeCopy(Te::CopyL12L0B{}, Te::CopyL12L0BTraitDefault{}), l0B, l1W);
}

__aicore__ inline void CubeMmadAccumulate(
    __cc__ float* l0c_mem, __ca__ bfloat16_t* l0a_mem, __cb__ bfloat16_t* l0b_mem,
    std::uint32_t m_extent, std::uint32_t n_extent, std::uint32_t k_extent, bool init) {
    auto l0A = Te::MakeTensor(Te::MakeMemPtr(l0a_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, bfloat16_t>(m_extent, k_extent));
    auto l0B = Te::MakeTensor(Te::MakeMemPtr(l0b_mem),
                              Te::MakeFrameLayout<Te::ZNLayoutPtn, bfloat16_t>(k_extent, n_extent));
    auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(m_extent, n_extent));
    Te::MmadParams params(static_cast<uint16_t>(m_extent), static_cast<uint16_t>(n_extent),
                          static_cast<uint16_t>(k_extent), 0, init);
    Te::Mmad(Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{}).with(params), l0C, l0A, l0B);
}

__aicore__ inline void CubeMoveOutDualDst(
    __cc__ float* l0c_mem, std::uint32_t m_extent, std::uint32_t n_extent) {
    auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(m_extent, n_extent));
    auto ub  = Te::MakeTensor(Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kCubeLandingUbOffset)),
                              Te::MakeFrameLayout<Te::NDExtLayoutPtn>(m_extent, n_extent));
    Te::FixpipeParams params;
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(params), ub, l0C);
}

// ---------------------------------------------------------------------------------------------
// The startup rendezvous, through GM rather than a cross-core flag
// ---------------------------------------------------------------------------------------------
__aicore__ inline void StartupWord(GM_ADDR shared_workspace, std::uint32_t idx,
                                   GlobalTensor<int64_t>& word) {
    word.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(Bytes(shared_workspace) + idx * kStartupLineBytes), 1);
}

__aicore__ inline void AivSignalStartup(GM_ADDR shared_workspace, std::uint32_t sub_idx) {
    GlobalTensor<int64_t> word;
    StartupWord(shared_workspace, sub_idx, word);
    word.SetValue(0, 1);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(word);
}

__aicore__ inline void CubeWaitStartup(GM_ADDR shared_workspace) {
    GlobalTensor<int64_t> word[kCubeAivPerAic];
    for (std::uint32_t sub = 0; sub < kCubeAivPerAic; ++sub) {
        StartupWord(shared_workspace, sub, word[sub]);
    }
    for (;;) {
        bool all_ready = true;
        for (std::uint32_t sub = 0; sub < kCubeAivPerAic; ++sub) {
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(word[sub]);
            if (word[sub].GetValue(0) == 0) {
                all_ready = false;
            }
        }
        if (all_ready) {
            return;
        }
    }
}

}  // namespace

// =================================================================================================
// THE VECTOR CORE SCHEDULER -- every AIV-side synchronisation primitive lives here
// =================================================================================================
__aicore__ inline void RunVectorCore(
    const Geometry& geom, std::uint32_t sub_idx, GM_ADDR shared_workspace,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut,
    const GlobalTensor<float>& tokens_in_gm, const GlobalTensor<float>& tokens_out_gm,
    const AivBuffers& buf,
    TBuf<TPosition::A1> (&l1_tokens_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1_fwd_weights_buf)[kCubeRingSlots][kCubeWeightStreams],
    TBuf<TPosition::A1> (&l1_act_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1_down_weights_buf)[kCubeRingSlots]) {
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;
    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx()) /
                                      static_cast<std::uint32_t>(GetSubBlockNum());
    const std::uint32_t row_base    = sub_idx * kCubeMHalf;

    AivRingState ring;
    bool startup_signalled = false;

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;
        const std::uint32_t remaining       = geom.total_tokens - token_row_start;
        const std::uint32_t valid_rows      = remaining < geom.token_tile ? remaining : geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            continue;
        }
        const ExpertWeights expert = ResolveExpert(gate_weights_packed, gate_weight_scales,
                                                   up_weights_packed,   up_weight_scales,
                                                   down_weights_packed, down_weight_scales,
                                                   expert_id, geom);

        DSV4_PIPELINE_TRACE("[AIV %u] tile %u rows %u expert %u\n", sub_idx, tile_idx, valid_rows, expert_id);

        const std::uint32_t subgemm_fwd = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            AivResetRing(ring);

            AsyncLoadTokensChunk(tokens_in_gm, buf.tokens_f32[0], token_row_start,
                                 sub_idx * kCubeKHalf, kCubeKHalf, valid_rows, geom.hidden);
            AsyncLoadWeightsSubTile(expert.gate_packed, expert.gate_scales, inter_base,
                                    sub_idx * kCubeKHalf, kCubeRSub, kCubeKHalf, geom.hidden,
                                    buf.weights[0][kStreamGate]);
            AsyncLoadWeightsSubTile(expert.up_packed, expert.up_scales, inter_base,
                                    sub_idx * kCubeKHalf, kCubeRSub, kCubeKHalf, geom.hidden,
                                    buf.weights[0][kStreamUp]);
            SetFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[0]);

            for (std::uint32_t step = 0; step < subgemm_fwd; ++step) {
                const std::uint32_t slot      = step % kCubeRingSlots;
                const std::uint32_t next_slot = (step + 1u) % kCubeRingSlots;

                if (step + 1u < subgemm_fwd) {
                    if (ring.stage_live[next_slot]) {
                        WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[next_slot]);
                        ring.stage_live[next_slot] = false;
                    }
                    const std::uint32_t col_start = (step + 1u) * kCubeKSub + sub_idx * kCubeKHalf;
                    AsyncLoadTokensChunk(tokens_in_gm, buf.tokens_f32[next_slot], token_row_start,
                                         col_start, kCubeKHalf, valid_rows, geom.hidden);
                    AsyncLoadWeightsSubTile(expert.gate_packed, expert.gate_scales, inter_base,
                                            col_start, kCubeRSub, kCubeKHalf, geom.hidden,
                                            buf.weights[next_slot][kStreamGate]);
                    AsyncLoadWeightsSubTile(expert.up_packed, expert.up_scales, inter_base,
                                            col_start, kCubeRSub, kCubeKHalf, geom.hidden,
                                            buf.weights[next_slot][kStreamUp]);
                    SetFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[next_slot]);
                }

                if (step >= kCubeRingSlots) {
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                if (ring.pushed_live[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                    ring.pushed_live[slot] = false;
                }

                WaitFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[slot]);
                DequantSubTileVector(kCubeRSub, kCubeKHalf, buf.weights[slot][kStreamGate]);
                DequantSubTileVector(kCubeRSub, kCubeKHalf, buf.weights[slot][kStreamUp]);
                CastTokensToBf16(buf.tokens_bf16[slot], buf.tokens_f32[slot], kCubeMTile * kCubeKHalf);

                SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
                ring.stage_live[slot] = true;

                SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
                WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);

                const std::uint32_t col_base = sub_idx * kCubeKHalf;
                PushUbTileToL1Nz(l1_tokens_buf[slot].Get<bfloat16_t>(), buf.tokens_bf16[slot],
                                 kCubeMTile, kCubeKHalf, kCubeKHalf, kCubeMTile, col_base, 0);
                PushUbTileToL1Nz(l1_fwd_weights_buf[slot][kStreamGate].Get<bfloat16_t>(),
                                 buf.weights[slot][kStreamGate].weight_expanded,
                                 kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
                PushUbTileToL1Nz(l1_fwd_weights_buf[slot][kStreamUp].Get<bfloat16_t>(),
                                 buf.weights[slot][kStreamUp].weight_expanded,
                                 kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);

                SetFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                ring.pushed_live[slot] = true;

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));

                if (!startup_signalled) {
                    AivSignalStartup(shared_workspace, sub_idx);
                    startup_signalled = true;
                }
            }

            for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
                if (ring.stage_live[slot]) {
                    WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
                    ring.stage_live[slot] = false;
                }
                if (ring.pushed_live[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                    ring.pushed_live[slot] = false;
                }
            }

            const std::uint32_t elems = kCubeMHalf * kCubeRSub;

            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            Mins(buf.gate_acc, buf.landing, geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);

            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivVector>(kCubeFlagProduct);
            Maxs(buf.up_acc, buf.landing, -geom.swiglu_limit, elems);
            CrossCoreSetFlag<kCubeSyncMode, PIPE_V>(kCubeFlagLandingFree);
            Mins(buf.up_acc, buf.up_acc, geom.swiglu_limit, elems);

            ApplySwiGluVector(buf, elems);

            SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
            WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
            StoreSwiGluToActPlane(buf, inter_base, geom.inter);
        }

        const std::uint32_t subgemm_down = geom.inter / kCubeRSub2;
        const std::uint32_t owned        = valid_rows > row_base ? (valid_rows - row_base) : 0u;
        const std::uint32_t store_rows   = owned < kCubeMHalf ? owned : kCubeMHalf;

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            AivResetRing(ring);

            AsyncLoadWeightsSubTile(expert.down_packed, expert.down_scales, hidden_base,
                                    sub_idx * kCubeRHalf2, kCubeHSub, kCubeRHalf2, geom.inter,
                                    buf.weights[0][kStreamDown]);
            SetFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[0]);

            for (std::uint32_t step = 0; step < subgemm_down; ++step) {
                const std::uint32_t slot      = step % kCubeRingSlots;
                const std::uint32_t next_slot = (step + 1u) % kCubeRingSlots;

                if (step + 1u < subgemm_down) {
                    if (ring.stage_live[next_slot]) {
                        WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[next_slot]);
                        ring.stage_live[next_slot] = false;
                    }
                    const std::uint32_t col_start = (step + 1u) * kCubeRSub2 + sub_idx * kCubeRHalf2;
                    AsyncLoadWeightsSubTile(expert.down_packed, expert.down_scales, hidden_base,
                                            col_start, kCubeHSub, kCubeRHalf2, geom.inter,
                                            buf.weights[next_slot][kStreamDown]);
                    SetFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[next_slot]);
                }

                if (step >= kCubeRingSlots) {
                    CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStep>(
                        static_cast<uint16_t>(kCubeFlagStepFree + slot));
                }

                if (ring.pushed_live[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                    ring.pushed_live[slot] = false;
                }

                WaitFlag<HardEvent::MTE2_V>(kEvCubeSlotFull[slot]);
                DequantSubTileVector(kCubeHSub, kCubeRHalf2, buf.weights[slot][kStreamDown]);

                SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
                ring.stage_live[slot] = true;

                SetFlag<HardEvent::V_MTE3>(kEvCubeWideReady);
                WaitFlag<HardEvent::V_MTE3>(kEvCubeWideReady);

                PushUbTileToL1Nz(l1_down_weights_buf[slot].Get<bfloat16_t>(),
                                 buf.weights[slot][kStreamDown].weight_expanded,
                                 kCubeHSub, kCubeRHalf2, kCubeRHalf2, kCubeHSub,
                                 sub_idx * kCubeRHalf2, 0);
                PushUbTileToL1Nz(l1_act_buf[slot].Get<bfloat16_t>(), buf.act[step * kCubeRSub2],
                                 kCubeMHalf, kCubeRSub2, geom.inter, kCubeMTile, 0,
                                 sub_idx * kCubeMHalf);

                SetFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                ring.pushed_live[slot] = true;

                CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));
            }

            for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
                if (ring.stage_live[slot]) {
                    WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
                    ring.stage_live[slot] = false;
                }
                if (ring.pushed_live[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(kEvCubeSlotPushed[slot]);
                    ring.pushed_live[slot] = false;
                }
            }

            CrossCoreWaitFlag<kCubeIntraSyncMode, kCubeWaitAivStore>(kCubeFlagProduct);
            if (store_rows > 0u) {
                StoreLandingToTokensOut(tokens_out_gm, buf, token_row_start, row_base,
                                        hidden_base, store_rows, geom.hidden);
            }
            CrossCoreSetFlag<kCubeSyncMode, PIPE_MTE3>(kCubeFlagLandingFree);
        }
    }
}

// =================================================================================================
// THE CUBE CORE SCHEDULER -- every AIC-side synchronisation primitive lives here
// =================================================================================================
__aicore__ inline void RunCubeCore(
    const Geometry& geom, GM_ADDR token_to_expert_lut, GM_ADDR shared_workspace,
    TBuf<TPosition::A1> (&l1_tokens_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1_fwd_weights_buf)[kCubeRingSlots][kCubeWeightStreams],
    TBuf<TPosition::A1> (&l1_act_buf)[kCubeRingSlots],
    TBuf<TPosition::B1> (&l1_down_weights_buf)[kCubeRingSlots]) {
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;
    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx());

    __ca__ bfloat16_t l0a[kCubeL0AElems];
    __cb__ bfloat16_t l0b_gate[kCubeL0BElems];
    __cb__ bfloat16_t l0b_up[kCubeL0BElems];
    __cc__ float      l0c_gate[kCubeL0CElems];
    __cc__ float      l0c_up[kCubeL0CElems];

    bool startup_awaited = false;

    for (std::uint32_t tile_idx = core_idx; tile_idx < total_tiles; tile_idx += geom.block_dim) {
        const std::uint32_t token_row_start = tile_idx * geom.token_tile;

        std::uint32_t expert_id = 0;
        if (!TileAdmitted(token_to_expert_lut, token_row_start, geom.num_experts, expert_id)) {
            continue;
        }

        DSV4_PIPELINE_TRACE("[AIC] tile %u expert %u\n", tile_idx, expert_id);

        const std::uint32_t subgemm_fwd = geom.hidden / kCubeKSub;

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += kCubeRSub) {
            for (std::uint32_t step = 0; step < subgemm_fwd; ++step) {
                const std::uint32_t slot = step % kCubeRingSlots;

                if (!startup_awaited) {
                    CubeWaitStartup(shared_workspace);
                    startup_awaited = true;
                }

                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));

                CubeLoadTokensToL0A(l0a, l1_tokens_buf[slot].Get<bfloat16_t>(), kCubeMTile, kCubeKSub);
                CubeLoadWeightsToL0B(l0b_gate, l1_fwd_weights_buf[slot][kStreamGate].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);
                CubeLoadWeightsToL0B(l0b_up, l1_fwd_weights_buf[slot][kStreamUp].Get<bfloat16_t>(),
                                     kCubeKSub, kCubeRSub);

                if (step + kCubeRingSlots < subgemm_fwd) {
                    for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
                        CrossCoreSetFlag<kCubeIntraSyncMode, PIPE_MTE1>(
                            static_cast<uint16_t>(kCubeFlagStepFree + slot + sub * kCubeIntraFlagStride));
                    }
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate, kCubeMTile, kCubeRSub, kCubeKSub, step == 0u);
                CubeMmadAccumulate(l0c_up, l0a, l0b_up, kCubeMTile, kCubeRSub, kCubeKSub, step == 0u);

                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
                WaitFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);

            CubeMoveOutDualDst(l0c_gate, kCubeMTile, kCubeRSub);
            PipeBarrier<PIPE_FIX>();
            for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
                CrossCoreSetFlag<kCubeIntraSyncMode, PIPE_FIX>(
                    static_cast<uint16_t>(kCubeFlagProduct + sub * kCubeIntraFlagStride));
            }
            CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);

            CubeMoveOutDualDst(l0c_up, kCubeMTile, kCubeRSub);
            PipeBarrier<PIPE_FIX>();
            for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
                CrossCoreSetFlag<kCubeIntraSyncMode, PIPE_FIX>(
                    static_cast<uint16_t>(kCubeFlagProduct + sub * kCubeIntraFlagStride));
            }
            CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
        }

        const std::uint32_t subgemm_down = geom.inter / kCubeRSub2;

        for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += kCubeHSub) {
            for (std::uint32_t step = 0; step < subgemm_down; ++step) {
                const std::uint32_t slot = step % kCubeRingSlots;

                CrossCoreWaitFlag<kCubeSyncMode, kCubeWaitAicStep>(
                    static_cast<uint16_t>(kCubeFlagStepReady + slot));

                CubeLoadTokensToL0A(l0a, l1_act_buf[slot].Get<bfloat16_t>(), kCubeMTile, kCubeRSub2);
                CubeLoadWeightsToL0B(l0b_gate, l1_down_weights_buf[slot].Get<bfloat16_t>(),
                                     kCubeRSub2, kCubeHSub);

                if (step + kCubeRingSlots < subgemm_down) {
                    for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
                        CrossCoreSetFlag<kCubeIntraSyncMode, PIPE_MTE1>(
                            static_cast<uint16_t>(kCubeFlagStepFree + slot + sub * kCubeIntraFlagStride));
                    }
                }

                SetFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);
                WaitFlag<HardEvent::MTE1_M>(kEvCubeMte1ToM);

                CubeMmadAccumulate(l0c_gate, l0a, l0b_gate, kCubeMTile, kCubeHSub, kCubeRSub2, step == 0u);

                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
                WaitFlag<HardEvent::M_MTE1>(kEvCubeMToMte1);
            }

            SetFlag<HardEvent::M_FIX>(kEvCubeMToFix);
            WaitFlag<HardEvent::M_FIX>(kEvCubeMToFix);

            CubeMoveOutDualDst(l0c_gate, kCubeMTile, kCubeHSub);
            PipeBarrier<PIPE_FIX>();
            for (std::uint16_t sub = 0; sub < kCubeAivPerAic; ++sub) {
                CrossCoreSetFlag<kCubeIntraSyncMode, PIPE_FIX>(
                    static_cast<uint16_t>(kCubeFlagProduct + sub * kCubeIntraFlagStride));
            }
            CrossCoreWaitFlag<kCubeSyncMode, PIPE_FIX>(kCubeFlagLandingFree);
        }
    }
}

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_pipeline(
    GM_ADDR tokens_in,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut, GM_ADDR tokens_out,
    GM_ADDR shared_workspace,    GM_ADDR tiling_config) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (unlikely(tiling_config == nullptr || shared_workspace == nullptr)) {
        return;
    }
    if (unlikely(tokens_in == nullptr || tokens_out == nullptr || token_to_expert_lut == nullptr)) {
        return;
    }
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data =
        reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
    if (unlikely(TilingRejected(tiling_data))) {
        return;
    }
    const Geometry geom = LoadGeometry(tiling_data);

    TPipe pipe;
    TBuf<TPosition::A1> l1_tokens_buf[kCubeRingSlots];
    TBuf<TPosition::B1> l1_fwd_weights_buf[kCubeRingSlots][kCubeWeightStreams];
    TBuf<TPosition::A1> l1_act_buf[kCubeRingSlots];
    TBuf<TPosition::B1> l1_down_weights_buf[kCubeRingSlots];
    for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
        pipe.InitBuffer(l1_tokens_buf[slot], kCubeL1ASlotBytes);
    }
    for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
        for (std::uint32_t stream = 0; stream < kCubeWeightStreams; ++stream) {
            pipe.InitBuffer(l1_fwd_weights_buf[slot][stream], kCubeL1WSlotBytes);
        }
    }
    for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
        pipe.InitBuffer(l1_act_buf[slot], kCubeL1A2SlotBytes);
    }
    for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
        pipe.InitBuffer(l1_down_weights_buf[slot], kCubeL1W2SlotBytes);
    }

    if ASCEND_IS_AIC {
        RunCubeCore(geom, token_to_expert_lut, shared_workspace,
                    l1_tokens_buf, l1_fwd_weights_buf, l1_act_buf, l1_down_weights_buf);
    }

    if ASCEND_IS_AIV {
        GlobalTensor<float> tokens_in_gm;
        GlobalTensor<float> tokens_out_gm;
        const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
        tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
        tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

        TQue<TPosition::VECIN,   1> q_landing;
        TQue<TPosition::VECCALC, 1> q_act;
        TQue<TPosition::VECCALC, 1> q_gate_acc;
        TQue<TPosition::VECCALC, 1> q_up_acc;
        TQue<TPosition::VECOUT,  1> q_act_sub;
        TQue<TPosition::VECCALC, 1> q_tokens_f32[kCubeRingSlots];
        TQue<TPosition::VECOUT,  1> q_tokens_bf16[kCubeRingSlots];
        TQue<TPosition::VECCALC, 1> q_nibbles[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_scales[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_expanded[kCubeRingSlots][kCubeWeightStreams];
        TQue<TPosition::VECCALC, 1> q_scale_plane;
        TQue<TPosition::VECCALC, 1> q_scale_temp;

        pipe.InitBuffer(q_landing,  1, kCubeLandingBytes);
        pipe.InitBuffer(q_act,      1, kCubeActPlaneBytes);
        pipe.InitBuffer(q_gate_acc, 1, kCubeAccBytes);
        pipe.InitBuffer(q_up_acc,   1, kCubeAccBytes);
        pipe.InitBuffer(q_act_sub,  1, kCubeActSubBytes);
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            pipe.InitBuffer(q_tokens_f32[slot], 1, kCubeTokenF32Bytes);
        }
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            pipe.InitBuffer(q_tokens_bf16[slot], 1, kCubeTokenSubBytes);
        }
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            for (std::uint32_t stream = 0; stream < kCubeWeightStreams; ++stream) {
                pipe.InitBuffer(q_nibbles[slot][stream], 1, kCubeNibHalfBytes);
            }
        }
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            for (std::uint32_t stream = 0; stream < kCubeWeightStreams; ++stream) {
                pipe.InitBuffer(q_scales[slot][stream], 1, kCubeScaleBytes);
            }
        }
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            for (std::uint32_t stream = 0; stream < kCubeWeightStreams; ++stream) {
                pipe.InitBuffer(q_expanded[slot][stream], 1, kCubeWideHalfBytes);
            }
        }
        pipe.InitBuffer(q_scale_plane, 1, kCubeScaleRowBytes);
        pipe.InitBuffer(q_scale_temp,  1, kCubeScaleTempBytes);

        AivBuffers buf;
        buf.landing  = q_landing.AllocTensor<float>();
        buf.act      = q_act.AllocTensor<bfloat16_t>();
        buf.gate_acc = q_gate_acc.AllocTensor<float>();
        buf.up_acc   = q_up_acc.AllocTensor<float>();
        buf.act_sub  = q_act_sub.AllocTensor<bfloat16_t>();

        LocalTensor<bfloat16_t> scale_plane = q_scale_plane.AllocTensor<bfloat16_t>();
        LocalTensor<uint16_t>   scale_temp  = q_scale_temp.AllocTensor<uint16_t>();

        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            buf.tokens_f32[slot]  = q_tokens_f32[slot].AllocTensor<float>();
            buf.tokens_bf16[slot] = q_tokens_bf16[slot].AllocTensor<bfloat16_t>();
            for (std::uint32_t stream = 0; stream < kCubeWeightStreams; ++stream) {
                buf.weights[slot][stream].weight_nibbles  = q_nibbles[slot][stream].AllocTensor<fp4x2_e2m1_t>();
                buf.weights[slot][stream].weight_scales   = q_scales[slot][stream].AllocTensor<uint8_t>();
                buf.weights[slot][stream].weight_expanded = q_expanded[slot][stream].AllocTensor<bfloat16_t>();
                buf.weights[slot][stream].scale_plane     = scale_plane;
                buf.weights[slot][stream].scale_temporary = scale_temp;
            }
        }

        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            Duplicate(buf.tokens_f32[slot], 0.0F, kCubeMTile * kCubeKHalf);
        }
        for (std::uint32_t slot = 0; slot < kCubeRingSlots; ++slot) {
            SetFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
            WaitFlag<HardEvent::V_MTE2>(kEvCubeSlotStage[slot]);
        }

        const std::uint32_t sub_idx = static_cast<std::uint32_t>(GetSubBlockIdx()) %
                                      static_cast<std::uint32_t>(GetSubBlockNum());

        RunVectorCore(geom, sub_idx, shared_workspace,
                      gate_weights_packed, gate_weight_scales,
                      up_weights_packed,   up_weight_scales,
                      down_weights_packed, down_weight_scales,
                      token_to_expert_lut, tokens_in_gm, tokens_out_gm, buf,
                      l1_tokens_buf, l1_fwd_weights_buf, l1_act_buf, l1_down_weights_buf);

        for (std::uint32_t slot = kCubeRingSlots; slot-- > 0;) {
            for (std::uint32_t stream = kCubeWeightStreams; stream-- > 0;) {
                q_expanded[slot][stream].FreeTensor(buf.weights[slot][stream].weight_expanded);
                q_scales[slot][stream].FreeTensor(buf.weights[slot][stream].weight_scales);
                q_nibbles[slot][stream].FreeTensor(buf.weights[slot][stream].weight_nibbles);
            }
            q_tokens_bf16[slot].FreeTensor(buf.tokens_bf16[slot]);
            q_tokens_f32[slot].FreeTensor(buf.tokens_f32[slot]);
        }
        q_scale_temp.FreeTensor(scale_temp);
        q_scale_plane.FreeTensor(scale_plane);
        q_act_sub.FreeTensor(buf.act_sub);
        q_up_acc.FreeTensor(buf.up_acc);
        q_gate_acc.FreeTensor(buf.gate_acc);
        q_act.FreeTensor(buf.act);
        q_landing.FreeTensor(buf.landing);
    }
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube_pipeline
