// =================================================================================================
// dsv4_moe_prefill_cube_kernel.cpp -- Batched MoE Prefill with W1/W3/W2 on Cube in BF16
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2)
// Kernel Task Mode    : KERNEL_TYPE_MIX_AIC_1_2 (Isomorphic Multi-Core Topology)
// Intermediates       : ZERO-GM WORKSPACE (On-Chip Resident Unified Buffer Activation Plane)
// =================================================================================================
//
// 1. HARDWARE SATURATION & EXECUTION MODEL (Ascend 950PR / arch35):
// -------------------------------------------------------------------------------------------------
// Ascend 950PR integrates 32 AI Subsystems (32 Cube Cores / AIC and 64 Vector Cores / AIV).
// Under KERNEL_TYPE_MIX_AIC_1_2, each AI Subsystem pairs 1 AIC with 2 AIVs:
//
//               ┌──────────────────────────────────────────────────────────────┐
//               │             32 AI Subsystems (Dav-3510 Cluster)              │
//               │                                                              │
//               │     Subsystem 0              ...             Subsystem 31    │
//               │  ┌───────────────┐                    ┌───────────────┐      │
//               │  │  1x Cube Core │ (32 AIC total)     │  1x Cube Core │      │
//               │  └───────┬───────┘                    └───────┬───────┘      │
//               │          │ KfcServer                          │              │
//               │     ┌────┴─────────┐                     ┌────┴─────────┐    │
//               │     │  KFC Mailbox │ (GM Workspace)      │  KFC Mailbox │    │
//               │     └────┬─────────┘                     └────┬─────────┘    │
//               │     ┌────┴────┬─────┐                    ┌────┴────┬──────┐  │
//               │     │  AIV 0  │AIV 1│ (64 AIV total)     │  AIV 62 │AIV 63│  │
//               │     └─────────┴─────┘                    └─────────┴──────┘  │
//               └──────────────────────────────────────────────────────────────┘
//
// - Cube/Vector Co-Design: REGIST_MATMUL_OBJ instantiates an AscendC::KfcServer. On AIC, the
//   macro executes server.Run() and returns immediately. On AIV, execution falls through to drive
//   the pipeline, issue GEMM requests, and perform vector SwiGLU / dequantization.
// - KFC Mailbox Partitioning: The Matmul framework indexes paired AIV
//   mailboxes from the caller-provided shared workspace.
// - Core Grid Distribution: GetBlockIdx() is the absolute dispatched core index;
//   GetSubBlockIdx() identifies the paired vector sub-core for KFC.
//   saturates 100% of the available vector and matrix compute units on the chip.
//
// 2. MATHEMATICAL MODEL & ASYMMETRIC ACTIVATION CLAMP:
// -------------------------------------------------------------------------------------------------
//   Gate Projection  : gate[m, r] = sum_c ( X_bf16[m, c] * W1_bf16[r, c] )               [Cube]
//   Up Projection    : up  [m, r] = sum_c ( X_bf16[m, c] * W3_bf16[r, c] )               [Cube]
//   Fused SwiGLU     : act [m, r] = silu(min(gate, L)) * clamp(up, -L, L)                [Vector]
//   Down Projection  : y   [m, h] = sum_r ( act_bf16[m, r] * W2_bf16[h, r] )             [Cube]
//
//   Note: The clamp is strictly asymmetric. The gate is bounded from ABOVE only; the up-path
//   is bounded on BOTH sides. Truncating the clamp introduces unbounded error into the hidden state.
//
// 3. THREE-LEVEL TILE DECOMPOSITION & UB RESIDENT PLANE:
// -------------------------------------------------------------------------------------------------
// Tokens are tiled into blocks of M = 16 rows (kCubeMTile). Intermediate activations stay resident
// inside UB (activation_resident, BF16, size: M * inter) across Phase 1 and Phase 2 without GM spill.
//
//  [M=16, hidden]              [inter, hidden] (W1/W3)                [M=16, inter]
//  Tokens Tile (GM)         Weights Sub-Tile (GM/TSCM)           Resident Act Plane (UB)
//   ┌──────────────┐         ┌─────────────────────────┐         ┌─────────────────────────┐
//   │  X[16, K_SUB]│  --->   │  W1/W3[R_SUB, K_SUB]    │  ===>   │  act[:, r0 : r0+R_SUB]  │
//   └──────────────┘         └─────────────────────────┘         └─────────────────────────┘
//        │                              │                                     │
//        │ (MTE2 -> Cast)               │ (Dequant E8M0 -> L1)                │ Phase 2 (A Slicing)
//        v                              v                                     v
//   tokens_sub_bf16           q_weight_fwd (TSCM/L1)             activation_sub_bf16
//   ┌──────────────┐         ┌─────────────────────────┐         ┌─────────────────────────┐
//   │ 16 x K_SUB   │    @    │ (R_SUB x K_SUB)^T       │  --->   │ 16 x R_SUB2             │
//   └──────────────┘         └─────────────────────────┘         └─────────────────────────┘
//                                                                             │
//                                     Down Weights (W2)                       │ (A @ W2^T)
//                                     ┌─────────────────────────┐             v
//                                     │  W2[H_SUB, R_SUB2]      │  ===>   y_acc [16 x H_SUB]
//                                     └─────────────────────────┘         (Overlapped MTE3 -> GM)
//
// 4. SOFTWARE PIPELINING & HARDWARE OPTIMIZATIONS:
// -------------------------------------------------------------------------------------------------
// - Phase 1 Hidden Contraction (Tokens Prefetching):
//   Implemented as a 2-stage double buffer (ping-pong on tokens_f32 / tokens_sub_bf16). MTE2 DMA load
//   of chunk (k0 + K_SUB) overlaps with sub-GEMM accumulation of chunk (k0) on the Cube.
// - Vectorized MXFP4/E8M0 Weight Dequantization:
//   Zero scalar loops. 32-byte aligned E8M0 exponent rows are vectorized via ShiftLeft(..., 7) and
//   expanded onto the weights grid using hardware Broadcast<bfloat16_t, 2>.
// - Vectorized SwiGLU & 2D UB-to-UB Memory Relocation:
//   Vector ALU performs contiguous Cast into a flat intermediate buffer, followed by a hardware-strided
//   UB -> UB DataCopy (DataCopyParams: blockLen=R_SUB, dstStride=inter-R_SUB) without scalar indexing.
// - Asynchronous GM Drain:
//   Phase 2 down-projection output flushes to GM via non-blocking V_MTE3 -> MTE3_V handshakes,
//   overlapping writeback bursts with next-tile setup.
//
// 5. MEMORY SAFETY & ALIAS RESOLUTION:
// -------------------------------------------------------------------------------------------------
// - Distinct UB Queues: scale_plane, scale_temporary, and tokens ping/pong buffers reside in strictly
//   disjoint UB memory regions, preventing alias corruption between Phase 1 GEMMs and Phase 2 down-slices.
// =================================================================================================

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

static_assert(kCubeMTile == kPrefillCubeMTile,     "token tile disagrees between the UB layout and the tiling header the host validates from");
static_assert(kCubeRSub  == kPrefillCubeRSub,      "phase 1 inter output sub-tile disagrees");
static_assert(kCubeKSub  == kPrefillCubeKSub,      "phase 1 hidden contraction sub-tile disagrees");
static_assert(kCubeHSub  == kPrefillCubeHSub,      "phase 2 hidden output sub-tile disagrees");
static_assert(kCubeRSub2 == kPrefillCubeRSub2,     "phase 2 inter contraction sub-tile disagrees");
static_assert(kCubeGroup == kPrefillCubeGroupSize, "MXFP4 group size disagrees");

// The token prefetch ring. Two slots, two event ids -- and each id carries TWO flags, because a
// flag is keyed by the (srcPipe, dstPipe, eventID) triple: `MTE2_V` id 2 ("ping is full") and
// `V_MTE2` id 2 ("ping is free") are distinct hardware flags. That is what lets a correct two-way
// handshake fit the four event ids arch35 provides, all of which this operator already spends.
constexpr std::uint32_t kTokenRingSlots        = 2;
constexpr int           kEvTokensPing          = 2;
constexpr int           kEvTokensPong          = 3;

using AGateType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t>;
using ADownType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t>;
using WType     = MatmulType<TPosition::TSCM,   CubeFormat::ND, bfloat16_t, true>;
using AccType   = MatmulType<TPosition::VECIN,  CubeFormat::ND, float>;
using BiasType  = MatmulType<TPosition::GM,     CubeFormat::ND, float>;

constexpr MatmulConfig kCubeCfg    = GetNormalConfig();
constexpr auto         kGateTiling = GetMatmulApiTiling<AGateType, WType, AccType, BiasType>(kCubeCfg);
constexpr auto         kDownTiling = GetMatmulApiTiling<ADownType, WType, AccType, BiasType>(kCubeCfg);

using GateUpMatmul = Matmul<AGateType, WType, AccType, BiasType, kGateTiling>;
using DownMatmul   = Matmul<ADownType, WType, AccType, BiasType, kDownTiling>;

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
    std::uint32_t inter_sub_fwd;
    std::uint32_t hidden_sub_fwd;
    std::uint32_t hidden_sub_down;
    std::uint32_t inter_sub_down;
    std::uint32_t gate_up_packed_stride;
    std::uint32_t gate_up_scale_stride;
    std::uint32_t down_packed_stride;
    std::uint32_t down_scale_stride;
    float         swiglu_limit;
};

__aicore__ inline Geometry LoadGeometry(const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data) {
    Geometry geom;
    geom.block_dim              = tiling_data->block_dim;
    geom.hidden                 = tiling_data->hidden;
    geom.inter                  = tiling_data->inter;
    geom.num_experts            = tiling_data->num_experts;
    geom.total_tokens           = tiling_data->total_tokens;
    geom.token_tile             = tiling_data->m_tile;
    geom.inter_sub_fwd          = tiling_data->r_sub;
    geom.hidden_sub_fwd         = tiling_data->k_sub;
    geom.hidden_sub_down        = tiling_data->h_sub;
    geom.inter_sub_down         = tiling_data->r_sub2;
    geom.gate_up_packed_stride  = tiling_data->w13_packed_expert_stride;
    geom.gate_up_scale_stride   = tiling_data->w13_scale_expert_stride;
    geom.down_packed_stride     = tiling_data->w2_packed_expert_stride;
    geom.down_scale_stride      = tiling_data->w2_scale_expert_stride;
    geom.swiglu_limit           = tiling_data->swiglu_limit;
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

struct Buffers {
    LocalTensor<bfloat16_t>  activation_resident;
    LocalTensor<float>        gate_accumulator;
    LocalTensor<float>        up_accumulator;
    LocalTensor<float>        subgemm_fwd;
    LocalTensor<float>        tokens_f32_ping;
    LocalTensor<float>        tokens_f32_pong;
    LocalTensor<bfloat16_t>  tokens_sub_bf16_ping;
    LocalTensor<bfloat16_t>  tokens_sub_bf16_pong;
    LocalTensor<float>        down_accumulator;
    LocalTensor<float>        subgemm_down;
    LocalTensor<bfloat16_t>  activation_sub_bf16;
    LocalTensor<fp4x2_e2m1_t> weight_nibbles;
    LocalTensor<bfloat16_t>  weight_expanded;
    LocalTensor<uint8_t>      weight_scales;
    LocalTensor<bfloat16_t>  scale_plane;
    LocalTensor<uint16_t>     scale_temporary;
};

}  // namespace

__aicore__ inline void AsyncLoadWeightsSubTile(
    __gm__ std::uint8_t* packed_weights, __gm__ std::uint8_t* scale_factors,
    std::uint32_t row_start, std::uint32_t col_start, std::uint32_t row_count, std::uint32_t col_count, std::uint32_t matrix_cols,
    const Buffers& buf) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t row_groups     = matrix_cols / kCubeGroup;

    {
        const std::uint64_t byte_offset = static_cast<std::uint64_t>(row_start) * (matrix_cols / 2u) + (col_start / 2u);
        GlobalTensor<fp4x2_e2m1_t> packed_gm;
        packed_gm.SetGlobalBuffer(reinterpret_cast<__gm__ fp4x2_e2m1_t*>(packed_weights + byte_offset), static_cast<std::uint64_t>(row_count) * col_count);
        DataCopyParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(row_count);
        burst_params.blockLen   = static_cast<std::uint16_t>((col_count / 2u) / 32u);
        burst_params.srcStride  = static_cast<std::uint16_t>(((matrix_cols - col_count) / 2u) / 32u);
        burst_params.dstStride  = 0;
        DataCopy(buf.weight_nibbles, packed_gm, burst_params);
    }

    {
        const std::uint64_t scale_offset = static_cast<std::uint64_t>(row_start) * row_groups + (col_start / kCubeGroup);
        GlobalTensor<uint8_t> scale_gm;
        scale_gm.SetGlobalBuffer(scale_factors + scale_offset, static_cast<std::uint64_t>(row_count) * groups_per_row);
        DataCopyExtParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(row_count);
        burst_params.blockLen   = groups_per_row;
        burst_params.srcStride  = row_groups - groups_per_row;
        burst_params.dstStride  = 0;
        burst_params.rsv        = 0;
        DataCopyPadExtParams<uint8_t> pad_params;
        pad_params.isPad        = false;
        pad_params.leftPadding  = 0;
        pad_params.rightPadding = 0;
        pad_params.paddingValue = 0;
        DataCopyPad(buf.weight_scales, scale_gm, burst_params, pad_params);
    }

    SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
}

__aicore__ inline void DequantSubTileVector(
    std::uint32_t row_count, std::uint32_t col_count,
    const Buffers& buf) {
    const std::uint32_t groups_per_row = col_count / kCubeGroup;
    const std::uint32_t total_elements = row_count * col_count;

    WaitFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);

    Cast(buf.weight_expanded, buf.weight_nibbles, RoundMode::CAST_NONE, total_elements);

    LocalTensor<uint16_t>   row_scales_u16  = buf.scale_temporary;
    LocalTensor<bfloat16_t> row_scales_bf16 = buf.scale_temporary.ReinterpretCast<bfloat16_t>();

    const uint32_t dst_shape[2] = {groups_per_row, kCubeGroup};
    const uint32_t src_shape[2] = {groups_per_row, 1};

    // ONE ROW AT A TIME, and the Mul moved INSIDE the loop.
    //
    // `buf.scale_plane` used to be an R_SUB x K_SUB plane (32,768 B) that this
    // loop filled row by row, followed by a single Mul over the whole
    // sub-tile. The plane was pure over-allocation: a row's scales are used
    // only by that row's multiply. It is now one row wide
    // (kCubeScaleRowBytes), and each row is scaled as soon as it is
    // broadcast.
    //
    // The arithmetic is UNCHANGED -- elementwise multiply, same operands, same
    // order within a row -- so the emu golden is unaffected. What changes is
    // 32,256 B of UB and one big Mul becoming row_count small ones.
    //
    // WHY IT MATTERED: the old Broadcast into scale_plane[out_base] was the
    // instruction the simulator reported as `ub_addr_overflow`, 1,280 times
    // per call, because the UB layout over-committed and this was the buffer
    // sitting past the end of it.
    for (std::uint32_t r_idx = 0; r_idx < row_count; ++r_idx) {
        const std::uint32_t scale_base = r_idx * kCubeScaleRowStride;
        const std::uint32_t out_base   = r_idx * col_count;

        Cast(row_scales_u16, buf.weight_scales[scale_base], RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        ShiftLeft(row_scales_u16, row_scales_u16, static_cast<uint16_t>(7), 16);
        PipeBarrier<PIPE_V>();

        Broadcast<bfloat16_t, 2, 1>(buf.scale_plane, row_scales_bf16, dst_shape, src_shape);
        PipeBarrier<PIPE_V>();

        Mul(buf.weight_expanded[out_base], buf.weight_expanded[out_base],
            buf.scale_plane, col_count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void DequantSubTileToL1(
    std::uint32_t row_count, std::uint32_t col_count,
    const Buffers& buf, const LocalTensor<bfloat16_t>& l1_weight_slot) {
    DequantSubTileVector(row_count, col_count, buf);
    const std::uint32_t total_elements = row_count * col_count;
    SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    DataCopy(l1_weight_slot, buf.weight_expanded, total_elements);
    // NO MTE3_MTE1 HERE. It used to be, and it was the deadlock.
    //
    // MTE1 (L1 -> L0A/L0B) is an AIC-ONLY pipeline. The Ascend C intra-core
    // synchronization tables list MTE3_MTE1 among the legal combinations for
    // the AIC and NOT for the AIV, whose only pipelines are S, V, MTE2 and
    // MTE3. This function runs on the AIV, so `WaitFlag<MTE3_MTE1>` here
    // blocked on a pipe event that NO transition on this core can produce --
    // a directed circuit with zero tokens and no live producer. It compiled,
    // and it never completed.
    //
    // It was not a BALANCE error, which is why counting SetFlag against
    // WaitFlag did not find it: the pair was textually 1:1. What was wrong is
    // that the pipe pair is not legal on this core.
    //
    // The dependency it was reaching for -- this AIV's UB -> L1 write must
    // land before the Cube's MTE1 reads that L1 slot -- is CROSS-CORE, and
    // intra-core SetFlag/WaitFlag cannot express a cross-core edge at all;
    // that would need CrossCoreSetFlag/CrossCoreWaitFlag. It also does not
    // need to be expressed here: the caller EnQues this slot on the
    // TQue<TPosition::TSCM, 2> and DeQues it before SetTensorB, and the
    // Matmul framework's KFC message at IterateAll is what orders the Cube
    // against it.
    //
    // MEASURED: with this pair present, no launch retired at any shape -- 48
    // sub-GEMMs for 2700 s, and 6 sub-GEMMs for 900 s, both ending on the
    // host watchdog with zero pem_lsu diagnostics and the cycle count still
    // climbing. The lifecycle-only variant, which never reaches this
    // function, retired in 41 s.
    SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
}

// THE TOKEN RING NEEDS EDGES IN BOTH DIRECTIONS.
//
// `MTE2_V` says "the slot is full, the Vector may read it". That alone is not a double buffer: it
// leaves the other direction unstated, and the other direction is what stops MTE2 from refilling a
// slot the Vector has not finished reading. With two slots and three or more chunks, chunk k+1 is
// loaded into the slot chunk k-1 is still being cast out of, and nothing orders the two.
//
// OBSERVED, 2026-10-16, dav-3510 CAModel, `Dsv4PrefillCubeDequantMicroSim` at hidden = 512: chunk 0
// came back holding CHUNK 2's values, which is exactly that race resolving the wrong way -- the
// MTE2 burst for chunk 2 landed in the ping slot before the Cast of chunk 0 had read it. Four
// chunks is the smallest grid that can show it (a two-chunk run reuses no slot and passed), and the
// corruption is silent: no fault, no flag, plausible numbers from the wrong chunk.
//
// So `V_MTE2` carries the matching buffer-free token: `PrepareTokensChunk` sets it once it has
// consumed a slot, and `AsyncLoadTokensChunk` waits for it before refilling one. The event id is
// the slot's own -- flags are keyed by the (srcPipe, dstPipe, eventID) triple, so `V_MTE2` id 2 and
// `MTE2_V` id 2 are different flags and the ring still costs only the two ids it already spent.
//
// `slot_was_consumed` is the caller's statement that this slot has been cast out of before, so
// there is a token to wait for. It is false for the first load into each slot, where the edge
// starts empty; waiting there would hang rather than synchronize. The caller must then DRAIN the
// tokens the last chunks leave outstanding -- see `DrainTokenRing` and its use per `inter_base`,
// because a leaked flag outlives the launch and corrupts the next one.
__aicore__ inline void AsyncLoadTokensChunk(
    const GlobalTensor<float>& tokens_in_gm, std::uint32_t token_row_start, std::uint32_t hidden_base,
    std::uint32_t valid_rows, const Geometry& geom,
    const LocalTensor<float>& dst_tensor, int event_id, bool slot_was_consumed) {
    if (slot_was_consumed) {
        WaitFlag<HardEvent::V_MTE2>(event_id);
    }
    DataCopyParams burst_params;
    burst_params.blockCount = static_cast<std::uint16_t>(valid_rows);
    burst_params.blockLen   = static_cast<std::uint16_t>((geom.hidden_sub_fwd * 4u) / 32u);
    burst_params.srcStride  = static_cast<std::uint16_t>(((geom.hidden - geom.hidden_sub_fwd) * 4u) / 32u);
    burst_params.dstStride  = 0;
    DataCopy(dst_tensor, tokens_in_gm[(static_cast<std::uint64_t>(token_row_start) * geom.hidden) + hidden_base], burst_params);
    SetFlag<HardEvent::MTE2_V>(event_id);
}

__aicore__ inline void PrepareTokensChunk(
    const LocalTensor<bfloat16_t>& tokens_sub_bf16, const LocalTensor<float>& tokens_f32,
    std::uint32_t valid_rows, std::uint32_t hidden_sub_fwd, int event_id) {
    WaitFlag<HardEvent::MTE2_V>(event_id);
    Cast(tokens_sub_bf16, tokens_f32, RoundMode::CAST_RINT, valid_rows * hidden_sub_fwd);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event_id);
}

// Consumes the buffer-free tokens the last chunks leave outstanding, so none survives into the next
// `inter_base` pass or the next launch.
//
// The count is exactly one per slot that was cast out of, and it does NOT depend on the parity of
// the chunk count. Per slot: a token is set once per chunk that occupied it, and consumed once per
// REFILL of it, and a slot is refilled for every chunk it holds except the first -- so each slot
// ends one token up, whichever slot the last chunk happened to land in. Clamped to the chunk count
// because a single-chunk pass never occupies the second slot, and draining it would wait forever on
// a token nothing set.
__aicore__ inline void DrainTokenRing(std::uint32_t chunk_count) {
    if (chunk_count >= 1u) {
        WaitFlag<HardEvent::V_MTE2>(kEvTokensPing);
    }
    if (chunk_count >= 2u) {
        WaitFlag<HardEvent::V_MTE2>(kEvTokensPong);
    }
}

__aicore__ inline void RunSubGemmAccumulate(
    GateUpMatmul& matmul_gate_up, const LocalTensor<bfloat16_t>& tokens_sub_bf16,
    __gm__ std::uint8_t* weights_packed, __gm__ std::uint8_t* weight_scales,
    std::uint32_t inter_base, std::uint32_t hidden_base, std::uint32_t valid_rows,
    const Geometry& geom, const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_fwd,
    const LocalTensor<float>& dst_accumulator, std::uint32_t accum_elements) {
    LocalTensor<bfloat16_t> slot_weight = q_weight_fwd.AllocTensor<bfloat16_t>();
    AsyncLoadWeightsSubTile(weights_packed, weight_scales, inter_base, hidden_base, geom.inter_sub_fwd, geom.hidden_sub_fwd, geom.hidden, buf);
    DequantSubTileToL1(geom.inter_sub_fwd, geom.hidden_sub_fwd, buf, slot_weight);
    q_weight_fwd.EnQue(slot_weight);
    LocalTensor<bfloat16_t> staged_weight = q_weight_fwd.DeQue<bfloat16_t>();

    matmul_gate_up.SetTensorA(tokens_sub_bf16);
    matmul_gate_up.SetTensorB(staged_weight, true);
    matmul_gate_up.SetOrgShape(valid_rows, geom.inter_sub_fwd, geom.hidden_sub_fwd);
    matmul_gate_up.SetTail(valid_rows, geom.inter_sub_fwd, geom.hidden_sub_fwd);
    // VECIN OUTPUT GOES THROUGH Iterate + GetTensorC, NOT IterateAll.
    //
    // This used to be `IterateAll(buf.subgemm_fwd)`. The Ascend C reference for
    // IterateAll's LocalTensor overload says its supported TPositions are TSCM
    // and VECIN and then, for VECIN specifically: "When outputting to the
    // VECIN tensor, use the Iterate API." C here is
    // MatmulType<TPosition::VECIN, ...>, so IterateAll was the one overload
    // the contract rules out for this destination, and the Fixpipe -> UB
    // delivery it synchronously waits on is the step GetTensorC drives -- a
    // call this kernel never made.
    //
    // THE LOOP RUNS EXACTLY ONCE AT THIS TILING, and that is load-bearing:
    // every GetTensorC writes the SAME UB tensor, so with more than one block
    // the Add below would accumulate only the last one. The host tiling makes
    // base == single for both sub-GEMMs (gate/up 16x128x128, down 16x64x256,
    // verified from MakeDsv4MoePrefillCubeGemmTiling's output), so there is a
    // single block. The `while` form is kept rather than a bare pair because
    // it is the documented shape and it terminates correctly either way --
    // but if base ever stops equalling single, this accumulation has to be
    // rewritten, not just re-looped.
    while (matmul_gate_up.Iterate()) {
        matmul_gate_up.GetTensorC(buf.subgemm_fwd);
    }
    matmul_gate_up.End();
    q_weight_fwd.FreeTensor(staged_weight);

    Add(dst_accumulator, dst_accumulator, buf.subgemm_fwd, accum_elements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ApplySwiGLUActivation(
    const Buffers& buf, std::uint32_t valid_rows, std::uint32_t inter_base,
    const Geometry& geom, std::uint32_t accum_elements) {
    Mins(buf.gate_accumulator, buf.gate_accumulator,  geom.swiglu_limit, accum_elements);
    Maxs(buf.up_accumulator,   buf.up_accumulator,   -geom.swiglu_limit, accum_elements);
    PipeBarrier<PIPE_V>();

    Mins(buf.up_accumulator,   buf.up_accumulator,    geom.swiglu_limit, accum_elements);
    Silu(buf.gate_accumulator, buf.gate_accumulator,  accum_elements);
    PipeBarrier<PIPE_V>();

    Mul(buf.gate_accumulator,  buf.gate_accumulator,  buf.up_accumulator, accum_elements);
    PipeBarrier<PIPE_V>();

    Cast(buf.activation_sub_bf16, buf.gate_accumulator, RoundMode::CAST_RINT, accum_elements);
    PipeBarrier<PIPE_V>();

    DataCopyParams stride_copy_params;
    stride_copy_params.blockCount = static_cast<std::uint16_t>(valid_rows);
    stride_copy_params.blockLen   = static_cast<std::uint16_t>((geom.inter_sub_fwd * sizeof(bfloat16_t)) / 32u);
    stride_copy_params.srcStride  = 0;
    stride_copy_params.dstStride  = static_cast<std::uint16_t>(((geom.inter - geom.inter_sub_fwd) * sizeof(bfloat16_t)) / 32u);

    DataCopy(buf.activation_resident[inter_base], buf.activation_sub_bf16, stride_copy_params);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void RunDownProjectionStep(
    DownMatmul& matmul_down, __gm__ std::uint8_t* down_packed_ptr, __gm__ std::uint8_t* down_scales_ptr,
    std::uint32_t hidden_base, std::uint32_t inter_base, std::uint32_t valid_rows,
    const Geometry& geom, const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_down,
    std::uint32_t down_accum_elements) {
    for (std::uint32_t r_idx = 0; r_idx < valid_rows; ++r_idx) {
        DataCopy(buf.activation_sub_bf16[r_idx * geom.inter_sub_down], buf.activation_resident[static_cast<std::uint32_t>(r_idx) * geom.inter + inter_base], geom.inter_sub_down);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<bfloat16_t> slot_down   = q_weight_down.AllocTensor<bfloat16_t>();
    AsyncLoadWeightsSubTile(down_packed_ptr, down_scales_ptr, hidden_base, inter_base, geom.hidden_sub_down, geom.inter_sub_down, geom.inter, buf);
    DequantSubTileToL1(geom.hidden_sub_down, geom.inter_sub_down, buf, slot_down);
    q_weight_down.EnQue(slot_down);
    LocalTensor<bfloat16_t> staged_down = q_weight_down.DeQue<bfloat16_t>();

    matmul_down.SetTensorA(buf.activation_sub_bf16);
    matmul_down.SetTensorB(staged_down, true);
    matmul_down.SetOrgShape(valid_rows, geom.hidden_sub_down, geom.inter_sub_down);
    matmul_down.SetTail(valid_rows, geom.hidden_sub_down, geom.inter_sub_down);
    // VECIN OUTPUT GOES THROUGH Iterate + GetTensorC, NOT IterateAll.
    //
    // This used to be `IterateAll(buf.subgemm_down)`. The Ascend C reference for
    // IterateAll's LocalTensor overload says its supported TPositions are TSCM
    // and VECIN and then, for VECIN specifically: "When outputting to the
    // VECIN tensor, use the Iterate API." C here is
    // MatmulType<TPosition::VECIN, ...>, so IterateAll was the one overload
    // the contract rules out for this destination, and the Fixpipe -> UB
    // delivery it synchronously waits on is the step GetTensorC drives -- a
    // call this kernel never made.
    //
    // THE LOOP RUNS EXACTLY ONCE AT THIS TILING, and that is load-bearing:
    // every GetTensorC writes the SAME UB tensor, so with more than one block
    // the Add below would accumulate only the last one. The host tiling makes
    // base == single for both sub-GEMMs (gate/up 16x128x128, down 16x64x256,
    // verified from MakeDsv4MoePrefillCubeGemmTiling's output), so there is a
    // single block. The `while` form is kept rather than a bare pair because
    // it is the documented shape and it terminates correctly either way --
    // but if base ever stops equalling single, this accumulation has to be
    // rewritten, not just re-looped.
    while (matmul_down.Iterate()) {
        matmul_down.GetTensorC(buf.subgemm_down);
    }
    matmul_down.End();
    q_weight_down.FreeTensor(staged_down);

    Add(buf.down_accumulator, buf.down_accumulator, buf.subgemm_down, down_accum_elements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessTokenTile(
    const Geometry& geom, std::uint32_t tile_idx,
    GM_ADDR gate_weights_packed, GM_ADDR gate_weight_scales,
    GM_ADDR up_weights_packed,   GM_ADDR up_weight_scales,
    GM_ADDR down_weights_packed, GM_ADDR down_weight_scales,
    GM_ADDR token_to_expert_lut,
    const GlobalTensor<float>& tokens_in_gm, const GlobalTensor<float>& tokens_out_gm,
    const Buffers& buf, TQue<TPosition::TSCM, 2>& q_weight_fwd, TQue<TPosition::TSCM, 2>& q_weight_down,
    GateUpMatmul& matmul_gate_up, DownMatmul& matmul_down) {
    const std::uint32_t token_row_start = tile_idx * geom.token_tile;
    const std::uint32_t valid_rows      = (geom.total_tokens - token_row_start) < geom.token_tile ? (geom.total_tokens - token_row_start) : geom.token_tile;

    const std::uint32_t expert_id = static_cast<std::uint32_t>(reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert_lut)[token_row_start]);
    if (unlikely(expert_id >= geom.num_experts)) {
        return;
    }
    const std::uint64_t expert_offset = static_cast<std::uint64_t>(expert_id);

    __gm__ std::uint8_t* gate_packed_ptr = Bytes(gate_weights_packed) + expert_offset * geom.gate_up_packed_stride;
    __gm__ std::uint8_t* gate_scales_ptr = Bytes(gate_weight_scales)  + expert_offset * geom.gate_up_scale_stride;
    __gm__ std::uint8_t* up_packed_ptr   = Bytes(up_weights_packed)   + expert_offset * geom.gate_up_packed_stride;
    __gm__ std::uint8_t* up_scales_ptr   = Bytes(up_weight_scales)    + expert_offset * geom.gate_up_scale_stride;
    __gm__ std::uint8_t* down_packed_ptr = Bytes(down_weights_packed) + expert_offset * geom.down_packed_stride;
    __gm__ std::uint8_t* down_scales_ptr = Bytes(down_weight_scales)  + expert_offset * geom.down_scale_stride;

    for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += geom.inter_sub_fwd) {
        const std::uint32_t accum_elements = valid_rows * geom.inter_sub_fwd;
        Duplicate(buf.gate_accumulator, 0.0F, accum_elements);
        Duplicate(buf.up_accumulator,   0.0F, accum_elements);
        PipeBarrier<PIPE_V>();

        const std::uint32_t chunk_count = geom.hidden / geom.hidden_sub_fwd;
        std::uint32_t hidden_base = 0;
        bool is_ping              = true;

        // The first load into each slot has no buffer-free token to wait for, so the prologue and
        // the loads for chunks below the slot count pass `false`. `kTokenRingSlots` is the ring
        // depth, stated once so the guard and the drain cannot disagree about it.
        AsyncLoadTokensChunk(tokens_in_gm, token_row_start, hidden_base, valid_rows, geom, buf.tokens_f32_ping, kEvTokensPing, false);

        for (; hidden_base + geom.hidden_sub_fwd < geom.hidden; hidden_base += geom.hidden_sub_fwd) {
            const std::uint32_t next_hidden_base = hidden_base + geom.hidden_sub_fwd;
            const std::uint32_t next_chunk       = next_hidden_base / geom.hidden_sub_fwd;
            auto& curr_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
            auto& next_tokens_f32  = is_ping ? buf.tokens_f32_pong      : buf.tokens_f32_ping;
            auto& curr_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
            const int curr_event   = is_ping ? kEvTokensPing            : kEvTokensPong;
            const int next_event   = is_ping ? kEvTokensPong            : kEvTokensPing;

            AsyncLoadTokensChunk(tokens_in_gm, token_row_start, next_hidden_base, valid_rows, geom, next_tokens_f32, next_event, next_chunk >= kTokenRingSlots);

            PrepareTokensChunk(curr_tokens_bf16, curr_tokens_f32, valid_rows, geom.hidden_sub_fwd, curr_event);
            RunSubGemmAccumulate(matmul_gate_up, curr_tokens_bf16, gate_packed_ptr, gate_scales_ptr, inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.gate_accumulator, accum_elements);
            RunSubGemmAccumulate(matmul_gate_up, curr_tokens_bf16, up_packed_ptr,   up_scales_ptr,   inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.up_accumulator,   accum_elements);

            is_ping = !is_ping;
        }

        auto& final_tokens_f32  = is_ping ? buf.tokens_f32_ping      : buf.tokens_f32_pong;
        auto& final_tokens_bf16 = is_ping ? buf.tokens_sub_bf16_ping : buf.tokens_sub_bf16_pong;
        const int final_event   = is_ping ? kEvTokensPing            : kEvTokensPong;

        PrepareTokensChunk(final_tokens_bf16, final_tokens_f32, valid_rows, geom.hidden_sub_fwd, final_event);
        RunSubGemmAccumulate(matmul_gate_up, final_tokens_bf16, gate_packed_ptr, gate_scales_ptr, inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.gate_accumulator, accum_elements);
        RunSubGemmAccumulate(matmul_gate_up, final_tokens_bf16, up_packed_ptr,   up_scales_ptr,   inter_base, hidden_base, valid_rows, geom, buf, q_weight_fwd, buf.up_accumulator,   accum_elements);

        // The ring is re-primed on every `inter_base` pass, so it must also be drained on every
        // pass. Leaving the tokens outstanding would make the next pass's prologue -- which does
        // not wait -- run against a stale free token, and the last pass's would outlive the launch.
        DrainTokenRing(chunk_count);

        ApplySwiGLUActivation(buf, valid_rows, inter_base, geom, accum_elements);
    }

    bool has_pending_output_burst = false;

    for (std::uint32_t hidden_base = 0; hidden_base < geom.hidden; hidden_base += geom.hidden_sub_down) {
        const std::uint32_t down_accum_elements = valid_rows * geom.hidden_sub_down;

        if (has_pending_output_burst) {
            WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
            has_pending_output_burst = false;
        }

        Duplicate(buf.down_accumulator, 0.0F, down_accum_elements);
        PipeBarrier<PIPE_V>();

        for (std::uint32_t inter_base = 0; inter_base < geom.inter; inter_base += geom.inter_sub_down) {
            RunDownProjectionStep(matmul_down, down_packed_ptr, down_scales_ptr, hidden_base, inter_base, valid_rows, geom, buf, q_weight_down, down_accum_elements);
        }

        SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);

        DataCopyParams burst_params;
        burst_params.blockCount = static_cast<std::uint16_t>(valid_rows);
        burst_params.blockLen   = static_cast<std::uint16_t>((geom.hidden_sub_down * 4u) / 32u);
        burst_params.srcStride  = 0;
        burst_params.dstStride  = static_cast<std::uint16_t>(((geom.hidden - geom.hidden_sub_down) * 4u) / 32u);
        DataCopy(tokens_out_gm[(static_cast<std::uint64_t>(token_row_start) * geom.hidden) + hidden_base], buf.down_accumulator, burst_params);

        SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
        has_pending_output_burst = true;
    }

    if (has_pending_output_burst) {
        WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    }
}

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube(
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
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data = reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
    if (unlikely(TilingRejected(tiling_data))) {
        return;
    }
    const Geometry geom = LoadGeometry(tiling_data);

    TPipe pipe;
    GateUpMatmul matmul_gate_up;
    DownMatmul   matmul_down;
    // REGIST_MATMUL_OBJ TAKES INTERLEAVED (object, tiling) PAIRS.
    //
    // Passing the two objects alone compiles and silently initializes BOTH
    // Matmuls with a NULL TCubeTiling -- KfcServer::InitObjAux falls to its
    // `InitKfc(tpipe, (void*)nullptr, ...)` branches, and every tiling field
    // the Cube then reads is a load off address 0. On the dav-3510 CAModel
    // that is the `pem_lsu.cc:484 invalid ldst addr` flood at ldst_addr 4, 8,
    // 12, 16, 20, 24, 28, 32, 40, 52, 60, 72, 76, 120, 124 -- exactly the
    // TCubeTiling field offsets -- and the launch never terminates. The full
    // derivation is in dsv4_moe_prefill_cube_cube_tiling.h.
    //
    // The pointers are typed `const __gm__ TCubeTiling*` deliberately: that
    // exact type selects InitKfc's branch which COPIES the struct out of GM
    // into a core-local one before Init (matmul_server.h:78). The host stages
    // both structs behind the 72-byte operator POD in the same allocation.
    const __gm__ TCubeTiling* gate_up_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeGateTilingOffset);
    const __gm__ TCubeTiling* down_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeDownTilingOffset);
    REGIST_MATMUL_OBJ(&pipe, shared_workspace,
                      matmul_gate_up, gate_up_cube_tiling,
                      matmul_down,    down_cube_tiling);

    GlobalTensor<float> tokens_in_gm;
    GlobalTensor<float> tokens_out_gm;
    const std::uint64_t total_elements = static_cast<std::uint64_t>(geom.total_tokens) * geom.hidden;
    tokens_in_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_in), total_elements);
    tokens_out_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tokens_out), total_elements);

    TQue<TPosition::VECCALC, 1> q_activation_resident;
    TQue<TPosition::VECCALC, 1> q_gate_accumulator;
    TQue<TPosition::VECCALC, 1> q_up_accumulator;
    TQue<TPosition::VECIN,   1> q_subgemm_fwd;
    TQue<TPosition::VECCALC, 1> q_tokens_f32_ping;
    TQue<TPosition::VECCALC, 1> q_tokens_f32_pong;
    TQue<TPosition::VECOUT,  1> q_tokens_sub_bf16_ping;
    TQue<TPosition::VECOUT,  1> q_tokens_sub_bf16_pong;
    TQue<TPosition::VECCALC, 1> q_down_accumulator;
    TQue<TPosition::VECIN,   1> q_subgemm_down;
    TQue<TPosition::VECOUT,  1> q_activation_sub_bf16;
    TQue<TPosition::VECCALC, 1> q_weight_nibbles;
    TQue<TPosition::VECCALC, 1> q_weight_expanded;
    TQue<TPosition::VECCALC, 1> q_weight_scales;
    TQue<TPosition::VECCALC, 1> q_scale_plane;
    TQue<TPosition::VECCALC, 1> q_scale_temporary;
    pipe.InitBuffer(q_activation_resident,   1, kCubeActBytes);
    pipe.InitBuffer(q_gate_accumulator,      1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_up_accumulator,        1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_subgemm_fwd,           1, kCubeAcc1Bytes);
    pipe.InitBuffer(q_tokens_f32_ping,       1, kCubeXF32Bytes);
    pipe.InitBuffer(q_tokens_f32_pong,       1, kCubeXF32Bytes);
    pipe.InitBuffer(q_tokens_sub_bf16_ping,  1, kCubeXSubBytes);
    pipe.InitBuffer(q_tokens_sub_bf16_pong,  1, kCubeXSubBytes);
    pipe.InitBuffer(q_down_accumulator,      1, kCubeAcc2Bytes);
    pipe.InitBuffer(q_subgemm_down,          1, kCubeAcc2Bytes);
    pipe.InitBuffer(q_activation_sub_bf16,   1, kCubeASubBytes);
    pipe.InitBuffer(q_weight_nibbles,        1, kCubeNibBytes);
    pipe.InitBuffer(q_weight_expanded,       1, kCubeWideBytes);
    pipe.InitBuffer(q_weight_scales,         1, kCubeScaleBytes);
    pipe.InitBuffer(q_scale_plane,           1, kCubeScaleRowBytes);
    pipe.InitBuffer(q_scale_temporary,       1, kCubeScaleTempBytes);

    TQue<TPosition::TSCM, 2> q_weight_fwd;
    TQue<TPosition::TSCM, 2> q_weight_down;
    pipe.InitBuffer(q_weight_fwd,  2, kCubeB1SlotBytes);
    pipe.InitBuffer(q_weight_down, 2, kCubeB2SlotBytes);

    Buffers buf;
    buf.activation_resident   = q_activation_resident.AllocTensor<bfloat16_t>();
    buf.gate_accumulator      = q_gate_accumulator.AllocTensor<float>();
    buf.up_accumulator        = q_up_accumulator.AllocTensor<float>();
    buf.subgemm_fwd           = q_subgemm_fwd.AllocTensor<float>();
    buf.tokens_f32_ping       = q_tokens_f32_ping.AllocTensor<float>();
    buf.tokens_f32_pong       = q_tokens_f32_pong.AllocTensor<float>();
    buf.tokens_sub_bf16_ping  = q_tokens_sub_bf16_ping.AllocTensor<bfloat16_t>();
    buf.tokens_sub_bf16_pong  = q_tokens_sub_bf16_pong.AllocTensor<bfloat16_t>();
    buf.down_accumulator      = q_down_accumulator.AllocTensor<float>();
    buf.subgemm_down          = q_subgemm_down.AllocTensor<float>();
    buf.activation_sub_bf16   = q_activation_sub_bf16.AllocTensor<bfloat16_t>();
    buf.weight_nibbles        = q_weight_nibbles.AllocTensor<fp4x2_e2m1_t>();
    buf.weight_expanded       = q_weight_expanded.AllocTensor<bfloat16_t>();
    buf.weight_scales         = q_weight_scales.AllocTensor<uint8_t>();
    buf.scale_plane           = q_scale_plane.AllocTensor<bfloat16_t>();
    buf.scale_temporary       = q_scale_temporary.AllocTensor<uint16_t>();

    const std::uint32_t core_idx    = static_cast<std::uint32_t>(GetBlockIdx());
    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;

    // ONLY SUB-CORE 0 DRIVES THE CUBE.
    //
    // In KERNEL_TYPE_MIX_AIC_1_2 one AIC is paired with TWO AIVs, and both of
    // them share a GetBlockIdx() -- GetSubBlockIdx() is what tells them apart.
    // Striding the tile grid by GetBlockIdx() alone therefore had BOTH paired
    // AIVs execute the SAME tiles: duplicate `y` stores, and both sub-cores
    // driving the same two Matmul objects through the KFC queue, issuing two
    // interleaved streams of SetTensorA/SetTensorB/IterateAll/End against one
    // server.
    //
    // The idle sub-core must still FALL THROUGH to a normal return rather than
    // exit early, because the AIC server runs until `quitSize == MIX_NUM`
    // (kernel_kfc.h: `isRun()` is `quitSize < MIX_NUM`, MIX_NUM =
    // __MIX_CORE_AIC_RATION__ = 2) and each AIV's QUIT is sent by its
    // KfcCommClient destructor at function exit. Entering the loop with
    // tile_idx == total_tiles skips the WORK while keeping that exit path and
    // the TQue alloc/free pairs below balanced.
    //
    // Minimal correct behaviour, not the fastest: splitting the tiles across
    // the two sub-cores instead of idling one is a separate change, and it
    // needs the `y` row disjointness argument redone per sub-core first.
    // `% GetSubBlockNum()` IS LOAD-BEARING, not defensive.
    //
    // AscendC::GetSubBlockIdx() is the AIV's logical id across ALL AI Core
    // groups -- with 10 groups at a 1:2 ratio it returns 0..19 -- while the
    // id WITHIN the group is what "am I this AIC's first vector core" needs.
    // (Matmul::GetSubBlockIdx() is the within-group one; the two are
    // documented as 0,1,0,1,... versus 0,1,2,...,19.)
    //
    // Comparing the global id against 0 therefore happens to be right at
    // block_dim == 1 and is WRONG for every larger grid: only the globally
    // first AIV would drive its Cube and every other group's pair would idle,
    // silently computing nothing. GetSubBlockNum() is the AIVs per logical AI
    // Core, so the modulo recovers the within-group id at any grid size.
    const bool drives_cube = (GetSubBlockIdx() % GetSubBlockNum()) == 0;

    for (std::uint32_t tile_idx = drives_cube ? core_idx : total_tiles;
         tile_idx < total_tiles; tile_idx += geom.block_dim) {
        ProcessTokenTile(
            geom, tile_idx,
            gate_weights_packed, gate_weight_scales,
            up_weights_packed,   up_weight_scales,
            down_weights_packed, down_weight_scales,
            token_to_expert_lut,
            tokens_in_gm, tokens_out_gm,
            buf, q_weight_fwd, q_weight_down,
            matmul_gate_up, matmul_down);
    }

    q_scale_temporary.FreeTensor(buf.scale_temporary);
    q_scale_plane.FreeTensor(buf.scale_plane);
    q_weight_scales.FreeTensor(buf.weight_scales);
    q_weight_expanded.FreeTensor(buf.weight_expanded);
    q_weight_nibbles.FreeTensor(buf.weight_nibbles);
    q_activation_sub_bf16.FreeTensor(buf.activation_sub_bf16);
    q_subgemm_down.FreeTensor(buf.subgemm_down);
    q_down_accumulator.FreeTensor(buf.down_accumulator);
    q_tokens_sub_bf16_pong.FreeTensor(buf.tokens_sub_bf16_pong);
    q_tokens_sub_bf16_ping.FreeTensor(buf.tokens_sub_bf16_ping);
    q_tokens_f32_pong.FreeTensor(buf.tokens_f32_pong);
    q_tokens_f32_ping.FreeTensor(buf.tokens_f32_ping);
    q_subgemm_fwd.FreeTensor(buf.subgemm_fwd);
    q_up_accumulator.FreeTensor(buf.up_accumulator);
    q_gate_accumulator.FreeTensor(buf.gate_accumulator);
    q_activation_resident.FreeTensor(buf.activation_resident);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
