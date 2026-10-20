/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY. Pure payloads and manual UB/L1 placement: no sync primitive of any kind here.
// Include order is load-bearing: a textual continuation of the production kernel TU.

#pragma once

#include "kernel_operator.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

// --- microstep schedule ----------------------------------------------------

enum class StepStage : std::uint8_t {
    Prologue,   // DMA in flight, nothing to dequant or contract yet
    Steady,     // DMA(s) overlapping dequant(s-1) and mmad(s-1)
    Epilogue    // DMA done, final dequant and mmad retire
};

/// Every buffer index and predicate of microstep `s`, derived from `s` alone.
struct StepSlotMapping {
    std::uint32_t slot_dma;      // s % kCubeRingSlots
    std::uint32_t slot_dequant;  // (s > 0) ? ((s - 1) % kCubeRingSlots) : 0
    std::uint32_t slot_l1;       // == slot_dequant
    std::uint32_t tile_dma;      // s
    std::uint32_t tile_mmad;     // (s > 0) ? (s - 1) : 0

    bool dma_active;
    bool dequant_active;
    bool mmad_active;
    bool is_first_mmad;
    bool is_last_mmad;

    StepStage stage;             // narration only; classifies the three predicates
};

__aicore__ inline StepSlotMapping ResolveStepSlots(std::uint32_t s, std::uint32_t total_steps) {
    StepSlotMapping m{};
    m.slot_dma      = s % kCubeRingSlots;
    m.slot_dequant  = (s > 0u) ? ((s - 1u) % kCubeRingSlots) : 0u;
    m.slot_l1       = m.slot_dequant;
    m.tile_dma      = s;
    m.tile_mmad     = (s > 0u) ? (s - 1u) : 0u;

    m.dma_active     = (s < total_steps);
    m.dequant_active = (s >= 1u && s <= total_steps);
    m.mmad_active    = m.dequant_active;
    m.is_first_mmad  = (s == 1u);
    m.is_last_mmad   = (s == total_steps);

    m.stage = !m.dequant_active ? StepStage::Prologue
            : (m.dma_active ? StepStage::Steady : StepStage::Epilogue);
    return m;
}

/// First microstep that REUSES an L1 slot. Guarding below this deadlocks: the release
/// would come from the same microstep the Cube cannot reach until this AIV publishes.
inline constexpr std::uint32_t kCubeFirstL1Reuse = 1u + kCubeRingSlots;

enum class MicrostepPhase : std::uint8_t { Forward, Down };

/// Epilogue microsteps per phase, counted from the handshakes they have to carry.
/// Forward: Fixpipe(gate), Fixpipe(up), SwiGLU, then the Cube's block rendezvous.
/// Down:    Fixpipe(down) and the GM store, then the rendezvous.
inline constexpr std::uint32_t kEpilogueStepsFwd  = 4;
inline constexpr std::uint32_t kEpilogueStepsDown = 2;

/// What one BLOCK of microsteps is: which projection, which output base, how many steps.
struct MicrostepPlan {
    MicrostepPhase phase;
    std::uint32_t  base;         // inter_base in phase 1, hidden_base in phase 2
    std::uint32_t  total_steps;  // contraction steps only

    __aicore__ inline bool Forward() const { return phase == MicrostepPhase::Forward; }
    __aicore__ inline std::uint32_t EpilogueSteps() const {
        return Forward() ? kEpilogueStepsFwd : kEpilogueStepsDown;
    }
    __aicore__ inline std::uint32_t LastStep() const {
        return total_steps + EpilogueSteps();
    }
};

/// What ONE microstep is: every slot, predicate and direction it needs, and nothing else.
/// Exactly one of the contraction predicates or the epilogue selectors is ever set.
struct MicrostepIter {
    StepSlotMapping slots;
    std::uint32_t   step;
    std::uint32_t   base;
    std::uint32_t   total_steps;
    std::uint32_t   stage;        // index within the epilogue, 0 during the contraction
    bool            forward;
    bool            epilogue;

    bool drain_step_free;         // reconcile the block's StepFree releases
    bool aiv_wait_product;
    bool aiv_gate;
    bool aiv_up;
    bool aiv_swiglu;
    bool aiv_store;
    bool aiv_set_landing_free;
    bool aiv_flag_after_store;    // scope the LandingFree set to MTE3 instead of V

    bool aic_retire_mmad;
    bool aic_fixpipe;
    bool aic_fixpipe_up;
    bool aic_set_product;
    bool aic_wait_landing_free;
    bool aic_wait_rendezvous;     // last stage: the wait gates the NEXT block's L0 loads
};

/// Loop step 1, plan: derive the whole iteration from the block plan and the counter.
__aicore__ inline MicrostepIter PlanMicrostep(const MicrostepPlan& plan, std::uint32_t s) {
    MicrostepIter it{};
    it.slots       = ResolveStepSlots(s, plan.total_steps);
    it.step        = s;
    it.base        = plan.base;
    it.total_steps = plan.total_steps;
    it.forward     = plan.Forward();
    it.epilogue    = (s > plan.total_steps);
    if (!it.epilogue) {
        return it;
    }

    // Past the contraction every slot predicate is off, so step 3 issues epilogue work only.
    it.slots.dma_active     = false;
    it.slots.dequant_active = false;
    it.slots.mmad_active    = false;
    it.stage                = s - plan.total_steps - 1u;

    it.drain_step_free    = (it.stage == 0u);
    it.aic_retire_mmad    = (it.stage == 0u);
    it.aic_wait_rendezvous = (it.stage + 1u == plan.EpilogueSteps());

    if (it.forward) {
        // Gate then up, in that order: swapping the two handovers is a wrong answer.
        it.aic_fixpipe           = (it.stage <= 1u);
        it.aic_fixpipe_up        = (it.stage == 1u);
        it.aic_set_product       = (it.stage <= 1u);
        it.aic_wait_landing_free = (it.stage >= 1u);

        it.aiv_wait_product      = (it.stage <= 1u);
        it.aiv_gate              = (it.stage == 0u);
        it.aiv_up                = (it.stage == 1u);
        it.aiv_swiglu            = (it.stage == 2u);
        it.aiv_set_landing_free  = (it.stage <= 2u);
        it.aiv_flag_after_store  = (it.stage == 2u);
    } else {
        it.aic_fixpipe           = (it.stage == 0u);
        it.aic_set_product       = (it.stage == 0u);
        it.aic_wait_landing_free = (it.stage == 1u);

        it.aiv_wait_product      = (it.stage == 0u);
        it.aiv_store             = (it.stage == 0u);
        it.aiv_set_landing_free  = (it.stage == 0u);
        it.aiv_flag_after_store  = (it.stage == 0u);
    }
    return it;
}

// --- manual UB arena, byte-identical to the layout the TPipe queues produced ---
// sub_idx-INDEPENDENT by contract: each AIV owns its own 256 KiB UB, offset 0 is a
// different physical location on each, and the AIC names one literal Fixpipe offset.

inline constexpr std::uint32_t kUbOffLanding    = 0;
inline constexpr std::uint32_t kUbOffAct        = kUbOffLanding + kCubeLandingBytes;
inline constexpr std::uint32_t kUbOffGateAcc    = kUbOffAct + kCubeActPlaneBytes;
inline constexpr std::uint32_t kUbOffUpAcc      = kUbOffGateAcc + kCubeAccBytes;
inline constexpr std::uint32_t kUbOffActSub     = kUbOffUpAcc + kCubeAccBytes;
inline constexpr std::uint32_t kUbOffXF32       = kUbOffActSub + kCubeActSubBytes;
inline constexpr std::uint32_t kUbOffXBf16      =
    kUbOffXF32 + kCubeRingSlots * kCubeTokenF32Bytes;
inline constexpr std::uint32_t kUbOffNib        =
    kUbOffXBf16 + kCubeRingSlots * kCubeTokenSubBytes;
inline constexpr std::uint32_t kUbOffScale      =
    kUbOffNib + kCubeRingSlots * kCubeWeightStreams * kCubeNibHalfBytes;
inline constexpr std::uint32_t kUbOffWide       =
    kUbOffScale + kCubeRingSlots * kCubeWeightStreams * kCubeScaleBytes;
inline constexpr std::uint32_t kUbOffScalePlane =
    kUbOffWide + kCubeRingSlots * kCubeWeightStreams * kCubeWideHalfBytes;
inline constexpr std::uint32_t kUbOffScaleTemp  = kUbOffScalePlane + kCubeScaleRowBytes;
inline constexpr std::uint32_t kUbArenaBytes    = kUbOffScaleTemp + kCubeScaleTempBytes;

static_assert(kUbOffLanding == kCubeLandingUbOffset,
              "the landing zone must sit at the offset the AIC's Fixpipe names");
static_assert(kUbArenaBytes == kCubeUbBytes,
              "the manual arena must charge exactly what the operator's UB census says");
// 32 is a power of two, so OR-ing the offsets checks every one of them at once.
static_assert(((kUbOffLanding | kUbOffAct | kUbOffGateAcc | kUbOffUpAcc | kUbOffActSub |
                kUbOffXF32 | kUbOffXBf16 | kUbOffNib | kUbOffScale | kUbOffWide |
                kUbOffScalePlane | kUbOffScaleTemp) & 31u) == 0u,
              "every UB offset must be 32-byte aligned: a burst block is 32 B");

// --- manual L1 arena, carved identically on both cores ---
// That is what makes the push and the load agree; nothing in the toolchain checks it.

inline constexpr std::uint32_t kL1OffA  = 0;
inline constexpr std::uint32_t kL1OffW  = kL1OffA + kCubeRingSlots * kCubeL1ASlotBytes;
inline constexpr std::uint32_t kL1OffA2 =
    kL1OffW + kCubeRingSlots * kCubeWeightStreams * kCubeL1WSlotBytes;
inline constexpr std::uint32_t kL1OffW2 = kL1OffA2 + kCubeRingSlots * kCubeL1A2SlotBytes;
inline constexpr std::uint32_t kL1ArenaBytes =
    kL1OffW2 + kCubeRingSlots * kCubeL1W2SlotBytes;

static_assert(kL1ArenaBytes == kCubeL1Bytes,
              "the manual L1 arena must charge exactly what the operator's L1 census says");
static_assert(((kL1OffA | kL1OffW | kL1OffA2 | kL1OffW2) & 31u) == 0u,
              "every L1 offset must be 32-byte aligned");

/// Placed the way TBuf::GetWithOffset does: dataLen in BYTES, bufferAddr a raw offset.
template <typename T>
__aicore__ inline LocalTensor<T> PlaceTensor(TPosition pos, std::uint32_t byte_offset,
                                             std::uint32_t byte_len) {
    TBuffAddr addr{};
    addr.dataLen    = byte_len;
    addr.bufferAddr = byte_offset;
    addr.logicPos   = static_cast<std::uint8_t>(pos);
    LocalTensor<T> tensor;
    tensor.SetAddr(addr);
    return tensor;
}

/// The L1 operand ring, hand-placed. Both cores call this with the same base.
struct L1Operands {
    LocalTensor<bfloat16_t> a[kCubeRingSlots];
    LocalTensor<bfloat16_t> w[kCubeRingSlots][kCubeWeightStreams];
    LocalTensor<bfloat16_t> a2[kCubeRingSlots];
    LocalTensor<bfloat16_t> w2[kCubeRingSlots];
};

__aicore__ inline void PlaceL1Operands(L1Operands& l1, std::uint32_t base) {
    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        l1.a[s] = PlaceTensor<bfloat16_t>(
            TPosition::A1, base + kL1OffA + s * kCubeL1ASlotBytes, kCubeL1ASlotBytes);
        l1.a2[s] = PlaceTensor<bfloat16_t>(
            TPosition::A1, base + kL1OffA2 + s * kCubeL1A2SlotBytes, kCubeL1A2SlotBytes);
        l1.w2[s] = PlaceTensor<bfloat16_t>(
            TPosition::B1, base + kL1OffW2 + s * kCubeL1W2SlotBytes, kCubeL1W2SlotBytes);
        for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
            const std::uint32_t slot = s * kCubeWeightStreams + w;
            l1.w[s][w] = PlaceTensor<bfloat16_t>(
                TPosition::B1, base + kL1OffW + slot * kCubeL1WSlotBytes, kCubeL1WSlotBytes);
        }
    }
}

/// Hand-placed; scale_plane and scale_temporary are shared across every slot and stream.
__aicore__ inline void PlaceAivBuffers(AivBuffers& buf, std::uint32_t base) {
    buf.landing  = PlaceTensor<float>(TPosition::VECIN, base + kUbOffLanding,
                                      kCubeLandingBytes);
    buf.act      = PlaceTensor<bfloat16_t>(TPosition::VECCALC, base + kUbOffAct,
                                           kCubeActPlaneBytes);
    buf.gate_acc = PlaceTensor<float>(TPosition::VECCALC, base + kUbOffGateAcc, kCubeAccBytes);
    buf.up_acc   = PlaceTensor<float>(TPosition::VECCALC, base + kUbOffUpAcc, kCubeAccBytes);
    buf.act_sub  = PlaceTensor<bfloat16_t>(TPosition::VECOUT, base + kUbOffActSub,
                                           kCubeActSubBytes);

    const LocalTensor<bfloat16_t> scale_plane = PlaceTensor<bfloat16_t>(
        TPosition::VECCALC, base + kUbOffScalePlane, kCubeScaleRowBytes);
    const LocalTensor<uint16_t> scale_temp = PlaceTensor<uint16_t>(
        TPosition::VECCALC, base + kUbOffScaleTemp, kCubeScaleTempBytes);

    for (std::uint32_t s = 0; s < kCubeRingSlots; ++s) {
        buf.x_f32[s]  = PlaceTensor<float>(
            TPosition::VECCALC, base + kUbOffXF32 + s * kCubeTokenF32Bytes, kCubeTokenF32Bytes);
        buf.x_bf16[s] = PlaceTensor<bfloat16_t>(
            TPosition::VECOUT, base + kUbOffXBf16 + s * kCubeTokenSubBytes, kCubeTokenSubBytes);
        for (std::uint32_t w = 0; w < kCubeWeightStreams; ++w) {
            const std::uint32_t slot = s * kCubeWeightStreams + w;
            buf.w[s][w].weight_nibbles = PlaceTensor<fp4x2_e2m1_t>(
                TPosition::VECCALC, base + kUbOffNib + slot * kCubeNibHalfBytes,
                kCubeNibHalfBytes);
            buf.w[s][w].weight_scales = PlaceTensor<uint8_t>(
                TPosition::VECCALC, base + kUbOffScale + slot * kCubeScaleBytes,
                kCubeScaleBytes);
            buf.w[s][w].weight_expanded = PlaceTensor<bfloat16_t>(
                TPosition::VECCALC, base + kUbOffWide + slot * kCubeWideHalfBytes,
                kCubeWideHalfBytes);
            buf.w[s][w].scale_plane     = scale_plane;
            buf.w[s][w].scale_temporary = scale_temp;
        }
    }
}

// --- per-core contexts -----------------------------------------------------

/// GM bases are unoffset; the accessors apply the per-expert stride, widening first.
struct AivStepContext {
    const Geometry&            geom;
    const AivBuffers&          buf;
    AivRingState&              ring;
    const GlobalTensor<float>& tokens_in;
    const GlobalTensor<float>& tokens_out;
    const L1Operands&          l1;
    GM_ADDR gate_packed_base;
    GM_ADDR gate_scales_base;
    GM_ADDR up_packed_base;
    GM_ADDR up_scales_base;
    GM_ADDR down_packed_base;
    GM_ADDR down_scales_base;
    GM_ADDR lut;
    std::uint32_t sub_idx;
    std::uint32_t row_base;          // sub_idx * kCubeMHalf
    std::uint32_t token_row_start;   // per admitted tile
    std::uint32_t valid_rows;
    std::uint32_t store_rows;
    std::uint32_t expert_id;

    __aicore__ inline __gm__ std::uint8_t* GatePacked() const {
        return Bytes(gate_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* GateScales() const {
        return Bytes(gate_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_scale_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* UpPacked() const {
        return Bytes(up_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* UpScales() const {
        return Bytes(up_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.gate_up_scale_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* DownPacked() const {
        return Bytes(down_packed_base) +
               static_cast<std::uint64_t>(expert_id) * geom.down_packed_stride;
    }
    __aicore__ inline __gm__ std::uint8_t* DownScales() const {
        return Bytes(down_scales_base) +
               static_cast<std::uint64_t>(expert_id) * geom.down_scale_stride;
    }
};

/// L0 members are address-space-qualified POINTERS, never dereferenced in this file.
struct CubeStepContext {
    const Geometry&    geom;
    const L1Operands&  l1;
    __ca__ bfloat16_t* l0a;
    __cb__ bfloat16_t* l0b_gate;
    __cb__ bfloat16_t* l0b_up;
    __cc__ float*      l0c_gate;
    __cc__ float*      l0c_up;
    GM_ADDR lut;
    std::uint32_t expert_id;
};

/// Refresh the per-tile fields, or report the tile is not this expert's.
__aicore__ inline bool AivAdmitTile(AivStepContext& ctx, std::uint32_t tile_idx) {
    const std::uint32_t token_row_start = tile_idx * ctx.geom.token_tile;
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, token_row_start, ctx.geom.num_experts, expert_id)) {
        return false;
    }
    const std::uint32_t remaining  = ctx.geom.total_tokens - token_row_start;
    const std::uint32_t valid_rows = remaining < ctx.geom.token_tile
                                         ? remaining : ctx.geom.token_tile;
    const std::uint32_t owned = valid_rows > ctx.row_base ? (valid_rows - ctx.row_base) : 0u;

    ctx.token_row_start = token_row_start;
    ctx.valid_rows      = valid_rows;
    ctx.store_rows      = owned < kCubeMHalf ? owned : kCubeMHalf;
    ctx.expert_id       = expert_id;
    return true;
}

__aicore__ inline bool CubeAdmitTile(CubeStepContext& ctx, std::uint32_t tile_idx) {
    std::uint32_t expert_id = 0;
    if (!TileAdmitted(ctx.lut, tile_idx * ctx.geom.token_tile,
                      ctx.geom.num_experts, expert_id)) {
        return false;
    }
    ctx.expert_id = expert_id;
    return true;
}

// --- AIV payloads, phase 1 -------------------------------------------------

/// GM -> UB slot_dma. Leaves ONE token outstanding; the caller takes it a microstep later.
__aicore__ inline void PayloadAiv_DmaInFwd(
    const AivStepContext& ctx, const StepSlotMapping& m, std::uint32_t inter_base) {
    AivPrefetchFwdStep(ctx.tokens_in, ctx.token_row_start,
                       ctx.GatePacked(), ctx.GateScales(), ctx.UpPacked(), ctx.UpScales(),
                       inter_base, m.tile_dma * kCubeKSub, ctx.valid_rows,
                       ctx.sub_idx, ctx.geom, ctx.buf, m.slot_dma, ctx.ring);
}

/// FP4 -> BF16 both streams, then the token half; event_id = -1, the caller takes the token.
__aicore__ inline void PayloadAiv_DequantAndCast(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[m.slot_dequant][0], -1);
    DequantSubTileVector(kCubeRSub, kCubeKHalf, ctx.buf.w[m.slot_dequant][1], -1);
    Cast(ctx.buf.x_bf16[m.slot_dequant], ctx.buf.x_f32[m.slot_dequant],
         RoundMode::CAST_RINT, kCubeMTile * kCubeKHalf);
}

/// UB -> L1 slot_l1: A, Wg, Wu. col_base is this AIV's contraction half.
__aicore__ inline void PayloadAiv_PushL1(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    const std::uint32_t col_base = ctx.sub_idx * kCubeKHalf;
    PushUbTileToL1Nz(ctx.l1.a[m.slot_l1], ctx.buf.x_bf16[m.slot_dequant],
                     kCubeMTile, kCubeKHalf, kCubeKHalf, kCubeMTile, col_base, 0);
    PushUbTileToL1Nz(ctx.l1.w[m.slot_l1][0], ctx.buf.w[m.slot_dequant][0].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
    PushUbTileToL1Nz(ctx.l1.w[m.slot_l1][1], ctx.buf.w[m.slot_dequant][1].weight_expanded,
                     kCubeRSub, kCubeKHalf, kCubeKHalf, kCubeRSub, col_base, 0);
}

/// Gate pre-activation out of the landing zone, clamped ABOVE only.
__aicore__ inline void PayloadAiv_EpilogueGate(const AivStepContext& ctx) {
    Mins(ctx.buf.gate_acc, ctx.buf.landing, ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// Up pre-activation, lower clamp; this is also the move OUT of the landing zone.
__aicore__ inline void PayloadAiv_EpilogueUp(const AivStepContext& ctx) {
    Maxs(ctx.buf.up_acc, ctx.buf.landing, -ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// Up pre-activation, upper clamp; reads up_acc, not the landing zone.
__aicore__ inline void PayloadAiv_EpilogueUpClamp(const AivStepContext& ctx) {
    Mins(ctx.buf.up_acc, ctx.buf.up_acc, ctx.geom.swiglu_limit, kCubeMHalf * kCubeRSub);
}

/// y = Silu(gate) * up -> BF16 -> this AIV's half of the resident plane.
__aicore__ inline void PayloadAiv_ApplySwiGLU(
    const AivStepContext& ctx, std::uint32_t inter_base) {
    AivApplySwiGLU(ctx.buf, inter_base, ctx.geom);
}

// --- AIV payloads, phase 2 -------------------------------------------------

/// GM -> UB slot_dma: one weight stream. No token burst; A comes from the plane.
__aicore__ inline void PayloadAiv_DmaInDown(
    const AivStepContext& ctx, const StepSlotMapping& m, std::uint32_t hidden_base) {
    AivPrefetchDownStep(ctx.DownPacked(), ctx.DownScales(),
                        hidden_base, m.tile_dma * kCubeRSub2,
                        ctx.sub_idx, ctx.geom, ctx.buf, m.slot_dma, ctx.ring);
}

/// event_id = -1: the slot's token is consumed by the caller, not here.
__aicore__ inline void PayloadAiv_DequantDown(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    DequantSubTileVector(kCubeHSub, kCubeRHalf2, ctx.buf.w[m.slot_dequant][0], -1);
}

/// The two pushes split on DIFFERENT axes: a contraction half vs a token-row range.
__aicore__ inline void PayloadAiv_PushL1Down(
    const AivStepContext& ctx, const StepSlotMapping& m) {
    PushUbTileToL1Nz(ctx.l1.w2[m.slot_l1], ctx.buf.w[m.slot_dequant][0].weight_expanded,
                     kCubeHSub, kCubeRHalf2, kCubeRHalf2,
                     kCubeHSub, ctx.sub_idx * kCubeRHalf2, 0);
    PushUbTileToL1Nz(ctx.l1.a2[m.slot_l1], ctx.buf.act[m.tile_mmad * kCubeRSub2],
                     kCubeMHalf, kCubeRSub2, ctx.geom.inter,
                     kCubeMTile, 0, ctx.sub_idx * kCubeMHalf);
}

/// store_rows == 0 when the tail misses this half, and a zero A row gave a zero C row.
__aicore__ inline void PayloadAiv_StoreLandingPlane(
    const AivStepContext& ctx, std::uint32_t hidden_base) {
    if (ctx.store_rows == 0u) {
        return;
    }
    DataCopyParams out_burst;
    out_burst.blockCount = static_cast<std::uint16_t>(ctx.store_rows);
    out_burst.blockLen   = static_cast<std::uint16_t>((kCubeHSub * 4u) / 32u);
    out_burst.srcStride  = 0;
    out_burst.dstStride  = static_cast<std::uint16_t>(((ctx.geom.hidden - kCubeHSub) * 4u) / 32u);
    DataCopy(ctx.tokens_out[(static_cast<std::uint64_t>(ctx.token_row_start + ctx.row_base) *
                             ctx.geom.hidden) + hidden_base],
             ctx.buf.landing, out_burst);
}

// --- Cube payloads ---------------------------------------------------------

__aicore__ inline void PayloadCube_LoadL0(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1.a[m.slot_l1], kCubeMTile, kCubeKSub);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1.w[m.slot_l1][0], kCubeKSub, kCubeRSub);
    CubeLoadWeightsToL0B(ctx.l0b_up, ctx.l1.w[m.slot_l1][1], kCubeKSub, kCubeRSub);
}

__aicore__ inline void PayloadCube_LoadL0Down(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeLoadTokensToL0A(ctx.l0a, ctx.l1.a2[m.slot_l1], kCubeMTile, kCubeRSub2);
    CubeLoadWeightsToL0B(ctx.l0b_gate, ctx.l1.w2[m.slot_l1], kCubeRSub2, kCubeHSub);
}

/// No MTE1->M event: the caller drain carries it. The K-axis interlock is NOT subsumed.
__aicore__ inline void PayloadCube_MmadAccumulate(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeRSub, kCubeKSub, m.is_first_mmad);
    CubeMmadAccumulate(ctx.l0c_up, ctx.l0a, ctx.l0b_up,
                       kCubeMTile, kCubeRSub, kCubeKSub, m.is_first_mmad);
    CubeCloseAccumulationStep();
}

__aicore__ inline void PayloadCube_MmadDown(
    const CubeStepContext& ctx, const StepSlotMapping& m) {
    CubeMmadAccumulate(ctx.l0c_gate, ctx.l0a, ctx.l0b_gate,
                       kCubeMTile, kCubeHSub, kCubeRSub2, m.is_first_mmad);
    CubeCloseAccumulationStep();
}

/// L0C -> UB offset 0 in EACH AIV's own address space, full M (CubeL0c2UbTraitDualDstM).
__aicore__ inline void PayloadCube_FixpipeMoveOut(
    __cc__ float* l0c_mem, std::uint32_t extent_m, std::uint32_t extent_n) {
    auto l0C = Te::MakeTensor(Te::MakeMemPtr(l0c_mem),
                              Te::MakeFrameLayout<Te::NZLayoutPtn, 16>(extent_m, extent_n));
    auto ub  = Te::MakeTensor(
        Te::MakeMemPtr(reinterpret_cast<__ubuf__ float*>(kUbOffLanding)),
        Te::MakeFrameLayout<Te::NDExtLayoutPtn>(extent_m, extent_n));
    Te::FixpipeParams fixParams;
    Te::Copy(Te::MakeCopy(Te::CopyL0C2UB{}, CubeL0c2UbTraitDualDstM{}).with(fixParams), ub, l0C);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
