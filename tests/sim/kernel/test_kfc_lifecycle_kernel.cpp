/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// SIM ONLY -- VARIANT (a) of the dsv4_moe_prefill_cube bisection.
//
// ---------------------------------------------------------------------------
// WHAT THIS ISOLATES
// ---------------------------------------------------------------------------
// dsv4_moe_prefill_cube does not retire a single token tile on the dav-3510
// CAModel. Six causes have been ruled out: MIX compilation (the object carries
// both _mix_aic and _mix_aiv entries in .aicore_binary and both
// SPLIT_CORE_CUBE and SPLIT_CORE_VEC phases compile), a null TCubeTiling, the
// V_MTE2 token ring at every chunk count, static flag balance and event-id
// collisions, work VOLUME (cutting 48 sub-GEMMs to 6 did not move the finish
// line), and sub-core duplication (gating on the within-group sub-block index
// changed 700,000 cycles to 660,000 -- noise).
//
// So stop varying the schedule and bisect by TRUNCATION, using COMPLETION
// itself as the signal rather than a progress marker the host would have to
// read out of GM while it is blocked in aclrtSynchronizeStream.
//
// THIS variant keeps ONLY:
//   * TPipe construction
//   * the two production Matmul types, declared from the production header
//   * REGIST_MATMUL_OBJ with the two host-built GM TCubeTilings
//   * the production UB buffer allocation, so the UB budget is the real one
//   * the sub-core gate, so the grid topology is the real one
//
// and then RETURNS. No DataCopy, no dequantization, no Cast, no SetFlag, no
// WaitFlag, no Iterate, no store. Not one compute instruction.
//
// WHAT EACH OUTCOME MEANS
//   RETIRES  -- the Matmul/KFC lifecycle is healthy: server start-up, the two
//               clients, and the QUIT handshake `isRun()` gates on all work in
//               this configuration. The hang is then inside a compute stage and
//               the bisection moves forward to variant (b).
//   HANGS    -- the lifecycle itself does not complete, which would explain why
//               three schedule-level fixes changed nothing, and localizes the
//               defect to REGIST_MATMUL_OBJ / KfcServer / KfcCommClient in this
//               build and launch configuration rather than to anything this
//               operator computes.
//
// Either way it is decisive, and it costs almost no simulated time: there is no
// work to simulate, so anything beyond simulator bring-up is the handshake.
//
// The buffers are allocated and freed EXACTLY as production does even though
// nothing reads them. That is deliberate -- the UB high-water mark and the
// TQue alloc/free balance are part of the configuration under test, and
// dropping them would make a pass here say less than it looks like it says.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"

namespace moe::kernels::arch35::kfc_lifecycle {

using namespace AscendC;

// The UB byte budget lives in the operator's private layout header, in its own
// namespace. Pulled in rather than re-stated: this variant must allocate the
// SAME UB plane production does, or the configuration under test is not
// production's (CLAUDE.md section 1.2 keeps that header operator-private, and
// this file is compiled only into the sim targets).
using namespace moe::kernels::arch35::dsv4_moe_prefill_cube;

// The SAME operand declarations as production. If these drift, this variant
// stops testing the configuration production uses.
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

static_assert(sizeof(TCubeTiling) == kPrefillCubeTCubeTilingBytes,
              "the staged TCubeTiling size disagrees with the wire format in "
              "dsv4_moe_prefill_cube_cube_tiling.h");

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR ptr) {
    return reinterpret_cast<__gm__ std::uint8_t*>(ptr);
}

extern "C" __global__ __aicore__ void test_kfc_lifecycle_kernel(
    GM_ADDR shared_workspace, GM_ADDR tiling_config, GM_ADDR status_out) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (unlikely(tiling_config == nullptr || shared_workspace == nullptr)) {
        return;
    }

    TPipe pipe;
    GateUpMatmul matmul_gate_up;
    DownMatmul   matmul_down;

    // The two tilings sit behind the 72-byte operator POD in the SAME staging
    // layout production uses, and are typed `const __gm__ TCubeTiling*` so
    // InitKfc takes its copy-out-of-GM branch.
    const __gm__ TCubeTiling* gate_up_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeGateTilingOffset);
    const __gm__ TCubeTiling* down_cube_tiling =
        reinterpret_cast<const __gm__ TCubeTiling*>(Bytes(tiling_config) + kPrefillCubeDownTilingOffset);
    REGIST_MATMUL_OBJ(&pipe, shared_workspace,
                      matmul_gate_up, gate_up_cube_tiling,
                      matmul_down,    down_cube_tiling);

    // Production's UB plane, allocated and freed, never read.
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

    LocalTensor<bfloat16_t>    t_act    = q_activation_resident.AllocTensor<bfloat16_t>();
    LocalTensor<float>         t_gate   = q_gate_accumulator.AllocTensor<float>();
    LocalTensor<float>         t_up     = q_up_accumulator.AllocTensor<float>();
    LocalTensor<float>         t_sgf    = q_subgemm_fwd.AllocTensor<float>();
    LocalTensor<float>         t_xping  = q_tokens_f32_ping.AllocTensor<float>();
    LocalTensor<float>         t_xpong  = q_tokens_f32_pong.AllocTensor<float>();
    LocalTensor<bfloat16_t>    t_bping  = q_tokens_sub_bf16_ping.AllocTensor<bfloat16_t>();
    LocalTensor<bfloat16_t>    t_bpong  = q_tokens_sub_bf16_pong.AllocTensor<bfloat16_t>();
    LocalTensor<float>         t_down   = q_down_accumulator.AllocTensor<float>();
    LocalTensor<float>         t_sgd    = q_subgemm_down.AllocTensor<float>();
    LocalTensor<bfloat16_t>    t_asub   = q_activation_sub_bf16.AllocTensor<bfloat16_t>();
    LocalTensor<fp4x2_e2m1_t>  t_nib    = q_weight_nibbles.AllocTensor<fp4x2_e2m1_t>();
    LocalTensor<bfloat16_t>    t_wide   = q_weight_expanded.AllocTensor<bfloat16_t>();
    LocalTensor<uint8_t>       t_wscale = q_weight_scales.AllocTensor<uint8_t>();
    LocalTensor<bfloat16_t>    t_splane = q_scale_plane.AllocTensor<bfloat16_t>();
    LocalTensor<uint16_t>      t_stemp  = q_scale_temporary.AllocTensor<uint16_t>();

    // The sub-core gate, as production has it. Nothing is driven either way;
    // it is here so the grid topology under test is production's.
    const bool drives_cube = (GetSubBlockIdx() % GetSubBlockNum()) == 0;

    // A liveness word, so "retired" is distinguishable from "the host gave up
    // before the kernel was ever entered". Written only by the driving
    // sub-core, and only on the AIV side -- the Cube has no business storing
    // to a scalar GM word, and on the AIC path this whole function body is
    // reached only after the KfcServer's service loop has already returned.
    if (drives_cube && status_out != nullptr) {
        GlobalTensor<int32_t> status_gm;
        status_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(status_out), 8);
        status_gm.SetValue(0, 1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                 DcciDst::CACHELINE_OUT>(status_gm);
    }

    q_scale_temporary.FreeTensor(t_stemp);
    q_scale_plane.FreeTensor(t_splane);
    q_weight_scales.FreeTensor(t_wscale);
    q_weight_expanded.FreeTensor(t_wide);
    q_weight_nibbles.FreeTensor(t_nib);
    q_activation_sub_bf16.FreeTensor(t_asub);
    q_subgemm_down.FreeTensor(t_sgd);
    q_down_accumulator.FreeTensor(t_down);
    q_tokens_sub_bf16_pong.FreeTensor(t_bpong);
    q_tokens_sub_bf16_ping.FreeTensor(t_bping);
    q_tokens_f32_pong.FreeTensor(t_xpong);
    q_tokens_f32_ping.FreeTensor(t_xping);
    q_subgemm_fwd.FreeTensor(t_sgf);
    q_up_accumulator.FreeTensor(t_up);
    q_gate_accumulator.FreeTensor(t_gate);
    q_activation_resident.FreeTensor(t_act);
}

}  // namespace moe::kernels::arch35::kfc_lifecycle
