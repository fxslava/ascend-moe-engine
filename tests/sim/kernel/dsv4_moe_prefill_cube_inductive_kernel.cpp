/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */

// Inductive boundary-grid simulator wrapper for dsv4_moe_prefill_cube.
// Textually includes the production kernel to execute verbatim compute bodies
// over a collapsed trip set S instead of all tiles.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"
#include "dsv4_moe_prefill_cube_cube_tiling.h"
#include "dsv4_moe_prefill_cube_sample_set.h"

// Textually include the production TU for verbatim compute logic
#include "../../../src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;
using moe::sim::prefill_cube::BuildTileSampleSet;
using moe::sim::prefill_cube::kMaxSamples;
using moe::sim::prefill_cube::kRingDepth;

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_inductive_sim(
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
    const __gm__ Dsv4MoePrefillCubeTilingData* tiling_data =
        reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling_config);
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
    pipe.InitBuffer(q_scale_plane,           1, kCubeRSub * kCubeKSub * sizeof(bfloat16_t));
    pipe.InitBuffer(q_scale_temporary,       1, 16 * sizeof(uint16_t));

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

    const std::uint32_t total_tiles = (geom.total_tokens + geom.token_tile - 1u) / geom.token_tile;

    std::uint32_t sample_indices[kMaxSamples] = {};
    std::uint32_t sample_count = BuildTileSampleSet(total_tiles, kRingDepth, sample_indices);

    // Behind BOTH Cube tilings, not at sizeof(the POD): the gate tiling
    // occupies offset 72 now. Kept in step with the sim launch wrapper.
    const std::uint32_t cap = *reinterpret_cast<const __gm__ std::uint32_t*>(
        Bytes(tiling_config) + kPrefillCubeStagingBytes);
    if (cap != 0u && cap < sample_count) {
        sample_count = cap;
    }

    const std::uint32_t core_idx = static_cast<std::uint32_t>(GetBlockIdx());

    for (std::uint32_t s_idx = 0; s_idx < sample_count; ++s_idx) {
        const std::uint32_t tile_idx = sample_indices[s_idx];
        if ((tile_idx % geom.block_dim) == core_idx) {
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
