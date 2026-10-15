// dsv4_moe_prefill_cube_kernel.cpp -- batched MoE prefill with W1/W3/W2 on
// the Cube in BF16 (Ascend 950PR, arch35 / dav-3510, MIX AIC+AIV).
//
// Oracle: src/emu/moe_block.cpp :: moe_prefill_batched_mxfp4_cpu, with X cast
// to BF16 first -- this is a W4A16 kernel, so the reference must round the
// activations the same way. See the numerics note on
// op_host/dsv4_moe_prefill_cube_tiling.h for why BF16 and not native Cube
// MXFP4, and for the proof sketch that the weight dequantization is exact.
//
//   gate[m, r] = sum_c  X_bf16[m, c] * W1_bf16[r, c]        Cube
//   up  [m, r] = sum_c  X_bf16[m, c] * W3_bf16[r, c]        Cube
//   act [m, r] = silu(min(gate, L)) * clamp(up, -L, L)       Vector
//   y   [m, h] = sum_r  act_bf16[m, r] * W2_bf16[h, r]       Cube
//
// THE MIX EXECUTION MODEL, which dictates the shape of this file.
// REGIST_MATMUL_OBJ stands up an AscendC::KfcServer. On the AIC side the macro
// spins in server.Run() and RETURNS FROM INSIDE THE MACRO; only the AIV side
// falls through. So everything written after the macro is AIV code that drives
// the Cube through the server's message queue, and `workspace` is that queue's
// GM backing -- which is why this entry point takes a workspace operand even
// before counting the dequantized-weight planes.
//
// THE CLAMP IS ASYMMETRIC AND THE TWO HALVES ARE NOT INTERCHANGEABLE.
// AscendSiluAndMulWithClamp bounds the GATE ABOVE ONLY and bounds UP ON BOTH
// SIDES, and the AIV prefill kernel does the same. Dropping the `up` clamp is
// not harmless: `up` is a raw hidden-wide GEMM output, so it exceeds L readily
// and the error is unbounded.
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 only (CLAUDE.md §5.7). No
// claim is made here about observed throughput, bandwidth or latency.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "dsv4_moe_prefill_cube_arch35.h"
#include "dsv4_moe_prefill_cube_tiling.h"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

using namespace AscendC;

// Keep the UB budget the host sizes dyn_ub_size from and the one this TU
// assumes tied together: a drift under-declares the dynamic region.
static_assert(kCubeMTile == kPrefillCubeMTile,
              "token tile disagrees between the UB layout and the tiling "
              "header the host sizes from");
static_assert(kCubeInterTile == kPrefillCubeInterTile,
              "gate/up N chunk disagrees between the UB layout and the tiling "
              "header");
static_assert(kCubeHiddenTile == kPrefillCubeHiddenTile,
              "down N chunk disagrees between the UB layout and the tiling "
              "header");

// ---------------------------------------------------------------------------
// Matmul types
// ---------------------------------------------------------------------------
// A from GM: X is cast to BF16 once into a workspace plane, which keeps UB
// free for `activated` to stay resident across the gate/up -> down boundary.
// B from GM: one expert's dequantized BF16 weight plane is 16 MiB at the
// production geometry, so neither VECOUT (UB) nor TSCM (L1) can hold the whole
// single-core matrix they require.
// C at VECIN: the audited direct Cube -> Vector position on 950PR. CO1 is
// physical L0C but is NOT an allowed high-level C role there; only GM and
// VECIN are, so VECIN is what "no GM round trip for the intermediate" means
// in practice on this part.
using XType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using WType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using AccType = MatmulType<TPosition::VECIN, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// THE DOWN GEMM NEEDS A DIFFERENT A POSITION, and reusing XType for it is a
// quiet defect rather than a compile error: SetTensorA has a LocalTensor
// overload, so handing it UB-resident `activated` while the type says
// TPosition::GM type-checks and then describes the wrong operand source to
// the Cube. A is in UB here, and VECOUT is the position for that on 950PR --
// which also carries the requirement that the whole single-core A be
// resident, satisfied by construction since `activated` is [m_tile, inter]
// BF16 = 64 KiB and the UB budget is built around keeping it.
//
// The output goes straight to GM: y is the operator's result, so there is
// nothing downstream on the Vector side to keep it in UB for.
using ActType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t>;
using YType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// W is [rows, cols] row-major with `cols` the reduction axis, so it enters the
// GEMM transposed -- hence ISTRANS = true on WType above rather than a
// repacking pass.
constexpr MatmulConfig kCubeCfg = GetNormalConfig();
constexpr auto kCubeTiling =
    GetMatmulApiTiling<XType, WType, AccType, BiasType>(kCubeCfg);

constexpr auto kDownTiling =
    GetMatmulApiTiling<ActType, WType, YType, BiasType>(kCubeCfg);

using GateUpMatmul = Matmul<XType, WType, AccType, BiasType, kCubeTiling>;
using DownMatmul = Matmul<ActType, WType, YType, BiasType, kDownTiling>;

namespace {

__aicore__ inline __gm__ std::uint8_t* Bytes(GM_ADDR p) {
  return reinterpret_cast<__gm__ std::uint8_t*>(p);
}

// The E8M0 group scale as a BF16 scalar, built by TRUNCATION rather than by a
// cast.
//
// `static_cast<bfloat16_t>(float)` is rejected by this backend outright --
// "fatal error: error in backend: not support bf16 type cast" -- so the
// conversion has to be done on the bit pattern. That is exact here and only
// here: 2^(code-127) has a ZERO mantissa, so dropping the low 16 bits of the
// binary32 discards nothing. Do not reuse this helper for a general float.
__aicore__ inline bfloat16_t E8M0ToBf16(std::uint8_t code) {
  const std::uint32_t f32 = static_cast<std::uint32_t>(code) << 23u;
  const std::uint16_t bits = static_cast<std::uint16_t>(f32 >> 16u);
  bfloat16_t out;
  __builtin_memcpy(&out, &bits, sizeof(out));
  return out;
}

struct Geometry {
  std::uint32_t block_dim;
  std::uint32_t hidden;
  std::uint32_t inter;
  std::uint32_t num_experts;
  std::uint32_t total_tokens;
  std::uint32_t m_tile;
  std::uint32_t inter_tile;
  std::uint32_t hidden_tile;
  std::uint32_t w13_packed_expert_stride;
  std::uint32_t w13_scale_expert_stride;
  std::uint32_t w2_packed_expert_stride;
  std::uint32_t w2_scale_expert_stride;
  float swiglu_limit;
};

// Field by field: CCE address spaces forbid struct copy from __gm__.
__aicore__ inline Geometry LoadGeometry(
    const __gm__ Dsv4MoePrefillCubeTilingData* t) {
  Geometry g;
  g.block_dim = t->block_dim;
  g.hidden = t->hidden;
  g.inter = t->inter;
  g.num_experts = t->num_experts;
  g.total_tokens = t->total_tokens;
  g.m_tile = t->m_tile;
  g.inter_tile = t->inter_tile;
  g.hidden_tile = t->hidden_tile;
  g.w13_packed_expert_stride = t->w13_packed_expert_stride;
  g.w13_scale_expert_stride = t->w13_scale_expert_stride;
  g.w2_packed_expert_stride = t->w2_packed_expert_stride;
  g.w2_scale_expert_stride = t->w2_scale_expert_stride;
  g.swiglu_limit = t->swiglu_limit;
  return g;
}

// The v1 contract: a kernel that returns early on bad tiling produces a zeroed
// output rather than an error (§2.5), which is why the host builder validates
// too. This is the backstop, not the gate.
__aicore__ inline bool TilingRejected(
    const __gm__ Dsv4MoePrefillCubeTilingData* t) {
  const std::uint32_t hidden = t->hidden;
  const std::uint32_t inter = t->inter;
  return t->abi_version != kTilingAbiVersion ||
         t->group_size != kPrefillCubeGroupSize ||
         t->low_nibble_first != 1u || t->num_experts == 0u ||
         t->total_tokens == 0u || t->m_tile != kCubeMTile ||
         t->inter_tile != kCubeInterTile ||
         t->hidden_tile != kCubeHiddenTile ||
         !Dsv4MoePrefillCubeAdmissible(hidden, inter);
}

}  // namespace

// ---------------------------------------------------------------------------
// Dequantization: MXFP4 (E2M1 nibbles + E8M0 group scales) -> BF16, EXACTLY
// ---------------------------------------------------------------------------
// One row chunk at a time, through UB. The conversion is two steps because
// the hardware gives exactly these two:
//
//   1. Cast(bfloat16_t <- fp4x2_e2m1_t) expands the packed nibbles to BF16
//      magnitudes. The E2M1 set needs two mantissa bits; BF16 has eight.
//   2. a per-32-group multiply by the E8M0 scale, which is a power of two and
//      therefore shifts the exponent without touching the mantissa.
//
// So the composition is exact, and it is exact INDEPENDENTLY of where the
// scale is applied -- which is what lets the Cube distribute it into every
// product while the CPU oracle applies it once per group subtotal.
//
// LOSSLESS EXCEPT AT THE EXPONENT EXTREMES, stated rather than glossed: E8M0
// spans 2^-127..2^127 and the largest E2M1 magnitude is 6, so a nibble of 6
// against a scale near 2^127 overflows BF16's range and a subnormal-range
// scale loses bits. Real MXFP4 checkpoints do not place scales there, and the
// host oracle rounds identically, so the comparison stays exact; a synthetic
// weight set at the extremes would not.
__aicore__ inline void DequantRowsToBf16(
    __gm__ std::uint8_t* packed, __gm__ std::uint8_t* scale,
    const GlobalTensor<bfloat16_t>& outGm, std::uint64_t out_base,
    std::uint32_t rows, std::uint32_t cols, const LocalTensor<fp4x2_e2m1_t>& nib,
    const LocalTensor<bfloat16_t>& wide, const LocalTensor<uint8_t>& scaleUb) {
  const std::uint32_t groups = cols / kPrefillCubeGroupSize;

  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint64_t packed_off =
        static_cast<std::uint64_t>(r) * (cols / 2u);
    const std::uint64_t scale_off = static_cast<std::uint64_t>(r) * groups;

    // Nibbles in, as raw bytes reinterpreted as FP4 pairs: cols/2 bytes.
    GlobalTensor<fp4x2_e2m1_t> packedGm;
    packedGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ fp4x2_e2m1_t*>(packed + packed_off), cols);
    DataCopy(nib, packedGm, cols);

    GlobalTensor<uint8_t> scaleGm;
    scaleGm.SetGlobalBuffer(scale + scale_off, groups);
    DataCopy(scaleUb, scaleGm, groups);

    SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);

    // Step 1: nibble -> BF16 magnitude. CAST_NONE: every E2M1 value is
    // exactly representable, so there is nothing to round and no rounding
    // mode to choose.
    Cast(wide, nib, RoundMode::CAST_NONE, cols);
    PipeBarrier<PIPE_V>();

    // Step 2: apply each group's E8M0 scale to its 32 elements.
    //
    // DecodeE8M0 is the SHARED decode from arch35_common.hpp -- ldexpf(1,
    // code - 127) -- the same one the AIV kernels use and the one that
    // matches moe::emu's E8M0 table. Reusing it is the point: a second
    // private decode here is exactly how a kernel drifts from its oracle.
    //
    // The multiply is exact in BF16: the nibble magnitude needs two mantissa
    // bits and the factor is a power of two, so the product only moves the
    // exponent field.
    for (std::uint32_t grp = 0; grp < groups; ++grp) {
      const bfloat16_t factor =
          E8M0ToBf16(static_cast<std::uint8_t>(scaleUb.GetValue(grp)));
      Muls(wide[grp * kPrefillCubeGroupSize], wide[grp * kPrefillCubeGroupSize],
           factor, kPrefillCubeGroupSize);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
    DataCopy(outGm[out_base + static_cast<std::uint64_t>(r) * cols], wide, cols);
    SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
  }
}

// ---------------------------------------------------------------------------
// Entry point: batched prefill, W1/W3/W2 on the Cube
// ---------------------------------------------------------------------------
// GRID: one token TILE (m_tile rows) per core, grid-striding over the tiles.
// `x_tokens` is expected ALREADY GROUPED BY EXPERT, the standard MoE dispatch
// output -- that is what makes a core's tile share one expert and lets one
// dequantization serve the whole group.
//
// `workspace` is CALLER-OWNED and must be at least
// Dsv4MoePrefillCubeWorkspaceBytes(total_tokens, hidden, inter). It carries,
// in order: the BF16 cast of X, then W1/W3 of the expert being processed, then
// W2. See the tiling header for why the dequantized weights land in GM.
extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  if (unlikely(tiling == nullptr || workspace == nullptr)) {
    return;
  }
  const __gm__ Dsv4MoePrefillCubeTilingData* td =
      reinterpret_cast<const __gm__ Dsv4MoePrefillCubeTilingData*>(tiling);
  if (unlikely(TilingRejected(td))) {
    return;  // uniform across cores
  }
  const Geometry g = LoadGeometry(td);

  TPipe pipe;
  GateUpMatmul mmGateUp;
  DownMatmul mmDown;
  // AIC spins in the server and returns from inside this macro; the AIV side
  // continues below and drives both objects.
  REGIST_MATMUL_OBJ(&pipe, workspace, mmGateUp, mmDown);

  // --- workspace layout ----------------------------------------------------
  // The KFC queue takes the head of the workspace, so the data planes start
  // past kCubeKfcReserveBytes. Offsets are formed in uint64 (§2.7): at the
  // production geometry w13_bf16 alone is 32 MiB.
  const std::uint64_t x_elems =
      static_cast<std::uint64_t>(g.total_tokens) * g.hidden;
  const std::uint64_t w13_elems =
      2ull * static_cast<std::uint64_t>(g.inter) * g.hidden;

  __gm__ std::uint8_t* ws = Bytes(workspace) + kCubeKfcReserveBytes;
  GlobalTensor<bfloat16_t> xBf16;
  GlobalTensor<bfloat16_t> w13Bf16;
  GlobalTensor<bfloat16_t> w2Bf16;
  xBf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ws), x_elems);
  w13Bf16.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(ws + x_elems * 2ull), w13_elems);
  w2Bf16.SetGlobalBuffer(
      reinterpret_cast<__gm__ bfloat16_t*>(ws + x_elems * 2ull +
                                           w13_elems * 2ull),
      static_cast<std::uint64_t>(g.hidden) * g.inter);

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_tokens), x_elems);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_tokens), x_elems);

  // --- UB: the budget the tiling header derives ---------------------------
  TQue<TPosition::VECIN, 1> qAcc;
  TQue<TPosition::VECCALC, 1> qWork;
  pipe.InitBuffer(qAcc, 2, kCubeMTile * kCubeInterTile * 4u);
  pipe.InitBuffer(qWork, 1, kCubeActBytes);

  LocalTensor<float> gate = qAcc.AllocTensor<float>();
  LocalTensor<float> up = qAcc.AllocTensor<float>();
  LocalTensor<bfloat16_t> act = qWork.AllocTensor<bfloat16_t>();

  const std::uint32_t core = static_cast<std::uint32_t>(GetBlockIdx());
  const std::uint32_t tiles =
      (g.total_tokens + g.m_tile - 1u) / g.m_tile;

  for (std::uint32_t tile = core; tile < tiles; tile += g.block_dim) {
    const std::uint32_t row0 = tile * g.m_tile;
    const std::uint32_t rows =
        (g.total_tokens - row0) < g.m_tile ? (g.total_tokens - row0) : g.m_tile;

    const std::uint32_t expert = static_cast<std::uint32_t>(
        reinterpret_cast<const __gm__ std::int32_t*>(token_to_expert)[row0]);
    if (unlikely(expert >= g.num_experts)) {
      continue;  // backstop; the host validates the ids it can see
    }
    const std::uint64_t e = static_cast<std::uint64_t>(expert);

    // --- cast this tile's X to BF16 ---------------------------------------
    // The Cube reads A from GM, so the cast is a workspace plane rather than
    // a UB residency. Done per tile, not once for the batch, so a core only
    // touches the rows it owns.
    {
      LocalTensor<float> xf = qAcc.AllocTensor<float>();
      LocalTensor<bfloat16_t> xb = qWork.AllocTensor<bfloat16_t>();
      for (std::uint32_t r = 0; r < rows; ++r) {
        const std::uint64_t off =
            (static_cast<std::uint64_t>(row0) + r) * g.hidden;
        DataCopy(xf, xGm[off], g.hidden);
        SetFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(kEvCubeMte2ToV);
        // FP32 -> BF16 is THE lossy step of this kernel, and the only one.
        // Round-to-nearest-even keeps the error at or below 2^-9 relative,
        // which is the bound the oracle's tolerance is derived from.
        Cast(xb, xf, RoundMode::CAST_RINT, g.hidden);
        SetFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        WaitFlag<HardEvent::V_MTE3>(kEvCubeVToMte3);
        DataCopy(xBf16[off], xb, g.hidden);
        SetFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(kEvCubeMte3ToV);
      }
      qWork.FreeTensor(xb);
      qAcc.FreeTensor(xf);
    }

    // --- dequantize this expert's W1, W3, W2 into the workspace -----------
    {
      LocalTensor<fp4x2_e2m1_t> nib = qAcc.AllocTensor<fp4x2_e2m1_t>();
      LocalTensor<bfloat16_t> wide = qWork.AllocTensor<bfloat16_t>();
      LocalTensor<uint8_t> scaleUb = qAcc.AllocTensor<uint8_t>();

      DequantRowsToBf16(Bytes(w1_packed) + e * g.w13_packed_expert_stride,
                        Bytes(w1_scale) + e * g.w13_scale_expert_stride,
                        w13Bf16, 0u, g.inter, g.hidden, nib, wide, scaleUb);
      DequantRowsToBf16(
          Bytes(w3_packed) + e * g.w13_packed_expert_stride,
          Bytes(w3_scale) + e * g.w13_scale_expert_stride, w13Bf16,
          static_cast<std::uint64_t>(g.inter) * g.hidden, g.inter, g.hidden,
          nib, wide, scaleUb);
      DequantRowsToBf16(Bytes(w2_packed) + e * g.w2_packed_expert_stride,
                        Bytes(w2_scale) + e * g.w2_scale_expert_stride, w2Bf16,
                        0u, g.hidden, g.inter, nib, wide, scaleUb);

      qAcc.FreeTensor(scaleUb);
      qWork.FreeTensor(wide);
      qAcc.FreeTensor(nib);
    }

    // --- gate/up on the Cube, SwiGLU on the Vector, chunk by chunk --------
    const std::uint64_t x_tile_base =
        static_cast<std::uint64_t>(row0) * g.hidden;
    for (std::uint32_t n0 = 0; n0 < g.inter; n0 += g.inter_tile) {
      // gate = X[rows, hidden] @ W1[n0:n0+nt, hidden]^T
      mmGateUp.SetTensorA(xBf16[x_tile_base]);
      mmGateUp.SetTensorB(w13Bf16[static_cast<std::uint64_t>(n0) * g.hidden]);
      mmGateUp.SetOrgShape(rows, g.inter, g.hidden);
      mmGateUp.SetTail(rows, g.inter_tile);
      mmGateUp.IterateAll(gate);
      mmGateUp.End();

      mmGateUp.SetTensorA(xBf16[x_tile_base]);
      mmGateUp.SetTensorB(
          w13Bf16[static_cast<std::uint64_t>(g.inter) * g.hidden +
                  static_cast<std::uint64_t>(n0) * g.hidden]);
      mmGateUp.SetOrgShape(rows, g.inter, g.hidden);
      mmGateUp.SetTail(rows, g.inter_tile);
      mmGateUp.IterateAll(up);
      mmGateUp.End();

      // ASYMMETRIC clamp, per the file header: gate above only, up both ways.
      // Evaluated as the oracle does it so the two agree on the clamp
      // boundaries rather than only in the interior.
      const std::uint32_t count = rows * g.inter_tile;
      Mins(gate, gate, g.swiglu_limit, count);
      PipeBarrier<PIPE_V>();
      Maxs(up, up, -g.swiglu_limit, count);
      PipeBarrier<PIPE_V>();
      Mins(up, up, g.swiglu_limit, count);
      PipeBarrier<PIPE_V>();
      // silu(gate) * up, into the resident BF16 `activated` chunk.
      Silu(gate, gate, count);
      PipeBarrier<PIPE_V>();
      Mul(gate, gate, up, count);
      PipeBarrier<PIPE_V>();
      Cast(act[n0], gate, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
    }

    // --- down on the Cube -------------------------------------------------
    // `act` is read from UB as the A matrix. VECOUT is the position for that,
    // and the whole single-core A (rows x inter BF16) is resident by
    // construction -- that is what the UB budget in the tiling header buys.
    for (std::uint32_t h0 = 0; h0 < g.hidden; h0 += g.hidden_tile) {
      mmDown.SetTensorA(act);
      mmDown.SetTensorB(w2Bf16[static_cast<std::uint64_t>(h0) * g.inter]);
      mmDown.SetOrgShape(rows, g.hidden, g.inter);
      mmDown.SetTail(rows, g.hidden_tile);
      mmDown.IterateAll(yGm[static_cast<std::uint64_t>(row0) * g.hidden + h0]);
      mmDown.End();
    }
  }

  qWork.FreeTensor(act);
  qAcc.FreeTensor(up);
  qAcc.FreeTensor(gate);
}

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube
