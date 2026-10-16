// SIM ONLY -- a collapsed boundary-grid variant of dsv4_moe_prefill_cube for
// the dav-3510 CAModel (MIX AIC+AIV, KERNEL_TYPE_MIX_AIC_1_1).
//
// Lives under tests/sim/kernel/ and is compiled into test targets only; no
// production library links it, and tests/sim/CMakeLists.txt asserts that
// rather than trusting it. Production op_kernel/ carries no verification
// branch and no #ifdef SIM because of this file.
//
// WHAT IS SHARED AND WHAT IS NOT
// This TU #includes the production kernel, so every COMPUTE body here is
// verbatim production code: ProcessTokenTile, DequantSubTileToL1, E8M0ToBf16,
// LoadGeometry, TilingRejected, the Buffers bundle, the Matmul type aliases
// and their compile-time tiling. Only the TRIP SCHEDULE differs -- which of
// the token tiles run. A collapsed run against a forked body would prove
// nothing about the kernel that ships, so the body is not copied, it is
// called.
//
// Including a .cpp is deliberate and is the same choice the two
// dsv4_moe_block sim kernels make: ProcessTokenTile is `inline` inside the
// production TU and cannot be referenced across a link. The production
// __global__ entry comes along unused, so this object must NOT be linked
// beside dsv4_moe_prefill_cube_kernel.o.
//
// THE SCHEDULE, AND WHY IT IS THE PRODUCTION MAP
// Production grid-strides over tiles:
//
//     for (tile = core; tile < tiles; tile += block_dim)
//
// This file strides over SAMPLE POSITIONS instead:
//
//     for (i = core; i < |S|; i += block_dim)   ->  ProcessTokenTile(S[i])
//
// so a core still takes every block_dim-th element of the sequence it is
// given, and adjacent sampled tiles still land on adjacent cores. The map
// under test is production's; only the sequence is shorter. Sampling global
// tile indices per core instead would hand a core tiles it does not own and
// exercise an addressing pattern production never performs.
//
// THERE IS NO CROSS-CORE HANDSHAKE TO UNBALANCE, which is what makes this
// collapse safe. The kernel issues no CrossCoreSetFlag / CrossCoreWaitFlag
// and no SyncAll; its only cross-core traffic is the Matmul KfcServer, and on
// the AIC side that is a SERVICE loop -- it answers however many messages the
// AIV client sends and quits on the client's teardown. Collapsing the AIV
// trip count cannot stand the AIC up waiting for a message that never comes,
// which is the artefact this technique normally has to design around. The
// sample set is still built ONCE, from the tiling, so both sides see one
// sequence.
//
// THE SAMPLE CAP IS A SEPARATE GM WORD, NOT A TILING FIELD. The driver needs
// to run 1 or 2 sampled tiles to MEASURE this gate's fixed cost (the cost
// model for CAModel is measured, never derived -- three predictions for these
// suites have now been wrong). Adding a field to Dsv4MoePrefillCubeTilingData
// would break its 72-byte wire format, which is public ABI, so the cap
// travels as its own operand. A cap of 0 means "run all of S".
//
// NEVER EXECUTED ON HARDWARE: compiled for dav-3510 and run under CAModel
// only (CLAUDE.md §5.7).

#include "dsv4_moe_prefill_cube_kernel.cpp"  // verbatim production bodies

#include "dsv4_moe_prefill_cube_sample_set.h"

extern "C" __global__ __aicore__ void dsv4_moe_prefill_cube_inductive_sim(
    GM_ADDR x_tokens, GM_ADDR w1_packed, GM_ADDR w1_scale, GM_ADDR w3_packed,
    GM_ADDR w3_scale, GM_ADDR w2_packed, GM_ADDR w2_scale,
    GM_ADDR token_to_expert, GM_ADDR y_tokens, GM_ADDR workspace,
    GM_ADDR sample_cap, GM_ADDR tiling) {
  using namespace moe::kernels::arch35;
  using namespace moe::kernels::arch35::dsv4_moe_prefill_cube;
  using moe::sim::prefill_cube::BuildTileSampleSet;
  using moe::sim::prefill_cube::kMaxSamples;
  using moe::sim::prefill_cube::kRingDepth;

  // --- prologue: production's, unchanged ----------------------------------
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
  // continues below. This is the handshake the gate exists to execute.
  REGIST_MATMUL_OBJ(&pipe, workspace, mmGateUp, mmDown);

  GlobalTensor<float> xGm;
  GlobalTensor<float> yGm;
  const std::uint64_t elems =
      static_cast<std::uint64_t>(g.total_tokens) * g.hidden;
  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_tokens), elems);
  yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y_tokens), elems);

  // THE WORKSPACE IS THE KFC QUEUE AND NOTHING ELSE, which is why this file
  // no longer builds any GlobalTensor over it. The predecessor laid three
  // dequantized planes out here -- X in BF16, W1+W3, W2 -- and the driver
  // bounds-checked them. There is nothing left to view: every weight sub-tile
  // is decoded into UB and pushed to L1, so the driver's workspace check
  // became an assertion that the region past the KFC reserve is NEVER
  // written, which is a stronger statement than the old bounds check and the
  // one this schedule is entitled to.

  // --- UB and L1: production's layout, verbatim ---------------------------
  TQue<TPosition::VECCALC, 1> qAct;
  TQue<TPosition::VECCALC, 1> qGate;
  TQue<TPosition::VECCALC, 1> qUp;
  TQue<TPosition::VECIN, 1> qC1;
  TQue<TPosition::VECCALC, 1> qXF32;
  TQue<TPosition::VECOUT, 1> qXSub;
  TQue<TPosition::VECCALC, 1> qYAcc;
  TQue<TPosition::VECIN, 1> qC2;
  TQue<TPosition::VECOUT, 1> qASub;
  TQue<TPosition::VECCALC, 1> qNib;
  TQue<TPosition::VECCALC, 1> qWide;
  TQue<TPosition::VECCALC, 1> qScale;
  pipe.InitBuffer(qAct, 1, kCubeActBytes);
  pipe.InitBuffer(qGate, 1, kCubeAcc1Bytes);
  pipe.InitBuffer(qUp, 1, kCubeAcc1Bytes);
  pipe.InitBuffer(qC1, 1, kCubeAcc1Bytes);
  pipe.InitBuffer(qXF32, 1, kCubeXF32Bytes);
  pipe.InitBuffer(qXSub, 1, kCubeXSubBytes);
  pipe.InitBuffer(qYAcc, 1, kCubeAcc2Bytes);
  pipe.InitBuffer(qC2, 1, kCubeAcc2Bytes);
  pipe.InitBuffer(qASub, 1, kCubeASubBytes);
  pipe.InitBuffer(qNib, 1, kCubeNibBytes);
  pipe.InitBuffer(qWide, 1, kCubeWideBytes);
  pipe.InitBuffer(qScale, 1, kCubeScaleBytes);

  TQue<TPosition::TSCM, 2> qB1;
  TQue<TPosition::TSCM, 1> qB2;
  pipe.InitBuffer(qB1, 2, kCubeB1SlotBytes);
  pipe.InitBuffer(qB2, 1, kCubeB2SlotBytes);

  Buffers b;
  b.act = qAct.AllocTensor<bfloat16_t>();
  b.gate_acc = qGate.AllocTensor<float>();
  b.up_acc = qUp.AllocTensor<float>();
  b.c1 = qC1.AllocTensor<float>();
  b.x_f32 = qXF32.AllocTensor<float>();
  b.x_sub = qXSub.AllocTensor<bfloat16_t>();
  b.y_acc = qYAcc.AllocTensor<float>();
  b.c2 = qC2.AllocTensor<float>();
  b.a_sub = qASub.AllocTensor<bfloat16_t>();
  b.w_nib = qNib.AllocTensor<fp4x2_e2m1_t>();
  b.w_wide = qWide.AllocTensor<bfloat16_t>();
  b.w_scale = qScale.AllocTensor<uint8_t>();

  const std::uint32_t core = static_cast<std::uint32_t>(GetBlockIdx());
  const std::uint32_t tiles = (g.total_tokens + g.m_tile - 1u) / g.m_tile;

  // --- THE ONLY DEVIATION: the trip schedule ------------------------------
  std::uint32_t sample[kMaxSamples];
  std::uint32_t n = BuildTileSampleSet(tiles, kRingDepth, sample);
  if (sample_cap != nullptr) {
    const std::uint32_t cap =
        reinterpret_cast<const __gm__ std::uint32_t*>(sample_cap)[0];
    // 0xFFFFFFFF -- SETUP ONLY: stand up the TPipe, the Matmul objects and the
    // KfcServer, then tear them down without running a single tile. A
    // diagnostic, and a load-bearing one: it bisects a run-time fault into
    // "the MIX bring-up" versus "the tile body", which from the outside look
    // identical -- both present as the simulator spinning. Nothing else can
    // tell them apart without hardware.
    if (cap == 0xFFFFFFFFu) {
      n = 0u;
    } else if (cap != 0u && cap < n) {
      n = cap;
    }
  }

  for (std::uint32_t i = core; i < n; i += g.block_dim) {
    ProcessTokenTile(g, sample[i], w1_packed, w1_scale, w3_packed, w3_scale,
                     w2_packed, w2_scale, token_to_expert, xGm, yGm, b, qB1,
                     qB2, mmGateUp, mmDown);
  }

  qScale.FreeTensor(b.w_scale);
  qWide.FreeTensor(b.w_wide);
  qNib.FreeTensor(b.w_nib);
  qASub.FreeTensor(b.a_sub);
  qC2.FreeTensor(b.c2);
  qYAcc.FreeTensor(b.y_acc);
  qXSub.FreeTensor(b.x_sub);
  qXF32.FreeTensor(b.x_f32);
  qC1.FreeTensor(b.c1);
  qUp.FreeTensor(b.up_acc);
  qGate.FreeTensor(b.gate_acc);
  qAct.FreeTensor(b.act);
}
