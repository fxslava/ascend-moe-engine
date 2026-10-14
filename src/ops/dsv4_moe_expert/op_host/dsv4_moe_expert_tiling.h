#ifndef DSV4_MOE_EXPERT_TILING_H
#define DSV4_MOE_EXPERT_TILING_H

// Tiling POD and v1 contract geometry for dsv4_moe_expert, the fused
// DeepSeek-V4 routed-expert runner: gate GEMV, up GEMV, the model's
// asymmetric clamped SwiGLU, and the down GEMV, for one token.
//
// Included by both op_kernel/ and op_host/; no <acl/acl.h> (CLAUDE.md §2.4).

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kDsv4ExpertRowsPerTile = 32;
inline constexpr std::uint32_t kDsv4ExpertGroupSize = 32;
inline constexpr std::uint32_t kDsv4ExpertBlockDim = 32;  // 32 AIVs, grid-stride

// The DeepSeek-V4 SwiGLU clamp bound. Carried in tiling rather than hard-coded
// so the value is part of the host-side contract instead of being buried in
// device code -- and so a model with a different limit can reuse the kernel.
//
// The clamp is ASYMMETRIC (AscendSiluAndMulWithClamp in
// vllm_ascend/models/deepseek_v4.py, ported in
// src/emu/moe_block.cpp::ExpertSwiglu):
//
//     gate = min(gate, +limit)              upper bound ONLY
//     up   = min(+limit, max(-limit, up))   two-sided
//     out  = silu(gate) * up
//
// Both halves matter and they are not interchangeable: see the note in the
// kernel. This is part of the model's function, not an overflow workaround.
inline constexpr float kDsv4ExpertSwigluLimit = 10.0F;

// ---------------------------------------------------------------------------
// Dynamic UB size (dyn_ub_size, the second chevron argument)
// ---------------------------------------------------------------------------
// Both entry points address their buffers from a SIMT VF, so both allocate
// from the DYNAMIC UB region via LocalMemAllocator -- the region whose size
// the host declares at launch. Each entry point has its own footprint, so
// each gets its own formula; the host passes the matching one.
//
// Allocation order in each Init() fixes the offsets and MUST match these sums.
//
// gate_up:  xF32       hidden  floats   (the token, resident for all tiles)
//           actTile    32      floats   (one 32-row tile of `activated`)
// down:     actF32     inter   floats   (activated, resident for all tiles)
//           yTile      32      floats
//
// uint64 because the extents are caller-supplied and §2.7 requires widening
// before the multiply; the caller checks the ceiling, then narrows.
inline constexpr std::uint64_t Dsv4ExpertGateUpDynUbBytes(
    std::uint32_t hidden) noexcept {
  return static_cast<std::uint64_t>(hidden) * 4u +
         static_cast<std::uint64_t>(kDsv4ExpertRowsPerTile) * 4u;
}

inline constexpr std::uint64_t Dsv4ExpertDownDynUbBytes(
    std::uint32_t inter) noexcept {
  return static_cast<std::uint64_t>(inter) * 4u +
         static_cast<std::uint64_t>(kDsv4ExpertRowsPerTile) * 4u;
}

// dyn_ub_size must not exceed 256 KiB - 8 KiB reserved - 32 KiB minimum
// DataCache - static. Neither entry point allocates statically (no TPipe), so
// the whole 216 KiB is available to the dynamic region.
inline constexpr std::uint32_t kDsv4ExpertDynUbCeilingBytes = 221184u;  // 216 KiB

// GM scratch: `activated`, written by the gate_up pass and read by the down
// pass. This is the operator's workspace, and the reason it is two entry
// points rather than one -- see the launch wrapper.
inline constexpr std::uint64_t Dsv4ExpertWorkspaceBytes(
    std::uint32_t inter) noexcept {
  return static_cast<std::uint64_t>(inter) * 4u;
}

// ---------------------------------------------------------------------------
// Admissible geometry -- ONE predicate, both sides
// ---------------------------------------------------------------------------
// The kernel's lane decomposition hands each of a row's 16 lanes
// cols/(16*2) pairs and fetches one E8M0 scale per 16 pairs, so it requires
//
//     cols % (kLanesPerRow * kDsv4ExpertGroupSize) == 0   i.e. cols % 512 == 0
//
// on BOTH reduction widths. Below that the per-lane group count floors to
// zero, every lane's group loop never runs, and the GEMV returns zero with no
// error -- the silent-wrong-answer case §2.5 describes.
//
// This predicate is the single definition of "admissible", consumed by
// MakeDsv4MoeExpertTiling() (the gate, §2.5) and by the kernel's
// TilingRejected() (the backstop). Keeping it in one place is the point:
// expressed twice, the host gate and the device backstop drift, and a
// geometry the host admits but the device rejects is exactly a zeroed output
// with no error reported.
//
// 16 lanes per row is the kernel's private thread geometry
// (op_kernel/dsv4_moe_expert_arch35.h, §1.2), so it is not imported here; the
// quantum it implies is restated as a contract constant with its derivation.
inline constexpr std::uint32_t kDsv4ExpertReductionQuantum = 512u;

// The admissibility TEST, written once as a macro.
//
// It has to be a macro, not a function, because BiSheng partitions functions
// by execution space and will not let one definition serve both: a plain
// function is __host__ and the kernel's call to it fails with "call to
// __host__ function from __aicore__ function", while an __aicore__ function is
// equally invisible to MakeDsv4MoeExpertTiling(), which is host code in the
// same ASC translation unit. Writing the condition twice is what this header
// exists to prevent, so the condition lives in the macro and the two typed
// wrappers below -- one per execution space, same name, same body -- are what
// call sites actually use.
//
// This is the SHAPE contract only. The UB ceiling is deliberately not part of
// it: the only path that allocates is LaunchDsv4MoeExpert(), which checks
// kDsv4ExpertDynUbCeilingBytes against both entry points' footprints itself,
// and the byte-count helpers above are host-space functions that an
// __aicore__ wrapper cannot call (they would link as undefined symbols on the
// device side).
//
// DSV4_TILING_TEST_OVERRIDE selects the SCALED-SIMULATION geometry: any width
// satisfying the reduction quantum, so the dav-3510 CAModel suite can execute
// the real kernel at a size the simulator finishes in seconds. No production
// configuration defines it; only tests/sim/CMakeLists.txt does, and it defines
// it for the BiSheng device TU and the host TUs alike so both sides agree.
// Everything the quantum protects is still enforced -- only the fixed
// 4096/2048 pin is relaxed.
#if defined(DSV4_TILING_TEST_OVERRIDE) && DSV4_TILING_TEST_OVERRIDE
#define MOE_DSV4_GEOMETRY_ADMISSIBLE(hidden, inter)                          \
  ((hidden) >= ::moe::kernels::arch35::kDsv4ExpertReductionQuantum &&        \
   (inter) >= ::moe::kernels::arch35::kDsv4ExpertReductionQuantum &&         \
   (hidden) % ::moe::kernels::arch35::kDsv4ExpertReductionQuantum == 0u &&   \
   (inter) % ::moe::kernels::arch35::kDsv4ExpertReductionQuantum == 0u)
#else
// v1 production geometry: DeepSeek-V4-Flash, hidden_size = 4096,
// moe_intermediate_size = 2048. Anything else is a geometry v1 does not
// implement, not a caller mistake (§2.6).
#define MOE_DSV4_GEOMETRY_ADMISSIBLE(hidden, inter)                          \
  ((hidden) == 4096u && (inter) == 2048u)
#endif

// Host wrapper. constexpr so the production pin folds at compile time.
inline constexpr bool Dsv4ExpertGeometryAdmissible(
    std::uint32_t hidden, std::uint32_t inter) noexcept {
  return MOE_DSV4_GEOMETRY_ADMISSIBLE(hidden, inter);
}

#if defined(__CCE__)
// Device wrapper -- same name, so the kernel's call site is identical and
// BiSheng picks this one by execution space. Not constexpr: the inputs are
// read from the tiling POD in GM at run time.
__aicore__ inline bool Dsv4ExpertGeometryAdmissible(std::uint32_t hidden,
                                                    std::uint32_t inter) {
  return MOE_DSV4_GEOMETRY_ADMISSIBLE(hidden, inter);
}
#endif

// ---------------------------------------------------------------------------
// sizeof == 56
// ---------------------------------------------------------------------------
struct Dsv4MoeExpertTilingData {
  std::uint32_t abi_version;            // 1
  std::uint32_t block_dim;              // 32 AIVs
  std::uint32_t hidden;                 // 4096
  std::uint32_t inter;                  // 2048
  std::uint32_t rows_per_block;         // 32
  std::uint32_t group_size;             // 32
  std::uint32_t w13_packed_row_stride;  // hidden/2 bytes
  std::uint32_t w13_scale_row_stride;   // hidden/32 bytes
  std::uint32_t w2_packed_row_stride;   // inter/2 bytes
  std::uint32_t w2_scale_row_stride;    // inter/32 bytes
  std::uint32_t low_nibble_first;       // 1; reject 0
  float swiglu_limit;                   // 10.0
  std::uint32_t reserved0;              // 0
  std::uint32_t reserved1;              // 0
};

MOE_ARCH35_ASSERT_TILING_POD(Dsv4MoeExpertTilingData, 56);

Status MakeDsv4MoeExpertTiling(std::uint32_t hidden, std::uint32_t inter,
                               std::uint32_t w13_packed_row_stride,
                               std::uint32_t w13_scale_row_stride,
                               std::uint32_t w2_packed_row_stride,
                               std::uint32_t w2_scale_row_stride,
                               Dsv4MoeExpertTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_EXPERT_TILING_H
