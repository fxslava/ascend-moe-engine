#ifndef DSV4_MOE_BLOCK_TILING_H
#define DSV4_MOE_BLOCK_TILING_H

// Tiling POD and contract for the two whole-MoE-block entry points:
//
//   dsv4_moe_decode_topk      M = 1,   top_k routed experts, router-weighted
//   dsv4_moe_prefill_batched  M >> 1,  one expert per token, grouped by expert
//
// Included by both op_kernel/ and op_host/; no <acl/acl.h> (CLAUDE.md §2.4).
//
// ONE POD, TWO ENTRY POINTS, following the shape fast_fp8_indexer_topk
// established and that dsv4_moe_expert_arch35.h already cites: one directory,
// one host ABI, one tiling POD (§1.5). The alternative -- a POD per entry
// point -- duplicates the geometry contract, and the two legs validate
// identically on everything except which of `top_k` / `total_tokens` carries
// the work.
//
// `mode` is in the POD on purpose. Each kernel rejects a tiling built for the
// other entry point, so handing the prefill kernel a decode plan is an early
// return rather than a plausible-looking wrong answer.
//
// WHY THE DECODE AND PREFILL KERNELS ARE SEPARATE AT ALL
// They have irreconcilable parallel decompositions. At M = 1 there is only
// one token, so the only intra-expert parallelism is over output rows -- and
// `down` reduces over all of `activated` while `gate_up` produces it tiled
// across cores, which is a cross-core dependency with no cheap AIV barrier.
// Decode therefore takes its parallelism from the top_k EXPERTS instead, one
// core per expert slot, and each core owns a whole expert end to end so
// `activated` never leaves UB. At M >> 1 every core owns whole tokens, so the
// same monolithic body is sound per token and the weight stream is reused
// across the tokens in a core's bucket.

#include "moe/kernels/arch35/status.hpp"
#include "moe/kernels/arch35/tiling_abi.hpp"

#include <cstdint>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kMoeBlockRowsPerTile = 32;
inline constexpr std::uint32_t kMoeBlockGroupSize = 32;

// The reduction quantum, as in the retired single-expert path: the SIMT lane
// decomposition gives each of a row's 16 lanes one 32-column scale group per
// warp step, so a row's group count must divide evenly across the lanes. See
// the derivation on dsv4_moe_expert_tiling.h.
inline constexpr std::uint32_t kMoeBlockReductionQuantum = 512u;

inline constexpr float kMoeBlockSwigluLimit = 10.0F;

// mode
inline constexpr std::uint32_t kMoeBlockModeDecode = 0u;
inline constexpr std::uint32_t kMoeBlockModePrefill = 1u;

// DECODE GRID: one core per expert slot.
//
// This is the resolution of a real fork, recorded because the alternatives
// are tempting and worse. Cores could instead SPLIT an expert's row tiles,
// which engages all 32 AIVs -- but then `activated` for that expert is again
// produced tiled across cores, and the down leg needs all of it. The ways out
// are (a) every core in the slot recomputes the whole gate/up leg, which
// multiplies the w1/w3 weight stream by the number of cores per slot, or
// (b) a GM round trip plus a barrier, which is the two-dispatch design this
// replaces. On a kernel measured at 3.76 FLOP/B -- weight-stream bound -- (a)
// pays the one resource that is actually scarce to buy idle-core occupancy.
//
// So: block_dim = top_k. With top_k = 6 that leaves 26 of 32 AIVs idle on the
// decode path, which is a real cost and is NOT claimed here to be the right
// trade on hardware -- it has never been measured (CLAUDE.md §5.7). It is the
// trade that keeps the weight stream at 1x and needs no barrier.
inline constexpr std::uint32_t kMoeBlockMaxTopK = 8u;

// PREFILL GRID: 32 AIVs, grid-stride over tokens.
inline constexpr std::uint32_t kMoeBlockPrefillBlockDim = 32u;

// ---------------------------------------------------------------------------
// UB footprint
// ---------------------------------------------------------------------------
// Both entry points run the same monolithic body on one core:
//
//   xF32        hidden floats   the token, resident for the whole expert
//   actF32      inter  floats   `activated`, written by gate/up, read by down
//                               -- NEVER goes to GM, which is the point
//   tile ring   2 x 32  floats  the down phase's double-buffered output tiles,
//                               with kMoeBlockTileSkewBytes BETWEEN them
//
// THE RING CONSTANTS LIVE HERE, not in op_kernel/<op>_arch35.h, because the
// host wrapper has to declare dyn_ub_size at the chevron and §1.2 makes
// op_kernel/ private -- a public header may not depend on it. The kernel's
// own copies are checked against these in dsv4_moe_block_kernel.cpp, so a
// drift between the two is a compile error rather than an under-declared
// dynamic region (which is the error-341 class).
inline constexpr std::uint32_t kMoeBlockTileSlots = 2u;
inline constexpr std::uint32_t kMoeBlockTileSkewBytes = 32u;

// uint64 because the extents are caller-supplied and §2.7 requires widening
// before the multiply; the caller checks the ceiling, then narrows.
inline constexpr std::uint64_t Dsv4MoeBlockDynUbBytes(
    std::uint32_t hidden, std::uint32_t inter) noexcept {
  return static_cast<std::uint64_t>(hidden) * 4u +
         static_cast<std::uint64_t>(inter) * 4u +
         static_cast<std::uint64_t>(kMoeBlockTileSlots) *
             kMoeBlockRowsPerTile * 4u +
         static_cast<std::uint64_t>(kMoeBlockTileSlots - 1u) *
             kMoeBlockTileSkewBytes;
}

inline constexpr std::uint32_t kMoeBlockDynUbCeilingBytes = 221184u;  // 216 KiB

// ---------------------------------------------------------------------------
// Admissible geometry -- ONE predicate, both execution spaces
// ---------------------------------------------------------------------------
// Macro plus two same-name wrappers, for the reason recorded on
// Dsv4ExpertGeometryAdmissible: BiSheng partitions functions into __host__
// and __aicore__ and will not let one definition serve both, so a single
// plain function cannot be called by the kernel and a single __aicore__
// function cannot be called by the host tiling builder in the same ASC TU.
// The condition lives in the macro so the host gate and the device backstop
// cannot drift.
//
// Shape only. The UB ceiling is deliberately excluded -- the byte-count
// helper above is host-space and a device wrapper cannot link it -- and is
// enforced in the launch wrappers, the only path that allocates.
#if defined(DSV4_TILING_TEST_OVERRIDE) && DSV4_TILING_TEST_OVERRIDE
#define MOE_DSV4_BLOCK_GEOMETRY_ADMISSIBLE(hidden, inter)                    \
  ((hidden) >= ::moe::kernels::arch35::kMoeBlockReductionQuantum &&          \
   (inter) >= ::moe::kernels::arch35::kMoeBlockReductionQuantum &&           \
   (hidden) % ::moe::kernels::arch35::kMoeBlockReductionQuantum == 0u &&     \
   (inter) % ::moe::kernels::arch35::kMoeBlockReductionQuantum == 0u)
#else
#define MOE_DSV4_BLOCK_GEOMETRY_ADMISSIBLE(hidden, inter)                    \
  ((hidden) == 4096u && (inter) == 2048u)
#endif

inline constexpr bool Dsv4MoeBlockGeometryAdmissible(
    std::uint32_t hidden, std::uint32_t inter) noexcept {
  return MOE_DSV4_BLOCK_GEOMETRY_ADMISSIBLE(hidden, inter);
}

#if defined(__CCE__)
__aicore__ inline bool Dsv4MoeBlockGeometryAdmissible(std::uint32_t hidden,
                                                      std::uint32_t inter) {
  return MOE_DSV4_BLOCK_GEOMETRY_ADMISSIBLE(hidden, inter);
}
#endif

// ---------------------------------------------------------------------------
// sizeof == 80
// ---------------------------------------------------------------------------
// EXPERT STRIDES ARE EXPLICIT FIELDS, not derived on device. Expert e's w1
// base is e * w13_packed_expert_stride, and for the production geometry that
// stride is 2048 * 2048 = 4,194,304 B; at 256 experts the largest offset is
// ~1.07 GB, which still fits uint32 but leaves one doubling of headroom. The
// kernel therefore widens to uint64 BEFORE multiplying by the expert id
// (§2.7), and the stride itself is validated host-side rather than
// recomputed from two caller extents in device code.
struct Dsv4MoeBlockTilingData {
  std::uint32_t abi_version;                // 1
  std::uint32_t mode;                       // kMoeBlockMode{Decode,Prefill}
  std::uint32_t block_dim;                  // decode: top_k; prefill: 32
  std::uint32_t hidden;                     // 4096
  std::uint32_t inter;                      // 2048
  std::uint32_t num_experts;                // total experts in the buffers
  std::uint32_t top_k;                      // decode: routed slots; prefill: 1
  std::uint32_t total_tokens;               // decode: 1; prefill: M
  std::uint32_t rows_per_block;             // 32
  std::uint32_t group_size;                 // 32
  std::uint32_t low_nibble_first;           // 1; reject 0
  std::uint32_t w13_packed_row_stride;      // hidden/2 bytes
  std::uint32_t w13_scale_row_stride;       // hidden/32 bytes
  std::uint32_t w2_packed_row_stride;       // inter/2 bytes
  std::uint32_t w2_scale_row_stride;        // inter/32 bytes
  std::uint32_t w13_packed_expert_stride;   // inter * hidden/2 bytes
  std::uint32_t w13_scale_expert_stride;    // inter * hidden/32 bytes
  std::uint32_t w2_packed_expert_stride;    // hidden * inter/2 bytes
  std::uint32_t w2_scale_expert_stride;     // hidden * inter/32 bytes
  float swiglu_limit;                       // 10.0
};

MOE_ARCH35_ASSERT_TILING_POD(Dsv4MoeBlockTilingData, 80);

// Both builders validate the same geometry and differ only in which extent
// carries the work, so they share one implementation behind two names -- the
// names are the contract, and §2.6 requires the two rejections stay distinct:
// invalid_argument for a caller mistake, unsupported_shape for a geometry
// this version does not implement.
Status MakeDsv4MoeDecodeTopkTiling(std::uint32_t hidden, std::uint32_t inter,
                                   std::uint32_t num_experts,
                                   std::uint32_t top_k,
                                   Dsv4MoeBlockTilingData* out) noexcept;

Status MakeDsv4MoePrefillTiling(std::uint32_t hidden, std::uint32_t inter,
                                std::uint32_t num_experts,
                                std::uint32_t total_tokens,
                                Dsv4MoeBlockTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_BLOCK_TILING_H
