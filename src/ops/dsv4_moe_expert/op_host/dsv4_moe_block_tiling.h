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

inline constexpr std::uint32_t kMoeBlockMaxTopK = 8u;

// ---------------------------------------------------------------------------
// GRIDS ARE SKU-DERIVED, NEVER LITERAL
// ---------------------------------------------------------------------------
// Both grids are now a function of the AIV count the runtime reports, because
// there is no literal that is right. Ascend 950PR ships as several parts and
// the toolkit's own platform configs
// (x86_64-linux/data/platform_config/Ascend950PR_*.ini) enumerate them --
// always 1 Cube : 2 Vector:
//
//     950z (dev)                     4 cube /  8 vector
//     9579 / 957b / 957c / 957d     28 cube / 56 vector
//     9589 / 958b                   32 cube / 64 vector
//     9599                          36 cube / 72 vector
//
// The previous constant here was 32, which is the CUBE count of the 9589
// part, not its Vector count: a pure-AIV grid of 32 therefore ran at HALF
// occupancy on a 64-vector part and 44% on a 72-vector one. See
// detail::QueryAivCoreCount() in src/kernels/arch35/launch_common.hpp.
//
// The fallback is used only when the capability query fails. It is the
// smallest production part's Vector count, so it never over-subscribes a real
// device; a grid that is too small costs throughput, while one that is too
// large is rejected at launch.
inline constexpr std::uint32_t kMoeBlockAivFallback = 56u;

// A sane upper bound for a queried core count, so a garbage capability value
// cannot be turned into an absurd grid.
inline constexpr std::uint32_t kMoeBlockAivMax = 256u;

// DECODE GRID: a GANG OF CORES PER ROUTED SLOT.
//
// This REVERSES the earlier `block_dim = top_k` decision, and the reason the
// old note gave for it does not survive contact with the hardware. It said
// the alternative to one-core-per-slot was either (a) recomputing the gate/up
// leg per core, multiplying the weight stream, or (b) "a GM round trip plus a
// barrier, which is the two-dispatch design this replaces". (b) was dismissed
// by conflating two very different costs:
//
//   * the BARRIER is a hardware primitive on this part, not a re-dispatch.
//     `hardware_sync_betweencore=1` in every Ascend950PR ini, and
//     AscendC::SyncAll<true>() has a workspace-free form gated on
//     __NPU_ARCH__ == 3510. Verified: it compiles for dav-3510.
//   * the GM ROUND TRIP is `inter` floats per slot -- 8 KiB at the production
//     geometry -- against a weight stream of ~12 MiB per expert. Three orders
//     of magnitude smaller than the thing it buys occupancy for.
//
// So the trade the old note refused is actually ~0.07% extra traffic to go
// from top_k cores busy to the whole device busy. At top_k = 6 on a 64-vector
// part the old grid left 58 of 64 AIVs idle.
//
// Gang layout: core c owns slot `c / gang_size`, lane `c % gang_size`, and
// lanes grid-stride over their slot's row tiles in BOTH phases. gang_size is
// floor(aiv / top_k), so the grid is gang_size * top_k and never exceeds the
// device; the remainder cores (aiv % top_k) are left unlaunched rather than
// given a partial slot, because an idle core costs nothing while a core that
// must still reach the barrier costs everyone.
//
// NEVER MEASURED on hardware (CLAUDE.md §5.7): this is a traffic and
// occupancy argument, not a benchmark result.
inline constexpr std::uint32_t kMoeBlockMinGangSize = 1u;

// PREFILL GRID: every AIV, grid-stride over tokens.

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
  std::uint32_t block_dim;                  // decode: gang_size*top_k; prefill: aiv
  std::uint32_t hidden;                     // 4096
  std::uint32_t inter;                      // 2048
  std::uint32_t num_experts;                // total experts in the buffers
  std::uint32_t top_k;                      // decode: routed slots; prefill: 1
  std::uint32_t total_tokens;               // decode: 1; prefill: M
  std::uint32_t rows_per_block;             // 32
  std::uint32_t group_size;                 // 32
  std::uint32_t low_nibble_first;           // 1; reject 0
  // Cores per routed slot. Decode: floor(aiv / top_k), >= 1. Prefill: 1,
  // because there a core owns whole tokens and never shares an expert slot.
  // The kernel divides GetBlockIdx() by it, so a zero here would fault --
  // the device backstop rejects that rather than trusting the host.
  std::uint32_t gang_size;
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

// 84, not 80: `gang_size` is new. Growing this struct IS an ABI break and the
// byte count is the wire format (§2.2) -- host and device are rebuilt
// together from this header, so the break is caught here rather than as a
// misparsed field on device.
MOE_ARCH35_ASSERT_TILING_POD(Dsv4MoeBlockTilingData, 84);

// ---------------------------------------------------------------------------
// Decode `activated` workspace -- CALLER-OWNED
// ---------------------------------------------------------------------------
// The gang decode needs `activated` to cross cores, so it needs GM scratch.
// The earlier ABI deliberately had no workspace, on the grounds that
// `activated` never reaches GM; gang-splitting one expert across cores is
// exactly what retires that premise, so the operand comes back -- but as a
// CALLER-OWNED buffer, not a per-launch allocation.
//
// WHY THE CALLER OWNS IT. Decode is the latency path: one token, and the
// legacy single-expert kernel it is measured against ran in ~281 us. An
// aclrtMalloc/aclrtFree pair inside the wrapper would add device-allocator
// latency to every decode step for a buffer whose size never changes between
// steps. The caller allocates once and reuses. This does not weaken §3.4 --
// the wrapper still allocates no OPERAND data; it just no longer pretends
// there is no scratch.
//
// Contents are write-before-read within a single launch, so the caller need
// not initialize it, and nothing in it survives the launch.
//
// THE SIZE HELPER IS PUBLIC, AND DELIBERATELY NOT HERE.
// Dsv4MoeDecodeWorkspaceBytes() lives in include/moe/kernels/arch35/launch.hpp
// because the CALLER allocates this buffer, which makes its size part of the
// public ABI rather than operator-private detail. This header is
// operator-private (§1.2) -- tests/bench spells the block geometry locally
// rather than include it, and op_host/ is not on the npu or bench include
// path -- so a size helper defined here is unreachable by exactly the callers
// that need it. That was a real build break, not a theoretical one.

// Both builders validate the same geometry and differ only in which extent
// carries the work, so they share one implementation behind two names -- the
// names are the contract, and §2.6 requires the two rejections stay distinct:
// invalid_argument for a caller mistake, unsupported_shape for a geometry
// this version does not implement.
// `aiv_count` is the Vector-core count to build the grid from -- what
// detail::QueryAivCoreCount() returned, or 0 to take kMoeBlockAivFallback.
// It is a PARAMETER rather than something these builders query themselves
// because this header is included by device TUs and must stay ACL-free
// (§2.4); the query lives in the launch layer, and the grid derivation stays
// here with the rest of the validation (§2.5).
Status MakeDsv4MoeDecodeTopkTiling(std::uint32_t hidden, std::uint32_t inter,
                                   std::uint32_t num_experts,
                                   std::uint32_t top_k,
                                   std::uint32_t aiv_count,
                                   Dsv4MoeBlockTilingData* out) noexcept;

Status MakeDsv4MoePrefillTiling(std::uint32_t hidden, std::uint32_t inter,
                                std::uint32_t num_experts,
                                std::uint32_t total_tokens,
                                std::uint32_t aiv_count,
                                Dsv4MoeBlockTilingData* out) noexcept;

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_BLOCK_TILING_H
