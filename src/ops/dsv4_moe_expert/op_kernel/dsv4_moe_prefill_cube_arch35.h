// dsv4_moe_prefill_cube_arch35.h -- UB budget and event plan for the
// Cube-accelerated batched prefill entry point (Ascend 950PR, arch35 /
// dav-3510, MIX AIC+AIV).
//
// op_kernel/ is PRIVATE (CLAUDE.md §1.2): nothing outside this operator may
// depend on these numbers. The tile extents the HOST must also know -- because
// it declares dyn_ub_size at the chevron and validates the geometry -- live in
// op_host/dsv4_moe_prefill_cube_tiling.h instead, and the kernel TU
// static_asserts the two copies against each other so a drift is a compile
// error rather than an under-declared dynamic region.
//
// PER-CORE CAPACITIES, from the toolkit's platform configs rather than the
// whitepaper's per-subsystem figures:
//
//     L1          524,288 B    shared with the paired Cube
//     L0A / L0B    65,536 B    Cube operand buffers, Matmul-managed
//     L0C         262,144 B    Cube accumulator, Matmul-managed
//     UB          253,952 B    PER VECTOR CORE (the paper's 512 KiB is per
//                              subsystem, and a subsystem holds two AIVs)
//
// L0A/L0B/L0C are not budgeted here: the Matmul template owns them, and its
// compile-time tiling (GetMatmulApiTiling) is what sizes them from the
// declared operand types. What this file budgets is the UB the VECTOR side
// holds while the Cube runs.

#ifndef MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
#define MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP

#include <cstdint>

#include "moe/kernels/arch35/arch35_common.hpp"

namespace moe::kernels::arch35::dsv4_moe_prefill_cube {

inline constexpr std::uint32_t kCubeMTile = 16;       // tokens per Cube pass
inline constexpr std::uint32_t kCubeInterTile = 512;  // gate/up N chunk
inline constexpr std::uint32_t kCubeHiddenTile = 512; // down N chunk

inline constexpr std::uint32_t kCubeHiddenMax = 4096;
inline constexpr std::uint32_t kCubeInterMax = 2048;

// --- the UB budget, and why M_TILE is 16 -----------------------------------
//
//     gate chunk   FP32  [16, 512]   32,768 B   Cube output, at VECIN
//     up   chunk   FP32  [16, 512]   32,768 B   Cube output, at VECIN
//     activated    BF16  [16, 2048]  65,536 B   resident across the boundary
//                                    ---------
//                                    131,072 B  of 221,184 B usable
//
// At M_TILE = 32 the same table is 262,144 B and does not fit, which is the
// whole derivation: `activated` must survive from the gate/up GEMMs to the
// down GEMM without a GM round trip, and on this part that residency is what
// the token tile is sized around.
//
// X is deliberately NOT in this budget. It is cast to BF16 into a workspace
// plane and read by the Cube from GM, because holding [16, 4096] BF16 in UB
// would cost another 131,072 B and leave no room for `activated`.
inline constexpr std::uint32_t kCubeAccBytes = kCubeMTile * kCubeInterTile * 4u;
inline constexpr std::uint32_t kCubeActBytes = kCubeMTile * kCubeInterMax * 2u;

static_assert(kCubeAccBytes == 32768, "gate/up chunk must be 32,768 B");
static_assert(kCubeActBytes == 65536, "activated must be 65,536 B");

// Two accumulator slots (gate and up) plus the activated plane. The dequant
// scratch below reuses these same two queues rather than adding segments, so
// the peak is the larger of the two phases, not their sum.
inline constexpr std::uint32_t kCubeUbBytes =
    2u * kCubeAccBytes + kCubeActBytes;

static_assert(kCubeUbBytes == 131072,
              "the Cube prefill UB budget must sum to 131,072 B");
static_assert(kCubeUbBytes <= kUsableUbBytes,
              "the Cube prefill UB budget exceeds the arch35 usable ceiling");

// The dequantization scratch must also fit the SAME two queues, since it
// reuses them: one row of nibbles expanded to BF16 is the widest single
// buffer, at hidden * 2 B.
static_assert(kCubeHiddenMax * 2u <= kCubeActBytes,
              "a dequantized weight row must fit the activated-sized buffer "
              "the dequant phase borrows");
static_assert(kCubeHiddenMax <= kCubeAccBytes,
              "a packed nibble row must fit the accumulator-sized buffer the "
              "dequant phase borrows");

// --- KFC queue reserve ----------------------------------------------------
// REGIST_MATMUL_OBJ stands up a KfcServer whose message queue lives at the
// head of the workspace, so the operator's own data planes must start past it.
// 1 MiB is the toolkit's own conservative reserve for the mix message queue
// plus its alignment; the planes that follow are tens of MiB, so rounding up
// here costs nothing and a too-small reserve would corrupt the queue.
inline constexpr std::uint64_t kCubeKfcReserveBytes = 1u << 20;

// Event ids. EVENT_ID0..2 only, balanced 1:1 at every use so no token
// outlives a loop iteration.
inline constexpr std::int32_t kEvCubeMte2ToV = 0;
inline constexpr std::int32_t kEvCubeVToMte3 = 1;
inline constexpr std::int32_t kEvCubeMte3ToV = 2;

}  // namespace moe::kernels::arch35::dsv4_moe_prefill_cube

#endif  // MOE_KERNELS_ARCH35_DSV4_MOE_PREFILL_CUBE_UB_LAYOUT_HPP
