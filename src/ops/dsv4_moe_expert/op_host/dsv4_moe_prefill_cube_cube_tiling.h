#ifndef DSV4_MOE_PREFILL_CUBE_CUBE_TILING_H
#define DSV4_MOE_PREFILL_CUBE_CUBE_TILING_H

// The Matmul framework's own TCubeTiling for this operator's two sub-GEMMs,
// and the GM staging layout that carries them to the kernel.
//
// ---------------------------------------------------------------------------
// WHY THIS EXISTS: THE DEFECT IT FIXES
// ---------------------------------------------------------------------------
// The kernel used to call
//
//     REGIST_MATMUL_OBJ(&pipe, shared_workspace, matmul_gate_up, matmul_down);
//
// with the two Matmul OBJECTS and no tiling operands. That is not the macro's
// contract: it takes INTERLEAVED (object, tiling) pairs. With no tiling after
// an object, KfcServer::InitObjAux (CANN 9.2.0-beta.2,
// asc/impl/adv_api/detail/kfc/kernel_kfc.h:366) takes a branch that calls
//
//     InitKfc(tpipe, (void*)nullptr, ...)
//
// for BOTH objects -- the `sizeof...(b) == 0` branch for the last one and the
// trailing `else` for the first. It compiles silently, because the variadic
// template simply selects the no-tiling overload; `isTiling<Args...>()` is
// `sizeof(T) == sizeof(void*)`, and a Matmul object is not pointer-sized.
//
// The static tiling could not cover for it either. `GetNormalConfig()`
// hard-codes basicM/N/K and singleCoreM/N/K to 0 (asc/include/adv_api/matmul/
// tiling.h:36), so `GetMatmulApiTiling` takes its early return
// (constant_tiling.h:41) and every field of the resulting
// MatmulApiStaticTiling keeps its -1 default -- and -1 means "read this field
// from the runtime TCubeTiling". So every field access went through a null
// pointer.
//
// MEASURED on the dav-3510 CAModel, 2026-10-16, at the minimum geometry (16
// tokens, one tile, one AIC cluster): 66 x `pem_lsu.cc:484
// mem_map_calc_49bit_op0 invalid ldst addr`, at ldst_addr
//
//     4 8 12 16 20 24 28 32 40 52 60 72 76 120 124
//
// and the run never terminated. Every one of those 15 offsets is a field of
// TCubeTiling -- M, N, Ka, Kb, singleCoreM/N/K, baseM, baseK, stepM, isBias,
// shareMode, shareL1Size, dbL0A, dbL0B -- i.e. a null-pointer struct read, not
// an unaligned UB access, not a bad burst stride and not a null GM operand
// base. Recorded here because the offset list is the evidence, and a future
// reader seeing `pem_lsu` faults should check this first.
//
// ---------------------------------------------------------------------------
// WHY THE TILING IS BUILT ON THE HOST AND STAGED THROUGH GM
// ---------------------------------------------------------------------------
// InitKfc has a dedicated branch for it (asc/impl/adv_api/detail/matmul/kfc/
// matmul_server.h:75): when the tiling argument is typed
// `const __gm__ TCubeTiling*` it COPIES the struct out of GM into a core-local
// one before Init. Any other pointer type is stored as-is, which is how a null
// got through. So GM is the supported path, and the type at the call site is
// load-bearing -- it selects the copy.
//
// Building it on the host also keeps the rule in CLAUDE.md section 2.5: the
// geometry decisions are made before the enqueue, where a rejection is
// observable, rather than on a device whose early return is a silent zeroed
// output.
//
// ---------------------------------------------------------------------------
// WHY THE 72-BYTE OPERATOR ABI IS NOT TOUCHED
// ---------------------------------------------------------------------------
// Dsv4MoePrefillCubeTilingData stays exactly 72 bytes and keeps its
// MOE_ARCH35_ASSERT_TILING_POD. What grows is the launch wrapper's own STAGING
// BUFFER, which is wrapper-private and not a published contract: the POD sits
// at offset 0 and the two TCubeTilings follow it. The sim launch wrapper
// already appends a uint32 to that same buffer for its sample cap, so this
// follows an established precedent in this tree rather than inventing one.
//
// The two tilings are held as OPAQUE BYTE BLOCKS on purpose. This header is
// included by both the host builder (plain g++) and the device kernel
// (BiSheng), and `TCubeTiling` is declared in two different toolkit trees
// (`AscendC::tiling::` under asc/include, `optiling::` under include/register).
// Naming the type here would drag one of those headers into both sides; naming
// a size does not. Each side casts to its own TCubeTiling and static_asserts
// that it is exactly kPrefillCubeTCubeTilingBytes, so a toolkit that changed
// the struct breaks the build instead of silently misreading the wire format.

#include "moe/kernels/arch35/status.hpp"

#include <cstddef>
#include <cstdint>

namespace moe::kernels::arch35 {

// sizeof(AscendC::tiling::TCubeTiling) on CANN 9.2.0-beta.2, VERIFIED by
// compiling and printing it rather than read off the struct definition. It is
// a flat run of int32 fields, so 4-byte alignment is all it needs; the blocks
// below are placed at 8-byte multiples anyway.
inline constexpr std::size_t kPrefillCubeTCubeTilingBytes = 200;

// Which of the two sub-GEMMs a staged block belongs to.
//
//   gate_up  phase 1: [M_TILE, K_SUB] x [R_SUB, K_SUB]^T -> [M_TILE, R_SUB]
//   down     phase 2: [M_TILE, R_SUB2] x [H_SUB, R_SUB2]^T -> [M_TILE, H_SUB]
//
// Two tilings and not one because the shapes genuinely differ (128x128
// contracting hidden, against 64x256 contracting inter), which is also why the
// kernel carries two Matmul types.
enum class PrefillCubeGemm : std::uint32_t { gate_up = 0, down = 1 };

// Builds the TCubeTiling for one sub-GEMM into `out`, which must point at
// kPrefillCubeTCubeTilingBytes writable bytes.
//
// HOST ONLY. It links the toolkit's matmul tiling API, so it is compiled by
// the host compiler and not by BiSheng: it contains no device code, and
// CLAUDE.md section 3.3 requires BiSheng only for the TU holding the chevron
// launch.
//
// `rows` is the M the Cube is told about at Init time. The kernel still calls
// SetTail per trip for a ragged tile, so this is the TILE's M (M_TILE), not a
// particular trip's valid row count.
Status MakeDsv4MoePrefillCubeGemmTiling(PrefillCubeGemm which,
                                        std::uint32_t rows, void* out) noexcept;

// ---------------------------------------------------------------------------
// The staging buffer layout
// ---------------------------------------------------------------------------
// Offsets rather than a struct, because the production POD's type is not
// visible here (this header is included by the kernel, which gets the POD from
// dsv4_moe_prefill_cube_tiling.h) and because the kernel needs the offsets as
// byte arithmetic on a GM_ADDR anyway.
inline constexpr std::size_t kPrefillCubeTilingPodBytes = 72;
inline constexpr std::size_t kPrefillCubeGateTilingOffset = kPrefillCubeTilingPodBytes;
inline constexpr std::size_t kPrefillCubeDownTilingOffset =
    kPrefillCubeGateTilingOffset + kPrefillCubeTCubeTilingBytes;
inline constexpr std::size_t kPrefillCubeStagingBytes =
    kPrefillCubeDownTilingOffset + kPrefillCubeTCubeTilingBytes;

static_assert(kPrefillCubeGateTilingOffset % 8u == 0u &&
                  kPrefillCubeDownTilingOffset % 8u == 0u,
              "each staged TCubeTiling must start 8-byte aligned, or the "
              "device-side copy out of GM reads it misaligned");

}  // namespace moe::kernels::arch35

#endif  // DSV4_MOE_PREFILL_CUBE_CUBE_TILING_H
