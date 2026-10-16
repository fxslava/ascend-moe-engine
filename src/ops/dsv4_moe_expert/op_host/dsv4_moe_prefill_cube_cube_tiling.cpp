// Host-side TCubeTiling construction for dsv4_moe_prefill_cube's two
// sub-GEMMs. See dsv4_moe_prefill_cube_cube_tiling.h for why this exists and
// why it is host-compiled.

#include "dsv4_moe_prefill_cube_cube_tiling.h"

#include "dsv4_moe_prefill_cube_tiling.h"

#include "matmul_tiling.h"

#include <cstring>

namespace moe::kernels::arch35 {
namespace {

// The wire format this operator stages. Asserted against the toolkit's own
// struct rather than assumed: if a CANN update changes TCubeTiling, this is
// the build break that catches it, instead of the kernel reading 200 bytes of
// a differently-shaped struct out of GM.
static_assert(sizeof(AscendC::tiling::TCubeTiling) == kPrefillCubeTCubeTilingBytes,
              "the toolkit's TCubeTiling is no longer "
              "kPrefillCubeTCubeTilingBytes; the staging layout in "
              "dsv4_moe_prefill_cube_cube_tiling.h is the wire format and must "
              "be updated together with the kernel's static_assert");

// DaVinci v3 per-core capacities, from the toolkit's platform configs
// (x86_64-linux/data/platform_config/Ascend950PR_*.ini) and matching the table
// in dsv4_moe_prefill_cube_tiling.h. Supplied explicitly through PlatformInfo
// rather than queried through platform_ascendc::PlatformAscendC, because that
// class is constructed from the op framework's TilingContext, which a
// standalone launch wrapper does not have.
//
// UB is the 248 KiB PER VECTOR CORE figure, not the whitepaper's 512 KB per AI
// subsystem (a subsystem holds two vector cores). It is passed only so the
// tiling API can bound its own staging; this operator's C lands at VECIN from
// buffers the kernel allocates itself.
constexpr std::uint64_t kL1Bytes  = 524288;
constexpr std::uint64_t kL0CBytes = 262144;
constexpr std::uint64_t kUbBytes  = 253952;
constexpr std::uint64_t kL0ABytes = 65536;
constexpr std::uint64_t kL0BBytes = 65536;

}  // namespace

Status MakeDsv4MoePrefillCubeGemmTiling(PrefillCubeGemm which,
                                       std::uint32_t rows,
                                       void* out) noexcept {
  if (out == nullptr || rows == 0u || rows > kPrefillCubeMTile) {
    return Status::invalid_argument;
  }

  // n contracts nothing; k is the axis reduced inside ONE sub-GEMM. Phase 1
  // produces R_SUB columns of `act` while contracting K_SUB of hidden; phase 2
  // produces H_SUB columns of Y while contracting R_SUB2 of inter. These are
  // the SUB-TILE extents, not the operator's axes: each (r0, k0) step is its
  // own single-core GEMM, which is the whole reason a TPosition::TSCM B
  // operand is affordable here.
  const std::int32_t n = (which == PrefillCubeGemm::gate_up)
                             ? static_cast<std::int32_t>(kPrefillCubeRSub)
                             : static_cast<std::int32_t>(kPrefillCubeHSub);
  const std::int32_t k = (which == PrefillCubeGemm::gate_up)
                             ? static_cast<std::int32_t>(kPrefillCubeKSub)
                             : static_cast<std::int32_t>(kPrefillCubeRSub2);
  const std::int32_t m = static_cast<std::int32_t>(rows);

  matmul_tiling::PlatformInfo platform;
  platform.socVersion = platform_ascendc::SocVersion::ASCEND950;
  platform.l1Size  = kL1Bytes;
  platform.l0CSize = kL0CBytes;
  platform.ubSize  = kUbBytes;
  platform.l0ASize = kL0ABytes;
  platform.l0BSize = kL0BBytes;

  matmul_tiling::MatmulApiTiling mm(platform);

  // These four MUST mirror the kernel's MatmulType declarations exactly. If
  // they drift, the Cube is initialized for a different operand layout than
  // the one the kernel then feeds it, which is a wrong answer and not a fault.
  //   A: TPosition::VECOUT, ND, bfloat16_t          (the cast token sub-tile)
  //   B: TPosition::TSCM,   ND, bfloat16_t, ibShare (dequantized into L1)
  //   C: TPosition::VECIN,  ND, float               (FP32 sub-GEMM subtotal)
  //   Bias: unused
  int32_t rc = mm.SetAType(matmul_tiling::TPosition::VECOUT,
                           matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_BF16, false);
  // isTrans = true, matching the kernel's SetTensorB(staged, true): the
  // weights are stored [n, k] and contracted along k.
  rc |= mm.SetBType(matmul_tiling::TPosition::TSCM,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, true);
  rc |= mm.SetCType(matmul_tiling::TPosition::VECIN,
                    matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
  rc |= mm.SetBiasType(matmul_tiling::TPosition::GM,
                       matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT);
  rc |= mm.SetBias(false);
  rc |= mm.SetShape(m, n, k);
  // Org == single, because the B operand of one sub-GEMM IS the whole
  // single-core B matrix. The kernel re-points A and B and calls SetOrgShape /
  // SetTail per trip; what Init needs is a shape consistent with those.
  rc |= mm.SetOrgShape(m, n, k);
  rc |= mm.SetBufferSpace(static_cast<int32_t>(kL1Bytes),
                          static_cast<int32_t>(kL0CBytes),
                          static_cast<int32_t>(kUbBytes));
  if (rc != 0) {
    return Status::unsupported_shape;
  }

  AscendC::tiling::TCubeTiling tiling;
  if (mm.GetTiling(tiling) != 0) {
    // The tiling API could not fit this sub-GEMM in the declared L1/L0C/UB.
    // unsupported_shape and not invalid_argument: the caller passed nothing
    // illegal, this geometry is simply not one the Cube can be tiled for
    // (CLAUDE.md section 2.6).
    return Status::unsupported_shape;
  }

  // A tiling that came back "successful" with a zero base tile would init the
  // Cube to do nothing, which on device is the silent zeroed output section
  // 2.5 exists to prevent. Checked here, host-side, where it is observable.
  if (tiling.baseM <= 0 || tiling.baseN <= 0 || tiling.baseK <= 0 ||
      tiling.M != m || tiling.N != n) {
    return Status::unsupported_shape;
  }

  std::memcpy(out, &tiling, sizeof(tiling));
  return Status::success;
}

}  // namespace moe::kernels::arch35
