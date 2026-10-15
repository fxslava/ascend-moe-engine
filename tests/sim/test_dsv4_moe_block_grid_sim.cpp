// Host verification of the REAL dsv4_moe_block tiling builders' grid
// derivation (CLAUDE.md section 4.3: tests/sim verifies host-side operator
// plumbing -- tiling plans -- and links the operator's own host sources).
//
// WHY BOTH THIS AND tests/emu/test_dsv4_moe_block_gang_emu.cpp
// They discharge different obligations and neither covers the other:
//
//   * the emu oracle proves the FORMULA is sound -- that lanes
//     (tile_lo = lane, tile_stride = gang_size) partition a phase's tiles
//     exactly once, for every SKU and gang size. It mirrors the arithmetic,
//     because the emu suite links moe_emu only (section 4.1).
//   * this file proves THE SHIPPING BUILDER COMPUTES THAT FORMULA, by calling
//     MakeDsv4Moe*Tiling itself. A mirrored constant that drifts from the
//     operator is invisible to the emu oracle by construction; it fails here.
//
// It also covers what a formula check cannot: the rejection semantics, which
// section 2.6 requires stay honest -- invalid_argument for a caller mistake,
// unsupported_shape for a geometry this version does not implement.

#include "dsv4_moe_block_tiling.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>

namespace {

// Dsv4MoeDecodeWorkspaceBytes is deliberately NOT exercised here. It is
// public ABI in include/moe/kernels/arch35/launch.hpp, and that header pulls
// in <acl/acl.h> -- which this target must not require, because it is built in
// the host-emu configuration too. The launch-layer suites that do include
// launch.hpp are where the workspace size is used and therefore pinned.
using moe::kernels::arch35::Dsv4MoeBlockTilingData;
using moe::kernels::arch35::MakeDsv4MoeDecodeTopkTiling;
using moe::kernels::arch35::MakeDsv4MoePrefillTiling;
using moe::kernels::arch35::Status;

constexpr std::uint32_t kHidden = 4096u;
constexpr std::uint32_t kInter = 2048u;
constexpr std::uint32_t kExperts = 256u;

int g_failures = 0;

void Fail(const char* what) {
  std::printf("[FAIL] %s\n", what);
  ++g_failures;
}

void ExpectEq(const char* what, std::uint32_t got, std::uint32_t want,
              std::uint32_t aiv, std::uint32_t k) {
  if (got != want) {
    std::printf("[FAIL] %s: got %u want %u (aiv=%u top_k/tokens=%u)\n", what,
                got, want, aiv, k);
    ++g_failures;
  }
}

}  // namespace

int main() {
  // --- decode: the grid is the gang product, for every 950PR Vector count --
  const std::uint32_t aivs[] = {8u, 56u, 64u, 72u, 0u, 1000u};
  for (std::uint32_t aiv : aivs) {
    // 0 means the capability query failed and 1000 is a nonsense reading; the
    // builder must take its documented fallback for both rather than build a
    // grid from the raw value.
    const std::uint32_t resolved = (aiv == 0u || aiv > 256u) ? 56u : aiv;
    for (std::uint32_t top_k : {1u, 2u, 4u, 6u, 8u}) {
      Dsv4MoeBlockTilingData t{};
      if (MakeDsv4MoeDecodeTopkTiling(kHidden, kInter, kExperts, top_k, aiv,
                                      &t) != Status::success) {
        Fail("decode builder rejected the production geometry");
        continue;
      }
      const std::uint32_t want_gang =
          (resolved / top_k == 0u) ? 1u : resolved / top_k;
      ExpectEq("gang_size", t.gang_size, want_gang, aiv, top_k);
      ExpectEq("block_dim", t.block_dim, want_gang * top_k, aiv, top_k);
      ExpectEq("top_k", t.top_k, top_k, aiv, top_k);
      ExpectEq("total_tokens", t.total_tokens, 1u, aiv, top_k);
      if (t.gang_size == 0u) {
        Fail("gang_size is zero -- the kernel divides GetBlockIdx() by it");
      }
      if (aiv != 0u && aiv <= 256u && t.block_dim > aiv) {
        Fail("decode grid oversubscribes the device");
      }
    }
  }

  // --- prefill: every AIV, but never more cores than tokens ---------------
  for (std::uint32_t aiv : {56u, 64u, 72u}) {
    for (std::uint32_t tokens : {1u, 7u, 64u, 256u, 4096u}) {
      Dsv4MoeBlockTilingData t{};
      if (MakeDsv4MoePrefillTiling(kHidden, kInter, kExperts, tokens, aiv,
                                   &t) != Status::success) {
        Fail("prefill builder rejected the production geometry");
        continue;
      }
      ExpectEq("prefill block_dim", t.block_dim,
               tokens < aiv ? tokens : aiv, aiv, tokens);
      // A core owns whole tokens on this path, so there is no slot sharing.
      ExpectEq("prefill gang_size", t.gang_size, 1u, aiv, tokens);
      ExpectEq("prefill top_k", t.top_k, 1u, aiv, tokens);
    }
  }

  // --- the grid must never be zero: it is a launch argument ---------------
  {
    Dsv4MoeBlockTilingData t{};
    if (MakeDsv4MoePrefillTiling(kHidden, kInter, kExperts, 1u, 64u, &t) ==
            Status::success &&
        t.block_dim == 0u) {
      Fail("prefill produced a zero grid");
    }
  }

  // --- rejections stay distinct (section 2.6) -----------------------------
  {
    Dsv4MoeBlockTilingData t{};
    // Caller mistakes.
    if (MakeDsv4MoeDecodeTopkTiling(kHidden, kInter, kExperts, 0u, 64u, &t) !=
        Status::invalid_argument) {
      Fail("top_k == 0 must be invalid_argument");
    }
    if (MakeDsv4MoeDecodeTopkTiling(kHidden, kInter, 4u, 8u, 64u, &t) ==
        Status::success) {
      Fail("top_k > num_experts must be rejected");
    }
    if (MakeDsv4MoeDecodeTopkTiling(kHidden, kInter, kExperts, 6u, 64u,
                                    nullptr) != Status::invalid_argument) {
      Fail("a null out pointer must be invalid_argument");
    }
    if (MakeDsv4MoePrefillTiling(kHidden, kInter, kExperts, 0u, 64u, &t) !=
        Status::invalid_argument) {
      Fail("total_tokens == 0 must be invalid_argument");
    }
    // A geometry this version does not implement -- not a caller mistake.
    if (MakeDsv4MoeDecodeTopkTiling(512u, 512u, kExperts, 6u, 64u, &t) !=
        Status::unsupported_shape) {
      Fail("a scaled geometry must be unsupported_shape, not invalid_argument");
    }
    // top_k above the routed-slot ceiling is a shape, not a caller mistake:
    // the workspace and the slot map are sized by it.
    if (MakeDsv4MoeDecodeTopkTiling(kHidden, kInter, kExperts, 9u, 64u, &t) !=
        Status::unsupported_shape) {
      Fail("top_k past the ceiling must be unsupported_shape");
    }
  }

  if (g_failures != 0) {
    std::printf("\n%d failure(s)\n", g_failures);
    return 1;
  }
  std::puts("[PASS] the shipping tiling builders derive the SKU-aware grids "
            "and keep their rejections distinct");
  return 0;
}
