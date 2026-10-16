/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

// =================================================================================================
// dsv4_prefill_cube_dequant_golden.hpp -- audited CPU oracle for DequantSubTileVector
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2)
// =================================================================================================
//
// 1. WHY A SECOND ORACLE EXISTS AT ALL
// -------------------------------------------------------------------------------------------------
// `moe::emu::mxfp4_dequant_element_bf16_cpu` already decodes one MXFP4 element, but it reads an
// `Mxfp4Matrix` -- a WHOLE weight matrix whose E8M0 plane has a COMPACT row stride of `cols / 32`.
// That is not the layout `DequantSubTileVector` sees. The device stages one sub-tile into UB and
// the scale burst lands ONE 32-BYTE BLOCK PER ROW, because a sub-tile row contributes only
// `cols / 32` scale bytes while a UB burst block is 32 B. The row stride in UB is therefore 32
// ENTRIES, not `cols / 32`, and the remaining bytes of each block are padding the kernel must
// never read.
//
// An oracle that cannot express that padded stride cannot verify the indexing most likely to be
// wrong. So `CpuDequantSubTile` below models the UB layout directly, and is written against raw
// bit arithmetic rather than by calling the emu helper, so that the two remain INDEPENDENT
// transcriptions of the nibble order and the group-scale indexing.
//
// 2. HOW THIS ORACLE IS ITSELF AUDITED (two-phase oracle, phase 1)
// -------------------------------------------------------------------------------------------------
// Never diff a kernel against an unaudited CPU model. `AuditDequantSubTileGolden` discharges four
// obligations on the host, before any device work:
//
//     a. EQUIVALENCE    -- over a compact-stride `Mxfp4Matrix` view of the same bytes, every
//                          element agrees bit-for-bit with `mxfp4_dequant_element_bf16_cpu`. Two
//                          independent transcriptions agreeing is what makes either trustworthy.
//     b. PAD BLINDNESS  -- re-running with every padding byte of every 32-byte scale block set to
//                          a different poison must not move one output bit. That is the actual
//                          claim about the padded stride, asserted rather than assumed.
//     c. NIBBLE ORDER   -- a hand-written single-pair probe pins low-nibble-first, which a fixture
//                          of random magnitudes cannot distinguish.
//     d. NON-DEGENERACY -- the golden must carry a nonzero norm and real variance. Comparing a
//                          value with itself is not a degeneracy check (CLAUDE.md section 4.8).
//
// 3. WHY THE DEVICE COMPARISON IS BIT-EXACT, FROM THE FORMAT
// -------------------------------------------------------------------------------------------------
// The device path is `Cast(fp4 -> bf16)`, then `bf16 = Bits(code << 7)` for the scale, then `Mul`.
// The host path is `DecodeFp4E2M1(nibble) * DecodeFp8E8M0(code)` rounded to BF16. These agree
// EXACTLY, and the reason is arithmetic rather than empirical:
//
//     * E2M1 decodes to {0, 0.5, 1, 1.5, 2, 3, 4, 6}, every one of which needs at most two
//       mantissa bits -- BF16 keeps seven, so each is exact in BF16.
//     * E8M0 code c denotes 2^(c-127), i.e. the value whose exponent FIELD is c and whose mantissa
//       is zero. The device builds it as BF16 bits `c << 7`; `DecodeFp8E8M0` builds it as FP32 bits
//       `c << 23`. Both are that same value, for all 256 codes, so the bitshift is not an
//       approximation of the table -- it is the table.
//     * A product of a <=2-bit significand with a power of two shifts the exponent and leaves the
//       significand alone, so no rounding occurs on EITHER side and the two paths cannot differ.
//
// That argument fails only where the product leaves BF16's normal range, or where the scale code is
// the 0 / 0xFF degenerate pair. `moe::emu::Mxfp4DequantIsExactInBf16` is the repository's own
// predicate for exactly that domain, and the fixture builder below REJECTS every pair it excludes
// -- so the bit-exact bound is enforced by construction at the fixture rather than fitted to an
// observed error.

#ifndef TESTS_SIM_DSV4_PREFILL_CUBE_DEQUANT_GOLDEN_HPP
#define TESTS_SIM_DSV4_PREFILL_CUBE_DEQUANT_GOLDEN_HPP

#include "moe/emu/cpu_kernels.h"
#include "moe/emu/dsv4_layers.h"
#include "moe/kernels/arch35/status.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe::sim::prefill_cube {

using moe::kernels::arch35::E8M0;
using moe::kernels::arch35::Mxfp4Pair;

// ---------------------------------------------------------------------------------------------
// Layout constants, restated rather than included from the operator's PRIVATE op_kernel header
// ---------------------------------------------------------------------------------------------
// `dsv4_moe_prefill_cube_arch35.h` is op-private (CLAUDE.md section 1.2) and nothing outside the
// operator may depend on it. A host test is outside the operator. So these are an independent
// statement of the same numbers, and the device micro-kernel -- which is inside the operator's
// include path and may read the private header -- static_asserts the two copies against each
// other. A drift is then a compile error in the micro-kernel rather than a silently passing test.
inline constexpr std::uint32_t kGoldenGroupSize      = 32;
inline constexpr std::uint32_t kGoldenScaleRowStride = 32;

// The byte written into every padding lane of a 32-byte scale block. 0xFF and not zero: zero is a
// valid E8M0 code denoting 2^-127, so a kernel that wrongly broadcast a padding lane would merely
// scale by something small and could still pass a loose compare. 0xFF denotes an infinity, so the
// same mistake produces an infinity or a NaN and cannot be mistaken for a rounding difference.
inline constexpr std::uint8_t kScalePadPoison    = 0xFFU;
inline constexpr std::uint8_t kScalePadPoisonAlt = 0xA5U;

// The eight E2M1 magnitudes by code. Stated here rather than taken from `DecodeFp4E2M1` so that the
// equivalence check in the audit compares two transcriptions and not one against itself.
inline constexpr float kE2m1Magnitudes[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};

[[noreturn]] inline void GoldenFail(const std::string& message) {
    throw std::runtime_error("dequant golden: " + message);
}

inline void GoldenRequire(bool condition, const std::string& message) {
    if (!condition) GoldenFail(message);
}

// One sub-tile as the device stages it in UB.
//
//   packed         [rows, cols / 2]                   contiguous, low nibble = even column
//   padded_scales  [rows, kGoldenScaleRowStride]      cols / 32 valid bytes, then padding
//   compact_scales [rows, cols / 32]                  the same codes at the Mxfp4Matrix stride, so
//                                                     the emu helper can view the same packed bytes
struct DequantSubTile {
    std::uint32_t          rows = 0;
    std::uint32_t          cols = 0;
    std::vector<Mxfp4Pair> packed;
    std::vector<E8M0>      padded_scales;
    std::vector<E8M0>      compact_scales;

    std::uint32_t GroupsPerRow() const noexcept { return cols / kGoldenGroupSize; }
    std::size_t   Elements()     const noexcept { return static_cast<std::size_t>(rows) * cols; }

    moe::emu::Mxfp4Matrix CompactView() const noexcept {
        return {packed.data(), compact_scales.data(), static_cast<int>(rows),
                static_cast<int>(cols)};
    }
};

// The oracle. Returns BF16 BIT PATTERNS, not floats: the device hands back the raw
// `weight_expanded` store and the comparison is bit-exact, so converting to float on either side
// would only add a place for a mismatch to hide.
inline std::vector<std::uint16_t> CpuDequantSubTile(
    const Mxfp4Pair* packed, const E8M0* padded_scales, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t scale_row_stride) {
    GoldenRequire(packed != nullptr && padded_scales != nullptr, "null sub-tile operand");
    GoldenRequire(rows > 0 && cols > 0, "empty sub-tile");
    GoldenRequire(cols % kGoldenGroupSize == 0, "sub-tile width must be group aligned");
    GoldenRequire(cols / kGoldenGroupSize <= scale_row_stride,
                  "a row of E8M0 scales must fit one scale burst block");

    const std::size_t          byte_stride = cols / 2U;
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * cols, 0U);

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
            const std::uint8_t byte       = packed[row * byte_stride + col / 2U];
            const std::uint8_t nibble     = (col % 2U == 0U) ? static_cast<std::uint8_t>(byte & 0x0FU)
                                                             : static_cast<std::uint8_t>(byte >> 4U);
            const float        magnitude  = kE2m1Magnitudes[nibble & 0x07U];
            const float        decoded    = (nibble & 0x08U) != 0U ? -magnitude : magnitude;
            const std::uint8_t scale_code = padded_scales[row * scale_row_stride +
                                                          col / kGoldenGroupSize];
            const float        scale      = moe::emu::Bf16ToFloat(
                static_cast<std::uint16_t>(static_cast<std::uint16_t>(scale_code) << 7U));
            out[static_cast<std::size_t>(row) * cols + col] =
                moe::emu::FloatToBf16(decoded * scale);
        }
    }
    return out;
}

inline std::vector<std::uint16_t> CpuDequantSubTile(const DequantSubTile& tile) {
    return CpuDequantSubTile(tile.packed.data(), tile.padded_scales.data(), tile.rows, tile.cols,
                             kGoldenScaleRowStride);
}

// References are generated in memory, per run, from a printed seed (CLAUDE.md section 4.5). No disk
// fixture can go stale and let a broken kernel pass.
//
// The scale codes span [96, 158] rather than the 126..128 an earlier fixture used: a three-code
// window leaves the exponent arithmetic of the `code << 7` path almost untested, since every scale
// is then within one power of two of unity. Every (nibble, code) pair that reaches the fixture is
// screened through `Mxfp4DequantIsExactInBf16`, which is what licenses the bit-exact bound.
inline DequantSubTile BuildDequantSubTile(std::uint32_t rows, std::uint32_t cols, std::uint64_t seed,
                                          std::uint8_t pad_poison = kScalePadPoison) {
    GoldenRequire(cols % kGoldenGroupSize == 0, "sub-tile width must be group aligned");
    GoldenRequire(cols / kGoldenGroupSize <= kGoldenScaleRowStride,
                  "a row of E8M0 scales must fit one scale burst block");

    DequantSubTile tile;
    tile.rows = rows;
    tile.cols = cols;
    tile.packed.assign(static_cast<std::size_t>(rows) * (cols / 2U), 0U);
    tile.padded_scales.assign(static_cast<std::size_t>(rows) * kGoldenScaleRowStride, pad_poison);
    tile.compact_scales.assign(static_cast<std::size_t>(rows) * (cols / kGoldenGroupSize), 0U);

    std::mt19937_64                         rng(seed);
    std::uniform_int_distribution<unsigned> nibble_source(0U, 15U);
    std::uniform_int_distribution<unsigned> scale_source(96U, 158U);
    const std::uint32_t                     groups_per_row = cols / kGoldenGroupSize;

    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t group = 0; group < groups_per_row; ++group) {
            const std::uint8_t scale_code = static_cast<std::uint8_t>(scale_source(rng));
            tile.padded_scales[static_cast<std::size_t>(row) * kGoldenScaleRowStride + group] =
                scale_code;
            tile.compact_scales[static_cast<std::size_t>(row) * groups_per_row + group] = scale_code;

            for (std::uint32_t col = group * kGoldenGroupSize; col < (group + 1U) * kGoldenGroupSize;
                 col += 2U) {
                std::uint8_t low  = static_cast<std::uint8_t>(nibble_source(rng));
                std::uint8_t high = static_cast<std::uint8_t>(nibble_source(rng));
                while (!moe::emu::Mxfp4DequantIsExactInBf16(low, scale_code)) {
                    low = static_cast<std::uint8_t>(nibble_source(rng));
                }
                while (!moe::emu::Mxfp4DequantIsExactInBf16(high, scale_code)) {
                    high = static_cast<std::uint8_t>(nibble_source(rng));
                }
                tile.packed[static_cast<std::size_t>(row) * (cols / 2U) + col / 2U] =
                    static_cast<Mxfp4Pair>(low | (high << 4U));
            }
        }
    }
    return tile;
}

// ---------------------------------------------------------------------------------------------
// Phase 1 of the two-phase oracle: audit the golden before trusting it
// ---------------------------------------------------------------------------------------------
inline void AuditDequantSubTileGolden(std::uint32_t rows, std::uint32_t cols, std::uint64_t seed) {
    const DequantSubTile             tile   = BuildDequantSubTile(rows, cols, seed);
    const std::vector<std::uint16_t> golden = CpuDequantSubTile(tile);

    // (a) EQUIVALENCE against the repository's independent element decoder.
    const moe::emu::Mxfp4Matrix view = tile.CompactView();
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
            const std::uint16_t want = moe::emu::FloatToBf16(moe::emu::mxfp4_dequant_element_bf16_cpu(
                view, static_cast<int>(row), static_cast<int>(col)));
            const std::uint16_t got = golden[static_cast<std::size_t>(row) * cols + col];
            if (got != want) {
                GoldenFail("disagrees with mxfp4_dequant_element_bf16_cpu at (" +
                           std::to_string(row) + ", " + std::to_string(col) + "): got " +
                           std::to_string(got) + ", want " + std::to_string(want));
            }
        }
    }

    // (b) PAD BLINDNESS. Same codes, a different poison in every padding lane, identical output.
    DequantSubTile      repoisoned     = tile;
    const std::uint32_t groups_per_row = tile.GroupsPerRow();
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t lane = groups_per_row; lane < kGoldenScaleRowStride; ++lane) {
            repoisoned.padded_scales[static_cast<std::size_t>(row) * kGoldenScaleRowStride + lane] =
                kScalePadPoisonAlt;
        }
    }
    const std::vector<std::uint16_t> repoisoned_golden = CpuDequantSubTile(repoisoned);
    for (std::size_t i = 0; i < golden.size(); ++i) {
        if (golden[i] != repoisoned_golden[i]) {
            GoldenFail("reads a scale-block padding lane: output moved at flat index " +
                       std::to_string(i) + " when only padding changed");
        }
    }

    // (c) NIBBLE ORDER. One byte, one scale, two distinct halves -- the only probe that can tell
    // low-nibble-first from high-nibble-first.
    {
        const std::uint8_t     probe_byte  = static_cast<std::uint8_t>(0x02U | (0x06U << 4U));
        const std::uint8_t     probe_scale = 127U;
        std::vector<E8M0>      probe_scales(kGoldenScaleRowStride, kScalePadPoison);
        std::vector<Mxfp4Pair> probe_packed(kGoldenGroupSize / 2U, probe_byte);
        probe_scales[0] = probe_scale;
        const std::vector<std::uint16_t> probe = CpuDequantSubTile(
            probe_packed.data(), probe_scales.data(), 1U, kGoldenGroupSize, kGoldenScaleRowStride);
        GoldenRequire(moe::emu::Bf16ToFloat(probe[0]) == 1.0F,
                      "an even column must decode the LOW nibble (got " +
                          std::to_string(moe::emu::Bf16ToFloat(probe[0])) + ", want 1.0)");
        GoldenRequire(moe::emu::Bf16ToFloat(probe[1]) == 4.0F,
                      "an odd column must decode the HIGH nibble (got " +
                          std::to_string(moe::emu::Bf16ToFloat(probe[1])) + ", want 4.0)");
    }

    // (d) NON-DEGENERACY of the golden itself: a real norm and real variance, and no element may
    // collide with the guard-band canary word -- otherwise a never-written output would be
    // indistinguishable from a correct one.
    double sum    = 0.0;
    double sum_sq = 0.0;
    for (const std::uint16_t bits : golden) {
        GoldenRequire(bits != 0x5A5AU, "a golden element equals the guard-band canary word 0x5A5A");
        const float value = moe::emu::Bf16ToFloat(bits);
        GoldenRequire(std::isfinite(value), "a golden element is not finite");
        sum    += value;
        sum_sq += static_cast<double>(value) * value;
    }
    const double count    = static_cast<double>(golden.size());
    const double mean     = sum / count;
    const double mean_sq  = sum_sq / count;
    const double variance = std::max(0.0, mean_sq - mean * mean);
    GoldenRequire(std::sqrt(mean_sq) > 1.0e-12, "the golden has no magnitude");
    GoldenRequire(std::sqrt(variance) > 1.0e-12, "the golden has no variance");

    std::printf("[PASS] golden self-audit %ux%u (seed 0x%016llX): equivalence, pad blindness, "
                "nibble order, non-degeneracy\n",
                rows, cols, static_cast<unsigned long long>(seed));
}

}  // namespace moe::sim::prefill_cube

#endif  // TESTS_SIM_DSV4_PREFILL_CUBE_DEQUANT_GOLDEN_HPP
