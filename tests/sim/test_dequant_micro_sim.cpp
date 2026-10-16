/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

// =================================================================================================
// test_dequant_micro_sim.cpp -- host harness for the dsv4_moe_prefill_cube routine micro-gates
// Target Architecture : Huawei Ascend 950PR (arch35 / dav-3510 / CANN 9.2), dav-3510 CAModel
// =================================================================================================
//
// 0. SCOPE: ONE BINARY, SEVERAL ROUTINES
// -------------------------------------------------------------------------------------------------
// This binary carries the inductive assembly of dsv4_moe_prefill_cube one routine at a time:
//
//     Step 1   DequantSubTileVector                        MXFP4 / E8M0 conversion in UB
//     Step 2   AsyncLoadTokensChunk + PrepareTokensChunk    the two-slot token prefetch ring
//
// They share one executable rather than getting one each, and the reason is measured: a run is
// ~290 s of which almost all is fixed simulator bring-up, because the launches themselves are tens
// of thousands of cycles. Another shape in this binary is nearly free; another binary is another
// ~290 s. Later steps belong here too.
//
// 1. WHAT A PASS HERE LICENSES, EXACTLY
// -------------------------------------------------------------------------------------------------
// "The production routines execute on the dav-3510 CAModel over the stated shapes, in isolation --
// AIV only, no KfcServer, no Cube -- and their BF16 output matches self-audited host goldens bit
// for bit." Nothing about hardware, nothing about the Cube path, nothing about throughput, and for
// Step 2 nothing about whether the prefetch actually overlapped the conversion IN TIME; see
// RunTokenRing for what rotation plus values does and does not establish. The simulator is
// cycle-accurate but it is not an NPU, and this gate is deliberately the second cheapest in the
// ladder -- after the host oracles and before the end-to-end collapsed run.
//
// 2. WHY M = 16 AND WHY [16, 64] IS THE PRIMARY CASE
// -------------------------------------------------------------------------------------------------
// The operator's token tile is forced to 16 by the residency of the on-chip activation plane, so 16
// is the row count every later step of the assembly will carry. The [16, 64] sub-tile is that row
// count against the narrowest group-aligned width that still spans more than one MXFP4 group:
// `groups_per_row` is 2, so the per-row `Broadcast<bfloat16_t, 2, 1>` has a real second dimension
// to expand and the 32-byte scale block is mostly padding -- which is the arrangement where a
// stride confusion between "4 scale bytes" and "32 scale entries" shows up as a wrong answer
// rather than as a coincidence.
//
// The two production sub-tile shapes follow it, because a routine that works at the small shape and
// not at [128, 128] or [64, 256] has only been proven at the small shape. They are cheap: one
// sub-tile each, no Cube, no KFC.
//
// 3. TOLERANCES, AND WHY THEY ARE WHAT THE FORMAT ALLOWS
// -------------------------------------------------------------------------------------------------
// BIT-EXACT. Not a bound fitted to an observed error -- the dequantization is a nibble decode, a
// bitshift and a multiply by a power of two, and `dsv4_prefill_cube_dequant_golden.hpp` derives in
// its header why no rounding occurs on either side and screens the fixture through
// `Mxfp4DequantIsExactInBf16` so the derivation holds for every generated pair. A mismatch of one
// ULP here is therefore a defect, not noise.
//
// Cosine similarity is still reported and still gated at 0.999, for the reason it always is: a
// bit-exact compare that fails tells you WHICH element moved, while cosine tells you whether the
// whole tile moved in DIRECTION -- an index transposition or a stride error -- or only in
// magnitude. Both are printed on a failure because they discriminate different bugs.
//
// 4. CANARY DISCIPLINE
// -------------------------------------------------------------------------------------------------
// Three bands, all poisoned with 0x5A5A before every launch:
//
//     * the guard tail past the logical output tile  -- catches an MTE burst that overruns
//     * the logical output tile itself               -- a word still holding 0x5A5A was never
//                                                       written, which a tolerance compare against
//                                                       a plausible value cannot distinguish from
//                                                       a wrong answer
//     * a UB-resident sentinel after `scale_temporary` -- catches a conversion that overruns the
//                                                       32-byte per-row scale temporary
//
// 0x5A5A as BF16 is about 1.5e16, and the golden audit asserts no golden element collides with it,
// so "still canary" and "correct" are disjoint by construction rather than by luck.

#include "dsv4_moe_block_sim_common.hpp"
#include "dsv4_prefill_cube_dequant_golden.hpp"
#include "dsv4_prefill_cube_tokens_golden.hpp"

extern "C" void LaunchDsv4PrefillCubeDequantMicro(void* packed, void* scales, void* output,
                                                   void* shape, void* status, void* ub_sentinel,
                                                   aclrtStream stream);

extern "C" void LaunchDsv4PrefillCubeTokensMicro(void* tokens_in, void* output, void* shape,
                                                  void* status, void* trace, void* ub_sentinel,
                                                  aclrtStream stream);

namespace {

using moe::sim::prefill_cube::AuditDequantSubTileGolden;
using moe::sim::prefill_cube::AuditTokenChunkPlan;
using moe::sim::prefill_cube::BuildDequantSubTile;
using moe::sim::prefill_cube::BuildTokenPlane;
using moe::sim::prefill_cube::CpuDequantSubTile;
using moe::sim::prefill_cube::CpuPrepareTokensChunk;
using moe::sim::prefill_cube::CpuTokenChunkPlan;
using moe::sim::prefill_cube::DequantSubTile;
using moe::sim::prefill_cube::kGoldenScaleRowStride;
using moe::sim::prefill_cube::kTokenMinChunks;
using moe::sim::prefill_cube::kTokenSlots;
using moe::sim::prefill_cube::TokenChunkPlan;

constexpr std::uint16_t kCanaryWord          = 0x5A5AU;
constexpr std::size_t   kGuardWords          = 128;  // 256 B past every output tile
constexpr std::uint32_t kSentinelBytes       = 256;
constexpr std::uint8_t  kSentinelByte        = 0x3CU;
constexpr std::uint32_t kStatusWords         = 8;
constexpr std::uint32_t kStatusPoison        = 0xFFFFFFFFU;
constexpr double        kCosineFloor         = 0.999;
constexpr int           kMicroBudgetSeconds  = 900;
constexpr std::uint64_t kMicroSeed           = 0xD5A4DE9CAFE1701ULL;

// The token micro-kernel's trace: two UB addresses per chunk, and 0xFFFFFFFF where there is no
// concurrent load (the epilogue chunk) or the chunk was never reached.
constexpr std::uint32_t kTraceMaxChunks = 32;
constexpr std::uint32_t kTraceWords     = 2U * kTraceMaxChunks;
constexpr std::uint32_t kTraceAbsent    = 0xFFFFFFFFU;

// M is forced to 16 by the activation plane's residency, and the contraction chunk is the
// operator's K_SUB. Both are fixed rather than swept: this gate is about the ring, and a chunk
// width the UB slots were not sized for is a different test.
constexpr std::uint32_t kTokenRows       = 16;
constexpr std::uint32_t kTokenChunkWidth = 128;

// `std::to_string` prints decimal, so a message that writes "0x" in front of it lies about the
// value. Every bit pattern below goes through this instead.
std::string Hex(std::uint64_t value, int digits) {
    static const char kDigits[] = "0123456789ABCDEF";
    std::string       out(static_cast<std::size_t>(digits), '0');
    for (int i = digits - 1; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kDigits[value & 0xFULL];
        value >>= 4U;
    }
    return "0x" + out;
}

// The status words the micro-kernel reports. Kept in step with the enum in
// tests/sim/kernel/test_dequant_micro_kernel.cpp; a code this table does not know is still
// reported, by number, rather than silently read as a pass.
const char* StatusName(std::uint32_t code) {
    switch (code) {
        case 0U:          return "ok";
        case 1U:          return "null GM operand";
        case 2U:          return "shape rejected (empty, unaligned, or scales exceed the row stride)";
        case 3U:          return "sub-tile exceeds the operator UB staging budget";
        case 4U:          return "an MTE burst would violate the 32-byte rule";
        case kStatusPoison: return "NOT WRITTEN -- the kernel never reached its status store";
        default:          return "<unknown status code>";
    }
}

struct TileMetrics {
    std::size_t mismatches     = 0;
    std::size_t unwritten      = 0;
    std::size_t first_bad      = 0;
    std::uint16_t first_got    = 0;
    std::uint16_t first_want   = 0;
    double      cosine         = 0.0;
};

TileMetrics CompareTile(const std::vector<std::uint16_t>& actual,
                        const std::vector<std::uint16_t>& golden) {
    Require(actual.size() == golden.size(), "tile comparison size mismatch");
    TileMetrics metrics;
    double      dot        = 0.0;
    double      norm_got   = 0.0;
    double      norm_want  = 0.0;

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float got  = moe::emu::Bf16ToFloat(actual[i]);
        const float want = moe::emu::Bf16ToFloat(golden[i]);
        dot       += static_cast<double>(got) * want;
        norm_got  += static_cast<double>(got) * got;
        norm_want += static_cast<double>(want) * want;
        if (actual[i] == kCanaryWord) ++metrics.unwritten;
        if (actual[i] != golden[i]) {
            if (metrics.mismatches == 0) {
                metrics.first_bad  = i;
                metrics.first_got  = actual[i];
                metrics.first_want = golden[i];
            }
            ++metrics.mismatches;
        }
    }
    metrics.cosine = (norm_got > 0.0 && norm_want > 0.0)
                         ? dot / (std::sqrt(norm_got) * std::sqrt(norm_want))
                         : 0.0;
    return metrics;
}

void RunShape(Runtime& runtime, LaunchWatchdog& watchdog, std::uint32_t rows, std::uint32_t cols,
              std::uint64_t seed) {
    const DequantSubTile             tile   = BuildDequantSubTile(rows, cols, seed);
    const std::vector<std::uint16_t> golden = CpuDequantSubTile(tile);
    const std::size_t                logical = tile.Elements();

    DeviceBuffer packed_dev(tile.packed.size());
    DeviceBuffer scales_dev(tile.padded_scales.size());
    DeviceBuffer output_dev((logical + kGuardWords) * sizeof(std::uint16_t));
    DeviceBuffer shape_dev(4 * sizeof(std::uint32_t));
    DeviceBuffer status_dev(kStatusWords * sizeof(std::uint32_t));
    DeviceBuffer sentinel_dev(kSentinelBytes);

    CopyToDevice(packed_dev, tile.packed);
    CopyToDevice(scales_dev, tile.padded_scales);
    CopyToDevice(shape_dev,
                 std::vector<std::uint32_t>{rows, cols, kGoldenScaleRowStride, 0U});
    CopyToDevice(output_dev, std::vector<std::uint16_t>(logical + kGuardWords, kCanaryWord));
    CopyToDevice(status_dev, std::vector<std::uint32_t>(kStatusWords, kStatusPoison));
    CopyToDevice(sentinel_dev, std::vector<std::uint8_t>(kSentinelBytes, kSentinelByte));

    const std::string tag = std::to_string(rows) + "x" + std::to_string(cols);
    watchdog.Arm("dequant micro-kernel launch at " + tag);
    LaunchDsv4PrefillCubeDequantMicro(packed_dev.get(), scales_dev.get(), output_dev.get(),
                                      shape_dev.get(), status_dev.get(), sentinel_dev.get(),
                                      runtime.stream());
    RequireAcl(aclrtSynchronizeStream(runtime.stream()), "aclrtSynchronizeStream(dequant micro)");
    watchdog.Disarm();

    const DumpCensus census = ExceptionDumps();
    if (census.non_empty > 0) {
        Fail(tag + ": the launch returned success but left " + std::to_string(census.non_empty) +
             " non-empty exception dump(s) behind; a device-side fault is not a pass");
    }

    std::vector<std::uint32_t> status(kStatusWords, 0U);
    CopyFromDevice(status, status_dev);
    if (status[0] != 0U) {
        Fail(tag + ": the kernel refused the launch with status " + std::to_string(status[0]) +
             " (" + StatusName(status[0]) + "); it observed rows=" + std::to_string(status[1]) +
             ", cols=" + std::to_string(status[2]) + ", operand=" + std::to_string(status[3]) +
             ", and its own budget is " + std::to_string(status[4]) + " elements over " +
             std::to_string(status[5]) + " scale rows at stride " + std::to_string(status[6]));
    }

    std::vector<std::uint8_t> sentinel(kSentinelBytes, 0U);
    CopyFromDevice(sentinel, sentinel_dev);
    for (std::size_t i = 0; i < sentinel.size(); ++i) {
        if (sentinel[i] != kSentinelByte) {
            Fail(tag + ": the UB sentinel following scale_temporary was modified at byte +" +
                 std::to_string(i) + " (" + Hex(sentinel[i], 2) +
                 "); the per-row scale conversion has overrun its 32-byte temporary");
        }
    }

    std::vector<std::uint16_t> readback(logical + kGuardWords, 0U);
    CopyFromDevice(readback, output_dev);
    for (std::size_t i = logical; i < readback.size(); ++i) {
        if (readback[i] != kCanaryWord) {
            Fail(tag + ": guard band corrupted at BF16 word +" + std::to_string(i - logical) +
                 " past the logical extent (" + std::to_string(logical) +
                 " words); the output burst overran its tile, leaving " + Hex(readback[i], 4));
        }
    }

    const std::vector<std::uint16_t> actual(readback.begin(),
                                            readback.begin() + static_cast<std::ptrdiff_t>(logical));
    const TileMetrics               metrics = CompareTile(actual, golden);

    if (metrics.unwritten > 0) {
        Fail(tag + ": " + std::to_string(metrics.unwritten) + " of " + std::to_string(logical) +
             " output words still hold the canary 0x5A5A, so the tile was only partially written");
    }
    if (metrics.mismatches > 0) {
        Fail(tag + ": " + std::to_string(metrics.mismatches) + " of " + std::to_string(logical) +
             " BF16 words differ from the golden, first at flat index " +
             std::to_string(metrics.first_bad) + " (row " +
             std::to_string(metrics.first_bad / cols) + ", col " +
             std::to_string(metrics.first_bad % cols) + "): got " +
             Hex(metrics.first_got, 4) + ", want " + Hex(metrics.first_want, 4) +
             "; cosine " + std::to_string(metrics.cosine) +
             (metrics.cosine < kCosineFloor
                  ? " -- the tile moved in DIRECTION, so suspect an index transposition or a stride"
                  : " -- the tile kept its direction, so suspect the scale decode rather than the "
                    "indexing"));
    }
    if (metrics.cosine < kCosineFloor) {
        Fail(tag + ": every word matches but cosine similarity is " +
             std::to_string(metrics.cosine) + ", below " + std::to_string(kCosineFloor) +
             "; the golden is degenerate and the compare proved nothing");
    }

    std::printf("[PASS] device dequant %ux%u: %zu BF16 words bit-exact, cosine %.6f, guard band "
                "and UB sentinel intact\n",
                rows, cols, logical, metrics.cosine);
}

// ---------------------------------------------------------------------------------------------
// STEP 2 -- the token prefetch ring on device
// ---------------------------------------------------------------------------------------------
// Checks four things, in the order that makes a failure attributable:
//
//   1. STATUS     -- the kernel reached its store and did not refuse the geometry.
//   2. ROTATION   -- from the kernel's own UB address trace: for every chunk that has a successor,
//                    the address the concurrent load targeted differs from the address the cast
//                    read, and the cast addresses alternate between exactly two slots. A ring that
//                    does not rotate fails here, before any value is compared, which is the
//                    difference between "the ring is wrong" and "the gather stride is wrong".
//   3. VALUES     -- every chunk bit-exact against CpuPrepareTokensChunk. Bit-exact because a
//                    single FP32-to-BF16 conversion in a specified rounding mode is deterministic
//                    and the fixture excludes ties, which are the only inputs where a converter may
//                    legitimately differ by one ULP on its tie-breaking convention.
//   4. CANARIES   -- guard tail past the chunk plane, no chunk word left holding the canary, and
//                    the UB sentinel behind the ring slots untouched.
//
// WHAT IT DOES NOT CHECK: that the loads and the casts overlapped IN TIME. A correct rotation plus
// correct values establishes that the slots are independent and no chunk is overwritten before it
// is consumed -- the correctness content of double buffering, and what a wrong ring breaks. Actual
// overlap is a scheduling outcome, and the only honest evidence for it is hardware.
void RunTokenRing(Runtime& runtime, LaunchWatchdog& watchdog, std::uint32_t hidden,
                  std::uint64_t seed) {
    const std::uint32_t      rows        = kTokenRows;
    const std::uint32_t      chunk_width = kTokenChunkWidth;
    const TokenChunkPlan     plan        = CpuTokenChunkPlan(hidden, chunk_width);
    const std::vector<float> tokens      = BuildTokenPlane(rows, hidden, seed);
    const std::size_t        logical =
        static_cast<std::size_t>(plan.chunks) * rows * chunk_width;

    DeviceBuffer tokens_dev(tokens.size() * sizeof(float));
    DeviceBuffer output_dev((logical + kGuardWords) * sizeof(std::uint16_t));
    DeviceBuffer shape_dev(4 * sizeof(std::uint32_t));
    DeviceBuffer status_dev(kStatusWords * sizeof(std::uint32_t));
    DeviceBuffer trace_dev(kTraceWords * sizeof(std::uint32_t));
    DeviceBuffer sentinel_dev(kSentinelBytes);

    CopyToDevice(tokens_dev, tokens);
    CopyToDevice(shape_dev, std::vector<std::uint32_t>{rows, hidden, chunk_width, 0U});
    CopyToDevice(output_dev, std::vector<std::uint16_t>(logical + kGuardWords, kCanaryWord));
    CopyToDevice(status_dev, std::vector<std::uint32_t>(kStatusWords, kStatusPoison));
    CopyToDevice(trace_dev, std::vector<std::uint32_t>(kTraceWords, kStatusPoison));
    CopyToDevice(sentinel_dev, std::vector<std::uint8_t>(kSentinelBytes, kSentinelByte));

    const std::string tag = "hidden=" + std::to_string(hidden) + " (" +
                            std::to_string(plan.chunks) + " chunks)";
    watchdog.Arm("token ring launch at " + tag);
    LaunchDsv4PrefillCubeTokensMicro(tokens_dev.get(), output_dev.get(), shape_dev.get(),
                                     status_dev.get(), trace_dev.get(), sentinel_dev.get(),
                                     runtime.stream());
    RequireAcl(aclrtSynchronizeStream(runtime.stream()), "aclrtSynchronizeStream(token ring)");
    watchdog.Disarm();

    const DumpCensus census = ExceptionDumps();
    if (census.non_empty > 0) {
        Fail(tag + ": the launch returned success but left " + std::to_string(census.non_empty) +
             " non-empty exception dump(s) behind; a device-side fault is not a pass");
    }

    std::vector<std::uint32_t> status(kStatusWords, 0U);
    CopyFromDevice(status, status_dev);
    if (status[0] != 0U) {
        Fail(tag + ": the kernel refused the launch with status " + std::to_string(status[0]) +
             " (" + StatusName(status[0]) + "); it observed rows=" + std::to_string(status[1]) +
             ", hidden=" + std::to_string(status[2]) + ", operand=" + std::to_string(status[3]));
    }

    // (2) ROTATION, from the addresses the kernel actually handed the production routines.
    std::vector<std::uint32_t> trace(kTraceWords, 0U);
    CopyFromDevice(trace, trace_dev);
    std::vector<std::uint32_t> cast_slots;
    for (std::uint32_t chunk = 0; chunk < plan.chunks; ++chunk) {
        const std::uint32_t load_destination = trace[2U * chunk];
        const std::uint32_t cast_source      = trace[2U * chunk + 1U];
        if (cast_source == kTraceAbsent) {
            Fail(tag + ": chunk " + std::to_string(chunk) +
                 " was never converted -- the trace slot is still absent, so the loop did not "
                 "reach it");
        }
        const bool has_successor = chunk + 1U < plan.chunks;
        if (has_successor && load_destination == kTraceAbsent) {
            Fail(tag + ": chunk " + std::to_string(chunk) +
                 " has a successor but issued no concurrent load; the prefetch is not running "
                 "ahead");
        }
        if (has_successor && load_destination == cast_source) {
            Fail(tag + ": ROTATION -- at chunk " + std::to_string(chunk) +
                 " the prefetch targets UB " + Hex(load_destination, 8) +
                 ", the same slot the cast is reading. The ring is not rotating, so chunk " +
                 std::to_string(chunk + 1U) + " is overwriting chunk " + std::to_string(chunk) +
                 " while it is still live.");
        }
        cast_slots.push_back(cast_source);
    }
    for (std::uint32_t chunk = 1; chunk < plan.chunks; ++chunk) {
        if (cast_slots[chunk] == cast_slots[chunk - 1U]) {
            Fail(tag + ": ROTATION -- chunks " + std::to_string(chunk - 1U) + " and " +
                 std::to_string(chunk) + " were both converted out of UB " +
                 Hex(cast_slots[chunk], 8));
        }
    }
    std::vector<std::uint32_t> distinct_slots = cast_slots;
    std::sort(distinct_slots.begin(), distinct_slots.end());
    distinct_slots.erase(std::unique(distinct_slots.begin(), distinct_slots.end()),
                         distinct_slots.end());
    const std::size_t expected_slots = plan.chunks < kTokenSlots ? plan.chunks : kTokenSlots;
    if (distinct_slots.size() != expected_slots) {
        Fail(tag + ": ROTATION -- the casts read from " + std::to_string(distinct_slots.size()) +
             " distinct UB slots over " + std::to_string(plan.chunks) + " chunks, want " +
             std::to_string(expected_slots));
    }

    // (4) Guard tail, before trusting anything inside the plane.
    std::vector<std::uint16_t> readback(logical + kGuardWords, 0U);
    CopyFromDevice(readback, output_dev);
    for (std::size_t i = logical; i < readback.size(); ++i) {
        if (readback[i] != kCanaryWord) {
            Fail(tag + ": guard band corrupted at BF16 word +" + std::to_string(i - logical) +
                 " past the chunk plane (" + std::to_string(logical) + " words), holding " +
                 Hex(readback[i], 4));
        }
    }

    std::vector<std::uint8_t> sentinel(kSentinelBytes, 0U);
    CopyFromDevice(sentinel, sentinel_dev);
    for (std::size_t i = 0; i < sentinel.size(); ++i) {
        if (sentinel[i] != kSentinelByte) {
            Fail(tag + ": the UB sentinel behind the ring slots was modified at byte +" +
                 std::to_string(i) + " (" + Hex(sentinel[i], 2) +
                 "); a token burst has overrun its slot");
        }
    }

    // (3) VALUES, per chunk, so a mismatch names the chunk and therefore the hidden_base.
    const std::size_t chunk_elements = static_cast<std::size_t>(rows) * chunk_width;
    double            dot            = 0.0;
    double            norm_got       = 0.0;
    double            norm_want      = 0.0;
    for (std::uint32_t chunk = 0; chunk < plan.chunks; ++chunk) {
        const std::vector<std::uint16_t> golden = CpuPrepareTokensChunk(
            tokens.data(), rows, hidden, chunk * chunk_width, chunk_width);
        for (std::size_t i = 0; i < chunk_elements; ++i) {
            const std::uint16_t got  = readback[chunk * chunk_elements + i];
            const std::uint16_t want = golden[i];
            const float         got_f  = moe::emu::Bf16ToFloat(got);
            const float         want_f = moe::emu::Bf16ToFloat(want);
            dot       += static_cast<double>(got_f) * want_f;
            norm_got  += static_cast<double>(got_f) * got_f;
            norm_want += static_cast<double>(want_f) * want_f;
            if (got == kCanaryWord) {
                Fail(tag + ": chunk " + std::to_string(chunk) + " word " + std::to_string(i) +
                     " still holds the canary, so the chunk was never stored");
            }
            if (got != want) {
                // Which chunk the WRONG value belongs to is the diagnosis: a neighbouring chunk
                // means the ring handed over the wrong slot, anything else means the gather.
                std::string origin = "no chunk in this plane";
                for (std::uint32_t other = 0; other < plan.chunks; ++other) {
                    const std::vector<std::uint16_t> candidate = CpuPrepareTokensChunk(
                        tokens.data(), rows, hidden, other * chunk_width, chunk_width);
                    if (candidate[i] == got) {
                        origin = "chunk " + std::to_string(other);
                        break;
                    }
                }
                Fail(tag + ": chunk " + std::to_string(chunk) + " (hidden_base " +
                     std::to_string(chunk * chunk_width) + ") differs at row " +
                     std::to_string(i / chunk_width) + ", col " + std::to_string(i % chunk_width) +
                     ": got " + Hex(got, 4) + ", want " + Hex(want, 4) +
                     "; the value read belongs to " + origin);
            }
        }
    }
    const double cosine = (norm_got > 0.0 && norm_want > 0.0)
                              ? dot / (std::sqrt(norm_got) * std::sqrt(norm_want))
                              : 0.0;
    if (cosine < kCosineFloor) {
        Fail(tag + ": every word matches but cosine similarity is " + std::to_string(cosine) +
             ", below " + std::to_string(kCosineFloor) + "; the token plane is degenerate");
    }

    std::printf("[PASS] device token ring %s: %zu BF16 words bit-exact over %u slots, cosine "
                "%.6f, rotation verified from %u traced UB addresses, guard band and UB sentinel "
                "intact\n",
                tag.c_str(), logical, static_cast<std::uint32_t>(distinct_slots.size()), cosine,
                2U * plan.chunks);
}

}  // namespace

int main() {
    try {
        std::printf("[SEED] 0x%016llX -- every fixture below is derived from it in memory\n",
                    static_cast<unsigned long long>(kMicroSeed));

        // Phase 1. Audit both oracles on the host, before any device work: a device comparison
        // against an unaudited golden proves nothing about the device. The token ring's schedule
        // audit is the cheaper half and goes first -- a marked-graph token imbalance presents on
        // CAModel as a hang, and counting the tokens in a host loop costs microseconds against the
        // ~290 s and a watchdog trip that diagnosing the hang would cost.
        AuditDequantSubTileGolden(16, 64, kMicroSeed);
        AuditDequantSubTileGolden(128, 128, kMicroSeed + 1U);
        AuditDequantSubTileGolden(64, 256, kMicroSeed + 2U);

        // Swept well past the shapes the device cases run, because it is free here: every
        // chunk-aligned hidden extent the operator admits, at the operator's K_SUB.
        for (std::uint32_t hidden = kTokenChunkWidth; hidden <= 4096U; hidden += kTokenChunkWidth) {
            CpuTokenChunkPlan(hidden, kTokenChunkWidth);
        }
        AuditTokenChunkPlan(kTokenChunkWidth * 1U, kTokenChunkWidth);              // degenerate
        AuditTokenChunkPlan(kTokenChunkWidth * 2U, kTokenChunkWidth);              // one flip
        AuditTokenChunkPlan(kTokenChunkWidth * kTokenMinChunks, kTokenChunkWidth);  // minimal ring
        AuditTokenChunkPlan(kTokenChunkWidth * 6U, kTokenChunkWidth);              // two wraps
        AuditTokenChunkPlan(4096U, kTokenChunkWidth);                              // production

        // Phase 2. Equivalence on device, primary shape first.
        Runtime        runtime;
        LaunchWatchdog watchdog(kMicroBudgetSeconds, "[PREFILL-CUBE-MICRO]");

        // Step 1 -- the MXFP4/E8M0 conversion.
        RunShape(runtime, watchdog, 16, 64, kMicroSeed);
        RunShape(runtime, watchdog, 128, 128, kMicroSeed + 1U);
        RunShape(runtime, watchdog, 64, 256, kMicroSeed + 2U);

        // Step 2 -- the two-slot token prefetch ring. kTokenMinChunks is the smallest grid in
        // which every structural branch fires: one chunk primes, one flips, one wraps back onto a
        // slot that has already been used, one drains. The 6-chunk case wraps twice, and the
        // 2-chunk case is below the minimum so it runs the loop body exactly once -- the boundary
        // where a prologue or epilogue off-by-one lives.
        RunTokenRing(runtime, watchdog, kTokenChunkWidth * kTokenMinChunks, kMicroSeed + 3U);
        RunTokenRing(runtime, watchdog, kTokenChunkWidth * 6U, kMicroSeed + 4U);
        RunTokenRing(runtime, watchdog, kTokenChunkWidth * 2U, kMicroSeed + 5U);
        runtime.Close();

        std::printf("[PASS] DequantSubTileVector and the token prefetch ring verified in "
                    "isolation on the dav-3510 CAModel; never executed on hardware\n");
        return 0;
    } catch (const std::exception& error) {
        std::printf("[FAIL] %s\n", error.what());
        return 1;
    }
}
