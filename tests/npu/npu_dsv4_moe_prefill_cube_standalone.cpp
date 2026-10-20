// =================================================================================================
// test_npu_dsv4_moe_prefill_cube_standalone -- on-device verification of the
// dsv4_moe_prefill_cube_pipeline entry point, isolated from the rest of the engine.
//
// Links one standalone BiSheng-built device library -- the pipeline kernel, its op_host chevron
// wrapper, the tiling builder and launch_common -- plus moe_emu, which supplies the in-memory CPU
// golden. Nothing else of the engine is compiled or linked, and nothing is read from disk.
//
// -------------------------------------------------------------------------------------------------
// ENTRY GUARD
// -------------------------------------------------------------------------------------------------
// An absent NPU is an environment without hardware, not a test failure. main() probes
// aclrtGetDeviceCount before any case runs and, on zero devices or a failed init, prints a banner
// containing "Physical Ascend NPU not detected" and exits 0 -- the phrase this target's ctest
// registration matches as its SKIP_REGULAR_EXPRESSION. A device that IS present but fails bring-up
// is kept apart from that and reported as a failure, because collapsing the two makes the suite
// either cry wolf on every laptop or pass silently on a broken machine.
//
// Past the guard a device exists, so every ACL error and every non-success Status is a hard FAIL
// carrying its code, the recorded LastLaunchFailure() and aclGetRecentErrMsg().
//
// The guard can only run where the DRIVER is installed: libascendcl pulls in libascend_hal.so,
// which ships with the driver and not the toolkit, and the build links the driver stubs with
// -rpath-link only. On a host with neither driver nor device this binary fails to load before
// main() and exits 127, which is the correct outcome and must not be "fixed" by putting the stub
// directory on the runtime search path.
//
// -------------------------------------------------------------------------------------------------
// THE CASES
// -------------------------------------------------------------------------------------------------
//   --gtest_filter=*Tile1*   one token tile, RAMP fixture. Cold-start rendezvous, the prologue MTE2
//                            burst that precedes the step loop, the first Fixpipe DUAL_DST_SPLIT_M
//                            hand-over. Dumps the realised weight ramp, the staged tiling POD, the
//                            head of the activation plane input and the worst element of every row.
//   --gtest_filter=*Tile2*   two token tiles over two contraction steps in each phase, RAMP
//                            fixture, so every L1 ring slot is written, read and flipped once:
//                            the ping-pong state transition and the per-slot event-id economy.
//   --gtest_filter=*Tile4*   four token tiles over four phase-1 contraction steps, random fixture.
//                            The first grid that recycles a ring slot, so the AIC's StepFree
//                            release and the AIVs' deferred MTE3_V drain both fire, then drain.
//   --gtest_filter=*Full*    every token over every expert with a ragged tail tile, random
//                            fixture, compared element by element. Reports the worst element's row
//                            and column alongside the max absolute and relative error.
//
// Phase 2 runs at most two contraction steps in every case: the test build defines
// DSV4_TILING_TEST_OVERRIDE=1, which caps `inter` at 512 while R_SUB2 is 256. The steady-state
// slot recycle is therefore covered on the phase-1 axis only.
//
// -------------------------------------------------------------------------------------------------
// THE RAMP FIXTURE, AND WHY IT IS SCALED
// -------------------------------------------------------------------------------------------------
// Tile1 and Tile2 exist to be read, so their operands are a ramp rather than noise:
//
//     activation   x[token][col] = ((token + col) % 16) + 1          literally 1, 2, 3 ... 16
//     weight       W[row][col]   = 0.1 * (1 + ((row + col) % 16))    the 0.1, 0.2, 0.3 ... ramp
//
// The activation ramp is phase-shifted by the token index so that no two token rows are identical:
// an unshifted ramp makes every row of the output the same number, which is exactly the state in
// which a mis-split DUAL_DST_SPLIT_M or a wrong row base cannot be seen.
//
// THE WEIGHT RAMP CARRIES A 1/cols FACTOR IN ITS E8M0 SCALE, and that is not cosmetic. SwiGLU
// clamps its pre-activations to +/- 10; a contraction of `cols` terms of an unscaled 0.1..1.6 ramp
// against a 1..16 ramp reaches several thousand, so EVERY pre-activation would saturate, the
// activation plane would be the constant Silu(10)*10, and the output would carry no trace of
// either ramp. Dividing the weight magnitude by the contraction length puts the pre-activations
// just under the clamp, with a minority above it, so the clamp is still exercised and the ramp
// still reaches the output. The E8M0 exponent is therefore 127 - 2 - log2(cols): the -2 places the
// 0.1..1.6 ramp on the E2M1 grid at a step of 0.125, and the -log2(cols) is the scaling.
//
// MXFP4 cannot represent a decimal ramp exactly -- E2M1 magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}
// -- so the sixteen requested values land on eight distinct codes. Tile1 prints the requested and
// the realised value side by side; both the device and the golden read the SAME stored nibbles, so
// the quantization costs nothing in parity.
//
// Tile4 and Full use a seeded normal fixture instead, because a ramp is a highly correlated input
// and correlated inputs hide cancellation errors.
//
// -------------------------------------------------------------------------------------------------
// FIXTURES AND TOLERANCES
// -------------------------------------------------------------------------------------------------
// Weights, activations and routing are generated in memory, per run; the random cases derive from
// the seed printed at startup. There are no disk fixtures: a stale asset must not be able to make
// a broken kernel pass.
//
// The bound is set by the format, not fitted to the observed error. The device dequantizes MXFP4 to
// BF16 and contracts BF16 x BF16 on the Cube with an FP32 L0C accumulator; the golden contracts in
// FP32 from the same nibbles and scales. BF16 keeps an 8-bit mantissa, so each operand carries up
// to 2^-9 = 1.95e-3 relative error, and a dot product of K terms accumulates it incoherently.
// Across two chained GEMMs with a BF16-rounded activation between them, 1e-2 relative plus 1e-2
// absolute is the first round power-of-ten bound above that; the absolute term covers rows whose
// output is near zero, where a relative bound is meaningless.
//
// Cosine stays strict at 0.999 where relative-L2 must be loose: rounding moves magnitude, while a
// structural bug -- a wrong operand transpose, a mis-split contraction half, a slot read before it
// was filled -- moves direction.
//
// Degeneracy is guarded on the GOLDEN, not by comparing a value with itself: every case asserts
// that the reference has a nonzero L2 norm and real variance before it compares anything to it.
// =================================================================================================

#include <gtest/gtest.h>

#include "dsv4_moe_prefill_cube_pipeline.h"

#include "moe/emu/dsv4_layers.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using moe::kernels::arch35::Dsv4MoePrefillCubeTilingData;
using moe::kernels::arch35::E8M0;
using moe::kernels::arch35::kPrefillCubeHSub;
using moe::kernels::arch35::kPrefillCubeKSub;
using moe::kernels::arch35::kPrefillCubeMTile;
using moe::kernels::arch35::kPrefillCubePipelineWorkspaceBytes;
using moe::kernels::arch35::kPrefillCubeRSub;
using moe::kernels::arch35::kPrefillCubeRSub2;
using moe::kernels::arch35::kPrefillCubeSwigluLimit;
using moe::kernels::arch35::LastLaunchFailure;
using moe::kernels::arch35::LaunchDsv4MoePrefillCubePipeline;
using moe::kernels::arch35::Mxfp4Pair;
using moe::kernels::arch35::Status;
using moe::kernels::arch35::ToString;

constexpr std::uint32_t kGroupSize   = 32;
constexpr std::uint32_t kRampPeriod  = 16;
constexpr float         kAbsTol      = 1.0e-2F;
constexpr float         kRelTol      = 1.0e-2F;
constexpr double        kCosineFloor = 0.999;

aclrtStream   g_stream = nullptr;
std::uint64_t g_seed   = 0;

std::string AclTail() {
    const char* msg = aclGetRecentErrMsg();
    return (msg == nullptr || *msg == '\0') ? std::string("<none>") : std::string(msg);
}

std::string LaunchTail(Status status) {
    const auto  failure = LastLaunchFailure();
    std::string out     = std::string("Status=") + ToString(status);
    if (failure.op != nullptr && failure.call != nullptr) {
        out += std::string("  op=") + failure.op + "  call=" + failure.call +
               "  aclError=" + std::to_string(failure.acl_error);
    }
    return out + "  aclGetRecentErrMsg=" + AclTail();
}

class DeviceBuffer {
public:
    DeviceBuffer()                               = default;
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)aclrtFree(ptr_);
        }
    }

    bool Allocate(std::size_t bytes) {
        bytes_ = bytes;
        return aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    }

    bool Upload(const void* host, std::size_t bytes) {
        return aclrtMemcpy(ptr_, bytes_, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }

    bool Download(void* host, std::size_t bytes) const {
        return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    }

    void*       get() const noexcept { return ptr_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void*       ptr_   = nullptr;
    std::size_t bytes_ = 0;
};

std::uint8_t EncodeFp4(float value) {
    std::uint8_t best  = 0;
    float        error = std::numeric_limits<float>::infinity();
    for (std::uint8_t code = 0; code < 16; ++code) {
        const float candidate = std::fabs(value - moe::emu::DecodeFp4E2M1(code));
        if (candidate < error) {
            best  = code;
            error = candidate;
        }
    }
    return best;
}

std::uint32_t Log2Exact(std::uint32_t value) {
    std::uint32_t bits = 0;
    while ((value >> bits) > 1u) {
        ++bits;
    }
    return bits;
}

// The ramp's E8M0 exponent: -2 puts the 0.1 .. 1.6 sequence on the E2M1 grid at a step of 0.125,
// and -log2(cols) divides the weight magnitude by the contraction length so SwiGLU does not clamp
// every pre-activation. See "THE RAMP FIXTURE" above.
E8M0 RampScaleCode(std::uint32_t cols) {
    return static_cast<E8M0>(127u - 2u - Log2Exact(cols));
}

float RampRequestedWeight(std::uint32_t row, std::uint32_t col) {
    return 0.1F * static_cast<float>(1u + ((row + col) % kRampPeriod));
}

float RampActivation(std::uint32_t token, std::uint32_t col) {
    return static_cast<float>(((token + col) % kRampPeriod) + 1u);
}

enum class Fill {
    kRamp,
    kRandom
};

struct ExpertBank {
    std::uint32_t          rows = 0;
    std::uint32_t          cols = 0;
    std::vector<Mxfp4Pair> packed;
    std::vector<E8M0>      scales;

    std::size_t PackedStride() const { return static_cast<std::size_t>(rows) * (cols / 2u); }
    std::size_t ScaleStride() const { return static_cast<std::size_t>(rows) * (cols / kGroupSize); }

    moe::emu::Mxfp4Matrix View(std::uint32_t expert) const {
        return {packed.data() + static_cast<std::size_t>(expert) * PackedStride(),
                scales.data() + static_cast<std::size_t>(expert) * ScaleStride(),
                static_cast<int>(rows), static_cast<int>(cols)};
    }
};

void AllocateBank(ExpertBank& bank, std::uint32_t rows, std::uint32_t cols, std::uint32_t experts) {
    bank.rows = rows;
    bank.cols = cols;
    bank.packed.assign(static_cast<std::size_t>(experts) * rows * (cols / 2u), 0);
    bank.scales.assign(static_cast<std::size_t>(experts) * rows * (cols / kGroupSize), 0);
}

void StorePair(ExpertBank& bank, std::uint32_t expert, std::uint32_t row, std::uint32_t col,
               std::uint8_t low, std::uint8_t high) {
    bank.packed[static_cast<std::size_t>(expert) * bank.PackedStride() +
                static_cast<std::size_t>(row) * (bank.cols / 2u) + col / 2u] =
        static_cast<Mxfp4Pair>(low | (high << 4u));
}

void StoreScale(ExpertBank& bank, std::uint32_t expert, std::uint32_t row, std::uint32_t group,
                E8M0 code) {
    bank.scales[static_cast<std::size_t>(expert) * bank.ScaleStride() +
                static_cast<std::size_t>(row) * (bank.cols / kGroupSize) + group] = code;
}

// The ramp bank. One E8M0 code for every group so the stored magnitudes are directly comparable
// across the whole matrix, and the expert index shifts the ramp phase so a wrong-expert read
// cannot pass unnoticed.
ExpertBank MakeRampBank(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts) {
    ExpertBank bank;
    AllocateBank(bank, rows, cols, experts);

    const E8M0  code  = RampScaleCode(cols);
    const float scale = std::ldexp(1.0F, static_cast<int>(code) - 127);

    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t group = 0; group < cols / kGroupSize; ++group) {
                StoreScale(bank, expert, row, group, code);
            }
            for (std::uint32_t col = 0; col < cols; col += 2u) {
                const std::uint8_t low  = EncodeFp4(RampRequestedWeight(row + expert, col) / scale);
                const std::uint8_t high = EncodeFp4(RampRequestedWeight(row + expert, col + 1u) / scale);
                StorePair(bank, expert, row, col, low, high);
            }
        }
    }
    return bank;
}

ExpertBank MakeRandomBank(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts,
                          std::mt19937_64& rng) {
    ExpertBank bank;
    AllocateBank(bank, rows, cols, experts);

    std::normal_distribution<float> source(0.0F, 1.7F);
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t group = 0; group < cols / kGroupSize; ++group) {
                // The expert id participates so no two experts share a scale pattern.
                const E8M0 code = static_cast<E8M0>(126u + ((row + group + expert) % 3u));
                StoreScale(bank, expert, row, group, code);
                const float scale = std::ldexp(1.0F, static_cast<int>(code) - 127);
                for (std::uint32_t col = group * kGroupSize; col < (group + 1u) * kGroupSize; col += 2u) {
                    const std::uint8_t low  = EncodeFp4(source(rng) / scale);
                    const std::uint8_t high = EncodeFp4(source(rng) / scale);
                    StorePair(bank, expert, row, col, low, high);
                }
            }
        }
    }
    return bank;
}

struct Shape {
    std::uint32_t hidden;
    std::uint32_t inter;
    std::uint32_t experts;
    std::uint32_t tokens;
    std::uint32_t aiv_count;
    Fill          fill;
};

struct Fixture {
    Shape                     shape;
    std::vector<float>        x;
    std::vector<std::int32_t> route;
    ExpertBank                gate_weights;
    ExpertBank                up_weights;
    ExpertBank                down_weights;
    std::vector<float>        golden;
};

Fixture BuildFixture(const Shape& shape) {
    Fixture fixture;
    fixture.shape = shape;

    std::mt19937_64 rng(g_seed ^ (static_cast<std::uint64_t>(shape.tokens) << 32) ^ shape.hidden);

    fixture.x.resize(static_cast<std::size_t>(shape.tokens) * shape.hidden);
    if (shape.fill == Fill::kRamp) {
        for (std::uint32_t token = 0; token < shape.tokens; ++token) {
            for (std::uint32_t col = 0; col < shape.hidden; ++col) {
                fixture.x[static_cast<std::size_t>(token) * shape.hidden + col] =
                    RampActivation(token, col);
            }
        }
    } else {
        std::normal_distribution<float> activation(0.0F, 1.0F);
        for (float& value : fixture.x) {
            value = activation(rng);
        }
    }

    // One expert per TILE, not per token: the device reads the routing of the tile's first row once
    // and contracts the whole tile against that expert, which is the operator's contract.
    fixture.route.resize(shape.tokens);
    const std::uint32_t tiles = (shape.tokens + kPrefillCubeMTile - 1u) / kPrefillCubeMTile;
    for (std::uint32_t tile = 0; tile < tiles; ++tile) {
        const std::int32_t  expert = static_cast<std::int32_t>(tile % shape.experts);
        const std::uint32_t last   = std::min(shape.tokens, (tile + 1u) * kPrefillCubeMTile);
        for (std::uint32_t row = tile * kPrefillCubeMTile; row < last; ++row) {
            fixture.route[row] = expert;
        }
    }

    if (shape.fill == Fill::kRamp) {
        fixture.gate_weights = MakeRampBank(shape.inter, shape.hidden, shape.experts);
        fixture.up_weights   = MakeRampBank(shape.inter, shape.hidden, shape.experts);
        fixture.down_weights = MakeRampBank(shape.hidden, shape.inter, shape.experts);
    } else {
        fixture.gate_weights = MakeRandomBank(shape.inter, shape.hidden, shape.experts, rng);
        fixture.up_weights   = MakeRandomBank(shape.inter, shape.hidden, shape.experts, rng);
        fixture.down_weights = MakeRandomBank(shape.hidden, shape.inter, shape.experts, rng);
    }

    moe::emu::CubeTileGeometry geometry;
    geometry.m_tile       = static_cast<int>(kPrefillCubeMTile);
    geometry.r_tile       = static_cast<int>(kPrefillCubeRSub);
    geometry.k_tile       = static_cast<int>(kPrefillCubeKSub);
    geometry.h_tile       = static_cast<int>(kPrefillCubeHSub);
    geometry.r_tile_down  = static_cast<int>(kPrefillCubeRSub2);
    geometry.buffer_depth = 2;

    std::vector<moe::emu::Mxfp4Matrix> gate_views, up_views, down_views;
    for (std::uint32_t expert = 0; expert < shape.experts; ++expert) {
        gate_views.push_back(fixture.gate_weights.View(expert));
        up_views.push_back(fixture.up_weights.View(expert));
        down_views.push_back(fixture.down_weights.View(expert));
    }

    fixture.golden.assign(fixture.x.size(), 0.0F);
    std::vector<float> scratch;
    moe::emu::moe_prefill_cube_flash_2d_w4a16_cpu(
        gate_views.data(), up_views.data(), down_views.data(), fixture.x.data(),
        fixture.route.data(), static_cast<int>(shape.tokens), static_cast<int>(shape.experts),
        fixture.golden.data(), kPrefillCubeSwigluLimit, geometry, nullptr, &scratch);

    return fixture;
}

struct Spread {
    double l2       = 0.0;
    double variance = 0.0;
};

Spread MeasureSpread(const std::vector<float>& values) {
    Spread spread;
    if (values.empty()) {
        return spread;
    }
    double sum = 0.0, sum_sq = 0.0;
    for (const float value : values) {
        sum += value;
        sum_sq += static_cast<double>(value) * value;
    }
    const double n    = static_cast<double>(values.size());
    const double mean = sum / n;
    spread.l2         = std::sqrt(sum_sq);
    spread.variance   = std::max(0.0, sum_sq / n - mean * mean);
    return spread;
}

struct Mismatch {
    double        max_abs   = 0.0;
    double        max_rel   = 0.0;
    double        cosine    = 1.0;
    std::uint32_t worst_row = 0;
    std::uint32_t worst_col = 0;
    double        worst_got = 0.0;
    double        worst_exp = 0.0;
    bool          finite    = true;
    bool          within    = true;
};

Mismatch Compare(const std::vector<float>& actual, const std::vector<float>& expected,
                 std::uint32_t rows, std::uint32_t cols) {
    Mismatch diff;
    double   dot = 0.0, norm_actual = 0.0, norm_expected = 0.0;
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * cols + col;
            const double      got   = actual[index];
            const double      want  = expected[index];
            if (!std::isfinite(got) || !std::isfinite(want)) {
                diff.finite = false;
                diff.within = false;
                continue;
            }
            const double delta = std::fabs(got - want);
            if (delta > diff.max_abs) {
                diff.max_abs   = delta;
                diff.worst_row = row;
                diff.worst_col = col;
                diff.worst_got = got;
                diff.worst_exp = want;
            }
            if (std::fabs(want) > 1.0e-6) {
                diff.max_rel = std::max(diff.max_rel, delta / std::fabs(want));
            }
            if (delta > kAbsTol + kRelTol * std::fabs(want)) {
                diff.within = false;
            }
            dot += got * want;
            norm_actual += got * got;
            norm_expected += want * want;
        }
    }
    if (norm_actual == 0.0 && norm_expected == 0.0) {
        diff.cosine = 1.0;
    } else if (norm_actual == 0.0 || norm_expected == 0.0) {
        diff.cosine = 0.0;
    } else {
        diff.cosine = dot / (std::sqrt(norm_actual) * std::sqrt(norm_expected));
    }
    return diff;
}

void DumpTiling(const Dsv4MoePrefillCubeTilingData& tiling) {
    std::printf(
        "    tiling: abi=%u block_dim=%u hidden=%u inter=%u experts=%u tokens=%u m_tile=%u\n"
        "            r_sub=%u k_sub=%u h_sub=%u r_sub2=%u group=%u low_nibble_first=%u limit=%.3f\n"
        "            strides: w13 packed=%u scale=%u  w2 packed=%u scale=%u\n",
        tiling.abi_version, tiling.block_dim, tiling.hidden, tiling.inter, tiling.num_experts,
        tiling.total_tokens, tiling.m_tile, tiling.r_sub, tiling.k_sub, tiling.h_sub,
        tiling.r_sub2, tiling.group_size, tiling.low_nibble_first,
        static_cast<double>(tiling.swiglu_limit), tiling.w13_packed_expert_stride,
        tiling.w13_scale_expert_stride, tiling.w2_packed_expert_stride,
        tiling.w2_scale_expert_stride);
}

void DumpRampRealisation(const Fixture& fixture) {
    const std::uint32_t cols  = fixture.shape.hidden;
    const E8M0          code  = RampScaleCode(cols);
    const float         scale = std::ldexp(1.0F, static_cast<int>(code) - 127);

    std::printf("    gate/up ramp, E8M0 code %u (scale 2^%d = %.9g):\n", static_cast<unsigned>(code),
                static_cast<int>(code) - 127, static_cast<double>(scale));
    for (std::uint32_t step = 0; step < kRampPeriod; ++step) {
        const float requested = 0.1F * static_cast<float>(step + 1u);
        const float realised  = moe::emu::DecodeFp4E2M1(EncodeFp4(requested / scale)) * scale;
        std::printf("      step %2u  requested %.4f  nibble x scale -> realised %.9g  (x %u = %.6g)\n",
                    step + 1u, static_cast<double>(requested), static_cast<double>(realised), cols,
                    static_cast<double>(realised) * cols);
    }
    std::printf("    activation ramp head, token 0: ");
    for (std::uint32_t col = 0; col < std::min<std::uint32_t>(kRampPeriod, cols); ++col) {
        std::printf("%g ", static_cast<double>(fixture.x[col]));
    }
    std::printf("\n    activation ramp head, token 1: ");
    if (fixture.shape.tokens > 1u) {
        for (std::uint32_t col = 0; col < std::min<std::uint32_t>(kRampPeriod, cols); ++col) {
            std::printf("%g ", static_cast<double>(fixture.x[cols + col]));
        }
    }
    std::printf("\n");
}

void DumpPerRowError(const std::vector<float>& actual, const std::vector<float>& expected,
                     std::uint32_t rows, std::uint32_t cols) {
    for (std::uint32_t row = 0; row < rows; ++row) {
        double        worst  = 0.0;
        std::uint32_t at_col = 0;
        for (std::uint32_t col = 0; col < cols; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * cols + col;
            const double      delta = std::fabs(static_cast<double>(actual[index]) - expected[index]);
            if (delta > worst) {
                worst  = delta;
                at_col = col;
            }
        }
        const std::size_t head = static_cast<std::size_t>(row) * cols;
        std::printf("    row %3u  worst abs %.6e at col %4u   y[0]=%+.6f golden[0]=%+.6f\n", row,
                    worst, at_col, static_cast<double>(actual[head]),
                    static_cast<double>(expected[head]));
    }
}

// Runs one fixture on the device. Every ACL error past the entry guard is a hard failure, so this
// asserts rather than returning a code; callers wrap it in ASSERT_NO_FATAL_FAILURE.
void RunOnDevice(const Fixture& fixture, std::vector<float>& y_out,
                 Dsv4MoePrefillCubeTilingData& tiling_out, double& seconds) {
    const Shape&      shape   = fixture.shape;
    const std::size_t x_bytes = fixture.x.size() * sizeof(float);

    DeviceBuffer x_dev, gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale;
    DeviceBuffer route_dev, y_dev, workspace;

    ASSERT_TRUE(x_dev.Allocate(x_bytes)) << "aclrtMalloc(x): " << AclTail();
    ASSERT_TRUE(gate_packed.Allocate(fixture.gate_weights.packed.size())) << AclTail();
    ASSERT_TRUE(gate_scale.Allocate(fixture.gate_weights.scales.size())) << AclTail();
    ASSERT_TRUE(up_packed.Allocate(fixture.up_weights.packed.size())) << AclTail();
    ASSERT_TRUE(up_scale.Allocate(fixture.up_weights.scales.size())) << AclTail();
    ASSERT_TRUE(down_packed.Allocate(fixture.down_weights.packed.size())) << AclTail();
    ASSERT_TRUE(down_scale.Allocate(fixture.down_weights.scales.size())) << AclTail();
    ASSERT_TRUE(route_dev.Allocate(fixture.route.size() * sizeof(std::int32_t))) << AclTail();
    ASSERT_TRUE(y_dev.Allocate(x_bytes)) << AclTail();
    ASSERT_TRUE(workspace.Allocate(kPrefillCubePipelineWorkspaceBytes)) << AclTail();

    ASSERT_TRUE(x_dev.Upload(fixture.x.data(), x_bytes)) << AclTail();
    ASSERT_TRUE(gate_packed.Upload(fixture.gate_weights.packed.data(), gate_packed.bytes())) << AclTail();
    ASSERT_TRUE(gate_scale.Upload(fixture.gate_weights.scales.data(), gate_scale.bytes())) << AclTail();
    ASSERT_TRUE(up_packed.Upload(fixture.up_weights.packed.data(), up_packed.bytes())) << AclTail();
    ASSERT_TRUE(up_scale.Upload(fixture.up_weights.scales.data(), up_scale.bytes())) << AclTail();
    ASSERT_TRUE(down_packed.Upload(fixture.down_weights.packed.data(), down_packed.bytes())) << AclTail();
    ASSERT_TRUE(down_scale.Upload(fixture.down_weights.scales.data(), down_scale.bytes())) << AclTail();
    ASSERT_TRUE(route_dev.Upload(fixture.route.data(), route_dev.bytes())) << AclTail();

    // A canary the kernel must overwrite: a row left at the pattern is a tile the schedule skipped.
    const std::vector<float> canary(fixture.x.size(), -1.0e30F);
    ASSERT_TRUE(y_dev.Upload(canary.data(), x_bytes)) << AclTail();

    const auto   started = std::chrono::steady_clock::now();
    const Status status  = LaunchDsv4MoePrefillCubePipeline(
        static_cast<const float*>(x_dev.get()),
        static_cast<const Mxfp4Pair*>(gate_packed.get()),
        static_cast<const E8M0*>(gate_scale.get()),
        static_cast<const Mxfp4Pair*>(up_packed.get()),
        static_cast<const E8M0*>(up_scale.get()),
        static_cast<const Mxfp4Pair*>(down_packed.get()),
        static_cast<const E8M0*>(down_scale.get()),
        static_cast<const std::int32_t*>(route_dev.get()),
        shape.hidden, shape.inter, shape.experts, shape.tokens, shape.aiv_count,
        static_cast<float*>(y_dev.get()), workspace.get(), &tiling_out, g_stream);
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    ASSERT_EQ(status, Status::success) << LaunchTail(status);

    y_out.assign(fixture.x.size(), 0.0F);
    ASSERT_TRUE(y_dev.Download(y_out.data(), x_bytes)) << AclTail();

    for (std::size_t index = 0; index < y_out.size(); ++index) {
        ASSERT_NE(y_out[index], -1.0e30F)
            << "output element " << index << " (row " << index / shape.hidden << ", col "
            << index % shape.hidden << ") still holds the canary: the kernel never wrote it";
    }
}

void AssertGoldenIsNonDegenerate(const Fixture& fixture) {
    const Spread spread = MeasureSpread(fixture.golden);
    ASSERT_GT(spread.l2, 1.0e-3) << "the golden is all but zero; it cannot discriminate anything";
    ASSERT_GT(spread.variance, 1.0e-6)
        << "the golden is constant across every element; it cannot discriminate anything";
}

void VerifyAgainstGolden(const Fixture& fixture, const std::vector<float>& y) {
    const Mismatch diff = Compare(y, fixture.golden, fixture.shape.tokens, fixture.shape.hidden);

    std::printf("    diff: max abs %.6e  max rel %.6e  cosine %.8f\n", diff.max_abs, diff.max_rel,
                diff.cosine);
    std::printf("    worst element: row %u col %u  device %+.8f  golden %+.8f\n", diff.worst_row,
                diff.worst_col, diff.worst_got, diff.worst_exp);
    std::fflush(stdout);

    EXPECT_TRUE(diff.finite) << "the device produced a non-finite element";
    EXPECT_TRUE(diff.within) << "element [" << diff.worst_row << ", " << diff.worst_col
                             << "] device " << diff.worst_got << " vs golden " << diff.worst_exp
                             << " exceeds abs " << kAbsTol << " + rel " << kRelTol;
    EXPECT_GE(diff.cosine, kCosineFloor)
        << "direction moved, which rounding does not do: a structural defect";
}

class Dsv4MoePrefillCubePipeline : public ::testing::Test {
protected:
    void RunCase(const Shape& shape, bool verbose) {
        const Fixture fixture = BuildFixture(shape);
        ASSERT_NO_FATAL_FAILURE(AssertGoldenIsNonDegenerate(fixture));

        std::printf("    shape: hidden=%u inter=%u experts=%u tokens=%u tiles=%u aiv_hint=%u fill=%s\n",
                    shape.hidden, shape.inter, shape.experts, shape.tokens,
                    (shape.tokens + kPrefillCubeMTile - 1u) / kPrefillCubeMTile, shape.aiv_count,
                    shape.fill == Fill::kRamp ? "ramp" : "random");
        std::printf("    phase 1 contraction steps=%u   phase 2 contraction steps=%u\n",
                    shape.hidden / kPrefillCubeKSub, shape.inter / kPrefillCubeRSub2);
        if (verbose) {
            DumpRampRealisation(fixture);
        }
        std::fflush(stdout);

        std::vector<float>           y;
        Dsv4MoePrefillCubeTilingData tiling{};
        double                       seconds = 0.0;
        ASSERT_NO_FATAL_FAILURE(RunOnDevice(fixture, y, tiling, seconds));

        DumpTiling(tiling);
        std::printf("    launch + sync: %.6f s\n", seconds);
        if (verbose) {
            DumpPerRowError(y, fixture.golden, shape.tokens, shape.hidden);
        }

        VerifyAgainstGolden(fixture, y);
    }
};

// One tile, one AI Core group, ramp fixture: the cold-start rendezvous and the phase-1 prologue
// burst with nothing yet in flight behind them.
TEST_F(Dsv4MoePrefillCubePipeline, Tile1ColdStartAndPrologue) {
    RunCase(Shape{256, 256, 2, 16, 1, Fill::kRamp}, /*verbose=*/true);
}

// Two tiles over two contraction steps per phase, ramp fixture: every ring slot is filled, pushed,
// consumed and flipped exactly once.
TEST_F(Dsv4MoePrefillCubePipeline, Tile2BufferFlipPingPong) {
    RunCase(Shape{256, 512, 2, 32, 2, Fill::kRamp}, /*verbose=*/true);
}

// Four tiles over four phase-1 contraction steps: the first grid that reuses a ring slot, so the
// AIC's StepFree release and the AIVs' deferred MTE3_V drain are both exercised, then drained.
TEST_F(Dsv4MoePrefillCubePipeline, Tile4SteadyStateAndDrain) {
    RunCase(Shape{512, 512, 4, 64, 2, Fill::kRandom}, /*verbose=*/false);
}

// Every token over every expert, with a ragged tail tile, compared element by element.
TEST_F(Dsv4MoePrefillCubePipeline, FullReferenceParity) {
    RunCase(Shape{512, 512, 4, 100, 4, Fill::kRandom}, /*verbose=*/false);
}

class Runtime {
public:
    bool Open(std::string* why) {
        const aclError init_rc = aclInit(nullptr);
        if (init_rc != ACL_SUCCESS) {
            *why = "aclInit failed (no CANN runtime available), aclError=" + std::to_string(init_rc);
            return false;
        }
        initialized_ = true;

        std::uint32_t  count    = 0;
        const aclError count_rc = aclrtGetDeviceCount(&count);
        if (count_rc != ACL_SUCCESS) {
            *why = "aclrtGetDeviceCount failed, aclError=" + std::to_string(count_rc);
            return false;
        }
        if (count == 0) {
            *why = "aclrtGetDeviceCount reported 0 devices";
            return false;
        }

        const aclError set_rc = aclrtSetDevice(0);
        if (set_rc != ACL_SUCCESS) {
            *why = "aclrtSetDevice(0) failed on a host that reports " + std::to_string(count) +
                   " device(s), aclError=" + std::to_string(set_rc) + ": " + AclTail();
            bring_up_failed_ = true;
            return false;
        }
        device_ = true;

        const aclError stream_rc = aclrtCreateStream(&stream_);
        if (stream_rc != ACL_SUCCESS) {
            *why = "aclrtCreateStream failed, aclError=" + std::to_string(stream_rc) + ": " + AclTail();
            bring_up_failed_ = true;
            return false;
        }
        return true;
    }

    ~Runtime() {
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_) {
            (void)aclrtResetDevice(0);
        }
        if (initialized_) {
            (void)aclFinalize();
        }
    }

    aclrtStream stream() const noexcept { return stream_; }
    bool        bring_up_failed() const noexcept { return bring_up_failed_; }

private:
    bool        initialized_     = false;
    bool        device_          = false;
    bool        bring_up_failed_ = false;
    aclrtStream stream_          = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    // Scanned before InitGoogleTest consumes it, so the case names stay discoverable on a host
    // with no NPU. argv is scanned directly rather than read back through a gtest flag accessor,
    // whose spelling moved between gtest 1.12 and 1.13.
    bool list_only = false;
    for (int arg = 1; arg < argc; ++arg) {
        if (std::string(argv[arg]).rfind("--gtest_list_tests", 0) == 0) {
            list_only = true;
        }
    }

    ::testing::InitGoogleTest(&argc, argv);

    if (list_only) {
        return RUN_ALL_TESTS();
    }

    Runtime     runtime;
    std::string why;
    if (!runtime.Open(&why)) {
        if (runtime.bring_up_failed()) {
            // A device exists and bring-up failed anyway. That is a defect on this machine, not an
            // environment without hardware, so it must not be collapsed into the skip.
            std::printf("[ FAIL ] device present but bring-up failed: %s\n", why.c_str());
            return 1;
        }
        std::printf("[ SKIP ALL: Physical Ascend NPU not detected ] %s\n", why.c_str());
        return 0;
    }
    g_stream = runtime.stream();

    const char* seed_env = std::getenv("DSV4_PREFILL_CUBE_SEED");
    g_seed = (seed_env != nullptr)
                 ? std::strtoull(seed_env, nullptr, 0)
                 : static_cast<std::uint64_t>(
                       std::chrono::steady_clock::now().time_since_epoch().count());

    std::printf("dsv4_moe_prefill_cube_pipeline standalone NPU suite\n");
    std::printf("  seed 0x%016llX  (override with DSV4_PREFILL_CUBE_SEED; the Tile1/Tile2 ramp\n"
                "                    fixtures are deterministic and ignore it)\n",
                static_cast<unsigned long long>(g_seed));
    std::printf("  m_tile %u  r_sub %u  k_sub %u  h_sub %u  r_sub2 %u  swiglu limit %.3f\n",
                kPrefillCubeMTile, kPrefillCubeRSub, kPrefillCubeKSub, kPrefillCubeHSub,
                kPrefillCubeRSub2, static_cast<double>(kPrefillCubeSwigluLimit));
    std::printf("  tolerance: abs %.3e  rel %.3e  cosine floor %.4f\n",
                static_cast<double>(kAbsTol), static_cast<double>(kRelTol), kCosineFloor);
    std::fflush(stdout);

    return RUN_ALL_TESTS();
}
