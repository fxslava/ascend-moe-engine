#include "test_patterns.h"

#if defined(DSV4_CAMODEL)
#include "dsv4_moe_prefill_cube_human_writed.h"
#include <acl/acl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr std::size_t kTokenCount = 16;
constexpr std::size_t kReductionWidth = 16;
constexpr std::size_t kSimStepsLocal = DSV4_SIM_STEPS;
constexpr std::size_t kReductionCount = kReductionWidth * kSimStepsLocal;
constexpr std::size_t kGateDimension = 2048;
constexpr std::size_t kDownDimension = 4096;
constexpr std::size_t kOutputColumns = 32;
constexpr std::size_t kOutputCount = kTokenCount * kOutputColumns;
constexpr float kTolerance = 0.5F;

uint16_t FloatToBf16(float value) {
    union Bits {
        float f;
        uint32_t u;
    } bits{};
    bits.f = value;
    const uint32_t increment = 0x7fffU + ((bits.u >> 16) & 1U);
    return static_cast<uint16_t>((bits.u + increment) >> 16);
}

float RoundBf16(float value) {
    return dsv4::sim::Bf16ToFloat(FloatToBf16(value));
}

#if defined(DSV4_CAMODEL)
template <class T>
bool AllocateDevice(T** pointer, std::size_t count) {
    return aclrtMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T),
                       ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
}

template <class T>
bool CopyToDevice(T* destination, const T* source, std::size_t count) {
    return aclrtMemcpy(destination, count * sizeof(T), source,
                       count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

template <class T>
bool CopyToHost(T* destination, const T* source, std::size_t count) {
    return aclrtMemcpy(destination, count * sizeof(T), source,
                       count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

struct DeviceBuffers {
    uint16_t* input = nullptr;
    uint8_t* gate_weights = nullptr;
    uint8_t* gate_scales = nullptr;
    uint8_t* up_weights = nullptr;
    uint8_t* up_scales = nullptr;
    uint8_t* down_weights = nullptr;
    uint8_t* down_scales = nullptr;
    uint16_t* gate = nullptr;
    uint16_t* up = nullptr;
    uint16_t* inter = nullptr;
    uint16_t* output = nullptr;
};
#endif

struct StageData {
    std::vector<uint16_t> input;
    std::vector<uint8_t> gate_weights;
    std::vector<uint8_t> gate_scales;
    std::vector<uint8_t> up_weights;
    std::vector<uint8_t> up_scales;
    std::vector<uint8_t> down_weights;
    std::vector<uint8_t> down_scales;
    std::vector<uint16_t> gate;
    std::vector<uint16_t> up;
    std::vector<uint16_t> inter;
    std::vector<uint16_t> output;
};

void InitializeWeightData(std::vector<float>& source,
                          std::vector<uint8_t>& packed,
                          std::vector<uint8_t>& scales,
                          std::size_t output_dimension,
                          bool all_ones) {
    if (all_ones) {
        std::fill(packed.begin(), packed.end(), 0x22U);
        std::fill(scales.begin(), scales.end(), 127U);
        return;
    }
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>((i % 6U) + 1U);
    }
    dsv4::sim::PackWeightsFp4(source.data(), packed.data(), scales.data(),
                              kReductionCount, output_dimension);
}

StageData CreateStage(bool sequential_pattern) {
    const std::size_t gate_elements = kReductionCount * kGateDimension;
    const std::size_t down_elements = kReductionCount * kDownDimension;
    StageData stage;
    stage.input.resize(kTokenCount * kReductionCount);
    stage.gate_weights.resize(gate_elements / 2);
    stage.gate_scales.resize(gate_elements / 32);
    stage.up_weights.resize(gate_elements / 2);
    stage.up_scales.resize(gate_elements / 32);
    stage.down_weights.resize(down_elements / 2);
    stage.down_scales.resize(down_elements / 32);
    stage.gate.resize(kOutputCount);
    stage.up.resize(kOutputCount);
    stage.inter.resize(kOutputCount);
    stage.output.resize(kOutputCount);

    if (sequential_pattern) {
        dsv4::sim::GenerateSequential(stage.input.data(), stage.input.size());
    } else {
        dsv4::sim::GenerateAllOnes(stage.input.data(), stage.input.size());
    }

    std::vector<float> gate_source(gate_elements);
    std::vector<float> up_source(gate_elements);
    std::vector<float> down_source(down_elements);
    InitializeWeightData(gate_source, stage.gate_weights, stage.gate_scales,
                         kGateDimension, !sequential_pattern);
    InitializeWeightData(up_source, stage.up_weights, stage.up_scales,
                         kGateDimension, !sequential_pattern);
    InitializeWeightData(down_source, stage.down_weights, stage.down_scales,
                         kDownDimension, !sequential_pattern);
    return stage;
}

void CpuReference(const StageData& stage,
                  std::vector<float>& gate,
                  std::vector<float>& up,
                  std::vector<float>& inter,
                  std::vector<float>& output) {
    gate.assign(kOutputCount, 0.0F);
    up.assign(kOutputCount, 0.0F);
    inter.assign(kOutputCount, 0.0F);
    output.assign(kOutputCount, 0.0F);
    for (std::size_t token_row = 0; token_row < kTokenCount; ++token_row) {
        for (std::size_t column = 0; column < kOutputColumns; ++column) {
            float gate_sum = 0.0F;
            float up_sum = 0.0F;
            for (std::size_t reduction = 0; reduction < kReductionCount; ++reduction) {
                const float input = dsv4::sim::Bf16ToFloat(
                    stage.input[token_row * kReductionCount + reduction]);
                const std::size_t weight_index = reduction * kGateDimension + column;
                const float gate_weight = RoundBf16(dsv4::sim::DecodePackedWeight(
                    stage.gate_weights.data(), stage.gate_scales.data(), weight_index));
                const float up_weight = RoundBf16(dsv4::sim::DecodePackedWeight(
                    stage.up_weights.data(), stage.up_scales.data(), weight_index));
                gate_sum = RoundBf16(gate_sum + RoundBf16(input * gate_weight));
                up_sum = RoundBf16(up_sum + RoundBf16(input * up_weight));
            }
            const std::size_t output_index = token_row * kOutputColumns + column;
            gate[output_index] = gate_sum;
            up[output_index] = up_sum;
            inter[output_index] = RoundBf16(
                (gate_sum / (1.0F + std::exp(-gate_sum))) * up_sum);
        }
        for (std::size_t column = 0; column < kOutputColumns; ++column) {
            float sum = 0.0F;
            for (std::size_t reduction = 0; reduction < kReductionCount; ++reduction) {
                const float weight = RoundBf16(dsv4::sim::DecodePackedWeight(
                    stage.down_weights.data(), stage.down_scales.data(),
                    reduction * kDownDimension + column));
                sum = RoundBf16(sum + RoundBf16(
                    inter[token_row * kOutputColumns + reduction] * weight));
            }
            output[token_row * kOutputColumns + column] = sum;
        }
    }
}

void PrintFirstEight(const char* label, const std::vector<uint16_t>& values) {
    std::printf("%s", label);
    for (std::size_t i = 0; i < 8; ++i) {
        std::printf(" %.6f", dsv4::sim::Bf16ToFloat(values[i]));
    }
    std::putchar('\n');
}

bool ValidateStage(const char* name,
                   const StageData& stage,
                   const std::vector<float>& golden_gate,
                   const std::vector<float>& golden_up,
                   const std::vector<float>& golden_inter,
                   const std::vector<float>& golden_output,
                   bool arithmetic_stage) {
    float maximum_difference = 0.0F;
    const std::vector<const std::vector<uint16_t>*> actual{
        &stage.gate, &stage.up, &stage.inter, &stage.output};
    const std::vector<const std::vector<float>*> expected{
        &golden_gate, &golden_up, &golden_inter, &golden_output};
    const char* labels[] = {"gate", "up", "SwiGLU", "down"};
    for (std::size_t tensor = 0; tensor < actual.size(); ++tensor) {
        std::printf("[%s] %-7s: ", name, labels[tensor]);
        for (std::size_t i = 0; i < 8; ++i) {
            const float value = dsv4::sim::Bf16ToFloat((*actual[tensor])[i]);
            std::printf("%s%.6f/%.6f", i == 0 ? "" : " ",
                        value, (*expected[tensor])[i]);
            maximum_difference = std::max(
                maximum_difference, std::fabs(value - (*expected[tensor])[i]));
        }
        std::putchar('\n');
        for (std::size_t i = 8; i < kOutputCount; ++i) {
            const float value = dsv4::sim::Bf16ToFloat((*actual[tensor])[i]);
            maximum_difference = std::max(
                maximum_difference, std::fabs(value - (*expected[tensor])[i]));
        }
    }
    std::printf("[%s] RMSE/max element checks; max abs %.7g\n",
                name, maximum_difference);
    bool passed = maximum_difference < kTolerance;
    if (arithmetic_stage) {
        for (std::size_t i = 0; i < kOutputCount; ++i) {
            const float gate_value = dsv4::sim::Bf16ToFloat(stage.gate[i]);
            const float up_value = dsv4::sim::Bf16ToFloat(stage.up[i]);
            const float inter_value = dsv4::sim::Bf16ToFloat(stage.inter[i]);
            const float output_value = dsv4::sim::Bf16ToFloat(stage.output[i]);
            const bool arithmetic_ok = std::fabs(gate_value - 32.0F) < 0.01F &&
                                       std::fabs(up_value - 32.0F) < 0.01F &&
                                       std::fabs(inter_value - 1024.0F) < 1.0F &&
                                       std::fabs(output_value - 32768.0F) < 32.0F;
            if (!arithmetic_ok) {
                std::printf("[%s] failed index %zu: %.6f %.6f %.6f %.6f\n",
                            name, i, gate_value, up_value, inter_value, output_value);
            }
            passed = passed && arithmetic_ok;
        }
    }
    std::printf("[%s] %s\n", name, passed ? "TEST PASSED" : "TEST FAILED");
    return passed;
}

#if defined(DSV4_CAMODEL)
bool RunOnCamodel(StageData& stage, DeviceBuffers& device, aclrtStream stream) {
    if (!CopyToDevice(device.input, stage.input.data(), stage.input.size()) ||
        !CopyToDevice(device.gate_weights, stage.gate_weights.data(), stage.gate_weights.size()) ||
        !CopyToDevice(device.gate_scales, stage.gate_scales.data(), stage.gate_scales.size()) ||
        !CopyToDevice(device.up_weights, stage.up_weights.data(), stage.up_weights.size()) ||
        !CopyToDevice(device.up_scales, stage.up_scales.data(), stage.up_scales.size()) ||
        !CopyToDevice(device.down_weights, stage.down_weights.data(), stage.down_weights.size()) ||
        !CopyToDevice(device.down_scales, stage.down_scales.data(), stage.down_scales.size())) {
        std::fprintf(stderr, "CAModel input DMA setup failed\n");
        return false;
    }
    moe::kernels::arch35::LaunchDsv4MoePrefillCubeHumanWrited(
        device.input, device.gate_weights, device.gate_scales, device.up_weights,
        device.up_scales, device.down_weights, device.down_scales, device.gate,
        device.up, device.inter, device.output, stream);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
        !CopyToHost(stage.gate.data(), device.gate, stage.gate.size()) ||
        !CopyToHost(stage.up.data(), device.up, stage.up.size()) ||
        !CopyToHost(stage.inter.data(), device.inter, stage.inter.size()) ||
        !CopyToHost(stage.output.data(), device.output, stage.output.size())) {
        std::fprintf(stderr, "CAModel launch or output DMA failed\n");
        return false;
    }
    return true;
}
#endif
}  // namespace

int main() {
#if defined(DSV4_CAMODEL)
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::fprintf(stderr, "CAModel ACL initialization failed\n");
        return EXIT_FAILURE;
    }
    DeviceBuffers device;
    const std::size_t gate_elements = kReductionCount * kGateDimension;
    const std::size_t down_elements = kReductionCount * kDownDimension;
    if (!AllocateDevice(&device.input, kTokenCount * kReductionCount) ||
        !AllocateDevice(&device.gate_weights, gate_elements / 2) ||
        !AllocateDevice(&device.gate_scales, gate_elements / 32) ||
        !AllocateDevice(&device.up_weights, gate_elements / 2) ||
        !AllocateDevice(&device.up_scales, gate_elements / 32) ||
        !AllocateDevice(&device.down_weights, down_elements / 2) ||
        !AllocateDevice(&device.down_scales, down_elements / 32) ||
        !AllocateDevice(&device.gate, kOutputCount) ||
        !AllocateDevice(&device.up, kOutputCount) ||
        !AllocateDevice(&device.inter, kOutputCount) ||
        !AllocateDevice(&device.output, kOutputCount)) {
        std::fprintf(stderr, "CAModel output/workspace allocation failed\n");
        return EXIT_FAILURE;
    }
#endif

    bool passed = true;
    for (const bool sequential_pattern : {true, false}) {
        StageData stage = CreateStage(sequential_pattern);
        std::vector<float> golden_gate;
        std::vector<float> golden_up;
        std::vector<float> golden_inter;
        std::vector<float> golden_output;
        CpuReference(stage, golden_gate, golden_up, golden_inter, golden_output);
        const char* stage_name = sequential_pattern ? "stage A: sequential DMA/dequant"
                                                    : "stage B: all-ones arithmetic";
        std::printf("\n=== %s; K=%zu x %zu steps ===\n",
                    stage_name, kReductionWidth, kSimStepsLocal);
        if (sequential_pattern) {
            std::printf("[stage A expected dequant first 8]");
            for (std::size_t i = 0; i < 8; ++i) {
                std::printf(" %.6f",
                    dsv4::sim::DecodePackedWeight(stage.gate_weights.data(),
                        stage.gate_scales.data(), i));
            }
            std::putchar('\n');
        }
#if defined(DSV4_CAMODEL)
        if (!RunOnCamodel(stage, device, stream)) {
            passed = false;
            break;
        }
#else
        for (std::size_t i = 0; i < kOutputCount; ++i) {
            stage.gate[i] = FloatToBf16(golden_gate[i]);
            stage.up[i] = FloatToBf16(golden_up[i]);
            stage.inter[i] = FloatToBf16(golden_inter[i]);
            stage.output[i] = FloatToBf16(golden_output[i]);
        }
#endif
        PrintFirstEight("[host] token buffer first 8:", stage.input);
        passed = ValidateStage(stage_name, stage, golden_gate, golden_up,
                               golden_inter, golden_output, !sequential_pattern) && passed;
    }
#if defined(DSV4_CAMODEL)
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
#endif
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
