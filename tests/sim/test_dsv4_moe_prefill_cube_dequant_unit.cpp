#include "dsv4_moe_block_sim_common.hpp"
#include "moe/emu/cpu_kernels.h"

extern "C" void LaunchDsv4MoePrefillCubeDequantUnit(
    void* packed, void* scales, void* output, void* shape, aclrtStream stream);

int main() {
  try {
    Runtime runtime;
    for (const auto& [rows, cols] : {
             std::pair<std::uint32_t, std::uint32_t>{128, 128},
             std::pair<std::uint32_t, std::uint32_t>{64, 256}}) {
      std::vector<std::uint8_t> packed(rows * cols / 2);
      std::vector<std::uint8_t> scales(rows * cols / 32);
      std::vector<std::uint8_t> padded_scales(rows * 32, 0);
      for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; col += 2) {
          const std::uint8_t low = static_cast<std::uint8_t>((row + col) % 16);
          const std::uint8_t high = static_cast<std::uint8_t>((row * 3 + col / 2 + 7) % 16);
          packed[row * (cols / 2) + col / 2] =
              static_cast<std::uint8_t>(low | (high << 4));
        }
        for (std::uint32_t group = 0; group < cols / 32; ++group) {
          scales[row * (cols / 32) + group] =
              static_cast<std::uint8_t>(126 + (row + group) % 3);
          padded_scales[row * 32 + group] = scales[row * (cols / 32) + group];
        }
      }

      DeviceBuffer packed_dev(packed.size());
      DeviceBuffer scales_dev(padded_scales.size());
      DeviceBuffer output_dev(rows * cols * sizeof(std::uint16_t));
      DeviceBuffer shape_dev(2 * sizeof(std::uint32_t));
      CopyToDevice(packed_dev, packed);
      CopyToDevice(scales_dev, padded_scales);
      CopyToDevice(shape_dev, std::vector<std::uint32_t>{rows, cols});

      LaunchDsv4MoePrefillCubeDequantUnit(
          packed_dev.get(), scales_dev.get(), output_dev.get(), shape_dev.get(),
          runtime.stream());
      RequireAcl(aclrtSynchronizeStream(runtime.stream()), "dequant unit synchronize");

      std::vector<std::uint16_t> actual(rows * cols);
      CopyFromDevice(actual, output_dev);
      const moe::emu::Mxfp4Matrix matrix{packed.data(), scales.data(),
                                          static_cast<int>(rows), static_cast<int>(cols)};
      for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t col = 0; col < cols; ++col) {
          const auto want = moe::emu::FloatToBf16(
              moe::emu::mxfp4_dequant_element_bf16_cpu(matrix, row, col));
          const auto got = actual[row * cols + col];
          if (got != want) {
            Fail("dequant mismatch at (" + std::to_string(row) + ", " +
                 std::to_string(col) + "): got " + std::to_string(got) +
                 ", want " + std::to_string(want));
          }
        }
      }
      std::printf("[PASS] vector dequant %ux%u: all %u BF16 values match CPU oracle\n",
                  rows, cols, rows * cols);
    }
    runtime.Close();
    return 0;
  } catch (const std::exception& error) {
    std::printf("[FAIL] %s\n", error.what());
    return 1;
  }
}
