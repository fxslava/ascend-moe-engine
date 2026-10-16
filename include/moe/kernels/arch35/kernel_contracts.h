/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// DeepSeek V4 Flash decode kernel contracts, Ascend 950PR / arch35.
//
// This header is the ONE definition of the host/device ABI for KER-01..KER-05
// as specified in docs/kernel_contracts_spec.md. It is included by both the
// host tilers (op_host/) and the device kernels (op_kernel/), so every field
// offset is shared by construction rather than by review.
//
// RULES THIS HEADER ENFORCES (see the spec's "Universal ABI" section)
//   * Every TilingData is a fixed-width, standard-layout, trivially-copyable
//     POD with no pointers. Pointer addresses travel as kernel GM_ADDR
//     arguments, never inside tiling.
//   * Every field is 32-bit, so no host/device pointer-width padding can
//     appear and the static_asserts below pin the exact byte sizes.
//   * All `reserved` fields are zero and an unknown abi_version is rejected.
//
// Strides are in ELEMENTS of the pointer's type, never bytes.

#ifndef MOE_KERNELS_ARCH35_KERNEL_CONTRACTS_H
#define MOE_KERNELS_ARCH35_KERNEL_CONTRACTS_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace moe::kernels::arch35 {

// --- storage types ---------------------------------------------------------
using Bf16 = std::uint16_t;      // storage bits, round-to-nearest-even on store
using Fp8E4M3 = std::uint8_t;    // storage bits
using Mxfp4Pair = std::uint8_t;  // low nibble = even logical column
using E8M0 = std::uint8_t;       // one exponent scale code

enum class Status : std::uint32_t {
  success = 0,
  invalid_argument = 1,
  unsupported_shape = 2,
  launch_error = 3,
};

// --- fixed token geometry (spec: "Fixed token geometry") -------------------
constexpr std::uint32_t kAbiVersion = 1U;

constexpr std::uint32_t kHcStreams = 4U;         // N
constexpr std::uint32_t kHiddenSize = 4096U;     // H
constexpr std::uint32_t kHcFlatWidth = kHcStreams * kHiddenSize;  // 16384
constexpr std::uint32_t kHcProjectionRows = kHcStreams * kHcStreams + 2U * kHcStreams;  // 24
constexpr std::uint32_t kHcSinkhornIters = 20U;
constexpr float kHcEps = 1e-6F;
constexpr float kRmsNormEps = 1e-6F;

constexpr std::uint32_t kCompressorHeadDim = 512U;   // D
constexpr std::uint32_t kCompressorCoff = 2U;        // reject 1
constexpr std::uint32_t kCompressorRatio = 4U;
constexpr std::uint32_t kCompressorRows = kCompressorCoff * kCompressorRatio;  // 8
constexpr std::uint32_t kRopeHeadDim = 64U;
constexpr std::uint32_t kCompressorStateFloats = 12288U;         // 49152 B
constexpr std::uint32_t kCompressorStateMetaWords = 4U;          // 16 B

constexpr std::uint32_t kIndexerHeads = 64U;
constexpr std::uint32_t kIndexerHeadDim = 128U;
constexpr std::uint32_t kIndexerTopK = 512U;

constexpr std::uint32_t kMxfp4GroupSize = 32U;
constexpr std::uint32_t kMxfp4LowNibbleFirst = 1U;  // reject 0

// Rotary layout. 1 = half mode, which is what the checkpoint C4A path uses.
constexpr std::uint32_t kRotaryModeHalf = 1U;

// --- KER-01 ----------------------------------------------------------------
struct MhcPreNormTilingData {
  std::uint32_t abi_version;          // 1
  std::uint32_t block_dim;            // 1 AIV
  std::uint32_t hidden;               // 4096
  std::uint32_t streams;              // 4
  std::uint32_t projection_rows;      // 24
  std::uint32_t projection_tile;      // 4096 FP32 elements
  std::uint32_t projection_chunks;    // 4 per row
  std::uint32_t queue_depth;          // 2
  std::uint32_t sinkhorn_iterations;  // 20
  std::uint32_t x_stream_stride;      // 4096 BF16 elements
  std::uint32_t phi_row_stride;       // 16384 FP32 elements
  std::uint32_t reserved;             // 0
  float hc_eps;                       // 1e-6
  float bundle_norm_eps;              // 1e-6
  float activation_norm_eps;          // 1e-6
  float reserved_fp32;                // 0
};

// --- KER-02 ----------------------------------------------------------------
struct MhcPostTilingData {
  std::uint32_t abi_version;       // 1
  std::uint32_t block_dim;         // 4 AIV, one output stream per block
  std::uint32_t hidden;            // 4096
  std::uint32_t streams;           // 4
  std::uint32_t feature_tile;      // 512 elements, 8 tiles per stream
  std::uint32_t x_stream_stride;   // 4096
  std::uint32_t y_stream_stride;   // 4096
  std::uint32_t b_row_stride;      // 4
};

// --- KER-03 ----------------------------------------------------------------
struct CompressorTilingData {
  std::uint32_t abi_version;         // 1
  std::uint32_t block_dim;           // 1 AIV
  std::uint32_t hidden;              // 4096
  std::uint32_t head_dim;            // 512
  std::uint32_t coff;                // 2, reject 1
  std::uint32_t ratio;               // 4
  std::uint32_t reduction_rows;      // 8
  std::uint32_t rope_head_dim;       // 64
  std::uint32_t rotary_mode;         // 1 = half, not interleave
  std::uint32_t projection_row_tile; // 1 row at a time
  std::uint32_t wkv_row_stride;      // 4096 BF16 elements
  std::uint32_t wgate_row_stride;    // 4096 BF16 elements
  std::uint32_t ape_row_stride;      // 1024 FP32 elements
  std::uint32_t reserved;            // 0
  float norm_eps;                    // 1e-6
  float reserved_fp32;               // 0
};

// --- KER-04 ----------------------------------------------------------------
struct IndexerTilingData {
  std::uint32_t abi_version;       // 1
  std::uint32_t block_dim;         // 1 AIV, deterministic total order
  std::uint32_t heads;             // 64
  std::uint32_t head_dim;          // 128
  std::uint32_t top_k;             // 512
  std::uint32_t query_head_stride; // 128 FP8 elements
  std::uint32_t key_block_stride;  // block_size*128 FP8 elements
  std::uint32_t block_size;        // 128 in v3 semi-E2E
  std::uint32_t seq_len;           // 2048 in v3 semi-E2E
  std::uint32_t block_table_len;   // >= ceil(seq_len/block_size)
  std::uint32_t num_physical_blocks;
  std::uint32_t score_tile;        // 512 FP32 elements
};

// --- KER-05 ----------------------------------------------------------------
struct Mxfp4GemvTilingData {
  std::uint32_t abi_version;       // 1
  std::uint32_t block_dim;         // 32 AIVs; rows assigned in 32-row tiles
  std::uint32_t rows;              // 2048 or 4096
  std::uint32_t cols;              // 4096 or 2048
  std::uint32_t rows_per_block;    // 32
  std::uint32_t group_size;        // 32
  std::uint32_t packed_row_stride; // cols/2 bytes
  std::uint32_t scale_row_stride;  // cols/32 bytes
  std::uint32_t low_nibble_first;  // 1; reject 0
  std::uint32_t reserved;          // 0
};

// --- the ABI pins the spec commits to -------------------------------------
static_assert(sizeof(MhcPreNormTilingData) == 64, "KER-01 tiling ABI is 64 B");
static_assert(sizeof(MhcPostTilingData) == 32, "KER-02 tiling ABI is 32 B");
static_assert(sizeof(CompressorTilingData) == 64, "KER-03 tiling ABI is 64 B");
static_assert(sizeof(IndexerTilingData) == 48, "KER-04 tiling ABI is 48 B");
static_assert(sizeof(Mxfp4GemvTilingData) == 40, "KER-05 tiling ABI is 40 B");

#define MOE_ARCH35_ASSERT_POD(T)                                            \
  static_assert(std::is_standard_layout<T>::value, #T " must be standard layout"); \
  static_assert(std::is_trivially_copyable<T>::value, #T " must be trivially copyable")

MOE_ARCH35_ASSERT_POD(MhcPreNormTilingData);
MOE_ARCH35_ASSERT_POD(MhcPostTilingData);
MOE_ARCH35_ASSERT_POD(CompressorTilingData);
MOE_ARCH35_ASSERT_POD(IndexerTilingData);
MOE_ARCH35_ASSERT_POD(Mxfp4GemvTilingData);
#undef MOE_ARCH35_ASSERT_POD

// All fields are 32-bit, so each struct's alignment must be 4: anything larger
// would mean a field was widened and the device layout silently diverged.
static_assert(alignof(MhcPreNormTilingData) == 4, "KER-01 tiling must be 4-aligned");
static_assert(alignof(MhcPostTilingData) == 4, "KER-02 tiling must be 4-aligned");
static_assert(alignof(CompressorTilingData) == 4, "KER-03 tiling must be 4-aligned");
static_assert(alignof(IndexerTilingData) == 4, "KER-04 tiling must be 4-aligned");
static_assert(alignof(Mxfp4GemvTilingData) == 4, "KER-05 tiling must be 4-aligned");

// --- host-side tiling builders --------------------------------------------
//
// These validate the declared geometry and fill the POD. They are pure host
// functions with no CANN dependency so they are unit-testable without a
// device; the Launch* wrappers in op_host/ call them before enqueueing.

Status BuildMhcPreNormTiling(std::uint32_t x_stream_stride,
                             std::uint32_t phi_row_stride,
                             MhcPreNormTilingData* out);

Status BuildMhcPostTiling(std::uint32_t x_stream_stride,
                          std::uint32_t y_stream_stride,
                          std::uint32_t b_row_stride,
                          MhcPostTilingData* out);

Status BuildCompressorTiling(std::uint32_t wkv_row_stride,
                             std::uint32_t wgate_row_stride,
                             std::uint32_t ape_row_stride,
                             std::uint32_t coff,
                             std::uint32_t rotary_mode,
                             CompressorTilingData* out);

Status BuildIndexerTiling(std::uint32_t query_head_stride,
                          std::uint32_t key_block_stride,
                          std::uint32_t block_size,
                          std::uint32_t seq_len,
                          std::uint32_t block_table_len,
                          std::uint32_t num_physical_blocks,
                          IndexerTilingData* out);

Status BuildMxfp4GemvTiling(std::uint32_t rows,
                            std::uint32_t cols,
                            std::uint32_t packed_row_stride,
                            std::uint32_t scale_row_stride,
                            std::uint32_t low_nibble_first,
                            Mxfp4GemvTilingData* out);

// --- GM workspace contracts ----------------------------------------------
//
// Spec: KER-01/02/03/05 require ZERO GM scratch. KER-04 needs one FP32 score
// per key, 32-byte aligned, which must not alias its inputs or output.

constexpr std::size_t kMhcPreNormWorkspaceBytes = 0U;
constexpr std::size_t kMhcPostWorkspaceBytes = 0U;
constexpr std::size_t kCompressorWorkspaceBytes = 0U;
constexpr std::size_t kMxfp4GemvWorkspaceBytes = 0U;

constexpr std::size_t IndexerWorkspaceBytes(std::uint32_t seq_len) {
  return ((static_cast<std::size_t>(seq_len) * 4U) + 31U) & ~static_cast<std::size_t>(31U);
}

// --- static UB budgets, bytes (see op_kernel/arch35_ub_layout.h) ----------
//
// These are the byte-exact static layouts the kernels allocate, measured by
// the generator that emits the layout header. Every one is below the 950PR
// usable ceiling of 216 KiB = 221,184 B (256 KiB physical, less the 32 KiB
// SIMT DataCache floor and the 8 KiB compiler reserve).
constexpr std::size_t kArch35UsableUbBytes = 216U * 1024U;

}  // namespace moe::kernels::arch35

#endif  // MOE_KERNELS_ARCH35_KERNEL_CONTRACTS_H
