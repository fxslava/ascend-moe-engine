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

// Host tilers for the arch35 Stage-1 kernels, KER-01..KER-05.
//
// These fill the versioned tiling PODs and reject a geometry the kernels do
// not implement, BEFORE anything is enqueued. They are deliberately free of
// any CANN dependency so the whole validation surface is unit-testable on a
// host with no device and no driver: the Launch* wrappers call into these and
// then enqueue.
//
// Everything here is a rejection, never a silent clamp. The ABI says "no
// silently clipped shape is permitted", and a clamp in a tiler is exactly the
// kind of defect that turns into a wrong answer much later.

#include "moe/kernels/arch35/kernel_contracts.h"

#include "../op_kernel/arch35_ub_layout.h"

namespace moe::kernels::arch35 {
namespace {

// v1 tile constants that are part of the declared ABI, not free parameters.
constexpr std::uint32_t kPreNormProjectionTile = 4096U;
constexpr std::uint32_t kPreNormQueueDepth = 2U;
constexpr std::uint32_t kPostFeatureTile = 512U;
constexpr std::uint32_t kPostBlockDim = 4U;   // one output stream per AIV
constexpr std::uint32_t kGemvBlockDim = 32U;  // 32 AIVs, 32-row tiles
constexpr std::uint32_t kGemvRowsPerBlock = 32U;
constexpr std::uint32_t kIndexerScoreTile = 512U;

// KER-04's selection buffers are statically sized for the v3 semi-E2E
// geometry, and `Sort` consumes 32 elements per repeat.
constexpr std::uint32_t kIndexerMaxSeqLen = 2048U;
constexpr std::uint32_t kIndexerSortGroup = 32U;

}  // namespace

Status BuildMhcPreNormTiling(std::uint32_t x_stream_stride,
                             std::uint32_t phi_row_stride,
                             MhcPreNormTilingData* out) {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // Strides are minima, not free shapes: a smaller stride would make the
  // flattened view of x overlap itself.
  if (x_stream_stride < kHiddenSize || phi_row_stride < kHcFlatWidth) {
    return Status::unsupported_shape;
  }
  *out = MhcPreNormTilingData{};
  out->abi_version = kAbiVersion;
  out->block_dim = 1U;
  out->hidden = kHiddenSize;
  out->streams = kHcStreams;
  out->projection_rows = kHcProjectionRows;
  out->projection_tile = kPreNormProjectionTile;
  out->projection_chunks = kHcFlatWidth / kPreNormProjectionTile;
  out->queue_depth = kPreNormQueueDepth;
  out->sinkhorn_iterations = kHcSinkhornIters;
  out->x_stream_stride = x_stream_stride;
  out->phi_row_stride = phi_row_stride;
  out->reserved = 0U;
  out->hc_eps = kHcEps;
  out->bundle_norm_eps = kRmsNormEps;
  out->activation_norm_eps = kRmsNormEps;
  out->reserved_fp32 = 0.0F;
  return Status::success;
}

Status BuildMhcPostTiling(std::uint32_t x_stream_stride,
                          std::uint32_t y_stream_stride,
                          std::uint32_t b_row_stride,
                          MhcPostTilingData* out) {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (x_stream_stride < kHiddenSize || y_stream_stride < kHiddenSize ||
      b_row_stride < kHcStreams) {
    return Status::unsupported_shape;
  }
  *out = MhcPostTilingData{};
  out->abi_version = kAbiVersion;
  out->block_dim = kPostBlockDim;
  out->hidden = kHiddenSize;
  out->streams = kHcStreams;
  out->feature_tile = kPostFeatureTile;
  out->x_stream_stride = x_stream_stride;
  out->y_stream_stride = y_stream_stride;
  out->b_row_stride = b_row_stride;
  return Status::success;
}

Status BuildCompressorTiling(std::uint32_t wkv_row_stride,
                             std::uint32_t wgate_row_stride,
                             std::uint32_t ape_row_stride,
                             std::uint32_t coff,
                             std::uint32_t rotary_mode,
                             CompressorTilingData* out) {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // coff == 1 is the vendored compressor's DEFAULT and is NOT what the DSV4
  // checkpoint uses. Accepting it here would silently compute a four-row
  // window, so it is rejected rather than adapted.
  if (coff != kCompressorCoff) {
    return Status::unsupported_shape;
  }
  // Half mode, not interleave. The two differ only in which lanes pair up, so
  // an interleave-mode launch would produce a plausible but wrong token.
  if (rotary_mode != kRotaryModeHalf) {
    return Status::unsupported_shape;
  }
  if (wkv_row_stride < kHiddenSize || wgate_row_stride < kHiddenSize ||
      ape_row_stride < kCompressorCoff * kCompressorHeadDim) {
    return Status::unsupported_shape;
  }
  *out = CompressorTilingData{};
  out->abi_version = kAbiVersion;
  out->block_dim = 1U;
  out->hidden = kHiddenSize;
  out->head_dim = kCompressorHeadDim;
  out->coff = kCompressorCoff;
  out->ratio = kCompressorRatio;
  out->reduction_rows = kCompressorRows;
  out->rope_head_dim = kRopeHeadDim;
  out->rotary_mode = kRotaryModeHalf;
  out->projection_row_tile = 1U;
  out->wkv_row_stride = wkv_row_stride;
  out->wgate_row_stride = wgate_row_stride;
  out->ape_row_stride = ape_row_stride;
  out->reserved = 0U;
  out->norm_eps = kRmsNormEps;
  out->reserved_fp32 = 0.0F;
  return Status::success;
}

Status BuildIndexerTiling(std::uint32_t query_head_stride,
                          std::uint32_t key_block_stride,
                          std::uint32_t block_size,
                          std::uint32_t seq_len,
                          std::uint32_t block_table_len,
                          std::uint32_t num_physical_blocks,
                          IndexerTilingData* out) {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  if (query_head_stride < kIndexerHeadDim) {
    return Status::unsupported_shape;
  }
  if (block_size == 0U || num_physical_blocks == 0U) {
    return Status::invalid_argument;
  }
  if (key_block_stride < block_size * kIndexerHeadDim) {
    return Status::unsupported_shape;
  }
  // seq_len == 0 is legal and selects nothing; the kernel still fills all 512
  // output lanes with -1.
  if (seq_len > 0U) {
    const std::uint32_t needed = (seq_len + block_size - 1U) / block_size;
    if (block_table_len < needed) {
      return Status::unsupported_shape;
    }
    // v1 limits, both from the static UB layout: `Sort` works in 32-element
    // groups, and the selection buffers are sized for the v3 geometry.
    if ((seq_len % kIndexerSortGroup) != 0U || seq_len > kIndexerMaxSeqLen) {
      return Status::unsupported_shape;
    }
  }
  *out = IndexerTilingData{};
  out->abi_version = kAbiVersion;
  out->block_dim = 1U;
  out->heads = kIndexerHeads;
  out->head_dim = kIndexerHeadDim;
  out->top_k = kIndexerTopK;
  out->query_head_stride = query_head_stride;
  out->key_block_stride = key_block_stride;
  out->block_size = block_size;
  out->seq_len = seq_len;
  out->block_table_len = block_table_len;
  out->num_physical_blocks = num_physical_blocks;
  out->score_tile = kIndexerScoreTile;
  return Status::success;
}

Status BuildMxfp4GemvTiling(std::uint32_t rows,
                            std::uint32_t cols,
                            std::uint32_t packed_row_stride,
                            std::uint32_t scale_row_stride,
                            std::uint32_t low_nibble_first,
                            Mxfp4GemvTilingData* out) {
  if (out == nullptr) {
    return Status::invalid_argument;
  }
  // The byte-for-byte packing is the contract. An FP8-style shared 128x128
  // scale grid read with this indexing yields finite, plausibly-scaled
  // garbage, so the strides are required to be exact rather than minima.
  if (low_nibble_first != kMxfp4LowNibbleFirst) {
    return Status::unsupported_shape;
  }
  if (rows == 0U || cols == 0U) {
    return Status::invalid_argument;
  }
  if ((cols % kMxfp4GroupSize) != 0U) {
    return Status::unsupported_shape;
  }
  if (packed_row_stride != cols / 2U || scale_row_stride != cols / kMxfp4GroupSize) {
    return Status::unsupported_shape;
  }
  // The resident FP32 x and the per-row decode buffers are statically sized.
  if (cols > ker05_ub::kXf32Count) {
    return Status::unsupported_shape;
  }
  *out = Mxfp4GemvTilingData{};
  out->abi_version = kAbiVersion;
  out->block_dim = kGemvBlockDim;
  out->rows = rows;
  out->cols = cols;
  out->rows_per_block = kGemvRowsPerBlock;
  out->group_size = kMxfp4GroupSize;
  out->packed_row_stride = packed_row_stride;
  out->scale_row_stride = scale_row_stride;
  out->low_nibble_first = kMxfp4LowNibbleFirst;
  out->reserved = 0U;
  return Status::success;
}

}  // namespace moe::kernels::arch35
