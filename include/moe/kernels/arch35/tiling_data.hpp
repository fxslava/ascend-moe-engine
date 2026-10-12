// tiling_data.hpp -- versioned, fixed-width tiling PODs for the arch35
// Stage-1 direct kernels (fused_mhc_pre_norm_sinkhorn..mxfp4_expert_gemv, docs/kernel_contracts_spec.md).
//
// Every struct is a standard-layout, trivially-copyable POD of 32-bit fields
// only, so host and device translation units see the identical layout with no
// pointer-width padding. The device kernel reads it through the `tiling`
// GM_ADDR argument; the host copies it into the device tiling buffer verbatim.
// `reserved` fields must be zero; kernels reject unknown `abi_version`.
//
// These PODs carry no pointers. Pointer addresses are kernel GM_ADDR
// arguments; the numerical order, not any private UB layout, is normative.

#ifndef MOE_KERNELS_ARCH35_TILING_DATA_HPP
#define MOE_KERNELS_ARCH35_TILING_DATA_HPP

#include <cstdint>
#include <type_traits>

namespace moe::kernels::arch35 {

inline constexpr std::uint32_t kTilingAbiVersion = 1;

// ---------------------------------------------------------------------------
// fused_mhc_pre_norm_sinkhorn -- fused_mhc_pre_norm_sinkhorn (sizeof == 64)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// fused_mhc_post_residual -- fused_mhc_post_residual (sizeof == 32)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// paged_compressor_rope -- fused_paged_compressor_rope, C4A coff=2 (sizeof == 64)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// fast_fp8_indexer_topk -- fused_fp8_indexer_topk (sizeof == 48)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// mxfp4_expert_gemv -- mxfp4 expert GEMV (sizeof == 40)
// ---------------------------------------------------------------------------
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

// The fixed-size ABI. Host and device TUs must compile matching asserts.
static_assert(sizeof(MhcPreNormTilingData) == 64, "fused_mhc_pre_norm_sinkhorn tiling must be 64 B");
static_assert(sizeof(MhcPostTilingData) == 32, "fused_mhc_post_residual tiling must be 32 B");
static_assert(sizeof(CompressorTilingData) == 64, "paged_compressor_rope tiling must be 64 B");
static_assert(sizeof(IndexerTilingData) == 48, "fast_fp8_indexer_topk tiling must be 48 B");
static_assert(sizeof(Mxfp4GemvTilingData) == 40, "mxfp4_expert_gemv tiling must be 40 B");
static_assert(std::is_standard_layout<MhcPreNormTilingData>::value &&
                  std::is_trivially_copyable<MhcPreNormTilingData>::value,
              "fused_mhc_pre_norm_sinkhorn tiling must be a standard-layout trivially-copyable POD");
static_assert(std::is_standard_layout<MhcPostTilingData>::value &&
                  std::is_trivially_copyable<MhcPostTilingData>::value,
              "fused_mhc_post_residual tiling must be a standard-layout trivially-copyable POD");
static_assert(std::is_standard_layout<CompressorTilingData>::value &&
                  std::is_trivially_copyable<CompressorTilingData>::value,
              "paged_compressor_rope tiling must be a standard-layout trivially-copyable POD");
static_assert(std::is_standard_layout<IndexerTilingData>::value &&
                  std::is_trivially_copyable<IndexerTilingData>::value,
              "fast_fp8_indexer_topk tiling must be a standard-layout trivially-copyable POD");
static_assert(std::is_standard_layout<Mxfp4GemvTilingData>::value &&
                  std::is_trivially_copyable<Mxfp4GemvTilingData>::value,
              "mxfp4_expert_gemv tiling must be a standard-layout trivially-copyable POD");

}  // namespace moe::kernels::arch35

#endif  // MOE_KERNELS_ARCH35_TILING_DATA_HPP
