#pragma once

#include <array>
#include <cstdint>

namespace ascend_moe {

// Compressor's TH layout reserves min(T, T/ratio+B) rows. For a single
// complete window and B=1 this is two rows; only the first is emitted.
inline constexpr int64_t kCompressorWindowOutputRows = 2;
inline constexpr int64_t kIndexerMetadataElements = 1024;

// Single-sequence, single-key-head decode schedule. LI core 0 owns the
// half-open BN2 interval [0,1); the kernel derives M/S2 ends from live lengths.
// All other LI cores and all LD (split-reduction) cores are disabled. See
// vllm_quant_lightning_indexer_metadata.h and QLIPreload::SplitCoreByAICPU.
// This deliberately trades parallelism for a static, replayable schedule.
inline std::array<int32_t, kIndexerMetadataElements> DecodeIndexerMetadata() {
  std::array<int32_t, kIndexerMetadataElements> metadata{};
  metadata[0] = 1;  // LI_CORE_ENABLE
  metadata[4] = 1;  // LI_BN2_END (exclusive)
  return metadata;
}

}  // namespace ascend_moe
