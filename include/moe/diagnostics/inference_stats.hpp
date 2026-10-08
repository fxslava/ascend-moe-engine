#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ascend_moe {

struct MoeCacheStats {
  uint64_t total_expert_requests = 0;
  uint64_t hbm_slot_hits = 0;
  uint64_t host_promotions = 0;
  uint64_t evictions_to_host = 0;

  double HitRate() const {
    return total_expert_requests == 0 ? 0.0 :
        static_cast<double>(hbm_slot_hits) / static_cast<double>(total_expert_requests);
  }
};

struct PagedAttentionStats {
  // Physical block reservations summed across all layers (allocated at Build).
  uint64_t total_blocks_allocated = 0;
  uint64_t active_context_tokens = 0;
  uint64_t block_size = 128;
  // Occupied token slots / reserved token slots; identical across layers.
  double kv_cache_utilization = 0.0;
};

struct AttentionDiagnostics {
  // No attention probabilities are read back solely for telemetry.
  std::optional<double> attention_entropy;
  // Fraction of context excluded by actual attention. This runner uses dense
  // MLA, so 0; never report the hypothetical Top-512 reduction as observed.
  double sparsity_ratio = 0.0;
};

struct InferenceDiagnostics {
  MoeCacheStats moe_cache;
  PagedAttentionStats paged_attention;
  AttentionDiagnostics attention;
  // CLI timing includes synchronized readback. null means not measured.
  std::optional<double> ttft_ms;
  std::optional<double> tpot_ms;
  std::optional<double> perplexity;
  bool mock_runtime = false;
  bool synthetic_weights = false;
  bool dry_run = false;
  uint64_t prompt_tokens = 0;
  uint64_t generated_tokens = 0;
  uint64_t decoded_steps = 0;

  void DumpJson(const std::string& path) const;
};

}  // namespace ascend_moe
