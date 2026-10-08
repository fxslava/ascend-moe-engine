#include "moe/diagnostics/inference_stats.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <locale>

#include "moe/core/error.hpp"

namespace ascend_moe {
namespace {
void Number(std::ostream& out, const std::optional<double>& value) {
  if (value && std::isfinite(*value)) out << *value;
  else out << "null";
}
}  // namespace

void InferenceDiagnostics::DumpJson(const std::string& path) const {
  std::ofstream out(path);
  DSV4_REQUIRE(out.is_open(), "cannot write inference diagnostics to " << path);
  out.imbue(std::locale::classic());
  out << std::setprecision(12) << std::boolalpha;
  out << "{\n  \"schema_version\": 1,\n"
      << "  \"mock_runtime\": " << mock_runtime << ",\n"
      << "  \"synthetic_weights\": " << synthetic_weights << ",\n"
      << "  \"dry_run\": " << dry_run << ",\n"
      << "  \"prompt_tokens\": " << prompt_tokens << ",\n"
      << "  \"generated_tokens\": " << generated_tokens << ",\n"
      << "  \"decoded_steps\": " << decoded_steps << ",\n  \"ttft_ms\": ";
  Number(out, ttft_ms);
  out << ",\n  \"tpot_ms\": ";
  Number(out, tpot_ms);
  out << ",\n  \"perplexity\": ";
  Number(out, perplexity);
  out << ",\n  \"moe_cache\": {\n"
      << "    \"total_expert_requests\": " << moe_cache.total_expert_requests << ",\n"
      << "    \"hbm_slot_hits\": " << moe_cache.hbm_slot_hits << ",\n"
      << "    \"host_promotions\": " << moe_cache.host_promotions << ",\n"
      << "    \"evictions_to_host\": " << moe_cache.evictions_to_host << ",\n"
      << "    \"hit_rate\": ";
  Number(out, moe_cache.HitRate());
  out << "\n  },\n  \"paged_attention\": {\n"
      << "    \"total_blocks_allocated\": " << paged_attention.total_blocks_allocated << ",\n"
      << "    \"active_context_tokens\": " << paged_attention.active_context_tokens << ",\n"
      << "    \"block_size\": " << paged_attention.block_size << ",\n"
      << "    \"kv_cache_utilization\": ";
  Number(out, paged_attention.kv_cache_utilization);
  out << "\n  },\n  \"attention\": {\n    \"mode\": \"dense_mla\",\n    \"attention_entropy\": ";
  Number(out, attention.attention_entropy);
  out << ",\n    \"sparsity_ratio\": ";
  Number(out, attention.sparsity_ratio);
  out << "\n  }\n}\n";
  out.close();
  DSV4_REQUIRE(static_cast<bool>(out), "failed writing inference diagnostics to " << path);
}
}  // namespace ascend_moe
