#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace ascend_moe {

using CheckpointHasTensor = std::function<bool(const std::string&)>;
using CheckpointIndex = std::map<std::string, std::string>;

// Auxiliary prediction heads are not inputs to the backbone decode graph.
bool IsAuxiliaryCheckpointTensor(const std::string& name);

// Canonical name wins; an empty result means no safe alias exists. This is
// shared by real ingestion and metadata-only probes. Never aliases HC mixing
// parameters to RMSNorm weights, or raw projections to absorbed MLA weights.
std::string ResolveCheckpointTensorName(const std::string& requested,
                                        const CheckpointHasTensor& has_tensor);

struct CheckpointTopology {
  std::map<std::string, std::string> globals;
  std::vector<std::map<std::string, std::string>> layers;
  size_t ignored_auxiliary_tensors = 0;
};

// Validates index-level backbone roles and shard targets, without opening any
// shard. HC bases describe topology only, not executable folded-MLA bindings.
// No dtype, shape, payload or numerical compatibility is implied by success.
CheckpointTopology ValidateCheckpointTopology(const CheckpointIndex& index, int64_t num_layers);

}  // namespace ascend_moe
