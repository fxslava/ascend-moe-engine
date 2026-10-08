#pragma once
#include <string>
#include <vector>

namespace ascend_moe::mock {
/// Test-only deterministic faults; an empty name disables injection.
void FailRuntimeCall(const std::string& name, int successful_calls_before_failure = 0);
void ClearRuntimeTrace();
const std::vector<std::string>& RuntimeTrace();
int RecordRuntimeCall(const char* name);
const char* RuntimeRecentError();
}  // namespace ascend_moe::mock
