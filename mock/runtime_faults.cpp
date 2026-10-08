#include "runtime_faults.hpp"
namespace ascend_moe::mock {
namespace {
std::vector<std::string> trace;
std::string failing, recent;
int remaining = 0;
}  // namespace
void FailRuntimeCall(const std::string& name, int calls) {
  failing = name;
  remaining = calls;
  recent.clear();
}
void ClearRuntimeTrace() { trace.clear(); }
const std::vector<std::string>& RuntimeTrace() { return trace; }
int RecordRuntimeCall(const char* name) {
  trace.emplace_back(name);
  if (failing == name && remaining-- == 0) {
    recent = std::string("injected failure: ") + name;
    failing.clear();
    return 361001;
  }
  return 0;
}
const char* RuntimeRecentError() { return recent.empty() ? nullptr : recent.c_str(); }
}  // namespace ascend_moe::mock
