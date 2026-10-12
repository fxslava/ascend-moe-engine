# -----------------------------------------------------------------------------
# Benchmarks. Timing only, never a correctness gate (CLAUDE.md section 4).
#
# A timing threshold on shared CI hardware is a flake generator rather than a
# signal, so nothing here is registered with pass/fail semantics on its
# numbers. What IS registered is a smoke test that the binary runs and exits
# clean with the iteration counts turned right down, which catches the harness
# bit-rotting unnoticed just because nobody reads timings in CI.
# -----------------------------------------------------------------------------

add_executable(bench_custom_kernels_performance
  bench_custom_kernels_performance.cpp
)
target_link_libraries(bench_custom_kernels_performance PRIVATE moe_emu)
target_include_directories(bench_custom_kernels_performance PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/tests/common"
)
target_compile_options(bench_custom_kernels_performance PRIVATE
  ${MOE_EMU_WARNINGS}
  -ffp-contract=off
)

# The device side is optional: the launch layer needs BiSheng, not an NPU, so
# a build check is possible without hardware. When arch35_device_launch exists
# it carries MOE_KERNEL_LAUNCH_LAYER_AVAILABLE, the CANN include dirs and the
# AscendCL link interface, and the benchmark's device sections compile in.
if(TARGET arch35_device_launch)
  target_link_libraries(bench_custom_kernels_performance PRIVATE
    arch35_device_launch)
endif()

set_target_properties(bench_custom_kernels_performance PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bench"
)

add_test(NAME bench_custom_kernels_performance_smoke
         COMMAND bench_custom_kernels_performance 1 2)
set_tests_properties(bench_custom_kernels_performance_smoke PROPERTIES
  LABELS "bench;smoke"
  TIMEOUT 600
)

# The --device-only path is the one a profiling session actually uses, and it
# takes a different route through every section (host loops skipped, device
# sections still dispatched). Smoke it separately so the flag cannot rot
# unnoticed behind the default run -- on a host with no device it does almost
# nothing and costs almost nothing, which is exactly why it is cheap to keep.
add_test(NAME bench_custom_kernels_performance_device_only_smoke
         COMMAND bench_custom_kernels_performance 1 2 --device-only)
set_tests_properties(bench_custom_kernels_performance_device_only_smoke
  PROPERTIES
  LABELS "bench;smoke"
  TIMEOUT 600
)
