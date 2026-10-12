# -----------------------------------------------------------------------------
# Two completely decoupled binaries for the five arch35 custom kernels.
#
#   test_custom_kernels_verification  -- correctness vs the in-memory CPU golden
#   bench_custom_kernels_performance  -- latency / throughput / efficiency
#
# Decoupled means decoupled: separate executables, no shared translation unit
# beyond the header-only kernel_test_harness.hpp, and no build-order dependency
# between them. Either can be built and run without the other.
#
# Both link only moe_emu (host-portable C++ over libstdc++/libm). Neither links
# any host CPU debug/interpretation runtime -- no cpudebug, no tikicpulib, no
# libascendc_cpudebug.so -- and `custom_kernels_no_cpudebug` below checks that
# against the linked artifacts rather than trusting the comment.
# -----------------------------------------------------------------------------

set(_custom_kernel_binaries
  test_custom_kernels_verification
  bench_custom_kernels_performance
)

foreach(_bin IN LISTS _custom_kernel_binaries)
  add_executable(${_bin} "${_bin}.cpp")
  target_link_libraries(${_bin} PRIVATE moe_emu)
  target_include_directories(${_bin} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(${_bin} PRIVATE ${MOE_EMU_WARNINGS})
  # Same reproducibility rule as moe_emu itself: the harness compares two
  # implementations of identical arithmetic, so the host side must not be free
  # to contract a * b + c into an FMA at its own discretion.
  target_compile_options(${_bin} PRIVATE -ffp-contract=off)
  set_target_properties(${_bin} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/custom_kernels"
  )
endforeach()

# The verification binary is the one with pass/fail semantics, so it is the one
# registered with ctest. The benchmark is a reporting tool: it is built and
# runnable but not a gate, since a timing threshold on shared CI hardware is a
# flake generator rather than a signal.
add_test(NAME test_custom_kernels_verification
         COMMAND test_custom_kernels_verification)
set_tests_properties(test_custom_kernels_verification PROPERTIES
  LABELS "custom_kernels;verification"
  TIMEOUT 600
)

# A smoke test that the benchmark binary runs and exits clean, with the
# iteration counts turned right down. Guards against the harness bit-rotting
# unnoticed just because nobody looks at timings in CI.
add_test(NAME bench_custom_kernels_performance_smoke
         COMMAND bench_custom_kernels_performance 1 2)
set_tests_properties(bench_custom_kernels_performance_smoke PROPERTIES
  LABELS "custom_kernels;benchmark"
  TIMEOUT 600
)

# ---------------------------------------------------------------------------
# The "no host CPU debug runtime" claim, checked against the built artifacts.
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  find_program(_ck_ldd ldd)
  if(_ck_ldd)
    add_test(NAME custom_kernels_no_cpudebug
      COMMAND ${CMAKE_COMMAND}
        -D "LDD=${_ck_ldd}"
        -D "BINARIES=${CMAKE_BINARY_DIR}/tests/custom_kernels/test_custom_kernels_verification;${CMAKE_BINARY_DIR}/tests/custom_kernels/bench_custom_kernels_performance"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_no_cpudebug.cmake"
    )
    set_tests_properties(custom_kernels_no_cpudebug PROPERTIES
      LABELS "custom_kernels;hygiene"
      TIMEOUT 120
    )
  endif()
endif()
