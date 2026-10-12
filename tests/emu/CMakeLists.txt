# -----------------------------------------------------------------------------
# The host emulation test suite. Registered under ctest; no device, no CANN.
# -----------------------------------------------------------------------------

set(MOE_EMU_TESTS
  test_mhc_sinkhorn_emu
  test_paged_compressor_emu
  test_fp8_indexer_emu
  test_safetensors_loader
  test_semi_e2e_pipeline
  test_custom_kernels_emu
)

foreach(emu_test IN LISTS MOE_EMU_TESTS)
  add_executable(${emu_test} "${emu_test}.cpp")
  target_link_libraries(${emu_test} PRIVATE moe_emu)
  target_include_directories(${emu_test} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # kernel_test_harness.hpp (metrics, Birkhoff residual, RNG, BF16 helpers)
    # is shared with the npu and bench suites, so it lives in tests/common.
    "${CMAKE_SOURCE_DIR}/tests/common"
  )
  target_compile_options(${emu_test} PRIVATE ${MOE_EMU_WARNINGS})
  set_target_properties(${emu_test} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/emu"
  )
  # WORKING_DIRECTORY is the source root so test_safetensors_loader finds
  # data/dsv4_layer0/ by its relative path when the shard has been downloaded.
  # It SKIPs that section rather than failing when it has not, so ctest passes
  # on a clean checkout.
  # Same reproducibility rule as moe_emu itself: these suites compare two
  # implementations of identical arithmetic, so the host side must not be free
  # to contract a * b + c into an FMA at its own discretion.
  target_compile_options(${emu_test} PRIVATE -ffp-contract=off)
  add_test(NAME ${emu_test}
           COMMAND ${emu_test}
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(${emu_test} PROPERTIES
    LABELS "emu"
    TIMEOUT 300
  )
endforeach()

# ---------------------------------------------------------------------------
# "No host CPU debug runtime", checked against the built artifacts rather than
# trusted to a comment (CLAUDE.md section 5.2). The emu suite is the one that
# must be provably free of cpudebug / tikicpulib: it is the golden oracle, and
# a reference validated by an interpreter would be circular.
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  find_program(_emu_ldd ldd)
  if(_emu_ldd)
    set(_emu_binaries "")
    foreach(emu_test IN LISTS MOE_EMU_TESTS)
      list(APPEND _emu_binaries "${CMAKE_BINARY_DIR}/tests/emu/${emu_test}")
    endforeach()
    add_test(NAME emu_no_cpudebug
      COMMAND ${CMAKE_COMMAND}
        -D "LDD=${_emu_ldd}"
        -D "BINARIES=${_emu_binaries}"
        -P "${CMAKE_SOURCE_DIR}/tests/common/verify_no_cpudebug.cmake"
    )
    set_tests_properties(emu_no_cpudebug PROPERTIES
      LABELS "emu;hygiene"
      TIMEOUT 120
    )
  endif()
endif()
