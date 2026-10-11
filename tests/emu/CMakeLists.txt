# -----------------------------------------------------------------------------
# The host emulation test suite. Registered under ctest; no device, no CANN.
# -----------------------------------------------------------------------------

set(MOE_EMU_TESTS
  test_mhc_sinkhorn_emu
  test_paged_compressor_emu
  test_fp8_indexer_emu
  test_safetensors_loader
  test_semi_e2e_pipeline
)

foreach(emu_test IN LISTS MOE_EMU_TESTS)
  add_executable(${emu_test} "${emu_test}.cpp")
  target_link_libraries(${emu_test} PRIVATE moe_emu)
  target_include_directories(${emu_test} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(${emu_test} PRIVATE ${MOE_EMU_WARNINGS})
  set_target_properties(${emu_test} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/emu"
  )
  # WORKING_DIRECTORY is the source root so test_safetensors_loader finds
  # data/dsv4_layer0/ by its relative path when the shard has been downloaded.
  # It SKIPs that section rather than failing when it has not, so ctest passes
  # on a clean checkout.
  add_test(NAME ${emu_test}
           COMMAND ${emu_test}
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(${emu_test} PROPERTIES TIMEOUT 300)
endforeach()
