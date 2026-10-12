# ---------------------------------------------------------------------------
# The npu suite has two groups with different dependencies, so each is gated
# on its own:
#
#   * the backend/ingest/e2e probes need the full
#     engine (ascend_moe_engine) and the AclNN headers, i.e. a CANN build;
#   * test_custom_kernels_npu_device needs only the BiSheng-built launch layer
#     and moe_emu, so it is also available in a host-emu build configured with
#     -DENABLE_ARCH35_DEVICE_LAUNCH=ON -- which is the configuration used to
#     verify that the device suite BUILDS on a machine with no NPU.
#
# Gating the whole directory on one of those would make the other unbuildable.
# ---------------------------------------------------------------------------
set(_npu_tests
  npu_backend_probe
  npu_weights_ingest_probe
  npu_e2e_single_token
)

if(NOT TARGET ascend_moe_engine)
  message(STATUS
    "tests/npu: engine-dependent probes not built (no ascend_moe_engine in "
    "this configuration)")
  set(_npu_tests "")
endif()

foreach(_test IN LISTS _npu_tests)
  # Suite sources sit next to this file.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cpp")
    set(_test_src "${_test}.cpp")
  else()
    set(_test_src "../${_test}.cpp")
  endif()
  add_executable(${_test} "${_test_src}")
  target_link_libraries(${_test} PRIVATE ascend_moe_engine)
  target_include_directories(${_test} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(${_test} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_test} PRIVATE -Werror)
  endif()
  set_target_properties(${_test} PROPERTIES BUILD_RPATH "$ORIGIN/../..")
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES
    LABELS "npu"
    TIMEOUT 120
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
endforeach()

# ---------------------------------------------------------------------------
# On-device verification of the five arch35 operators (CLAUDE.md section 4).
#
# Separate from the loop above because it links the BiSheng-built launch layer
# rather than ascend_moe_engine, and because it only exists when that layer
# has been built. arch35_device_launch carries
# MOE_KERNEL_LAUNCH_LAYER_AVAILABLE, the CANN include dirs and the AscendCL
# link interface, so nothing has to be spelled out here.
#
# Building it needs BiSheng and a CANN toolkit, NOT an NPU. RUNNING it needs
# an NPU, and the binary's own entry guard turns the absence of one into a
# skip with exit 0 -- see the SKIP_REGULAR_EXPRESSION below, which matches the
# phrase the guard prints.
# ---------------------------------------------------------------------------
if(TARGET arch35_device_launch)
  add_executable(test_custom_kernels_npu_device npu_custom_kernels_device.cpp)
  target_link_libraries(test_custom_kernels_npu_device PRIVATE
    arch35_device_launch
    moe_emu          # the in-memory CPU golden it compares against
    pthread
    dl
  )
  target_include_directories(test_custom_kernels_npu_device PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/common"
  )
  target_compile_options(test_custom_kernels_npu_device PRIVATE
    -Wall -Wextra
    # Compares device output against a host golden computed from the same
    # arithmetic, so the host side must not contract into an FMA at will.
    -ffp-contract=off
  )
  if(DSV4_WERROR)
    target_compile_options(test_custom_kernels_npu_device PRIVATE -Werror)
  endif()
  set_target_properties(test_custom_kernels_npu_device PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/npu"
    BUILD_RPATH "$ORIGIN/../.."
  )
  add_test(NAME test_custom_kernels_npu_device
           COMMAND test_custom_kernels_npu_device)
  set_tests_properties(test_custom_kernels_npu_device PROPERTIES
    LABELS "npu;arch35;verification"
    TIMEOUT 600
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
else()
  message(STATUS
    "test_custom_kernels_npu_device: not built (needs "
    "-DENABLE_ARCH35_DEVICE_LAUNCH=ON)")
endif()
