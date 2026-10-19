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
# On-device verification of the six arch35 operators (CLAUDE.md section 4).
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
  # -------------------------------------------------------------------------
  # bf16_block_quant_fp8, on its own binary.
  #
  # Separate from test_custom_kernels_npu_device on purpose: this one exists
  # to settle the __asc_fp8x2_storage_t byte order, and a dedicated binary
  # keeps that answer readable instead of buried in a seven-operator run. It
  # has the same dependencies and the same entry-guard contract.
  # -------------------------------------------------------------------------
  add_executable(test_bf16_block_quant_fp8_npu
    test_bf16_block_quant_fp8_npu.cpp)
  target_link_libraries(test_bf16_block_quant_fp8_npu PRIVATE
    arch35_device_launch
    moe_emu          # the in-memory CPU golden it compares against
    pthread
    dl
  )
  target_include_directories(test_bf16_block_quant_fp8_npu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/tests/common"
  )
  target_compile_options(test_bf16_block_quant_fp8_npu PRIVATE
    -Wall -Wextra
    # Compares device output against a host golden computed from the same
    # arithmetic, so the host side must not contract into an FMA at will.
    -ffp-contract=off
  )
  if(DSV4_WERROR)
    target_compile_options(test_bf16_block_quant_fp8_npu PRIVATE -Werror)
  endif()
  set_target_properties(test_bf16_block_quant_fp8_npu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/npu"
    BUILD_RPATH "$ORIGIN/../.."
  )
  add_test(NAME test_bf16_block_quant_fp8_npu
           COMMAND test_bf16_block_quant_fp8_npu)
  set_tests_properties(test_bf16_block_quant_fp8_npu PROPERTIES
    LABELS "npu;arch35;verification"
    TIMEOUT 600
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
else()
  message(STATUS
    "test_custom_kernels_npu_device: not built (needs "
    "-DENABLE_ARCH35_DEVICE_LAUNCH=ON)")
  message(STATUS
    "test_bf16_block_quant_fp8_npu: not built (needs "
    "-DENABLE_ARCH35_DEVICE_LAUNCH=ON)")
endif()


# ---------------------------------------------------------------------------
# test_npu_dsv4_moe_prefill_cube_standalone
#
# On-device verification of the dsv4_moe_prefill_cube_pipeline entry point, as
# an ISOLATED target: it does NOT link arch35_device_launch, so building it
# compiles four device TUs and one host TU and nothing else of the engine.
#
#   src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_pipeline_kernel.cpp
#   src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_pipeline.cpp
#   src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_tiling.cpp
#   src/kernels/arch35/launch_common.cpp
#
# All four are ASC (BiSheng) TUs for the reason in CLAUDE.md section 3.3: the
# chevron dispatch is the only launch form this toolkit supports and only
# BiSheng lowers it. The gtest driver stays on the host compiler; mixing the
# two at link time works.
#
# moe_emu is linked because it IS the golden the suite compares against
# (CLAUDE.md section 4), not because the engine is needed. It links nothing but
# libstdc++/libm.
#
# DSV4_TILING_TEST_OVERRIDE=1 is PUBLIC on the device library so the kernel,
# the tiling builder and the gtest driver agree on kPrefillCubeInterCap and on
# the admissibility predicate. Without it the only admissible shape is
# hidden 4096 / inter 2048, which no bring-up case wants to carry.
#
# Filterable: --gtest_filter=*Tile1* / *Tile2* / *Tile4* / *Full*
# Device printf under Release: -DENABLE_KERNEL_PRINTF=ON
# ---------------------------------------------------------------------------
option(ENABLE_KERNEL_PRINTF
       "Enable __asc_aicore::printf tracing in the standalone pipeline kernel, even under Release"
       OFF)
set(DSV4_GTEST_SOURCE_DIR "" CACHE PATH
    "Local googletest checkout to build against instead of fetching one")

if(COMMAND ascendc_library)
  set(_pipeline_device dsv4_moe_prefill_cube_pipeline_device)
  set(_standalone_target test_npu_dsv4_moe_prefill_cube_standalone)
  set(_dsv4_op_dir "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert")

  # The toolkit's arch-qualified tree, which carries <acl/acl.h> for the host
  # driver and the driver stub the link needs.
  set(_dsv4_cann "$ENV{ASCEND_HOME_PATH}")
  if(NOT _dsv4_cann)
    set(_dsv4_cann "${ASCEND_HOME_PATH}")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_dsv4_cann_arch "${_dsv4_cann}/aarch64-linux")
  else()
    set(_dsv4_cann_arch "${_dsv4_cann}/x86_64-linux")
  endif()

  # -------------------------------------------------------------------------
  # Google Test: an installed one if there is one, otherwise a source build.
  # DSV4_GTEST_SOURCE_DIR points FetchContent at a local checkout so the target
  # also configures with no network.
  # -------------------------------------------------------------------------
  if(NOT TARGET GTest::gtest)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::gtest)
    include(FetchContent)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    if(DSV4_GTEST_SOURCE_DIR)
      FetchContent_Declare(googletest SOURCE_DIR "${DSV4_GTEST_SOURCE_DIR}")
    else()
      FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
      )
    endif()
    FetchContent_MakeAvailable(googletest)
  endif()

  ascendc_library(${_pipeline_device} STATIC
    "${_dsv4_op_dir}/op_kernel/dsv4_moe_prefill_cube_pipeline_kernel.cpp"
    "${_dsv4_op_dir}/op_host/dsv4_moe_prefill_cube_pipeline.cpp"
    "${_dsv4_op_dir}/op_host/dsv4_moe_prefill_cube_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )

  # Bare-filename includes, resolved by the per-op include directories
  # (CLAUDE.md section 1.3).
  ascendc_include_directories(${_pipeline_device} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${_dsv4_op_dir}/op_kernel"
    "${_dsv4_op_dir}/op_host"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35"
  )
  # PUBLIC: the gtest driver includes op_host/dsv4_moe_prefill_cube_pipeline.h
  # and must see the same geometry constants the device TUs were built with.
  target_include_directories(${_pipeline_device} PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${_dsv4_op_dir}/op_host"
    "${_dsv4_cann_arch}/include"
  )
  target_compile_definitions(${_pipeline_device} PUBLIC DSV4_TILING_TEST_OVERRIDE=1)

  # -Werror on the device TUs, and deliberately WITHOUT -Wall -Wextra.
  #
  # Those two are the host driver's contract, not the kernel's: turning them on
  # for an ASC TU makes BiSheng report ~20 -Wunused-parameter and
  # -Wmissing-braces diagnostics inside the TOOLKIT's own headers
  # (impl/basic_api/kernel_event.h, kernel_utils_ceil_oom_que.h,
  # impl/utils/debug/asc_*printf*.h), which no change to this operator can
  # silence. arch35_device_launch and scripts/device_compile_kernels.sh leave
  # them off for the same reason. BiSheng's default warning set on these four
  # TUs is clean and -Werror keeps it that way.
  if(DSV4_WERROR)
    target_compile_options(${_pipeline_device} PRIVATE -Werror)
  endif()

  # -------------------------------------------------------------------------
  # Release-mode device printf
  # -------------------------------------------------------------------------
  # __asc_aicore::printf compiles away when ASCENDC_DUMP is defined to 0, and a
  # Release configuration also carries -DNDEBUG. Both are neutralised here so
  # the trace survives -DCMAKE_BUILD_TYPE=Release: the kernel's
  # DSV4_PIPELINE_TRACE macro is keyed on ENABLE_KERNEL_PRINTF alone.
  if(ENABLE_KERNEL_PRINTF)
    target_compile_definitions(${_pipeline_device} PRIVATE
      ENABLE_KERNEL_PRINTF=1
      ASCENDC_DUMP=1
    )
    target_compile_options(${_pipeline_device} PRIVATE -UNDEBUG)
    message(STATUS
      "test_npu_dsv4_moe_prefill_cube_standalone: device printf ENABLED "
      "(ASCENDC_DUMP=1, -UNDEBUG)")
  endif()

  add_executable(${_standalone_target} npu_dsv4_moe_prefill_cube_standalone.cpp)

  target_link_libraries(${_standalone_target} PRIVATE
    ${_pipeline_device}
    moe_emu            # the in-memory CPU golden it compares against
    GTest::gtest
    Threads::Threads
    dl
  )
  target_include_directories(${_standalone_target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_compile_options(${_standalone_target} PRIVATE
    -Wall -Wextra
    # Compares device output against a host golden computed from the same
    # arithmetic, so the host side must not contract into an FMA at will.
    -ffp-contract=off
  )
  if(DSV4_WERROR)
    target_compile_options(${_standalone_target} PRIVATE -Werror)
  endif()

  # -------------------------------------------------------------------------
  # LINK-TIME DRIVER STUBS, -rpath-link ONLY -- never -rpath (CLAUDE.md 4.4).
  #
  # libruntime.so's DT_NEEDED closure reaches libascend_hal.so, which ships
  # with the DRIVER and not the toolkit. The stub makes the LINK possible on a
  # driver-less workstation; keeping it off the runtime search path is what
  # stops a fake driver from satisfying the suite's device probe.
  # -------------------------------------------------------------------------
  foreach(_candidate
      "${_dsv4_cann_arch}/devlib/linux/x86_64"
      "${_dsv4_cann_arch}/devlib"
      "${_dsv4_cann}/devlib/linux/x86_64"
      "${_dsv4_cann}/devlib")
    if(EXISTS "${_candidate}/libascend_hal.so")
      target_link_options(${_standalone_target} PRIVATE
                          "LINKER:-rpath-link,${_candidate}")
    endif()
  endforeach()
  foreach(_candidate "${_dsv4_cann_arch}/lib64" "${_dsv4_cann}/lib64")
    if(IS_DIRECTORY "${_candidate}")
      target_link_options(${_standalone_target} PRIVATE
                          "LINKER:-rpath-link,${_candidate}")
    endif()
  endforeach()

  set_target_properties(${_standalone_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/npu"
    BUILD_RPATH "$ORIGIN/../.."
  )

  add_test(NAME ${_standalone_target} COMMAND ${_standalone_target})
  set_tests_properties(${_standalone_target} PROPERTIES
    LABELS "npu;arch35;pipeline;verification"
    TIMEOUT 900
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
else()
  message(STATUS
    "test_npu_dsv4_moe_prefill_cube_standalone: not built (needs "
    "-DENABLE_ARCH35_DEVICE_LAUNCH=ON, which enables the ASC language)")
endif()
