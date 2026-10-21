# Flat MoE simulator test: one host runner, one procedural data source.
add_executable(sim_moe_test main_sim.cpp test_patterns.cpp)
target_compile_features(sim_moe_test PRIVATE cxx_std_17)
target_compile_definitions(sim_moe_test PRIVATE DSV4_SIM_STEPS=2)
target_compile_options(sim_moe_test PRIVATE -O3 -Wall -Wextra)
target_include_directories(sim_moe_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(sim_moe_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/sim")
add_test(NAME sim_moe_test COMMAND sim_moe_test)

if(COMMAND ascendc_library)
  set(_human_cann "$ENV{ASCEND_HOME_PATH}")
  if(NOT _human_cann)
    set(_human_cann "${ASCEND_HOME_PATH}")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_human_arch "${_human_cann}/aarch64-linux")
  else()
    set(_human_arch "${_human_cann}/x86_64-linux")
  endif()
  set(_human_device dsv4_moe_human_camodel_device)
  ascendc_library(${_human_device} STATIC
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_human_writed_kernel.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_human_writed.cpp")
  ascendc_include_directories(${_human_device} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${CMAKE_SOURCE_DIR}/src/ops/common"
    "${_human_arch}/include")
  target_compile_definitions(${_human_device} PRIVATE ENABLE_KERNEL_PRINTF=1 ASCENDC_DUMP=1 DSV4_SIM_MICROTILE=1 DSV4_SIM_STEPS=2)
  target_compile_options(${_human_device} PRIVATE --cce-enable-print -UNDEBUG)
  target_compile_definitions(sim_moe_test PRIVATE DSV4_CAMODEL=1 DSV4_SIM_MICROTILE=1 DSV4_SIM_STEPS=2)
  target_include_directories(sim_moe_test PRIVATE
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${_human_arch}/include")
  target_link_libraries(sim_moe_test PRIVATE ${_human_device})
  foreach(_candidate
      "${_human_arch}/devlib/linux/x86_64"
      "${_human_arch}/devlib"
      "${_human_cann}/devlib/linux/x86_64"
      "${_human_cann}/devlib")
    if(EXISTS "${_candidate}/libascend_hal.so")
      target_link_options(sim_moe_test PRIVATE
        "LINKER:-rpath-link,${_candidate}")
    endif()
  endforeach()
  foreach(_candidate "${_human_arch}/lib64" "${_human_cann}/lib64")
    if(IS_DIRECTORY "${_candidate}")
      target_link_options(sim_moe_test PRIVATE
        "LINKER:-rpath-link,${_candidate}")
    endif()
  endforeach()
  set(_human_sim "${_human_arch}/simulator/dav_3510")
  set_tests_properties(sim_moe_test PROPERTIES
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;ASCEND_HOME_PATH=${_human_cann};LD_LIBRARY_PATH=${_human_sim}/camodel:${_human_sim}/lib:${_human_arch}/lib64:${_human_cann}/lib64:$ENV{LD_LIBRARY_PATH}"
    LABELS "sim;camodel;functional"
    TIMEOUT 3600)

  # ---------------------------------------------------------------------------
  # PROGRESS TRACE PROBE -- dsv4_moe_prefill_cube, SIM ONLY (CLAUDE.md 5.7)
  # ---------------------------------------------------------------------------
  # tests/sim/kernel/dsv4_moe_prefill_cube_trace_kernel.cpp is a transcription
  # of RunVectorCore/RunCubeCore with __asc_aicore::printf at every stage
  # boundary, so a stall is NAMED ("the AIV posted StepReady and the AIC never
  # returned from its StepReady wait") instead of inferred. It is the probe
  # that MEASURED the AIV GetBlockIdx()/GetSubBlockNum() table quoted in
  # src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp, which
  # is why it is kept rather than deleted.
  #
  # A DIAGNOSTIC, NOT A GATE, and not even a correctness probe: the trace
  # perturbs timing, so a retirement here would not license the schedule -- it
  # would say the stall is timing-sensitive, which is itself the finding.
  # Registered for reproducibility only.
  #
  # Rebuilt on ascendc_library() rather than the hand-rolled bisheng
  # add_custom_command() this used to carry. Same dav-3510 ASC compile, but the
  # toolkit's own macro attaches the CANN link interface and tracks header
  # dependencies, so the probe cannot drift from the production kernel it
  # textually includes -- which is exactly how its previous DEPENDS list came
  # to name dsv4_moe_prefill_cube_legacy_matmul.h after that header was gone.
  if(TARGET moe_emu)
    # MakeDsv4MoePrefillCubeGemmTiling, called from the probe's launch TU. The
    # module is idempotent and is also included by cmake/arch35_device.cmake.
    include("${CMAKE_SOURCE_DIR}/cmake/dsv4_prefill_cube_cube_tiling.cmake")
    find_package(Threads REQUIRED)

    set(_trace_device dsv4_moe_prefill_cube_trace_device)
    ascendc_library(${_trace_device} STATIC
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_trace_kernel.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_trace_launch.cpp"
      "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_tiling.cpp"
      "${CMAKE_SOURCE_DIR}/src/ops/common/launch_common.cpp")
    # asc/include is on the path for utils/debug/asc_printf.h, which no other
    # device TU in this tree uses.
    ascendc_include_directories(${_trace_device} PRIVATE
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
      "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
      "${CMAKE_SOURCE_DIR}/src/ops/common"
      "${CMAKE_CURRENT_SOURCE_DIR}/kernel"
      "${_human_arch}/include"
      "${_human_arch}/asc/include")
    target_include_directories(${_trace_device} PUBLIC
      "${CMAKE_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
      "${_human_arch}/include")
    # The probe runs the scaled geometry the registered case passes on argv,
    # which production tiling would reject; same override the other CAModel
    # targets use.
    target_compile_definitions(${_trace_device} PUBLIC DSV4_TILING_TEST_OVERRIDE=1)

    set(_trace_target test_dsv4_moe_prefill_cube_trace_sim)
    add_executable(${_trace_target} test_dsv4_moe_prefill_cube_sim.cpp)
    target_include_directories(${_trace_target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(${_trace_target} PRIVATE
      DSV4_BOUNDARY_TRACE_SIM=1)
    target_compile_options(${_trace_target} PRIVATE -Wall -Wextra)
    if(DSV4_WERROR)
      target_compile_options(${_trace_target} PRIVATE -Werror)
    endif()
    # THE GRAPH LAYER, LISTED EXPLICITLY.
    #
    # dsv4_prefill_cube_cube_tiling drags in libregister.so (for
    # optiling::TilingDef behind MatmulApiTiling), and libregister's own
    # DT_NEEDED closure reaches libgraph.so -> libmetadef.so for the ge::,
    # ops::, optiling::, domi:: and gert:: symbols. ld resolves a shared
    # library's own undefineds from -rpath-link and the referrer's RUNPATH, not
    # from -L, so without these the link dies in a page of
    # "libregister.so: undefined reference to ge::..." that names none of this
    # target's code.
    #
    # The old hand-rolled version of this target got them from
    # ${MOE_CANN_LIBRARIES}, which only the CANN branch of the root
    # CMakeLists defines -- so the probe built only in a full CANN
    # configuration and not in the host-emu + ENABLE_ARCH35_DEVICE_LAUNCH one
    # that scripts/verify_npu_build.sh uses. Searched rather than assumed, as
    # in cmake/dsv4_prefill_cube_cube_tiling.cmake.
    set(_trace_graph_libs)
    foreach(_lib libgraph.so libmetadef.so libopp_registry.so libexe_graph.so)
      if(EXISTS "${_human_arch}/lib64/${_lib}")
        list(APPEND _trace_graph_libs "${_human_arch}/lib64/${_lib}")
      endif()
    endforeach()

    target_link_libraries(${_trace_target} PRIVATE
      ${_trace_device}
      dsv4_prefill_cube_cube_tiling
      moe_emu                # the in-memory CPU golden it compares against
      ${_trace_graph_libs}
      Threads::Threads)

    # Driver stubs on -rpath-link ONLY, never -rpath (CLAUDE.md 4.4): the stub
    # must be invisible at run time or it could satisfy the device probe in
    # place of the CAModel's own libascend_hal.so, which the ctest ENVIRONMENT
    # below puts first on LD_LIBRARY_PATH.
    foreach(_candidate
        "${_human_arch}/devlib/linux/x86_64"
        "${_human_arch}/devlib"
        "${_human_cann}/devlib/linux/x86_64"
        "${_human_cann}/devlib")
      if(EXISTS "${_candidate}/libascend_hal.so")
        target_link_options(${_trace_target} PRIVATE
          "LINKER:-rpath-link,${_candidate}")
      endif()
    endforeach()
    foreach(_candidate "${_human_arch}/lib64" "${_human_cann}/lib64")
      if(IS_DIRECTORY "${_candidate}")
        target_link_options(${_trace_target} PRIVATE
          "LINKER:-rpath-link,${_candidate}")
      endif()
    endforeach()

    set_target_properties(${_trace_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/sim")

    # argv: [hidden] [inter] [budget_seconds]. The defaults ARE these.
    add_test(NAME Dsv4MoePrefillCubeTraceProbe
             COMMAND ${_trace_target} 128 256 5400)
    set_tests_properties(Dsv4MoePrefillCubeTraceProbe PROPERTIES
      ENVIRONMENT
        "ASCEND_SIMULATOR_MODE=1;ASCEND_HOME_PATH=${_human_cann};LD_LIBRARY_PATH=${_human_sim}/camodel:${_human_sim}/lib:${_human_arch}/lib64:${_human_cann}/lib64:$ENV{LD_LIBRARY_PATH}"
      LABELS "sim;arch35;camodel;probe;trace;cube;mix"
      TIMEOUT 7200)
  else()
    message(STATUS
      "test_dsv4_moe_prefill_cube_trace_sim: not built (needs the moe_emu CPU "
      "golden)")
  endif()

  # ISOLATION, ASSERTED rather than assumed: no production target may list a
  # tests/sim/kernel/ source. The whole point of the directory is that
  # verification scaffolding cannot reach a shipped artifact, and that holds
  # only until someone adds a source to the wrong target during a refactor.
  foreach(_prod_target moe_emu ascend_moe_engine arch35_device_launch)
    if(TARGET ${_prod_target})
      get_target_property(_prod_type ${_prod_target} TYPE)
      if(NOT _prod_type STREQUAL "INTERFACE_LIBRARY")
        get_target_property(_prod_sources ${_prod_target} SOURCES)
        if(_prod_sources)
          foreach(_prod_source IN LISTS _prod_sources)
            if(_prod_source MATCHES "/sim/kernel/")
              message(FATAL_ERROR
                "Production target '${_prod_target}' lists a sim/kernel/ "
                "source:
  ${_prod_source}
"
                "Verification scaffolding must not link into a production "
                "artifact.")
            endif()
          endforeach()
        endif()
      endif()
    endif()
  endforeach()
else()
  message(STATUS
    "sim_moe_test: using its hardware-free CPU launch stub (enable "
    "ENABLE_ARCH35_DEVICE_LAUNCH with CANN 9.2 to run CAModel)")
  set_tests_properties(sim_moe_test PROPERTIES
    LABELS "sim;stub;functional"
    TIMEOUT 300)
endif()
