if(TARGET test_common AND TARGET cpu_ref)
  add_executable(test_mhc_fused_sinkhorn_gemv_sim
    test_mhc_fused_sinkhorn_gemv_sim.cpp
  )
  target_link_libraries(test_mhc_fused_sinkhorn_gemv_sim PRIVATE
    test_common
    cpu_ref
  )
  target_compile_options(test_mhc_fused_sinkhorn_gemv_sim PRIVATE
    -Wall -Wextra
  )
  target_compile_definitions(test_mhc_fused_sinkhorn_gemv_sim PRIVATE
    ENABLE_MOCK_RUNTIME=1
  )
  add_test(NAME MhcFusedSim COMMAND test_mhc_fused_sinkhorn_gemv_sim)
  set_tests_properties(MhcFusedSim PROPERTIES LABELS "sim" TIMEOUT 120)
endif()

if(TARGET moe_emu)
  set(_dsv4_tiling_source
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_expert_tiling.cpp"
  )

  add_executable(test_dsv4_moe_expert_tiling_prod
    test_dsv4_moe_expert_tiling_prod.cpp
    "${_dsv4_tiling_source}"
  )
  target_include_directories(test_dsv4_moe_expert_tiling_prod PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
  )
  target_compile_options(test_dsv4_moe_expert_tiling_prod PRIVATE
    -Wall -Wextra
  )
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_expert_tiling_prod PRIVATE -Werror)
  endif()
  add_test(NAME Dsv4MoeExpertProductionTiling
           COMMAND test_dsv4_moe_expert_tiling_prod)
  set_tests_properties(Dsv4MoeExpertProductionTiling PROPERTIES
    LABELS "sim;arch35;contract"
    TIMEOUT 30
  )

  # The whole-block builders' SKU-aware grid derivation. Links the operator's
  # own tiling source, as the expert tiler above does: these are host plumbing
  # tests (section 4.3), and the tiling header is ACL-free by section 2.4 so
  # neither needs a toolkit or a device.
  add_executable(test_dsv4_moe_block_grid_sim
    test_dsv4_moe_block_grid_sim.cpp
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.cpp"
  )
  target_include_directories(test_dsv4_moe_block_grid_sim PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
  )
  target_compile_options(test_dsv4_moe_block_grid_sim PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_block_grid_sim PRIVATE -Werror)
  endif()
  add_test(NAME Dsv4MoeBlockGridTiling
           COMMAND test_dsv4_moe_block_grid_sim)
  set_tests_properties(Dsv4MoeBlockGridTiling PROPERTIES
    LABELS "sim;arch35;contract"
    TIMEOUT 30
  )
endif()

option(ENABLE_DSV4_CAMODEL_TEST
       "Build the real dav-3510 DSV4 expert kernel test against CAModel" OFF)

if(ENABLE_DSV4_CAMODEL_TEST)
  if(ENABLE_MOCK_RUNTIME OR ENABLE_HOST_EMU)
    message(FATAL_ERROR
      "ENABLE_DSV4_CAMODEL_TEST requires a CANN build, not mock or host emulation")
  endif()
  if(NOT TARGET moe_emu)
    message(FATAL_ERROR "The CAModel test requires the moe_emu CPU golden target")
  endif()
  if(NOT ASCEND_HOME_PATH OR NOT EXISTS "${ASCEND_HOME_PATH}/bin/bisheng")
    message(FATAL_ERROR
      "ENABLE_DSV4_CAMODEL_TEST requires ASCEND_HOME_PATH/bin/bisheng; "
      "source the CANN set_env.sh before configuring")
  endif()
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "The installed DAV3510 CAModel test requires an x86_64 host")
  endif()

  set(_dsv4_sim_root
      "${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_3510")
  if(NOT EXISTS "${_dsv4_sim_root}/lib/libruntime_camodel.so")
    message(FATAL_ERROR
      "DAV3510 CAModel runtime not found under ${_dsv4_sim_root}")
  endif()
  if(NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
     OR NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
     OR NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
     OR NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
     OR NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
     OR NOT EXISTS "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so")
    message(FATAL_ERROR
      "CANN host runtime support libraries are missing under "
      "${ASCEND_HOME_PATH}/x86_64-linux/lib64")
  endif()

  set(_dsv4_asc_sources
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_expert_kernel.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_expert.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_expert_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_block_kernel.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )
  set(_dsv4_asc_headers
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_expert_arch35.h"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_expert_tiling.h"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_block_arch35.h"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.h"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.hpp"
    "${CMAKE_SOURCE_DIR}/include/moe/kernels/arch35/arch35_common.hpp"
    "${CMAKE_SOURCE_DIR}/include/moe/kernels/arch35/launch.hpp"
    "${CMAKE_SOURCE_DIR}/include/moe/kernels/arch35/status.hpp"
    "${CMAKE_SOURCE_DIR}/include/moe/kernels/arch35/tiling_abi.hpp"
  )
  set(_dsv4_asc_objects)
  foreach(_source IN LISTS _dsv4_asc_sources)
    get_filename_component(_source_name "${_source}" NAME_WE)
    set(_object
        "${CMAKE_CURRENT_BINARY_DIR}/dsv4_camodel/${_source_name}.dav-3510.o")
    add_custom_command(
      OUTPUT "${_object}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${CMAKE_CURRENT_BINARY_DIR}/dsv4_camodel"
      COMMAND "${ASCEND_HOME_PATH}/bin/bisheng"
              -O2 --npu-arch=dav-3510 --asc-aicore-lang
              -DDSV4_TILING_TEST_OVERRIDE=1
              "-I${CMAKE_SOURCE_DIR}/include"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
              "-I${CMAKE_SOURCE_DIR}/src/kernels/arch35"
              "-I${ASCEND_HOME_PATH}/include"
              -c "${_source}" -o "${_object}"
      DEPENDS "${_source}" ${_dsv4_asc_headers}
      VERBATIM
      COMMENT "Compiling ${_source_name} for dav-3510 with BiSheng")
    list(APPEND _dsv4_asc_objects "${_object}")
  endforeach()
  set_source_files_properties(${_dsv4_asc_objects}
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_custom_target(dsv4_moe_expert_camodel_objects
                    DEPENDS ${_dsv4_asc_objects})

  add_executable(test_dsv4_moe_expert_sim
    test_dsv4_moe_expert_sim.cpp
    ${_dsv4_asc_objects}
  )
  add_dependencies(test_dsv4_moe_expert_sim dsv4_moe_expert_camodel_objects)
  target_include_directories(test_dsv4_moe_expert_sim PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_expert_sim PRIVATE -Wall -Wextra)
  target_compile_definitions(test_dsv4_moe_expert_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_expert_sim PRIVATE
    moe_emu
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
    ${MOE_CANN_LIBRARIES}
  )
  set_target_properties(test_dsv4_moe_expert_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_expert_sim PRIVATE -Werror)
  endif()
  # The whole-block entry points. Reuses the SAME ASC objects -- both suites
  # link one arch35 device object set, so the two kernels are compiled once.
  add_executable(test_dsv4_moe_block_sim
    test_dsv4_moe_block_sim.cpp
    ${_dsv4_asc_objects}
  )
  add_dependencies(test_dsv4_moe_block_sim dsv4_moe_expert_camodel_objects)
  # The suite runs each launch under a hang watchdog on its own thread: an
  # unbalanced event handshake in the tile ring presents as a hang, and
  # without this the only symptom is ctest expiring with no launch named.
  find_package(Threads REQUIRED)
  target_link_libraries(test_dsv4_moe_block_sim PRIVATE Threads::Threads)
  target_include_directories(test_dsv4_moe_block_sim PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_block_sim PRIVATE -Wall -Wextra)
  target_compile_definitions(test_dsv4_moe_block_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_block_sim PRIVATE
    moe_emu
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
    ${MOE_CANN_LIBRARIES}
  )
  set_target_properties(test_dsv4_moe_block_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_block_sim PRIVATE -Werror)
  endif()
  # TIMEOUT is 6 hours. MEASURED, 2026-10-15: this suite takes ~4,930 s
  # (~82 min) at hidden = inter = 512, top_k = 2, tokens = 2, block_dim = 2,
  # over ~1.5 M simulated cycles. So the budget is ~4.4x headroom, which is
  # about right for a shared host -- do not tune it to the measurement.
  #
  # DO NOT PREDICT THIS NUMBER FROM THE SINGLE-EXPERT SUITE. Two attempts at
  # a closed-form cost model were both wrong by a wide margin, in opposite
  # directions, and the error in each was extrapolating across KERNELS:
  #
  #   per-core reading   (one core runs a whole expert, 786 K MACs, ~32x the
  #                      retired two-phase kernel's ~24.6 K per core)
  #                      -> predicts ~4.6 h.   Observed ~82 min.
  #   total-work reading (786 K MACs -> ~260 s, 1.57 M -> ~540 s, taken from
  #                      the single-expert suite and scaled linearly)
  #                      -> predicts ~18 min.  Observed ~82 min.
  #
  # The single-expert suite is a DIFFERENT kernel on a DIFFERENT grid (32
  # cores, two-phase) and CAModel joins simulated threads in parallel, so
  # neither its per-core nor its total-work rate carries over. Time this
  # suite; do not derive it.
  #
  # The geometry cannot be tuned down either: the width cannot go below 512,
  # the reduction quantum (kMoeBlockReductionQuantum), because below it a
  # row's group count no longer divides across the 16 lanes.
  #
  # So this is a SOAK gate, not something to run in a tight loop. The fast
  # gates for these kernels are tests/emu (host, seconds) and the BiSheng
  # compile gate; this one answers "does it execute and compute the right
  # answer", which nothing else on a workstation without an NPU can.
  add_test(NAME Dsv4MoeBlockSim COMMAND test_dsv4_moe_block_sim)
  set_tests_properties(Dsv4MoeBlockSim PROPERTIES
    LABELS "sim;arch35;camodel;functional;soak"
    TIMEOUT 21600
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )

  # -------------------------------------------------------------------------
  # Collapsed boundary-grid gate: tests/sim/kernel/ is SIM ONLY
  # -------------------------------------------------------------------------
  # The sim kernel TU textually #includes the production kernel so the compute
  # bodies are verbatim, which means it also defines the production __global__
  # entries. It therefore must NOT be linked beside dsv4_moe_block_kernel.o --
  # this target links the sim objects and the tiling/launch_common TUs only.
  set(_dsv4_sim_kernel_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_block_inductive_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_block_inductive_launch.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )
  set(_dsv4_sim_kernel_objects)
  foreach(_source IN LISTS _dsv4_sim_kernel_sources)
    get_filename_component(_source_name "${_source}" NAME_WE)
    set(_object
        "${CMAKE_CURRENT_BINARY_DIR}/dsv4_sim_kernel/${_source_name}.dav-3510.o")
    add_custom_command(
      OUTPUT "${_object}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${CMAKE_CURRENT_BINARY_DIR}/dsv4_sim_kernel"
      COMMAND "${ASCEND_HOME_PATH}/bin/bisheng"
              -O2 --npu-arch=dav-3510 --asc-aicore-lang
              -DDSV4_TILING_TEST_OVERRIDE=1
              "-I${CMAKE_SOURCE_DIR}/include"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
              "-I${CMAKE_SOURCE_DIR}/src/kernels/arch35"
              "-I${ASCEND_HOME_PATH}/include"
              -c "${_source}" -o "${_object}"
      DEPENDS "${_source}" ${_dsv4_asc_headers}
      VERBATIM
      COMMENT "Compiling SIM-ONLY ${_source_name} for dav-3510 with BiSheng")
    list(APPEND _dsv4_sim_kernel_objects "${_object}")
  endforeach()
  set_source_files_properties(${_dsv4_sim_kernel_objects}
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_custom_target(dsv4_moe_block_sim_kernel_objects
                    DEPENDS ${_dsv4_sim_kernel_objects})

  add_executable(test_dsv4_moe_block_inductive_sim
    test_dsv4_moe_block_inductive_sim.cpp
    ${_dsv4_sim_kernel_objects}
  )
  add_dependencies(test_dsv4_moe_block_inductive_sim
                   dsv4_moe_block_sim_kernel_objects)
  target_include_directories(test_dsv4_moe_block_inductive_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_block_inductive_sim PRIVATE -Wall -Wextra)
  target_compile_definitions(test_dsv4_moe_block_inductive_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_block_inductive_sim PRIVATE
    moe_emu
    Threads::Threads
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
    ${MOE_CANN_LIBRARIES}
  )
  set_target_properties(test_dsv4_moe_block_inductive_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )

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
                "source:\n  ${_prod_source}\n"
                "Verification scaffolding must not link into a production "
                "artifact.")
            endif()
          endforeach()
        endif()
      endif()
    endif()
  endforeach()

  # TIMEOUT 1800. MEASURED: this gate takes ~644 s against the full suite's
  # ~4,932 s -- a 7.7x cut. 1800 s is ~2.8x headroom.
  #
  # IT IS 7.7x AND NOT 16x, AND THE REASON IS WORTH KEEPING. The collapsed
  # grid runs 4 of 16 tiles per phase on one core: ~196 K MACs against the
  # full suite's ~3.14 M over four cores, so a sixteenth of the WORK. Solving
  # the two measurements as t = F + W with a 1/16 work ratio:
  #
  #     W (variable, full grid) ~= 4,574 s
  #     F (fixed: simulator bring-up, binary load, teardown) ~= 358 s
  #
  # F does not shrink. So ~358 s is the FLOOR for any CAModel run of this
  # kernel on this host, the best possible speedup is 4932/358 ~= 13.8x even
  # at zero work, and collapsing further than this buys very little: the
  # variable part is already down to ~286 s.
  #
  # Anyone asked for a "sub-30-second" CAModel gate here should be shown this
  # arithmetic: the fixed cost alone is ~12x that target.
  add_test(NAME Dsv4MoeBlockInductiveSim
           COMMAND test_dsv4_moe_block_inductive_sim)
  set_tests_properties(Dsv4MoeBlockInductiveSim PROPERTIES
    LABELS "sim;arch35;camodel;inductive;functional"
    TIMEOUT 1800
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )

  add_test(NAME Dsv4MoeExpertSim COMMAND test_dsv4_moe_expert_sim)
  set_tests_properties(Dsv4MoeExpertSim PROPERTIES
    LABELS "sim;arch35;camodel;functional"
    TIMEOUT 1800
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )
endif()
