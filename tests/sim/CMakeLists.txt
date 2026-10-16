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

  # The collapsed trip schedule for the Cube prefill gate. Registered here,
  # OUTSIDE the CAModel guard, and deliberately so: the sample-set builder is
  # pure index arithmetic, it decides what the simulator run proves, and
  # section 5.5 puts that class of logic on the host. It needs no toolkit and
  # no device, so it must run everywhere the CI gate runs -- including on a
  # machine that can never build the CAModel target it serves.
  #
  # It compiles the SAME header the device TU includes (kernel/..._sample_set.h
  # is ACL-free for this reason), so a drift between the audited schedule and
  # the executed one is impossible rather than merely unlikely.
  add_executable(test_dsv4_moe_prefill_cube_sample_set_sim
    test_dsv4_moe_prefill_cube_sample_set_sim.cpp
  )
  target_include_directories(test_dsv4_moe_prefill_cube_sample_set_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_compile_options(test_dsv4_moe_prefill_cube_sample_set_sim PRIVATE
    -Wall -Wextra
  )
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_prefill_cube_sample_set_sim PRIVATE
                           -Werror)
  endif()
  add_test(NAME Dsv4MoePrefillCubeSampleSet
           COMMAND test_dsv4_moe_prefill_cube_sample_set_sim)
  set_tests_properties(Dsv4MoePrefillCubeSampleSet PROPERTIES
    LABELS "sim;arch35;contract;inductive"
    TIMEOUT 60
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
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_arch35.h"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_tiling.h"
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

  # -------------------------------------------------------------------------
  # Collapsed MULTI-CORE gate for the gang decode path
  # -------------------------------------------------------------------------
  # Same scaffolding as above, different kernel. It needs its own target
  # because its sim TU also textually includes the production kernel, so the
  # two sim objects define the same production __global__ entries and cannot
  # be linked into one binary.
  #
  # This is the ONLY gate that executes the cross-core barrier: the prefill
  # inductive gate above runs one core, where SyncAll is satisfied trivially.
  set(_dsv4_decode_sim_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_block_decode_inductive_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_block_decode_inductive_launch.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_block_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )
  set(_dsv4_decode_sim_objects)
  foreach(_source IN LISTS _dsv4_decode_sim_sources)
    get_filename_component(_source_name "${_source}" NAME_WE)
    set(_object
        "${CMAKE_CURRENT_BINARY_DIR}/dsv4_decode_sim_kernel/${_source_name}.dav-3510.o")
    add_custom_command(
      OUTPUT "${_object}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${CMAKE_CURRENT_BINARY_DIR}/dsv4_decode_sim_kernel"
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
    list(APPEND _dsv4_decode_sim_objects "${_object}")
  endforeach()
  set_source_files_properties(${_dsv4_decode_sim_objects}
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_custom_target(dsv4_moe_block_decode_sim_kernel_objects
                    DEPENDS ${_dsv4_decode_sim_objects})

  add_executable(test_dsv4_moe_block_decode_inductive_sim
    test_dsv4_moe_block_decode_inductive_sim.cpp
    ${_dsv4_decode_sim_objects}
  )
  add_dependencies(test_dsv4_moe_block_decode_inductive_sim
                   dsv4_moe_block_decode_sim_kernel_objects)
  target_include_directories(test_dsv4_moe_block_decode_inductive_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_block_decode_inductive_sim
                         PRIVATE -Wall -Wextra)
  target_compile_definitions(test_dsv4_moe_block_decode_inductive_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_block_decode_inductive_sim PRIVATE
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
  set_target_properties(test_dsv4_moe_block_decode_inductive_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )

  # TIMEOUT 1800. MEASURED, 2026-10-15: this gate takes **426 s** at
  # hidden = inter = 512, top_k = 2, aiv_count = 4 -> gang_size = 2,
  # block_dim = 4, sampling 4 phase-1 and 5 phase-2 tiles per lane. The
  # simulated task ran cycle 400 -> 88,883 and the run ended with all threads
  # joined, zero excp_log*.dump files and the output guard band intact.
  # 1800 s is ~4.2x headroom, in line with the sibling gates.
  #
  # IT IS FASTER THAN THE SINGLE-CORE PREFILL GATE (644 s) DESPITE RUNNING
  # FOUR CORES, which is worth keeping because it is counter-intuitive and it
  # is the third time a cost prediction for this simulator has been wrong.
  # A pre-run estimate here reasoned "4 cores and ~213 K MACs per core against
  # the prefill gate's 1 core and ~196 K, so it cannot be cheaper" and
  # predicted >600 s. Observed 426 s. The error was the same one the note on
  # Dsv4MoeBlockSim already records: extrapolating across suites. CAModel
  # joins simulated threads in parallel, so added cores are close to free here
  # and only the per-core trip count moved.
  #
  # Do NOT subtract the prefill gate's ~358 s fixed cost from this 426 s to
  # infer a variable part: that figure was measured for a different suite and
  # carrying it across is the very extrapolation that has now failed three
  # times. If this gate's fixed cost is needed, measure it here by running at
  # one sampled trip.
  add_test(NAME Dsv4MoeBlockDecodeInductiveSim
           COMMAND test_dsv4_moe_block_decode_inductive_sim)
  set_tests_properties(Dsv4MoeBlockDecodeInductiveSim PROPERTIES
    LABELS "sim;arch35;camodel;inductive;functional;gang"
    TIMEOUT 1800
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )

  # -------------------------------------------------------------------------
  # Collapsed boundary-grid gate for the CUBE PREFILL path
  # -------------------------------------------------------------------------
  # Third sim kernel, third target, for the same reason as the second: its TU
  # textually includes dsv4_moe_prefill_cube_kernel.cpp so the compute bodies
  # are verbatim production code, which also brings along the production
  # __global__ entry. Two such objects cannot be linked into one binary, so
  # each collapsed gate gets its own executable.
  #
  # This is the ONLY gate that executes a MIX AIC+AIV kernel: the other two
  # are KERNEL_TYPE_MIX_AIV_1_0 and never stand up a KfcServer. What it adds
  # over them is therefore the Cube side -- the AIV->AIC message queue, the
  # Matmul framework's L0A/L0B/L0C staging, C at TPosition::VECIN and A taken
  # from VECOUT -- none of which any other hardware-free gate touches.
  set(_dsv4_prefill_cube_sim_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_inductive_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_inductive_launch.cpp"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_tiling.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )
  set(_dsv4_prefill_cube_sim_objects)
  foreach(_source IN LISTS _dsv4_prefill_cube_sim_sources)
    get_filename_component(_source_name "${_source}" NAME_WE)
    set(_object
        "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim/${_source_name}.dav-3510.o")
    add_custom_command(
      OUTPUT "${_object}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim"
      # ASCEND_HOME_PATH IS PASSED EXPLICITLY, and it is not redundant.
      # BiSheng's ASC plugin loader reads that variable from the ENVIRONMENT
      # and aborts with "PlugIn Err: can not find ASCEND_HOME_PATH" (an
      # LLVM report_fatal_error, exit 134, no diagnostic naming the source) if
      # it is unset. A CMake custom command does not inherit the shell that
      # configured the build, so sourcing set_env.sh before `cmake` is not
      # enough -- it has to be sourced before every `cmake --build` too. The
      # sibling gates above rely on exactly that; this one does not need to.
      COMMAND "${CMAKE_COMMAND}" -E env
              "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
              "ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}"
              "${ASCEND_HOME_PATH}/bin/bisheng"
              -O2 --npu-arch=dav-3510 --asc-aicore-lang
              -DDSV4_TILING_TEST_OVERRIDE=1
              "-I${CMAKE_SOURCE_DIR}/include"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
              "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
              "-I${CMAKE_SOURCE_DIR}/src/kernels/arch35"
              "-I${CMAKE_CURRENT_SOURCE_DIR}/kernel"
              "-I${ASCEND_HOME_PATH}/include"
              -c "${_source}" -o "${_object}"
      DEPENDS "${_source}" ${_dsv4_asc_headers}
              "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_sample_set.h"
              "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"
      VERBATIM
      COMMENT "Compiling SIM-ONLY ${_source_name} for dav-3510 with BiSheng")
    list(APPEND _dsv4_prefill_cube_sim_objects "${_object}")
  endforeach()
  set_source_files_properties(${_dsv4_prefill_cube_sim_objects}
    PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_custom_target(dsv4_moe_prefill_cube_sim_kernel_objects
                    DEPENDS ${_dsv4_prefill_cube_sim_objects})

  # The host-compiled Cube tiling builder, shared with the arch35_device_launch
  # build. Defines dsv4_prefill_cube_cube_tiling; see the module for why that
  # one TU is not built with BiSheng.
  include("${CMAKE_SOURCE_DIR}/cmake/dsv4_prefill_cube_cube_tiling.cmake")

  add_executable(test_dsv4_moe_prefill_cube_inductive_sim
    test_dsv4_moe_prefill_cube_inductive_sim.cpp
    ${_dsv4_prefill_cube_sim_objects}
  )
  add_dependencies(test_dsv4_moe_prefill_cube_inductive_sim
                   dsv4_moe_prefill_cube_sim_kernel_objects)
  target_include_directories(test_dsv4_moe_prefill_cube_inductive_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_prefill_cube_inductive_sim
                         PRIVATE -Wall -Wextra)
  target_compile_definitions(test_dsv4_moe_prefill_cube_inductive_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_prefill_cube_inductive_sim PRIVATE
    dsv4_prefill_cube_cube_tiling
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
  set_target_properties(test_dsv4_moe_prefill_cube_inductive_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )

  # LINK-TIME DRIVER STUBS, and -rpath-link ONLY -- never -rpath (§4.4).
  #
  # libruntime.so's DT_NEEDED closure reaches libascend_hal.so, which ships
  # with the DRIVER and not the toolkit; CANN carries a link-time stub for it
  # so a host with no driver can still link. Without the stub on the link path
  # this target fails with "undefined reference to drvHdcGetCapacity" and a
  # page of sibling drv*/hal* symbols.
  #
  # It goes on -rpath-link and NOT on BUILD_RPATH deliberately: the stub must
  # be invisible at RUN time, or it could satisfy the device probe in place of
  # a real driver -- or, here, in place of the CAModel's own
  # simulator/dav_3510/camodel/libascend_hal.so, which the ctest ENVIRONMENT
  # below puts first on LD_LIBRARY_PATH. That is the whole point of the
  # distinction §4.4 draws.
  #
  # The search list is wider than the root CMakeLists' MOE_CANN_DEVLIB_DIRS,
  # which looks under ${ASCEND_HOME_PATH}/devlib{,/linux/<arch>}. On
  # 9.2.0-beta.2 the stub is under ${ASCEND_HOME_PATH}/<arch>/devlib instead,
  # so the root's list resolves to a directory that exists and does not hold
  # it. Searched here rather than assumed.
  set(_dsv4_devlib_dirs)
  foreach(_candidate
      "${ASCEND_HOME_PATH}/x86_64-linux/devlib"
      "${ASCEND_HOME_PATH}/x86_64-linux/devlib/linux/x86_64"
      "${ASCEND_HOME_PATH}/devlib/linux/x86_64-linux"
      "${ASCEND_HOME_PATH}/devlib")
    if(EXISTS "${_candidate}/libascend_hal.so")
      list(APPEND _dsv4_devlib_dirs "${_candidate}")
    endif()
  endforeach()
  if(NOT _dsv4_devlib_dirs)
    message(FATAL_ERROR
      "No libascend_hal.so link-time stub found under ${ASCEND_HOME_PATH}; "
      "test_dsv4_moe_prefill_cube_inductive_sim cannot link without it. See "
      "CLAUDE.md §4.4.")
  endif()
  foreach(_dir IN LISTS MOE_CANN_LIB_DIRS _dsv4_devlib_dirs)
    target_link_options(test_dsv4_moe_prefill_cube_inductive_sim PRIVATE
                        "LINKER:-rpath-link,${_dir}")
  endforeach()

  # ---------------------------------------------------------------------
  # SINGLE-TILE BRING-UP: one ProcessTokenTile trip, one AIC cluster
  # ---------------------------------------------------------------------
  # Links the SAME device objects as the inductive gate above -- same kernel,
  # same launch wrapper, same tiling TU -- and differs only in its host driver,
  # which passes total_tokens = 16 so the grid collapses to a single full tile
  # at block_dim 1. Nothing is recompiled for it and no kernel source is
  # duplicated, so a pass here is a statement about the production
  # ProcessTokenTile body and not about a reduced copy of it.
  #
  # It exists because the gate above has never completed a trip and cannot
  # attribute its pem_lsu flood to any one of the KfcServer, the two Matmul
  # objects, the L1 ring, the grid stride and the ragged epilogue it stands up
  # together. This target removes every one of those except the Cube path.
  add_executable(test_dsv4_moe_prefill_cube_single_tile_sim
    test_dsv4_moe_prefill_cube_single_tile_sim.cpp
    ${_dsv4_prefill_cube_sim_objects}
  )
  add_dependencies(test_dsv4_moe_prefill_cube_single_tile_sim
                   dsv4_moe_prefill_cube_sim_kernel_objects)
  target_include_directories(test_dsv4_moe_prefill_cube_single_tile_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include"
  )
  target_compile_options(test_dsv4_moe_prefill_cube_single_tile_sim
                         PRIVATE -Wall -Wextra)
  # Must match the flag the device objects were compiled with, or the host and
  # the kernel disagree on which geometries MOE_DSV4_PREFILL_CUBE_ADMISSIBLE
  # accepts and 512/512 is rejected on one side only.
  target_compile_definitions(test_dsv4_moe_prefill_cube_single_tile_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  target_link_libraries(test_dsv4_moe_prefill_cube_single_tile_sim PRIVATE
    dsv4_prefill_cube_cube_tiling
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
  set_target_properties(test_dsv4_moe_prefill_cube_single_tile_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}"
  )
  # -rpath-link ONLY, never -rpath: §4.4, same reasoning as the gate above.
  foreach(_dir IN LISTS MOE_CANN_LIB_DIRS _dsv4_devlib_dirs)
    target_link_options(test_dsv4_moe_prefill_cube_single_tile_sim PRIVATE
                        "LINKER:-rpath-link,${_dir}")
  endforeach()

  add_test(NAME Dsv4MoePrefillCubeSingleTileSim
           COMMAND test_dsv4_moe_prefill_cube_single_tile_sim)
  set_tests_properties(Dsv4MoePrefillCubeSingleTileSim PROPERTIES
    LABELS "sim;arch35;camodel;cube;mix;bringup"
    TIMEOUT 3600
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )

  # Standalone vector conversion check, without KFC or any GEMM work.
  set(_dsv4_dequant_unit_object
      "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim/dsv4_moe_prefill_cube_dequant_unit.dav-3510.o")
  add_custom_command(
    OUTPUT "${_dsv4_dequant_unit_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim"
    COMMAND "${CMAKE_COMMAND}" -E env
            "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
            "ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}"
            "${ASCEND_HOME_PATH}/bin/bisheng"
            -O2 --npu-arch=dav-3510 --asc-aicore-lang
            "-I${CMAKE_SOURCE_DIR}/include"
            "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
            "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
            "-I${CMAKE_SOURCE_DIR}/src/kernels/arch35"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/kernel"
            "-I${ASCEND_HOME_PATH}/include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_dequant_unit.cpp"
            -o "${_dsv4_dequant_unit_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/dsv4_moe_prefill_cube_dequant_unit.cpp"
            "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"
    VERBATIM)
  set_source_files_properties("${_dsv4_dequant_unit_object}"
                              PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  list(GET _dsv4_prefill_cube_sim_objects 3 _dsv4_dequant_launch_common_object)
  add_executable(test_dsv4_moe_prefill_cube_dequant_unit
    test_dsv4_moe_prefill_cube_dequant_unit.cpp "${_dsv4_dequant_unit_object}"
    "${_dsv4_dequant_launch_common_object}")
  add_dependencies(test_dsv4_moe_prefill_cube_dequant_unit
                   dsv4_moe_prefill_cube_sim_kernel_objects)
  target_include_directories(test_dsv4_moe_prefill_cube_dequant_unit PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include")
  target_compile_options(test_dsv4_moe_prefill_cube_dequant_unit PRIVATE -Wall -Wextra)
  target_link_libraries(test_dsv4_moe_prefill_cube_dequant_unit PRIVATE
    moe_emu Threads::Threads
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
    ${MOE_CANN_LIBRARIES})
  set_target_properties(test_dsv4_moe_prefill_cube_dequant_unit PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}")
  foreach(_dir IN LISTS MOE_CANN_LIB_DIRS _dsv4_devlib_dirs)
    target_link_options(test_dsv4_moe_prefill_cube_dequant_unit PRIVATE
                        "LINKER:-rpath-link,${_dir}")
  endforeach()

  # ---------------------------------------------------------------------
  # THE INDUCTIVE ASSEMBLY GATE: one routine of the Cube prefill at a time
  # ---------------------------------------------------------------------
  # Step 1  DequantSubTileVector                        MXFP4/E8M0 in UB
  # Step 2  AsyncLoadTokensChunk + PrepareTokensChunk   token prefetch ring
  #
  # Every case poisons its canary bands (output guard tail, output tile, and a
  # UB sentinel behind the buffers under test), audits its golden on the host
  # before touching the device, and reports a named refusal through a status
  # word instead of the zeroed output an early `return` would leave behind.
  #
  # It is AIV-only on purpose: no REGIST_MATMUL_OBJ, no KfcServer, no Cube. The
  # end-to-end gate below has never completed a trip and floods pem_lsu with
  # small-offset faults; a gate that shares none of that bring-up is what makes
  # "these routines are not the fault site" a statement rather than a guess.
  #
  # ONE BINARY FOR ALL STEPS, and the reason is the measurement below: the wall
  # time is dominated by simulator bring-up, not by the launches. Add a shape or
  # a routine here; do not add a second executable.
  set(_dsv4_dequant_micro_object
      "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim/test_dequant_micro_kernel.dav-3510.o")
  add_custom_command(
    OUTPUT "${_dsv4_dequant_micro_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/dsv4_prefill_cube_sim"
    COMMAND "${CMAKE_COMMAND}" -E env
            "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
            "ASCEND_TOOLKIT_HOME=${ASCEND_HOME_PATH}"
            "${ASCEND_HOME_PATH}/bin/bisheng"
            -O2 --npu-arch=dav-3510 --asc-aicore-lang
            "-I${CMAKE_SOURCE_DIR}/include"
            "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
            "-I${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel"
            "-I${CMAKE_SOURCE_DIR}/src/kernels/arch35"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/kernel"
            "-I${ASCEND_HOME_PATH}/include"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_dequant_micro_kernel.cpp"
            -o "${_dsv4_dequant_micro_object}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/test_dequant_micro_kernel.cpp"
            ${_dsv4_asc_headers}
            "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_prefill_cube_kernel.cpp"
    VERBATIM
    COMMENT "Compiling SIM-ONLY test_dequant_micro_kernel.cpp for dav-3510 with BiSheng")
  set_source_files_properties("${_dsv4_dequant_micro_object}"
                              PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  add_executable(test_dequant_micro_sim
    test_dequant_micro_sim.cpp "${_dsv4_dequant_micro_object}"
    "${_dsv4_dequant_launch_common_object}")
  add_dependencies(test_dequant_micro_sim
                   dsv4_moe_prefill_cube_sim_kernel_objects)
  target_include_directories(test_dequant_micro_sim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
    "${ASCEND_HOME_PATH}/include")
  target_compile_options(test_dequant_micro_sim PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(test_dequant_micro_sim PRIVATE -Werror)
  endif()
  target_link_libraries(test_dequant_micro_sim PRIVATE
    moe_emu Threads::Threads
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascendc_runtime.a"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libruntime.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libprofapi.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libunified_dlog.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libmmpa.so"
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
    ${MOE_CANN_LIBRARIES})
  set_target_properties(test_dequant_micro_sim PROPERTIES
    BUILD_RPATH "${MOE_CANN_LIB_DIR}")
  foreach(_dir IN LISTS MOE_CANN_LIB_DIRS _dsv4_devlib_dirs)
    target_link_options(test_dequant_micro_sim PRIVATE
                        "LINKER:-rpath-link,${_dir}")
  endforeach()

  # MEASURED on this workstation, 2026-10-16.
  #
  #   Step 1 only, 3 sub-tiles          274.6 s / 290.6 s over two runs
  #   Step 1 + Step 2, 6 launches       469.6 s
  #
  # Every word bit-exact in each, and no pem_lsu diagnostic in any of them.
  # Nearly all of the wall time is FIXED cost -- simulator bring-up, binary load
  # and teardown -- because the launches themselves are tens of thousands of
  # cycles. That is why all steps share this one binary: another shape here is
  # nearly free, another executable is another ~300 s.
  #
  # 1500 s is ~3.2x the current measurement. Re-measure when a step is added
  # rather than letting the budget drift into the range where a hang sits
  # undetected for half an hour.
  add_test(NAME Dsv4PrefillCubeDequantMicroSim COMMAND test_dequant_micro_sim)
  set_tests_properties(Dsv4PrefillCubeDequantMicroSim PROPERTIES
    LABELS "sim;arch35;camodel;functional;vector;dequant;prefetch"
    TIMEOUT 1500
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )

  # ---------------------------------------------------------------------
  # THIS GATE DOES NOT PASS YET, AND THAT IS THE POINT OF REGISTERING IT.
  # ---------------------------------------------------------------------
  # MEASURED, 2026-10-15, first ever execution of dsv4_moe_prefill_cube on
  # anything. The CAModel brings up a simulated Ascend950PR (dav_3510, 32
  # cores) and the launch starts, then:
  #
  #   * at ONE sampled tile, over both launches: killed at 1800 s without
  #     completing. From [TASK_BEGIN] onward the simulator floods
  #     `[ERROR] pem_lsu.cc:484 mem_map_calc_49bit_op0 invalid ldst addr`
  #     with a REPEATING cycle of small addresses (4, 8, 12 ... 60, 72, 76,
  #     120, 124) -- the shape of struct field offsets against a base of
  #     zero -- and never terminates.
  #   * at ZERO sampled tiles (the 0xFFFFFFFF setup-only diagnostic): NOT ONE
  #     pem_lsu error, so the fault is produced by ProcessTokenTile and not by
  #     the MIX bring-up. That run did not return either, but its hang is most
  #     likely an artefact of the diagnostic rather than a kernel defect: the
  #     AIC's REGIST_CUBE_OBJ does ClearWorkspace() on the KFC queue before
  #     entering `while (server.isRun())`, and with no tiles to run the AIV
  #     posts its quit and exits before that clear, which would zero the flag
  #     the server is waiting on. Stated as a hypothesis, not a finding.
  #
  # So the TIMEOUT below is NOT derived from a completed run -- no trip of this
  # kernel has ever completed, so there is no per-trip cost to measure and
  # nothing to extrapolate. 7200 s is a holding value chosen to be clearly
  # longer than the 1800 s that was already insufficient; replace it with a
  # measurement once the kernel executes.
  #
  # The gate is registered red on purpose. It sits behind
  # ENABLE_DSV4_CAMODEL_TEST (OFF by default), so it does not break a normal
  # build, and a gate that fails on a broken kernel is a working gate. It goes
  # green when the kernel does.
  #
  # NOTE ON COST, for whoever fixes this: the dominant simulated work per tile
  # is the FULL re-dequantization of the expert's W1/W3/W2 -- 3 x 512 rows at
  # the gate's geometry, each row a DataCopy / Cast / 16 x Muls / DataCopy
  # sandwiched between three SetFlag/WaitFlag pairs. ProcessTokenTile does that
  # on EVERY tile, including consecutive tiles of the same expert, which the
  # operator's own tiling header says it does not ("the dequantization is
  # O(weights) and runs ONCE per expert"). Making that claim true would cut
  # this gate's cost by the number of tiles per expert as a side effect.
  add_test(NAME Dsv4MoePrefillCubeInductiveSim
           COMMAND test_dsv4_moe_prefill_cube_inductive_sim)
  set_tests_properties(Dsv4MoePrefillCubeInductiveSim PROPERTIES
    LABELS "sim;arch35;camodel;inductive;functional;cube;mix"
    TIMEOUT 7200
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
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
