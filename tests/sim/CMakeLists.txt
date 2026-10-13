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
    "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
  )
  set(_dsv4_asc_headers
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_kernel/dsv4_moe_expert_arch35.h"
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_expert_tiling.h"
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
  add_test(NAME Dsv4MoeExpertSim COMMAND test_dsv4_moe_expert_sim)
  set_tests_properties(Dsv4MoeExpertSim PROPERTIES
    LABELS "sim;arch35;camodel;functional"
    TIMEOUT 1800
    ENVIRONMENT
      "ASCEND_SIMULATOR_MODE=1;LD_LIBRARY_PATH=${_dsv4_sim_root}/camodel:${_dsv4_sim_root}/lib:${MOE_CANN_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
  )
endif()
