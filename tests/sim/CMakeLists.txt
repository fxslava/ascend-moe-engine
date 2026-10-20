# Flat MoE simulator test: one host runner, one procedural data source.
add_executable(sim_moe_test main_sim.cpp test_patterns.cpp)
target_compile_features(sim_moe_test PRIVATE cxx_std_17)
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
  target_compile_definitions(${_human_device} PRIVATE ENABLE_KERNEL_PRINTF=1 ASCENDC_DUMP=1)
  target_compile_options(${_human_device} PRIVATE --cce-enable-print -UNDEBUG)
  target_compile_definitions(sim_moe_test PRIVATE DSV4_CAMODEL=1)
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
else()
  message(STATUS
    "sim_moe_test: using its hardware-free CPU launch stub (enable "
    "ENABLE_ARCH35_DEVICE_LAUNCH with CANN 9.2 to run CAModel)")
  set_tests_properties(sim_moe_test PROPERTIES
    LABELS "sim;stub;functional"
    TIMEOUT 300)
endif()
