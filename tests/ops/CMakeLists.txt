add_executable(aclnn_emu_parity_npu aclnn_emu_parity_npu.cpp)
target_link_libraries(aclnn_emu_parity_npu PRIVATE ascend_moe_engine moe_emu)
target_include_directories(aclnn_emu_parity_npu PRIVATE
  "${CMAKE_SOURCE_DIR}/tests/npu"
)
target_compile_options(aclnn_emu_parity_npu PRIVATE -Wall -Wextra -ffp-contract=off)
if(DSV4_WERROR)
  target_compile_options(aclnn_emu_parity_npu PRIVATE -Werror)
endif()
set_target_properties(aclnn_emu_parity_npu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/ops"
  BUILD_RPATH "$ORIGIN/../.."
)

add_test(NAME aclnn_emu_parity_npu COMMAND aclnn_emu_parity_npu)
set_tests_properties(aclnn_emu_parity_npu PROPERTIES
  LABELS "ops;npu;verification"
  TIMEOUT 600
  SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
)
