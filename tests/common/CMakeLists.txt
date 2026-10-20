add_library(test_common STATIC
  test_utils.cpp
)
target_include_directories(test_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/src/ops/mhc_fused_sinkhorn_gemv/cpu_ref"
)
target_link_libraries(test_common PUBLIC cpu_ref)
target_compile_options(test_common PRIVATE -Wall -Wextra)
