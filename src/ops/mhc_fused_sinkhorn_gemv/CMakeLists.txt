add_library(cpu_ref STATIC
  cpu_ref/mhc_fused_sinkhorn_gemv_ref.cpp
  op_host/mhc_fused_sinkhorn_gemv.cpp
  op_host/mhc_fused_sinkhorn_gemv_tiling.cpp
)
target_include_directories(cpu_ref PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/cpu_ref"
  "${CMAKE_SOURCE_DIR}/include"
)
target_compile_options(cpu_ref PRIVATE -Wall -Wextra)
