add_executable(test_mhc_fused_sinkhorn_gemv_sim
  test_mhc_fused_sinkhorn_gemv_sim.cpp
)
target_link_libraries(test_mhc_fused_sinkhorn_gemv_sim PRIVATE
  test_common
  cpu_ref
)
target_compile_options(test_mhc_fused_sinkhorn_gemv_sim PRIVATE -Wall -Wextra)
target_compile_definitions(test_mhc_fused_sinkhorn_gemv_sim PRIVATE
  ENABLE_MOCK_RUNTIME=1
)
add_test(NAME MhcFusedSim COMMAND test_mhc_fused_sinkhorn_gemv_sim)
set_tests_properties(MhcFusedSim PROPERTIES LABELS "sim" TIMEOUT 120)
