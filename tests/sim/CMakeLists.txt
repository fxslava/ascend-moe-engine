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

  add_executable(test_dsv4_moe_expert_sim
    test_dsv4_moe_expert_sim.cpp
    "${_dsv4_tiling_source}"
  )
  target_link_libraries(test_dsv4_moe_expert_sim PRIVATE moe_emu)
  target_include_directories(test_dsv4_moe_expert_sim PRIVATE
    "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
  )
  target_compile_options(test_dsv4_moe_expert_sim PRIVATE
    -Wall -Wextra -ffp-contract=off
  )
  target_compile_definitions(test_dsv4_moe_expert_sim PRIVATE
    DSV4_TILING_TEST_OVERRIDE=1
  )
  if(DSV4_WERROR)
    target_compile_options(test_dsv4_moe_expert_sim PRIVATE -Werror)
  endif()
  add_test(NAME Dsv4MoeExpertSim COMMAND test_dsv4_moe_expert_sim)
  set_tests_properties(Dsv4MoeExpertSim PROPERTIES
    LABELS "sim;arch35;functional"
    TIMEOUT 120
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
