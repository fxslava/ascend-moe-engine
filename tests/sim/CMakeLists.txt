# Lightweight hardware-free simulator contract harness. CANN is optional; the
# CPU launch stub keeps this target useful on developer machines without it.
add_executable(sim_moe_test main_sim.cpp test_patterns.cpp)
target_include_directories(sim_moe_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(sim_moe_test PRIVATE cxx_std_17)
target_compile_options(sim_moe_test PRIVATE -Wall -Wextra)
if(DSV4_WERROR)
  target_compile_options(sim_moe_test PRIVATE -Werror)
endif()
set_target_properties(sim_moe_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/sim")
add_test(NAME sim_moe_test COMMAND sim_moe_test)
set_tests_properties(sim_moe_test PROPERTIES LABELS "sim;contract" TIMEOUT 30)
