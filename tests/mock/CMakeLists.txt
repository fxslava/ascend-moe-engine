set(_mock_tests
  mock_contract_smoke
  mock_pipeline_e2e
  mock_safetensors_index_probe
  mock_failure_teardown
  vendor_ops_table_test
  dsv4_operator_hypotheses_test
)

foreach(_test IN LISTS _mock_tests)
  # Suite sources sit next to this file. The shared hypothesis suite is the one
  # exception: it lives in tests/ because the CANN build registers it too.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cpp")
    set(_test_src "${_test}.cpp")
  else()
    set(_test_src "../${_test}.cpp")
  endif()
  add_executable(${_test} "${_test_src}")
  target_link_libraries(${_test} PRIVATE ascend_moe_engine)
  target_compile_options(${_test} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_test} PRIVATE -Werror)
  endif()
  set_target_properties(${_test} PROPERTIES BUILD_RPATH "$ORIGIN/../..")
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES LABELS "mock" TIMEOUT 120)
endforeach()
