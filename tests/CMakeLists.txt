# dsv4_operator_hypotheses_test is in BOTH lists on purpose: it is the one
# suite whose claims have to be the same sentence in either build. The
# plan-phase hypotheses run against the symbolic backend here and against
# libcust_opapi.so there; the numeric sections (Birkhoff, index range) hold the
# host reference in a mock build and the device readback in a CANN one. See the
# file header for how it reports the difference.
if(ENABLE_MOCK_RUNTIME)
  set(_tests mock_contract_smoke mock_pipeline_e2e mock_safetensors_index_probe mock_failure_teardown
      vendor_ops_table_test dsv4_operator_hypotheses_test)
  set(_test_label mock)
else()
  set(_tests npu_backend_probe npu_weights_ingest_probe npu_e2e_single_token
      dsv4_operator_hypotheses_test)
  set(_test_label npu)
endif()

foreach(_test IN LISTS _tests)
  add_executable(${_test} ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE ascend_moe_engine)
  target_compile_options(${_test} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_test} PRIVATE -Werror)
  endif()
  set_target_properties(${_test} PROPERTIES BUILD_RPATH "$ORIGIN/..")
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES LABELS "${_test_label}" TIMEOUT 120)
  if(NOT ENABLE_MOCK_RUNTIME)
    set_tests_properties(${_test} PROPERTIES SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected")
  endif()
endforeach()
