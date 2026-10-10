set(_npu_tests
  npu_backend_probe
  npu_weights_ingest_probe
  npu_e2e_single_token
)

foreach(_test IN LISTS _npu_tests)
  add_executable(${_test} ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE ascend_moe_engine)
  target_compile_options(${_test} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_test} PRIVATE -Werror)
  endif()
  set_target_properties(${_test} PROPERTIES BUILD_RPATH "$ORIGIN/..")
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES
    LABELS "npu"
    TIMEOUT 120
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
endforeach()
