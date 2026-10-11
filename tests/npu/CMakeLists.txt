set(_npu_tests
  npu_backend_probe
  npu_weights_ingest_probe
  npu_e2e_single_token
  dsv4_operator_hypotheses_test
)

foreach(_test IN LISTS _npu_tests)
  # Suite sources sit next to this file. The shared hypothesis suite is the one
  # exception: it lives in tests/ because the mock build registers it too.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_test}.cpp")
    set(_test_src "${_test}.cpp")
  else()
    set(_test_src "../${_test}.cpp")
  endif()
  add_executable(${_test} "${_test_src}")
  target_link_libraries(${_test} PRIVATE ascend_moe_engine)
  # npu_test_support.hpp lives here, so the out-of-directory hypothesis suite
  # needs this directory on its include path to reach it.
  target_include_directories(${_test} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(${_test} PRIVATE -Wall -Wextra)
  if(DSV4_WERROR)
    target_compile_options(${_test} PRIVATE -Werror)
  endif()
  set_target_properties(${_test} PROPERTIES BUILD_RPATH "$ORIGIN/../..")
  add_test(NAME ${_test} COMMAND ${_test})
  set_tests_properties(${_test} PROPERTIES
    LABELS "npu"
    TIMEOUT 120
    SKIP_REGULAR_EXPRESSION "Physical Ascend NPU not detected"
  )
endforeach()
