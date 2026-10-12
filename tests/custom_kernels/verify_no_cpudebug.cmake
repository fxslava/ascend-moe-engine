# -----------------------------------------------------------------------------
# Checks the acceptance criterion "complete absence of invocations of cpudebug /
# tikicpulib" against the linked artifacts instead of against a promise.
#
# Invoked by ctest with:
#   -D LDD=<path to ldd> -D BINARIES=<;-separated executable paths>
# -----------------------------------------------------------------------------

if(NOT DEFINED LDD OR NOT DEFINED BINARIES)
  message(FATAL_ERROR "verify_no_cpudebug.cmake requires -D LDD and -D BINARIES")
endif()

set(_banned
  cpudebug
  libascendc_cpudebug
  tikicpulib
  tikcpulib
)

set(_violations "")

foreach(_bin IN LISTS BINARIES)
  if(NOT EXISTS "${_bin}")
    message(FATAL_ERROR "expected binary not found: ${_bin}")
  endif()

  execute_process(
    COMMAND "${LDD}" "${_bin}"
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
    RESULT_VARIABLE _rc
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "ldd failed on ${_bin}: ${_err}")
  endif()

  string(TOLOWER "${_out}" _lower)
  foreach(_bad IN LISTS _banned)
    string(FIND "${_lower}" "${_bad}" _at)
    if(NOT _at EQUAL -1)
      list(APPEND _violations "${_bin} links '${_bad}'")
    endif()
  endforeach()

  message(STATUS "clean: ${_bin}")
endforeach()

if(_violations)
  foreach(_v IN LISTS _violations)
    message(SEND_ERROR "${_v}")
  endforeach()
  message(FATAL_ERROR
    "a custom-kernel binary links a host CPU debug runtime; the verification "
    "and benchmark targets must not depend on cpudebug / tikicpulib")
endif()

message(STATUS
  "no host CPU debug runtime linked into any custom-kernel binary")
