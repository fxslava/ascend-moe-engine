# -----------------------------------------------------------------------------
# Verifies that the host emulation bench really is standalone.
#
# ENABLE_HOST_EMU returns out of the root CMakeLists before any CANN
# discovery, so in principle nothing toolkit-shaped CAN be linked. This script
# checks that against the built binary instead of taking it on trust, because
# "no CANN dependency" is an acceptance criterion and a stray
# target_link_libraries in a future edit would otherwise go unnoticed.
#
# Invoked as: cmake -DBENCH=<path> -P cmake/verify_emu_standalone.cmake
# -----------------------------------------------------------------------------

if(NOT DEFINED BENCH)
  message(FATAL_ERROR "verify_emu_standalone.cmake needs -DBENCH=<binary>")
endif()
if(NOT EXISTS "${BENCH}")
  message(FATAL_ERROR "no such binary: ${BENCH}")
endif()

find_program(LDD_EXECUTABLE ldd)
if(NOT LDD_EXECUTABLE)
  message(STATUS "ldd not found; skipping the linkage check")
  return()
endif()

execute_process(
  COMMAND "${LDD_EXECUTABLE}" "${BENCH}"
  OUTPUT_VARIABLE ldd_output
  ERROR_VARIABLE ldd_error
  RESULT_VARIABLE ldd_status
)
if(NOT ldd_status EQUAL 0)
  message(FATAL_ERROR "ldd ${BENCH} failed: ${ldd_error}")
endif()

# Every library family this tree must NOT pull in.
set(forbidden
  libascendcl
  libopapi
  libnnopbase
  libacl_op_compiler
  libge_runner
  libgraph
  libcust_opapi
  libascend_hal
  libruntime
  libpython
  libtorch
  libc10
)

set(violations "")
foreach(library IN LISTS forbidden)
  if(ldd_output MATCHES "${library}")
    list(APPEND violations "${library}")
  endif()
endforeach()

if(violations)
  string(REPLACE ";" ", " violation_text "${violations}")
  message(FATAL_ERROR
    "the host emu bench links forbidden libraries: ${violation_text}\n"
    "ldd output:\n${ldd_output}")
endif()

# Undefined symbols would mean the static library is expecting something the
# link line never supplied -- e.g. a dlsym'd operator entry point.
execute_process(
  COMMAND "${LDD_EXECUTABLE}" -r "${BENCH}"
  OUTPUT_VARIABLE ldd_r_output
  ERROR_VARIABLE ldd_r_error
  RESULT_VARIABLE ldd_r_status
)
set(combined "${ldd_r_output}${ldd_r_error}")
if(combined MATCHES "undefined symbol")
  message(FATAL_ERROR "the host emu bench has undefined symbols:\n${combined}")
endif()

message(STATUS "emu standalone linkage OK -- no CANN, Python or Torch dependency")
message(STATUS "${ldd_output}")
