add_subdirectory(common)

if(NOT TARGET moe_emu)
  if(NOT DEFINED MOE_EMU_WARNINGS)
    set(MOE_EMU_WARNINGS -Wall -Wextra)
  endif()
  if(NOT DEFINED MOE_EMU_ARCH_FLAGS)
    set(MOE_EMU_ARCH_FLAGS "")
  endif()
  if(NOT DEFINED MOE_EMU_OPENMP)
    set(MOE_EMU_OPENMP OFF)
  endif()
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/emu" "${CMAKE_BINARY_DIR}/src/emu")
endif()

add_subdirectory(sim)

# The two decoupled custom-kernel binaries (verification + benchmark).
#
# They need moe_emu, which supplies the in-memory CPU golden. The root
# CMakeLists.txt only adds src/emu inside its ENABLE_HOST_EMU branch (which
# then return()s), so in a mock or CANN build the target does not exist yet and
# has to be brought in here. moe_emu links nothing but libstdc++/libm, so this
# adds no backend dependency to either of those builds.
if(NOT TARGET moe_emu)
  # src/emu/CMakeLists.txt reads these; they are only set in the host-emu
  # branch. Empty is the right value elsewhere -- an empty compile-option list
  # is a no-op, and -ffp-contract=off is applied there unconditionally.
  if(NOT DEFINED MOE_EMU_WARNINGS)
    set(MOE_EMU_WARNINGS -Wall -Wextra)
  endif()
  if(NOT DEFINED MOE_EMU_ARCH_FLAGS)
    set(MOE_EMU_ARCH_FLAGS "")
  endif()
  if(NOT DEFINED MOE_EMU_OPENMP)
    set(MOE_EMU_OPENMP OFF)
  endif()
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/emu" "${CMAKE_BINARY_DIR}/src/emu")
endif()

# The host emulation suite -- the golden oracle (CLAUDE.md section 4). In a
# host-emu build the root CMakeLists adds this itself and then return()s, so
# this call is only ever reached in a mock or CANN build, where the suite has
# not been added yet. It needs nothing but moe_emu, bootstrapped above.
add_subdirectory(emu)

# Timing only, never a gate (CLAUDE.md section 4).
add_subdirectory(bench)

if(ENABLE_MOCK_RUNTIME)
  add_subdirectory(mock)
else()
  add_subdirectory(npu)
  add_subdirectory(ops)
endif()
