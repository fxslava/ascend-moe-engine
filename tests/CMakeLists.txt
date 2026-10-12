add_subdirectory(common)
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

add_subdirectory(custom_kernels)

# dsv4_operator_hypotheses_test.cpp deliberately sits here rather than in either
# suite directory: it is registered by BOTH tests/mock and tests/npu, because it
# is the one suite whose claims have to be the same sentence in either build.
# The plan-phase hypotheses run against the symbolic backend in a mock build and
# against libcust_opapi.so in a CANN one; the numeric sections (Birkhoff, index
# range) hold the host reference in a mock build and the device readback in a
# CANN one. See the file header for how it reports the difference.
if(ENABLE_MOCK_RUNTIME)
 add_subdirectory(mock)
else()
 add_subdirectory(npu)
endif()
