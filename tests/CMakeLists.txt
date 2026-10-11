add_subdirectory(common)
add_subdirectory(sim)

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
