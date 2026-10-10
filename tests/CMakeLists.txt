add_subdirectory(common)
add_subdirectory(sim)

if(ENABLE_MOCK_RUNTIME)
 add_subdirectory(mock)
else()
 add_subdirectory(npu)
endif()
