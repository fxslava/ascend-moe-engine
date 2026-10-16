# dsv4_prefill_cube_cube_tiling -- the HOST-compiled Matmul tiling builder for
# dsv4_moe_prefill_cube.
#
# WHY IT IS A SEPARATE, HOST-COMPILED TARGET
# Every other op_host TU of this operator is compiled by BiSheng, because
# CLAUDE.md section 3.3 requires it for the TU holding the chevron launch.
# This one holds no device code and no chevron: it only fills two TCubeTiling
# structs for the host to stage. It is built with the host compiler because it
# includes the toolkit's matmul tiling API, which pulls in
# register/tilingdata_base.h and the graph/protobuf headers behind it -- not
# something to push through the ASC frontend.
#
# WHY IT IS NEEDED AT ALL
# REGIST_MATMUL_OBJ takes interleaved (object, tiling) pairs. Called with the
# objects alone it silently initializes every Matmul with a NULL TCubeTiling,
# which on the dav-3510 CAModel is an unterminating `pem_lsu invalid ldst addr`
# storm at the TCubeTiling field offsets. The derivation and the measured
# offset list are in
# src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_cube_tiling.h.
#
# Included from more than one place (the CAModel sim suite and the
# arch35_device_launch build), so it is idempotent.

if(TARGET dsv4_prefill_cube_cube_tiling)
  return()
endif()

if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR
    "dsv4_prefill_cube_cube_tiling needs ASCEND_HOME_PATH: it links the "
    "toolkit's matmul tiling API.")
endif()

# The include roots are SEARCHED, not assumed. On 9.2.0-beta.2 the matmul
# tiling headers are under asc/include/tiling/matrix, while the register/ and
# kernel_tiling/ headers they include sit in two other trees entirely.
set(_dsv4_cube_tiling_incs
  "${ASCEND_HOME_PATH}/x86_64-linux/asc/include"
  "${ASCEND_HOME_PATH}/x86_64-linux/asc/include/tiling/matrix"
  "${ASCEND_HOME_PATH}/x86_64-linux/asc/include/utils"
  "${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/highlevel_api"
  "${ASCEND_HOME_PATH}/x86_64-linux/include"
)
foreach(_inc IN LISTS _dsv4_cube_tiling_incs)
  if(NOT EXISTS "${_inc}")
    message(FATAL_ERROR
      "matmul tiling API include root ${_inc} not found under "
      "${ASCEND_HOME_PATH}; dsv4_moe_prefill_cube_cube_tiling.cpp cannot be "
      "compiled without it.")
  endif()
endforeach()

# libtiling_api.a holds MatmulApiTiling. It needs optiling::TilingDef from
# libregister, memcpy_s from libc_sec, and libplatform for SocVersion; a static
# archive pulls none of them in by itself, so they are listed explicitly.
set(_dsv4_cube_tiling_libs
  "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libtiling_api.a"
  "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libplatform.so"
  "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libregister.so"
  "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libc_sec.so"
  "${ASCEND_HOME_PATH}/x86_64-linux/lib64/libascend_protobuf.so"
)
foreach(_lib IN LISTS _dsv4_cube_tiling_libs)
  if(NOT EXISTS "${_lib}")
    message(FATAL_ERROR
      "dsv4_prefill_cube_cube_tiling needs ${_lib}, which is not present "
      "under ${ASCEND_HOME_PATH}.")
  endif()
endforeach()

add_library(dsv4_prefill_cube_cube_tiling STATIC
  "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host/dsv4_moe_prefill_cube_cube_tiling.cpp")
target_include_directories(dsv4_prefill_cube_cube_tiling PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/src/ops/dsv4_moe_expert/op_host"
  ${_dsv4_cube_tiling_incs})
target_link_libraries(dsv4_prefill_cube_cube_tiling PUBLIC
  ${_dsv4_cube_tiling_libs})
set_target_properties(dsv4_prefill_cube_cube_tiling PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
