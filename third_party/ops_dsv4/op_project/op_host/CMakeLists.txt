# The thirteen vendored arch35 operators' op_host layer, compiled DIRECTLY into
# the two vendor libraries the CANN planner loads:
#
#   cust_op_proto  -> libcust_opsproto_rt2.0.so  (infershape registration)
#   cust_optiling  -> libcust_opmaster_rt2.0.so  (the tiling phase)
#
# There is deliberately NO opbuild step. The toolkit's op_build generator
# produces aclnn wrappers (this repository ships its own libcust_opapi.so with
# hand-written entry points) and proto glue (the vendored *_proto.cpp files
# self-register through IMPL_OP_INFERSHAPE), and in this 9.1.0-950 image its
# dlopen probe cannot work at all: libop_common.so requires
# ge::GetViewErrorCodeStr, which no shipped library exports. The vendored
# common/src/toolkit_stubs.cpp supplies the one symbol (Ops::Base::ToString)
# that op_common/log/log.h declares and this image defines nowhere.

set(_dsv4 "${CMAKE_CURRENT_SOURCE_DIR}/../..")

set(ops_srcs
    # ---- mhc: ops-transformer trio ----
    ${_dsv4}/mhc/mhc_pre/op_host/mhc_pre_def.cpp
    ${_dsv4}/mhc/mhc_pre/op_host/mhc_pre_infershape.cpp
    ${_dsv4}/mhc/mhc_pre/op_host/op_tiling/arch35/mhc_pre_tiling.cpp
    ${_dsv4}/mhc/mhc_sinkhorn/op_host/mhc_sinkhorn_def.cpp
    ${_dsv4}/mhc/mhc_sinkhorn/op_host/mhc_sinkhorn_infershape.cpp
    ${_dsv4}/mhc/mhc_sinkhorn/op_host/mhc_sinkhorn_tiling.cpp
    ${_dsv4}/mhc/mhc_post/op_host/mhc_post_def.cpp
    ${_dsv4}/mhc/mhc_post/op_host/mhc_post_infershape.cpp
    ${_dsv4}/mhc/mhc_post/op_host/mhc_post_tiling.cpp
    ${_dsv4}/mhc/mhc_post/op_host/mhc_post_tiling_base.cpp
    # ---- mhc: vllm-ascend hc_* family ----
    ${_dsv4}/mhc/hc_pre/op_host/hc_pre_def.cpp
    ${_dsv4}/mhc/hc_pre/op_host/hc_pre_proto.cpp
    ${_dsv4}/mhc/hc_pre/op_host/hc_pre_tiling.cpp
    ${_dsv4}/mhc/hc_pre_inv_rms/op_host/hc_pre_inv_rms_def.cpp
    ${_dsv4}/mhc/hc_pre_inv_rms/op_host/hc_pre_inv_rms_proto.cpp
    ${_dsv4}/mhc/hc_pre_inv_rms/op_host/hc_pre_inv_rms_tiling.cpp
    ${_dsv4}/mhc/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_def.cpp
    ${_dsv4}/mhc/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_proto.cpp
    ${_dsv4}/mhc/hc_pre_sinkhorn/op_host/hc_pre_sinkhorn_tiling.cpp
    ${_dsv4}/mhc/hc_post/op_host/hc_post_def.cpp
    ${_dsv4}/mhc/hc_post/op_host/hc_post_proto.cpp
    ${_dsv4}/mhc/hc_post/op_host/hc_post_tiling.cpp
    # ---- attention ----
    ${_dsv4}/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp
    ${_dsv4}/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_infershape.cpp
    ${_dsv4}/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp
    ${_dsv4}/attention/compressor/op_host/compressor_def.cpp
    ${_dsv4}/attention/compressor/op_host/compressor_proto.cpp
    ${_dsv4}/attention/compressor/op_host/arch35/compressor_tiling.cpp
    ${_dsv4}/attention/vllm_quant_lightning_indexer/op_host/vllm_quant_lightning_indexer_def.cpp
    ${_dsv4}/attention/vllm_quant_lightning_indexer/op_host/vllm_quant_lightning_indexer_infershape.cpp
    ${_dsv4}/attention/vllm_quant_lightning_indexer/op_host/vllm_quant_lightning_indexer_tiling.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_def.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_proto.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_tiling.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_consistancy.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_existance.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_feature.cpp
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp
    ${_dsv4}/attention/kv_compress_epilog/op_host/kv_compress_epilog_def.cpp
    ${_dsv4}/attention/kv_compress_epilog/op_host/kv_compress_epilog_tiling_arch35.cpp
    ${_dsv4}/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_def.cpp
    ${_dsv4}/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_proto.cpp
    ${_dsv4}/attention/indexer_compress_epilog_v2/op_host/indexer_compress_epilog_v2_tiling.cpp
    # ---- toolkit gap stubs, compiled into BOTH vendor libraries ----
    ${_dsv4}/common/src/toolkit_stubs.cpp
)

set(_dsv4_host_dirs "")
foreach(_ops_dsv4_dir
    ${_dsv4}/mhc/mhc_pre/op_host
    ${_dsv4}/mhc/mhc_pre/op_host/op_tiling/arch35
    ${_dsv4}/mhc/mhc_sinkhorn/op_host
    ${_dsv4}/mhc/mhc_post/op_host
    ${_dsv4}/mhc/hc_pre/op_host
    ${_dsv4}/mhc/hc_pre_inv_rms/op_host
    ${_dsv4}/mhc/hc_pre_sinkhorn/op_host
    ${_dsv4}/mhc/hc_post/op_host
    ${_dsv4}/attention/quant_lightning_indexer/op_host
    ${_dsv4}/attention/compressor/op_host
    ${_dsv4}/attention/compressor/op_host/arch35
    ${_dsv4}/attention/vllm_quant_lightning_indexer/op_host
    ${_dsv4}/attention/kv_quant_sparse_attn_sharedkv/op_host
    ${_dsv4}/attention/kv_compress_epilog/op_host
    ${_dsv4}/attention/indexer_compress_epilog_v2/op_host
    ${_dsv4}/common/include
    ${_dsv4}/common/include/external
)
    list(APPEND _dsv4_host_dirs "${_ops_dsv4_dir}")
endforeach()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_dsv4_arch "aarch64-linux")
else()
    set(_dsv4_arch "x86_64-linux")
endif()
foreach(_dsv4_toolkit_inc
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${_dsv4_arch}/include/op_common"
    "${ASCEND_CANN_PACKAGE_PATH}/${_dsv4_arch}/include/op_common/op_host"
    "${ASCEND_CANN_PACKAGE_PATH}/${_dsv4_arch}/pkg_inc/aicpu"
    "${ASCEND_CANN_PACKAGE_PATH}/${_dsv4_arch}/pkg_inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")
    if(EXISTS "${_dsv4_toolkit_inc}")
        list(APPEND _dsv4_host_dirs "${_dsv4_toolkit_inc}")
    endif()
endforeach()

# The toolkit's own libraries. libop_common is deliberately NOT linked: this
# image's copy requires ge::GetViewErrorCodeStr, which nothing exports, and
# everything the tiling sources need from it is covered by libnnopbase.
set(_dsv4_lib_dir "${ASCEND_CANN_PACKAGE_PATH}/${_dsv4_arch}/lib64")
if(NOT EXISTS "${_dsv4_lib_dir}/libexe_graph.so")
    set(_dsv4_lib_dir "${ASCEND_CANN_PACKAGE_PATH}/lib64")
endif()
set(_dsv4_link_libs "")
foreach(_dsv4_lib exe_graph register tiling_api nnopbase graph ge_common_base c_sec)
    if(EXISTS "${_dsv4_lib_dir}/lib${_dsv4_lib}.so")
        list(APPEND _dsv4_link_libs "${_dsv4_lib_dir}/lib${_dsv4_lib}.so")
    endif()
endforeach()

add_library(cust_op_proto SHARED ${ops_srcs})
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
# intf_pub defaults to -std=c++11; the vendored tiling sources use
# std::is_same_v and friends, and a later -std wins on the command line.
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden -std=c++17)
target_include_directories(cust_op_proto PRIVATE ${_dsv4_host_dirs})
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        ${_dsv4_link_libs}
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

add_library(cust_optiling SHARED ${ops_srcs})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden -std=c++17)
target_include_directories(cust_optiling PRIVATE ${_dsv4_host_dirs})
target_link_libraries(cust_optiling PRIVATE
        nnopbase
        intf_pub
        ${_dsv4_link_libs}
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)
