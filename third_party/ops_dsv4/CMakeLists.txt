# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
#
# The consolidated DeepSeek-V4-Flash arch35 (Ascend 950PR) operator set.
#
# PROVENANCE
#   mhc/{mhc_pre,mhc_sinkhorn,mhc_post}       ops-transformer
#   attention/quant_lightning_indexer         ops-transformer
#   mhc/{hc_pre,hc_pre_inv_rms,
#        hc_pre_sinkhorn,hc_post}             vllm-ascend csrc/moe
#   attention/compressor                      vllm-ascend csrc/attention
#   attention/vllm_quant_lightning_indexer    vllm-ascend csrc/attention
#   attention/kv_quant_sparse_attn_sharedkv   vllm-ascend csrc/attention
#   attention/kv_compress_epilog              vllm-ascend csrc/attention
#   attention/indexer_compress_epilog_v2      vllm-ascend csrc/attention
#
#   THE TWO mHC FAMILIES COEXIST ON PURPOSE. mhc_* is the ops-transformer
#   decomposition (MhcPre -> MhcSinkhorn -> MhcPost), where the Sinkhorn
#   normalization is a separate in-place operator. hc_* is the vllm-ascend
#   decomposition of the same mapping, in which HcPre fuses the whole prologue
#   -- reciprocal RMS, mixing projection and Sinkhorn -- into one launch, and
#   HcPreInvRms + a mixing GEMM + HcPreSinkhorn is the staged alternative.
#   Neither family's Sinkhorn stage exists as a standalone tensor in the hc_*
#   form, which is what removes the manual-4.31 self-copy hazard that
#   mhc_sinkhorn can only work around. See README.md for which to prefer.
#
#   The four ops-transformer trees arrive with a hand-written op_api layer.
#   The nine vllm-ascend trees do NOT: upstream reaches them through torch-npu
#   EXEC_NPU_CMD, which dlsyms the aclnn entry points that the CANN custom-op
#   package generator emits from op_host/*_def.cpp. This repository is
#   torch-free, so their op_host/op_api/ layer is written here by hand against
#   the operator definitions, in the same shape as the ops-transformer ones.
#
# WHAT IS BUILT HERE
#   cust_opapi -> libcust_opapi.so : the thirteen operators' op_api layer (the
#   aclnn*GetWorkspaceSize / aclnn* C entry points and their l0op builders),
#   compiled against the CANN toolkit the engine itself is built against and
#   linked as a DT_NEEDED library of libascend_moe_engine.so. OpTable resolves
#   the entry points out of it via dlsym(RTLD_DEFAULT, ...); the library lands
#   in the build root so the $ORIGIN-relative rpaths of moe_runner and the
#   tests resolve it without LD_LIBRARY_PATH.
#
# WHAT IS VENDORED BUT NOT COMPILED HERE
#   op_host/ (proto defs, infershape, tiling, parameter checks) and
#   op_kernel/arch35/ (Ascend C kernels) deploy through the opp/vendor package
#   flow: they are cross-compiled by the toolkit's opbuild/ccec machinery into
#   binaries + libcust_opproto.so + libcust_tiling.so registered under
#   $ASCEND_OPP_PATH/vendors. DSV4_VENDOR_ASCENDC_KERNELS=ON wires the
#   in-tree kernel compilation when a toolkit that provides the
#   ascendc_library() CMake command is used (see the option below).
#
#   Because the tiling and proto registrations live in that package rather
#   than in this library, the hand-written l0op builders here size their
#   outputs from the caller descriptors instead of calling INFER_SHAPE. That
#   also suits the engine, which sizes its own static arena slots.
#
# MOCK MODE
#   With ENABLE_MOCK_RUNTIME=ON this directory contributes nothing: the mock
#   stubs for all thirteen entry points live in the engine's mock/ tree and fold
#   into libopapi_mock.so, which keeps the whole symbolic build free of any
#   CANN dependency.

if(ENABLE_MOCK_RUNTIME)
  message(STATUS "ops_dsv4: mock runtime selected; the vendored operator "
                 "stubs are provided by mock/mock_ops.cpp inside libopapi_mock.so")
  return()
endif()

# The parent sets ASCEND_HOME_PATH (CACHE) before adding this directory; the
# fallback keeps this file usable when configured standalone.
if(NOT ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  set(ASCEND_HOME_PATH "${ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit install root")
endif()

if(NOT EXISTS "${ASCEND_HOME_PATH}/include/aclnn/aclnn_base.h")
  message(FATAL_ERROR
    "ops_dsv4: no CANN toolkit at ${ASCEND_HOME_PATH} (expected "
    "${ASCEND_HOME_PATH}/include/aclnn/aclnn_base.h). Source set_env.sh or pass "
    "-DASCEND_HOME_PATH=<toolkit root>.")
endif()

set(_ops_dsv4 "${CMAKE_CURRENT_SOURCE_DIR}")

# The op_api layer: one aclnn entry-point file and one l0op builder per op.
#
# DSV4_VENDOR_OPS is the single list every other reference derives from -- the
# sources, the per-op include roots and, with DSV4_VENDOR_ASCENDC_KERNELS, the
# kernel entry files. Each entry is "<subtree>|<op_file_stem>".
set(DSV4_VENDOR_OPS
  "mhc/mhc_pre|mhc_pre"
  "mhc/mhc_sinkhorn|mhc_sinkhorn"
  "mhc/mhc_post|mhc_post"
  "mhc/hc_pre|hc_pre"
  "mhc/hc_pre_inv_rms|hc_pre_inv_rms"
  "mhc/hc_pre_sinkhorn|hc_pre_sinkhorn"
  "mhc/hc_post|hc_post"
  "attention/quant_lightning_indexer|quant_lightning_indexer"
  "attention/compressor|compressor"
  "attention/vllm_quant_lightning_indexer|vllm_quant_lightning_indexer"
  "attention/kv_quant_sparse_attn_sharedkv|kv_quant_sparse_attn_sharedkv"
  "attention/kv_compress_epilog|kv_compress_epilog"
  "attention/indexer_compress_epilog_v2|indexer_compress_epilog_v2"
)

set(CUST_OPAPI_SOURCES "")
set(_ops_dsv4_api_includes "")
foreach(_ops_dsv4_entry IN LISTS DSV4_VENDOR_OPS)
  string(REPLACE "|" ";" _ops_dsv4_parts "${_ops_dsv4_entry}")
  list(GET _ops_dsv4_parts 0 _ops_dsv4_subtree)
  list(GET _ops_dsv4_parts 1 _ops_dsv4_stem)
  set(_ops_dsv4_api "${_ops_dsv4}/${_ops_dsv4_subtree}/op_host/op_api")
  foreach(_ops_dsv4_file "aclnn_${_ops_dsv4_stem}.cpp" "${_ops_dsv4_stem}.cpp")
    if(NOT EXISTS "${_ops_dsv4_api}/${_ops_dsv4_file}")
      message(FATAL_ERROR
        "ops_dsv4: ${_ops_dsv4_subtree} is listed in DSV4_VENDOR_OPS but "
        "${_ops_dsv4_api}/${_ops_dsv4_file} is missing. Re-vendor the subtree or "
        "drop it from the list.")
    endif()
    list(APPEND CUST_OPAPI_SOURCES "${_ops_dsv4_api}/${_ops_dsv4_file}")
  endforeach()
  list(APPEND _ops_dsv4_api_includes "${_ops_dsv4_api}")
endforeach()

add_library(cust_opapi SHARED ${CUST_OPAPI_SOURCES})

# Include roots, in resolution order:
#   per-op op_api dir      aclnn_<op>.h and <op>.h (local prototypes)
#   common/include         external/aclnn_kernels/*, err/ops_err.h (vendored)
#   common/include/external  plain "aclnn_util.h" (vendored upstream copy)
#   toolkit include        acl/, aclnn/, aclnn_kernels/, register/, platform/
#   toolkit include/aclnn  opdev/* (op_executor, make_op_executor, ...)
#   arch-qualified op_common  log/log.h, op_host util/shape_util.h
target_include_directories(cust_opapi PRIVATE
  ${_ops_dsv4_api_includes}
  "${_ops_dsv4}/common/include"
  "${_ops_dsv4}/common/include/external"
  "${ASCEND_HOME_PATH}/include"
  "${ASCEND_HOME_PATH}/include/aclnn"
  "${ASCEND_HOME_PATH}/include/aclnn/opdev"
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ops_dsv4_arch "aarch64-linux")
else()
  set(_ops_dsv4_arch "x86_64-linux")
endif()
foreach(_ops_dsv4_candidate
    "${ASCEND_HOME_PATH}/${_ops_dsv4_arch}/include/op_common"
    "${ASCEND_HOME_PATH}/${_ops_dsv4_arch}/include/op_common/op_host"
    "${ASCEND_HOME_PATH}/${_ops_dsv4_arch}/pkg_inc/aicpu"  # opdev/aicpu/* include aicpu_engine_struct.h bare
    "${ASCEND_HOME_PATH}/${_ops_dsv4_arch}/pkg_inc")
  if(EXISTS "${_ops_dsv4_candidate}")
    target_include_directories(cust_opapi PRIVATE "${_ops_dsv4_candidate}")
  endif()
endforeach()

# Locate the toolkit libraries with the same arch-qualified preference the
# engine's root CMakeLists uses (an x86_64 build host must not pick up an
# aarch64 copy or the reverse).
set(_ops_dsv4_lib_dirs "")
foreach(_ops_dsv4_candidate
    "${ASCEND_HOME_PATH}/${_ops_dsv4_arch}/lib64"
    "${ASCEND_HOME_PATH}/lib64")
  if(EXISTS "${_ops_dsv4_candidate}/libascendcl.so")
    list(APPEND _ops_dsv4_lib_dirs "${_ops_dsv4_candidate}")
  endif()
endforeach()
if(NOT _ops_dsv4_lib_dirs)
  message(FATAL_ERROR "ops_dsv4: no libascendcl.so for ${CMAKE_SYSTEM_PROCESSOR} under ${ASCEND_HOME_PATH}")
endif()

# op_api calls the aclnnBase framework (CREATE_EXECUTOR / CommonOpExecutorRun),
# the l0op helpers (l0op::Contiguous / ViewCopy / ...) and the aclTensor C++
# accessors -- all exported by the toolkit's own libopapi / libnnopbase /
# libascendcl. libc_sec backs the securec.h memcpy_s calls.
set(_ops_dsv4_link_libs "")
foreach(_ops_dsv4_name ascendcl nnopbase opapi runtime c_sec)
  find_library(_ops_dsv4_lib_${_ops_dsv4_name} NAMES ${_ops_dsv4_name} PATHS ${_ops_dsv4_lib_dirs} NO_DEFAULT_PATH)
  if(_ops_dsv4_lib_${_ops_dsv4_name})
    list(APPEND _ops_dsv4_link_libs "${_ops_dsv4_lib_${_ops_dsv4_name}}")
  else()
    message(STATUS "ops_dsv4: lib${_ops_dsv4_name}.so absent from ${_ops_dsv4_lib_dirs}; skipping")
  endif()
endforeach()
target_link_libraries(cust_opapi PRIVATE ${_ops_dsv4_link_libs})

# The engine reaches these entry points with dlsym, not direct calls, so the
# library must survive --as-needed on the consumers' link lines.
target_link_options(cust_opapi INTERFACE "LINKER:--no-as-needed")
foreach(_ops_dsv4_dir IN LISTS _ops_dsv4_lib_dirs)
  target_link_options(cust_opapi PRIVATE "LINKER:-rpath-link,${_ops_dsv4_dir}")
endforeach()

set_target_properties(cust_opapi PROPERTIES
  OUTPUT_NAME "cust_opapi"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
# Vendored upstream code: keep the engine's warning hygiene out of it -- the
# tree is compiled upstream with its own flag set, and -Wall/-Wextra noise
# here would only invite blanket suppressions.
target_compile_options(cust_opapi PRIVATE -w)
# The toolkit libraries (libopapi, libnnopbase, ...) export their C++ surface
# under the pre-cxx11 libstdc++ string ABI (op::OpDfxGuard::StringToVec,
# ReportErrorMessageInner, op::internal::GetLogApiInfo all mangle with `Ss`,
# not `__cxx11::basic_string`). Upstream ops-transformer builds with the old
# ABI for the same reason; compiling the vendored sources with the host
# default would leave libcust_opapi.so with unsatisfiable undefined symbols.
target_compile_definitions(cust_opapi PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

# -----------------------------------------------------------------------------
# Optional: compile the vendored Ascend C kernels in-tree.
#
# ascendc_library() is provided by the toolkit's op-project template cmake
# (tools/op_project_templates/ascendc/aclnn/cmake), not by the toolkit root,
# and the x86_64 verification host has no NPU to run the result against --
# so kernel compilation is opt-in and OFF by default. The deployment path for
# the 950PR host is the opp/vendor package flow over the same
# vendored sources.
# -----------------------------------------------------------------------------
option(DSV4_VENDOR_ASCENDC_KERNELS "Compile the vendored arch35 Ascend C kernels in-tree" OFF)
set(DSV4_VENDOR_SOC_VERSION "Ascend950PR" CACHE STRING "SoC the vendored kernels compile for")

if(DSV4_VENDOR_ASCENDC_KERNELS)
  # Resolve the kernel entry files FIRST, before the toolkit-capability check
  # below: a missing or renamed kernel source is a vendoring mistake in this
  # tree and should be reported on any toolkit, not hidden behind a host that
  # happens not to ship ascendc_library(). The entry file is <stem>.cpp for
  # most trees; the mHC ops that upstream builds through the APT template carry
  # the _apt suffix instead.
  set(_ops_dsv4_kernel_sources "")
  foreach(_ops_dsv4_entry IN LISTS DSV4_VENDOR_OPS)
    string(REPLACE "|" ";" _ops_dsv4_parts "${_ops_dsv4_entry}")
    list(GET _ops_dsv4_parts 0 _ops_dsv4_subtree)
    list(GET _ops_dsv4_parts 1 _ops_dsv4_stem)
    set(_ops_dsv4_kernel_dir "${_ops_dsv4}/${_ops_dsv4_subtree}/op_kernel")
    unset(_ops_dsv4_kernel)
    foreach(_ops_dsv4_candidate "${_ops_dsv4_stem}.cpp" "${_ops_dsv4_stem}_apt.cpp")
      if(EXISTS "${_ops_dsv4_kernel_dir}/${_ops_dsv4_candidate}")
        set(_ops_dsv4_kernel "${_ops_dsv4_kernel_dir}/${_ops_dsv4_candidate}")
        break()
      endif()
    endforeach()
    if(NOT _ops_dsv4_kernel)
      message(FATAL_ERROR
        "ops_dsv4: no Ascend C kernel entry file for ${_ops_dsv4_subtree} (looked for "
        "${_ops_dsv4_stem}.cpp and ${_ops_dsv4_stem}_apt.cpp under ${_ops_dsv4_kernel_dir}).")
    endif()
    list(APPEND _ops_dsv4_kernel_sources "${_ops_dsv4_kernel}")
  endforeach()
  list(LENGTH _ops_dsv4_kernel_sources _ops_dsv4_kernel_count)
  message(STATUS "ops_dsv4: resolved ${_ops_dsv4_kernel_count} arch35 Ascend C kernel entry files")

  find_file(_ops_dsv4_ascendc_cmake
    NAMES ascendc.cmake func.cmake
    PATHS "${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/aclnn/cmake"
          "${ASCEND_HOME_PATH}/tools/op_project_templates/ascendc/cmake"
    NO_DEFAULT_PATH)
  if(NOT _ops_dsv4_ascendc_cmake)
    message(FATAL_ERROR
      "ops_dsv4: DSV4_VENDOR_ASCENDC_KERNELS=ON but no ascendc cmake module was "
      "found under ${ASCEND_HOME_PATH}/tools/op_project_templates. Kernel compilation "
      "needs a toolkit whose op-project templates ship ascendc_library(); otherwise "
      "deploy the vendored op_kernel/ trees through the opp/vendor package flow.")
  endif()
  include("${_ops_dsv4_ascendc_cmake}")
  if(NOT COMMAND ascendc_library)
    message(FATAL_ERROR "ops_dsv4: ${_ops_dsv4_ascendc_cmake} does not define ascendc_library()")
  endif()

  ascendc_library(cust_opapi_kernels
    SOC_VERSION ${DSV4_VENDOR_SOC_VERSION}
    SOURCES ${_ops_dsv4_kernel_sources}
  )
  add_dependencies(cust_opapi cust_opapi_kernels)
else()
  message(STATUS "ops_dsv4: Ascend C kernels not compiled (DSV4_VENDOR_ASCENDC_KERNELS=OFF); "
                 "host library libcust_opapi.so only")
endif()
