# -----------------------------------------------------------------------------
# arch35_device.cmake -- build the five arch35 kernels and the host launch
# layer with BiSheng, and expose them as the `arch35_device_launch` target.
#
# Include this only when ENABLE_ARCH35_DEVICE_LAUNCH is ON. It requires a CANN
# 9.2 toolkit (ASCEND_HOME_PATH set in the environment) and BiSheng; it does
# NOT require an NPU to be present, because configuring and building are
# independent of device availability.
#
# WHY THE ASC LANGUAGE IS USED FOR THE LAUNCH LAYER TOO
# src/ops/common/launch_common.cpp dispatches with the triple-chevron syntax
# (`kernel<<<blockDim, l2ctrl, stream>>>`), which only BiSheng can lower. The
# alternative -- ACLRT_LAUNCH_KERNEL / `aclrtlaunch_<kernel>` host stubs -- is
# NOT available in CANN 9.2.0-beta.2: the ASC flow passes -DGEN_ACLRT=<dir>
# but nothing in the toolchain consumes it, so no stub is emitted and the link
# fails with five undefined symbols. Verified against this toolkit, not
# assumed.
#
# Host TUs (the verification harness) stay on the normal host compiler; only
# this library is built with BiSheng. Mixing the two at link time works.
# -----------------------------------------------------------------------------

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR
    "ENABLE_ARCH35_DEVICE_LAUNCH requires the ASCEND_HOME_PATH environment "
    "variable (the CANN toolkit root).")
endif()

set(_cann "$ENV{ASCEND_HOME_PATH}")

# asc-config.cmake sits directly in <arch>/lib64/cmake/, not in a
# lib64/cmake/asc/ subdirectory, so CMake's normal config search misses it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_cann_arch "${_cann}/aarch64-linux")
else()
  set(_cann_arch "${_cann}/x86_64-linux")
endif()
set(asc_DIR "${_cann_arch}/lib64/cmake")

find_package(asc REQUIRED)

# CMakeASCInformation appends --npu-arch=${CMAKE_ASC_ARCHITECTURES}, so this
# must be set before the language is enabled.
set(CMAKE_ASC_ARCHITECTURES "${ARCH35_NPU_ARCH}")

# Deliberately NOT adding cpp to CMAKE_ASC_SOURCE_FILE_EXTENSIONS: that would
# make every .cpp in the project a device TU, including the host harness.
# ascendc_library() marks its own sources LANGUAGE ASC individually.
enable_language(ASC)

# The source list comes from the operator directories themselves, which each
# publish their own TUs (CLAUDE.md section 1); src/ops/CMakeLists.txt collects
# them. Per operator that is three ASC TUs -- the device kernel in op_kernel/,
# the chevron launch wrapper in op_host/, and the host-side tiling builder it
# calls -- plus the shared launch_common.cpp.
include("${CMAKE_SOURCE_DIR}/cmake/arch35_ops.cmake")

ascendc_library(arch35_device_launch STATIC
  ${MOE_ARCH35_OP_ASC_SOURCES}
)

# dsv4_moe_prefill_cube's launch wrapper calls MakeDsv4MoePrefillCubeGemmTiling
# to build the two TCubeTilings it stages for REGIST_MATMUL_OBJ. That builder
# is the one op_host TU compiled by the HOST compiler rather than BiSheng, so
# it cannot live in MOE_ARCH35_OP_ASC_SOURCES and is linked in as its own
# target. PUBLIC so consumers of arch35_device_launch resolve the symbol too.
include("${CMAKE_SOURCE_DIR}/cmake/dsv4_prefill_cube_cube_tiling.cmake")
target_link_libraries(arch35_device_launch PUBLIC dsv4_prefill_cube_cube_tiling)
# Each operator's private headers (op_kernel/ and op_host/) are on the include
# path so the bare-filename includes the layout mandates resolve.
ascendc_include_directories(arch35_device_launch PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  ${MOE_ARCH35_OP_INCLUDE_DIRS}
)

# Consumers get the public header and <acl/acl.h>; ascendc_library() already
# attached the CANN runtime link interface (ascendcl, runtime, ...).
target_include_directories(arch35_device_launch INTERFACE
  "${CMAKE_SOURCE_DIR}/include"
  "${_cann_arch}/include"
)

# The harness compiles its device cases only when this is defined.
target_compile_definitions(arch35_device_launch INTERFACE
  MOE_KERNEL_LAUNCH_LAYER_AVAILABLE=1
)

# libruntime.so has a DT_NEEDED on libascend_hal.so, which ships with the NPU
# DRIVER, not the toolkit. On a machine with no driver installed the link fails
# with "libascend_hal.so, needed by libruntime.so, not found". CANN ships
# driver stubs under <arch>/devlib for exactly this case.
#
# -rpath-link (NOT -rpath) is deliberate: it resolves the transitive dependency
# at LINK time only and puts nothing into the binary's runtime search path, so
# a stub can never be silently loaded instead of the real driver. Running still
# requires a genuine driver and device; this only makes the build possible on a
# driver-less workstation.
if(EXISTS "${_cann_arch}/devlib/linux/x86_64")
  set(_hal_stub_dir "${_cann_arch}/devlib/linux/x86_64")
elseif(EXISTS "${_cann_arch}/devlib")
  set(_hal_stub_dir "${_cann_arch}/devlib")
endif()

if(DEFINED _hal_stub_dir)
  target_link_options(arch35_device_launch INTERFACE
    "-Wl,-rpath-link,${_hal_stub_dir}"
  )
  message(STATUS "  driver stubs (link-time only): ${_hal_stub_dir}")
endif()

message(STATUS "arch35 device launch: ON (npu-arch=${ARCH35_NPU_ARCH})")
message(STATUS "  CANN: ${_cann}")
message(STATUS "  NOTE: builds without an NPU; the device test cases skip at "
               "runtime when no device is present.")
