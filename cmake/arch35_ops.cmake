# -----------------------------------------------------------------------------
# arch35_ops.cmake -- the arch35 / dav-3510 operator registry.
#
# One list, and the file paths derived from it by the layout rule in
# CLAUDE.md section 1. Deliberately NOT built out of per-operator
# add_subdirectory() scope: arch35_device.cmake is included from two different
# places in the root CMakeLists (the host-emu branch and the full build), and
# a variable that has to be published upward from a subdirectory would make
# the result depend on which ran first. Deriving the paths here is
# order-independent and fails loudly if a file is missing.
#
# Adding an operator: create src/ops/<op>/ following the layout, add it to
# MOE_ARCH35_OPS below, and add it to scripts/device_compile_kernels.sh.
# -----------------------------------------------------------------------------

set(MOE_ARCH35_OPS
  fused_mhc_pre_norm_sinkhorn
  fused_mhc_post_residual
  paged_compressor_rope
  fast_fp8_indexer_topk
  mxfp4_expert_gemv
  dsv4_moe_expert
)

set(MOE_ARCH35_OP_ASC_SOURCES "")
set(MOE_ARCH35_OP_INCLUDE_DIRS "")

foreach(_op IN LISTS MOE_ARCH35_OPS)
  set(_dir "${CMAKE_SOURCE_DIR}/src/ops/${_op}")
  # Three ASC translation units per operator: the device kernel, the chevron
  # launch wrapper, and the host-side tiling builder the wrapper calls. All
  # three must be compiled by BiSheng (CLAUDE.md section 3.3); the tiling TU
  # is listed here rather than built as a separate host library so that no
  # binary can end up with two copies of it.
  foreach(_src
      "${_dir}/op_kernel/${_op}_kernel.cpp"
      "${_dir}/op_host/${_op}.cpp"
      "${_dir}/op_host/${_op}_tiling.cpp")
    if(NOT EXISTS "${_src}")
      message(FATAL_ERROR
        "arch35 operator '${_op}' is missing ${_src}. The layout in "
        "CLAUDE.md section 1 is mandatory; either add the file or remove the "
        "operator from MOE_ARCH35_OPS.")
    endif()
    list(APPEND MOE_ARCH35_OP_ASC_SOURCES "${_src}")
  endforeach()
  list(APPEND MOE_ARCH35_OP_INCLUDE_DIRS
    "${_dir}/op_kernel"
    "${_dir}/op_host"
  )
endforeach()

# -----------------------------------------------------------------------------
# Additional device-kernel TUs
# -----------------------------------------------------------------------------
# The three-TUs-per-operator derivation above assumes one kernel entry symbol
# per operator. fast_fp8_indexer_topk has three: the fused single-core
# reference and the score/select pair that LaunchFp8IndexerTopK actually
# dispatches, because scoring wants all 32 AIVs and the bitonic selection
# network wants a total order on one. They are extra ENTRY POINTS of one
# operator, not new operators -- one directory, one host ABI, one tiling POD --
# so they are listed here rather than added to MOE_ARCH35_OPS, which would
# demand their own op_host/ and op_kernel/ trees and a second launch wrapper.
#
# Adding one: put it in the owning operator's op_kernel/ and list it here, and
# add it to scripts/device_compile_kernels.sh (CLAUDE.md §5.1).
# dsv4_moe_expert likewise has two entry points, but both live in its single
# op_kernel/<op>_kernel.cpp, so the derivation above already covers it and it
# needs no entry here: the gate_up/down split exists because the down leg
# reduces over the whole `activated` vector while the gate/up legs are split
# across cores by row tile, and the barrier between them is stream ordering.
set(MOE_ARCH35_EXTRA_KERNEL_SOURCES
  "${CMAKE_SOURCE_DIR}/src/ops/fast_fp8_indexer_topk/op_kernel/fast_fp8_indexer_score_kernel.cpp"
  "${CMAKE_SOURCE_DIR}/src/ops/fast_fp8_indexer_topk/op_kernel/fast_fp8_indexer_select_kernel.cpp"
)

foreach(_src IN LISTS MOE_ARCH35_EXTRA_KERNEL_SOURCES)
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR
      "arch35 extra kernel TU ${_src} is listed in "
      "MOE_ARCH35_EXTRA_KERNEL_SOURCES but does not exist. Either add the "
      "file or remove it from the list.")
  endif()
  list(APPEND MOE_ARCH35_OP_ASC_SOURCES "${_src}")
endforeach()

# The shared launch-layer TU (the failure record and Status::ToString) belongs
# to no single operator but is an ASC TU for the same reason the op_host
# wrappers are.
list(APPEND MOE_ARCH35_OP_ASC_SOURCES
  "${CMAKE_SOURCE_DIR}/src/kernels/arch35/launch_common.cpp"
)
list(APPEND MOE_ARCH35_OP_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/src/kernels/arch35"
)
