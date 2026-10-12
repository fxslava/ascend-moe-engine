# -----------------------------------------------------------------------------
# One UB allocation mechanism per device kernel, checked against the sources
# instead of against a comment.
#
# TPipe/TBuf and LocalMemAllocator both bump a head pointer from the SAME
# Unified Buffer base address. A kernel that holds a TBuf alongside the
# allocator therefore has its TBufs overlaying the allocator's leading
# segments. Nothing diagnoses it: the link is clean, the UB budget is legal,
# and there is no fault -- the two buffers simply share bytes.
#
# WHY THIS CHECK EXISTS AND NOT JUST A RULE IN CLAUDE.md
# paged_compressor_rope carried two 1 KiB BF16 staging TBufs next to a
# LocalMemAllocator layout, and they aliased kvCur_ row 0: normWBf16 over
# feature lanes [0,256), outBf16 over [256,512). Every step's norm-gain
# DataCopy overwrote a quarter of the first state row, and the whole-region
# writeback persisted it. It cost two rounds of hardware debugging and
# presented as something else entirely -- a group-boundary bug -- because step
# 0's projection writes slot 0 and therefore repaired row 0 after the clobber,
# so the first EMIT was clean and only the SECOND one, which reduces the rolled
# a-half of row 0, was wrong.
#
# That is precisely the shape of defect a comment does not prevent: the wrong
# code looks conventional, the symptom appears four launches away from the
# cause, and the only cheap witness is "these two mechanisms are both present
# in one translation unit".
#
# Invoked by ctest with:
#   -D KERNEL_SOURCES=<;-separated .cpp/.h paths>
# -----------------------------------------------------------------------------

if(NOT DEFINED KERNEL_SOURCES)
  message(FATAL_ERROR
    "verify_ub_allocator_exclusive.cmake requires -D KERNEL_SOURCES")
endif()

set(_violations "")
set(_checked 0)

foreach(_src IN LISTS KERNEL_SOURCES)
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "expected kernel source not found: ${_src}")
  endif()
  math(EXPR _checked "${_checked} + 1")

  file(READ "${_src}" _text)

  # Strip // comments so the explanatory prose in these very files -- which
  # necessarily names both mechanisms -- cannot trip the check. Block comments
  # are not used for this in the tree, and a false positive here is a loud
  # failure rather than a silent pass, so line comments are enough.
  string(REGEX REPLACE "//[^\n]*" "" _code "${_text}")

  # A declared TPipe member or local, not a mention. `TBuf` alone is not
  # enough: it appears as a template argument in unrelated places.
  set(_has_tpipe FALSE)
  if(_code MATCHES "(^|[^A-Za-z_])TPipe[ \t]+[A-Za-z_]" OR
     _code MATCHES "InitBuffer[ \t]*\\(")
    set(_has_tpipe TRUE)
  endif()

  set(_has_alloc FALSE)
  if(_code MATCHES "LocalMemAllocator[ \t]*<")
    set(_has_alloc TRUE)
  endif()

  if(_has_tpipe AND _has_alloc)
    get_filename_component(_name "${_src}" NAME)
    list(APPEND _violations
      "${_name}: declares a TPipe/TBuf AND a LocalMemAllocator. They share the "
      "UB base address, so the TBufs alias the allocator's first segments. "
      "Move the TBuf contents into the allocator (and add their bytes to the "
      "operator's dyn_ub_size formula), or drop the allocator.")
  endif()
endforeach()

if(_violations)
  foreach(_v IN LISTS _violations)
    message(SEND_ERROR "${_v}")
  endforeach()
  message(FATAL_ERROR "UB allocation mechanisms are not exclusive")
endif()

message(STATUS
  "UB allocator exclusivity: OK (${_checked} device kernel sources; no TU "
  "mixes TPipe/TBuf with LocalMemAllocator)")
