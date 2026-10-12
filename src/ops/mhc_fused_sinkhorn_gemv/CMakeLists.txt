# -----------------------------------------------------------------------------
# DEPRECATED OPERATOR -- see CLAUDE.md section 6.
#
# mhc_fused_sinkhorn_gemv predates the arch35 / dav-3510 operator contracts:
#   * its kernel is not one of the five arch35 operators and is NOT covered by
#     scripts/device_compile_kernels.sh, so it has no device compile gate;
#   * its launch path is not the chevron-dispatch layer in src/kernels/arch35;
#   * the target below was originally named plain `cpu_ref`, which is a
#     global-namespace collision as soon as a second operator follows the same
#     layout -- exactly what CLAUDE.md section 1.4 now forbids.
#
# It stays in the build for two reasons and no others: tests/common and
# tests/sim depend on it, and it is the FORMAT REFERENCE for the per-operator
# directory layout every arch35 operator now follows.
#
# Do not extend this operator, add features to it, or copy its target naming.
# New mHC/Sinkhorn work belongs in src/ops/fused_mhc_pre_norm_sinkhorn and
# src/ops/fused_mhc_post_residual.
# -----------------------------------------------------------------------------

message(STATUS
  "mhc_fused_sinkhorn_gemv: DEPRECATED (format reference only; see CLAUDE.md section 6)")

add_library(mhc_fused_sinkhorn_gemv_cpu_ref STATIC
  cpu_ref/mhc_fused_sinkhorn_gemv_ref.cpp
  op_host/mhc_fused_sinkhorn_gemv.cpp
  op_host/mhc_fused_sinkhorn_gemv_tiling.cpp
)
target_include_directories(mhc_fused_sinkhorn_gemv_cpu_ref PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/cpu_ref"
  "${CMAKE_CURRENT_SOURCE_DIR}/op_host"
  "${CMAKE_SOURCE_DIR}/include"
)
target_compile_options(mhc_fused_sinkhorn_gemv_cpu_ref PRIVATE -Wall -Wextra)

# The old unprefixed name, kept so tests/common and tests/sim keep linking
# while they still reference it. New code must use the prefixed target.
add_library(cpu_ref ALIAS mhc_fused_sinkhorn_gemv_cpu_ref)
