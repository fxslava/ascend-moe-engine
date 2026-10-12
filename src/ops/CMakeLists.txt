# -----------------------------------------------------------------------------
# Operator aggregation.
#
# The five arch35 / dav-3510 operators contribute ASC translation units only.
# Their file list is derived from the operator registry in
# cmake/arch35_ops.cmake and compiled by cmake/arch35_device.cmake into
# arch35_device_launch, so nothing is built from their directories here --
# each src/ops/<op>/CMakeLists.txt documents the operator's files and exists
# for per-operator host targets if one is ever needed.
#
# This file therefore only has to bring in the deprecated operator, which does
# build host targets.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# DEPRECATED (CLAUDE.md section 6)
#
# mhc_fused_sinkhorn_gemv predates the arch35 contracts: its kernel is not one
# of the five dav-3510 operators and is not covered by the device compile
# gate. It is kept building because tests/common and tests/sim depend on it,
# and because it remains the format reference for the operator layout.
# Do not extend it.
# -----------------------------------------------------------------------------
add_subdirectory(mhc_fused_sinkhorn_gemv)
