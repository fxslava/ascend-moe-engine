# -----------------------------------------------------------------------------
# bf16_block_quant_fp8 -- arch35 / dav-3510 operator.
# Layout and naming per CLAUDE.md section 1.
#
# Nothing is built here. All three of this operator's translation units are
# ASC (BiSheng) TUs:
#
#   op_kernel/bf16_block_quant_fp8_kernel.cpp   the device kernel
#   op_host/bf16_block_quant_fp8.cpp            the chevron launch wrapper
#   op_host/bf16_block_quant_fp8_tiling.cpp     the host-side tiling builder
#
# They need the ASC language enabled, which only happens under
# ENABLE_ARCH35_DEVICE_LAUNCH, so they are compiled into arch35_device_launch
# by cmake/arch35_device.cmake. That file derives this operator's paths from
# the registry in cmake/arch35_ops.cmake rather than from this directory's
# scope, so the build does not depend on subdirectory ordering.
#
# This file exists to keep the operator layout uniform and to hold any
# per-operator HOST target that is ever needed. If you add one, name it
# bf16_block_quant_fp8_<role> per CLAUDE.md section 1.4, and do not compile
# bf16_block_quant_fp8_tiling.cpp into it -- that TU is already in
# arch35_device_launch, and a binary linking both would get duplicate symbols.
# -----------------------------------------------------------------------------
