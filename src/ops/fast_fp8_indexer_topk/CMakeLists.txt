# -----------------------------------------------------------------------------
# fast_fp8_indexer_topk -- arch35 / dav-3510 operator.
# Layout and naming per CLAUDE.md section 1.
#
# Nothing is built here. All five of this operator's translation units are
# ASC (BiSheng) TUs:
#
#   op_kernel/fast_fp8_indexer_topk_kernel.cpp    fused single-core reference
#   op_kernel/fast_fp8_indexer_score_kernel.cpp   stage 1, 32 AIV scoring
#   op_kernel/fast_fp8_indexer_select_kernel.cpp  stage 2, 1 AIV selection
#   op_host/fast_fp8_indexer_topk.cpp             the chevron launch wrapper
#   op_host/fast_fp8_indexer_topk_tiling.cpp      the host-side tiling builder
#
# Three device entry symbols, one operator: one directory, one host ABI
# (LaunchFp8IndexerTopK) and one tiling POD. The wrapper dispatches the
# score+select pair on one stream; the fused kernel is currently unlaunched and
# kept only as the single-launch shape (see its file header). The two SIMT
# bodies they share live in
# op_kernel/fast_fp8_indexer_{score,select}_simt.h -- operator-private headers
# (CLAUDE.md §1.2), included by bare filename, so there is one transcription of
# the scoring loop and one of the selection network rather than three.
#
# They need the ASC language enabled, which only happens under
# ENABLE_ARCH35_DEVICE_LAUNCH, so they are compiled into arch35_device_launch
# by cmake/arch35_device.cmake. That file derives this operator's paths from
# the registry in cmake/arch35_ops.cmake rather than from this directory's
# scope, so the build does not depend on subdirectory ordering. The two extra
# kernel TUs are listed explicitly there, in
# MOE_ARCH35_EXTRA_KERNEL_SOURCES, because the registry's path derivation
# assumes one kernel entry per operator.
#
# This file exists to keep the operator layout uniform and to hold any
# per-operator HOST target that is ever needed. If you add one, name it
# fast_fp8_indexer_topk_<role> per CLAUDE.md section 1.4, and do not
# compile fast_fp8_indexer_topk_tiling.cpp into it -- that TU is
# already in arch35_device_launch, and a binary linking both would get
# duplicate symbols.
# -----------------------------------------------------------------------------
