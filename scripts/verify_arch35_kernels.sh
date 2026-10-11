#!/usr/bin/env bash
# Stage-1 arch35 kernel gate: static audit + device compile for KER-01..KER-05.
#
# Steps 3 and 4 of the engineering loop in one reproducible command:
#   1. regenerate the UB layout header and the injected analyzer annotations,
#      and fail if either was stale (so offsets cannot drift from the sources)
#   2. ascend_analyzer --chip ascend950pr --ub-bytes 221184 --warnings-as-errors
#   3. bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang
#
# The analyzer and PipeSim live outside the repo and outside the CANN
# container, so the audit runs on the host from a venv and the compile runs in
# the container. Override either location with the environment variables below.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZER_ROOT="${ANALYZER_ROOT:-/mnt/d/Projects/ascend-kernel-analyzer}"
VENV_PY="${VENV_PY:-}"
CANN_CONTAINER="${CANN_CONTAINER:-cann92_dev}"
CANN_REPO_MOUNT="${CANN_REPO_MOUNT:-/home/ascend-moe-engine}"
BISHENG="${BISHENG:-/usr/local/Ascend/cann-9.2.0-beta.2/bin/bisheng}"

# 216 KiB: 256 KiB physical UB less the 32 KiB SIMT DataCache floor and the
# 8 KiB compiler reserve. The built-in profile reports the physical 256 KiB,
# so the usable ceiling has to be passed explicitly.
UB_BYTES=221184
KERNEL_DIR="src/kernels/arch35/op_kernel"
KERNELS=(
  ker01_mhc_pre_norm_sinkhorn
  ker02_mhc_post_residual
  ker03_paged_compressor_rope
  ker04_fp8_indexer_topk
  ker05_mxfp4_expert_gemv
)

cd "$REPO" || exit 2
fail=0

if [[ -z "$VENV_PY" ]]; then
  echo "VENV_PY is not set: point it at a python with the analyzer's deps" >&2
  echo "  (networkx, tree-sitter, tree-sitter-cpp, pcpp, z3-solver, numpy)" >&2
  exit 2
fi

echo "== 1. generated artefacts are up to date =="
if ! python3 scripts/gen_arch35_ub_layout.py \
      --header "$KERNEL_DIR/arch35_ub_layout.h" \
      --inject "$KERNEL_DIR" --check; then
  echo "  FAIL: regenerate with scripts/gen_arch35_ub_layout.py" >&2
  fail=1
else
  echo "  OK"
fi

echo
echo "== 2. static audit (ascend_analyzer, chip ascend950pr, UB $UB_BYTES B) =="
for k in "${KERNELS[@]}"; do
  out=$(PYTHONPATH="$ANALYZER_ROOT" "$VENV_PY" -m ascend_analyzer \
          "$KERNEL_DIR/$k.cpp" --chip ascend950pr --ub-bytes "$UB_BYTES" \
          --warnings-as-errors 2>&1)
  rc=$?
  summary=$(printf '%s\n' "$out" | grep -E "fatal .*warning .*info" | tail -1)
  hw=$(printf '%s\n' "$out" | grep -oE "high-water [0-9]+ B \([0-9.]+%\)" | tail -1)
  if [[ $rc -eq 0 ]]; then
    printf '  OK    %-32s %s  %s\n' "$k" "${summary:-no summary}" "$hw"
  else
    printf '  FAIL  %-32s rc=%d\n' "$k" "$rc"
    printf '%s\n' "$out" | sed -n '/^Findings/,/^SRAM/p' | sed 's/^/        /'
    fail=1
  fi
done

echo
echo "== 3. device compile (bisheng, --npu-arch=dav-3510) =="
for k in "${KERNELS[@]}"; do
  log=$(docker exec "$CANN_CONTAINER" bash -lc "
    cd $CANN_REPO_MOUNT/$KERNEL_DIR &&
    $BISHENG -O2 --npu-arch=dav-3510 --asc-aicore-lang \
      -I$CANN_REPO_MOUNT/include -I. -c $k.cpp -o /tmp/$k.dav-3510.o 2>&1
    echo \"RC=\$?\"
    ls -l /tmp/$k.dav-3510.o 2>/dev/null | awk '{print \"SIZE=\" \$5}'
  " 2>&1)
  if printf '%s\n' "$log" | grep -q "^SIZE="; then
    size=$(printf '%s\n' "$log" | grep "^SIZE=" | cut -d= -f2)
    printf '  OK    %-32s %s bytes of dav-3510 object\n' "$k" "$size"
  else
    printf '  FAIL  %-32s\n' "$k"
    printf '%s\n' "$log" | grep -E "error:" | head -5 | sed 's/^/        /'
    fail=1
  fi
done

echo
if [[ $fail -eq 0 ]]; then
  echo "ALL GATES PASSED for KER-01..KER-05 (layout, audit, device compile)."
else
  echo "GATE FAILURES present; see above." >&2
fi
exit $fail
