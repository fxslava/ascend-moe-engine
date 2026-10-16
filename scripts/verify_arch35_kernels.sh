#!/usr/bin/env bash
# Static-only arch35 gate for KER-01..KER-05. Never invokes a kernel runtime.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR=src/kernels/arch35/op_kernel
KERNELS=(ker01_mhc_pre_norm_sinkhorn ker02_mhc_post_residual ker03_paged_compressor_rope ker04_fp8_indexer_topk ker05_mxfp4_expert_gemv)
UB_BYTES=221184
PIPESIM_BIN="${PIPESIM_BIN:-/mnt/d/Projects/PipeSim/.venv/Scripts/npu-pipesim.exe}"
ANALYZER_PYTHONPATH="${ANALYZER_PYTHONPATH:-}"
CANN_CONTAINER="${CANN_CONTAINER:-cann92_dev}"
CONTAINER_MOUNT="${CONTAINER_MOUNT:-/home/ascend-moe-engine}"

cd "$REPO"
mkdir -p work/arch35-objects

# Resolve the checkout through the container's actual bind mount. In this
# environment it is /home/ascend-moe-engine, including nested worktrees.
host_mount="$(docker inspect --format "{{range .Mounts}}{{if eq .Destination \"$CONTAINER_MOUNT\"}}{{.Source}}{{end}}{{end}}" "$CANN_CONTAINER")"
if [[ -z "$host_mount" || "$REPO" != "$host_mount"* ]]; then
  echo "Checkout $REPO is not under container mount $host_mount" >&2
  exit 1
fi
container_workdir="$CONTAINER_MOUNT${REPO#"$host_mount"}"

# A site-installed analyzer needs no PYTHONPATH. The local, ignored copies of
# analyzer source and dependencies are used when present; callers can override.
if [[ -z "$ANALYZER_PYTHONPATH" && -d work/analyzer-src ]]; then
  ANALYZER_PYTHONPATH="$container_workdir/work/analyzer-src:$container_workdir/work/analyzer-deps"
fi

# PipeSim is a tile/budget model; its timing estimates are not execution proof.
echo '== PipeSim ascend950pr budget =='
profile_json="$("$PIPESIM_BIN" archs ascend950pr --format json)"
profile_ub="$(printf '%s' "$profile_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pools"]["UB"]["capacity_bytes"])')"
if [[ "$profile_ub" != "$UB_BYTES" ]]; then
  echo "PipeSim UB budget $profile_ub differs from required $UB_BYTES" >&2
  exit 1
fi
echo "  usable AIV UB: $profile_ub bytes"

# The generator pins every static buffer's offset and checks the same cap.
echo '== generated UB layout =='
python3 scripts/gen_arch35_ub_layout.py --header "$KERNEL_DIR/arch35_ub_layout.h" --inject "$KERNEL_DIR" --check

# Static source audit runs inside CANN, with the same paths as device compile.
echo '== ascend_analyzer (container) =='
for k in "${KERNELS[@]}"; do
  if audit="$(docker exec -u "$(id -u):$(id -g)" -e "PYTHONPATH=$ANALYZER_PYTHONPATH" \
      -w "$container_workdir" "$CANN_CONTAINER" \
      python3 -m ascend_analyzer "$KERNEL_DIR/$k.cpp" \
      --chip ascend950pr --ub-bytes "$UB_BYTES" --warnings-as-errors 2>&1)"; then
    summary="$(printf '%s\n' "$audit" | grep -E '  0 fatal +0 warning +0 info' | tail -1)"
    high_water="$(printf '%s\n' "$audit" | grep -oE 'high-water [0-9]+ B' | tail -1)"
    printf '  %-32s %s, %s\n' "$k" "$high_water" "$summary"
  else
    printf '%s\n' "$audit" >&2
    exit 1
  fi
done

echo '== non-root BiSheng device compilation =='
for k in "${KERNELS[@]}"; do
  docker exec -u "$(id -u):$(id -g)" -w "$container_workdir" "$CANN_CONTAINER" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
    bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang \
      -Iinclude -Isrc/kernels/arch35/op_kernel \
      -c "src/kernels/arch35/op_kernel/$1.cpp" \
      -o "work/arch35-objects/$1.dav-3510.o"
  ' _ "$k"
  object="work/arch35-objects/$k.dav-3510.o"
  [[ -s "$object" ]] || { echo "Missing or empty $object" >&2; exit 1; }
  printf '  %-32s %s bytes\n' "$k" "$(stat -c %s "$object")"
done

echo '== SIMT vector-core definitions =='
for k in ker01_mhc_pre_norm_sinkhorn ker04_fp8_indexer_topk ker05_mxfp4_expert_gemv; do
  definitions="$(docker exec -u "$(id -u):$(id -g)" -w "$container_workdir" "$CANN_CONTAINER" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
    bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang \
      -Iinclude -Isrc/kernels/arch35/op_kernel -dM -E \
      "src/kernels/arch35/op_kernel/$1.cpp" | grep -E "^#define (SPLIT_CORE_VEC|__DAV_VEC__)"
  ' _ "$k")"
  printf '%s\n' "$definitions" | grep -q '^#define SPLIT_CORE_VEC' || exit 1
  printf '%s\n' "$definitions" | grep -q '^#define __DAV_VEC__' || exit 1
  echo "  $k: SPLIT_CORE_VEC and __DAV_VEC__ defined"
done

echo 'ALL STATIC AND DEVICE-COMPILE GATES PASSED. No kernel execution was run.'
