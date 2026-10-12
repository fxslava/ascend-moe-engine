#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# The device compilation gate for the five arch35 custom kernels.
#
#   bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang -c <kernel.cpp> \
#           -o <kernel.dav-3510.o>
#
# run inside the ascend-c-dev container. This is the sole compiler verification
# gate: no host CPU debug runtime (cpudebug / tikicpulib) is involved, and no
# cycle-accurate simulator is run over model-scale tensors.
#
# WHY THE TOOLKIT IS BIND-MOUNTED
# -------------------------------
# The ascend-c-dev:9.2.0-beta.2 image is a thin shell. Its environment is set up
# as though CANN were installed (ASCEND_TOOLKIT_HOME, PATH and LD_LIBRARY_PATH
# all point at /usr/local/Ascend/cann-9.2.0-beta.2) but that directory is NOT in
# the image -- it is ~257 MB, essentially bare Ubuntu 22.04, and contains no
# bisheng. The toolkit lives on the Docker host at the same path, so it has to
# be mounted in. Without -v /usr/local/Ascend:/usr/local/Ascend the compile
# fails with "bisheng: command not found".
#
# Also note there is no /usr/local/Ascend/ascend-toolkit/set_env.sh in this
# layout; sourcing it is unnecessary (and fails) because the image environment
# already points at the toolkit.
#
# Usage:
#   scripts/device_compile_kernels.sh [output-dir]
#
# From Windows, Docker lives inside WSL, so drive it as:
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/ascend-moe-engine/scripts/device_compile_kernels.sh
# -----------------------------------------------------------------------------
set -uo pipefail

IMAGE="${ASCEND_DEV_IMAGE:-ascend-c-dev:9.2.0-beta.2}"
TOOLKIT_HOST="${ASCEND_TOOLKIT_HOST:-/usr/local/Ascend}"
OUT_DIR="${1:-build_dav3510}"

# Resolve the repository root from this script's location so the mount is right
# regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found on PATH." >&2
  echo "       On Windows the daemon runs inside WSL; invoke this script with" >&2
  echo "       wsl -d <distro> -- bash <path to this script>" >&2
  exit 2
fi

if [ ! -d "$TOOLKIT_HOST" ]; then
  echo "error: CANN toolkit not found at $TOOLKIT_HOST on the Docker host." >&2
  echo "       Override with ASCEND_TOOLKIT_HOST=<path>." >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/$OUT_DIR"

echo "image:    $IMAGE"
echo "toolkit:  $TOOLKIT_HOST (bind-mounted read-only)"
echo "repo:     $REPO_ROOT"
echo "output:   $OUT_DIR"
echo

# -i so the heredoc reaches bash on stdin; every -v/-w flag must precede the
# image name or docker treats it as the command.
docker run --rm -i \
  -v "$TOOLKIT_HOST":/usr/local/Ascend:ro \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  -e OUT_DIR="$OUT_DIR" \
  "$IMAGE" bash -s <<'INNER'
set -uo pipefail

ROOT="${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME unset in the image}"
if ! command -v bisheng >/dev/null 2>&1; then
  echo "error: bisheng not on PATH inside the container." >&2
  echo "       Expected the toolkit at $ROOT -- is it mounted?" >&2
  exit 3
fi

INC="-Iinclude"
for d in \
  "$ROOT/x86_64-linux/asc/include" \
  "$ROOT/x86_64-linux/asc/include/simt_api" \
  "$ROOT/x86_64-linux/include" \
  "$ROOT/compiler/tikcpp/tikcfw" \
  "$ROOT/compiler/tikcpp/tikcfw/impl" \
  "$ROOT/compiler/tikcpp/tikcfw/interface" ; do
  [ -d "$d" ] && INC="$INC -I$d"
done

echo "bisheng:  $(command -v bisheng)"
bisheng --version 2>&1 | head -1
echo

KERNELS="
fused_mhc_pre_norm_sinkhorn
fused_mhc_post_residual
paged_compressor_rope
fast_fp8_indexer_topk
mxfp4_expert_gemv
"

pass=0; fail=0
for k in $KERNELS; do
  src="src/ops/$k.cpp"
  obj="$OUT_DIR/$k.dav-3510.o"
  log="$OUT_DIR/$k.log"
  printf '%-34s ' "$k"
  if bisheng -O2 --npu-arch=dav-3510 --asc-aicore-lang $INC \
       -c "$src" -o "$obj" > "$log" 2>&1 ; then
    warns=$(grep -ciE 'warning' "$log" || true)
    printf 'PASS  %7s B  warnings=%s\n' "$(stat -c%s "$obj")" "$warns"
    pass=$((pass+1))
  else
    printf 'FAIL\n'
    sed 's/^/    /' "$log" | grep -iE 'error|fatal' | head -10
    fail=$((fail+1))
  fi
done

echo
echo "dav-3510 device compilation: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
INNER
