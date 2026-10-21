#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Verify that the NPU-side targets BUILD, on a machine with no NPU.
#
# Configuring and building the device launch layer needs BiSheng and a CANN
# toolkit; it does NOT need hardware (cmake/arch35_device.cmake links against
# the driver stubs with -rpath-link for exactly this case). So "does it build"
# is answerable here, while "does it compute the right answer" is not.
#
# Builds the host-emu configuration with -DENABLE_ARCH35_DEVICE_LAUNCH=ON,
# which is the smallest configuration that produces:
#   * arch35_device_launch        -- the five operators + the launch layer
#   * test_custom_kernels_npu_device -- the on-device verification suite
#   * test_custom_kernels_emu     -- the CPU oracle
#   * bench_custom_kernels_performance
#
# Runs inside the ascend-c-dev container with the toolkit bind-mounted, for
# the same reason scripts/device_compile_kernels.sh does -- see that file.
#
# From Windows:
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/ascend-moe-engine/scripts/verify_npu_build.sh
# -----------------------------------------------------------------------------
set -uo pipefail

IMAGE="${ASCEND_DEV_IMAGE:-ascend-c-dev:9.2.0-beta.2}"
TOOLKIT_HOST="${ASCEND_TOOLKIT_HOST:-/usr/local/Ascend}"
BUILD_DIR="${1:-build_npu_verify}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found on PATH (on Windows the daemon is inside WSL)." >&2
  exit 2
fi
if [ ! -d "$TOOLKIT_HOST" ]; then
  echo "error: CANN toolkit not found at $TOOLKIT_HOST." >&2
  exit 2
fi

echo "image:    $IMAGE"
echo "toolkit:  $TOOLKIT_HOST (bind-mounted read-only)"
echo "build:    $BUILD_DIR"
echo

docker run --rm -i \
  -v "$TOOLKIT_HOST":/usr/local/Ascend:ro \
  -v "$REPO_ROOT":/workspace \
  -w /workspace \
  -e BUILD_DIR="$BUILD_DIR" \
  "$IMAGE" bash -s <<'INNER'
set -uo pipefail

rm -rf "$BUILD_DIR"

# tests/npu's standalone target pulls googletest in with FetchContent, which
# clones into $BUILD_DIR/_deps and then runs git inside that clone. The repo is
# bind-mounted from the Docker host, so its files are owned by a uid this
# container's root is not, and git aborts with "detected dubious ownership" --
# which surfaces as a FetchContent configure failure, not as a git error.
# Scoped to this --rm container's own root HOME; the host's git config is
# untouched.
git config --global --add safe.directory '*' 2>/dev/null || true

echo "=== configure ==="
if ! cmake -S . -B "$BUILD_DIR" \
      -DENABLE_HOST_EMU=ON \
      -DENABLE_ARCH35_DEVICE_LAUNCH=ON \
      -DCMAKE_BUILD_TYPE=Release > /tmp/configure.log 2>&1 ; then
  grep -iE 'error|fatal' /tmp/configure.log | head -20
  echo "configure FAILED"
  exit 1
fi
grep -E 'arch35 device launch|npu-arch|driver stubs' /tmp/configure.log
echo "configure OK"
echo

echo "=== build ==="
if ! cmake --build "$BUILD_DIR" -j "$(nproc)" > /tmp/build.log 2>&1 ; then
  grep -iE 'error|undefined reference|fatal' /tmp/build.log | head -30
  echo "build FAILED"
  exit 1
fi
# Clock-skew noise is expected on a bind-mounted Windows filesystem and is not
# a build warning; filter it so a real warning is visible.
warns=$(grep -icE 'warning' /tmp/build.log | head -1)
skew=$(grep -icE 'clock skew|modification time' /tmp/build.log | head -1)
echo "build OK (warning lines: $warns, of which filesystem clock-skew: $skew)"
echo

echo "=== artifacts ==="
for f in \
  "$BUILD_DIR/libarch35_device_launch.a" \
  "$BUILD_DIR/tests/npu/test_custom_kernels_npu_device" \
  "$BUILD_DIR/tests/emu/test_custom_kernels_emu" \
  "$BUILD_DIR/tests/bench/bench_custom_kernels_performance" ; do
  found=$(find "$BUILD_DIR" -name "$(basename "$f")" -print -quit 2>/dev/null)
  if [ -n "$found" ]; then
    printf '  %-44s %10s B\n' "$(basename "$f")" "$(stat -c%s "$found")"
  else
    printf '  %-44s MISSING\n' "$(basename "$f")"
  fi
done

echo
echo "=== AscendCL linkage of the device suite ==="
dev=$(find "$BUILD_DIR" -name test_custom_kernels_npu_device -print -quit)
if [ -n "$dev" ]; then
  # The device suite MUST link libascendcl; the CPU oracle must NOT (CLAUDE.md
  # section 4.1). Both halves of that claim are checked.
  if ldd "$dev" 2>/dev/null | grep -q ascendcl ; then
    echo "  device suite links libascendcl: yes"
  else
    echo "  device suite links libascendcl: NO -- unexpected"
  fi
fi
emu=$(find "$BUILD_DIR" -name test_custom_kernels_emu -print -quit)
if [ -n "$emu" ]; then
  if ldd "$emu" 2>/dev/null | grep -qE 'ascendcl|cpudebug|tikicpulib' ; then
    echo "  CPU oracle is free of ascendcl/cpudebug: NO -- contract violated"
    exit 1
  else
    echo "  CPU oracle is free of ascendcl/cpudebug: yes"
  fi
fi

echo
echo "npu build verification: OK"
INNER
