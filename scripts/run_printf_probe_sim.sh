#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build and run the device-printf probe on the dav-3510 CAModel.
#
# Answers "does __asc_aicore::printf actually produce output when the kernel
# executes?" -- the toolchain question, separate from any operator. The probe is
# tests/sim/kernel/dsv4_printf_probe_sim.cpp.
#
# NO DOCKER. Unlike scripts/device_compile_kernels.sh, the CAModel runs directly
# in WSL against the installed toolkit.
#
# From Windows:
#   wsl -d Ubuntu-22.04 -- bash /mnt/c/ascend-moe-engine/scripts/run_printf_probe_sim.sh
#
# From inside WSL:
#   bash scripts/run_printf_probe_sim.sh
#
# Options:
#   --build-only        compile and link, do not run the simulator (seconds)
#   --variant on|off    build/run only one variant (default: both)
#   --keep <dir>        build directory (default: ~/build/printf-probe)
#
# `on` passes --cce-enable-print as well as the macros; `off` passes only the
# macros. Both are expected to print -- see the measurement in the probe's own
# header and in tests/npu/CMakeLists.txt. The pair exists so the flag's effect
# stays a measured fact rather than a belief.
#
# COST: ~200-320 s PER VARIANT, almost all of it fixed simulator bring-up. Run
# it in the background; a foreground call will time out. Do NOT put the build
# directory under /tmp -- WSL wipes it when the distro goes idle, which deletes
# a configured tree with no error anywhere.
# -----------------------------------------------------------------------------
# NOT `set -u`: set_env.sh reads CMAKE_PREFIX_PATH / PYTHONPATH / TOOLCHAIN_HOME before it
# writes them, so -u makes the toolkit's own script emit "unbound variable" noise that looks
# like a failure of this one.
set -o pipefail

VARIANTS="on off"
BUILD_ONLY=0
WORK="$HOME/build/printf-probe"

while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) BUILD_ONLY=1; shift ;;
    --variant)    VARIANTS="$2"; shift 2 ;;
    --keep)       WORK="$2"; shift 2 ;;
    -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROBE="$REPO_ROOT/tests/sim/kernel/dsv4_printf_probe_sim.cpp"

if [ ! -f "$PROBE" ]; then
  echo "error: probe source not found at $PROBE" >&2
  exit 2
fi

# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  echo "error: ASCEND_HOME_PATH unset after sourcing set_env.sh" >&2
  exit 2
fi

R="$ASCEND_HOME_PATH"
A="$R/x86_64-linux"
SIM="$A/simulator/dav_3510"

if [ ! -f "$SIM/lib/libruntime_camodel.so" ]; then
  echo "error: dav-3510 CAModel runtime not found under $SIM" >&2
  exit 2
fi

mkdir -p "$WORK"
cd "$WORK"

INC="-I$A/asc/include -I$A/asc/include/simt_api -I$A/include -I$R/include"
for d in "$R/compiler/tikcpp/tikcfw" \
         "$R/compiler/tikcpp/tikcfw/impl" \
         "$R/compiler/tikcpp/tikcfw/interface"; do
  [ -d "$d" ] && INC="$INC -I$d"
done

# The same library set tests/sim/CMakeLists.txt links its CAModel targets with.
LIBS="$A/lib64/libascendc_runtime.a $A/lib64/libruntime.so $A/lib64/libascendcl.so \
$A/lib64/libprofapi.so $A/lib64/libunified_dlog.so $A/lib64/libmmpa.so $A/lib64/libc_sec.so"

# libruntime.so's DT_NEEDED closure reaches libascend_hal.so, which ships with
# the DRIVER. -rpath-link only, never -rpath: the stub must be invisible at run
# time so it cannot be loaded in place of the CAModel's own libascend_hal.so,
# which the LD_LIBRARY_PATH below puts first (CLAUDE.md section 4.4).
RPL=""
for d in "$A/lib64" "$A/devlib/linux/x86_64" "$A/devlib"; do
  [ -d "$d" ] && RPL="$RPL -Wl,-rpath-link,$d"
done

flags_for() {
  case "$1" in
    on)  echo "--cce-enable-print -DENABLE_KERNEL_PRINTF=1 -DASCENDC_DUMP=1 -UNDEBUG" ;;
    off) echo "-DENABLE_KERNEL_PRINTF=1 -DASCENDC_DUMP=1 -UNDEBUG" ;;
    *)   echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}

# bisheng -c then link with the host compiler, as tests/sim does. Handing the
# .a straight to bisheng --asc-aicore-lang makes it treat the archive as a
# SOURCE file ("error: expected unqualified-id / !<arch>"), so the two steps
# are not interchangeable with one.
build_one() {
  tag="$1"
  echo "=== build $tag ==="
  # shellcheck disable=SC2086
  if ! bisheng -O2 -DNDEBUG --npu-arch=dav-3510 --asc-aicore-lang \
        $(flags_for "$tag") $INC -c "$PROBE" -o "$WORK/probe_$tag.o" 2>&1 | head -30; then
    echo "$tag: COMPILE FAILED"
    return 1
  fi
  # shellcheck disable=SC2086
  if ! g++ -o "$WORK/printf_probe_$tag" "$WORK/probe_$tag.o" $LIBS $RPL \
        -Wl,-rpath,"$A/lib64" -lpthread -ldl 2>&1 | head -30; then
    echo "$tag: LINK FAILED"
    return 1
  fi
  printf '%s: built  object=%s B  binary=%s B  print_symbols=%s\n' "$tag" \
    "$(stat -c%s "$WORK/probe_$tag.o")" \
    "$(stat -c%s "$WORK/printf_probe_$tag")" \
    "$(nm -C "$WORK/probe_$tag.o" 2>/dev/null | grep -ci 'PrintPayload\|printf')"
}

run_one() {
  tag="$1"
  log="$WORK/run_$tag.log"
  echo "=== run $tag on the CAModel (expect 200-320 s) ==="
  start=$(date +%s)
  ASCEND_SIMULATOR_MODE=1 \
  LD_LIBRARY_PATH="$SIM/camodel:$SIM/lib:$A/lib64:${LD_LIBRARY_PATH:-}" \
    timeout 1500 "$WORK/printf_probe_$tag" > "$log" 2>&1
  rc=$?
  echo "$tag: exit=$rc wall=$(( $(date +%s) - start ))s log=$log"
  echo "--- device records ---"
  grep "DEVPRINT" "$log" || echo "  (none -- see the witness line below before blaming printf)"
  echo "--- host ---"
  grep "^HOST" "$log"
  # pmu.cc "Failed to open dir: ./log_ca/" is simulator noise on every run.
  grep -iE "error|assert|FAIL" "$log" | grep -v "log_ca" | head -5
  echo
}

fail=0
for v in $VARIANTS; do
  build_one "$v" || fail=1
done
[ "$fail" -ne 0 ] && exit 1

if [ "$BUILD_ONLY" -eq 1 ]; then
  echo "build-only requested; not running the simulator."
  exit 0
fi

mkdir -p "$WORK/log_ca"
for v in $VARIANTS; do
  run_one "$v"
done

echo "================ SUMMARY ================"
for v in $VARIANTS; do
  log="$WORK/run_$v.log"
  [ -f "$log" ] || continue
  printf '%-4s records=%s  %s\n' "$v" \
    "$(grep -c DEVPRINT "$log")" \
    "$(grep -o 'witness\[0\]=[0-9-]*' "$log" | head -1)"
done
echo
echo "4 records and witness[0]=1234 is a pass. 0 records with witness[0]=1234"
echo "means the kernel ran and printf is not reaching the host. No witness at"
echo "all means the launch itself failed -- read the log, not the records."
