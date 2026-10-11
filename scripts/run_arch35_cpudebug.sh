#!/usr/bin/env bash
# Numerical acceptance for the arch35 Stage-1 kernels, with no NPU present.
#
# Builds each kernel's REAL Ascend C source together with a host harness under
# CANN's CPU interpreter for Ascend950PR_9599 and runs it against the fixtures
# that apps/dsv4_extract_kernel_fixtures writes. The kernel sources here are
# byte-identical to the ones bisheng compiles for the device, so this verifies
# the shipped code and not a re-implementation.
#
# Note the interpreter requires _GLIBCXX_USE_CXX11_ABI=0, which is why the
# harnesses cannot link libmoe_emu and read pre-computed fixture bytes instead.
#
# usage: run_arch35_cpudebug.sh <fixture-root> [kernel ...]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CANN_CONTAINER:-cann92_dev}"
MOUNT="${CANN_REPO_MOUNT:-/home/ascend-moe-engine}"
CANN="${CANN_ROOT:-/usr/local/Ascend/cann-9.2.0-beta.2}"

FIXTURE_ROOT="${1:?usage: run_arch35_cpudebug.sh <fixture-root> [kernel ...]}"
shift || true
KERNELS=("$@")
if [[ ${#KERNELS[@]} -eq 0 ]]; then
  KERNELS=(ker01 ker02 ker03 ker04 ker05)
fi

# kernel -> "<kernel source stem>|<fixture subdir>[,<fixture subdir>...]"
declare -A PLAN=(
  [ker01]="ker01_mhc_pre_norm_sinkhorn|ker01_layer0,ker01_layer2,ker01_moe"
  [ker02]="ker02_mhc_post_residual|ker02_layer0,ker02_layer2,ker02_moe"
  [ker03]="ker03_paged_compressor_rope|ker03_group1,ker03_group2"
  [ker04]="ker04_fp8_indexer_topk|ker04_indexer"
  [ker05]="ker05_mxfp4_expert_gemv|ker05_w1,ker05_w3,ker05_w2"
)

CPUD="$CANN/tools/cpudebug"
SIM="$CANN/x86_64-linux/simulator/Ascend950PR_9599/lib"
DEFS="-D__CCE_AICORE__=310 -D__DAV_C310__ -D__NPU_ARCH__=3510 \
      -D_GLIBCXX_USE_CXX11_ABI=0 -D__CCE_KT_TEST__=1 -DASCENDC_CPU_DEBUG=1"
INCS="-I$CPUD/include -I$CPUD/lib/include -I$CANN/include \
      -I$CANN/compiler/tikcpp/tikcfw -I$CANN/compiler/tikcpp/tikcfw/impl \
      -I$CANN/compiler/tikcpp/tikcfw/interface -I$CANN/include/base \
      -I$MOUNT/include -I$MOUNT/src/kernels/arch35/op_kernel \
      -I$MOUNT/tests/npu_cpudebug"
# The simulator library path the shipped cmake config points at does not exist
# in this layout; the real one is under x86_64-linux/simulator.
LIBS="-L$CPUD/lib -L$CPUD/lib/Ascend950PR_9599 -L$SIM -L$CANN/lib64 \
      -Wl,-Bdynamic,--no-as-needed -lc_sec -lpem_davinci -lcpudebug \
      -lcpudebug_stubreg -lcpudebug_cceprint -lcpudebug_npuchk -lpthread \
      -Wl,--as-needed"

fail=0
for k in "${KERNELS[@]}"; do
  entry="${PLAN[$k]:-}"
  if [[ -z "$entry" ]]; then
    echo "unknown kernel '$k'" >&2
    fail=1
    continue
  fi
  stem="${entry%%|*}"
  fixtures="${entry##*|}"

  echo "== building $k harness =="
  build_log=$(docker exec "$CONTAINER" bash -lc "
    mkdir -p /tmp/arch35_cpudebug && cd /tmp/arch35_cpudebug &&
    g++ -std=c++17 -O1 -w $DEFS $INCS \
      $MOUNT/src/kernels/arch35/op_kernel/$stem.cpp \
      $MOUNT/src/kernels/arch35/op_host/arch35_tiling.cpp \
      $MOUNT/tests/npu_cpudebug/${k}_cpudebug_main.cpp \
      -o ${k}_cpudebug $LIBS > ${k}_build.log 2>&1
    echo \"RC=\$?\"
    [[ -s ${k}_cpudebug ]] && echo LINKED
  " 2>&1)
  if ! grep -q LINKED <<<"$build_log"; then
    echo "  BUILD FAILED for $k" >&2
    docker exec "$CONTAINER" bash -lc \
      "grep -E 'error:|undefined reference|cannot find' /tmp/arch35_cpudebug/${k}_build.log | head -15" >&2
    fail=1
    continue
  fi
  echo "  linked"

  IFS=',' read -ra cases <<<"$fixtures"
  for case_dir in "${cases[@]}"; do
    host_fixture="$FIXTURE_ROOT/$case_dir"
    if [[ ! -d "$host_fixture" ]]; then
      echo "  SKIP $case_dir (no fixture at $host_fixture)" >&2
      fail=1
      continue
    fi
    # The fixture root lives outside the container's bind mount, so stage it.
    docker exec "$CONTAINER" bash -lc "rm -rf /tmp/arch35_fixtures/$case_dir && mkdir -p /tmp/arch35_fixtures"
    docker cp -q "$host_fixture" "$CONTAINER:/tmp/arch35_fixtures/$case_dir" 2>/dev/null \
      || docker cp "$host_fixture" "$CONTAINER:/tmp/arch35_fixtures/$case_dir"
    out=$(docker exec "$CONTAINER" bash -lc "
      cd /tmp/arch35_cpudebug && chmod +x ${k}_cpudebug &&
      LD_LIBRARY_PATH=$CPUD/lib:$CPUD/lib/Ascend950PR_9599:$SIM:$CANN/lib64 \
        ${HARNESS_ENV:-} ./${k}_cpudebug /tmp/arch35_fixtures/$case_dir 2>&1
      echo \"EXIT=\$?\"
    " 2>&1)
    # CANN 9.2's interpreter raises a spurious SIGSEGV backtrace inside
    # Cast<float, bfloat16_t> on one core, at every count, while still
    # computing the correct values and exiting 0. It is counted and reported
    # rather than hidden, but it is not a kernel failure; the harness verdict
    # and the process exit code are authoritative. See ISSUES_AND_GAPS.md.
    bf16_cast_segv=$(printf '%s\n' "$out" | grep -c "SIGSEGV Signal")
    if [[ "$bf16_cast_segv" -gt 0 ]]; then
      echo "  note: $bf16_cast_segv spurious interpreter SIGSEGV in Cast<float,bfloat16_t> (known CANN 9.2 defect, values unaffected)"
    fi
    # Drop the interpreter's own chatter; keep any real [ERROR] diagnostic that
    # is not part of that known backtrace, since that is how an illegal
    # intrinsic type or mode shows up.
    printf '%s\n' "$out" \
      | grep -vE "^\[#|^\[SUCCESS\]|^\[TmSim\]|^\[INFO\]|_start at|SIGABRT Signal|SIGSEGV Signal|error happened!|^$" \
      | sed 's/^/  /'
    if ! printf '%s\n' "$out" | grep -q "EXIT=0"; then
      fail=1
    fi
  done
  echo
done

if [[ $fail -eq 0 ]]; then
  echo "ALL CPU-DEBUG NUMERICAL CHECKS PASSED."
else
  echo "CPU-DEBUG FAILURES present; see above." >&2
fi
exit $fail
