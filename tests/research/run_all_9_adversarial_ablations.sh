#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/adversarial_multi_needle_benchmark.py" \
  --verify-2bit-packing-only

configs=(
  cfg1_tq2_lloydmax
  cfg2_tq2_lloydmax_e8
  cfg3_tq2_lloydmax_e24
  cfg4_tq3_lloydmax
  cfg5_tq3_lloydmax_e8
  cfg6_tq3_lloydmax_e24
  cfg7_tq4_lloydmax
  cfg8_tq4_lloydmax_e8
  cfg9_tq4_lloydmax_e24
)

for config in "${configs[@]}"; do
  echo
  echo "## ${config}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/adversarial_multi_needle_benchmark.py" \
    --config "${config}"
done

echo
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_9_adversarial_ablations.py"
