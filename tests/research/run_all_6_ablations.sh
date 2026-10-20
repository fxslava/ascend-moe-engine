#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

configs=(
  cfg1_tq4_lloydmax
  cfg2_tq4_lloydmax_e8
  cfg3_tq4_lloydmax_e24
  cfg4_tq3_lloydmax
  cfg5_tq3_lloydmax_e8
  cfg6_tq3_lloydmax_e24
)

for config in "${configs[@]}"; do
  echo
  echo "## ${config}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extreme_niah_benchmark.py" --config "${config}"
done

echo
echo "## Six-way comparison"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_6_ablations.py"
