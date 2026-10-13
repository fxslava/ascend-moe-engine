#!/usr/bin/env bash
# Nine faithful DSV4 MLA ablations: {2,3,4} bits x {scalar Lloyd-Max, E8, E24},
# each over the full context grid, each writing one
# faithful_dsv4_results_*.csv beside this script.
#
# Every run uses the decoupled cache: a 512-lane quantized semantic latent and
# 64 UNQUANTIZED rotary lanes. The bit width and lattice mode name what happens
# to the latent only; the rotary path is identical in all nine.
#
# The dimension contracts are asserted first. They parse the engine's own
# headers, so if the geometry has moved under us the sweep stops before burning
# an hour producing numbers for the wrong shape.
#
# Runtime: the E24 Chase decoder dominates and the 1M context is 262144 slots
# of 512 lanes. Budget a few hours for the full grid; override the grid to cut
# it down, e.g.
#   CONTEXT_LENGTHS=8192,32768 ./run_faithful_dsv4_ablations.sh
#
# Usage:
#   ./run_faithful_dsv4_ablations.sh [extra benchmark args...]
#   SEED=123 CONTEXT_LENGTHS=8192 ./run_faithful_dsv4_ablations.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SEED="${SEED:-20261013}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-8192,32768,262144,1048576}"
CHUNK_TOKENS="${CHUNK_TOKENS:-8192}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

cd "$HERE"

echo "== dimension contracts =="
"$PYTHON" -m unittest test_dsv4_mla_dimensions 2>&1 | tail -3

echo
echo "== audited geometry =="
"$PYTHON" dsv4_mla_niah_benchmark.py --print-geometry-only

# `tr -d '\r'` is not decoration: on Windows the interpreter's stdout newline
# translation puts a CR in every line, and argparse then rejects the config
# name as an invalid choice.
CONFIGS=$(
  "$PYTHON" - <<'PY' | tr -d '\r'
from dsv4_mla_niah_benchmark import CONFIGS
print("\n".join(CONFIGS))
PY
)

echo
echo "== sweep: seed=$SEED contexts=$CONTEXT_LENGTHS chunk=$CHUNK_TOKENS =="
for config in $CONFIGS; do
  echo
  echo "--- $config ---"
  "$PYTHON" dsv4_mla_niah_benchmark.py \
    --config "$config" \
    --context-lengths "$CONTEXT_LENGTHS" \
    --chunk-tokens "$CHUNK_TOKENS" \
    --seed "$SEED" \
    "$@"
done

echo
echo "== combined comparison =="
"$PYTHON" summarize_faithful_dsv4_ablations.py

echo
echo "Wrote:"
ls -1 "$HERE"/faithful_dsv4_results_*.csv
