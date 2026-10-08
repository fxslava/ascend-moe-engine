#
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Operator-timeline profiling for moe_runner via msprof, no MindStudio UI.

Wraps ``msprof --application="./moe_runner ..."`` and parses the
``op_summary_*.csv`` timeline msprof leaves behind into a compact
per-operator table (count, total/min/avg/max duration), so the question
"which operator owns the step" gets answered on the command line.

ZERO PyTorch / torch_npu. Requires the msprof binary that ships with the
CANN toolkit (source set_env.sh so it is on PATH).

Usage::

    python -m benchmarks.profile_cann --engine build_cann/apps/moe_runner \
        --engine-args "--synthetic-weights --prompt-ids 7 --max-new-tokens 4" \
        --output prof_out --top 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def find_summary_csv(output_dir: Path) -> list[Path]:
    """msprof scatters op_summary_*.csv under the output tree; find them all."""
    return sorted(output_dir.rglob("op_summary*.csv"))


def pick(columns: list[str], *needles: str) -> str | None:
    """First column whose lowercase name contains any needle.

    msprof column names drift between releases ("Op Name" vs "Name",
    "Duration(us)" vs "Total Time(us)"), so both the name and the duration
    columns are matched by substring.
    """
    for column in columns:
        lowered = column.lower()
        if any(needle in lowered for needle in needles):
            return column
    return None


def summarize(csv_paths: list[Path]) -> tuple[dict[str, int], dict[str, float], dict[str, float],
                                              dict[str, float]]:
    """Aggregates every operator row across all summary CSVs.

    Returns (counts, totals_us, minimums_us, maximums_us) keyed by operator
    name.
    """
    counts: dict[str, int] = collections.Counter()
    totals: dict[str, float] = collections.defaultdict(float)
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = collections.defaultdict(float)
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            name_column = pick(reader.fieldnames, "op name", "name")
            duration_column = pick(reader.fieldnames, "duration", "total time")
            if name_column is None or duration_column is None:
                print(f"  (skipped {csv_path.name}: no name/duration columns in {reader.fieldnames})")
                continue
            for row in reader:
                name = (row.get(name_column) or "").strip()
                if not name:
                    continue
                try:
                    duration_us = float((row.get(duration_column) or "0").replace(",", ""))
                except ValueError:
                    continue
                counts[name] += 1
                totals[name] += duration_us
                if name not in minimums or duration_us < minimums[name]:
                    minimums[name] = duration_us
                maximums[name] = max(maximums[name], duration_us)
    return counts, totals, minimums, maximums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.profile_cann",
        description="msprof moe_runner and summarize the operator timeline (no MindStudio UI).",
    )
    parser.add_argument("--engine", default="build_cann/apps/moe_runner", help="moe_runner binary to profile")
    parser.add_argument("--engine-args", default="--dry-run --synthetic-weights",
                        help="arguments for moe_runner, one shell-like string")
    parser.add_argument("--output", default="moe_prof_out", help="msprof --output directory")
    parser.add_argument("--msprof", default=None, help="explicit msprof path (default: PATH lookup)")
    parser.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    arguments = parser.parse_args(argv)

    msprof = arguments.msprof or shutil.which("msprof")
    if not msprof:
        raise SystemExit("msprof not found on PATH. Source the toolkit env "
                         "(source /usr/local/Ascend/ascend-toolkit/set_env.sh) or pass --msprof.")
    engine = Path(arguments.engine)
    if not engine.is_file():
        raise SystemExit(f"engine binary not found: {engine}")

    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    application = f"./{engine}" if not engine.is_absolute() else str(engine)
    command = [
        msprof,
        f"--application={application} {arguments.engine_args}",
        f"--output={output_dir}",
    ]
    print(f"profiling: {application} {arguments.engine_args}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"msprof exited with {completed.returncode}")

    csv_paths = find_summary_csv(output_dir)
    if not csv_paths:
        produced = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
        hint = f"; msprof produced: {produced[:10]}" if produced else "; msprof produced nothing"
        raise SystemExit("no op_summary_*.csv under " + str(output_dir) + hint)
    for path in csv_paths:
        print(f"timeline: {path}")

    counts, totals, minimums, maximums = summarize(csv_paths)
    if not counts:
        raise SystemExit("the summary CSVs parsed to zero operator rows")
    ranked = sorted(counts, key=lambda name: totals[name], reverse=True)
    total_all = sum(totals[name] for name in ranked)
    print(f"\n{'operator':<52} {'count':>7} {'total_us':>12} {'avg_us':>10} {'min_us':>10} {'max_us':>10} {'share':>7}")
    for name in ranked[: arguments.top]:
        share = 100.0 * totals[name] / total_all if total_all else 0.0
        print(f"{name[:52]:<52} {counts[name]:>7} {totals[name]:>12.0f} {totals[name] / counts[name]:>10.1f} "
              f"{minimums[name]:>10.1f} {maximums[name]:>10.1f} {share:>6.1f}%")
    print(f"\n{len(ranked)} operators, {total_all:.0f} us total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
