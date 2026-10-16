#!/usr/bin/env python3
"""Print one markdown comparison row for each completed NIAH ablation."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


RESULTS = (
    ("cfg1", "4", "none", "niah_results_4bit_lloydmax.csv"),
    ("cfg2", "4", "E8", "niah_results_4bit_lloydmax_e8.csv"),
    ("cfg3", "4", "E24", "niah_results_4bit_lloydmax_e24.csv"),
    ("cfg4", "3", "none", "niah_results_3bit_lloydmax.csv"),
    ("cfg5", "3", "E8", "niah_results_3bit_lloydmax_e8.csv"),
    ("cfg6", "3", "E24", "niah_results_3bit_lloydmax_e24.csv"),
)


def main() -> int:
    root = Path(__file__).resolve().parent
    print(
        "| Config | Bits | Lattice | Recall@128 | Mean rho | Min rho | "
        "Mean dilution | 1M latency | 1M packed keys | Peak RSS |"
    )
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, bits, lattice, filename in RESULTS:
        path = root / filename
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 20:
            raise ValueError(f"{path} has {len(rows)} rows; expected 20")
        recalls = sum(int(row["recall_at_k"]) for row in rows)
        rhos = [float(row["spearman_rho"]) for row in rows]
        dilutions = [float(row["dilution"]) for row in rows]
        million = [row for row in rows if int(row["context_length"]) == 1048576]
        if len(million) != 5:
            raise ValueError(f"{path} does not contain five 1M-depth rows")
        print(
            f"| {name} | {bits} | {lattice} | {recalls}/20 | "
            f"{mean(rhos):.6f} | {min(rhos):.6f} | {mean(dilutions):.6f} | "
            f"{float(million[0]['stream_latency_s']):.3f}s | "
            f"{float(million[0]['packed_key_stream_mib']):.1f} MiB | "
            f"{max(float(row['peak_rss_mib']) for row in rows):.1f} MiB |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
