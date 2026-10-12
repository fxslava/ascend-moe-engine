#!/usr/bin/env python3
"""Summarize the nine completed anisotropic multi-needle ablations."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


CONFIGS = (
    ("cfg1_tq2_lloydmax", "adversarial_results_2bit_lloydmax.csv"),
    ("cfg2_tq2_lloydmax_e8", "adversarial_results_2bit_lloydmax_e8.csv"),
    ("cfg3_tq2_lloydmax_e24", "adversarial_results_2bit_lloydmax_e24.csv"),
    ("cfg4_tq3_lloydmax", "adversarial_results_3bit_lloydmax.csv"),
    ("cfg5_tq3_lloydmax_e8", "adversarial_results_3bit_lloydmax_e8.csv"),
    ("cfg6_tq3_lloydmax_e24", "adversarial_results_3bit_lloydmax_e24.csv"),
    ("cfg7_tq4_lloydmax", "adversarial_results_4bit_lloydmax.csv"),
    ("cfg8_tq4_lloydmax_e8", "adversarial_results_4bit_lloydmax_e8.csv"),
    ("cfg9_tq4_lloydmax_e24", "adversarial_results_4bit_lloydmax_e24.csv"),
)


def main() -> int:
    root = Path(__file__).resolve().parent
    print(
        "| Config | Mean Recall@128 | Distractor FP Rate | "
        "Mean Spearman rho | 1M Latency (s) | 1M Packed Keys (MiB) |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for config, filename in CONFIGS:
        path = root / filename
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        contexts = {int(row["context_length"]) for row in rows}
        if contexts != {8192, 32768, 262144, 1048576}:
            raise ValueError(f"{path} does not contain the complete four-context grid")
        million = [row for row in rows if int(row["context_length"]) == 1048576]
        if len(million) != 1:
            raise ValueError(f"{path} must contain exactly one 1M-token row")
        row = million[0]
        print(
            f"| {config} | "
            f"{mean(float(item['mean_needle_recall_at_128']) for item in rows):.4f} | "
            f"{mean(float(item['distractor_false_positive_rate']) for item in rows):.4f} | "
            f"{mean(float(item['mean_spearman_rho']) for item in rows):.6f} | "
            f"{float(row['stream_latency_s']):.3f} | "
            f"{float(row['packed_key_stream_mib']):.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
