#!/usr/bin/env python3
"""Summarize the nine faithful DSV4 MLA ablations as a markdown comparison.

Takes the config list from the benchmark module rather than restating it, so
the two cannot drift apart. Tolerates a partial context grid -- the sweep is
often re-run over a cut-down grid -- but says so in the output instead of
silently averaging over fewer points than the reader expects.

It also re-asserts the deliverable's central claim across every row: the
rotary path is unquantized in all nine configurations. If any row says
otherwise the summary fails rather than printing a table that implies the
experiment was controlled when it was not.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .dsv4_mla_niah_benchmark import CONFIGS, DEFAULT_CONTEXTS
except ImportError:  # Direct execution from tests/research.
    from dsv4_mla_niah_benchmark import CONFIGS, DEFAULT_CONTEXTS  # type: ignore


def _load(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _check_rope_is_unquantized(config: str, rows: Sequence[Dict[str, str]]) -> None:
    for row in rows:
        if row["rope_quantized"].strip().lower() != "false":
            raise ValueError(
                f"{config}: a row reports rope_quantized="
                f"{row['rope_quantized']!r}; the decoupled-RoPE control is broken"
            )
        if int(row["rope_dim"]) != 64 or int(row["latent_dim"]) != 512:
            raise ValueError(
                f"{config}: geometry drifted to latent_dim={row['latent_dim']} "
                f"rope_dim={row['rope_dim']}; expected 512/64"
            )


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = []
    loaded: Dict[str, List[Dict[str, str]]] = {}
    for config, (_bits, _mode, filename) in CONFIGS.items():
        path = root / filename
        if not path.is_file():
            missing.append(filename)
            continue
        rows = _load(path)
        if not rows:
            missing.append(filename)
            continue
        _check_rope_is_unquantized(config, rows)
        loaded[config] = rows

    if not loaded:
        print("No faithful_dsv4_results_*.csv found; run the sweep first.")
        return 1

    contexts = sorted({int(row["context_length"]) for rows in loaded.values() for row in rows})
    complete = contexts == sorted(DEFAULT_CONTEXTS)

    print("## Faithful DSV4 MLA ablations")
    print()
    print(
        "512-lane quantized semantic latent + 64-lane UNQUANTIZED decoupled "
        "RoPE, 64 heads, head-max aggregation, 4:1 CSA compression."
    )
    print()
    print(f"Context grid: {', '.join(f'{value:,}' for value in contexts)}")
    if not complete:
        print()
        print(
            f"> Partial grid. The full grid is "
            f"{', '.join(f'{value:,}' for value in DEFAULT_CONTEXTS)}; "
            "averages below cover only the contexts actually present."
        )
    if missing:
        print()
        print(f"> Missing {len(missing)} of {len(CONFIGS)} result files: "
              f"{', '.join(sorted(missing))}")
    print()
    print(
        "| Config | Bits | Lattice | Bits/lane | Recall@128 | Distractor FP | "
        "Mean rank | Spearman rho | Dilution | RoPE share | CF phase err (deg) | "
        "Latency (s) | Packed latent (MiB) |"
    )
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    widest = max(contexts)
    for config, rows in loaded.items():
        bits, mode, _filename = CONFIGS[config]
        lattice = mode.removeprefix("lloydmax").lstrip("_") or "scalar"
        widest_rows = [row for row in rows if int(row["context_length"]) == widest]
        reference = widest_rows[0] if widest_rows else rows[-1]

        # The ACTUAL stored cost per semantic lane, measured from the payload
        # rather than taken from the config's name. `bits` names the Lloyd-Max
        # index width only; the group scales and -- far more significantly --
        # the int16 lattice coordinates are also on the wire.
        slots = int(reference["compressed_slots"])
        latent_dim = int(reference["latent_dim"])
        payload_bytes = float(reference["packed_latent_stream_mib"]) * 1024.0 * 1024.0
        bits_per_lane = payload_bytes * 8.0 / (slots * latent_dim) if slots else 0.0

        print(
            f"| {config} | {bits} | {lattice} | {bits_per_lane:.2f} | "
            f"{mean(float(row['mean_needle_recall_at_128']) for row in rows):.4f} | "
            f"{mean(float(row['distractor_false_positive_rate']) for row in rows):.4f} | "
            f"{mean(float(row['mean_needle_absolute_rank']) for row in rows):.1f} | "
            f"{mean(float(row['mean_spearman_rho']) for row in rows):.6f} | "
            f"{mean(float(row['mean_dilution']) for row in rows):.4f} | "
            f"{mean(float(row['mean_rope_logit_share']) for row in rows):.3f} | "
            f"{mean(float(row['counterfactual_rope_phase_error_deg']) for row in rows):.2f} | "
            f"{float(reference['stream_latency_s']):.1f} | "
            f"{float(reference['packed_latent_stream_mib']):.1f} |"
        )

    print()
    print(
        f"Recall, FP rate, rank, rho, dilution and phase error are averaged over "
        f"the context grid. Latency and packed size are for the widest context "
        f"present ({widest:,} tokens)."
    )
    print()
    print(
        "**`Bits/lane` is measured from the payload, and the lattice rows are "
        "NOT iso-bitrate with the scalar rows.** `Bits` names the Lloyd-Max "
        "index width only. `PackedVector` stores every lattice coordinate as a "
        "raw int16, so an E8/E24 row carries 2 bytes per semantic lane on top "
        "of its 2-4 bit index -- ~19-21 bits/lane against the scalar path's "
        "3-5. So \"2-bit E8 matches 4-bit scalar\" is a statement about "
        "quality at ~4x the storage, not a rate-distortion win; at equal "
        "footprint the scalar path is ahead. Entropy-coding the coset index "
        "(E8 needs ~8 bits per 8D block, ~1 bit/lane) is what would make the "
        "comparison fair, and it is not implemented here."
    )
    print()
    print(
        "`CF phase err` is the COUNTERFACTUAL rotary phase error: what the same "
        "codec would have done to the 64 rotary lanes had they been quantized "
        "with the latent. The shipped path's rotary error is zero, and that is "
        "structural -- `DecoupledMlaCache` has no code path that quantizes rope."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
