#!/usr/bin/env python3
"""Deterministic CPU Needle-in-a-Haystack probe for the quantized CSA cache."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Iterable, List, Sequence, Tuple

try:
    from .turbo_lattice_vq import (
        CSACompressor,
        PagedQuantizedCache,
        TurboLatticeQuantizer,
        cosine_similarity,
    )
except ImportError:  # Direct execution from tests/research.
    from turbo_lattice_vq import (  # type: ignore
        CSACompressor,
        PagedQuantizedCache,
        TurboLatticeQuantizer,
        cosine_similarity,
    )


@dataclass(frozen=True)
class ProbeResult:
    context_length: int
    depth: float
    needle_position: int
    compressed_slot: int
    needle_rank: int
    effective_top_k: int
    recall: int
    spearman_rho: float
    dilution: float


def _unit_gaussian(rng: random.Random, dimension: int) -> Tuple[float, ...]:
    vector = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def _correlated_query(
    needle: Sequence[float], rng: random.Random, noise: float
) -> Tuple[float, ...]:
    perturbation = _unit_gaussian(rng, len(needle))
    vector = [float(value) + noise * perturbation[index] for index, value in enumerate(needle)]
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def _rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) * 0.5 + 1.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must have the same non-zero length")
    if len(left) == 1:
        return 1.0
    rank_left = _rankdata(left)
    rank_right = _rankdata(right)
    mean_left = sum(rank_left) / len(rank_left)
    mean_right = sum(rank_right) / len(rank_right)
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(rank_left, rank_right)
    )
    variance_left = sum((value - mean_left) ** 2 for value in rank_left)
    variance_right = sum((value - mean_right) ** 2 for value in rank_right)
    denominator = math.sqrt(variance_left * variance_right)
    return covariance / denominator if denominator else 1.0


def build_quantizer(
    dimension: int,
    group_size: int,
    seed: int,
    calibration_vectors: int,
    lattice_step: float,
) -> TurboLatticeQuantizer:
    rng = random.Random(seed ^ 0x5A17C9E3)
    calibration = [_unit_gaussian(rng, dimension) for _ in range(calibration_vectors)]
    return TurboLatticeQuantizer.calibrate(
        calibration,
        group_size=group_size,
        levels=16,
        lattice_step=lattice_step,
    )


def run_probe(
    context_length: int,
    depth: float,
    quantizer: TurboLatticeQuantizer,
    *,
    seed: int = 20251012,
    compression_ratio: int = 4,
    block_size: int = 128,
    requested_top_k: int = 512,
    query_noise: float = 0.05,
) -> ProbeResult:
    if context_length <= 0 or context_length % compression_ratio:
        raise ValueError("context length must be positive and divisible by compression_ratio")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("needle depth must be in [0, 1]")
    dimension = quantizer.group_size * (128 // quantizer.group_size)
    if dimension != 128:
        raise ValueError("the NIAH PA_BSND probe requires head_dim=128")

    case_seed = seed + context_length * 1009 + int(round(depth * 1000.0)) * 9176
    rng = random.Random(case_seed)
    needle_rng = random.Random(seed ^ 0x0EED1E)
    needle = _unit_gaussian(needle_rng, dimension)
    query = _correlated_query(needle, needle_rng, query_noise)
    needle_position = min(context_length - 1, int(round(depth * (context_length - 1))))

    tokens: List[Tuple[float, ...]] = []
    for position in range(context_length):
        tokens.append(needle if position == needle_position else _unit_gaussian(rng, dimension))

    compressed_length = context_length // compression_ratio
    blocks = (compressed_length + block_size - 1) // block_size
    cache = PagedQuantizedCache(blocks, block_size, dimension, quantizer)
    compressor = CSACompressor(compression_ratio)
    exact_compressed = compressor.compress_into_cache(tokens, tokens, cache)

    effective_top_k = min(
        compressed_length,
        max(1, (requested_top_k + compression_ratio - 1) // compression_ratio),
    )
    attention = cache.attention(query, compressed_length, top_k=effective_top_k)
    baseline_scores = [
        sum(q * key_lane for q, key_lane in zip(query, key)) / math.sqrt(dimension)
        for key in exact_compressed
    ]
    rho = spearman_rank_correlation(baseline_scores, attention.scores)

    needle_slot = needle_position // compression_ratio
    needle_rank = attention.ranked_indices.index(needle_slot) + 1
    decoded_key, _ = cache.read(needle_slot)
    exact_cosine = cosine_similarity(query, needle)
    quantized_compressed_cosine = cosine_similarity(query, decoded_key)
    dilution = quantized_compressed_cosine / exact_cosine if exact_cosine else 0.0
    return ProbeResult(
        context_length=context_length,
        depth=depth,
        needle_position=needle_position,
        compressed_slot=needle_slot,
        needle_rank=needle_rank,
        effective_top_k=effective_top_k,
        recall=int(needle_rank <= effective_top_k),
        spearman_rho=rho,
        dilution=dilution,
    )


def run_grid(
    context_lengths: Iterable[int],
    depths: Iterable[float],
    *,
    seed: int = 20251012,
    group_size: int = 32,
    lattice_step: float = 4.0,
    calibration_vectors: int = 256,
    compression_ratio: int = 4,
    block_size: int = 128,
    requested_top_k: int = 512,
    query_noise: float = 0.05,
) -> List[ProbeResult]:
    contexts = tuple(context_lengths)
    depth_values = tuple(depths)
    quantizer = build_quantizer(
        128, group_size, seed, calibration_vectors, lattice_step
    )
    return [
        run_probe(
            context,
            depth,
            quantizer,
            seed=seed,
            compression_ratio=compression_ratio,
            block_size=block_size,
            requested_top_k=requested_top_k,
            query_noise=query_noise,
        )
        for context in contexts
        for depth in depth_values
    ]


def format_grid(results: Sequence[ProbeResult]) -> str:
    contexts = sorted({result.context_length for result in results})
    depths = sorted({result.depth for result in results})
    by_case = {(result.context_length, result.depth): result for result in results}
    header = ["context"] + [f"{depth:.0%}" for depth in depths]
    rows = [header]
    for context in contexts:
        row = [str(context)]
        for depth in depths:
            result = by_case[(context, depth)]
            row.append(
                f"R={result.recall} rank={result.needle_rank} "
                f"rho={result.spearman_rho:.4f} dil={result.dilution:.4f}"
            )
        rows.append(row)
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    rendered = []
    for row_number, row in enumerate(rows):
        rendered.append(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_number == 0:
            rendered.append("-+-".join("-" * width for width in widths))
    return "\n".join(rendered)


def _parse_ints(text: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_depths(text: str) -> Tuple[float, ...]:
    output = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item[:-1]) / 100.0 if item.endswith("%") else float(item)
        output.append(value)
    return tuple(output)


def _write_csv(path: Path, results: Sequence[ProbeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(ProbeResult.__dataclass_fields__.keys())
        for result in results:
            writer.writerow(
                getattr(result, field) for field in ProbeResult.__dataclass_fields__.keys()
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-lengths", default="1024,4096,8192,16384")
    parser.add_argument("--depths", default="10%,30%,50%,70%,90%")
    parser.add_argument("--seed", type=int, default=20251012)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--lattice-step", type=float, default=4.0)
    parser.add_argument("--calibration-vectors", type=int, default=256)
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--query-noise", type=float, default=0.05)
    parser.add_argument("--csv", type=Path, help="also write one row per grid point")
    args = parser.parse_args(argv)

    results = run_grid(
        _parse_ints(args.context_lengths),
        _parse_depths(args.depths),
        seed=args.seed,
        group_size=args.group_size,
        lattice_step=args.lattice_step,
        calibration_vectors=args.calibration_vectors,
        compression_ratio=args.compression_ratio,
        block_size=args.block_size,
        requested_top_k=args.top_k,
        query_noise=args.query_noise,
    )
    effective = results[0].effective_top_k if results else 0
    print(
        f"seed={args.seed} PA_BSND head_dim=128 ratio={args.compression_ratio} "
        f"requested_top_k={args.top_k} effective_compressed_top_k={effective}"
    )
    print("cell: Recall, 1-based needle slot rank, Spearman rho, dilution")
    print(format_grid(results))
    if args.csv:
        _write_csv(args.csv, results)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
