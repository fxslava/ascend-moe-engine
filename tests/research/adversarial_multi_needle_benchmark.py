#!/usr/bin/env python3
"""Streaming anisotropic 16-needle / 16-distractor cache benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import math
from pathlib import Path
import time
from typing import Sequence

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - target environment has NumPy.
    raise SystemExit("This research benchmark requires NumPy") from error

try:
    from .extreme_niah_benchmark import (
        _decode_batch,
        _encode_batch,
        _memory_mib,
        _pack_indices_batch,
        _unpack_indices_batch,
        _top_heap_update,
    )
    from .niah_attention_regression import spearman_rank_correlation
    from .turbo_lattice_vq import TurboLatticeQuantizer, _pack_indices, _unpack_indices
except ImportError:  # Direct execution from tests/research.
    from extreme_niah_benchmark import (  # type: ignore
        _decode_batch,
        _encode_batch,
        _memory_mib,
        _pack_indices_batch,
        _unpack_indices_batch,
        _top_heap_update,
    )
    from niah_attention_regression import spearman_rank_correlation  # type: ignore
    from turbo_lattice_vq import (  # type: ignore
        TurboLatticeQuantizer,
        _pack_indices,
        _unpack_indices,
    )


HEAD_DIM = 128
COMPRESSION_RATIO = 4
TOP_K = 128
NEEDLE_COUNT = 16
DISTRACTOR_COUNT = 16
OUTLIER_CHANNELS = (0, 16, 64, 96)
OUTLIER_GAIN = 8.0
QUERY_SNR_DB = 9.0
DEFAULT_CONTEXTS = (8192, 32768, 262144, 1048576)
DEFAULT_CHUNK_TOKENS = 8192
DEFAULT_SEED = 20251012
MAX_PEAK_RSS_MIB = 120.0


def _result_filename(bits: int, mode: str) -> str:
    suffix = "" if mode == "lloydmax" else f"_{mode.removeprefix('lloydmax_')}"
    return f"adversarial_results_{bits}bit_lloydmax{suffix}.csv"


CONFIGS = {
    f"cfg{index}_tq{bits}_{mode}": (bits, mode, _result_filename(bits, mode))
    for index, (bits, mode) in enumerate(
        (
            (2, "lloydmax"),
            (2, "lloydmax_e8"),
            (2, "lloydmax_e24"),
            (3, "lloydmax"),
            (3, "lloydmax_e8"),
            (3, "lloydmax_e24"),
            (4, "lloydmax"),
            (4, "lloydmax_e8"),
            (4, "lloydmax_e24"),
        ),
        start=1,
    )
}


@dataclass(frozen=True)
class AdversarialResult:
    context_length: int
    bit_width: int
    mode: str
    mean_needle_recall_at_128: float
    min_needle_recall: float
    needles_found: int
    total_needles: int
    distractor_false_positive_rate: float
    mean_needle_rank: float
    mean_spearman_rho: float
    mean_dilution: float
    stream_latency_s: float
    peak_rss_mib: float
    packed_key_stream_mib: float
    seed: int


def verify_2bit_packing() -> None:
    quantizer = TurboLatticeQuantizer.for_standard_normal(2, "none", group_size=32)
    expected_centres = (-1.510418, -0.452794, 0.452794, 1.510418)
    expected_thresholds = (-0.981606, 0.0, 0.981606)
    if quantizer.lloyd.centres != expected_centres:
        raise AssertionError("2-bit Gaussian Lloyd-Max centres do not match specification")
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(quantizer.lloyd.thresholds, expected_thresholds)
    ):
        raise AssertionError("2-bit Gaussian Lloyd-Max thresholds do not match specification")

    indices = tuple(index % 4 for index in range(128))
    packed = _pack_indices(indices, 2)
    if len(packed) != 32:
        raise AssertionError("128 two-bit indices must occupy exactly 32 bytes")
    if packed != bytes((0xE4,)) * 32:
        raise AssertionError("two-bit packing must store four low-to-high indices per byte")
    if _unpack_indices(packed, 128, 2) != indices:
        raise AssertionError("two-bit packed indices did not round-trip")
    batched = _pack_indices_batch(np.asarray(indices, dtype=np.uint8)[None, :], 2)
    if batched.shape != (1, 32) or batched[0].tobytes() != packed:
        raise AssertionError("vectorized two-bit packing differs from the scalar format")
    if tuple(_unpack_indices_batch(batched, 128, 2)[0].tolist()) != indices:
        raise AssertionError("vectorized two-bit packed indices did not round-trip")


def _make_transform(seed: int) -> "np.ndarray":
    rng = np.random.default_rng(seed & ((1 << 64) - 1))
    basis, triangular = np.linalg.qr(rng.standard_normal((HEAD_DIM, HEAD_DIM)))
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    eigenvectors = basis * signs[None, :]
    eigenvalues = np.exp(
        -2.5 * np.arange(HEAD_DIM, dtype=np.float64) / HEAD_DIM
    )
    return (
        np.sqrt(eigenvalues).astype(np.float32)[:, None]
        * eigenvectors.T.astype(np.float32)
    )


def _transform_and_normalize(
    standard_normal: "np.ndarray", transform: "np.ndarray"
) -> "np.ndarray":
    vectors = standard_normal @ transform
    vectors[:, OUTLIER_CHANNELS] *= np.float32(OUTLIER_GAIN)
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, keepdims=True, dtype=np.float32))
    vectors /= np.maximum(norms, np.float32(1e-12))
    return vectors


def _background_chunk(
    rng: "np.random.Generator", count: int, transform: "np.ndarray"
) -> "np.ndarray":
    standard_normal = rng.standard_normal((count, HEAD_DIM), dtype=np.float32)
    return _transform_and_normalize(standard_normal, transform)


def _unit(vector: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalize a degenerate vector")
    return (vector / np.float32(norm)).astype(np.float32, copy=False)


def _orthogonal_unit(
    reference: "np.ndarray", rng: "np.random.Generator", transform: "np.ndarray | None"
) -> "np.ndarray":
    for _ in range(8):
        vector = rng.standard_normal(HEAD_DIM, dtype=np.float32)
        if transform is not None:
            vector = _transform_and_normalize(vector[None, :], transform)[0]
        vector = vector - np.dot(vector, reference) * reference
        norm = float(np.linalg.norm(vector))
        if norm > 1e-7:
            return (vector / np.float32(norm)).astype(np.float32, copy=False)
    raise RuntimeError("failed to construct a non-degenerate orthogonal vector")


def _make_needles_queries_and_distractors(
    seed: int, transform: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    seed = seed & ((1 << 64) - 1)
    needle_rng = np.random.default_rng(seed ^ 0x4E4545444C455F31)
    noise_rng = np.random.default_rng(seed ^ 0x51554552595F3031)
    distractor_rng = np.random.default_rng(seed ^ 0x4449535452414354)
    needles = _transform_and_normalize(
        needle_rng.standard_normal((NEEDLE_COUNT, HEAD_DIM), dtype=np.float32),
        transform,
    )
    queries = np.empty_like(needles)
    noise_norm = np.float32(10.0 ** (-QUERY_SNR_DB / 20.0))
    for index, needle in enumerate(needles):
        noise = _orthogonal_unit(needle, noise_rng, transform)
        queries[index] = _unit(needle + noise_norm * noise)

    distractors = np.empty_like(needles)
    cosine_targets = distractor_rng.uniform(0.70, 0.82, NEEDLE_COUNT)
    for index, query in enumerate(queries):
        orthogonal = _orthogonal_unit(query, distractor_rng, None)
        cosine = np.float32(cosine_targets[index])
        distractors[index] = (
            cosine * query + np.sqrt(np.float32(1.0) - cosine * cosine) * orthogonal
        )
    return needles, queries, distractors


def _place_adversaries(
    context_length: int, seed: int
) -> tuple["np.ndarray", "np.ndarray"]:
    depths = np.linspace(0.05, 0.95, NEEDLE_COUNT)
    needle_positions = np.rint(depths * (context_length - 1)).astype(np.int64)
    needle_slots = needle_positions // COMPRESSION_RATIO
    if len(np.unique(needle_slots)) != NEEDLE_COUNT:
        raise ValueError("context is too short to place 16 distinct needle slots")

    compressed_slots = context_length // COMPRESSION_RATIO
    allowed = np.ones(compressed_slots, dtype=np.bool_)
    for slot in needle_slots:
        allowed[max(0, int(slot) - 1) : min(compressed_slots, int(slot) + 2)] = False
    available = np.flatnonzero(allowed)
    if available.size < DISTRACTOR_COUNT:
        raise ValueError("context is too short to place distinct distractors")

    rng = np.random.default_rng((seed ^ context_length ^ 0x44535452504C4143) & ((1 << 64) - 1))
    distractor_slots = rng.choice(available, size=DISTRACTOR_COUNT, replace=False)
    distractor_offsets = rng.integers(0, COMPRESSION_RATIO, size=DISTRACTOR_COUNT)
    distractor_positions = distractor_slots * COMPRESSION_RATIO + distractor_offsets
    return needle_positions, distractor_positions.astype(np.int64)


def _cosine(left: "np.ndarray", right: "np.ndarray") -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right)) / denominator if denominator else 0.0


def _precompute_needle_scores(
    context_length: int,
    chunk_tokens: int,
    seed: int,
    transform: "np.ndarray",
    needles: "np.ndarray",
    queries: "np.ndarray",
    needle_positions: "np.ndarray",
    quantizer: TurboLatticeQuantizer,
) -> "np.ndarray":
    """Score needle slots in matching-sized chunks before the rank pass."""
    rng = np.random.default_rng((seed ^ 0x484159535441434B) & ((1 << 64) - 1))
    scores = np.empty(NEEDLE_COUNT, dtype=np.float32)
    inverse_temperature = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for token_start in range(0, context_length, chunk_tokens):
        token_stop = min(context_length, token_start + chunk_tokens)
        standard_normal = rng.standard_normal(
            (token_stop - token_start, HEAD_DIM), dtype=np.float32
        )
        first_needle = int(np.searchsorted(needle_positions, token_start))
        last_needle = int(np.searchsorted(needle_positions, token_stop))
        if first_needle == last_needle:
            del standard_normal
            continue
        tokens = _transform_and_normalize(standard_normal, transform)
        del standard_normal
        for needle_index in range(first_needle, last_needle):
            tokens[int(needle_positions[needle_index]) - token_start] = needles[needle_index]
        compressed = np.mean(
            tokens.reshape(-1, COMPRESSION_RATIO, HEAD_DIM), axis=1, dtype=np.float32
        )
        packed = _encode_batch(compressed, quantizer)
        decoded = _decode_batch(packed, quantizer)
        quantized_scores = (decoded @ queries.T) * inverse_temperature
        first_slot = token_start // COMPRESSION_RATIO
        for needle_index in range(first_needle, last_needle):
            slot = int(needle_positions[needle_index] // COMPRESSION_RATIO)
            scores[needle_index] = quantized_scores[slot - first_slot, needle_index]
        del tokens, compressed, packed, decoded, quantized_scores
    return scores


def run_context(
    context_length: int,
    bits: int,
    mode: str,
    *,
    seed: int = DEFAULT_SEED,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    sample_slots: int = 4096,
) -> AdversarialResult:
    if context_length <= 0 or context_length % COMPRESSION_RATIO:
        raise ValueError("context length must be a positive multiple of four")
    if chunk_tokens <= 0 or chunk_tokens % COMPRESSION_RATIO:
        raise ValueError("chunk size must be a positive multiple of four")
    if sample_slots <= 0:
        raise ValueError("sample_slots must be positive")
    if mode not in ("lloydmax", "lloydmax_e8", "lloydmax_e24"):
        raise ValueError("mode must be lloydmax, lloydmax_e8, or lloydmax_e24")

    verify_2bit_packing()
    quantizer = TurboLatticeQuantizer.for_standard_normal(
        bits, mode.removeprefix("lloydmax").lstrip("_") or "none", group_size=32
    )
    started = time.perf_counter()
    transform = _make_transform(seed ^ 0x434F56415249414E)
    needles, queries, distractors = _make_needles_queries_and_distractors(seed, transform)
    needle_positions, distractor_positions = _place_adversaries(context_length, seed)
    needle_scores = _precompute_needle_scores(
        context_length,
        chunk_tokens,
        seed,
        transform,
        needles,
        queries,
        needle_positions,
        quantizer,
    )
    special_positions = np.concatenate((needle_positions, distractor_positions))
    special_vectors = np.concatenate((needles, distractors), axis=0)
    special_order = np.argsort(special_positions)
    special_positions = special_positions[special_order]
    special_vectors = special_vectors[special_order]
    special_slots = special_positions // COMPRESSION_RATIO
    special_slot_lookup = {
        int(slot): int(original_index)
        for slot, original_index in zip(special_slots, special_order)
    }

    compressed_count = context_length // COMPRESSION_RATIO
    if compressed_count <= TOP_K:
        raise ValueError("context must contain more than 128 compressed slots")
    background_slots = np.ones(compressed_count, dtype=np.bool_)
    background_slots[special_slots] = False
    sample_candidates = np.flatnonzero(background_slots)
    sample_count = min(sample_slots, sample_candidates.size)
    sample_rng = np.random.default_rng((seed ^ context_length ^ 0x53504541524D414E) & ((1 << 64) - 1))
    sampled_slots = np.sort(sample_rng.choice(sample_candidates, size=sample_count, replace=False))
    sampled_exact = np.empty((sample_count, NEEDLE_COUNT), dtype=np.float32)
    sampled_quantized = np.empty_like(sampled_exact)
    needle_slots = (needle_positions // COMPRESSION_RATIO).astype(np.int64)
    needle_ranks = np.zeros(NEEDLE_COUNT, dtype=np.int64)
    top_heaps: list[list[tuple[float, int]]] = [[] for _ in range(NEEDLE_COUNT)]
    special_exact_scores = np.empty((NEEDLE_COUNT + DISTRACTOR_COUNT, NEEDLE_COUNT), dtype=np.float32)
    special_quantized_scores = np.empty_like(special_exact_scores)
    special_exact_keys = np.empty((NEEDLE_COUNT + DISTRACTOR_COUNT, HEAD_DIM), dtype=np.float32)
    special_quantized_keys = np.empty_like(special_exact_keys)
    packed_key_bytes = 0
    inverse_temperature = np.float32(1.0 / math.sqrt(HEAD_DIM))
    stream_rng = np.random.default_rng((seed ^ 0x484159535441434B) & ((1 << 64) - 1))

    for token_start in range(0, context_length, chunk_tokens):
        token_stop = min(context_length, token_start + chunk_tokens)
        tokens = _background_chunk(stream_rng, token_stop - token_start, transform)
        first_special = int(np.searchsorted(special_positions, token_start))
        last_special = int(np.searchsorted(special_positions, token_stop))
        for special_index in range(first_special, last_special):
            tokens[int(special_positions[special_index]) - token_start] = special_vectors[special_index]

        compressed = np.mean(
            tokens.reshape(-1, COMPRESSION_RATIO, HEAD_DIM), axis=1, dtype=np.float32
        )
        packed = _encode_batch(compressed, quantizer)
        decoded = _decode_batch(packed, quantizer)
        packed_key_bytes += packed.nbytes
        exact_scores = (compressed @ queries.T) * inverse_temperature
        quantized_scores = (decoded @ queries.T) * inverse_temperature
        first_slot = token_start // COMPRESSION_RATIO
        last_slot = first_slot + compressed.shape[0]
        candidate_slots = np.arange(first_slot, last_slot)
        for query_index in range(NEEDLE_COUNT):
            _top_heap_update(top_heaps[query_index], quantized_scores[:, query_index], first_slot, TOP_K)
            target_score = needle_scores[query_index]
            target_slot = needle_slots[query_index]
            needle_ranks[query_index] += int(
                np.count_nonzero(
                    (quantized_scores[:, query_index] > target_score)
                    | (
                        (quantized_scores[:, query_index] == target_score)
                        & (candidate_slots < target_slot)
                    )
                )
            )

        sample_begin = int(np.searchsorted(sampled_slots, first_slot, side="left"))
        sample_end = int(np.searchsorted(sampled_slots, last_slot, side="left"))
        if sample_begin < sample_end:
            local = sampled_slots[sample_begin:sample_end] - first_slot
            sampled_exact[sample_begin:sample_end] = exact_scores[local]
            sampled_quantized[sample_begin:sample_end] = quantized_scores[local]

        for special_slot, special_index in special_slot_lookup.items():
            if first_slot <= special_slot < last_slot:
                local = special_slot - first_slot
                special_exact_scores[special_index] = exact_scores[local]
                special_quantized_scores[special_index] = quantized_scores[local]
                special_exact_keys[special_index] = compressed[local]
                special_quantized_keys[special_index] = decoded[local]
        del tokens, compressed, packed, decoded, exact_scores, quantized_scores

    ranks = []
    distractor_ranks = []
    recalls = []
    dilutions = []
    rhos = []
    distractor_slots = (distractor_positions // COMPRESSION_RATIO).astype(np.int64)
    for query_index in range(NEEDLE_COUNT):
        needle_score = float(special_quantized_scores[query_index, query_index])
        distractor_index = NEEDLE_COUNT + query_index
        distractor_score = float(special_quantized_scores[distractor_index, query_index])
        needle_rank = int(needle_ranks[query_index])
        ranks.append(needle_rank)
        recalls.append(int(needle_rank <= TOP_K))
        distractor_candidate = (distractor_score, -int(distractor_slots[query_index]))
        distractor_in_top_k = distractor_candidate in top_heaps[query_index]
        distractor_ranks.append(int(distractor_in_top_k))
        exact_cosine = _cosine(queries[query_index], special_exact_keys[query_index])
        decoded_cosine = _cosine(queries[query_index], special_quantized_keys[query_index])
        dilutions.append(decoded_cosine / exact_cosine if exact_cosine else 0.0)

        exact_samples = np.concatenate(
            (sampled_exact[:, query_index], special_exact_scores[:, query_index])
        )
        quantized_samples = np.concatenate(
            (sampled_quantized[:, query_index], special_quantized_scores[:, query_index])
        )
        rhos.append(spearman_rank_correlation(exact_samples.tolist(), quantized_samples.tolist()))

    elapsed = time.perf_counter() - started
    _, peak_rss_mib = _memory_mib()
    if peak_rss_mib >= MAX_PEAK_RSS_MIB:
        raise MemoryError(
            f"peak RSS {peak_rss_mib:.1f} MiB exceeds the {MAX_PEAK_RSS_MIB:.0f} MiB limit"
        )
    return AdversarialResult(
        context_length=context_length,
        bit_width=bits,
        mode=mode,
        mean_needle_recall_at_128=float(sum(recalls) / NEEDLE_COUNT),
        min_needle_recall=float(min(recalls)),
        needles_found=sum(recalls),
        total_needles=NEEDLE_COUNT,
        distractor_false_positive_rate=float(
            sum(distractor_ranks) / DISTRACTOR_COUNT
        ),
        mean_needle_rank=float(sum(ranks) / NEEDLE_COUNT),
        mean_spearman_rho=float(sum(rhos) / NEEDLE_COUNT),
        mean_dilution=float(sum(dilutions) / NEEDLE_COUNT),
        stream_latency_s=elapsed,
        peak_rss_mib=peak_rss_mib,
        packed_key_stream_mib=packed_key_bytes / (1024.0 * 1024.0),
        seed=seed,
    )


def write_results(path: Path, rows: Sequence[AdversarialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(AdversarialResult)]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: getattr(row, column) for column in columns} for row in rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=tuple(CONFIGS))
    parser.add_argument("--context-lengths", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--sample-slots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--verify-2bit-packing-only", action="store_true")
    args = parser.parse_args(argv)

    verify_2bit_packing()
    if args.verify_2bit_packing_only:
        print("2-bit Lloyd-Max codebook and four-indices-per-byte packing: PASS")
        return 0
    if args.config is None:
        parser.error("--config is required unless --verify-2bit-packing-only is used")
    bits, mode, filename = CONFIGS[args.config]
    contexts = tuple(int(value.strip()) for value in args.context_lengths.split(",") if value.strip())
    if not contexts:
        parser.error("at least one context length is required")

    results = [
        run_context(
            context,
            bits,
            mode,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            sample_slots=args.sample_slots,
        )
        for context in contexts
    ]
    path = args.csv or Path(__file__).with_name(filename)
    write_results(path, results)
    print(f"{args.config} (seed={args.seed}, chunk={args.chunk_tokens})")
    print("| Context | Recall@128 | Distractor FP | Mean rank | Mean rho | Latency | Peak RSS | Packed keys |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row.context_length:,} | {row.mean_needle_recall_at_128:.4f} | "
            f"{row.distractor_false_positive_rate:.4f} | {row.mean_needle_rank:.1f} | "
            f"{row.mean_spearman_rho:.6f} | {row.stream_latency_s:.3f}s | "
            f"{row.peak_rss_mib:.1f} MiB | {row.packed_key_stream_mib:.1f} MiB |"
        )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
