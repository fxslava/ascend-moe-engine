#!/usr/bin/env python3
"""Streaming extreme-context NIAH benchmark for CSA + Turbo-Lattice VQ.

The benchmark shares one deterministic haystack pass across all requested
needle depths for a context length.  A chunk is generated, reduced at the 4:1
CSA cadence, encoded to the selected packed TurboQuant/Lloyd-Max/lattice
representation, decoded for scoring, and immediately discarded.  Resident
state is limited to a top-K heap, sampled score pairs, and the current chunk.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass, fields
import heapq
import math
from pathlib import Path
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - exercised on minimal hosts.
    raise SystemExit(
        "extreme_niah_benchmark.py requires NumPy for bounded-memory vectorized "
        "execution; run it in the documented CANN Python 3.12 container"
    ) from error

try:
    from .niah_attention_regression import build_quantizer, spearman_rank_correlation
    from .turbo_lattice_vq import (
        E4M3FN,
        TurboLatticeQuantizer,
        golay_decoder_tables,
    )
except ImportError:  # Direct execution from tests/research.
    from niah_attention_regression import (  # type: ignore
        build_quantizer,
        spearman_rank_correlation,
    )
    from turbo_lattice_vq import (  # type: ignore
        E4M3FN,
        TurboLatticeQuantizer,
        golay_decoder_tables,
    )


DEFAULT_CONTEXTS = (8192, 32768, 262144, 1048576)
DEFAULT_DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_RESULTS = Path(__file__).with_name("extreme_niah_results.csv")
HEAD_DIM = 128
LEECH_DIMENSION = 24
LEECH_NORMALIZATION = math.sqrt(8.0)

ABLATION_CONFIGS = {
    "cfg1_tq4_lloydmax": (4, "none", "niah_results_4bit_lloydmax.csv"),
    "cfg2_tq4_lloydmax_e8": (4, "e8", "niah_results_4bit_lloydmax_e8.csv"),
    "cfg3_tq4_lloydmax_e24": (4, "e24", "niah_results_4bit_lloydmax_e24.csv"),
    "cfg4_tq3_lloydmax": (3, "none", "niah_results_3bit_lloydmax.csv"),
    "cfg5_tq3_lloydmax_e8": (3, "e8", "niah_results_3bit_lloydmax_e8.csv"),
    "cfg6_tq3_lloydmax_e24": (3, "e24", "niah_results_3bit_lloydmax_e24.csv"),
}


@dataclass(frozen=True)
class ExtremeProbeResult:
    context_length: int
    depth: float
    needle_position: int
    compressed_slot: int
    compressed_slots: int
    needle_absolute_rank: int
    effective_top_k: int
    recall_at_k: int
    spearman_rho: float
    spearman_samples: int
    dilution: float
    stream_latency_s: float
    microseconds_per_compressed_slot: float
    peak_rss_mib: float
    packed_key_stream_mib: float
    chunk_tokens: int
    seed: int


@dataclass(frozen=True)
class PackedBatch:
    """Packed chunk payload; no unquantized vector is retained here."""

    index_bits: int
    lattice_mode: str
    padded_dimension: int
    scales_f32: "np.ndarray"
    lloyd_indices_packed: "np.ndarray"
    lattice_codewords_i16: "np.ndarray"
    # Logical width of one encoded row. Defaults to the 128-lane probe geometry
    # so the original callers are unchanged; the MLA benchmark encodes 512-lane
    # latents through the same codec and must carry its own width here, because
    # decode cannot recover it from the packed payload alone.
    dimension: int = HEAD_DIM

    @property
    def nbytes(self) -> int:
        return int(
            self.scales_f32.nbytes
            + self.lloyd_indices_packed.nbytes
            + self.lattice_codewords_i16.nbytes
        )


def _parse_ints(text: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _parse_depths(text: str) -> Tuple[float, ...]:
    output = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        output.append(float(item[:-1]) / 100.0 if item.endswith("%") else float(item))
    return tuple(output)


def _memory_mib() -> Tuple[float, float]:
    """Return current RSS and high-water RSS, in MiB.

    The peak-RSS budget is a hard gate on these benchmarks, so the probe has
    to work on the workstation they are authored on as well as in the CANN
    container. `resource` is Unix-only and `/proc` does not exist on Windows,
    so this reads whichever of the three sources the host actually has:
    `/proc/self/status` (exact, Linux), `getrusage` (Unix fallback), or
    `GetProcessMemoryInfo` (Windows). It never raises -- a benchmark must not
    die because it could not measure itself -- and returns (0.0, 0.0) if no
    source is available, which the callers' thresholds then read as "unknown"
    rather than "fine".
    """
    current_kib = 0.0
    peak_kib = 0.0

    try:  # Linux: exact current and high-water marks.
        with Path("/proc/self/status").open("r", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    current_kib = float(line.split()[1])
                elif line.startswith("VmHWM:"):
                    peak_kib = float(line.split()[1])
    except OSError:
        pass
    if current_kib or peak_kib:
        return current_kib / 1024.0, peak_kib / 1024.0

    try:  # Other Unix: ru_maxrss is the high-water mark only.
        import resource  # noqa: PLC0415 - optional, platform-specific.

        peak_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak_kib / 1024.0, peak_kib / 1024.0
    except (ImportError, OSError, ValueError):
        pass

    try:  # Windows: PROCESS_MEMORY_COUNTERS via psapi.
        import ctypes  # noqa: PLC0415 - optional, platform-specific.
        from ctypes import wintypes  # noqa: PLC0415

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # The signatures must be declared: GetCurrentProcess returns the
        # pseudo-handle (HANDLE)-1, and without an explicit HANDLE restype
        # ctypes hands it on as a 32-bit int, which the call then rejects.
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return (
                counters.WorkingSetSize / (1024.0 * 1024.0),
                counters.PeakWorkingSetSize / (1024.0 * 1024.0),
            )
    except (ImportError, AttributeError, OSError):
        pass

    return 0.0, 0.0


def _cosine(left: "np.ndarray", right: "np.ndarray") -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right) / denominator if denominator else 0.0


def _splitmix64(values: "np.ndarray") -> "np.ndarray":
    """Vectorized counter hash used for random-access, chunk-stable keys."""
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _haystack_tokens(
    positions: "np.ndarray", dimension: int, stream_seed: int
) -> "np.ndarray":
    """Generate unit-norm Rademacher keys for arbitrary token positions."""
    token_counters = positions.astype(np.uint64, copy=False)[:, None]
    lanes = np.arange(dimension, dtype=np.uint64)[None, :]
    counters = (
        np.uint64(stream_seed)
        ^ ((token_counters + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93))
        ^ ((lanes + np.uint64(1)) * np.uint64(0xA5A3564E27F8862B))
    )
    signs = ((_splitmix64(counters) >> np.uint64(63)).astype(np.float32) * 2.0) - 1.0
    return signs * np.float32(1.0 / math.sqrt(dimension))


def _needle_and_query(seed: int, noise: float) -> Tuple["np.ndarray", "np.ndarray"]:
    needle = _haystack_tokens(
        np.array([0], dtype=np.int64), HEAD_DIM, seed ^ 0x6E6565646C65
    )[0]
    perturbation = _haystack_tokens(
        np.array([1], dtype=np.int64), HEAD_DIM, seed ^ 0x7175657279
    )[0]
    query = needle + np.float32(noise) * perturbation
    query /= np.linalg.norm(query)
    return needle, query.astype(np.float32, copy=False)


def _nearest_dn_batch(values: "np.ndarray") -> "np.ndarray":
    rounded = np.floor(values + np.float32(0.5)).astype(np.int32)
    odd_rows = (np.sum(rounded, axis=-1) & 1) != 0
    if not np.any(odd_rows):
        return rounded
    active_values = values[odd_rows]
    active_rounded = rounded[odd_rows]
    base_error = (active_values - active_rounded) ** 2
    down_error = (active_values - (active_rounded - 1)) ** 2 - base_error
    up_error = (active_values - (active_rounded + 1)) ** 2 - base_error
    adjustments = np.where(down_error <= up_error, -1, 1).astype(np.int32)
    deltas = np.minimum(down_error, up_error)
    best_lanes = np.argmin(deltas, axis=-1)
    active_rounded[np.arange(active_rounded.shape[0]), best_lanes] += adjustments[
        np.arange(active_rounded.shape[0]), best_lanes
    ]
    rounded[odd_rows] = active_rounded
    return rounded


def _nearest_e8_batch(values: "np.ndarray") -> "np.ndarray":
    shape = values.shape
    lattice_input = values.reshape(-1, 8).astype(np.float64, copy=False)
    integer_point = _nearest_dn_batch(lattice_input)
    half_base = _nearest_dn_batch(lattice_input - 0.5)
    half_point = half_base.astype(np.float64) + 0.5
    integer_error = np.sum((lattice_input - integer_point) ** 2, axis=-1)
    half_error = np.sum((lattice_input - half_point) ** 2, axis=-1)
    use_integer = integer_error <= half_error
    doubled = np.where(
        use_integer[:, None], 2 * integer_point, 2 * half_base + 1
    )
    if np.any(doubled < -32768) or np.any(doubled > 32767):
        raise OverflowError("E8 codeword does not fit packed int16")
    return doubled.astype(np.int16).reshape(shape)


def _nearest_leech24_batch(
    values: "np.ndarray", chase_bits: int = 3
) -> "np.ndarray":
    """Vectorized mirrored-Golay Chase decoder for 24D residual blocks."""
    if values.shape[-1] != LEECH_DIMENSION:
        raise ValueError("Leech lattice input must end in 24 coordinates")
    if not 0 <= chase_bits <= 8:
        raise ValueError("chase_bits must be in [0, 8]")
    original_shape = values.shape
    target = values.reshape(-1, LEECH_DIMENSION).astype(np.float64, copy=False)
    target = target * LEECH_NORMALIZATION
    rows = target.shape[0]
    row_indices = np.arange(rows)
    bit_positions = np.arange(LEECH_DIMENSION, dtype=np.uint32)
    bit_powers = np.left_shift(np.uint32(1), bit_positions)
    low, high, leaders = golay_decoder_tables()
    low_lut = np.asarray(low, dtype=np.uint32)
    high_lut = np.asarray(high, dtype=np.uint32)
    leader_lut = np.asarray(leaders, dtype=np.uint32)
    best_error = np.full(rows, np.inf, dtype=np.float64)
    best_point = np.zeros((rows, LEECH_DIMENSION), dtype=np.int32)

    for parity in (0, 1):
        offsets = np.asarray((parity, parity + 2), dtype=np.float64)
        ks = np.floor((target[:, :, None] - offsets) / 4.0 + 0.5)
        coordinates = offsets.astype(np.int32) + 4 * ks.astype(np.int32)
        costs = (target[:, :, None] - coordinates.astype(np.float64)) ** 2
        hard_bits = costs[:, :, 1] < costs[:, :, 0]
        hard_words = np.sum(
            hard_bits.astype(np.uint32) * bit_powers[None, :],
            axis=1,
            dtype=np.uint32,
        )
        if chase_bits:
            reliability = np.abs(costs[:, :, 1] - costs[:, :, 0])
            least_reliable = np.argsort(
                reliability, axis=1, kind="stable"
            )[:, :chase_bits]
        else:
            least_reliable = np.empty((rows, 0), dtype=np.int64)

        for chase_pattern in range(1 << chase_bits):
            received = hard_words.copy()
            for bit in range(chase_bits):
                if chase_pattern >> bit & 1:
                    received ^= bit_powers[least_reliable[:, bit]]
            syndrome = low_lut[received & np.uint32(0xFFF)] ^ high_lut[
                (received >> np.uint32(12)) & np.uint32(0xFFF)
            ]
            codewords = received ^ leader_lut[syndrome]
            residue_bits = (
                (codewords[:, None] >> bit_positions[None, :]) & np.uint32(1)
            ).astype(np.int64)
            point = np.take_along_axis(
                coordinates, residue_bits[:, :, None], axis=2
            )[:, :, 0]
            selected_costs = np.take_along_axis(
                costs, residue_bits[:, :, None], axis=2
            )[:, :, 0]
            errors = np.sum(selected_costs, axis=1, dtype=np.float32)
            selected_k = (
                point - parity - 2 * residue_bits.astype(np.int32)
            ) // 4
            needs_repair = (np.sum(selected_k, axis=1) & 1) != parity
            if np.any(needs_repair):
                down_delta = (target - (point - 4)) ** 2 - (target - point) ** 2
                up_delta = (target - (point + 4)) ** 2 - (target - point) ** 2
                use_down = down_delta <= up_delta
                repair_cost = np.where(use_down, down_delta, up_delta)
                repair_lane = np.argmin(repair_cost, axis=1)
                active_rows = row_indices[needs_repair]
                active_lanes = repair_lane[needs_repair]
                point = point.copy()
                point[active_rows, active_lanes] += np.where(
                    use_down[active_rows, active_lanes], -4, 4
                )
                errors = errors.copy()
                errors[active_rows] += repair_cost[active_rows, active_lanes]
            improve = errors < best_error
            best_error[improve] = errors[improve]
            best_point[improve] = point[improve]

    if np.any(best_point < -32768) or np.any(best_point > 32767):
        raise OverflowError("Leech lattice codeword does not fit packed int16")
    return best_point.astype(np.int16).reshape(original_shape)


def _pack_indices_batch(indices: "np.ndarray", bits: int) -> "np.ndarray":
    shifts = np.arange(bits, dtype=np.uint8)
    planes = ((indices[:, :, None] >> shifts) & np.uint8(1)).reshape(
        indices.shape[0], -1
    )
    return np.packbits(planes, axis=1, bitorder="little")


def _unpack_indices_batch(
    payload: "np.ndarray", count: int, bits: int
) -> "np.ndarray":
    planes = np.unpackbits(
        payload, axis=1, count=count * bits, bitorder="little"
    ).reshape(payload.shape[0], count, bits)
    powers = np.left_shift(np.uint8(1), np.arange(bits, dtype=np.uint8))
    return np.sum(planes * powers, axis=2, dtype=np.uint8)


def _fp8_quantize(values: "np.ndarray") -> "np.ndarray":
    positive_values = np.asarray(E4M3FN._positive_values(), dtype=np.float32)
    positive_codes = np.asarray(
        [code for _, code in E4M3FN._positive_table()], dtype=np.uint8
    )
    magnitude = np.minimum(np.abs(values), np.float32(448.0))
    insertion = np.searchsorted(positive_values, magnitude, side="left")
    high = np.minimum(insertion, len(positive_values) - 1)
    low = np.maximum(high - 1, 0)
    choose_low = (magnitude - positive_values[low]) <= (
        positive_values[high] - magnitude
    )
    indices = np.where((insertion > 0) & choose_low, low, high)
    codes = positive_codes[indices]
    codes = np.where((values < 0.0) & (codes != 0), codes | np.uint8(0x80), codes)
    decode_table = np.asarray(
        [E4M3FN.decode(code) for code in range(256)], dtype=np.float32
    )
    return decode_table[codes]


def _encode_batch(
    vectors: "np.ndarray", quantizer: TurboLatticeQuantizer, dimension: int = HEAD_DIM
) -> PackedBatch:
    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError(f"batch vectors must have shape [N, {dimension}]")
    if dimension % quantizer.group_size:
        raise ValueError("dimension must be divisible by the quantizer group_size")
    group_size = quantizer.group_size
    grouped = vectors.reshape(vectors.shape[0], dimension // group_size, group_size)
    if quantizer.scale_mode == "rms":
        scales = np.sqrt(np.mean(grouped * grouped, axis=-1, dtype=np.float32))
    else:
        scales = np.max(np.abs(grouped), axis=-1).astype(np.float32) / np.float32(448.0)
    scales[scales == 0.0] = 1.0
    normalized = (grouped / scales[..., None]).reshape(vectors.shape)
    fp8_values = _fp8_quantize(normalized)

    thresholds = np.asarray(quantizer.lloyd.thresholds, dtype=np.float32)
    centres = np.asarray(quantizer.lloyd.centres, dtype=np.float32)
    indices = np.searchsorted(thresholds, fp8_values, side="right").astype(np.uint8)
    residual = fp8_values.astype(np.float64) - np.asarray(
        quantizer.lloyd.centres, dtype=np.float64
    )[indices]
    padded_dimension = dimension
    if quantizer.lattice_mode == "none":
        codewords = np.empty((vectors.shape[0], 0), dtype=np.int16)
    elif quantizer.lattice_mode == "e8":
        if dimension % 8:
            raise ValueError("E8 requires a dimension divisible by eight")
        codewords = _nearest_e8_batch(
            residual / quantizer.lattice_step
        )
    else:
        padded_dimension = (
            (dimension + LEECH_DIMENSION - 1) // LEECH_DIMENSION
        ) * LEECH_DIMENSION
        padded = np.pad(residual, ((0, 0), (0, padded_dimension - dimension)))
        blocks = padded.reshape(-1, LEECH_DIMENSION)
        codewords = _nearest_leech24_batch(
            blocks / quantizer.lattice_step,
            quantizer.leech_chase_bits,
        ).reshape(vectors.shape[0], padded_dimension)
    return PackedBatch(
        index_bits=quantizer.index_bits,
        lattice_mode=quantizer.lattice_mode,
        padded_dimension=padded_dimension,
        scales_f32=scales,
        lloyd_indices_packed=_pack_indices_batch(indices, quantizer.index_bits),
        lattice_codewords_i16=codewords,
        dimension=dimension,
    )


def _decode_batch(
    packed: PackedBatch, quantizer: TurboLatticeQuantizer
) -> "np.ndarray":
    rows = packed.scales_f32.shape[0]
    if packed.index_bits != quantizer.index_bits:
        raise ValueError("packed index width does not match quantizer")
    if packed.lattice_mode != quantizer.lattice_mode:
        raise ValueError("packed lattice mode does not match quantizer")
    indices = _unpack_indices_batch(
        packed.lloyd_indices_packed, packed.dimension, packed.index_bits
    )
    centres = np.asarray(quantizer.lloyd.centres, dtype=np.float32)
    normalized = centres[indices]
    if packed.lattice_mode == "e8":
        normalized = normalized + packed.lattice_codewords_i16.astype(
            np.float32
        ) * np.float32(quantizer.lattice_step * 0.5)
    elif packed.lattice_mode == "e24":
        residual = packed.lattice_codewords_i16[:, : packed.dimension].astype(np.float32)
        normalized = normalized + residual * np.float32(
            quantizer.lattice_step / LEECH_NORMALIZATION
        )
    scales = np.repeat(packed.scales_f32, quantizer.group_size, axis=1)
    return normalized * scales


def _quantize_decode_batch(
    vectors: "np.ndarray", quantizer: TurboLatticeQuantizer
) -> Tuple["np.ndarray", int]:
    packed = _encode_batch(vectors, quantizer)
    return _decode_batch(packed, quantizer), packed.nbytes


def _top_heap_update(
    heap: List[Tuple[float, int]],
    scores: "np.ndarray",
    first_slot: int,
    capacity: int,
) -> None:
    take = min(capacity, scores.size)
    if take == scores.size:
        local_indices = np.arange(scores.size)
    else:
        local_indices = np.argpartition(scores, scores.size - take)[-take:]
    for local_index in local_indices:
        logical_index = first_slot + int(local_index)
        candidate = (float(scores[local_index]), -logical_index)
        if len(heap) < capacity:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)


def _sample_indices(compressed_slots: int, count: int, seed: int) -> Tuple[int, ...]:
    actual = min(count, compressed_slots)
    return tuple(sorted(random.Random(seed).sample(range(compressed_slots), actual)))


def _needle_positions(context_length: int, depths: Sequence[float]) -> Tuple[int, ...]:
    positions = tuple(
        min(context_length - 1, int(round(depth * (context_length - 1))))
        for depth in depths
    )
    if len(set(position // 4 for position in positions)) != len(positions):
        raise ValueError("depths map to duplicate compressed slots")
    return positions


def _replacement_vectors(
    context_length: int,
    depths: Sequence[float],
    stream_seed: int,
    needle: "np.ndarray",
) -> Tuple[Tuple[int, ...], "np.ndarray", "np.ndarray"]:
    positions = _needle_positions(context_length, depths)
    exact = []
    base = []
    for position in positions:
        group_start = (position // 4) * 4
        group_positions = np.arange(group_start, group_start + 4, dtype=np.int64)
        tokens = _haystack_tokens(group_positions, HEAD_DIM, stream_seed)
        base.append(np.mean(tokens, axis=0, dtype=np.float32))
        tokens[position - group_start] = needle
        exact.append(np.mean(tokens, axis=0, dtype=np.float32))
    return positions, np.stack(exact), np.stack(base)


def run_context_streaming(
    context_length: int,
    depths: Sequence[float],
    quantizer: TurboLatticeQuantizer,
    *,
    seed: int = 20251012,
    chunk_tokens: int = 8192,
    sample_slots: int = 4096,
    compression_ratio: int = 4,
    requested_top_k: int = 512,
    query_noise: float = 0.05,
) -> List[ExtremeProbeResult]:
    if compression_ratio != 4:
        raise ValueError("the extreme benchmark currently models CSA ratio 4")
    if context_length <= 0 or context_length % compression_ratio:
        raise ValueError("context length must be positive and divisible by four")
    if chunk_tokens <= 0 or chunk_tokens % compression_ratio:
        raise ValueError("chunk_tokens must be a positive multiple of four")
    if any(not 0.0 <= depth <= 1.0 for depth in depths):
        raise ValueError("depths must be in [0, 1]")

    started = time.perf_counter()
    stream_seed = seed ^ (context_length * 0x9E3779B1)
    needle, query = _needle_and_query(seed, query_noise)
    positions, replacement_exact, replacement_base = _replacement_vectors(
        context_length, depths, stream_seed, needle
    )
    replacement_decoded, replacement_packed_bytes = _quantize_decode_batch(
        replacement_exact, quantizer
    )
    base_target_decoded, _ = _quantize_decode_batch(replacement_base, quantizer)
    inverse_temperature = np.float32(1.0 / math.sqrt(HEAD_DIM))
    needle_scores = (replacement_decoded @ query) * inverse_temperature
    needle_slots = np.asarray([position // compression_ratio for position in positions])
    base_target_scores = (base_target_decoded @ query) * inverse_temperature

    compressed_slots = context_length // compression_ratio
    effective_top_k = min(
        compressed_slots,
        max(1, (requested_top_k + compression_ratio - 1) // compression_ratio),
    )
    samples = _sample_indices(
        compressed_slots, sample_slots, seed ^ context_length ^ 0x53504541524D414E
    )
    sample_exact: Dict[int, float] = {}
    sample_quantized: Dict[int, float] = {}
    rank_counts = np.zeros(len(depths), dtype=np.int64)
    top_heap: List[Tuple[float, int]] = []
    heap_capacity = min(compressed_slots, effective_top_k + 1)
    packed_stream_bytes = 0

    for token_start in range(0, context_length, chunk_tokens):
        token_stop = min(context_length, token_start + chunk_tokens)
        token_positions = np.arange(token_start, token_stop, dtype=np.int64)
        tokens = _haystack_tokens(token_positions, HEAD_DIM, stream_seed)
        compressed = np.mean(
            tokens.reshape(-1, compression_ratio, HEAD_DIM), axis=1, dtype=np.float32
        )
        decoded, packed_bytes = _quantize_decode_batch(compressed, quantizer)
        packed_stream_bytes += packed_bytes
        exact_scores = (compressed @ query) * inverse_temperature
        quantized_scores = (decoded @ query) * inverse_temperature
        first_slot = token_start // compression_ratio
        slot_indices = np.arange(
            first_slot, first_slot + quantized_scores.size, dtype=np.int64
        )

        better = quantized_scores[:, None] > needle_scores[None, :]
        tied_earlier = (quantized_scores[:, None] == needle_scores[None, :]) & (
            slot_indices[:, None] < needle_slots[None, :]
        )
        rank_counts += np.sum(better | tied_earlier, axis=0, dtype=np.int64)
        _top_heap_update(top_heap, quantized_scores, first_slot, heap_capacity)

        sample_begin = bisect_left(samples, first_slot)
        sample_end = bisect_left(samples, first_slot + quantized_scores.size)
        for slot in samples[sample_begin:sample_end]:
            local = slot - first_slot
            sample_exact[slot] = float(exact_scores[local])
            sample_quantized[slot] = float(quantized_scores[local])

        # Drop the only dense uncompressed allocation before the next chunk.
        del tokens, compressed, decoded, exact_scores, quantized_scores

    ranks: List[int] = []
    recalls: List[int] = []
    rhos: List[float] = []
    dilutions: List[float] = []
    exact_needle_cosine = _cosine(query, needle)
    base_top = list(top_heap)
    for case, (depth, position) in enumerate(zip(depths, positions)):
        slot = int(needle_slots[case])
        base_qualifies = (
            base_target_scores[case] > needle_scores[case]
            or (
                base_target_scores[case] == needle_scores[case]
                and slot < needle_slots[case]
            )
        )
        absolute_rank = int(rank_counts[case] - int(base_qualifies) + 1)

        scenario_candidates = [entry for entry in base_top if -entry[1] != slot]
        scenario_candidates.append((float(needle_scores[case]), -slot))
        selected = heapq.nlargest(effective_top_k, scenario_candidates)
        candidate_recall = int(any(-entry[1] == slot for entry in selected))
        rank_recall = int(absolute_rank <= effective_top_k)
        if candidate_recall != rank_recall:
            raise AssertionError("streaming top-K heap and exact rank disagree")

        exact_sample = [sample_exact[index] for index in samples]
        quantized_sample = [sample_quantized[index] for index in samples]
        if slot in sample_exact:
            sample_position = bisect_left(samples, slot)
            exact_sample[sample_position] = float(replacement_exact[case] @ query) * float(
                inverse_temperature
            )
            quantized_sample[sample_position] = float(needle_scores[case])
        else:
            exact_sample.append(
                float(replacement_exact[case] @ query) * float(inverse_temperature)
            )
            quantized_sample.append(float(needle_scores[case]))
        rhos.append(spearman_rank_correlation(exact_sample, quantized_sample))
        ranks.append(absolute_rank)
        recalls.append(rank_recall)
        dilutions.append(
            _cosine(query, replacement_decoded[case]) / exact_needle_cosine
        )

    elapsed = time.perf_counter() - started
    _, peak_rss_mib = _memory_mib()
    # The replacement payloads are real writes too, but they replace slots
    # already counted in the stream rather than increasing cache capacity.
    if replacement_packed_bytes <= 0:
        raise AssertionError("replacement vectors did not produce packed payloads")
    return [
        ExtremeProbeResult(
            context_length=context_length,
            depth=depth,
            needle_position=positions[index],
            compressed_slot=int(needle_slots[index]),
            compressed_slots=compressed_slots,
            needle_absolute_rank=ranks[index],
            effective_top_k=effective_top_k,
            recall_at_k=recalls[index],
            spearman_rho=rhos[index],
            spearman_samples=len(samples) + int(int(needle_slots[index]) not in sample_exact),
            dilution=dilutions[index],
            stream_latency_s=elapsed,
            microseconds_per_compressed_slot=elapsed * 1e6 / compressed_slots,
            peak_rss_mib=peak_rss_mib,
            packed_key_stream_mib=packed_stream_bytes / (1024.0 * 1024.0),
            chunk_tokens=chunk_tokens,
            seed=seed,
        )
        for index, depth in enumerate(depths)
    ]


def run_extreme_grid(
    contexts: Iterable[int],
    depths: Sequence[float],
    *,
    config: str | None = None,
    seed: int = 20251012,
    chunk_tokens: int = 8192,
    sample_slots: int = 4096,
    group_size: int = 32,
    lattice_step: float = 4.0,
    calibration_vectors: int = 256,
    requested_top_k: int = 512,
    query_noise: float = 0.05,
) -> List[ExtremeProbeResult]:
    if config is None:
        quantizer = build_quantizer(
            HEAD_DIM, group_size, seed, calibration_vectors, lattice_step
        )
    else:
        if config not in ABLATION_CONFIGS:
            raise ValueError(f"unknown ablation configuration: {config}")
        bits, lattice_mode, _ = ABLATION_CONFIGS[config]
        quantizer = TurboLatticeQuantizer.for_standard_normal(
            bits=bits,
            lattice_mode=lattice_mode,
            group_size=group_size,
        )
    results: List[ExtremeProbeResult] = []
    for context in contexts:
        results.extend(
            run_context_streaming(
                context,
                depths,
                quantizer,
                seed=seed,
                chunk_tokens=chunk_tokens,
                sample_slots=sample_slots,
                requested_top_k=requested_top_k,
                query_noise=query_noise,
            )
        )
    return results


def write_results(path: Path, results: Sequence[ExtremeProbeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(ExtremeProbeResult)]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in names})


def format_markdown(results: Sequence[ExtremeProbeResult]) -> str:
    lines = [
        "| Context | Depth | Rank | Recall@128 | Spearman rho | Dilution | "
        "Latency | us/slot | Peak RSS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.context_length:,} | {result.depth:.0%} | "
            f"{result.needle_absolute_rank:,} | {result.recall_at_k} | "
            f"{result.spearman_rho:.6f} | {result.dilution:.6f} | "
            f"{result.stream_latency_s:.3f}s | "
            f"{result.microseconds_per_compressed_slot:.3f} | "
            f"{result.peak_rss_mib:.1f} MiB |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-lengths", default=",".join(str(value) for value in DEFAULT_CONTEXTS)
    )
    parser.add_argument(
        "--depths", default=",".join(str(value) for value in DEFAULT_DEPTHS)
    )
    parser.add_argument("--seed", type=int, default=20251012)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--sample-slots", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--lattice-step", type=float, default=4.0)
    parser.add_argument("--calibration-vectors", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--query-noise", type=float, default=0.05)
    parser.add_argument("--config", choices=tuple(ABLATION_CONFIGS))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args(argv)

    results = run_extreme_grid(
        _parse_ints(args.context_lengths),
        _parse_depths(args.depths),
        config=args.config,
        seed=args.seed,
        chunk_tokens=args.chunk_tokens,
        sample_slots=args.sample_slots,
        group_size=args.group_size,
        lattice_step=args.lattice_step,
        calibration_vectors=args.calibration_vectors,
        requested_top_k=args.top_k,
        query_noise=args.query_noise,
    )
    effective_top_k = results[0].effective_top_k if results else 0
    config_label = args.config or "legacy_calibrated_e8"
    print(
        f"config={config_label} seed={args.seed} chunk_tokens={args.chunk_tokens} ratio=4 "
        f"requested_top_k={args.top_k} effective_compressed_top_k={effective_top_k}"
    )
    print(format_markdown(results))
    if not args.no_csv:
        if args.csv is not None:
            csv_path = args.csv
        elif args.config is not None:
            csv_path = Path(__file__).with_name(ABLATION_CONFIGS[args.config][2])
        else:
            csv_path = DEFAULT_RESULTS
        write_results(csv_path, results)
        print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
