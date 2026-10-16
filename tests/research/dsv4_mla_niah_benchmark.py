#!/usr/bin/env python3
"""Faithful DSV4 MLA needle-in-a-haystack benchmark over a decoupled CSA cache.

What makes this "faithful" as against the earlier flat-128 probe, and where
each number comes from:

* THE CACHE IS DECOUPLED. One slot is a 512-lane semantic latent
  (`kDefaultKvLoraRank`) plus a 64-lane rotary vector (`kDefaultQkRopeHeadDim`).
  The latent goes through FP8 -> Lloyd-Max -> optional E8/E24 and is stored
  packed. The rotary lanes are stored as float32 and are NEVER handed to a
  quantizer -- `DecoupledMlaCache` is the only writer and it has no path that
  would. `test_dsv4_mla_dimensions.py` asserts the invariant directly.

* ROPE IS THE COMPRESSOR'S ROPE. Half mode (`RotaryMode::kHalf`), theta =
  160000 -- "compress_rope_theta", NOT the attention path's 10000 -- and the
  table is indexed by the COMPRESSED slot index, per
  src/emu/semi_e2e_pipeline.cpp:524 and paged_compressor_decode.cpp:179.
  Order within a slot is pool -> RMSNorm -> RoPE, matching the C4A epilog.

* SCORING IS MULTI-HEAD AND DECOMPOSED. 64 heads (`kIndexerHeads`), query head
  dim 128 + 64 (`kDefaultQkNopeHeadDim` + rope), so the temperature is
  1/sqrt(192). Per head the logit is
  <W_UK[h]^T q_nope[h], C_kv[i]> + <q_rope[h], K_rope[i]>, the standard MLA
  absorption, and candidates rank by head-max so one specialized retrieval
  head can surface a needle the other 63 are indifferent to. Each needle is
  assigned a different retrieval head, so all 64 are exercised.

* RMSNORM BOUNDS THE OUTLIERS. The pooled latent is RMSNorm'd
  (`rms_norm`, mirroring `moe::emu::rms_norm_cpu`, eps 1e-6) before scoring,
  so a few high-gain channels cannot set the scale of every logit.

The headline control is the COUNTERFACTUAL rope column: the same quantizer is
applied to the needle slots' rotary lanes and discarded, purely to report the
phase error in degrees that the shipped path avoids. The pipeline's own rope
error is zero by construction, which a column of zeros alone would not prove.

References are generated in memory from a printed seed; there are no disk
fixtures, and token content is position-keyed so any slot is reproducible on
its own. Peak RSS is gated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - target environment has NumPy.
    raise SystemExit("This research benchmark requires NumPy") from error

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .extreme_niah_benchmark import (
        _decode_batch,
        _encode_batch,
        _haystack_tokens,
        _memory_mib,
    )
    from .niah_attention_regression import spearman_rank_correlation
    from .turbo_lattice_vq import (
        CSA_COMPRESS_RATIO,
        MLA_INDEXER_HEADS,
        MLA_LATENT_DIM,
        MLA_NOPE_HEAD_DIM,
        MLA_ROPE_DIM,
        RMS_NORM_EPSILON,
        TurboLatticeQuantizer,
    )
except ImportError:  # Direct execution from tests/research.
    from extreme_niah_benchmark import (  # type: ignore
        _decode_batch,
        _encode_batch,
        _haystack_tokens,
        _memory_mib,
    )
    from niah_attention_regression import spearman_rank_correlation  # type: ignore
    from turbo_lattice_vq import (  # type: ignore
        CSA_COMPRESS_RATIO,
        MLA_INDEXER_HEADS,
        MLA_LATENT_DIM,
        MLA_NOPE_HEAD_DIM,
        MLA_ROPE_DIM,
        RMS_NORM_EPSILON,
        TurboLatticeQuantizer,
    )


LATENT_DIM = MLA_LATENT_DIM          # 512, kDefaultKvLoraRank
ROPE_DIM = MLA_ROPE_DIM              # 64,  kDefaultQkRopeHeadDim
NOPE_HEAD_DIM = MLA_NOPE_HEAD_DIM    # 128, kDefaultQkNopeHeadDim
HEADS = MLA_INDEXER_HEADS            # 64,  kIndexerHeads
COMPRESSION_RATIO = CSA_COMPRESS_RATIO  # 4, kCompressRatioCsa
# "compress_rope_theta", the compressor path's base. The attention path's
# "rope_theta" is 10000 and is NOT what the compressed stream uses.
COMPRESS_ROPE_THETA = 160000.0

TOP_K = 128
NEEDLE_COUNT = 16
DISTRACTOR_COUNT = 16
QUERY_SNR_DB = 9.0
DISTRACTOR_COSINE_LOW = 0.70
DISTRACTOR_COSINE_HIGH = 0.80
GROUP_SIZE = 32

DEFAULT_CONTEXTS = (8192, 32768, 262144, 1048576)
DEFAULT_CHUNK_TOKENS = 8192
DEFAULT_SLOT_BATCH = 512
DEFAULT_SAMPLE_SLOTS = 4096
DEFAULT_SEED = 20261013
MAX_PEAK_RSS_MIB = 150.0

MODES = ("lloydmax", "lloydmax_e8", "lloydmax_e24")


def _result_filename(bits: int, mode: str) -> str:
    suffix = "" if mode == "lloydmax" else f"_{mode.removeprefix('lloydmax_')}"
    return f"faithful_dsv4_results_{bits}bit_lloydmax{suffix}.csv"


CONFIGS: Dict[str, Tuple[int, str, str]] = {
    f"cfg{index}_tq{bits}_{mode}": (bits, mode, _result_filename(bits, mode))
    for index, (bits, mode) in enumerate(
        tuple((bits, mode) for bits in (2, 3, 4) for mode in MODES), start=1
    )
}


@dataclass(frozen=True)
class MlaNiahResult:
    context_length: int
    compressed_slots: int
    bit_width: int
    mode: str
    heads: int
    latent_dim: int
    rope_dim: int
    rope_quantized: bool
    mean_needle_recall_at_128: float
    min_needle_recall: float
    needles_found: int
    total_needles: int
    distractor_false_positive_rate: float
    mean_needle_absolute_rank: float
    max_needle_absolute_rank: int
    mean_spearman_rho: float
    # Quantized/exact score ratio at the needle slot; see run_context.
    mean_dilution: float
    mean_rope_logit_share: float
    counterfactual_rope_phase_error_deg: float
    max_counterfactual_rope_phase_error_deg: float
    stream_latency_s: float
    peak_rss_mib: float
    packed_latent_stream_mib: float
    rope_stream_mib: float
    seed: int


# ---------------------------------------------------------------------------
# RoPE, half mode, on the compressed index
# ---------------------------------------------------------------------------


def _rope_tables(
    positions: "np.ndarray", rope_dim: int = ROPE_DIM, theta: float = COMPRESS_ROPE_THETA
) -> Tuple["np.ndarray", "np.ndarray"]:
    """cos/sin tables for `positions`, mirroring emu `BuildRopeTables`.

    The engine duplicates each of the `rope_dim / 2` frequencies across both
    halves of the table, so a half-mode rotation can index `[i]` and
    `[half + i]` with the same value. The duplication is reproduced rather
    than optimized away, to keep the indexing identical to the reference.
    """
    if rope_dim % 2:
        raise ValueError("rope_dim must be even")
    half = rope_dim // 2
    exponents = 2.0 * np.arange(half, dtype=np.float64) / float(rope_dim)
    inverse_frequency = 1.0 / np.power(theta, exponents)
    angles = positions.astype(np.float64)[:, None] * inverse_frequency[None, :]
    cosines = np.cos(angles).astype(np.float32)
    sines = np.sin(angles).astype(np.float32)
    return (
        np.concatenate((cosines, cosines), axis=1),
        np.concatenate((sines, sines), axis=1),
    )


def _apply_rope_half(
    vectors: "np.ndarray", cosines: "np.ndarray", sines: "np.ndarray"
) -> "np.ndarray":
    """Half-mode rotation, per src/emu/paged_compressor_decode.cpp:189.

        out[i]        = cos[i] * lo        - sin[i] * hi
        out[half + i] = sin[half + i] * lo + cos[half + i] * hi
    """
    if vectors.shape[1] % 2:
        raise ValueError("rope width must be even")
    half = vectors.shape[1] // 2
    low = vectors[:, :half]
    high = vectors[:, half:]
    rotated = np.empty_like(vectors)
    rotated[:, :half] = cosines[:, :half] * low - sines[:, :half] * high
    rotated[:, half:] = sines[:, half:] * low + cosines[:, half:] * high
    return rotated


def _rms_norm_rows(vectors: "np.ndarray", eps: float = RMS_NORM_EPSILON) -> "np.ndarray":
    """Row-wise ``x / sqrt(mean(x^2) + eps)``; the no-weight `rms_norm_cpu`."""
    mean_square = np.mean(
        vectors.astype(np.float32) * vectors.astype(np.float32),
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )
    return vectors / np.sqrt(mean_square + np.float32(eps))


# ---------------------------------------------------------------------------
# Needles, distractors, placement
# ---------------------------------------------------------------------------


def _unit_rows(vectors: "np.ndarray") -> "np.ndarray":
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, keepdims=True, dtype=np.float32))
    return vectors / np.maximum(norms, np.float32(1e-12))


def _orthogonal_unit(
    reference: "np.ndarray", rng: "np.random.Generator"
) -> "np.ndarray":
    for _ in range(8):
        vector = rng.standard_normal(reference.shape[0], dtype=np.float32)
        vector = vector - float(np.dot(vector, reference)) * reference
        norm = float(np.linalg.norm(vector))
        if norm > 1e-7:
            return (vector / np.float32(norm)).astype(np.float32, copy=False)
    raise RuntimeError("failed to construct a non-degenerate orthogonal vector")


@dataclass(frozen=True)
class Adversaries:
    needle_latent: "np.ndarray"       # [16, 512] unit
    needle_rope: "np.ndarray"         # [16, 64] unit, pre-rotation carrier
    distractor_latent: "np.ndarray"   # [16, 512] unit, cos 0.70..0.80 to needle
    distractor_rope: "np.ndarray"     # [16, 64] unit
    retrieval_heads: "np.ndarray"     # [16] which head specializes per needle


def _make_adversaries(seed: int) -> Adversaries:
    seed &= (1 << 64) - 1
    needle_rng = np.random.default_rng(seed ^ 0x4E4545444C455F34)
    distractor_rng = np.random.default_rng(seed ^ 0x4449535452414354)

    needle_latent = _unit_rows(
        needle_rng.standard_normal((NEEDLE_COUNT, LATENT_DIM), dtype=np.float32)
    )
    needle_rope = _unit_rows(
        needle_rng.standard_normal((NEEDLE_COUNT, ROPE_DIM), dtype=np.float32)
    )

    # Hard distractors: a controlled cosine to the needle they shadow, so the
    # separation the benchmark measures is the quantizer's, not the geometry's.
    distractor_latent = np.empty_like(needle_latent)
    targets = distractor_rng.uniform(
        DISTRACTOR_COSINE_LOW, DISTRACTOR_COSINE_HIGH, NEEDLE_COUNT
    )
    for index in range(NEEDLE_COUNT):
        orthogonal = _orthogonal_unit(needle_latent[index], distractor_rng)
        cosine = np.float32(targets[index])
        distractor_latent[index] = (
            cosine * needle_latent[index]
            + np.sqrt(np.float32(1.0) - cosine * cosine) * orthogonal
        )
    distractor_rope = _unit_rows(
        distractor_rng.standard_normal((NEEDLE_COUNT, ROPE_DIM), dtype=np.float32)
    )

    # A different specialized head per needle, so head-max is exercised across
    # all 64 rather than always resolving through head 0.
    retrieval_heads = np.arange(NEEDLE_COUNT, dtype=np.int64) * (
        HEADS // NEEDLE_COUNT
    )
    return Adversaries(
        needle_latent=needle_latent,
        needle_rope=needle_rope,
        distractor_latent=distractor_latent,
        distractor_rope=distractor_rope,
        retrieval_heads=retrieval_heads,
    )


def _place_adversaries(
    context_length: int, seed: int
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Needle token positions across 5%..95% depth, plus disjoint distractors."""
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

    rng = np.random.default_rng(
        (seed ^ context_length ^ 0x44535452504C4143) & ((1 << 64) - 1)
    )
    distractor_slots = rng.choice(available, size=DISTRACTOR_COUNT, replace=False)
    offsets = rng.integers(0, COMPRESSION_RATIO, size=DISTRACTOR_COUNT)
    distractor_positions = distractor_slots * COMPRESSION_RATIO + offsets
    return needle_positions, np.sort(distractor_positions.astype(np.int64))


# ---------------------------------------------------------------------------
# Queries: per-head absorption
# ---------------------------------------------------------------------------


def _build_queries(
    seed: int, adversaries: Adversaries, query_position: int
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Absorbed and rotary queries, flattened to [(needle, head), ...] rows.

    For head `h`, `W_UK[h]` is [128, 512] and the query carries a 128-lane
    nope part. The up-projection is absorbed into the query --
    `absorbed[h] = W_UK[h]^T q_nope[h]` -- which is what lets a 512-lane
    latent be scored without reconstructing a per-slot key, and is how the
    deployed MLA path does it.

    `W_UK` is built one head at a time from a per-head seed: materializing all
    64 at once would cost 16 MiB against the 150 MiB budget for no benefit.

    W_UK IS LEARNED, AND MODELLING IT AS PURELY RANDOM IS WRONG. A random
    [128, 512] projection retains only sqrt(128/512) ~ 0.48 of any given
    latent direction, so it discards three quarters of the semantic energy
    while the 64-lane rotary term loses none -- which inflates the rotary
    share of the logit to ~0.5 against the 64/192 ~ 0.33 the geometry implies,
    and buries the deepest needles for a reason that has nothing to do with
    quantization. A trained up-projection reads the directions that carry
    meaning. So each head's row space is seeded with an orthonormal basis of
    the needle directions and filled out with random rows: every head CAN read
    them, and head specialization comes from the query, which is where it
    comes from in the model.
    """
    noise_scale = np.float32(10.0 ** (-QUERY_SNR_DB / 20.0))
    absorbed = np.zeros((NEEDLE_COUNT, HEADS, LATENT_DIM), dtype=np.float32)
    rope_queries = np.zeros((NEEDLE_COUNT, HEADS, ROPE_DIM), dtype=np.float32)

    # The query sits at the end of the compressed stream, so the rotary term
    # is a genuine relative-position function of (query_slot - slot): that is
    # the long-range cue a quantized rope would corrupt.
    query_cos, query_sin = _rope_tables(np.asarray([query_position], dtype=np.int64))

    needle_latent_normed = _rms_norm_rows(adversaries.needle_latent)

    # The semantic rows every head shares: an orthonormal basis of the needle
    # directions. QR of the [512, 16] stack, taken back as [16, 512].
    semantic_basis, _ = np.linalg.qr(needle_latent_normed.T.astype(np.float64))
    semantic_basis = semantic_basis.T.astype(np.float32)

    for head in range(HEADS):
        head_rng = np.random.default_rng(
            (seed ^ 0x57554B5F48454144 ^ (head * 0x9E3779B1)) & ((1 << 64) - 1)
        )
        projection = head_rng.standard_normal(
            (NOPE_HEAD_DIM, LATENT_DIM), dtype=np.float32
        ) / np.float32(math.sqrt(LATENT_DIM))
        projection[: semantic_basis.shape[0]] = semantic_basis
        for needle in range(NEEDLE_COUNT):
            if int(adversaries.retrieval_heads[needle]) == head:
                # The specialized head's nope query points at where this
                # needle lands in that head's 128-lane subspace, plus noise at
                # the stated SNR.
                target = projection @ needle_latent_normed[needle]
                target = target / np.float32(
                    max(float(np.linalg.norm(target)), 1e-12)
                )
                noise = _orthogonal_unit(target, head_rng)
                q_nope = target + noise_scale * noise
                q_rope_carrier = adversaries.needle_rope[needle]
            else:
                q_nope = head_rng.standard_normal(NOPE_HEAD_DIM, dtype=np.float32)
                q_rope_carrier = head_rng.standard_normal(ROPE_DIM, dtype=np.float32)

            # The query head vector is RMSNorm'd as one 192-lane row and only
            # then rotated, mirroring `q[h] = rope_half(rms_norm(q[h]))` in
            # dsv4_layers.h:231-232. This is what keeps the query's two halves
            # on the same scale as the cached row's.
            head_vector = _rms_norm_rows(
                np.concatenate((q_nope, q_rope_carrier))[None, :]
            )[0]
            absorbed[needle, head] = projection.T @ head_vector[:NOPE_HEAD_DIM]
            rope_queries[needle, head] = _apply_rope_half(
                np.ascontiguousarray(head_vector[NOPE_HEAD_DIM:])[None, :],
                query_cos,
                query_sin,
            )[0]
        del projection
    return (
        absorbed.reshape(NEEDLE_COUNT * HEADS, LATENT_DIM),
        rope_queries.reshape(NEEDLE_COUNT * HEADS, ROPE_DIM),
    )


# ---------------------------------------------------------------------------
# Slot construction: pool -> RMSNorm -> RoPE
# ---------------------------------------------------------------------------


def _slot_rows(
    slots: "np.ndarray",
    stream_seed: int,
    special_positions: "np.ndarray",
    special_latent: "np.ndarray",
    special_rope: "np.ndarray",
    rope_weight: float,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Build (latent, rope) for arbitrary compressed `slots`.

    Token content is position-keyed, so this is random access: the needle
    pre-pass and the streaming pass regenerate identical slots without
    sharing an RNG stream, and no slot has to be held between them.
    """
    token_positions = (
        slots[:, None] * COMPRESSION_RATIO
        + np.arange(COMPRESSION_RATIO, dtype=np.int64)[None, :]
    ).ravel()

    latent = _haystack_tokens(token_positions, LATENT_DIM, stream_seed ^ 0x4C41544E)
    rope = _haystack_tokens(token_positions, ROPE_DIM, stream_seed ^ 0x524F5045)

    if special_positions.size:
        located = np.searchsorted(special_positions, token_positions)
        located = np.minimum(located, special_positions.size - 1)
        hit = special_positions[located] == token_positions
        if np.any(hit):
            rows = np.flatnonzero(hit)
            which = located[rows]
            latent[rows] = special_latent[which]
            rope[rows] = special_rope[which]

    # 4:1 pooling, then the C4A epilog's order: RMSNorm the WHOLE row, then
    # RoPE the rotary lanes at the compressed index.
    #
    # Normalizing the row as one vector is not a detail. `rms_norm_cpu` reduces
    # over the whole row, so it is what fixes the relative scale of the two
    # paths: afterwards every lane has RMS ~1, the 64 rotary lanes carry norm
    # ~sqrt(64) against the semantic lanes' ~sqrt(512), and the rotary term is
    # worth its natural 64/192 share of the logit. Normalizing the halves
    # separately instead leaves the rotary term at ~1% and would make the
    # whole decoupled-RoPE question unmeasurable.
    pooled_latent = np.mean(
        latent.reshape(-1, COMPRESSION_RATIO, LATENT_DIM), axis=1, dtype=np.float32
    )
    pooled_rope = np.mean(
        rope.reshape(-1, COMPRESSION_RATIO, ROPE_DIM), axis=1, dtype=np.float32
    )
    del latent, rope

    row = _rms_norm_rows(np.concatenate((pooled_latent, pooled_rope), axis=1))
    del pooled_latent, pooled_rope
    cosines, sines = _rope_tables(slots)
    rotated = _apply_rope_half(
        np.ascontiguousarray(row[:, LATENT_DIM:]), cosines, sines
    )
    if rope_weight != 1.0:
        rotated = rotated * np.float32(rope_weight)
    return np.ascontiguousarray(row[:, :LATENT_DIM]), rotated


def _aggregate_scores(
    latent: "np.ndarray",
    rope: "np.ndarray",
    absorbed: "np.ndarray",
    rope_queries: "np.ndarray",
    inverse_temperature: "np.floating",
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Head-max aggregated scores [slots, 16] and the rope term's share.

    Both terms are formed for all (needle, head) pairs in one GEMM each, then
    reduced with a max over the head axis.
    """
    semantic = latent @ absorbed.T
    rotary = rope @ rope_queries.T
    per_head = (semantic + rotary) * inverse_temperature
    per_head = per_head.reshape(latent.shape[0], NEEDLE_COUNT, HEADS)
    chosen = np.argmax(per_head, axis=2)
    scores = np.take_along_axis(per_head, chosen[:, :, None], axis=2)[:, :, 0]

    rotary = rotary.reshape(latent.shape[0], NEEDLE_COUNT, HEADS)
    semantic = semantic.reshape(latent.shape[0], NEEDLE_COUNT, HEADS)
    chosen_rotary = np.abs(np.take_along_axis(rotary, chosen[:, :, None], axis=2))
    chosen_semantic = np.abs(
        np.take_along_axis(semantic, chosen[:, :, None], axis=2)
    )
    share = chosen_rotary / np.maximum(chosen_rotary + chosen_semantic, np.float32(1e-20))
    return scores, share[:, :, 0]


def _counterfactual_rope_phase_error_deg(
    rope: "np.ndarray", quantizer: TurboLatticeQuantizer
) -> "np.ndarray":
    """Angle between true and would-be-quantized rotary vectors, in degrees.

    This is the control, not part of the pipeline: it quantizes the rotary
    lanes with the same codec the latent uses and throws the result away, so
    the report can state what the decoupling buys instead of printing a column
    of zeros and asserting it meant something. 64 lanes is divisible by the
    32-lane group and by 8, so E8 applies directly; E24 pads to 72 internally.
    """
    packed = _encode_batch(rope, quantizer, dimension=ROPE_DIM)
    decoded = _decode_batch(packed, quantizer)
    dot = np.sum(rope * decoded, axis=1, dtype=np.float64)
    norms = np.linalg.norm(rope, axis=1).astype(np.float64) * np.linalg.norm(
        decoded, axis=1
    ).astype(np.float64)
    cosine = np.where(norms > 0.0, dot / np.maximum(norms, 1e-300), 0.0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_context(
    context_length: int,
    bits: int,
    mode: str,
    *,
    seed: int = DEFAULT_SEED,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    slot_batch: int = DEFAULT_SLOT_BATCH,
    sample_slots: int = DEFAULT_SAMPLE_SLOTS,
    rope_weight: float = 1.0,
    max_peak_rss_mib: float = MAX_PEAK_RSS_MIB,
) -> MlaNiahResult:
    if context_length <= 0 or context_length % COMPRESSION_RATIO:
        raise ValueError("context length must be a positive multiple of four")
    if chunk_tokens <= 0 or chunk_tokens % COMPRESSION_RATIO:
        raise ValueError("chunk size must be a positive multiple of four")
    if slot_batch <= 0:
        raise ValueError("slot_batch must be positive")
    if sample_slots <= 0:
        raise ValueError("sample_slots must be positive")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    compressed_slots = context_length // COMPRESSION_RATIO
    if compressed_slots <= TOP_K:
        raise ValueError("context must contain more than 128 compressed slots")

    lattice_mode = mode.removeprefix("lloydmax").lstrip("_") or "none"
    quantizer = TurboLatticeQuantizer.for_standard_normal(
        bits, lattice_mode, group_size=GROUP_SIZE
    )
    if LATENT_DIM % quantizer.group_size:
        raise ValueError("the 512-lane latent must divide the quantizer group size")

    started = time.perf_counter()
    stream_seed = seed & ((1 << 64) - 1)
    adversaries = _make_adversaries(seed)
    needle_positions, distractor_positions = _place_adversaries(context_length, seed)
    needle_slots = needle_positions // COMPRESSION_RATIO
    distractor_slots = distractor_positions // COMPRESSION_RATIO

    absorbed, rope_queries = _build_queries(
        seed, adversaries, query_position=compressed_slots - 1
    )
    inverse_temperature = np.float32(1.0 / math.sqrt(NOPE_HEAD_DIM + ROPE_DIM))

    # The insertion table, sorted by token position for the searchsorted probe.
    special_positions = np.concatenate((needle_positions, distractor_positions))
    special_latent = np.concatenate(
        (adversaries.needle_latent, adversaries.distractor_latent), axis=0
    )
    special_rope = np.concatenate(
        (adversaries.needle_rope, adversaries.distractor_rope), axis=0
    )
    order = np.argsort(special_positions, kind="stable")
    special_positions = special_positions[order]
    special_latent = special_latent[order]
    special_rope = special_rope[order]

    def build(slots: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
        return _slot_rows(
            slots,
            stream_seed,
            special_positions,
            special_latent,
            special_rope,
            rope_weight,
        )

    def quantize(latent: "np.ndarray") -> Tuple["np.ndarray", int]:
        packed = _encode_batch(latent, quantizer, dimension=LATENT_DIM)
        return _decode_batch(packed, quantizer), packed.nbytes

    # --- Pass 1: the needle and distractor slots' own scores ----------------
    # Only 32 slots, so this is cheap, and it is what the rank count in pass 2
    # compares against. Position-keyed generation makes it exact.
    needle_latent_slots, needle_rope_slots = build(needle_slots)
    needle_decoded, _ = quantize(needle_latent_slots)
    needle_scores_all, needle_share = _aggregate_scores(
        needle_decoded, needle_rope_slots, absorbed, rope_queries, inverse_temperature
    )
    exact_needle_scores_all, _ = _aggregate_scores(
        needle_latent_slots, needle_rope_slots, absorbed, rope_queries, inverse_temperature
    )
    # Needle n is probed by query n: take the diagonal.
    diagonal = np.arange(NEEDLE_COUNT)
    needle_scores = needle_scores_all[diagonal, diagonal].astype(np.float32)
    exact_needle_scores = exact_needle_scores_all[diagonal, diagonal].astype(np.float32)
    rope_logit_share = float(np.mean(needle_share[diagonal, diagonal]))

    # Dilution here is the QUANTIZED/EXACT score ratio at the needle's own
    # slot, both after 4:1 pooling. It isolates what the latent codec costs the
    # needle's logit; the pooling loss is common to both terms and cancels. A
    # value near 1.0 means the codec preserved the needle's score, below 1.0
    # that it eroded it.
    dilutions = []
    for index in range(NEEDLE_COUNT):
        exact = float(exact_needle_scores[index])
        dilutions.append(float(needle_scores[index]) / exact if exact else 0.0)

    counterfactual = _counterfactual_rope_phase_error_deg(needle_rope_slots, quantizer)
    del needle_latent_slots, needle_decoded, needle_scores_all, exact_needle_scores_all

    distractor_latent_slots, distractor_rope_slots = build(distractor_slots)
    distractor_decoded, _ = quantize(distractor_latent_slots)
    distractor_scores_all, _ = _aggregate_scores(
        distractor_decoded,
        distractor_rope_slots,
        absorbed,
        rope_queries,
        inverse_temperature,
    )
    distractor_scores = distractor_scores_all[diagonal, diagonal].astype(np.float32)
    del distractor_latent_slots, distractor_decoded, distractor_scores_all
    del distractor_rope_slots, needle_rope_slots

    # --- Spearman sample: background slots, drawn once ---------------------
    background = np.ones(compressed_slots, dtype=np.bool_)
    background[needle_slots] = False
    background[distractor_slots] = False
    candidates = np.flatnonzero(background)
    sample_count = min(sample_slots, candidates.size)
    sample_rng = np.random.default_rng(
        (seed ^ context_length ^ 0x53504541524D414E) & ((1 << 64) - 1)
    )
    sampled_slots = np.sort(
        sample_rng.choice(candidates, size=sample_count, replace=False)
    )
    del background, candidates
    sampled_exact = np.empty((sample_count, NEEDLE_COUNT), dtype=np.float32)
    sampled_quantized = np.empty_like(sampled_exact)

    # --- Pass 2: stream every slot once ------------------------------------
    # Both the needle and its paired distractor are scored by RANK -- the count
    # of slots that strictly beat them -- rather than by testing membership of
    # a top-K heap. A heap test has to compare the pass-1 score against the
    # pass-2 one for exact float equality, and those are computed in batches of
    # different shapes (32 rows versus `slot_batch`), which BLAS is free to
    # evaluate with different kernels. Rank counting is invariant to that.
    needle_ranks = np.zeros(NEEDLE_COUNT, dtype=np.int64)
    distractor_ranks = np.zeros(DISTRACTOR_COUNT, dtype=np.int64)
    packed_latent_bytes = 0
    slots_per_chunk = chunk_tokens // COMPRESSION_RATIO

    for chunk_start in range(0, compressed_slots, slots_per_chunk):
        chunk_stop = min(compressed_slots, chunk_start + slots_per_chunk)
        for batch_start in range(chunk_start, chunk_stop, slot_batch):
            batch_stop = min(chunk_stop, batch_start + slot_batch)
            slots = np.arange(batch_start, batch_stop, dtype=np.int64)
            latent, rope = build(slots)
            decoded, packed_bytes = quantize(latent)
            packed_latent_bytes += packed_bytes

            exact_scores, _ = _aggregate_scores(
                latent, rope, absorbed, rope_queries, inverse_temperature
            )
            quantized_scores, _ = _aggregate_scores(
                decoded, rope, absorbed, rope_queries, inverse_temperature
            )
            del latent, rope, decoded

            for needle in range(NEEDLE_COUNT):
                column = quantized_scores[:, needle]
                for ranks, scores_of, slot_of in (
                    (needle_ranks, needle_scores, needle_slots),
                    (distractor_ranks, distractor_scores, distractor_slots),
                ):
                    target = scores_of[needle]
                    target_slot = int(slot_of[needle])
                    ranks[needle] += int(
                        np.count_nonzero(
                            (column > target)
                            | ((column == target) & (slots < target_slot))
                        )
                    )

            begin = int(np.searchsorted(sampled_slots, batch_start, side="left"))
            end = int(np.searchsorted(sampled_slots, batch_stop, side="left"))
            if begin < end:
                local = sampled_slots[begin:end] - batch_start
                sampled_exact[begin:end] = exact_scores[local]
                sampled_quantized[begin:end] = quantized_scores[local]
            del exact_scores, quantized_scores

    # --- Metrics ----------------------------------------------------------
    recalls: List[int] = []
    false_positives = 0
    rhos: List[float] = []
    for needle in range(NEEDLE_COUNT):
        # `rank` is the number of slots that strictly beat this one, so the
        # needle is inside the top-K window exactly when rank < TOP_K. Using
        # `<=` would count the 129th-placed slot as recalled at K = 128.
        recalls.append(int(int(needle_ranks[needle]) < TOP_K))
        false_positives += int(int(distractor_ranks[needle]) < TOP_K)
        rhos.append(
            spearman_rank_correlation(
                sampled_exact[:, needle].tolist(),
                sampled_quantized[:, needle].tolist(),
            )
        )

    elapsed = time.perf_counter() - started
    _, peak_rss_mib = _memory_mib()
    if peak_rss_mib and peak_rss_mib >= max_peak_rss_mib:
        raise MemoryError(
            f"peak RSS {peak_rss_mib:.1f} MiB exceeds the "
            f"{max_peak_rss_mib:.0f} MiB limit"
        )

    return MlaNiahResult(
        context_length=context_length,
        compressed_slots=compressed_slots,
        bit_width=bits,
        mode=mode,
        heads=HEADS,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
        # Structural, not measured: the cache has no path that quantizes rope.
        rope_quantized=False,
        mean_needle_recall_at_128=float(sum(recalls) / NEEDLE_COUNT),
        min_needle_recall=float(min(recalls)),
        needles_found=sum(recalls),
        total_needles=NEEDLE_COUNT,
        distractor_false_positive_rate=float(false_positives / DISTRACTOR_COUNT),
        mean_needle_absolute_rank=float(needle_ranks.mean()),
        max_needle_absolute_rank=int(needle_ranks.max()),
        mean_spearman_rho=float(sum(rhos) / NEEDLE_COUNT),
        mean_dilution=float(sum(dilutions) / NEEDLE_COUNT),
        mean_rope_logit_share=rope_logit_share,
        counterfactual_rope_phase_error_deg=float(np.mean(counterfactual)),
        max_counterfactual_rope_phase_error_deg=float(np.max(counterfactual)),
        stream_latency_s=elapsed,
        peak_rss_mib=peak_rss_mib,
        packed_latent_stream_mib=packed_latent_bytes / (1024.0 * 1024.0),
        rope_stream_mib=compressed_slots * ROPE_DIM * 4 / (1024.0 * 1024.0),
        seed=seed,
    )


def write_results(path: Path, rows: Sequence[MlaNiahResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [field.name for field in fields(MlaNiahResult)]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: getattr(row, column) for column in columns} for row in rows
        )


def print_table(config: str, rows: Sequence[MlaNiahResult]) -> None:
    print(f"\n### {config}")
    print(
        "| Context | Slots | Recall@128 | Distractor FP | Mean rank | Max rank | "
        "Spearman | Dilution | RoPE share | CF phase err | Latency | Peak RSS |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.context_length:,} | {row.compressed_slots:,} | "
            f"{row.mean_needle_recall_at_128:.4f} | "
            f"{row.distractor_false_positive_rate:.4f} | "
            f"{row.mean_needle_absolute_rank:.1f} | "
            f"{row.max_needle_absolute_rank:,} | "
            f"{row.mean_spearman_rho:.6f} | {row.mean_dilution:.4f} | "
            f"{row.mean_rope_logit_share:.3f} | "
            f"{row.counterfactual_rope_phase_error_deg:.2f}deg | "
            f"{row.stream_latency_s:.3f}s | {row.peak_rss_mib:.1f} MiB |"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=tuple(CONFIGS))
    parser.add_argument(
        "--context-lengths", default=",".join(map(str, DEFAULT_CONTEXTS))
    )
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--slot-batch", type=int, default=DEFAULT_SLOT_BATCH)
    parser.add_argument("--sample-slots", type=int, default=DEFAULT_SAMPLE_SLOTS)
    parser.add_argument("--rope-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-peak-rss-mib", type=float, default=MAX_PEAK_RSS_MIB
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--print-geometry-only",
        action="store_true",
        help="print the audited geometry and exit",
    )
    args = parser.parse_args(argv)

    if args.print_geometry_only:
        print(
            f"latent d_c={LATENT_DIM} rope d_rope={ROPE_DIM} "
            f"nope d_h={NOPE_HEAD_DIM} heads H={HEADS} "
            f"ratio={COMPRESSION_RATIO} compress_rope_theta={COMPRESS_ROPE_THETA:.0f} "
            f"temperature=1/sqrt({NOPE_HEAD_DIM + ROPE_DIM}) rope_quantized=False"
        )
        return 0
    if args.config is None:
        parser.error("--config is required unless --print-geometry-only is used")

    bits, mode, filename = CONFIGS[args.config]
    contexts = tuple(
        int(value.strip())
        for value in args.context_lengths.split(",")
        if value.strip()
    )
    if not contexts:
        parser.error("at least one context length is required")

    print(
        f"{args.config}: {bits}-bit {mode}, seed={args.seed}, "
        f"chunk={args.chunk_tokens} tokens, slot_batch={args.slot_batch}"
    )
    rows = []
    for context in contexts:
        row = run_context(
            context,
            bits,
            mode,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            slot_batch=args.slot_batch,
            sample_slots=args.sample_slots,
            rope_weight=args.rope_weight,
            max_peak_rss_mib=args.max_peak_rss_mib,
        )
        rows.append(row)
        print(
            f"  {context:>9,} tokens -> recall@128 {row.mean_needle_recall_at_128:.4f}"
            f"  rho {row.mean_spearman_rho:.6f}  {row.stream_latency_s:.2f}s"
        )

    path = args.csv or Path(__file__).with_name(filename)
    write_results(path, rows)
    print_table(args.config, rows)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
