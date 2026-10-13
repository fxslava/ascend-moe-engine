"""Host validation of the MLA benchmark's rotary and scoring index arithmetic.

The vectorized RoPE and the head-max reduction are the parts of this
benchmark that are wrong the first time, and neither needs hardware to check.
So each is compared against a scalar transcription of the reference it claims
to implement:

* `_rope_tables` against `BuildRopeTables` (src/emu/semi_e2e_pipeline.cpp:85)
* `_apply_rope_half` against the half-mode branch of
  `paged_compressor_decode.cpp:179-190`

Requires NumPy, like the benchmark itself.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
except ImportError as error:  # pragma: no cover
    raise SystemExit("test_dsv4_mla_niah_benchmark requires NumPy") from error

from dsv4_mla_niah_benchmark import (  # noqa: E402
    COMPRESSION_RATIO,
    COMPRESS_ROPE_THETA,
    HEADS,
    LATENT_DIM,
    NEEDLE_COUNT,
    NOPE_HEAD_DIM,
    ROPE_DIM,
    TOP_K,
    _aggregate_scores,
    _apply_rope_half,
    _counterfactual_rope_phase_error_deg,
    _rms_norm_rows,
    _rope_tables,
    run_context,
)
from extreme_niah_benchmark import _decode_batch, _encode_batch  # noqa: E402
from turbo_lattice_vq import RMS_NORM_EPSILON, TurboLatticeQuantizer  # noqa: E402


def _reference_rope_tables(position: int, rope_dim: int, theta: float):
    """Scalar transcription of emu `BuildRopeTables`."""
    cosines = [0.0] * rope_dim
    sines = [0.0] * rope_dim
    half = rope_dim // 2
    for index in range(half):
        inverse_frequency = 1.0 / math.pow(theta, (2.0 * index) / rope_dim)
        angle = position * inverse_frequency
        cosines[index] = math.cos(angle)
        sines[index] = math.sin(angle)
        cosines[half + index] = cosines[index]
        sines[half + index] = sines[index]
    return cosines, sines


def _reference_apply_half(vector, cosines, sines):
    """Scalar transcription of paged_compressor_decode.cpp:179-190."""
    rope_dim = len(vector)
    half = rope_dim // 2
    out = [0.0] * rope_dim
    for index in range(half):
        low = vector[index]
        high = vector[half + index]
        out[index] = cosines[index] * low - sines[index] * high
        out[half + index] = sines[half + index] * low + cosines[half + index] * high
    return out


class RopeTableTest(unittest.TestCase):
    def test_matches_the_reference_tables(self) -> None:
        positions = np.asarray([0, 1, 7, 4095, 262143], dtype=np.int64)
        cosines, sines = _rope_tables(positions)
        self.assertEqual(cosines.shape, (positions.size, ROPE_DIM))
        for row, position in enumerate(positions.tolist()):
            want_cos, want_sin = _reference_rope_tables(
                position, ROPE_DIM, COMPRESS_ROPE_THETA
            )
            for lane in range(ROPE_DIM):
                self.assertAlmostEqual(
                    float(cosines[row, lane]), want_cos[lane], places=5
                )
                self.assertAlmostEqual(
                    float(sines[row, lane]), want_sin[lane], places=5
                )

    def test_halves_are_duplicated(self) -> None:
        cosines, sines = _rope_tables(np.asarray([12345], dtype=np.int64))
        half = ROPE_DIM // 2
        np.testing.assert_array_equal(cosines[:, :half], cosines[:, half:])
        np.testing.assert_array_equal(sines[:, :half], sines[:, half:])

    def test_uses_the_compressor_theta_not_the_attention_theta(self) -> None:
        # 160000 is "compress_rope_theta"; 10000 is the attention path's and
        # would give visibly different low-frequency lanes.
        self.assertEqual(COMPRESS_ROPE_THETA, 160000.0)
        position = np.asarray([1000], dtype=np.int64)
        compressor, _ = _rope_tables(position, theta=COMPRESS_ROPE_THETA)
        attention, _ = _rope_tables(position, theta=10000.0)
        self.assertFalse(np.allclose(compressor, attention))

    def test_position_zero_is_the_identity(self) -> None:
        cosines, sines = _rope_tables(np.asarray([0], dtype=np.int64))
        vector = np.arange(1, ROPE_DIM + 1, dtype=np.float32)[None, :]
        np.testing.assert_allclose(
            _apply_rope_half(vector, cosines, sines), vector, atol=1e-6
        )


class RopeRotationTest(unittest.TestCase):
    def test_matches_the_reference_rotation(self) -> None:
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((5, ROPE_DIM), dtype=np.float32)
        positions = np.asarray([0, 3, 64, 1024, 99991], dtype=np.int64)
        cosines, sines = _rope_tables(positions)
        rotated = _apply_rope_half(vectors, cosines, sines)
        for row in range(vectors.shape[0]):
            want = _reference_apply_half(
                vectors[row].tolist(),
                cosines[row].tolist(),
                sines[row].tolist(),
            )
            for lane in range(ROPE_DIM):
                self.assertAlmostEqual(
                    float(rotated[row, lane]), want[lane], places=5
                )

    def test_rotation_preserves_the_norm(self) -> None:
        rng = np.random.default_rng(11)
        vectors = rng.standard_normal((4, ROPE_DIM), dtype=np.float32)
        cosines, sines = _rope_tables(np.asarray([1, 2, 3, 4], dtype=np.int64))
        rotated = _apply_rope_half(vectors, cosines, sines)
        np.testing.assert_allclose(
            np.linalg.norm(rotated, axis=1),
            np.linalg.norm(vectors, axis=1),
            rtol=1e-5,
        )

    def test_inner_product_depends_only_on_the_position_difference(self) -> None:
        # This is the property the decoupled rotary path exists to preserve: a
        # relative-position kernel. If it holds, phase coherence is intact; a
        # quantized rope is exactly what breaks it.
        rng = np.random.default_rng(13)
        carrier = rng.standard_normal((1, ROPE_DIM), dtype=np.float32)
        query = rng.standard_normal((1, ROPE_DIM), dtype=np.float32)

        def product(query_position: int, key_position: int) -> float:
            q_cos, q_sin = _rope_tables(
                np.asarray([query_position], dtype=np.int64)
            )
            k_cos, k_sin = _rope_tables(np.asarray([key_position], dtype=np.int64))
            rotated_query = _apply_rope_half(query, q_cos, q_sin)
            rotated_key = _apply_rope_half(carrier, k_cos, k_sin)
            return float(np.sum(rotated_query * rotated_key, dtype=np.float64))

        baseline = product(500, 200)
        for shift in (1, 17, 1000):
            self.assertAlmostEqual(
                product(500 + shift, 200 + shift), baseline, places=3
            )
        self.assertNotAlmostEqual(product(500, 400), baseline, places=3)


class RmsNormRowsTest(unittest.TestCase):
    def test_matches_the_scalar_formula(self) -> None:
        rng = np.random.default_rng(17)
        vectors = rng.standard_normal((3, 64), dtype=np.float32)
        normalized = _rms_norm_rows(vectors)
        for row in range(vectors.shape[0]):
            values = vectors[row].astype(np.float64)
            scale = 1.0 / math.sqrt(float(np.mean(values * values)) + RMS_NORM_EPSILON)
            for lane in range(vectors.shape[1]):
                self.assertAlmostEqual(
                    float(normalized[row, lane]),
                    float(values[lane]) * scale,
                    places=4,
                )

    def test_normalizing_the_whole_row_sets_the_two_paths_relative_scale(self) -> None:
        # Normalizing a 576-lane row leaves each lane at RMS ~1, so the 64
        # rotary lanes carry ~sqrt(64) against the 512 semantic lanes'
        # ~sqrt(512). Halving that ratio is what made the rotary term worth
        # ~1% of the logit in the first draft.
        rng = np.random.default_rng(19)
        row = rng.standard_normal((1, LATENT_DIM + ROPE_DIM), dtype=np.float32)
        normalized = _rms_norm_rows(row)
        semantic = float(np.linalg.norm(normalized[0, :LATENT_DIM]))
        rotary = float(np.linalg.norm(normalized[0, LATENT_DIM:]))
        self.assertAlmostEqual(
            rotary / semantic, math.sqrt(ROPE_DIM / LATENT_DIM), delta=0.05
        )


class AggregateScoresTest(unittest.TestCase):
    """The (needle, head) row ordering of the flattened query matrices."""

    def test_head_max_picks_the_right_needle_and_head(self) -> None:
        slots = 3
        absorbed = np.zeros((NEEDLE_COUNT * HEADS, LATENT_DIM), dtype=np.float32)
        rope_queries = np.zeros((NEEDLE_COUNT * HEADS, ROPE_DIM), dtype=np.float32)
        # Needle 2 is read by head 5 through latent lane 9.
        absorbed[2 * HEADS + 5, 9] = 1.0
        latent = np.zeros((slots, LATENT_DIM), dtype=np.float32)
        rope = np.zeros((slots, ROPE_DIM), dtype=np.float32)
        latent[1, 9] = 4.0

        inverse_temperature = np.float32(1.0 / math.sqrt(NOPE_HEAD_DIM + ROPE_DIM))
        scores, share = _aggregate_scores(
            latent, rope, absorbed, rope_queries, inverse_temperature
        )
        self.assertEqual(scores.shape, (slots, NEEDLE_COUNT))
        self.assertAlmostEqual(
            float(scores[1, 2]), 4.0 / math.sqrt(192), places=5
        )
        # No other (slot, needle) cell sees it.
        self.assertAlmostEqual(float(scores[0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(scores[1, 3]), 0.0, places=6)
        # A pure semantic hit carries no rotary share.
        self.assertAlmostEqual(float(share[1, 2]), 0.0, places=6)

    def test_rotary_only_hit_reports_a_full_rope_share(self) -> None:
        absorbed = np.zeros((NEEDLE_COUNT * HEADS, LATENT_DIM), dtype=np.float32)
        rope_queries = np.zeros((NEEDLE_COUNT * HEADS, ROPE_DIM), dtype=np.float32)
        rope_queries[0 * HEADS + 1, 3] = 1.0
        latent = np.zeros((1, LATENT_DIM), dtype=np.float32)
        rope = np.zeros((1, ROPE_DIM), dtype=np.float32)
        rope[0, 3] = 2.0
        _, share = _aggregate_scores(
            latent,
            rope,
            absorbed,
            rope_queries,
            np.float32(1.0 / math.sqrt(192)),
        )
        self.assertAlmostEqual(float(share[0, 0]), 1.0, places=6)


class LatentCodecTest(unittest.TestCase):
    def test_512_lane_round_trip_carries_its_own_width(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "e8", group_size=32)
        rng = np.random.default_rng(23)
        latent = rng.standard_normal((6, LATENT_DIM), dtype=np.float32)
        packed = _encode_batch(latent, quantizer, dimension=LATENT_DIM)
        self.assertEqual(packed.dimension, LATENT_DIM)
        decoded = _decode_batch(packed, quantizer)
        self.assertEqual(decoded.shape, (6, LATENT_DIM))
        # Lossy, but directionally faithful at four bits plus an E8 residual.
        for row in range(latent.shape[0]):
            cosine = float(
                np.dot(latent[row], decoded[row])
                / (np.linalg.norm(latent[row]) * np.linalg.norm(decoded[row]))
            )
            self.assertGreater(cosine, 0.99)

    def test_a_width_mismatch_is_refused(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "none", group_size=32)
        latent = np.zeros((2, LATENT_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            _encode_batch(latent, quantizer, dimension=ROPE_DIM)

    def test_counterfactual_phase_error_is_real_and_bit_ordered(self) -> None:
        rng = np.random.default_rng(29)
        rope = rng.standard_normal((8, ROPE_DIM), dtype=np.float32)
        errors = {}
        for bits in (2, 3, 4):
            quantizer = TurboLatticeQuantizer.for_standard_normal(
                bits, "none", group_size=32
            )
            errors[bits] = float(
                np.mean(_counterfactual_rope_phase_error_deg(rope, quantizer))
            )
        # More bits, less phase damage -- and all of it strictly nonzero, which
        # is what makes the shipped path's zero meaningful.
        self.assertGreater(errors[2], errors[3])
        self.assertGreater(errors[3], errors[4])
        self.assertGreater(errors[4], 0.0)


class EndToEndTest(unittest.TestCase):
    """One small context through the real driver."""

    def test_small_context_runs_and_reports_an_unquantized_rope(self) -> None:
        context = 8192
        result = run_context(
            context,
            4,
            "lloydmax",
            seed=4242,
            chunk_tokens=4096,
            slot_batch=256,
            sample_slots=256,
        )
        self.assertEqual(result.context_length, context)
        self.assertEqual(result.compressed_slots, context // COMPRESSION_RATIO)
        self.assertEqual(result.latent_dim, LATENT_DIM)
        self.assertEqual(result.rope_dim, ROPE_DIM)
        self.assertEqual(result.heads, HEADS)
        self.assertFalse(result.rope_quantized)
        self.assertEqual(result.total_needles, NEEDLE_COUNT)

        # The retrieval has to actually work, or the ablation measures nothing.
        self.assertGreaterEqual(result.mean_needle_recall_at_128, 0.75)
        self.assertGreater(result.mean_spearman_rho, 0.9)
        self.assertLess(result.mean_needle_absolute_rank, TOP_K)
        # The counterfactual control must be live.
        self.assertGreater(result.counterfactual_rope_phase_error_deg, 0.0)
        # So must the distractor metric. Distractors sit at cosine 0.70-0.80 to
        # their query, so a rate pinned at exactly 0.0 means the measurement is
        # broken, not that the codec is perfect -- which is how the heap-
        # membership version of this metric failed.
        self.assertGreater(result.distractor_false_positive_rate, 0.0)
        self.assertLessEqual(result.distractor_false_positive_rate, 1.0)
        # The rotary term must be a real but not dominant share of the logit.
        self.assertGreater(result.mean_rope_logit_share, 0.02)
        self.assertLess(result.mean_rope_logit_share, 0.6)
        self.assertGreater(result.packed_latent_stream_mib, 0.0)

    def test_a_short_context_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            run_context(256, 4, "lloydmax")
        with self.assertRaises(ValueError):
            run_context(8190, 4, "lloydmax")
        with self.assertRaises(ValueError):
            run_context(8192, 4, "turboquant")


if __name__ == "__main__":
    unittest.main()
