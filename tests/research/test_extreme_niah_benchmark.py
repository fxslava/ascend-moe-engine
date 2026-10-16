"""Small-scale invariants for the streaming extreme NIAH benchmark."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy is available in the target CANN container")
class ExtremeNiahStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from niah_attention_regression import build_quantizer

        cls.quantizer = build_quantizer(128, 32, 99, 32, 4.0)

    def test_chunk_size_does_not_change_metrics(self) -> None:
        from extreme_niah_benchmark import run_context_streaming

        first = run_context_streaming(
            1024,
            (0.25, 0.75),
            self.quantizer,
            seed=99,
            chunk_tokens=256,
            sample_slots=128,
        )
        second = run_context_streaming(
            1024,
            (0.25, 0.75),
            self.quantizer,
            seed=99,
            chunk_tokens=512,
            sample_slots=128,
        )
        for left, right in zip(first, second):
            self.assertEqual(left.needle_absolute_rank, right.needle_absolute_rank)
            self.assertEqual(left.recall_at_k, right.recall_at_k)
            self.assertAlmostEqual(left.spearman_rho, right.spearman_rho, places=7)
            self.assertAlmostEqual(left.dilution, right.dilution, places=7)

    def test_one_million_configuration_has_bounded_resident_chunk(self) -> None:
        from extreme_niah_benchmark import HEAD_DIM

        chunk_tokens = 8192
        dense_chunk_bytes = chunk_tokens * HEAD_DIM * 4
        full_context_bytes = 1048576 * HEAD_DIM * 4
        self.assertLess(dense_chunk_bytes, 5 * 1024 * 1024)
        self.assertGreater(full_context_bytes, 500 * 1024 * 1024)

    def test_vectorized_packing_matches_scalar_reference(self) -> None:
        from extreme_niah_benchmark import _quantize_decode_batch

        rng = np.random.default_rng(17)
        vectors = rng.normal(size=(4, 128)).astype(np.float32)
        decoded_batch, packed_bytes = _quantize_decode_batch(vectors, self.quantizer)
        self.assertGreater(packed_bytes, 0)
        for row, vector in enumerate(vectors):
            packed = self.quantizer.encode_vector(vector)
            decoded_scalar = self.quantizer.decode_vector(packed)
            np.testing.assert_allclose(decoded_batch[row], decoded_scalar, rtol=0.0, atol=1e-6)

    def test_all_ablation_batch_formats_match_scalar_reference(self) -> None:
        from extreme_niah_benchmark import _quantize_decode_batch
        from turbo_lattice_vq import TurboLatticeQuantizer

        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(3, 128)).astype(np.float32)
        for bits in (2, 3, 4):
            for lattice_mode in ("none", "e8", "e24"):
                quantizer = TurboLatticeQuantizer.for_standard_normal(
                    bits, lattice_mode
                )
                decoded_batch, _ = _quantize_decode_batch(vectors, quantizer)
                for row, vector in enumerate(vectors):
                    packed = quantizer.encode_vector(vector)
                    decoded_scalar = quantizer.decode_vector(packed)
                    np.testing.assert_allclose(
                        decoded_batch[row], decoded_scalar, rtol=0.0, atol=1e-6
                    )

    def test_two_bit_preflight_and_adversarial_geometry(self) -> None:
        from adversarial_multi_needle_benchmark import (
            NEEDLE_COUNT,
            QUERY_SNR_DB,
            _make_needles_queries_and_distractors,
            _make_transform,
            _place_adversaries,
            _cosine,
            verify_2bit_packing,
        )

        verify_2bit_packing()
        transform = _make_transform(20251012)
        needles, queries, distractors = _make_needles_queries_and_distractors(
            20251012, transform
        )
        expected_cosine = 1.0 / math.sqrt(1.0 + 10.0 ** (-QUERY_SNR_DB / 10.0))
        self.assertEqual(needles.shape, (NEEDLE_COUNT, 128))
        for needle, query, distractor in zip(needles, queries, distractors):
            self.assertAlmostEqual(_cosine(needle, query), expected_cosine, places=6)
            self.assertGreaterEqual(_cosine(query, distractor), 0.70)
            self.assertLessEqual(_cosine(query, distractor), 0.82)
        needle_positions, distractor_positions = _place_adversaries(8192, 20251012)
        needle_slots = needle_positions // 4
        distractor_slots = distractor_positions // 4
        self.assertEqual(len(set(needle_slots)), NEEDLE_COUNT)
        self.assertEqual(len(set(distractor_slots)), NEEDLE_COUNT)
        self.assertTrue(
            all(min(abs(slot - needle_slots)) > 1 for slot in distractor_slots)
        )

    def test_adversarial_streaming_metrics_are_chunk_invariant(self) -> None:
        from adversarial_multi_needle_benchmark import run_context

        first = run_context(1024, 2, "lloydmax", seed=91, chunk_tokens=256, sample_slots=128)
        second = run_context(1024, 2, "lloydmax", seed=91, chunk_tokens=512, sample_slots=128)
        self.assertEqual(first.needles_found, second.needles_found)
        self.assertEqual(first.mean_needle_rank, second.mean_needle_rank)
        self.assertEqual(first.distractor_false_positive_rate, second.distractor_false_positive_rate)
        self.assertAlmostEqual(first.mean_spearman_rho, second.mean_spearman_rho, places=6)


if __name__ == "__main__":
    unittest.main()
