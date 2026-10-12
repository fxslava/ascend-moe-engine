"""Focused CPU tests for the CSA quantized-cache research path."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from niah_attention_regression import build_quantizer, run_probe  # noqa: E402
from turbo_lattice_vq import (  # noqa: E402
    CSACompressor,
    E4M3FN,
    LloydMaxQuantizer,
    PagedQuantizedCache,
    TurboLatticeQuantizer,
    decode_e8_doubled,
    is_leech_integer_point,
    nearest_e8_doubled,
    nearest_leech24_integer,
)


class TurboLatticeCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.quantizer = build_quantizer(
            dimension=128,
            group_size=32,
            seed=1234,
            calibration_vectors=64,
            lattice_step=4.0,
        )

    def test_e4m3fn_known_boundaries_and_saturation(self) -> None:
        self.assertEqual(E4M3FN.decode(E4M3FN.encode(1.0)), 1.0)
        self.assertEqual(E4M3FN.decode(E4M3FN.encode(448.0)), 448.0)
        self.assertEqual(E4M3FN.decode(E4M3FN.encode(1000.0)), 448.0)
        self.assertEqual(E4M3FN.decode(E4M3FN.encode(-1.0)), -1.0)

    def test_e8_codeword_membership_and_nearby_reconstruction(self) -> None:
        source = (0.49,) * 8
        codeword = nearest_e8_doubled(source)
        reconstructed = decode_e8_doubled(codeword)
        self.assertTrue(all(abs(value - 0.5) < 1e-12 for value in reconstructed))
        self.assertTrue(all((coordinate & 1) == (codeword[0] & 1) for coordinate in codeword))

    def test_standard_normal_lloyd_max_codebooks(self) -> None:
        two_bit = LloydMaxQuantizer.standard_normal(4)
        self.assertEqual(
            two_bit.centres, (-1.510418, -0.452794, 0.452794, 1.510418)
        )
        self.assertEqual(two_bit.thresholds, (-0.981606, 0.0, 0.981606))
        for levels in (8, 16):
            quantizer = LloydMaxQuantizer.standard_normal(levels)
            self.assertEqual(len(quantizer.centres), levels)
            self.assertEqual(len(quantizer.thresholds), levels - 1)
            for left, right in zip(quantizer.centres, reversed(quantizer.centres)):
                self.assertAlmostEqual(left, -right, places=12)
            self.assertTrue(
                all(a < b for a, b in zip(quantizer.centres, quantizer.centres[1:]))
            )

    def test_leech_decoder_returns_valid_integer_embedding(self) -> None:
        source = tuple(math.sin(index * 0.71) for index in range(24))
        codeword = nearest_leech24_integer(source)
        self.assertTrue(is_leech_integer_point(codeword))

    def test_ablation_payload_widths_and_e24_padding(self) -> None:
        vector = tuple(math.sin(index * 0.17) for index in range(128))
        expected_index_bytes = {2: 32, 3: 48, 4: 64}
        expected_lattice_bytes = {"none": 0, "e8": 256, "e24": 288}
        for bits in (2, 3, 4):
            for lattice_mode in ("none", "e8", "e24"):
                quantizer = TurboLatticeQuantizer.for_standard_normal(
                    bits, lattice_mode
                )
                packed = quantizer.encode_vector(vector)
                self.assertEqual(
                    len(packed.lloyd_indices_packed), expected_index_bytes[bits]
                )
                self.assertEqual(
                    len(packed.lattice_codewords_i16),
                    expected_lattice_bytes[lattice_mode],
                )
                self.assertEqual(
                    packed.padded_dimension, 144 if lattice_mode == "e24" else 128
                )
                self.assertEqual(len(quantizer.decode_vector(packed)), 128)

    def test_quantized_cache_contains_packed_payload_and_decodes_on_read(self) -> None:
        block_table = (1, 0)
        cache = PagedQuantizedCache(2, 16, 128, self.quantizer, block_table)
        key = tuple(math.sin(index * 0.17) for index in range(128))
        value = tuple(math.cos(index * 0.11) for index in range(128))
        cache.write(0, key, value)
        slot = cache.packed_slot(0)
        self.assertIsInstance(slot.key.lloyd_indices_u4, bytes)
        self.assertIsInstance(slot.key.e8_codewords_i16, bytes)
        self.assertLess(slot.key.nbytes, 128 * 4)
        decoded_key, decoded_value = cache.read(0)
        key_rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(key, decoded_key)) / 128)
        value_rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(value, decoded_value)) / 128)
        self.assertLess(key_rmse, 0.03)
        self.assertLess(value_rmse, 0.03)

    def test_compressor_writes_only_emitted_slots_and_attention_reads_them(self) -> None:
        tokens = [
            tuple(math.sin((token + 1) * (lane + 1) * 0.01) for lane in range(128))
            for token in range(8)
        ]
        cache = PagedQuantizedCache(1, 16, 128, self.quantizer)
        exact = CSACompressor(4).compress_into_cache(tokens, tokens, cache)
        self.assertEqual(len(exact), 2)
        with self.assertRaises(KeyError):
            cache.packed_slot(2)
        result = cache.attention(tokens[0], seq_len=2, top_k=1)
        self.assertEqual(len(result.scores), 2)
        self.assertEqual(len(result.output), 128)

    def test_small_niah_probe_recalls_needle_deterministically(self) -> None:
        first = run_probe(256, 0.5, self.quantizer, seed=77, requested_top_k=64)
        second = run_probe(256, 0.5, self.quantizer, seed=77, requested_top_k=64)
        self.assertEqual(first, second)
        self.assertEqual(first.recall, 1)
        self.assertGreater(first.spearman_rho, 0.98)
        self.assertGreater(first.dilution, 0.35)


if __name__ == "__main__":
    unittest.main()
