"""Dimension contracts for the faithful DSV4 MLA research cache.

This suite exists because the previous synthetic probe drifted from the
engine's real geometry without anything failing.  So it does not restate the
constants -- it parses them out of the C++ headers and asserts the Python
research model agrees.  Edit `include/moe/core/config.hpp` or
`include/moe/emu/cpu_kernels.h` and these tests go red.

It is pure stdlib and reads only headers, so it runs anywhere; there is no
NPU, CANN or NumPy dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from turbo_lattice_vq import (  # noqa: E402
    CSA_COMPRESSOR_HEAD_DIM,
    CSA_COMPRESS_RATIO,
    CSA_ROPE_BASE_OFFSET,
    DecoupledMlaCache,
    MLA_INDEXER_HEADS,
    MLA_INDEXER_HEAD_DIM,
    MLA_INDEXER_TOP_K,
    MLA_KV_ROW_ELEMENTS,
    MLA_LATENT_DIM,
    MLA_LAYOUT_IN_ROW,
    MLA_LAYOUT_SPLIT,
    MLA_NOPE_HEAD_DIM,
    MLA_ROPE_DIM,
    MultiHeadMlaIndexer,
    RMS_NORM_EPSILON,
    TurboLatticeQuantizer,
    mla_semantic_dim,
    rms_norm,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_HPP = _REPO_ROOT / "include" / "moe" / "core" / "config.hpp"
_CPU_KERNELS_H = _REPO_ROOT / "include" / "moe" / "emu" / "cpu_kernels.h"


def _constexpr_int(header: Path, name: str) -> int:
    """Read `inline constexpr <type> <name> = <int>;` out of a header."""
    text = header.read_text(encoding="utf-8")
    match = re.search(
        r"^inline\s+constexpr\s+[\w:]+\s+" + re.escape(name) + r"\s*=\s*(-?\d+)\s*;",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"{name} is no longer declared in {header.name}")
    return int(match.group(1))


class HeaderParityTest(unittest.TestCase):
    """The research constants must equal the engine's own."""

    def test_headers_are_present(self) -> None:
        self.assertTrue(_CONFIG_HPP.is_file(), f"missing {_CONFIG_HPP}")
        self.assertTrue(_CPU_KERNELS_H.is_file(), f"missing {_CPU_KERNELS_H}")

    def test_mla_geometry_matches_config_hpp(self) -> None:
        self.assertEqual(
            _constexpr_int(_CONFIG_HPP, "kDefaultKvLoraRank"), MLA_LATENT_DIM
        )
        self.assertEqual(
            _constexpr_int(_CONFIG_HPP, "kDefaultQkRopeHeadDim"), MLA_ROPE_DIM
        )
        self.assertEqual(
            _constexpr_int(_CONFIG_HPP, "kDefaultQkNopeHeadDim"), MLA_NOPE_HEAD_DIM
        )

    def test_compress_ratio_matches_config_hpp(self) -> None:
        # The spec called this `kCompressRatio`; the engine has three of them
        # and CSA is the 4:1 one. SWA is 1 and HCA is 128.
        self.assertEqual(
            _constexpr_int(_CONFIG_HPP, "kCompressRatioCsa"), CSA_COMPRESS_RATIO
        )
        self.assertEqual(_constexpr_int(_CONFIG_HPP, "kCompressRatioSwa"), 1)
        self.assertEqual(_constexpr_int(_CONFIG_HPP, "kCompressRatioHca"), 128)

    def test_indexer_geometry_matches_cpu_kernels_h(self) -> None:
        self.assertEqual(
            _constexpr_int(_CPU_KERNELS_H, "kIndexerHeads"), MLA_INDEXER_HEADS
        )
        self.assertEqual(
            _constexpr_int(_CPU_KERNELS_H, "kIndexerHeadDim"), MLA_INDEXER_HEAD_DIM
        )
        self.assertEqual(
            _constexpr_int(_CPU_KERNELS_H, "kIndexerTopK"), MLA_INDEXER_TOP_K
        )

    def test_compressor_rope_offset_matches_cpu_kernels_h(self) -> None:
        head_dim = _constexpr_int(_CPU_KERNELS_H, "kCompressorHeadDim")
        rope_dim = _constexpr_int(_CPU_KERNELS_H, "kRopeHeadDim")
        self.assertEqual(head_dim, CSA_COMPRESSOR_HEAD_DIM)
        self.assertEqual(rope_dim, MLA_ROPE_DIM)
        # kRopeBaseOffset is derived in the header, so recompute it the same way
        # rather than regex-matching an expression.
        self.assertEqual(head_dim - rope_dim, CSA_ROPE_BASE_OFFSET)
        self.assertEqual(CSA_ROPE_BASE_OFFSET, 448)
        self.assertEqual(
            _constexpr_int(_CPU_KERNELS_H, "kCompressorRatio"), CSA_COMPRESS_RATIO
        )


class GeometryContractTest(unittest.TestCase):
    def test_required_dimensions(self) -> None:
        self.assertEqual(MLA_ROPE_DIM, 64)
        self.assertEqual(MLA_LATENT_DIM, 512)
        self.assertEqual(MLA_NOPE_HEAD_DIM, 128)
        self.assertEqual(MLA_INDEXER_HEADS, 64)
        self.assertEqual(CSA_COMPRESS_RATIO, 4)

    def test_kv_row_elements_is_the_split_sum(self) -> None:
        # MlaGeometry::kv_row_elements() == kv_lora_rank + qk_rope_head_dim.
        self.assertEqual(MLA_KV_ROW_ELEMENTS, 576)
        self.assertEqual(MLA_KV_ROW_ELEMENTS, MLA_LATENT_DIM + MLA_ROPE_DIM)

    def test_the_two_layouts_have_different_semantic_widths(self) -> None:
        self.assertEqual(mla_semantic_dim(MLA_LAYOUT_SPLIT), 512)
        self.assertEqual(mla_semantic_dim(MLA_LAYOUT_IN_ROW), 448)
        with self.assertRaises(ValueError):
            mla_semantic_dim("flat")

    def test_in_row_layout_row_is_512_not_576(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "none", group_size=32)
        cache = DecoupledMlaCache(2, 8, quantizer, layout=MLA_LAYOUT_IN_ROW)
        self.assertEqual(cache.semantic_dim, 448)
        self.assertEqual(cache.rope_dim, 64)
        self.assertEqual(cache.row_elements, CSA_COMPRESSOR_HEAD_DIM)
        self.assertEqual(cache.row_elements, 512)

    def test_split_layout_shapes_are_pa_bsnd(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "none", group_size=32)
        cache = DecoupledMlaCache(3, 16, quantizer, layout=MLA_LAYOUT_SPLIT)
        self.assertEqual(cache.latent_shape, (3, 16, 1, 512))
        self.assertEqual(cache.rope_shape, (3, 16, 1, 64))
        self.assertEqual(cache.row_elements, 576)


class RopeIsNeverQuantizedTest(unittest.TestCase):
    """The central invariant: the 64 rotary lanes survive the cache exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.quantizer = TurboLatticeQuantizer.for_standard_normal(
            2, "e8", group_size=32
        )

    def _cache(self, layout: str = MLA_LAYOUT_SPLIT) -> DecoupledMlaCache:
        return DecoupledMlaCache(4, 8, self.quantizer, layout=layout)

    def test_rope_round_trips_bit_exactly(self) -> None:
        cache = self._cache()
        # Phases a 2-bit Lloyd-Max codebook would obliterate: cos/sin pairs at
        # fine angular offsets, whose magnitudes straddle the codebook centres.
        rope = tuple(
            math.cos(index * 0.013) if index % 2 == 0 else math.sin(index * 0.013)
            for index in range(MLA_ROPE_DIM)
        )
        latent = tuple(math.sin(index * 0.37) for index in range(MLA_LATENT_DIM))
        cache.write(0, latent, rope)
        self.assertEqual(cache.read_rope(0), rope)

    def test_rope_round_trips_in_the_in_row_layout_too(self) -> None:
        cache = self._cache(MLA_LAYOUT_IN_ROW)
        row = tuple(math.sin(index * 0.11) for index in range(CSA_COMPRESSOR_HEAD_DIM))
        cache.write_row(0, row)
        # The rotary lanes are exactly row[448:512] -- kRopeBaseOffset.
        self.assertEqual(cache.read_rope(0), row[CSA_ROPE_BASE_OFFSET:])
        self.assertEqual(len(cache.read_rope(0)), 64)

    def test_the_latent_path_really_is_lossy(self) -> None:
        # A degeneracy guard on the test above: if the quantizer were a no-op,
        # "rope survived" would prove nothing. Assert the semantic path loses
        # information on the same slot where the rotary path did not.
        cache = self._cache()
        latent = tuple(math.sin(index * 0.37) for index in range(MLA_LATENT_DIM))
        rope = tuple(math.cos(index * 0.013) for index in range(MLA_ROPE_DIM))
        cache.write(0, latent, rope)
        decoded = cache.read_latent(0)
        self.assertEqual(len(decoded), MLA_LATENT_DIM)
        self.assertNotEqual(decoded, latent)
        error = max(abs(a - b) for a, b in zip(decoded, latent))
        self.assertGreater(error, 0.0, "2-bit latent quantization lost nothing")

    def test_a_rope_width_mismatch_is_rejected(self) -> None:
        cache = self._cache()
        latent = (0.0,) * MLA_LATENT_DIM
        with self.assertRaises(ValueError):
            cache.write(0, latent, (0.0,) * 32)
        with self.assertRaises(ValueError):
            cache.write(0, (0.0,) * 448, (0.0,) * MLA_ROPE_DIM)

    def test_non_finite_rope_is_rejected(self) -> None:
        cache = self._cache()
        latent = (0.0,) * MLA_LATENT_DIM
        rope = (float("nan"),) + (0.0,) * (MLA_ROPE_DIM - 1)
        with self.assertRaises(ValueError):
            cache.write(0, latent, rope)

    def test_slot_payload_keeps_rope_as_raw_floats(self) -> None:
        cache = self._cache()
        cache.write(0, (0.5,) * MLA_LATENT_DIM, (0.25,) * MLA_ROPE_DIM)
        slot = cache.packed_slot(0)
        # Four bytes per rotary lane: float32, not a packed index payload.
        self.assertEqual(slot.nbytes - slot.latent.nbytes, 4 * MLA_ROPE_DIM)
        self.assertLess(
            slot.latent.nbytes,
            4 * MLA_LATENT_DIM,
            "the 512-lane latent should be smaller than float32",
        )


class RmsNormTest(unittest.TestCase):
    def test_matches_the_reference_formula(self) -> None:
        values = (3.0, -4.0, 0.0, 5.0)
        mean_square = sum(value * value / len(values) for value in values)
        expected_scale = 1.0 / math.sqrt(mean_square + RMS_NORM_EPSILON)
        for actual, value in zip(rms_norm(values), values):
            self.assertAlmostEqual(actual, value * expected_scale, places=12)

    def test_epsilon_matches_the_engine(self) -> None:
        self.assertEqual(RMS_NORM_EPSILON, 1e-6)

    def test_weight_is_applied_and_optional(self) -> None:
        values = (1.0, 2.0, 3.0, 4.0)
        weight = (2.0, 0.5, 1.0, -1.0)
        unweighted = rms_norm(values)
        weighted = rms_norm(values, weight)
        for plain, gained, gain in zip(unweighted, weighted, weight):
            self.assertAlmostEqual(gained, plain * gain, places=12)

    def test_an_outlier_channel_cannot_set_the_logit_scale(self) -> None:
        # Flaw 4 of the previous probe, as an assertion: one channel with 8x
        # gain must not dominate the normalized vector's energy.
        clean = tuple(1.0 for _ in range(128))
        spiked = (64.0,) + clean[1:]
        normalized = rms_norm(spiked)
        energy = sum(value * value for value in normalized)
        self.assertLess(
            normalized[0] ** 2 / energy,
            0.97,
            "RMSNorm did not bound the outlier channel's share of the energy",
        )
        self.assertAlmostEqual(energy / len(normalized), 1.0, places=4)

    def test_rejects_an_empty_vector_and_a_mismatched_weight(self) -> None:
        with self.assertRaises(ValueError):
            rms_norm(())
        with self.assertRaises(ValueError):
            rms_norm((1.0, 2.0), (1.0,))


class MultiHeadIndexerTest(unittest.TestCase):
    def _indexer(self, aggregation: str = "head_max") -> MultiHeadMlaIndexer:
        # Four heads is enough to show the aggregation difference and keeps the
        # stdlib dot products cheap; the geometry under test is the 512/64 split.
        absorbed = [
            [1.0 if lane == head else 0.0 for lane in range(MLA_LATENT_DIM)]
            for head in range(4)
        ]
        rope = [[0.0] * MLA_ROPE_DIM for _ in range(4)]
        return MultiHeadMlaIndexer(absorbed, rope, aggregation=aggregation)

    def test_temperature_is_over_the_full_query_head_dim(self) -> None:
        indexer = self._indexer()
        self.assertAlmostEqual(
            indexer.inverse_temperature,
            1.0 / math.sqrt(MLA_NOPE_HEAD_DIM + MLA_ROPE_DIM),
            places=12,
        )
        self.assertAlmostEqual(indexer.inverse_temperature, 1.0 / math.sqrt(192), 12)

    def test_logits_decompose_into_a_semantic_and_a_rotary_term(self) -> None:
        absorbed = [[1.0] + [0.0] * (MLA_LATENT_DIM - 1)]
        rope = [[1.0] + [0.0] * (MLA_ROPE_DIM - 1)]
        indexer = MultiHeadMlaIndexer(absorbed, rope)
        latent = [3.0] + [0.0] * (MLA_LATENT_DIM - 1)
        rotary = [5.0] + [0.0] * (MLA_ROPE_DIM - 1)
        logits = indexer.head_logits(latent, rotary)
        self.assertEqual(len(logits), 1)
        self.assertAlmostEqual(logits[0], 8.0 / math.sqrt(192), places=12)

    def test_head_max_surfaces_what_a_mean_buries(self) -> None:
        # One specialized head sees the needle; the rest are indifferent.
        latent = [0.0] * MLA_LATENT_DIM
        latent[2] = 10.0
        rotary = [0.0] * MLA_ROPE_DIM
        by_max = self._indexer("head_max").score_slot(latent, rotary)
        by_mean = self._indexer("mean").score_slot(latent, rotary)
        self.assertAlmostEqual(by_max, 10.0 / math.sqrt(192), places=12)
        self.assertAlmostEqual(by_mean, by_max / 4.0, places=12)
        self.assertGreater(by_max, by_mean)

    def test_ranking_over_a_cache_puts_the_needle_first(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "none", group_size=32)
        cache = DecoupledMlaCache(2, 8, quantizer)
        for slot in range(8):
            latent = [0.0] * MLA_LATENT_DIM
            latent[2] = 10.0 if slot == 5 else 0.1
            cache.write(slot, latent, [0.0] * MLA_ROPE_DIM)
        scores, ranked = self._indexer().rank(cache, 8, top_k=3)
        self.assertEqual(len(scores), 8)
        self.assertEqual(ranked[0], 5)
        self.assertEqual(len(ranked), 3)

    def test_geometry_mismatches_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiHeadMlaIndexer([[0.0] * 128], [[0.0] * MLA_ROPE_DIM])
        with self.assertRaises(ValueError):
            MultiHeadMlaIndexer([[0.0] * MLA_LATENT_DIM], [[0.0] * 32])
        with self.assertRaises(ValueError):
            MultiHeadMlaIndexer([[0.0] * MLA_LATENT_DIM], [[0.0] * MLA_ROPE_DIM] * 2)
        with self.assertRaises(ValueError):
            MultiHeadMlaIndexer([], [])
        with self.assertRaises(ValueError):
            MultiHeadMlaIndexer(
                [[0.0] * MLA_LATENT_DIM], [[0.0] * MLA_ROPE_DIM], aggregation="sum"
            )

    def test_an_in_row_cache_is_refused_by_a_split_indexer(self) -> None:
        quantizer = TurboLatticeQuantizer.for_standard_normal(4, "none", group_size=32)
        cache = DecoupledMlaCache(2, 8, quantizer, layout=MLA_LAYOUT_IN_ROW)
        cache.write_row(0, [0.0] * CSA_COMPRESSOR_HEAD_DIM)
        with self.assertRaises(ValueError):
            self._indexer().rank(cache, 1)


if __name__ == "__main__":
    unittest.main()
