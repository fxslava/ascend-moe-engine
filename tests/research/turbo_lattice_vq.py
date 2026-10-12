"""TurboQuant + Lloyd-Max + E8/E24 quantization for a CPU PA_BSND cache.

This module deliberately models only the CSA cache boundary:

* :class:`CSACompressor` emits one compressed key/value for each group of four
  input tokens.
* :class:`PagedQuantizedCache.write` quantizes that emitted slot.  The cache
  never retains the unquantized key or value.
* :class:`PagedQuantizedCache.read` and ``attention`` decode the packed slot
  before scoring or accumulating values.

The packed representation applies the operations in the requested order.  On
write, a group scale normalizes values, the normalized values are rounded
through FP8, a scalar Lloyd-Max centre is selected, and the residual is either
omitted or mapped to E8/E24.  On read, any lattice residual is decoded, added
to the Lloyd-Max reconstruction, and finally multiplied by the block scale.

It is a deterministic research model, not a wire-compatible CANN format.
Only the Python standard library is required and no NPU runtime is imported.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from statistics import NormalDist
import struct
from typing import Iterable, List, Optional, Sequence, Tuple


_FP8_MAX = 448.0
_E8_DIMENSION = 8
_LEECH_DIMENSION = 24
_LEECH_NORMALIZATION = math.sqrt(8.0)


class E4M3FN:
    """Table-driven finite E4M3 encoder/decoder.

    Encoding uses round-to-nearest with a deterministic lower-code tie break.
    NaN is rejected and infinities saturate.  Code 0x7f/0xff is the E4M3FN
    NaN encoding; all finite magnitudes, including the extended exponent-15
    range through 448, are represented in the lookup table.
    """

    @staticmethod
    def decode(code: int) -> float:
        if not 0 <= code <= 255:
            raise ValueError("an FP8 code must fit in one byte")
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            value = mantissa * (2.0 ** -9)
        elif exponent == 0x0F:
            if mantissa == 0x07:
                return math.nan
            value = (1.0 + mantissa / 8.0) * (2.0 ** 8)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        return sign * value

    @classmethod
    @lru_cache(maxsize=1)
    def _positive_table(cls) -> Tuple[Tuple[float, int], ...]:
        return tuple(
            (cls.decode(code), code)
            for code in range(0x80)
            if code != 0x7F
        )

    @classmethod
    def encode(cls, value: float) -> int:
        if math.isnan(value):
            raise ValueError("NaN cannot be encoded in the research cache")
        negative = math.copysign(1.0, value) < 0.0
        magnitude = min(abs(value), _FP8_MAX)
        table = cls._positive_table()
        values = cls._positive_values()
        insertion = bisect_left(values, magnitude)
        if insertion == 0:
            code = table[0][1]
        elif insertion == len(table):
            code = table[-1][1]
        else:
            low_value, low_code = table[insertion - 1]
            high_value, high_code = table[insertion]
            code = low_code if magnitude - low_value <= high_value - magnitude else high_code
        if negative and code != 0:
            code |= 0x80
        return code

    @classmethod
    @lru_cache(maxsize=1)
    def _positive_values(cls) -> Tuple[float, ...]:
        return tuple(item[0] for item in cls._positive_table())

    @classmethod
    def quantize(cls, value: float) -> float:
        return cls.decode(cls.encode(value))


class LloydMaxQuantizer:
    """Deterministic scalar Lloyd-Max quantizer with at most 16 levels."""

    def __init__(self, centres: Sequence[float]):
        if not centres or len(centres) > 16:
            raise ValueError("Lloyd-Max requires between 1 and 16 centres")
        ordered = tuple(float(value) for value in centres)
        if any(not math.isfinite(value) for value in ordered):
            raise ValueError("Lloyd-Max centres must be finite")
        if tuple(sorted(ordered)) != ordered:
            raise ValueError("Lloyd-Max centres must be sorted")
        self.centres = ordered
        self.thresholds = tuple(
            (ordered[index] + ordered[index + 1]) * 0.5
            for index in range(len(ordered) - 1)
        )

    @classmethod
    def fit(
        cls,
        samples: Iterable[float],
        levels: int = 16,
        iterations: int = 64,
    ) -> "LloydMaxQuantizer":
        values = sorted(float(value) for value in samples)
        if not values:
            raise ValueError("at least one Lloyd-Max calibration sample is required")
        if not 1 <= levels <= 16:
            raise ValueError("levels must be in [1, 16] so indices fit in nibbles")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Lloyd-Max samples must be finite")
        if values[0] == values[-1]:
            return cls((values[0],))

        centres = []
        for level in range(levels):
            position = (level + 0.5) * len(values) / levels
            index = min(len(values) - 1, int(position))
            centres.append(values[index])

        for _ in range(iterations):
            thresholds = [
                (centres[index] + centres[index + 1]) * 0.5
                for index in range(levels - 1)
            ]
            sums = [0.0] * levels
            counts = [0] * levels
            for value in values:
                bucket = bisect_right(thresholds, value)
                sums[bucket] += value
                counts[bucket] += 1
            updated = [
                sums[index] / counts[index] if counts[index] else centres[index]
                for index in range(levels)
            ]
            if max(abs(updated[index] - centres[index]) for index in range(levels)) < 1e-7:
                centres = updated
                break
            centres = updated
        return cls(tuple(centres))

    @classmethod
    @lru_cache(maxsize=3)
    def standard_normal(
        cls, levels: int, iterations: int = 256
    ) -> "LloydMaxQuantizer":
        """Optimal MSE Lloyd-Max codebook for a unit Gaussian source."""
        if levels == 4:
            return cls((-1.510418, -0.452794, 0.452794, 1.510418))
        if levels not in (8, 16):
            raise ValueError("the ablation uses 4-, 8-, or 16-level Gaussian codebooks")
        normal = NormalDist()
        centres = [
            normal.inv_cdf((index + 0.5) / levels) for index in range(levels)
        ]
        root_two_pi = math.sqrt(2.0 * math.pi)

        def density(value: float) -> float:
            if math.isinf(value):
                return 0.0
            return math.exp(-0.5 * value * value) / root_two_pi

        for _ in range(iterations):
            thresholds = [-math.inf]
            thresholds.extend(
                (centres[index] + centres[index + 1]) * 0.5
                for index in range(levels - 1)
            )
            thresholds.append(math.inf)
            updated = []
            for index in range(levels):
                low = thresholds[index]
                high = thresholds[index + 1]
                probability = normal.cdf(high) - normal.cdf(low)
                updated.append((density(low) - density(high)) / probability)
            if max(abs(a - b) for a, b in zip(updated, centres)) < 1e-14:
                centres = updated
                break
            centres = updated
        return cls(tuple(centres))

    def encode(self, value: float) -> int:
        return bisect_right(self.thresholds, value)

    def decode(self, index: int) -> float:
        if not 0 <= index < len(self.centres):
            raise ValueError("Lloyd-Max index is outside the codebook")
        return self.centres[index]


def _nearest_dn(values: Sequence[float]) -> Tuple[int, ...]:
    """Nearest point in D_n: integer coordinates with even coordinate sum."""
    rounded = [int(math.floor(value + 0.5)) for value in values]
    if sum(rounded) % 2 == 0:
        return tuple(rounded)

    best_index = 0
    best_delta = math.inf
    best_adjustment = 0
    for index, (value, integer) in enumerate(zip(values, rounded)):
        down = integer - 1
        up = integer + 1
        down_cost = (value - down) ** 2 - (value - integer) ** 2
        up_cost = (value - up) ** 2 - (value - integer) ** 2
        adjustment, delta = (-1, down_cost) if down_cost <= up_cost else (1, up_cost)
        if delta < best_delta:
            best_index = index
            best_delta = delta
            best_adjustment = adjustment
    rounded[best_index] += best_adjustment
    return tuple(rounded)


def nearest_e8_doubled(values: Sequence[float]) -> Tuple[int, ...]:
    """Return twice the nearest E8 lattice point.

    ``E8 = D8 union (D8 + (1/2,...,1/2))``.  Doubled coordinates are packed
    because they exactly encode both cosets without a separate tag.
    """
    if len(values) != _E8_DIMENSION:
        raise ValueError("E8 nearest-point search requires exactly eight values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("E8 input must be finite")

    integer_point = _nearest_dn(values)
    half_base = _nearest_dn(tuple(value - 0.5 for value in values))
    half_point = tuple(value + 0.5 for value in half_base)
    integer_error = sum((value - point) ** 2 for value, point in zip(values, integer_point))
    half_error = sum((value - point) ** 2 for value, point in zip(values, half_point))
    if integer_error <= half_error:
        return tuple(2 * value for value in integer_point)
    return tuple(2 * value + 1 for value in half_base)


def decode_e8_doubled(codeword: Sequence[int]) -> Tuple[float, ...]:
    if len(codeword) != _E8_DIMENSION:
        raise ValueError("an E8 codeword has eight doubled coordinates")
    parity = codeword[0] & 1
    if any((coordinate & 1) != parity for coordinate in codeword):
        raise ValueError("an E8 codeword must lie wholly in one coset")
    if parity == 0:
        if sum(coordinate // 2 for coordinate in codeword) % 2:
            raise ValueError("integer E8 codeword violates D8 parity")
    else:
        if sum((coordinate - 1) // 2 for coordinate in codeword) % 2:
            raise ValueError("half-integer E8 codeword violates D8 parity")
    return tuple(coordinate * 0.5 for coordinate in codeword)


def _carryless_multiply(left: int, right: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        left <<= 1
        right >>= 1
    return result


@lru_cache(maxsize=1)
def extended_golay_codewords() -> Tuple[int, ...]:
    """The binary extended Golay [24,12,8] code in coordinate-bit order."""
    words = []
    for message in range(4096):
        cyclic = _carryless_multiply(message, 0xC75)
        parity = cyclic.bit_count() & 1
        words.append(cyclic | (parity << 23))
    return tuple(sorted(words))


def _golay_syndrome(word: int, basis: Sequence[int]) -> int:
    syndrome = 0
    for index, row in enumerate(basis):
        syndrome |= ((word & row).bit_count() & 1) << index
    return syndrome


@lru_cache(maxsize=1)
def golay_decoder_tables() -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Return low/high syndrome LUTs and the radius-4 syndrome leaders.

    The extended Golay code is self-dual, so twelve independent generator
    rows also form a parity-check matrix.  Its covering radius is four; the
    first weight-ordered error pattern stored for each syndrome is therefore
    a deterministic nearest-Hamming coset leader.
    """
    basis = tuple(
        _carryless_multiply(1 << bit, 0xC75)
        | ((_carryless_multiply(1 << bit, 0xC75).bit_count() & 1) << 23)
        for bit in range(12)
    )
    if any(_golay_syndrome(word, basis) != 0 for word in extended_golay_codewords()):
        raise AssertionError("the Golay generator basis is not self-orthogonal")

    leaders = [-1] * 4096
    for weight in range(5):
        for positions in combinations(range(24), weight):
            error = sum(1 << position for position in positions)
            syndrome = _golay_syndrome(error, basis)
            if leaders[syndrome] < 0:
                leaders[syndrome] = error
    if any(leader < 0 for leader in leaders):
        raise AssertionError("Golay radius-4 syndrome table is incomplete")
    low = tuple(_golay_syndrome(word, basis) for word in range(4096))
    high = tuple(_golay_syndrome(word << 12, basis) for word in range(4096))
    return low, high, tuple(leaders)


def golay_bounded_decode(word: int) -> int:
    """Decode a 24-bit word with the deterministic radius-4 Golay table."""
    word &= (1 << 24) - 1
    low, high, leaders = golay_decoder_tables()
    syndrome = low[word & 0xFFF] ^ high[(word >> 12) & 0xFFF]
    return word ^ leaders[syndrome]


def is_leech_integer_point(point: Sequence[int]) -> bool:
    """Membership in the integer embedding ``sqrt(8) * Lambda_24``."""
    if len(point) != _LEECH_DIMENSION:
        return False
    parity = point[0] & 1
    if any((coordinate & 1) != parity for coordinate in point):
        return False
    residue_word = 0
    for index, coordinate in enumerate(point):
        residue_word |= (((coordinate - parity) // 2) & 1) << index
    if golay_bounded_decode(residue_word) != residue_word:
        return False
    return sum(point) % 8 == 4 * parity


def nearest_leech24_integer(
    values: Sequence[float], chase_bits: int = 3
) -> Tuple[int, ...]:
    """Fast mirrored-Golay bounded-distance decoder for ``Lambda_24``.

    Both integer-coordinate parity cosets are tried.  Within each coset a
    hard Golay decision plus all flips of the ``chase_bits`` least-reliable
    residue decisions are syndrome-decoded, scored in Euclidean distance,
    and repaired to the required global sum congruence.  The returned point
    is always a valid member of ``sqrt(8) * Lambda_24``.

    This is a Chase/Construction-A bounded-distance decoder.  It avoids an
    exhaustive 4,096-codeword soft search and intentionally does not claim
    maximum-likelihood nearest-neighbour behavior outside its decoding
    radius.
    """
    if len(values) != _LEECH_DIMENSION:
        raise ValueError("the Leech decoder requires exactly 24 coordinates")
    if not 0 <= chase_bits <= 8:
        raise ValueError("chase_bits must be in [0, 8]")
    target = tuple(float(value) * _LEECH_NORMALIZATION for value in values)
    if any(not math.isfinite(value) for value in target):
        raise ValueError("Leech decoder input must be finite")

    best_error = math.inf
    best_point: Optional[Tuple[int, ...]] = None
    for parity in (0, 1):
        coordinate_x: List[Tuple[int, int]] = []
        coordinate_cost: List[Tuple[float, float]] = []
        hard_word = 0
        reliabilities = []
        for index, value in enumerate(target):
            candidates_x = []
            candidates_cost = []
            for residue_bit in (0, 1):
                offset = parity + 2 * residue_bit
                k = int(math.floor((value - offset) / 4.0 + 0.5))
                coordinate = offset + 4 * k
                candidates_x.append(coordinate)
                candidates_cost.append((value - coordinate) ** 2)
            coordinate_x.append((candidates_x[0], candidates_x[1]))
            coordinate_cost.append((candidates_cost[0], candidates_cost[1]))
            hard_bit = int(candidates_cost[1] < candidates_cost[0])
            hard_word |= hard_bit << index
            reliabilities.append((abs(candidates_cost[1] - candidates_cost[0]), index))

        least_reliable = tuple(
            index for _, index in sorted(reliabilities)[:chase_bits]
        )
        decoded_words = set()
        for chase_pattern in range(1 << chase_bits):
            received = hard_word
            for bit, coordinate in enumerate(least_reliable):
                if chase_pattern >> bit & 1:
                    received ^= 1 << coordinate
            decoded_words.add(golay_bounded_decode(received))

        for codeword in sorted(decoded_words):
            point = []
            ks = []
            error = 0.0
            for index, value in enumerate(target):
                bit = codeword >> index & 1
                coordinate = coordinate_x[index][bit]
                point.append(coordinate)
                ks.append((coordinate - parity - 2 * bit) // 4)
                error += coordinate_cost[index][bit]
            if sum(ks) % 2 != parity:
                repair_delta = math.inf
                repair_index = 0
                repair_shift = 0
                for index, (value, coordinate) in enumerate(zip(target, point)):
                    down_delta = (value - (coordinate - 4)) ** 2 - (value - coordinate) ** 2
                    up_delta = (value - (coordinate + 4)) ** 2 - (value - coordinate) ** 2
                    shift, delta = (-4, down_delta) if down_delta <= up_delta else (4, up_delta)
                    if delta < repair_delta:
                        repair_delta = delta
                        repair_index = index
                        repair_shift = shift
                point[repair_index] += repair_shift
                error += repair_delta
            candidate = tuple(point)
            if error < best_error or (error == best_error and candidate < best_point):
                best_error = error
                best_point = candidate
    if best_point is None or not is_leech_integer_point(best_point):
        raise AssertionError("mirrored Golay decoder produced a non-Leech point")
    return best_point


def _pack_indices(indices: Sequence[int], bits: int) -> bytes:
    if bits not in (2, 3, 4):
        raise ValueError("Lloyd-Max indices use two, three, or four bits")
    packed = bytearray((len(indices) * bits + 7) // 8)
    mask = (1 << bits) - 1
    bit_offset = 0
    for index in indices:
        if not 0 <= index <= mask:
            raise ValueError("a Lloyd-Max index does not fit its configured width")
        byte = bit_offset // 8
        shift = bit_offset & 7
        packed[byte] |= (index << shift) & 0xFF
        if shift + bits > 8:
            packed[byte + 1] |= index >> (8 - shift)
        bit_offset += bits
    return bytes(packed)


def _unpack_indices(payload: bytes, count: int, bits: int) -> Tuple[int, ...]:
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("corrupt packed Lloyd-Max payload")
    mask = (1 << bits) - 1
    output = []
    bit_offset = 0
    for _ in range(count):
        byte = bit_offset // 8
        shift = bit_offset & 7
        value = payload[byte] >> shift
        if shift + bits > 8:
            value |= payload[byte + 1] << (8 - shift)
        output.append(value & mask)
        bit_offset += bits
    return tuple(output)


@dataclass(frozen=True)
class PackedVector:
    """No-raw-float cache payload for one key or value vector."""

    dimension: int
    group_size: int
    index_bits: int
    lattice_mode: str
    padded_dimension: int
    scales_f32: bytes
    lloyd_indices_packed: bytes
    lattice_codewords_i16: bytes

    @property
    def lloyd_indices_u4(self) -> bytes:
        """Compatibility alias retained for the original E8 probe."""
        return self.lloyd_indices_packed

    @property
    def e8_codewords_i16(self) -> bytes:
        """Compatibility alias retained for the original E8 probe."""
        return self.lattice_codewords_i16

    @property
    def nbytes(self) -> int:
        return (
            len(self.scales_f32)
            + len(self.lloyd_indices_packed)
            + len(self.lattice_codewords_i16)
        )


class TurboLatticeQuantizer:
    """Block-scaled E4M3FN plus scalar, E8, or E24 residual quantization."""

    def __init__(
        self,
        lloyd: LloydMaxQuantizer,
        group_size: int = 32,
        lattice_step: float = 4.0,
        lattice_mode: str = "e8",
        index_bits: Optional[int] = None,
        scale_mode: str = "max_fp8",
        leech_chase_bits: int = 3,
    ):
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if lattice_step <= 0.0 or not math.isfinite(lattice_step):
            raise ValueError("lattice_step must be finite and positive")
        if lattice_mode not in ("none", "e8", "e24"):
            raise ValueError("lattice_mode must be none, e8, or e24")
        if scale_mode not in ("max_fp8", "rms"):
            raise ValueError("scale_mode must be max_fp8 or rms")
        inferred_bits = max(1, (len(lloyd.centres) - 1).bit_length())
        index_bits = inferred_bits if index_bits is None else index_bits
        if index_bits not in (2, 3, 4) or len(lloyd.centres) > 1 << index_bits:
            raise ValueError("index_bits cannot represent the Lloyd-Max codebook")
        self.lloyd = lloyd
        self.group_size = group_size
        self.lattice_step = float(lattice_step)
        self.lattice_mode = lattice_mode
        self.index_bits = index_bits
        self.scale_mode = scale_mode
        self.leech_chase_bits = leech_chase_bits

    @classmethod
    def calibrate(
        cls,
        vectors: Iterable[Sequence[float]],
        group_size: int = 32,
        levels: int = 16,
        lattice_step: float = 4.0,
        max_samples: int = 65536,
    ) -> "TurboLatticeQuantizer":
        normalized_fp8: List[float] = []
        for vector in vectors:
            if len(vector) % group_size:
                raise ValueError("calibration vector dimension must be divisible by group_size")
            for start in range(0, len(vector), group_size):
                group = vector[start : start + group_size]
                maximum = max(abs(float(value)) for value in group)
                scale = maximum / _FP8_MAX if maximum else 1.0
                for value in group:
                    normalized_fp8.append(E4M3FN.quantize(float(value) / scale))
            if len(normalized_fp8) >= max_samples:
                break
        if len(normalized_fp8) > max_samples:
            stride = len(normalized_fp8) / max_samples
            normalized_fp8 = [normalized_fp8[int(index * stride)] for index in range(max_samples)]
        return cls(
            LloydMaxQuantizer.fit(normalized_fp8, levels=levels),
            group_size=group_size,
            lattice_step=lattice_step,
            lattice_mode="e8",
            index_bits=max(3, (levels - 1).bit_length()),
            scale_mode="max_fp8",
        )

    @classmethod
    def for_standard_normal(
        cls,
        bits: int,
        lattice_mode: str,
        group_size: int = 32,
        lattice_step: Optional[float] = None,
        leech_chase_bits: int = 3,
    ) -> "TurboLatticeQuantizer":
        if bits not in (2, 3, 4):
            raise ValueError("TurboQuant ablation bits must be two, three, or four")
        lloyd = LloydMaxQuantizer.standard_normal(1 << bits)
        if lattice_step is None:
            centre_gaps = [
                lloyd.centres[index + 1] - lloyd.centres[index]
                for index in range(len(lloyd.centres) - 1)
            ]
            lattice_step = sorted(centre_gaps)[len(centre_gaps) // 2] * 0.5
        return cls(
            lloyd,
            group_size=group_size,
            lattice_step=lattice_step,
            lattice_mode=lattice_mode,
            index_bits=bits,
            scale_mode="rms",
            leech_chase_bits=leech_chase_bits,
        )

    def _scale(self, group: Sequence[float]) -> float:
        if self.scale_mode == "rms":
            return math.sqrt(sum(float(value) ** 2 for value in group) / len(group)) or 1.0
        maximum = max(abs(float(value)) for value in group)
        return maximum / _FP8_MAX if maximum else 1.0

    def encode_vector(self, vector: Sequence[float]) -> PackedVector:
        dimension = len(vector)
        if dimension == 0 or dimension % self.group_size:
            raise ValueError("vector dimension must be a non-zero multiple of group_size")
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("cache vectors must be finite")

        scales: List[float] = []
        indices: List[int] = []
        residuals: List[float] = []
        for start in range(0, dimension, self.group_size):
            group = vector[start : start + self.group_size]
            scale = self._scale(group)
            scales.append(scale)
            for value in group:
                fp8_value = E4M3FN.quantize(float(value) / scale)
                index = self.lloyd.encode(fp8_value)
                indices.append(index)
                residuals.append(fp8_value - self.lloyd.decode(index))

        lattice_coordinates: List[int] = []
        padded_dimension = dimension
        if self.lattice_mode == "e8":
            if dimension % _E8_DIMENSION:
                raise ValueError("E8 vectors must be divisible by eight")
            for start in range(0, dimension, _E8_DIMENSION):
                lattice_input = tuple(
                    residual / self.lattice_step
                    for residual in residuals[start : start + _E8_DIMENSION]
                )
                lattice_coordinates.extend(nearest_e8_doubled(lattice_input))
        elif self.lattice_mode == "e24":
            padded_dimension = (
                (dimension + _LEECH_DIMENSION - 1) // _LEECH_DIMENSION
            ) * _LEECH_DIMENSION
            padded = residuals + [0.0] * (padded_dimension - dimension)
            for start in range(0, padded_dimension, _LEECH_DIMENSION):
                lattice_input = tuple(
                    residual / self.lattice_step
                    for residual in padded[start : start + _LEECH_DIMENSION]
                )
                lattice_coordinates.extend(
                    nearest_leech24_integer(lattice_input, self.leech_chase_bits)
                )
        if any(not -32768 <= coordinate <= 32767 for coordinate in lattice_coordinates):
            raise OverflowError("lattice codeword does not fit packed int16")

        return PackedVector(
            dimension=dimension,
            group_size=self.group_size,
            index_bits=self.index_bits,
            lattice_mode=self.lattice_mode,
            padded_dimension=padded_dimension,
            scales_f32=struct.pack("<" + "f" * len(scales), *scales),
            lloyd_indices_packed=_pack_indices(indices, self.index_bits),
            lattice_codewords_i16=struct.pack(
                "<" + "h" * len(lattice_coordinates), *lattice_coordinates
            ),
        )

    def decode_vector(self, packed: PackedVector) -> Tuple[float, ...]:
        if packed.group_size != self.group_size:
            raise ValueError("packed vector group_size does not match the quantizer")
        if packed.index_bits != self.index_bits or packed.lattice_mode != self.lattice_mode:
            raise ValueError("packed vector format does not match the quantizer")
        if packed.dimension % self.group_size:
            raise ValueError("corrupt packed vector dimension")
        group_count = packed.dimension // self.group_size
        if len(packed.scales_f32) != group_count * 4:
            raise ValueError("corrupt packed scale payload")
        expected_lattice = 0 if self.lattice_mode == "none" else packed.padded_dimension * 2
        if len(packed.lattice_codewords_i16) != expected_lattice:
            raise ValueError("corrupt packed lattice payload")

        scales = struct.unpack("<" + "f" * group_count, packed.scales_f32)
        indices = _unpack_indices(
            packed.lloyd_indices_packed, packed.dimension, packed.index_bits
        )
        lattice_residual = [0.0] * packed.dimension
        if self.lattice_mode != "none":
            coordinates = struct.unpack(
                "<" + "h" * packed.padded_dimension,
                packed.lattice_codewords_i16,
            )
            if self.lattice_mode == "e8":
                for start in range(0, packed.dimension, _E8_DIMENSION):
                    decoded = decode_e8_doubled(
                        coordinates[start : start + _E8_DIMENSION]
                    )
                    lattice_residual[start : start + _E8_DIMENSION] = decoded
            else:
                for position in range(packed.dimension):
                    lattice_residual[position] = (
                        coordinates[position] / _LEECH_NORMALIZATION
                    )
        decoded: List[float] = []
        for position in range(packed.dimension):
            group = position // self.group_size
            normalized = self.lloyd.decode(indices[position])
            normalized += lattice_residual[position] * self.lattice_step
            decoded.append(normalized * scales[group])
        return tuple(decoded)


@dataclass(frozen=True)
class QuantizedKVSlot:
    key: PackedVector
    value: PackedVector


@dataclass(frozen=True)
class AttentionResult:
    scores: Tuple[float, ...]
    ranked_indices: Tuple[int, ...]
    output: Tuple[float, ...]


class PagedQuantizedCache:
    """CPU PA_BSND ``[blocks, block_size, 1, D]`` packed K/V cache."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        head_dim: int,
        quantizer: TurboLatticeQuantizer,
        block_table: Optional[Sequence[int]] = None,
    ):
        if num_blocks <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError("cache extents must be positive")
        if head_dim % quantizer.group_size:
            raise ValueError("head_dim must be divisible by the quantizer group_size")
        table = tuple(range(num_blocks)) if block_table is None else tuple(block_table)
        if len(table) > num_blocks or any(not 0 <= block < num_blocks for block in table):
            raise ValueError("block_table contains an invalid physical block")
        if len(set(table)) != len(table):
            raise ValueError("block_table physical blocks must be unique")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        self.quantizer = quantizer
        self.block_table = table
        self._slots: List[Optional[QuantizedKVSlot]] = [None] * (num_blocks * block_size)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, 1, self.head_dim)

    def _physical_slot(self, logical_index: int) -> int:
        if logical_index < 0:
            raise IndexError("logical cache index must be non-negative")
        logical_block, within = divmod(logical_index, self.block_size)
        if logical_block >= len(self.block_table):
            raise IndexError("logical cache index exceeds the block table")
        return self.block_table[logical_block] * self.block_size + within

    def write(self, logical_index: int, key: Sequence[float], value: Sequence[float]) -> None:
        if len(key) != self.head_dim or len(value) != self.head_dim:
            raise ValueError("key/value dimension does not match PA_BSND head_dim")
        physical = self._physical_slot(logical_index)
        self._slots[physical] = QuantizedKVSlot(
            key=self.quantizer.encode_vector(key),
            value=self.quantizer.encode_vector(value),
        )

    def packed_slot(self, logical_index: int) -> QuantizedKVSlot:
        slot = self._slots[self._physical_slot(logical_index)]
        if slot is None:
            raise KeyError("the requested cache slot has not been written")
        return slot

    def read(self, logical_index: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        slot = self.packed_slot(logical_index)
        return (
            self.quantizer.decode_vector(slot.key),
            self.quantizer.decode_vector(slot.value),
        )

    def read_key(self, logical_index: int) -> Tuple[float, ...]:
        return self.quantizer.decode_vector(self.packed_slot(logical_index).key)

    def read_value(self, logical_index: int) -> Tuple[float, ...]:
        return self.quantizer.decode_vector(self.packed_slot(logical_index).value)

    def attention(
        self,
        query: Sequence[float],
        seq_len: int,
        top_k: Optional[int] = None,
    ) -> AttentionResult:
        """Decode on read, rank keys, and accumulate decoded values."""
        if len(query) != self.head_dim:
            raise ValueError("query dimension does not match PA_BSND head_dim")
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative")
        inverse_temperature = 1.0 / math.sqrt(self.head_dim)
        scores: List[float] = []
        for index in range(seq_len):
            key = self.read_key(index)
            scores.append(
                sum(float(q) * k for q, k in zip(query, key)) * inverse_temperature
            )
        ranked = sorted(range(seq_len), key=lambda index: (-scores[index], index))
        selected = ranked if top_k is None else ranked[: min(top_k, seq_len)]
        output = [0.0] * self.head_dim
        if selected:
            maximum = max(scores[index] for index in selected)
            weights = [math.exp(scores[index] - maximum) for index in selected]
            denominator = sum(weights)
            for index, weight in zip(selected, weights):
                value = self.read_value(index)
                probability = weight / denominator
                for lane in range(self.head_dim):
                    output[lane] += probability * value[lane]
        return AttentionResult(tuple(scores), tuple(ranked), tuple(output))


class CSACompressor:
    """Small CPU model of CSA's lane-wise softmax reduction and 4:1 cadence.

    The checkpoint-backed C4A path projects hidden states into eight rows
    (previous ``a`` plus current ``b``) before a lane-wise softmax.  A
    standalone synthetic probe has no checkpoint projections, so this class
    exposes the same lane-wise reduction over one ratio-sized group.  Passing
    no gate logits selects deterministic uniform averaging.
    """

    def __init__(self, ratio: int = 4):
        if ratio <= 0:
            raise ValueError("compression ratio must be positive")
        self.ratio = ratio

    def compress_group(
        self,
        keys: Sequence[Sequence[float]],
        values: Optional[Sequence[Sequence[float]]] = None,
        gate_logits: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        if len(keys) != self.ratio:
            raise ValueError("one compressor group must contain exactly ratio tokens")
        values = keys if values is None else values
        if len(values) != self.ratio:
            raise ValueError("key and value groups must have the same token count")
        dimension = len(keys[0])
        if dimension == 0 or any(len(vector) != dimension for vector in keys):
            raise ValueError("all compressor keys must have one non-zero dimension")
        if any(len(vector) != dimension for vector in values):
            raise ValueError("all compressor values must match the key dimension")
        if gate_logits is None:
            gate_logits = tuple(tuple(0.0 for _ in range(dimension)) for _ in keys)
        if len(gate_logits) != self.ratio or any(
            len(vector) != dimension for vector in gate_logits
        ):
            raise ValueError("gate logits must have shape [ratio, dimension]")

        compressed_key = [0.0] * dimension
        compressed_value = [0.0] * dimension
        for lane in range(dimension):
            maximum = max(float(gate_logits[token][lane]) for token in range(self.ratio))
            exponentials = [
                math.exp(float(gate_logits[token][lane]) - maximum)
                for token in range(self.ratio)
            ]
            denominator = sum(exponentials)
            for token, exponential in enumerate(exponentials):
                weight = exponential / denominator
                compressed_key[lane] += weight * float(keys[token][lane])
                compressed_value[lane] += weight * float(values[token][lane])
        return tuple(compressed_key), tuple(compressed_value)

    def compress_into_cache(
        self,
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        cache: PagedQuantizedCache,
        start_slot: int = 0,
    ) -> Tuple[Tuple[float, ...], ...]:
        if len(keys) != len(values) or len(keys) % self.ratio:
            raise ValueError("token streams must be equal and divisible by the ratio")
        exact_compressed: List[Tuple[float, ...]] = []
        for offset in range(0, len(keys), self.ratio):
            compressed_key, compressed_value = self.compress_group(
                keys[offset : offset + self.ratio],
                values[offset : offset + self.ratio],
            )
            slot = start_slot + offset // self.ratio
            cache.write(slot, compressed_key, compressed_value)
            exact_compressed.append(compressed_key)
        return tuple(exact_compressed)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine operands must have the same non-zero dimension")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
