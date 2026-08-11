"""Generation-time packing for fixed-point parameters."""

from collections.abc import Iterable
import math


def pack_fixed_point_words(
    values: Iterable[float],
    *,
    width: int,
    integer: int,
    signed: bool,
    lanes: int,
) -> tuple[int, ...]:
    """Pack quantized values with lane zero in the least-significant bits."""

    fractional = _validate_format(width, integer, lanes)
    lane_mask = (1 << width) - 1
    encoded = tuple(
        _encode_fixed(value, width, fractional, signed) for value in values
    )
    words = []
    for offset in range(0, len(encoded), lanes):
        word = 0
        for lane, value in enumerate(encoded[offset : offset + lanes]):
            word |= (value & lane_mask) << (lane * width)
        words.append(word)
    return tuple(words)


def unpack_fixed_point_words(
    words: Iterable[int],
    *,
    count: int,
    width: int,
    integer: int,
    signed: bool,
    lanes: int,
) -> tuple[float, ...]:
    """Decode packed fixed-point words for validation."""

    fractional = _validate_format(width, integer, lanes)
    if count < 0:
        raise ValueError("count must be non-negative")
    packed = tuple(words)
    if len(packed) * lanes < count:
        raise ValueError("packed words do not contain the requested values")
    lane_mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    scale = 1 << fractional
    decoded = []
    for index in range(count):
        code = (packed[index // lanes] >> ((index % lanes) * width)) & lane_mask
        if signed and code & sign_bit:
            code -= 1 << width
        decoded.append(code / scale)
    return tuple(decoded)


def _validate_format(width: int, integer: int, lanes: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    if integer > width:
        raise ValueError("integer bits cannot exceed width")
    if lanes <= 0:
        raise ValueError("lanes must be positive")
    return width - integer


def _encode_fixed(value: float, width: int, fractional: int, signed: bool) -> int:
    scaled = float(value) * (1 << fractional)
    quantized = round(scaled)
    if not math.isclose(scaled, quantized, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"value {value!r} is not exactly representable")
    minimum = -(1 << (width - 1)) if signed else 0
    maximum = (1 << (width - (1 if signed else 0))) - 1
    if quantized < minimum or quantized > maximum:
        raise ValueError(f"value {value!r} is outside the fixed-point range")
    return quantized & ((1 << width) - 1)
