"""Exact sharing of one hardware multiplier between two products with a common operand."""


def signed_width(numeric) -> int:
    """Width of a fixed-point code as a signed integer."""
    return numeric.width + (0 if numeric.signed else 1)


def shared_operand_field(common_bits: int, other_bits: int, ports: tuple[int, int]) -> int | None:
    """Field width that lets one multiplier form two products with a common operand.

    For signed codes `(high * 2**field + low) * common` equals
    `high * common * 2**field + low * common`. With `field` equal to the signed
    width of one product, the low field holds `low * common` exactly. The
    packed operand then needs `field + other_bits + 1` bits, which must fit
    the wide multiplier port while the common operand fits the narrow one.
    """
    field = common_bits + other_bits
    wide, narrow = max(ports), min(ports)
    if field + other_bits + 1 > wide or common_bits > narrow:
        return None
    return field


def split_shared_product(product: int, field: int) -> tuple[int, int]:
    """Recover `(high * common, low * common)` from the product of a packed operand."""
    low = product & ((1 << field) - 1)
    if low >> (field - 1):
        low -= 1 << field
    return (product - low) >> field, low
