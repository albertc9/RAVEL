from itertools import product

import pytest

from ravel_hls.analysis.multipliers import shared_operand_field, split_shared_product


def signed_values(bits):
    return range(-(1 << (bits - 1)), 1 << (bits - 1))


def test_two_products_with_a_common_operand_are_recovered_exactly_from_one_multiplication():
    common_bits, other_bits = 5, 4
    field = shared_operand_field(common_bits, other_bits, ports=(27, 18))

    assert field == common_bits + other_bits
    operand_bits = field + other_bits + 1
    for common, high, low in product(signed_values(common_bits), signed_values(other_bits), signed_values(other_bits)):
        packed = (high << field) + low
        assert -(1 << (operand_bits - 1)) <= packed < 1 << (operand_bits - 1)
        assert split_shared_product(packed * common, field) == (high * common, low * common)


def test_the_extreme_operands_of_the_qualified_widths_are_recovered_exactly():
    field = shared_operand_field(11, 7, ports=(27, 18))

    extremes = lambda bits: (-(1 << (bits - 1)), -1, 0, 1, (1 << (bits - 1)) - 1)
    for common, high, low in product(extremes(11), extremes(7), extremes(7)):
        assert split_shared_product(((high << field) + low) * common, field) == (high * common, low * common)


@pytest.mark.parametrize("common_bits, other_bits, field", [(11, 7, 18), (10, 7, 17), (9, 5, 14), (11, 8, None), (19, 3, None)])
def test_the_packed_operand_must_fit_the_multiplier_ports(common_bits, other_bits, field):
    assert shared_operand_field(common_bits, other_bits, ports=(27, 18)) == field
