import pytest

from ravel_hls.packing import pack_fixed_point_words, unpack_fixed_point_words


def test_fixed_point_words_use_little_lane_order_and_twos_complement() -> None:
    words = pack_fixed_point_words(
        [0.5, -0.25, -2.0],
        width=4,
        integer=2,
        signed=True,
        lanes=4,
    )

    assert words == (0x08F2,)
    assert unpack_fixed_point_words(
        words,
        count=3,
        width=4,
        integer=2,
        signed=True,
        lanes=4,
    ) == pytest.approx((0.5, -0.25, -2.0))
