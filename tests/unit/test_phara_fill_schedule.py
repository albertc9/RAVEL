import pytest

from ravel_hls.analysis.phara import (
    build_fill_cycle_schedule,
    build_row_credit_schedule,
    canonical_row_credit,
)


def test_the_canonical_geometry_needs_no_fill_cycles_and_matches_the_row_credit_schedule():
    arguments = dict(input_rows=256, temporal_pack=8, kernel_rows=5, convolution_stride=3, pool_rows=2)
    filled = build_fill_cycle_schedule(**arguments)
    canonical = build_row_credit_schedule(**arguments)

    assert canonical_row_credit(kernel_rows=5, convolution_stride=3)
    assert (filled.cycles, filled.fill_cycles, filled.drain_cycles) == (canonical.cycles, 0, 0)
    assert (filled.output_words, filled.max_live_rows, filled.buffer_rows) == (42, canonical.max_live_rows, 16)


def test_a_supertile_wider_than_one_input_word_fills_before_its_first_output():
    schedule = build_fill_cycle_schedule(
        input_rows=256, temporal_pack=8, kernel_rows=7, convolution_stride=2, pool_rows=2)

    assert not canonical_row_credit(kernel_rows=7, convolution_stride=2)
    assert (schedule.input_words, schedule.output_words) == (32, 62)
    assert (schedule.cycles, schedule.fill_cycles, schedule.drain_cycles) == (63, 1, 0)
    assert (schedule.max_live_rows, schedule.buffer_rows) == (16, 16)


@pytest.mark.parametrize("kernel_rows", range(1, 12))
@pytest.mark.parametrize("stride", range(1, 7))
@pytest.mark.parametrize("input_rows", (64, 128, 200))
def test_every_supertile_is_buffered_when_it_is_computed_and_every_word_is_read(kernel_rows, stride, input_rows):
    pack, pool_rows = 8, 2
    schedule = build_fill_cycle_schedule(input_rows=input_rows, temporal_pack=pack, kernel_rows=kernel_rows,
                                         convolution_stride=stride, pool_rows=pool_rows)
    convolution_rows = (input_rows - kernel_rows) // stride + 1
    assert schedule.output_words == convolution_rows // pool_rows
    assert schedule.buffer_rows % pack == 0

    # Replay the lowering's loop: at most one read and one output per cycle.
    words_read = produced = 0
    for _ in range(schedule.cycles):
        producing = produced < schedule.output_words
        start = produced * pool_rows * stride
        last = start + (pool_rows - 1) * stride + kernel_rows - 1
        required = last // pack + 1
        if words_read < schedule.input_words and (not producing or words_read < required):
            words_read += 1
        if producing and words_read >= required:
            assert last < words_read * pack
            assert start >= words_read * pack - schedule.buffer_rows
            produced += 1
    assert produced == schedule.output_words
    assert words_read == schedule.input_words
    assert schedule.cycles == schedule.output_words + schedule.fill_cycles + schedule.drain_cycles
