from ravel_hls.analysis.phara import (
    build_pool_aligned_schedule,
    build_row_credit_schedule,
)


def test_phara_p4_schedule_emits_every_pool_group_while_reading_64_words() -> None:
    schedule = build_pool_aligned_schedule(
        input_rows=256,
        temporal_pack=4,
        kernel_rows=5,
        convolution_stride=3,
        pool_rows=2,
    )

    assert schedule.input_words == 64
    assert schedule.output_words == 42
    assert schedule.cycles == 64
    assert schedule.output_after_input_words[:6] == (1, 3, 4, 6, 7, 9)
    assert schedule.output_after_input_words[-1] == 63
    assert schedule.buffer_rows == 12


def test_phara_p8_credit_schedule_emits_continuously_and_reads_32_words() -> None:
    schedule = build_row_credit_schedule(
        input_rows=256,
        temporal_pack=8,
        kernel_rows=5,
        convolution_stride=3,
        pool_rows=2,
    )

    assert schedule.input_words == 32
    assert schedule.output_words == 42
    assert schedule.cycles == 42
    assert schedule.read_on_output[:12] == (
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    )
    assert sum(schedule.read_on_output) == 32
    assert schedule.read_on_output[-1] is True
    assert schedule.max_live_rows == 14
    assert schedule.buffer_rows == 16
