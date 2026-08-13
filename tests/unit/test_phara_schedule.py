from ravel_hls.analysis.phara import build_pool_aligned_schedule


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
