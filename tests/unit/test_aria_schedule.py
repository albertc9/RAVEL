import pytest

from ravel_hls.analysis.streaming import build_two_row_convolution_schedule


@pytest.mark.parametrize("kernel_rows", range(3, 9))
@pytest.mark.parametrize("stride_rows", range(2, 6))
def test_two_row_schedule_emits_every_valid_convolution_window_once(
    kernel_rows: int, stride_rows: int,
) -> None:
    schedule = build_two_row_convolution_schedule(
        input_rows=32,
        kernel_rows=kernel_rows,
        stride_rows=stride_rows,
    )
    expected_window_ends = tuple(range(kernel_rows - 1, 32, stride_rows))

    assert schedule.window_end_rows == expected_window_ends
    assert schedule.input_word_indices == tuple(
        end_row // 2 for end_row in expected_window_ends
    )
    assert schedule.input_word_lanes == tuple(
        end_row % 2 for end_row in expected_window_ends
    )
    assert schedule.output_rows == (32 - kernel_rows) // stride_rows + 1
    assert schedule.maximum_outputs_per_input_word == 1
