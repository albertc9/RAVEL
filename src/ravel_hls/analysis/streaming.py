"""Static schedule analysis for Aria streaming implementations."""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TwoRowConvolutionSchedule:
    """Emission schedule for valid convolution over two-row input words."""

    input_words: int
    output_rows: int
    window_end_rows: tuple[int, ...]
    input_word_indices: tuple[int, ...]
    input_word_lanes: tuple[int, ...]
    maximum_outputs_per_input_word: int


def build_two_row_convolution_schedule(
    *, input_rows: int, kernel_rows: int, stride_rows: int
) -> TwoRowConvolutionSchedule:
    """Derive which input row completes each valid convolution window."""

    window_end_rows = tuple(range(kernel_rows - 1, input_rows, stride_rows))
    input_word_indices = tuple(end_row // 2 for end_row in window_end_rows)
    input_word_lanes = tuple(end_row % 2 for end_row in window_end_rows)
    emissions_per_word = Counter(input_word_indices)
    return TwoRowConvolutionSchedule(
        input_words=(input_rows + 1) // 2,
        output_rows=len(window_end_rows),
        window_end_rows=window_end_rows,
        input_word_indices=input_word_indices,
        input_word_lanes=input_word_lanes,
        maximum_outputs_per_input_word=max(emissions_per_word.values(), default=0),
    )
