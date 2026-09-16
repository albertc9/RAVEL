"""Finite word-dependency traces; clock latency remains an explicit estimate."""

from dataclasses import dataclass
from fractions import Fraction
from math import prod

from ..domain.temporal import TemporalBlock

MAX_PRODUCTION_EVENTS = 4096


@dataclass(frozen=True)
class TokenSchedule:
    input_words: int
    output_words: int
    production_after_input_words: tuple[int, ...]

    def to_dict(self) -> dict:
        rate = Fraction(self.output_words, self.input_words)
        return {"id": "complete-window-dependencies", "version": 1,
                "timebase": "accepted-input-words-not-clock-cycles",
                "input_words": self.input_words, "output_words": self.output_words,
                "production_after_input_words": list(self.production_after_input_words),
                "fill_input_words": self.production_after_input_words[0],
                "drain_input_words": self.input_words - self.production_after_input_words[-1],
                "rational_rate": [rate.numerator, rate.denominator],
                "elastic_gaps": "arbitrary-with-eventual-progress",
                "consecutive_inference": "repeat-complete-frame",
                "fifo_depth_words": 4}


def temporal_schedule(block: TemporalBlock, input_lanes: int, output_lanes: int) -> TokenSchedule:
    conv, pool = block.convolution.attribute, block.pooling.attribute
    shape = block.pooling.outputs[0].shape
    output_words = (prod(shape) + output_lanes - 1) // output_lanes
    if output_words > MAX_PRODUCTION_EVENTS:
        raise ValueError("Complete production trace exceeds the declared 4096-event bound")
    events = []
    for word in range(output_words):
        scalar = min((word + 1) * output_lanes, prod(shape)) - 1
        output_row, output_column = scalar // (shape[1] * shape[2]), (scalar // shape[2]) % shape[1]
        last_conv_row = output_row * pool("stride_height") + pool("pool_height") - 1
        last_conv_column = output_column * pool("stride_width") + pool("pool_width") - 1
        row = last_conv_row * conv("stride_height") + conv("filt_height") - 1
        column = last_conv_column * conv("stride_width") + conv("filt_width") - 1
        scalar_count = (row * conv("in_width") + column + 1) * conv("n_chan")
        events.append((scalar_count + input_lanes - 1) // input_lanes)
    return TokenSchedule((prod(block.input.shape) + input_lanes - 1) // input_lanes, output_words, tuple(events))
