"""Geometry-derived schedules for independent-position temporal arithmetic."""

from dataclasses import asdict, dataclass

from ..domain.temporal import TemporalBlock


@dataclass(frozen=True)
class WindowSchedule:
    input_rows: int
    width: int
    channels: int
    filters: int
    kernel_rows: int
    stride_rows: int
    convolution_rows: int
    pool_rows: int
    output_rows: int
    positions: int

    @property
    def input_words(self) -> int:
        return self.input_rows * self.width // self.positions

    @property
    def products(self) -> int:
        return self.positions * self.kernel_rows * self.channels * self.filters

    def to_dict(self) -> dict:
        return {
            "id": "captured-window-positions",
            "version": 1,
            **asdict(self),
            "input_words": self.input_words,
            "arithmetic_order": "kernel-channel-filter",
            "storage": "partitioned-row-history-and-local-window",
            "target_loop_ii": 1,
            "drains_complete_frame": True,
        }


def window_schedules(block: TemporalBlock) -> tuple[WindowSchedule, ...]:
    """Only admit the geometry implemented by the registered window lowering."""
    conv, pool = block.convolution.attribute, block.pooling.attribute
    if any(operation.attribute(name, 0) for operation in (block.convolution, block.pooling)
           for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")):
        return ()
    if (conv("filt_width") != 1 or conv("stride_width") != 1
            or pool("pool_width") != 1 or pool("stride_width") != 1
            or pool("stride_height") != pool("pool_height")):
        return ()
    width = conv("in_width")
    return tuple(
        WindowSchedule(conv("in_height"), width, conv("n_chan"), conv("n_filt"),
                       conv("filt_height"), conv("stride_height"), conv("out_height"),
                       pool("pool_height"), pool("out_height"), positions)
        for positions in range(1, min(width, 4) + 1)
        if width % positions == 0
    )
