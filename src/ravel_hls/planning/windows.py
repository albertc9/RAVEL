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
    conv = block.convolution.attribute
    windows = tuple(operation for operation in (block.convolution, block.pooling) if operation is not None)
    if any(operation.attribute(name, 0) for operation in windows
           for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")):
        return ()
    if conv("filt_width") != 1 or conv("stride_width") != 1:
        return ()
    if block.pooling is not None:
        pool = block.pooling.attribute
        if (pool("pool_width") != 1 or pool("stride_width") != 1
                or pool("stride_height") != pool("pool_height")):
            return ()
    width = conv("in_width")
    return tuple(
        WindowSchedule(conv("in_height"), width, conv("n_chan"), conv("n_filt"),
                       conv("filt_height"), conv("stride_height"), conv("out_height"),
                       block.pool_rows, block.output_rows, positions)
        for positions in range(1, min(width, 4) + 1)
        if width % positions == 0
    )


@dataclass(frozen=True)
class DenseSchedule:
    """Sequential Dense with several outputs: one input group and some outputs per step."""

    inputs: int
    outputs: int
    mac_lanes: int
    output_lanes: int

    @property
    def steps(self) -> int:
        return self.inputs // self.mac_lanes * (self.outputs // self.output_lanes)

    @property
    def multipliers(self) -> int:
        return self.mac_lanes * self.output_lanes

    def to_dict(self) -> dict:
        return {"id": "matrix-sequential", "version": 1, **asdict(self),
                "steps": self.steps, "multipliers": self.multipliers,
                "accumulation": "ordered", "target_loop_ii": 1}


def dense_schedules(inputs: int, outputs: int, mac_lanes: int) -> tuple[DenseSchedule, ...]:
    """Every way to share the outputs evenly across steps."""
    if inputs % mac_lanes:
        return ()
    return tuple(DenseSchedule(inputs, outputs, mac_lanes, lanes)
                 for lanes in range(1, outputs + 1) if outputs % lanes == 0)


@dataclass(frozen=True)
class ArithmeticSchedule:
    """Share position engines across finite phases of an accepted stream word."""

    position_lanes: int
    engines: int

    @property
    def reuse_factor(self):
        return self.position_lanes // self.engines

    def to_dict(self):
        return {"id": "phased-position-engines", "version": 2,
                "position_lanes": self.position_lanes, "engines": self.engines,
                "reuse_factor": self.reuse_factor, "phase_loop_ii": 1,
                "extra_phases": "complete-convolution-windows-only"}


def arithmetic_schedules(positions):
    return tuple(ArithmeticSchedule(positions, engines) for engines in range(positions, 0, -1)
                 if positions % engines == 0)
