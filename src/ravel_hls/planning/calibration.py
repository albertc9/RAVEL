"""Pinned analytical coverage from the Aria 1.7.1 development experiments.

Coverage is a range of hardware facts, not a model identity. The envelope is a
conservative prediction at the requested clock, never a timing/fit certificate.
"""

from dataclasses import dataclass
from math import ceil, log2

from ..domain.temporal import TemporalBlock
from .windows import WindowSchedule


@dataclass(frozen=True)
class WindowCostProfile:
    id: str = "ku5p-captured-window-2023.2"
    version: int = 1

    def covers(self, block: TemporalBlock, schedule: WindowSchedule, part, clock) -> bool:
        weight = next(p for p in block.convolution.parameters if p.role == "weight")
        return (
            part == "xcku5p-ffvb676-2-e" and clock == 5
            and 1 <= schedule.positions <= 2 and 1 <= schedule.width <= 4
            and 1 <= schedule.channels <= 12 and 1 <= schedule.filters <= 12
            and 1 <= schedule.kernel_rows <= 5 and schedule.stride_rows >= 2
            and schedule.pool_rows == 2
            and schedule.input_rows <= 128
            and block.input.numeric_type.kind == "fixed"
            and block.input.numeric_type.width <= 10
            and weight.numeric_type.kind == "fixed" and weight.numeric_type.width <= 8
            and max(t.numeric_type.width for op in (block.convolution, block.activation, block.pooling)
                    for t in op.outputs) <= 24
        )

    def window_cycles(self, schedule: WindowSchedule) -> int:
        # Includes ordered MAC depth, local window capture, and process restart.
        return schedule.input_words + ceil(0.75 * schedule.kernel_rows * schedule.channels) + 9

    def bridge_cycles(self, bridge) -> int:
        bits = max(bridge.input.lanes, bridge.output.lanes) * bridge.input.numeric.width
        return bridge.cycles + 2 * ceil(log2(max(2, bits))) + 8

    def to_dict(self):
        return {"id": self.id, "version": self.version,
                "tool": "Vitis HLS 2023.2", "part": "xcku5p-ffvb676-2-e",
                "clock_period_ns": 5, "status": "predicted",
                "evidence": "references/qualification/aria_1_7_1_search/calibration.json"}


WINDOW_COST_PROFILE = WindowCostProfile()
