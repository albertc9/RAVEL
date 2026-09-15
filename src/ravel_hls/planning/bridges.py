"""Versioned lossless stream-layout bridges and their finite token events."""

from dataclasses import dataclass, field
from math import gcd
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .chain import StreamContract


@dataclass(frozen=True)
class Bridge:
    input: "StreamContract"
    output: "StreamContract"
    id: str = "lossless-stream-repack"
    version: int = 2

    @property
    def lanes_per_cycle(self) -> int:
        return gcd(self.input.lanes, self.output.lanes)

    @property
    def cycles(self) -> int:
        return (self.input.values + self.lanes_per_cycle - 1) // self.lanes_per_cycle

    @property
    def events(self) -> tuple[tuple[int, str], ...]:
        events = []
        for cycle, scalar in enumerate(range(0, self.input.values, self.lanes_per_cycle)):
            if scalar % self.input.lanes == 0:
                events.append((cycle, "consume"))
            if (scalar + self.lanes_per_cycle) % self.output.lanes == 0 or scalar + self.lanes_per_cycle >= self.input.values:
                events.append((cycle, "produce"))
        return tuple(events)

    def transfer(self, words: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        """Execute the contract on canonical codes, including final-word masks."""
        if len(words) != self.input.words or any(len(word) != self.input.lanes for word in words):
            raise ValueError("Input word count or packing disagrees with bridge contract")
        codes = tuple(code for word in words for code in word)[:self.input.values]
        return tuple(tuple(codes[start + lane] if start + lane < len(codes) else 0
                           for lane in range(self.output.lanes))
                     for start in range(0, len(codes), self.output.lanes))

    def to_dict(self) -> dict[str, object]:
        return {"strategy": {"id": self.id, "version": self.version},
                "input": self.input.to_dict(), "output": self.output.to_dict(),
                "schedule": {"id": "shared-lane-repack-events", "version": 2,
                             "cycles": self.cycles, "lanes_per_cycle": self.lanes_per_cycle,
                             "consume_every": self.input.lanes // self.lanes_per_cycle,
                             "produce_every": self.output.lanes // self.lanes_per_cycle, "drains_tail": True},
                "storage_scalars": self.input.lanes + self.output.lanes,
                "proof": "identical-code-layout-and-order"}


def bridge_for(source: "StreamContract", target: "StreamContract") -> Bridge | None:
    """Return the qualified bridge only when no arithmetic is required."""
    if (source.tensor_id != target.tensor_id or source.shape != target.shape
            or source.order != target.order or source.protocol != target.protocol
            or source.reset != target.reset or not source.numeric.preserves_codes_in(target.numeric)):
        return None
    return Bridge(source, target)


@dataclass(frozen=True)
class BridgeStrategy:
    id: str
    version: int
    evaluator: Callable[["StreamContract", "StreamContract"], Bridge | None] = field(compare=False, repr=False)

    def evaluate(self, source: "StreamContract", target: "StreamContract") -> Bridge | None:
        bridge = self.evaluator(source, target)
        if bridge is not None and (bridge.id, bridge.version) != (self.id, self.version):
            raise ValueError("Bridge capability returned a different registered identity")
        return bridge


LOSSLESS_BRIDGES = (BridgeStrategy("lossless-stream-repack", 2, bridge_for),)
