"""Versioned lossless stream-layout bridges and their finite token events."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chain import StreamContract


@dataclass(frozen=True)
class Bridge:
    input: "StreamContract"
    output: "StreamContract"
    id: str = "lossless-stream-repack"
    version: int = 1

    @property
    def events(self) -> tuple[tuple[int, str], ...]:
        events = []
        for scalar in range(self.input.values):
            if scalar % self.input.lanes == 0:
                events.append((scalar, "consume"))
            if scalar % self.output.lanes == self.output.lanes - 1 or scalar == self.input.values - 1:
                events.append((scalar, "produce"))
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
                "schedule": {"id": "scalar-repack-events", "version": 1,
                             "cycles": self.input.values, "consume_every": self.input.lanes,
                             "produce_every": self.output.lanes, "drains_tail": True},
                "storage_scalars": self.input.lanes + self.output.lanes,
                "proof": "identical-code-layout-and-order"}


def bridge_for(source: "StreamContract", target: "StreamContract") -> Bridge | None:
    """Return the qualified bridge only when no arithmetic is required."""
    if (source.tensor_id != target.tensor_id or source.shape != target.shape
            or source.order != target.order or source.protocol != target.protocol
            or source.reset != target.reset or not source.numeric.preserves_codes_in(target.numeric)):
        return None
    return Bridge(source, target)
