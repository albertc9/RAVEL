"""Immutable logical axes and reversible C-order scalar addressing."""

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class Axis:
    name: str
    extent: int

    def __post_init__(self):
        if not self.name or self.extent <= 0:
            raise ValueError("Logical axes require a name and positive extent")


@dataclass(frozen=True)
class FeatureLayout:
    axes: tuple[Axis, ...]

    @classmethod
    def temporal(cls, shape: tuple[int, ...]) -> "FeatureLayout":
        if len(shape) != 3:
            raise ValueError("Temporal layout requires temporal, feature and channel axes")
        return cls(tuple(Axis(name, size) for name, size in zip(("temporal", "feature", "channel"), shape)))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(prod(axis.extent for axis in self.axes[index + 1:]) for index in range(len(self.axes)))

    @property
    def scalar_count(self) -> int:
        return prod(axis.extent for axis in self.axes)

    def scalar_index(self, coordinates: tuple[int, ...]) -> int:
        if len(coordinates) != len(self.axes) or any(not 0 <= value < axis.extent for axis, value in zip(self.axes, coordinates)):
            raise ValueError("Coordinates are outside the logical layout")
        return sum(value * stride for value, stride in zip(coordinates, self.strides))

    def coordinates(self, scalar_index: int) -> tuple[int, ...]:
        if not 0 <= scalar_index < self.scalar_count:
            raise ValueError("Scalar index is outside the logical layout")
        return tuple(scalar_index // stride % axis.extent for axis, stride in zip(self.axes, self.strides))

    def to_dict(self) -> dict:
        return {"order": "C", "axes": [{"name": axis.name, "extent": axis.extent, "stride": stride}
                                        for axis, stride in zip(self.axes, self.strides)], "scalar_count": self.scalar_count}
