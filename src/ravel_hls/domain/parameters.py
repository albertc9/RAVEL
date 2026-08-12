"""Immutable hardware-authoritative parameter payload values."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ParameterTensor:
    """One read-only ModelGraph inference tensor addressed semantically."""

    id: str
    operation_id: str
    role: str
    symbol: str
    type_name: str
    numeric_type: Mapping[str, Any]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True, order="C")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "numeric_type", MappingProxyType(dict(self.numeric_type)))


@dataclass(frozen=True)
class ParameterPayload:
    """Ordered read-only parameter tensors extracted from one ModelGraph."""

    tensors: tuple[ParameterTensor, ...]

    def by_id(self) -> Mapping[str, ParameterTensor]:
        return MappingProxyType({tensor.id: tensor for tensor in self.tensors})
