"""Immutable, framework-independent projection of compiler graph semantics.

These values describe an existing graph. They cannot edit or lower it. Mapping
conversion is explicit and exists only at the extraction/serialization seams.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, TypeAlias


Scalar: TypeAlias = str | int | float | bool | None
AttributeValue: TypeAlias = Scalar | tuple["AttributeValue", ...]


def _value(value: object) -> AttributeValue:
    if isinstance(value, (tuple, list)):
        return tuple(_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported graph attribute value: {type(value).__name__}")


def _serialized(value: AttributeValue) -> object:
    return [_serialized(item) for item in value] if isinstance(value, tuple) else value


@dataclass(frozen=True)
class NumericType:
    kind: str
    width: int
    integer: int
    signed: bool
    rounding: str | None
    saturation: str | None
    saturation_bits: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def preserves_codes_in(self, target: NumericType) -> bool:
        """Whether an exact representable value survives a same-layout cast.

        Rounding cannot affect exact values. Symmetric saturation can still
        clip the most negative two's-complement code and must be respected.
        """
        return ((self.width, self.integer, self.signed) ==
                (target.width, target.integer, target.signed)
                and (not self.signed or target.saturation != "SAT_SYM"
                     or self.saturation == "SAT_SYM"))


@dataclass(frozen=True)
class TensorFacts:
    id: str
    shape: tuple[int, ...]
    numeric_type: NumericType

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorFacts:
        return cls(str(value["id"]), tuple(value["shape"]), NumericType(**value["numeric_type"]))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "shape": list(self.shape), "numeric_type": self.numeric_type.to_dict()}


@dataclass(frozen=True)
class ParameterFacts:
    role: str
    shape: tuple[int, ...]
    numeric_type: NumericType
    content_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ParameterFacts:
        return cls(str(value["role"]), tuple(value["shape"]),
                   NumericType(**value["numeric_type"]), str(value["content_sha256"]))

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "shape": list(self.shape),
                "numeric_type": self.numeric_type.to_dict(), "content_sha256": self.content_sha256}


@dataclass(frozen=True)
class Attribute:
    name: str
    value: AttributeValue


@dataclass(frozen=True)
class OperationFacts:
    id: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[TensorFacts, ...]
    attributes: tuple[Attribute, ...]
    parameters: tuple[ParameterFacts, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OperationFacts:
        return cls(
            str(value["id"]), str(value["kind"]), tuple(value["inputs"]),
            tuple(TensorFacts.from_dict(item) for item in value["outputs"]),
            tuple(Attribute(name, _value(item)) for name, item in sorted(value["attributes"].items())),
            tuple(ParameterFacts.from_dict(item) for item in value["parameters"]),
        )

    def attribute(self, name: str, default: AttributeValue = None) -> AttributeValue:
        return next((item.value for item in self.attributes if item.name == name), default)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, "inputs": list(self.inputs),
                "outputs": [item.to_dict() for item in self.outputs],
                "attributes": {item.name: _serialized(item.value) for item in self.attributes},
                "parameters": [item.to_dict() for item in self.parameters]}


@dataclass(frozen=True)
class GraphFacts:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operations: tuple[OperationFacts, ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GraphFacts:
        return cls(tuple(value["inputs"]), tuple(value["outputs"]),
                   tuple(OperationFacts.from_dict(item) for item in value["operations"]),
                   int(value.get("schema_version", 1)))

    def operation(self, operation_id: str) -> OperationFacts:
        for operation in self.operations:
            if operation.id == operation_id:
                return operation
        raise KeyError(operation_id)

    def tensor(self, tensor_id: str) -> TensorFacts:
        for operation in self.operations:
            for output in operation.outputs:
                if output.id == tensor_id:
                    return output
        raise KeyError(tensor_id)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "inputs": list(self.inputs),
                "outputs": list(self.outputs),
                "operations": [operation.to_dict() for operation in self.operations]}
