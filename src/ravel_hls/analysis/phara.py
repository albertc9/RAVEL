"""Exact modular arithmetic used by PHARA analysis."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class AffineNode:
    """One immutable operation in a PHARA affine graph."""

    id: str
    operation: str
    inputs: tuple[str, ...]
    value: int | None = None


@dataclass(frozen=True)
class AffineGraph:
    """An affine integer-code graph evaluated modulo ``modulus``."""

    input_ids: tuple[str, ...]
    nodes: tuple[AffineNode, ...]
    output_ids: tuple[str, ...]
    modulus: int


@dataclass(frozen=True)
class AffineProof:
    """Result of comparing a graph with reference coefficient vectors."""

    status: str
    modulus: int
    reference_coefficients: tuple[tuple[int, ...], ...]
    realized_coefficients: tuple[tuple[int, ...], ...]
    graph_sha256: str
    identity: str


def evaluate(graph: AffineGraph, input_codes: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate one PHARA graph over integer codes with explicit wrapping."""

    values = {
        input_id: value % graph.modulus
        for input_id, value in zip(graph.input_ids, input_codes)
    }
    for node in graph.nodes:
        operands = tuple(values[input_id] for input_id in node.inputs)
        if node.operation == "shift" and node.value is not None:
            value = operands[0] << node.value
        elif node.operation == "subtract":
            value = operands[0] - operands[1]
        else:
            raise ValueError(f"Unsupported PHARA affine operation: {node.operation}")
        values[node.id] = value % graph.modulus
    return tuple(values[output_id] for output_id in graph.output_ids)


def prove_equivalent(
    graph: AffineGraph,
    reference_coefficients: tuple[tuple[int, ...], ...],
) -> AffineProof:
    """Prove output coefficient vectors equal the reference modulo the graph width."""

    width = len(graph.input_ids)
    vectors: dict[str, tuple[int, ...]] = {
        input_id: tuple(
            1 if index == input_index else 0 for index in range(width)
        )
        for input_index, input_id in enumerate(graph.input_ids)
    }
    for node in graph.nodes:
        operands = tuple(vectors[input_id] for input_id in node.inputs)
        if node.operation == "shift" and node.value is not None:
            vector = tuple(coefficient << node.value for coefficient in operands[0])
        elif node.operation == "subtract":
            vector = tuple(left - right for left, right in zip(*operands))
        else:
            raise ValueError(f"Unsupported PHARA affine operation: {node.operation}")
        vectors[node.id] = tuple(value % graph.modulus for value in vector)

    realized = tuple(vectors[output_id] for output_id in graph.output_ids)
    reference = tuple(
        tuple(value % graph.modulus for value in coefficients)
        for coefficients in reference_coefficients
    )
    status = "proven" if realized == reference else "rejected"
    graph_sha256 = _canonical_sha256(
        {
            "input_ids": list(graph.input_ids),
            "modulus": graph.modulus,
            "nodes": [
                {
                    "id": node.id,
                    "inputs": list(node.inputs),
                    "operation": node.operation,
                    "value": node.value,
                }
                for node in graph.nodes
            ],
            "output_ids": list(graph.output_ids),
        }
    )
    proof_statement = {
        "graph_sha256": graph_sha256,
        "modulus": graph.modulus,
        "realized_coefficients": [list(vector) for vector in realized],
        "reference_coefficients": [list(vector) for vector in reference],
        "status": status,
    }
    return AffineProof(
        status=status,
        modulus=graph.modulus,
        reference_coefficients=reference,
        realized_coefficients=realized,
        graph_sha256=graph_sha256,
        identity=_canonical_sha256(proof_statement),
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
