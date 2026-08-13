"""Exact modular arithmetic used by PHARA analysis."""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from collections.abc import Mapping
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


@dataclass(frozen=True)
class DirectSupertileAnalysis:
    """Direct constant-multiply realization of one pool-aligned supertile."""

    graph: AffineGraph
    proof: AffineProof
    summary: Mapping[str, int]


def evaluate(graph: AffineGraph, input_codes: tuple[int, ...]) -> tuple[int, ...]:
    """Evaluate one PHARA graph over integer codes with explicit wrapping."""

    values = {
        input_id: value % graph.modulus
        for input_id, value in zip(graph.input_ids, input_codes)
    }
    for node in graph.nodes:
        operands = tuple(values[input_id] for input_id in node.inputs)
        if node.operation == "constant" and node.value is not None:
            value = node.value
        elif node.operation == "multiply" and node.value is not None:
            value = operands[0] * node.value
        elif node.operation == "shift" and node.value is not None:
            value = operands[0] << node.value
        elif node.operation == "add":
            value = operands[0] + operands[1]
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

    has_affine_constant = any(node.operation == "constant" for node in graph.nodes)
    width = len(graph.input_ids) + int(has_affine_constant)
    vectors: dict[str, tuple[int, ...]] = {
        input_id: tuple(
            1 if index == input_index else 0 for index in range(width)
        )
        for input_index, input_id in enumerate(graph.input_ids)
    }
    for node in graph.nodes:
        operands = tuple(vectors[input_id] for input_id in node.inputs)
        if node.operation == "constant" and node.value is not None:
            vector = tuple(
                node.value if index == width - 1 else 0 for index in range(width)
            )
        elif node.operation == "multiply" and node.value is not None:
            vector = tuple(coefficient * node.value for coefficient in operands[0])
        elif node.operation == "shift" and node.value is not None:
            vector = tuple(coefficient << node.value for coefficient in operands[0])
        elif node.operation == "add":
            vector = tuple(left + right for left, right in zip(*operands))
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


def analyze_direct_supertile(
    *,
    weight_codes: tuple[tuple[int, ...], ...],
    aligned_bias_codes: tuple[int, ...],
    convolution_stride: int,
    modulus: int,
) -> DirectSupertileAnalysis:
    """Build and prove the direct graph for two convolution rows pooled together."""

    kernel_rows = len(weight_codes)
    filter_lanes = len(aligned_bias_codes)
    convolution_rows = 2
    input_rows = kernel_rows + convolution_stride
    input_ids = tuple(f"x{row}" for row in range(input_rows))
    nodes: list[AffineNode] = []
    output_ids = []
    reference = []
    for convolution_row in range(convolution_rows):
        input_offset = convolution_row * convolution_stride
        for filter_lane in range(filter_lanes):
            prefix = f"y{convolution_row}_f{filter_lane}"
            accumulator = f"{prefix}_bias"
            nodes.append(
                AffineNode(
                    id=accumulator,
                    operation="constant",
                    inputs=(),
                    value=aligned_bias_codes[filter_lane],
                )
            )
            coefficients = [0] * input_rows
            for kernel_row in range(kernel_rows):
                coefficient = weight_codes[kernel_row][filter_lane]
                input_row = input_offset + kernel_row
                product = f"{prefix}_product{kernel_row}"
                nodes.append(
                    AffineNode(
                        id=product,
                        operation="multiply",
                        inputs=(input_ids[input_row],),
                        value=coefficient,
                    )
                )
                next_accumulator = f"{prefix}_sum{kernel_row}"
                nodes.append(
                    AffineNode(
                        id=next_accumulator,
                        operation="add",
                        inputs=(accumulator, product),
                    )
                )
                accumulator = next_accumulator
                coefficients[input_row] = coefficient
            output_ids.append(accumulator)
            reference.append((*coefficients, aligned_bias_codes[filter_lane]))

    graph = AffineGraph(
        input_ids=input_ids,
        nodes=tuple(nodes),
        output_ids=tuple(output_ids),
        modulus=modulus,
    )
    proof = prove_equivalent(graph, tuple(reference))
    operation_counts = {
        operation: sum(node.operation == operation for node in graph.nodes)
        for operation in ("constant", "multiply", "add")
    }
    depths = {input_id: 0 for input_id in graph.input_ids}
    fanout = {input_id: 0 for input_id in graph.input_ids}
    for node in graph.nodes:
        depths[node.id] = (
            0 if node.operation == "constant" else 1 + max(depths[i] for i in node.inputs)
        )
        fanout.setdefault(node.id, 0)
        for input_id in node.inputs:
            fanout[input_id] = fanout.get(input_id, 0) + 1
    return DirectSupertileAnalysis(
        graph=graph,
        proof=proof,
        summary=MappingProxyType(
            {
                "input_rows": input_rows,
                "convolution_rows": convolution_rows,
                "filter_lanes": filter_lanes,
                "output_values": len(output_ids),
                "constant_nodes": operation_counts["constant"],
                "multiply_nodes": operation_counts["multiply"],
                "add_nodes": operation_counts["add"],
                "depth": max(depths[output_id] for output_id in graph.output_ids),
                "max_fanout": max(fanout.values()),
            }
        ),
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
