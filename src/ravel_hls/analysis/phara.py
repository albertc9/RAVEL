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


@dataclass(frozen=True)
class DaSupertileAnalysis:
    """Multiplierless realization of one pool-aligned supertile."""

    graph: AffineGraph
    proof: AffineProof
    summary: Mapping[str, int]


@dataclass(frozen=True)
class PoolAlignedSchedule:
    """Static production schedule for one pool-aligned temporal packing."""

    input_words: int
    output_words: int
    cycles: int
    output_after_input_words: tuple[int, ...]
    buffer_rows: int


def build_pool_aligned_schedule(
    *,
    input_rows: int,
    temporal_pack: int,
    kernel_rows: int,
    convolution_stride: int,
    pool_rows: int,
) -> PoolAlignedSchedule:
    """Derive the q1 production points for a valid convolution and pool."""

    input_words = input_rows // temporal_pack
    convolution_rows = (input_rows - kernel_rows) // convolution_stride + 1
    output_words = convolution_rows // pool_rows
    production = tuple(
        (
            pool_index * pool_rows * convolution_stride
            + (pool_rows - 1) * convolution_stride
            + kernel_rows
            - 1
        )
        // temporal_pack
        for pool_index in range(output_words)
    )
    return PoolAlignedSchedule(
        input_words=input_words,
        output_words=output_words,
        cycles=input_words,
        output_after_input_words=production,
        buffer_rows=kernel_rows + convolution_stride + temporal_pack,
    )


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
        elif node.operation == "negate":
            value = -operands[0]
        else:
            raise ValueError(f"Unsupported PHARA affine operation: {node.operation}")
        values[node.id] = value % graph.modulus
    return tuple(values[output_id] for output_id in graph.output_ids)


def prove_equivalent(
    graph: AffineGraph,
    reference_coefficients: tuple[tuple[int, ...], ...],
) -> AffineProof:
    """Prove output coefficient vectors equal the reference modulo the graph width."""

    has_affine_constant = any(
        node.operation == "constant" for node in graph.nodes
    ) or any(len(vector) == len(graph.input_ids) + 1 for vector in reference_coefficients)
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
        elif node.operation == "negate":
            vector = tuple(-coefficient for coefficient in operands[0])
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


def analyze_da_supertile(
    *,
    weight_codes: tuple[tuple[int, ...], ...],
    aligned_bias_codes: tuple[int, ...],
    convolution_stride: int,
    modulus: int,
) -> DaSupertileAnalysis:
    """Build and prove a CSD realization with deterministic product sharing."""

    kernel_rows = len(weight_codes)
    filter_lanes = len(aligned_bias_codes)
    convolution_rows = 2
    input_rows = kernel_rows + convolution_stride
    input_ids = tuple(f"x{row}" for row in range(input_rows))
    nodes: list[AffineNode] = []
    shifted_nodes: dict[tuple[str, int], str] = {}
    product_nodes: dict[tuple[str, int], str] = {}
    constant_nodes: dict[int, str] = {}
    pair_nodes: dict[tuple[str, str, str], str] = {}
    output_ids = []
    reference = []
    product_uses = 0
    shared_pair_uses = 0

    def combine(operation: str, left: str, right: str, node_id: str) -> str:
        nonlocal shared_pair_uses
        operands = tuple(sorted((left, right))) if operation == "add" else (left, right)
        key = (operation, *operands)
        if key in pair_nodes:
            shared_pair_uses += 1
            return pair_nodes[key]
        nodes.append(AffineNode(node_id, operation, operands))
        pair_nodes[key] = node_id
        return node_id

    def shifted(input_id: str, amount: int) -> str:
        if amount == 0:
            return input_id
        key = (input_id, amount)
        if key not in shifted_nodes:
            node_id = f"{input_id}_shift{amount}"
            nodes.append(
                AffineNode(
                    id=node_id,
                    operation="shift",
                    inputs=(input_id,),
                    value=amount,
                )
            )
            shifted_nodes[key] = node_id
        return shifted_nodes[key]

    def product(input_id: str, coefficient: int) -> str:
        key = (input_id, coefficient)
        if key in product_nodes:
            return product_nodes[key]
        digits = _canonical_signed_digits(coefficient)
        positive = [shifted(input_id, shift) for shift, sign in digits if sign > 0]
        negative = [shifted(input_id, shift) for shift, sign in digits if sign < 0]
        if positive:
            accumulator = positive[0]
            for ordinal, term in enumerate(positive[1:], start=1):
                node_id = f"{input_id}_c{coefficient}_positive{ordinal}"
                accumulator = combine("add", accumulator, term, node_id)
        else:
            accumulator = f"{input_id}_c{coefficient}_negate"
            nodes.append(AffineNode(accumulator, "negate", (negative[0],)))
            negative = negative[1:]
        for ordinal, term in enumerate(negative, start=1):
            node_id = f"{input_id}_c{coefficient}_negative{ordinal}"
            accumulator = combine("subtract", accumulator, term, node_id)
        product_nodes[key] = accumulator
        return accumulator

    def constant(value: int) -> str:
        if value not in constant_nodes:
            node_id = f"constant{len(constant_nodes)}"
            nodes.append(AffineNode(node_id, "constant", (), value))
            constant_nodes[value] = node_id
        return constant_nodes[value]

    expressions: list[list[str]] = []
    output_names = []
    for convolution_row in range(convolution_rows):
        input_offset = convolution_row * convolution_stride
        for filter_lane in range(filter_lanes):
            terms = []
            coefficients = [0] * input_rows
            for kernel_row in range(kernel_rows):
                coefficient = _centered_code(
                    weight_codes[kernel_row][filter_lane], modulus
                )
                input_row = input_offset + kernel_row
                coefficients[input_row] = coefficient
                if coefficient != 0:
                    product_uses += 1
                    terms.append(product(input_ids[input_row], coefficient))
            bias = _centered_code(aligned_bias_codes[filter_lane], modulus)
            if bias or not terms:
                terms.append(constant(bias))
            expressions.append(terms)
            output_names.append(f"y{convolution_row}_f{filter_lane}")
            reference.append((*coefficients, bias))

    cse_ordinal = 0
    while True:
        pair_occurrences: dict[tuple[str, str], set[int]] = {}
        for expression_index, terms in enumerate(expressions):
            for left_index in range(len(terms)):
                for right_index in range(left_index + 1, len(terms)):
                    pair = tuple(sorted((terms[left_index], terms[right_index])))
                    pair_occurrences.setdefault(pair, set()).add(expression_index)
        candidates = [
            (pair, indexes)
            for pair, indexes in pair_occurrences.items()
            if len(indexes) >= 2
        ]
        if not candidates:
            break
        pair, indexes = min(
            candidates,
            key=lambda candidate: (
                -(len(candidate[1]) - 1) * modulus.bit_length(),
                candidate[0],
            ),
        )
        node_id = f"cse{cse_ordinal}"
        cse_ordinal += 1
        key = ("add", *pair)
        if key in pair_nodes:
            shared = pair_nodes[key]
        else:
            nodes.append(AffineNode(node_id, "add", pair))
            pair_nodes[key] = node_id
            shared = node_id
        shared_pair_uses += len(indexes) - 1
        for expression_index in indexes:
            terms = expressions[expression_index]
            left_index = terms.index(pair[0])
            terms.pop(left_index)
            right_index = terms.index(pair[1])
            terms.pop(right_index)
            terms.append(shared)

    for output_name, terms in zip(output_names, expressions):
        accumulator = terms[0]
        for ordinal, term in enumerate(terms[1:], start=1):
            accumulator = combine(
                "add", accumulator, term, f"{output_name}_sum{ordinal}"
            )
        output_ids.append(accumulator)

    graph = AffineGraph(input_ids, tuple(nodes), tuple(output_ids), modulus)
    proof = prove_equivalent(graph, tuple(reference))
    operations = ("shift", "add", "subtract", "negate", "constant")
    operation_counts = {
        operation: sum(node.operation == operation for node in graph.nodes)
        for operation in operations
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
    for output_id in graph.output_ids:
        fanout[output_id] = fanout.get(output_id, 0) + 1
    return DaSupertileAnalysis(
        graph=graph,
        proof=proof,
        summary=MappingProxyType(
            {
                "input_rows": input_rows,
                "convolution_rows": convolution_rows,
                "filter_lanes": filter_lanes,
                "output_values": len(output_ids),
                "shift_nodes": operation_counts["shift"],
                "add_nodes": operation_counts["add"],
                "subtract_nodes": operation_counts["subtract"],
                "negate_nodes": operation_counts["negate"],
                "constant_nodes": operation_counts["constant"],
                "depth": max(depths[output_id] for output_id in graph.output_ids),
                "max_fanout": max(fanout.values()),
                "shared_product_uses": product_uses - len(product_nodes),
                "shared_pair_uses": shared_pair_uses,
            }
        ),
    )


def _canonical_signed_digits(value: int) -> tuple[tuple[int, int], ...]:
    sign = -1 if value < 0 else 1
    remaining = abs(value)
    digits = []
    shift = 0
    while remaining:
        if remaining & 1:
            digit = 2 - (remaining & 3)
            digits.append((shift, digit * sign))
            remaining -= digit
        remaining >>= 1
        shift += 1
    return tuple(digits)


def _centered_code(value: int, modulus: int) -> int:
    normalized = value % modulus
    return normalized - modulus if normalized >= modulus // 2 else normalized


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
