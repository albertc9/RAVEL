"""Typed repeated-block recognition over wiring, independent of strategies."""

from dataclasses import dataclass
from math import prod

from .graph import GraphFacts, OperationFacts, TensorFacts


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    operation_id: str | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message,
                "operation_id": self.operation_id, "severity": self.severity}


@dataclass(frozen=True)
class TemporalBlock:
    convolution: OperationFacts
    activation: OperationFacts
    pooling: OperationFacts
    input: TensorFacts


@dataclass(frozen=True)
class LayoutView:
    operation: OperationFacts
    input: TensorFacts
    output: TensorFacts

    def to_dict(self) -> dict[str, object]:
        return {"operation_id": self.operation.id, "input_shape": list(self.input.shape),
                "output_shape": list(self.output.shape), "order": "channels_last"}


@dataclass(frozen=True)
class TemporalChain:
    input: TensorFacts
    blocks: tuple[TemporalBlock, ...]
    layout: LayoutView
    head: OperationFacts

    def to_dict(self) -> dict[str, object]:
        return {"parser": {"id": "temporal-block-grammar", "version": 1},
                "block_count": len(self.blocks),
                "blocks": [{"convolution": b.convolution.id, "activation": b.activation.id,
                            "pooling": b.pooling.id} for b in self.blocks],
                "layout": self.layout.to_dict(), "head": self.head.id}


@dataclass(frozen=True)
class Recognition:
    chain: TemporalChain | None
    findings: tuple[Finding, ...] = ()


def recognize_temporal_chain(graph: GraphFacts) -> Recognition:
    """Recognize Input -> (Conv/ReLU/Pool)+ -> layout -> Dense1.

    Shape equations validate facts; no artifact extent or hardware packing is
    a grammar constant. Unsupported operations stay visible as local findings.
    """
    def failure(code, message, operation=None):
        return Recognition(None, (Finding(code, message, operation.id if operation else None),))

    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        return failure("family.topology.io", "Temporal family requires one input and one output")
    producers = {tensor.id: op for op in graph.operations for tensor in op.outputs}
    consumers: dict[str, list[OperationFacts]] = {}
    for op in graph.operations:
        for tensor in op.inputs:
            consumers.setdefault(tensor, []).append(op)
    first = producers.get(graph.inputs[0])
    if first is None or first.kind != "input" or first.inputs:
        return failure("family.topology.input", "Graph input must be an Input operation", first)
    ordered = [first]
    visited = {first.id}
    current = first
    while current.outputs and current.outputs[0].id != graph.outputs[0]:
        if len(current.outputs) != 1:
            return failure("family.topology.wiring", "Temporal operations require one output", current)
        successors = consumers.get(current.outputs[0].id, [])
        if len(successors) != 1:
            return failure("family.topology.wiring", "Temporal family requires a direct linear chain", current)
        successor = successors[0]
        if successor.id in visited or successor.inputs != (current.outputs[0].id,):
            return failure("family.topology.wiring", "Temporal chain contains a cycle, merge, or inconsistent port", successor)
        ordered.append(successor)
        visited.add(successor.id)
        current = successor
    if len(visited) != len(graph.operations) or len(current.outputs) != 1:
        return failure("family.topology.wiring", "Graph contains disconnected or extra operations", current)

    position = 1
    previous = first.outputs[0]
    if position < len(ordered) and ordered[position].kind == "repack":
        repack = ordered[position]
        shape = repack.outputs[0].shape
        if shape != (*previous.shape, 1) or not previous.numeric_type.preserves_codes_in(repack.outputs[0].numeric_type):
            return failure("family.geometry.repack", "Input layout must preserve codes and logical axes", repack)
        previous = repack.outputs[0]
        position += 1
    blocks = []
    while position < len(ordered) and ordered[position].kind == "conv2d":
        triple = ordered[position:position + 3]
        if tuple(op.kind for op in triple) != ("conv2d", "relu", "max_pool2d"):
            return failure("family.topology.sequence", "Temporal block requires Conv2D, explicit ReLU, and MaxPool2D", triple[0])
        convolution, activation, pooling = triple
        for op, input_tensor in ((convolution, previous), (pooling, activation.outputs[0])):
            findings = _window_geometry(op, input_tensor)
            if findings:
                return Recognition(None, findings)
        if activation.outputs[0].shape != convolution.outputs[0].shape:
            return failure("family.geometry.activation", "ReLU must preserve convolution shape", activation)
        blocks.append(TemporalBlock(convolution, activation, pooling, previous))
        previous = pooling.outputs[0]
        position += 3
    tail = ordered[position:]
    if not blocks or tuple(op.kind for op in tail) != ("reshape", "dense"):
        return failure("family.topology.sequence", "Temporal family requires repeated blocks followed by layout and Dense1", tail[0] if tail else current)
    layout, head = tail
    layout_output = layout.outputs[0]
    if layout_output.shape != (prod(previous.shape),) or not previous.numeric_type.preserves_codes_in(layout_output.numeric_type):
        return failure("family.geometry.layout", "Flatten must preserve feature order, shape product, and integer codes", layout)
    if head.attribute("n_in") != prod(layout_output.shape) or head.outputs[0].shape != (1,) or head.attribute("n_out") != 1:
        return failure("family.geometry.head", "Temporal family requires a single raw Dense output", head)
    return Recognition(TemporalChain(first.outputs[0], tuple(blocks), LayoutView(layout, previous, layout_output), head))


def _window_geometry(operation: OperationFacts, input_tensor: TensorFacts) -> tuple[Finding, ...]:
    attribute = operation.attribute
    convolution = operation.kind == "conv2d"
    prefix = "filt" if convolution else "pool"
    required = ("in_height", "in_width", "out_height", "out_width", "n_filt",
                f"{prefix}_height", f"{prefix}_width", "stride_height", "stride_width")
    if any(not isinstance(attribute(name), int) or attribute(name) <= 0 for name in required):
        return (Finding("family.geometry.window", "Window dimensions and strides must be positive", operation.id),)
    expected_input = (attribute("in_height"), attribute("in_width"), attribute("n_chan") if convolution else attribute("n_filt"))
    if input_tensor.shape != expected_input:
        return (Finding("family.geometry.relation", "Window input dimensions disagree with graph wiring", operation.id),)
    for axis, before, after in (("height", "top", "bottom"), ("width", "left", "right")):
        extent = (attribute(f"in_{axis}") + attribute(f"pad_{before}", 0) + attribute(f"pad_{after}", 0) - attribute(f"{prefix}_{axis}")) // attribute(f"stride_{axis}") + 1
        if extent != attribute(f"out_{axis}"):
            return (Finding("family.geometry.window", "Window output extent violates kernel/stride/padding algebra", operation.id),)
    expected_output = (attribute("out_height"), attribute("out_width"), attribute("n_filt"))
    if operation.outputs[0].shape != expected_output:
        return (Finding("family.geometry.relation", "Window output dimensions disagree with graph facts", operation.id),)
    return ()
