"""Project the compiler-owned ModelGraph into frozen facts and native bindings."""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any
from ..domain import ParameterPayload, ParameterTensor
from ..domain.graph import GraphFacts
def _semantic_kind(layer: Any) -> str:
    if layer.class_name == "Activation" and layer.get_attr("activation") == "relu":
        return "relu"
    if layer.class_name == "Pooling2D" and layer.get_attr("pool_op") == "Max":
        return "max_pool2d"
    names = {
        "Input": "input",
        "Repack": "repack",
        "Conv2D": "conv2d",
        "PointwiseConv2D": "conv2d",
        "Reshape": "reshape",
        "Dense": "dense",
    }
    return names.get(layer.class_name, layer.class_name.lower())

def _extract_parameter_payload(layers: list[Any]) -> ParameterPayload:
    tensors = []
    ordinals: dict[str, int] = {}
    for layer in layers:
        kind = _semantic_kind(layer)
        ordinal = ordinals.get(kind, 0)
        ordinals[kind] = ordinal + 1
        operation_id = f"{kind}_{ordinal}"
        for role, weight in layer.weights.items():
            tensors.append(
                ParameterTensor(
                    id=f"{operation_id}:{role}",
                    operation_id=operation_id,
                    role=role,
                    symbol=weight.name,
                    type_name=weight.type.name,
                    numeric_type=_numeric_type(weight.type.precision),
                    values=weight.data,
                )
            )
    return ParameterPayload(tuple(sorted(tensors, key=lambda tensor: tensor.id)))

def _native_rendering_contract(layers: list[Any]) -> dict[str, Any]:
    """Capture qualified native bindings before the renderer boundary."""
    result = {}
    ordinals = {}
    for layer in layers:
        kind = _semantic_kind(layer)
        ordinal = ordinals.get(kind, 0)
        ordinals[kind] = ordinal + 1
        output = layer.get_output_variable()
        result[f"{kind}_{ordinal}"] = {
            "output_symbol": output.name, "output_type": output.type.name,
            "output_precision_cpp": output.type.precision.definition_cpp(),
            "output_shape": [int(value) for value in output.shape],
            "native_call": layer.get_attr("function_cpp"),
            "config_symbol": f"config{layer.get_attr('index')}",
            "input_symbol": layer.get_input_variable().name if layer.inputs and kind != "input" else None,
            **({"accumulator_numeric": _numeric_type(layer.get_attr("accum_t").precision)}
               if kind == "conv2d" else {}),
        }
    return result

def _numeric_type(precision: Any) -> dict[str, Any]:
    rounding = getattr(precision, "rounding_mode", None)
    saturation = getattr(precision, "saturation_mode", None)
    return {
        "kind": "fixed" if hasattr(precision, "integer") else "integer",
        "width": int(precision.width),
        "integer": int(getattr(precision, "integer", precision.width)),
        "signed": bool(getattr(precision, "signed", True)),
        "rounding": str(rounding) if rounding is not None else None,
        "saturation": str(saturation) if saturation is not None else None,
        "saturation_bits": int(getattr(precision, "saturation_bits", 0)),
    }

def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value

def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

_OPERATION_ATTRIBUTES = {
    "input": ("input_shape",),
    "repack": ("target_shape",),
    "conv2d": (
        "in_height",
        "in_width",
        "n_chan",
        "filt_height",
        "filt_width",
        "n_filt",
        "stride_height",
        "stride_width",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "out_height",
        "out_width",
    ),
    "relu": ("activation", "n_in"),
    "max_pool2d": (
        "in_height",
        "in_width",
        "n_filt",
        "pool_height",
        "pool_width",
        "stride_height",
        "stride_width",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "pool_op",
        "out_height",
        "out_width",
    ),
    "reshape": ("target_shape",),
    "dense": ("n_in", "n_out"),
}



def _extract_model_facts(
    layers: list[Any], *, input_ports: tuple[str, ...], output_ports: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, str]]:
    import numpy as np

    operations: list[dict[str, Any]] = []
    raw_outputs: dict[str, str] = {}
    ordinals: dict[str, int] = {}
    parameter_identities: list[dict[str, Any]] = []
    for layer in layers:
        kind = _semantic_kind(layer)
        ordinal = ordinals.get(kind, 0)
        ordinals[kind] = ordinal + 1
        operation_id = f"{kind}_{ordinal}"
        inputs = [] if kind == "input" else [raw_outputs.get(name, f"unresolved:port{port}") for port, name in enumerate(layer.inputs)]
        outputs = []
        for port, raw_name in enumerate(layer.outputs):
            variable = layer.get_output_variable(raw_name)
            tensor_id = f"{operation_id}:out{port}"
            raw_outputs[raw_name] = tensor_id
            outputs.append(
                {
                    "id": tensor_id,
                    "shape": [int(dimension) for dimension in variable.shape],
                    "numeric_type": _numeric_type(variable.type.precision),
                }
            )
        parameters = []
        for role, weight in layer.weights.items():
            values = np.ascontiguousarray(weight.data)
            numeric_type = _numeric_type(weight.type.precision)
            fractional = numeric_type["width"] - numeric_type["integer"]
            codes = np.rint(values * (2**fractional)).astype("<i8", copy=False)
            content = hashlib.sha256(codes.tobytes(order="C")).hexdigest()
            descriptor = {
                "role": role,
                "shape": [int(dimension) for dimension in values.shape],
                "numeric_type": numeric_type,
                "content_sha256": content,
            }
            parameters.append(descriptor)
            parameter_identities.append(
                {
                    "operation_id": operation_id,
                    **descriptor,
                }
            )
        attributes = {
            name: _json_value(layer.get_attr(name))
            for name in _OPERATION_ATTRIBUTES.get(kind, ())
            if layer.get_attr(name) is not None
        }
        operations.append(
            {
                "id": operation_id,
                "kind": kind,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": attributes,
                "parameters": parameters,
            }
        )

    facts = {
        "schema_version": 1,
        "inputs": [raw_outputs[name] for name in input_ports],
        "outputs": [raw_outputs[name] for name in output_ports],
        "operations": operations,
    }
    structure = deepcopy(facts)
    for operation in structure["operations"]:
        for parameter in operation["parameters"]:
            parameter.pop("content_sha256")
    return facts, {
        "model_structure_sha256": _canonical_sha256(structure),
        "parameter_state_sha256": _canonical_sha256(
            sorted(
                parameter_identities,
                key=lambda parameter: (
                    parameter["operation_id"],
                    parameter["role"],
                ),
            )
        ),
    }


def ordered_layers(graph) -> list[Any]:
    """Traverse declared graph ports and their dependencies, independent of names/list order."""
    nodes = list(graph.get_layers())
    by_name = {layer.name: layer for layer in nodes}
    producers = {output: layer for layer in nodes for output in layer.outputs}
    ordered, active, visited = [], set(), set()
    def visit(layer):
        if id(layer) in visited:
            return
        if id(layer) in active:
            from ..exceptions import CompatibilityError
            raise CompatibilityError("Compiler graph contains a directed cycle")
        active.add(id(layer))
        for name in layer.inputs:
            if name in producers and producers[name] is not layer:
                visit(producers[name])
        active.remove(id(layer))
        visited.add(id(layer))
        ordered.append(layer)
    for name in graph.inputs:
        visit(by_name[name])
    for output in graph.outputs:
        visit(producers[output])
    # Disconnected opaque compiler nodes remain visible to family diagnostics.
    for layer in sorted(nodes, key=lambda item: (_semantic_kind(item), tuple(tuple(item.get_output_variable(name).shape) for name in item.outputs))):
        visit(layer)
    return ordered
