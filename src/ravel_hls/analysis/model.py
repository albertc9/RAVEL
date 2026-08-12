"""Side-effect-free public model analysis."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..analysis.dense import analyze_dense_facts
from ..compatibility.dependencies import inspect_dependencies
from ..domain import ParameterPayload, ParameterTensor
from ..exceptions import CompatibilityError, ConfigurationError
from ..generations.aria import resolve_aria_design
from ..profiles.aria.plan import build_implementation_plan


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class ModelAnalysis:
    """Stable read-only result returned by :func:`analyze`."""

    _report: Mapping[str, Any]

    @classmethod
    def _from_report(cls, report: Mapping[str, Any]) -> "ModelAnalysis":
        return cls(_freeze(report))

    @property
    def applicable(self) -> bool:
        """Whether one model family and implementation design are applicable."""

        return self._report["applicability"]["status"] == "applicable"

    @property
    def model_family(self) -> Mapping[str, Any] | None:
        """The uniquely matched model family, if any."""

        return self._report["model_family"]

    @property
    def findings(self) -> tuple[Mapping[str, Any], ...]:
        """Structured applicability findings."""

        return self._report["applicability"]["findings"]

    @property
    def model_facts(self) -> Mapping[str, Any]:
        """Immutable normalized hardware-semantic model facts."""

        return self._report["model_facts"]

    @property
    def resolved_design(self) -> Mapping[str, Any] | None:
        """The immutable render-ready design when analysis is applicable."""

        return self._report["resolved_design"]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent schema representation."""

        return _thaw(self._report)


@dataclass(frozen=True)
class _AnalyzedModel:
    graph: Any
    source_model: Any
    analysis: ModelAnalysis
    parameter_payload: ParameterPayload


def analyze(model: Any, config: Mapping[str, Any]) -> ModelAnalysis:
    """Analyze a qualified Keras/HGQ2 model without publishing a project."""

    return _analyze_model(model, config).analysis


def _analyze_model(model: Any, config: Mapping[str, Any]) -> _AnalyzedModel:
    """Return the private graph-bearing analysis used by conversion."""

    hls_values = config.get("HLS")
    if not isinstance(hls_values, Mapping):
        raise ConfigurationError("HLS must be a mapping")
    backend = hls_values.get("Backend", "Vitis")
    io_type = hls_values.get("IOType", "io_stream")
    if backend != "Vitis":
        raise ConfigurationError("HLS.Backend must be Vitis")
    if io_type != "io_stream":
        raise ConfigurationError("HLS.IOType must be io_stream")

    optimization = config.get("Optimization", {})
    if not isinstance(optimization, Mapping):
        raise ConfigurationError("Optimization must be a mapping")
    unknown_optimization = sorted(
        set(optimization) - {"TemporalPacking", "DenseParallelism"}
    )
    if unknown_optimization:
        raise ConfigurationError(
            f"Unknown optimization field: {unknown_optimization[0]}"
        )
    choices = {
        "TemporalPacking": optimization.get("TemporalPacking", 4),
        "DenseParallelism": optimization.get("DenseParallelism", 2),
    }
    for field, allowed in (
        ("TemporalPacking", {2, 4}),
        ("DenseParallelism", {1, 2}),
    ):
        value = choices[field]
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            options = ", ".join(str(item) for item in sorted(allowed))
            raise ConfigurationError(f"Optimization.{field} must be one of {options}")

    dependencies = inspect_dependencies()
    if dependencies["dependency_qualification"] != "qualified":
        failed = [
            name
            for name, facts in dependencies["dependencies"].items()
            if facts["status"] != "qualified"
        ]
        raise CompatibilityError(
            "Aria 1.5.0 dependency stack is not qualified: " + ", ".join(failed)
        )

    import hls4ml

    normalized_model = model
    if isinstance(model, (str, os.PathLike)):
        import keras
        from hgq.layers import QConv2D, QDense

        model_path = Path(model)
        if not model_path.is_file():
            raise CompatibilityError(f"Keras model file does not exist: {model_path}")
        normalized_model = keras.models.load_model(
            model_path,
            custom_objects={"QConv2D": QConv2D, "QDense": QDense},
        )

    frontend_provenance = _frontend_provenance(normalized_model)

    hls_config = hls4ml.utils.config_from_keras_model(
        normalized_model, granularity="name", backend=backend
    )
    hls_config["Model"].update({"Strategy": "Latency", "ReuseFactor": 1})
    conversion: dict[str, Any] = {
        "model": normalized_model,
        "output_dir": str(Path.cwd() / "ravel_analysis"),
        "project_name": "ravel_analysis",
        "hls_config": hls_config,
        "backend": backend,
        "io_type": io_type,
    }
    if hls_values.get("Part") is not None:
        conversion["part"] = hls_values["Part"]
    if hls_values.get("ClockPeriod") is not None:
        conversion["clock_period"] = hls_values["ClockPeriod"]
    graph = hls4ml.converters.convert_from_keras_model(**conversion)
    layers = list(graph.get_layers())
    model_facts, fingerprints = _extract_model_facts(layers)
    parameter_payload = _extract_parameter_payload(layers)
    fingerprints["frontend_provenance_sha256"] = _canonical_sha256(
        frontend_provenance
    )
    model_family, applicability = _match_model_family(
        model_facts, frontend_provenance
    )

    dense_facts: dict[str, Any] = {}
    resolved_design = None
    if model_family is not None:
        dense_facts = {"dense": analyze_dense_facts(layers)}
        plan = build_implementation_plan(choices, {**model_facts, **dense_facts})
        strategy_findings = _strategy_geometry_findings(
            model_facts["operations"], choices, plan
        )
        if strategy_findings:
            applicability = {
                "status": "unsupported",
                "findings": strategy_findings,
            }
        else:
            interface = _predicted_interface(model_facts, plan)
            resolved_design = resolve_aria_design(
                model_facts=model_facts,
                implementation_plan=plan,
                interfaces=interface,
                parameter_bindings=_parameter_bindings(
                    model_facts, parameter_payload
                ),
                rendering=_rendering_contract(layers, plan),
            )
    analysis = ModelAnalysis._from_report(
        {
            "schema_version": 1,
            "generation": {"id": "aria", "version": "1.5.0"},
            "model_family": model_family,
            "applicability": applicability,
            "frontend_provenance": frontend_provenance,
            "model_facts": {**model_facts, **dense_facts},
            "resolved_design": resolved_design,
            "fingerprints": fingerprints,
        }
    )
    return _AnalyzedModel(graph, normalized_model, analysis, parameter_payload)


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


_CANONICAL_SEQUENCE = (
    "input",
    "repack",
    "conv2d",
    "relu",
    "max_pool2d",
    "reshape",
    "dense",
)

def _match_model_family(
    facts: Mapping[str, Any], provenance: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    operations = facts["operations"]
    observed_sequence = tuple(operation["kind"] for operation in operations)
    findings: list[dict[str, Any]] = []
    if observed_sequence != _CANONICAL_SEQUENCE:
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_sequence, _CANONICAL_SEQUENCE)
                )
                if observed != expected
            ),
            min(len(observed_sequence), len(_CANONICAL_SEQUENCE)),
        )
        findings.append(
            {
                "code": "family.topology.sequence",
                "severity": "error",
                "operation_id": (
                    operations[mismatch]["id"] if mismatch < len(operations) else None
                ),
                "expected": list(_CANONICAL_SEQUENCE),
                "observed": list(observed_sequence),
                "message": "Model operation sequence is outside this family",
            }
        )
    else:
        findings.extend(_family_geometry_findings(operations))
        for previous, current in zip(operations, operations[1:]):
            if current["inputs"] != [previous["outputs"][0]["id"]]:
                findings.append(
                    {
                        "code": "family.topology.wiring",
                        "severity": "error",
                        "operation_id": current["id"],
                        "expected": [previous["outputs"][0]["id"]],
                        "observed": current["inputs"],
                        "message": "Model family requires a direct linear chain",
                    }
                )
        source_layers = provenance["source_layers"]
        for class_name in ("QConv2D", "QDense"):
            matching_sources = [
                source
                for source in source_layers
                if source["class_name"] == class_name
                and source["module"].startswith("hgq.layers")
            ]
            if len(matching_sources) != 1:
                findings.append(
                    {
                        "code": "family.provenance.hgq2",
                        "severity": "error",
                        "operation_id": "conv2d_0" if class_name == "QConv2D" else "dense_0",
                        "expected": f"one hgq.layers {class_name}",
                        "observed": len(matching_sources),
                        "message": "Model family requires HGQ2 Conv2D and Dense origins",
                    }
                )
    if findings:
        return None, {"status": "unsupported", "findings": findings}
    return (
        {"id": "hgq-conv-pool-dense", "version": 1},
        {"status": "applicable", "findings": []},
    )


def _family_geometry_findings(
    operations: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate cross-operation geometry without fixing one trained instance."""

    by_id = {operation["id"]: operation for operation in operations}
    input_shape = by_id["input_0"]["outputs"][0]["shape"]
    repack_shape = by_id["repack_0"]["outputs"][0]["shape"]
    convolution_shape = by_id["conv2d_0"]["outputs"][0]["shape"]
    activation_shape = by_id["relu_0"]["outputs"][0]["shape"]
    pooling_shape = by_id["max_pool2d_0"]["outputs"][0]["shape"]
    reshape_shape = by_id["reshape_0"]["outputs"][0]["shape"]
    output_shape = by_id["dense_0"]["outputs"][0]["shape"]
    convolution = by_id["conv2d_0"]["attributes"]
    activation = by_id["relu_0"]["attributes"]
    pooling = by_id["max_pool2d_0"]["attributes"]
    dense = by_id["dense_0"]["attributes"]
    expected = {
        ("repack_0", "shape"): [*input_shape, 1],
        ("conv2d_0", "input_geometry"): repack_shape,
        ("conv2d_0", "output_geometry"): [
            convolution["out_height"],
            convolution["out_width"],
            convolution["n_filt"],
        ],
        ("relu_0", "shape"): convolution_shape,
        ("relu_0", "n_in"): _shape_size(convolution_shape),
        ("max_pool2d_0", "input_geometry"): convolution_shape,
        ("max_pool2d_0", "output_geometry"): [
            pooling["out_height"],
            pooling["out_width"],
            pooling["n_filt"],
        ],
        ("reshape_0", "shape"): [_shape_size(pooling_shape)],
        ("dense_0", "n_in"): _shape_size(reshape_shape),
        ("dense_0", "output_geometry"): [dense["n_out"]],
    }
    observed = {
        ("repack_0", "shape"): repack_shape,
        ("conv2d_0", "input_geometry"): [
            convolution["in_height"],
            convolution["in_width"],
            convolution["n_chan"],
        ],
        ("conv2d_0", "output_geometry"): convolution_shape,
        ("relu_0", "shape"): activation_shape,
        ("relu_0", "n_in"): activation["n_in"],
        ("max_pool2d_0", "input_geometry"): [
            pooling["in_height"],
            pooling["in_width"],
            pooling["n_filt"],
        ],
        ("max_pool2d_0", "output_geometry"): pooling_shape,
        ("reshape_0", "shape"): reshape_shape,
        ("dense_0", "n_in"): dense["n_in"],
        ("dense_0", "output_geometry"): output_shape,
    }
    findings = []
    for (operation_id, field), expected_value in expected.items():
        observed_value = observed[(operation_id, field)]
        if observed_value != expected_value:
            findings.append(
                {
                    "code": "family.geometry.relation",
                    "severity": "error",
                    "operation_id": operation_id,
                    "field": field,
                    "expected": expected_value,
                    "observed": observed_value,
                    "message": f"{operation_id}.{field} is inconsistent",
                }
            )
    return findings


def _shape_size(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _strategy_geometry_findings(
    operations: list[Mapping[str, Any]],
    choices: Mapping[str, int],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Check the selected renderer's real compile-time geometry contract."""

    by_id = {operation["id"]: operation for operation in operations}
    input_shape = by_id["input_0"]["outputs"][0]["shape"]
    convolution = by_id["conv2d_0"]["attributes"]
    pooling = by_id["max_pool2d_0"]["attributes"]
    temporal_pack = choices["TemporalPacking"]
    dense_parallelism = choices["DenseParallelism"]
    findings: list[dict[str, Any]] = []

    def require(
        condition: bool,
        *,
        code: str,
        operation_id: str,
        field: str,
        expected: Any,
        observed: Any,
        message: str,
    ) -> None:
        if not condition:
            findings.append(
                {
                    "code": code,
                    "severity": "error",
                    "operation_id": operation_id,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "message": message,
                }
            )

    require(
        input_shape[0] % temporal_pack == 0,
        code="strategy.geometry.input_pack",
        operation_id="input_0",
        field="height",
        expected=f"divisible by {temporal_pack}",
        observed=input_shape[0],
        message="Selected temporal packing must divide the input height",
    )
    common_convolution = {
        "n_chan": 1,
        "filt_width": 1,
        "stride_width": 1,
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
    }
    for field, expected in common_convolution.items():
        require(
            convolution[field] == expected,
            code="strategy.geometry.convolution",
            operation_id="conv2d_0",
            field=field,
            expected=expected,
            observed=convolution[field],
            message=f"Aria streaming convolution does not support {field}={convolution[field]}",
        )

    if temporal_pack == 4:
        p4_contract = {"filt_height": 5, "stride_height": 3}
        for field, expected in p4_contract.items():
            require(
                convolution[field] == expected,
                code="strategy.geometry.p4",
                operation_id="conv2d_0",
                field=field,
                expected=expected,
                observed=convolution[field],
                message=f"Aria P4 requires Conv2D {field}={expected}",
            )
    else:
        require(
            convolution["filt_height"] >= 3,
            code="strategy.geometry.p2",
            operation_id="conv2d_0",
            field="filt_height",
            expected=">= 3",
            observed=convolution["filt_height"],
            message="Aria P2 requires Conv2D filt_height>=3",
        )
        require(
            convolution["stride_height"] >= 2,
            code="strategy.geometry.p2",
            operation_id="conv2d_0",
            field="stride_height",
            expected=">= 2",
            observed=convolution["stride_height"],
            message="Aria P2 requires Conv2D stride_height>=2",
        )

    pooling_contract = {
        "pool_height": 2,
        "pool_width": 1,
        "stride_height": 2,
        "stride_width": 1,
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
        "pool_op": "Max",
    }
    for field, expected in pooling_contract.items():
        require(
            pooling[field] == expected,
            code="strategy.geometry.pooling",
            operation_id="max_pool2d_0",
            field=field,
            expected=expected,
            observed=pooling[field],
            message=f"Aria streaming pooling requires {field}={expected}",
        )

    require(
        convolution["out_width"] % dense_parallelism == 0,
        code="strategy.geometry.dense_stream",
        operation_id="dense_0",
        field="DenseParallelism",
        expected=f"divisor of output width {convolution['out_width']}",
        observed=dense_parallelism,
        message="Dense parallelism must divide each streamed width group",
    )
    for reason in plan["weight_delivery"]["applicability"]["reasons"]:
        findings.append(
            {
                "code": "strategy.parameters.dense",
                "severity": "error",
                "operation_id": "dense_0",
                "field": "weight_delivery",
                "expected": "wide-sequential-v1 compatible Dense",
                "observed": reason,
                "message": reason,
            }
        )
    return findings


def _parameter_bindings(
    model_facts: Mapping[str, Any], parameter_payload: ParameterPayload
) -> list[dict[str, Any]]:
    payload = parameter_payload.by_id()
    bindings = []
    for operation in model_facts["operations"]:
        for parameter in operation["parameters"]:
            binding_id = f"{operation['id']}:{parameter['role']}"
            descriptor = {
                key: deepcopy(value)
                for key, value in parameter.items()
                if key != "content_sha256"
            }
            bindings.append(
                {
                    "id": binding_id,
                    "operation_id": operation["id"],
                    "role": parameter["role"],
                    "symbol": payload[binding_id].symbol,
                    "type_name": payload[binding_id].type_name,
                    "descriptor": descriptor,
                }
            )
    return sorted(bindings, key=lambda binding: binding["id"])


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


def _rendering_contract(
    layers: list[Any], implementation_plan: Mapping[str, Any]
) -> dict[str, Any]:
    operations: dict[str, dict[str, Any]] = {}
    ordinals: dict[str, int] = {}
    for layer in layers:
        kind = _semantic_kind(layer)
        ordinal = ordinals.get(kind, 0)
        ordinals[kind] = ordinal + 1
        operation_id = f"{kind}_{ordinal}"
        output = layer.get_output_variable()
        operation = {
            "output_symbol": output.name,
            "output_type": output.type.name,
            "output_precision_cpp": output.type.precision.definition_cpp(),
        }
        index = layer.get_attr("index")
        if operation_id == "conv2d_0":
            operation["config_symbol"] = f"config{index}"
        elif operation_id == "relu_0":
            operation["config_symbol"] = f"relu_config{index}"
        elif operation_id in {"max_pool2d_0", "dense_0"}:
            operation["config_symbol"] = f"config{index}"
        operations[operation_id] = operation
    temporal_pack = implementation_plan["temporal_pack"]
    width_lanes = implementation_plan["width_lanes"]
    return {
        "operations": operations,
        "types": {
            "input_wide": _wide_type_name(
                operations["input_0"]["output_type"], f"x{temporal_pack}"
            ),
            "convolution_wide": _wide_type_name(
                operations["conv2d_0"]["output_type"], f"x{width_lanes}"
            ),
            "activation_wide": _wide_type_name(
                operations["relu_0"]["output_type"], f"x{width_lanes}"
            ),
            "pooling_wide": _wide_type_name(
                operations["max_pool2d_0"]["output_type"], f"x{width_lanes}"
            ),
        },
        "streams": {
            "convolution": (
                f"{operations['conv2d_0']['output_symbol']}_x{width_lanes}"
            ),
            "activation": f"{operations['relu_0']['output_symbol']}_x{width_lanes}",
            "pooling": (
                f"{operations['max_pool2d_0']['output_symbol']}_x{width_lanes}"
            ),
        },
        "first_convolution_function": (
            f"first_conv_{temporal_pack}row_4lane_temporal_wide_cl"
        ),
    }


def _wide_type_name(type_name: str, suffix: str) -> str:
    stem = type_name[:-2] if type_name.endswith("_t") else type_name
    return f"{stem}_{suffix}_t"


_QUANTIZER_ROLES = {
    "iq_conf": "input",
    "kq_conf": "weight",
    "bq_conf": "bias",
    "oq_conf": "output",
}


def _frontend_provenance(model: Any) -> dict[str, Any]:
    source_layers = []
    quantizer_contracts = []
    for ordinal, layer in enumerate(model.layers):
        source_layers.append(
            {
                "ordinal": ordinal,
                "module": type(layer).__module__,
                "class_name": type(layer).__name__,
            }
        )
        config = layer.get_config()
        for field, role in _QUANTIZER_ROLES.items():
            serialized = config.get(field)
            if not isinstance(serialized, Mapping):
                continue
            quantizer = serialized.get("config")
            if not isinstance(quantizer, Mapping):
                continue
            quantizer_contracts.append(
                {
                    "source_ordinal": ordinal,
                    "role": role,
                    "q_type": quantizer.get("q_type"),
                    "rounding": quantizer.get("round_mode"),
                    "overflow": quantizer.get("overflow_mode"),
                    "homogeneous_axis": _json_value(
                        quantizer.get("homogeneous_axis")
                    ),
                    "heterogeneous_axis": _json_value(
                        quantizer.get("heterogeneous_axis")
                    ),
                    "is_weight": quantizer.get("is_weight"),
                }
            )
    return {
        "adapter": {"id": "keras-hgq2", "version": 1},
        "source_layers": source_layers,
        "quantizer_contracts": quantizer_contracts,
    }


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
    layers: list[Any],
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
        inputs = [raw_outputs[name] for name in layer.inputs if name in raw_outputs]
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
        "inputs": [operations[0]["outputs"][0]["id"]],
        "outputs": [operations[-1]["outputs"][0]["id"]],
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


def _predicted_interface(
    model_facts: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    operations = model_facts["operations"]
    input_tensor = operations[0]["outputs"][0]
    output_tensor = operations[-1]["outputs"][0]
    input_width = input_tensor["numeric_type"]["width"]
    output_width = output_tensor["numeric_type"]["width"]
    input_slot_width = max(8, 1 << (input_width - 1).bit_length())
    output_slot_width = max(8, 1 << (output_width - 1).bit_length())
    return {
        "logical": {
            "input_shape": input_tensor["shape"],
            "output_shape": output_tensor["shape"],
        },
        "hls_stream": {
            "input_rows_per_word": plan["temporal_pack"],
            "values_per_input_word": plan["values_per_input_word"],
            "input_words_per_inference": plan["input_words_per_inference"],
        },
        "rtl": {
            "input_tdata_bits": plan["values_per_input_word"] * input_slot_width,
            "output_tdata_bits": _shape_size(output_tensor["shape"])
            * output_slot_width,
        },
    }
