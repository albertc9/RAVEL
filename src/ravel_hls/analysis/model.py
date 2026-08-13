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
from ..analysis.phara import analyze_direct_parameters
from ..compatibility.dependencies import inspect_dependencies
from ..config import validate_public_config
from ..domain import ParameterPayload, ParameterTensor
from ..exceptions import CompatibilityError, ConfigurationError
from ..generations import builtin_generation
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

    generation = builtin_generation("aria", "1.5.1")
    config = validate_public_config(config)
    hls_values = config["HLS"]
    backend = hls_values.get("Backend", "Vitis")
    io_type = hls_values.get("IOType", "io_stream")
    if backend != "Vitis":
        raise ConfigurationError("HLS.Backend must be Vitis")
    if io_type != "io_stream":
        raise ConfigurationError("HLS.IOType must be io_stream")

    choices = config["Optimization"]

    dependencies = inspect_dependencies()
    if dependencies["dependency_qualification"] != "qualified":
        failed = [
            name
            for name, facts in dependencies["dependencies"].items()
            if facts["status"] != "qualified"
        ]
        raise CompatibilityError(
            "Aria 1.5.1 dependency stack is not qualified: " + ", ".join(failed)
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
    model_family, applicability = generation.match_model_family(
        model_facts, frontend_provenance
    )

    dense_facts: dict[str, Any] = {}
    resolved_design = None
    if model_family is not None:
        dense_facts = {"dense": analyze_dense_facts(layers)}
        plan = build_implementation_plan(choices, {**model_facts, **dense_facts})
        strategy = generation.strategy(
            "phara" if "phara" in plan else "aria-wide-stream",
            1 if "phara" in plan else 2,
        )
        strategy_findings = strategy.evaluate(
            model_facts["operations"], choices, plan
        )
        if strategy_findings:
            applicability = {
                "status": "unsupported",
                "findings": strategy_findings,
            }
        else:
            interface = _predicted_interface(model_facts, plan)
            coefficient_realization = (
                analyze_direct_parameters(model_facts, parameter_payload)
                if "phara" in plan
                else None
            )
            resolved_design = generation.resolver.resolve(
                model_facts=model_facts,
                implementation_plan=plan,
                interfaces=interface,
                parameter_bindings=_parameter_bindings(
                    model_facts, parameter_payload
                ),
                rendering=_rendering_contract(layers, plan),
                coefficient_realization=coefficient_realization,
            )
    analysis = ModelAnalysis._from_report(
        {
            "schema_version": 1,
            "generation": generation.identity,
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


def _shape_size(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


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
