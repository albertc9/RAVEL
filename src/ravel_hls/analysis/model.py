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
from ..compatibility.model_profile import validate_aria_model_profile
from ..exceptions import CompatibilityError, ConfigurationError
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

    def to_dict(self) -> dict[str, Any]:
        """Return an independent schema representation."""

        return _thaw(self._report)


def analyze(model: Any, config: Mapping[str, Any]) -> ModelAnalysis:
    """Analyze a qualified Keras/HGQ2 model without publishing a project."""

    hls_values = config.get("HLS")
    if not isinstance(hls_values, Mapping):
        raise ConfigurationError("HLS must be a mapping")
    backend = hls_values.get("Backend", "Vitis")
    io_type = hls_values.get("IOType", "io_stream")
    if backend != "Vitis":
        raise ConfigurationError("HLS.Backend must be Vitis")
    if io_type != "io_stream":
        raise ConfigurationError("HLS.IOType must be io_stream")

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
    validate_aria_model_profile(layers)

    model_facts, fingerprints = _extract_model_facts(layers)

    optimization = config.get("Optimization", {})
    if not isinstance(optimization, Mapping):
        raise ConfigurationError("Optimization must be a mapping")
    choices = {
        "TemporalPacking": optimization.get("TemporalPacking", 4),
        "DenseParallelism": optimization.get("DenseParallelism", 2),
    }
    dense_facts = {"dense": analyze_dense_facts(layers)}
    plan = build_implementation_plan(choices, dense_facts)
    interface = _predicted_interface(layers, plan)
    return ModelAnalysis._from_report(
        {
            "schema_version": 1,
            "generation": {"id": "aria", "version": "1.5.0"},
            "model_family": {"id": "hgq-conv-pool-dense", "version": 1},
            "applicability": {"status": "applicable", "findings": []},
            "model_facts": {**model_facts, **dense_facts},
            "resolved_design": {
                "specialization": {
                    "temporal_packing": plan["temporal_pack"],
                    "dense_parallelism": plan["dense_parallelism"],
                },
                "interfaces": interface,
            },
            "fingerprints": fingerprints,
        }
    )


def _semantic_kind(layer: Any) -> str:
    if layer.class_name == "Activation" and layer.get_attr("activation") == "relu":
        return "relu"
    if layer.class_name == "Pooling2D" and layer.get_attr("pool_op") == "Max":
        return "max_pool2d"
    names = {
        "Input": "input",
        "Repack": "repack",
        "Conv2D": "conv2d",
        "Reshape": "reshape",
        "Dense": "dense",
    }
    return names.get(layer.class_name, layer.class_name.lower())


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
        "parameter_state_sha256": _canonical_sha256(parameter_identities),
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
    layers: list[Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    input_width = layers[0].get_output_variable().type.precision.width
    output_width = layers[-1].get_output_variable().type.precision.width
    input_slot_width = max(8, 1 << (input_width - 1).bit_length())
    output_slot_width = max(8, 1 << (output_width - 1).bit_length())
    return {
        "logical": {"input_shape": [256, 4], "output_shape": [1]},
        "hls_stream": {
            "input_rows_per_word": plan["temporal_pack"],
            "values_per_input_word": plan["values_per_input_word"],
            "input_words_per_inference": plan["input_words_per_inference"],
        },
        "rtl": {
            "input_tdata_bits": plan["values_per_input_word"] * input_slot_width,
            "output_tdata_bits": output_slot_width,
        },
    }
