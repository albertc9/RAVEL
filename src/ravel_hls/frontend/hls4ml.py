"""The only adapter allowed to load Keras or mutate hls4ml conversion state."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock
from typing import Any
from ..compatibility.legacy_frontend import _normalize_singleton_channel_input
from ..exceptions import CompatibilityError
from .provenance import _frontend_provenance
_HLS4ML_CONVERSION_LOCK = RLock()

@contextmanager
def _homogeneous_stream_quantizer_compatibility():
    from hls4ml.backends.fpga.passes.hgq_proxy_model import (
        ProcessFixedPointQuantizerLayer,
    )
    from hls4ml.model.optimizer.passes.bit_exact import (
        get_input_layers,
        get_output_layers,
    )
    from hls4ml.model.optimizer.passes.hgq_proxy_model import (
        FuseFixedPointQuantizer,
    )
    from hls4ml.model.types import FixedPrecisionType

    original_transform = ProcessFixedPointQuantizerLayer.transform

    def transform(pass_instance, graph, node):
        if (
            graph.config.config["IOType"] != "io_stream"
            or not node.get_attr("fusible", False)
        ):
            return original_transform(pass_instance, graph, node)
        input_layer = get_input_layers(node)[0]
        if len(get_output_layers(input_layer)) != 1:
            return original_transform(pass_instance, graph, node)
        output_precision = node.get_output_variable().type.precision
        precision = FixedPrecisionType(
            output_precision.width,
            output_precision.integer,
            output_precision.signed,
            node.RND,
            node.SAT,
        )
        FuseFixedPointQuantizer().propagate(input_layer, precision)
        graph.remove_node(node)
        return True

    with _HLS4ML_CONVERSION_LOCK:
        ProcessFixedPointQuantizerLayer.transform = transform
        try:
            yield
        finally:
            ProcessFixedPointQuantizerLayer.transform = original_transform

def _normalize_plain_relu_layers(model: Any) -> Any:
    """Express a plain Keras ReLU layer as a ReLU activation.

    The pinned hls4ml converts a standalone ReLU layer as a thresholded ReLU
    whose zero threshold gets a degenerate fixed-point type; its C model then
    passes negative values. An Activation("relu") layer converts correctly and
    computes the same function.
    """
    def plain(layer):
        return (type(layer).__name__ == "ReLU" and type(layer).__module__.startswith("keras")
                and not getattr(layer, "negative_slope", 0) and not getattr(layer, "threshold", 0)
                and getattr(layer, "max_value", None) is None)

    if not any(plain(layer) for layer in getattr(model, "layers", ())):
        return model
    import keras

    return keras.models.clone_model(
        model, clone_function=lambda layer: keras.layers.Activation("relu", name=layer.name) if plain(layer) else layer)


def _require_convertible_operations(model: Any) -> None:
    """Report source operations the pinned hls4ml cannot convert, before it fails."""
    findings = [
        {"code": "frontend.operation.grouped_convolution", "severity": "error", "operation_id": None,
         "source_layer": layer.name,
         "message": "Grouped convolution is not converted by the pinned hls4ml; "
                    "export it as an ordinary convolution with a block-diagonal kernel"}
        for layer in getattr(model, "layers", ())
        if isinstance(getattr(layer, "groups", 1), int) and getattr(layer, "groups", 1) != 1
    ]
    if findings:
        raise CompatibilityError("; ".join(finding["message"] for finding in findings), findings=findings)


def convert_model(model, hls_values):
    backend = hls_values.get("Backend", "Vitis")
    io_type = hls_values.get("IOType", "io_stream")
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
    normalized_model = _normalize_singleton_channel_input(normalized_model)
    normalized_model = _normalize_plain_relu_layers(normalized_model)
    _require_convertible_operations(normalized_model)

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
    with _homogeneous_stream_quantizer_compatibility():
        graph = hls4ml.converters.convert_from_keras_model(**conversion)
    return graph, normalized_model, frontend_provenance
