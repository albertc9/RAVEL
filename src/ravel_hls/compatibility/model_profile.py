"""Aria semantic model-profile checks."""

from collections.abc import Iterable
from typing import Any

from ..exceptions import CompatibilityError


def validate_aria_model_profile(layers: Iterable[Any]) -> None:
    """Reject model graphs outside the fixed Aria 1.4.0 topology family."""

    layer_list = list(layers)
    expected_sequence = [
        "Input",
        "Repack",
        "Conv2D",
        "Activation",
        "Pooling2D",
        "Reshape",
        "Dense",
    ]
    actual_sequence = [layer.class_name for layer in layer_list]
    if actual_sequence != expected_sequence:
        raise CompatibilityError(
            "Aria 1.4.0 layer sequence must be " + " -> ".join(expected_sequence)
        )
    for producer, consumer in zip(layer_list, layer_list[1:]):
        if list(getattr(consumer, "inputs", ())) != list(
            getattr(producer, "outputs", ())
        ):
            raise CompatibilityError(
                "Aria graph wiring must be a direct linear chain"
            )
    for layer in (layer_list[2], layer_list[-1]):
        module = layer.get_attr("module")
        if not isinstance(module, str) or not module.startswith("hgq.layers"):
            raise CompatibilityError(
                f"Aria {layer.class_name} must originate from an HGQ quantized layer"
            )
        if layer.get_attr("strategy") != "latency":
            raise CompatibilityError(
                f"Aria {layer.class_name}.strategy must be latency"
            )
    for layer in layer_list:
        reuse_factor = layer.get_attr("reuse_factor")
        if reuse_factor is not None and reuse_factor != 1:
            raise CompatibilityError(
                f"Aria {layer.class_name}.reuse_factor must be 1"
            )
    required_attributes = [
        {"target_shape": [256, 4, 1]},
        {
            "in_height": 256,
            "in_width": 4,
            "n_chan": 1,
            "filt_height": 5,
            "filt_width": 1,
            "n_filt": 7,
            "stride_height": 3,
            "stride_width": 1,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
            "out_height": 84,
            "out_width": 4,
        },
        {"activation": "relu", "n_in": 2352},
        {
            "in_height": 84,
            "in_width": 4,
            "n_filt": 7,
            "pool_height": 2,
            "pool_width": 1,
            "stride_height": 2,
            "stride_width": 1,
            "pad_top": 0,
            "pad_bottom": 0,
            "pad_left": 0,
            "pad_right": 0,
            "pool_op": "Max",
            "out_height": 42,
            "out_width": 4,
        },
        {"target_shape": [1176]},
        {"n_in": 1176, "n_out": 1},
    ]
    for layer, expected in zip(layer_list[1:], required_attributes):
        for attribute, expected_value in expected.items():
            if layer.get_attr(attribute) != expected_value:
                raise CompatibilityError(
                    f"Aria {layer.class_name}.{attribute} must be {expected_value}"
                )
