"""Dense facts derived from a converted hls4ml model graph."""

from collections.abc import Iterable
from typing import Any

import numpy as np


def analyze_dense_facts(layers: Iterable[Any]) -> list[dict[str, Any]]:
    """Describe Dense dimensions and parameter statistics without choosing a plan."""

    layer_list = list(layers)
    producers = {
        output: layer.class_name
        for layer in layer_list
        for output in getattr(layer, "outputs", ())
    }
    consumers: dict[str, list[str]] = {}
    for layer in layer_list:
        for input_name in getattr(layer, "inputs", ()):
            consumers.setdefault(input_name, []).append(layer.class_name)
    facts = []
    for layer in layer_list:
        if layer.class_name != "Dense":
            continue
        kernel, bias = list(layer.get_weights())
        kernel_values = np.asarray(kernel.data)
        bias_values = np.asarray(bias.data)
        zero = int(np.count_nonzero(kernel_values == 0))
        nonzero_values = kernel_values[kernel_values != 0]
        mantissas, _ = np.frexp(np.abs(nonzero_values))
        facts.append(
            {
                "role": "output",
                "n_in": layer.get_attr("n_in"),
                "n_out": layer.get_attr("n_out"),
                "kernel": {
                    "shape": list(kernel_values.shape),
                    "elements": int(kernel_values.size),
                    "statistics": {
                        "zero": zero,
                        "nonzero": int(kernel_values.size) - zero,
                        "power_of_two": int(np.count_nonzero(mantissas == 0.5)),
                        "unique": int(np.unique(kernel_values).size),
                    },
                },
                "bias": {
                    "shape": list(bias_values.shape),
                    "elements": int(bias_values.size),
                },
                "graph": {
                    "predecessors": [
                        producers[input_name]
                        for input_name in getattr(layer, "inputs", ())
                    ],
                    "successors": [
                        consumer
                        for output_name in getattr(layer, "outputs", ())
                        for consumer in consumers.get(output_name, ())
                    ],
                },
                "feature_ordering": {
                    "kind": "identity",
                    "order": "C",
                },
                "parameter_representation": (
                    "compressed"
                    if hasattr(kernel.type, "index_precision")
                    else "dense"
                ),
                "numeric": {
                    "input": _precision_facts(layer.get_input_variable().type),
                    "output": _precision_facts(layer.get_output_variable().type),
                    "weight": _precision_facts(kernel.type),
                    "bias": _precision_facts(bias.type),
                    "accumulator": _precision_facts(layer.get_attr("accum_t")),
                },
            }
        )
    return facts


def _precision_facts(type_value: Any) -> dict[str, Any]:
    precision = type_value.precision
    return {
        "kind": "fixed",
        "width": int(precision.width),
        "integer": int(precision.integer),
        "fractional": int(precision.fractional),
        "signed": bool(precision.signed),
        "rounding": str(precision.rounding_mode).removeprefix("AP_"),
        "overflow": str(precision.saturation_mode).removeprefix("AP_"),
        "saturation_bits": int(precision.saturation_bits),
    }
