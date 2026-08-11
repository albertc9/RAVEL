"""Dense facts derived from a converted hls4ml model graph."""

from collections.abc import Iterable
from typing import Any

import numpy as np


def analyze_dense_facts(layers: Iterable[Any]) -> list[dict[str, Any]]:
    """Describe Dense dimensions and parameter statistics without choosing a plan."""

    facts = []
    for layer in layers:
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
            }
        )
    return facts
