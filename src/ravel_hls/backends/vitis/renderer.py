"""Strict rendering of the Aria Vitis project specialization."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ...exceptions import ProjectGenerationError


_TEMPLATE_ROOT = Path(__file__).with_name("templates")


def render_aria_project(
    project_path: Path,
    project_name: str,
    layers: list[Any],
    *,
    optimization: Mapping[str, int],
) -> list[str]:
    """Render all Aria-owned firmware files from a typed model context."""

    input_layer, _, convolution, activation, pooling, _, dense = layers
    input_variable = input_layer.get_output_variable()
    convolution_variable = convolution.get_output_variable()
    activation_variable = activation.get_output_variable()
    pooling_variable = pooling.get_output_variable()
    output_variable = dense.get_output_variable()
    convolution_weights = list(convolution.get_weights())
    dense_weights = list(dense.get_weights())
    if len(convolution_weights) != 2 or len(dense_weights) != 2:
        raise ProjectGenerationError(
            "Aria requires Conv2D and Dense weight/bias pairs in the hls4ml graph"
        )

    firmware = project_path / "firmware"
    defines_path = firmware / "defines.h"
    if not defines_path.is_file():
        raise ProjectGenerationError("hls4ml baseline is missing firmware/defines.h")
    baseline_defines = defines_path.read_text(encoding="utf-8")
    defines_body, marker, _ = baseline_defines.rpartition("#endif")
    if not marker:
        raise ProjectGenerationError("hls4ml firmware/defines.h has no include guard")

    context = {
        "project_name": project_name,
        "input_name": input_variable.name,
        "output_name": output_variable.name,
        "input_wide_type": _wide_type_name(input_variable.type.name, "x2"),
        "conv_wide_type": _wide_type_name(convolution_variable.type.name, "x4"),
        "activation_wide_type": _wide_type_name(activation_variable.type.name, "x4"),
        "pool_wide_type": _wide_type_name(pooling_variable.type.name, "x4"),
        "output_type": output_variable.type.name,
        "input_precision": input_variable.type.precision.definition_cpp(),
        "conv_precision": convolution_variable.type.precision.definition_cpp(),
        "activation_precision": activation_variable.type.precision.definition_cpp(),
        "pool_precision": pooling_variable.type.precision.definition_cpp(),
        "conv_config": f"config{convolution.get_attr('index')}",
        "activation_config": f"relu_config{activation.get_attr('index')}",
        "pool_config": f"config{pooling.get_attr('index')}",
        "dense_config": f"config{dense.get_attr('index')}",
        "dense_parallelism": optimization["DenseParallelism"],
        "conv_stream": f"{convolution_variable.name}_x4",
        "activation_stream": f"{activation_variable.name}_x4",
        "pool_stream": f"{pooling_variable.name}_x4",
        "conv_weight": _weight_context(convolution_weights[0]),
        "conv_bias": _weight_context(convolution_weights[1]),
        "dense_weight": _weight_context(dense_weights[0]),
        "dense_bias": _weight_context(dense_weights[1]),
        "baseline_defines_body": defines_body.rstrip(),
    }
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATE_ROOT),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    outputs = {
        f"firmware/{project_name}.cpp": "aria/firmware/top.cpp.j2",
        f"firmware/{project_name}.h": "aria/firmware/top.h.j2",
        "firmware/defines.h": "aria/firmware/defines.h.j2",
        "firmware/nnet_utils/nnet_aria.h": "aria/firmware/nnet_aria.h.j2",
        f"{project_name}_bridge.cpp": "aria/bridge/bridge.cpp.j2",
        f"{project_name}_test.cpp": "aria/testbench/test.cpp.j2",
    }
    for relative_path, template_name in outputs.items():
        output = project_path / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            environment.get_template(template_name).render(**context), encoding="utf-8"
        )
    return sorted(outputs)


def _wide_type_name(type_name: str, suffix: str) -> str:
    stem = type_name[:-2] if type_name.endswith("_t") else type_name
    return f"{stem}_{suffix}_t"


def _weight_context(weight: Any) -> dict[str, Any]:
    return {
        "name": weight.name,
        "type_name": weight.type.name,
        "length": weight.data_length,
    }
