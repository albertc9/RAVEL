"""Render a Vitis project from a resolved design and parameter payload only."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np

from ...domain import ParameterPayload, ParameterTensor
from ...exceptions import ProjectGenerationError
from ...packing import pack_fixed_point_words


_TEMPLATE_ROOT = (
    Path(__file__).parents[2] / "backends" / "vitis" / "templates"
)


def render_aria_project(
    project_path: Path,
    project_name: str,
    resolved_design: Mapping[str, Any],
    parameter_payload: ParameterPayload,
) -> list[str]:
    """Render every Aria-owned file without inspecting a model graph."""

    rendering = resolved_design["rendering"]
    plan = resolved_design["implementation_plan"]
    parameters = parameter_payload.by_id()
    required = {
        "conv2d_0:weight",
        "conv2d_0:bias",
        "dense_0:weight",
        "dense_0:bias",
    }
    if not required.issubset(parameters):
        raise ProjectGenerationError(
            "Resolved Aria parameter payload is missing required semantic bindings"
        )
    temporal_pack = plan["temporal_pack"]
    weight_delivery = plan["weight_delivery"]
    dense_packed = None
    if weight_delivery["id"] == "wide-sequential":
        dense_packed = _packed_weight_context(
            parameters["dense_0:weight"], weight_delivery
        )

    firmware = project_path / "firmware"
    defines_path = firmware / "defines.h"
    if not defines_path.is_file():
        raise ProjectGenerationError("hls4ml baseline is missing firmware/defines.h")
    baseline_defines = defines_path.read_text(encoding="utf-8")
    defines_body, marker, _ = baseline_defines.rpartition("#endif")
    if not marker:
        raise ProjectGenerationError("hls4ml firmware/defines.h has no include guard")

    input_value = rendering["operations"]["input_0"]
    convolution = rendering["operations"]["conv2d_0"]
    activation = rendering["operations"]["relu_0"]
    pooling = rendering["operations"]["max_pool2d_0"]
    dense = rendering["operations"]["dense_0"]
    streams = rendering["streams"]
    context = {
        "project_name": project_name,
        "input_name": input_value["output_symbol"],
        "output_name": dense["output_symbol"],
        "input_wide_type": rendering["types"]["input_wide"],
        "conv_wide_type": rendering["types"]["convolution_wide"],
        "activation_wide_type": rendering["types"]["activation_wide"],
        "pool_wide_type": rendering["types"]["pooling_wide"],
        "output_type": dense["output_type"],
        "input_precision": input_value["output_precision_cpp"],
        "conv_precision": convolution["output_precision_cpp"],
        "activation_precision": activation["output_precision_cpp"],
        "pool_precision": pooling["output_precision_cpp"],
        "conv_config": convolution["config_symbol"],
        "activation_config": activation["config_symbol"],
        "pool_config": pooling["config_symbol"],
        "dense_config": dense["config_symbol"],
        "dense_parallelism": plan["dense_parallelism"],
        "temporal_pack": temporal_pack,
        "channels_per_row": plan["channels_per_row"],
        "input_values_per_inference": (
            plan["input_words_per_inference"] * plan["values_per_input_word"]
        ),
        "input_words_per_inference": plan["input_words_per_inference"],
        "width_lanes": plan["width_lanes"],
        "filter_lanes": plan["filter_lanes"],
        "values_per_internal_word": plan["values_per_internal_word"],
        "first_conv_function": rendering["first_convolution_function"],
        "conv_stream": streams["convolution"],
        "activation_stream": streams["activation"],
        "pool_stream": streams["pooling"],
        "conv_weight": _weight_context(parameters["conv2d_0:weight"]),
        "conv_bias": _weight_context(parameters["conv2d_0:bias"]),
        "dense_weight": _weight_context(parameters["dense_0:weight"]),
        "dense_packed": dense_packed,
        "dense_bias": _weight_context(parameters["dense_0:bias"]),
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
    if dense_packed is not None:
        outputs[dense_packed["path"]] = "aria/firmware/dense_weights.h.j2"
    for relative_path, template_name in outputs.items():
        output = project_path / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            environment.get_template(template_name).render(**context), encoding="utf-8"
        )
    return sorted(outputs)


def _weight_context(parameter: ParameterTensor) -> dict[str, Any]:
    return {
        "name": parameter.symbol,
        "type_name": parameter.type_name,
        "length": int(parameter.values.size),
    }


def _packed_weight_context(
    parameter: ParameterTensor, weight_delivery: Mapping[str, Any]
) -> dict[str, Any]:
    precision = parameter.numeric_type
    values = np.asarray(parameter.values).reshape(-1)
    try:
        words = pack_fixed_point_words(
            values,
            width=precision["width"],
            integer=precision["integer"],
            signed=precision["signed"],
            lanes=weight_delivery["mac_lanes"],
        )
    except ValueError as error:
        raise ProjectGenerationError(
            f"Dense weight {parameter.symbol} cannot be packed: {error}"
        ) from error
    if len(words) != weight_delivery["depth"]:
        raise ProjectGenerationError(
            f"Dense weight {parameter.symbol} packed depth does not match its plan"
        )
    word_bits = weight_delivery["word_bits"]
    hex_digits = (word_bits + 3) // 4
    return {
        "name": f"{parameter.symbol}_ravel_packed",
        "guard": f"RAVEL_{parameter.symbol.upper()}_PACKED_H_",
        "path": f"firmware/weights/{parameter.symbol}_ravel_packed.h",
        "word_bits": word_bits,
        "lane_bits": precision["width"],
        "mac_lanes": weight_delivery["mac_lanes"],
        "depth": len(words),
        "tail_elements": weight_delivery["tail_elements"],
        "valid_last_lanes": (
            weight_delivery["tail_elements"] or weight_delivery["mac_lanes"]
        ),
        "words": [f"0x{word:0{hex_digits}x}" for word in words],
    }
