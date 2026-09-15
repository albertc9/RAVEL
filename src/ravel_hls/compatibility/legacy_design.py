"""Preserved legacy plan/render bindings from the immutable projection."""
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from ..domain import ParameterPayload

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

def _rendering_contract(
    native: Mapping[str, Any], implementation_plan: Mapping[str, Any]
) -> dict[str, Any]:
    operations = {}
    for operation_id, binding in native.items():
        operation = {key: binding[key] for key in ("output_symbol", "output_type", "output_precision_cpp")}
        if operation_id in {"conv2d_0", "relu_0", "max_pool2d_0", "dense_0"}:
            operation["config_symbol"] = binding["config_symbol"].replace("config", "relu_config") if operation_id == "relu_0" else binding["config_symbol"]
        operations[operation_id] = operation
    temporal_pack = implementation_plan["temporal_pack"]
    width_lanes = implementation_plan["width_lanes"]
    contract = {
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
    if "phara" in implementation_plan:
        realization = implementation_plan["phara"]["realization"]
        contract["phara_fused_function"] = (
            f"phara_pool_aligned_{realization}_p{temporal_pack}_cl"
        )
        contract["dense_function"] = "dense_wide_stream"
    return contract

def _wide_type_name(type_name: str, suffix: str) -> str:
    stem = type_name[:-2] if type_name.endswith("_t") else type_name
    return f"{stem}_{suffix}_t"

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
