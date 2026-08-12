"""Actually executed immutable Aria resolved-design transformations."""

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable, Mapping


PassEffect = Callable[[dict[str, Any]], dict[str, Any]]


def resolve_aria_design(
    *,
    model_facts: Mapping[str, Any],
    implementation_plan: Mapping[str, Any],
    interfaces: Mapping[str, Any],
    parameter_bindings: list[dict[str, Any]],
    rendering: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve and execute the versioned Aria strategy over one design value."""

    operations = {
        operation["id"]: operation for operation in model_facts["operations"]
    }
    state: dict[str, Any] = {
        "schema_version": 1,
        "generation": {"id": "aria", "version": "1.5.0"},
        "model_family": {"id": "hgq-conv-pool-dense", "version": 1},
        "strategy": {"id": "aria-wide-stream", "version": 1},
        "resolver": {"id": "aria-explicit-pd", "version": 1},
        "specialization": {
            "temporal_packing": implementation_plan["temporal_pack"],
            "dense_parallelism": implementation_plan["dense_parallelism"],
        },
        "implementation_plan": deepcopy(dict(implementation_plan)),
        "interfaces": deepcopy(dict(interfaces)),
        "parameter_bindings": deepcopy(parameter_bindings),
        "rendering": deepcopy(dict(rendering)),
        "streaming": {},
    }
    transformations: tuple[tuple[str, int, PassEffect], ...] = (
        (
            "pack-temporal-input",
            1,
            lambda current: _set_streaming(
                current,
                "input",
                {
                    "rows_per_word": implementation_plan["temporal_pack"],
                    "values_per_word": implementation_plan["values_per_input_word"],
                    "words_per_inference": implementation_plan[
                        "input_words_per_inference"
                    ],
                },
            ),
        ),
        (
            "fuse-repack-into-first-conv",
            1,
            lambda current: _set_streaming(
                current,
                "first_convolution",
                {
                    "operation_id": "conv2d_0",
                    "kernel": [
                        operations["conv2d_0"]["attributes"]["filt_height"],
                        operations["conv2d_0"]["attributes"]["filt_width"],
                    ],
                    "stride": [
                        operations["conv2d_0"]["attributes"]["stride_height"],
                        operations["conv2d_0"]["attributes"]["stride_width"],
                    ],
                    "width_lanes": implementation_plan["width_lanes"],
                    "filter_lanes": implementation_plan["filter_lanes"],
                },
            ),
        ),
        (
            "propagate-wide-relu-stream",
            1,
            lambda current: _set_streaming(
                current,
                "activation",
                {
                    "operation_id": "relu_0",
                    "values_per_word": implementation_plan[
                        "values_per_internal_word"
                    ],
                },
            ),
        ),
        (
            "specialize-nonoverlapping-maxpool",
            1,
            lambda current: _set_streaming(
                current,
                "pooling",
                {
                    "operation_id": "max_pool2d_0",
                    "window": [
                        operations["max_pool2d_0"]["attributes"]["pool_height"],
                        operations["max_pool2d_0"]["attributes"]["pool_width"],
                    ],
                    "stride": [
                        operations["max_pool2d_0"]["attributes"]["stride_height"],
                        operations["max_pool2d_0"]["attributes"]["stride_width"],
                    ],
                },
            ),
        ),
        (
            "stream-flatten-into-dense",
            1,
            lambda current: _set_streaming(
                current,
                "dense",
                {
                    "operation_id": "dense_0",
                    "inputs": implementation_plan["dense_inputs"],
                    "parallelism": implementation_plan["dense_parallelism"],
                    "steps": implementation_plan["dense_steps"],
                    "weight_delivery": implementation_plan["weight_delivery"],
                },
            ),
        ),
        (
            "bind-shallow-internal-fifos",
            1,
            lambda current: _set_streaming(
                current,
                "buffers",
                {
                    "depth": implementation_plan["internal_fifo_depth"],
                    "storage": "srl",
                },
            ),
        ),
        (
            "elide-dataflow-start-propagation",
            1,
            lambda current: _set_streaming(
                current,
                "dataflow_control",
                {
                    "start_propagation": implementation_plan[
                        "dataflow_start_propagation"
                    ],
                    "block_control": "ap_ctrl_hs",
                },
            ),
        ),
    )
    records = []
    for order, (pass_id, version, transform) in enumerate(transformations, start=1):
        input_fingerprint = _fingerprint(state)
        transformed = transform(state)
        output_fingerprint = _fingerprint(transformed)
        if output_fingerprint == input_fingerprint:
            raise RuntimeError(f"Aria pass {pass_id} did not transform the design")
        records.append(
            {
                "id": pass_id,
                "version": version,
                "order": order,
                "result": "applied",
                "input_design_sha256": input_fingerprint,
                "output_design_sha256": output_fingerprint,
            }
        )
        state = transformed
    state["executed_passes"] = records
    state["resolved_design_sha256"] = records[-1]["output_design_sha256"]
    return state


def _set_streaming(
    state: dict[str, Any], name: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    transformed = deepcopy(state)
    transformed["streaming"][name] = deepcopy(dict(value))
    return transformed


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
