"""Actually executed immutable Aria resolved-design transformations."""

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable, Mapping

from ..registry import ComponentDefinition


PassEffect = Callable[[dict[str, Any]], dict[str, Any]]

ARIA_PASS_DEFINITIONS = tuple(
    ComponentDefinition(pass_id, version)
    for pass_id, version in (
        ("pack-temporal-input", 1),
        ("fuse-repack-into-first-conv", 2),
        ("propagate-wide-relu-stream", 1),
        ("specialize-nonoverlapping-maxpool", 1),
        ("stream-flatten-into-dense", 1),
        ("bind-shallow-internal-fifos", 1),
        ("elide-dataflow-start-propagation", 1),
    )
)

PHARA_FUSION_PASS = ComponentDefinition(
    "fuse-pool-aligned-conv-relu-maxpool", 1
)


def resolve_aria_design(
    *,
    model_facts: Mapping[str, Any],
    implementation_plan: Mapping[str, Any],
    interfaces: Mapping[str, Any],
    parameter_bindings: list[dict[str, Any]],
    rendering: Mapping[str, Any],
    coefficient_realization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and execute the versioned Aria strategy over one design value."""

    operations = {
        operation["id"]: operation for operation in model_facts["operations"]
    }
    phara = implementation_plan.get("phara")
    state: dict[str, Any] = {
        "schema_version": 1,
        "generation": {"id": "aria", "version": "1.5.1"},
        "model_family": {"id": "hgq-conv-pool-dense", "version": 1},
        "strategy": (
            {"id": "phara", "version": 1}
            if phara is not None
            else {"id": "aria-wide-stream", "version": 2}
        ),
        "resolver": (
            {"id": "aria-aggressive-phara", "version": 1}
            if phara is not None
            else {"id": "aria-explicit-pd", "version": 2}
        ),
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
    if coefficient_realization is not None:
        state["coefficient_realization"] = deepcopy(
            dict(coefficient_realization)
        )
    input_transform = (
        ARIA_PASS_DEFINITIONS[0],
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
        )
    )
    if phara is not None:
        region_transformations: tuple[tuple[ComponentDefinition, PassEffect], ...] = (
            (
                PHARA_FUSION_PASS,
                lambda current: _set_streaming(
                    current,
                    "phara_fused_region",
                    {
                        "operation_ids": [
                            "conv2d_0",
                            "relu_0",
                            "max_pool2d_0",
                        ],
                        "pool_rows_per_supertile": phara[
                            "pool_rows_per_supertile"
                        ],
                        "supertile_input_rows": phara["supertile_input_rows"],
                        "pooled_words": phara["pooled_words"],
                        "scheduler": deepcopy(phara.get("scheduler")),
                        "realization": phara["realization"],
                    },
                ),
            ),
        )
    else:
        region_transformations = (
            (
                ARIA_PASS_DEFINITIONS[1],
                lambda current: _set_streaming(
                    current,
                    "first_convolution",
                    {
                        "operation_id": "conv2d_0",
                        "kernel": [
                            operations["conv2d_0"]["attributes"][
                                "filt_height"
                            ],
                            operations["conv2d_0"]["attributes"]["filt_width"],
                        ],
                        "stride": [
                            operations["conv2d_0"]["attributes"][
                                "stride_height"
                            ],
                            operations["conv2d_0"]["attributes"]["stride_width"],
                        ],
                        "width_lanes": implementation_plan["width_lanes"],
                        "filter_lanes": implementation_plan["filter_lanes"],
                    },
                ),
            ),
            (
                ARIA_PASS_DEFINITIONS[2],
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
                ARIA_PASS_DEFINITIONS[3],
                lambda current: _set_streaming(
                    current,
                    "pooling",
                    {
                        "operation_id": "max_pool2d_0",
                        "window": [
                            operations["max_pool2d_0"]["attributes"][
                                "pool_height"
                            ],
                            operations["max_pool2d_0"]["attributes"]["pool_width"],
                        ],
                        "stride": [
                            operations["max_pool2d_0"]["attributes"][
                                "stride_height"
                            ],
                            operations["max_pool2d_0"]["attributes"]["stride_width"],
                        ],
                    },
                ),
            ),
        )
    transformations: tuple[tuple[ComponentDefinition, PassEffect], ...] = (
        input_transform,
        *region_transformations,
        (
            ARIA_PASS_DEFINITIONS[4],
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
            ARIA_PASS_DEFINITIONS[5],
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
            ARIA_PASS_DEFINITIONS[6],
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
    for order, (definition, transform) in enumerate(transformations, start=1):
        input_fingerprint = _fingerprint(state)
        transformed = transform(state)
        output_fingerprint = _fingerprint(transformed)
        if output_fingerprint == input_fingerprint:
            raise RuntimeError(f"Aria pass {definition.id} did not transform the design")
        records.append(
            {
                "id": definition.id,
                "version": definition.version,
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
