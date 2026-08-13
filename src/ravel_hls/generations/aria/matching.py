"""Aria family matching and implementation-strategy applicability."""

from collections.abc import Mapping
from typing import Any


_CANONICAL_SEQUENCE = (
    "input",
    "repack",
    "conv2d",
    "relu",
    "max_pool2d",
    "reshape",
    "dense",
)


def match_hgq_conv_pool_dense(
    facts: Mapping[str, Any], provenance: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    operations = facts["operations"]
    observed_sequence = tuple(operation["kind"] for operation in operations)
    findings: list[dict[str, Any]] = []
    if observed_sequence != _CANONICAL_SEQUENCE:
        mismatch = next(
            (
                index
                for index, (observed, expected) in enumerate(
                    zip(observed_sequence, _CANONICAL_SEQUENCE)
                )
                if observed != expected
            ),
            min(len(observed_sequence), len(_CANONICAL_SEQUENCE)),
        )
        findings.append(
            {
                "code": "family.topology.sequence",
                "severity": "error",
                "operation_id": (
                    operations[mismatch]["id"] if mismatch < len(operations) else None
                ),
                "expected": list(_CANONICAL_SEQUENCE),
                "observed": list(observed_sequence),
                "message": "Model operation sequence is outside this family",
            }
        )
    else:
        findings.extend(_family_geometry_findings(operations))
        for previous, current in zip(operations, operations[1:]):
            if current["inputs"] != [previous["outputs"][0]["id"]]:
                findings.append(
                    {
                        "code": "family.topology.wiring",
                        "severity": "error",
                        "operation_id": current["id"],
                        "expected": [previous["outputs"][0]["id"]],
                        "observed": current["inputs"],
                        "message": "Model family requires a direct linear chain",
                    }
                )
        source_layers = provenance["source_layers"]
        for class_name in ("QConv2D", "QDense"):
            matching_sources = [
                source
                for source in source_layers
                if source["class_name"] == class_name
                and source["module"].startswith("hgq.layers")
            ]
            if len(matching_sources) != 1:
                findings.append(
                    {
                        "code": "family.provenance.hgq2",
                        "severity": "error",
                        "operation_id": (
                            "conv2d_0" if class_name == "QConv2D" else "dense_0"
                        ),
                        "expected": f"one hgq.layers {class_name}",
                        "observed": len(matching_sources),
                        "message": "Model family requires HGQ2 Conv2D and Dense origins",
                    }
                )
    if findings:
        return None, {"status": "unsupported", "findings": findings}
    return (
        {"id": "hgq-conv-pool-dense", "version": 1},
        {"status": "applicable", "findings": []},
    )


def _family_geometry_findings(
    operations: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {operation["id"]: operation for operation in operations}
    input_shape = by_id["input_0"]["outputs"][0]["shape"]
    repack_shape = by_id["repack_0"]["outputs"][0]["shape"]
    convolution_shape = by_id["conv2d_0"]["outputs"][0]["shape"]
    activation_shape = by_id["relu_0"]["outputs"][0]["shape"]
    pooling_shape = by_id["max_pool2d_0"]["outputs"][0]["shape"]
    reshape_shape = by_id["reshape_0"]["outputs"][0]["shape"]
    output_shape = by_id["dense_0"]["outputs"][0]["shape"]
    convolution = by_id["conv2d_0"]["attributes"]
    activation = by_id["relu_0"]["attributes"]
    pooling = by_id["max_pool2d_0"]["attributes"]
    dense = by_id["dense_0"]["attributes"]
    expected = {
        ("repack_0", "shape"): [*input_shape, 1],
        ("conv2d_0", "input_geometry"): repack_shape,
        ("conv2d_0", "output_geometry"): [
            convolution["out_height"],
            convolution["out_width"],
            convolution["n_filt"],
        ],
        ("relu_0", "shape"): convolution_shape,
        ("relu_0", "n_in"): _shape_size(convolution_shape),
        ("max_pool2d_0", "input_geometry"): convolution_shape,
        ("max_pool2d_0", "output_geometry"): [
            pooling["out_height"],
            pooling["out_width"],
            pooling["n_filt"],
        ],
        ("reshape_0", "shape"): [_shape_size(pooling_shape)],
        ("dense_0", "n_in"): _shape_size(reshape_shape),
        ("dense_0", "output_geometry"): [dense["n_out"]],
    }
    observed = {
        ("repack_0", "shape"): repack_shape,
        ("conv2d_0", "input_geometry"): [
            convolution["in_height"],
            convolution["in_width"],
            convolution["n_chan"],
        ],
        ("conv2d_0", "output_geometry"): convolution_shape,
        ("relu_0", "shape"): activation_shape,
        ("relu_0", "n_in"): activation["n_in"],
        ("max_pool2d_0", "input_geometry"): [
            pooling["in_height"],
            pooling["in_width"],
            pooling["n_filt"],
        ],
        ("max_pool2d_0", "output_geometry"): pooling_shape,
        ("reshape_0", "shape"): reshape_shape,
        ("dense_0", "n_in"): dense["n_in"],
        ("dense_0", "output_geometry"): output_shape,
    }
    findings = []
    for (operation_id, field), expected_value in expected.items():
        observed_value = observed[(operation_id, field)]
        if observed_value != expected_value:
            findings.append(
                {
                    "code": "family.geometry.relation",
                    "severity": "error",
                    "operation_id": operation_id,
                    "field": field,
                    "expected": expected_value,
                    "observed": observed_value,
                    "message": f"{operation_id}.{field} is inconsistent",
                }
            )
    return findings


def evaluate_aria_wide_stream(
    operations: list[Mapping[str, Any]],
    choices: Mapping[str, int],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {operation["id"]: operation for operation in operations}
    input_shape = by_id["input_0"]["outputs"][0]["shape"]
    convolution = by_id["conv2d_0"]["attributes"]
    pooling = by_id["max_pool2d_0"]["attributes"]
    temporal_pack = choices["TemporalPacking"]
    dense_parallelism = choices["DenseParallelism"]
    findings: list[dict[str, Any]] = []

    def require(
        condition: bool,
        *,
        code: str,
        operation_id: str,
        field: str,
        expected: Any,
        observed: Any,
        message: str,
    ) -> None:
        if not condition:
            findings.append(
                {
                    "code": code,
                    "severity": "error",
                    "operation_id": operation_id,
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "message": message,
                }
            )

    require(
        input_shape[0] % temporal_pack == 0,
        code="strategy.geometry.input_pack",
        operation_id="input_0",
        field="height",
        expected=f"divisible by {temporal_pack}",
        observed=input_shape[0],
        message="Selected temporal packing must divide the input height",
    )
    for field, expected in {
        "n_chan": 1,
        "filt_width": 1,
        "stride_width": 1,
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
    }.items():
        require(
            convolution[field] == expected,
            code="strategy.geometry.convolution",
            operation_id="conv2d_0",
            field=field,
            expected=expected,
            observed=convolution[field],
            message=f"Aria streaming convolution does not support {field}={convolution[field]}",
        )
    if temporal_pack in {4, 8}:
        geometry_code = (
            "strategy.geometry.phara"
            if temporal_pack == 8
            else "strategy.geometry.p4"
        )
        for field, expected in {"filt_height": 5, "stride_height": 3}.items():
            require(
                convolution[field] == expected,
                code=geometry_code,
                operation_id="conv2d_0",
                field=field,
                expected=expected,
                observed=convolution[field],
                message=(
                    f"Aria P{temporal_pack} requires Conv2D {field}={expected}"
                ),
            )
    else:
        require(
            convolution["filt_height"] >= 3,
            code="strategy.geometry.p2",
            operation_id="conv2d_0",
            field="filt_height",
            expected=">= 3",
            observed=convolution["filt_height"],
            message="Aria P2 requires Conv2D filt_height>=3",
        )
        require(
            convolution["stride_height"] >= 2,
            code="strategy.geometry.p2",
            operation_id="conv2d_0",
            field="stride_height",
            expected=">= 2",
            observed=convolution["stride_height"],
            message="Aria P2 requires Conv2D stride_height>=2",
        )
    for field, expected in {
        "pool_height": 2,
        "pool_width": 1,
        "stride_height": 2,
        "stride_width": 1,
        "pad_top": 0,
        "pad_bottom": 0,
        "pad_left": 0,
        "pad_right": 0,
        "pool_op": "Max",
    }.items():
        require(
            pooling[field] == expected,
            code="strategy.geometry.pooling",
            operation_id="max_pool2d_0",
            field=field,
            expected=expected,
            observed=pooling[field],
            message=f"Aria streaming pooling requires {field}={expected}",
        )
    require(
        convolution["out_width"] % dense_parallelism == 0,
        code="strategy.geometry.dense_stream",
        operation_id="dense_0",
        field="DenseParallelism",
        expected=f"divisor of output width {convolution['out_width']}",
        observed=dense_parallelism,
        message="Dense parallelism must divide each streamed width group",
    )
    for reason in plan["weight_delivery"]["applicability"]["reasons"]:
        findings.append(
            {
                "code": "strategy.parameters.dense",
                "severity": "error",
                "operation_id": "dense_0",
                "field": "weight_delivery",
                "expected": "wide-sequential-v1 compatible Dense",
                "observed": reason,
                "message": reason,
            }
        )
    return findings


def _shape_size(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result
