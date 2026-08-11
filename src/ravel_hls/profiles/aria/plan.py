"""Resolved implementation plans for the Aria specialization set."""

from collections.abc import Mapping
import hashlib
import json
from typing import Any


def build_implementation_plan(optimization: Mapping[str, int]) -> dict[str, Any]:
    """Return the deterministic plan for one resolved specialization."""

    temporal_pack = optimization["TemporalPacking"]
    dense_parallelism = optimization["DenseParallelism"]
    return {
        "template_profile": f"aria-p{temporal_pack}-d{dense_parallelism}-v1",
        "temporal_pack": temporal_pack,
        "channels_per_row": 4,
        "values_per_input_word": temporal_pack * 4,
        "input_words_per_inference": 256 // temporal_pack,
        "width_lanes": 4,
        "filter_lanes": 7,
        "values_per_internal_word": 28,
        "dense_inputs": 1176,
        "dense_parallelism": dense_parallelism,
        "dense_steps": 168 // dense_parallelism,
        "internal_fifo_depth": 4,
        "dataflow_start_propagation": False,
    }


def build_pass_records(
    optimization: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return ordered records for the selected legal pass sequence."""

    temporal_pack = optimization["TemporalPacking"]
    dense_parallelism = optimization["DenseParallelism"]
    pass_ids = (
        f"PackTemporalInput{temporal_pack}x",
        "FuseRepackReshapeIntoFirstConv",
        "PropagateWideReLUStream",
        "SpecializeNonOverlappingMaxPool",
        "StreamFlattenIntoDense",
        "BindShallowInternalFifos",
        "ElideDataflowStartPropagation",
    )
    effects = [
        (
            {
                "rows_per_word": temporal_pack,
                "values_per_word": temporal_pack * 4,
            },
            ["firmware/defines.h", "bridge", "testbench"],
        ),
        (
            {"width_lanes": 4, "filter_lanes": 7},
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            {"values_per_word": 28},
            ["firmware/top.cpp", "firmware/defines.h"],
        ),
        (
            {"pool_height": 2, "pool_width": 1, "stride_height": 2},
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            {
                "dense_inputs": 1176,
                "filter_lanes": 7,
                "dense_parallelism": dense_parallelism,
                "dense_steps": 168 // dense_parallelism,
            },
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            {"fifo_depth": 4, "storage": "srl"},
            ["firmware/top.cpp"],
        ),
        (
            {"start_propagation": False, "block_control": "ap_ctrl_hs"},
            ["firmware/top.cpp"],
        ),
    ]
    state: dict[str, Any] = {"profile": "aria", "streaming": {}}
    records = []
    for order, (pass_id, effect) in enumerate(zip(pass_ids, effects), start=1):
        parameters, artifacts = effect
        input_fingerprint = _fingerprint(state)
        state = {
            **state,
            "streaming": {**state["streaming"], pass_id: parameters},
        }
        records.append(
            {
                "id": pass_id,
                "version": 1,
                "order": order,
                "legality": "passed",
                "resolved_parameters": parameters,
                "input_ir_sha256": input_fingerprint,
                "output_ir_sha256": _fingerprint(state),
                "affected_artifacts": artifacts,
            }
        )
    return records


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
