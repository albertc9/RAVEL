"""Resolved implementation plan for the fixed Aria 1.0 profile."""

import hashlib
import json
from typing import Any


PASS_IDS = (
    "PackTemporalInput2x",
    "FuseRepackReshapeIntoFirstConv",
    "PropagateWideReLUStream",
    "SpecializeNonOverlappingMaxPool",
    "StreamFlattenIntoDense",
    "BindShallowInternalFifos",
)


def build_implementation_plan() -> dict[str, Any]:
    """Return the deterministic P2 wide-stream plan selected by Aria 1.0."""

    return {
        "template_profile": "aria-2x-v1",
        "temporal_pack": 2,
        "channels_per_row": 4,
        "values_per_input_word": 8,
        "input_words_per_inference": 128,
        "width_lanes": 4,
        "filter_lanes": 7,
        "values_per_internal_word": 28,
        "dense_inputs": 1176,
        "internal_fifo_depth": 4,
    }


def build_pass_records() -> list[dict[str, Any]]:
    """Return ordered, versioned records for the fixed legal pass sequence."""

    effects = [
        (
            {"rows_per_word": 2, "values_per_word": 8},
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
            {"dense_inputs": 1176, "filter_lanes": 7},
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            {"fifo_depth": 4, "storage": "srl"},
            ["firmware/top.cpp"],
        ),
    ]
    state: dict[str, Any] = {"profile": "aria", "streaming": {}}
    records = []
    for order, (pass_id, effect) in enumerate(zip(PASS_IDS, effects), start=1):
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
