"""Resolved implementation plan for the fixed Aria 1.0 profile."""

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

    return [
        {"id": pass_id, "version": 1, "order": order, "legality": "passed"}
        for order, pass_id in enumerate(PASS_IDS, start=1)
    ]
