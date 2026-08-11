"""Resolved implementation plans for the Aria specialization set."""

from collections.abc import Mapping
import hashlib
import json
from typing import Any


def build_implementation_plan(
    optimization: Mapping[str, int], model_facts: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the deterministic plan for one resolved specialization."""

    temporal_pack = optimization["TemporalPacking"]
    dense_parallelism = optimization["DenseParallelism"]
    dense_facts = model_facts["dense"][0]
    dense_inputs = dense_facts["n_in"]
    dense_outputs = dense_facts["n_out"]
    dense_group_size = dense_facts["input_group_size"]
    mac_lanes = dense_group_size * dense_parallelism
    total_products = dense_inputs * dense_outputs
    dense_steps = (total_products + mac_lanes - 1) // mac_lanes
    tail_elements = total_products % mac_lanes
    valid_tail_lanes = tail_elements or mac_lanes
    weight_delivery = {
        "id": "wide-sequential",
        "version": 1,
        "mac_lanes": mac_lanes,
        "word_bits": mac_lanes * dense_facts["numeric"]["weight"]["width"],
        "depth": dense_steps,
        "tail_elements": tail_elements,
        "tail_mask": (1 << valid_tail_lanes) - 1,
        "storage": {"type": "rom_1p", "implementation": "bram"},
        "multipliers": {"implementation": "dsp", "instances": mac_lanes},
        "accumulation": {"policy": "ordered"},
    }
    return {
        "template_profile": f"aria-p{temporal_pack}-d{dense_parallelism}-v2",
        "temporal_pack": temporal_pack,
        "channels_per_row": 4,
        "values_per_input_word": temporal_pack * 4,
        "input_words_per_inference": 256 // temporal_pack,
        "width_lanes": 4,
        "filter_lanes": dense_group_size,
        "values_per_internal_word": 4 * dense_group_size,
        "dense_inputs": dense_inputs,
        "dense_parallelism": dense_parallelism,
        "dense_steps": dense_steps,
        "weight_delivery": weight_delivery,
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
