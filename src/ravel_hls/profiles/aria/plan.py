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
    operations = {
        operation["id"]: operation
        for operation in model_facts.get("operations", ())
    }
    input_shape = (
        operations["input_0"]["outputs"][0]["shape"]
        if operations
        else [256, 4]
    )
    convolution = (
        operations["conv2d_0"]["attributes"]
        if operations
        else {"out_width": 4, "n_filt": dense_facts["input_group_size"]}
    )
    dense_inputs = dense_facts["n_in"]
    dense_outputs = dense_facts["n_out"]
    dense_group_size = dense_facts["input_group_size"]
    mac_lanes = dense_group_size * dense_parallelism
    total_products = dense_inputs * dense_outputs
    dense_steps = (total_products + mac_lanes - 1) // mac_lanes
    tail_elements = total_products % mac_lanes
    valid_tail_lanes = tail_elements or mac_lanes
    applicability_reasons = []
    if dense_facts["n_out"] != 1:
        applicability_reasons.append("wide-sequential-v1 requires one Dense output")
    if dense_facts["parameter_representation"] != "dense":
        applicability_reasons.append(
            "wide-sequential-v1 requires dense parameter representation"
        )
    if dense_facts["feature_ordering"]["kind"] != "identity":
        applicability_reasons.append(
            "wide-sequential-v1 requires identity feature ordering"
        )
    if any(
        numeric["kind"] != "fixed" for numeric in dense_facts["numeric"].values()
    ):
        applicability_reasons.append("wide-sequential-v1 requires fixed-point types")
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
        "applicability": {
            "status": "applicable" if not applicability_reasons else "inapplicable",
            "reasons": applicability_reasons,
        },
    }
    return {
        "template_profile": f"aria-p{temporal_pack}-d{dense_parallelism}-v2",
        "temporal_pack": temporal_pack,
        "channels_per_row": input_shape[1],
        "values_per_input_word": temporal_pack * input_shape[1],
        "input_words_per_inference": input_shape[0] // temporal_pack,
        "width_lanes": convolution["out_width"],
        "filter_lanes": convolution["n_filt"],
        "values_per_internal_word": (
            convolution["out_width"] * convolution["n_filt"]
        ),
        "dense_inputs": dense_inputs,
        "dense_parallelism": dense_parallelism,
        "dense_steps": dense_steps,
        "weight_delivery": weight_delivery,
        "internal_fifo_depth": 4,
        "dataflow_start_propagation": False,
    }


def build_pass_records(
    implementation_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return ordered records for the selected legal pass sequence."""

    temporal_pack = implementation_plan["temporal_pack"]
    dense_parallelism = implementation_plan["dense_parallelism"]
    weight_delivery = implementation_plan["weight_delivery"]
    dense_parameters = {
        "dense_inputs": implementation_plan["dense_inputs"],
        "filter_lanes": implementation_plan["filter_lanes"],
        "dense_parallelism": dense_parallelism,
        "dense_steps": implementation_plan["dense_steps"],
        "weight_delivery": (
            f"{weight_delivery['id']}-v{weight_delivery['version']}"
        ),
    }
    if weight_delivery["id"] == "wide-sequential":
        dense_parameters.update(
            {
                "weight_word_bits": weight_delivery["word_bits"],
                "weight_depth": weight_delivery["depth"],
                "tail_elements": weight_delivery["tail_elements"],
                "tail_mask": weight_delivery["tail_mask"],
                "accumulation": weight_delivery["accumulation"]["policy"],
            }
        )
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
            {
                "width_lanes": implementation_plan["width_lanes"],
                "filter_lanes": implementation_plan["filter_lanes"],
            },
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            {"values_per_word": implementation_plan["values_per_internal_word"]},
            ["firmware/top.cpp", "firmware/defines.h"],
        ),
        (
            {"pool_height": 2, "pool_width": 1, "stride_height": 2},
            ["firmware/top.cpp", "firmware/nnet_utils/nnet_aria.h"],
        ),
        (
            dense_parameters,
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
