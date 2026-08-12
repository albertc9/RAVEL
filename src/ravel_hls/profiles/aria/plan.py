"""Resolved implementation plans for the Aria specialization set."""

from collections.abc import Mapping
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
    input_shape = operations["input_0"]["outputs"][0]["shape"]
    convolution = operations["conv2d_0"]["attributes"]
    products_per_window = (
        convolution["filt_height"]
        * convolution["filt_width"]
        * convolution["n_chan"]
        * convolution["n_filt"]
    )
    first_convolution = {
        "id": "full-width-latency",
        "version": 1,
        "parallel_windows": convolution["out_width"],
        "products_per_window": products_per_window,
        "multiplier_limit": products_per_window * convolution["out_width"],
        "target_loop_ii": 1,
    }
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
        "template_profile": f"aria-p{temporal_pack}-d{dense_parallelism}-v3",
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
        "first_convolution": first_convolution,
        "weight_delivery": weight_delivery,
        "internal_fifo_depth": 4,
        "dataflow_start_propagation": False,
    }
