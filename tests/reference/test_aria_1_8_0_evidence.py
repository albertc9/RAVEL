"""Source-bound calibration evidence for the v1.8.0 separable-block coverage."""

import hashlib
import json
from pathlib import Path

from ravel_hls.planning.calibration import CONSTANT_MATRIX_COST_PROFILE

ROOT = Path(__file__).parents[2] / "references/qualification/aria_1_8_0_separable"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_record_matches_the_profile_and_its_frozen_reports():
    calibration = read(ROOT / "calibration.json")
    profile = CONSTANT_MATRIX_COST_PROFILE.to_dict()

    assert (calibration["profile"], calibration["version"]) == (profile["id"], profile["version"])
    assert profile["evidence"] == "references/qualification/aria_1_8_0_separable/calibration.json"
    assert len(calibration["measurements"]) >= 13
    for item in calibration["measurements"]:
        assert digest(ROOT / item["hls_report_file"]) == item["hls_report_sha256"]
        assert item["c_verification"] == "passed"
        assert item["rtl_cosimulation"] == "passed"
        assert item["rtl_protocol"] == "passed"


def test_shared_arithmetic_predictions_stay_close_to_the_measured_intervals():
    measurements = read(ROOT / "calibration.json")["measurements"]
    engines = [item for item in measurements
               if all(stage.get("arithmetic") for stage in item["stages"] if stage["operation_ids"][0].startswith("conv2d_"))
               and not any(stage["strategy"]["id"] == "aria-dense-matrix" for stage in item["stages"])]

    assert len(engines) >= 7
    assert all(-2 <= item["ii"] - item["predicted_frame_cycles"] <= 3 for item in engines)


def test_the_dense_matrix_overhead_matches_its_recorded_stage_measurements():
    from ravel_hls.planning.calibration import DENSE_MATRIX_OVERHEAD_CYCLES

    record = read(ROOT / "calibration.json")["dense_matrix_overhead"]

    assert record["cycles"] == DENSE_MATRIX_OVERHEAD_CYCLES
    overheads = [item["stage_ii"] - item["steps"] for item in record["measurements"]]
    assert len(overheads) >= 3 and max(overheads) == DENSE_MATRIX_OVERHEAD_CYCLES and min(overheads) >= 9


def test_final_projects_bind_their_frozen_files_and_pass_every_recorded_layer():
    final = read(ROOT / "final.json")

    assert {item["project"] for item in final} >= {"final_target", "final_m09de", "final_heldout"}
    for item in final:
        assert item["generation"] == {"id": "aria", "version": "1.8.0"}
        assert item["manifest_schema_version"] == 9
        for filename, expected in item["files"].items():
            assert digest(ROOT / item["project"] / filename) == expected
        assert item["c_verification"] == item["stage_boundaries"] == item["rtl_cosimulation"] == "passed"
        assert item["rtl_protocol"]["status"] == "passed" and item["rtl_protocol"]["output_stall_cycles"] > 0
        assert item["search_status"] == "complete"
    routed = {item["project"]: item["ooc"]["timing"] for item in final if "ooc" in item}
    assert set(routed) >= {"final_target", "final_m09de", "final_m13", "final_m08", "final2_m07", "final2_dense32"}
    # The Dense-32 shape missed timing before the accumulator loop carried one addition.
    assert routed.pop("final_dense32")["wns_ns"] < 0
    assert all(timing["wns_ns"] >= 0 and timing["tns_ns"] == 0 for timing in routed.values())


def test_the_target_shape_is_selected_with_shared_arithmetic_in_its_separable_group():
    target = next(item for item in read(ROOT / "final.json") if item["project"] == "final_target")

    assert [stage["strategy"] for stage in target["stages"]] == [
        "phara", "aria-affine-window", "aria-affine-window", "identity-layout-view", "aria-dense-wide"]
    assert target["first_stage_scheduler"]["version"] == 2
    record = read(ROOT / "final_target/qualification.json")
    assert record["manifest_sha256"] == target["manifest_sha256"]
    assert record["performance"] == {"initiation_interval": target["ii"], "latency_cycles": target["latency"]}
