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
               if all(stage.get("arithmetic") for stage in item["stages"] if stage["operation_ids"][0].startswith("conv2d_"))]

    assert len(engines) >= 7
    assert all(-2 <= item["ii"] - item["predicted_frame_cycles"] <= 3 for item in engines)
