"""Source-bound release gates, independent of analytical cost predictions."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / "references/qualification/aria_1_7_1_search"
PRIOR = ROOT.with_name("aria_1_7_0_multi_conv")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,count", [("composed_target", 99), ("two_block_c3", 95), ("heldout", 95), ("legacy_p2d2", 77)])
def test_search_release_evidence_binds_exact_c_rtl_and_vendor_results(name, count):
    project = ROOT / name
    manifest = read(project / "manifest.json")
    record = read(project / "qualification.json")
    protocol = read(project / "rtl/protocol.json")
    assert manifest["schema_version"] == 7
    assert manifest["ravel"]["release"] == "1.7.1"
    assert record["manifest_sha256"] == digest(project / "manifest.json")
    assert record["source_closure_sha256"] == manifest["source_closure_sha256"] == protocol["source_closure_sha256"]
    assert record["rtl_cosimulation"] == protocol["status"] == "passed"
    assert protocol["completed_samples"] == count
    assert protocol["reset_aborts"] and protocol["reset_between_epochs"]
    assert protocol["input_gap_cycles"] and protocol["output_stall_cycles"]
    assert protocol["log_sha256"] == digest(project / "rtl/simulation.txt")
    assert protocol["testbench_sha256"] == digest(project / "rtl/testbench.sv")
    for filename, expected in protocol["vector_files"].items():
        assert digest(project / "rtl" / filename) == expected
    for filename, expected in record["report_files"].items():
        assert digest(project / "hls" / filename) == expected
    if record["ooc"]:
        assert record["ooc"]["manifest_sha256"] == record["manifest_sha256"]
        for filename, expected in record["ooc"]["report_files"].items():
            assert digest(project / "ooc" / filename) == expected
    assert manifest["verification"]["transformation_equivalence"] == "passed"
    assert manifest["verification"]["source_conversion_consistency"] == "passed"
    if name != "legacy_p2d2":
        assert manifest["resolved_design"]["stages"][1]["strategy"]["id"] == "aria-window-stream"
        assert manifest["verification"]["stage_boundaries"]["status"] == "passed"
        assert manifest["resolved_design"]["optimization_search"]["performance_qualification"] == "not_run"


def test_target_beats_the_matched_aria170_interval_and_closes_the_requested_clock():
    current = read(ROOT / "composed_target/qualification.json")
    previous = read(PRIOR / "composed_target/qualification.json")
    assert current["performance"]["initiation_interval"] < previous["performance"]["initiation_interval"] == 844
    assert current["ooc"]["timing"]["target_clock_ns"] == previous["ooc"]["timing"]["target_clock_ns"] == 5
    assert current["ooc"]["timing"]["wns_ns"] >= 0
    manifest = read(ROOT / "composed_target/manifest.json")
    corpora = manifest["verification"]["corpora"]
    assert corpora["built_in"]["sample_count"] == 96
    assert corpora["supplied"]["sample_count"] == 1000


def test_legacy_performance_and_all_nineteen_firmware_configurations_are_preserved():
    current = read(ROOT / "legacy_p2d2/qualification.json")
    previous = read(PRIOR / "legacy_p2d2/qualification.json")
    assert current["performance"] == previous["performance"]
    assert current["ooc"]["resources"] == previous["ooc"]["resources"]
    assert current["ooc"]["timing"] == previous["ooc"]["timing"]
    comparisons = read(ROOT / "legacy-comparison.json")
    assert len(comparisons) == 19
    assert all(row["identical_to_aria170"] and row["firmware_files"] == 86 for row in comparisons)


def test_reviewed_sources_reproduce_the_qualified_hardware_and_evidence_hashes():
    reproduction = read(ROOT / "reproducibility.json")["projects"]
    assert set(reproduction) == {"composed_target", "two_block_c3", "heldout", "legacy_p2d2"}
    for result in reproduction.values():
        assert result["original"] == result["repeat"]
        assert result["compilation_inputs_identical"] and result["selected_candidate_identical"]
        assert set(result["host_stamp_only_differences"]) <= {"build_lib.sh", "hls4ml_config.yml"}
    provenance = read(ROOT / "provenance.json")
    for name, filename in (("two_block_c3", "two_block_c3.keras"), ("heldout", "two_block_heldout.keras")):
        assert digest(Path(__file__).parent / "fixtures" / filename) == provenance["models"][name]["model_sha256"]
    for filename, expected in read(ROOT / "checksums.json").items():
        assert digest(ROOT / filename) == expected
