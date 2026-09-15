"""Immutable qualification evidence is bound to the exact tested sources."""
import hashlib
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).parents[2] / "references/qualification/aria_1_7_0_multi_conv"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,count", [("composed_target", 99), ("two_block_c3", 95), ("legacy_p2d2", 77)])
def test_aria17_qualification_binds_c_rtl_protocol_and_vendor_evidence(name, count):
    project = ROOT / name
    manifest = read(project / "manifest.json")
    record = read(project / "qualification.json")
    protocol = read(project / "rtl/protocol.json")
    provenance = read(ROOT / "provenance.json")["models"][name]
    assert manifest["schema_version"] == 6
    assert record["schema_version"] == 5
    assert provenance["manifest_sha256"] == record["manifest_sha256"] == digest(project / "manifest.json")
    assert provenance["qualification_sha256"] == digest(project / "qualification.json")
    assert record["source_closure_sha256"] == manifest["source_closure_sha256"] == protocol["source_closure_sha256"]
    assert manifest["verification"]["transformation_equivalence"] == "passed"
    assert manifest["verification"]["source_conversion_consistency"] == "passed"
    assert record["rtl_cosimulation"] == protocol["status"] == "passed"
    assert protocol["completed_samples"] == count
    assert protocol["reset_aborts"] > 0 and protocol["reset_between_epochs"] > 0
    assert protocol["output_stall_cycles"] > 0 and protocol["input_gap_cycles"] > 0
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
    if name == "composed_target":
        assert manifest["verification"]["corpora"]["supplied"]["sample_count"] == 1000
        assert manifest["verification"]["stage_boundaries"]["status"] == "passed"
    if name == "two_block_c3":
        fixture = Path(__file__).with_name("fixtures") / "two_block_c3.keras"
        assert digest(fixture) == provenance["model_sha256"]


def test_aria17_legacy_pair_has_identical_ii_latency_resources_and_routed_slack():
    before = read(ROOT / "baseline/legacy_p2d2/ravel_qualification.json")
    after = read(ROOT / "legacy_p2d2/qualification.json")
    assert before["performance"] == after["performance"] == {"initiation_interval": 135, "latency_cycles": 140}
    before_manifest = read(ROOT / "baseline/legacy_p2d2/ravel_manifest.json")
    after_manifest = read(ROOT / "legacy_p2d2/manifest.json")
    assert before_manifest["implementation_plan"] == after_manifest["implementation_plan"]
    assert before_manifest["interfaces"] == after_manifest["interfaces"]
    utilization = (ROOT / "baseline/legacy_p2d2/ooc/ooc_utilization.rpt").read_text()
    for label, key in (("CLB LUTs", "LUT"), ("CLB Registers", "FF"), ("Block RAM Tile", "BRAM_TILES"), ("DSPs", "DSP")):
        observed = float(re.search(r"\|\s*" + label + r"\*?\s*\|\s*([\d,.]+)\s*\|", utilization)[1].replace(",", ""))
        assert after["ooc"]["resources"][key] == observed
    assert after["ooc"]["timing"]["wns_ns"] == 0.673


def test_aria17_fresh_process_reproduction_and_evidence_checksums_are_complete():
    for comparison in read(ROOT / "reproducibility.json").values():
        assert comparison["identical"]
        assert comparison["original"] == comparison["repeat"]
    for filename, expected in read(ROOT / "checksums.json").items():
        assert digest(ROOT / filename) == expected
