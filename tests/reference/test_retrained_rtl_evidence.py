import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).parents[2]
EVIDENCE_ROOT = REPOSITORY / "references" / "qualification" / "aria_1_5_retrained"
MODELS = (
    "adam_p1_step2",
    "adam_hgq_replicate_s2",
    "adam_hgq_replicate_s2_300ep",
)


@pytest.mark.parametrize("name", MODELS)
def test_representative_retrained_rtl_cosim_evidence_is_source_backed(
    name: str,
) -> None:
    model = (
        REPOSITORY
        / "references"
        / "fLow_0.08-fhigh_0.23-rate_0.5"
        / name
        / f"{name}_best.keras"
    )
    manifest_path = EVIDENCE_ROOT / f"{name}_manifest.json"
    qualification_path = EVIDENCE_ROOT / f"{name}_qualification.json"
    synthesis_path = EVIDENCE_ROOT / f"{name}_csynth.xml"
    cosimulation_path = EVIDENCE_ROOT / f"{name}_cosim.rpt"
    provenance = json.loads(
        (EVIDENCE_ROOT / "provenance.json").read_text(encoding="utf-8")
    )["models"][name]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))

    assert provenance["model_sha256"] == _sha256(model)
    assert provenance["manifest_sha256"] == _sha256(manifest_path)
    assert provenance["qualification_sha256"] == _sha256(qualification_path)
    assert qualification["manifest_sha256"] == _sha256(manifest_path)
    assert qualification["generation_fingerprint"] == manifest[
        "generation_fingerprint"
    ]
    assert qualification["source_closure_sha256"] == manifest[
        "source_closure_sha256"
    ]
    assert qualification["rtl_cosimulation"] == "passed"
    assert qualification["tool"] == {"name": "Vitis HLS", "version": "2023.2"}
    assert qualification["part"] == "xcku5p-ffvb676-2-e"
    assert qualification["timing"]["target_clock_ns"] == 5.0
    assert qualification["report_files"] == {
        "project_prj/solution1/sim/report/project_cosim.rpt": _sha256(
            cosimulation_path
        ),
        "project_prj/solution1/syn/report/project_csynth.xml": _sha256(
            synthesis_path
        ),
    }
    assert "|   Verilog|      Pass|" in cosimulation_path.read_text(
        encoding="utf-8"
    )


def test_representative_set_and_environment_are_fixed() -> None:
    provenance = json.loads(
        (EVIDENCE_ROOT / "provenance.json").read_text(encoding="utf-8")
    )

    assert provenance["ravel_commit"] == "0ea7b80"
    assert provenance["generation_commit"] == "cca8710"
    assert provenance["dependencies"] == {
        "hls4ml": "1.2.0",
        "hgq2": "0.1.7",
        "keras": "3.12.1",
        "numpy": "1.26.4",
    }
    assert tuple(provenance["models"]) == MODELS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
