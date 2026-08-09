import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ravel_hls import ProjectGenerationError, open_project


def test_open_project_rejects_an_unmanaged_directory(tmp_path: Path) -> None:
    (tmp_path / "hls4ml_config.yml").write_text("Backend: Vitis\n", encoding="utf-8")

    with pytest.raises(ProjectGenerationError, match="not a RAVEL project"):
        open_project(tmp_path)


def test_open_project_reconstructs_public_project_state(tmp_path: Path) -> None:
    manifest = _write_valid_project(tmp_path)

    project = open_project(tmp_path)

    assert project.path == tmp_path
    assert project.config["Profile"] == "aria"
    assert project.implementation_plan["temporal_pack"] == 2
    assert project.status == manifest["status"]


def test_open_project_marks_modified_managed_sources_and_stale_evidence(
    tmp_path: Path,
) -> None:
    _write_valid_project(tmp_path)
    manifest_before = (tmp_path / "ravel_manifest.json").read_bytes()
    (tmp_path / "firmware" / "aria_top.cpp").write_text(
        "void manually_edited() {}\n", encoding="utf-8"
    )

    project = open_project(tmp_path)

    assert project.status["source_integrity"] == "modified"
    assert project.status["correctness_verification"] == "stale"
    assert (tmp_path / "ravel_manifest.json").read_bytes() == manifest_before


def test_open_project_reports_a_corrupt_manifest_as_a_project_error(tmp_path: Path) -> None:
    (tmp_path / "ravel_manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(ProjectGenerationError, match="manifest"):
        open_project(tmp_path)


def test_open_project_rejects_an_unsupported_manifest_schema(tmp_path: Path) -> None:
    manifest = _write_valid_project(tmp_path)
    manifest["schema_version"] = 99
    (tmp_path / "ravel_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProjectGenerationError, match="schema_version"):
        open_project(tmp_path)


def test_inspect_cli_reports_project_status_as_json(tmp_path: Path) -> None:
    _write_valid_project(tmp_path)
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")

    result = subprocess.run(
        [sys.executable, "-m", "ravel_hls.cli", "inspect", str(tmp_path), "--json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["ravel"] == {"product": "RAVEL", "generation": "Aria", "release": "1.0"}
    assert report["status"]["source_integrity"] == "clean"
    assert result.stderr == ""


def test_inspect_cli_reports_expected_project_errors_without_traceback(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")

    result = subprocess.run(
        [sys.executable, "-m", "ravel_hls.cli", "inspect", str(tmp_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not a RAVEL project" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def _write_valid_project(project_path: Path) -> dict[str, object]:
    source = "void aria_top() {}\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    (project_path / "firmware").mkdir()
    (project_path / "firmware" / "aria_top.cpp").write_text(source, encoding="utf-8")
    (project_path / "hls4ml_config.yml").write_text("Backend: Vitis\n", encoding="utf-8")
    (project_path / "ravel_config.yml").write_text(
        "Profile: aria\nVerification:\n  Mode: required\n", encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "ravel": {"product": "RAVEL", "generation": "Aria", "release": "1.0"},
        "implementation_plan": {"template_profile": "aria-2x-v1", "temporal_pack": 2},
        "status": {
            "generation": "complete",
            "dependency_qualification": "qualified",
            "correctness_verification": "passed",
            "model_fidelity": "reported",
            "source_integrity": "clean",
            "performance_qualification": "not_run",
        },
        "managed_files": {"firmware/aria_top.cpp": source_hash},
    }
    (project_path / "ravel_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest
