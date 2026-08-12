from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from ravel_hls import (
    BuildError,
    Project,
    ProjectGenerationError,
    VerificationError,
    convert,
)


MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
    / "adam_p1_step2"
    / "adam_p1_step2_best.keras"
)


@pytest.fixture(scope="module")
def generated_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("lifecycle") / "aria_template"
    return convert(
        MODEL,
        output,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "disabled"},
        },
    ).path


@pytest.fixture
def generated_project(tmp_path: Path, generated_template: Path) -> Project:
    output = tmp_path / "aria_project"
    shutil.copytree(generated_template, output)
    return Project.open(output)


def test_project_build_runs_vitis_in_place_and_records_only_success(
    generated_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    invocation: dict[str, Any] = {}
    record = object()
    monkeypatch.setattr(
        "ravel_hls.project.shutil.which",
        lambda command: "/opt/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        invocation.update({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, 0, "synthesis complete\n", "")

    monkeypatch.setattr("ravel_hls.project.subprocess.run", fake_run)
    monkeypatch.setattr(Project, "record", lambda self, report_dir: record)

    assert generated_project.build() is record
    assert invocation == {
        "args": [
            "/opt/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
            "-f",
            "build_prj.tcl",
        ],
        "cwd": generated_project.path,
        "check": False,
        "capture_output": True,
        "text": True,
        "shell": False,
    }
    assert (generated_project.path / "ravel_vitis.log").read_text() == (
        "synthesis complete\n"
    )


def test_project_build_failure_keeps_a_bounded_log_without_qualification(
    generated_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ravel_hls.project.shutil.which", lambda command: "/tools/vitis_hls"
    )
    monkeypatch.setattr(
        "ravel_hls.project.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, "starting\n", "synthesis failed\n"
        ),
    )
    recorded: list[Path] = []
    monkeypatch.setattr(
        Project, "record", lambda self, path: recorded.append(Path(path))
    )

    with pytest.raises(BuildError, match="exit code 1"):
        generated_project.build()

    assert (generated_project.path / "ravel_vitis.log").read_text() == (
        "starting\nsynthesis failed\n"
    )
    assert recorded == []
    assert not (generated_project.path / "ravel_qualification.json").exists()


def test_project_build_rejects_missing_tools_and_modified_sources(
    generated_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ravel_hls.project.shutil.which", lambda command: None)
    with pytest.raises(BuildError, match="Cannot find.*vitis_hls"):
        generated_project.build()

    top = generated_project.manifest["normalized_configuration"]["hls4ml"][
        "ProjectName"
    ]
    (generated_project.path / "firmware" / f"{top}.cpp").write_text(
        "void modified() {}\n", encoding="utf-8"
    )
    with pytest.raises(VerificationError, match="modified RAVEL project"):
        generated_project.build()


def test_conversion_requires_force_for_an_unrecognized_target(
    tmp_path: Path,
) -> None:
    output = tmp_path / "unmanaged_target"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unmanaged\n", encoding="utf-8")
    config = {
        "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
        "Verification": {"Mode": "disabled"},
    }

    with pytest.raises(ProjectGenerationError, match="unrecognized target"):
        convert(MODEL, output, config)
    assert marker.is_file()

    forced = convert(
        MODEL,
        output,
        {**config, "Project": {"ForceReplace": True}},
    )
    assert forced.path == output
    assert not marker.exists()


def test_failed_regeneration_preserves_the_previous_project_atomically(
    generated_project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_manifest = (
        generated_project.path / "ravel_manifest.json"
    ).read_bytes()

    class _FailingBinding:
        def render(self, *args: object) -> list[str]:
            raise ProjectGenerationError("synthetic renderer failure")

    class _FailingGeneration:
        def backend_binding(self, *args: object) -> _FailingBinding:
            return _FailingBinding()

    monkeypatch.setattr(
        "ravel_hls.api.builtin_generation",
        lambda generation_id, version: _FailingGeneration(),
    )

    with pytest.raises(ProjectGenerationError, match="synthetic renderer failure"):
        convert(
            MODEL,
            generated_project.path,
            {
                "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
                "Verification": {"Mode": "disabled"},
            },
        )

    assert (generated_project.path / "ravel_manifest.json").read_bytes() == (
        original_manifest
    )
    assert Project.open(generated_project.path).status["source_integrity"] == "clean"
    assert not list(
        generated_project.path.parent.glob(
            f".{generated_project.path.name}.ravel-*"
        )
    )


def test_generation_writes_only_explicitly_selected_vitis_stages(
    tmp_path: Path,
) -> None:
    project = convert(
        MODEL,
        tmp_path / "aria_stages",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "disabled"},
            "Vitis": {
                "Run": False,
                "Stages": {"Synth": False, "CoSim": True},
            },
        },
    )

    options = (project.path / "build_opt.tcl").read_text(encoding="utf-8")
    assert "synth      0" in options
    assert "cosim      1" in options
    assert not (project.path / "ravel_vitis.log").exists()
