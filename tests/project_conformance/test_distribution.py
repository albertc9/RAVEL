from pathlib import Path
import json
import os
import shutil
import subprocess
import tomllib
import zipfile

import pytest
import yaml


def test_distribution_version_is_derived_from_git_tags() -> None:
    repository = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" not in metadata["project"]
    assert "version" in metadata["project"]["dynamic"]
    assert any(
        requirement.startswith("setuptools-scm")
        for requirement in metadata["build-system"]["requires"]
    )
    assert metadata["tool"]["setuptools_scm"]["version_scheme"] == "guess-next-dev"
    assert metadata["tool"]["setuptools_scm"]["version_file"] == (
        "src/ravel_hls/_version.py"
    )


def test_release_build_uses_the_dedicated_runner_without_moving_pypi_publish() -> None:
    repository = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (repository / ".github/workflows/publish.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    assert "workflow_dispatch" in workflow["on"]
    assert workflow["jobs"]["build"]["runs-on"] == [
        "self-hosted",
        "linux",
        "x64",
        "ravel-release",
    ]
    assert workflow["jobs"]["build"]["env"] == {
        "TMPDIR": "${{ github.workspace }}/.runner-temp"
    }
    build_steps = workflow["jobs"]["build"]["steps"]
    assert not any(
        step.get("uses", "").startswith("actions/setup-python@")
        for step in build_steps
    )
    uv_setup = next(
        step
        for step in build_steps
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    )
    assert uv_setup == {
        "name": "Set up Python 3.11 with uv",
        "uses": (
            "astral-sh/setup-uv@"
            "c771a70e6277c0a99b617c7a806ffedaca235ff9"
        ),
        "with": {
            "version": "0.11.32",
            "python-version": "3.11",
            "activate-environment": "true",
            "enable-cache": "true",
            "cache-python": "true",
            "prune-cache": "true",
        },
    }
    dependency_install = next(
        step
        for step in build_steps
        if step.get("name") == "Install build and test dependencies"
    )
    assert dependency_install["run"] == (
        "uv pip install --upgrade build twine '.[test]'"
    )
    test_step = next(
        step for step in build_steps if step.get("name") == "Run tests"
    )
    assert test_step["run"] == (
        'mkdir -p "$TMPDIR"\npython .github/scripts/run_release_tests.py\n'
    )
    version_check = next(
        step
        for step in build_steps
        if step.get("name") == "Verify tag matches package version"
    )
    assert version_check["if"] == (
        "${{ startsWith(github.ref, 'refs/tags/v') }}"
    )
    assert workflow["jobs"]["publish"]["runs-on"] == "ubuntu-latest"
    assert workflow["jobs"]["publish"]["if"] == (
        "${{ startsWith(github.ref, 'refs/tags/v') }}"
    )


def test_committed_hls4ml_configs_do_not_expose_generation_machine_paths() -> None:
    repository = Path(__file__).resolve().parents[2]

    for config_path in repository.rglob("hls4ml_config.yml"):
        config = config_path.read_text(encoding="utf-8")
        assert "/home/" not in config
        assert "/Users/" not in config


def test_public_namespace_exposes_only_the_aria_1_5_lifecycle() -> None:
    import ravel_hls

    assert {"analyze", "convert", "refresh", "Parameters", "Project"} <= set(
        ravel_hls.__all__
    )
    for removed_name in (
        "RavelConfig",
        "RavelProject",
        "convert_from_keras_model",
        "optimize_project",
        "refresh_model",
        "open_project",
        "import_vitis_reports",
    ):
        assert removed_name not in ravel_hls.__all__
        assert not hasattr(ravel_hls, removed_name)


def test_removed_internal_generation_paths_are_absent() -> None:
    repository = Path(__file__).resolve().parents[2]

    assert not (repository / "src/ravel_hls/backends/vitis/renderer.py").exists()
    assert not (repository / "src/ravel_hls/compatibility/model_profile.py").exists()

    api = (repository / "src/ravel_hls/api.py").read_text(encoding="utf-8")
    for removed_name in (
        "convert_from_keras_model",
        "optimize_project",
        "refresh_model",
        "build_pass_records",
    ):
        assert f"def {removed_name}(" not in api


def test_config_schema_describes_the_unified_aria_1_5_mapping() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_config.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["title"] == "RAVEL Aria 1.5.1 configuration"
    assert schema["required"] == ["HLS"]
    assert set(schema["properties"]) == {
        "Project",
        "HLS",
        "Optimization",
        "Verification",
        "Vitis",
    }
    assert schema["properties"]["Optimization"]["properties"] == {
        "TemporalPacking": {"enum": [2, 4, 8], "default": 8},
        "DenseParallelism": {"enum": [1, 2, 4], "default": 4},
    }
    assert len(schema["properties"]["Optimization"]["allOf"]) == 2
    assert schema["properties"]["Vitis"]["properties"]["Run"]["default"] is False


def test_manifest_schema_describes_the_v5_envelope_and_realization() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 5}
    assert schema["properties"]["ravel"]["properties"]["release"] == {
        "const": "1.5.1"
    }
    assert "source_closure" in schema["required"]
    assert "source_closure_sha256" in schema["required"]
    assert "managed_files" not in schema["properties"]
    assert "resolved_design" in schema["required"]
    assert "architecture_contract_sha256" in schema["required"]


def test_parameter_package_schema_describes_portable_inference_state() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_parameters.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 2}
    assert {
        "format",
        "model_family",
        "model_structure_sha256",
        "model_facts",
        "entries",
        "compatibility_sha256",
        "parameter_state_sha256",
        "package_content_sha256",
    } <= set(schema["required"])


def test_qualification_schema_binds_v3_evidence_identity_and_stages() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_qualification.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 4}
    assert {
        "manifest_sha256",
        "generation_fingerprint",
        "source_closure_sha256",
        "top",
        "rtl_cosimulation",
        "stages",
    } <= set(schema["required"])
    assert schema["properties"]["rtl_cosimulation"] == {
        "enum": ["not_run", "passed"]
    }
    assert schema["properties"]["status"] == {"const": "recorded"}


def test_wheel_contains_aria_rendering_templates(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for distribution conformance tests")
    repository = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheel"
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")

    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("ravel_hls-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")
    assert "ravel_hls/backends/vitis/templates/aria/firmware/top.cpp.j2" in names
    assert "ravel_hls/backends/vitis/templates/aria/bridge/bridge.cpp.j2" in names
    assert "ravel_hls/backends/vitis/templates/aria/testbench/test.cpp.j2" in names
    assert "ravel_hls/schemas/ravel_config.schema.json" in names
    assert "ravel_hls/schemas/ravel_manifest.schema.json" in names
    assert "ravel_hls/schemas/ravel_parameters.schema.json" in names
    assert "ravel_hls/schemas/ravel_qualification.schema.json" in names
    assert 'Requires-Dist: tensorflow-cpu==2.20.0; platform_system == "Linux"' in (
        metadata.splitlines()
    )
    assert "Provides-Extra: reference" not in metadata
