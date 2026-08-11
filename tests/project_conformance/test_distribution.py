from pathlib import Path
import json
import os
import shutil
import subprocess
import tomllib
import zipfile

import pytest


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


def test_committed_hls4ml_configs_do_not_expose_generation_machine_paths() -> None:
    repository = Path(__file__).resolve().parents[2]

    for config_path in repository.rglob("hls4ml_config.yml"):
        config = config_path.read_text(encoding="utf-8")
        assert "/home/" not in config
        assert "/Users/" not in config


def test_public_namespace_exposes_only_the_aria_1_4_lifecycle() -> None:
    import ravel_hls

    assert {"convert", "Project"} <= set(ravel_hls.__all__)
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


def test_config_schema_describes_the_unified_aria_1_4_mapping() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_config.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["title"] == "RAVEL Aria 1.4.0 configuration"
    assert schema["required"] == ["Project", "HLS"]
    assert set(schema["properties"]) == {
        "Project",
        "HLS",
        "Optimization",
        "Verification",
        "Vitis",
    }
    assert schema["properties"]["Optimization"]["properties"] == {
        "TemporalPacking": {"enum": [2, 4], "default": 4},
        "DenseParallelism": {"enum": [1, 2], "default": 2},
    }
    assert schema["properties"]["Vitis"]["properties"]["Run"]["default"] is False


def test_manifest_schema_describes_the_v3_source_closure() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 3}
    assert schema["properties"]["ravel"]["properties"]["release"] == {
        "const": "1.4.0"
    }
    assert "source_closure" in schema["required"]
    assert "source_closure_sha256" in schema["required"]
    assert "managed_files" not in schema["properties"]


def test_parameter_package_schema_describes_portable_inference_state() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_parameters.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 1}
    assert {
        "frontend_contract",
        "entries",
        "compatibility_sha256",
        "parameter_state_sha256",
        "package_content_sha256",
    } <= set(schema["required"])


def test_qualification_schema_binds_v2_evidence_identity() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_qualification.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 2}
    assert {
        "manifest_sha256",
        "generation_fingerprint",
        "source_closure_sha256",
        "top",
    } <= set(schema["required"])
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
