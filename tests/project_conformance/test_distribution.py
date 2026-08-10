from pathlib import Path
import json
import os
import shutil
import subprocess
import zipfile

import pytest


def test_public_namespace_exposes_only_the_aria_1_1_lifecycle() -> None:
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


def test_config_schema_describes_the_unified_aria_1_1_mapping() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_config.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["title"] == "RAVEL Aria 1.1 configuration"
    assert schema["required"] == ["Project", "HLS"]
    assert set(schema["properties"]) == {
        "Project",
        "HLS",
        "Verification",
        "Vitis",
    }
    assert schema["properties"]["Vitis"]["properties"]["Run"]["default"] is False


def test_manifest_schema_describes_the_v2_source_closure() -> None:
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads(
        (repository / "src/ravel_hls/schemas/ravel_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"] == {"const": 2}
    assert "source_closure" in schema["required"]
    assert "source_closure_sha256" in schema["required"]
    assert "managed_files" not in schema["properties"]


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
    assert "ravel_hls/schemas/ravel_qualification.schema.json" in names
    assert 'Requires-Dist: tensorflow-cpu==2.20.0; platform_system == "Linux"' in (
        metadata.splitlines()
    )
    assert "Provides-Extra: reference" not in metadata
