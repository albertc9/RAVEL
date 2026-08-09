from pathlib import Path
import os
import shutil
import subprocess
import zipfile

import pytest


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
