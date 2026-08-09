import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest


def test_installed_cli_reports_package_version(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for distribution conformance tests")

    repository = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheel"
    build_environment = os.environ.copy()
    build_environment.pop("PYTHONPATH", None)
    build_environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repository,
        env=build_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("ravel_hls-*.whl"))
    virtual_environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(virtual_environment)
    python = virtual_environment / "bin" / "python"
    runtime_environment = os.environ.copy()
    runtime_environment.pop("PYTHONPATH", None)
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", wheel],
        cwd=tmp_path,
        env=runtime_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    package_version = subprocess.check_output(
        [
            python,
            "-c",
            "from importlib.metadata import version; print(version('ravel-hls'))",
        ],
        cwd=tmp_path,
        env=runtime_environment,
        text=True,
    ).strip()
    result = subprocess.run(
        [virtual_environment / "bin" / "ravel-hls", "--version"],
        cwd=tmp_path,
        env=runtime_environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == f"ravel-hls {package_version} (RAVEL Aria 1.0)\n"
    assert result.stderr == ""
