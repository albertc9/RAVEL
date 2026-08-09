import json
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


def test_doctor_reports_missing_required_dependency(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for distribution conformance tests")

    repository = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheel"
    isolated_environment = os.environ.copy()
    isolated_environment.pop("PYTHONPATH", None)
    isolated_environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repository,
        env=isolated_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("ravel_hls-*.whl"))
    virtual_environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(virtual_environment)
    python = virtual_environment / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", wheel],
        cwd=tmp_path,
        env=isolated_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [virtual_environment / "bin" / "ravel-hls", "doctor", "--json"],
        cwd=tmp_path,
        env=isolated_environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["dependencies"]["hls4ml"] == {
        "installed": None,
        "required": "==1.2.0",
        "status": "missing",
    }
    assert report["dependency_qualification"] == "failed"
    assert result.stderr == ""


def test_doctor_reports_incompatible_dependency_version(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for distribution conformance tests")

    repository = Path(__file__).resolve().parents[2]
    wheel_dir = tmp_path / "wheel"
    isolated_environment = os.environ.copy()
    isolated_environment.pop("PYTHONPATH", None)
    isolated_environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=repository,
        env=isolated_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("ravel_hls-*.whl"))
    virtual_environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(virtual_environment)
    python = virtual_environment / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", wheel],
        cwd=tmp_path,
        env=isolated_environment,
        check=True,
        capture_output=True,
        text=True,
    )

    site_packages = Path(
        subprocess.check_output(
            [
                python,
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            cwd=tmp_path,
            env=isolated_environment,
            text=True,
        ).strip()
    )
    distribution = site_packages / "hls4ml-9.9.9.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: hls4ml\nVersion: 9.9.9\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [virtual_environment / "bin" / "ravel-hls", "doctor", "--json"],
        cwd=tmp_path,
        env=isolated_environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["dependencies"]["hls4ml"] == {
        "installed": "9.9.9",
        "required": "==1.2.0",
        "status": "incompatible",
    }
    assert report["dependency_qualification"] == "failed"
    assert result.stderr == ""


def test_doctor_requires_hgq2_alongside_hls4ml(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    distributions = tmp_path / "distributions"
    _write_distribution(distributions, "ravel-hls", "1.0.0.dev0")
    _write_distribution(distributions, "hls4ml", "1.2.0")
    isolated_environment = os.environ.copy()
    isolated_environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(distributions)]
    )

    result = subprocess.run(
        [sys.executable, "-S", "-m", "ravel_hls.cli", "doctor", "--json"],
        cwd=tmp_path,
        env=isolated_environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["dependencies"]["hgq2"] == {
        "installed": None,
        "required": "==0.1.7",
        "status": "missing",
    }
    assert report["dependencies"]["hls4ml"]["status"] == "qualified"
    assert report["dependency_qualification"] == "failed"
    assert result.stderr == ""


def test_doctor_rejects_legacy_hgq_namespace_conflict(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    distributions = tmp_path / "distributions"
    _write_distribution(distributions, "ravel-hls", "1.0.0.dev0")
    _write_distribution(distributions, "hls4ml", "1.2.0")
    _write_distribution(distributions, "hgq2", "0.1.7")
    _write_distribution(distributions, "HGQ", "0.2.6")
    isolated_environment = os.environ.copy()
    isolated_environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), str(distributions)]
    )

    result = subprocess.run(
        [sys.executable, "-S", "-m", "ravel_hls.cli", "doctor", "--json"],
        cwd=tmp_path,
        env=isolated_environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["dependencies"]["hgq"] == {
        "installed": "0.2.6",
        "required": "not installed",
        "status": "conflict",
    }
    assert report["dependencies"]["hgq2"]["status"] == "qualified"
    assert report["dependency_qualification"] == "failed"
    assert result.stderr == ""


def test_doctor_reports_runtime_and_simulation_capabilities(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")

    result = subprocess.run(
        [sys.executable, "-m", "ravel_hls.cli", "doctor", "--json"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["python"]["required"] == ">=3.10"
    assert report["python"]["status"] == "qualified"
    assert report["platform"]["status"] in {"full", "generation_only"}
    assert report["compiler"]["status"] in {"available", "missing"}
    assert report["hls_simulation_headers"]["status"] in {"available", "missing"}


def _write_distribution(root: Path, name: str, package_version: str) -> None:
    distribution = root / f"{name.replace('-', '_')}-{package_version}.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {package_version}\n",
        encoding="utf-8",
    )
