from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[2]
CHECKER = REPOSITORY / ".github" / "scripts" / "verify_release_version.py"


def _metadata(version: str) -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: ravel-hls\n"
        f"Version: {version}\n"
        "\n"
    ).encode()


def _wheel(tmp_path: Path, version: str) -> Path:
    wheel = tmp_path / f"ravel_hls-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"ravel_hls-{version}.dist-info/METADATA",
            _metadata(version),
        )
    return wheel


def _sdist(tmp_path: Path, version: str) -> Path:
    sdist = tmp_path / f"ravel_hls-{version}.tar.gz"
    payload = _metadata(version)
    info = tarfile.TarInfo(f"ravel_hls-{version}/PKG-INFO")
    info.size = len(payload)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))
    return sdist


def test_release_version_checker_accepts_matching_tag_and_artifacts(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "v1.2.2",
            str(_wheel(tmp_path, "1.2.2")),
            str(_sdist(tmp_path, "1.2.2")),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "release version verified: 1.2.2" in result.stdout


def test_release_version_checker_rejects_tag_package_mismatch(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "v1.2.2",
            str(_wheel(tmp_path, "1.1.0")),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "tag v1.2.2 does not match package version 1.1.0" in result.stderr
