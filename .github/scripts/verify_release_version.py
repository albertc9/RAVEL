#!/usr/bin/env python3
"""Verify that release artifacts use the version in the triggering Git tag."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import zipfile


def _metadata_version(payload: bytes, artifact: Path) -> str:
    metadata = BytesParser().parsebytes(payload)
    if metadata.get("Name") != "ravel-hls":
        raise ValueError(f"{artifact}: expected package name ravel-hls")
    version = metadata.get("Version")
    if not version:
        raise ValueError(f"{artifact}: package metadata has no Version")
    return version


def _wheel_version(artifact: Path) -> str:
    with zipfile.ZipFile(artifact) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"{artifact}: expected exactly one .dist-info/METADATA file"
            )
        return _metadata_version(archive.read(metadata_files[0]), artifact)


def _sdist_version(artifact: Path) -> str:
    with tarfile.open(artifact, "r:gz") as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if member.name.endswith("/PKG-INFO") and member.isfile()
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{artifact}: expected exactly one top-level PKG-INFO")
        extracted = archive.extractfile(metadata_files[0])
        if extracted is None:
            raise ValueError(f"{artifact}: cannot read PKG-INFO")
        return _metadata_version(extracted.read(), artifact)


def artifact_version(artifact: Path) -> str:
    if artifact.name.endswith(".whl"):
        return _wheel_version(artifact)
    if artifact.name.endswith(".tar.gz"):
        return _sdist_version(artifact)
    raise ValueError(f"{artifact}: unsupported distribution format")


def main(arguments: list[str]) -> int:
    if len(arguments) < 2:
        raise SystemExit("usage: verify_release_version.py TAG ARTIFACT [ARTIFACT ...]")
    tag, *artifact_names = arguments
    if not tag.startswith("v") or len(tag) == 1:
        raise SystemExit(f"release tag must have the form vVERSION, got {tag!r}")

    expected = tag[1:]
    try:
        versions = {
            artifact_version(Path(artifact_name)) for artifact_name in artifact_names
        }
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit(str(error)) from error

    if len(versions) != 1:
        raise SystemExit(
            "release artifacts disagree on package version: "
            + ", ".join(sorted(versions))
        )
    actual = versions.pop()
    if actual != expected:
        raise SystemExit(f"tag {tag} does not match package version {actual}")

    print(f"release version verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
