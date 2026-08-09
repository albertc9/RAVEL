"""RAVEL generated-project access."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import RavelConfig
from .exceptions import ProjectGenerationError


@dataclass(frozen=True)
class RavelProject:
    """Read-only view of a generated RAVEL project."""

    path: Path
    config: RavelConfig
    manifest: dict[str, Any]

    @property
    def implementation_plan(self) -> dict[str, Any]:
        return self.manifest.get("implementation_plan", {})

    @property
    def status(self) -> dict[str, str]:
        status = dict(self.manifest.get("status", {}))
        managed_files = self.manifest.get("managed_files", {})
        integrity_clean = all(
            _file_sha256(self.path / relative_path) == expected_sha256
            for relative_path, expected_sha256 in managed_files.items()
        )
        if not integrity_clean:
            status["source_integrity"] = "modified"
            if status.get("correctness_verification") == "passed":
                status["correctness_verification"] = "stale"
        return status


def open_project(path: str | Path) -> RavelProject:
    """Open an existing RAVEL project without modifying it."""

    project_path = Path(path)
    manifest_path = project_path / "ravel_manifest.json"
    if not manifest_path.is_file():
        raise ProjectGenerationError(f"{project_path} is not a RAVEL project")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProjectGenerationError(
            f"Cannot read RAVEL project manifest at {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ProjectGenerationError(
            "RAVEL project manifest schema_version must be 1"
        )
    config_path = project_path / "ravel_config.yml"
    config = RavelConfig.from_yaml(config_path.read_text(encoding="utf-8"))
    return RavelProject(project_path, config, manifest)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
