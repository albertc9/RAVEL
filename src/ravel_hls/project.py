"""RAVEL generated-project access."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import RavelConfig
from .exceptions import ProjectGenerationError
from .manifest import _build_source_closure


@dataclass(frozen=True)
class Project:
    """Read-only view of a generated RAVEL project."""

    path: Path
    config: RavelConfig
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        """Open an existing RAVEL project without modifying it."""

        return open_project(path)

    @property
    def implementation_plan(self) -> dict[str, Any]:
        return self.manifest.get("implementation_plan", {})

    @property
    def status(self) -> dict[str, str]:
        return self._status(check_integrity=True)

    def _status(self, *, check_integrity: bool) -> dict[str, str]:
        status = dict(self.manifest.get("status", {}))
        if not check_integrity:
            status["source_integrity"] = "not_checked"
            if _qualification_matches_manifest(self.path):
                status["performance_qualification"] = "recorded"
            return status
        source_closure = self.manifest.get("source_closure")
        if isinstance(source_closure, list):
            integrity_clean = _build_source_closure(self.path) == source_closure
        else:
            managed_files = self.manifest.get("managed_files", {})
            integrity_clean = all(
                _file_sha256(self.path / relative_path) == expected_sha256
                for relative_path, expected_sha256 in managed_files.items()
            )
        if not integrity_clean:
            status["source_integrity"] = "modified"
            if status.get("correctness_verification") == "passed":
                status["correctness_verification"] = "stale"
            if (self.path / "ravel_qualification.json").is_file():
                status["performance_qualification"] = "stale"
        elif _qualification_matches_manifest(self.path):
            status["performance_qualification"] = "recorded"
        return status

    def link(self) -> Any:
        """Return hls4ml's restricted existing-project compile/predict/build view."""

        from hls4ml.utils.link import FilesystemModelGraph

        return FilesystemModelGraph(self.path)

    link_hls4ml = link

    def refresh(self, model: Any) -> "Project":
        """Regenerate this project with a new compatible model."""

        from .api import refresh_model

        return refresh_model(self, model)


RavelProject = Project


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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {1, 2}:
        raise ProjectGenerationError(
            "RAVEL project manifest schema_version must be 1 or 2"
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


def _qualification_matches_manifest(project_path: Path) -> bool:
    qualification_path = project_path / "ravel_qualification.json"
    if not qualification_path.is_file():
        return False
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return qualification.get("manifest_sha256") == _file_sha256(
        project_path / "ravel_manifest.json"
    )
