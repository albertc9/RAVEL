"""RAVEL generated-project access."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .config import RavelConfig
from .exceptions import BuildError, ProjectGenerationError, VerificationError
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
            if _qualification_matches_manifest(self.path, self.manifest):
                status["performance_qualification"] = "recorded"
            elif (self.path / "ravel_qualification.json").is_file():
                status["performance_qualification"] = "stale"
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
        elif _qualification_matches_manifest(self.path, self.manifest):
            status["performance_qualification"] = "recorded"
        elif (self.path / "ravel_qualification.json").is_file():
            status["performance_qualification"] = "stale"
        return status

    def link(self) -> Any:
        """Return hls4ml's restricted existing-project compile/predict/build view."""

        from hls4ml.utils.link import FilesystemModelGraph

        return FilesystemModelGraph(self.path)

    link_hls4ml = link

    def refresh(
        self, model: Any, *, verification_inputs: Any | None = None
    ) -> "Project":
        """Regenerate this project with a new compatible model."""

        from .api import refresh

        return refresh(self, model, verification_inputs=verification_inputs)

    def record(self, report_dir: str | Path) -> Any:
        """Attach measured Vitis HLS evidence without launching the tool."""

        from .qualification.vitis import import_vitis_reports

        return import_vitis_reports(self, report_dir=report_dir)

    def build(self) -> Any:
        """Run Vitis HLS for this project and attach its synthesis measurements."""

        if self.manifest.get("schema_version") not in {2, 3, 4}:
            raise BuildError(
                "Vitis builds require a schema-v2, schema-v3, or schema-v4 RAVEL project"
            )
        if self.status.get("source_integrity") != "clean":
            raise VerificationError(
                "Cannot build a modified RAVEL project; regenerate or restore sources"
            )
        build_script = self.path / "build_prj.tcl"
        if not build_script.is_file():
            raise BuildError(f"Vitis build script does not exist: {build_script}")
        launcher = shutil.which("vitis_hls")
        if launcher is None:
            raise BuildError("Cannot find the Vitis HLS 2023.2 launcher: vitis_hls")
        command = [launcher, "-f", "build_prj.tcl"]
        try:
            completed = subprocess.run(
                command,
                cwd=self.path,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
            )
        except OSError as error:
            raise BuildError(f"Cannot launch Vitis HLS: {error}") from error
        log_path = self.path / "ravel_vitis.log"
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise BuildError(
                f"Vitis HLS failed with exit code {completed.returncode}; "
                f"see {log_path}"
            )
        return self.record(self.path)


RavelProject = Project


def open_project(path: str | Path) -> RavelProject:
    """Open an existing RAVEL project without modifying it."""

    import yaml

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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
        1,
        2,
        3,
        4,
    }:
        raise ProjectGenerationError(
            "RAVEL project manifest schema_version must be 1, 2, 3, or 4"
        )
    implementation_plan = manifest.get("implementation_plan")
    if (
        manifest["schema_version"] in {1, 2}
        and isinstance(implementation_plan, dict)
        and "weight_delivery" not in implementation_plan
    ):
        manifest = dict(manifest)
        manifest["implementation_plan"] = {
            **implementation_plan,
            "weight_delivery": {"id": "complete-partition", "version": 1},
        }
    config_path = project_path / "ravel_config.yml"
    config_text = config_path.read_text(encoding="utf-8")
    config = RavelConfig.from_yaml(config_text)
    recorded_config = yaml.safe_load(config_text)
    if isinstance(recorded_config, dict) and "Optimization" not in recorded_config:
        implementation_plan = manifest.get("implementation_plan", {})
        temporal_pack = implementation_plan.get("temporal_pack")
        if temporal_pack in {2, 4}:
            dense_parallelism = implementation_plan.get(
                "dense_parallelism", 1 if temporal_pack == 2 else 2
            )
            if dense_parallelism in {1, 2}:
                migrated_config = config.to_dict()
                migrated_config["Optimization"] = {
                    "TemporalPacking": temporal_pack,
                    "DenseParallelism": dense_parallelism,
                }
                config = RavelConfig(migrated_config)
    return RavelProject(project_path, config, manifest)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualification_matches_manifest(
    project_path: Path, manifest: dict[str, Any]
) -> bool:
    qualification_path = project_path / "ravel_qualification.json"
    if not qualification_path.is_file():
        return False
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected_top = (
        manifest.get("normalized_configuration", {})
        .get("hls4ml", {})
        .get("ProjectName")
    )
    return (
        qualification.get("schema_version") == 2
        and qualification.get("status") == "recorded"
        and qualification.get("manifest_sha256")
        == _file_sha256(project_path / "ravel_manifest.json")
        and qualification.get("generation_fingerprint")
        == manifest.get("generation_fingerprint")
        and qualification.get("source_closure_sha256")
        == manifest.get("source_closure_sha256")
        and qualification.get("top") == expected_top
    )
