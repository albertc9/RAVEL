"""Deterministic generation-manifest construction."""

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .compatibility.dependencies import inspect_dependencies
from .config import AGGRESSIVE_SPECIALIZATION_POLICY, RavelConfig


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_generation_manifest(
    *,
    project_path: Path,
    hls_config: dict[str, Any],
    ravel_config: RavelConfig,
    semantic_model: dict[str, Any],
    implementation_plan: dict[str, Any],
    pass_records: list[dict[str, Any]],
    verification_report: dict[str, Any],
    interface_contract: dict[str, Any],
) -> dict[str, Any]:
    dependency_report = inspect_dependencies()
    recorded_configuration = {
        "hls4ml": hls_config,
        "ravel": ravel_config.to_dict(),
    }
    generation_configuration = {
        "hls4ml": hls_config,
        "ravel": {
            "Profile": "aria",
            "OptimizationPolicy": AGGRESSIVE_SPECIALIZATION_POLICY,
            "Optimization": ravel_config["Optimization"],
        },
    }
    configuration_sha256 = canonical_sha256(generation_configuration)
    semantic_model_sha256 = canonical_sha256(semantic_model)
    implementation_sha256 = canonical_sha256(
        {
            "plan": implementation_plan,
            "passes": pass_records,
            "compatibility_profile": "hls4ml-1.2.0-hgq2-0.1.7",
        }
    )
    generation_fingerprint = canonical_sha256(
        {
            "semantic_model_sha256": semantic_model_sha256,
            "configuration_sha256": configuration_sha256,
            "implementation_sha256": implementation_sha256,
        }
    )
    source_artifact = project_path / "keras_model.keras"
    source_closure = _build_source_closure(project_path)
    try:
        package_version = version("ravel-hls")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "schema_version": 2,
        "ravel": {
            "product": "RAVEL",
            "generation": "Aria",
            "release": "1.3.0",
            "package_version": package_version,
        },
        "source_model": {
            "source_artifact_sha256": (
                file_sha256(source_artifact) if source_artifact.is_file() else None
            ),
            "semantic_model_sha256": semantic_model_sha256,
            "facts": semantic_model["facts"],
        },
        "dependencies": dependency_report["dependencies"],
        "normalized_configuration": recorded_configuration,
        "generation_configuration": generation_configuration,
        "configuration_sha256": configuration_sha256,
        "profile": {"id": "aria", "version": 1},
        "implementation_plan": implementation_plan,
        "implementation_sha256": implementation_sha256,
        "pipeline": {
            "phases": [
                "ValidateProfile",
                "GenerateBaseline",
                "BuildIR",
                "ApplyAriaPasses",
                "RenderProject",
                "ValidateProject",
                "PromoteProject",
            ],
            "passes": pass_records,
        },
        "interfaces": interface_contract,
        "verification": verification_report,
        "status": {
            "generation": "complete",
            "dependency_qualification": dependency_report["dependency_qualification"],
            "correctness_verification": verification_report[
                "transformation_equivalence"
            ],
            "model_fidelity": verification_report["model_fidelity"],
            "source_integrity": "clean",
            "performance_qualification": "not_run",
        },
        "source_closure": source_closure,
        "source_closure_sha256": canonical_sha256(source_closure),
        "generation_fingerprint": generation_fingerprint,
    }


def _build_source_closure(project_path: Path) -> list[dict[str, Any]]:
    entries = []
    for path in _iter_source_files(project_path):
        relative = path.relative_to(project_path).as_posix()
        entries.append(
            {
                "role": _source_role(relative),
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return entries


def _iter_source_files(project_path: Path) -> list[Path]:
    paths = []
    for directory, child_directories, filenames in os.walk(
        project_path, topdown=True, followlinks=False
    ):
        child_directories[:] = sorted(
            name
            for name in child_directories
            if not name.startswith(".") and not name.endswith("_prj")
        )
        root = Path(directory)
        for filename in sorted(filenames):
            path = root / filename
            relative = path.relative_to(project_path).as_posix()
            if not _excluded_from_source_closure(relative) and path.is_file():
                paths.append(path)
    return sorted(paths)


def _excluded_from_source_closure(relative_path: str) -> bool:
    parts = relative_path.split("/")
    return (
        relative_path in {"ravel_manifest.json", "ravel_qualification.json"}
        or relative_path.endswith(".log")
        or any(part.endswith("_prj") for part in parts)
        or any(part.startswith(".") for part in parts)
    )


def _source_role(relative_path: str) -> str:
    if relative_path == "keras_model.keras":
        return "model"
    if relative_path.endswith((".yml", ".yaml", ".json")):
        return "configuration"
    if relative_path.endswith(".tcl"):
        return "vendor_script"
    if relative_path.startswith("firmware/weights/"):
        return "parameter"
    if relative_path.startswith("firmware/"):
        return "firmware"
    if relative_path.endswith((".cpp", ".h")):
        return "simulation"
    return "project"
