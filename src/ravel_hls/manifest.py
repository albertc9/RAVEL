"""Deterministic generation-manifest construction."""

from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
from typing import Any

from .compatibility.dependencies import inspect_dependencies
from .config import RavelConfig


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
    managed_paths: list[str],
    verification_report: dict[str, Any],
    interface_contract: dict[str, Any],
) -> dict[str, Any]:
    dependency_report = inspect_dependencies()
    normalized_configuration = {
        "hls4ml": hls_config,
        "ravel": ravel_config.to_dict(),
    }
    configuration_sha256 = canonical_sha256(normalized_configuration)
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
    try:
        package_version = version("ravel-hls")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "schema_version": 1,
        "ravel": {
            "product": "RAVEL",
            "generation": "Aria",
            "release": "1.0",
            "package_version": package_version,
        },
        "source_model": {
            "source_artifact_sha256": (
                file_sha256(source_artifact) if source_artifact.is_file() else None
            ),
            "semantic_model_sha256": semantic_model_sha256,
        },
        "dependencies": dependency_report["dependencies"],
        "normalized_configuration": normalized_configuration,
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
        "managed_files": {
            relative_path: file_sha256(project_path / relative_path)
            for relative_path in sorted(managed_paths)
        },
        "generation_fingerprint": generation_fingerprint,
    }
