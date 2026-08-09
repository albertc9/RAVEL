"""Read-only dependency compatibility inspection."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any


def inspect_dependencies() -> dict[str, Any]:
    """Report the dependency qualification facts currently implemented."""

    requirements = {"hls4ml": "==1.2.0", "hgq2": "==0.1.7"}
    dependencies: dict[str, dict[str, str | None]] = {}
    for name, required_version in requirements.items():
        try:
            installed_version = version(name)
        except PackageNotFoundError:
            installed_version = None

        if installed_version is None:
            status = "missing"
        elif installed_version == required_version.removeprefix("=="):
            status = "qualified"
        else:
            status = "incompatible"
        dependencies[name] = {
            "installed": installed_version,
            "required": required_version,
            "status": status,
        }

    try:
        legacy_hgq_version = version("HGQ")
    except PackageNotFoundError:
        legacy_hgq_version = None
    if legacy_hgq_version is not None:
        dependencies["hgq"] = {
            "installed": legacy_hgq_version,
            "required": "not installed",
            "status": "conflict",
        }

    qualified = all(facts["status"] == "qualified" for facts in dependencies.values())
    return {
        "dependencies": dependencies,
        "dependency_qualification": "qualified" if qualified else "failed",
    }
