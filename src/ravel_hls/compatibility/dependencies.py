"""Read-only dependency and host capability inspection."""

from importlib.metadata import PackageNotFoundError, distribution, version
import platform
from pathlib import Path
import shutil
import sys
from typing import Any


def inspect_dependencies() -> dict[str, Any]:
    """Report the dependency qualification facts currently implemented."""

    requirements = {
        "hls4ml": "==1.2.0",
        "hgq2": "==0.1.7",
        "keras": "==3.12.1",
        "numpy": "==1.26.4",
        "quantizers": "==1.2.2",
        "Jinja2": "==3.1.6",
        "PyYAML": "==6.0.3",
    }
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
    system = platform.system()
    platform_status = {
        "Linux": "full",
        "Darwin": "generation_only",
    }.get(system, "unsupported")
    compiler = next(
        (
            path
            for name in ("g++-16", "g++-15", "g++-14", "g++", "c++", "clang++")
            if (path := shutil.which(name))
        ),
        None,
    )
    header_path = _hls_simulation_header()
    return {
        "dependencies": dependencies,
        "dependency_qualification": "qualified" if qualified else "failed",
        "python": {
            "installed": platform.python_version(),
            "required": ">=3.10",
            "status": "qualified" if sys.version_info >= (3, 10) else "incompatible",
        },
        "platform": {"system": system, "status": platform_status},
        "compiler": {
            "command": compiler,
            "status": "available" if compiler is not None else "missing",
        },
        "hls_simulation_headers": {
            "path": str(header_path) if header_path is not None else None,
            "status": "available" if header_path is not None else "missing",
        },
    }


def _hls_simulation_header() -> Path | None:
    try:
        hls4ml_distribution = distribution("hls4ml")
    except PackageNotFoundError:
        return None
    candidate = Path(
        hls4ml_distribution.locate_file(
            "hls4ml/templates/vivado/ap_types/ap_fixed.h"
        )
    )
    return candidate if candidate.is_file() else None
