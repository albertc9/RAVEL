"""Read-only dependency compatibility inspection."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any


def inspect_dependencies() -> dict[str, Any]:
    """Report the dependency qualification facts currently implemented."""

    required_version = "==1.2.0"
    try:
        installed_version = version("hls4ml")
    except PackageNotFoundError:
        installed_version = None

    status = "missing" if installed_version is None else "qualified"
    return {
        "dependencies": {
            "hls4ml": {
                "installed": installed_version,
                "required": required_version,
                "status": status,
            }
        },
        "dependency_qualification": "failed" if status == "missing" else "qualified",
    }
