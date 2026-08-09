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

    if installed_version is None:
        status = "missing"
    elif installed_version == required_version.removeprefix("=="):
        status = "qualified"
    else:
        status = "incompatible"
    return {
        "dependencies": {
            "hls4ml": {
                "installed": installed_version,
                "required": required_version,
                "status": status,
            }
        },
        "dependency_qualification": "qualified" if status == "qualified" else "failed",
    }
