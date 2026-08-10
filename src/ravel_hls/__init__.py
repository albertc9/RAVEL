"""Public Python API for RAVEL."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .exceptions import (
    CompatibilityError,
    ConfigurationError,
    OptimizationError,
    ProjectGenerationError,
    RavelError,
    VerificationError,
)

if TYPE_CHECKING:
    from .api import convert
    from .parameters import Parameters
    from .project import Project
    from .qualification.vitis import QualificationRecord

__all__ = [
    "CompatibilityError",
    "ConfigurationError",
    "OptimizationError",
    "Parameters",
    "Project",
    "ProjectGenerationError",
    "QualificationRecord",
    "RavelError",
    "VerificationError",
    "convert",
]


def __getattr__(name: str) -> Any:
    lazy_exports = {
        "Parameters": (".parameters", "Parameters"),
        "Project": (".project", "Project"),
        "QualificationRecord": (".qualification.vitis", "QualificationRecord"),
        "convert": (".api", "convert"),
    }
    target = lazy_exports.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
