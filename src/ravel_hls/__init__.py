"""Public Python API for RAVEL."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .exceptions import (
    BuildError,
    CompatibilityError,
    ConfigurationError,
    OptimizationError,
    ProjectGenerationError,
    RavelError,
    VerificationError,
)

if TYPE_CHECKING:
    from .analysis.model import ModelAnalysis, analyze
    from .api import convert, refresh
    from .parameters import Parameters
    from .project import Project
    from .qualification.vitis import QualificationRecord

__all__ = [
    "BuildError",
    "CompatibilityError",
    "ConfigurationError",
    "ModelAnalysis",
    "OptimizationError",
    "Parameters",
    "Project",
    "ProjectGenerationError",
    "QualificationRecord",
    "RavelError",
    "VerificationError",
    "analyze",
    "convert",
    "refresh",
]


def __getattr__(name: str) -> Any:
    lazy_exports = {
        "ModelAnalysis": (".analysis.model", "ModelAnalysis"),
        "Parameters": (".parameters", "Parameters"),
        "Project": (".project", "Project"),
        "QualificationRecord": (".qualification.vitis", "QualificationRecord"),
        "analyze": (".analysis.model", "analyze"),
        "convert": (".api", "convert"),
        "refresh": (".api", "refresh"),
    }
    target = lazy_exports.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
