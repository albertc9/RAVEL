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
    from .api import convert_from_keras_model, optimize_project, refresh_model
    from .config import RavelConfig
    from .project import RavelProject, open_project

__all__ = [
    "CompatibilityError",
    "ConfigurationError",
    "OptimizationError",
    "ProjectGenerationError",
    "RavelConfig",
    "RavelError",
    "RavelProject",
    "VerificationError",
    "convert_from_keras_model",
    "open_project",
    "optimize_project",
    "refresh_model",
]


def __getattr__(name: str) -> Any:
    lazy_exports = {
        "RavelConfig": (".config", "RavelConfig"),
        "RavelProject": (".project", "RavelProject"),
        "convert_from_keras_model": (".api", "convert_from_keras_model"),
        "open_project": (".project", "open_project"),
        "optimize_project": (".api", "optimize_project"),
        "refresh_model": (".api", "refresh_model"),
    }
    target = lazy_exports.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
