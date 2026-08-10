"""Public RAVEL exception categories."""


class RavelError(Exception):
    """Base class for expected RAVEL failures."""


class ConfigurationError(RavelError, ValueError):
    """Raised when RAVEL configuration is invalid."""


class CompatibilityError(RavelError):
    """Raised when an input or dependency is outside a qualified profile."""


class ProjectGenerationError(RavelError):
    """Raised when a project cannot be generated or opened safely."""


class BuildError(RavelError):
    """Raised when an explicitly requested vendor build cannot complete."""


class OptimizationError(RavelError):
    """Raised when an Aria transformation cannot be applied."""


class VerificationError(RavelError):
    """Raised when required project verification fails."""
