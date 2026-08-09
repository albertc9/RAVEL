from ravel_hls import (
    CompatibilityError,
    ConfigurationError,
    OptimizationError,
    ProjectGenerationError,
    RavelError,
    VerificationError,
)


def test_public_failures_share_a_ravel_base_exception() -> None:
    assert issubclass(ConfigurationError, RavelError)
    assert issubclass(CompatibilityError, RavelError)
    assert issubclass(ProjectGenerationError, RavelError)
    assert issubclass(OptimizationError, RavelError)
    assert issubclass(VerificationError, RavelError)
