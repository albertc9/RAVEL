"""Code-owned product-generation identity, independent of package build tags."""

ARIA_ID = "aria"
ARIA_VERSION = "1.7.0"


def aria_identity() -> dict[str, str]:
    """Return a fresh serialization of the current compiler-policy identity."""

    return {"id": ARIA_ID, "version": ARIA_VERSION}
