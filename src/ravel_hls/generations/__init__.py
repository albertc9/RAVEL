"""Built-in versioned RAVEL generations."""

from .aria import ARIA
from .registry import GenerationDefinition
from dataclasses import replace


# These generations share the retained legacy/native lowering implementations.
# Refresh supplies their recorded plans; their old selection policy is not rerun.
_BUILTIN_GENERATIONS = (ARIA, *(replace(ARIA, version=version) for version in ("1.7.1", "1.7.0", "1.6.2", "1.6.1", "1.6.0")))


def builtin_generation(generation_id: str, version: str) -> GenerationDefinition:
    """Resolve one closed built-in generation without mutable registration."""

    for generation in _BUILTIN_GENERATIONS:
        if (generation.id, generation.version) == (generation_id, version):
            return generation
    raise LookupError(f"unknown RAVEL generation: {generation_id} {version}")


__all__ = ["GenerationDefinition", "builtin_generation"]
