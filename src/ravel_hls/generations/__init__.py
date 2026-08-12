"""Built-in versioned RAVEL generations."""

from .aria import ARIA_1_5_1
from .registry import GenerationDefinition


_BUILTIN_GENERATIONS = (ARIA_1_5_1,)


def builtin_generation(generation_id: str, version: str) -> GenerationDefinition:
    """Resolve one closed built-in generation without mutable registration."""

    for generation in _BUILTIN_GENERATIONS:
        if (generation.id, generation.version) == (generation_id, version):
            return generation
    raise LookupError(f"unknown RAVEL generation: {generation_id} {version}")


__all__ = ["GenerationDefinition", "builtin_generation"]
