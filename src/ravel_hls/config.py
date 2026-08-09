"""RAVEL-owned configuration."""

from collections.abc import Iterator, Mapping
from typing import Any


class RavelConfig(Mapping[str, Any]):
    """Typed, mapping-compatible configuration for a RAVEL run."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {"Verification": {"Mode": "auto"}}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
