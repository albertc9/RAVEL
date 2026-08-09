"""RAVEL-owned configuration."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any

import yaml

from .exceptions import ConfigurationError


class RavelConfig(Mapping[str, Any]):
    """Typed, mapping-compatible configuration for a RAVEL run."""

    @classmethod
    def from_yaml(cls, text: str) -> "RavelConfig":
        """Construct a validated configuration from YAML text."""

        try:
            values = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigurationError(f"Invalid RAVEL configuration YAML: {error}") from error
        if values is not None and not isinstance(values, Mapping):
            raise ConfigurationError("RAVEL configuration YAML must contain a mapping")
        return cls(values)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(values or {})
        unknown_fields = sorted(self._data.keys() - {"Profile", "Verification"})
        if unknown_fields:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: {', '.join(unknown_fields)}"
            )
        verification_values = self._data.get("Verification", {})
        if not isinstance(verification_values, Mapping):
            raise ConfigurationError("Verification must be a mapping")
        unknown_verification_fields = sorted(
            verification_values.keys() - {"Mode", "Samples", "Seed"}
        )
        if unknown_verification_fields:
            field = unknown_verification_fields[0]
            raise ConfigurationError(f"Unknown RAVEL configuration field: Verification.{field}")
        verification = {"Mode": "auto"}
        verification.update(verification_values)
        if verification["Mode"] not in {"auto", "required", "disabled"}:
            raise ConfigurationError(
                "Verification.Mode must be one of: auto, required, disabled"
            )
        samples = verification.get("Samples")
        if samples is not None and (
            not isinstance(samples, int) or isinstance(samples, bool) or samples < 1
        ):
            raise ConfigurationError("Verification.Samples must be a positive integer")
        seed = verification.get("Seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ConfigurationError("Verification.Seed must be a nonnegative integer")
        self._data["Verification"] = verification

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent dictionary representation."""

        return deepcopy(self._data)

    def to_yaml(self) -> str:
        """Serialize the configuration using stable field ordering."""

        return yaml.safe_dump(self.to_dict(), sort_keys=False)
