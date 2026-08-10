"""RAVEL-owned configuration."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any

from .exceptions import ConfigurationError


class RavelConfig(Mapping[str, Any]):
    """Typed, mapping-compatible configuration for a RAVEL run."""

    @classmethod
    def from_yaml(cls, text: str) -> "RavelConfig":
        """Construct a validated configuration from YAML text."""

        import yaml

        try:
            values = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ConfigurationError(f"Invalid RAVEL configuration YAML: {error}") from error
        if values is not None and not isinstance(values, Mapping):
            raise ConfigurationError("RAVEL configuration YAML must contain a mapping")
        return cls(values)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(values or {})
        if self._data.keys() & {"Project", "HLS", "Vitis"}:
            self._init_run_config()
            return
        unknown_fields = sorted(self._data.keys() - {"Profile", "Verification"})
        if unknown_fields:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: {', '.join(unknown_fields)}"
            )
        if "Profile" in self._data and self._data["Profile"] != "aria":
            raise ConfigurationError("Profile must be aria for RAVEL Aria 1.0")
        self._data.setdefault("Profile", "aria")
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

    def _init_run_config(self) -> None:
        unknown_fields = sorted(
            self._data.keys() - {"Project", "HLS", "Verification", "Vitis"}
        )
        if unknown_fields:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: {unknown_fields[0]}"
            )
        project = self._data.get("Project")
        hls = self._data.get("HLS")
        if not isinstance(project, Mapping):
            raise ConfigurationError("Project must be a mapping")
        if not isinstance(hls, Mapping):
            raise ConfigurationError("HLS must be a mapping")
        verification = self._data.get("Verification", {})
        if not isinstance(verification, Mapping):
            raise ConfigurationError("Verification must be a mapping")
        vitis = self._data.get("Vitis", {})
        if not isinstance(vitis, Mapping):
            raise ConfigurationError("Vitis must be a mapping")
        run_vitis = vitis.get("Run", False)
        if not isinstance(run_vitis, bool):
            raise ConfigurationError("Vitis.Run must be a boolean")
        stage_defaults = {
            "Reset": True,
            "CSim": False,
            "Synth": True,
            "CoSim": False,
            "Validation": False,
            "Export": False,
            "VSynth": False,
        }
        stages = vitis.get("Stages", {})
        if not isinstance(stages, Mapping):
            raise ConfigurationError("Vitis.Stages must be a mapping")
        unknown_stages = sorted(stages.keys() - stage_defaults.keys())
        if unknown_stages:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: Vitis.Stages.{unknown_stages[0]}"
            )
        for stage, enabled in stages.items():
            if not isinstance(enabled, bool):
                raise ConfigurationError(f"Vitis.Stages.{stage} must be a boolean")
        stage_defaults.update(stages)
        self._data = {
            "Project": {**project, "OutputDir": str(project["OutputDir"])},
            "HLS": deepcopy(dict(hls)),
            "Verification": {"Mode": "auto", **verification},
            "Vitis": {
                "Run": run_vitis,
                "Stages": stage_defaults,
            },
        }

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

        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)
