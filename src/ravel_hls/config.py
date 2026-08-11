"""RAVEL-owned configuration."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
import os
from typing import Any

from .exceptions import ConfigurationError


_AGGRESSIVE_SPECIALIZATION = {
    "TemporalPacking": 4,
    "DenseParallelism": 2,
}


def _resolve_optimization(values: Any) -> dict[str, int]:
    if values is None:
        return dict(_AGGRESSIVE_SPECIALIZATION)
    if not isinstance(values, Mapping):
        raise ConfigurationError("Optimization must be a mapping")
    fields = {"TemporalPacking", "DenseParallelism"}
    unknown_fields = sorted(values.keys() - fields)
    if unknown_fields:
        raise ConfigurationError(
            f"Unknown RAVEL configuration field: Optimization.{unknown_fields[0]}"
        )
    missing_fields = sorted(fields - values.keys())
    if missing_fields:
        raise ConfigurationError(
            f"Optimization.{missing_fields[0]} is required when Optimization is supplied"
        )
    temporal_packing = values["TemporalPacking"]
    if temporal_packing not in {2, 4} or isinstance(temporal_packing, bool):
        raise ConfigurationError("Optimization.TemporalPacking must be one of: 2, 4")
    dense_parallelism = values["DenseParallelism"]
    if dense_parallelism not in {1, 2} or isinstance(dense_parallelism, bool):
        raise ConfigurationError("Optimization.DenseParallelism must be one of: 1, 2")
    return {
        "TemporalPacking": temporal_packing,
        "DenseParallelism": dense_parallelism,
    }


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
            raise ConfigurationError("Profile must be aria for RAVEL Aria 1.1.0")
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
            self._data.keys()
            - {"Project", "HLS", "Optimization", "Verification", "Vitis"}
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
        unknown_project_fields = sorted(
            project.keys() - {"Name", "OutputDir", "ForceReplace"}
        )
        if unknown_project_fields:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: Project.{unknown_project_fields[0]}"
            )
        for required in ("Name", "OutputDir"):
            if required not in project:
                raise ConfigurationError(f"Project.{required} is required")
        project_name = project["Name"]
        if not isinstance(project_name, str) or not project_name:
            raise ConfigurationError("Project.Name must be a nonempty string")
        output_dir = project["OutputDir"]
        if not isinstance(output_dir, (str, os.PathLike)):
            raise ConfigurationError("Project.OutputDir must be a path")
        force_replace = project.get("ForceReplace", False)
        if not isinstance(force_replace, bool):
            raise ConfigurationError("Project.ForceReplace must be a boolean")
        unknown_hls_fields = sorted(
            hls.keys()
            - {"Backend", "IOType", "Part", "ClockPeriod", "Config"}
        )
        if unknown_hls_fields:
            raise ConfigurationError(
                f"Unknown RAVEL configuration field: HLS.{unknown_hls_fields[0]}"
            )
        if "Config" not in hls or not isinstance(hls["Config"], Mapping):
            raise ConfigurationError("HLS.Config must be a mapping")
        backend = hls.get("Backend", "Vitis")
        if backend != "Vitis":
            raise ConfigurationError("HLS.Backend must be Vitis")
        io_type = hls.get("IOType", "io_stream")
        if io_type != "io_stream":
            raise ConfigurationError("HLS.IOType must be io_stream")
        part = hls.get("Part")
        if part is not None and (not isinstance(part, str) or not part):
            raise ConfigurationError("HLS.Part must be a nonempty string or null")
        clock_period = hls.get("ClockPeriod")
        if clock_period is not None and (
            not isinstance(clock_period, (int, float))
            or isinstance(clock_period, bool)
            or clock_period <= 0
        ):
            raise ConfigurationError("HLS.ClockPeriod must be a positive number or null")
        verification = self._data.get("Verification", {})
        if not isinstance(verification, Mapping):
            raise ConfigurationError("Verification must be a mapping")
        unknown_verification_fields = sorted(
            verification.keys() - {"Mode", "Samples", "Seed"}
        )
        if unknown_verification_fields:
            raise ConfigurationError(
                "Unknown RAVEL configuration field: "
                f"Verification.{unknown_verification_fields[0]}"
            )
        normalized_verification = {"Mode": "auto", **verification}
        if normalized_verification["Mode"] not in {"auto", "required", "disabled"}:
            raise ConfigurationError(
                "Verification.Mode must be one of: auto, required, disabled"
            )
        samples = normalized_verification.get("Samples")
        if samples is not None and (
            not isinstance(samples, int) or isinstance(samples, bool) or samples < 1
        ):
            raise ConfigurationError("Verification.Samples must be a positive integer")
        seed = normalized_verification.get("Seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ConfigurationError("Verification.Seed must be a nonnegative integer")
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
            "Project": {
                "Name": project_name,
                "OutputDir": str(output_dir),
                "ForceReplace": force_replace,
            },
            "HLS": {
                "Backend": backend,
                "IOType": io_type,
                "Part": part,
                "ClockPeriod": clock_period,
                "Config": deepcopy(dict(hls["Config"])),
            },
            "Optimization": _resolve_optimization(self._data.get("Optimization")),
            "Verification": normalized_verification,
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
