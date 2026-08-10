"""Portable, deterministic RAVEL inference-parameter packages."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import os
import stat
from typing import Any
import zipfile

import numpy as np

from .exceptions import ConfigurationError


_MANIFEST_NAME = "parameter_package.json"
_MAX_ARCHIVE_ENTRIES = 4096
_MAX_MANIFEST_SIZE = 8 * 1024 * 1024
_MAX_ARRAY_SIZE = 2 * 1024**3
_MAX_TOTAL_SIZE = 8 * 1024**3
_INFERENCE_VARIABLES = {"kernel", "bias", "k", "i", "f"}
_STATIC_QUANTIZER_KEYS = {
    "affine",
    "enable_iq",
    "enable_oq",
    "heterogeneous_axis",
    "homogeneous_axis",
    "overflow",
    "overflow_mode",
    "q_type",
    "round_mode",
    "rounding",
    "scaler",
}


@dataclass(frozen=True)
class Parameters:
    """A validated, non-executable inference-parameter package."""

    _manifest: dict[str, Any]
    _arrays: tuple[np.ndarray, ...]

    @property
    def compatibility_sha256(self) -> str:
        return self._manifest["compatibility_sha256"]

    @property
    def parameter_state_sha256(self) -> str:
        return self._manifest["parameter_state_sha256"]

    @property
    def package_content_sha256(self) -> str:
        return self._manifest["package_content_sha256"]

    @classmethod
    def extract(cls, model: Any) -> "Parameters":
        """Extract generation-relevant inference state from a loaded model."""

        if isinstance(model, (str, os.PathLike)):
            import keras
            from hgq.layers import QConv2D, QDense

            model_path = Path(model)
            if not model_path.is_file():
                raise ConfigurationError(f"Keras model file does not exist: {model_path}")
            model = keras.models.load_model(
                model_path, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
            )
        layers = getattr(model, "layers", None)
        if not isinstance(layers, (list, tuple)):
            raise ConfigurationError("Parameters.extract requires a loaded model")
        arrays: list[np.ndarray] = []
        entries: list[dict[str, Any]] = []
        topology: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(layers):
            layer_name = str(getattr(layer, "name", ""))
            config = layer.get_config() if callable(getattr(layer, "get_config", None)) else {}
            topology.append(
                {
                    "class_name": getattr(layer, "_class_name", type(layer).__name__),
                    "quantizer_contract": _quantizer_contract(config),
                }
            )
            for variable in getattr(layer, "weights", ()):
                kind = str(getattr(variable, "name", "")).rsplit("/", 1)[-1]
                if kind not in _INFERENCE_VARIABLES:
                    continue
                values = np.ascontiguousarray(variable.numpy())
                if values.dtype.hasobject:
                    raise ConfigurationError("Parameter arrays cannot use object dtype")
                role = _variable_role(variable, layer_name)
                array_index = len(arrays)
                storage = f"arrays/{array_index:04d}.npy"
                payload = _npy_bytes(values)
                entries.append(
                    {
                        "slot": f"layer-{layer_index:03d}/{role}",
                        "kind": kind,
                        "shape": list(values.shape),
                        "dtype": values.dtype.str,
                        "storage": storage,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                arrays.append(values)
        compatibility = {
            "frontend_contract": {"id": "keras-hgq2", "version": 1},
            "topology": topology,
            "slots": [
                {key: entry[key] for key in ("slot", "kind", "shape", "dtype")}
                for entry in entries
            ],
        }
        parameter_state = [
            {"slot": entry["slot"], "sha256": entry["sha256"]} for entry in entries
        ]
        manifest = {
            "schema_version": 1,
            "frontend_contract": compatibility["frontend_contract"],
            "topology": topology,
            "entries": entries,
            "compatibility_sha256": _canonical_sha256(compatibility),
            "parameter_state_sha256": _canonical_sha256(parameter_state),
        }
        manifest["package_content_sha256"] = _package_content_sha256(manifest, arrays)
        return cls(manifest, tuple(arrays))

    def save(self, path: str | Path) -> None:
        """Write this package as a deterministic `.ravelparams` archive."""

        output = Path(path)
        with zipfile.ZipFile(output, "w") as archive:
            _write_zip_entry(archive, _MANIFEST_NAME, _manifest_bytes(self._manifest))
            for entry, values in zip(self._manifest["entries"], self._arrays):
                _write_zip_entry(archive, entry["storage"], _npy_bytes(values))

    def _apply(self, model: Any) -> Any:
        template = type(self).extract(model)
        if template.compatibility_sha256 != self.compatibility_sha256:
            raise ConfigurationError(
                "Parameter package is incompatible with the project model template"
            )
        variables = list(_inference_variables(model))
        if len(variables) != len(self._arrays):
            raise ConfigurationError(
                "Parameter package does not match the project model variable slots"
            )
        for variable, values in zip(variables, self._arrays):
            variable.assign(values)
        return model

    @classmethod
    def load(cls, path: str | Path) -> "Parameters":
        """Load and validate a `.ravelparams` archive."""

        with zipfile.ZipFile(path) as archive:
            archive_entries = archive.infolist()
            if len(archive_entries) > _MAX_ARCHIVE_ENTRIES:
                raise ConfigurationError(
                    "Parameter package exceeds the 4096-entry safety limit"
                )
            for entry in archive_entries:
                if stat.S_ISLNK(getattr(entry, "external_attr", 0) >> 16):
                    raise ConfigurationError(
                        f"Parameter package contains a symlink entry: {entry.filename}"
                    )
                if (
                    entry.filename.startswith("arrays/")
                    and entry.file_size > _MAX_ARRAY_SIZE
                ):
                    raise ConfigurationError(
                        "Parameter package array exceeds the 2 GiB safety limit"
                    )
            if sum(entry.file_size for entry in archive_entries) > _MAX_TOTAL_SIZE:
                raise ConfigurationError(
                    "Parameter package exceeds the 8 GiB uncompressed safety limit"
                )
            names = [entry.filename for entry in archive_entries]
            if len(names) != len(set(names)):
                raise ConfigurationError("Parameter package contains duplicate entries")
            for name in names:
                _require_safe_archive_path(name)
            if _MANIFEST_NAME not in names:
                raise ConfigurationError("Parameter package has no manifest")
            if archive.getinfo(_MANIFEST_NAME).file_size > _MAX_MANIFEST_SIZE:
                raise ConfigurationError(
                    "Parameter package manifest exceeds the 8 MiB safety limit"
                )
            manifest = json.loads(archive.read(_MANIFEST_NAME))
            if manifest.get("schema_version") != 1:
                raise ConfigurationError(
                    "Parameter package schema_version must be 1"
                )
            arrays = []
            for entry in manifest.get("entries", []):
                storage = entry["storage"]
                _require_safe_archive_path(storage)
                if storage not in names:
                    raise ConfigurationError(f"Parameter package payload is missing: {storage}")
                payload = archive.read(storage)
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ConfigurationError(f"Parameter payload digest mismatch: {storage}")
                try:
                    values = np.load(BytesIO(payload), allow_pickle=False)
                except ValueError as error:
                    if "Object arrays" in str(error):
                        raise ConfigurationError(
                            "Parameter arrays cannot use object dtype"
                        ) from error
                    raise ConfigurationError(
                        f"Parameter payload is not a valid NPY array: {storage}"
                    ) from error
                if values.dtype.hasobject:
                    raise ConfigurationError("Parameter arrays cannot use object dtype")
                if list(values.shape) != entry.get("shape"):
                    raise ConfigurationError(
                        f"Parameter payload shape does not match its manifest: {storage}"
                    )
                if values.dtype.str != entry.get("dtype"):
                    raise ConfigurationError(
                        f"Parameter payload dtype does not match its manifest: {storage}"
                    )
                arrays.append(np.ascontiguousarray(values))
        expected_content = _package_content_sha256(manifest, arrays)
        expected_compatibility = _canonical_sha256(
            {
                "frontend_contract": manifest.get("frontend_contract"),
                "topology": manifest.get("topology"),
                "slots": [
                    {key: entry.get(key) for key in ("slot", "kind", "shape", "dtype")}
                    for entry in manifest.get("entries", [])
                ],
            }
        )
        if expected_compatibility != manifest.get("compatibility_sha256"):
            raise ConfigurationError("Parameter package compatibility digest mismatch")
        expected_parameter_state = _canonical_sha256(
            [
                {"slot": entry.get("slot"), "sha256": entry.get("sha256")}
                for entry in manifest.get("entries", [])
            ]
        )
        if expected_parameter_state != manifest.get("parameter_state_sha256"):
            raise ConfigurationError("Parameter package state digest mismatch")
        if expected_content != manifest.get("package_content_sha256"):
            raise ConfigurationError("Parameter package content digest mismatch")
        return cls(manifest, tuple(arrays))


def _quantizer_contract(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _quantizer_contract(item)
            for key, item in sorted(value.items())
            if key in _STATIC_QUANTIZER_KEYS or isinstance(item, dict)
        }
    if isinstance(value, (list, tuple)):
        return [_quantizer_contract(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _inference_variables(model: Any):
    for layer in model.layers:
        for variable in getattr(layer, "weights", ()):
            kind = str(getattr(variable, "name", "")).rsplit("/", 1)[-1]
            if kind in _INFERENCE_VARIABLES:
                yield variable


def _variable_role(variable: Any, layer_name: str) -> str:
    path = str(getattr(variable, "path", getattr(variable, "name", "")))
    parts = path.split("/")
    if parts and parts[0] == layer_name:
        parts = parts[1:]
    return "/".join(
        part.removeprefix(f"{layer_name}_") if layer_name else part for part in parts
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return _canonical_bytes(manifest) + b"\n"


def _npy_bytes(values: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def _package_content_sha256(
    manifest: dict[str, Any], arrays: list[np.ndarray] | tuple[np.ndarray, ...]
) -> str:
    digest = hashlib.sha256()
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "package_content_sha256"
    }
    digest.update(_manifest_bytes(unsigned_manifest))
    for entry, values in zip(manifest.get("entries", []), arrays):
        digest.update(entry["storage"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(_npy_bytes(values))
    return digest.hexdigest()


def _write_zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def _require_safe_archive_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ConfigurationError(f"Parameter package contains unsafe archive path: {value}")
