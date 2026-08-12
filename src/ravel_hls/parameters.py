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
from .domain import ParameterPayload, ParameterTensor


_MANIFEST_NAME = "parameter_package.json"
_MAX_ARCHIVE_ENTRIES = 4096
_MAX_MANIFEST_SIZE = 8 * 1024 * 1024
_MAX_ARRAY_SIZE = 2 * 1024**3
_MAX_TOTAL_SIZE = 8 * 1024**3


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
    def model_structure_sha256(self) -> str:
        """Canonical ModelGraph structure and numeric-contract identity."""

        value = self._manifest.get("model_structure_sha256")
        if not isinstance(value, str):
            raise ConfigurationError("Parameter package has no ModelGraph structure identity")
        return value

    @property
    def package_content_sha256(self) -> str:
        return self._manifest["package_content_sha256"]

    @classmethod
    def extract(cls, model: Any) -> "Parameters":
        """Extract generation-relevant inference state from a loaded model."""

        if not (
            isinstance(model, (str, os.PathLike))
            or (hasattr(model, "inputs") and hasattr(model, "outputs"))
        ):
            raise ConfigurationError(
                "Parameters.extract requires a model path or loaded Keras model"
            )
        return cls._extract_modelgraph(model)

    @classmethod
    def _extract_modelgraph(cls, model: Any) -> "Parameters":
        from .analysis.model import _analyze_model

        analyzed = _analyze_model(
            model, {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}}
        )
        report = analyzed.analysis.to_dict()
        arrays = []
        entries = []
        for index, tensor in enumerate(analyzed.parameter_payload.tensors):
            values = np.ascontiguousarray(tensor.values)
            payload = _npy_bytes(values)
            storage = f"arrays/{index:04d}.npy"
            entries.append(
                {
                    "id": tensor.id,
                    "operation_id": tensor.operation_id,
                    "role": tensor.role,
                    "shape": list(values.shape),
                    "dtype": values.dtype.str,
                    "numeric_type": dict(tensor.numeric_type),
                    "storage": storage,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            arrays.append(values)
        compatibility = {
            "model_family": report["model_family"],
            "model_structure_sha256": report["fingerprints"][
                "model_structure_sha256"
            ],
            "entries": [
                {
                    key: entry[key]
                    for key in (
                        "id",
                        "operation_id",
                        "role",
                        "shape",
                        "numeric_type",
                    )
                }
                for entry in entries
            ],
        }
        manifest = {
            "schema_version": 2,
            "format": "ravel-modelgraph-parameters",
            "generation": report["generation"],
            "model_family": report["model_family"],
            "model_structure_sha256": report["fingerprints"][
                "model_structure_sha256"
            ],
            "frontend_provenance": report["frontend_provenance"],
            "model_facts": report["model_facts"],
            "entries": entries,
            "compatibility_sha256": _canonical_sha256(compatibility),
            "parameter_state_sha256": report["fingerprints"][
                "parameter_state_sha256"
            ],
            "known_answer_evidence": None,
        }
        manifest["package_content_sha256"] = _package_content_sha256(
            manifest, arrays
        )
        return cls(manifest, tuple(arrays))

    def save(self, path: str | Path) -> None:
        """Write this package as a deterministic `.ravelparams` archive."""

        output = Path(path)
        with zipfile.ZipFile(output, "w") as archive:
            _write_zip_entry(archive, _MANIFEST_NAME, _manifest_bytes(self._manifest))
            for entry, values in zip(self._manifest["entries"], self._arrays):
                _write_zip_entry(archive, entry["storage"], _npy_bytes(values))

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
            if manifest.get("schema_version") != 2:
                raise ConfigurationError("Parameter package schema_version must be 2")
            if manifest.get("format") != "ravel-modelgraph-parameters":
                raise ConfigurationError(
                    "Parameter package format must be ravel-modelgraph-parameters"
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
                "model_family": manifest.get("model_family"),
                "model_structure_sha256": manifest.get(
                    "model_structure_sha256"
                ),
                "entries": [
                    {
                        key: entry.get(key)
                        for key in (
                            "id",
                            "operation_id",
                            "role",
                            "shape",
                            "numeric_type",
                        )
                    }
                    for entry in manifest.get("entries", [])
                ],
            }
        )
        if expected_compatibility != manifest.get("compatibility_sha256"):
            raise ConfigurationError("Parameter package compatibility digest mismatch")
        expected_parameter_state = _modelgraph_parameter_state(manifest, arrays)
        if expected_parameter_state != manifest.get("parameter_state_sha256"):
            raise ConfigurationError("Parameter package state digest mismatch")
        if expected_content != manifest.get("package_content_sha256"):
            raise ConfigurationError("Parameter package content digest mismatch")
        return cls(manifest, tuple(arrays))

    def _payload_for(self, template: ParameterPayload) -> ParameterPayload:
        template_by_id = template.by_id()
        tensors = []
        for entry, values in zip(self._manifest["entries"], self._arrays):
            template_tensor = template_by_id.get(entry["id"])
            if template_tensor is None:
                raise ConfigurationError(
                    f"Parameter package binding is absent from template: {entry['id']}"
                )
            tensors.append(
                ParameterTensor(
                    id=entry["id"],
                    operation_id=entry["operation_id"],
                    role=entry["role"],
                    symbol=template_tensor.symbol,
                    type_name=template_tensor.type_name,
                    numeric_type=entry["numeric_type"],
                    values=values,
                )
            )
        return ParameterPayload(tuple(tensors))

    def _apply_to_analysis(self, analyzed: Any) -> tuple[ParameterPayload, dict[str, Any]]:
        """Replace one analyzed ModelGraph payload by canonical operation bindings."""

        payload = self._payload_for(analyzed.parameter_payload)
        replacements = payload.by_id()
        from .analysis.model import _semantic_kind

        ordinals: dict[str, int] = {}
        for layer in analyzed.graph.get_layers():
            kind = _semantic_kind(layer)
            ordinal = ordinals.get(kind, 0)
            ordinals[kind] = ordinal + 1
            operation_id = f"{kind}_{ordinal}"
            for role, weight in layer.weights.items():
                tensor = replacements.get(f"{operation_id}:{role}")
                if tensor is None:
                    continue
                if tuple(weight.data.shape) != tuple(tensor.values.shape):
                    raise ConfigurationError(
                        f"Parameter payload shape changed for {tensor.id}"
                    )
                weight.data[...] = tensor.values
                weight.nzeros = int(np.count_nonzero(weight.data == 0))
                weight.nonzeros = int(weight.data.size - weight.nzeros)
                weight.min = np.min(weight.data)
                weight.max = np.max(weight.data)
        report = analyzed.analysis.to_dict()
        report["frontend_provenance"] = self._manifest["frontend_provenance"]
        report["model_facts"] = self._manifest["model_facts"]
        report["fingerprints"]["parameter_state_sha256"] = self._manifest[
            "parameter_state_sha256"
        ]
        return payload, report


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


def _modelgraph_parameter_state(
    manifest: dict[str, Any], arrays: list[np.ndarray]
) -> str:
    identities = []
    for entry, values in zip(manifest.get("entries", []), arrays):
        numeric_type = entry["numeric_type"]
        fractional = numeric_type["width"] - numeric_type["integer"]
        codes = np.rint(np.asarray(values) * (2**fractional)).astype(
            "<i8", copy=False
        )
        identities.append(
            {
                "operation_id": entry["operation_id"],
                "role": entry["role"],
                "shape": entry["shape"],
                "numeric_type": numeric_type,
                "content_sha256": hashlib.sha256(
                    codes.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return _canonical_sha256(identities)


def _write_zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def _require_safe_archive_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ConfigurationError(f"Parameter package contains unsafe archive path: {value}")
