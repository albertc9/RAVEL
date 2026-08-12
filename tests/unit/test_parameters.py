from io import BytesIO
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from ravel_hls import ConfigurationError, Parameters


MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
    / "adam_p1_step2"
    / "adam_p1_step2_best.keras"
)


@pytest.fixture(scope="module")
def valid_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("parameters") / "valid.ravelparams"
    Parameters.extract(MODEL).save(path)
    return path


def _rewrite_package(
    source_path: Path,
    target_path: Path,
    mutate_manifest,
    replace_payload=None,
) -> None:
    with zipfile.ZipFile(source_path) as source:
        manifest = json.loads(source.read("parameter_package.json"))
        mutate_manifest(manifest)
        with zipfile.ZipFile(target_path, "w") as target:
            target.writestr("parameter_package.json", json.dumps(manifest))
            for entry in manifest.get("entries", []):
                payload = source.read(entry["storage"])
                if replace_payload is not None:
                    payload = replace_payload(entry, payload)
                target.writestr(entry["storage"], payload)


def test_parameters_reject_legacy_object_extraction() -> None:
    class _LegacyModel:
        layers = []

    with pytest.raises(ConfigurationError, match="model path or loaded Keras model"):
        Parameters.extract(_LegacyModel())


def test_parameters_reject_path_traversal_before_loading_payload(
    tmp_path: Path, valid_package: Path
) -> None:
    with zipfile.ZipFile(valid_package) as archive:
        manifest = json.loads(archive.read("parameter_package.json"))
        payload = archive.read(manifest["entries"][0]["storage"])
    manifest["entries"][0]["storage"] = "../payload.npy"
    malicious = tmp_path / "malicious.ravelparams"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("parameter_package.json", json.dumps(manifest))
        archive.writestr("../payload.npy", payload)

    with pytest.raises(ConfigurationError, match="unsafe.*path"):
        Parameters.load(malicious)


def test_parameters_reject_an_archive_with_too_many_entries(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.ravelparams"
    with zipfile.ZipFile(oversized, "w") as archive:
        archive.writestr("parameter_package.json", "{}")
        for index in range(4096):
            archive.writestr(f"arrays/{index:04d}.npy", b"")

    with pytest.raises(ConfigurationError, match="4096"):
        Parameters.load(oversized)


def test_parameters_reject_an_oversized_manifest(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized-manifest.ravelparams"
    with zipfile.ZipFile(oversized, "w") as archive:
        archive.writestr("parameter_package.json", b" " * (8 * 1024 * 1024 + 1))

    with pytest.raises(ConfigurationError, match="8 MiB"):
        Parameters.load(oversized)


def test_parameters_reject_a_non_v2_schema(
    tmp_path: Path, valid_package: Path
) -> None:
    unknown = tmp_path / "unknown.ravelparams"
    _rewrite_package(
        valid_package,
        unknown,
        lambda manifest: manifest.update(schema_version=1),
    )

    with pytest.raises(ConfigurationError, match="schema_version must be 2"):
        Parameters.load(unknown)


@pytest.mark.parametrize(
    ("array_sizes", "message"),
    [
        ([2 * 1024**3 + 1], "2 GiB"),
        ([2 * 1024**3] * 4, "8 GiB"),
    ],
)
def test_parameters_reject_oversized_payload_metadata_before_reading(
    monkeypatch: pytest.MonkeyPatch, array_sizes: list[int], message: str
) -> None:
    entries = [SimpleNamespace(filename="parameter_package.json", file_size=2)]
    entries.extend(
        SimpleNamespace(filename=f"arrays/{index:04d}.npy", file_size=size)
        for index, size in enumerate(array_sizes)
    )

    class _Archive:
        def __enter__(self) -> "_Archive":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def infolist(self) -> list[SimpleNamespace]:
            return entries

        def getinfo(self, name: str) -> SimpleNamespace:
            return next(entry for entry in entries if entry.filename == name)

        def read(self, name: str) -> bytes:
            raise AssertionError(f"payload was read before size validation: {name}")

    monkeypatch.setattr("ravel_hls.parameters.zipfile.ZipFile", lambda path: _Archive())

    with pytest.raises(ConfigurationError, match=message):
        Parameters.load("oversized.ravelparams")


def test_parameters_reject_a_zip_symlink_entry(
    tmp_path: Path, valid_package: Path
) -> None:
    linked = tmp_path / "linked.ravelparams"
    with zipfile.ZipFile(valid_package) as source, zipfile.ZipFile(
        linked, "w"
    ) as target:
        for source_info in source.infolist():
            target_info = zipfile.ZipInfo(source_info.filename)
            target_info.compress_type = zipfile.ZIP_DEFLATED
            target_info.external_attr = source_info.external_attr
            if source_info.filename.startswith("arrays/"):
                target_info.external_attr = (stat.S_IFLNK | 0o777) << 16
            target.writestr(target_info, source.read(source_info.filename))

    with pytest.raises(ConfigurationError, match="symlink"):
        Parameters.load(linked)


def test_parameters_reject_an_object_dtype_payload(
    tmp_path: Path, valid_package: Path
) -> None:
    unsafe = tmp_path / "unsafe.ravelparams"
    object_payload = BytesIO()
    np.save(object_payload, np.asarray([object()], dtype=object), allow_pickle=True)
    payload = object_payload.getvalue()

    def mutate(manifest: dict[str, object]) -> None:
        manifest["entries"][0]["sha256"] = hashlib.sha256(payload).hexdigest()

    first_storage = None
    with zipfile.ZipFile(valid_package) as source:
        first_storage = json.loads(source.read("parameter_package.json"))["entries"][0][
            "storage"
        ]
    _rewrite_package(
        valid_package,
        unsafe,
        mutate,
        lambda entry, original: payload
        if entry["storage"] == first_storage
        else original,
    )

    with pytest.raises(ConfigurationError, match="object dtype"):
        Parameters.load(unsafe)


def test_parameters_reject_a_payload_shape_mismatch(
    tmp_path: Path, valid_package: Path
) -> None:
    mismatched = tmp_path / "mismatched.ravelparams"

    def mutate(manifest: dict[str, object]) -> None:
        manifest["entries"][0]["shape"] = [999]

    _rewrite_package(valid_package, mismatched, mutate)

    with pytest.raises(ConfigurationError, match="shape"):
        Parameters.load(mismatched)


def test_parameters_recompute_the_compatibility_identity(
    tmp_path: Path, valid_package: Path
) -> None:
    mismatched = tmp_path / "mismatched-identity.ravelparams"
    _rewrite_package(
        valid_package,
        mismatched,
        lambda manifest: manifest.update(compatibility_sha256="0" * 64),
    )

    with pytest.raises(ConfigurationError, match="compatibility"):
        Parameters.load(mismatched)
