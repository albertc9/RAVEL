from pathlib import Path
from io import BytesIO
import hashlib
import json
import stat
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest

from ravel_hls import ConfigurationError, Parameters


class _Variable:
    def __init__(self, path: str, values: list[float]) -> None:
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._values.copy()


class _Layer:
    def __init__(self, name: str, class_name: str, weights: list[_Variable]) -> None:
        self.name = name
        self._class_name = class_name
        self.weights = weights

    def get_config(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kq_conf": {
                "q_type": "kif",
                "round_mode": "RND",
                "overflow_mode": "SAT_SYM",
                "homogeneous_axis": [0],
                "heterogeneous_axis": None,
            },
        }


class _Model:
    def __init__(self) -> None:
        self.layers = [
            _Layer(
                "private_conv_name",
                "QConv2D",
                [
                    _Variable("private_conv_name/kernel", [1.0, 2.0]),
                    _Variable("private_conv_name/bias", [0.5]),
                    _Variable("private_conv_name/private_conv_name_kq/k", [1.0]),
                    _Variable("private_conv_name/private_conv_name_kq/i", [2.0]),
                    _Variable("private_conv_name/private_conv_name_kq/f", [3.0]),
                    _Variable("private_conv_name/beta", [0.1]),
                    _Variable("private_conv_name/ebops", [99.0]),
                ],
            )
        ]


def test_parameters_round_trip_is_deterministic_and_private(tmp_path: Path) -> None:
    parameters = Parameters.extract(_Model())
    first = tmp_path / "first.ravelparams"
    second = tmp_path / "second.ravelparams"

    parameters.save(first)
    parameters.save(second)
    loaded = Parameters.load(first)

    assert first.read_bytes() == second.read_bytes()
    assert loaded.compatibility_sha256 == parameters.compatibility_sha256
    assert loaded.parameter_state_sha256 == parameters.parameter_state_sha256
    assert loaded.package_content_sha256 == parameters.package_content_sha256
    assert b"private_conv_name" not in first.read_bytes()
    assert b"beta" not in first.read_bytes()
    assert b"ebops" not in first.read_bytes()


def test_parameters_reject_path_traversal_before_loading_payload(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    with zipfile.ZipFile(valid) as archive:
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


def test_parameters_extract_loads_a_keras_path_with_hgq2_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    model = _Model()
    load_call: dict[str, object] = {}
    qconv2d = object()
    qdense = object()

    def fake_load(path: Path, **kwargs: object) -> _Model:
        load_call.update({"path": path, **kwargs})
        return model

    fake_keras = ModuleType("keras")
    fake_keras.models = SimpleNamespace(load_model=fake_load)
    fake_hgq = ModuleType("hgq")
    fake_hgq_layers = ModuleType("hgq.layers")
    fake_hgq_layers.QConv2D = qconv2d
    fake_hgq_layers.QDense = qdense
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.setitem(sys.modules, "hgq", fake_hgq)
    monkeypatch.setitem(sys.modules, "hgq.layers", fake_hgq_layers)

    Parameters.extract(model_path)

    assert load_call == {
        "path": model_path,
        "custom_objects": {"QConv2D": qconv2d, "QDense": qdense},
    }


def test_parameters_reject_an_unknown_schema_version(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    unknown = tmp_path / "unknown.ravelparams"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(unknown, "w") as target:
        manifest = json.loads(source.read("parameter_package.json"))
        manifest["schema_version"] = 2
        target.writestr("parameter_package.json", json.dumps(manifest))
        for entry in manifest["entries"]:
            target.writestr(entry["storage"], source.read(entry["storage"]))

    with pytest.raises(ConfigurationError, match="schema_version.*1"):
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


def test_parameters_reject_a_zip_symlink_entry(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    linked = tmp_path / "linked.ravelparams"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(linked, "w") as target:
        for source_info in source.infolist():
            target_info = zipfile.ZipInfo(source_info.filename)
            target_info.compress_type = zipfile.ZIP_DEFLATED
            target_info.external_attr = source_info.external_attr
            if source_info.filename.startswith("arrays/"):
                target_info.external_attr = (stat.S_IFLNK | 0o777) << 16
            target.writestr(target_info, source.read(source_info.filename))

    with pytest.raises(ConfigurationError, match="symlink"):
        Parameters.load(linked)


def test_parameters_reject_an_object_dtype_payload(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    unsafe = tmp_path / "unsafe.ravelparams"
    object_payload = BytesIO()
    np.save(object_payload, np.asarray([object()], dtype=object), allow_pickle=True)
    payload = object_payload.getvalue()
    with zipfile.ZipFile(valid) as source:
        manifest = json.loads(source.read("parameter_package.json"))
        replaced_storage = manifest["entries"][0]["storage"]
        manifest["entries"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
        with zipfile.ZipFile(unsafe, "w") as target:
            target.writestr("parameter_package.json", json.dumps(manifest))
            for entry in manifest["entries"]:
                target.writestr(
                    entry["storage"],
                    payload
                    if entry["storage"] == replaced_storage
                    else source.read(entry["storage"]),
                )

    with pytest.raises(ConfigurationError, match="object dtype"):
        Parameters.load(unsafe)


def test_parameters_reject_a_payload_shape_mismatch(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    mismatched = tmp_path / "mismatched.ravelparams"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(mismatched, "w") as target:
        manifest = json.loads(source.read("parameter_package.json"))
        manifest["entries"][0]["shape"] = [999]
        target.writestr("parameter_package.json", json.dumps(manifest))
        for entry in manifest["entries"]:
            target.writestr(entry["storage"], source.read(entry["storage"]))

    with pytest.raises(ConfigurationError, match="shape"):
        Parameters.load(mismatched)


def test_parameters_recompute_the_compatibility_identity(tmp_path: Path) -> None:
    valid = tmp_path / "valid.ravelparams"
    Parameters.extract(_Model()).save(valid)
    mismatched = tmp_path / "mismatched-identity.ravelparams"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(mismatched, "w") as target:
        manifest = json.loads(source.read("parameter_package.json"))
        manifest["compatibility_sha256"] = "0" * 64
        target.writestr("parameter_package.json", json.dumps(manifest))
        for entry in manifest["entries"]:
            target.writestr(entry["storage"], source.read(entry["storage"]))

    with pytest.raises(ConfigurationError, match="compatibility"):
        Parameters.load(mismatched)
