import json
from pathlib import Path
import zipfile

import pytest

from ravel_hls import Parameters, VerificationError, analyze, convert, refresh


MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
    / "adam_p1_step2"
    / "adam_p1_step2_best.keras"
)


def test_modelgraph_parameter_package_is_canonical_and_deterministic(
    tmp_path: Path,
) -> None:
    analysis = analyze(
        MODEL, {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}}
    ).to_dict()
    parameters = Parameters.extract(MODEL)
    first = tmp_path / "first.ravelparams"
    second = tmp_path / "second.ravelparams"

    parameters.save(first)
    parameters.save(second)
    loaded = Parameters.load(first)
    with zipfile.ZipFile(first) as archive:
        manifest_bytes = archive.read("parameter_package.json")
        manifest = json.loads(manifest_bytes)

    assert first.read_bytes() == second.read_bytes()
    assert manifest["schema_version"] == 2
    assert manifest["format"] == "ravel-modelgraph-parameters"
    assert [entry["id"] for entry in manifest["entries"]] == [
        "conv2d_0:bias",
        "conv2d_0:weight",
        "dense_0:bias",
        "dense_0:weight",
    ]
    assert b"q_conv2d" not in manifest_bytes
    assert b"q_dense" not in manifest_bytes
    assert parameters.model_structure_sha256 == analysis["fingerprints"][
        "model_structure_sha256"
    ]
    assert parameters.parameter_state_sha256 == analysis["fingerprints"][
        "parameter_state_sha256"
    ]
    assert loaded.package_content_sha256 == parameters.package_content_sha256


def test_parameter_package_refreshes_modelgraph_payload_without_framework_names(
    tmp_path: Path,
) -> None:
    import keras
    from hgq.layers import QConv2D, QDense

    project = convert(
        MODEL,
        tmp_path / "package_refresh",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "disabled"},
        },
    )
    original_contract = project.manifest["architecture_contract_sha256"]
    original_header = next(
        (project.path / "firmware" / "weights").glob("*_ravel_packed.h")
    ).read_text(encoding="utf-8")
    updated_model = keras.models.load_model(
        MODEL, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
    )
    kernel = updated_model.layers[-1].kernel
    changed = kernel.numpy()
    changed.reshape(-1)[0] += 0.125
    kernel.assign(changed)
    parameters = Parameters.extract(updated_model)

    refreshed = refresh(project, parameters)

    refreshed_header = next(
        (refreshed.path / "firmware" / "weights").glob("*_ravel_packed.h")
    ).read_text(encoding="utf-8")
    assert refreshed.manifest["architecture_contract_sha256"] == original_contract
    assert refreshed.manifest["source_model"]["fingerprints"][
        "parameter_state_sha256"
    ] == parameters.parameter_state_sha256
    assert refreshed_header != original_header


def test_parameter_package_without_known_answers_has_explicit_verification_limits(
    tmp_path: Path,
) -> None:
    parameters = Parameters.extract(MODEL)
    auto_project = convert(
        MODEL,
        tmp_path / "package_auto",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "auto", "Samples": 6, "Seed": 29},
        },
    )

    refreshed = refresh(auto_project, parameters)

    assert refreshed.manifest["verification"][
        "transformation_equivalence"
    ] == "passed"
    assert refreshed.manifest["verification"][
        "source_conversion_consistency"
    ] == "not_run"
    assert refreshed.manifest["verification"]["model_fidelity"] == "not_run"

    required_project = convert(
        MODEL,
        tmp_path / "package_required",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "required", "Samples": 6, "Seed": 29},
        },
    )
    original_manifest = required_project.manifest
    with pytest.raises(VerificationError, match="known-answer evidence"):
        refresh(required_project, parameters)
    assert required_project.path.is_dir()
    assert required_project.manifest == original_manifest
