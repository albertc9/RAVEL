from pathlib import Path

import pytest

from ravel_hls import CompatibilityError, Project, convert, refresh


MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
    / "adam_p1_step2"
    / "adam_p1_step2_best.keras"
)
REFERENCE_MODELS = sorted(
    path
    for path in (Path(__file__).parents[2] / "references").glob("**/*.keras")
    if "cnn_for_arianna" not in path.parts
)

assert len(REFERENCE_MODELS) == 12


def test_user_can_convert_a_retrained_model_without_building_hls4ml_config(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "aria_step2"

    project = convert(
        MODEL,
        output_dir,
        {
            "HLS": {
                "Backend": "Vitis",
                "IOType": "io_stream",
                "Part": "xcku5p-ffvb676-2-e",
                "ClockPeriod": 5.0,
            },
            "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
            "Verification": {"Mode": "disabled"},
        },
    )

    assert isinstance(project, Project)
    assert project.path == output_dir
    assert project.manifest["schema_version"] == 4
    assert project.manifest["ravel"]["release"] == "1.5.0"
    assert project.manifest["source_model"]["model_family"] == {
        "id": "hgq-conv-pool-dense",
        "version": 1,
    }
    assert (
        project.manifest["source_model"]["facts"]["operations"][0]["outputs"][0][
            "numeric_type"
        ]["width"]
        == 8
    )
    assert (
        project.manifest["interfaces"]["rtl_interface"]["expected"][
            "input_tdata_bits"
        ]
        == 128
    )


def test_conversion_checks_implementation_consistency_without_accuracy_labels(
    tmp_path: Path,
) -> None:
    project = convert(
        MODEL,
        tmp_path / "aria_consistency",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "required", "Samples": 8, "Seed": 7},
        },
    )

    verification = project.manifest["verification"]
    assert verification["source_conversion_consistency"] == "passed"
    assert verification["transformation_equivalence"] == "passed"
    assert verification["stimuli"]["kind"] == "numeric_contract"
    assert verification["stimuli"]["input_numeric_type"]["width"] == 8
    assert verification["stimuli"]["sample_count"] == 8
    assert "accuracy" not in verification


def test_extracted_geometry_drives_generated_cpp_and_consistency(
    tmp_path: Path, noncanonical_geometry_model,
) -> None:
    project = convert(
        noncanonical_geometry_model,
        tmp_path / "aria_dynamic_geometry",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 2, "DenseParallelism": 2},
            "Verification": {"Mode": "required", "Samples": 8, "Seed": 11},
        },
    )

    assert project.manifest["interfaces"]["logical_model_interface"] == {
        "input_shape": [128, 4],
        "output_shape": [1],
    }
    assert project.manifest["interfaces"]["hls_stream_interface"][
        "values_per_input_word"
    ] == 8
    assert project.manifest["verification"][
        "source_conversion_consistency"
    ] == "passed"
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"


@pytest.mark.parametrize(
    "model_path", REFERENCE_MODELS, ids=lambda path: path.parent.name
)
def test_every_retrained_reference_model_generates_consistent_default_cpp(
    tmp_path: Path, model_path: Path
) -> None:
    project = convert(
        model_path,
        tmp_path / model_path.parent.name,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
            "Verification": {"Mode": "required", "Samples": 8, "Seed": 23},
        },
    )

    assert project.manifest["source_model"]["model_family"] == {
        "id": "hgq-conv-pool-dense",
        "version": 1,
    }
    assert project.manifest["verification"][
        "source_conversion_consistency"
    ] == "passed"
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"


def test_refresh_replaces_parameters_while_preserving_the_recorded_architecture(
    tmp_path: Path,
) -> None:
    import keras
    from hgq.layers import QConv2D, QDense

    project = convert(
        MODEL,
        tmp_path / "aria_refresh",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "disabled"},
        },
    )
    original = project.manifest
    updated_model = keras.models.load_model(
        MODEL, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
    )
    kernel = updated_model.layers[1].kernel
    changed = kernel.numpy()
    changed.reshape(-1)[0] += 0.125
    kernel.assign(changed)

    refreshed = refresh(project, updated_model)

    assert refreshed.path == project.path
    assert (
        refreshed.manifest["source_model"]["fingerprints"][
            "parameter_state_sha256"
        ]
        != original["source_model"]["fingerprints"]["parameter_state_sha256"]
    )
    assert (
        refreshed.manifest["source_model"]["fingerprints"][
            "model_structure_sha256"
        ]
        == original["source_model"]["fingerprints"]["model_structure_sha256"]
    )
    assert refreshed.manifest["architecture_contract_sha256"] == original[
        "architecture_contract_sha256"
    ]
    assert refreshed.manifest["implementation_plan"] == original[
        "implementation_plan"
    ]


def test_refresh_rejects_a_model_that_changes_the_recorded_architecture(
    tmp_path: Path,
) -> None:
    project = convert(
        MODEL,
        tmp_path / "aria_refresh_reject",
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Verification": {"Mode": "disabled"},
        },
    )
    changed_precision_model = (
        Path(__file__).parents[2]
        / "references"
        / "fLow_0.08-fhigh_0.23-rate_0.5"
        / "adam_hgq_replicate_s2"
        / "adam_hgq_replicate_s2_best.keras"
    )

    with pytest.raises(CompatibilityError, match="architecture contract"):
        refresh(project, changed_precision_model)
