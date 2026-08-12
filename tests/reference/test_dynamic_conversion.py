from pathlib import Path

from ravel_hls import Project, convert


MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
    / "adam_p1_step2"
    / "adam_p1_step2_best.keras"
)


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
