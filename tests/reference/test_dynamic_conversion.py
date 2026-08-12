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
