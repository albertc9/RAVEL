from pathlib import Path

from ravel_hls import analyze


REFERENCE_MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "cnn_for_arianna"
    / "models"
    / "cnn_for_arianna.keras"
)


def test_user_can_analyze_the_canonical_model_without_publishing_a_project(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    analysis = analyze(
        REFERENCE_MODEL,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
        },
    )

    report = analysis.to_dict()
    assert report["generation"] == {"id": "aria", "version": "1.5.0"}
    assert report["model_family"] == {
        "id": "hgq-conv-pool-dense",
        "version": 1,
    }
    assert report["applicability"] == {"status": "applicable", "findings": []}
    assert [operation["id"] for operation in report["model_facts"]["operations"]] == [
        "input_0",
        "repack_0",
        "conv2d_0",
        "relu_0",
        "max_pool2d_0",
        "reshape_0",
        "dense_0",
    ]
    assert report["resolved_design"]["specialization"] == {
        "temporal_packing": 4,
        "dense_parallelism": 2,
    }
    assert report["resolved_design"]["interfaces"]["rtl"]["input_tdata_bits"] == 256
    assert report["resolved_design"]["interfaces"]["rtl"]["output_tdata_bits"] == 32
    assert list(tmp_path.iterdir()) == []
