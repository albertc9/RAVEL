from pathlib import Path

from ravel_hls import analyze


REFERENCE_MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "cnn_for_arianna"
    / "models"
    / "cnn_for_arianna.keras"
)
RETRAINED_ROOT = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
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


def test_retrained_models_share_a_family_while_exposing_learned_numeric_types() -> None:
    narrow = analyze(
        RETRAINED_ROOT / "adam_p1_step2" / "adam_p1_step2_best.keras",
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()
    wide = analyze(
        RETRAINED_ROOT
        / "adam_hgq_replicate_s2"
        / "adam_hgq_replicate_s2_best.keras",
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()

    assert narrow["model_family"] == wide["model_family"]
    narrow_input = narrow["model_facts"]["operations"][0]["outputs"][0]
    wide_input = wide["model_facts"]["operations"][0]["outputs"][0]
    assert narrow_input == {
        "id": "input_0:out0",
        "shape": [256, 4],
        "numeric_type": {
            "kind": "fixed",
            "width": 8,
            "integer": 4,
            "signed": True,
            "rounding": "RND",
            "saturation": "SAT_SYM",
            "saturation_bits": 0,
        },
    }
    assert wide_input["numeric_type"]["width"] == 11
    assert narrow["resolved_design"]["interfaces"]["rtl"]["input_tdata_bits"] == 128
    assert wide["resolved_design"]["interfaces"]["rtl"]["input_tdata_bits"] == 256
    assert (
        narrow["fingerprints"]["model_structure_sha256"]
        != wide["fingerprints"]["model_structure_sha256"]
    )
    assert (
        narrow["fingerprints"]["parameter_state_sha256"]
        != wide["fingerprints"]["parameter_state_sha256"]
    )


def test_future_topology_is_analyzed_before_it_is_reported_as_unsupported() -> None:
    import keras

    inputs = keras.Input(shape=(256, 4))
    x = keras.layers.Reshape((256, 4, 1))(inputs)
    x = keras.layers.Conv2D(
        7, (5, 1), strides=(3, 1), padding="valid", activation="relu"
    )(x)
    x = keras.layers.Conv2D(7, (1, 1), padding="same")(x)
    x = keras.layers.MaxPool2D((2, 1), strides=(2, 1))(x)
    x = keras.layers.Flatten()(x)
    model = keras.Model(inputs, keras.layers.Dense(1)(x))

    report = analyze(
        model,
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()

    assert "conv2d_1" in [
        operation["id"] for operation in report["model_facts"]["operations"]
    ]
    assert report["model_family"] is None
    assert report["applicability"]["status"] == "unsupported"
    assert "family.topology.sequence" in {
        finding["code"] for finding in report["applicability"]["findings"]
    }
    assert report["resolved_design"] is None


def test_retraining_does_not_change_the_static_hgq2_frontend_contract() -> None:
    first = analyze(
        RETRAINED_ROOT / "adam_p1_step2" / "adam_p1_step2_best.keras",
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()
    second = analyze(
        RETRAINED_ROOT
        / "adam_hgq_replicate_s2"
        / "adam_hgq_replicate_s2_best.keras",
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()

    provenance = first["frontend_provenance"]
    assert provenance["adapter"] == {"id": "keras-hgq2", "version": 1}
    assert [layer["class_name"] for layer in provenance["source_layers"]] == [
        "Reshape",
        "QConv2D",
        "MaxPooling2D",
        "Flatten",
        "QDense",
    ]
    weight_contract = next(
        contract
        for contract in provenance["quantizer_contracts"]
        if contract["source_ordinal"] == 1 and contract["role"] == "weight"
    )
    assert weight_contract == {
        "source_ordinal": 1,
        "role": "weight",
        "q_type": "kif",
        "rounding": "RND",
        "overflow": "SAT_SYM",
        "homogeneous_axis": None,
        "heterogeneous_axis": [],
        "is_weight": True,
    }
    assert (
        first["fingerprints"]["frontend_provenance_sha256"]
        == second["fingerprints"]["frontend_provenance_sha256"]
    )
