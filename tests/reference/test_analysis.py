from pathlib import Path

import pytest

from ravel_hls import ConfigurationError, analyze


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
REFERENCE_MODELS = sorted(
    path
    for path in (Path(__file__).parents[2] / "references").glob("**/*.keras")
    if "cnn_for_arianna" not in path.parts
)
SNAPSHOT_ROOT = Path(__file__).with_name("analysis_snapshots")

assert len(REFERENCE_MODELS) == 12


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
    assert report["generation"] == {"id": "aria", "version": "1.5.1"}
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


def test_model_analysis_exposes_read_only_public_properties() -> None:
    analysis = analyze(
        REFERENCE_MODEL,
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    )

    assert analysis.applicable is True
    assert analysis.model_family == {"id": "hgq-conv-pool-dense", "version": 1}
    assert analysis.findings == ()
    assert analysis.model_facts["operations"][0]["id"] == "input_0"
    assert analysis.resolved_design["specialization"]["temporal_packing"] == 4
    with pytest.raises(TypeError):
        analysis.model_facts["schema_version"] = 99

    mutable_copy = analysis.to_dict()
    mutable_copy["model_facts"]["schema_version"] = 99
    assert analysis.model_facts["schema_version"] == 1


def test_same_topology_accepts_extracted_geometry_instead_of_arianna_constants(
    noncanonical_geometry_model,
) -> None:
    report = analyze(
        noncanonical_geometry_model,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 2, "DenseParallelism": 2},
        },
    ).to_dict()

    assert report["applicability"] == {"status": "applicable", "findings": []}
    operations = {item["id"]: item for item in report["model_facts"]["operations"]}
    assert operations["conv2d_0"]["attributes"]["filt_height"] == 3
    assert operations["conv2d_0"]["attributes"]["n_filt"] == 5
    assert operations["dense_0"]["attributes"] == {"n_in": 620, "n_out": 1}
    design = report["resolved_design"]
    assert design["interfaces"]["logical"] == {
        "input_shape": [128, 4],
        "output_shape": [1],
    }
    assert design["interfaces"]["hls_stream"] == {
        "input_rows_per_word": 2,
        "values_per_input_word": 8,
        "input_words_per_inference": 64,
    }
    assert design["implementation_plan"]["first_convolution"] == {
        "id": "full-width-latency",
        "version": 1,
        "parallel_windows": 4,
        "products_per_window": 15,
        "multiplier_limit": 60,
        "target_loop_ii": 1,
    }


def test_same_family_reports_an_unsupported_strategy_before_rendering(
    noncanonical_geometry_model,
) -> None:
    report = analyze(
        noncanonical_geometry_model,
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()

    assert report["model_family"] == {
        "id": "hgq-conv-pool-dense",
        "version": 1,
    }
    assert report["applicability"]["status"] == "unsupported"
    assert report["resolved_design"] is None
    assert "strategy.geometry.p4" in {
        finding["code"] for finding in report["applicability"]["findings"]
    }


@pytest.mark.parametrize(
    ("optimization", "message"),
    [
        ({"TemporalPacking": 3}, "TemporalPacking"),
        ({"DenseParallelism": 5}, "DenseParallelism"),
        ({"Unknown": 1}, "Optimization.Unknown"),
    ],
)
def test_analysis_rejects_unregistered_strategy_choices(
    optimization: dict[str, int], message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        analyze(
            REFERENCE_MODEL,
            {
                "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
                "Optimization": optimization,
            },
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
                "Unknown": {},
            },
            "Unknown RAVEL configuration field",
        ),
        (
            {
                "HLS": {
                    "Backend": "Vitis",
                    "IOType": "io_stream",
                    "Config": {},
                }
            },
            "HLS.Config",
        ),
        (
            {
                "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
                "Verification": {"Samples": 0},
            },
            "Verification.Samples",
        ),
        (
            {
                "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
                "Vitis": {"Stages": {"CoSim": "yes"}},
            },
            "Vitis.Stages.CoSim",
        ),
    ],
)
def test_analysis_validates_the_complete_public_configuration(
    config: dict[str, object], message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        analyze(REFERENCE_MODEL, config)


@pytest.mark.parametrize(
    "model_path", REFERENCE_MODELS, ids=lambda path: path.parent.name
)
def test_retrained_model_analysis_matches_its_reviewed_snapshot(
    model_path: Path,
) -> None:
    import json

    report = analyze(
        model_path,
        {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
    ).to_dict()
    snapshot_path = SNAPSHOT_ROOT / f"{model_path.parent.name}.json"

    assert report == json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_resolved_design_records_the_actual_versioned_resolution_and_pass_chain() -> None:
    design = analyze(
        REFERENCE_MODEL,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
        },
    ).to_dict()["resolved_design"]

    assert design["strategy"] == {"id": "aria-wide-stream", "version": 2}
    assert design["resolver"] == {"id": "aria-explicit-pd", "version": 2}
    assert design["implementation_plan"]["template_profile"] == "aria-p4-d2-v3"
    assert [binding["id"] for binding in design["parameter_bindings"]] == [
        "conv2d_0:bias",
        "conv2d_0:weight",
        "dense_0:bias",
        "dense_0:weight",
    ]
    passes = design["executed_passes"]
    assert [item["id"] for item in passes] == [
        "pack-temporal-input",
        "fuse-repack-into-first-conv",
        "propagate-wide-relu-stream",
        "specialize-nonoverlapping-maxpool",
        "stream-flatten-into-dense",
        "bind-shallow-internal-fifos",
        "elide-dataflow-start-propagation",
    ]
    assert all(item["result"] == "applied" for item in passes)
    assert passes[1]["version"] == 2
    assert all(
        previous["output_design_sha256"] == current["input_design_sha256"]
        for previous, current in zip(passes, passes[1:])
    )
    assert passes[-1]["output_design_sha256"] == design["resolved_design_sha256"]


def test_public_analysis_reports_the_selected_phara_fused_region() -> None:
    design = analyze(
        REFERENCE_MODEL,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 8, "DenseParallelism": 4},
        },
    ).to_dict()["resolved_design"]

    assert design["strategy"] == {"id": "phara", "version": 1}
    assert design["resolver"] == {"id": "aria-aggressive-phara", "version": 1}
    assert design["specialization"] == {
        "temporal_packing": 8,
        "dense_parallelism": 4,
    }
    assert design["streaming"]["phara_fused_region"] == {
        "operation_ids": ["conv2d_0", "relu_0", "max_pool2d_0"],
        "pool_rows_per_supertile": 2,
        "supertile_input_rows": 8,
        "pooled_words": 42,
        "scheduler": {
            "id": "row-credit",
            "version": 1,
            "buffer_rows": 16,
            "max_live_rows": 14,
            "read_cycles": 32,
        },
        "realization": "direct",
    }
    assert "fuse-pool-aligned-conv-relu-maxpool" in {
        item["id"] for item in design["executed_passes"]
    }
    realization = design["coefficient_realization"]
    assert realization["kind"] == "direct"
    assert realization["policy"] == {"id": "phara-direct", "version": 1}
    assert realization["proof"]["status"] == "proven"
    assert len(realization["proof"]["identity"]) == 64
    assert len(realization["graph_sha256"]) == 64
    assert realization["graph_summary"] == {
        "input_rows": 8,
        "convolution_rows": 2,
        "filter_lanes": 7,
        "output_values": 14,
        "constant_nodes": 14,
        "multiply_nodes": 70,
        "add_nodes": 70,
        "depth": 6,
        "max_fanout": 14,
    }
