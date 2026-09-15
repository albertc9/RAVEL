from pathlib import Path

import pytest

from ravel_hls import analyze, convert


def make_temporal_model(*, blocks=2, height=128, width=3, filters=5, prefix="renamed"):
    import keras
    from hgq.layers import QConv2D, QDense

    reference = Path(__file__).parents[2] / "references/cnn_for_arianna/models/cnn_for_arianna.keras"
    base = keras.models.load_model(reference, custom_objects={"QConv2D": QConv2D, "QDense": QDense})
    convolution = next(layer for layer in base.layers if isinstance(layer, QConv2D)).get_config()
    dense = next(layer for layer in base.layers if isinstance(layer, QDense)).get_config()
    inputs = keras.Input((height, width, 1), name=f"{prefix}_input")
    value = inputs
    for block in range(blocks):
        config = {**convolution, "name": f"{prefix}_conv_{block}", "filters": filters,
                  "kernel_size": (3, 1), "strides": (2, 1)}
        value = QConv2D.from_config(config)(value)
        value = keras.layers.MaxPool2D((2, 1), strides=(2, 1), name=f"{prefix}_pool_{block}")(value)
    value = keras.layers.Flatten(name=f"{prefix}_view")(value)
    config = {**dense, "name": f"{prefix}_head", "units": 1, "enable_iq": False, "iq_conf": None}
    return keras.Model(inputs, QDense.from_config(config)(value))


def test_analysis_recognizes_a_repeated_temporal_family_before_plan_qualification():
    report = analyze(make_temporal_model(), {
        "HLS": {},
        "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()

    assert report["model_family"] == {"id": "hgq-temporal-block-chain", "version": 1}
    assert report["recognition"]["block_count"] == 2
    assert report["recognition"]["layout"]["input_shape"] == [7, 3, 5]
    assert report["recognition"]["layout"]["output_shape"] == [105]


def test_three_blocks_are_recognized_but_outside_the_qualified_release_domain():
    report = analyze(make_temporal_model(blocks=3, height=512, width=2, filters=3), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()

    assert report["recognition"]["block_count"] == 3
    assert report["applicability"]["status"] == "unsupported"
    assert "family.support.block_count" in {item["code"] for item in report["applicability"]["findings"]}


def test_two_blocks_resolve_existing_specializations_and_explicit_delegation():
    report = analyze(make_temporal_model(), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()

    assert report["applicability"] == {"status": "applicable", "findings": []}
    stages = report["resolved_design"]["stages"]
    assert [stage["strategy"]["id"] for stage in stages] == [
        "aria-wide-stream", "hls4ml-temporal-block", "identity-layout-view", "aria-dense-wide",
    ]
    assert stages[0]["strategy"]["version"] == 2
    assert stages[1]["operation_ids"] == ["conv2d_1", "relu_1", "max_pool2d_1"]
    assert report["resolved_design"]["components"]["bridge_strategies"] == [{"id": "lossless-stream-repack", "version": 2}]
    assert report["resolved_design"]["bridges"]
    assert report["resolved_design"]["delegation"]["hls4ml_version"] == "1.2.0"


def test_composed_two_block_project_is_bit_exact_against_its_clean_baseline(tmp_path):
    project = convert(make_temporal_model(height=64, width=2, filters=3), tmp_path / "composed", {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
        "Verification": {"Mode": "required", "Samples": 16},
    })

    assert project.status["correctness_verification"] == "passed"
    assert project.manifest["verification"]["source_conversion_consistency"] == "passed"
    assert project.manifest["resolved_design"]["strategy"]["id"] == "aria-composed"
    boundaries = project.manifest["verification"]["stage_boundaries"]
    assert boundaries["status"] == "passed"
    assert {entry["tensor_id"] for entry in boundaries["observations"]} >= {"max_pool2d_0:out0", "max_pool2d_1:out0", "reshape_0:out0", "dense_0:out0"}
    assert "nnet::conv_2d_cl" in (project.path / "firmware/composed.cpp").read_text()
    owned = {item["path"] for item in project.manifest["source_ownership"]}
    assert "firmware/nnet_utils/ravel_bridges.h" in owned
    assert "firmware/nnet_utils/nnet_conv2d_stream.h" not in owned


def test_supplied_vectors_augment_the_mandatory_corpus_and_rtl_uses_the_builtin_vectors(tmp_path):
    import numpy as np

    supplied = np.zeros((2, 64, 2, 1), dtype=np.float32)
    project = convert(make_temporal_model(height=64, width=2, filters=3), tmp_path / "corpora", {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
        "Verification": {"Mode": "required", "Samples": 8},
    }, verification_inputs=supplied)

    corpora = project.manifest["verification"]["corpora"]
    assert corpora["built_in"]["recipe"] == {"id": "numeric-contract", "version": 2}
    assert corpora["built_in"]["sample_count"] >= 8
    assert corpora["supplied"]["sample_count"] == 2
    assert all(record["transformation_equivalence"] == "passed" for record in corpora.values())
    assert project.manifest["verification"]["rtl_reference"]["sample_count"] == corpora["built_in"]["sample_count"]
    assert (project.path / "tb_data/rtl_expected_words.hex").is_file()
    rtl_inputs = np.loadtxt(project.path / "tb_data/tb_input_features.dat")
    assert rtl_inputs.shape[0] == corpora["built_in"]["sample_count"]


def test_refresh_preserves_the_composed_plan_and_binds_it_in_the_architecture(tmp_path):
    from ravel_hls import refresh

    model = make_temporal_model(height=64, width=2, filters=3)
    config = {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
              "Verification": {"Mode": "disabled"}}
    project = convert(model, tmp_path / "refreshable", config)
    architecture = project.manifest["architecture_envelope"]
    assert architecture["stages"] == project.manifest["resolved_design"]["stages"]
    assert architecture["bridges"] == project.manifest["resolved_design"]["bridges"]
    renewed = refresh(project, model)
    assert renewed.manifest["architecture_envelope_sha256"] == project.manifest["architecture_envelope_sha256"]
    assert renewed.manifest["resolved_design"]["stages"] == project.manifest["resolved_design"]["stages"]


def test_analysis_preserves_all_declared_ports_before_reporting_a_multi_input_graph_unsupported():
    import keras

    left = keras.Input((8,), name="port_left")
    right = keras.Input((8,), name="port_right")
    model = keras.Model([left, right], keras.layers.Dense(1)(keras.layers.Add()([right, left])))
    report = analyze(model, {"HLS": {}}).to_dict()
    assert report["model_facts"]["inputs"] == ["input_0:out0", "input_1:out0"]
    assert report["applicability"]["status"] == "unsupported"
    assert "family.topology.io" in {finding["code"] for finding in report["applicability"]["findings"]}
