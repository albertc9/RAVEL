from pathlib import Path

import pytest

from ravel_hls import analyze, convert


def make_temporal_model(*, blocks=2, height=128, width=3, filters=5, prefix="renamed", padding="valid"):
    import keras
    from hgq.layers import QConv2D, QDense

    keras.utils.set_random_seed(1700)
    reference = Path(__file__).parents[2] / "references/cnn_for_arianna/models/cnn_for_arianna.keras"
    base = keras.models.load_model(reference, custom_objects={"QConv2D": QConv2D, "QDense": QDense})
    convolution = next(layer for layer in base.layers if isinstance(layer, QConv2D)).get_config()
    dense = next(layer for layer in base.layers if isinstance(layer, QDense)).get_config()
    inputs = keras.Input((height, width, 1), name=f"{prefix}_input")
    value = inputs
    for block in range(blocks):
        config = {**convolution, "name": f"{prefix}_conv_{block}", "filters": filters[block] if isinstance(filters, tuple) else filters,
                  "kernel_size": (3, 1), "strides": (2, 1),
                  "padding": padding[block] if isinstance(padding, tuple) else padding}
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


def test_analysis_explains_its_analytical_search_without_claiming_vendor_measurements():
    report = analyze(make_temporal_model(height=64, width=2, filters=3), {
        "HLS": {},
        "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()

    search = report["resolved_design"]["optimization_search"]
    assert search["mode"] == "analytical"
    assert search["status"] == "complete"
    assert search["candidate_count"] >= 1
    assert search["selected_candidate"] in {
        candidate["id"] for candidate in search["candidates"]
    }
    assert search["performance_qualification"] == "not_run"


@pytest.mark.parametrize("width, expected_positions", [(1, {1}), (2, {1, 2}), (3, {1, 3})])
def test_analysis_derives_position_candidates_from_geometry_without_promoting_unknown_costs(
    width, expected_positions,
):
    report = analyze(make_temporal_model(height=64, width=width, filters=3), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()

    search = report["resolved_design"]["optimization_search"]
    generated = [entry for entry in search["candidates"] if entry["schedule"] is not None]
    assert {entry["schedule"]["positions"] for entry in generated} == expected_positions
    assert all(entry["confidence"] == "uncalibrated" for entry in generated)
    selected = next(entry for entry in search["candidates"] if entry["id"] == search["selected_candidate"])
    assert selected["schedule"] is None


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
    assert report["resolved_design"]["control"]["reset"]["scope"] == "all-registers"
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
    bridge_owner = next(item["owner"] for item in project.manifest["source_ownership"] if item["path"] == "firmware/nnet_utils/ravel_bridges.h")
    assert bridge_owner == {"id": "lossless-stream-repack", "version": 2}
    assert "firmware/nnet_utils/ravel_bridges.h" in owned
    assert "firmware/nnet_utils/nnet_conv2d_stream.h" not in owned


def test_calibrated_window_conversion_preserves_intermediate_and_final_codes(tmp_path):
    project = convert(make_temporal_model(height=64, width=2, filters=3), tmp_path / "window", {
        "HLS": {"Part": "xcku5p-ffvb676-2-e", "ClockPeriod": 5},
        "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
        "Verification": {"Mode": "required", "Samples": 16},
    })

    stages = project.manifest["resolved_design"]["stages"]
    assert stages[1]["strategy"]["id"] == "aria-window-stream"
    assert project.status["correctness_verification"] == "passed"
    boundaries = project.manifest["verification"]["stage_boundaries"]
    assert boundaries["status"] == "passed"
    assert {entry["tensor_id"] for entry in boundaries["observations"]} >= {
        "conv2d_1:out0", "relu_1:out0", "max_pool2d_1:out0", "dense_0:out0",
    }


def test_search_reports_resource_predictions_and_rejects_an_impossible_core_limit():
    model = make_temporal_model(height=64, width=2, filters=3)
    config = {"HLS": {"Part": "xcku5p-ffvb676-2-e", "ClockPeriod": 5},
              "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1}}
    analysis = analyze(model, config).to_dict()
    search = analysis["resolved_design"]["optimization_search"]
    selected = next(c for c in search["candidates"] if c["id"] == search["selected_candidate"])
    assert set(selected["resources"]["values"]) == {"LUT", "FF", "DSP", "BRAM"}
    assert selected["resources"]["status"] == "predicted"
    assert search["constraints"]["clock_period_ns"] == 5
    assert search["constraints"]["limits_source"] == "device-capacity"
    limited = analyze(model, {**config, "Optimization": {
        **config["Optimization"], "ResourceLimits": {"LUT": 1},
    }}).to_dict()
    assert limited["applicability"]["status"] == "unsupported"
    assert "search.no_feasible_plan" in {f["code"] for f in limited["applicability"]["findings"]}
    assert limited["optimization_search"]["status"] == "complete"


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
    assert len(project.manifest["generated_plan_sha256"]) == 64
    architecture = project.manifest["architecture_envelope"]
    assert architecture["stages"] == project.manifest["resolved_design"]["stages"]
    assert architecture["bridges"] == project.manifest["resolved_design"]["bridges"]
    renewed = refresh(project, model)
    assert renewed.manifest["generated_plan_sha256"] == project.manifest["generated_plan_sha256"]
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


def test_each_block_owns_its_own_channel_and_filter_geometry(tmp_path):
    project = convert(make_temporal_model(height=64, width=2, filters=(3, 4)), tmp_path / "mixed_filters", {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
        "Verification": {"Mode": "required", "Samples": 8},
    })
    stages = project.manifest["resolved_design"]["stages"]
    assert stages[0]["output"]["shape"] == [15, 2, 3]
    assert stages[1]["input"]["lanes"] == 3
    assert stages[1]["output"]["shape"] == [3, 2, 4]
    assert stages[1]["schedule"]["production_after_input_words"] == [9, 10, 17, 18, 25, 26]
    assert stages[1]["schedule"]["drain_input_words"] == 4
    assert project.manifest["resolved_design"]["semantic_stages"][1]["dropped_pool_rows"] == 1
    assert stages[-1]["input"]["lanes"] == 8
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"


def test_padding_remains_visible_as_an_opaque_operation_in_family_diagnostics():
    report = analyze(make_temporal_model(height=64, width=2, filters=3, padding=("valid", "same")), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()
    assert report["applicability"]["status"] == "unsupported"
    assert any(operation["kind"] == "zeropadding2d" for operation in report["model_facts"]["operations"])
    assert any(finding["code"] == "family.topology.sequence" and finding["operation_id"] == "zeropadding2d_0"
               for finding in report["applicability"]["findings"])


def test_conversion_failure_carries_the_same_structured_findings_as_analysis(tmp_path):
    from ravel_hls import CompatibilityError

    model = make_temporal_model(blocks=3, height=512, width=2, filters=3)
    config = {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1}}
    expected = analyze(model, config).to_dict()["applicability"]["findings"]
    with pytest.raises(CompatibilityError) as caught:
        convert(model, tmp_path / "unsupported", config)
    assert list(caught.value.findings) == expected
    assert not (tmp_path / "unsupported").exists()


def test_declared_schedule_bound_is_a_local_failure_without_truncating_the_chain():
    report = analyze(make_temporal_model(height=8192, width=16, filters=3), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()
    assert report["recognition"]["block_count"] == 2
    assert report["resolved_design"] is None
    assert any(finding["code"] == "strategy.schedule.event_bound" and finding["operation_id"] == "conv2d_1"
               for finding in report["applicability"]["findings"])


def test_every_selected_stage_declares_schedule_layout_control_and_source_ownership():
    report = analyze(make_temporal_model(height=64, width=2, filters=3), {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }).to_dict()
    design = report["resolved_design"]
    for stage in design["stages"]:
        assert stage["schedule"]["input_words"] == stage["input"]["words"]
        assert stage["schedule"]["output_words"] == stage["output"]["words"]
        assert stage["execution"]["control"] == "ap_ctrl_hs-dataflow-process"
        assert stage["execution"]["reset"] == "discard-in-flight"
        assert stage["execution"]["source_owner"]
    layout = next(stage for stage in design["semantic_stages"] if stage["kind"] == "layout-view")
    assert layout["mapping"]["input"]["axes"][-1] == {"name": "channel", "extent": 3, "stride": 1}
    assert design["semantic_stages"][-1]["kind"] == "dense-head"


def test_parameter_package_refreshes_second_block_codes_without_changing_the_selected_plan(tmp_path):
    from ravel_hls import Parameters, refresh
    model = make_temporal_model(height=64, width=2, filters=3)
    config = {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
              "Verification": {"Mode": "auto", "Samples": 16}}
    original = convert(model, tmp_path / "parameter_refresh", config)
    kernel = [layer.kernel for layer in model.layers if hasattr(layer, "kernel")][1]
    values = kernel.numpy()
    values.reshape(-1)[0] += 0.125
    kernel.assign(values)
    package = Parameters.extract(model)
    renewed = refresh(original, package)
    assert renewed.manifest["architecture_envelope_sha256"] == original.manifest["architecture_envelope_sha256"]
    assert renewed.manifest["source_model"]["fingerprints"]["parameter_state_sha256"] != original.manifest["source_model"]["fingerprints"]["parameter_state_sha256"]
    assert renewed.manifest["verification"]["transformation_equivalence"] == "passed"
    assert renewed.manifest["verification"]["stage_boundaries"]["status"] == "passed"
    assert renewed.manifest["verification"]["source_conversion_consistency"] == "not_run"


def test_fresh_process_generation_preserves_complete_source_and_plan_fingerprints(tmp_path):
    import json
    import os
    import subprocess
    import sys
    model_path = tmp_path / "model.keras"
    import keras
    model = make_temporal_model(height=64, width=2, filters=3)
    shared_initializer = keras.initializers.GlorotUniform(seed=1700)
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel_initializer = shared_initializer
    model.save(model_path)
    code = '''import sys
from ravel_hls import convert
convert(sys.argv[1], sys.argv[2], {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1}, "Verification": {"Mode": "disabled"}})
'''
    results = []
    for run in ("first", "second"):
        output = tmp_path / run / "same_top"
        subprocess.run([sys.executable, "-c", code, str(model_path), str(output)], env=os.environ.copy(), check=True, capture_output=True, text=True)
        results.append(json.loads((output / "ravel_manifest.json").read_text()))
    for key in ("generated_plan_sha256", "source_closure_sha256", "generation_fingerprint", "architecture_envelope_sha256"):
        assert results[0][key] == results[1][key], key
    restored = keras.models.load_model(output / "keras_model.keras")
    initializers = [layer.kernel_initializer for layer in restored.layers if hasattr(layer, "kernel_initializer")]
    assert len(initializers) == 3
    assert [initializer.get_config() for initializer in initializers] == [{"seed": 1700}] * 3


def test_generated_bridge_preserves_codes_and_zeroes_tail_padding_across_consecutive_calls(tmp_path):
    import subprocess
    from ravel_hls.compatibility.dependencies import inspect_dependencies
    project = convert(make_temporal_model(height=64, width=2, filters=3), tmp_path / "bridge", {
        "HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
        "Verification": {"Mode": "disabled"},
    })
    source = tmp_path / "bridge_test.cpp"
    source.write_text('''#include "defines.h"
#include "nnet_utils/ravel_bridges.h"
#include <cassert>
template<unsigned IN, unsigned OUT, unsigned LANES> void check() {
    using scalar = ap_fixed<8,4>;
    using input_word = nnet::array<scalar, IN>;
    using output_word = nnet::array<scalar, OUT>;
    hls::stream<input_word> input;
    hls::stream<output_word> output;
    const unsigned codes[5] = {128, 127, 255, 1, 73};
    for (unsigned frame=0; frame<3; ++frame) {
        for (unsigned i=0; i<5; i+=IN) {
            input_word word;
            for (unsigned lane=0; lane<IN; ++lane) word[lane].range(7,0)=i+lane<5 ? codes[i+lane] : 99;
            input.write(word);
        }
        ravel::repack<input_word, output_word, 5, LANES>(input, output);
        for (unsigned i=0; i<5; i+=OUT) {
            output_word word=output.read();
            for (unsigned lane=0; lane<OUT; ++lane) assert(word[lane].range(7,0).to_uint() == (i+lane<5 ? codes[i+lane] : 0));
        }
        assert(input.empty() && output.empty());
    }
}
int main() { check<4,2,2>(); check<2,4,2>(); check<3,2,1>(); }
''')
    binary = tmp_path / "bridge_test"
    compiler = inspect_dependencies()["compiler"]["command"]
    assert compiler is not None
    subprocess.run([compiler, "-std=c++17", "-Wno-unknown-pragmas", "-I" + str(project.path / "firmware"),
                    "-I" + str(project.path / "firmware/ap_types"), str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
