from pathlib import Path

import numpy as np
import pytest

from ravel_hls import analyze, convert
from ravel_hls.exceptions import CompatibilityError


PACKING = {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1}}


def conv(filters, kernel=3, stride=2, *, relu=True, pool=True, diagonal=False, groups=1, across=1):
    return {"filters": filters, "kernel": kernel, "stride": stride, "relu": relu,
            "pool": pool, "diagonal": diagonal, "groups": groups, "across": across}


def make_block_chain(blocks, *, height=64, width=2, prefix="renamed", hidden=()):
    """Build an HGQ2 chain of convolutions with optional ReLU, pool and block-diagonal kernels.

    `hidden` lists the units of Dense layers, each followed by ReLU, before the single output.
    """
    import keras
    from hgq.layers import QConv2D, QDense

    keras.utils.set_random_seed(1800)
    reference = Path(__file__).parents[2] / "references/cnn_for_arianna/models/cnn_for_arianna.keras"
    base = keras.models.load_model(reference, custom_objects={"QConv2D": QConv2D, "QDense": QDense})
    convolution = next(layer for layer in base.layers if isinstance(layer, QConv2D)).get_config()
    dense = next(layer for layer in base.layers if isinstance(layer, QDense)).get_config()
    inputs = keras.Input((height, width, 1), name=f"{prefix}_input")
    value = inputs
    diagonal = []
    for index, block in enumerate(blocks):
        layer = QConv2D.from_config({
            **convolution, "name": f"{prefix}_conv_{index}", "filters": block["filters"],
            "kernel_size": (block["kernel"], block["across"]), "strides": (block["stride"], 1), "padding": "valid",
            "activation": "relu" if block["relu"] else "linear", "groups": block["groups"]})
        value = layer(value)
        if block["diagonal"]:
            diagonal.append(layer)
        if block["pool"]:
            value = keras.layers.MaxPool2D((2, 1), strides=(2, 1), name=f"{prefix}_pool_{index}")(value)
    value = keras.layers.Flatten(name=f"{prefix}_view")(value)
    for index, units in enumerate(hidden):
        value = QDense.from_config({**dense, "name": f"{prefix}_hidden_{index}", "units": units})(value)
        value = keras.layers.ReLU(name=f"{prefix}_hidden_relu_{index}")(value)
    head = QDense.from_config({**dense, "name": f"{prefix}_head", "units": 1,
                               **({} if hidden else {"enable_iq": False, "iq_conf": None})})
    model = keras.Model(inputs, head(value))
    for layer in diagonal:
        kernel = np.asarray(keras.ops.convert_to_numpy(layer._kernel))
        channels, filters = kernel.shape[2], kernel.shape[3]
        mask = np.eye(channels, filters, dtype=bool)[None, None]
        layer._kernel.assign(np.where(mask, kernel, 0.0).astype(kernel.dtype))
    return model


def separable_chain(**arguments):
    return make_block_chain([conv(3), conv(3, relu=False, pool=False, diagonal=True),
                             conv(4, kernel=1, stride=1)], **arguments)


def test_a_separable_group_is_recognized_as_blocks_with_optional_activation_and_pool():
    report = analyze(separable_chain(), PACKING).to_dict()

    assert report["recognition"]["block_count"] == 3
    assert report["recognition"]["blocks"][1] == {"convolution": "conv2d_1", "activation": None, "pooling": None}
    assert report["recognition"]["blocks"][2] == {"convolution": "conv2d_2", "activation": "relu_1", "pooling": "max_pool2d_1"}
    assert report["applicability"] == {"status": "applicable", "findings": []}
    stages = report["resolved_design"]["stages"]
    assert [stage["operation_ids"] for stage in stages[:3]] == [
        ["conv2d_0", "relu_0", "max_pool2d_0"], ["conv2d_1"], ["conv2d_2", "relu_1", "max_pool2d_1"]]
    assert report["resolved_design"]["semantic_stages"][1]["dropped_pool_rows"] == 0


def test_a_separable_group_converts_bit_exactly_through_every_owned_boundary(tmp_path):
    project = convert(separable_chain(), tmp_path / "separable", {
        **PACKING, "Verification": {"Mode": "required", "Samples": 8}})

    manifest = project.manifest
    assert manifest["verification"]["stage_boundaries"]["status"] == "passed"
    assert [stage["strategy"]["id"] for stage in manifest["resolved_design"]["stages"]][0] == "aria-wide-stream"


@pytest.mark.parametrize("name, blocks", [
    ("relu_unpooled", [conv(3), conv(3, pool=False), conv(4, kernel=1, stride=1)]),
    ("linear_before_wide", [conv(3), conv(3, relu=False, pool=False), conv(4)]),
    ("three_relu", [conv(3, kernel=5, stride=3), conv(3, kernel=3, stride=1, pool=False), conv(4, kernel=3, stride=3)]),
])
def test_one_unpooled_convolution_block_of_any_form_converts_bit_exactly(tmp_path, name, blocks):
    project = convert(make_block_chain(blocks, height=128), tmp_path / name, {
        **PACKING, "Verification": {"Mode": "required", "Samples": 8}})

    manifest = project.manifest
    assert len(manifest["resolved_design"]["stages"]) == 5
    assert manifest["verification"]["transformation_equivalence"] == "passed"
    assert manifest["verification"]["stage_boundaries"]["status"] == "passed"


@pytest.mark.parametrize("blocks, operation", [
    ([conv(3), conv(3, pool=False), conv(3, pool=False), conv(4)], "conv2d_2"),
    ([conv(3, pool=False), conv(3), conv(4)], "conv2d_0"),
])
def test_other_unpooled_block_chains_are_recognized_but_outside_the_support_domain(blocks, operation):
    report = analyze(make_block_chain(blocks, height=256), PACKING).to_dict()

    assert report["recognition"]["block_count"] == len(blocks)
    assert report["applicability"]["status"] == "unsupported"
    assert [(item["code"], item["operation_id"]) for item in report["applicability"]["findings"]] == [
        ("family.support.block_form", operation)]
    assert report["resolved_design"] is None


def grouped_chain():
    return make_block_chain([conv(4), conv(4, relu=False, pool=False, groups=4), conv(5, kernel=1, stride=1)])


def test_a_grouped_convolution_is_rewritten_with_a_block_diagonal_kernel_of_equal_outputs():
    import numpy as np
    from ravel_hls.frontend.hls4ml import _normalize_grouped_convolutions

    model = grouped_chain()
    rewritten = _normalize_grouped_convolutions(model)
    inputs = np.random.default_rng(1800).normal(0.0, 2.0, (16, 64, 2, 1)).astype("float32")

    assert [layer.groups for layer in rewritten.layers if hasattr(layer, "groups")] == [1, 1, 1]
    kernel = np.asarray(rewritten.layers[3].kernel)
    assert kernel.shape[2:] == (4, 4) and np.count_nonzero(kernel[:, :, ~np.eye(4, dtype=bool)]) == 0
    assert np.array_equal(model.predict(inputs, verbose=0), rewritten.predict(inputs, verbose=0))


def test_a_grouped_convolution_converts_bit_exactly_as_a_separable_group(tmp_path):
    project = convert(grouped_chain(), tmp_path / "grouped", {**PACKING, "Verification": {"Mode": "required", "Samples": 8}})

    stages = project.manifest["resolved_design"]["stages"]
    assert [stage["operation_ids"] for stage in stages[1:3]] == [["conv2d_1"], ["conv2d_2", "relu_1", "max_pool2d_1"]]
    assert project.manifest["verification"]["source_conversion_consistency"] == "passed"
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"


def test_a_grouped_convolution_outside_hgq2_is_reported_before_the_compiler_fails():
    import keras

    inputs = keras.Input((64, 2, 1))
    value = keras.layers.Conv2D(4, (3, 1), strides=(2, 1), activation="relu")(inputs)
    value = keras.layers.MaxPool2D((2, 1), strides=(2, 1))(value)
    value = keras.layers.Conv2D(4, (3, 1), strides=(2, 1), groups=4)(value)
    value = keras.layers.Flatten()(value)
    model = keras.Model(inputs, keras.layers.Dense(1)(value))

    with pytest.raises(CompatibilityError) as raised:
        analyze(model, PACKING)

    assert [finding["code"] for finding in raised.value.findings] == ["frontend.operation.grouped_convolution"]


KU5P = {"Part": "xcku5p-ffvb676-2-e", "ClockPeriod": 5}


@pytest.mark.parametrize("engine, window, matrix", [
    ("aria-affine-window", True, True),
    ("aria-window-stream", True, False),
])
def test_ravel_engines_lower_unpooled_and_pointwise_blocks_bit_exactly(tmp_path, monkeypatch, engine, window, matrix):
    # Selection needs calibrated coverage; open it here to check the lowering itself.
    from ravel_hls.planning import calibration
    monkeypatch.setattr(calibration.WindowCostProfile, "covers", lambda self, *a, **k: window)
    monkeypatch.setattr(calibration.ConstantMatrixCostProfile, "covers", lambda self, *a, **k: matrix)
    model = make_block_chain([conv(3, kernel=5, stride=3), conv(3, kernel=5, stride=2, relu=False, pool=False, diagonal=True),
                              conv(4, kernel=1, stride=1)], height=128, width=4)

    project = convert(model, tmp_path / "engines", {"HLS": KU5P, "Verification": {"Mode": "required", "Samples": 8}})

    design = project.manifest["resolved_design"]
    assert [stage["strategy"]["id"] for stage in design["stages"][1:3]] == [engine, engine]
    assert [stage["operation_ids"] for stage in design["stages"][1:3]] == [["conv2d_1"], ["conv2d_2", "relu_1", "max_pool2d_1"]]
    assert design["optimization_search"]["status"] == "complete"
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"


def test_new_block_forms_use_shared_arithmetic_inside_calibrated_coverage_only():
    model = make_block_chain([conv(3, kernel=5, stride=3), conv(3, kernel=5, stride=2, relu=False, pool=False, diagonal=True),
                              conv(4, kernel=1, stride=1)], height=128, width=4)

    covered = analyze(model, {"HLS": KU5P}).to_dict()["resolved_design"]
    uncovered = analyze(model, {"HLS": {}}).to_dict()["resolved_design"]

    assert [stage["strategy"]["id"] for stage in covered["stages"][1:3]] == ["aria-affine-window", "aria-affine-window"]
    assert covered["optimization_search"]["selection_reason"] == "calibrated-predicted-frame-interval"
    assert covered["optimization_search"]["status"] == "complete"
    assert covered["components"]["cost_policy"]["calibration_profile"]["version"] == 3
    assert [stage["strategy"]["id"] for stage in uncovered["stages"][1:3]] == ["hls4ml-temporal-block", "hls4ml-temporal-block"]
    assert uncovered["optimization_search"]["selection_reason"] == "retain-incumbent-without-calibrated-coverage"


def test_refresh_preserves_the_plan_and_rebuilds_the_arithmetic_of_a_separable_project(tmp_path):
    from ravel_hls import Project, refresh

    model = make_block_chain([conv(3, kernel=7, stride=2), conv(3, kernel=7, stride=2, relu=False, pool=False, diagonal=True),
                              conv(4, kernel=1, stride=1)], height=128, width=4)
    project = convert(model, tmp_path / "separable_refresh", {"HLS": KU5P, "Verification": {"Mode": "required", "Samples": 8}})
    original = project.manifest
    assert [stage["strategy"]["id"] for stage in original["resolved_design"]["stages"][:3]] == [
        "phara", "aria-affine-window", "aria-affine-window"]
    assert Project.open(project.path).status["source_integrity"] == "clean"

    renewed = refresh(project, model).manifest

    assert renewed["schema_version"] == 9
    assert renewed["architecture_envelope_sha256"] == original["architecture_envelope_sha256"]
    for before, after in zip(original["resolved_design"]["stages"], renewed["resolved_design"]["stages"]):
        assert (before["strategy"], before["operation_ids"], before.get("arithmetic_schedule")) == (
            after["strategy"], after["operation_ids"], after.get("arithmetic_schedule"))
    assert renewed["verification"]["stage_boundaries"]["status"] == "passed"


def width_mixing_chain():
    return make_block_chain([conv(3), conv(3, kernel=1, stride=1, pool=False, across=2), conv(4)], height=128, width=2)


def test_dense_parallelism_is_checked_against_the_width_that_reaches_the_dense_head():
    report = analyze(width_mixing_chain(), {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 2}}).to_dict()

    assert report["applicability"]["status"] == "unsupported"
    finding = next(item for item in report["applicability"]["findings"] if item["code"] == "strategy.geometry.dense_stream")
    assert finding["expected"] == "divisor of output width 1"


def test_a_convolution_across_the_width_converts_through_a_delegated_stage(tmp_path):
    project = convert(width_mixing_chain(), tmp_path / "width_mixing", {
        **PACKING, "Verification": {"Mode": "required", "Samples": 8}})

    stages = project.manifest["resolved_design"]["stages"]
    assert (stages[1]["strategy"]["id"], stages[1]["operation_ids"]) == ("hls4ml-temporal-block", ["conv2d_1", "relu_1"])
    assert stages[2]["input"]["shape"][1] == 1
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"
