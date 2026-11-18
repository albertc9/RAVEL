from pathlib import Path

import numpy as np
import pytest

from ravel_hls import analyze, convert
from ravel_hls.exceptions import CompatibilityError


PACKING = {"HLS": {}, "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1}}


def conv(filters, kernel=3, stride=2, *, relu=True, pool=True, diagonal=False, groups=1):
    return {"filters": filters, "kernel": kernel, "stride": stride, "relu": relu,
            "pool": pool, "diagonal": diagonal, "groups": groups}


def make_block_chain(blocks, *, height=64, width=2, prefix="renamed"):
    """Build an HGQ2 chain of convolutions with optional ReLU, pool and block-diagonal kernels."""
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
            "kernel_size": (block["kernel"], 1), "strides": (block["stride"], 1), "padding": "valid",
            "activation": "relu" if block["relu"] else "linear", "groups": block["groups"]})
        value = layer(value)
        if block["diagonal"]:
            diagonal.append(layer)
        if block["pool"]:
            value = keras.layers.MaxPool2D((2, 1), strides=(2, 1), name=f"{prefix}_pool_{index}")(value)
    value = keras.layers.Flatten(name=f"{prefix}_view")(value)
    head = QDense.from_config({**dense, "name": f"{prefix}_head", "units": 1, "enable_iq": False, "iq_conf": None})
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


@pytest.mark.parametrize("blocks", [
    [conv(3), conv(3, pool=False), conv(4, kernel=1, stride=1)],
    [conv(3), conv(3, relu=False, pool=False), conv(4)],
])
def test_other_unpooled_block_forms_are_recognized_but_outside_the_support_domain(blocks):
    report = analyze(make_block_chain(blocks, height=128), PACKING).to_dict()

    assert report["recognition"]["block_count"] == 3
    assert report["applicability"]["status"] == "unsupported"
    assert {item["code"] for item in report["applicability"]["findings"]} == {"family.support.block_form"}
    assert report["resolved_design"] is None


def test_a_grouped_convolution_is_reported_before_the_compiler_fails():
    model = make_block_chain([conv(3), conv(3, relu=False, pool=False, groups=3), conv(4, kernel=1, stride=1)])

    with pytest.raises(CompatibilityError) as raised:
        analyze(model, PACKING)

    assert [finding["code"] for finding in raised.value.findings] == ["frontend.operation.grouped_convolution"]
