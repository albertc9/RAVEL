from pathlib import Path

import pytest

from ravel_hls import analyze


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
