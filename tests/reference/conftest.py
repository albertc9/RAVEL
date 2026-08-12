from pathlib import Path

import pytest


RETRAINED_ROOT = (
    Path(__file__).parents[2]
    / "references"
    / "fLow_0.08-fhigh_0.23-rate_0.5"
)


@pytest.fixture(scope="session")
def noncanonical_geometry_model():
    import keras
    from hgq.layers import QConv2D, QDense

    base = keras.models.load_model(
        RETRAINED_ROOT / "adam_p1_step2" / "adam_p1_step2_best.keras",
        custom_objects={"QConv2D": QConv2D, "QDense": QDense},
    )
    inputs = keras.Input((128, 4), name="input_layer")
    x = keras.layers.Reshape((128, 4, 1), name="reshape")(inputs)
    convolution = base.layers[1].get_config()
    convolution.update(
        {
            "name": "q_conv2d",
            "filters": 5,
            "kernel_size": (3, 1),
            "strides": (2, 1),
        }
    )
    x = QConv2D.from_config(convolution)(x)
    x = keras.layers.MaxPooling2D(
        (2, 1), strides=(2, 1), name="max_pooling2d"
    )(x)
    x = keras.layers.Flatten(name="flatten")(x)
    dense = base.layers[-1].get_config()
    dense.update(
        {"name": "q_dense", "units": 1, "enable_iq": False, "iq_conf": None}
    )
    return keras.Model(inputs, QDense.from_config(dense)(x))
