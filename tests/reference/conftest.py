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


@pytest.fixture(scope="session")
def stride_two_position_model():
    import keras
    import numpy as np
    from hgq.layers import QConv2D, QDense

    base = keras.models.load_model(
        RETRAINED_ROOT / "adam_p1_step2" / "adam_p1_step2_best.keras",
        custom_objects={"QConv2D": QConv2D, "QDense": QDense},
    )
    inputs = keras.Input((32, 4), name="input_layer")
    x = keras.layers.Reshape((32, 4, 1), name="reshape")(inputs)
    convolution = base.layers[1].get_config()
    convolution.update(
        {
            "name": "q_conv2d",
            "filters": 1,
            "kernel_size": (7, 1),
            "strides": (2, 1),
        }
    )
    convolution_layer = QConv2D.from_config(convolution)
    x = convolution_layer(x)
    x = keras.layers.MaxPooling2D(
        (2, 1), strides=(2, 1), name="max_pooling2d"
    )(x)
    x = keras.layers.Flatten(name="flatten")(x)
    dense = base.layers[-1].get_config()
    dense.update(
        {"name": "q_dense", "units": 1, "enable_iq": False, "iq_conf": None}
    )
    dense_layer = QDense.from_config(dense)
    model = keras.Model(inputs, dense_layer(x))

    convolution_kernel = np.zeros(convolution_layer.kernel.shape, dtype=np.float32)
    convolution_kernel[0, 0, 0, 0] = 1.0
    convolution_kernel[1, 0, 0, 0] = -1.0
    convolution_layer.kernel.assign(convolution_kernel)
    convolution_layer.bias.assign(
        np.zeros(convolution_layer.bias.shape, dtype=np.float32)
    )
    dense_kernel = np.zeros(dense_layer.kernel.shape, dtype=np.float32)
    dense_kernel[0, 0] = 1.0
    dense_kernel[1, 0] = -1.0
    dense_layer.kernel.assign(dense_kernel)
    dense_layer.bias.assign(np.zeros(dense_layer.bias.shape, dtype=np.float32))
    return model
