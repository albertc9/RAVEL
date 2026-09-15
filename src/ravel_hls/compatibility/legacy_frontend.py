"""The preserved single-block source-input adapter, isolated from the frontend."""
from typing import Any
_DIRECT_ARIA_LAYERS = ("QConv2D", "MaxPooling2D", "Flatten", "QDense")

def _normalize_singleton_channel_input(model: Any) -> Any:
    inputs = tuple(getattr(model, "inputs", ()))
    outputs = tuple(getattr(model, "outputs", ()))
    if len(inputs) != 1 or len(outputs) != 1:
        return model
    input_shape = tuple(inputs[0].shape[1:])
    if (
        len(input_shape) != 3
        or input_shape[-1] != 1
        or any(dimension is None for dimension in input_shape)
    ):
        return model
    layers = [
        layer for layer in model.layers if type(layer).__name__ != "InputLayer"
    ]
    if tuple(type(layer).__name__ for layer in layers) != _DIRECT_ARIA_LAYERS:
        return model

    import keras

    input_name = str(inputs[0].name).split(":", maxsplit=1)[0]
    canonical_input = keras.Input(shape=input_shape[:-1], name=input_name)
    value = keras.layers.Reshape(input_shape, name="ravel_repack")(
        canonical_input
    )
    for layer in layers:
        value = layer(value)
    return keras.Model(canonical_input, value, name=model.name)
