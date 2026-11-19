# Aria v1.8.0

## What's New

Aria 1.8.0 converts more model shapes at the default P8/D4 setting, while
existing configurations generate the same sources as 1.7.2.

- Convolution blocks may omit the ReLU or the pool, which adds
  depthwise-separable layers and three-convolution chains.
- P8 derives its schedule from the kernel, stride and pool of the first layer
  and is no longer limited to K5/S3.
- One hidden Dense layer is supported. `TargetII` trades its multipliers for
  interval, and two of its products share one DSP multiplier where they fit.
- A standalone Keras `ReLU` layer, which hls4ml 1.2.0 converts incorrectly, is
  now handled.

A trained depthwise-separable model achieves II 48 with 13.86% LUT on KU5P at
200 MHz; C verification, RTL CoSim and protocol checks pass. See the
[design and limits](../design/aria-1.8.0-block-forms.md) and the
[measured qualification](../../references/qualification/aria_1_8_0_separable/README.md).
