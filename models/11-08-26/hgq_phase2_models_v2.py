import os
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'torch'

import torch
import torch.nn as nn
from keras.layers import MaxPooling2D
from hgq.config import QuantizerConfigScope
from hgq.layers import QDense, QConv2D


class HGQPhase2ModelV2(nn.Module):
    """
    Quantized Phase 2 CNN v2: increased precision + wider conv2, smaller dense.

    Changes vs V1:
      b0=10, i0=5   (vs b0=8, i0=4)  — finer weight resolution, reduces logit
                                         saturation (signal outputs ~450 in V1)
      conv2: 32 filters (vs 20)        — more feature detectors for arbitrary
                                         signal position in Phase 2 window
      dense1: 64 units (vs 256)        — 4× fewer MACs on the most expensive
                                         layer; offsets FPGA cost of b0 increase

    FPGA resource estimate vs V1:
      Dense MACs: 1792×64 = 114,688  vs  1120×256 = 286,720  (−60%)
      Bit factor:  (10/8)² = 1.56×
      Net dense:   114,688 × 1.56 / 286,720 ≈ 0.62  (−38% vs V1 dense)

    Base class is nn.Module — see V1 docstring for why keras.Model breaks autograd.
    """

    network_blocks = [None]

    def __init__(self, betavalue=1e-7, input_shape=(256, 4)):
        super().__init__()
        scope = QuantizerConfigScope(
            place='all',
            heterogeneous_axis=(),
            default_q_type='kif',
            overflow_mode='SAT_SYM',
            b0=10, i0=5,
        )
        with scope:
            self.conv1 = QConv2D(
                20, kernel_size=(7, 1), strides=(2, 1),
                padding='valid', activation='relu',
                beta0=betavalue, parallelization_factor=1,
            )
            self.conv2 = QConv2D(
                32, kernel_size=(7, 1), strides=(2, 1),
                padding='valid', activation='relu',
                beta0=betavalue, parallelization_factor=1,
            )
            self.dense1 = QDense(64, beta0=betavalue)
            self.dense2 = QDense(1, beta0=betavalue)

        self.pool1 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))
        self.pool2 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))

    def parameters(self, recurse=True):
        seen = set()
        for p in super().parameters(recurse=recurse):
            if id(p) not in seen:
                seen.add(id(p))
                yield p
        for layer in [self.conv1, self.conv2, self.dense1, self.dense2]:
            for var in layer.trainable_weights:
                t = getattr(var, '_value', None)
                if isinstance(t, torch.Tensor) and t.requires_grad and id(t) not in seen:
                    seen.add(id(t))
                    yield t

    def forward(self, x):
        # x: (B, 4, 256) from DatasetContinuousStreamStitchless
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, 256, 4, 1)
        x = self.conv1(x)              # (B, 125, 4, 20)
        x = self.pool1(x)              # (B,  62, 4, 20)
        x = self.conv2(x)              # (B,  28, 4, 32)
        x = self.pool2(x)              # (B,  14, 4, 32)
        x = x.reshape(x.shape[0], -1)  # (B, 1792)
        x = self.dense1(x)             # (B,   64)
        x = torch.relu(x)
        x = self.dense2(x)             # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)
