"""
hgq_phase2_models_ablation.py — Systematic 2×2 ablation models between mini and Run 4.

Architecture grid (all models: b0=8, i0=4, heterogeneous_axis=(), kif/SAT_SYM):

  Conv layers →       1 conv            |  2 convs
  Hidden dense ↓      (k7,s2,20 filt)   |  (k7,s2,20+20 filt)
  ─────────────────────────────────────────────────────────────
  None                Ab_1cv_nd         |  Ab_2cv_nd
                      ~5,121 params     |  ~4,101 params
  ─────────────────────────────────────────────────────────────
  Dense(64)           Ab_1cv_d64        |  Ab_2cv_d64
                      ~317,729 params   |  ~74,789 params

Reference points (outside this grid):
  mini          Phase 1 arch: conv1(7,k5,s3)→pool→Dense(1)        b0=8,  i0=4  ~1,219   params [trained]
  ab_20f        Bridge:       conv1(20,k5,s3)→pool→Dense(1)        b0=8,  i0=4  ~3,481   params [chain only]
  ab_2cv32_d64  Bridge:       conv1(20)+conv2(32,k7,s2)→Dense(64)  b0=8,  i0=4  ~119,489 params [chain only]
  Run 4         V2 arch:      conv1(20)+conv2(32)→Dense(64)        b0=10, i0=5  ~119,525 params [trained]

Progression chain (one change per arrow):
  mini → [+filters 7→20] → ab_20f → [+kernel k5→k7, +stride s3→s2*] → ab_1cv_nd
       → [+2nd conv] → ab_2cv_nd → [+Dense(64)] → ab_2cv_d64
       → [+conv2 filters 20→32] → ab_2cv32_d64 → [+b0=8→10] → Run 4

  * mini→ab_1cv_nd still changes two things (kernel+stride). Add ab_20f_k7 later if needed.

Shape traces (input (B, 4, 256) → permute+unsqueeze → (B, 256, 4, 1)):
  1-conv path:
    conv1(20,k7,s2,relu)       → (B, 125, 4, 20)
    pool1(2,1)                 → (B,  62, 4, 20)
    flat                       → (B, 4960)

  2-conv path:
    conv1(20,k7,s2,relu)       → (B, 125, 4, 20)
    pool1(2,1)                 → (B,  62, 4, 20)
    conv2(20,k7,s2,relu)       → (B,  28, 4, 20)
    pool2(2,1)                 → (B,  14, 4, 20)
    flat                       → (B, 1120)

Parameter counts (approximate, before HGQ quantizer overhead):
  Ab_1cv_nd:  160 (conv1) + 4,961 (dense_out)              = 5,121
  Ab_1cv_d64: 160 (conv1) + 317,504 (dense64) + 65         = ~317,729  ← dense dominates
  Ab_2cv_nd:  160 (conv1) + 2,820 (conv2) + 1,121          = 4,101
  Ab_2cv_d64: 160 (conv1) + 2,820 (conv2) + 71,744 + 65    = 74,789

  Note: Ab_1cv_d64 has ~4× more params than Ab_2cv_d64 because the 1-conv flat
  size is 4960 vs 1120 — the 2nd conv acts as a strong spatial compressor.
  Ab_1cv_d64 dense1: 4960×64 + 64 = 317,504 (not 76K — easy to confuse with mini-flat).

Build call reference (for TrainChunked_hgq_ablation.py and eval):
  Ab_1cv_nd:  conv1.build((1,256,4,1)); dense1.build((1,4960))
  Ab_1cv_d64: conv1.build((1,256,4,1)); dense1.build((1,4960)); dense2.build((1,64))
  Ab_2cv_nd:  conv1.build((1,256,4,1)); conv2.build((1,62,4,20)); dense1.build((1,1120))
  Ab_2cv_d64: conv1.build((1,256,4,1)); conv2.build((1,62,4,20)); dense1.build((1,1120)); dense2.build((1,64))
"""

import os
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'torch'

import torch
import torch.nn as nn
from keras.layers import MaxPooling2D
from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QDense, QConv2D

# Shared quantizer config for all ablation models — holds bit width constant
# so architecture is the only variable. b0=8, i0=4 matches mini (Phase 1 baseline).
_AB_B0, _AB_I0 = 8, 4
_AB_BETAVALUE   = 1e-7


def _make_scope():
    return QuantizerConfigScope(
        place='all',
        heterogeneous_axis=(),     # single bit-width per tensor, not per-axis
        default_q_type='kif',
        overflow_mode='SAT_SYM',
        b0=_AB_B0, i0=_AB_I0,
    )


def _fixed_output_quantizer():
    """Albert's request (2026-09-24): score fixed at ap_fixed<12,8> = sign + 7 integer + 4 fractional bits,
    range [-128, 127.9375], step 0.0625, round-to-nearest, plain saturation. Built OUTSIDE _make_scope() so the
    ablation defaults (i0=4, SAT_SYM) cannot leak in; frozen after construction (see _K5S3TwoConv)."""
    return QuantizerConfig('kif', 'datalane', k0=True, i0=7, f0=4, round_mode='RND', overflow_mode='SAT',
                           heterogeneous_axis=(), homogeneous_axis=None)


def _hgq_parameters(model, layer_list):
    """Yield deduplicated parameters: torch nn.Module params + HGQ Keras weight tensors."""
    seen = set()
    for p in nn.Module.parameters(model):
        if id(p) not in seen:
            seen.add(id(p))
            yield p
    for layer in layer_list:
        for var in layer.trainable_weights:
            t = getattr(var, '_value', None)
            if isinstance(t, torch.Tensor) and t.requires_grad and id(t) not in seen:
                seen.add(id(t))
                yield t


# ── Ab_20f: 1 conv (Phase 1 geometry), 20 filters, no hidden dense ───────────
# Chain model between mini and ab_1cv_nd. Isolates filter count (7→20) while
# keeping mini's kernel size (k5) and stride (s3).

class HGQPhase2Ab_20f(nn.Module):
    """
    Chain bridge: 1 conv (Phase 1 kernel/stride), 20 filters, no hidden dense.
    Conv(20, k5×1, s3×1) → Pool(2,1) → Flat(3360) → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~3,481 params.

    Isolates the effect of filter count (7→20) from mini while keeping Phase 1
    conv geometry (k5, s3). The next step in the chain (ab_1cv_nd) then changes
    kernel (k5→k7) and stride (s3→s2) together.

    Shape trace:
      (B,256,4,1) → Conv(20,k5,s3) → (B,84,4,20) → Pool(2,1) → (B,42,4,20) → Flat(3360)
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4, 1)
        x = self.conv1(x)                     # (B,  84, 4, 20)
        x = self.pool1(x)                     # (B,  42, 4, 20)
        x = x.reshape(x.shape[0], -1)         # (B, 3360)
        x = self.dense1(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Ab_2cv32_d64: 2 convs (conv2=32 filters) + Dense(64) ─────────────────────
# Chain model between ab_2cv_d64 and Run 4. Isolates conv2 filter count (20→32)
# while keeping b0=8 (next step to Run 4 changes only b0=8→10).

class HGQPhase2Ab_2cv32_d64(nn.Module):
    """
    Chain bridge: 2 conv layers (conv2=32 filters), Dense(64) hidden.
    Conv(20,k7,s2) → Pool → Conv(32,k7,s2) → Pool → Flat(1792) → QDense(64) → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~119,489 params.

    Differs from ab_2cv_d64 ONLY by conv2 filter count (20→32).
    Differs from Run 4 (V2) ONLY by bit widths (b0=8,i0=4 vs b0=10,i0=5).
    Isolates the two remaining confounders between the ablation grid and Run 4.

    Shape trace:
      (B,256,4,1) → Conv(20,k7,s2) → (B,125,4,20) → Pool → (B,62,4,20)
                  → Conv(32,k7,s2) → (B,28,4,32)  → Pool → (B,14,4,32) → Flat(1792)
                  → Dense(64) + ReLU → Dense(1)
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(32, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(64, beta0=betavalue)
            self.dense2 = QDense(1,  beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1, self.dense2])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4,  1)
        x = self.conv1(x)                     # (B, 125, 4, 20)
        x = self.pool1(x)                     # (B,  62, 4, 20)
        x = self.conv2(x)                     # (B,  28, 4, 32)
        x = self.pool2(x)                     # (B,  14, 4, 32)
        x = x.reshape(x.shape[0], -1)         # (B, 1792)
        x = self.dense1(x)                    # (B,   64)
        x = torch.relu(x)
        x = self.dense2(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Ab_1cv_nd: 1 conv, no hidden dense ───────────────────────────────────────

class HGQPhase2Ab_1cv_nd(nn.Module):
    """
    Ablation: 1 conv layer, no hidden dense.
    Conv(20,k7,s2) → Pool → Flat(4960) → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~5,121 params.
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4, 1)
        x = self.conv1(x)                     # (B, 125, 4, 20)
        x = self.pool1(x)                     # (B,  62, 4, 20)
        x = x.reshape(x.shape[0], -1)         # (B, 4960)
        x = self.dense1(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Ab_1cv_d64: 1 conv + Dense(64) ───────────────────────────────────────────

class HGQPhase2Ab_1cv_d64(nn.Module):
    """
    Ablation: 1 conv layer, Dense(64) hidden layer.
    Conv(20,k7,s2) → Pool → Flat(4960) → QDense(64) → ReLU → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~317,729 params  (dense1 dominates: 4960×64+64 = 317,504 — large flat from 1-conv).
    WARNING: commonly confused with ~76K — that figure assumed mini's flat size (1176),
    not the correct 1-conv flat size (4960). Ab_1cv_d64 is the heaviest ablation model.

    The 1-conv flat size (4960) is ~4× larger than the 2-conv flat (1120), making
    Ab_1cv_d64 ~4× larger than Ab_2cv_d64 despite identical dense architecture.
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(64, beta0=betavalue)
            self.dense2 = QDense(1,  beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.dense1, self.dense2])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4, 1)
        x = self.conv1(x)                     # (B, 125, 4, 20)
        x = self.pool1(x)                     # (B,  62, 4, 20)
        x = x.reshape(x.shape[0], -1)         # (B, 4960)
        x = self.dense1(x)                    # (B,   64)
        x = torch.relu(x)
        x = self.dense2(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Ab_2cv_nd: 2 convs, no hidden dense ──────────────────────────────────────

class HGQPhase2Ab_2cv_nd(nn.Module):
    """
    Ablation: 2 conv layers, no hidden dense.
    Conv(20,k7,s2) → Pool → Conv(20,k7,s2) → Pool → Flat(1120) → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~4,101 params.

    The 2nd conv compresses the spatial dimension (62→14 time steps), giving a
    small flat size (1120) despite the same conv filter count as Ab_1cv_nd.
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4, 1)
        x = self.conv1(x)                     # (B, 125, 4, 20)
        x = self.pool1(x)                     # (B,  62, 4, 20)
        x = self.conv2(x)                     # (B,  28, 4, 20)
        x = self.pool2(x)                     # (B,  14, 4, 20)
        x = x.reshape(x.shape[0], -1)         # (B, 1120)
        x = self.dense1(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Ab_2cv_d64: 2 convs + Dense(64) ──────────────────────────────────────────

class HGQPhase2Ab_2cv_d64(nn.Module):
    """
    Ablation: 2 conv layers, Dense(64) hidden layer.
    Conv(20,k7,s2) → Pool → Conv(20,k7,s2) → Pool → Flat(1120) → QDense(64) → ReLU → QDense(1)
    b0=8, i0=4, heterogeneous_axis=(), betavalue=1e-7.
    ~74,789 params.

    Architecturally closest to Run 4 (V2), but with:
      - conv2 = 20 filters (V2 uses 32)
      - b0=8, i0=4 (V2 uses b0=10, i0=5)
    These two differences are the remaining confounders between Ab_2cv_d64 and Run 4.
    """
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(20, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(64, beta0=betavalue)
            self.dense2 = QDense(1,  beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1, self.dense2])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)   # (B, 256, 4, 1)
        x = self.conv1(x)                     # (B, 125, 4, 20)
        x = self.pool1(x)                     # (B,  62, 4, 20)
        x = self.conv2(x)                     # (B,  28, 4, 20)
        x = self.pool2(x)                     # (B,  14, 4, 20)
        x = x.reshape(x.shape[0], -1)         # (B, 1120)
        x = self.dense1(x)                    # (B,   64)
        x = torch.relu(x)
        x = self.dense2(x)                    # (B,    1)
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── K5S3 convertible candidates ───────────────────────────────────────────────
# Mini conv geometry (k5, s3, pool 2): the 8-row pooled region the FPGA converter's fast P4/P8 schedulers support.

class _K5S3OneConv(nn.Module):
    """Conv(F, k5×1, s3×1) → Pool(2,1) → Flat(168·F) → QDense(1). Mini structure with F filters.
    Shapes: (B,256,4,1) → (B,84,4,F) → (B,42,4,F) → (B,168·F) → (B,1)."""
    network_blocks = [None]
    FILTERS = None

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(self.FILTERS, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


class _K5S3TwoConv(nn.Module):
    """Conv(F, k5, s3) → Pool → Conv(F, kK2, s3) → Pool → Flat → QDense(1).
    K2=5: (B,256,4,1) → (B,84,4,F) → (B,42,4,F) → (B,13,4,F) → (B,6,4,F) → (B,24·F) → (B,1)  [pool2 drops 1 row]
    K2=3: (B,256,4,1) → (B,84,4,F) → (B,42,4,F) → (B,14,4,F) → (B,7,4,F) → (B,28·F) → (B,1)  [non-overlapping, no drop]"""
    network_blocks = [None]
    FILTERS = None
    CONV2_KERNEL = 5
    FIXED_OUTPUT = False   # True -> dense1 output quantized to ap_fixed<12,8> (frozen, not trained)

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        oq_conf = _fixed_output_quantizer() if self.FIXED_OUTPUT else None
        with _make_scope():
            self.conv1  = QConv2D(self.FILTERS, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(self.FILTERS, kernel_size=(self.CONV2_KERNEL,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            if self.FIXED_OUTPUT:
                self.dense1 = QDense(1, beta0=betavalue, enable_oq=True, oq_conf=oq_conf)
            else:
                self.dense1 = QDense(1, beta0=betavalue)
        if self.FIXED_OUTPUT:
            self.dense1.oq.trainable = False   # keras-level freeze (drops out of trainable_weights)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        params = _hgq_parameters(self, [self.conv1, self.conv2, self.dense1])
        if not self.FIXED_OUTPUT:
            return params
        # torch-level freeze: on the torch backend every keras variable is ALSO a registered torch Parameter, and
        # _hgq_parameters starts from nn.Module.parameters(), which ignores keras' trainable flag. Exclude the output
        # quantizer's k/i/f explicitly (caught by CheckO12Model: 1123 vs 1120) so the ap_fixed<12,8> format is fixed.
        frozen = set()
        for w in self.dense1.oq.weights:
            t = getattr(w, '_value', w)
            if isinstance(t, torch.Tensor):
                t.requires_grad_(False)
                frozen.add(id(t))
        return (p for p in params if id(p) not in frozen)

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


class HGQPhase2K5S3_f12(_K5S3OneConv):
    FILTERS = 12  # ~2,089 params, ~22K MACs


class HGQPhase2K5S3_f32(_K5S3OneConv):
    FILTERS = 32  # ~5,569 params, ~59K MACs


class HGQPhase2K5S3_2cv_f12(_K5S3TwoConv):
    FILTERS = 12  # ~1,093 params, ~58K MACs


class HGQPhase2K5S3_2cv_f12_o12(_K5S3TwoConv):
    """2cv_k5s3_f12 with the score fixed at ap_fixed<12,8> (Albert, 2026-09-24). Otherwise identical, so a retrain
    on the same data/seed is a clean A/B of whether constraining the output shrinks the learned internal widths."""
    FILTERS = 12
    FIXED_OUTPUT = True


class HGQPhase2K5S3_2cv_f20(_K5S3TwoConv):
    FILTERS = 20  # ~2,621 params, ~138K MACs


class HGQPhase2K5S3_2cv_k3s3_f20(_K5S3TwoConv):
    FILTERS = 20
    CONV2_KERNEL = 3  # ~1,901 params, ~101K MACs


# ── Task D candidates (2026-10-26): between f12 (~58K MACs) and Run 4 (~687K), one change each vs f12 ──────────────
# The ablation pointed at time resolution (conv stride), not width, as what separates f12 from Run 4.

class HGQPhase2K5S3_2cv_f12_d16(_K5S3TwoConv):
    """A1: f12 + a small hidden dense layer (Run 4 has Dense(64)).
    ... → Flat(288) → QDense(16) → ReLU → QDense(1).  ~5,445 params, ~62K MACs."""
    FILTERS = 12

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__(betavalue, input_shape)
        with _make_scope():
            self.dense1 = QDense(16, beta0=betavalue)
            self.dense2 = QDense(1, beta0=betavalue)

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1, self.dense2])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = torch.relu(self.dense1(x.reshape(x.shape[0], -1)))
        return self.dense2(x)


class HGQPhase2K5S3_K3S1_f12(nn.Module):
    """A2: f12 with the second conv at stride 1 (finer time resolution in layer 2 only).
    Conv(12,k5,s3) → Pool → Conv(12,k3,s1) → Pool → Flat(960) → QDense(1)
    (B,256,4,1) → (B,84,4,12) → (B,42,4,12) → (B,40,4,12) → (B,20,4,12) → (B,960).  ~1,477 params, ~90K MACs."""
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(3,1), strides=(1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


class HGQPhase2K7S2_2cv_f12(HGQPhase2Ab_2cv_nd):
    """B1: the ab_2cv_nd geometry (k7s2 twice, no hidden dense; ~96% of Run 4 on prod_p2_v1) at f12's width.
    (B,256,4,1) → (B,125,4,12) → (B,62,4,12) → (B,28,4,12) → (B,14,4,12) → (B,672).  ~1,789 params, ~156K MACs."""

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        nn.Module.__init__(self)
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(7,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))


class HGQPhase2Dec2_K5S1K5S2_f12(nn.Module):
    """C1: for band-limited input (amphp_lp150: < 0.5% of the noise power above 220 MHz) keep every 2nd sample
    (0.5 GSa/s; in firmware a decimation after the digital low-pass), then finer convs at the same width as f12.
    x[..., ::2] → Conv(12,k5,s1) → Pool → Conv(12,k5,s2) → Pool → Flat(672) → QDense(1)
    (B,128,4,1) → (B,124,4,12) → (B,62,4,12) → (B,29,4,12) → (B,14,4,12) → (B,672).  ~1,477 params, ~114K MACs.
    At 0.5 GSa/s conv1 spans 10 ns in 2-ns steps (f12: 5 ns in 3-ns steps)."""
    network_blocks = [None]

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(5,1), strides=(1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(5,1), strides=(2,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1])

    def forward(self, x):
        x = x[..., ::2]                       # (B, 4, 128): 1 GSa/s -> 0.5 GSa/s
        x = x.permute(0,2,1).unsqueeze(-1)    # (B, 128, 4, 1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


# ── Next round (2026-11-01): one change each vs the current model (2cv_k5s3_f12, k5s3 12+12), all on amphp ───────────

class HGQPhase2K5S3_2cv_f8(_K5S3TwoConv):
    """Filter count alone: 8+8 instead of 12+12.  (B,256,4,1) → ... → (B,6,4,8) → (B,192).  ~569 params, ~30K MACs."""
    FILTERS = 8


class HGQPhase2Dec2_2cv_k5s3_f12(HGQPhase2K5S3_2cv_f12):
    """Sample rate alone: same layer settings as the current model, FPGA keeps every 2nd sample (no low-pass).
    x[..., ::2] → (B,128,4,1) → (B,42,4,12) → (B,21,4,12) → (B,6,4,12) → (B,3,4,12) → (B,144).  ~949 params, ~28K MACs."""

    def forward(self, x):
        return super().forward(x[..., ::2])    # (B, 4, 128): 1 GSa/s -> 0.5 GSa/s


class _FirstLayerVariant(nn.Module):
    """Conv(12, kK1, sS1) → Pool → Conv(12, k5, s3) → Pool → Flat → QDense(1): only the first layer differs from f12."""
    network_blocks = [None]
    K1, S1 = None, None

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(self.K1,1), strides=(self.S1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


class HGQPhase2K7S3K5S3_f12(_FirstLayerVariant):
    """First-layer kernel alone: k7 s3.  (B,256,4,1) → (B,84,4,12) → same as f12.  ~1,117 params, ~66K MACs."""
    K1, S1 = 7, 3


class HGQPhase2K5S2K5S3_f12(_FirstLayerVariant):
    """First-layer step alone: k5 s2.  (B,126,4,12) → (B,63,4,12) → (B,20,4,12) → (B,10,4,12) → (B,480).  ~1,285 params, ~88K MACs."""
    K1, S1 = 5, 2


class _ThreeLayer(nn.Module):
    """conv1 → Pool → conv2 → conv3 → Pool → Flat → QDense(1); subclasses define the three conv layers."""
    network_blocks = [None]

    def _pools(self):
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool3 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    def parameters(self, recurse=True):
        return _hgq_parameters(self, [self.conv1, self.conv2, self.conv3, self.dense1])

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.pool3(self.conv3(self.conv2(x)))
        return self.dense1(x.reshape(x.shape[0], -1))

    def obtain_pre_activation(self, x):
        return self.forward(x)


class HGQPhase2Light3cv(_ThreeLayer):
    """More, lighter layers (Albert 2026-10-28: the slowest layer sets the throughput): same total compute, end positions
    and time span as f12, with a lighter heaviest layer.
    Conv(8,k5,s3) → Pool → Conv(8,k3,s1) → Conv(12,k3,s3) → Pool → Flat(288) → QDense(1)
    (B,256,4,1) → (B,84,4,8) → (B,42,4,8) → (B,40,4,8) → (B,13,4,12) → (B,6,4,12) → (B,288).  ~837 params, ~59K MACs."""

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(8, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(8, kernel_size=(3,1), strides=(1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv3  = QConv2D(12, kernel_size=(3,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self._pools()


class HGQPhase2SepConv2_f12(_ThreeLayer):
    """Depthwise-separable second layer (suggested by Erin Ewart, reading A). Only conv2: conv1 has one input channel, so a
    depthwise conv1 would make every filter a scaled copy of one shape.
    Conv(12,k5,s3) → Pool → depthwise k5 s3 (each of the 12 channels filtered separately, no activation) → Conv(12,1x1) → Pool
    → Flat(288) → QDense(1).  Same time axis as f12.  ~589 params, ~31K MACs."""

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(5,1), strides=(3,1), groups=12,
                                  padding='valid', activation=None,
                                  beta0=betavalue, parallelization_factor=1)
            self.conv3  = QConv2D(12, kernel_size=(1,1), strides=(1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self._pools()


class HGQPhase2MixL1_f12(_ThreeLayer):
    """Antennas mixed after the first layer (suggested by Erin Ewart, reading B). The (1,4) kernel combines the 4 antennas at
    each time position; everything after it sees one combined channel set.
    Conv(12,k5,s3) → Pool → Conv(12,(1,4)) → Conv(12,k5,s3) → Pool → Flat(72) → QDense(1)
    (B,256,4,1) → (B,84,4,12) → (B,42,4,12) → (B,42,1,12) → (B,13,1,12) → (B,6,1,12) → (B,72).  ~1,465 params, ~54K MACs."""

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1  = QConv2D(12, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv2  = QConv2D(12, kernel_size=(1,4), strides=(1,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.conv3  = QConv2D(12, kernel_size=(5,1), strides=(3,1),
                                  padding='valid', activation='relu',
                                  beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(1, beta0=betavalue)
        self._pools()


class HGQPhase2AntCh_f12(HGQPhase2K5S3_2cv_f12):
    """Antennas as input channels (earliest possible mixing): the first-layer filters see all 4 antennas at once.
    (B,256,1,4) → (B,84,1,12) → (B,42,1,12) → (B,13,1,12) → (B,6,1,12) → (B,72).  ~1,057 params, ~30K MACs."""

    def forward(self, x):
        x = x.permute(0,2,1).unsqueeze(2)      # (B, 256, 1, 4)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.dense1(x.reshape(x.shape[0], -1))


# ── Compute-saver round (2026-11-09): cheaper versions of the two best designs + "swing" models ────────────────────
# One configurable class: [keep every 2nd sample] → conv1 → Pool → conv2 (ordinary, or depthwise + 1x1 conv3) → Pool →
# Flat → [QDense(H) → ReLU] → QDense(1). The depthwise conv has no activation, as in sepconv2_f12.

class _FlexCNN(nn.Module):
    network_blocks = [None]
    DEC = False            # True: x[..., ::2] (0.5 GSa/s; the 150 MHz low-pass is applied before, in the amp chain)
    F1, K1, S1 = 12, 5, 3  # conv1 filters / kernel / step
    F2, K2, S2 = 12, 5, 3  # conv2 filters (ordinary conv; with DW: 1x1 output channels) / kernel / step
    DW = False             # depthwise conv2 (groups = F1) + 1x1 conv3 to F2 channels
    HIDDEN = 0             # hidden dense units (0 = none)

    def __init__(self, betavalue=_AB_BETAVALUE, input_shape=(256, 4)):
        super().__init__()
        with _make_scope():
            self.conv1 = QConv2D(self.F1, kernel_size=(self.K1, 1), strides=(self.S1, 1), padding='valid',
                                 activation='relu', beta0=betavalue, parallelization_factor=1)
            if self.DW:
                self.conv2 = QConv2D(self.F1, kernel_size=(self.K2, 1), strides=(self.S2, 1), groups=self.F1,
                                     padding='valid', activation=None, beta0=betavalue, parallelization_factor=1)
                self.conv3 = QConv2D(self.F2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                     activation='relu', beta0=betavalue, parallelization_factor=1)
            else:
                self.conv2 = QConv2D(self.F2, kernel_size=(self.K2, 1), strides=(self.S2, 1), padding='valid',
                                     activation='relu', beta0=betavalue, parallelization_factor=1)
            self.dense1 = QDense(self.HIDDEN or 1, beta0=betavalue)
            if self.HIDDEN:
                self.dense2 = QDense(1, beta0=betavalue)
        self.pool1 = MaxPooling2D(pool_size=(2,1), strides=(2,1))
        self.pool2 = MaxPooling2D(pool_size=(2,1), strides=(2,1))

    @classmethod
    def shapes(cls):
        t0 = 128 if cls.DEC else 256
        t1 = (t0 - cls.K1) // cls.S1 + 1
        t2 = (t1 // 2 - cls.K2) // cls.S2 + 1
        return t0, t1, t2, (t2 // 2) * 4 * cls.F2

    def layers(self):
        return [l for l in (getattr(self, n, None) for n in ("conv1", "conv2", "conv3", "dense1", "dense2")) if l is not None]

    def parameters(self, recurse=True):
        return _hgq_parameters(self, self.layers())

    def forward(self, x):
        if self.DEC:
            x = x[..., ::2]
        x = x.permute(0,2,1).unsqueeze(-1)
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        if self.DW:
            x = self.conv3(x)
        x = self.dense1(self.pool2(x).reshape(x.shape[0], -1))
        if self.HIDDEN:
            x = self.dense2(torch.relu(x))
        return x

    def obtain_pre_activation(self, x):
        return self.forward(x)


def _flex_build(m):
    t0, t1, t2, flat = m.shapes()
    m.conv1.build((1, t0, 4, 1))
    m.conv2.build((1, t1 // 2, 4, m.F1))
    if m.DW:
        m.conv3.build((1, t2, 4, m.F1))
    m.dense1.build((1, flat))
    if m.HIDDEN:
        m.dense2.build((1, m.HIDDEN))


class HGQPhase2K5S3_2cv_f6(_K5S3TwoConv):
    FILTERS = 6   # filter floor: 367 params, ~20K MACs


class HGQPhase2K5S3_2cv_f4(_K5S3TwoConv):
    FILTERS = 4   # 205 params, ~11K MACs


class HGQPhase2K5S3_2cv_f2(_K5S3TwoConv):
    FILTERS = 2   # 83 params, ~4K MACs


class HGQPhase2K7S2_f8(_FlexCNN):          # B1 at 8 filters
    F1, K1, S1, F2, K2, S2 = 8, 7, 2, 8, 7, 2


class HGQPhase2K7S2_dw12(_FlexCNN):        # B1 with a depthwise-separable conv2
    F1, K1, S1, F2, K2, S2, DW = 12, 7, 2, 12, 7, 2, True


class HGQPhase2K7S2_f8_dw(_FlexCNN):       # B1 at 8 filters + depthwise-separable conv2
    F1, K1, S1, F2, K2, S2, DW = 8, 7, 2, 8, 7, 2, True


class HGQPhase2Dec2_f8(_FlexCNN):          # C1 (low-pass + half rate, k5s1 / k5s2) at 8 filters
    DEC, F1, K1, S1, F2, K2, S2 = True, 8, 5, 1, 8, 5, 2


class HGQPhase2Dec2_dw12(_FlexCNN):        # C1 with a depthwise-separable conv2
    DEC, F1, K1, S1, F2, K2, S2, DW = True, 12, 5, 1, 12, 5, 2, True


class HGQPhase2Dec2_f8_dw(_FlexCNN):       # C1 at 8 filters + depthwise-separable conv2
    DEC, F1, K1, S1, F2, K2, S2, DW = True, 8, 5, 1, 8, 5, 2, True


class HGQPhase2SwingS1(_FlexCNN):
    """Swing model S1: low-pass + half rate, conv1 16 filters k5 s1 (10 ns / 2 ns), depthwise k5 s2 + 1x1 to 16,
    Dense(32) + ReLU (Run 4's hidden layer at half width). Heaviest layer conv1 39,680 MACs (1.06x the current CNN)."""
    DEC, F1, K1, S1, F2, K2, S2, DW, HIDDEN = True, 16, 5, 1, 16, 5, 2, True, 32


class HGQPhase2SwingS2(_FlexCNN):
    """Swing model S2: full rate (no low-pass), conv1 12 filters k7 s2, depthwise k7 s2 + 1x1 to 16, Dense(32) + ReLU.
    Heaviest layer conv1 42,000 MACs (1.12x the current CNN)."""
    F1, K1, S1, F2, K2, S2, DW, HIDDEN = 12, 7, 2, 16, 7, 2, True, 32


# ── Registry ──────────────────────────────────────────────────────────────────

ABLATION_REGISTRY = {
    # model_type_key : (ModelClass, build_fn, checkpoint_name)
    "k5s3_f12": (
        HGQPhase2K5S3_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.dense1.build((1, 2016))),
        "adam_p2v1_hgq_k5s3_f12",
    ),
    "k5s3_f32": (
        HGQPhase2K5S3_f32,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.dense1.build((1, 5376))),
        "adam_p2v1_hgq_k5s3_f32",
    ),
    "2cv_k5s3_f12": (
        HGQPhase2K5S3_2cv_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.dense1.build((1, 288))),
        "adam_p2v1_hgq_2cv_k5s3_f12",
    ),
    "2cv_k5s3_f12_o12": (
        HGQPhase2K5S3_2cv_f12_o12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.dense1.build((1, 288))),
        "adam_p2v1_hgq_2cv_k5s3_f12_o12",
    ),
    "2cv_k5s3_f20": (
        HGQPhase2K5S3_2cv_f20,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 20)), m.dense1.build((1, 480))),
        "adam_p2v1_hgq_2cv_k5s3_f20",
    ),
    "2cv_k3s3_f20": (
        HGQPhase2K5S3_2cv_k3s3_f20,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 20)), m.dense1.build((1, 560))),
        "adam_p2v1_hgq_2cv_k3s3_f20",
    ),
    # Task D candidates (2026-10-26)
    "2cv_k5s3_f12_d16": (
        HGQPhase2K5S3_2cv_f12_d16,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.dense1.build((1, 288)),
                   m.dense2.build((1, 16))),
        "adam_p2v6_hgq_2cv_k5s3_f12_d16",
    ),
    "k5s3k3s1_f12": (
        HGQPhase2K5S3_K3S1_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.dense1.build((1, 960))),
        "adam_p2v6_hgq_k5s3k3s1_f12",
    ),
    "2cv_k7s2_f12": (
        HGQPhase2K7S2_2cv_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 62, 4, 12)), m.dense1.build((1, 672))),
        "adam_p2v6_hgq_2cv_k7s2_f12",
    ),
    "dec2_k5s1k5s2_f12": (
        HGQPhase2Dec2_K5S1K5S2_f12,
        lambda m: (m.conv1.build((1, 128, 4, 1)), m.conv2.build((1, 62, 4, 12)), m.dense1.build((1, 672))),
        "adam_p2v6_hgq_dec2_k5s1k5s2_f12",
    ),
    # Next round (2026-11-01)
    "2cv_k5s3_f8": (
        HGQPhase2K5S3_2cv_f8,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 8)), m.dense1.build((1, 192))),
        "adam_p2v6_hgq_2cv_k5s3_f8",
    ),
    "dec2_2cv_k5s3_f12": (
        HGQPhase2Dec2_2cv_k5s3_f12,
        lambda m: (m.conv1.build((1, 128, 4, 1)), m.conv2.build((1, 21, 4, 12)), m.dense1.build((1, 144))),
        "adam_p2v6_hgq_dec2_2cv_k5s3_f12",
    ),
    "k7s3k5s3_f12": (
        HGQPhase2K7S3K5S3_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.dense1.build((1, 288))),
        "adam_p2v6_hgq_k7s3k5s3_f12",
    ),
    "k5s2k5s3_f12": (
        HGQPhase2K5S2K5S3_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 63, 4, 12)), m.dense1.build((1, 480))),
        "adam_p2v6_hgq_k5s2k5s3_f12",
    ),
    "light3cv_f8f8f12": (
        HGQPhase2Light3cv,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 8)), m.conv3.build((1, 40, 4, 8)),
                   m.dense1.build((1, 288))),
        "adam_p2v6_hgq_light3cv_f8f8f12",
    ),
    "sepconv2_f12": (
        HGQPhase2SepConv2_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.conv3.build((1, 13, 4, 12)),
                   m.dense1.build((1, 288))),
        "adam_p2v6_hgq_sepconv2_f12",
    ),
    "mixl1_f12": (
        HGQPhase2MixL1_f12,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 12)), m.conv3.build((1, 42, 1, 12)),
                   m.dense1.build((1, 72))),
        "adam_p2v6_hgq_mixl1_f12",
    ),
    "antch_f12": (
        HGQPhase2AntCh_f12,
        lambda m: (m.conv1.build((1, 256, 1, 4)), m.conv2.build((1, 42, 1, 12)), m.dense1.build((1, 72))),
        "adam_p2v6_hgq_antch_f12",
    ),
    # Compute-saver round (2026-11-09)
    "2cv_k5s3_f6": (
        HGQPhase2K5S3_2cv_f6,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 6)), m.dense1.build((1, 144))),
        "adam_p2v6_hgq_2cv_k5s3_f6",
    ),
    "2cv_k5s3_f4": (
        HGQPhase2K5S3_2cv_f4,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 4)), m.dense1.build((1, 96))),
        "adam_p2v6_hgq_2cv_k5s3_f4",
    ),
    "2cv_k5s3_f2": (
        HGQPhase2K5S3_2cv_f2,
        lambda m: (m.conv1.build((1, 256, 4, 1)), m.conv2.build((1, 42, 4, 2)), m.dense1.build((1, 48))),
        "adam_p2v6_hgq_2cv_k5s3_f2",
    ),
    "k7s2_f8": (HGQPhase2K7S2_f8, _flex_build, "adam_p2v6_hgq_k7s2_f8"),
    "k7s2_dw12": (HGQPhase2K7S2_dw12, _flex_build, "adam_p2v6_hgq_k7s2_dw12"),
    "k7s2_f8_dw": (HGQPhase2K7S2_f8_dw, _flex_build, "adam_p2v6_hgq_k7s2_f8_dw"),
    "dec2_k5s1k5s2_f8": (HGQPhase2Dec2_f8, _flex_build, "adam_p2v6_hgq_dec2_k5s1k5s2_f8"),
    "dec2_k5s1k5s2_dw12": (HGQPhase2Dec2_dw12, _flex_build, "adam_p2v6_hgq_dec2_k5s1k5s2_dw12"),
    "dec2_k5s1k5s2_f8_dw": (HGQPhase2Dec2_f8_dw, _flex_build, "adam_p2v6_hgq_dec2_k5s1k5s2_f8_dw"),
    "swing_s1": (HGQPhase2SwingS1, _flex_build, "adam_p2v6_hgq_swing_s1"),
    "swing_s2": (HGQPhase2SwingS2, _flex_build, "adam_p2v6_hgq_swing_s2"),
    "ab_20f": (
        HGQPhase2Ab_20f,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.dense1.build((1, 3360)),
        ),
        "adam_p2v1_hgq_ab_20f",
    ),
    "ab_2cv32_d64": (
        HGQPhase2Ab_2cv32_d64,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.conv2.build((1, 62, 4, 20)),
            m.dense1.build((1, 1792)),
            m.dense2.build((1, 64)),
        ),
        "adam_p2v1_hgq_ab_2cv32_d64",
    ),
    "ab_1cv_nd": (
        HGQPhase2Ab_1cv_nd,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.dense1.build((1, 4960)),
        ),
        "adam_p2v1_hgq_ab_1cv_nd",
    ),
    "ab_1cv_d64": (
        HGQPhase2Ab_1cv_d64,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.dense1.build((1, 4960)),
            m.dense2.build((1, 64)),
        ),
        "adam_p2v1_hgq_ab_1cv_d64",
    ),
    "ab_2cv_nd": (
        HGQPhase2Ab_2cv_nd,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.conv2.build((1, 62, 4, 20)),
            m.dense1.build((1, 1120)),
        ),
        "adam_p2v1_hgq_ab_2cv_nd",
    ),
    "ab_2cv_d64": (
        HGQPhase2Ab_2cv_d64,
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.conv2.build((1, 62, 4, 20)),
            m.dense1.build((1, 1120)),
            m.dense2.build((1, 64)),
        ),
        "adam_p2v1_hgq_ab_2cv_d64",
    ),
    # Run 4 architecture (HGQPhase2ModelV2: conv1 20 k7s2 + conv2 32 k7s2 + Dense(64) + Dense(1), b0=10, i0=5,
    # ~119,525 params, ~687k MACs) so the largest research model can be retrained with this trainer on newer datasets
    # (config name decides the checkpoint folder, e.g. adam_p2v5_hgq_run4). Not FPGA-sized.
    "run4": (
        None,  # filled below (lazy import keeps this module importable without hgq_phase2_models_v2)
        lambda m: (
            m.conv1.build((1, 256, 4, 1)),
            m.conv2.build((1, 62, 4, 20)),
            m.dense1.build((1, 1792)),
            m.dense2.build((1, 64)),
        ),
        "adam_p2v1_hgq_v2_es0",
    ),
}


def _register_run4():
    from hgq_phase2_models_v2 import HGQPhase2ModelV2
    ABLATION_REGISTRY["run4"] = (HGQPhase2ModelV2,) + ABLATION_REGISTRY["run4"][1:]


_register_run4()
