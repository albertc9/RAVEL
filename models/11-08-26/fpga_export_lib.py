"""fpga_export_lib.py -- Keras graphs and architecture logs for every v6 CNN architecture (FPGA package, 2026-11-08).

Used by ExportFPGA_v6_all.py (HPC3, trained weights) and testable locally on freshly built models (no data needed).

keras_versions(model, arch) -> {version: (keras.Model, note)}
  Builds a keras functional model from the SAME trained layer objects the torch wrapper uses, mirroring its forward()
  exactly (checked by the caller: reloaded .keras scores == torch scores). Version "main" is the firmware-matching one;
  some architectures get alternative versions with identical math, in case a converter route is unavailable:
    dec2_*      main: input (128, 4, 1) = already low-passed (if any) and decimated outside the CNN, as firmware would
                alt "decimate_in_model": input (256, 4, 1), first layer = plain 1x1 conv, step 2, weight 1 (keeps
                every 2nd sample, exactly x[..., ::2])
    antch_f12   main: input (256, 4, 1) + Reshape to (256, 1, 4) (pure relabelling: antennas become channels)
                alt "channels_input": input (256, 1, 4) directly
    sepconv2    main: grouped conv (groups = 12) + 1x1 conv
                alt "dense_equivalent": the grouped conv rewritten as an ordinary conv with a block-diagonal kernel
                (off-diagonal weights exactly 0; same quantizers), for converters without grouped conv
architecture_card(km) -> (text, dict): per layer type, settings, shapes, weights+biases, MACs, learned bit widths.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np  # noqa: E402
import keras  # noqa: E402

# Same compatibility patches as ExportFPGA_v6_ampchain.py: keras.save() lazily imports tensorflow, and the old TF in
# ~/.local on HPC3 crashes on import with numpy 2 (np.complex_); treat TF as unavailable instead. HGQ base_conv rename.
from keras.src.utils import module_utils as _mu  # noqa: E402
_orig_lazy_init = _mu.LazyModule.initialize


def _safe_lazy_init(self):
    try:
        _orig_lazy_init(self)
    except Exception:
        self.module = None


def _safe_lazy_getattr(self, name):
    if self.module is None:
        self.initialize()
    if self.module is None:
        class _Unavailable:
            pass
        return _Unavailable
    return getattr(self.module, name)


_mu.LazyModule.initialize = _safe_lazy_init
_mu.LazyModule.__getattr__ = _safe_lazy_getattr
import keras.src.layers.convolutional.base_conv as _base_conv  # noqa: E402
if not hasattr(_base_conv, "Conv") and hasattr(_base_conv, "BaseConv"):
    _base_conv.Conv = _base_conv.BaseConv

TWO_CONV = {"2cv_k5s3_f12", "2cv_k5s3_f8", "2cv_k7s2_f12", "k5s3k3s1_f12", "k7s3k5s3_f12", "k5s2k5s3_f12"}
HIDDEN_DENSE = {"2cv_k5s3_f12_d16", "run4"}
THREE_CONV = {"light3cv_f8f8f12", "sepconv2_f12", "mixl1_f12"}
DEC2 = {"dec2_k5s1k5s2_f12", "dec2_2cv_k5s3_f12"}
ANTCH = {"antch_f12"}
ALL_ARCHS = TWO_CONV | HIDDEN_DENSE | THREE_CONV | DEC2 | ANTCH

# expected weights+biases and MACs per 256-ns window (hgq_phase2_models_ablation.py docstrings / v3_models.MODEL_CATALOG;
# the catalog's run4 count 119,525 also includes its 36 quantizer scalars)
EXPECTED = {
    "2cv_k5s3_f12": (1_093, 57_888), "2cv_k5s3_f8": (569, 30_272), "2cv_k7s2_f12": (1_789, 155_568),
    "k5s3k3s1_f12": (1_477, 90_240), "k7s3k5s3_f12": (1_117, 65_952), "k5s2k5s3_f12": (1_285, 88_320),
    "2cv_k5s3_f12_d16": (5_445, 62_224), "run4": (119_489, 686_512), "light3cv_f8f8f12": (837, 59_424),
    "sepconv2_f12": (589, 31_056), "mixl1_f12": (1_465, 53_784), "dec2_k5s1k5s2_f12": (1_477, 113_952),
    "dec2_2cv_k5s3_f12": (949, 27_504), "antch_f12": (1_057, 29_592),
}


ONE_HZ = 256e-9     # background fraction per 256-ns window at 1 Hz


def thr_1hz(tasks):
    """Score at which the trigger fires at exactly 1 Hz on noise (fires at score > threshold); same as
    MakeTalkPlots_v4.thr_1hz for CNNs (kept here so the HPC export needs no plotting modules)."""
    total = sum(int(x["n_windows"]) for x in tasks)

    def frac(t):
        n = 0.0
        for x in tasks:
            top, hist, edges = x["top"], x["hist"], x["hist_edges"]
            tail = np.concatenate([np.cumsum(hist[::-1])[::-1], [0]])
            from_top = len(top) - np.searchsorted(np.sort(top), t, side="right")
            n += from_top if t >= top.min() else max(float(np.interp(t, edges, tail)), from_top)
        return n / total
    lo, hi = -20.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if frac(mid) > ONE_HZ else (lo, mid)
    return 0.5 * (lo + hi)


def _two_conv(m, x):
    x = m.pool2(m.conv2(m.pool1(m.conv1(x))))
    return m.dense1(keras.layers.Flatten(name="flatten")(x))


def _hidden_dense(m, x):
    x = m.pool2(m.conv2(m.pool1(m.conv1(x))))
    x = keras.layers.ReLU(name="relu_hidden")(m.dense1(keras.layers.Flatten(name="flatten")(x)))
    return m.dense2(x)


def _three_conv(m, x, conv2=None):
    x = m.pool1(m.conv1(x))
    x = m.pool3(m.conv3((conv2 or m.conv2)(x)))
    return m.dense1(keras.layers.Flatten(name="flatten")(x))


def _decimator():
    """Plain 1x1 conv, step 2 along time, weight 1, no bias: output[t] = input[2t] (= x[..., ::2])."""
    d = keras.layers.Conv2D(1, (1, 1), strides=(2, 1), use_bias=False, trainable=False, name="decimate_keep_every_2nd")
    d.build((None, 256, 4, 1))
    d.kernel.assign(np.ones((1, 1, 1, 1), np.float32))
    return d


def _dense_equivalent(conv):
    """Ordinary (groups=1) QConv2D that computes exactly what the grouped (depthwise) `conv` computes."""
    from hgq.layers import QConv2D
    cfg = conv.get_config()
    groups, filters = cfg["groups"], cfg["filters"]
    if groups != filters:
        raise ValueError("dense equivalent written for depthwise convs (groups == filters)")
    cfg.update(groups=1, name=conv.name + "_dense_equivalent")
    new = QConv2D.from_config(cfg)
    kh, kw, _, _ = conv.kernel.shape
    new.build((None, 42, 4, filters))
    k = np.asarray(keras.ops.convert_to_numpy(conv._kernel))                     # (kh, kw, 1, F)
    big = np.zeros((kh, kw, filters, filters), np.float32)
    for c in range(filters):
        big[:, :, c, c] = k[:, :, 0, c]
    new._kernel.assign(big)
    src = {_role(w.path): w for w in conv.weights if _role(w.path)}
    for w in new.weights:
        r = _role(w.path)
        if r and r in src and r != "kernel":
            w.assign(src[r])
    return new


def _role(path):
    """'.../q_conv2d_iq/fixed_point_quantizer_kif/f' -> 'iq.f'; '.../kernel' -> 'kernel'; '.../bias' -> 'bias'."""
    parts = path.split("/")
    if parts[-1] in ("kernel", "bias", "beta", "ebops"):
        return parts[-1]
    for p in parts:
        for q in ("_iq", "_kq", "_bq", "_oq"):
            if p.endswith(q) and parts[-1] in ("k", "i", "f"):
                return f"{q[1:]}.{parts[-1]}"
    return None


def keras_versions(m, arch, label):
    """{version: (keras.Model, input description)}; 'main' always present."""
    out = {}
    if arch in DEC2:
        inp = keras.Input((128, 4, 1), name="waveform_0p5GSa")
        out["main"] = (keras.Model(inp, _two_conv(m, inp), name=label),
                       "(128, 4, 1): 0.5 GSa/s, every 2nd sample of the 1 GSa/s trace, taken outside the CNN")
        inp = keras.Input((256, 4, 1), name="waveform")
        out["decimate_in_model"] = (keras.Model(inp, _two_conv(m, _decimator()(inp)), name=label + "_decimate_in_model"),
                                    "(256, 4, 1): 1 GSa/s; the first layer keeps every 2nd sample")
        return out
    if arch in ANTCH:
        inp = keras.Input((256, 4, 1), name="waveform")
        x = keras.layers.Reshape((256, 1, 4), name="antennas_to_channels")(inp)
        out["main"] = (keras.Model(inp, _two_conv(m, x), name=label),
                       "(256, 4, 1), reshaped inside the model to (256, 1, 4): antennas become input channels")
        inp = keras.Input((256, 1, 4), name="waveform_antennas_as_channels")
        out["channels_input"] = (keras.Model(inp, _two_conv(m, inp), name=label + "_channels_input"),
                                 "(256, 1, 4): time, 1, antenna (same numbers as (256, 4, 1), relabelled)")
        return out
    inp = keras.Input((256, 4, 1), name="waveform")
    if arch in TWO_CONV:
        y = _two_conv(m, inp)
    elif arch in HIDDEN_DENSE:
        y = _hidden_dense(m, inp)
    elif arch in THREE_CONV:
        y = _three_conv(m, inp)
    else:
        raise KeyError(arch)
    out["main"] = (keras.Model(inp, y, name=label), "(256, 4, 1): 1 GSa/s, time x antenna x 1")
    if arch == "sepconv2_f12":
        inp = keras.Input((256, 4, 1), name="waveform")
        out["dense_equivalent"] = (keras.Model(inp, _three_conv(m, inp, _dense_equivalent(m.conv2)),
                                               name=label + "_dense_equivalent"),
                                   "(256, 4, 1); 2nd layer as an ordinary conv with a block-diagonal kernel")
    return out


def to_keras_input(x, version, arch):
    """x: (N, 4, 256) float32 as the torch model takes it -> the array this keras version takes."""
    if arch in DEC2 and version == "main":
        x = x[..., ::2]
    xk = x.transpose(0, 2, 1)[:, :, :, None]                                 # (N, T, 4, 1)
    if arch in ANTCH and version == "channels_input":
        xk = xk.reshape(len(xk), xk.shape[1], 1, 4)
    return np.ascontiguousarray(xk, dtype=np.float32)


def keras_scores(km, xk):
    return np.asarray(keras.ops.convert_to_numpy(km(xk, training=False))).reshape(-1)


def _shape(t):
    return tuple(None if d is None else int(d) for d in t.shape)


def layer_names(m):
    """{id(keras layer): attribute name in the torch wrapper} so the logs say conv1 / pool1 / dense1, not q_conv2d_57."""
    return {id(getattr(m, a)): a for a in ("conv1", "conv2", "conv3", "pool1", "pool2", "pool3", "dense1", "dense2")
            if hasattr(m, a)}


def architecture_card(km, title="", names=None):
    """Layer table + learned bit widths, read from the keras model itself."""
    names = names or {}
    rows, total_w, total_mac, quant = [], 0, 0, []
    for L in km.layers:
        if isinstance(L, keras.layers.InputLayer):
            continue
        cfg = L.get_config()
        ish, osh = _shape(L.input), _shape(L.output)
        nm = names.get(id(L), L.name)
        if L.name.endswith("_dense_equivalent"):
            nm = "conv2_dense_equivalent"
        d = dict(name=nm, keras_name=L.name, type=type(L).__name__, input=ish[1:], output=osh[1:])
        for k in ("filters", "kernel_size", "strides", "padding", "groups", "activation", "use_bias", "pool_size",
                  "units", "target_shape"):
            if k in cfg:
                v = cfg[k]
                d[k] = list(v) if isinstance(v, tuple) else v
        if isinstance(L, keras.layers.MaxPooling2D):
            d["strides"] = list(cfg["strides"])
        nw = sum(int(np.prod(w.shape)) for w in L.weights if w.path.split("/")[-1] in ("kernel", "bias"))
        mac = 0
        if "kernel_size" in cfg:
            cin = ish[-1] // cfg.get("groups", 1)
            mac = int(np.prod(osh[1:-1])) * cfg["filters"] * int(np.prod(cfg["kernel_size"])) * cin
        elif "units" in cfg:
            mac = ish[-1] * cfg["units"]
        d["weights_and_biases"], d["macs"] = nw, mac
        if L.name.startswith("decimate"):
            d["note"] = "fixed selection, not a learned layer (weight 1): keeps every 2nd sample"
            d["macs"] = 0
        bw = {}
        for w in L.weights:
            r = _role(w.path)
            if r and "." in r:
                bw.setdefault(r.split(".")[0], {})[r.split(".")[1]] = float(np.asarray(keras.ops.convert_to_numpy(w)).reshape(-1)[0])
        if bw:
            d["bit_widths"] = {}
            for q in ("iq", "kq", "bq", "oq"):
                if q in bw:
                    v = bw[q]
                    k, i, f = round(v.get("k", 0)), round(v.get("i", 0)), round(v.get("f", 0))
                    d["bit_widths"][q] = dict(v, total_rounded=int(k + i + f), ap_fixed=f"ap_fixed<{k + i + f},{k + i}>")
            quant.append((nm, d["bit_widths"]))
        total_w += nw
        total_mac += d["macs"]
        rows.append(d)
    heavy = max(rows, key=lambda r: r["macs"])
    info = dict(model=km.name, input=_shape(km.inputs[0])[1:], output=_shape(km.outputs[0])[1:], layers=rows,
                total_weights_and_biases=total_w, total_macs=total_mac,
                heaviest_layer=dict(name=heavy["name"], macs=heavy["macs"]))

    def fmt(v):
        return "-" if v is None else (",".join(str(x) for x in v) if isinstance(v, list) else str(v))
    lines = [title or km.name, "=" * len(title or km.name),
             f"input {info['input']} -> output {info['output']} (raw score; trigger when score > threshold)", "",
             f"{'layer':30s} {'type':16s} {'filters/units':>13s} {'kernel':>7s} {'step':>6s} {'groups':>6s} {'act':>7s} "
             f"{'input shape':>15s} {'output shape':>15s} {'w+b':>8s} {'MACs':>9s}"]
    for d in rows:
        lines.append(f"{d['name']:30s} {d['type']:16s} {fmt(d.get('filters', d.get('units'))):>13s} "
                     f"{fmt(d.get('kernel_size', d.get('pool_size'))):>7s} {fmt(d.get('strides')):>6s} "
                     f"{fmt(d.get('groups')):>6s} {fmt(d.get('activation')):>7s} {str(d['input']):>15s} "
                     f"{str(d['output']):>15s} {d['weights_and_biases']:8,d} {d['macs']:9,d}"
                     + (f"   ({d['note']})" if "note" in d else ""))
    lines += ["", f"total weights+biases {total_w:,}   total MACs per window {total_mac:,}   "
              f"heaviest layer {heavy['name']} ({heavy['macs']:,} MACs)", "",
              "Learned bit widths (HGQ kif fixed point, one width per tensor; iq = layer input, kq = kernel, bq = bias;",
              "k = sign bit, i = integer bits, f = fractional bits; values as stored in the .keras file;",
              "width = k + round(i) + round(f), i.e. ap_fixed<k+i+f, k+i>; the .keras file is authoritative):"]
    for name, bw in quant:
        for q, v in bw.items():
            lines.append(f"  {name:30s} {q}: k {v.get('k', 0):.0f}  i {v.get('i', 0):7.3f}  f {v.get('f', 0):7.3f}"
                         f"   -> {v['total_rounded']:2d} bits  {v['ap_fixed']}")
    return "\n".join(lines) + "\n", info
