"""Proven constant-matrix capabilities shared by both convolution stages."""

import numpy as np

from ..analysis.phara import analyze_constant_matrix


def constant_arithmetic(block, parameters, native, *, paired=False, dsp_budgets=(0, 16)):
    """Admit only linear product casts followed by modular accumulation.

    Flatten kernel/channel coordinates in native C order. Exact products may
    wrap at any accumulation step because addition is associative modulo 2**W.
    Fractional truncation and saturating accumulators are deliberately excluded.
    """
    if parameters is None or not native:
        return ()
    conv = block.convolution
    accumulator = native[conv.id].get("accumulator_numeric")
    if not accumulator or accumulator["kind"] != "fixed":
        return ()
    if accumulator["saturation"] not in (None, "WRAP") or accumulator["saturation_bits"]:
        return ()
    if paired and accumulator != conv.outputs[0].numeric_type.to_dict():
        return ()
    payload = parameters.by_id()
    weight, bias = payload[f"{conv.id}:weight"], payload[f"{conv.id}:bias"]
    source = block.input.numeric_type
    if source.kind != "fixed" or weight.numeric_type["kind"] != "fixed":
        return ()
    if paired and not source.signed:
        return ()
    acc_fraction = accumulator["width"] - accumulator["integer"]
    weight_fraction = weight.numeric_type["width"] - weight.numeric_type["integer"]
    product_fraction = source.width - source.integer + weight_fraction
    shift = acc_fraction - product_fraction
    if shift < 0 or accumulator["width"] > 32:
        return ()
    bias_scaled = np.asarray(bias.values) * (2.0 ** acc_fraction)
    if not np.all(bias_scaled == np.floor(bias_scaled)):
        return ()
    matrix = np.rint(np.asarray(weight.values) * (2.0 ** weight_fraction)).astype(np.int64)
    matrix = matrix.reshape(-1, conv.attribute("n_filt")) * (1 << shift)
    # Bound graph construction independently of the search's candidate count.
    if matrix.size > 2048:
        return ()
    arguments = dict(weight_codes=tuple(tuple(int(v) for v in row) for row in matrix),
                     aligned_bias_codes=tuple(int(v) for v in bias_scaled.flat),
                     modulus=1 << accumulator["width"],
                     row_offset=conv.attribute("stride_height") * conv.attribute("n_chan") if paired else 0,
                     output_rows=2 if paired else 1)
    choices = []
    seen = set()
    for budget in dsp_budgets:
        realization = analyze_constant_matrix(**arguments, dsp_product_budget=budget)
        if realization["graph_sha256"] not in seen:
            choices.append({**realization, "operation_id": conv.id,
                            "input_numeric": source.to_dict(),
                            "accumulator_numeric": accumulator, "paired": paired})
            seen.add(realization["graph_sha256"])
    return tuple(choices)
