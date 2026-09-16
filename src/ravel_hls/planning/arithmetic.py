"""Proven constant-matrix capabilities shared by both convolution stages."""

from dataclasses import dataclass

import numpy as np

from ..analysis.phara import AffineGraph, AffineProof, analyze_constant_matrix
from ..domain.graph import NumericType


@dataclass(frozen=True)
class ConstantArithmetic:
    """An immutable, proved implementation of one modular coefficient matrix."""

    graph: AffineGraph
    proof: AffineProof
    summary: tuple[tuple[str, int], ...]
    operation_id: str
    input_numeric: NumericType
    accumulator_numeric: NumericType
    paired: bool
    dsp_product_budget: int

    def count(self, name: str) -> int:
        return next((value for key, value in self.summary if key == name), 0)

    def to_dict(self):
        return {"kind": "hybrid", "policy": {"id": "constant-matrix-csd-cse-dsp", "version": 1},
                "graph": {"input_ids": list(self.graph.input_ids), "output_ids": list(self.graph.output_ids),
                          "modulus": self.graph.modulus,
                          "nodes": [{"id": node.id, "operation": node.operation,
                                     "inputs": list(node.inputs), "value": node.value} for node in self.graph.nodes]},
                "graph_sha256": self.proof.graph_sha256,
                "proof": {"status": self.proof.status, "identity": self.proof.identity, "modulus": self.proof.modulus},
                "graph_summary": dict(self.summary), "dsp_product_budget": self.dsp_product_budget,
                "operation_id": self.operation_id, "input_numeric": self.input_numeric.to_dict(),
                "accumulator_numeric": self.accumulator_numeric.to_dict(), "paired": self.paired}


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
        if realization.proof.status == "proven" and realization.proof.graph_sha256 not in seen:
            choices.append(ConstantArithmetic(realization.graph, realization.proof,
                                              tuple(sorted(realization.summary.items())), conv.id, source,
                                              NumericType(**accumulator), paired, budget))
            seen.add(realization.proof.graph_sha256)
    return tuple(choices)
