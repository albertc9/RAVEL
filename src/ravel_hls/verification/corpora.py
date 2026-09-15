"""Deterministic model-derived integer-code corpora with additive caller data."""

from dataclasses import dataclass
import hashlib
from math import prod
from typing import Any, Mapping

import numpy as np

from ..exceptions import ConfigurationError
from .equivalence import prepare_stimuli


@dataclass(frozen=True)
class Corpus:
    name: str
    inputs: np.ndarray
    record: dict[str, Any]


def prepare_corpora(config: Mapping[str, Any], supplied: Any, facts: Mapping[str, Any]) -> tuple[Corpus, ...]:
    """Always generate numeric-contract-v2; normalize each supplied input once."""
    tensor = facts["operations"][0]["outputs"][0]
    numeric = tensor["numeric_type"]
    shape = tuple(tensor["shape"])
    initial, metadata = prepare_stimuli(config, None, tensor)
    scale = 2 ** (numeric["width"] - numeric["integer"])
    codes = [item.copy() for item in np.rint(initial.astype(np.float64) * scale).astype(np.int64)]
    patterns = list(metadata["patterns"])
    limit = (1 << (numeric["width"] - int(numeric["signed"]))) - 1
    lower = -limit if numeric["signed"] and numeric["saturation"] == "SAT_SYM" else (-(limit + 1) if numeric["signed"] else 0)
    rows = {0, shape[0] - 1}
    receptive_field, jump = 1, 1
    for operation in facts["operations"]:
        kind = operation["kind"]
        if kind not in {"conv2d", "max_pool2d"}:
            continue
        attr = operation["attributes"]
        kernel = attr["filt_height" if kind == "conv2d" else "pool_height"]
        receptive_field += (kernel - 1) * jump
        jump *= attr["stride_height"]
        rows.update((receptive_field - 1, receptive_field, jump - 1, jump,
                     (attr["out_height"] - 1) * jump + receptive_field - 1))
    row_size = prod(shape[1:])
    for row in sorted(row for row in rows if 0 <= row < shape[0]):
        for offset in sorted({0, row_size - 1}):
            vector = np.zeros(shape, dtype=np.int64)
            vector.reshape(-1)[row * row_size + offset] = limit
            codes.append(vector)
            patterns.append(f"receptive_boundary_row_{row}_lane_{offset}")
    for value, label in ((1, "positive_lsb"), (-1 if numeric["signed"] else 0, "negative_lsb")):
        codes.append(np.full(shape, value, dtype=np.int64))
        patterns.append(label)
    ordered = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    codes.append(ordered % (limit - lower + 1) + lower)
    patterns.append("feature_channel_order")
    builtin_codes = np.stack(codes)
    builtin = builtin_codes.astype(np.float64) / scale
    builtin_record = _record(builtin, builtin_codes, numeric, "numeric_contract")
    builtin_record.update(recipe={"id": "numeric-contract", "version": 2},
                          seed=config["Verification"].get("Seed", 0), patterns=patterns)
    result = [Corpus("built_in", builtin, builtin_record)]
    if supplied is not None:
        inputs, supplied_record = prepare_stimuli(config, supplied, tensor)
        if not np.all(np.isfinite(inputs)):
            raise ConfigurationError("verification_inputs must contain finite numeric values")
        rounded = np.floor(inputs.astype(np.float64) * scale + 0.5)
        if numeric["saturation"] in {"SAT", "SAT_SYM"}:
            rounded = np.clip(rounded, lower, limit)
        elif numeric["saturation"] == "WRAP":
            rounded = (rounded - lower) % (limit - lower + 1) + lower
        else:
            raise ConfigurationError("Unsupported input quantization contract for verification")
        supplied_codes = rounded.astype(np.int64)
        normalized = supplied_codes.astype(np.float64) / scale
        record = _record(normalized, supplied_codes, numeric, "supplied")
        record["supplied_content_sha256"] = supplied_record["content_sha256"]
        result.append(Corpus("supplied", normalized, record))
    return tuple(result)


def _record(inputs, codes, numeric, kind):
    return {"kind": kind, "shape": list(inputs.shape), "dtype": str(inputs.dtype),
            "sample_count": len(inputs), "input_numeric_type": dict(numeric),
            "content_sha256": hashlib.sha256(inputs.tobytes(order="C")).hexdigest(),
            "integer_code_sha256": hashlib.sha256(codes.astype("<i8").tobytes(order="C")).hexdigest()}
