"""Exact AXI word vectors bound to the clean baseline and scalar contract."""

import json
from pathlib import Path

import numpy as np

from ..domain.graph import NumericType
from ..manifest import file_sha256


def write_rtl_vectors(directory: Path, inputs: np.ndarray, expected: np.ndarray,
                      input_type: NumericType, output_type: NumericType, lanes: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    input_slot = max(8, 1 << (input_type.width - 1).bit_length())
    output_slot = max(8, 1 << (output_type.width - 1).bit_length())
    input_codes = np.rint(inputs.reshape(len(inputs), -1) * 2 ** (input_type.width - input_type.integer)).astype(np.int64)
    output_codes = np.rint(expected.reshape(len(inputs), -1) * 2 ** (output_type.width - output_type.integer)).astype(np.int64)
    if output_codes.shape[1] != 1:
        raise ValueError("RTL qualification requires the single-output contract")
    words = []
    for sample in input_codes:
        for offset in range(0, len(sample), lanes):
            word = sum((int(code) & ((1 << input_type.width) - 1)) << (lane * input_slot)
                       for lane, code in enumerate(sample[offset:offset + lanes]))
            words.append(f"{word:0{input_slot * lanes // 4}x}")
    (directory / "rtl_input_words.hex").write_text("\n".join(words) + "\n")
    (directory / "rtl_expected_words.hex").write_text("\n".join(
        f"{int(code) & ((1 << output_type.width) - 1):0{output_slot // 4}x}" for code in output_codes[:, 0]) + "\n")
    record = {"schema_version": 1, "reference": "clean-hls4ml", "corpus": "built_in",
              "sample_count": len(inputs), "input_words_per_inference": len(words) // len(inputs),
              "input_tdata_bits": input_slot * lanes, "output_tdata_bits": output_slot,
              "padding": "zero", "files": {name: file_sha256(directory / name)
                    for name in ("rtl_input_words.hex", "rtl_expected_words.hex")}}
    (directory / "rtl_vectors.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return record
