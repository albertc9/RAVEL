"""Test-only integer-bit observations in disposable copies of generated C++."""

import hashlib
import json
from math import prod
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..exceptions import VerificationError
from .equivalence import predict_optimized


_HEADER = r'''
#include "hls_stream.h"
#include <cstdint>
#include <fstream>
#include <vector>
namespace ravel_observation {
template<class T> void snapshot(hls::stream<T>& stream, unsigned count, const char* path) {
    std::vector<T> words;
    std::ofstream file(path, std::ios::binary | std::ios::app);
    for (unsigned offset = 0; offset < count; offset += T::size) {
        T word = stream.read();
        words.push_back(word);
        for (unsigned lane = 0; lane < T::size && offset + lane < count; ++lane) {
            typedef typename T::value_type scalar;
            static_assert(scalar::width <= 64, "Observation supports at most 64-bit scalar codes");
            std::uint64_t bits = word[lane].range(scalar::width - 1, 0).to_uint64();
            file.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
        }
    }
    for (const auto& word : words) stream.write(word);
}
}
'''


def capture_boundaries(path, name, design, facts, inputs, compiler, *, baseline):
    """Compile observations only in a temporary project; never alter production."""
    with tempfile.TemporaryDirectory(prefix="ravel-boundaries-") as work:
        work = Path(work)
        copied = work / "project"
        shutil.copytree(path, copied, ignore=shutil.ignore_patterns("*_prj", "*.so", "*.log"))
        source = copied / "firmware" / f"{name}.cpp"
        body = source.read_text()
        sites = []
        tensors = {tensor["id"]: tensor for op in facts["operations"] for tensor in op["outputs"]}
        if baseline:
            operations = {op["id"]: op for op in facts["operations"]}
            native = design["rendering"]["native_operations"]
            for operation in facts["operations"]:
                if operation["kind"] in {"input", "repack"}:
                    continue
                anchor = operation
                while not native[anchor["id"]]["native_call"]:
                    if len(anchor["inputs"]) != 1:
                        raise VerificationError("Boundary has no unique native producer")
                    anchor = operations[anchor["inputs"][0].split(":")[0]]
                binding = native[anchor["id"]]
                call = binding["native_call"]
                if body.count(call) != 1:
                    raise VerificationError(f"Cannot locate clean boundary {operation['id']}")
                tensor = operation["outputs"][0]
                index = len(sites)
                sites.append((tensor["id"], prod(tensor["shape"]), work / f"{index}.bin"))
                statement = _statement(binding["output_symbol"], sites[-1])
                body = body.replace(call, call + "\n" + statement, 1)
        else:
            def instrument(match):
                tensor, symbol, count = match.groups()
                site = (tensor, int(count), work / f"{len(sites)}.bin")
                sites.append(site)
                return _statement(symbol, site)
            body = re.sub(r"// RAVEL_OBSERVE (\S+) (\w+) (\d+)", instrument, body)
        if not sites:
            raise VerificationError("No owned stage boundaries were instrumented")
        first_include = body.find("\n", body.index("#include"))
        body = body[:first_include + 1] + _HEADER + body[first_include + 1:]
        source.write_text(body)
        predict_optimized(copied, inputs, compiler)
        results = []
        for tensor, count, file in sites:
            codes = np.fromfile(file, dtype="<u8") if file.exists() else np.array([], dtype="<u8")
            if codes.size != len(inputs) * count:
                raise VerificationError(f"Boundary {tensor} produced {codes.size} codes; expected {len(inputs) * count}")
            results.append((tensor, codes.reshape(len(inputs), count)))
        return results


def _statement(symbol, site):
    return f"    ravel_observation::snapshot({symbol}, {site[1]}, {json.dumps(str(site[2]))});"


def compare_boundaries(baseline, optimized):
    reference = dict(baseline)
    observations = []
    for tensor, observed in optimized:
        expected = reference.get(tensor)
        if expected is None or not np.array_equal(expected, observed):
            raise VerificationError(f"Stage boundary integer-code mismatch at {tensor}")
        observations.append({"tensor_id": tensor, "sample_count": len(observed),
                             "codes_per_sample": observed.shape[1], "status": "passed",
                             "integer_bits_sha256": hashlib.sha256(observed.astype("<u8").tobytes()).hexdigest()})
    return {"status": "passed", "corpus": "built_in", "encoding": "unsigned-scalar-bits-le64", "observations": observations}
