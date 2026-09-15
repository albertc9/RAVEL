"""Behavioral verification of a staged Aria transformation."""

import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from ..config import RavelConfig
from ..exceptions import ConfigurationError, VerificationError


def prepare_stimuli(
    config: RavelConfig,
    verification_inputs: Any | None,
    input_tensor: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize supplied data or create deterministic synthetic inputs."""

    verification = config["Verification"]
    if input_tensor is not None:
        input_numeric_type = input_tensor["numeric_type"]
        input_shape = tuple(input_tensor["shape"])
    else:
        input_numeric_type = None
        input_shape = (256, 4)
    if verification_inputs is None and input_numeric_type is not None:
        sample_count = verification.get("Samples", 32)
        seed = verification.get("Seed", 0)
        codes, patterns = _numeric_contract_codes(
            sample_count, input_shape, input_numeric_type, seed
        )
        fractional = input_numeric_type["width"] - input_numeric_type["integer"]
        inputs = (codes.astype(np.float64) / (2**fractional)).astype(np.float32)
        kind = "numeric_contract"
    elif verification_inputs is None:
        sample_count = verification.get("Samples", 32)
        seed = verification.get("Seed", 0)
        inputs = np.random.default_rng(seed).uniform(
            -1.0, 1.0, size=(sample_count, *input_shape)
        ).astype(np.float32)
        kind = "synthetic"
    else:
        inputs = np.asarray(verification_inputs)
        if (
            inputs.ndim == len(input_shape) + 2
            and tuple(inputs.shape[1:-1]) == input_shape
            and inputs.shape[-1] == 1
        ):
            inputs = inputs[..., 0]
        seed = None
        kind = "supplied"
    if (
        inputs.ndim != len(input_shape) + 1
        or tuple(inputs.shape[1:]) != input_shape
        or inputs.shape[0] < 1
    ):
        raise ConfigurationError(
            "verification_inputs must have shape "
            f"[samples, {', '.join(str(value) for value in input_shape)}]"
        )
    inputs = np.ascontiguousarray(inputs)
    record = {
        "kind": kind,
        "shape": list(inputs.shape),
        "dtype": str(inputs.dtype),
        "sample_count": int(inputs.shape[0]),
        "seed": seed,
        "content_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
    }
    if input_numeric_type is not None:
        record["input_numeric_type"] = dict(input_numeric_type)
    if verification_inputs is None and input_numeric_type is not None:
        record["integer_code_sha256"] = hashlib.sha256(
            codes.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest()
        record["patterns"] = patterns
    return inputs, record


def _numeric_contract_codes(
    sample_count: int,
    shape: tuple[int, ...],
    numeric_type: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    width = numeric_type["width"]
    signed = numeric_type["signed"]
    if signed:
        maximum = (1 << (width - 1)) - 1
        minimum = (
            -maximum
            if numeric_type["saturation"] == "SAT_SYM"
            else -(1 << (width - 1))
        )
    else:
        minimum = 0
        maximum = (1 << width) - 1
    codes = np.random.default_rng(seed).integers(
        minimum, maximum + 1, size=(sample_count, *shape), dtype=np.int64
    )
    pattern_names = ["seeded_random"] * sample_count
    fixed_patterns = [0, minimum, maximum]
    fixed_pattern_names = ["zeros", "minimum", "maximum"]
    for index, value in enumerate(fixed_patterns[:sample_count]):
        codes[index].fill(value)
        pattern_names[index] = fixed_pattern_names[index]
    if sample_count > 3:
        flat = codes[3].reshape(-1)
        flat[0::2] = minimum
        flat[1::2] = maximum
        pattern_names[3] = "alternating_extrema"
    probe_rows = _spatial_probe_rows(shape[0])
    row_size = int(np.prod(shape[1:], dtype=np.int64))
    probe_stop = min(sample_count, 12)
    for sample_index in range(4, probe_stop):
        probe_index = sample_index - 4
        row = probe_rows[probe_index % len(probe_rows)]
        within_row = (probe_index // len(probe_rows)) % row_size
        value = -1 if signed and probe_index % 2 else 1
        codes[sample_index].fill(0)
        codes[sample_index].reshape(-1)[row * row_size + within_row] = value
        polarity = "negative" if value < 0 else "positive"
        suffix = f"_offset_{within_row}" if within_row else ""
        pattern_names[sample_index] = f"{polarity}_impulse_row_{row}{suffix}"
    return codes, pattern_names


def _spatial_probe_rows(input_rows: int) -> tuple[int, ...]:
    anchors = (0, 1, 2, input_rows - 3, input_rows - 2, input_rows - 1)
    ordered = (*anchors, *range(3, max(3, input_rows - 3)))
    return tuple(dict.fromkeys(row for row in ordered if 0 <= row < input_rows))


def predict_baseline(
    hls_model: Any, inputs: np.ndarray, compiler: str | None = None
) -> np.ndarray:
    """Compile and predict with the clean hls4ml baseline boundary."""

    if not callable(getattr(hls_model, "compile", None)) or not callable(
        getattr(hls_model, "predict", None)
    ):
        raise VerificationError(
            "Required hls4ml compile/predict capability is unavailable"
        )
    try:
        with _compiler_environment(compiler):
            hls_model.compile()
    except Exception as error:
        raise VerificationError(
            f"hls4ml baseline compilation failed: {error}"
        ) from error
    try:
        return np.asarray(hls_model.predict(inputs))
    except Exception as error:
        raise VerificationError(f"hls4ml baseline prediction failed: {error}") from error


def predict_optimized(
    project_path: Path, inputs: np.ndarray, compiler: str | None = None
) -> np.ndarray:
    """Compile and predict the rendered project in a fresh process."""

    with tempfile.TemporaryDirectory(prefix="ravel-predict-") as directory:
        interchange_path = Path(directory)
        input_path = interchange_path / "inputs.npy"
        output_path = interchange_path / "outputs.npy"
        np.save(input_path, inputs, allow_pickle=False)
        command = [
            sys.executable,
            "-m",
            "ravel_hls.verification._predict_project",
            str(project_path.resolve()),
            str(input_path),
            str(output_path),
        ]
        with _compiler_environment(compiler):
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            stage = {10: "compilation", 11: "prediction"}.get(
                result.returncode, "worker"
            )
            raise VerificationError(
                f"RAVEL optimized project {stage} failed: {detail}"
            )
        try:
            return np.load(output_path, allow_pickle=False)
        except Exception as error:
            raise VerificationError(
                f"RAVEL optimized project prediction failed: {error}"
            ) from error


def require_bit_exact(baseline: np.ndarray, optimized: np.ndarray) -> None:
    """Require the public Aria transformation-equivalence contract."""

    if baseline.shape != optimized.shape or not np.array_equal(baseline, optimized):
        maximum_difference = (
            float(np.max(np.abs(baseline - optimized)))
            if baseline.shape == optimized.shape and baseline.size
            else None
        )
        raise VerificationError(
            "Aria transformation equivalence failed: "
            f"baseline shape {baseline.shape}, optimized shape {optimized.shape}, "
            f"max abs diff {maximum_difference}"
        )


def require_source_consistency(
    source_model: Any,
    inputs: np.ndarray,
    baseline_predictions: np.ndarray,
    output_numeric_type: dict[str, Any],
) -> dict[str, Any]:
    """Require Keras/HGQ and clean hls4ml to agree as output integer codes."""

    try:
        source_predictions = source_model.predict(inputs, verbose=0)
    except TypeError:
        source_predictions = source_model.predict(inputs)
    source_predictions = np.asarray(source_predictions)
    if source_predictions.shape != baseline_predictions.shape:
        raise VerificationError(
            "Source-conversion consistency failed: "
            f"source shape {source_predictions.shape}, baseline shape {baseline_predictions.shape}"
        )
    fractional = output_numeric_type["width"] - output_numeric_type["integer"]
    scale = 2**fractional
    source_codes = np.rint(source_predictions * scale).astype(np.int64)
    baseline_codes = np.rint(baseline_predictions * scale).astype(np.int64)
    if not np.array_equal(source_codes, baseline_codes):
        maximum_code_difference = int(
            np.max(np.abs(source_codes - baseline_codes))
        )
        raise VerificationError(
            "Source-conversion consistency failed: "
            f"max integer-code difference {maximum_code_difference}"
        )
    return {
        "status": "passed",
        "comparison": "canonical_fixed_point_integer_codes",
        "max_abs_float_difference": float(
            np.max(np.abs(source_predictions - baseline_predictions))
        ),
    }


@contextmanager
def _compiler_environment(compiler: str | None):
    if compiler is None:
        yield
        return
    previous_path = os.environ.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="ravel-cxx-") as directory:
        shim = Path(directory) / "g++"
        shim.symlink_to(compiler)
        os.environ["PATH"] = directory + os.pathsep + previous_path
        try:
            yield
        finally:
            os.environ["PATH"] = previous_path


def report_model_fidelity(
    source_model: Any, inputs: np.ndarray, hls_predictions: np.ndarray
) -> dict[str, Any] | None:
    """Return the informational Keras/HGQ versus HLS score difference."""

    if not callable(getattr(source_model, "predict", None)):
        return None
    try:
        source_predictions = source_model.predict(inputs, verbose=0)
    except TypeError:
        source_predictions = source_model.predict(inputs)
    source_predictions = np.asarray(source_predictions)
    if source_predictions.shape != hls_predictions.shape:
        return {
            "status": "reported",
            "shape_match": False,
            "max_abs_score_diff": None,
        }
    return {
        "status": "reported",
        "shape_match": True,
        "max_abs_score_diff": float(
            np.max(np.abs(source_predictions - hls_predictions))
        ),
    }
