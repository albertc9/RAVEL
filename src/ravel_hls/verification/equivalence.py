"""Behavioral verification of a staged Aria transformation."""

import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
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
        codes = _numeric_contract_codes(
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
    return inputs, record


def _numeric_contract_codes(
    sample_count: int,
    shape: tuple[int, ...],
    numeric_type: dict[str, Any],
    seed: int,
) -> np.ndarray:
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
    patterns = [0, minimum, maximum]
    for index, value in enumerate(patterns[:sample_count]):
        codes[index].fill(value)
    if sample_count > 3:
        flat = codes[3].reshape(-1)
        flat[0::2] = minimum
        flat[1::2] = maximum
    if sample_count > 4:
        codes[4].fill(0)
        codes[4].reshape(-1)[0] = 1
    if sample_count > 5 and signed:
        codes[5].fill(0)
        codes[5].reshape(-1)[0] = -1
    return codes


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
    """Compile and predict through hls4ml's existing-project boundary."""

    from hls4ml.utils.link import FilesystemModelGraph

    try:
        linked = FilesystemModelGraph(project_path)
        with _compiler_environment(compiler):
            linked.compile()
    except Exception as error:
        raise VerificationError(
            f"RAVEL optimized project compilation failed: {error}"
        ) from error
    try:
        return np.asarray(linked.predict(inputs))
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
