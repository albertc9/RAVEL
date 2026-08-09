"""Behavioral verification of a staged Aria transformation."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RavelConfig
from ..exceptions import ConfigurationError, VerificationError


def prepare_stimuli(
    config: RavelConfig, verification_inputs: Any | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize supplied data or create deterministic synthetic inputs."""

    verification = config["Verification"]
    if verification_inputs is None:
        sample_count = verification.get("Samples", 32)
        seed = verification.get("Seed", 0)
        inputs = np.random.default_rng(seed).uniform(
            -1.0, 1.0, size=(sample_count, 256, 4)
        ).astype(np.float32)
        kind = "synthetic"
    else:
        inputs = np.asarray(verification_inputs)
        seed = None
        kind = "supplied"
    if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (256, 4) or inputs.shape[0] < 1:
        raise ConfigurationError(
            "verification_inputs must have shape [samples, 256, 4]"
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
    return inputs, record


def predict_baseline(hls_model: Any, inputs: np.ndarray) -> np.ndarray:
    """Compile and predict with the clean hls4ml baseline boundary."""

    if not callable(getattr(hls_model, "compile", None)) or not callable(
        getattr(hls_model, "predict", None)
    ):
        raise VerificationError(
            "Required hls4ml compile/predict capability is unavailable"
        )
    try:
        hls_model.compile()
    except Exception as error:
        raise VerificationError(
            f"hls4ml baseline compilation failed: {error}"
        ) from error
    try:
        return np.asarray(hls_model.predict(inputs))
    except Exception as error:
        raise VerificationError(f"hls4ml baseline prediction failed: {error}") from error


def predict_optimized(project_path: Path, inputs: np.ndarray) -> np.ndarray:
    """Compile and predict through hls4ml's existing-project boundary."""

    from hls4ml.utils.link import FilesystemModelGraph

    try:
        linked = FilesystemModelGraph(project_path)
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
