#!/usr/bin/env python3
"""Convert the homogeneous low-BOP Keras model to an IOStream HLS baseline.

This flow is intentionally direct. The low-BOP model is already homogeneous, so
there is no IOParallel precision oracle, HGQ reference extraction, or stream
quantizer patching step.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "models/hgq_config_beta7_gamma6_p1_cl_lowbop.keras"
DEFAULT_OUTPUT_DIR = "cnn_core_project"
DEFAULT_DATA_DIR = "data"
DEFAULT_BACKEND = "Vitis"
DEFAULT_PART = "xcku5p-ffvb676-2-e"
DEFAULT_PROJECT_NAME = "cnn_core"
DEFAULT_IO_TYPE = "io_stream"
DEFAULT_CLOCK_PERIOD = 5
DEFAULT_INPUT_STREAM_DEPTH = 16
DEFAULT_DEFAULT_PRECISION = "ap_fixed<16,6>"
DEFAULT_INPUT_PRECISION = "ap_fixed<12,6>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the homogeneous low-BOP CNN model with hls4ml."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--part", default=DEFAULT_PART)
    parser.add_argument("--io-type", default=DEFAULT_IO_TYPE, choices=("io_stream", "io_parallel"))
    parser.add_argument("--clock-period", type=float, default=DEFAULT_CLOCK_PERIOD)
    parser.add_argument("--reuse-factor", type=int, default=1)
    parser.add_argument("--strategy", default="Latency", choices=("Latency", "Resource"))
    parser.add_argument("--default-precision", default=DEFAULT_DEFAULT_PRECISION)
    parser.add_argument("--input-precision", default=DEFAULT_INPUT_PRECISION)
    parser.add_argument("--input-stream-depth", type=int, default=DEFAULT_INPUT_STREAM_DEPTH)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not remove the existing output directory before conversion.",
    )
    args = parser.parse_args()
    if args.skip_compile and not args.skip_verify:
        parser.error("--skip-compile requires --skip-verify because hls_model.predict() needs compile().")
    return args


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def assert_safe_replace_path(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {Path("/"), REPO_ROOT, REPO_ROOT.parent}:
        raise ValueError(f"Refusing to replace unsafe path: {resolved}")


def prepare_output_dir(path: Path, keep_output: bool) -> None:
    assert_safe_replace_path(path)
    if path.exists() and not keep_output:
        print(f"[INFO] Removing existing output directory: {path}")
        shutil.rmtree(path)
    elif not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def import_dependencies() -> tuple[Any, Any, dict[str, Any]]:
    try:
        import hls4ml
    except ImportError as exc:
        print(f"[ERROR] Missing hls4ml: {exc}", file=sys.stderr)
        raise

    try:
        import keras
    except ImportError:
        try:
            from tensorflow import keras
        except ImportError as exc:
            print(f"[ERROR] Missing keras/tensorflow: {exc}", file=sys.stderr)
            raise

    custom_objects: dict[str, Any] = {}
    try:
        from hgq.layers import QConv2D, QDense
    except ImportError:
        print("[INFO] HGQ package not found; loading model with standard Keras objects only.")
    else:
        custom_objects = {"QConv2D": QConv2D, "QDense": QDense}
        print("[INFO] HGQ package found; registering QConv2D/QDense for model loading.")

    return hls4ml, keras, custom_objects


def supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def get_model_input_shape(model: Any) -> tuple[int, ...]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape is None:
        raise ValueError("Unable to determine model input shape")
    return tuple(input_shape[1:])


def find_input_layer_name(config: dict[str, Any]) -> str | None:
    for name in config.get("LayerName", {}):
        if "input" in name.lower():
            return name
    return None


def build_hls_config(hls4ml: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    config_kwargs = supported_kwargs(
        hls4ml.utils.config_from_keras_model,
        {
            "model": model,
            "granularity": "name",
            "backend": args.backend,
            "default_precision": args.default_precision,
        },
    )
    config = hls4ml.utils.config_from_keras_model(**config_kwargs)

    model_config = config.setdefault("Model", {})
    model_config["Strategy"] = args.strategy
    model_config["ReuseFactor"] = args.reuse_factor

    input_layer_name = find_input_layer_name(config)
    if input_layer_name is not None:
        config["LayerName"][input_layer_name]["Precision"] = {"result": args.input_precision}
        if args.io_type == "io_stream":
            config["LayerName"][input_layer_name]["StreamDepth"] = args.input_stream_depth
        print(f"[CONFIG] Input layer '{input_layer_name}' result -> {args.input_precision}")
    else:
        print("[WARNING] No input layer found in hls4ml config; input precision not set.")

    return config


def convert_model(hls4ml: Any, model: Any, hls_config: dict[str, Any], args: argparse.Namespace) -> Any:
    kwargs = supported_kwargs(
        hls4ml.converters.convert_from_keras_model,
        {
            "model": model,
            "hls_config": hls_config,
            "output_dir": str(rel(args.output_dir)),
            "project_name": args.project_name,
            "io_type": args.io_type,
            "backend": args.backend,
            "part": args.part,
            "clock_period": args.clock_period,
        },
    )

    print("[INFO] Converting homogeneous model with settings:")
    for key in ("output_dir", "project_name", "io_type", "backend", "part", "clock_period"):
        if key in kwargs:
            print(f"   {key}: {kwargs[key]}")
    return hls4ml.converters.convert_from_keras_model(**kwargs)


def prepare_test_data(data_dir: Path, model_input_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    x_path = data_dir / "X_test_data.npy"
    y_path = data_dir / "y_test_labels.npy"
    if not x_path.exists() or not y_path.exists():
        print("[WARNING] Test data not found; skipping verification.")
        return None

    x_test = np.load(x_path)
    y_test = np.load(y_path)
    if x_test.ndim == 4 and x_test.shape[-1] == 1:
        x_test = x_test.squeeze(axis=-1)
    if tuple(x_test.shape[1:]) == model_input_shape:
        x_prepared = x_test
    elif x_test.ndim == 3 and tuple(x_test.shape[1:][::-1]) == model_input_shape:
        x_prepared = np.transpose(x_test, (0, 2, 1))
    else:
        raise ValueError(f"Unsupported X_test shape {x_test.shape}; model expects {model_input_shape}.")
    return np.ascontiguousarray(x_prepared), y_test


def verify_model(hls_model: Any, keras_model: Any, data_dir: Path, model_input_shape: tuple[int, ...]) -> None:
    prepared = prepare_test_data(data_dir, model_input_shape)
    if prepared is None:
        return

    x_test, y_test = prepared
    print("=" * 40)
    print("[TEST] Homogeneous IOStream baseline")
    print("=" * 40)
    y_hls = hls_model.predict(x_test)
    y_keras = keras_model.predict(x_test)

    y_hls_bin = (y_hls > 0).astype(int).flatten()
    y_keras_bin = (y_keras > 0).astype(int).flatten()
    y_label_bin = y_test.astype(int).flatten()
    accuracy = float(np.mean(y_label_bin == y_hls_bin))
    fidelity = float(np.mean(y_keras_bin == y_hls_bin))
    max_abs_diff = float(np.max(np.abs(y_hls.flatten() - y_keras.flatten())))
    print(f"   HLS accuracy:       {accuracy:.4f}")
    print(f"   HLS/Keras fidelity: {fidelity:.4f}")
    print(f"   Max abs score diff: {max_abs_diff:.6g}")


def main() -> int:
    args = parse_args()
    hls4ml, keras, custom_objects = import_dependencies()

    model_path = rel(args.model_path)
    output_dir = rel(args.output_dir)
    data_dir = rel(args.data_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    print(f"[INFO] Loading homogeneous Keras model: {model_path}")
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    model_input_shape = get_model_input_shape(model)
    print(f"[INFO] Model input shape: {model_input_shape}")

    hls_config = build_hls_config(hls4ml, model, args)
    prepare_output_dir(output_dir, args.keep_output)
    hls_model = convert_model(hls4ml, model, hls_config, args)
    hls_model.write()
    print(f"[INFO] Project written to {output_dir}")

    if args.skip_compile:
        print("[INFO] Skipping hls_model.compile().")
    else:
        print("[INFO] Compiling C simulation library...")
        hls_model.compile()
        print("[INFO] Compile success.")

    if args.skip_verify:
        print("[INFO] Skipping verification.")
    else:
        verify_model(hls_model, model, data_dir, model_input_shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
