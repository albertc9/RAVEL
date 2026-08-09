#!/usr/bin/env python3
"""Generate the canonical CNN-for-Arianna project through RAVEL's public API."""

import argparse
import json
from pathlib import Path
from typing import Sequence


REFERENCE_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the CNN-for-Arianna Aria project"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REFERENCE_ROOT / "models" / "cnn_for_arianna.keras",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_ROOT / "generated" / "cnn_core",
    )
    parser.add_argument("--project-name", default="cnn_core")
    parser.add_argument("--part", default="xcku5p-ffvb676-2-e")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument(
        "--verification",
        choices=("auto", "required", "disabled"),
        default="required",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--inputs",
        type=Path,
        help="Optional NumPy .npy verification tensor with shape [samples, 256, 4]",
    )
    parser.add_argument("--force-replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    import hls4ml
    import keras
    import numpy as np
    from hgq.layers import QConv2D, QDense
    from ravel_hls import RavelConfig, convert_from_keras_model

    model = keras.models.load_model(
        args.model, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
    )
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vitis"
    )
    hls_config.setdefault("Model", {}).update(
        {"Strategy": "Latency", "ReuseFactor": 1}
    )
    verification_inputs = np.load(args.inputs) if args.inputs is not None else None
    project = convert_from_keras_model(
        model,
        output_dir=args.output,
        project_name=args.project_name,
        hls_config=hls_config,
        ravel_config=RavelConfig(
            {
                "Profile": "aria",
                "Verification": {
                    "Mode": args.verification,
                    "Samples": args.samples,
                    "Seed": args.seed,
                },
            }
        ),
        part=args.part,
        clock_period=args.clock_period,
        force_replace=args.force_replace,
        verification_inputs=verification_inputs,
    )
    print(
        json.dumps(
            {
                "output": str(project.path),
                "generation_fingerprint": project.manifest["generation_fingerprint"],
                "status": project.status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
