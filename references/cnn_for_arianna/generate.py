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
    parser.add_argument("--part", default="xcku5p-ffvb676-2-e")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument("--temporal-packing", type=int, choices=(2, 4))
    parser.add_argument("--dense-parallelism", type=int, choices=(1, 2))
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
    parser.add_argument(
        "--vitis",
        action="store_true",
        help="Run Vitis HLS synthesis and RTL co-simulation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    import numpy as np
    import ravel_hls as ravel

    verification_inputs = np.load(args.inputs) if args.inputs is not None else None
    config = {
        "Project": {
            "ForceReplace": args.force_replace,
        },
        "HLS": {
            "Backend": "Vitis",
            "IOType": "io_stream",
            "Part": args.part,
            "ClockPeriod": args.clock_period,
        },
        "Verification": {
            "Mode": args.verification,
            "Samples": args.samples,
            "Seed": args.seed,
        },
        "Vitis": {
            "Run": args.vitis,
            "Stages": {"CoSim": args.vitis},
        },
    }
    optimization = {}
    if args.temporal_packing is not None:
        optimization["TemporalPacking"] = args.temporal_packing
    if args.dense_parallelism is not None:
        optimization["DenseParallelism"] = args.dense_parallelism
    if optimization:
        config["Optimization"] = optimization
    project = ravel.convert(
        args.model,
        args.output,
        config,
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
