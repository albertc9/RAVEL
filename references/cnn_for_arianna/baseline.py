#!/usr/bin/env python3
"""Generate and optionally synthesize the unmodified hls4ml baseline."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET


REFERENCE_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the vanilla hls4ml CNN-for-Arianna baseline"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REFERENCE_ROOT / "models" / "cnn_for_arianna.keras",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REFERENCE_ROOT / "generated" / "hls4ml_baseline",
    )
    parser.add_argument("--project-name", default="cnn_core")
    parser.add_argument("--part", default="xcku5p-ffvb676-2-e")
    parser.add_argument("--clock-period", type=float, default=5.0)
    parser.add_argument(
        "--vitis",
        action="store_true",
        help="Run synthesis through hls4ml's public build API",
    )
    return parser


def _metrics(output: Path, project_name: str) -> dict[str, Any]:
    report = (
        output
        / f"{project_name}_prj"
        / "solution1"
        / "syn"
        / "report"
        / f"{project_name}_csynth.xml"
    )
    root = ET.parse(report).getroot()

    def required(path: str) -> str:
        value = root.findtext(path)
        if value is None:
            raise RuntimeError(f"hls4ml baseline report is missing {path}")
        return value

    return {
        "tool": {
            "name": "Vitis HLS",
            "version": required("./ReportVersion/Version"),
        },
        "part": required("./UserAssignments/Part"),
        "timing": {
            "target_clock_ns": float(
                required("./UserAssignments/TargetClockPeriod")
            ),
            "estimated_clock_ns": float(
                required(
                    "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"
                )
            ),
        },
        "performance": {
            "initiation_interval": int(
                required(
                    "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min"
                )
            ),
            "latency_cycles": int(
                required(
                    "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"
                )
            ),
        },
        "resources": {
            name: int(required(f"./AreaEstimates/Resources/{name}"))
            for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
        },
        "report": report.relative_to(output).as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(
            f"Refusing to reuse baseline output directory: {args.output}"
        )

    import hls4ml
    import keras
    from hgq.layers import QConv2D, QDense

    model = keras.models.load_model(
        args.model, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
    )
    hls_config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vitis"
    )
    hls_config["Model"].update({"Strategy": "Latency", "ReuseFactor": 1})
    hls_model = hls4ml.converters.convert_from_keras_model(
        model=model,
        output_dir=str(args.output),
        project_name=args.project_name,
        hls_config=hls_config,
        backend="Vitis",
        io_type="io_stream",
        part=args.part,
        clock_period=args.clock_period,
    )
    hls_model.write()
    result: dict[str, Any] = {
        "flow": "hls4ml-vanilla",
        "model": args.model.name,
        "output": str(args.output),
        "source_policy": "no generated source edits",
        "status": "generated",
    }
    if args.vitis:
        hls_model.build(
            reset=True,
            csim=False,
            synth=True,
            cosim=False,
            validation=False,
            export=False,
            vsynth=False,
            fifo_opt=False,
            log_to_stdout=False,
        )
        result.update(_metrics(args.output, args.project_name))
        result["status"] = "synthesized"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
