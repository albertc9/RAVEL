"""Regenerate reviewed analysis reports for every retrained reference model."""

import json
from pathlib import Path

from ravel_hls import analyze


REFERENCE_ROOT = Path(__file__).parents[2] / "references"
SNAPSHOT_ROOT = Path(__file__).with_name("analysis_snapshots")


def main() -> int:
    models = sorted(
        path
        for path in REFERENCE_ROOT.glob("**/*.keras")
        if "cnn_for_arianna" not in path.parts
    )
    if len(models) != 12:
        raise RuntimeError(f"Expected 12 retrained models, found {len(models)}")
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    for model_path in models:
        report = analyze(
            model_path,
            {"HLS": {"Backend": "Vitis", "IOType": "io_stream"}},
        ).to_dict()
        output = SNAPSHOT_ROOT / f"{model_path.parent.name}.json"
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
