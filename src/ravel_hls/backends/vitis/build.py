"""Vitis HLS build configuration and execution support."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_DEFAULT_STAGES = {
    "Reset": True,
    "CSim": False,
    "Synth": True,
    "CoSim": False,
    "Validation": False,
    "Export": False,
    "VSynth": False,
}


def write_build_options(project_path: Path, config: Mapping[str, Any]) -> None:
    """Write the stable hls4ml build controls selected for this invocation."""

    stages = dict(_DEFAULT_STAGES)
    vitis = config.get("Vitis", {})
    if isinstance(vitis, Mapping):
        configured = vitis.get("Stages", {})
        if isinstance(configured, Mapping):
            stages.update(configured)
    values = {
        "reset": stages["Reset"],
        "csim": stages["CSim"],
        "synth": stages["Synth"],
        "cosim": stages["CoSim"],
        "validation": stages["Validation"],
        "export": stages["Export"],
        "vsynth": stages["VSynth"],
        "fifo_opt": False,
    }
    lines = ["array set opt {"]
    lines.extend(
        f"    {name:<10} {int(bool(enabled))}" for name, enabled in values.items()
    )
    lines.append("}")
    (project_path / "build_opt.tcl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
