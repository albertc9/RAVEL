"""Vitis HLS build configuration and execution support."""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from ...exceptions import ProjectGenerationError


_DEFAULT_STAGES = {
    "Reset": True,
    "CSim": False,
    "Synth": True,
    "CoSim": False,
    "Validation": False,
    "Export": False,
    "VSynth": False,
}

_UNSUPPORTED_ARRAY_PARTITION = re.compile(
    r"^\s*catch\s+\{config_array_partition\s+-maximum_size\s+\$maximum_size\}\s*$"
)


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


def normalize_build_script(project_path: Path) -> None:
    """Remove hls4ml commands that Vitis HLS 2023.2 does not support."""

    script_path = project_path / "build_prj.tcl"
    try:
        lines = script_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ProjectGenerationError(
            f"Cannot normalize generated Vitis build script: {error}"
        ) from error
    normalized = [
        line for line in lines if _UNSUPPORTED_ARRAY_PARTITION.fullmatch(line) is None
    ]
    script_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")
