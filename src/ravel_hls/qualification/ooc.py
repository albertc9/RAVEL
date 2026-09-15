"""Read routed Vivado measurements and their pre-build source binding."""

import json
from pathlib import Path
import re

from ..exceptions import ProjectGenerationError
from ..manifest import file_sha256


def import_ooc(directory: Path, expected: dict) -> dict:
    try:
        binding = json.loads((directory / "binding.json").read_text())
        timing = (directory / "timing.rpt").read_text()
        utilization = (directory / "utilization.rpt").read_text()
    except (OSError, ValueError) as error:
        raise ProjectGenerationError(f"Cannot read OOC evidence: {error}") from error
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ProjectGenerationError(f"OOC {key} disagrees with the bound project")
    for pattern in (r"Tool Version\s*:\s*Vivado v\.2023\.2\b",
                    r"Design\s*:\s*" + re.escape(expected["top"]) + r"\s*(?:\n|$)",
                    r"Design State\s*:\s*Routed\b"):
        if not re.search(pattern, timing):
            raise ProjectGenerationError("OOC timing report has an unqualified tool, top or implementation state")
    rows = timing.splitlines()
    try:
        index = next(i for i, row in enumerate(rows) if "WNS(ns)" in row and "TNS(ns)" in row)
        values = next(row.split() for row in rows[index + 1:] if re.match(r"\s*-?\d+\.\d+", row))
        wns, tns = map(float, values[:2])
        resources = {}
        for label, key in (("CLB LUTs", "LUT"), ("CLB Registers", "FF"), ("Block RAM Tile", "BRAM_TILES"), ("DSPs", "DSP")):
            match = re.search(r"\|\s*" + label + r"\*?\s*\|\s*([\d,.]+)\s*\|", utilization)
            resources[key] = float(match[1].replace(",", ""))
    except (StopIteration, ValueError, TypeError) as error:
        raise ProjectGenerationError("Incomplete routed timing or utilization report") from error
    return {**binding, "timing": {"wns_ns": wns, "tns_ns": tns, "target_clock_ns": expected["clock_period_ns"]},
            "resources": resources, "status": "recorded", "report_files": {
                name: file_sha256(directory / name) for name in ("binding.json", "timing.rpt", "utilization.rpt")}}
