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
        if not all(re.search(pattern, report) for report in (timing, utilization)):
            raise ProjectGenerationError("OOC timing report has an unqualified tool, top or implementation state")
    device = re.search(r"Device\s*:\s*(\S+)", utilization)
    if device is None or device[1].lower() != expected["part"].lower():
        raise ProjectGenerationError("OOC report part disagrees with the bound project")
    clocks = re.findall(r"^ap_clk\s+\{[^}]+\}\s+(\d+(?:\.\d+)?)\s+", timing, re.MULTILINE)
    if len(clocks) != 1 or abs(float(clocks[0]) - expected["clock_period_ns"]) > 0.0005:
        raise ProjectGenerationError("OOC report clock disagrees with the bound project")
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
