"""Import measured Vitis HLS evidence without launching vendor tools."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from ..exceptions import ProjectGenerationError, VerificationError
from ..project import RavelProject, open_project


@dataclass(frozen=True)
class QualificationRecord:
    """Typed view of one imported Vitis synthesis report."""

    manifest_sha256: str
    tool_version: str
    part: str
    target_clock_ns: float
    estimated_clock_ns: float
    initiation_interval: int
    latency_cycles: int
    resources: dict[str, int]
    rtl_ports: dict[str, dict[str, int | str]]
    report_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "tool": {"name": "Vitis HLS", "version": self.tool_version},
            "part": self.part,
            "timing": {
                "target_clock_ns": self.target_clock_ns,
                "estimated_clock_ns": self.estimated_clock_ns,
            },
            "performance": {
                "initiation_interval": self.initiation_interval,
                "latency_cycles": self.latency_cycles,
            },
            "resources": self.resources,
            "rtl_ports": self.rtl_ports,
            "report_files": self.report_files,
            "status": "recorded",
        }


def import_vitis_reports(
    project: RavelProject | str | os.PathLike[str],
    *,
    report_dir: str | os.PathLike[str],
) -> QualificationRecord:
    """Parse a completed Vitis report tree and atomically attach its measurements."""

    project_view = project if isinstance(project, RavelProject) else open_project(project)
    if project_view.status.get("source_integrity") != "clean":
        raise VerificationError(
            "Cannot qualify a modified RAVEL project; regenerate or restore managed files"
        )
    report_root = Path(report_dir)
    candidates = sorted(report_root.rglob("*_csynth.xml"))
    if not candidates:
        raise ProjectGenerationError(
            f"No Vitis top-level csynth XML report found under {report_root}"
        )
    parsed: list[tuple[Path, ET.Element]] = []
    for candidate in candidates:
        try:
            root = ET.parse(candidate).getroot()
        except (OSError, ET.ParseError):
            continue
        top_name = root.findtext("./UserAssignments/TopModelName")
        if top_name and candidate.name == f"{top_name}_csynth.xml":
            parsed.append((candidate, root))
    if len(parsed) != 1:
        raise ProjectGenerationError(
            "Expected exactly one complete top-level Vitis csynth XML report"
        )
    report_path, root = parsed[0]
    tool_version = _required_text(root, "./ReportVersion/Version")
    if tool_version != "2023.2":
        raise ProjectGenerationError(
            f"Unsupported Vitis report version {tool_version}; qualified version is 2023.2"
        )
    manifest_path = project_view.path / "ravel_manifest.json"
    manifest_sha256 = _file_sha256(manifest_path)
    resources_node = root.find("./AreaEstimates/Resources")
    if resources_node is None:
        raise ProjectGenerationError("Vitis report is missing AreaEstimates/Resources")
    resources = {
        name: int(_required_text(resources_node, f"./{name}"))
        for name in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    }
    rtl_ports: dict[str, dict[str, int | str]] = {}
    for port in root.findall("./InterfaceSummary/RtlPorts"):
        name = _required_text(port, "./name")
        rtl_ports[name] = {
            "direction": _required_text(port, "./Dir"),
            "bits": int(_required_text(port, "./Bits")),
        }
    expected_rtl = (
        project_view.manifest.get("interfaces", {})
        .get("rtl_interface", {})
        .get("expected", {})
    )
    for manifest_name, port_name in (
        ("input_tdata_bits", "input_TDATA"),
        ("output_tdata_bits", "output_TDATA"),
    ):
        expected_bits = expected_rtl.get(manifest_name)
        measured_bits = rtl_ports.get(port_name, {}).get("bits")
        if not isinstance(expected_bits, int) or measured_bits != expected_bits:
            raise ProjectGenerationError(
                f"Vitis {port_name} expected {expected_bits} bits but measured "
                f"{measured_bits}"
            )
    record = QualificationRecord(
        manifest_sha256=manifest_sha256,
        tool_version=tool_version,
        part=_required_text(root, "./UserAssignments/Part"),
        target_clock_ns=float(
            _required_text(root, "./UserAssignments/TargetClockPeriod")
        ),
        estimated_clock_ns=float(
            _required_text(
                root,
                "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
            )
        ),
        initiation_interval=int(
            _required_text(
                root,
                "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min",
            )
        ),
        latency_cycles=int(
            _required_text(
                root,
                "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
            )
        ),
        resources=resources,
        rtl_ports=rtl_ports,
        report_files={
            report_path.relative_to(report_root).as_posix(): _file_sha256(report_path)
        },
    )
    qualification_path = project_view.path / "ravel_qualification.json"
    temporary_path = qualification_path.with_name(".ravel_qualification.json.tmp")
    temporary_path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, qualification_path)
    return record


def _required_text(node: ET.Element, path: str) -> str:
    value = node.findtext(path)
    if value is None or not value.strip():
        raise ProjectGenerationError(f"Vitis report is missing {path}")
    return value.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
