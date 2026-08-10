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
    generation_fingerprint: str
    source_closure_sha256: str
    top: str
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
            "schema_version": 2,
            "manifest_sha256": self.manifest_sha256,
            "generation_fingerprint": self.generation_fingerprint,
            "source_closure_sha256": self.source_closure_sha256,
            "top": self.top,
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
    if project_view.manifest.get("schema_version") != 2:
        raise ProjectGenerationError(
            "Vitis evidence can only be recorded for a schema-v2 project"
        )
    if project_view.status.get("source_integrity") != "clean":
        raise VerificationError(
            "Cannot qualify a modified RAVEL project; regenerate or restore managed files"
        )
    expected_hls = project_view.manifest.get("normalized_configuration", {}).get(
        "hls4ml", {}
    )
    expected_top = expected_hls.get("ProjectName")
    if not isinstance(expected_top, str) or not expected_top.isidentifier():
        raise ProjectGenerationError(
            "RAVEL manifest does not contain a valid hls4ml ProjectName"
        )
    report_root = Path(report_dir)
    candidates = sorted(report_root.rglob(f"{expected_top}_csynth.xml"))
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
        if top_name == expected_top:
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
    reported_top = _required_text(root, "./UserAssignments/TopModelName")
    reported_part = _required_text(root, "./UserAssignments/Part")
    reported_clock = float(
        _required_text(root, "./UserAssignments/TargetClockPeriod")
    )
    for field, expected, measured in (
        ("ProjectName", expected_hls.get("ProjectName"), reported_top),
        ("Part", expected_hls.get("Part"), reported_part),
        ("ClockPeriod", expected_hls.get("ClockPeriod"), reported_clock),
    ):
        if expected != measured:
            raise ProjectGenerationError(
                f"Vitis {field} expected {expected} but measured {measured}"
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
    for bits_name, port_name_key, expected_direction in (
        ("input_tdata_bits", "input_tdata_port", "in"),
        ("output_tdata_bits", "output_tdata_port", "out"),
    ):
        expected_bits = expected_rtl.get(bits_name)
        port_name = expected_rtl.get(port_name_key)
        measured_port = rtl_ports.get(port_name, {})
        measured_bits = measured_port.get("bits")
        measured_direction = measured_port.get("direction")
        if (
            not isinstance(port_name, str)
            or not isinstance(expected_bits, int)
            or measured_bits != expected_bits
            or measured_direction != expected_direction
        ):
            raise ProjectGenerationError(
                f"Vitis {port_name} expected {expected_bits} bits but measured "
                f"{measured_bits}; expected direction {expected_direction} but measured "
                f"{measured_direction}"
            )
    record = QualificationRecord(
        manifest_sha256=manifest_sha256,
        generation_fingerprint=_required_manifest_sha256(
            project_view.manifest, "generation_fingerprint"
        ),
        source_closure_sha256=_required_manifest_sha256(
            project_view.manifest, "source_closure_sha256"
        ),
        top=reported_top,
        tool_version=tool_version,
        part=reported_part,
        target_clock_ns=reported_clock,
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


def _required_manifest_sha256(manifest: dict[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ProjectGenerationError(f"RAVEL manifest has no valid {key}")
    return value
