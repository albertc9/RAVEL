"""Import measured Vitis HLS evidence without launching vendor tools."""

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from ..exceptions import ProjectGenerationError, VerificationError
from ..identity import QUALIFICATION_SCHEMA_VERSION
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
    stages: dict[str, dict[str, Any]]
    resources: dict[str, int]
    rtl_ports: dict[str, dict[str, int | str]]
    rtl_cosimulation: str
    report_files: dict[str, str]
    stage_plan: tuple[dict[str, Any], ...] = ()
    interfaces: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[dict[str, Any], ...] = ()
    ooc: dict[str, Any] | None = None
    rtl_protocol: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
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
            "stages": self.stages,
            "resources": self.resources,
            "rtl_ports": self.rtl_ports,
            "rtl_cosimulation": self.rtl_cosimulation,
            "report_files": self.report_files,
            "stage_plan": list(self.stage_plan),
            "interfaces": self.interfaces,
            "warnings": list(self.warnings),
            "status": "recorded",
            "ooc": self.ooc,
            "rtl_protocol": self.rtl_protocol,
        }


def import_vitis_reports(
    project: RavelProject | str | os.PathLike[str],
    *,
    report_dir: str | os.PathLike[str],
    ooc_dir: str | os.PathLike[str] | None = None,
    protocol_report: str | os.PathLike[str] | None = None,
) -> QualificationRecord:
    """Parse a completed Vitis report tree and atomically attach its measurements."""

    project_view = project if isinstance(project, RavelProject) else open_project(project)
    if project_view.manifest.get("schema_version") not in {2, 3, 4, 5, 6, 7, 8}:
        raise ProjectGenerationError(
            "Vitis evidence can only be recorded for a schema-v2 through "
            "schema-v7 project"
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
    cosim_requested = bool(project_view.config["Vitis"]["Stages"]["CoSim"])
    cosim_report = _rtl_cosimulation_report(report_root, reported_top)
    if cosim_requested and cosim_report is None:
        raise ProjectGenerationError(
            "Vitis run requested RTL CoSim but no passing top-level CoSim report "
            f"was found for {reported_top}"
        )
    rtl_cosimulation = "passed" if cosim_report is not None else "not_run"
    report_files = {
        report_path.relative_to(report_root).as_posix(): _file_sha256(report_path)
    }
    if cosim_report is not None:
        report_files[
            cosim_report.relative_to(report_root).as_posix()
        ] = _file_sha256(cosim_report)
    stages, stage_reports = _stage_evidence(
        report_root,
        project_view.manifest,
        expected_tool_version=tool_version,
        expected_part=reported_part,
        expected_clock=reported_clock,
    )
    for stage_report in stage_reports:
        report_files[stage_report.relative_to(report_root).as_posix()] = _file_sha256(
            stage_report
        )
    ooc = None
    warnings = list(project_view.manifest.get("resolved_design", {}).get("warnings", ()))
    if ooc_dir is not None:
        from .ooc import import_ooc
        ooc = import_ooc(Path(ooc_dir), {"manifest_sha256": manifest_sha256,
            "source_closure_sha256": project_view.manifest["source_closure_sha256"],
            "top": reported_top, "part": reported_part, "clock_period_ns": reported_clock, "tool_version": "2023.2"})
        if ooc["timing"]["wns_ns"] < 0:
            warnings.append({"code": "ooc.timing_miss", "message": "Routed design misses the requested clock period"})
    protocol = None
    if protocol_report is not None:
        protocol = json.loads(Path(protocol_report).read_text())
        for key, expected in {"manifest_sha256": manifest_sha256, "top": reported_top,
                              "source_closure_sha256": project_view.manifest["source_closure_sha256"],
                              "vector_files": project_view.manifest.get("verification", {}).get("rtl_reference", {}).get("files")}.items():
            if protocol.get(key) != expected or expected is None:
                raise ProjectGenerationError(f"RTL protocol {key} disagrees with the bound project")
        if protocol.get("status") != "passed" or any(protocol.get(key, 0) < 1 for key in ("completed_samples", "reset_aborts", "output_stall_cycles")):
            raise ProjectGenerationError("RTL protocol evidence is incomplete")
        protocol["report_sha256"] = _file_sha256(Path(protocol_report))
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
        stages=stages,
        resources=resources,
        rtl_ports=rtl_ports,
        rtl_cosimulation=rtl_cosimulation,
        report_files=report_files,
        stage_plan=tuple(project_view.manifest.get("resolved_design", {}).get("stages", ())),
        interfaces=project_view.manifest.get("interfaces", {}),
        warnings=tuple(warnings),
        ooc=ooc,
        rtl_protocol=protocol,
    )
    qualification_path = project_view.path / "ravel_qualification.json"
    temporary_path = qualification_path.with_name(".ravel_qualification.json.tmp")
    temporary_path.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, qualification_path)
    return record


def _stage_evidence(
    report_root: Path,
    manifest: dict[str, Any],
    *,
    expected_tool_version: str,
    expected_part: str,
    expected_clock: float,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    resolved_design = manifest.get("resolved_design", {})
    if resolved_design.get("strategy", {}).get("id") == "aria-composed":
        return _composed_stage_evidence(report_root, resolved_design, expected_tool_version, expected_part, expected_clock)
    rendering = resolved_design.get("rendering", {})
    if resolved_design.get("strategy", {}).get("id") == "phara":
        stage_functions = {
            "phara_fused_region": rendering.get("phara_fused_function"),
            "dense": rendering.get("dense_function"),
        }
    else:
        stage_functions = {
            "first_convolution": rendering.get("first_convolution_function")
        }
    if not all(isinstance(name, str) for name in stage_functions.values()):
        return {}, []
    stages = {}
    stage_reports = []
    for stage, function_name in stage_functions.items():
        evidence, report_paths = _one_stage_evidence(
            report_root,
            stage=stage,
            function_name=function_name,
            expected_tool_version=expected_tool_version,
            expected_part=expected_part,
            expected_clock=expected_clock,
        )
        stages[stage] = evidence
        stage_reports.extend(report_paths)
    return stages, stage_reports


def _one_stage_evidence(
    report_root: Path,
    *,
    stage: str,
    function_name: str,
    expected_tool_version: str,
    expected_part: str,
    expected_clock: float,
) -> tuple[dict[str, Any], list[Path]]:
    matches: list[tuple[Path, ET.Element]] = []
    for candidate in sorted(report_root.rglob("*_csynth.xml")):
        try:
            root = ET.parse(candidate).getroot()
        except (OSError, ET.ParseError):
            continue
        top = root.findtext("./UserAssignments/TopModelName", "").strip()
        if top == function_name or top.startswith(f"{function_name}_"):
            matches.append((candidate, root))
    if not matches:
        raise ProjectGenerationError(
            f"Expected at least one {stage} Vitis csynth XML report"
        )
    pipelined = []
    for _, root in matches:
        for field, expected, observed in (
            (
                "version",
                expected_tool_version,
                _required_text(root, "./ReportVersion/Version"),
            ),
            ("part", expected_part, _required_text(root, "./UserAssignments/Part")),
            (
                "clock",
                expected_clock,
                float(
                    _required_text(
                        root, "./UserAssignments/TargetClockPeriod"
                    )
                ),
            ),
        ):
            if expected != observed:
                raise ProjectGenerationError(
                    f"{stage} Vitis {field} expected {expected} "
                    f"but measured {observed}"
                )
        pipelined.extend(
            loop
            for loop in root.findall(
                "./PerformanceEstimates/SummaryOfLoopLatency/*"
            )
            if loop.findtext("./PipelineII") is not None
        )
    if len(pipelined) != 1:
        raise ProjectGenerationError(
            f"Expected exactly one pipelined {stage} loop"
        )
    loop = pipelined[0]
    _, root = max(
        matches,
        key=lambda match: (
            int(
                _required_text(
                    match[1],
                    "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min",
                )
            ),
            int(
                _required_text(
                    match[1],
                    "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
                )
            ),
            -len(_required_text(match[1], "./UserAssignments/TopModelName")),
        ),
    )
    return {
        "top": _required_text(root, "./UserAssignments/TopModelName"),
        "initiation_interval": int(
            _required_text(
                root,
                "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min",
            )
        ),
        "latency_cycles": int(
            _required_text(
                root,
                "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
            )
        ),
        "loop": {
            "name": loop.tag,
            "trip_count": int(_required_text(loop, "./TripCount")),
            "pipeline_ii": int(_required_text(loop, "./PipelineII")),
            "pipeline_depth": int(_required_text(loop, "./PipelineDepth")),
        },
    }, [path for path, _ in matches]


def _rtl_cosimulation_report(report_root: Path, top: str) -> Path | None:
    candidates = sorted(report_root.rglob(f"{top}_cosim.rpt"))
    passing = []
    for candidate in candidates:
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(
            "|" in line and "Verilog" in line and "Pass" in line
            for line in content.splitlines()
        ):
            passing.append(candidate)
    if len(passing) > 1:
        raise ProjectGenerationError(
            "Expected at most one passing top-level Vitis RTL CoSim report"
        )
    return passing[0] if passing else None


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


def _composed_stage_evidence(report_root, design, version, part, clock):
    bindings = design.get("report_bindings")
    if not bindings:
        raise ProjectGenerationError("Composed manifest has no stage report bindings")
    reports = []
    for path in sorted(report_root.rglob("*_csynth.xml")):
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        reports.append((path, root))
    stages, used = {}, []
    for binding in bindings:
        functions = []
        for selector in binding["functions"]:
            matches = [(path, root) for path, root in reports
                       if (top := root.findtext("./UserAssignments/TopModelName", "")).startswith(selector["name"] + "_")
                       and ("config" not in selector or re.search(r"(?:^|_)" + re.escape(selector["config"]) + r"(?:_|$)", top))]
            if not matches:
                raise ProjectGenerationError(f"No Vitis report for selected stage {binding['stage_id']}: {selector}")
            for path, root in matches:
                if (_required_text(root, "./ReportVersion/Version") != version
                        or _required_text(root, "./UserAssignments/Part") != part
                        or float(_required_text(root, "./UserAssignments/TargetClockPeriod")) != clock):
                    raise ProjectGenerationError(f"Selected stage report disagrees with top environment: {path.name}")
                functions.append({"top": _required_text(root, "./UserAssignments/TopModelName"),
                                  "initiation_interval": int(_required_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Interval-min")),
                                  "latency_cycles": int(_required_text(root, "./PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")),
                                  "loops": [{"name": loop.tag, "pipeline_ii": int(loop.findtext("PipelineII"))}
                                            for loop in root.findall("./PerformanceEstimates/SummaryOfLoopLatency/*") if loop.findtext("PipelineII") is not None]})
                used.append(path)
        stages[binding["stage_id"]] = {"functions": functions, "realization": binding.get("realization", "measured")}
    return stages, used
