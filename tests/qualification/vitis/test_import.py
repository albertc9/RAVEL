import hashlib
import json
from pathlib import Path

from ravel_hls import import_vitis_reports, open_project


def test_import_vitis_reports_links_measured_evidence_to_the_manifest(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path)
    manifest_before = (project_path / "ravel_manifest.json").read_bytes()
    report_dir = tmp_path / "reports"
    report_path = report_dir / "solution1" / "syn" / "report" / "aria_top_csynth.xml"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(_CSYNTH_XML, encoding="utf-8")

    record = import_vitis_reports(project_path, report_dir=report_dir)

    assert record.tool_version == "2023.2"
    assert record.initiation_interval == 178
    assert record.latency_cycles == 183
    assert record.resources == {"BRAM_18K": 0, "DSP": 4, "FF": 3483, "LUT": 28922, "URAM": 0}
    assert record.rtl_ports["input_TDATA"] == {"direction": "in", "bits": 128}
    assert record.rtl_ports["output_TDATA"] == {"direction": "out", "bits": 32}
    assert (project_path / "ravel_qualification.json").is_file()
    assert (project_path / "ravel_manifest.json").read_bytes() == manifest_before
    assert open_project(project_path).status["performance_qualification"] == "recorded"


def _write_project(project_path: Path) -> None:
    source = "void aria_top() {}\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    (project_path / "firmware").mkdir(parents=True)
    (project_path / "firmware" / "aria_top.cpp").write_text(source, encoding="utf-8")
    (project_path / "hls4ml_config.yml").write_text(
        "Backend: Vitis\nIOType: io_stream\n", encoding="utf-8"
    )
    (project_path / "ravel_config.yml").write_text(
        "Profile: aria\nVerification:\n  Mode: required\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "ravel": {"product": "RAVEL", "generation": "Aria", "release": "1.0"},
        "implementation_plan": {"template_profile": "aria-2x-v1"},
        "status": {
            "generation": "complete",
            "dependency_qualification": "qualified",
            "correctness_verification": "passed",
            "model_fidelity": "reported",
            "source_integrity": "clean",
            "performance_qualification": "not_run",
        },
        "managed_files": {"firmware/aria_top.cpp": source_hash},
    }
    (project_path / "ravel_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


_CSYNTH_XML = """\
<profile>
  <ReportVersion><Version>2023.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcku5p-ffvb676-2-e</Part><TopModelName>aria_top</TopModelName>
    <TargetClockPeriod>5.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis><EstimatedClockPeriod>3.647</EstimatedClockPeriod></SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Best-caseLatency>183</Best-caseLatency><Interval-min>178</Interval-min>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates><Resources>
    <DSP>4</DSP><FF>3483</FF><LUT>28922</LUT><BRAM_18K>0</BRAM_18K><URAM>0</URAM>
  </Resources></AreaEstimates>
  <InterfaceSummary>
    <RtlPorts><name>input_TDATA</name><Dir>in</Dir><Bits>128</Bits></RtlPorts>
    <RtlPorts><name>output_TDATA</name><Dir>out</Dir><Bits>32</Bits></RtlPorts>
  </InterfaceSummary>
</profile>
"""
