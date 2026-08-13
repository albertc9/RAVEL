import hashlib
import json
from pathlib import Path

import pytest

from ravel_hls import Project, ProjectGenerationError
from ravel_hls.project import open_project
from ravel_hls.qualification.vitis import import_vitis_reports


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
    (report_path.parent / "conv_kernel_csynth.xml").write_text(
        _CSYNTH_XML.replace("aria_top", "conv_kernel"), encoding="utf-8"
    )

    record = Project.open(project_path).record(report_dir)

    assert record.tool_version == "2023.2"
    assert record.initiation_interval == 178
    assert record.latency_cycles == 183
    assert record.resources == {"BRAM_18K": 0, "DSP": 4, "FF": 3483, "LUT": 28922, "URAM": 0}
    assert record.rtl_ports["input_layer_TDATA"] == {"direction": "in", "bits": 128}
    assert record.rtl_ports["layer9_out_TDATA"] == {"direction": "out", "bits": 32}
    assert record.rtl_cosimulation == "not_run"
    assert (project_path / "ravel_qualification.json").is_file()
    qualification = json.loads(
        (project_path / "ravel_qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["schema_version"] == 4
    assert qualification["stages"] == {}
    assert qualification["generation_fingerprint"] == "1" * 64
    assert qualification["source_closure_sha256"] == json.loads(
        (project_path / "ravel_manifest.json").read_text(encoding="utf-8")
    )["source_closure_sha256"]
    assert qualification["top"] == "aria_top"
    assert (project_path / "ravel_manifest.json").read_bytes() == manifest_before
    assert open_project(project_path).status["performance_qualification"] == "recorded"


def test_import_vitis_reports_links_schema_v3_evidence_to_a_v3_manifest(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path, schema_version=3)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    report_path.write_text(_CSYNTH_XML, encoding="utf-8")

    Project.open(project_path).record(report_dir)

    qualification = json.loads(
        (project_path / "ravel_qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["schema_version"] == 4
    assert Project.open(project_path).status["performance_qualification"] == "recorded"


def test_import_vitis_reports_rejects_an_rtl_width_mismatch(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    report_path.write_text(
        _CSYNTH_XML.replace("<Bits>128</Bits>", "<Bits>64</Bits>", 1),
        encoding="utf-8",
    )

    with pytest.raises(ProjectGenerationError, match="input_layer_TDATA.*128.*64"):
        import_vitis_reports(project_path, report_dir=report_dir)

    assert not (project_path / "ravel_qualification.json").exists()


def test_import_vitis_reports_rejects_a_different_target_part(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    report_path.write_text(
        _CSYNTH_XML.replace(
            "xcku5p-ffvb676-2-e", "xczu28dr-ffvg1517-2-e", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectGenerationError, match="Part.*xcku5p.*xczu28dr"):
        import_vitis_reports(project_path, report_dir=report_dir)

    assert not (project_path / "ravel_qualification.json").exists()


def test_import_records_performance_without_target_thresholds(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    report_path.write_text(
        _CSYNTH_XML.replace(
            "<EstimatedClockPeriod>3.647</EstimatedClockPeriod>",
            "<EstimatedClockPeriod>7.500</EstimatedClockPeriod>",
        ).replace("<Interval-min>178</Interval-min>", "<Interval-min>999</Interval-min>"),
        encoding="utf-8",
    )

    record = Project.open(project_path).record(report_dir)

    assert record.estimated_clock_ns == 7.5
    assert record.initiation_interval == 999
    assert Project.open(project_path).status["performance_qualification"] == "recorded"


def test_import_records_a_requested_passing_rtl_cosimulation(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path, cosim=True)
    report_dir = tmp_path / "reports"
    synthesis = report_dir / "solution1" / "syn" / "report" / "aria_top_csynth.xml"
    synthesis.parent.mkdir(parents=True)
    synthesis.write_text(_CSYNTH_XML, encoding="utf-8")
    cosimulation = report_dir / "solution1" / "sim" / "report" / "aria_top_cosim.rpt"
    cosimulation.parent.mkdir(parents=True)
    cosimulation.write_text(_COSIM_REPORT, encoding="utf-8")

    record = Project.open(project_path).record(report_dir)

    assert record.rtl_cosimulation == "passed"
    qualification = record.to_dict()
    assert qualification["rtl_cosimulation"] == "passed"
    assert qualification["report_files"][
        "solution1/sim/report/aria_top_cosim.rpt"
    ] == hashlib.sha256(_COSIM_REPORT.encode()).hexdigest()


def test_import_records_first_convolution_pipeline_evidence(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    first_convolution = "first_conv_4row_4lane_temporal_wide_cl"
    _write_project(project_path, first_convolution=first_convolution)
    report_dir = tmp_path / "reports"
    top_report = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    top_report.write_text(_CSYNTH_XML, encoding="utf-8")
    stage_report = report_dir / f"{first_convolution}_specialized_csynth.xml"
    stage_report.write_text(_FIRST_CONV_CSYNTH_XML, encoding="utf-8")

    record = Project.open(project_path).record(report_dir)

    assert record.stages == {
        "first_convolution": {
            "top": f"{first_convolution}_specialized",
            "initiation_interval": 258,
            "latency_cycles": 258,
            "loop": {
                "name": "ReadAndDrainP4",
                "trip_count": 85,
                "pipeline_ii": 3,
                "pipeline_depth": 5,
            },
        }
    }
    qualification = record.to_dict()
    assert qualification["schema_version"] == 4
    assert qualification["stages"] == record.stages
    assert qualification["report_files"][stage_report.name] == hashlib.sha256(
        _FIRST_CONV_CSYNTH_XML.encode()
    ).hexdigest()


def test_import_records_phara_fused_region_and_dense_stage_evidence(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path, schema_version=5)
    manifest_path = project_path / "ravel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_design"] = {
        "strategy": {"id": "phara", "version": 1},
        "rendering": {
            "phara_fused_function": "phara_pool_aligned_direct_p8_cl",
            "dense_function": "dense_wide_stream",
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "aria_top_csynth.xml").write_text(
        _CSYNTH_XML, encoding="utf-8"
    )
    fused_report = report_dir / "phara_pool_aligned_direct_p8_cl_1_csynth.xml"
    fused_report.write_text(
        _stage_csynth_xml(
            top="phara_pool_aligned_direct_p8_cl_1",
            latency=42,
            interval=42,
            loop="ProduceP8Words",
            trip_count=42,
            pipeline_ii=1,
            pipeline_depth=8,
        ),
        encoding="utf-8",
    )
    dense_report = report_dir / "dense_wide_stream_array_s_csynth.xml"
    dense_report.write_text(
        _stage_wrapper_csynth_xml(
            top="dense_wide_stream_array_s",
            latency=50,
            interval=50,
        ),
        encoding="utf-8",
    )
    dense_loop_report = (
        report_dir
        / "dense_wide_stream_array_Pipeline_DenseValues_csynth.xml"
    )
    dense_loop_report.write_text(
        _stage_csynth_xml(
            top="dense_wide_stream_array_Pipeline_DenseValues",
            latency=47,
            interval=47,
            loop="DenseValues",
            trip_count=42,
            pipeline_ii=1,
            pipeline_depth=4,
        ),
        encoding="utf-8",
    )

    record = Project.open(project_path).record(report_dir)

    assert record.to_dict()["schema_version"] == 4
    assert record.stages == {
        "phara_fused_region": {
            "top": "phara_pool_aligned_direct_p8_cl_1",
            "initiation_interval": 42,
            "latency_cycles": 42,
            "loop": {
                "name": "ProduceP8Words",
                "trip_count": 42,
                "pipeline_ii": 1,
                "pipeline_depth": 8,
            },
        },
        "dense": {
            "top": "dense_wide_stream_array_s",
            "initiation_interval": 50,
            "latency_cycles": 50,
            "loop": {
                "name": "DenseValues",
                "trip_count": 42,
                "pipeline_ii": 1,
                "pipeline_depth": 4,
            },
        },
    }
    assert dense_report.name in record.report_files
    assert dense_loop_report.name in record.report_files


def test_import_rejects_a_missing_requested_rtl_cosimulation_report(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path, cosim=True)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "aria_top_csynth.xml").write_text(
        _CSYNTH_XML, encoding="utf-8"
    )

    with pytest.raises(ProjectGenerationError, match="requested RTL CoSim"):
        Project.open(project_path).record(report_dir)

    assert not (project_path / "ravel_qualification.json").exists()


def test_project_marks_qualification_with_a_foreign_fingerprint_as_stale(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project"
    _write_project(project_path)
    report_dir = tmp_path / "reports"
    report_path = report_dir / "aria_top_csynth.xml"
    report_dir.mkdir()
    report_path.write_text(_CSYNTH_XML, encoding="utf-8")
    Project.open(project_path).record(report_dir)
    qualification_path = project_path / "ravel_qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["generation_fingerprint"] = "2" * 64
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")

    assert Project.open(project_path).status["performance_qualification"] == "stale"


def _write_project(
    project_path: Path,
    *,
    schema_version: int = 2,
    cosim: bool = False,
    first_convolution: str | None = None,
) -> None:
    source = "void aria_top() {}\n"
    (project_path / "firmware").mkdir(parents=True)
    (project_path / "firmware" / "aria_top.cpp").write_text(source, encoding="utf-8")
    hls_config = "Backend: Vitis\nIOType: io_stream\n"
    ravel_config = (
        "Project:\n"
        "  Name: aria_top\n"
        "  OutputDir: .\n"
        "HLS:\n"
        "  Backend: Vitis\n"
        "  IOType: io_stream\n"
        "  Config: {}\n"
        "Verification:\n"
        "  Mode: required\n"
        "Vitis:\n"
        "  Run: false\n"
        "  Stages:\n"
        f"    CoSim: {'true' if cosim else 'false'}\n"
    )
    (project_path / "hls4ml_config.yml").write_text(
        hls_config, encoding="utf-8"
    )
    (project_path / "ravel_config.yml").write_text(
        ravel_config, encoding="utf-8"
    )
    source_closure = [
        _closure_entry("firmware/aria_top.cpp", "firmware", source.encode()),
        _closure_entry("hls4ml_config.yml", "configuration", hls_config.encode()),
        _closure_entry("ravel_config.yml", "configuration", ravel_config.encode()),
    ]
    source_closure_sha256 = hashlib.sha256(
        json.dumps(
            source_closure, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": schema_version,
        "ravel": {"product": "RAVEL", "generation": "Aria", "release": "1.1.0"},
        "implementation_plan": {"template_profile": "aria-2x-v1"},
        "normalized_configuration": {
            "hls4ml": {
                "ProjectName": "aria_top",
                "Part": "xcku5p-ffvb676-2-e",
                "ClockPeriod": 5.0,
            },
            "ravel": {"Profile": "aria"},
        },
        "interfaces": {
            "rtl_interface": {
                "expected": {
                    "input_tdata_bits": 128,
                    "output_tdata_bits": 32,
                    "input_tdata_port": "input_layer_TDATA",
                    "output_tdata_port": "layer9_out_TDATA",
                },
                "measured": None,
            }
        },
        "status": {
            "generation": "complete",
            "dependency_qualification": "qualified",
            "correctness_verification": "passed",
            "model_fidelity": "reported",
            "source_integrity": "clean",
            "performance_qualification": "not_run",
        },
        "source_closure": source_closure,
        "source_closure_sha256": source_closure_sha256,
        "generation_fingerprint": "1" * 64,
    }
    if first_convolution is not None:
        manifest["resolved_design"] = {
            "rendering": {"first_convolution_function": first_convolution}
        }
    (project_path / "ravel_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _closure_entry(path: str, role: str, payload: bytes) -> dict[str, object]:
    return {
        "role": role,
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


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
    <RtlPorts><name>input_layer_TDATA</name><Dir>in</Dir><Bits>128</Bits></RtlPorts>
    <RtlPorts><name>layer9_out_TDATA</name><Dir>out</Dir><Bits>32</Bits></RtlPorts>
  </InterfaceSummary>
</profile>
"""


_COSIM_REPORT = """\
Report time       : Tue Aug 12 13:00:00 2026.
Solution          : solution1.
Simulation tool   : xsim.

+----------+----------+
|       RTL|    Status|
+----------+----------+
|   Verilog|      Pass|
+----------+----------+
"""


_FIRST_CONV_CSYNTH_XML = """\
<profile>
  <ReportVersion><Version>2023.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcku5p-ffvb676-2-e</Part>
    <TopModelName>first_conv_4row_4lane_temporal_wide_cl_specialized</TopModelName>
    <TargetClockPeriod>5.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Best-caseLatency>258</Best-caseLatency><Interval-min>258</Interval-min>
    </SummaryOfOverallLatency>
    <SummaryOfLoopLatency>
      <ReadAndDrainP4>
        <TripCount>85</TripCount><PipelineII>3</PipelineII><PipelineDepth>5</PipelineDepth>
      </ReadAndDrainP4>
    </SummaryOfLoopLatency>
  </PerformanceEstimates>
</profile>
"""


def _stage_csynth_xml(
    *,
    top: str,
    latency: int,
    interval: int,
    loop: str,
    trip_count: int,
    pipeline_ii: int,
    pipeline_depth: int,
) -> str:
    return f"""\
<profile>
  <ReportVersion><Version>2023.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcku5p-ffvb676-2-e</Part>
    <TopModelName>{top}</TopModelName>
    <TargetClockPeriod>5.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Best-caseLatency>{latency}</Best-caseLatency>
      <Interval-min>{interval}</Interval-min>
    </SummaryOfOverallLatency>
    <SummaryOfLoopLatency>
      <{loop}>
        <TripCount>{trip_count}</TripCount>
        <PipelineII>{pipeline_ii}</PipelineII>
        <PipelineDepth>{pipeline_depth}</PipelineDepth>
      </{loop}>
    </SummaryOfLoopLatency>
  </PerformanceEstimates>
</profile>
"""


def _stage_wrapper_csynth_xml(
    *,
    top: str,
    latency: int,
    interval: int,
) -> str:
    return f"""\
<profile>
  <ReportVersion><Version>2023.2</Version></ReportVersion>
  <UserAssignments>
    <Part>xcku5p-ffvb676-2-e</Part>
    <TopModelName>{top}</TopModelName>
    <TargetClockPeriod>5.00</TargetClockPeriod>
  </UserAssignments>
  <PerformanceEstimates>
    <SummaryOfOverallLatency>
      <Best-caseLatency>{latency}</Best-caseLatency>
      <Interval-min>{interval}</Interval-min>
    </SummaryOfOverallLatency>
    <SummaryOfLoopLatency />
  </PerformanceEstimates>
</profile>
"""
