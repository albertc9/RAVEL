#!/usr/bin/env python3
"""Qualify an external model using pinned Python and Vitis HLS/Vivado 2023.2.

Creates a project and sibling protocol/OOC evidence directories. Model and
optional NPZ assets remain external to the repository.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from ravel_hls import Project, convert
from ravel_hls.manifest import file_sha256
from ravel_hls.verification.rtl_protocol import verify_project_protocol

OOC_SCRIPT = """read_verilog [glob -directory $::env(RAVEL_RTL_DIR) *.v]
synth_design -mode out_of_context -top $::env(RAVEL_TOP) -part $::env(RAVEL_PART)
create_clock -period $::env(RAVEL_CLOCK_NS) [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -file timing.rpt
report_utilization -file utilization.rpt
report_utilization -hierarchical -file utilization_hierarchical.rpt
write_checkpoint -force routed.dcp
exit
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, help="Optional additive NPZ corpus containing X")
    parser.add_argument("--temporal-packing", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--dense-parallelism", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--part", default="xcku5p-ffvb676-2-e")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument("--target-ii", type=int, help="Optional best-effort analytical II target")
    parser.add_argument("--skip-ooc", action="store_true")
    args = parser.parse_args()
    inputs = None
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as data:
            inputs = data["X"].copy()
    project = convert(args.model, args.output, {
        "HLS": {"Part": args.part, "ClockPeriod": args.clock_ns},
        "Optimization": {"TemporalPacking": args.temporal_packing, "DenseParallelism": args.dense_parallelism,
                         **({"TargetII": args.target_ii} if args.target_ii is not None else {})},
        "Verification": {"Mode": "required", "Samples": 32},
        "Vitis": {"Stages": {"CSim": True, "Synth": True, "CoSim": True}},
    }, verification_inputs=inputs)
    print("C and stage boundaries passed", flush=True)
    with (project.path / "vendor.log").open("w") as log:
        subprocess.run(["vitis_hls", "-f", "build_prj.tcl"], cwd=project.path,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print("HLS and CoSim passed", flush=True)
    top = project.path.name
    rtl = project.path / f"{top}_prj/solution1/syn/verilog"
    protocol_dir = project.path.with_name(top + "_protocol")
    protocol = verify_project_protocol(project, rtl, protocol_dir)
    print(f"RTL protocol passed: {protocol['completed_samples']} outputs", flush=True)
    ooc_dir = None
    if not args.skip_ooc:
        ooc_dir = project.path.with_name(top + "_ooc")
        ooc_dir.mkdir(exist_ok=True)
        binding = {"manifest_sha256": file_sha256(project.path / "ravel_manifest.json"),
                   "source_closure_sha256": project.manifest["source_closure_sha256"],
                   "top": top, "part": args.part, "clock_period_ns": args.clock_ns, "tool_version": "2023.2"}
        (ooc_dir / "binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        script = ooc_dir / "implement.tcl"
        script.write_text(OOC_SCRIPT)
        environment = {**os.environ, "RAVEL_RTL_DIR": str(rtl), "RAVEL_TOP": top,
                       "RAVEL_PART": args.part, "RAVEL_CLOCK_NS": str(args.clock_ns)}
        with (ooc_dir / "run.log").open("w") as log:
            subprocess.run(["vivado", "-mode", "batch", "-source", str(script)], cwd=ooc_dir,
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    record = Project.open(project.path).record(project.path, ooc_dir=ooc_dir,
                                               protocol_report=protocol_dir / "rtl_protocol.json")
    print(json.dumps(record.to_dict()["performance"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
