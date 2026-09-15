"""Executable AXI handshake checks against exact clean-baseline word vectors."""

import json
from pathlib import Path
import re
import shutil
import subprocess

from ..exceptions import VerificationError
from ..manifest import file_sha256


def run_protocol(rtl_dir: Path, vectors: Path, output: Path, *, top: str, input_port: str, output_port: str) -> dict:
    for identifier in (top, input_port, output_port):
        if not re.fullmatch(r"[A-Za-z_]\w*", identifier):
            raise VerificationError("Invalid RTL interface identifier")
    reference = json.loads((vectors / "rtl_vectors.json").read_text())
    for name, digest in reference["files"].items():
        if file_sha256(vectors / name) != digest:
            raise VerificationError("RTL vectors have changed since baseline binding")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(rtl_dir.glob("*.v"))
    if not files:
        raise VerificationError("No synthesized Verilog files for protocol verification")
    for data in rtl_dir.glob("*.dat"):
        shutil.copyfile(data, output / data.name)
    for name in reference["files"]:
        shutil.copyfile(vectors / name, output / name)
    testbench = output / "ravel_protocol_tb.sv"
    testbench.write_text(_testbench(top, input_port, output_port, reference))
    if all(shutil.which(command) for command in ("xvlog", "xelab", "xsim")):
        tool = "xsim"
        compile_command = ["xvlog", "--sv", str(testbench), *map(str, files)]
        run_command = ["xsim", "ravel_protocol_snapshot", "-runall"]
    elif shutil.which("verilator"):
        tool = "verilator"
        compile_command = [tool, "--binary", "--timing", "-Wno-fatal", "--top-module", "ravel_protocol_tb", "--Mdir", str(output / "obj"), str(testbench), *map(str, files)]
        run_command = [str(output / "obj/Vravel_protocol_tb")]
    elif shutil.which("iverilog"):
        tool = "iverilog"
        binary = output / "simulation.vvp"
        compile_command = [tool, "-g2012", "-s", "ravel_protocol_tb", "-o", str(binary), str(testbench), *map(str, files)]
        run_command = ["vvp", str(binary)]
    else:
        raise VerificationError("RTL protocol verification requires Icarus Verilog or Verilator")
    commands = [("compile", compile_command)]
    if tool == "xsim":
        commands.append(("elaborate", ["xelab", "ravel_protocol_tb", "--snapshot", "ravel_protocol_snapshot", "--timescale", "1ns/1ps"]))
    commands.append(("simulation", run_command))
    for label, command in commands:
        with (output / f"{label}.log").open("w") as log:
            result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise VerificationError(f"RTL protocol {label} failed; see {output / (label + '.log')}")
    log = (output / "simulation.log").read_text()
    matched = re.search(r"RAVEL_PROTOCOL_PASS completed=(\d+) stalls=(\d+) gaps=(\d+)", log)
    if not matched or int(matched[2]) < 1:
        raise VerificationError("RTL protocol run has no complete passing handshake evidence")
    report = {"schema_version": 1, "status": "passed", "top": top,
              "simulator": {"name": tool, "version": subprocess.run([tool, "-V" if tool == "iverilog" else ("-version" if tool == "xsim" else "--version")], capture_output=True, text=True).stdout.splitlines()[0]},
              "completed_samples": int(matched[1]), "output_stall_cycles": int(matched[2]), "input_gap_cycles": int(matched[3]),
              "reset_aborts": 1, "reset_between_epochs": 1, "consecutive_samples": reference["sample_count"],
              "vector_files": reference["files"], "testbench_sha256": file_sha256(testbench),
              "rtl_files": {path.name: file_sha256(path) for path in files},
              "log_sha256": file_sha256(output / "simulation.log")}
    (output / "rtl_protocol.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def _testbench(top, input_port, output_port, reference):
    count = reference["sample_count"]
    words = reference["input_words_per_inference"]
    bits_in = reference["input_tdata_bits"]
    bits_out = reference["output_tdata_bits"]
    return f'''`timescale 1ns/1ps
module ravel_protocol_tb;
reg clk=0; always #2.5 clk=~clk;
reg rst=0, start=0, in_valid=0, out_ready=0;
reg [{bits_in-1}:0] in_data=0;
wire [{bits_out-1}:0] out_data;
wire in_ready, out_valid, control_ready;
reg [{bits_in-1}:0] inputs[0:{count*words-1}];
reg [{bits_out-1}:0] expected[0:{count-1}];
integer stalls=0, gaps=0, completed=0;
{top} dut(.ap_clk(clk), .ap_rst_n(rst), .ap_start(start), .ap_ready(control_ready),
 .{input_port}_TDATA(in_data), .{input_port}_TVALID(in_valid), .{input_port}_TREADY(in_ready),
 .{output_port}_TDATA(out_data), .{output_port}_TVALID(out_valid), .{output_port}_TREADY(out_ready));
task reset_dut;
begin
 @(negedge clk); rst=0; start=0; in_valid=0; out_ready=0;
 repeat(8) @(posedge clk);
 @(negedge clk); rst=1; start=0;
end
endtask
task epoch(input integer samples);
integer sent, received, cycles, launched;
reg pending, blocked;
reg [{bits_out-1}:0] held;
begin
 sent=0; received=0; cycles=0; launched=0; pending=0; blocked=0;
 $display("RAVEL_EPOCH samples=%0d", samples);
 while(sent < samples*{words} || received < samples || launched < samples) begin
   @(negedge clk);
   if(!pending && sent < samples*{words} && cycles%7 != 0) begin
     pending=1; in_data=inputs[sent];
   end
   in_valid=pending; start=(launched < samples);
   out_ready=(cycles%17 >= 5);
   @(posedge clk);
   if(start && control_ready) launched=launched+1;
   if(blocked && (!out_valid || out_data !== held)) $fatal(1, "AXI output changed while stalled");
   blocked=out_valid && !out_ready; held=out_data;
   if(blocked) stalls=stalls+1;
   if(!in_valid && sent < samples*{words}) gaps=gaps+1;
   if(in_valid && in_ready) begin sent=sent+1; pending=0; end
   if(out_valid && out_ready) begin
     if(received >= samples || out_data !== expected[received])
       $fatal(1, "Integer-code mismatch sample %0d got %h expected %h", received, out_data, expected[received]);
     received=received+1; completed=completed+1;
   end
   cycles=cycles+1;
   if(cycles > 10000000) $fatal(1, "Protocol deadlock or incomplete output");
 end
 @(negedge clk); in_valid=0; start=0; out_ready=1;
 repeat(32) begin @(posedge clk); if(out_valid) $fatal(1, "Extra output after inference drain"); end
end
endtask
integer sent_prefix, cycles_prefix;
initial begin
 $readmemh("rtl_input_words.hex", inputs);
 $readmemh("rtl_expected_words.hex", expected);
 reset_dut;
 sent_prefix=0; cycles_prefix=0;
 while(sent_prefix < {max(1, words//2)}) begin
   @(negedge clk); start=1; in_valid=1; in_data=inputs[sent_prefix]; out_ready=0;
   @(posedge clk); if(in_ready) sent_prefix=sent_prefix+1;
   cycles_prefix=cycles_prefix+1;
   if(cycles_prefix > 100000) $fatal(1, "Prefix deadlock");
 end
 reset_dut;
 epoch({count});
 reset_dut;
 epoch({min(3,count)});
 $display("RAVEL_PROTOCOL_PASS completed=%0d stalls=%0d gaps=%0d", completed, stalls, gaps);
 $finish;
end
endmodule
'''


def verify_project_protocol(project, rtl_dir: Path, output: Path) -> dict:
    """Verify synthesized interfaces and bind the trace to a fresh project view."""
    from ..project import Project

    view = Project.open(project.path if hasattr(project, "path") else project)
    if view.status["source_integrity"] != "clean":
        raise VerificationError("Cannot verify modified project sources")
    expected = view.manifest["interfaces"]["rtl_interface"]["expected"]
    top = view.manifest["normalized_configuration"]["hls4ml"]["ProjectName"]
    report = run_protocol(Path(rtl_dir).resolve(), view.path / "tb_data", Path(output).resolve(), top=top,
                          input_port=expected["input_tdata_port"].removesuffix("_TDATA"),
                          output_port=expected["output_tdata_port"].removesuffix("_TDATA"))
    report.update(manifest_sha256=file_sha256(view.path / "ravel_manifest.json"),
                  source_closure_sha256=view.manifest["source_closure_sha256"], interfaces=view.manifest["interfaces"])
    (output / "rtl_protocol.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
