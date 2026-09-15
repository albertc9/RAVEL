from pathlib import Path
import shutil

import numpy as np
import pytest

from ravel_hls.domain.graph import NumericType
from ravel_hls.verification.rtl_vectors import write_rtl_vectors
from ravel_hls.verification.rtl_protocol import run_protocol


@pytest.mark.skipif(not (shutil.which("iverilog") or shutil.which("verilator")), reason="RTL simulator unavailable")
def test_protocol_runner_checks_stalls_reset_abort_and_consecutive_code_outputs(tmp_path):
    numeric = NumericType("fixed", 8, 8, False, "TRN", "WRAP")
    values = np.array([[1], [127], [255], [42]], dtype=np.float64)
    vectors = tmp_path / "vectors"
    write_rtl_vectors(vectors, values, values, numeric, numeric, 1)
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "identity.v").write_text('''module identity(input ap_clk, input ap_rst_n, input ap_start,
        input [7:0] input_TDATA, input input_TVALID, output input_TREADY,
        output reg [7:0] output_TDATA, output reg output_TVALID, input output_TREADY);
        assign input_TREADY = ap_start && (!output_TVALID || output_TREADY);
        always @(posedge ap_clk) begin
          if (!ap_rst_n) begin output_TVALID <= 0; output_TDATA <= 0; end
          else if (input_TREADY) begin output_TVALID <= input_TVALID; output_TDATA <= input_TDATA; end
        end
        endmodule''')
    report = run_protocol(rtl, vectors, tmp_path / "run", top="identity", input_port="input", output_port="output")
    assert report["status"] == "passed"
    assert report["completed_samples"] == 7
    assert report["reset_aborts"] == 1
    assert report["output_stall_cycles"] > 0
