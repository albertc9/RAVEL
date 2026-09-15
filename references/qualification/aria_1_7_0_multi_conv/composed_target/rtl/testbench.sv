`timescale 1ns/1ps
module ravel_protocol_tb;
reg clk=0; always #2.5 clk=~clk;
reg rst=0, start=0, in_valid=0, out_ready=0;
reg [511:0] in_data=0;
wire [31:0] out_data;
wire in_ready, out_valid, control_ready;
reg [511:0] inputs[0:3071];
reg [31:0] expected[0:95];
integer stalls=0, gaps=0, completed=0;
composed_target dut(.ap_clk(clk), .ap_rst_n(rst), .ap_start(start), .ap_ready(control_ready),
 .waveform_TDATA(in_data), .waveform_TVALID(in_valid), .waveform_TREADY(in_ready),
 .layer12_out_TDATA(out_data), .layer12_out_TVALID(out_valid), .layer12_out_TREADY(out_ready));
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
reg [31:0] held;
begin
 sent=0; received=0; cycles=0; launched=0; pending=0; blocked=0;
 $display("RAVEL_EPOCH samples=%0d", samples);
 while(sent < samples*32 || received < samples || launched < samples) begin
   @(negedge clk);
   if(!pending && sent < samples*32 && cycles%7 != 0) begin
     pending=1; in_data=inputs[sent];
   end
   in_valid=pending; start=(launched < samples);
   out_ready=(cycles%17 >= 5);
   @(posedge clk);
   if(start && control_ready) launched=launched+1;
   if(blocked && (!out_valid || out_data !== held)) $fatal(1, "AXI output changed while stalled");
   blocked=out_valid && !out_ready; held=out_data;
   if(blocked) stalls=stalls+1;
   if(!in_valid && sent < samples*32) gaps=gaps+1;
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
 while(sent_prefix < 16) begin
   @(negedge clk); start=1; in_valid=1; in_data=inputs[sent_prefix]; out_ready=0;
   @(posedge clk); if(in_ready) sent_prefix=sent_prefix+1;
   cycles_prefix=cycles_prefix+1;
   if(cycles_prefix > 100000) $fatal(1, "Prefix deadlock");
 end
 reset_dut;
 epoch(96);
 reset_dut;
 epoch(3);
 $display("RAVEL_PROTOCOL_PASS completed=%0d stalls=%0d gaps=%0d", completed, stalls, gaps);
 $finish;
end
endmodule
