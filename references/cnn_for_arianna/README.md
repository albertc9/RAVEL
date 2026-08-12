# CNN for Arianna reference

This directory is the executable Aria 1.5.1 reference consumer. It owns the
canonical trained Keras/HGQ2 model and uses only RAVEL's public API.

```bash
python -m pip install -c constraints/aria-reference.txt -e .
ravel-hls doctor --json
python references/cnn_for_arianna/generate.py
```

The default run performs required bit-exact baseline/optimized C++ verification
with 32 deterministic synthetic samples, targets `xcku5p-ffvb676-2-e` at 5 ns,
selects P4/D2 through RAVEL's omitted-option default, and publishes below
`generated/`. Supply `--inputs test_vectors.npy` for a local
tensor shaped `[samples, 256, 4]`; its source path is not recorded.

Use `--temporal-packing {2,4}` and `--dense-parallelism {1,2}` for an explicit
specialization. The flags are independent. If neither is present, the script
does not add an `Optimization` section and RAVEL resolves P4/D2.

Vitis is off by default. Add `--vitis` to set `Vitis.Run` true in the same Python
configuration. RAVEL then runs the standalone Vitis HLS 2023.2 launcher after
publication. This reference enables reset, synthesis, and RTL CoSim, then
automatically writes `ravel_qualification.json` on success. No manual
report-import step is required.

The qualification records the measured result for that exact project. It does
not impose an application-specific performance threshold. Additional CSim,
validation, export, or Vivado-synthesis stages remain explicit choices.

## Historical Aria 1.4 default result

The default P4/D2 project was generated with 32 deterministic verification
samples and synthesized on Vitis HLS 2023.2. Verilog RTL CoSim passed.

| Flow | II | Latency (cycles) | Est. clock (ns) | BRAM_18K | DSP | FF | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Aria 1.1.0 P2/D1 | 178 | 183 | 3.647 | 0 | 4 | 3483 | 28922 |
| Aria 1.3.0 P4/D2 | 94 | 99 | 3.502 | 0 | 8 | 4436 | 53502 |
| Aria 1.4.0 P4/D2 | 94 | 99 | 3.402 | 3 | 8 | 4368 | 15623 |

Aria 1.4 retains the Aria 1.3 P4/D2 II and latency. The sequential packed Dense
weight ROM reduces estimated LUT by 70.8% and FF by 1.5%, uses three BRAM_18Ks,
and leaves DSP usage unchanged. The [evidence record](reports/aria_1_4_p4d2.json),
adjacent synthesis XML, and RTL CoSim report provide the audit trail.

## Vanilla hls4ml baseline

Use the companion baseline to measure what hls4ml produces before RAVEL applies
the Aria transformation:

```bash
export PATH="$PWD/references/cnn_for_arianna/tools:$PATH"
python references/cnn_for_arianna/baseline.py --vitis
```

The script calls hls4ml's public conversion, writer, and build APIs. The output
directory must not already exist, and no generated C++, headers, Tcl, or YAML
are edited after hls4ml writes them. The small external `vitis-run` adapter only
translates hls4ml 1.2.0's launcher command to the `vitis_hls -f` command provided
by Vitis HLS 2023.2; it never writes into the generated project.

This baseline descends from CNN-Core-Generator commit
`6e16cd474bcf45e41b173734b59e70ddd6ed6323`, the first direct homogeneous
IOStream conversion of the low-BOP model. The reference copy and that historical
model are byte-identical, with SHA-256
`65021d84030d9c09a7f1fd541221b150dad14858ad85458912a1a6a6b40a9978`.
Both comparison flows use hls4ml's generated per-layer precision, `IOStream`,
`Latency`, reuse factor 1, `xcku5p-ffvb676-2-e`, and a 5 ns target. Unlike the
historical script, this baseline does not add a separate input-precision or FIFO
depth override, so the only source-level difference in the comparison is the
RAVEL transformation.

### Exact-current-model result

The reproducible baseline was synthesized on Vitis HLS 2023.2 from RAVEL commit
`103f55e`. It used the exact reference model SHA-256 above and the same part,
clock target, hls4ml version, strategy, reuse factor, and generated precision
configuration as the RAVEL flow.

| Flow | II | Latency (cycles) | Est. clock (ns) | BRAM_18K | DSP | FF | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla hls4ml | 3076 | 3084 | 3.619 | 18 | 0 | 26275 | 38365 |
| RAVEL Aria 1.1.0 P2/D1 | 178 | 183 | 3.647 | 0 | 4 | 3483 | 28922 |

RAVEL reduces the measured initiation interval by 17.3x and cycle latency by
16.9x. It also uses 86.7% fewer FF, 24.6% fewer LUT, and no BRAM_18K. The tradeoff
is four DSPs instead of zero and a 0.028 ns higher estimated clock period; both
flows remain below the 5 ns target. These are measurements, not performance
gates for the general-purpose RAVEL API.

To audit the no-edit condition, the baseline was generated a second time into a
control directory without synthesis. Its complete firmware tree, bridge,
testbench, and build script were byte-identical to the synthesized project. The
only YAML differences were the output-specific paths and hls4ml's random stamp.
The report, hashes, environment, build stages, and comparison ratios are stored
in `legacy/reports/hls4ml_exact_current.json` with the adjacent original XML.

### Historical context (not like-for-like)

CNN-Core-Generator commit `c389552` contains a Vitis HLS 2023.2 report for an
older, topologically equivalent direct-hls4ml project. No generated C++, header,
or hls4ml YAML changed between its generation commit (`2b01843`) and the report
commit. The report commit enabled `fifo_opt` and `vsynth`, but Vitis emitted the
listed C-synthesis result before FIFO optimization. The source-backed evidence
is preserved in `legacy/reports/hls4ml_c389552.json` and its adjacent XML.

| Flow | II | Latency (cycles) | BRAM_18K | DSP | FF | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla hls4ml (`c389552`) | 3076 | 3082 | 19 | 14 | 30061 | 34994 |
| RAVEL Aria 1.1.0 P2/D1 | 178 | 183 | 0 | 4 | 3483 | 28922 |

The historical observation is a 17.3x lower initiation interval and a 16.8x
lower cycle latency for RAVEL, with fewer reported resources. This is context,
not a qualification gate or a strict speedup claim: the historical project uses
an older parameter set, materialized `Conv2D`/`Dense` layers, and a 64-bit input
port, while the HGQ2 P2/D1 reference retains `QConv2D`/`QDense` and produces a
128-bit input port. Aria 1.3 and 1.4 P4/D2 use a 256-bit input port. Run the
exact-current-model baseline above before using
the historical comparison as a like-for-like result; the exact result above is
the primary comparison.
