# CNN for Arianna reference

This directory is the executable Aria 1.1.0 reference consumer. It owns the
canonical trained Keras/HGQ2 model and uses only RAVEL's public API.

```bash
python -m pip install -c constraints/aria-reference.txt -e .
ravel-hls doctor --json
python references/cnn_for_arianna/generate.py
```

The default run performs required bit-exact baseline/optimized C++ verification
with 32 deterministic synthetic samples, targets `xcku5p-ffvb676-2-e` at 5 ns,
and publishes below `generated/`. Supply `--inputs test_vectors.npy` for a local
tensor shaped `[samples, 256, 4]`; its source path is not recorded.

Vitis is off by default. Add `--vitis` to set `Vitis.Run` true in the same Python
configuration. RAVEL then runs the standalone Vitis HLS 2023.2 launcher after
publication, using the default reset+synthesis stage profile, and automatically
writes `ravel_qualification.json` on success. No user-written conditional or
manual report-import step is required.

The qualification records the measured result for that exact project. It does
not require II 178 or any other application-specific performance number. Enable
additional `Vitis.Stages` only when their separate CSim, CoSim, validation,
export, or Vivado-synthesis evidence is wanted.

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
| RAVEL Aria 1.1.0 | 178 | 183 | 0 | 4 | 3483 | 28922 |

The historical observation is a 17.3x lower initiation interval and a 16.8x
lower cycle latency for RAVEL, with fewer reported resources. This is context,
not a qualification gate or a strict speedup claim: the historical project uses
an older parameter set, materialized `Conv2D`/`Dense` layers, and a 64-bit input
port, while the current HGQ2 reference retains `QConv2D`/`QDense` and produces a
128-bit RAVEL input port. Run the exact-current-model baseline above before using
the comparison as a like-for-like result.
